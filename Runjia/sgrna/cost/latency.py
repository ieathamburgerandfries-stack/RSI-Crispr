"""Scoring latency of SLICER against crisprHAL 2, on one thread, in the three forms it has been measured: the model alone (0.002 ms per guide for SLICER's default model against 1.32 ms for crisprHAL 2 at a batch of 100), end to end with featurisation, and over a stretch of chromosome the way a design tool scores it.

The repeated, quiet-session timings that the report quotes are `cost_repeats`; these parts are the single-pass benchmarks that came before it, kept because their result files are audited. Training cost is the wrong clock for a
guide-design tool (a user fits once and then scores every candidate guide for every target), and these parts measure the clock that matters. Models are fitted on random data of the real width and tree count, so
timings do not depend on what was learned; every library runs on one thread (XGBoost `n_jobs=1`, LightGBM `n_jobs=1`, CatBoost `thread_count=1`, TensorFlow intra- and inter-op threads 1, and the BLAS thread variables below).

  model        the model alone on random input of the right width, batches of 1, 100 and 10,000               -> cost_slicer.csv       (--members adds the mixed ensemble)
  featurise    the 6,517 columns per guide, including the 1,000-nt flank windows, warm and on one thread       -> cost_featurise.csv
  e2e          featurisation + column selection + imputation + model, on the cohort's guides                  -> e2e_slicer.csv
  locus        every NGG site of a stretch of chromosome through the per-locus precompute: the design path    -> e2e_slicer_locus.csv
  inference    one fit of the champion, then latency and throughput of scoring 1, 100 and 10,000 guides      -> inference_cost.csv    (the oldest of the three; crisprHAL 2's numbers are recorded in HAL_INFERENCE)
  hal-model    crisprHAL 2's shipped WT-SpCas9 network alone, TensorFlow, one thread                          -> cost_crisprhal.csv
  hal-e2e      crisprHAL 2 with its one-hot encoding of random 378-nt contexts, then the network              -> e2e_crisprhal.csv

    python -m sgrna.cost.latency --parts model featurise e2e locus --members d099x,...
    python -m sgrna.cost.latency --parts inference --reps 5
    .venv-crisprhal/bin/python sgrna/cost/latency.py --parts hal-model hal-e2e --hal <project root>      # the TensorFlow side runs as a script, under the other interpreter
    python -m sgrna.cost.latency --parts model --quick                                                   # one repeat, batches 1 and 100: a smoke test that writes the same columns

The two `hal-*` parts write to `<project root>/RSI-Crispr-fork/Runjia/results/` and import only numpy, pandas and TensorFlow; the other parts need the package. Timings move with machine load and cannot be reproduced to the
byte, so no `reproduce` case covers this module (`--quick` checks the code path and the columns, not the numbers). It replaced `cost_bench`, `e2e_bench` and `inference_cost`.

Report section: Part 5.
"""
import os
import sys

# The end-to-end and model timings run on one thread for every library. The variables must be set before numpy loads, which is why they are set here: for every part except the single-fit `inference`
# (whose fit time is a different measurement, at the library's default threads), and whenever the module is imported by another one (`cost_repeats` imports `_make_models` and has always run this way).
if not (__name__ == "__main__" and {a for a in sys.argv[1:] if not a.startswith("-") and a in {"model", "featurise", "e2e", "locus", "inference", "hal-model", "hal-e2e"}} == {"inference"}):
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"

import argparse
import json
import time

import numpy as np
import pandas as pd

QUICK = False


def _batches():
    return (1, 100) if QUICK else (1, 100, 10000)


def _reps(full: int) -> int:
    return 1 if QUICK else full


def _time(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1000.0


# crisprHAL 2's shipped WT-SpCas9 model, measured on the same machine and the same CPU with `colab/crisprhal_inference_bench.py` under `.venv-crisprhal` (TensorFlow 2.19.1). It is kept here as recorded numbers
# because it needs the other interpreter; re-run that script to refresh them. Both models are on this machine's CPU. The like-for-like GPU comparison was run on a Colab T4 (`colab/inference_bench_t4.py`,
# `results/inference_colab.csv`, `results/inference_t4.csv`, superseded by `results/inference_onecore.csv`): crisprHAL 2 takes 78.0 / 99.7 / 2,768 ms for 1 / 100 / 10,000 guides there and 9.8 s to cold-start,
# and SLICER on the same instance's CPU is 38-160x faster.
HAL_INFERENCE = {
    "import_and_load_s": 13.4,
    1: dict(latency_s=0.04426, guides_per_s=23),
    100: dict(latency_s=0.10168, guides_per_s=983),
    10_000: dict(latency_s=8.16176, guides_per_s=1225),
}


# ---------------------------------------------------------------------------------------------------------------- the model alone
def model_side(mix_names):
    import xgboost as xgb
    from ..common import OUT
    from ..tuning import mix_ecoli as M
    from ..tuning import tune_ecoli as T
    sel = T.selected()
    cfgs = {**json.loads((OUT / "tune_configs.json").read_text()), **M.all_cfgs()}
    fin = pd.read_csv(OUT / "mix_finalists.csv").set_index("name") if (OUT / "mix_finalists.csv").exists() else None
    rng = np.random.default_rng(0)
    rows = []

    def make(name):
        kind = "xgb" if (fin is None or name not in fin.index) else fin.loc[name, "kind"]
        k = int(sel.loc[name, "k"]) if kind == "xgb" else int(fin.loc[name, "k"])
        cfg = dict(cfgs[name], n_estimators=k)
        nf = int(cfg.get("nf", 0)) or 6517
        Xf = rng.standard_normal((3000, nf)).astype(np.float32); yf = rng.standard_normal(3000)
        if kind == "xgb":
            m = xgb.XGBRegressor(**dict(T.xgb_params(cfg, 0), n_jobs=1)).fit(Xf, yf)
            fn = lambda X: m.predict(X[:, :nf])
        elif kind == "lgb":
            import lightgbm as lgb
            m = lgb.LGBMRegressor(n_estimators=k, learning_rate=cfg["learning_rate"], num_leaves=int(cfg["num_leaves"]), max_depth=int(cfg["max_depth"]),
                                  n_jobs=1, verbose=-1).fit(Xf, yf)
            fn = lambda X: m.predict(X[:, :nf])
        elif kind == "cat":
            from catboost import CatBoostRegressor
            m = CatBoostRegressor(iterations=k, learning_rate=cfg["learning_rate"], depth=int(cfg["depth"]), thread_count=1, verbose=0).fit(Xf, yf)
            fn = lambda X: m.predict(X[:, :nf])
        else:
            return None, nf, kind, k
        return fn, nf, kind, k

    sets = {"original champion": ["champion"], "tuned XGBoost (d099x)": ["d099x"],
            "XGBoost average of 5": ["d099x", "d077x", "d021x", "d080x", "d000x"]}
    if mix_names:
        sets["mixed ensemble"] = mix_names
    for lab, names in sets.items():
        fns = [make(n) for n in names]
        fns = [f for f in fns if f[0] is not None]
        for b in _batches():
            X = rng.standard_normal((b, 6517)).astype(np.float32)
            reps = _reps(200 if b == 1 else (20 if b == 100 else 3))
            ms = _time(lambda: [f[0](X) for f in fns], reps)
            rows.append(dict(model=lab, batch=b, latency_ms=ms, per_guide_ms=ms / b, n_members=len(fns),
                             total_trees=int(sum(f[3] for f in fns if f[2] != "et"))))
            print(f"  {lab:26s} batch {b:6d}: {ms:9.2f} ms", flush=True)
    return rows


def featurise_side(n=300):
    """Warm, one-thread cost of the 6,517 columns per guide (tables preloaded, 5 guides first to warm caches)."""
    from sgrna import config
    from sgrna.build import featurise, qct
    from sgrna.features import a_flank
    from sgrna.harness.curated import load_base
    n = 30 if QUICK else n
    _, _, _, protos = load_base()
    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv").set_index("sgRNAID")
    tables, vm = qct.load_tables(), featurise.load_v_mapping()
    P = list(protos[:n]); sub = idx.reindex(P).reset_index().rename(columns={"index": "sgRNAID"}); sub["located"] = True
    featurise.matrix(P[:5], v_mapping=vm, tables=tables, verbose=False); a_flank.build(sub.iloc[:5], tables=tables)
    t = time.perf_counter(); featurise.matrix(P, v_mapping=vm, tables=tables, verbose=False); b = time.perf_counter() - t
    t = time.perf_counter(); a_flank.build(sub, tables=tables); a = time.perf_counter() - t
    return [dict(part="base (6,169 columns from the 20-mer)", n_guides=n, per_guide_ms=b / n * 1000),
            dict(part="family A (348 columns from the 1,000-nt flank windows)", n_guides=n, per_guide_ms=a / n * 1000),
            dict(part="total", n_guides=n, per_guide_ms=(a + b) / n * 1000)]


# ---------------------------------------------------------------------------------------------------------------- end to end
def _make_models(sets):
    """One callable per model, fitted on random data of the real width and tree count (imported by `cost_repeats`)."""
    import xgboost as xgb
    from ..common import OUT
    from ..tuning import mix_ecoli as M
    from ..tuning import tune_ecoli as T
    sel = T.selected()
    cfgs = {**json.loads((OUT / "tune_configs.json").read_text()), **M.all_cfgs()}
    fin = pd.read_csv(OUT / "mix_finalists.csv").set_index("name")
    rng = np.random.default_rng(0)

    def make(name):
        kind = "xgb" if name not in fin.index else fin.loc[name, "kind"]
        k = int(sel.loc[name, "k"]) if kind == "xgb" else int(fin.loc[name, "k"])
        cfg = dict(cfgs[name], n_estimators=k); nf = int(cfg["nf"])
        cols = np.sort(rng.choice(6517, nf, replace=False)); med = rng.standard_normal(nf).astype(np.float32)
        Xf = rng.standard_normal((3000, nf)).astype(np.float32); yf = rng.standard_normal(3000)
        if kind == "xgb":
            m = xgb.XGBRegressor(**dict(T.xgb_params(cfg, 0), n_jobs=1)).fit(Xf, yf)
        elif kind == "lgb":
            import lightgbm as lgb
            m = lgb.LGBMRegressor(n_estimators=k, learning_rate=cfg["learning_rate"], num_leaves=int(cfg["num_leaves"]), max_depth=int(cfg["max_depth"]),
                                  n_jobs=1, verbose=-1).fit(Xf, yf)
        else:
            from catboost import CatBoostRegressor
            m = CatBoostRegressor(iterations=k, learning_rate=cfg["learning_rate"], depth=int(cfg["depth"]), thread_count=1, verbose=0).fit(Xf, yf)

        def fn(X):
            B = X[:, cols]
            return m.predict(np.where(np.isnan(B), med, B))
        return fn

    return {lab: [make(n) for n in names] for lab, names in sets.items()}


def e2e_side(sets):
    from sgrna import config
    from sgrna.build import featurise, qct
    from ..build import a_flank_fast as AF
    from ..build import base_fast as BF
    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv"); idx["located"] = idx["located"].astype(bool)
    tables, vm = qct.load_tables(), featurise.load_v_mapping()
    models = _make_models(sets)

    rows = []
    for lab, fns in models.items():
        for b in _batches():
            sub = idx.iloc[:b].copy()

            def run():
                Mb, _ = BF.matrix_fast(sub["protospacer"].tolist(), vm, tables)
                Ma, _ = AF.build_matrix(sub, tables)
                X = np.hstack([Mb, Ma.astype(np.float32)])
                return [f(X) for f in fns]
            run()
            ms = _time(run, _reps(200 if b == 1 else (20 if b == 100 else 3)))
            rows.append(dict(model=lab, batch=b, ms_per_call=ms, ms_per_guide=ms / b))
            print(f"  {lab:26s} batch {b:6d}: {ms:9.2f} ms  ({ms / b:.4f} ms/guide)", flush=True)
    return rows


def locus_side(sets, widths=(1_000, 100_000, 1_000_000), start=1_000_000):
    """Score every NGG site in a stretch of chromosome, through the per-locus precompute: the design path, end to end."""
    from sgrna.build import featurise, qct
    from ..build import base_fast as BF
    from ..build import flank_locus as FL
    widths = widths[:2] if QUICK else widths
    tables, vm = qct.load_tables(), featurise.load_v_mapping()
    codes = FL.genome_codes()
    models = _make_models(sets)

    rows = []
    for lab, fns in models.items():
        for width in widths:
            l0, sg = FL.ngg_sites(codes, start, start + width)
            plus = sg > 0
            n = len(l0)

            def run():
                Mb, _ = BF.matrix_fast(None, vm, tables, codes=FL.protospacer_codes(codes, l0, plus))
                Ma, _ = FL.build_matrix(l0, sg, codes, tables)
                X = np.hstack([Mb, Ma.astype(np.float32)])
                return [f(X) for f in fns]
            run()
            ms = _time(run, _reps(10 if n < 5_000 else (3 if n < 50_000 else 1)))
            rows.append(dict(model=lab, width_nt=width, n_guides=n, ms_per_call=ms, ms_per_guide=ms / n))
            print(f"  {lab:26s} {width:>9,} nt, {n:6,} sites: {ms:9.2f} ms  ({ms / n:.4f} ms/guide)", flush=True)
    return rows


# ---------------------------------------------------------------------------------------------------------------- the first benchmark: one fit, then latency, throughput, cold start
def inference_side(n_score=(1, 100, 10_000), reps: int = 5, seed: int = 41, verbose: bool = True):
    """Fit the champion once; time scoring 1, 100 and 10,000 guides. A boosted ensemble's forward cost is one root-to-leaf walk per tree (n_trees x depth comparisons), not the same unit of work as a dense network's FLOPs."""
    import xgboost as xgb

    from .. import config
    from ..harness.curated import load_base, load_block
    from ..harness.run_ablation import _impute, champion_params, make_selector

    X, y, names, protos = load_base()
    A, _ = load_block("a_flank", protos)
    X = np.hstack([X, A]).astype(np.float32)

    tr = np.arange(len(X))
    Xi, _ = _impute(X[tr], X[:1])
    sel = make_selector(seed)
    sel.fit(Xi, y)
    top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
    m = xgb.XGBRegressor(**champion_params(seed))
    t0 = time.perf_counter()
    m.fit(Xi[:, top], y)
    fit_s = time.perf_counter() - t0
    booster = m.get_booster()
    n_trees = len(booster.get_dump())

    rows = []
    for n in (n_score[:2] if QUICK else n_score):
        batch = np.ascontiguousarray(Xi[:n, top])
        m.predict(batch)                       # warm the cache
        ts = []
        for _ in range(_reps(reps)):
            t0 = time.perf_counter()
            m.predict(batch)
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        rows.append(dict(model="SLICER champion", n_guides=n, n_columns=len(top), n_trees=n_trees, latency_s=med, guides_per_s=n / med if med else np.nan,
                         fit_seconds=fit_s, cold_start_s=np.nan))
        if n in HAL_INFERENCE:
            h = HAL_INFERENCE[n]
            rows.append(dict(model="crisprHAL 2, shipped WT-SpCas9", n_guides=n, n_columns=378, n_trees=np.nan, latency_s=h["latency_s"], guides_per_s=h["guides_per_s"],
                             fit_seconds=np.nan, cold_start_s=HAL_INFERENCE["import_and_load_s"]))
        if verbose:
            print(f"  score {n:>7,} guides: {med * 1e3:9.2f} ms  {n / med:12,.0f} guides/s", flush=True)
    out = pd.DataFrame(rows)
    if verbose:
        ours = out[out.model.str.startswith("SLICER")].iloc[-1]
        theirs = out[out.model.str.startswith("crisprHAL")]
        print(f"\n  one fit took {fit_s:.1f} s; scoring runs at {ours.guides_per_s:,.0f} guides/s")
        if len(theirs):
            t = theirs.iloc[-1]
            print(f"  against {t.guides_per_s:,.0f} guides/s for crisprHAL 2 on the same CPU, a factor of {ours.guides_per_s / t.guides_per_s:,.0f} on throughput")
            print(f"  and {t.cold_start_s:.1f} s of cold start before its first prediction, against a fraction of a second for ours")
        print("  these numbers move with machine load; the gap does not")
        print(f"  comparison of work per guide: {n_trees} trees x depth 4 = {n_trees * 4:,} comparisons against a dense network's forward pass")
    return out


# ---------------------------------------------------------------------------------------------------------------- crisprHAL 2 (run as a script under the TensorFlow interpreter)
def hal_model_side(root):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    os.chdir(os.path.join(root, "external_data", "crisprHAL"))
    model = keras.models.load_model("models/WT-SPCAS9.keras", compile=False)
    rows = []
    for b in _batches():
        X = np.random.default_rng(0).integers(0, 2, (b, 378, 4)).astype(np.float32)
        model.predict(X[:1], verbose=0)
        reps = _reps(30 if b == 1 else (8 if b == 100 else 2))
        ms = _time(lambda: model.predict(X, batch_size=min(b, 256), verbose=0), reps)
        rows.append(dict(model="crisprHAL 2 (shipped WT-SPCAS9, TensorFlow, one thread)", batch=b, latency_ms=ms, per_guide_ms=ms / b, n_members=1, total_trees=0))
        print(rows[-1], flush=True)
    return rows


def hal_e2e_side(root):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    hal = os.path.join(root, "external_data", "crisprHAL")
    sys.path.insert(0, hal); os.chdir(hal)
    from processing import processing
    proc = processing()
    model = keras.models.load_model("models/WT-SPCAS9.keras", compile=False)
    rng = np.random.default_rng(0)
    rows = []
    for b in _batches():
        ctx = ["".join(rng.choice(list("ACGT"), 378)) for _ in range(b)]
        model.predict(proc.onehotencode(ctx[:1]), verbose=0)
        enc = _time(lambda: proc.onehotencode(ctx), _reps(30 if b == 1 else (5 if b == 100 else 1)))
        X = proc.onehotencode(ctx)
        mod = _time(lambda: model.predict(X, batch_size=min(b, 256), verbose=0), _reps(30 if b == 1 else (8 if b == 100 else 2)))
        rows.append(dict(model="crisprHAL 2 (encode + model)", batch=b, ms_per_call=enc + mod, ms_per_guide=(enc + mod) / b, encode_ms=enc, model_ms=mod))
        print(rows[-1], flush=True)
    return rows


PARTS = ("model", "featurise", "e2e", "locus", "inference", "hal-model", "hal-e2e")


def main(argv=None) -> int:
    global QUICK
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--parts", nargs="+", choices=PARTS, required=True)
    ap.add_argument("--members", default="", help="the finalists of the mixed ensemble, comma-separated (model, e2e, locus)")
    ap.add_argument("--hal", default="", help="the project root, for the hal-* parts (TensorFlow environment)")
    ap.add_argument("--reps", type=int, default=5, help="repeats of the inference part")
    ap.add_argument("--quick", action="store_true", help="one repeat, batches 1 and 100, narrower loci: a smoke test that writes the same columns")
    a = ap.parse_args(argv)
    QUICK = a.quick
    mix = [m for m in a.members.split(",") if m]
    sets = {"original champion": ["champion"], "tuned XGBoost (d099x)": ["d099x"], "XGBoost average of 5": ["d099x", "d077x", "d021x", "d080x", "d000x"]}
    if mix:
        sets["mixed ensemble"] = mix
    for part in a.parts:
        if part.startswith("hal-"):
            if not a.hal:
                ap.error(f"--hal <project root> is needed for {part}")
            name, rows = (("cost_crisprhal.csv", hal_model_side(a.hal)) if part == "hal-model" else ("e2e_crisprhal.csv", hal_e2e_side(a.hal)))
            pd.DataFrame(rows).to_csv(os.path.join(a.hal, "RSI-Crispr-fork", "Runjia", "results", name), index=False)
            continue
        from ..common import OUT          # the package; the hal-* parts above run without it
        if part == "model":
            pd.DataFrame(model_side(mix)).to_csv(OUT / "cost_slicer.csv", index=False)
        elif part == "featurise":
            pd.DataFrame(featurise_side()).to_csv(OUT / "cost_featurise.csv", index=False)
        elif part == "e2e":
            pd.DataFrame(e2e_side(sets)).to_csv(OUT / "e2e_slicer.csv", index=False)
        elif part == "locus":
            pd.DataFrame(locus_side(sets)).to_csv(OUT / "e2e_slicer_locus.csv", index=False)
        elif part == "inference":
            inference_side(reps=a.reps).to_csv(OUT / "inference_cost.csv", index=False)
    return 0


if __name__ == "__main__":
    if __package__ in (None, ""):             # run as a script under the TensorFlow interpreter: no package, so no relative imports
        sys.path.insert(0, os.getcwd())
    raise SystemExit(main())
