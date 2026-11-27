"""End-to-end scoring cost per guide, one thread: featurisation + column selection + imputation + model, against crisprHAL 2 with its input encoding.

SLICER side: the array-based `base_fast` and `a_flank_fast` (identical to the original featurisers, see their `check()`), the selected
columns picked out of the 6,517, missing values filled with stored medians, then the model(s). Models are fitted on random data of
the real widths and tree counts so timings do not depend on what was learned (as in `cost_bench.py`). Every library runs on one
thread. The guide's flanking sequence is taken from the cohort index; fetching it from a genome is outside both models' cost.

crisprHAL 2 side (TensorFlow environment): `processing.onehotencode` on random 378-nt contexts, then the shipped model.

`--locus` scores the real design target instead: every NGG-flanked site in a stretch of chromosome, through the per-locus
precompute (`flank_locus`), which reads the protospacers and the flank tracks straight out of the genome array. That is the path
a design tool takes, and the one the cost comparison with crisprHAL 2 should be read against; the default table above is the
cost of re-scoring a sparse, already-extracted cohort, which is a different question.

    python -m sgrna.cost.e2e_bench --members d099x,...
    python -m sgrna.cost.e2e_bench --locus
    .venv-crisprhal/bin/python sgrna/cost/e2e_bench.py --hal <project root>

Report section: Part 5.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import time

import numpy as np
import pandas as pd

BATCHES = (1, 100, 10000)


def _time(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1000.0


def _make_models(sets):
    """One callable per model, fitted on random data of the real width and tree count, as in `cost_bench.py`."""
    import xgboost as xgb
    from ..tuning import mix_ecoli as M
    from ..tuning import tune_ecoli as T
    from ..common import OUT
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


def slicer_side(sets):
    from sgrna import config
    from sgrna.build import featurise, qct
    from ..build import a_flank_fast as AF
    from ..build import base_fast as BF
    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv"); idx["located"] = idx["located"].astype(bool)
    tables, vm = qct.load_tables(), featurise.load_v_mapping()
    models = _make_models(sets)

    rows = []
    for lab, fns in models.items():
        for b in BATCHES:
            sub = idx.iloc[:b].copy()

            def run():
                Mb, _ = BF.matrix_fast(sub["protospacer"].tolist(), vm, tables)
                Ma, _ = AF.build_matrix(sub, tables)
                X = np.hstack([Mb, Ma.astype(np.float32)])
                return [f(X) for f in fns]
            run()
            ms = _time(run, 200 if b == 1 else (20 if b == 100 else 3))
            rows.append(dict(model=lab, batch=b, ms_per_call=ms, ms_per_guide=ms / b))
            print(f"  {lab:26s} batch {b:6d}: {ms:9.2f} ms  ({ms / b:.4f} ms/guide)", flush=True)
    return rows


def locus_side(sets, widths=(1_000, 100_000, 1_000_000), start=1_000_000):
    """Score every NGG site in a stretch of chromosome, through the per-locus precompute: the design path, end to end."""
    from sgrna.build import featurise, qct
    from ..build import base_fast as BF
    from ..build import flank_locus as FL
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
            ms = _time(run, 10 if n < 5_000 else (3 if n < 50_000 else 1))
            rows.append(dict(model=lab, width_nt=width, n_guides=n, ms_per_call=ms, ms_per_guide=ms / n))
            print(f"  {lab:26s} {width:>9,} nt, {n:6,} sites: {ms:9.2f} ms  ({ms / n:.4f} ms/guide)", flush=True)
    return rows


def hal_side(root):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import sys
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
    for b in BATCHES:
        ctx = ["".join(rng.choice(list("ACGT"), 378)) for _ in range(b)]
        model.predict(proc.onehotencode(ctx[:1]), verbose=0)
        enc = _time(lambda: proc.onehotencode(ctx), 30 if b == 1 else (5 if b == 100 else 1))
        X = proc.onehotencode(ctx)
        mod = _time(lambda: model.predict(X, batch_size=min(b, 256), verbose=0), 30 if b == 1 else (8 if b == 100 else 2))
        rows.append(dict(model="crisprHAL 2 (encode + model)", batch=b, ms_per_call=enc + mod, ms_per_guide=(enc + mod) / b, encode_ms=enc, model_ms=mod))
        print(rows[-1], flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="")
    ap.add_argument("--hal", default="")
    ap.add_argument("--locus", action="store_true", help="the design path: every NGG site in a stretch of chromosome")
    a = ap.parse_args()
    if a.hal:
        out = os.path.join(a.hal, "RSI-Crispr-fork", "Runjia", "results", "e2e_crisprhal.csv")
        pd.DataFrame(hal_side(a.hal)).to_csv(out, index=False)
    else:
        import sys
        sys.path.insert(0, os.getcwd())
        from ..common import OUT
        mix = [m for m in a.members.split(",") if m]
        sets = {"original champion": ["champion"], "tuned XGBoost (d099x)": ["d099x"], "XGBoost average of 5": ["d099x", "d077x", "d021x", "d080x", "d000x"]}
        if mix:
            sets["mixed ensemble"] = mix
        if a.locus:
            pd.DataFrame(locus_side(sets)).to_csv(OUT / "e2e_slicer_locus.csv", index=False)
        else:
            pd.DataFrame(slicer_side(sets)).to_csv(OUT / "e2e_slicer.csv", index=False)
