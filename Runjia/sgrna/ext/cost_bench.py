"""What the extra accuracy costs: inference latency, model size, and featurisation, on one host thread, same machine.

Model-only timings use random input of the right width (as in the earlier Colab benchmark), one thread for every library
(XGBoost `n_jobs=1`, LightGBM `n_jobs=1`, CatBoost `thread_count=1`, TensorFlow intra/inter-op threads 1). Models are fitted on
random data first so their tree counts and depths are the real ones; timings do not depend on what was learned. Featurisation
(the 6,517 columns, including the 1,000-nt flank windows) is timed separately because a prediction needs it.

    python -m sgrna.ext.cost_bench --members d099x,...          # SLICER side
    .venv-crisprhal/bin/python sgrna/ext/cost_bench.py --hal ... # crisprHAL 2 side
"""
from __future__ import annotations

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


def slicer_side(mix_names):
    import xgboost as xgb
    from . import mix_ecoli as M
    from . import tune_ecoli as T
    from .common import OUT
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
        t0 = time.time()
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
        for b in BATCHES:
            X = rng.standard_normal((b, 6517)).astype(np.float32)
            reps = 200 if b == 1 else (20 if b == 100 else 3)
            ms = _time(lambda: [f[0](X) for f in fns], reps)
            rows.append(dict(model=lab, batch=b, latency_ms=ms, per_guide_ms=ms / b, n_members=len(fns),
                             total_trees=int(sum(f[3] for f in fns if f[2] != "et"))))
            print(f"  {lab:26s} batch {b:6d}: {ms:9.2f} ms", flush=True)
    return rows


def featurise_side(n=300):
    """Warm, one-thread cost of the 6,517 columns per guide (tables preloaded, 5 guides first to warm caches)."""
    from sgrna import config, featurise, qct
    from sgrna.curated_ablation import _base
    from sgrna.features import a_flank
    _, _, _, protos = _base()
    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv").set_index("sgRNAID")
    tables, vm = qct.load_tables(), featurise.load_v_mapping()
    P = list(protos[:n]); sub = idx.reindex(P).reset_index().rename(columns={"index": "sgRNAID"}); sub["located"] = True
    featurise.matrix(P[:5], v_mapping=vm, tables=tables, verbose=False); a_flank.build(sub.iloc[:5], tables=tables)
    t = time.perf_counter(); featurise.matrix(P, v_mapping=vm, tables=tables, verbose=False); b = time.perf_counter() - t
    t = time.perf_counter(); a_flank.build(sub, tables=tables); a = time.perf_counter() - t
    return [dict(part="base (6,169 columns from the 20-mer)", n_guides=n, per_guide_ms=b / n * 1000),
            dict(part="family A (348 columns from the 1,000-nt flank windows)", n_guides=n, per_guide_ms=a / n * 1000),
            dict(part="total", n_guides=n, per_guide_ms=(a + b) / n * 1000)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="")
    ap.add_argument("--hal", default="", help="project root, to time crisprHAL 2 in the TensorFlow environment")
    a = ap.parse_args()
    if a.hal:
        import os
        import sys
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
        import keras
        root = a.hal
        os.chdir(os.path.join(root, "external_data", "crisprHAL"))
        model = keras.models.load_model("models/WT-SPCAS9.keras", compile=False)
        rows = []
        for b in BATCHES:
            X = np.random.default_rng(0).integers(0, 2, (b, 378, 4)).astype(np.float32)
            model.predict(X[:1], verbose=0)
            reps = 30 if b == 1 else (8 if b == 100 else 2)
            ms = _time(lambda: model.predict(X, batch_size=min(b, 256), verbose=0), reps)
            rows.append(dict(model="crisprHAL 2 (shipped WT-SPCAS9, TensorFlow, one thread)", batch=b, latency_ms=ms, per_guide_ms=ms / b,
                             n_members=1, total_trees=0)); print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(root, "RSI-Crispr-fork", "Runjia", "results", "ext", "cost_crisprhal.csv"), index=False)
    else:
        from .common import OUT
        if a.members:
            pd.DataFrame(slicer_side([m for m in a.members.split(",") if m])).to_csv(OUT / "cost_slicer.csv", index=False)
        else:
            pd.DataFrame(featurise_side()).to_csv(OUT / "cost_featurise.csv", index=False)
