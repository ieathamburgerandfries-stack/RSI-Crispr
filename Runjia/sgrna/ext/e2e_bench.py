"""End-to-end scoring cost per guide, one thread: featurisation + column selection + imputation + model, against crisprHAL 2 with its input encoding.

SLICER side: the array-based `base_fast` and `a_flank_fast` (identical to the original featurisers, see their `check()`), the selected
columns picked out of the 6,517, missing values filled with stored medians, then the model(s). Models are fitted on random data of
the real widths and tree counts so timings do not depend on what was learned (as in `cost_bench.py`). Every library runs on one
thread. The guide's flanking sequence is taken from the cohort index; fetching it from a genome is outside both models' cost.

crisprHAL 2 side (TensorFlow environment): `processing.onehotencode` on random 378-nt contexts, then the shipped model.

    python -m sgrna.ext.e2e_bench --members d099x,...
    .venv-crisprhal/bin/python sgrna/ext/e2e_bench.py --hal <project root>
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


def slicer_side(sets):
    import xgboost as xgb
    from sgrna import config, featurise, qct
    from . import a_flank_fast as AF
    from . import base_fast as BF
    from . import mix_ecoli as M
    from . import tune_ecoli as T
    from .common import OUT
    sel = T.selected()
    cfgs = {**json.loads((OUT / "tune_configs.json").read_text()), **M.all_cfgs()}
    fin = pd.read_csv(OUT / "mix_finalists.csv").set_index("name")
    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv"); idx["located"] = idx["located"].astype(bool)
    tables, vm = qct.load_tables(), featurise.load_v_mapping()
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

    rows = []
    for lab, names in sets.items():
        fns = [make(n) for n in names]
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
    a = ap.parse_args()
    if a.hal:
        out = os.path.join(a.hal, "RSI-Crispr-fork", "Runjia", "results", "ext", "e2e_crisprhal.csv")
        pd.DataFrame(hal_side(a.hal)).to_csv(out, index=False)
    else:
        import sys
        sys.path.insert(0, os.getcwd())
        from .common import OUT
        mix = [m for m in a.members.split(",") if m]
        sets = {"original champion": ["champion"], "tuned XGBoost (d099x)": ["d099x"], "XGBoost average of 5": ["d099x", "d077x", "d021x", "d080x", "d000x"]}
        if mix:
            sets["mixed ensemble"] = mix
        pd.DataFrame(slicer_side(sets)).to_csv(OUT / "e2e_slicer.csv", index=False)
