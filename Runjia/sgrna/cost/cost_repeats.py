"""Cost with its spread: every timing repeated, every repeat kept, so a ratio comes with a distribution and not a bare range.

The earlier cost tables quote medians of very few repeats (three at 10,000 guides, one for crisprHAL 2), so a statement such as '13x to 21x faster' gave a range
that mixes the effect of batch size with measurement noise and cannot be read as an interval. Here each configuration is timed many times on an otherwise idle
machine, one host thread, and the raw timings are written out (`cost_repeats_inference.csv`, one row per repeat) for a violin plot; ratios are formed per
repeat pair and summarised by their median and 95% range. Fit times are repeated the same way (`cost_repeats_fit.csv`): SLICER's default model (300 columns
selected from 6,517 by a 300-tree XGBoost, then 400 trees) and the tuned model (4,000 trees, depth 5) on one training fold, process CPU-seconds and wall seconds,
each repeated; crisprHAL 2's fits are the stored ones (five folds on this CPU, `crisprhal_rerun.csv`; T4 fits of the width experiment), as one fit takes about
17 minutes of wall-clock on this machine.

    OMP_NUM_THREADS=1 python -m sgrna.cost.cost_repeats inference          # SLICER, main environment
    .venv-crisprhal/bin/python -m sgrna.cost.cost_repeats hal              # crisprHAL 2, TensorFlow environment
    python -m sgrna.cost.cost_repeats fit                                   # fit times, main environment
    python -m sgrna.cost.cost_repeats summary
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    if sys.argv[1:2] in (["inference"], ["hal"]):
        os.environ[_v] = "1"

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
RUNJIA = HERE.parents[2]
ROOT = HERE.parents[3]
RESULTS = RUNJIA / "results" / "post_freeze"
INF = RESULTS / "cost_repeats_inference.csv"
FIT = RESULTS / "cost_repeats_fit.csv"
REPS = {1: 300, 100: 60, 10_000: 15}


def _timings(fn, reps):
    fn()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


def inference():
    sys.path.insert(0, str(RUNJIA))
    from sgrna import config
    from sgrna.build import a_flank_fast as AF
    from sgrna.build import base_fast as BF
    from sgrna.build import featurise, qct
    from sgrna.cost.e2e_bench import _make_models

    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv")
    idx["located"] = idx["located"].astype(bool)
    tables, vm = qct.load_tables(), featurise.load_v_mapping()
    models = _make_models({"SLICER default": ["champion"], "SLICER tuned": ["d099x"]})
    rows = []
    for lab, fns in models.items():
        for b, reps in REPS.items():
            sub = idx.iloc[:b].copy()

            def run():
                Mb, _ = BF.matrix_fast(sub["protospacer"].tolist(), vm, tables)
                Ma, _ = AF.build_matrix(sub, tables)
                X = np.hstack([Mb, Ma.astype(np.float32)])
                return [f(X) for f in fns]
            for i, t in enumerate(_timings(run, reps)):
                rows.append(dict(model=lab, batch=b, rep=i, ms_per_guide=1000 * t / b))
            print(f"  {lab:16s} batch {b:6d}: median {np.median([r['ms_per_guide'] for r in rows if r['model'] == lab and r['batch'] == b]):.4f} ms/guide", flush=True)
    old = pd.read_csv(INF) if INF.exists() else pd.DataFrame()
    pd.concat([old[~old.model.isin(models)] if len(old) else old, pd.DataFrame(rows)]).to_csv(INF, index=False)


def hal():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import keras
    hal_dir = ROOT / "external_data" / "crisprHAL"
    sys.path.insert(0, str(hal_dir))
    cwd = os.getcwd()
    os.chdir(hal_dir)
    from processing import processing
    proc = processing()
    model = keras.models.load_model("models/WT-SPCAS9.keras", compile=False)
    os.chdir(cwd)
    rng = np.random.default_rng(0)
    rows = []
    for b, reps in {1: 200, 100: 40, 10_000: 8}.items():
        ctx = ["".join(rng.choice(list("ACGT"), 378)) for _ in range(b)]

        def run():
            X = proc.onehotencode(ctx)
            return model.predict(X, batch_size=min(b, 256), verbose=0)
        for i, t in enumerate(_timings(run, reps)):
            rows.append(dict(model="crisprHAL 2", batch=b, rep=i, ms_per_guide=1000 * t / b))
        print(f"  crisprHAL 2 batch {b:6d}: median {np.median([r['ms_per_guide'] for r in rows if r['batch'] == b]):.4f} ms/guide", flush=True)
    old = pd.read_csv(INF) if INF.exists() else pd.DataFrame()
    pd.concat([old[old.model != "crisprHAL 2"] if len(old) else old, pd.DataFrame(rows)]).to_csv(INF, index=False)


def fit():
    sys.path.insert(0, str(RUNJIA))
    import resource

    from sgrna.cohorts.curated_ablation import _base, _block
    from sgrna.compare.vs_crisprhal import member_specs, train_predict
    from sklearn.model_selection import KFold

    X, y, names, protos = _base()
    A, _ = _block("a_flank", protos)
    X = np.hstack([X, A]).astype(np.float32)
    tr, va = next(iter(KFold(5, shuffle=True, random_state=41).split(np.arange(len(y)))))
    rows = []
    for lab, member, reps in (("SLICER default", "champion", 10), ("SLICER tuned", "d099x", 5)):
        specs = member_specs([member])
        for i in range(reps):
            c0, w0 = time.process_time(), time.perf_counter()
            r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
            train_predict(specs, X[tr], y[tr], X[va][:10], seed=41 + i)
            cpu = time.process_time() - c0
            rows.append(dict(model=lab, rep=i, cpu_seconds=cpu, wall_seconds=time.perf_counter() - w0, n_train=len(tr)))
            print(f"  {lab} fit {i + 1}/{reps}: {cpu:.0f} CPU-s, {rows[-1]['wall_seconds']:.0f} s wall", flush=True)
            pd.DataFrame(rows).to_csv(FIT, index=False)


def summary():
    d = pd.read_csv(INF)
    g = d.groupby(["model", "batch"]).ms_per_guide
    s = g.agg(["count", "median", "mean", "std", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
    s.columns = ["reps", "median", "mean", "sd", "q025", "q975"]
    print(s.round(4).to_string())
    h = d[d.model == "crisprHAL 2"].set_index(["batch", "rep"]).ms_per_guide
    for lab in ("SLICER default", "SLICER tuned"):
        for b in REPS:
            sl = d[(d.model == lab) & (d.batch == b)].ms_per_guide.to_numpy()
            hh = h.loc[b].to_numpy()
            rng = np.random.default_rng(0)
            r = rng.choice(hh, 4000) / rng.choice(sl, 4000)
            print(f"  crisprHAL 2 / {lab} at batch {b:>6}: median {np.median(r):7.1f}x  95% range {np.percentile(r, 2.5):7.1f}x to {np.percentile(r, 97.5):7.1f}x")


if __name__ == "__main__":
    {"inference": inference, "hal": hal, "fit": fit, "summary": summary}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
