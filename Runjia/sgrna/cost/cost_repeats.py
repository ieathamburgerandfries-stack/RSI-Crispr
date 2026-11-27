"""Cost with its spread: every timing repeated, every repeat kept, so a ratio comes with a distribution and not a bare range.

The earlier cost tables quote medians of very few repeats (three at 10,000 guides, one for crisprHAL 2), so a statement such as '13x to 21x faster' gave a range
that mixes the effect of batch size with measurement noise and cannot be read as an interval. Here each configuration is timed many times on an otherwise idle
machine, one host thread, and the raw timings are written out (`cost_repeats_inference.csv`, one row per repeat) for a violin plot; ratios are formed per
repeat pair and summarised by their median and 95% range. Fit times are repeated the same way (`cost_repeats_fit.csv`): SLICER's default model (300 columns
selected from 6,517 by a 300-tree XGBoost, then 400 trees) and the tuned model (4,000 trees, depth 5) on one training fold, process CPU-seconds and wall seconds,
each repeated; crisprHAL 2's published configuration is fitted three times on the same training fold (`hal_fit`, about 17 minutes of wall-clock each). The T4 fits of
the width experiment (`crisprhal_wide_window_folds.csv`) are stored results, not repeated here; BoostMEC is timed by `cost_boostmec.py` into the same two files.

Quiet means that nothing of ours runs at the same time (the commands are chained one after another by `cost_chain.sh`) and that the machine's one-minute load
average is written next to every configuration (`load1_start`, `load1_end`; browsers and chat programs were open, nothing else). Each row carries the session it
was timed in, and `cost_repeats_machine.json` the processor and library versions; all rows of one table come from one session.

    OMP_NUM_THREADS=1 python -m sgrna.cost.cost_repeats inference          # SLICER, main environment
    .venv-crisprhal/bin/python -m sgrna.cost.cost_repeats hal              # crisprHAL 2, TensorFlow environment
    python -m sgrna.cost.cost_repeats fit                                   # fit times, main environment
    .venv-crisprhal/bin/python -m sgrna.cost.cost_repeats hal_fit          # crisprHAL 2 fit times on this CPU, TensorFlow environment
    OMP_NUM_THREADS=1 python -m sgrna.cost.cost_repeats featurisers         # the original featurisers against the array ones, alternating, one session
    python -m sgrna.cost.cost_repeats summary
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    if sys.argv[1:2] in (["inference"], ["hal"], ["featurisers"]):
        os.environ[_v] = "1"

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
RUNJIA = HERE.parents[2]
ROOT = HERE.parents[3]
RESULTS = RUNJIA / "results"
INF = RESULTS / "cost_repeats_inference.csv"
FIT = RESULTS / "cost_repeats_fit.csv"
REPS = {1: 300, 100: 60, 10_000: 15}


SESSION = time.strftime("%Y-%m-%d %H:%M")


def load1() -> float:
    return round(os.getloadavg()[0], 2)


def _timings(fn, reps):
    fn()
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


def machine(extra: dict | None = None) -> dict:
    """What the timings were taken on, appended to `cost_repeats_machine.json` (one entry per session and command)."""
    import platform
    import subprocess
    info = dict(session=SESSION, cpu=subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
                cores=os.cpu_count(), python=platform.python_version(), load1_at_start=load1(), argv=sys.argv[1:])
    for mod in ("numpy", "xgboost", "lightgbm", "tensorflow", "RNA"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    info.update(extra or {})
    path = RESULTS / "cost_repeats_machine.json"
    old = json.loads(path.read_text()) if path.exists() else []
    path.write_text(json.dumps(old + [info], indent=1))
    return info


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
    machine()
    rows = []
    for lab, fns in models.items():
        for b, reps in REPS.items():
            sub = idx.iloc[:b].copy()

            def run():
                Mb, _ = BF.matrix_fast(sub["protospacer"].tolist(), vm, tables)
                Ma, _ = AF.build_matrix(sub, tables)
                X = np.hstack([Mb, Ma.astype(np.float32)])
                return [f(X) for f in fns]
            l0 = load1()
            ts = _timings(run, reps)
            l1 = load1()
            for i, t in enumerate(ts):
                rows.append(dict(model=lab, batch=b, rep=i, ms_per_guide=1000 * t / b, session=SESSION, load1_start=l0, load1_end=l1))
            print(f"  {lab:16s} batch {b:6d}: median {np.median([r['ms_per_guide'] for r in rows if r['model'] == lab and r['batch'] == b]):.4f} ms/guide", flush=True)
    old = pd.read_csv(INF) if INF.exists() else pd.DataFrame()
    pd.concat([old[~old.model.isin(models)] if len(old) else old, pd.DataFrame(rows)]).to_csv(INF, index=False)


def featurisers():
    """The same SLICER default model scored through the original per-guide featurisers and through the array-based ones, alternating in one session.

    Both paths give identical columns (`controls/equiv_check.py`), so this isolates what the array rewrite is worth: raw sequence to score, one thread,
    `featurise.matrix` + `features.a_flank.build` against `base_fast.matrix_fast` + `a_flank_fast.build_matrix`, then the same column selection, imputation and
    400-tree model. Repeats alternate (original, array, original, ...) so that any drift of the machine hits both alike.
    """
    sys.path.insert(0, str(RUNJIA))
    from sgrna import config
    from sgrna.build import a_flank_fast as AF
    from sgrna.build import base_fast as BF
    from sgrna.build import featurise, qct
    from sgrna.cost.e2e_bench import _make_models
    from sgrna.features import a_flank as A

    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv")
    idx["located"] = idx["located"].astype(bool)
    tables, vm = qct.load_tables(), featurise.load_v_mapping()
    (fn,) = _make_models({"SLICER default": ["champion"]})["SLICER default"]
    machine()
    reps = {1: 100, 100: 20, 10_000: 3}
    rows = []
    for b, n in reps.items():
        sub = idx.iloc[:b].copy()

        def original():
            Mb, _ = featurise.matrix(sub["protospacer"].tolist(), v_mapping=vm, tables=tables, verbose=False)
            Ma = A.build(sub, tables=tables).drop(columns=config.ID_COL).to_numpy(np.float32)
            return fn(np.hstack([Mb.astype(np.float32), Ma]))

        def array():
            Mb, _ = BF.matrix_fast(sub["protospacer"].tolist(), vm, tables)
            Ma, _ = AF.build_matrix(sub, tables)
            return fn(np.hstack([Mb, Ma.astype(np.float32)]))

        assert np.allclose(original(), array(), atol=1e-5), "the two paths disagree"
        l0 = load1()
        for rep in range(n):
            for lab, f in (("original featurisers", original), ("array featurisers", array)):
                t = time.perf_counter()
                f()
                rows.append(dict(model="SLICER default, " + lab, batch=b, rep=rep, ms_per_guide=1000 * (time.perf_counter() - t) / b,
                                 session=SESSION, load1_start=l0, load1_end=load1()))
        for lab in ("original featurisers", "array featurisers"):
            print(f"  {lab:22s} batch {b:6d}: median {np.median([r['ms_per_guide'] for r in rows if r['batch'] == b and lab in r['model']]):.4f} ms/guide", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "cost_repeats_featurisers.csv", index=False)


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
    machine()
    rows = []
    for b, reps in {1: 200, 100: 40, 10_000: 8}.items():
        ctx = ["".join(rng.choice(list("ACGT"), 378)) for _ in range(b)]

        def run():
            X = proc.onehotencode(ctx)
            return model.predict(X, batch_size=min(b, 256), verbose=0)
        l0 = load1()
        ts = _timings(run, reps)
        l1 = load1()
        for i, t in enumerate(ts):
            rows.append(dict(model="crisprHAL 2", batch=b, rep=i, ms_per_guide=1000 * t / b, session=SESSION, load1_start=l0, load1_end=l1))
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
    machine()
    rows = []
    for lab, member, reps in (("SLICER default", "champion", 10), ("SLICER tuned", "d099x", 5)):
        specs = member_specs([member])
        for i in range(reps):
            l0 = load1()
            c0, w0 = time.process_time(), time.perf_counter()
            r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
            train_predict(specs, X[tr], y[tr], X[va][:10], seed=41 + i)
            cpu = time.process_time() - c0
            rows.append(dict(model=lab, rep=i, cpu_seconds=cpu, wall_seconds=time.perf_counter() - w0, n_train=len(tr), session=SESSION, load1_start=l0, load1_end=load1()))
            print(f"  {lab} fit {i + 1}/{reps}: {cpu:.0f} CPU-s, {rows[-1]['wall_seconds']:.0f} s wall", flush=True)
            save_fit(rows)


def save_fit(rows):
    """Write the fit timings of the models in `rows`, keeping the rows of the other models (SLICER, BoostMEC and crisprHAL 2 are timed by different commands)."""
    new = pd.DataFrame(rows)
    old = pd.read_csv(FIT) if FIT.exists() else pd.DataFrame()
    pd.concat([old[~old.model.isin(set(new.model))] if len(old) else old, new]).to_csv(FIT, index=False)


def hal_fit(reps: int = 3):
    """crisprHAL 2's published configuration (48 epochs, batch 1,024) fitted on this CPU on the training part of fold 1 of SLICER's cost folds (26,853 guides), timed on
    an otherwise idle machine; wall and process CPU seconds. Run from the TensorFlow environment; every thread TensorFlow chooses is used, as when one trains on a laptop.

        .venv-crisprhal/bin/python -m sgrna.cost.cost_repeats hal_fit
    """
    import json as _json

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from scipy.stats import spearmanr  # noqa: F401  (kept so a broken environment fails before the long fit)
    from sklearn.model_selection import KFold

    hal_dir = ROOT / "external_data" / "crisprHAL"
    sys.path.insert(0, str(hal_dir))
    cwd = os.getcwd()
    os.chdir(hal_dir)
    from models import models, modelVersionDefaultEpochs
    from processing import processing
    proc = processing()
    os.chdir(cwd)
    epochs = modelVersionDefaultEpochs["WT-SPCAS9"]
    meta = _json.loads((ROOT / "data" / "interim" / "expanded_meta.json").read_text())
    protos, y = np.asarray(meta["protospacers"]), np.asarray(meta["y"], dtype=float)
    up = 189

    def rd(split):
        d = pd.read_csv(hal_dir / "data" / f"WT-SpCas9_{split}ing_data.csv", header=None, names=["seq", "y"])
        d["protospacer"] = d["seq"].str[up:up + 20]
        return d
    hal = pd.concat([rd("train"), rd("test")]).drop_duplicates("protospacer")
    seqs = hal.set_index("protospacer")["seq"].reindex(protos).to_numpy()
    tr, _ = next(iter(KFold(5, shuffle=True, random_state=41).split(seqs)))
    X = proc.onehotencode(seqs[tr])
    machine(dict(tf_intra_threads=tf.config.threading.get_intra_op_parallelism_threads(), epochs=epochs, n_train=len(tr)))
    rows = []
    for i in range(reps):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(41 + i)
        m = models("WT-SPCAS9", summary=False)
        l0 = load1()
        c0, w0 = time.process_time(), time.perf_counter()
        m.train(X, y[tr], epochs=epochs, batch_size=1024, verbose=0)
        cpu, wall = time.process_time() - c0, time.perf_counter() - w0
        rows.append(dict(model="crisprHAL 2", rep=i, cpu_seconds=cpu, wall_seconds=wall, n_train=len(tr), session=SESSION, load1_start=l0, load1_end=load1()))
        print(f"  crisprHAL 2 fit {i + 1}/{reps}: {wall / 60:.1f} min wall, {cpu / 60:.1f} CPU-min, load {l0} to {load1()}", flush=True)
        save_fit(rows)


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
    {"inference": inference, "featurisers": featurisers, "hal": hal, "fit": fit, "hal_fit": hal_fit, "summary": summary}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
