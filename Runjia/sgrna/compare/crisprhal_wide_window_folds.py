"""crisprHAL 2 at wider input, on the folds the first run did not cover (local CPU; TensorFlow env `.venv-crisprhal`, not the main env).

The width experiment of Finding 2 (`colab/crisprhal_wide_window.py`) was first run on one fold of one seed. On a CPU the 378 nt arm takes
about 75 minutes a fold and the 2,023 nt arm hours, so this file was used only for the 23 nt arm (folds 2-5) and the 378 nt arm (folds 2 and 3); the single Colab file
`colab/crisprhal_wide_window.py` (one paste, a T4, about three hours) runs the rest. This repeats it on the other
folds of the same split -- `KFold(5, shuffle=True, random_state=41)` over the 33,567 curated guides, the split the SLICER arms in
`results/headtohead.csv` use -- with the same code path: crisprHAL's own `models` class, its published configuration (epochs, dropout,
filters, GRU width as released, tuned at 378 nt), trained through Keras directly (which raises on failure) at batch 256, the batch the
2,023 nt arm needs. Nothing is re-tuned: the question stays "is the published configuration degraded by input wider than it was
designed for", and this makes the answer a five-fold one.

Input: the curated guide index holds 1,000 nt of flank each side; a guide's window is `upstream + protospacer + PAM + downstream`
(2,023 nt), and an arm of width `up + 23 + down` is the central slice, so the 378 nt arm is crisprHAL's own context (189 / 166). The
378 nt slice is compared with crisprHAL's released training and test files before anything is fitted.

    ../.venv-crisprhal/bin/python -m sgrna.compare.crisprhal_wide_window_folds --fold 2            # one fold, all four widths
    ../.venv-crisprhal/bin/python -m sgrna.compare.crisprhal_wide_window_folds --fold 1 --arms native_378    # hardware check on fold 1

Each fold is written to `results/crisprhal_wide_window_f<fold>.csv` as soon as each arm finishes, so a crash loses one arm. The
fold-1 table of the first run (a Colab T4) is unchanged; `python -m sgrna.compare.crisprhal_wide_window_folds assemble` joins all folds into `results/crisprhal_wide_window_folds.csv` and
`crisprhal_wide_window_summary.csv` (mean and sd over folds, the paired change from 378 nt, and the lead over SLICER on the same folds).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]                 # the project root that holds data/ and external_data/
RUNJIA = Path(__file__).resolve().parents[2]
RESULTS = RUNJIA / "results"
HAL_DIR = ROOT / "external_data" / "crisprHAL"
INDEX = ROOT / "data" / "interim" / "curated_guide_index.csv"
MODEL, HALF, CORE, SEED = "WT-SPCAS9", 1000, 23, 41
WIDTHS = {"core_23": (0, 0), "native_378": (189, 166), "mid_1023": (500, 500), "wide_2023": (1000, 1000)}


SLOTS = {"mid_1023": 3, "wide_2023": 2}      # at most this many processes train the wide arms at once (16 GB of memory)


class slot:
    """A counting semaphore across processes: hold one of N lock files while an arm trains."""
    def __init__(self, name: str):
        self.n = SLOTS.get(name, 0); self.name = name; self.fh = None

    def __enter__(self):
        if not self.n: return self
        import fcntl
        d = RESULTS / ".slots"; d.mkdir(exist_ok=True)
        while True:
            for i in range(self.n):
                fh = open(d / f"{self.name}_{i}.lock", "w")
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB); self.fh = fh; return self
                except OSError:
                    fh.close()
            time.sleep(30)

    def __exit__(self, *a):
        if self.fh: self.fh.close()


def windows():
    """2,023 nt window per guide and the label, in the curated index's row order."""
    g = pd.read_csv(INDEX)
    full = (g.upstream.str.upper() + g.protospacer.str.upper() + g.pam.str.upper() + g.downstream.str.upper()).to_numpy()
    assert all(len(s) == 2 * HALF + CORE for s in full[:100]), "window length"
    return full, g.score.to_numpy(dtype=float)


def check_against_released(full, y):
    """The 378 nt slice must equal crisprHAL's own context for the same guide, with the same label."""
    lo, hi = HALF - 189, HALF + CORE + 166
    ours = {s[lo:hi]: v for s, v in zip(full, y)}
    n = bad = 0
    for part in ("training", "testing"):
        d = pd.read_csv(HAL_DIR / "data" / f"WT-SpCas9_{part}_data.csv", header=None)
        for seq, val in zip(d[0], d[1]):
            n += 1
            if seq not in ours or abs(ours[seq] - float(val)) > 1e-6:
                bad += 1
    print(f"released WT-SpCas9 rows: {n:,}; not matched by our 378 nt slice with the same label: {bad:,}", flush=True)
    return n, bad


def assemble() -> None:
    """Join every fold into one table, and compare the widths fold by fold against SLICER on the same folds (`headtohead.csv`)."""
    first = pd.read_csv(RESULTS / "crisprhal_wide_window.csv")
    first = first[(first.model == "crisprHAL 2") & (first.batch == 256)].assign(note="Colab T4, first run")
    parts = [first[["model", "arm", "input_nt", "batch", "fold", "spearman", "fit_minutes", "note"]]]
    for f in sorted(RESULTS.glob("crisprhal_wide_window_f*.csv")):
        parts.append(pd.read_csv(f)[["model", "arm", "input_nt", "batch", "fold", "spearman", "fit_minutes", "note"]])
    allf = pd.concat(parts, ignore_index=True)
    # one value per arm and fold: prefer the local CPU run when both exist (it covers every fold with one code path)
    allf["local"] = allf.note.str.startswith("local")
    allf = allf.sort_values("local").drop_duplicates(["arm", "fold"], keep="last").sort_values(["input_nt", "fold"])
    allf.drop(columns="local").to_csv(RESULTS / "crisprhal_wide_window_folds.csv", index=False)
    hh = pd.read_csv(RESULTS / "headtohead.csv")
    hh = hh[hh.seed == SEED].pivot_table(index="fold", columns="arm", values="spearman")
    slicer = {"core_23": "curated_all", "native_378": "curated_matched", "wide_2023": "curated_flank"}
    hal = allf.pivot_table(index="fold", columns="arm", values="spearman")
    rows = []
    for arm in ["core_23", "native_378", "mid_1023", "wide_2023"]:
        if arm not in hal: continue
        h = hal[arm].dropna()
        r = dict(arm=arm, input_nt=int(allf[allf.arm == arm].input_nt.iloc[0]), n_folds=len(h), crisprhal_mean=h.mean(), crisprhal_sd=h.std(ddof=1))
        if arm in slicer:
            sl = hh[slicer[arm]].loc[h.index]
            d = (sl - h)
            r.update(slicer_mean=sl.mean(), slicer_sd=sl.std(ddof=1), lead_mean=d.mean(), lead_sd=d.std(ddof=1), folds_slicer_ahead=int((d > 0).sum()))
        if arm != "native_378" and "native_378" in hal:
            dd = (h - hal["native_378"].loc[h.index].dropna()).dropna()
            r.update(vs_378_mean=dd.mean(), vs_378_sd=dd.std(ddof=1), folds_below_378=int((dd < 0).sum()), n_paired=len(dd))
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "crisprhal_wide_window_summary.csv", index=False)
    print(out.round(4).to_string(index=False))


def main(argv=None) -> int:
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == "assemble":
        assemble(); return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--arms", default=",".join(WIDTHS))
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=0, help="TensorFlow threads (0 = default)")
    ap.add_argument("--check", action="store_true", help="only compare the 378 nt slice with the released files, then stop")
    a = ap.parse_args(argv)
    if a.threads:
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"; os.environ["TF_NUM_INTRAOP_THREADS"] = str(a.threads)
        os.environ["OMP_NUM_THREADS"] = str(a.threads)
    full, y = windows()
    n, bad = check_against_released(full, y)
    if a.check:
        return 0 if bad == 0 else 1
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    sys.path.insert(0, str(HAL_DIR)); cwd = os.getcwd(); os.chdir(HAL_DIR)
    try:
        import models as hal_models
        from models import models
        from processing import processing
    finally:
        os.chdir(cwd)
    import tensorflow as tf
    if a.threads:
        tf.config.threading.set_intra_op_parallelism_threads(a.threads); tf.config.threading.set_inter_op_parallelism_threads(1)
    proc = processing()
    epochs = hal_models.modelVersionDefaultEpochs[MODEL]
    tr, va = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(y))))[a.fold - 1]
    out = RESULTS / f"crisprhal_wide_window_f{a.fold}.csv"
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {(r["arm"], r["batch"]) for r in rows}
    for name in a.arms.split(","):
        if (name, a.batch) in done:
            print(f"  {name}: already recorded", flush=True); continue
        up, down = WIDTHS[name]; length = up + CORE + down
        seqs = np.array([s[HALF - up:HALF + CORE + down] for s in full])
        hal_models.modelVersionInputLength[MODEL][0] = length
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
        m = models(MODEL, summary=False)
        if getattr(m, "model", None) is None or m.model.input_shape[1] != length:
            raise RuntimeError(f"{name}: did not build at {length} nt")
        X = proc.onehotencode(seqs[tr]).astype("float32"); Xv = proc.onehotencode(seqs[va]).astype("float32")
        with slot(name):
            t0 = time.time()
            m.model.fit(X, y[tr], epochs=epochs, batch_size=a.batch, verbose=0)
            pred = np.asarray(m.model.predict(Xv, batch_size=a.batch, verbose=0)).ravel()
        if pred.shape[0] != len(va) or not np.isfinite(pred).all() or np.std(pred) == 0:
            raise RuntimeError(f"{name}: degenerate predictions")
        rho = float(spearmanr(y[va], pred).statistic)
        if rho < 0.30:
            raise RuntimeError(f"{name}: rho {rho:.4f} implies the fit did not take")
        rows.append(dict(model="crisprHAL 2", arm=name, input_nt=length, batch=a.batch, fold=a.fold, spearman=rho,
                         r2=float(r2_score(y[va], pred)), fit_minutes=round((time.time() - t0) / 60, 2),
                         n_train=len(tr), n_test=len(va), note="local CPU, " + (f"{a.threads} threads" if a.threads else "default threads")))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  fold {a.fold} {name:>10} ({length:>4} nt, batch {a.batch}): rho {rho:.4f}  {rows[-1]['fit_minutes']:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
