"""crisprHAL 2 retrained by us with its own code reproduces its published hold-out (rho 0.694 against 0.695), and is then run on our rows and folds, searched over its own constructor's arguments, and fed wider input than the 378 nt it was designed for.

Every crisprHAL 2 that we train ourselves, in the TensorFlow environment (`.venv-crisprhal`: TensorFlow 2.19 on Python 3.12, their pinned versions, none of xgboost, lightgbm or catboost), with their model
class and encoder imported unchanged from `external_data/crisprHAL`. (Their shipped weights are not retrained: `vs_crisprhal.py` scores those.) Three runs, one `--run` each:

  rerun   their code and settings on our rows and folds, as a check that this code reproduces their hold-out and as the like-for-like comparison         crisprhal_rerun.csv
  tune    a hyperparameter search of their constructor's arguments, so that the head-to-head is not tuned-against-untuned                                    crisprhal_tuning.csv, crisprhal_tuned_folds.csv
  width   their configuration at wider input (23, 378, 1,023 and 2,023 nt), fold by fold                                                                      crisprhal_wide_window_f<fold>.csv, crisprhal_wide_window_folds.csv, crisprhal_wide_window_summary.csv

    .venv-crisprhal/bin/python -m sgrna.compare.crisprhal_runs --run rerun --arm their_split        # or our_folds
    .venv-crisprhal/bin/python -m sgrna.compare.crisprhal_runs --run tune --stage both               # or search, final, summary
    .venv-crisprhal/bin/python -m sgrna.compare.crisprhal_runs --run width --fold 2                  # one fold, all four widths; --arms native_378 for one; --check for the data check only
    python -m sgrna.compare.crisprhal_runs --run width --assemble                                    # joins every fold's file and compares with SLICER

------------------------------------------------------------------------------------------------------------------------------
rerun
------------------------------------------------------------------------------------------------------------------------------
`headtohead.py` compares our LightGBM against crisprHAL 2's *published*
Spearman (0.697). A published number is a weaker comparison than a model run
side by side, so this trains their WT-SpCas9 model -- their architecture,
their hyper-parameters, their 48 epochs at batch 1024, imported from their own
code in `external_data/crisprHAL` -- on exactly the rows and folds the
`curated_*` arms use: the 33,567 deduplicated curated guides,
`KFold(5, shuffle=True, random_state=seed)` over the order in
`expanded_meta.json`. Their input is their own 378 nt context
(189 up, 20 nt target, 3 nt PAM, 166 down), looked up by protospacer.

Two arms:

    their_split   their shipped train/test split, as a check that the code
                  reproduces their published hold-out (0.695) here
    our_folds     the like-for-like comparison against `curated_flank`

Resumable: one row per (arm, seed, fold) in `results/crisprhal_rerun.csv`.

------------------------------------------------------------------------------------------------------------------------------
tune
------------------------------------------------------------------------------------------------------------------------------
Why this exists. The head-to-head margin is +0.0107 in SLICER's favour, and
tuning SLICER alone is worth +0.0103 -- the whole margin. So the comparison as it
stands is tuned-against-untuned in our favour, and the report refuses to call it
a win for that reason. The fix is to search their settings too and then say
whichever thing is true.

Design, and where it is deliberately unequal. Their model costs ~17 min of CPU
per fit here, so a nested search matching SLICER's (12 draws inside each of 5
outer folds = 60 fits) would be about 17 hours. Instead:

  stage 1   12 random draws, evaluated on ONE inner split carved out of outer
            fold 1's training data (80/20). Nothing here ever sees an outer
            validation fold.
  stage 2   the winning configuration re-fitted on all 5 outer folds, which is
            the number comparable with `--run rerun --arm our_folds`.

That is a weaker search than SLICER received, in two ways worth stating in the
paper: fewer effective evaluations, and a single inner split rather than one per
fold. **If their tuned score closes the gap, parity is the honest conclusion and
the weakness of the search only strengthens it.** If it does not close the gap,
the asymmetry is a live caveat and must be quoted alongside the result.

The search space is their own constructor's arguments -- learning rate, CNN
width and window, recurrent width, the three dropout rates, dense widths -- plus
epochs and batch size from `train()`. Their published defaults are included as
draw 0, so the search cannot do worse than their paper configuration.

------------------------------------------------------------------------------------------------------------------------------
width
------------------------------------------------------------------------------------------------------------------------------
crisprHAL 2 at wider input, on the folds the first run did not cover (local CPU; TensorFlow env `.venv-crisprhal`, not the main env).

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
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from .. import config
from ..common import OUT

HAL_DIR = config.EXTERNAL / "crisprHAL"
META_PATH = config.INTERIM / "expanded_meta.json"
RESULTS_PATH = OUT / "crisprhal_rerun.csv"                     # rerun
SEARCH_PATH = OUT / "crisprhal_tuning.csv"                     # tune
FINAL_PATH = OUT / "crisprhal_tuned_folds.csv"
INDEX = config.INTERIM / "curated_guide_index.csv"             # width
MODEL = "WT-SPCAS9"
UP = 189                                # protospacer starts here in their 378 nt
HALF, CORE, SEED = 1000, 23, 41
WIDTHS = {"core_23": (0, 0), "native_378": (189, 166), "mid_1023": (500, 500), "wide_2023": (1000, 1000)}
SLOTS = {"mid_1023": 3, "wide_2023": 2}      # at most this many processes train the wide arms at once (16 GB of memory)

# Their published configuration, from models.__init__ defaults plus the model-specific epoch count. Draw 0 of the search is exactly this, so the search is anchored.
DEFAULTS = dict(learning_rate=0.0005, drop_rate=0.3, CNN_filters=128,
                window_size=3, CNN_drop=0.3, CNN_dense1=128, CNN_dense2=64,
                RNN_size=128, RNN_dense1=128, RNN_dense2=64, CNN_RNN_drop=0.3)
DEFAULT_BATCH = 1024
SPACE = dict(
    learning_rate=[1e-4, 2.5e-4, 5e-4, 1e-3],
    drop_rate=[0.2, 0.3, 0.4],
    CNN_filters=[64, 128, 192],
    window_size=[3, 5],
    CNN_drop=[0.2, 0.3, 0.4],
    CNN_dense1=[64, 128, 256],
    CNN_dense2=[32, 64, 128],
    RNN_size=[64, 128, 192],
    RNN_dense1=[64, 128, 256],
    RNN_dense2=[32, 64, 128],
    CNN_RNN_drop=[0.2, 0.3, 0.4],
)
EPOCH_CHOICES = [32, 48, 64]
BATCH_CHOICES = [512, 1024]


# ------------------------------------------------------------------------------------------------------------------------
# their code and their data
# ------------------------------------------------------------------------------------------------------------------------

def _load_hal():
    """Their model class and encoder, imported from their repo unchanged."""
    sys.path.insert(0, str(HAL_DIR))
    cwd = os.getcwd()
    os.chdir(HAL_DIR)                   # their module paths are repo-relative
    try:
        from models import models, modelVersionDefaultEpochs
        from processing import processing
    finally:
        os.chdir(cwd)
    return models, modelVersionDefaultEpochs[MODEL], processing()


def _read(split: str) -> pd.DataFrame:
    d = pd.read_csv(HAL_DIR / "data" / f"WT-SpCas9_{split}ing_data.csv",
                    header=None, names=["seq", "y"])
    d["protospacer"] = d["seq"].str[UP:UP + 20]
    return d


def _arm_data(arm: str, seed: int, n_splits: int):
    """Yield (fold, seqs_tr, y_tr, seqs_va, y_va)."""
    if arm == "their_split":
        tr, te = _read("train"), _read("test")
        yield 1, tr["seq"].to_numpy(), tr["y"].to_numpy(), te["seq"].to_numpy(), te["y"].to_numpy()
        return

    meta = json.loads(META_PATH.read_text())
    protos = np.asarray(meta["protospacers"])
    y = np.asarray(meta["y"], dtype=float)
    hal = pd.concat([_read("train"), _read("test")]).drop_duplicates("protospacer")
    seqs = hal.set_index("protospacer")["seq"].reindex(protos)
    if seqs.isna().any():
        raise SystemExit(f"{seqs.isna().sum()} curated guides have no crisprHAL context")
    seqs = seqs.to_numpy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(kf.split(seqs), start=1):
        yield fold, seqs[tr], y[tr], seqs[va], y[va]

# ------------------------------------------------------------------------------------------------------------------------
# rerun: their settings on our rows and folds
# ------------------------------------------------------------------------------------------------------------------------

def rerun(arm: str, seed: int = 41, n_splits: int = 5, epochs: int | None = None,
        verbose: bool = True) -> pd.DataFrame:
    import tensorflow as tf

    models, default_epochs, proc = _load_hal()
    epochs = epochs or default_epochs

    prev = pd.read_csv(RESULTS_PATH) if RESULTS_PATH.exists() else pd.DataFrame()
    done = (set(zip(prev["arm"], prev["seed"], prev["fold"])) if len(prev) else set())

    rows = []
    for fold, s_tr, y_tr, s_va, y_va in _arm_data(arm, seed, n_splits):
        if (arm, seed, fold) in done:
            if verbose:
                print(f"  {arm} fold {fold}: already done")
            continue
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed + fold)
        X_tr, X_va = proc.onehotencode(s_tr), proc.onehotencode(s_va)
        m = models(MODEL, summary=False)
        t0 = time.time()
        m.train(X_tr, y_tr, epochs=epochs, batch_size=1024, verbose=2 if verbose else 0)
        pred = np.asarray(m.predict(X_va, verbose=0)).ravel()
        if arm != "their_split":
            pd.DataFrame({"y": y_va, "pred": pred, "fold": fold}).to_csv(
                config.RESULTS / f"preds_crisprhal_default_fold{fold}.csv",
                index=False)
        row = dict(arm=arm, model="crisprHAL2_WT-SpCas9", seed=seed, fold=fold,
                   epochs=epochs, n_train=len(y_tr), n_val=len(y_va),
                   spearman=float(spearmanr(y_va, pred).statistic),
                   r2=float(r2_score(y_va, pred)),
                   fit_seconds=round(time.time() - t0, 1))
        rows.append(row)
        prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        prev.to_csv(RESULTS_PATH, index=False)
        if verbose:
            print(f"  {arm} fold {fold}  rho {row['spearman']:.4f}  "
                  f"({row['fit_seconds']:.0f} s)", flush=True)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------------------------------------------------------------
# tune: a search of their constructor's arguments
# ------------------------------------------------------------------------------------------------------------------------

def draws(n: int, seed: int = 41) -> list[dict]:
    """Draw 0 is their published configuration; the rest are random."""
    rng = np.random.default_rng(seed)
    out = [dict(DEFAULTS, epochs=None, batch_size=DEFAULT_BATCH)]
    for _ in range(n - 1):
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        cfg["epochs"] = int(EPOCH_CHOICES[int(rng.integers(len(EPOCH_CHOICES)))])
        cfg["batch_size"] = int(BATCH_CHOICES[int(rng.integers(len(BATCH_CHOICES)))])
        out.append(cfg)
    return out


def _curated():
    """The 33,567 curated guides in `expanded_meta.json` order, with their context."""
    meta = json.loads(META_PATH.read_text())
    protos = np.asarray(meta["protospacers"])
    y = np.asarray(meta["y"], dtype=float)
    hal = pd.concat([_read("train"), _read("test")]).drop_duplicates("protospacer")
    seqs = hal.set_index("protospacer")["seq"].reindex(protos)
    if seqs.isna().any():
        raise SystemExit(f"{seqs.isna().sum()} curated guides lack crisprHAL context")
    return seqs.to_numpy(), y


def _fit_eval(models, proc, cfg, s_tr, y_tr, s_va, y_va, seed, default_epochs,
              verbose=True, save_preds=None):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    kw = {k: v for k, v in cfg.items() if k not in ("epochs", "batch_size")}
    epochs = cfg["epochs"] or default_epochs
    m = models(MODEL, summary=False, **kw)
    t0 = time.time()
    m.train(proc.onehotencode(s_tr), y_tr, epochs=epochs,
            batch_size=cfg["batch_size"], verbose=0)
    pred = np.asarray(m.predict(proc.onehotencode(s_va), verbose=0)).ravel()
    if save_preds:
        pd.DataFrame({"y": y_va, "pred": pred}).to_csv(save_preds, index=False)
    return dict(spearman=float(spearmanr(y_va, pred).statistic),
                r2=float(r2_score(y_va, pred)),
                epochs=epochs,
                fit_seconds=round(time.time() - t0, 1))


def search(n_draws: int = 12, seed: int = 41, n_splits: int = 5,
           patience: int | None = None, deadline_hours: float | None = None,
           verbose: bool = True):
    """Stage 1: score each draw on one inner split of outer fold 1's training set.

    `n_draws` is a ceiling. With `patience` the search stops once the best inner
    score has not improved for that many consecutive draws, so "searched until
    it stopped improving" is a statement about the recorded curve rather than
    about running out of patience -- every draw is already one row of
    `crisprhal_tuning.csv`, which is the trace.
    """
    models, default_epochs, proc = _load_hal()
    seqs, y = _curated()

    outer = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, _ = next(iter(outer.split(seqs)))
    inner = KFold(n_splits=5, shuffle=True, random_state=seed + 1)
    i_tr, i_va = next(iter(inner.split(tr_idx)))
    s_tr, y_tr = seqs[tr_idx[i_tr]], y[tr_idx[i_tr]]
    s_va, y_va = seqs[tr_idx[i_va]], y[tr_idx[i_va]]
    if verbose:
        print(f"  inner split: {len(y_tr):,} train / {len(y_va):,} validation "
              f"(both inside outer fold 1's training rows)", flush=True)

    prev = pd.read_csv(SEARCH_PATH) if SEARCH_PATH.exists() else pd.DataFrame()
    done = set(prev["draw"]) if len(prev) else set()
    stop = time.time() + deadline_hours * 3600 if deadline_hours else None

    best = float(prev["spearman"].max()) if len(prev) else -np.inf
    since_improved = 0
    for i, cfg in enumerate(draws(n_draws, seed)):
        if i in done:
            if verbose:
                print(f"  draw {i}: already done", flush=True)
            continue
        if stop and time.time() > stop:
            print("  deadline reached; re-run to continue", flush=True)
            break
        if patience and since_improved >= patience:
            print(f"  no improvement in {patience} draws; search stopped at "
                  f"draw {i}", flush=True)
            break
        res = _fit_eval(models, proc, cfg, s_tr, y_tr, s_va, y_va,
                        seed=seed, default_epochs=default_epochs,
                        verbose=verbose)
        # `cfg` carries the requested epochs (possibly None) and `res` the
        # resolved one, so they must not be splatted into one dict literal.
        row = {"draw": i, "is_published_default": (i == 0)}
        row.update(cfg)
        row.update(res)
        prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        prev.to_csv(SEARCH_PATH, index=False)
        if res["spearman"] > best:
            best, since_improved = res["spearman"], 0
        else:
            since_improved += 1
        if verbose:
            tag = " (their published default)" if i == 0 else ""
            print(f"  draw {i:2d}{tag}: inner rho {res['spearman']:.4f}  "
                  f"best {best:.4f}  (+{since_improved} since improved, "
                  f"{res['fit_seconds'] / 60:.1f} min)", flush=True)
    return prev


def best_config() -> dict:
    d = pd.read_csv(SEARCH_PATH)
    row = d.loc[d["spearman"].idxmax()]
    keys = list(SPACE) + ["epochs", "batch_size"]
    cfg = {k: row[k] for k in keys}
    for k in ("CNN_filters", "window_size", "CNN_dense1", "CNN_dense2",
              "RNN_size", "RNN_dense1", "RNN_dense2", "epochs", "batch_size"):
        cfg[k] = int(cfg[k])
    return cfg, int(row["draw"]), float(row["spearman"])


def final(seed: int = 41, n_splits: int = 5, deadline_hours: float | None = None,
          verbose: bool = True):
    """Stage 2: the winning configuration on all 5 outer folds."""
    cfg, draw, inner_rho = best_config()
    if verbose:
        print(f"  winning draw {draw} (inner rho {inner_rho:.4f}): {cfg}",
              flush=True)
    models, default_epochs, proc = _load_hal()
    seqs, y = _curated()

    prev = pd.read_csv(FINAL_PATH) if FINAL_PATH.exists() else pd.DataFrame()
    done = set(prev["fold"]) if len(prev) else set()
    stop = time.time() + deadline_hours * 3600 if deadline_hours else None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(kf.split(seqs), start=1):
        if fold in done:
            continue
        if stop and time.time() > stop:
            print("  deadline reached; re-run to continue", flush=True)
            break
        res = _fit_eval(models, proc, cfg, seqs[tr], y[tr], seqs[va], y[va],
                        seed=seed + fold, default_epochs=default_epochs,
                        verbose=verbose,
                        save_preds=config.RESULTS /
                        f"preds_crisprhal_tuned_fold{fold}.csv")
        row = {"model": "crisprHAL2_WT-SpCas9_tuned", "draw": draw,
               "fold": fold, "seed": seed, "n_train": len(tr),
               "n_val": len(va)}
        row.update(cfg)
        row.update(res)
        prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        prev.to_csv(FINAL_PATH, index=False)
        if verbose:
            print(f"  fold {fold}: rho {res['spearman']:.4f} "
                  f"({res['fit_seconds'] / 60:.1f} min)", flush=True)
    return prev


def summary():
    if SEARCH_PATH.exists():
        d = pd.read_csv(SEARCH_PATH).sort_values("spearman", ascending=False)
        print("\n== search, best first ==")
        print(d[["draw", "is_published_default", "learning_rate", "CNN_filters",
                 "RNN_size", "epochs", "batch_size", "spearman",
                 "fit_seconds"]].round(4).to_string(index=False))
    if FINAL_PATH.exists():
        f = pd.read_csv(FINAL_PATH)
        print("\n== tuned, 5 outer folds ==")
        print(f[["fold", "spearman", "r2", "fit_seconds"]].round(4)
              .to_string(index=False))
        print(f"\n  mean rho {f['spearman'].mean():.4f} "
              f"± {f['spearman'].std():.4f}")

# ------------------------------------------------------------------------------------------------------------------------
# width: their configuration at wider input
# ------------------------------------------------------------------------------------------------------------------------

class slot:
    """A counting semaphore across processes: hold one of N lock files while an arm trains."""
    def __init__(self, name: str):
        self.n = SLOTS.get(name, 0); self.name = name; self.fh = None

    def __enter__(self):
        if not self.n: return self
        import fcntl
        d = OUT / ".slots"; d.mkdir(exist_ok=True)
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
    first = pd.read_csv(OUT / "crisprhal_wide_window.csv")
    first = first[(first.model == "crisprHAL 2") & (first.batch == 256)].assign(note="Colab T4, first run")
    parts = [first[["model", "arm", "input_nt", "batch", "fold", "spearman", "fit_minutes", "note"]]]
    for f in sorted(OUT.glob("crisprhal_wide_window_f*.csv")):
        parts.append(pd.read_csv(f)[["model", "arm", "input_nt", "batch", "fold", "spearman", "fit_minutes", "note"]])
    allf = pd.concat(parts, ignore_index=True)
    # one value per arm and fold: prefer the local CPU run when both exist (it covers every fold with one code path)
    allf["local"] = allf.note.str.startswith("local")
    allf = allf.sort_values("local").drop_duplicates(["arm", "fold"], keep="last").sort_values(["input_nt", "fold"])
    allf.drop(columns="local").to_csv(OUT / "crisprhal_wide_window_folds.csv", index=False)
    hh = pd.read_csv(OUT / "headtohead.csv")
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
    out.to_csv(OUT / "crisprhal_wide_window_summary.csv", index=False)
    print(out.round(4).to_string(index=False))


def width(a) -> int:
    """One fold of the width experiment: every requested arm, one result file per fold (`a` holds fold, arms, batch, threads and check)."""
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
    out = OUT / f"crisprhal_wide_window_f{a.fold}.csv"
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

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", choices=["rerun", "tune", "width"], required=True)
    ap.add_argument("--arm", choices=["their_split", "our_folds"], help="rerun: which arm")
    ap.add_argument("--seed", type=int, default=41, help="rerun, tune")
    ap.add_argument("--epochs", type=int, default=None, help="rerun: default their model-specific value (48)")
    ap.add_argument("--stage", choices=["search", "final", "both", "summary"], default="both", help="tune")
    ap.add_argument("--draws", type=int, default=12, help="tune: ceiling on draws")
    ap.add_argument("--patience", type=int, default=None, help="tune: stop after this many draws with no improvement")
    ap.add_argument("--deadline-hours", type=float, default=None, help="tune")
    ap.add_argument("--fold", type=int, help="width: the fold (1-5)")
    ap.add_argument("--arms", default=",".join(WIDTHS), help="width: comma-separated arms")
    ap.add_argument("--batch", type=int, default=256, help="width")
    ap.add_argument("--threads", type=int, default=0, help="width: TensorFlow threads (0 = default)")
    ap.add_argument("--check", action="store_true", help="width: only compare the 378 nt slice with the released files, then stop")
    ap.add_argument("--assemble", action="store_true", help="width: join every fold's file and compare with SLICER")
    a = ap.parse_args(argv)
    if a.run == "rerun":
        if not a.arm:
            ap.error("--run rerun needs --arm")
        rerun(a.arm, seed=a.seed, epochs=a.epochs)
        d = pd.read_csv(RESULTS_PATH)
        print(d.groupby("arm")["spearman"].agg(["count", "mean", "std"]).round(4))
        return 0
    if a.run == "tune":
        if a.stage in ("search", "both"):
            search(n_draws=a.draws, seed=a.seed, patience=a.patience, deadline_hours=a.deadline_hours)
        if a.stage in ("final", "both"):
            final(seed=a.seed, deadline_hours=a.deadline_hours)
        summary()
        return 0
    if a.assemble:
        assemble()
        return 0
    if not a.fold:
        ap.error("--run width needs --fold (or --assemble)")
    return width(a)


if __name__ == "__main__":
    raise SystemExit(main())
