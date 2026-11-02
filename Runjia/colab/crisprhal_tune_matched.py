# ============================================================================
# Give crisprHAL 2 a search of its own: nested, 12 draws per fold with patience.
#   -- run in Google Colab on a GPU runtime (Runtime -> Change runtime type -> T4)
#
# RAN 16 OCTOBER. Result: 0.6987 mean over the five folds (0.6890, 0.7011,
# 0.7044, 0.6993, 0.6997) against their published configuration's 0.6971 and
# SLICER's 0.7218. Their published configuration WON the inner search in four
# folds of five; the fifth picked a draw 0.0016 ahead on the inner split, which
# is noise. 33 fits, ~2.6 GPU-hours, because patience stopped every fold after
# 5-8 draws.
#
# CORRECTION to the premise below: this is NOT the search SLICER got. SLICER's
# tuned arm used 97-157 draws per fold (627 total, results/tuning.csv); the
# "12 per fold" came from tune.py's CLI default rather than from the run that
# produced the number. Equalising honestly is ~100 draws per fold, ~500 fits,
# ~40 GPU-hours. What this run establishes is weaker but still useful: a small
# search finds nothing better than their defaults, while the same small search
# on our side buys +0.0103.
#
# WHY. The held-out-fold margin is SLICER 0.7213 against crisprHAL 0.7040,
# +0.0172 with a guide-paired 95% CI of [+0.0136, +0.0209]. That margin is an
# UPPER BOUND, and the report says so, because the two searches were not equal:
#
#   SLICER      97-157 random draws inside EACH of 5 outer folds, chosen on an
#               inner 80/20 split of that fold's training rows (627 total)
#   crisprHAL   10 draws on ONE inner split of fold 1's training rows, because
#               each fit cost ~17 CPU-minutes locally (7.3 h for the search)
#
# Neither search ever touched an outer validation fold, so neither number is
# leaked -- but theirs is the weaker search, and a reviewer will say so. This
# script removes the asymmetry.
#
# WHY NOW. Measured, not assumed: their published configuration fits one fold in
# 5.48 min on a Colab T4 against 17.0 min locally, a 3.1x speed-up
# (results/crisprhal_wide_window.csv, results/crisprhal_rerun.csv). That is what
# makes a matched nested search affordable at all.
#
# COST, from those measurements rather than from the report's earlier guess.
# 12 draws x 5 folds + 5 final refits = 65 fits. The draws average larger than
# their default configuration -- the local search's median fit was 44 min against
# 17 for the default -- so budget ~14 min/fit on a T4 and therefore
# **10-15 GPU-hours**, i.e. 2-4 free sessions with the resume below. The report
# previously estimated "2-4 GPU-h, one free Colab session"; that was computed
# before any Colab timing existed and it is too optimistic by 3-5x.
#
# INTERPRETATION, FIXED IN ADVANCE so the result cannot be read after the fact.
# Their own fully-tuned PUBLISHED WT model scores 0.695 on their own WT split,
# where SLICER scores 0.6997 at the same data scale (~26.8k training rows). So:
#
#   * tuned crisprHAL reaches >= ~0.716 on our folds  ->  the +0.0172 was mostly
#     a tuning-effort artefact; the honest conclusion on our folds is parity,
#     matching what their published weights already say on their splits.
#   * tuned crisprHAL lands ~0.704-0.710              ->  the margin survives a
#     matched search, and ~+0.005-0.010 is the number to quote (consistent with
#     the +0.005 we see against their published model on their own split).
#   * tuned crisprHAL exceeds 0.7213                  ->  we report that we lose
#     on our own folds once their search is equalised. Say it plainly.
#
# The positive control is draw 0: it is their published configuration, so fold
# scores for it must land near 0.686-0.702 (results/crisprhal_rerun.csv). If
# they do not, the harness is wrong and nothing else here can be trusted.
# ============================================================================

# --- cell 1: environment --------------------------------------------------
# Do NOT pin TensorFlow. Colab's Python is often too new for older releases and
# the install fails. Draw 0 certifies the environment instead.
#
# import tensorflow as tf; print("tensorflow", tf.__version__)

# --- cell 2: upload ---------------------------------------------------------
# Upload two things to /content:
#   crisprhal_widewindow_input.csv.gz   (33,567 rows: protospacer, y, seq, ok)
#   the crisprHAL repo, either cloned or uploaded as a folder named crisprHAL
#
#   !git clone --depth 1 https://github.com/tbrowne5/crisprHAL.git
#
# The input file holds +/-1,000 nt per guide; this script slices their 378 nt
# window out of it (189 up, 20 core, 3 PAM, 166 down), so no new export is
# needed and the rows stay in `expanded_meta.json` order -- which is what makes
# the folds identical to every other arm in the report.

# --- cell 3: the search -----------------------------------------------------
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

DATA = "/content/crisprhal_widewindow_input.csv.gz"
HAL = "/content/crisprHAL"
OUT = "/content/crisprhal_matched_folds.csv"     # resume file; download this
                                                 # into results/ under this name
SEED = 41
HALF, CORE = 1000, 23
UP, DOWN = 189, 166                              # their WT-SpCas9 window
TRIALS = 12                                      # per fold, matching SLICER
PATIENCE = 4                                     # stop a fold's search early
INNER_FRAC = 0.2

sys.path.insert(0, HAL)
os.chdir(HAL)
from models import models, modelVersionDefaultEpochs
from processing import processing
MODEL = "WT-SPCAS9"
proc = processing()
default_epochs = modelVersionDefaultEpochs[MODEL]

DEFAULTS = dict(learning_rate=0.0005, drop_rate=0.3, CNN_filters=128,
                window_size=3, CNN_drop=0.3, CNN_dense1=128, CNN_dense2=64,
                RNN_size=128, RNN_dense1=128, RNN_dense2=64, CNN_RNN_drop=0.3)
SPACE = dict(
    learning_rate=[1e-4, 2.5e-4, 5e-4, 1e-3], drop_rate=[0.2, 0.3, 0.4],
    CNN_filters=[64, 128, 192], window_size=[3, 5], CNN_drop=[0.2, 0.3, 0.4],
    CNN_dense1=[64, 128, 256], CNN_dense2=[32, 64, 128],
    RNN_size=[64, 128, 192], RNN_dense1=[64, 128, 256],
    RNN_dense2=[32, 64, 128], CNN_RNN_drop=[0.2, 0.3, 0.4])
EPOCHS = [32, 48, 64]
BATCHES = [512, 1024]


def draws(n, fold):
    """Draw 0 is their published configuration, so the search is anchored."""
    rng = np.random.default_rng(SEED * 1000 + fold)
    out = [dict(DEFAULTS, epochs=default_epochs, batch_size=1024)]
    for _ in range(n - 1):
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        cfg["epochs"] = int(EPOCHS[int(rng.integers(len(EPOCHS)))])
        cfg["batch_size"] = int(BATCHES[int(rng.integers(len(BATCHES)))])
        out.append(cfg)
    return out


df = pd.read_csv(DATA)
seqs = np.array([s[HALF - UP:HALF + CORE + DOWN] for s in df["seq"]])
assert len(seqs[0]) == UP + CORE + DOWN, len(seqs[0])
y = df["y"].to_numpy(float)
print(f"{len(df):,} guides, {len(seqs[0])} nt, {default_epochs} default epochs")

done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
seen = set(zip(done.get("stage", []), done.get("fold", []), done.get("draw", [])))


def fit_score(cfg, tr_seq, tr_y, va_seq, va_y):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    kw = {k: v for k, v in cfg.items() if k not in ("epochs", "batch_size")}
    m = models(MODEL, summary=False, **kw)
    t0 = time.time()
    # Call .fit/.predict directly: crisprHAL's own train() catches its
    # exceptions and returns None identically on success and failure, which once
    # let an untrained network be scored here.
    m.model.fit(proc.onehotencode(tr_seq), tr_y, epochs=cfg["epochs"],
                batch_size=cfg["batch_size"], verbose=0)
    p = np.asarray(m.model.predict(proc.onehotencode(va_seq), verbose=0)).ravel()
    assert np.isfinite(p).all() and np.std(p) > 0, "degenerate predictions"
    return float(spearmanr(va_y, p).statistic), round(time.time() - t0, 1)


def record(row):
    global done
    done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
    done.to_csv(OUT, index=False)


outer = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(y))))
for fold, (tr, va) in enumerate(outer, start=1):
    inner = KFold(int(round(1 / INNER_FRAC)), shuffle=True,
                  random_state=SEED + fold)
    itr, iva = next(iter(inner.split(tr)))
    best, best_cfg, stale = -np.inf, None, 0
    for i, cfg in enumerate(draws(TRIALS, fold)):
        if ("search", fold, i) in seen:
            prev = done[(done["stage"] == "search") & (done["fold"] == fold)
                        & (done["draw"] == i)].iloc[0]
            if prev["spearman"] > best:
                best, best_cfg, stale = prev["spearman"], json.loads(prev["config"]), 0
            continue
        rho, secs = fit_score(cfg, seqs[tr][itr], y[tr][itr],
                              seqs[tr][iva], y[tr][iva])
        record(dict(stage="search", fold=fold, draw=i, spearman=rho,
                    is_published_config=(i == 0), seconds=secs,
                    config=json.dumps(cfg)))
        print(f"  fold {fold} draw {i:2d}: inner rho {rho:.4f}  ({secs/60:.1f} min)",
              flush=True)
        if rho > best:
            best, best_cfg, stale = rho, cfg, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"  fold {fold}: no improvement in {PATIENCE}, stopping")
                break
    if ("final", fold, -1) not in seen:
        rho, secs = fit_score(best_cfg, seqs[tr], y[tr], seqs[va], y[va])
        record(dict(stage="final", fold=fold, draw=-1, spearman=rho,
                    is_published_config=(best_cfg == draws(TRIALS, fold)[0]),
                    seconds=secs, config=json.dumps(best_cfg)))
        print(f"fold {fold} FINAL: outer rho {rho:.4f}  ({secs/60:.1f} min)",
              flush=True)

f = done[done["stage"] == "final"]
if len(f) == 5:
    print(f"\nmatched-search crisprHAL 2: mean rho {f['spearman'].mean():.4f} "
          f"(folds {', '.join(f'{v:.4f}' for v in f['spearman'])})")
    print("SLICER tuned on the same folds: 0.7213 pooled out-of-fold")
    print("crisprHAL at its 10-draw search: 0.7040 pooled out-of-fold")

# --- cell 4: bring it home --------------------------------------------------
# Download /content/crisprhal_tune_matched.csv and put it in results/, then:
#   python -m sgrna.heldout_compare        # re-run the paired bootstrap
# The pooled out-of-fold comparison needs per-guide predictions, so if the
# matched search wins, re-export them as preds_crisprhal_matched_fold*.csv
# (columns y,pred, one file per fold, validation rows in fold order) -- that is
# the format heldout_compare.py reconstructs the alignment for.
