# ============================================================================
# Re-tune crisprHAL 2's OTHER hyperparameters at each input width.
#   -- run in Google Colab on a GPU runtime (Runtime -> Change runtime type -> T4)
#
# This is item 7 of the report's remaining list, and the last unrun one that is
# not a 40-GPU-hour matched search.
#
# WHAT IS ALREADY KNOWN. Handed wider DNA at its published settings, their model
# gets worse, not better (fold 1, batch 256):
#
#       23 nt  0.5494      378 nt  0.6799      1,023 nt  0.6602      2,023 nt  0.6299
#
# and the report is careful about what that supports: "the published
# configuration, used as published, is degraded by input wider than it was tuned
# for" -- NOT "a convolutional-recurrent architecture cannot exploit long-range
# context". The gap between those two sentences is exactly this script, because
# their epochs, dropout, filters and GRU width were all chosen at 378 nt.
#
# Their own paper supplies part of the control and points the same way: they
# searched input length per dataset, stopped at 378 nt for E. coli SpCas9, and
# report the gains levelling off (+0.00034 rho per upstream nt, +0.00074 per
# downstream). So 378 nt is where THEIR search stopped improving. What follows
# asks whether the decline past it is a tuning artefact.
#
# INTERPRETATION, FIXED IN ADVANCE.
#   * If the tuned score at 1,023 / 2,023 nt still sits below the tuned score at
#     378 nt, then "degraded by wider input" survives re-tuning and Finding 2's
#     narrow claim can be widened to the architecture as searched here.
#   * If re-tuning closes the gap, Finding 2 keeps only the published-
#     configuration version of the claim -- which is all it currently asserts --
#     and SLICER's +0.065 at +-1,000 nt must be requoted against the tuned arm.
#   * Either way the 378 nt arm must reproduce ~0.68 at batch 256, or the
#     harness is wrong and nothing else here can be trusted.
#
# COST. 3 widths x (up to 12 search draws, patience 4, so usually 5-8) + 1 final
# + 1 published-config reference. Expect 25-30 fits; at 378 nt a fit is minutes
# and at 2,023 nt it is many times that, so budget 4-6 GPU-hours and let the
# resume file carry it across sessions.
#
# BATCH SIZE IS FIXED AT 256 ON PURPOSE. cuDNN cannot allocate the GRU workspace
# at 2,023 nt with their published batch of 1,024, so every width in the report
# was run at 256 and the 378 nt arm at both batches brackets that cost at
# -0.011. Keeping 256 everywhere -- including the published-config reference --
# is what makes these numbers comparable with the ones above.
# ============================================================================

# --- cell 1 ---------------------------------------------------------------
# Do NOT pin TensorFlow: Colab's Python is often too new for older releases.
# import tensorflow as tf; print("tensorflow", tf.__version__)
# !git clone -q https://github.com/tbrowne5/crisprHAL.git /content/crisprHAL

# --- cell 2: upload crisprhal_widewindow_input.csv.gz ---------------------
# from google.colab import files; files.upload()

# --- cell 3 ---------------------------------------------------------------
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

DATA = "/content/crisprhal_widewindow_input.csv.gz"
HAL = "/content/crisprHAL"
OUT = "/content/crisprhal_width_tuning.csv"      # resume file; download this
                                                 # into results/ under this name
SEED = 41
HALF, CORE = 1000, 23
# (label, up, down). 189/166 reproduces their released input exactly and is the
# positive control; it is listed first so a broken harness is caught cheaply.
WIDTHS = [("native_378", 189, 166),
          ("mid_1023", 500, 500),
          ("wide_2023", 1000, 1000)]
FOLD = 1                 # one fold, as in the width sweep this extends
TRIALS = 12
PATIENCE = 4
INNER_FRAC = 0.2
BATCH = 256              # fixed; see the header

sys.path.insert(0, HAL)
_cwd = os.getcwd()
os.chdir(HAL)
try:
    import models as hal_models
    from models import models
    from processing import processing
finally:
    os.chdir(_cwd)
MODEL = "WT-SPCAS9"
proc = processing()
default_epochs = hal_models.modelVersionDefaultEpochs[MODEL]

DEFAULTS = dict(learning_rate=0.0005, drop_rate=0.3, CNN_filters=128,
                window_size=3, CNN_drop=0.3, CNN_dense1=128, CNN_dense2=64,
                RNN_size=128, RNN_dense1=128, RNN_dense2=64, CNN_RNN_drop=0.3)
SPACE = dict(
    learning_rate=[1e-4, 2.5e-4, 5e-4, 1e-3], drop_rate=[0.2, 0.3, 0.4],
    CNN_filters=[64, 128, 192], window_size=[3, 5, 7], CNN_drop=[0.2, 0.3, 0.4],
    CNN_dense1=[64, 128, 256], CNN_dense2=[32, 64, 128],
    RNN_size=[64, 128, 192], RNN_dense1=[64, 128, 256],
    RNN_dense2=[32, 64, 128], CNN_RNN_drop=[0.2, 0.3, 0.4])
EPOCHS = [32, 48, 64, 96]
# Two deliberate widenings over the 378 nt search space, because the whole
# question is whether the space itself was the constraint: a 7-wide convolution
# and 96 epochs are both things a wider input might want and could not ask for.


def draws(n, width_label):
    """Draw 0 is their published configuration, so the search is anchored.

    The per-width seed is a CRC of the label and not `hash()`: Python salts
    string hashing per process, so a resumed session would redraw a different
    draw 3 and the resume file would silently mix two search spaces.
    """
    rng = np.random.default_rng(SEED * 1000 + zlib.crc32(width_label.encode()))
    out = [dict(DEFAULTS, epochs=default_epochs, batch_size=BATCH)]
    for _ in range(n - 1):
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        cfg["epochs"] = int(EPOCHS[int(rng.integers(len(EPOCHS)))])
        cfg["batch_size"] = BATCH
        out.append(cfg)
    return out


df = pd.read_csv(DATA)
y = df["y"].to_numpy(dtype=float)
full = df["seq"].to_numpy()
print(f"{len(df):,} guides, {len(full[0])} nt stored, "
      f"{default_epochs} default epochs")

done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame()
seen = set(zip(done.get("stage", []), done.get("width", []),
               done.get("draw", [])))


def window(up, down):
    lo, hi = HALF - up, HALF + CORE + down
    w = np.array([s[lo:hi] for s in full])
    assert len(w[0]) == up + CORE + down
    return w


def fit_score(cfg, length, tr_seq, tr_y, va_seq, va_y):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    # Their builder takes input_length from a registry rather than an argument,
    # so patch the registry before constructing.
    hal_models.modelVersionInputLength[MODEL][0] = length
    kw = {k: v for k, v in cfg.items() if k not in ("epochs", "batch_size")}
    m = models(MODEL, summary=False, **kw)
    # Their __init__ swallows build errors into a print and leaves self.model
    # unset, and their train() returns None identically on success and failure.
    # Both have already cost this project a scored run of an untrained network,
    # so check the object and call .fit/.predict directly.
    assert getattr(m, "model", None) is not None, f"build failed at {length} nt"
    assert m.model.input_shape[1] == length, (
        f"built at {m.model.input_shape[1]} nt, expected {length}")
    t0 = time.time()
    m.model.fit(proc.onehotencode(tr_seq), tr_y, epochs=cfg["epochs"],
                batch_size=cfg["batch_size"], verbose=0)
    p = np.asarray(m.model.predict(proc.onehotencode(va_seq),
                                   verbose=0)).ravel()
    assert np.isfinite(p).all() and np.std(p) > 0, "degenerate predictions"
    return float(spearmanr(va_y, p).statistic), round(time.time() - t0, 1)


def record(row):
    global done
    done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
    done.to_csv(OUT, index=False)


outer = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(y))))
tr, va = outer[FOLD - 1]
inner = KFold(int(round(1 / INNER_FRAC)), shuffle=True, random_state=SEED + FOLD)
itr, iva = next(iter(inner.split(tr)))

for label, up, down in WIDTHS:
    seqs = window(up, down)
    length = up + CORE + down
    print(f"\n=== {label}: {length} nt ===", flush=True)
    best, best_cfg, stale = -np.inf, None, 0
    for i, cfg in enumerate(draws(TRIALS, label)):
        if ("search", label, i) in seen:
            prev = done[(done["stage"] == "search") & (done["width"] == label)
                        & (done["draw"] == i)].iloc[0]
            if prev["spearman"] > best:
                best = prev["spearman"]
                best_cfg, stale = json.loads(prev["config"]), 0
            continue
        rho, secs = fit_score(cfg, length, seqs[tr][itr], y[tr][itr],
                              seqs[tr][iva], y[tr][iva])
        record(dict(stage="search", width=label, input_nt=length, draw=i,
                    spearman=rho, is_published_config=(i == 0), seconds=secs,
                    config=json.dumps(cfg)))
        print(f"  draw {i:2d}: inner rho {rho:.4f}  ({secs / 60:.1f} min)",
              flush=True)
        if rho > best:
            best, best_cfg, stale = rho, cfg, 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"  no improvement in {PATIENCE}, stopping this width")
                break

    # Two outer scores per width: the search winner, and their published
    # configuration at this width. The second is the number the report already
    # quotes, so it doubles as the check that this harness matches that one.
    for stage, cfg in (("final", best_cfg),
                       ("published", draws(TRIALS, label)[0])):
        if (stage, label, -1) in seen:
            continue
        rho, secs = fit_score(cfg, length, seqs[tr], y[tr], seqs[va], y[va])
        record(dict(stage=stage, width=label, input_nt=length, draw=-1,
                    spearman=rho, is_published_config=(stage == "published"),
                    seconds=secs, config=json.dumps(cfg)))
        print(f"  {stage.upper()} at {length} nt: outer rho {rho:.4f}  "
              f"({secs / 60:.1f} min)", flush=True)

f = done[done["stage"].isin(("final", "published"))]
if len(f):
    print()
    print(f.pivot_table(index="input_nt", columns="stage",
                        values="spearman").round(4).to_string())
    print("\nreference, published config at batch 256, fold 1: "
          "378 nt 0.6799 / 1,023 nt 0.6602 / 2,023 nt 0.6299")
    print("SLICER on the same fold: 0.6568 at their span, 0.6944 at +-1,000 nt")

# --- cell 4: bring it home ------------------------------------------------
# from google.colab import files; files.download(OUT)
# Put it in results/crisprhal_width_tuning.csv, then update Finding 2's
# "re-tune at each width has not been run" paragraph and Part 4 item 7.
