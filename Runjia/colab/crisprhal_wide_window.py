# ============================================================================
# Does crisprHAL 2 close the gap when it is given the DNA it currently cannot
# see?  --  run this in Google Colab on a GPU runtime.
#
# WHY. Our headline is SLICER 0.7072 against crisprHAL 0.697 on the same rows,
# label and folds. But the two models do not see the same DNA. crisprHAL reads
# 378 nt (189 up, 166 down); family A reaches +/-1,000 nt. Restricting family A
# to their span already showed their convolutions WIN by 0.025, so our margin
# comes from DNA they never see -- not from a better encoding. The obvious
# question, and the one a reviewer will ask, is what happens when they are
# handed the same reach.
#
# THE 2x2 THIS COMPLETES
#                     20-mer only    +/-175 nt     +/-1,000 nt
#   SLICER               0.5440        0.6724         0.7072
#   crisprHAL 2            ?           0.697            ?
#
# INTERPRETATION, FIXED IN ADVANCE so the result cannot be read after the fact:
#   * If crisprHAL at +/-1,000 nt reaches ~0.707 or better, our contribution is
#     "we identified the feature", and the efficiency claim stands only as
#     "a deep model is not required to exploit it".
#   * If it stays near 0.697, then a tuned CNN cannot use long-range
#     composition even when handed it, which vindicates the inductive-bias
#     argument on THEIR architecture rather than our in-house control.
#   Either way the 378 nt arm must reproduce ~0.697 or the harness is wrong and
#   nothing else here can be trusted.
#
# Runtime: GPU (Runtime -> Change runtime type -> T4). Their GRU is configured
# for cuDNN's fused kernel, so CPU is many times slower.
# ============================================================================

# --- cell 1 ---------------------------------------------------------------
# Colab ships TensorFlow already, so do NOT pin a version -- Colab's Python is
# often too new for older releases and the install will fail. Just check what is
# there. crisprHAL declares no TF requirement; we run it locally on 2.19.1 with
# Keras 3.15.1, and anything from 2.19 up should behave. The 378 nt positive
# control below is what actually certifies the environment: if it reproduces
# ~0.697, the TF version is fine, whatever it says here.
#
# import tensorflow as tf; print("tensorflow", tf.__version__)
# !git clone -q https://github.com/tbrowne5/crisprHAL.git /content/crisprHAL
#
# Only if the import fails outright:
# !pip -q install "tensorflow>=2.19"    (then Runtime -> Restart session)

# --- cell 2: upload crisprhal_widewindow_input.csv.gz ----------------------
# from google.colab import files; files.upload()

# --- cell 3 ---------------------------------------------------------------
# This cell is the third version of itself, and the first two are why it looks the way it does.
# v1 trained through crisprHAL's own train()/predict() with a fixed batch of 1,024: cuDNN cannot allocate the GRU
# workspace at 2,023 nt on a T4, and because those wrappers swallow their exceptions the failed fit returned None and
# predict() scored a randomly-initialised network. v2 added failure checks and a batch fallback but still trained through
# the wrappers, so the swallowed exception hid the failure from its checks and the 2,023 nt arm 'succeeded' at rho
# 0.0434 in 0.36 minutes, which is an untrained net and not a measurement. v3, below, calls
# Keras directly (which raises), retries the widest arm at smaller batches, and re-runs every other width at the batch
# it needed so the width curve is read within one optimiser setting. The table in the report (batch 256) is v3's.
# The two superseded drafts, colab/cell3_revised.py and colab/cell3_v3.py, were folded into this file on 22 Nov 2026.
import os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

HAL_DIR, DATA = "/content/crisprHAL", "/content/crisprhal_widewindow_input.csv.gz"
OUT = "/content/crisprhal_wide_window_results.csv"
MODEL, HALF, CORE, SEED, FOLD = "WT-SPCAS9", 1000, 23, 41, 1
WIDTHS = [("core_23", 0, 0), ("native_378", 189, 166),
          ("mid_1023", 500, 500), ("wide_2023", 1000, 1000)]

sys.path.insert(0, HAL_DIR)
_cwd = os.getcwd(); os.chdir(HAL_DIR)
try:
    import models as hal_models
    from models import models
    from processing import processing
finally:
    os.chdir(_cwd)
proc = processing()
EPOCHS = hal_models.modelVersionDefaultEpochs[MODEL]

df = pd.read_csv(DATA)
y = df["y"].to_numpy(dtype=float); full = df["seq"].to_numpy()
tr, va = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(y))))[FOLD - 1]
import tensorflow as tf


def fit(name, up, down, batch):
    length = up + CORE + down
    lo, hi = HALF - up, HALF + CORE + down
    seqs = np.array([s[lo:hi] for s in full])
    hal_models.modelVersionInputLength[MODEL][0] = length
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
    m = models(MODEL, summary=False)
    if getattr(m, "model", None) is None or m.model.input_shape[1] != length:
        raise RuntimeError(f"{name}: did not build at {length} nt")
    X = proc.onehotencode(seqs[tr]).astype("float32")
    Xv = proc.onehotencode(seqs[va]).astype("float32")
    t0 = time.time()
    # Keras directly, NOT m.train / m.predict -- those swallow the exception.
    m.model.fit(X, y[tr], epochs=EPOCHS, batch_size=batch, verbose=0)
    pred = np.asarray(m.model.predict(Xv, batch_size=batch, verbose=0)).ravel()
    if pred.shape[0] != len(va) or not np.isfinite(pred).all() or np.std(pred) == 0:
        raise RuntimeError(f"{name}: degenerate predictions")
    rho = float(spearmanr(y[va], pred).statistic)
    if rho < 0.30:          # backstop: an untrained net lands near 0
        raise RuntimeError(f"{name}: rho {rho:.4f} implies the fit did not take")
    return dict(arm=name, input_nt=length, batch=batch, fold=FOLD,
                n_train=len(tr), n_test=len(va), spearman=rho,
                r2=float(r2_score(y[va], pred)),
                fit_minutes=round((time.time() - t0) / 60, 2))


def run(name, up, down, batches):
    for b in batches:
        try:
            r = fit(name, up, down, b)
            print(f"  {name:>11} ({r['input_nt']:>4} nt, batch {b:>4}): "
                  f"rho {r['spearman']:.4f}   {r['fit_minutes']:.1f} min", flush=True)
            return r
        except Exception as e:
            print(f"  {name} FAILED at batch {b}: {str(e)[:110]}", flush=True)
    return None


# Step 1: find the largest batch the widest arm can actually train at.
print("== finding a workable batch for 2,023 nt ==", flush=True)
wide = run("wide_2023", 1000, 1000, (256, 64, 16))
if wide is None:
    raise SystemExit("2,023 nt will not train at any batch tried")
B = wide["batch"]
print(f"\n== re-running every width at batch {B} so the curve is comparable ==", flush=True)

rows = [wide]
for name, up, down in WIDTHS[:-1]:
    r = run(name, up, down, (B,))
    if r: rows.append(r)
    pd.DataFrame(rows).to_csv(OUT, index=False)

out = pd.DataFrame(rows).sort_values("input_nt")
print()
print(out[["arm", "input_nt", "batch", "spearman", "fit_minutes"]].to_string(index=False))
print("\nSLICER, same fold 1 split: 0.5473 (20-mer) / 0.6568 (matched) / 0.6944 (full family A)")

# --- cell 4 ---------------------------------------------------------------
# from google.colab import files; files.download(OUT)
