# ============================================================================
# crisprHAL 2 at wider input, on every fold  --  ONE Colab cell: paste the whole file and run it.
#
# Runtime: GPU (Runtime -> Change runtime type -> T4). crisprHAL 2's GRU is built for cuDNN's fused kernel.
# You will be asked once to upload crisprhal_widewindow_input.csv.gz (columns protospacer, y, seq (2,023 nt), ok; it is
# data/interim/crisprhal_widewindow_input.csv.gz in the project folder, 19.9 MB, the 33,567 curated guides in the order of
# data/interim/curated_guide_index.csv). Everything else is fetched by this cell.
#
# WHY. SLICER reads up to +/-1,000 nt of flank, crisprHAL 2 reads 378 nt (189 up, 166 down). Restricting SLICER to their span
# showed their convolutions win by about 0.02, so SLICER's lead comes from DNA their input never reaches. This retrains crisprHAL 2
# at four widths with its PUBLISHED configuration (epochs, dropout, filters and GRU width as released, tuned at 378 nt; nothing is
# re-tuned) on the same rows, label and folds as the SLICER arms: KFold(5, shuffle=True, random_state=41) over the 33,567 curated
# guides. The question is only whether the published configuration is degraded by input wider than it was designed for.
#
# STATUS: every arm has been run (fold 1 at all widths in the first Colab run; folds 2-5 in a second; the 23 nt arm and the 378 nt arms of
# folds 2 and 3 on a CPU), so with the default PLAN this file reproduces the whole experiment: all four widths on all five folds,
# about 3.5 hours on a T4 (6 + 14 + 27 minutes per fold plus 1.5 for the 23 nt arm). To run only part of it, edit PLAN. Fold 1's 378 nt arm
# was reproduced on the T4 as a harness check (0.6809 against 0.6799); CHECK = True repeats it first. A cheap robustness run: PLAN = {2: ["native_378"],
# 3: ["native_378"]} retrains the two 378 nt arms that were trained on a CPU (about 12 minutes).
#
# BATCH 256 throughout. The published batch is 1,024, but cuDNN cannot allocate the GRU workspace at 2,023 nt on a T4 at that batch.
# At 378 nt the two batches differ by -0.011 (0.6906 at 1,024, 0.6799 at 256), a fifth of the effect at 2,023 nt, so every width is
# read at one optimiser setting.
#
# HISTORY. Earlier drafts trained through crisprHAL's own train()/predict(), which swallow exceptions: a failed 2,023 nt fit came
# back as a randomly initialised network scored at rho 0.043. This file calls Keras directly (which raises) and refuses any fit
# whose predictions are constant or whose rho is below 0.30. It is one file in place of the earlier fold-1-only script, a
# separate folds 2-5 script that had to be pasted over cells of it, and two cell-3 drafts.
#
# AFTER IT RUNS: the cell downloads crisprhal_wide_window_fcolab.csv (if the session runs out of compute units first, paste the printed
# lines back). Save it as results/crisprhal_wide_window_fcolab3.csv in the project (the files fcolab.csv and fcolab2.csv hold the
# values already transcribed) and run   python -m sgrna.compare.crisprhal_wide_window_folds assemble   (the assemble step reads every
# results/crisprhal_wide_window_f*.csv; the harness check is printed, not saved).
# ============================================================================
import os, subprocess, sys, time

# ---- what to run: {fold: [arms]} ------------------------------------------------------------------------------------------------
# Arms: core_23 (guide + PAM), native_378 (their span: 189 up, 166 down), mid_1023 (+/-500), wide_2023 (+/-1,000).
PLAN = {f: ["core_23", "native_378", "mid_1023", "wide_2023"] for f in (1, 2, 3, 4, 5)}
# Only the two CPU-trained 378 nt arms:  PLAN = {2: ["native_378"], 3: ["native_378"]}
CHECK = False                # True: run fold 1's 378 nt arm first as the harness check (must give about 0.680)
BATCH, SEED, HALF, CORE = 256, 41, 1000, 23
WIDTHS = {"core_23": (0, 0), "native_378": (189, 166), "mid_1023": (500, 500), "wide_2023": (1000, 1000)}
HAL_DIR, DATA, OUT = "/content/crisprHAL", "/content/crisprhal_widewindow_input.csv.gz", "/content/crisprhal_wide_window_fcolab.csv"

# ---- environment ----------------------------------------------------------------------------------------------------------------
# Colab ships TensorFlow, so do not pin a version (an old pin often fails on Colab's newer Python). crisprHAL declares no TF
# requirement; anything from 2.19 up behaves, and the harness check below is what certifies the environment.
try:
    import tensorflow as tf
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "tensorflow>=2.19"], check=True)
    raise SystemExit("TensorFlow was just installed: Runtime -> Restart session, then run this cell again")
print("tensorflow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))
if not tf.config.list_physical_devices("GPU"):
    raise SystemExit("No GPU: Runtime -> Change runtime type -> T4 GPU, then run this cell again")
if not os.path.isdir(HAL_DIR):
    subprocess.run(["git", "clone", "-q", "https://github.com/tbrowne5/crisprHAL.git", HAL_DIR], check=True)
if not os.path.exists(DATA):
    from google.colab import files
    up = files.upload()                                      # choose crisprhal_widewindow_input.csv.gz
    os.replace(next(iter(up)), DATA)

import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

sys.path.insert(0, HAL_DIR); _cwd = os.getcwd(); os.chdir(HAL_DIR)
try:
    import models as hal_models
    from models import models
    from processing import processing
finally:
    os.chdir(_cwd)
MODEL = "WT-SPCAS9"
proc = processing(); EPOCHS = hal_models.modelVersionDefaultEpochs[MODEL]
df = pd.read_csv(DATA)
assert len(df) == 33567 and df["seq"].str.len().eq(2 * HALF + CORE).all(), "unexpected input file"
y = df["y"].to_numpy(dtype=float); full = df["seq"].to_numpy()
splits = list(KFold(5, shuffle=True, random_state=SEED).split(np.arange(len(y))))


def fit(fold, name):
    up, down = WIDTHS[name]; length = up + CORE + down
    tr, va = splits[fold - 1]
    seqs = np.array([s[HALF - up:HALF + CORE + down] for s in full])
    hal_models.modelVersionInputLength[MODEL][0] = length
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
    m = models(MODEL, summary=False)
    if getattr(m, "model", None) is None or m.model.input_shape[1] != length:
        raise RuntimeError(f"fold {fold} {name}: did not build at {length} nt")
    X = proc.onehotencode(seqs[tr]).astype("float32"); Xv = proc.onehotencode(seqs[va]).astype("float32")
    t0 = time.time()
    m.model.fit(X, y[tr], epochs=EPOCHS, batch_size=BATCH, verbose=0)           # Keras directly: it raises on failure
    pred = np.asarray(m.model.predict(Xv, batch_size=BATCH, verbose=0)).ravel()
    rho = float(spearmanr(y[va], pred).statistic)
    if pred.shape[0] != len(va) or not np.isfinite(pred).all() or np.std(pred) == 0 or rho < 0.30:
        raise RuntimeError(f"fold {fold} {name}: degenerate fit (rho {rho:.4f}); an untrained net lands near 0")
    return dict(model="crisprHAL 2", arm=name, input_nt=length, batch=BATCH, fold=fold, spearman=rho,
                r2=float(r2_score(y[va], pred)), fit_minutes=round((time.time() - t0) / 60, 2),
                n_train=len(tr), n_test=len(va), note="Colab T4")


if CHECK:
    print("== harness check: fold 1, 378 nt (first run: 0.6799) ==", flush=True)
    r = fit(1, "native_378")
    print(f"   rho {r['spearman']:.4f}   {r['fit_minutes']:.1f} min", flush=True)
    if abs(r["spearman"] - 0.6799) > 0.02:
        raise SystemExit(f"harness check failed: {r['spearman']:.4f} is not near 0.680, so nothing else here can be trusted")
    print("   harness reproduces the first run\n", flush=True)

rows = []
for fold, arms in PLAN.items():
    for name in arms:
        r = fit(fold, name); rows.append(r)
        pd.DataFrame(rows).to_csv(OUT, index=False)                              # saved after every arm: a crash loses one arm
        print(f"fold {fold} {name:>10} ({r['input_nt']:>4} nt): rho {r['spearman']:.4f}   {r['fit_minutes']:.1f} min", flush=True)

print("\n", pd.DataFrame(rows)[["fold", "arm", "input_nt", "spearman", "fit_minutes"]].to_string(index=False))
print("\nSLICER on the same folds (results/headtohead.csv): full flank 0.7079 / 0.7121 / 0.7112 / 0.7136 on folds 2-5 "
      "and 0.6944 on fold 1; matched to 378 nt 0.6724 / 0.6733 / 0.6793 / 0.6802 and 0.6568")
try:
    from google.colab import files
    files.download(OUT)
except Exception as e:                                                           # not in Colab, or the download was blocked
    print("download", OUT, "from the file browser:", e)
