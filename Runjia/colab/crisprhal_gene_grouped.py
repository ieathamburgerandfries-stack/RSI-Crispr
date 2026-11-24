# ============================================================================
# crisprHAL 2 on folds that hold WHOLE GENES out  --  ONE Colab cell: paste the whole file and run it.
#
# Runtime: GPU (Runtime -> Change runtime type -> T4). About 5 x 6 = 30 minutes.
# You will be asked once to upload crisprhal_widewindow_input_genefolds.csv.gz (data/interim/ in the project folder, about 20 MB: the 33,567 curated guides,
# their 2,023 nt windows and labels, and a column `gene_fold` (0 to 4) that puts every gene, and every guide outside a gene's 100 kb block, in one fold).
#
# WHY. The head-to-head of the report is on crisprHAL 2's own random splits. A random split lets a model that reads 378 nt of context recognise the locus of a held-out
# guide from its neighbours in training. These folds are the ones SLICER's gene-grouped numbers use (seed 41), so the comparison carries no locus leakage for either model.
# The model is crisprHAL 2's WT-SpCas9 network with its PUBLISHED configuration (48 epochs; nothing re-tuned), input = the 378 nt of its own context
# (189 nt upstream, the 20-nt guide and PAM, 166 nt downstream), at its PUBLISHED batch size of 1,024. (A first run of this file used batch 256, the batch of the 2,023 nt width
# experiment, where cuDNN cannot allocate the GRU workspace at 1,024; at 378 nt it can, and 256 cost crisprHAL 2 0.016 on random-row folds, 0.681 against 0.697, so that run
# handicapped it: it is kept as results/crisprhal_gene_grouped_fcolab_b256.csv and is not the comparison.)
#
# AFTER IT RUNS: the cell downloads crisprhal_gene_grouped_fcolab_b1024.csv. Save it as results/crisprhal_gene_grouped_fcolab_b1024.csv in the project and run
#   python -m sgrna.compare.gene_grouped_vs_crisprhal assemble
# If the session runs out of compute units first, paste the printed lines back: each fold prints its rho when it finishes and the csv is rewritten after every fold.
# ============================================================================
import os, subprocess, sys, time

BATCH, SEED, HALF, CORE = 1024, 41, 1000, 23
UP, DOWN = 189, 166
MODEL = "WT-SPCAS9"
HAL_DIR, DATA, OUT = "/content/crisprHAL", "/content/crisprhal_widewindow_input_genefolds.csv.gz", "/content/crisprhal_gene_grouped_fcolab_b1024.csv"

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
    up = files.upload()                                      # choose crisprhal_widewindow_input_genefolds.csv.gz
    os.replace(next(iter(up)), DATA)

import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

sys.path.insert(0, HAL_DIR); _cwd = os.getcwd(); os.chdir(HAL_DIR)
try:
    import models as hal_models
    from models import models
    from processing import processing
finally:
    os.chdir(_cwd)
proc = processing(); EPOCHS = hal_models.modelVersionDefaultEpochs[MODEL]
df = pd.read_csv(DATA)
assert len(df) == 33567 and df["seq"].str.len().eq(2 * HALF + CORE).all() and set(df.gene_fold) == {0, 1, 2, 3, 4}, "unexpected input file"
y = df["y"].to_numpy(dtype=float)
seqs = np.array([s[HALF - UP:HALF + CORE + DOWN] for s in df["seq"]])
fold = df["gene_fold"].to_numpy()
hal_models.modelVersionInputLength[MODEL][0] = UP + CORE + DOWN

rows = []
for k in range(5):
    tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
    m = models(MODEL, summary=False)
    if getattr(m, "model", None) is None or m.model.input_shape[1] != UP + CORE + DOWN:
        raise RuntimeError("did not build at 378 nt")
    X = proc.onehotencode(seqs[tr]).astype("float32"); Xv = proc.onehotencode(seqs[va]).astype("float32")
    t0 = time.time()
    m.model.fit(X, y[tr], epochs=EPOCHS, batch_size=BATCH, verbose=0)           # Keras directly: it raises on failure
    pred = np.asarray(m.model.predict(Xv, batch_size=BATCH, verbose=0)).ravel()
    rho = float(spearmanr(y[va], pred).statistic)
    if not np.isfinite(pred).all() or np.std(pred) == 0 or rho < 0.30:
        raise RuntimeError(f"fold {k}: degenerate fit (rho {rho:.4f})")
    rows.append(dict(model="crisprHAL 2", fold=k, spearman=rho, r2=float(r2_score(y[va], pred)), fit_minutes=round((time.time() - t0) / 60, 2), n_train=len(tr), n_test=len(va), note=f"Colab T4, 378 nt, batch {BATCH}"))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"fold {k}: rho {rho:.4f}   {rows[-1]['fit_minutes']:.1f} min", flush=True)

print("\n", pd.DataFrame(rows)[["fold", "spearman", "fit_minutes"]].to_string(index=False))
try:
    from google.colab import files
    files.download(OUT)
except Exception as e:
    print("download", OUT, "from the file browser:", e)
