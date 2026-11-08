"""Forward-pass timing on a Colab T4: crisprHAL 2 (GPU) and SLICER (CPU and GPU).

Paste into ONE Colab cell and run, with Runtime > Change runtime type > T4 GPU.
It installs what it needs, prints a table and writes inference_t4.json.

crisprHAL 2 WT-SpCas9 (their shipped .keras) is timed on the GPU that TensorFlow
sees, at 1 / 100 / 10,000 guides, median of 3 after a warm-up. Cold start is
import + load + the first predict, reported separately because on a GPU the
first call pays for CUDA/cuDNN initialisation.

SLICER is the champion's configuration (400 trees, depth 4, 300 columns, hist)
trained on synthetic data of the right shape, timed on the Colab CPU and on the
T4 (device='cuda'). Prediction cost of a tree ensemble depends on the number of
trees and their depth and not on the values in the data, and synthetic data with
signal grows trees to full depth, so this is if anything an upper bound.
"""
import subprocess, sys, os
if not os.path.isdir("/content/crisprHAL"):
    subprocess.run(["git", "clone", "-q", "https://github.com/tbrowne5/crisprHAL",
                    "/content/crisprHAL"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "xgboost"], check=True)
import json, os, time, sys
import numpy as np

HAL = "/content/crisprHAL"
out = {}

# ---------- crisprHAL 2 ----------
t_start = time.perf_counter()
sys.path.insert(0, HAL); cwd = os.getcwd(); os.chdir(HAL)
import tensorflow as tf
from processing import processing
os.chdir(cwd)
print("GPUs seen by TensorFlow:", tf.config.list_physical_devices("GPU"))
import_s = time.perf_counter() - t_start
t0 = time.perf_counter()
m = tf.keras.models.load_model(f"{HAL}/models/WT-SPCAS9.keras", compile=False)
load_s = time.perf_counter() - t0
proc = processing(); L = m.input_shape[1]
rng = np.random.default_rng(41)
bases = np.array(list("ACGT"))
seqs = np.array(["".join(rng.choice(bases, L)) for _ in range(10_000)])
X = proc.onehotencode(seqs)
t0 = time.perf_counter(); m.predict(X[:1], verbose=0)
first_s = time.perf_counter() - t0
out["hal_cold_start_s"] = dict(import_s=import_s, load_s=load_s, first_predict_s=first_s,
                               total_s=import_s + load_s + first_s)
print(f"crisprHAL cold start: import {import_s:.1f} + load {load_s:.1f} "
      f"+ first predict {first_s:.1f} = {import_s + load_s + first_s:.1f} s")
for n in (1, 100, 10_000):
    b = X[:n]; m.predict(b, verbose=0)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); m.predict(b, verbose=0); ts.append(time.perf_counter() - t0)
    med = float(np.median(ts))
    out[f"hal_gpu_{n}"] = dict(latency_s=med, guides_per_s=n / med)
    print(f"  crisprHAL 2 on T4, {n:>6,} guides: {med*1e3:10.2f} ms  {n/med:12,.0f} guides/s")

# ---------- SLICER ----------
import xgboost as xgb
n_tr, d = 20_000, 300
Xtr = rng.normal(size=(n_tr, d)).astype(np.float32)
w = rng.normal(size=d); ytr = (Xtr @ w + 0.5 * Xtr[:, 0] * Xtr[:, 1]
                               + rng.normal(size=n_tr)).astype(np.float32)
Xs = rng.normal(size=(10_000, d)).astype(np.float32)
for dev in ("cpu", "cuda"):
    try:
        mdl = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                               subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                               device=dev, n_jobs=-1, random_state=41)
        mdl.fit(Xtr, ytr)
        for n in (1, 100, 10_000):
            b = Xs[:n]; mdl.predict(b)
            ts = []
            for _ in range(5):
                t0 = time.perf_counter(); mdl.predict(b); ts.append(time.perf_counter() - t0)
            med = float(np.median(ts))
            out[f"slicer_{dev}_{n}"] = dict(latency_s=med, guides_per_s=n / med)
            print(f"  SLICER config on {dev:>4}, {n:>6,} guides: {med*1e3:10.3f} ms  "
                  f"{n/med:14,.0f} guides/s")
    except Exception as exc:
        print("  SLICER on", dev, "failed:", exc)

import platform
out["cpu_count"] = os.cpu_count()
json.dump(out, open("inference_t4.json", "w"), indent=2)
print("\nwrote inference_t4.json")
