"""Measured forward-pass cost for crisprHAL 2's shipped WT-SpCas9 model."""
import os, sys, time
import numpy as np
HAL = "/Users/runjia/Claude/Projects/RSI09/external_data/crisprHAL"
t_start = time.perf_counter()
sys.path.insert(0, HAL); cwd = os.getcwd(); os.chdir(HAL)
try:
    import tensorflow as tf
    from processing import processing
finally:
    os.chdir(cwd)
import_s = time.perf_counter() - t_start
t0 = time.perf_counter()
m = tf.keras.models.load_model(f"{HAL}/models/WT-SPCAS9.keras", compile=False)
load_s = time.perf_counter() - t0
proc = processing()
L = m.input_shape[1]
print(f"import {import_s:.1f} s, load {load_s:.1f} s, input length {L}")
rng = np.random.default_rng(41)
bases = np.array(list("ACGT"))
seqs = np.array(["".join(rng.choice(bases, L)) for _ in range(10_000)])
X = proc.onehotencode(seqs)
for n in (1, 100, 10_000):
    b = X[:n]
    m.predict(b, verbose=0)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); m.predict(b, verbose=0); ts.append(time.perf_counter()-t0)
    med = float(np.median(ts))
    print(f"  score {n:>7,} guides: {med*1e3:10.2f} ms  {n/med:12,.0f} guides/s")
print(f"COLD START (import + load) = {import_s + load_s:.1f} s")
