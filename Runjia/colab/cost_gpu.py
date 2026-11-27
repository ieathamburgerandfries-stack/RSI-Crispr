# ============================================================================
# What a GPU buys: crisprHAL 2 and SLICER-shaped trees on a Colab T4  --  ONE Colab cell: paste the whole file and run it.
#
# Runtime: GPU (Runtime -> Change runtime type -> T4). About 35 minutes (three fits of crisprHAL 2 take about 5.5 minutes each; set FIT_REPEATS = 0 below to skip them,
# and the run takes about 12 minutes). Nothing to upload: timing does not depend on the sequences, so the inputs are random 378-nt contexts and random labels.
#
# WHY. The cost section of the paper gives CPU times measured on a quiet laptop (one thread for scoring) and, separately, what a GPU changes. crisprHAL 2 is a network and
# meant for a GPU; SLICER is a tree model whose features are built on the host. This file times, with every repeat kept (so a spread can be drawn, as for the CPU):
#   1. crisprHAL 2's shipped WT-SpCas9 network scoring 1, 100 and 10,000 guides on the T4: (a) from raw 378-nt contexts on the host (one-hot encoding, copy to the card,
#      forward pass, copy back: what a request costs), (b) with the encoded batch already on the card (forward pass only). Its Keras batch size is swept (32, 256, 1024,
#      2048) and the best is kept, so it is not penalised by Keras' default of 32.
#   2. SLICER-shaped XGBoost models (default: 400 trees, depth 4, 300 columns; tuned: 4,000 trees, depth 5, 450 columns) scoring the same batch sizes on the T4 with the
#      matrix on the host or on the card, and on one CPU thread of this Colab machine. This is the forward pass only: SLICER's features are built on the host CPU in
#      either case (0.08 ms per guide on the laptop, `cost_repeats_inference.csv`), so the card can only ever speed up the part that costs least.
#   3. fits: crisprHAL 2's published configuration (48 epochs, batch 1,024, 26,853 guides: the training part of SLICER's cost fold) on the T4, and the two XGBoost shapes
#      on the T4 and on this machine's CPU threads.
# Scoring cost of a tree ensemble depends on the number of trees and their depth and not on the values, and so does the cost of a fixed number of epochs of a network, so
# random inputs are a fair stand-in for timing and say nothing about accuracy.
#
# AFTER IT RUNS: the cell downloads cost_gpu.csv (one row per repeat). Save it as results/cost_gpu.csv in the project; the cost figure and the audit read it.
# This file replaces colab/inference_bench_t4.py, inference_bench_onecore.py, slicer_cpu_check.py and crisprhal_inference_bench.py (their medians stay in results/inference_*.csv).
# ============================================================================
import json, os, subprocess, sys, textwrap, time
from collections import deque

FIT_REPEATS = 3
REPS = {1: 200, 100: 60, 10000: 30}                  # repeats per batch size
SMOKE = os.environ.get("COST_GPU_SMOKE") == "1"      # a few seconds on any machine, to check the file; the numbers it writes mean nothing
if SMOKE:
    REPS, FIT_REPEATS = {1: 2, 100: 2, 10000: 2}, 0
HAL_DIR = os.environ.get("HAL_DIR", "/content/crisprHAL")
OUT = os.environ.get("COST_OUT", "/content/cost_gpu.csv")

if not os.path.isdir(HAL_DIR):
    subprocess.run(["git", "clone", "-q", "https://github.com/tbrowne5/crisprHAL.git", HAL_DIR], check=True)
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "xgboost"], check=True)

WORKER = textwrap.dedent('''
    import os, sys, time, json, csv, subprocess
    phase, HAL, OUT = os.environ["COST_PHASE"], os.environ["HAL_DIR"], os.environ["COST_OUT"]
    REPS = {int(k): v for k, v in json.loads(os.environ["COST_REPS"]).items()}
    FIT = int(os.environ["COST_FIT"]); SKIP = os.environ.get("COST_SKIP", "").split(",")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if phase == "inference":                                   # one host thread, for every library, before anything is imported
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[k] = "1"
    import numpy as np
    rows = []
    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    has_gpu = bool(gpu_name)
    CARD = "T4" if has_gpu else "no GPU (smoke test)"            # what the rows are labelled with; the xgboost device falls back to the CPU without a card
    meta = dict(phase=phase, gpu=gpu_name or "none", cpu_count=os.cpu_count())

    def add(model, device, placement, what, batch, rep, value, unit):
        rows.append(dict(model=model, device=device, placement=placement, what=what, batch=batch, rep=rep, value=value, unit=unit))

    def timed(fn, reps):
        fn(); out = []
        for _ in range(reps):
            t = time.perf_counter(); fn(); out.append(time.perf_counter() - t)
        return out

    def best_batch_size(n, fn_of_bs):
        cands = sorted({min(n, b) for b in (32, 256, 1024, 2048)})
        return min(cands, key=lambda bs: float(np.median(timed(lambda: fn_of_bs(bs), 3 if n >= 10000 else 5))))

    rng = np.random.default_rng(41)

    def shaped_xgb(trees, depth, cols, n, device):
        import xgboost as xgb
        X = rng.normal(size=(n, cols)).astype(np.float32); w = rng.normal(size=cols)
        y = (X @ w + 0.5 * X[:, 0] * X[:, 1] + rng.normal(size=n)).astype(np.float32)
        return xgb.XGBRegressor(n_estimators=trees, learning_rate=0.05, max_depth=depth, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=41,
                                device=device if has_gpu else "cpu"), X, y

    SHAPES = (("SLICER default shape", 400, 4, 300), ("SLICER tuned shape", 4000, 5, 450))

    def write():
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "device", "placement", "what", "batch", "rep", "value", "unit"]); w.writeheader(); w.writerows(rows)

    if phase == "inference":
        if "hal" not in SKIP:
            import tensorflow as tf
            sys.path.insert(0, HAL); cwd = os.getcwd(); os.chdir(HAL)
            from processing import processing
            os.chdir(cwd)
            tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
            meta["tf"] = tf.__version__; meta["tf_gpus"] = [g.name for g in tf.config.list_physical_devices("GPU")]
            h = tf.keras.models.load_model(HAL + "/models/WT-SPCAS9.keras", compile=False)
            L = h.input_shape[1]; proc = processing()
            ctx_all = np.array(["".join(rng.choice(list("ACGT"), L)) for _ in range(max(REPS))])
            X_all = proc.onehotencode(ctx_all)
            for n, reps in REPS.items():
                ctx = ctx_all[:n]
                bs = best_batch_size(n, lambda b: h.predict(proc.onehotencode(ctx), batch_size=b, verbose=0))
                for i, t in enumerate(timed(lambda: h.predict(proc.onehotencode(ctx), batch_size=bs, verbose=0), reps)):
                    add("crisprHAL 2", CARD, "raw contexts on host", "score", n, i, 1000 * t / n, "ms/guide")
                add("crisprHAL 2", CARD, "raw contexts on host", "best batch size", n, 0, bs, "guides")
                with tf.device("/GPU:0" if has_gpu else "/CPU:0"):
                    xt = tf.constant(X_all[:n])
                def fwd(b):
                    return tf.concat([h(xt[j:j + b], training=False) for j in range(0, n, b)], 0).numpy()
                bs2 = best_batch_size(n, fwd)
                for i, t in enumerate(timed(lambda: fwd(bs2), reps)):
                    add("crisprHAL 2", CARD, "encoded batch on the card", "forward", n, i, 1000 * t / n, "ms/guide")
                add("crisprHAL 2", CARD, "encoded batch on the card", "best batch size", n, 0, bs2, "guides")
                med = np.median([r["value"] for r in rows if r["placement"] == "raw contexts on host" and r["batch"] == n and r["unit"] == "ms/guide"])
                print(f"  crisprHAL 2, {n} guides: {med:.4f} ms/guide from raw contexts (batch size {bs}), best batch size on the card {bs2}", flush=True)
            write()
        if "xgb" not in SKIP:
            try:
                import cupy as cp
            except Exception:
                cp = None
            for lab, trees, depth, cols in SHAPES:
                m, X, y = shaped_xgb(trees, depth, cols, 20000, "cuda")
                m.set_params(n_jobs=2); m.fit(X, y)                        # fitted on the card, then scored where each row says
                bst = m.get_booster(); Xs = rng.normal(size=(max(REPS), cols)).astype(np.float32)
                for dev, place in (("cpu", "one CPU thread"), ("cuda", "matrix on host"), ("cuda", "matrix on the card")):
                    if dev == "cuda" and not has_gpu:
                        continue
                    if place == "matrix on the card" and cp is None:
                        continue
                    bst.set_param({"device": dev, "nthread": 1})
                    for n, reps in REPS.items():
                        if place == "matrix on the card":
                            bd = cp.asarray(Xs[:n])
                            def f():
                                bst.inplace_predict(bd); cp.cuda.Stream.null.synchronize()
                        else:
                            def f():
                                bst.inplace_predict(Xs[:n])
                        for i, t in enumerate(timed(f, reps)):
                            add(lab, CARD if dev == "cuda" else "Colab CPU", place, "forward", n, i, 1000 * t / n, "ms/guide")
                print(f"  {lab}: scored", flush=True)
            write()
    else:
        n_train = 26853
        if "xgb" not in SKIP:
            for lab, trees, depth, cols in SHAPES:
                for dev, place in (("cuda", CARD), ("cpu", "Colab CPU, all threads")):
                    reps = (max(FIT, 1) if dev == "cuda" else (2 if trees == 400 else 1)) if FIT else 1
                    for i in range(reps):
                        m, X, y = shaped_xgb(trees, depth, cols, n_train, dev); m.set_params(n_jobs=os.cpu_count() if dev == "cpu" else 1)
                        t0 = time.perf_counter(); m.fit(X, y); t = time.perf_counter() - t0
                        add(lab, place, "synthetic matrix", "fit", n_train, i, t, "s")
                    print(f"  {lab} fit on {place}: {np.median([r['value'] for r in rows if r['model'] == lab and r['device'] == place]):.1f} s", flush=True)
            write()
        if FIT and "hal" not in SKIP:
            import tensorflow as tf
            sys.path.insert(0, HAL); cwd = os.getcwd(); os.chdir(HAL)
            from models import models, modelVersionDefaultEpochs
            from processing import processing
            os.chdir(cwd)
            meta["tf"] = tf.__version__; meta["tf_gpus"] = [g.name for g in tf.config.list_physical_devices("GPU")]
            proc = processing(); epochs = modelVersionDefaultEpochs["WT-SPCAS9"]
            for i in range(FIT):
                tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(41 + i)
                ctx = np.array(["".join(rng.choice(list("ACGT"), 378)) for _ in range(n_train)])
                X = proc.onehotencode(ctx); y = rng.normal(size=n_train)
                m = models("WT-SPCAS9", summary=False)
                t0 = time.perf_counter(); m.train(X, y, epochs=epochs, batch_size=1024, verbose=0); t = time.perf_counter() - t0
                add("crisprHAL 2", CARD, "synthetic contexts", "fit", n_train, i, t, "s")
                print(f"  crisprHAL 2 fit {i + 1}/{FIT}: {t / 60:.2f} min", flush=True)
                write()
    write()
    print("META" + json.dumps(meta))
''')

open("_cost_worker.py", "w").write(WORKER)
meta, parts = {}, []
for phase in ("inference", "fit"):
    print(f"phase: {phase} ...", flush=True)
    out = f"cost_gpu_{phase}.csv"
    env = {**os.environ, "COST_PHASE": phase, "HAL_DIR": HAL_DIR, "COST_OUT": out, "COST_REPS": json.dumps(REPS), "COST_FIT": str(FIT_REPEATS), "TF_CPP_MIN_LOG_LEVEL": "3"}
    p = subprocess.Popen([sys.executable, "_cost_worker.py"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail = deque(maxlen=40)
    for line in p.stdout:
        tail.append(line.rstrip())
        if line.startswith("META"):
            meta[phase] = json.loads(line[4:])
        elif line.startswith("  "):
            print(line.rstrip(), flush=True)
    if p.wait() != 0:
        print("FAILED", phase, "\n".join(tail)); continue
    parts.append(out)

import pandas as pd
if parts:
    d = pd.concat([pd.read_csv(q) for q in parts], ignore_index=True)
    d["gpu"] = next((m["gpu"] for m in meta.values() if m.get("gpu")), "")
    d["session"] = time.strftime("%Y-%m-%d %H:%M")
    d.to_csv(OUT, index=False)
    json.dump(meta, open(OUT.replace(".csv", "_machine.json"), "w"), indent=1)
    g = d[d.unit == "ms/guide"].groupby(["model", "device", "placement", "batch"]).value.agg(["count", "median", lambda x: x.quantile(0.025), lambda x: x.quantile(0.975)])
    g.columns = ["repeats", "median", "q025", "q975"]
    print("\nms per guide (every repeat is in the csv):"); print(g.round(4).to_string())
    f = d[d.unit == "s"].groupby(["model", "device"]).value.agg(["count", "median", "min", "max"]); print("\nfit seconds:"); print(f.round(1).to_string())
    print("\nmachine:", json.dumps(meta))
    if "google.colab" in sys.modules:
        from google.colab import files
        files.download(OUT)
    print(f"\nwrote {OUT}")
