"""Forward-pass cost, controlled: crisprHAL 2 vs SLICER with ONE host thread, on the
CPU and on the GPU, so that how many cores the machine has is not a factor.

Controls (the same for both models, both devices):
  * one host thread: OMP/MKL/TF/xgboost pinned to 1 before import; each timing also
    reports process CPU-seconds / wall-seconds, ~1.0 if the pin held;
  * everything in-process, median of repeated calls after a warm-up;
  * crisprHAL 2's batch size is swept (32, 256, 1024, 2048; Keras' default of 32 sends
    a 10,000-guide request to a GPU as 313 tiny batches) and its best is kept, so the
    comparison is against crisprHAL at its best, not at its default;
  * on the GPU, input is timed both on the host (what a request looks like: includes
    the transfer) and already on the card (a pipeline that keeps batches there).

Colab: Runtime > Change runtime type > T4 GPU. Paste into ONE cell. ~10 min. It runs the
CPU phase and the GPU phase in separate processes (TensorFlow cannot un-hide a GPU),
prints a table, and writes onecore.json. Paste the whole printout back.

SLICER is the champion's configuration (400 trees, depth 4, 300 columns, hist) on
synthetic data of that shape; scoring cost depends on tree count and depth, not values.
Cold start is not timed here (see colab/inference_bench_t4.py, whose results are
already in results/inference_colab.csv).
"""
import json, os, subprocess, sys, textwrap

WORKER = textwrap.dedent('''
    import os, sys, time, json
    mode = os.environ["BENCH_MODE"]
    parts = os.environ.get("BENCH_PARTS", "slicer,hal").split(",")
    HAL = os.environ["HAL_DIR"]
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    if mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import numpy as np
    out = {"mode": mode, "cpu_count": os.cpu_count()}
    SIZES = (1, 100, 10000)

    def timed(fn, reps):
        fn(); ts, cs = [], []
        for _ in range(reps):
            c0, t0 = time.process_time(), time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0); cs.append(time.process_time() - c0)
        return float(np.median(ts)), float(np.sum(cs) / np.sum(ts))

    rng = np.random.default_rng(41)
    if "slicer" in parts:
        import xgboost as xgb
        n, d = 20000, 300
        X = rng.normal(size=(n, d)).astype(np.float32); w = rng.normal(size=d)
        y = (X @ w + 0.5 * X[:, 0] * X[:, 1] + rng.normal(size=n)).astype(np.float32)
        Xs = rng.normal(size=(10000, d)).astype(np.float32)
        m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             n_jobs=1, random_state=41, device="cpu").fit(X, y)
        bst = m.get_booster(); bst.set_param({"nthread": 1})
        if mode == "gpu":
            m.set_params(device="cuda")
        for k in SIZES:
            b = Xs[:k]
            med, u = timed(lambda: m.predict(b), 9 if k < 10000 else 5)
            out["slicer_host_%d" % k] = dict(latency_s=med, cpu_over_wall=u)
        if mode == "gpu":
            try:
                import cupy as cp
                for k in SIZES:
                    bd = cp.asarray(Xs[:k])
                    def f():
                        bst.inplace_predict(bd); cp.cuda.Stream.null.synchronize()
                    med, u = timed(f, 9 if k < 10000 else 5)
                    out["slicer_device_%d" % k] = dict(latency_s=med, cpu_over_wall=u)
            except Exception as e:
                out["slicer_device_error"] = repr(e)

    if "hal" in parts:
        sys.path.insert(0, HAL); cwd = os.getcwd(); os.chdir(HAL)
        import tensorflow as tf
        from processing import processing
        os.chdir(cwd)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        out["tf_gpus"] = [g.name for g in tf.config.list_physical_devices("GPU")]
        h = tf.keras.models.load_model(HAL + "/models/WT-SPCAS9.keras", compile=False)
        L = h.input_shape[1]; proc = processing()
        bases = np.array(list("ACGT"))
        seqs = np.array(["".join(rng.choice(bases, L)) for _ in range(10000)])
        XH = proc.onehotencode(seqs)
        for k in SIZES:
            b = XH[:k]; best = None
            for bs in sorted({min(k, v) for v in (32, 256, 1024, 2048)}):
                med, u = timed(lambda: h.predict(b, batch_size=bs, verbose=0),
                               5 if k < 10000 else 2)
                if best is None or med < best[0]:
                    best = (med, u, bs)
            out["hal_host_%d" % k] = dict(latency_s=best[0], cpu_over_wall=best[1],
                                           batch_size=best[2])
            out["hal_default32_%d" % k] = dict(latency_s=timed(
                lambda: h.predict(b, batch_size=32, verbose=0), 2)[0])
        if mode == "gpu":
            for k in SIZES:
                with tf.device("/GPU:0"):
                    xt = tf.constant(XH[:k])
                best = None
                for bs in sorted({min(k, v) for v in (32, 256, 1024, 2048)}):
                    def f():
                        ys = [h(xt[i:i + bs], training=False) for i in range(0, k, bs)]
                        tf.concat(ys, 0).numpy()
                    med, u = timed(f, 5 if k < 10000 else 2)
                    if best is None or med < best[0]:
                        best = (med, u, bs)
                out["hal_device_%d" % k] = dict(latency_s=best[0], cpu_over_wall=best[1],
                                                 batch_size=best[2])
    print("JSON" + json.dumps(out))
''')

HAL_DIR = os.environ.get("HAL_DIR", "/content/crisprHAL")
if "BENCH_WRITE_ONLY" in os.environ:
    open("_bench_worker.py", "w").write(WORKER); sys.exit(0)
if not os.path.isdir(HAL_DIR):
    subprocess.run(["git", "clone", "-q", "https://github.com/tbrowne5/crisprHAL", HAL_DIR], check=True)
if "google.colab" in sys.modules:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "xgboost"], check=True)
open("_bench_worker.py", "w").write(WORKER)

res = {}
for mode in ("cpu", "gpu"):
    print(f"phase: {mode} ...", flush=True)
    r = subprocess.run([sys.executable, "_bench_worker.py"], capture_output=True, text=True,
                       env={**os.environ, "BENCH_MODE": mode, "HAL_DIR": HAL_DIR})
    line = [l for l in r.stdout.splitlines() if l.startswith("JSON")]
    if not line:
        print("FAILED", mode, r.stderr[-1500:]); continue
    res[mode] = json.loads(line[-1][4:])
json.dump(res, open("onecore.json", "w"), indent=2)

def g(mode, key, k, field="latency_s"):
    return res.get(mode, {}).get(f"{key}_{k}", {}).get(field)
def cell(x): return "       n/a" if x is None else f"{x*1e3:10.2f}"
S = (1, 100, 10000)
print(f"\nONE HOST THREAD (machine has {os.cpu_count()} vCPUs); median ms per call")
print(f"{'':44s}{'1 guide':>10s}{'100':>10s}{'10,000':>10s}   cpu/wall at 10,000")
rows = [("crisprHAL 2, CPU (best batch)", "cpu", "hal_host"), ("SLICER, CPU", "cpu", "slicer_host"),
        ("crisprHAL 2, GPU, input on host (best batch)", "gpu", "hal_host"),
        ("SLICER, GPU, input on host", "gpu", "slicer_host"),
        ("crisprHAL 2, GPU, input on card (best batch)", "gpu", "hal_device"),
        ("SLICER, GPU, input on card", "gpu", "slicer_device")]
for name, mode, key in rows:
    print(f"{name:44s}" + "".join(cell(g(mode, key, k)) for k in S)
          + (f"   {g(mode, key, 10000, 'cpu_over_wall'):.1f}" if g(mode, key, 10000, 'cpu_over_wall') else ""))
print("\nbest crisprHAL batch size:  " + "; ".join(
    f"{m} {key.split('_')[1]}: " + "/".join(str(g(m, key, k, 'batch_size')) for k in S)
    for m, key in (("cpu", "hal_host"), ("gpu", "hal_host"), ("gpu", "hal_device"))))
print("crisprHAL at Keras' default batch size 32 (ms):  " + "; ".join(
    f"{m}: " + "/".join(f"{g(m, 'hal_default32', k)*1e3:.0f}" for k in S if g(m, 'hal_default32', k))
    for m in ("cpu", "gpu") if m in res))
print("\nRatios, crisprHAL 2 / SLICER, same hardware:")
for lab, hm, hk, sm, sk in (("CPU vs CPU", "cpu", "hal_host", "cpu", "slicer_host"),
                            ("GPU vs GPU, input on host", "gpu", "hal_host", "gpu", "slicer_host"),
                            ("GPU vs GPU, input on card", "gpu", "hal_device", "gpu", "slicer_device")):
    print(f"  {lab:28s}" + "".join(
        f"{g(hm, hk, k) / g(sm, sk, k):9.0f}x" if g(hm, hk, k) and g(sm, sk, k) else "      n/a"
        for k in S))
print("\nwrote onecore.json -- paste this whole printout back")
