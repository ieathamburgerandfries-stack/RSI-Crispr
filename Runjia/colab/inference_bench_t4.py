"""Like-for-like forward-pass timing on ONE Colab instance: crisprHAL 2 vs SLICER,
on the CPU and on the GPU, with cold starts.

Paste into ONE Colab cell. Runtime > Change runtime type > T4 GPU. Takes ~10 min.
It prints a table and writes inference_colab.json (paste the printout back).

Four cells of the comparison, each pair on the same hardware type:
    CPU  : crisprHAL 2 (TensorFlow, GPU hidden)   vs  SLICER (xgboost, device=cpu)
    GPU  : crisprHAL 2 (TensorFlow on the T4)     vs  SLICER (xgboost, device=cuda)
For SLICER on the GPU two input placements are timed, because the host-to-device
copy is part of the cost of a request and not part of the cost of a batch that is
already on the card: 'host' (numpy in, as a web request would arrive) and
'device' (cupy array already on the card).

Cold start = fresh process: import + load model + first prediction, which on a
GPU includes CUDA/cuDNN initialisation. Each cold start is its own subprocess.

SLICER is the champion's configuration (400 trees, depth 4, 300 columns, hist)
trained on synthetic data of the right shape. Prediction cost of a tree ensemble
depends on the number of trees and their depth, not on the values, so synthetic
data is a fair stand-in for timing and says nothing about accuracy.
"""
import json, os, subprocess, sys, textwrap

if not os.path.isdir("/content/crisprHAL"):
    subprocess.run(["git", "clone", "-q", "https://github.com/tbrowne5/crisprHAL",
                    "/content/crisprHAL"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "xgboost"], check=True)

HAL_WORKER = textwrap.dedent('''
    import os, sys, time, json
    mode = sys.argv[1]
    if mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    HAL = "/content/crisprHAL"
    t_all = time.perf_counter()
    import numpy as np
    sys.path.insert(0, HAL); os.chdir(HAL)
    import tensorflow as tf
    from processing import processing
    t_imp = time.perf_counter() - t_all
    t0 = time.perf_counter()
    m = tf.keras.models.load_model(HAL + "/models/WT-SPCAS9.keras", compile=False)
    t_load = time.perf_counter() - t0
    proc = processing(); L = m.input_shape[1]
    rng = np.random.default_rng(41); bases = np.array(list("ACGT"))
    seqs = np.array(["".join(rng.choice(bases, L)) for _ in range(10000)])
    X = proc.onehotencode(seqs)
    t0 = time.perf_counter(); m.predict(X[:1], verbose=0); t_first = time.perf_counter() - t0
    out = dict(devices=[d.name for d in tf.config.list_physical_devices("GPU")],
               cold=dict(import_s=t_imp, load_s=t_load, first_predict_s=t_first,
                         total_s=t_imp + t_load + t_first))
    for n in (1, 100, 10000):
        b = X[:n]; m.predict(b, verbose=0)
        ts = []
        for _ in range(3 if n < 10000 else 2):
            t0 = time.perf_counter(); m.predict(b, verbose=0); ts.append(time.perf_counter() - t0)
        med = float(np.median(ts)); out[str(n)] = dict(latency_s=med, guides_per_s=n / med)
    print("JSON" + json.dumps(out))
''')

SLICER_TRAIN = textwrap.dedent('''
    import numpy as np, xgboost as xgb
    rng = np.random.default_rng(41); n, d = 20000, 300
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    y = (X @ w + 0.5 * X[:, 0] * X[:, 1] + rng.normal(size=n)).astype(np.float32)
    m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         n_jobs=-1, random_state=41)
    m.fit(X, y); m.save_model("slicer_synth.json")
    np.save("slicer_X.npy", rng.normal(size=(10000, d)).astype(np.float32))
''')

SLICER_WORKER = textwrap.dedent('''
    import sys, time, json
    dev = sys.argv[1]
    t_all = time.perf_counter()
    import numpy as np, xgboost as xgb
    t_imp = time.perf_counter() - t_all
    X = np.load("slicer_X.npy")
    t0 = time.perf_counter()
    m = xgb.XGBRegressor(); m.load_model("slicer_synth.json"); m.set_params(device=dev)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter(); m.predict(X[:1]); t_first = time.perf_counter() - t0
    out = dict(cold=dict(import_s=t_imp, load_s=t_load, first_predict_s=t_first,
                         total_s=t_imp + t_load + t_first))
    def timeit(fn, n):
        fn(); ts = []
        for _ in range(7 if n < 10000 else 5):
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        return float(np.median(ts))
    for n in (1, 100, 10000):
        b = X[:n]; med = timeit(lambda: m.predict(b), n)
        out["host_" + str(n)] = dict(latency_s=med, guides_per_s=n / med)
    if dev == "cuda":
        try:
            import cupy as cp
            for n in (1, 100, 10000):
                bd = cp.asarray(X[:n]); bst = m.get_booster()
                def f():
                    bst.inplace_predict(bd); cp.cuda.Stream.null.synchronize()
                med = timeit(f, n)
                out["device_" + str(n)] = dict(latency_s=med, guides_per_s=n / med)
        except Exception as e:
            out["device_error"] = repr(e)
    print("JSON" + json.dumps(out))
''')

def run(code, *args):
    open("_w.py", "w").write(code)
    r = subprocess.run([sys.executable, "_w.py", *args], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("JSON")]
    if not line:
        print("FAILED", args, r.stderr[-1500:]); return None
    return json.loads(line[-1][4:])

res = {}
print("crisprHAL 2 on GPU ..."); res["hal_gpu"] = run(HAL_WORKER, "gpu")
print("crisprHAL 2 on CPU ..."); res["hal_cpu"] = run(HAL_WORKER, "cpu")
run(SLICER_TRAIN)
print("SLICER on CPU ...");      res["slicer_cpu"] = run(SLICER_WORKER, "cpu")
print("SLICER on GPU ...");      res["slicer_gpu"] = run(SLICER_WORKER, "cuda")
res["cpu_count"] = os.cpu_count()
json.dump(res, open("inference_colab.json", "w"), indent=2)

def g(k, key):
    r = res.get(k) or {}
    return r.get(key, {}).get("latency_s")
def ms(x): return "      n/a" if x is None else f"{x*1e3:9.2f}"
print(f"\nvCPUs: {res['cpu_count']}   (ms per call; median)")
print(f"{'':34s}{'1 guide':>10s}{'100':>10s}{'10,000':>10s}   cold start (s)")
rows = [("crisprHAL 2  CPU", "hal_cpu", ""), ("SLICER       CPU", "slicer_cpu", "host_"),
        ("crisprHAL 2  GPU", "hal_gpu", ""), ("SLICER       GPU, host input", "slicer_gpu", "host_"),
        ("SLICER       GPU, device input", "slicer_gpu", "device_")]
for name, k, pre in rows:
    cold = (res.get(k) or {}).get("cold", {}).get("total_s")
    print(f"{name:34s}" + "".join(ms(g(k, pre + str(n))) for n in (1, 100, 10000))
          + ("" if cold is None or pre == "device_" else f"   {cold:8.2f}"))
print("\nSame-hardware ratios (crisprHAL 2 / SLICER):")
for lab, h, s, pre in (("CPU vs CPU", "hal_cpu", "slicer_cpu", "host_"),
                       ("GPU vs GPU, host input", "hal_gpu", "slicer_gpu", "host_"),
                       ("GPU vs GPU, device input", "hal_gpu", "slicer_gpu", "device_")):
    print(f"  {lab:26s}" + "".join(
        f"{(g(h, str(n)) / g(s, pre + str(n))):9.0f}x" if g(s, pre + str(n)) else "      n/a"
        for n in (1, 100, 10000)))
print("\nwrote inference_colab.json -- paste this whole printout back")
