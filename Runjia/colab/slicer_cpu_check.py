"""Why does SLICER's small-batch CPU time differ between two Colab runs?

Run 1 (model trained in the same process):      0.48 / 1.36 / 72.0 ms  for 1 / 100 / 10,000 guides
Run 2 (model saved, loaded in a fresh process): 8.00 / 8.01 / 98.0 ms

Same instance type, same configuration, a 16x gap at one guide. This cell times the
same model four ways on one instance, twice each, so the cause (or run-to-run noise)
shows. Paste into ONE Colab cell (CPU or T4 runtime; it only uses the CPU). ~2 min.
"""
import subprocess, sys, time, os, json
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "xgboost"], check=True)
import numpy as np, xgboost as xgb
print("xgboost", xgb.__version__, " vCPUs", os.cpu_count())
rng = np.random.default_rng(41); n, d = 20000, 300
X = rng.normal(size=(n, d)).astype(np.float32); w = rng.normal(size=d)
y = (X @ w + 0.5 * X[:, 0] * X[:, 1] + rng.normal(size=n)).astype(np.float32)
Xs = rng.normal(size=(10000, d)).astype(np.float32)
kw = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
          colsample_bytree=0.8, tree_method="hist", random_state=41)

def med(fn, reps):
    fn(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3

trained = xgb.XGBRegressor(n_jobs=-1, device="cpu", **kw).fit(X, y)
trained.save_model("m.json")
loaded = xgb.XGBRegressor(); loaded.load_model("m.json")
loaded_dev = xgb.XGBRegressor(); loaded_dev.load_model("m.json"); loaded_dev.set_params(device="cpu")
loaded_nj = xgb.XGBRegressor(n_jobs=-1); loaded_nj.load_model("m.json")
loaded_nj.set_params(device="cpu")
bst = trained.get_booster()
variants = {
    "trained in-process, n_jobs=-1, device=cpu": lambda b: trained.predict(b),
    "loaded, defaults": lambda b: loaded.predict(b),
    "loaded, set_params(device=cpu)  [run 2's path]": lambda b: loaded_dev.predict(b),
    "loaded, n_jobs=-1 + device=cpu": lambda b: loaded_nj.predict(b),
    "booster.inplace_predict (no sklearn wrapper)": lambda b: bst.inplace_predict(b),
}
print(f"\n{'ms per call (median)':50s}{'1':>9s}{'100':>9s}{'10,000':>9s}")
for rep in (1, 2):
    for name, f in variants.items():
        row = [med(lambda b=Xs[:k]: f(b), 9 if k < 10000 else 5) for k in (1, 100, 10000)]
        print(f"{name + f'  (pass {rep})':50s}" + "".join(f"{v:9.2f}" for v in row))
for nt in (1, 2):
    bst.set_param({"nthread": nt})
    row = [med(lambda b=Xs[:k]: bst.inplace_predict(b), 9 if k < 10000 else 5) for k in (1, 100, 10000)]
    print(f"{f'inplace_predict, nthread={nt}':50s}" + "".join(f"{v:9.2f}" for v in row))
print("\npaste this whole printout back")
