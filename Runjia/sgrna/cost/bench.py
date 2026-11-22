"""Peak memory and training cost, measured rather than estimated.

Two questions this project got asked and could not answer from the code it had,
because the original measurements were made in a throwaway script:

  memory   What does one fold of the pipeline actually cost in RAM, and can a
           different model or a different matrix representation reduce it?
  netcost  Does a neural network get much cheaper if it is given the reduced
           column set from `representation.py` instead of all 6,517 columns?

Both run each configuration in a **separate process** and read peak RSS from
`resource.getrusage(RUSAGE_SELF).ru_maxrss` in that child. Measuring peak
memory in-process is meaningless once an earlier configuration has already
allocated the high-water mark, and `ru_maxrss` never goes down.

Threads are pinned (`OMP_NUM_THREADS=4`) because this machine has 10 cores and
16 GB: an unpinned run swaps, and a swapping run reports both time and memory
that belong to the operating system rather than the model. That is the same
precaution behind `timing_controlled.csv`.

    python -m sgrna.cost.bench --what memory
    python -m sgrna.cost.bench --what netcost
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd

from .. import config

OUT_MEMORY = config.RESULTS / "memory_comparison.csv"
OUT_NETCOST = config.RESULTS / "net_cost.csv"

ENV = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
       "MKL_NUM_THREADS": "4", "PYTHONWARNINGS": "ignore"}

# The child prints one JSON line on stdout; everything else is noise.
PREAMBLE = """
import json, resource, sys, time
import numpy as np
sys.path.insert(0, {src!r})


def peak_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kibibytes.
    return rss / 2**30 if sys.platform == "darwin" else rss / 2**20


def emit(**kw):
    kw["peak_gb"] = round(peak_gb(), 2)
    print("RESULT " + json.dumps(kw))
"""


def _child(body: str) -> dict:
    src = str(config.ROOT / "src")
    code = PREAMBLE.format(src=src) + textwrap.dedent(body)
    env = {**os.environ, **ENV}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(
        f"child produced no result\n--- stdout ---\n{proc.stdout[-2000:]}"
        f"\n--- stderr ---\n{proc.stderr[-2000:]}")


# --------------------------------------------------------------------------
# The shared loader body: the expanded matrix one fold of the pipeline sees
# --------------------------------------------------------------------------

_LOAD = """
from sgrna import config
from sgrna.compare.headtohead import _assemble
X, y, names = _assemble("curated_flank", verbose=False)
"""

_SUBSET = """
import json as _json
_vmap = _json.loads((config.INTERIM / "v_column_mapping.json").read_text())["mapping"]
_keep = [i for i, n in enumerate(names)
         if len(_vmap.get(n, {}).get("kmer", "")) in (1, 2)
         or n.startswith("eng.flank.")]
X = np.ascontiguousarray(X[:, _keep])
names = [names[i] for i in _keep]
"""

# Loading *only* the reduced columns, which is the configuration a project that
# adopted `representation.py`'s result would actually run. The point of the
# separate arm: subsetting after the full matrix is in memory cannot help, because
# peak RSS is already set by then. The saving is only available if the reduced
# matrix is what gets stored and read.
_LOAD_STORED_SUBSET = """
import json as _json
import pandas as _pd
from sgrna import config

_cache = config.INTERIM / "bench_reduced_X.npy"
_cache_meta = config.INTERIM / "bench_reduced_meta.json"
if not (_cache.exists() and _cache_meta.exists()):
    # One-off: build the reduced store. Not part of the measured arm -- a
    # deployment would ship this file, not rebuild it.
    meta = _json.loads((config.INTERIM / "expanded_meta.json").read_text())
    names_all = list(meta["feature_names"])
    vmap = _json.loads(
        (config.INTERIM / "v_column_mapping.json").read_text())["mapping"]
    cols = [i for i, n in enumerate(names_all)
            if len(vmap.get(n, {}).get("kmer", "")) in (1, 2)]
    Xa = np.load(config.INTERIM / "expanded_X.npy", mmap_mode="r")
    out = np.empty((Xa.shape[0], len(cols)), dtype=np.float32)
    for lo in range(0, Xa.shape[0], 2000):      # chunked: never a full copy
        hi = min(lo + 2000, Xa.shape[0])
        out[lo:hi] = Xa[lo:hi][:, cols]
    protos = np.asarray(meta["protospacers"])
    from sgrna.compare.headtohead import FLANK_PATH
    blk = (_pd.read_csv(FLANK_PATH).set_index(config.ID_COL)
           .reindex(protos))                 # keyed by protospacer, like _assemble
    vals = blk.to_numpy(dtype=np.float32)
    good = ~np.all(np.isnan(vals), axis=0)
    out = np.hstack([out, vals[:, good]])
    np.save(_cache, out)
    _cache_meta.write_text(_json.dumps({
        "feature_names": [names_all[i] for i in cols]
                         + [c for c, k in zip(blk.columns, good) if k],
        "y": meta["y"]}))
    del out, Xa, vals

_m = _json.loads(_cache_meta.read_text())
X = np.load(_cache)
y = np.asarray(_m["y"], dtype=float)
names = list(_m["feature_names"])
"""

_FIT = """
from sklearn.model_selection import KFold
from sgrna.harness.run_ablation import _impute, make_selector, champion_params
import xgboost as xgb
tr, va = next(iter(KFold(n_splits=5, shuffle=True, random_state=41).split(X)))
X_tr, X_va = _impute(X[tr], X[va])
t0 = time.time()
sel = make_selector(41, kind={selector!r})
sel.fit(X_tr, y[tr])
top = np.argsort(sel.feature_importances_)[::-1][:min(300, X.shape[1])]
select_s = round(time.time() - t0, 1)
m = xgb.XGBRegressor(**champion_params(41))
m.fit(X_tr[:, top], y[tr])
from scipy.stats import spearmanr
rho = float(spearmanr(y[va], m.predict(X_va[:, top])).statistic)
emit(config={label!r}, columns=int(X.shape[1]),
     select_s=select_s, rho=round(rho, 4))
"""

MEMORY_CASES = [
    ("imports only (floor)", """
        import xgboost, lightgbm, sklearn, pandas
        emit(config="imports only (floor)")
    """),
    ("all columns + XGBoost selector (current)",
     _LOAD + _FIT.format(selector="xgboost",
                         label="all columns + XGBoost selector (current)")),
    ("all columns + LightGBM selector",
     _LOAD + _FIT.format(selector="lightgbm",
                         label="all columns + LightGBM selector")),
    ("reduced columns + XGBoost selector",
     _LOAD + _SUBSET + _FIT.format(selector="xgboost",
                                   label="reduced columns + XGBoost selector")),
    ("reduced columns + LightGBM selector",
     _LOAD + _SUBSET + _FIT.format(selector="lightgbm",
                                   label="reduced columns + LightGBM selector")),
    ("reduced columns, stored reduced + XGBoost",
     _LOAD_STORED_SUBSET + _FIT.format(
         selector="xgboost",
         label="reduced columns, stored reduced + XGBoost")),
    ("reduced columns, stored reduced + LightGBM",
     _LOAD_STORED_SUBSET + _FIT.format(
         selector="lightgbm",
         label="reduced columns, stored reduced + LightGBM")),
]


def memory(reps: int = 3, verbose: bool = True) -> pd.DataFrame:
    """Peak RSS per configuration, each repetition in its own fresh process.

    `reps` is not optional in practice. Two single-shot runs of the identical
    configuration came back 2.95 GB and 3.29 GB, so run-to-run spread is around
    0.3 GB -- larger than the gap between several of the arms. A single
    measurement here would support a ranking the data does not.
    """
    raw = []
    for label, body in MEMORY_CASES:
        if verbose:
            print(f"  {label} ...", flush=True)
        for rep in range(reps):
            row = _child(body)
            row.setdefault("config", label)
            row["rep"] = rep + 1
            raw.append(row)
        peaks = [r["peak_gb"] for r in raw if r["config"] == label]
        if verbose:
            print(f"    peak {min(peaks):.2f}-{max(peaks):.2f} GB "
                  f"(median {float(np.median(peaks)):.2f})", flush=True)
    df = pd.DataFrame(raw)
    agg = (df.groupby("config", sort=False)
             .agg(columns=("columns", "first"),
                  peak_gb_median=("peak_gb", "median"),
                  peak_gb_min=("peak_gb", "min"),
                  peak_gb_max=("peak_gb", "max"),
                  select_s_median=("select_s", "median"),
                  rho=("rho", "median"),
                  reps=("peak_gb", "size"))
             .reset_index())
    agg.to_csv(OUT_MEMORY, index=False)
    df.to_csv(config.RESULTS / "memory_comparison_runs.csv", index=False)
    return agg


# --------------------------------------------------------------------------
# netcost: does a neural net get much cheaper on the reduced column set?
# --------------------------------------------------------------------------

_NET = """
import torch
from torch import nn
torch.manual_seed(41)
torch.set_num_threads(4)
from sklearn.model_selection import KFold
from sgrna.harness.run_ablation import _impute
from scipy.stats import spearmanr

tr, va = next(iter(KFold(n_splits=5, shuffle=True, random_state=41).split(X)))
X_tr, X_va = _impute(X[tr], X[va])
mu, sd = X_tr.mean(0), X_tr.std(0)
sd[sd == 0] = 1.0
X_tr = (X_tr - mu) / sd
X_va = (X_va - mu) / sd
ymu, ysd = y[tr].mean(), y[tr].std()

net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
Xt = torch.from_numpy(np.ascontiguousarray(X_tr))
yt = torch.from_numpy(((y[tr] - ymu) / ysd).astype("float32")).unsqueeze(1)
t0 = time.time()
for epoch in range(20):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 256):
        b = perm[i:i + 256]
        opt.zero_grad()
        nn.functional.mse_loss(net(Xt[b]), yt[b]).backward()
        opt.step()
train_s = round(time.time() - t0, 1)
net.eval()
with torch.no_grad():
    pred = net(torch.from_numpy(np.ascontiguousarray(X_va))).squeeze(1).numpy()
rho = float(spearmanr(y[va], pred).statistic)
emit(config={label!r}, columns=int(X.shape[1]), params=sum(p.numel() for p in net.parameters()),
     train_s=train_s, rho=round(rho, 4))
"""

NETCOST_CASES = [
    ("dense net, all columns", _LOAD + _NET.format(label="dense net, all columns")),
    ("dense net, reduced columns",
     _LOAD + _SUBSET + _NET.format(label="dense net, reduced columns")),
]


def netcost(verbose: bool = True) -> pd.DataFrame:
    """Same net, same epochs, two column counts.

    The honest framing matters here. **The sequence CNN in `seqnet.py` never
    sees this matrix** -- it reads raw flanking DNA, so a reduced column set
    cannot speed it up; its cost is epochs x convolutions over sequence length.
    The question a reduced column set *can* answer is about a net that takes the
    feature table as input, which is what this measures: a 3-layer dense net,
    identical architecture apart from its input width, 20 epochs, batch 256.
    """
    rows = []
    for label, body in NETCOST_CASES:
        if verbose:
            print(f"  {label} ...", flush=True)
        row = _child(body)
        row.setdefault("config", label)
        rows.append(row)
        if verbose:
            print(f"    {row['train_s']} s, peak {row['peak_gb']} GB, "
                  f"rho {row['rho']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_NETCOST, index=False)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", choices=["memory", "netcost", "both"],
                    default="both")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args(argv)
    if args.what in ("memory", "both"):
        print("=== peak memory, one fold, each rep in its own process ===")
        print(memory(reps=args.reps).to_string(index=False))
    if args.what in ("netcost", "both"):
        print("\n=== dense net cost, all columns vs reduced ===")
        print(netcost().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
