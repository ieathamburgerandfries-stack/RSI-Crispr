"""Measures CPU-seconds and floating-point operations to fit SLICER, a cut-down SLICER and crisprHAL 2; superseded by cost_repeats, as its crisprHAL 2 CPU-seconds were a hand-entered estimate.

Three things the report has been quoting loosely and should quote precisely.

**CPU-seconds, not wall-clock.** Wall-clock on a ten-core laptop confuses "fast"
with "parallel". `resource.getrusage` in a child process gives user+system CPU
time summed over every thread, which is the quantity that transfers to another
machine and the one a cloud bill is computed from.

**FLOPs, analytically.** Measured CPU time depends on BLAS, thread count and
cache; floating-point operation counts do not. They are computed from the
architecture here rather than profiled, with the assumptions written out, because
an assumption you can read beats a number you cannot reproduce. Training is taken
as 3x forward (one forward, one backward of roughly twice the forward cost) --
the standard convention.

**A minimum-cost configuration.** The report's 17 s/fold is the frozen baseline,
which was never chosen for speed: it uses an XGBoost ranking pass over 6,517
columns to pick 300, on a matrix that section 10 shows is 7% information. Strip
every avoidable cost and measure what is left.

    python -m sgrna.cost --what cpu      # measured CPU-seconds per configuration
    python -m sgrna.cost --what flops    # analytic FLOP counts
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

OUT_CPU = config.RESULTS / "cost_cpu_seconds.csv"
OUT_FLOPS = config.RESULTS / "cost_flops.csv"

ENV = {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8",
       "MKL_NUM_THREADS": "8", "PYTHONWARNINGS": "ignore"}

PREAMBLE = """
import json, resource, sys, time
import numpy as np
sys.path.insert(0, {src!r})

def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (r.ru_utime + r.ru_stime + c.ru_utime + c.ru_stime)

def peak_gb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**30 if sys.platform == "darwin" else rss / 2**20

_T0W, _T0C = time.time(), cpu_s()

def emit(**kw):
    kw["wall_s"] = round(time.time() - _T0W, 2)
    kw["cpu_s"] = round(cpu_s() - _T0C, 2)
    kw["peak_gb"] = round(peak_gb(), 2)
    print("RESULT " + json.dumps(kw))
"""


def _child(body: str) -> dict:
    code = PREAMBLE.format(src=str(config.ROOT / "src")) + textwrap.dedent(body)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env={**os.environ, **ENV})
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(f"no result\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")


# --------------------------------------------------------------------------
# Measured CPU-seconds, one fold, each configuration in a fresh process
# --------------------------------------------------------------------------

_LOAD = """
from sgrna import config
from sgrna.compare.headtohead import _assemble
X, y, names = _assemble("curated_flank", verbose=False)
"""

_REDUCE = """
import json as _json
_vmap = _json.loads((config.INTERIM / "v_column_mapping.json").read_text())["mapping"]
_keep = [i for i, n in enumerate(names)
         if len(_vmap.get(n, {}).get("kmer", "")) in (1, 2)
         or n.startswith("eng.flank.")]
X = np.ascontiguousarray(X[:, _keep])
names = [names[i] for i in _keep]
"""

_FOLD = """
from sklearn.model_selection import KFold
from sgrna.harness.run_ablation import _impute, make_selector, champion_params
from scipy.stats import spearmanr
tr, va = next(iter(KFold(5, shuffle=True, random_state=41).split(X)))
X_tr, X_va = _impute(X[tr], X[va])
"""

_SELECT = """
sel = make_selector(41, kind={selector!r})
sel.fit(X_tr, y[tr])
top = np.argsort(sel.feature_importances_)[::-1][:300]
X_tr, X_va = X_tr[:, top], X_va[:, top]
"""

_FIT = """
{fitter}
rho = float(spearmanr(y[va], pred).statistic)
emit(config={label!r}, columns=int(X.shape[1]), rho=round(rho, 4))
"""

_XGB = """
import xgboost as xgb
m = xgb.XGBRegressor(**champion_params(41)); m.fit(X_tr, y[tr])
pred = m.predict(X_va)
"""

_LGB = """
import lightgbm as lgb
m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=16,
                      max_depth=4, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, random_state=41, n_jobs=-1,
                      verbose=-1); m.fit(X_tr, y[tr])
pred = m.predict(X_va)
"""

_RIDGE = """
from sklearn.linear_model import Ridge
mu, sd = X_tr.mean(0), X_tr.std(0); sd[sd == 0] = 1.0
m = Ridge(alpha=10.0).fit((X_tr - mu) / sd, y[tr])
pred = m.predict((X_va - mu) / sd)
"""

CPU_CASES = [
    ("imports only (floor)", "import xgboost, lightgbm, sklearn, pandas\n"
                             "emit(config='imports only (floor)')"),
    ("as published: all columns, XGBoost select, XGBoost fit",
     _LOAD + _FOLD + _SELECT.format(selector="xgboost")
     + _FIT.format(fitter=_XGB, label="as published: all columns, XGBoost select, XGBoost fit")),
    ("all columns, LightGBM select, LightGBM fit",
     _LOAD + _FOLD + _SELECT.format(selector="lightgbm")
     + _FIT.format(fitter=_LGB, label="all columns, LightGBM select, LightGBM fit")),
    ("reduced columns, no selection, LightGBM fit",
     _LOAD + _REDUCE + _FOLD
     + _FIT.format(fitter=_LGB, label="reduced columns, no selection, LightGBM fit")),
    ("reduced columns, no selection, ridge",
     _LOAD + _REDUCE + _FOLD
     + _FIT.format(fitter=_RIDGE, label="reduced columns, no selection, ridge")),
]


def cpu(reps: int = 3, verbose: bool = True) -> pd.DataFrame:
    raw = []
    for label, body in CPU_CASES:
        if verbose:
            print(f"  {label} ...", flush=True)
        for _ in range(reps):
            row = _child(body)
            row.setdefault("config", label)
            raw.append(row)
        got = [r for r in raw if r["config"] == label]
        if verbose:
            print(f"    cpu {np.median([r['cpu_s'] for r in got]):.1f} s  "
                  f"wall {np.median([r['wall_s'] for r in got]):.1f} s  "
                  f"peak {np.median([r['peak_gb'] for r in got]):.2f} GB",
                  flush=True)
    df = pd.DataFrame(raw)
    agg = (df.groupby("config", sort=False)
             .agg(columns=("columns", "first"), rho=("rho", "median"),
                  cpu_s=("cpu_s", "median"), wall_s=("wall_s", "median"),
                  peak_gb=("peak_gb", "median"), reps=("cpu_s", "size"))
             .reset_index())
    agg["cpu_per_wall"] = (agg["cpu_s"] / agg["wall_s"]).round(2)
    agg.to_csv(OUT_CPU, index=False)
    return agg


# --------------------------------------------------------------------------
# Analytic FLOPs
# --------------------------------------------------------------------------


def flops_crisprhal(n_train: int = 26853, epochs: int = 48, length: int = 378,
                    filters: int = 128, kernel: int = 3, gru: int = 128,
                    dense=(128, 64)) -> dict:
    """Forward FLOPs for one sequence, then scaled to a training run.

    Architecture from `external_data/crisprHAL/models.py`: four Conv1D blocks
    (`filters` channels, `kernel` wide, 'same' padding) each followed by
    MaxPool1d(2); a bidirectional GRU of `gru` units reading the output of the
    first block; two dense heads. A multiply-accumulate is counted as 2 FLOPs.
    """
    f = 0
    L, c_in = length, 4
    conv_out_len = []
    for _ in range(4):
        f += 2 * L * kernel * c_in * filters          # conv
        c_in = filters
        L //= 2                                        # maxpool
        conv_out_len.append(L)
    f_conv = f
    # GRU reads the first block's output: length length//2, 128 channels.
    # 3 gates, each a matmul on [input; hidden] -> hidden, both directions.
    L_rnn = length // 2
    f_gru = 2 * (2 * 3 * L_rnn * (filters + gru) * gru)
    # dense heads: flatten of the conv stack, plus the GRU branch
    f_dense = 2 * (conv_out_len[-1] * filters * dense[0] + dense[0] * dense[1]
                   + dense[1] + 2 * gru * dense[0] + dense[0] * dense[1])
    fwd = f_conv + f_gru + f_dense
    return dict(model="crisprHAL 2", forward_flops_per_guide=fwd,
                train_flops=3 * fwd * n_train * epochs,
                note=f"4 conv blocks + BiGRU({gru}), {epochs} epochs, "
                     f"{n_train:,} guides, train = 3x forward")


def flops_slicer(n_train: int = 26853, n_cols: int = 6517, n_trees: int = 400,
                 depth: int = 4, colsample: float = 0.8,
                 n_select_trees: int = 300, select_cols: int | None = None,
                 n_final_cols: int = 300, label: str = "SLICER, as published",
                 ) -> dict:
    """FLOPs for histogram-based boosting.

    The dominant cost is histogram construction: for every tree, at every
    level, each row contributes one add into one bin per candidate column. So
    ~1 add per (row, column, level), plus a small constant per bin for the
    split scan. Binning the matrix once is n_rows x n_cols compares. Prediction
    is depth compares per tree, negligible.
    """
    select_cols = n_cols if select_cols is None else select_cols
    binning = n_train * n_cols
    hist_select = n_select_trees * depth * n_train * select_cols * colsample
    hist_final = n_trees * depth * n_train * n_final_cols * colsample
    total = binning + hist_select + hist_final
    return dict(model=label, forward_flops_per_guide=np.nan,
                train_flops=total,
                note=f"{n_select_trees} selector trees over {select_cols:,} cols "
                     f"+ {n_trees} champion trees over {n_final_cols} cols, "
                     f"depth {depth}, ~1 add per (row, col, level)")


def bytes_crisprhal(n_train: int = 26853, epochs: int = 48, length: int = 378,
                    filters: int = 128, kernel: int = 3, gru: int = 128) -> int:
    """Bytes moved for one training run: activations in and out, plus weights.

    Counted per layer as (input activations + output activations + weights) x 4,
    which is what a cache-naive implementation would touch; a fused kernel moves
    less. The point of the number is the *ratio* to FLOPs, so a consistent
    convention matters more than the absolute figure.
    """
    b = 0
    L, c_in = length, 4
    for _ in range(4):
        b += (L * c_in + L * filters + kernel * c_in * filters) * 4
        c_in = filters
        L //= 2
    L_rnn = length // 2
    b += (L_rnn * filters + L_rnn * 2 * gru + 3 * (filters + gru) * gru * 2) * 4
    return 3 * b * n_train * epochs


def bytes_slicer(n_train: int = 26853, n_cols: int = 6517, n_trees: int = 400,
                 depth: int = 4, colsample: float = 0.8,
                 n_select_trees: int = 300, select_cols: int | None = None,
                 n_final_cols: int = 300) -> int:
    """Bytes moved: one float32 read per (row, candidate column, tree level).

    Histogram construction is a streaming scan with almost no reuse -- each
    value is read, binned and discarded -- which is why this pipeline is
    bandwidth-bound rather than compute-bound.
    """
    select_cols = n_cols if select_cols is None else select_cols
    reads = (n_select_trees * depth * n_train * select_cols * colsample
             + n_trees * depth * n_train * n_final_cols * colsample)
    return int(4 * reads + 4 * n_train * n_cols)


# Cloud rates, October 2026, on-demand, for the cheapest instance that fits.
# Quoted so the arithmetic is checkable, not because they are stable.
RATE_CPU_CORE_HOUR = 0.04      # GBP, general-purpose vCPU
RATE_GPU_HOUR = 0.45           # GBP, one T4/L4-class accelerator


def panel(verbose: bool = True) -> pd.DataFrame:
    """Every cost measure this project can compute, with its bias stated."""
    rows = []
    for label, f, byt, cpu_s, wall_s, cores, gb in (
        ("crisprHAL 2, published", flops_crisprhal()["train_flops"],
         bytes_crisprhal(), 10000.0, 1266.0, 8, 4.55),
        ("crisprHAL 2, tuned",
         flops_crisprhal(epochs=64, filters=192, gru=192,
                         dense=(64, 64))["train_flops"],
         bytes_crisprhal(epochs=64, filters=192, gru=192), 32000.0, 4148.0, 8, 4.6),
        ("SLICER, as published", flops_slicer()["train_flops"],
         bytes_slicer(), 65.8, 13.8, 8, 3.76),
        ("SLICER, minimum cost", flops_slicer(
            n_cols=712, select_cols=0, n_select_trees=0,
            n_final_cols=712)["train_flops"],
         bytes_slicer(n_cols=712, select_cols=0, n_select_trees=0,
                      n_final_cols=712), 7.5, 3.6, 8, 2.83),
        ("SLICER, ridge on reduced columns", np.nan, np.nan, 3.4, 2.9, 8, 3.09),
    ):
        rows.append(dict(
            model=label, train_flops=f, bytes_moved=byt,
            flops_per_byte=(f / byt if f == f and byt == byt else np.nan),
            cpu_seconds=cpu_s, wall_seconds=wall_s, peak_gb=gb,
            cpu_per_wall=round(cpu_s / wall_s, 2),
            gbp_per_fold_cpu=round(cpu_s / 3600 * RATE_CPU_CORE_HOUR, 4),
        ))
    df = pd.DataFrame(rows)
    base = df.loc[df["model"] == "crisprHAL 2, published"]
    for col in ("train_flops", "bytes_moved", "cpu_seconds", "wall_seconds"):
        df[f"ratio_{col}"] = (base[col].iloc[0] / df[col]).round(1)
    df.to_csv(config.RESULTS / "cost_panel.csv", index=False)
    if verbose:
        show = ["model", "train_flops", "bytes_moved", "flops_per_byte",
                "cpu_seconds", "wall_seconds", "peak_gb", "cpu_per_wall"]
        print(df[show].to_string(index=False))
    return df


def flops(verbose: bool = True) -> pd.DataFrame:
    rows = [
        flops_crisprhal(),
        flops_crisprhal(epochs=64, filters=192, gru=192,
                        dense=(64, 64)) | {"model": "crisprHAL 2, tuned"},
        flops_slicer(),
        flops_slicer(n_cols=712, select_cols=0, n_select_trees=0,
                     n_final_cols=712, label="SLICER, reduced, no selection"),
    ]
    df = pd.DataFrame(rows)
    base = df.loc[df["model"] == "crisprHAL 2", "train_flops"].iloc[0]
    df["vs_crisprHAL"] = (base / df["train_flops"]).round(1)
    df.to_csv(OUT_FLOPS, index=False)
    if verbose:
        for _, r in df.iterrows():
            print(f"  {r['model']:34s} {r['train_flops']:.3e} FLOPs  "
                  f"({r['vs_crisprHAL']:.1f}x lighter than crisprHAL)")
            print(f"      {r['note']}")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", choices=["cpu", "flops", "panel", "both"],
                    default="both")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args(argv)
    if args.what == "panel":
        print("=== every cost measure, one training fold ===")
        panel()
        return 0
    if args.what in ("flops", "both"):
        print("=== analytic FLOPs for one training run ===")
        flops()
    if args.what in ("cpu", "both"):
        print("\n=== measured CPU-seconds, one fold, median of reps ===")
        print(cpu(reps=args.reps).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
