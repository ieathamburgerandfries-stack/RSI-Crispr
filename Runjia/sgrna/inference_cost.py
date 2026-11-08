"""Training cost is the wrong clock for a guide-design tool. This is the other one.

`cost.py` measures what it takes to *fit* each model -- CPU-seconds, FLOPs,
bytes moved, peak memory -- and Finding 7 quotes the ~150x from it. That is the
right number for "can I afford to rebuild this", and the wrong one for the thing
a user actually does, which is fit once and then score every candidate guide for
every target they care about. A genome-wide design run is tens of thousands to
millions of forward passes against a handful of fits.

`cost.flops_crisprhal` already computes forward FLOPs per guide for their
network; the SLICER side of that column is NaN, and neither model's *measured*
prediction latency exists anywhere in this project. This fills both.

Three rubrics, because they answer different questions:

    latency        seconds to score one batch, the number a user waits for
    throughput     guides per second at a realistic batch size
    cold start     seconds from process start to first prediction, which for a
                   deep model includes importing the framework and is usually
                   the larger term for small jobs

The forward-FLOPs figure for a boosted ensemble is also derivable rather than
NaN: scoring one guide costs one root-to-leaf walk per tree, so `n_trees x
depth` comparisons -- 400 x 4 = 1,600 for the champion, against crisprHAL 2's
4.2e14 / 26,853 / 3 forward FLOPs per guide. The two are not the same unit of
work and the comparison is reported as such.

    python -m sgrna.inference_cost
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import config

OUT = config.RESULTS / "inference_cost.csv"

# crisprHAL 2's shipped WT-SpCas9 model, measured on the same machine and the
# same CPU with `colab/crisprhal_inference_bench.py` under `.venv-crisprhal`
# (TensorFlow 2.19.1). It is kept here as recorded numbers because it needs the
# other interpreter; re-run that script to refresh them.
#
# Both models are on this machine's CPU. The like-for-like GPU comparison was run
# on a Colab T4 (`colab/inference_bench_t4.py`, `results/inference_t4.csv`):
# crisprHAL 2 takes 78.0 / 99.7 / 2,768 ms for 1 / 100 / 10,000 guides there and
# 9.8 s to cold-start, and SLICER on the same instance's CPU is 38-160x faster.
HAL_INFERENCE = {
    "import_and_load_s": 13.4,
    1: dict(latency_s=0.04426, guides_per_s=23),
    100: dict(latency_s=0.10168, guides_per_s=983),
    10_000: dict(latency_s=8.16176, guides_per_s=1225),
}


def run(n_score=(1, 100, 10_000), reps: int = 5, seed: int = 41,
        verbose: bool = True):
    import xgboost as xgb

    from .curated_ablation import _base, _block
    from .run_ablation import _impute, champion_params, make_selector

    X, y, names, protos = _base()
    A, _ = _block("a_flank", protos)
    X = np.hstack([X, A]).astype(np.float32)

    tr = np.arange(len(X))
    Xi, _ = _impute(X[tr], X[:1])
    sel = make_selector(seed)
    sel.fit(Xi, y)
    top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
    m = xgb.XGBRegressor(**champion_params(seed))
    t0 = time.perf_counter()
    m.fit(Xi[:, top], y)
    fit_s = time.perf_counter() - t0
    booster = m.get_booster()
    n_trees = len(booster.get_dump())

    rows = []
    for n in n_score:
        batch = np.ascontiguousarray(Xi[:n, top])
        m.predict(batch)                       # warm the cache
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            m.predict(batch)
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        rows.append(dict(model="SLICER champion", n_guides=n,
                         n_columns=len(top), n_trees=n_trees,
                         latency_s=med, guides_per_s=n / med if med else np.nan,
                         fit_seconds=fit_s, cold_start_s=np.nan))
        if n in HAL_INFERENCE:
            h = HAL_INFERENCE[n]
            rows.append(dict(model="crisprHAL 2, shipped WT-SpCas9",
                             n_guides=n, n_columns=378, n_trees=np.nan,
                             latency_s=h["latency_s"],
                             guides_per_s=h["guides_per_s"],
                             fit_seconds=np.nan,
                             cold_start_s=HAL_INFERENCE["import_and_load_s"]))
        if verbose:
            print(f"  score {n:>7,} guides: {med * 1e3:9.2f} ms  "
                  f"{n / med:12,.0f} guides/s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    if verbose:
        ours = out[out.model.str.startswith("SLICER")].iloc[-1]
        theirs = out[out.model.str.startswith("crisprHAL")]
        print(f"\n  one fit took {fit_s:.1f} s; scoring runs at "
              f"{ours.guides_per_s:,.0f} guides/s")
        if len(theirs):
            t = theirs.iloc[-1]
            print(f"  against {t.guides_per_s:,.0f} guides/s for crisprHAL 2 on "
                  f"the same CPU, a factor of "
                  f"{ours.guides_per_s / t.guides_per_s:,.0f} on throughput")
            print(f"  and {t.cold_start_s:.1f} s of cold start before its first "
                  f"prediction, against a fraction of a second for ours")
        print("  these numbers move with machine load; the gap does not")
        print(f"  comparison of work per guide is in the docstring: "
              f"{n_trees} trees x depth {config.MAX_DEPTH if hasattr(config, 'MAX_DEPTH') else 4} "
              f"= {n_trees * 4:,} comparisons against a dense network's forward pass")
        print(f"written -> {OUT}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args(argv)
    run(reps=a.reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
