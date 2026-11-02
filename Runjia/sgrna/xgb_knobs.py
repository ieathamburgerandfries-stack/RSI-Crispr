"""Which XGBoost knobs has this project never touched, and do they matter?

The champion sets eight of XGBoost's roughly sixteen real tuning parameters and
inherits the rest from the library. The 627-draw search that the report quotes
was run on **LightGBM** over eight of its own knobs (`tune.py`), so several
XGBoost parameters have never been varied by anything here at any point:

    gamma                minimum loss reduction to make a split
    reg_alpha            L1 penalty on leaf weights (set to 0.0, never moved)
    colsample_bylevel    column subsampling per depth level
    colsample_bynode     column subsampling per split
    max_bin              histogram resolution
    grow_policy          depthwise against lossguide (and its max_leaves)
    max_delta_step       cap on a leaf's weight update
    num_parallel_tree    trees per boosting round (random-forest mode)

Three of those are *a priori* plausible on this matrix and that is the reason to
measure rather than assume: the base is wide and highly redundant, which is what
per-level and per-node column subsampling exist for; and it is largely binary and
sparse, which is what `max_bin` interacts with.

**The design is one knob at a time around the champion**, not a joint search.
That cannot find an interaction between two untouched knobs, and it is not meant
to: the question is whether any of them moves the number at all, and a sweep
answers that for a twentieth of the cost. Feature selection is model-independent
here -- the selector is its own fixed XGBoost -- so the 300 columns are chosen
once per fold and reused by every arm, which is what makes the sweep cheap.

    python -m sgrna.xgb_knobs
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from . import config

OUT = config.RESULTS / "xgb_knobs.csv"

# Everything the champion does not set, with values either side of the default.
SWEEP = {
    "gamma": [0.0, 0.1, 1.0, 10.0],
    "reg_alpha": [0.0, 0.1, 1.0, 10.0],
    "colsample_bylevel": [1.0, 0.8, 0.5],
    "colsample_bynode": [1.0, 0.8, 0.5],
    "max_bin": [256, 64, 512],
    "max_delta_step": [0, 1],
    "num_parallel_tree": [1, 4],
}
# grow_policy needs max_leaves alongside it, so it is swept as pairs.
GROW = [dict(grow_policy="depthwise"),
        dict(grow_policy="lossguide", max_depth=0, max_leaves=16),
        dict(grow_policy="lossguide", max_depth=0, max_leaves=63)]


def _folds(X, y, seed=41, n_splits=5, n_features=None):
    """Select once per fold; every arm then sees the same columns."""
    from .run_ablation import _impute, make_selector

    nf = config.N_FEATURES if n_features is None else n_features
    out = []
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        A, B = _impute(X[tr], X[va])
        sel = make_selector(seed)
        sel.fit(A, y[tr])
        top = np.argsort(sel.feature_importances_)[::-1][:nf]
        out.append((A[:, top], y[tr], B[:, top], y[va]))
    return out


def _score(folds, params, seed=41):
    import xgboost as xgb

    rhos = []
    for A, ytr, B, yva in folds:
        m = xgb.XGBRegressor(**params)
        m.fit(A, ytr)
        rhos.append(float(spearmanr(yva, m.predict(B)).statistic))
    return float(np.mean(rhos)), float(np.std(rhos, ddof=1))


def run(with_flank: bool = True, seed: int = 41, verbose: bool = True):
    from .curated_ablation import _base, _block
    from .run_ablation import champion_params

    X, y, names, protos = _base()
    if with_flank:
        v, _ = _block("a_flank", protos)
        X = np.hstack([X, v])
    if verbose:
        print(f"  matrix {X.shape[0]:,} x {X.shape[1]:,}; selecting once per fold",
              flush=True)
    t0 = time.time()
    folds = _folds(X, y, seed=seed)
    if verbose:
        print(f"  selection done in {time.time() - t0:.0f} s", flush=True)

    champ = dict(champion_params(seed))
    ref, ref_sd = _score(folds, champ, seed)
    rows = [dict(knob="(champion)", value="as configured", spearman=ref,
                 spearman_sd=ref_sd, d_spearman=0.0, is_default=True)]
    if verbose:
        print(f"  champion: rho {ref:.4f}", flush=True)

    for knob, values in SWEEP.items():
        for val in values:
            p = dict(champ)
            p[knob] = val
            rho, sd = _score(folds, p, seed)
            rows.append(dict(knob=knob, value=val, spearman=rho,
                             spearman_sd=sd, d_spearman=rho - ref,
                             is_default=(val == values[0])))
            if verbose:
                print(f"  {knob:20s} = {str(val):>9}: rho {rho:.4f}  "
                      f"d {rho - ref:+.4f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)

    for g in GROW:
        p = dict(champ)
        p.update(g)
        label = g["grow_policy"] + (f", max_leaves={g['max_leaves']}"
                                    if "max_leaves" in g else "")
        rho, sd = _score(folds, p, seed)
        rows.append(dict(knob="grow_policy", value=label, spearman=rho,
                         spearman_sd=sd, d_spearman=rho - ref,
                         is_default=(g is GROW[0])))
        if verbose:
            print(f"  {'grow_policy':20s} = {label:>9}: rho {rho:.4f}  "
                  f"d {rho - ref:+.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    if verbose:
        best = out.iloc[out["d_spearman"].idxmax()]
        print(f"\n  best single change: {best['knob']} = {best['value']}  "
              f"{best['d_spearman']:+.4f}")
        print(f"  for scale, family A is worth +0.164 and the LightGBM search "
              f"bought +0.0131")
        print(f"written -> {OUT}")
    return out


# --- the follow-up ------------------------------------------------------
# The sweep is 25 arms scored on one fold partition, so its best arm is the
# maximum of 25 draws and inherits the usual selection bias; the measured
# adaptive component elsewhere in this project is +0.0087, which is *larger*
# than the +0.0069 the sweep's winner gained. So the winner gets a second,
# paired test on fold partitions the sweep never saw.
#
# It also gets a control. `lossguide, max_leaves=63` is two changes at once:
# a different grow policy *and* four times the leaves. `depthwise, max_depth=6`
# is ~64 leaves under the original policy, so if that matches lossguide then
# the gain is capacity and the grow policy is incidental.
VERIFY = {
    "champion": {},
    "lossguide-63": dict(grow_policy="lossguide", max_depth=0, max_leaves=63),
    "depthwise-d6": dict(max_depth=6),
    "parallel-tree-4": dict(num_parallel_tree=4),
}


def _per_fold(folds, params, seed=41):
    import xgboost as xgb

    out = []
    for A, ytr, B, yva in folds:
        m = xgb.XGBRegressor(**params)
        m.fit(A, ytr)
        out.append(float(spearmanr(yva, m.predict(B)).statistic))
    return out


def verify(seeds=(41, 42, 43), with_flank: bool = True, verbose: bool = True):
    """Paired per-fold deltas against the champion, over several partitions."""
    from .curated_ablation import _base, _block
    from .run_ablation import champion_params

    X, y, names, protos = _base()
    if with_flank:
        v, _ = _block("a_flank", protos)
        X = np.hstack([X, v])

    rows = []
    for seed in seeds:
        t0 = time.time()
        folds = _folds(X, y, seed=seed)
        if verbose:
            print(f"  seed {seed}: selection {time.time() - t0:.0f} s", flush=True)
        champ = dict(champion_params(seed))
        scores = {}
        for label, override in VERIFY.items():
            p = dict(champ)
            p.update(override)
            scores[label] = _per_fold(folds, p, seed)
        for label, rhos in scores.items():
            for k, (r, c) in enumerate(zip(rhos, scores["champion"]), start=1):
                rows.append(dict(seed=seed, fold=k, arm=label, rho=r,
                                 d_rho=r - c))
            if verbose:
                d = np.mean(np.array(rhos) - np.array(scores["champion"]))
                print(f"    {label:16s} rho {np.mean(rhos):.4f}  d {d:+.4f}",
                      flush=True)
        pd.DataFrame(rows).to_csv(config.RESULTS / "xgb_knobs_verify.csv",
                                  index=False)

    d = pd.DataFrame(rows)
    print()
    rng = np.random.default_rng(0)
    for arm, g in d[d["arm"] != "champion"].groupby("arm"):
        v = g["d_rho"].to_numpy()
        boot = [rng.choice(v, v.size, replace=True).mean() for _ in range(20000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {arm:16s} d {v.mean():+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  "
              f"folds up {int((v > 0).sum())}/{v.size}")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-only", action="store_true",
                    help="sweep on the base without family A")
    ap.add_argument("--verify", action="store_true",
                    help="paired re-test of the sweep's winner on unseen "
                         "fold partitions, with a capacity control")
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    a = ap.parse_args(argv)
    if a.verify:
        verify(seeds=tuple(a.seeds), with_flank=not a.base_only)
    else:
        run(with_flank=not a.base_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
