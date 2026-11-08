"""Is "shape / 3D adds nothing on top of family A" a property of the data or of the boosted-tree
pipeline with its 300-column selector?  The same comparison with a ridge regression on all columns
(no selector, no trees), gene-grouped 5-fold CV, one seed, alpha chosen on an inner split of the
first training fold and then fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from sgrna.run_ablation import grouped_folds

from .common import OUT
from .shape_ablate import Cohort


def prep(X):
    X = np.asarray(X, np.float64)
    med = np.nanmedian(X, 0); med = np.where(np.isfinite(med), med, 0)
    X = np.where(np.isfinite(X), X, med)
    return X


def run():
    co = Cohort()
    A, nA = co.A()
    parts = {"A": A}
    for blk in ("mean", "vec", "geom"):
        parts[f"mc_{blk}"] = co.ext("mc", None, blk)[0]
        parts[f"mc_blockshuf_{blk}"] = co.ext("mc", "block", blk)[0]
    for k, args in {"mc_all": [("mc", None, b) for b in ("mean", "vec", "geom")],
                    "mc_armshuf_all": [("mc", "arm", b) for b in ("mean", "vec", "geom")],
                    "md_all": [("md", None, b) for b in ("mean", "vec", "geom")]}.items():
        parts[k] = np.hstack([co.ext(*a)[0] for a in args])
    arms = {"base": [], "base+A": ["A"], "base+A+mc_mean": ["A", "mc_mean"], "base+A+mc_vec": ["A", "mc_vec"],
            "base+A+mc_geom": ["A", "mc_geom"], "base+A+mc_blockshuf_mean": ["A", "mc_blockshuf_mean"],
            "base+A+mc_blockshuf_vec": ["A", "mc_blockshuf_vec"], "base+A+mc_blockshuf_geom": ["A", "mc_blockshuf_geom"], "base+mc_all": ["mc_all"], "base+A+mc_all": ["A", "mc_all"],
            "base+A+mc_armshuf_all": ["A", "mc_armshuf_all"], "base+A+md_all": ["A", "md_all"]}
    Xbase = prep(co.X)
    folds = list(grouped_folds(co.groups, 5, 41))
    rows = []
    alphas_seen = {}
    for arm, ps in arms.items():
        X = np.hstack([Xbase] + [prep(parts[p]) for p in ps])
        mu, sd = X[folds[0][0]].mean(0), X[folds[0][0]].std(0) + 1e-9
        X = (X - mu) / sd
        tr0 = folds[0][0]
        inner = np.random.default_rng(0).permutation(tr0)
        a_tr, a_va = inner[: int(0.8 * len(inner))], inner[int(0.8 * len(inner)):]
        best = max((spearmanr(co.y[a_va], Ridge(alpha=al).fit(X[a_tr], co.y[a_tr]).predict(X[a_va])).statistic, al)
                   for al in (3000, 10000, 30000, 100000, 300000, 1000000))
        alpha = best[1]
        alphas_seen[arm] = alpha
        rho = [spearmanr(co.y[te], Ridge(alpha=alpha).fit(X[tr], co.y[tr]).predict(X[te])).statistic for tr, te in folds]
        rows.append(dict(arm=arm, n_cols=X.shape[1], alpha=alpha, rho=float(np.mean(rho)),
                         fold_rho=";".join(f"{r:.5f}" for r in rho)))
        print(f"  {arm:24s} cols {X.shape[1]:6d} alpha {alpha:6.0f}  rho {np.mean(rho):.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "ridge_check.csv", index=False)
    # paired fold differences against base+A, at each arm's own alpha
    f = {r["arm"]: np.array([float(x) for x in r["fold_rho"].split(";")]) for r in rows}
    for arm in f:
        if arm in ("base", "base+A"):
            continue
        d = f[arm] - f["base+A"]
        print(f"  {arm:24s} minus base+A: mean {d.mean():+.4f}  folds {' '.join(f'{x:+.4f}' for x in d)}", flush=True)


if __name__ == "__main__":
    run()
