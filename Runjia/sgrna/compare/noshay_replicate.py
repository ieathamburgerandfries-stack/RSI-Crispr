"""Reproduce the inherited paper's own model, as its methods section describes it.

Everything in this project is measured against a *baseline* built from Noshay et
al.'s feature matrix with our own XGBoost. That is not the same as reproducing
*their model*, and until now the 0.502 in the comparison table was a number read
off their Table 1 rather than one we had seen happen. This module runs their
algorithm.

What the paper specifies (NAR 51:10147, 'Iterative Random Forest Model' and
Table 1):

  * iterative Random Forest (iRF), their Ranger-based implementation at
    github.com/jailGroup/RangerBasediRF
  * **1,000 decision trees per forest, ten iterations**
  * **five-fold cross-validation**, each run an 80/20 train/test split
  * the full 6,232-column matrix: one-hot positional encoding plus the quantum
    chemical tensors
  * reported for E. coli, full matrix: **R² 0.2491, Pearson 0.5019**

Two honest gaps between that and what can be run here, both of which belong in
any statement of the result:

  1. **Their row count is not reproducible from their own release.** Table 1
     says 40,468 sgRNAs (32,374 train); the published supplementary matrix
     covers 13,880. We run 13,880, so a lower score is expected and is not
     evidence their figure was wrong.
  2. **This is a reimplementation, not their code.** iRF as published (Basu et
     al., PNAS 2018) iterates a forest in which each tree's candidate features
     are drawn with probability proportional to the previous iteration's
     importances; that is what `_irf_fit` does, with scikit-learn trees. Their
     Ranger build will differ in details.

**One deviation that matters if you reuse this.** `_grow` draws `mtry` columns
**once per tree** and then grows the whole tree inside that subspace, rather than
resampling candidates at every split as a true random forest does. The
reweighting mechanism needs the candidate set chosen externally, so this is the
natural way to implement iRF — but it means **absolute `mtry` sets tree
capacity**, and `sqrt(p)` is therefore not portable across matrices of different
width. Measured on the same rows and fold
(`results/irf_mtry_sweep.csv`), sweeping `mtry` on the 427-column reduced basis:

    mtry  20 (= sqrt(427), the default)   rho 0.275
    mtry  78 (= sqrt(6232), matched)      rho 0.489
    mtry 200                              rho 0.501
    mtry 427 (every column, no subspace)  rho 0.497
    -- full 6,232 columns at mtry 78      rho 0.495 / 0.486

Same information, same algorithm, **0.21 of Spearman** between the first two
rows, entirely from how many columns each tree was allowed to see. The curve is
flat from 78 upward and peaks near half the columns, so the collapse at 20 is a
starved-tree effect and nothing to do with the reduced basis. Compare iRF across
column counts at **matched absolute `mtry`**, never at matched sqrt(p).

    python -m sgrna.compare.noshay_replicate --run
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

from .. import config

OUT = config.RESULTS / "noshay_replication.csv"

N_TREES = 1000
N_ITERATIONS = 10


def _peak_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**30 if sys.platform == "darwin" else rss / 2**20


def _grow(X, y, p, mtry, weights, uniform, tree_seed, row_seed):
    """One bootstrapped tree on a weighted-sampled feature subset."""
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(tree_seed)
    feat = rng.choice(p, size=mtry, replace=False,
                      p=None if uniform else weights)
    rows = np.random.default_rng(row_seed).integers(0, X.shape[0],
                                                    size=X.shape[0])
    t = DecisionTreeRegressor(max_features=None, random_state=int(tree_seed))
    t.fit(X[np.ix_(rows, feat)], y[rows])
    return t, feat


def _irf_fit(X, y, n_trees=N_TREES, n_iter=N_ITERATIONS, seed=41,
             n_jobs=-1, verbose=True):
    """iRF: a forest whose feature sampling is reweighted by the last forest's.

    Iteration 1 is a plain random forest (uniform feature weights). Each later
    iteration draws every tree's candidate feature set with probability
    proportional to the previous iteration's Gini importances, which is the
    'amplification' the paper describes -- features that mattered keep getting
    offered, so interactions among them get a chance to be found.

    Trees are grown individually because scikit-learn's forest has no
    per-feature sampling weights. `max_features='sqrt'` matches Ranger's
    regression default of floor(sqrt(p)).
    """
    n, p = X.shape
    mtry = max(1, int(np.sqrt(p)))
    weights = np.full(p, 1.0 / p)
    rng = np.random.default_rng(seed)

    from joblib import Parallel, delayed

    trees, cols = [], []
    for it in range(1, n_iter + 1):
        uniform = it == 1
        tree_seeds = rng.integers(1 << 31, size=n_trees)
        row_seeds = rng.integers(1 << 31, size=n_trees)
        grown = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=8)(
            delayed(_grow)(X, y, p, mtry, weights, uniform,
                           int(tree_seeds[i]), int(row_seeds[i]))
            for i in range(n_trees))
        trees = [g[0] for g in grown]
        cols = [g[1] for g in grown]
        imp = np.zeros(p)
        for t, feat in grown:
            imp[feat] += t.feature_importances_
        s = imp.sum()
        weights = (imp / s) if s > 0 else np.full(p, 1.0 / p)
        # keep every feature reachable, as iRF does, so a late interaction is
        # not permanently excluded by an unlucky first iteration
        weights = 0.9 * weights + 0.1 / p
        if verbose:
            print(f"      iteration {it}/{n_iter}", flush=True)
    return trees, cols


def _irf_predict(model, X):
    trees, cols = model
    out = np.zeros(X.shape[0])
    for t, feat in zip(trees, cols):
        out += t.predict(X[:, feat])
    return out / len(trees)


def run(seed: int = 41, n_splits: int = 5, n_trees: int = N_TREES,
        n_iter: int = N_ITERATIONS, n_jobs: int = -1,
        verbose: bool = True) -> pd.DataFrame:
    from sklearn.model_selection import KFold

    from ..build.build_matrix import load_dataset
    from ..harness.run_ablation import _impute

    X, y, names, _, _ = load_dataset([], verbose=verbose)
    if verbose:
        print(f"  their full matrix: {X.shape[0]:,} guides x {X.shape[1]:,} "
              f"columns (their Table 1 used 40,468 guides)", flush=True)

    prev = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    rows = prev.to_dict("records") if len(prev) else []
    done = set(prev["fold"]) if len(prev) else set()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        if fold in done:
            if verbose:
                print(f"    fold {fold}: already done", flush=True)
            continue
        X_tr, X_va = _impute(X[tr], X[va])
        if verbose:
            print(f"    fold {fold}: {len(tr):,} train / {len(va):,} test",
                  flush=True)
        t0 = time.time()
        model = _irf_fit(X_tr, y[tr], n_trees=n_trees, n_iter=n_iter,
                         seed=seed + fold, n_jobs=n_jobs, verbose=verbose)
        fit_s = time.time() - t0
        pred = _irf_predict(model, X_va)
        rows.append(dict(
            model="iRF (reimplementation of Noshay et al.)",
            fold=fold, seed=seed, n_train=len(tr), n_test=len(va),
            n_features=X.shape[1], n_trees=n_trees, n_iterations=n_iter,
            pearson=float(pearsonr(y[va], pred).statistic),
            spearman=float(spearmanr(y[va], pred).statistic),
            r2=float(r2_score(y[va], pred)),
            fit_seconds=round(fit_s, 1),
            peak_gb=round(_peak_gb(), 2)))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        if verbose:
            r = rows[-1]
            print(f"    fold {fold}: Pearson {r['pearson']:.4f}  "
                  f"rho {r['spearman']:.4f}  R2 {r['r2']:.4f}  "
                  f"({fit_s / 60:.1f} min)", flush=True)
    return pd.DataFrame(rows)


def summary():
    d = pd.read_csv(OUT)
    print(d[["fold", "pearson", "spearman", "r2", "fit_seconds",
             "peak_gb"]].round(4).to_string(index=False))
    print(f"\n  mean Pearson {d['pearson'].mean():.4f} ± {d['pearson'].std():.4f}"
          f"   (their Table 1: 0.5019)")
    print(f"  mean Spearman {d['spearman'].mean():.4f}"
          f"   (not reported in their paper)")
    print(f"  mean R2      {d['r2'].mean():.4f} ± {d['r2'].std():.4f}"
          f"   (their Table 1: 0.2491)")
    print(f"  total CPU time {d['fit_seconds'].sum() / 60:.1f} min for "
          f"{len(d)} folds; peak {d['peak_gb'].max():.2f} GB")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--trees", type=int, default=N_TREES)
    ap.add_argument("--iterations", type=int, default=N_ITERATIONS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=-1,
                    help="trees are grown in parallel; -1 uses every core")
    a = ap.parse_args(argv)
    if a.run:
        run(n_splits=a.folds, n_trees=a.trees, n_iter=a.iterations,
            n_jobs=a.jobs)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
