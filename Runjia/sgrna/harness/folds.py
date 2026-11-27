"""Fold assignment by whole groups, and the fit-and-score step that the held-out-gene experiments share.

`fold_ids` puts whole groups (a gene, or a 100 kb block for guides outside any gene) into folds, so that no group is split between training and test: this is the paper's fold design, and nine experiments
and the design benchmark use it. `fit_predict` is the one fit those experiments repeat: impute, rank the columns with the throw-away selector, keep the top `n_features`, fit the champion and predict.
Both were helpers of `controls/sealed_holdout_gene.py`, which other modules imported as `sg` and whose module-level `N_JOBS` five of them set (`folds.N_JOBS = threads`).
"""
from __future__ import annotations

import numpy as np

N_JOBS = 3


def fold_ids(groups: np.ndarray, n_splits: int = 5, seed: int = 41) -> np.ndarray:
    """Whole groups to folds: groups in random order, each to the currently smallest fold."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv)
    fold_of = np.empty(len(u), dtype=int)
    load = np.zeros(n_splits)
    for g in rng.permutation(len(u)):
        k = int(np.argmin(load))
        fold_of[g] = k
        load[k] += sizes[g]
    return fold_of[inv]


def fit_predict(X_tr, y_tr, X_te, params=None, n_features=None, seed=41, n_jobs=...):
    """Impute, select `n_features` columns by a throw-away XGBoost, fit the champion (with `params` on top) and predict `X_te`.

    `n_jobs`: the module's `N_JOBS` by default (set `folds.N_JOBS = threads` before a long run); an integer overrides it; `None` leaves both models at the library's own default, which is what
    `controls/sealed.py --parts holdout` has always used.
    """
    import xgboost as xgb

    from .run_ablation import _impute, champion_params, selector_params

    jobs = N_JOBS if n_jobs is ... else n_jobs
    X_tr, X_te = _impute(X_tr, X_te)
    if n_features and n_features < X_tr.shape[1]:
        sp = dict(selector_params(seed))
        if jobs is not None:
            sp["n_jobs"] = jobs
        sel = xgb.XGBRegressor(**sp)
        sel.fit(X_tr, y_tr)
        top = np.argsort(sel.feature_importances_)[::-1][:n_features]
        X_tr, X_te = X_tr[:, top], X_te[:, top]
    p = dict(champion_params(seed))
    p.update(params or {})
    if jobs is not None:
        p["n_jobs"] = jobs
    m = xgb.XGBRegressor(**p)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)
