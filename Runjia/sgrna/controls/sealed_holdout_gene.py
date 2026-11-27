"""Holding out whole genes in the pool cross-validation leaves the sealed-gene drop at +0.0081 against +0.0088 with random rows, so unseen genes explain almost none of it.

`sealed_holdout.py` measures an "adaptive component" as a difference of differences,

    (pool CV - sealed | frozen)  -  (pool CV - sealed | base only),

where the pool CV is a **random-row** 5-fold CV inside the development pool and the sealed score is on **whole genes**. For a feature
that reads the locus (family A) a random-row fold can score by recognising the neighbourhood of a guide whose neighbours sit in
training, so the difference of differences also contains what unseen genes cost such a feature. The frozen arm re-derives no
hyperparameter at all (champion settings, selection inside each training split), which is why the adaptive part of that number can be
small.

This module re-runs only the pool CV with **whole genes held out** (guides outside any gene are grouped by 100 kb block, so
neighbours stay together), for the same sealing draws and the same arms, with the settings the original run used (the dev-tuned arm's
settings are read from `sealed_holdout.csv`). The sealed scores are not recomputed: they are read from `sealed_holdout.csv`.

    random-row component  = gap_random(frozen) - gap_random(base)       (the report's +0.0087)
    gene-grouped component = gap_gene(frozen)   - gap_gene(base)         (both CVs and the sealed score hold out whole genes)
    cost of unseen genes   = random-row component - gene-grouped component

If the gene-grouped component is near zero, the whole +0.0087 is the cost of unseen genes to a locus-reading feature and none of it is
adaptive overfitting of the automatable choices; if it is near +0.0087 the reverse.

Stored result: `results/sealed_holdout_gene.csv` holds the first 8 of the 15 sealings (about 7 minutes per sealing at 3 threads
on a loaded machine; the run was stopped there). Their random-row component, +0.0088, matches the 15-draw +0.0087.

    OMP_NUM_THREADS=3 python -m sgrna.controls.sealed_holdout_gene [--max-draws N] [--arms base_only frozen dev_tuned]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..harness.curated import load_base, load_block
from .sealed_holdout import OUT as SEALED_CSV, _seal

N_JOBS = 3


def _out_path():
    from ..common import OUT as POST
    return POST / "sealed_holdout_gene.csv"


def _fold_ids(groups: np.ndarray, n_splits: int = 5, seed: int = 41) -> np.ndarray:
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


def _fit_predict(X_tr, y_tr, X_te, params=None, n_features=None, seed=41):
    import xgboost as xgb

    from ..harness.run_ablation import _impute, champion_params, selector_params

    X_tr, X_te = _impute(X_tr, X_te)
    if n_features and n_features < X_tr.shape[1]:
        sp = dict(selector_params(seed))
        sp["n_jobs"] = N_JOBS
        sel = xgb.XGBRegressor(**sp)
        sel.fit(X_tr, y_tr)
        top = np.argsort(sel.feature_importances_)[::-1][:n_features]
        X_tr, X_te = X_tr[:, top], X_te[:, top]
    p = dict(champion_params(seed))
    p.update(params or {})
    p["n_jobs"] = N_JOBS
    m = xgb.XGBRegressor(**p)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def _cv_gene(X, y, groups, params, n_features, seed=41, verbose=True):
    fold = _fold_ids(groups, 5, seed)
    pred = np.empty_like(y)
    for k in range(5):
        t0 = time.time()
        va = fold == k
        pred[va] = _fit_predict(X[~va], y[~va], X[va], params, n_features, seed)
        if verbose:
            print(f"      fold {k + 1}: {time.time() - t0:5.1f}s", flush=True)
    return float(spearmanr(y, pred).statistic)


def run(max_draws: int | None = None, arms=("base_only", "frozen", "dev_tuned"), verbose: bool = True) -> pd.DataFrame:
    from ..build.build_features import load_curated_index

    ref = pd.read_csv(SEALED_CSV)
    out_path = _out_path()
    done = pd.read_csv(out_path) if out_path.exists() and out_path.stat().st_size else pd.DataFrame()
    X, y, names, protos = load_base()
    A, anames = load_block("a_flank", protos)
    XA = np.hstack([X, A])
    idx = load_curated_index(verbose=False).set_index(config.ID_COL)
    gene = idx["gene_name"].reindex(protos)
    left = idx["left"].reindex(protos).to_numpy(float)
    groups = np.array([g if isinstance(g, str) and g else f"ig{int(l // 100_000)}" for g, l in zip(gene, left)], dtype=object)

    seeds = sorted(ref.seal_seed.unique())[:max_draws]
    rows = []
    for sd in seeds:
        sealed = _seal(protos, seed=int(sd), verbose=False)
        dev = np.where(~sealed)[0]
        for arm in arms:
            if len(done) and ((done.arm == arm) & (done.seal_seed == sd)).any():
                continue
            r = ref[(ref.arm == arm) & (ref.seal_seed == sd)].iloc[0]
            params = json.loads(r.params) if isinstance(r.params, str) else {}
            Xa, nf = (X, None) if arm == "base_only" else (XA, config.N_FEATURES)
            if verbose:
                print(f"  seed {sd} {arm}: gene-grouped CV in the pool ({len(dev):,} guides)", flush=True)
            cv = _cv_gene(Xa[dev], y[dev], groups[dev], params or None, nf)
            row = dict(arm=arm, seal_seed=int(sd), cv_gene=cv, cv_random=float(r.cv_in_pool), sealed=float(r.sealed),
                       gap_gene=cv - float(r.sealed), gap_random=float(r.gap), n_dev=int(len(dev)), params=r.params)
            if verbose:
                print(f"    cv_gene {cv:.4f}  cv_random {r.cv_in_pool:.4f}  sealed {r.sealed:.4f}", flush=True)
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(out_path, index=False)
            rows.append(row)
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-draws", type=int, default=None)
    ap.add_argument("--arms", nargs="*", default=["base_only", "frozen", "dev_tuned"])
    a = ap.parse_args(argv)
    run(a.max_draws, tuple(a.arms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
