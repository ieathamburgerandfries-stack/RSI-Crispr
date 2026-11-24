"""On genes sealed away from column selection and settings, family A's gain is 0.0087 lower than in cross-validation of the pool, and a settings search inside the pool does not enlarge the drop.

Part 4 says the curated five-fold figures "carry an unknown adaptive-overfitting
component" and leaves it unknown, because no untouched external dataset remains.
Sealing rows *now* cannot undo the history -- every choice in the pipeline was
made while looking at numbers computed on all 33,567 guides, and hiding a slice
afterwards does not unsee them. So this does not measure our actual history.

**What it can measure is the procedure.** Re-derive the mechanisable choices --
column selection and hyperparameters -- using only a development pool, then
score once on rows no step of that re-derivation ever saw. If a cross-validated
estimate computed inside the pool is systematically higher than the sealed-set
score, the pipeline inflates its own estimates, and by how much.

**The confound, and the design that removes it.** Sealed rows are a different
population: whole genes are sealed, and gene identity explains 29% of this
label's variance, so *any* model will score differently there for reasons that
have nothing to do with adaptivity. The fix is a difference of differences --
measure the CV-minus-sealed gap for a model whose configuration was **not**
chosen on this cohort, and subtract:

    adaptive component  =  (CV - sealed | tuned here)  -  (CV - sealed | not tuned here)

Three arms:

    base_only    6,169 regenerated columns, champion hyperparameters carried
                 over from the published cohort, no selection -- nothing about
                 it was chosen on these rows
    frozen       base + family A, champion parameters, 300 columns selected
                 inside each training split: the configuration the report quotes
    dev_tuned    the same, with an 8-draw search run inside the development pool
                 only -- the arm that can overfit the pool

**What it cannot capture, and this is the larger part.** Family A exists because
someone looked at residuals computed on the whole cohort. The window sizes, the
decision to build it at all, the choice of `n_features`, and five withdrawn
claims' worth of exploration are all human choices informed by the full data. No
re-derivation reaches them. **The number below is therefore a lower bound on the
adaptive component, and it bounds only the automatable part.**

    python -m sgrna.controls.sealed_holdout
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from .. import config
from ..cohorts.curated_ablation import _base, _block

OUT = config.RESULTS / "sealed_holdout.csv"
SEAL_FRACTION = 0.20
SEAL_SEED = 20261016          # fixed here, once, and not varied afterwards


def _seal(protos: np.ndarray, seed: int = SEAL_SEED, verbose: bool = True):
    """Seal whole genes until ~20% of rows are sealed; intergenic rows stay in."""
    from ..build.build_features import load_curated_index

    idx = load_curated_index(verbose=False).set_index(config.ID_COL)
    gene = idx["gene_name"].reindex(protos)
    gene = np.array([g if isinstance(g, str) and g else "" for g in gene],
                    dtype=object)
    genes = np.array(sorted({g for g in gene if g}))
    rng = np.random.default_rng(seed)
    rng.shuffle(genes)
    counts = pd.Series(gene).value_counts()
    sealed_genes, n = set(), 0
    target = int(len(protos) * SEAL_FRACTION)
    for g in genes:
        if n >= target:
            break
        sealed_genes.add(g)
        n += int(counts[g])
    sealed = np.array([g in sealed_genes for g in gene])
    if verbose:
        print(f"  sealed {sealed.sum():,} guides in {len(sealed_genes):,} genes "
              f"({sealed.mean():.1%}); development pool {(~sealed).sum():,}")
    return sealed


def _fit_predict(X_tr, y_tr, X_te, params=None, n_features=None, seed=41):
    import xgboost as xgb

    from ..harness.run_ablation import _impute, champion_params, make_selector

    X_tr, X_te = _impute(X_tr, X_te)
    if n_features and n_features < X_tr.shape[1]:
        sel = make_selector(seed)
        sel.fit(X_tr, y_tr)
        top = np.argsort(sel.feature_importances_)[::-1][:n_features]
        X_tr, X_te = X_tr[:, top], X_te[:, top]
    p = dict(champion_params(seed))
    p.update(params or {})
    m = xgb.XGBRegressor(**p)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def _cv_in_pool(X, y, params=None, n_features=None, n_splits=5, seed=41):
    """Out-of-fold ρ inside the development pool: the estimate we would report."""
    pred = np.empty_like(y)
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        pred[va] = _fit_predict(X[tr], y[tr], X[va], params, n_features, seed)
    return float(spearmanr(y, pred).statistic)


def _search_in_pool(X, y, draws=8, seed=41, verbose=True):
    """A small hyperparameter search, fitted and scored entirely inside the pool."""
    rng = np.random.default_rng(seed)
    space = dict(max_depth=[3, 4, 6, 8], learning_rate=[0.02, 0.05, 0.1],
                 subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0],
                 reg_lambda=[0.5, 1.0, 5.0], min_child_weight=[1, 5, 20])
    inner = KFold(5, shuffle=True, random_state=seed + 1)
    itr, iva = next(iter(inner.split(X)))
    best, best_p = -np.inf, {}
    for d in range(draws):
        p = {} if d == 0 else {k: v[int(rng.integers(len(v)))]
                               for k, v in space.items()}
        pred = _fit_predict(X[itr], y[itr], X[iva], p, config.N_FEATURES, seed)
        rho = float(spearmanr(y[iva], pred).statistic)
        if rho > best:
            best, best_p = rho, p
        if verbose:
            print(f"    draw {d}: inner rho {rho:.4f}", flush=True)
    return best_p


def run(match_train_size: bool = True, seal_seed: int = SEAL_SEED,
        verbose: bool = True) -> pd.DataFrame:
    """`match_train_size` fixes a confound the first version of this had.

    A 5-fold CV inside the pool trains each fold on 80% of it, while the
    sealed-set fit trains on 100%. The sealed number is therefore flattered by
    ~25% more training data -- worth roughly +0.005 ρ at this point on the
    sample-complexity curve, and *more* for arms that benefit most from data,
    which is exactly the arms under test. Subsampling the sealed-set fit to the
    same 80% removes it, so that the only remaining difference between the two
    estimates is the population.
    """
    X, y, names, protos = _base()
    A, anames = _block("a_flank", protos)
    XA = np.hstack([X, A])
    sealed = _seal(protos, seed=seal_seed, verbose=verbose)
    dev = ~sealed
    dev_idx = np.where(dev)[0]
    fit_idx = dev_idx
    if match_train_size:
        n = int(len(dev_idx) * (1 - 1 / 5))
        fit_idx = np.random.default_rng(seal_seed).choice(dev_idx, n, replace=False)
        if verbose:
            print(f"  sealed-set fits use {n:,} of {len(dev_idx):,} pool rows, "
                  f"matching what each CV fold trains on")

    arms = {
        "base_only":  dict(X=X,  n_features=None, tune=False),
        "frozen":     dict(X=XA, n_features=config.N_FEATURES, tune=False),
        "dev_tuned":  dict(X=XA, n_features=config.N_FEATURES, tune=True),
    }

    rows = []
    for arm, cfg in arms.items():
        Xa = cfg["X"]
        params = None
        if cfg["tune"]:
            if verbose:
                print(f"  {arm}: searching inside the development pool")
            params = _search_in_pool(Xa[dev], y[dev], verbose=verbose)
            if verbose:
                print(f"    chosen: {params}")
        cv = _cv_in_pool(Xa[dev], y[dev], params, cfg["n_features"])
        pred = _fit_predict(Xa[fit_idx], y[fit_idx], Xa[sealed], params,
                            cfg["n_features"])
        sealed_rho = float(spearmanr(y[sealed], pred).statistic)
        rows.append(dict(arm=arm, seal_seed=seal_seed, n_cols=Xa.shape[1],
                         n_dev=int(dev.sum()),
                         n_sealed=int(sealed.sum()), cv_in_pool=cv,
                         sealed=sealed_rho, gap=cv - sealed_rho,
                         params=json.dumps(params or {})))
        prev = (pd.read_csv(OUT) if OUT.exists() and OUT.stat().st_size
                else pd.DataFrame())
        if len(prev):
            prev = prev[~((prev["arm"] == arm)
                          & (prev.get("seal_seed", -1) == seal_seed))]
        if verbose:
            print(f"  {arm:>10}: pool CV {cv:.4f}   sealed {sealed_rho:.4f}   "
                  f"gap {cv - sealed_rho:+.4f}", flush=True)
        pd.concat([prev, pd.DataFrame([rows[-1]])],
                  ignore_index=True).to_csv(OUT, index=False)

    out = pd.DataFrame(rows).set_index("arm")
    if verbose:
        ref = out.loc["base_only", "gap"]
        print(f"\n  reference gap (nothing chosen on these rows): {ref:+.4f}")
        for arm in ("frozen", "dev_tuned"):
            print(f"  adaptive component, {arm:>9}: "
                  f"{out.loc[arm, 'gap'] - ref:+.4f}")
        print(f"\nwritten -> {OUT}")
    return out.reset_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-match-train-size", action="store_true",
                    help="reproduce the first, confounded version")
    ap.add_argument("--seal-seeds", type=int, nargs="*", default=[SEAL_SEED],
                    help="repeat with several sealing draws: one draw of 872 "
                         "genes is itself a sample, and the arms under test "
                         "read flank properties the reference arm cannot see")
    a = ap.parse_args(argv)
    for sd in a.seal_seeds:
        run(match_train_size=not a.no_match_train_size, seal_seed=sd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
