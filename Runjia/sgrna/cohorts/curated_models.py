"""Including family A does not make model class matter more: eight classes span 0.108 rho on the curated cohort, no wider than on the published matrix.

"Sixteen model classes span only 0.116 ρ" is a Finding 2 corollary, and it was
measured on the published matrix **without** family A — i.e. on a representation
with no large effect in it. The reading it supports ("the representation matters
and the model class barely does") is therefore circular unless it survives on a
representation that *does* contain the effect.

Same registry (`evaluate._make`), same folds, curated cohort, every model given
base + family A. One seed × five folds, because the question is the *spread*
across classes and not a per-class estimate to three decimals.

    python -m sgrna.cohorts.curated_models
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from .. import config
from .curated_ablation import _base, _block

OUT = config.RESULTS / "curated_models.csv"
MODELS = ("xgboost", "lightgbm", "random_forest", "extra_trees", "ridge",
          "elastic_net", "knn", "pls")


def run(models=MODELS, seed: int = 41, n_splits: int = 5, verbose: bool = True):
    from ..harness.evaluate import _make, _standardise
    from ..harness.run_ablation import _impute, make_selector

    X, y, names, protos = _base()
    A, _ = _block("a_flank", protos)
    X = np.hstack([X, A])
    if verbose:
        print(f"  curated base + family A: {X.shape[0]:,} x {X.shape[1]:,}")

    done = set()
    if OUT.exists() and OUT.stat().st_size:
        done = set(pd.read_csv(OUT)["model"])

    rows = [] if not done else pd.read_csv(OUT).to_dict("records")
    folds = list(KFold(n_splits, shuffle=True, random_state=seed).split(X))
    for name in models:
        if name in done:
            continue
        t0, rhos = time.time(), []
        for tr, va in folds:
            X_tr, X_va = _impute(X[tr], X[va])
            sel = make_selector(seed)
            sel.fit(X_tr, y[tr])
            top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
            X_tr, X_va = X_tr[:, top], X_va[:, top]
            model, scale = _make(name, seed)
            if scale:
                X_tr, X_va = _standardise(X_tr, X_va)
            model.fit(X_tr, y[tr])
            rhos.append(float(spearmanr(y[va], model.predict(X_va)).statistic))
        rows.append(dict(model=name, spearman=float(np.mean(rhos)),
                         spearman_sd=float(np.std(rhos, ddof=1)),
                         seconds=round(time.time() - t0, 1)))
        if verbose:
            print(f"  {name:>14}: rho {rows[-1]['spearman']:.4f} "
                  f"± {rows[-1]['spearman_sd']:.4f}  ({rows[-1]['seconds']:.0f}s)",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    out = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    out.to_csv(OUT, index=False)
    if verbose:
        span = out["spearman"].max() - out["spearman"].min()
        print()
        print(out.round(4).to_string(index=False))
        print(f"\n  spread across {len(out)} classes: {span:.4f} rho")
        print(f"written -> {OUT}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args(argv)
    run(models=tuple(a.models) if a.models else MODELS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
