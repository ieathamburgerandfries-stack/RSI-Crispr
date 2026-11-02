"""Cross-kingdom transfer, with the human side fitted properly.

Part 3's boundary claim is that transfer between *E. coli* and human is negative
in both directions: rho -0.048 going to human and -0.017 coming back
(`results/human_transfer.csv`). That file was written by a script that is not in
`src/`, and both directions used the champion hyperparameters -- which
`human_baseline.py` has since shown to be mis-specified on the human matrix by
0.10 rho, because the human signal is far less additive than *E. coli*'s
(ridge/boosting 63.7% against 95.6%).

That matters for exactly one direction. **human -> E. coli** trains on human, so
an undertuned human model makes the reported -0.017 pessimistic, and the report
should not rest a boundary claim on a model it knows is the wrong shape for its
training data. **E. coli -> human** trains on *E. coli*, where the champion is
appropriate, so it is a control here rather than a question.

So: both directions, at both settings, on the columns the two matrices share by
name -- no change of representation, which is what makes the comparison a
statement about kingdoms rather than about feature engineering.

    python -m sgrna.kingdom_transfer
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config

OUT = config.RESULTS / "kingdom_transfer.csv"
HUMAN = config.DATA / "raw" / "human_feature_matrix.csv"

# The settings jacky_400_1000 found, which reach 0.5076 within human against the
# champion's 0.4074. Library defaults apart from the tree count.
HUMAN_FITTED = dict(n_estimators=1000, max_depth=6, learning_rate=0.3,
                    subsample=1.0, colsample_bytree=1.0)


def _matrix(path, skiprows):
    d = pd.read_csv(path, skiprows=skiprows, low_memory=False)
    y = pd.to_numeric(d[config.TARGET_COL], errors="coerce")
    drop = [c for c in (config.ID_COL, "sgRNAID", config.TARGET_COL) if c in d]
    X = d.drop(columns=drop).apply(pd.to_numeric, errors="coerce")
    ok = np.isfinite(y.to_numpy(float))
    return X.loc[ok], y.to_numpy(float)[ok]


def load(verbose: bool = True):
    Xe, ye = _matrix(config.BASE_MATRIX, config.BASE_MATRIX_HEADER_ROW)
    Xh, yh = _matrix(HUMAN, 1)
    shared = [c for c in Xe.columns if c in set(Xh.columns)]
    if verbose:
        print(f"  E. coli {len(ye):,} x {Xe.shape[1]:,}  |  "
              f"human {len(yh):,} x {Xh.shape[1]:,}  |  "
              f"shared by name {len(shared):,}")
    return (Xe[shared].to_numpy(np.float32), ye,
            Xh[shared].to_numpy(np.float32), yh, shared)


def run(seeds=(41, 42, 43), verbose: bool = True):
    import xgboost as xgb

    from .run_ablation import _impute, champion_params, make_selector

    Xe, ye, Xh, yh, shared = load(verbose=verbose)
    directions = {"ecoli -> human": (Xe, ye, Xh, yh),
                  "human -> ecoli": (Xh, yh, Xe, ye)}

    rows = []
    for name, (X_tr0, y_tr, X_te0, y_te) in directions.items():
        for setting in ("champion", "human_fitted"):
            rhos = []
            for seed in seeds:
                X_tr, X_te = _impute(X_tr0, X_te0)
                sel = make_selector(seed)
                sel.fit(X_tr, y_tr)
                top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
                p = dict(champion_params(seed))
                if setting == "human_fitted":
                    p.update(HUMAN_FITTED)
                m = xgb.XGBRegressor(**p)
                m.fit(X_tr[:, top], y_tr)
                rhos.append(float(spearmanr(y_te, m.predict(X_te[:, top])).statistic))
            rows.append(dict(direction=name, settings=setting,
                             n_shared=len(shared), n_train=len(y_tr),
                             n_test=len(y_te), spearman=float(np.mean(rhos)),
                             spearman_sd=float(np.std(rhos, ddof=1)),
                             n_fits=len(rhos)))
            if verbose:
                print(f"  {name:>15} / {setting:>12}: rho "
                      f"{rows[-1]['spearman']:+.4f} ± {rows[-1]['spearman_sd']:.4f}",
                      flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)

    if verbose:
        print("  the figures in the report are -0.048 and -0.017, both at champion")
        print(f"written -> {OUT}")
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bootstrap", action="store_true",
                    help="resample test rows at the human-fitted settings")
    a = ap.parse_args(argv)
    bootstrap() if a.bootstrap else run()
    return 0



# --------------------------------------------------------------------------
# Does either direction actually exclude zero?
# --------------------------------------------------------------------------
#
# The seed-to-seed spread above is model noise, not sampling noise: three seeds
# of one train/test split say nothing about how much of the measured rho is the
# particular test set. With rho this close to zero that is the whole question --
# -0.006 on 13,880 rows and -0.037 on 17,421 rows are not the same claim, even
# though both are negative. So resample the test rows.

BOOT = config.RESULTS / "kingdom_transfer_boot.csv"


def bootstrap(seeds=(41, 42, 43), n_boot=2000, verbose: bool = True):
    import xgboost as xgb

    from .run_ablation import _impute, champion_params, make_selector

    Xe, ye, Xh, yh, shared = load(verbose=verbose)
    directions = {"ecoli -> human": (Xe, ye, Xh, yh),
                  "human -> ecoli": (Xh, yh, Xe, ye)}

    rows = []
    for name, (X_tr0, y_tr, X_te0, y_te) in directions.items():
        preds = []
        for seed in seeds:
            X_tr, X_te = _impute(X_tr0, X_te0)
            sel = make_selector(seed)
            sel.fit(X_tr, y_tr)
            top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
            p = dict(champion_params(seed))
            p.update(HUMAN_FITTED)
            m = xgb.XGBRegressor(**p)
            m.fit(X_tr[:, top], y_tr)
            preds.append(m.predict(X_te[:, top]))
        yhat = np.mean(preds, axis=0)

        rng = np.random.default_rng(41)
        n = len(y_te)
        draws = np.empty(n_boot)
        for b in range(n_boot):
            i = rng.integers(0, n, n)
            draws[b] = spearmanr(y_te[i], yhat[i]).statistic
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append(dict(direction=name, settings="human_fitted", n_test=n,
                         spearman=float(spearmanr(y_te, yhat).statistic),
                         ci_lo=float(lo), ci_hi=float(hi), n_boot=n_boot,
                         excludes_zero=bool(lo < 0 and hi < 0)))
        if verbose:
            r = rows[-1]
            print(f"  {name:>15}: rho {r['spearman']:+.4f} "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
                  f"{'excludes zero' if r['excludes_zero'] else 'INCLUDES ZERO'}",
                  flush=True)
        pd.DataFrame(rows).to_csv(BOOT, index=False)

    if verbose:
        print(f"written -> {BOOT}")
    return pd.DataFrame(rows)

if __name__ == "__main__":
    raise SystemExit(main())
