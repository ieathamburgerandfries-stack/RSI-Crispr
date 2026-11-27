"""The five most used geometry columns are excess twist over the flanks, which dinucleotide fractions predict out of fold at R² 0.86 to 0.97, so they re-express composition.

Fits the selector + model once on the whole cohort (base + mc geom), lists the geometry columns
that survive selection with their gain, then asks for each whether base composition in the arm
windows predicts it out of fold (gene-grouped). Three predictors, because they answer different questions:

  mononucleotide   the `comp` block, 40 columns of A/C/G/T fractions, by gradient boosting. A linear ridge from
                   family A could not do this (mean R2 0.07, `redundancy.py`); a nonlinear map from letters might.
  dinucleotide, GB the `dcomp` block, 128 columns of dinucleotide fractions, by gradient boosting.
  dinucleotide, ridge  the same 128 columns by a plain ridge. The excess-twist columns are a summed step twist
                   minus a constant per base pair, which is *linear* in dinucleotide counts, so if the finding
                   "excess twist is a weighted dinucleotide count" is what it says, a linear model must recover it.

The dinucleotide figure (R2 0.963, quoted in the report's Appendix B4) was originally computed in a session whose
code was not kept, so it had no producer and no audit check; this module is now that producer.

Report section: Appendix B4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from sgrna import config
from sgrna.harness.run_ablation import champion_params, make_selector, _impute

from sklearn.linear_model import RidgeCV

from ..common import OUT
from ..cohorts.shape_ablate import Cohort


def _oof_r2(X, y, groups, model):
    """Gene-grouped 5-fold out-of-fold R2 of `model(Xtr, ytr) -> predict(Xte)`."""
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, groups=groups):
        pred[te] = model(X[tr], y[tr]).predict(X[te])
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def _gb(Xtr, ytr):
    return xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=-1).fit(Xtr, ytr)


def _ridge(Xtr, ytr):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12            # standardise on the training fold only
    m = RidgeCV(alphas=np.logspace(-4, 3, 8)).fit((Xtr - mu) / sd, ytr)
    return type("R", (), {"predict": staticmethod(lambda X: m.predict((X - mu) / sd))})


def run():
    co = Cohort()
    G, nG = co.ext("mc", None, "geom")
    C, nC = co.ext("comp", None, "mean")
    D, nD = co.ext("dcomp", None, "mean")
    C, D = np.nan_to_num(C), np.nan_to_num(D)
    X = np.hstack([co.X, G]); names = list(co.names) + nG
    Xi, _ = _impute(X, X[:10])
    sel = make_selector(41); sel.fit(Xi, co.y)
    top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
    m = xgb.XGBRegressor(**champion_params(41)).fit(Xi[:, top], co.y)
    imp = pd.Series(m.feature_importances_, index=[names[i] for i in top]).sort_values(ascending=False)
    geo = imp[[i.startswith("mc.real.geom") for i in imp.index]]
    print(f"{len(geo)} geometry columns kept of {len(top)}; share of total importance {geo.sum():.3f}")
    rows = []
    for name, v in geo.head(12).items():
        j = nG.index(name)
        y = G[:, j]
        r2 = _oof_r2(C, y, co.groups, _gb)
        r2_d_gb = _oof_r2(D, y, co.groups, _gb)
        r2_d_ridge = _oof_r2(D, y, co.groups, _ridge)
        rows.append(dict(column=name.replace("mc.real.geom.", ""), importance=v, r2_from_composition=r2,
                         r2_from_dinucleotide_gb=r2_d_gb, r2_from_dinucleotide_ridge=r2_d_ridge))
        print(f"  {rows[-1]['column']:22s} importance {v:.4f}   out-of-fold R2: letters {r2:.3f}, "
              f"dinucleotides GB {r2_d_gb:.3f} / ridge {r2_d_ridge:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "geom_importance.csv", index=False)


if __name__ == "__main__":
    run()
