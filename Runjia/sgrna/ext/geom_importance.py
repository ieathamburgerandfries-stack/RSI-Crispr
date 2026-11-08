"""Which geometry columns does the model use, and are they nonlinear re-expressions of composition?

Fits the selector + model once on the whole cohort (base + mc geom), lists the geometry columns
that survive selection with their gain, then asks for each whether plain arm-window base
composition (the `comp` block, 40 columns) predicts it out of fold (gene-grouped, gradient
boosting). A linear ridge from family A could not (mean R2 0.07, `redundancy.py`); a nonlinear map
from composition might.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from sgrna import config
from sgrna.run_ablation import champion_params, make_selector, _impute

from .common import OUT
from .shape_ablate import Cohort


def run():
    co = Cohort()
    G, nG = co.ext("mc", None, "geom")
    C, nC = co.ext("comp", None, "mean")
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
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(C, groups=co.groups):
            pred[te] = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=-1).fit(
                np.nan_to_num(C[tr]), y[tr]).predict(np.nan_to_num(C[te]))
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        rows.append(dict(column=name.replace("mc.real.geom.", ""), importance=v, r2_from_composition=r2))
        print(f"  {rows[-1]['column']:22s} importance {v:.4f}   out-of-fold R2 from composition {r2:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "geom_importance.csv", index=False)


if __name__ == "__main__":
    run()
