"""What the stage-1 search says about each hyperparameter (marginal effects on the DEV-fold rho).

For each draw the tree count with the best mean rho is taken (that count is part of the configuration), so a draw's score
is the maximum over a tree grid, a small upward bias that is shared by all draws. Marginal effects are the mean score of the
draws in each level of a hyperparameter; because draws vary in all dimensions at once, a ridge regression of the score on
the standardised hyperparameters gives the effect of each one holding the others fixed.

    python -m sgrna.ext.tune_analyze
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from .common import OUT


def table():
    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    s = pd.read_csv(OUT / "tune_stage1.csv")
    m = s.groupby(["draw", "k"]).rho.mean().reset_index().sort_values("rho", ascending=False).drop_duplicates("draw")
    m = m.set_index("draw")
    d = pd.DataFrame({n: cfgs[n] for n in m.index}).T
    d["rho"] = m.rho.astype(float); d["best_k"] = m.k.astype(int)
    for c in d.columns:
        if c not in ("target",):
            d[c] = pd.to_numeric(d[c])
    return d


def main():
    pd.set_option("display.width", 200)
    d = table()
    ref = d.loc["champion", "rho"]
    draws = d.drop(index=["champion", "depth6"])
    print(f"{len(draws)} draws; champion {ref:.4f}, depth6 {d.loc['depth6', 'rho']:.4f}; best draw {draws.rho.max():.4f}; "
          f"median draw {draws.rho.median():.4f}; draws above champion: {(draws.rho > ref).sum()}")
    for c, bins in (("max_depth", None), ("nf", None), ("target", None), ("max_bin", None), ("gamma", None), ("reg_alpha", None)):
        print(f"\nby {c}:"); print(draws.groupby(c).rho.agg(["mean", "max", "count"]).round(4).to_string())
    for c in ("learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda"):
        q = pd.qcut(draws[c], 4, duplicates="drop")
        print(f"\nby {c} (quartiles):"); print(draws.groupby(q, observed=True).rho.agg(["mean", "max", "count"]).round(4).to_string())
    feats = ["max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda", "reg_alpha",
             "gamma", "nf", "best_k"]
    X = draws[feats].astype(float).copy()
    for c in ("learning_rate", "min_child_weight", "reg_lambda", "nf", "best_k"):
        X[c] = np.log(X[c])
    X["rankgauss"] = (draws.target == "rankgauss").astype(float); X["bin64"] = (draws.max_bin == 64).astype(float)
    Z = (X - X.mean()) / X.std()
    r = RidgeCV(alphas=np.logspace(-2, 2, 9)).fit(Z, draws.rho.astype(float))
    eff = pd.Series(r.coef_, index=Z.columns).sort_values(key=abs, ascending=False)
    print("\nridge effect of +1 sd of each hyperparameter on rho (others held fixed):"); print(eff.round(4).to_string())


if __name__ == "__main__":
    main()
