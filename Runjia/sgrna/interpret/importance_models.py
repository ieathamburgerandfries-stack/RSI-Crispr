"""Per-column importance depends on the model asked: five tree families agree on columns at only 0.496 even with SHAP, but on blocks of columns at 0.800.

`importance.py` compared two models (XGBoost, LightGBM) and two methods (split
gain, permutation) and found the uncomfortable result that they agree about
*blocks* of columns (rho 0.50) while disagreeing about *individual* columns
(rho -0.14), with only 138 of 300 selected columns in common. The explanation
offered was redundancy: where many columns carry the same information, which
one a model credits is close to arbitrary.

Two objections to that, and this module exists to answer both.

**"You only tried two models, and they are both boosted trees."** Boosting
grows trees on residuals, so two boosters might fail the same way for reasons
that have nothing to do with the data. The test is to ask model families that
reach a prediction by genuinely different routes -- bagged trees, which fit
independently and average; and penalised linear models, which have no column
selection step at all and assign every column a coefficient. If the
column-level disagreement is a property of the matrix, it should survive the
change of model class. If it is a property of boosting, it should not.

**"Split gain is a crude heuristic. Use SHAP."** This is the right objection to
raise and it does not survive contact with the problem. SHAP's uniqueness
theorem is about dividing credit for *one prediction* among the inputs that
produced it; it says nothing about which input is mechanistically responsible.
Given two columns carrying identical information, Shapley values split the
credit between them -- by symmetry, roughly evenly, and exactly how evenly
depends on how the particular trees happened to use them. So the prediction
here is explicit and falsifiable: **SHAP should show the same column-level
instability as gain, and the same block-level stability.** If SHAP instead
produces a ranking all models agree on, the redundancy account is wrong and
this project has been reading its own interpretability results incorrectly.

Three importance methods are computed where each applies:

  native       split gain (boosters), PredictionValuesChange (CatBoost),
               impurity decrease (forests), |coefficient| on standardised
               columns (linear). What a practitioner reads off by default.
  shap         TreeSHAP mean |phi| over validation rows, for the five tree
               models. Model-agnostic in principle, tractable here only for
               trees. For a linear model SHAP is proportional to the
               standardised coefficient, so the native column already is it.
  permutation  drop in validation rho when one column is shuffled. The only
               method that is defined identically for every model, and the
               only one measured against held-out data.

    python -m sgrna.interpret.importance_models --run

Report section: Finding 5.
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from .importance import block_of, kuncheva

PAIRS = config.RESULTS / "importance_model_pairs.csv"
METHODS = config.RESULTS / "importance_model_methods.csv"
SHARES = config.RESULTS / "importance_model_blocks.csv"


def _models(seed: int):
    """Four routes to a prediction: boosting, bagging, ordered boosting, linear."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge

    from ..harness.run_ablation import champion_params

    out = {
        "xgboost": xgb.XGBRegressor(**champion_params(seed)),
        "lightgbm": lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=16, max_depth=4,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            max_features="sqrt", random_state=seed, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            max_features="sqrt", random_state=seed, n_jobs=-1),
        "ridge": Ridge(alpha=10.0, random_state=seed),
        "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=seed,
                                  max_iter=5000),
    }
    try:
        from catboost import CatBoostRegressor
        out["catboost"] = CatBoostRegressor(
            iterations=400, learning_rate=0.05, depth=4, random_seed=seed,
            verbose=0, allow_writing_files=False)
    except ImportError:
        pass
    return out


IS_TREE = {"xgboost", "lightgbm", "catboost", "random_forest", "extra_trees"}
IS_LINEAR = {"ridge", "elastic_net"}


def _native(name, model, X_tr) -> np.ndarray:
    """Whatever this model reports when you ask it what mattered."""
    if name in IS_LINEAR:
        # |coef| on standardised columns -- the scale-free form, and what
        # SHAP reduces to for a linear model with independent inputs.
        sd = X_tr.std(axis=0)
        v = np.abs(np.asarray(model.coef_, dtype=float)) * sd
    else:
        v = np.abs(np.asarray(model.feature_importances_, dtype=float))
    s = v.sum()
    return v / s if s > 0 else v


def _shap(name, model, X_va, n_rows: int, seed: int) -> np.ndarray | None:
    """Mean |phi| per column, TreeSHAP, on a validation subsample."""
    if name not in IS_TREE:
        return None
    import shap
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_va.shape[0], size=min(n_rows, X_va.shape[0]),
                     replace=False)
    ex = shap.TreeExplainer(model)
    phi = ex.shap_values(X_va[idx], check_additivity=False)
    v = np.abs(np.asarray(phi, dtype=float)).mean(axis=0)
    s = v.sum()
    return v / s if s > 0 else v


def _permutation(model, X, y, cols, n_repeats: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = spearmanr(y, model.predict(X)).statistic
    out = np.zeros(len(cols))
    for j, c in enumerate(cols):
        drops = []
        for _ in range(n_repeats):
            saved = X[:, c].copy()
            X[:, c] = rng.permutation(saved)
            drops.append(base - spearmanr(y, model.predict(X)).statistic)
            X[:, c] = saved
        out[j] = float(np.mean(drops))
    return out


def _rho(a, b) -> float:
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else np.nan


def run(families=("a_flank",), seed: int = 41, n_splits: int = 3,
        n_features: int | None = None, shap_rows: int = 1000,
        perm_repeats: int = 2, verbose: bool = True):
    from sklearn.model_selection import KFold

    from ..build.build_matrix import load_dataset
    from ..harness.run_ablation import _impute

    n_features = n_features or config.N_FEATURES
    X, y, names, _, _ = load_dataset(list(families), verbose=verbose)
    names = np.asarray(names)
    blocks = np.array([block_of(n) for n in names])
    labels = sorted(set(blocks))
    if verbose:
        print(f"  {X.shape[1]:,} columns in {len(labels)} blocks")

    pair_rows, method_rows, share_rows = [], [], []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        X_tr, X_va = _impute(X[tr], X[va])
        y_tr, y_va = y[tr], y[va]

        imp = {}          # (model, method) -> full-length importance vector
        tops = {}         # model -> indices of its own top n_features
        score = {}
        for name, model in _models(seed).items():
            t0 = time.time()
            model.fit(X_tr, y_tr)
            nat = _native(name, model, X_tr)
            top = np.argsort(nat)[::-1][:n_features]
            imp[(name, "native")] = nat
            tops[name] = top
            score[name] = _rho(y_va, model.predict(X_va))

            sv = _shap(name, model, X_va, shap_rows, seed)
            if sv is not None:
                imp[(name, "shap")] = sv

            perm_full = np.zeros(X.shape[1])
            perm_full[top] = _permutation(model, X_va.copy(), y_va, top,
                                          perm_repeats, seed)
            imp[(name, "permutation")] = perm_full
            if verbose:
                print(f"    fold {fold} {name:14s} rho {score[name]:.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

        models = sorted(tops)

        # --- within a model: do its own methods agree? ----------------------
        for name in models:
            avail = [m for m in ("native", "shap", "permutation")
                     if (name, m) in imp]
            for m1, m2 in itertools.combinations(avail, 2):
                t = tops[name]
                method_rows.append(dict(
                    fold=fold, model=name, method_a=m1, method_b=m2,
                    rho_column=_rho(imp[(name, m1)][t], imp[(name, m2)][t]),
                    rho_block=_rho(
                        pd.Series(imp[(name, m1)]).groupby(blocks).sum()
                        .reindex(labels).fillna(0.0),
                        pd.Series(imp[(name, m2)]).groupby(blocks).sum()
                        .reindex(labels).fillna(0.0)),
                ))

        # --- between models: same method, different model -------------------
        for method in ("native", "shap", "permutation"):
            present = [m for m in models if (m, method) in imp]
            for a, b in itertools.combinations(present, 2):
                ia, ib = imp[(a, method)], imp[(b, method)]
                sa, sb = set(tops[a].tolist()), set(tops[b].tolist())
                union = sorted(sa | sb)
                ga = pd.Series(ia).groupby(blocks).sum().reindex(labels).fillna(0.0)
                gb = pd.Series(ib).groupby(blocks).sum().reindex(labels).fillna(0.0)
                pair_rows.append(dict(
                    fold=fold, method=method, model_a=a, model_b=b,
                    rho_a=score[a], rho_b=score[b],
                    rho_column=_rho(ia[union], ib[union]),
                    rho_block=_rho(ga, gb),
                    selected_overlap=len(sa & sb),
                    selected_kuncheva=kuncheva(sa, sb, X.shape[1]),
                ))

        # --- where each model puts its weight, by block ---------------------
        for (name, method), v in imp.items():
            g = pd.Series(v).groupby(blocks).sum().reindex(labels).fillna(0.0)
            tot = g.sum()
            for lab in labels:
                share_rows.append(dict(
                    fold=fold, model=name, method=method, block=lab,
                    n_cols=int((blocks == lab).sum()),
                    share=float(g[lab] / tot) if tot else 0.0,
                ))

        pd.DataFrame(pair_rows).to_csv(PAIRS, index=False)
        pd.DataFrame(method_rows).to_csv(METHODS, index=False)
        pd.DataFrame(share_rows).to_csv(SHARES, index=False)

    return (pd.DataFrame(pair_rows), pd.DataFrame(method_rows),
            pd.DataFrame(share_rows))


def summary():
    pairs = pd.read_csv(PAIRS)
    meth = pd.read_csv(METHODS)
    sh = pd.read_csv(SHARES)

    print("\n== between models, same method (mean over folds and pairs) ==")
    print(pairs.groupby("method")[["rho_column", "rho_block",
                                   "selected_overlap", "selected_kuncheva"]]
          .mean().round(3).to_string())

    print("\n== between models, per pair (native importance) ==")
    nat = pairs[pairs["method"] == "native"]
    print(nat.groupby(["model_a", "model_b"])[["rho_column", "rho_block",
                                               "selected_overlap"]]
          .mean().round(3).to_string())

    print("\n== within one model, between its methods ==")
    print(meth.groupby(["model", "method_a", "method_b"])[["rho_column", "rho_block"]]
          .mean().round(3).to_string())

    print("\n== share of importance per block, native (%) ==")
    piv = (sh[sh["method"] == "native"]
           .groupby(["block", "model"])["share"].mean().unstack() * 100).round(1)
    piv["n_cols"] = sh.groupby("block")["n_cols"].first()
    print(piv.sort_values("xgboost", ascending=False).to_string())
    return pairs, meth, sh


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--shap-rows", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.run:
        run(n_splits=args.folds, shap_rows=args.shap_rows)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
