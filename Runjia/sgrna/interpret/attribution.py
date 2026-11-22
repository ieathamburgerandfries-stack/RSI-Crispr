"""Two models, different features, same score. Are they disagreeing about biology?

The observation that prompts this: XGBoost's own two importance methods rank
its top features at rho -0.02 (i.e. independently), LightGBM's agree at +0.44,
the two models pick substantially different features, and they score the same.
That sounds like a contradiction. It is not -- but saying so is worth nothing
without a test, so this module runs one.

The hypothesis is **redundancy**: the matrix carries each underlying quantity in
many correlated columns, so there are many near-equivalent ways to reach the
same prediction. If that is right, then the two models' chosen columns should
be *different names for the same variables* -- different labels, same
information -- and three things should hold:

  1. Column overlap between the two top sets should be low.
  2. But every column one model picked and the other did not should have a
     close correlate inside the other model's set.
  3. And the *share of importance* each model gives to each kind of feature
     should agree, even though the individual columns do not.

The alternative hypothesis -- that the models have genuinely found different
biology -- predicts low overlap, *no* close correlates, and disagreeing shares.

A permuted-column control is included for scale: two selections that genuinely
share nothing still overlap a little by luck, and the matched-correlate
statistic needs a null to be read against.

    python -m sgrna.interpret.attribution
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config

OUT = config.RESULTS / "attribution.csv"
OUT_SUMMARY = config.RESULTS / "attribution_summary.csv"


def _paths(importance: str):
    """Keep the gain-based figures on disk beside the SHAP ones."""
    if importance == "native":
        return OUT, OUT_SUMMARY
    return (config.RESULTS / f"attribution_{importance}.csv",
            config.RESULTS / f"attribution_summary_{importance}.csv")


def _kind(name: str) -> str:
    """What quantity does this column describe? Coarse, hypothesis-level."""
    n = name
    if n.startswith("eng.flank.comp.") or n.startswith("eng.flank.win."):
        side = "up" if ".up" in n else ("dn" if ".dn" in n else "?")
        return f"flank window ({side}stream)"
    if n.startswith("eng.flank.seq.") or n.startswith("eng.flank.qct."):
        return "flank, nearest 10 nt"
    if n.startswith("V"):
        return "target: position/k-mer indicator"
    if "monomer" in n or "dimer" in n or "trimer" in n or "tetramer" in n:
        return "target: quantum descriptor"
    return "target: other published column"


def run(families=("a_flank",), seed: int = 41, n_features: int = 300,
        top: int = 50, verbose: bool = True,
        importance: str = "shap") -> pd.DataFrame:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import KFold

    from ..build.build_matrix import load_dataset
    from .importance import shap_importance
    from ..harness.run_ablation import _impute, champion_params

    if importance not in ("native", "shap"):
        raise ValueError(f"unknown importance {importance!r}")

    X, y, names, _, _ = load_dataset(list(families), verbose=verbose)
    names = np.asarray(names)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    tr, va = next(iter(kf.split(X)))
    X_tr, X_va = _impute(X[tr], X[va])
    y_tr, y_va = y[tr], y[va]

    # Each model selects with its own gain, as in the interpretability run --
    # the question is what each one chooses when left to itself.
    fits = {}
    for tag in ("xgboost", "lightgbm"):
        if tag == "xgboost":
            sel = xgb.XGBRegressor(**champion_params(seed))
        else:
            sel = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                    num_leaves=16, max_depth=4, subsample=0.8,
                                    subsample_freq=1, colsample_bytree=0.8,
                                    random_state=seed, n_jobs=-1, verbose=-1)
        sel.fit(X_tr, y_tr)
        if importance == "shap":
            imp = shap_importance(sel, X_va, seed=seed)
            if imp is None:
                raise RuntimeError("shap is not installed; pass "
                                   "importance='native' for the old figures")
        else:
            imp = np.asarray(sel.feature_importances_, dtype=float)
        order = np.argsort(imp)[::-1][:n_features]

        if tag == "xgboost":
            model = xgb.XGBRegressor(**champion_params(seed))
        else:
            model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                      num_leaves=16, max_depth=4, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.8,
                                      random_state=seed, n_jobs=-1, verbose=-1)
        model.fit(X_tr[:, order], y_tr)
        pred = np.asarray(model.predict(X_va[:, order])).ravel()
        fits[tag] = dict(order=order, imp=imp,
                         rho=float(spearmanr(y_va, pred).statistic))
        if verbose:
            print(f"  {tag:9s} held-out rho {fits[tag]['rho']:.4f}")

    a = fits["xgboost"]["order"][:top]
    b = fits["lightgbm"]["order"][:top]
    shared = np.intersect1d(a, b)

    # --- test 2: does each unshared pick have a close correlate in the other set?
    Xs = X_tr
    def _corr_matrix(idx_a, idx_b):
        A = Xs[:, idx_a]
        B = Xs[:, idx_b]
        A = (A - A.mean(0)) / np.where(A.std(0) == 0, 1, A.std(0))
        B = (B - B.mean(0)) / np.where(B.std(0) == 0, 1, B.std(0))
        return np.abs(A.T @ B) / len(Xs)

    only_a = np.setdiff1d(a, b)
    only_b = np.setdiff1d(b, a)
    M = _corr_matrix(only_a, b)
    best_a = M.max(axis=1) if M.size else np.array([])
    M2 = _corr_matrix(only_b, a)
    best_b = M2.max(axis=1) if M2.size else np.array([])

    # Null: random columns from the same matrix, matched for count.
    rng = np.random.default_rng(seed)
    rand = rng.choice(X.shape[1], size=len(only_a), replace=False)
    Mnull = _corr_matrix(rand, b)
    best_null = Mnull.max(axis=1) if Mnull.size else np.array([])

    rows = []
    for j in only_a:
        k = int(np.argmax(_corr_matrix(np.array([j]), b)[0]))
        rows.append(dict(picked_by="xgboost only", column=names[j],
                         kind=_kind(names[j]),
                         closest_in_other_set=names[b[k]],
                         abs_corr=float(_corr_matrix(np.array([j]), b)[0][k])))
    out, out_summary = _paths(importance)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    # --- test 3: importance share by kind of feature
    share = {}
    for tag in ("xgboost", "lightgbm"):
        imp = fits[tag]["imp"]
        idx = fits[tag]["order"]
        tot = imp[idx].sum()
        s = {}
        for j in idx:
            s[_kind(names[j])] = s.get(_kind(names[j]), 0.0) + imp[j] / tot
        share[tag] = s
    kinds = sorted(set(share["xgboost"]) | set(share["lightgbm"]))
    sh = pd.DataFrame({
        "kind": kinds,
        "xgboost_share": [round(share["xgboost"].get(k, 0.0), 4) for k in kinds],
        "lightgbm_share": [round(share["lightgbm"].get(k, 0.0), 4) for k in kinds],
    })
    sh["difference"] = (sh["xgboost_share"] - sh["lightgbm_share"]).round(4)
    sh.to_csv(out_summary, index=False)

    if verbose:
        print()
        print(f"  top-{top} columns shared between the two models: "
              f"{len(shared)} / {top}")
        print(f"  rank correlation of the two {importance} rankings, all columns: "
              f"{spearmanr(fits['xgboost']['imp'], fits['lightgbm']['imp']).statistic:+.3f}")
        print()
        print("  for columns only one model picked, |correlation| with its "
              "closest counterpart in the other model's set:")
        for lab, arr in (("xgboost-only -> lightgbm set", best_a),
                         ("lightgbm-only -> xgboost set", best_b),
                         ("random columns -> lightgbm set (null)", best_null)):
            if arr.size:
                print(f"    {lab:40s} median {np.median(arr):.3f}   "
                      f"share > 0.7: {np.mean(arr > 0.7):.0%}")
        print()
        print("  importance share by kind of feature:")
        print(sh.to_string(index=False))
    return sh


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--importance", choices=["native", "shap"], default="shap")
    args = ap.parse_args(argv)
    run(top=args.top, seed=args.seed, importance=args.importance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
