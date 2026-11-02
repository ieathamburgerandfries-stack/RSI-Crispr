"""Does the human baseline reproduce ρ ≈ 0.472 under this harness?

Part 3 says the cross-kingdom conclusion leans on the human side being weak, and
cites a teammate's ρ ≈ 0.472 against the 0.404 our harness measures — a figure
whose provenance was missing for days (item 4). The settings arrived on
17 October (`Jacky/human-boundary/HUMAN_BASELINE_SETTINGS.md`): **400 selected
columns, 1,000 estimators, every other XGBoost knob at library defaults**, from
an early personal script that predates the merged pipeline.

So the question is whether that configuration reproduces here. Three arms on the
same folds of the human iRF matrix (Noshay et al.'s Supplemental Table 3,
`data/raw/human_feature_matrix.csv`):

    harness_default   300 columns, the champion settings -- what produces 0.404
    jacky_400_1000    400 columns, n_estimators=1000, XGBoost defaults otherwise
    jacky_cols_only   400 columns, champion settings -- separates the column
                      count from the tree count, since only one of them can be
                      doing the work

Library defaults matter here and are not a detail: XGBoost's defaults are
`max_depth=6, learning_rate=0.3`, far deeper and faster-learning than the
champion's `max_depth=4, learning_rate=0.05`. On a 17,421-row matrix that is a
different model, not a tweak.

    python -m sgrna.human_baseline
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from . import config

OUT = config.RESULTS / "human_baseline.csv"
MATRIX = config.DATA / "raw" / "human_feature_matrix.csv"
TARGET = "cut.score"


def load(verbose: bool = True):
    d = pd.read_csv(MATRIX, skiprows=1, low_memory=False)
    y = pd.to_numeric(d[TARGET], errors="coerce").to_numpy(float)
    X = d.drop(columns=[c for c in (config.ID_COL, "sgRNAID", TARGET)
                        if c in d.columns])
    X = X.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    ok = np.isfinite(y)
    if verbose:
        print(f"  human matrix: {ok.sum():,} guides x {X.shape[1]:,} columns")
    return X[ok], y[ok]


def run(seeds=(41, 42, 43), n_splits=5, verbose: bool = True):
    import xgboost as xgb

    from .run_ablation import _impute, champion_params, make_selector

    X, y = load(verbose=verbose)
    arms = {
        "harness_default": dict(n_features=config.N_FEATURES, params=None),
        "jacky_400_1000": dict(n_features=400,
                               params=dict(n_estimators=1000, max_depth=6,
                                           learning_rate=0.3, subsample=1.0,
                                           colsample_bytree=1.0, reg_lambda=1.0,
                                           min_child_weight=1)),
        "jacky_cols_only": dict(n_features=400, params=None),
    }

    rows = []
    for arm, cfg in arms.items():
        rhos = []
        for seed in seeds:
            for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(X):
                X_tr, X_va = _impute(X[tr], X[va])
                sel = make_selector(seed)
                sel.fit(X_tr, y[tr])
                top = np.argsort(sel.feature_importances_)[::-1][:cfg["n_features"]]
                p = dict(champion_params(seed))
                p.update(cfg["params"] or {})
                m = xgb.XGBRegressor(**p)
                m.fit(X_tr[:, top], y[tr])
                rhos.append(float(spearmanr(y[va], m.predict(X_va[:, top])).statistic))
        rows.append(dict(arm=arm, n_features=cfg["n_features"],
                         spearman=float(np.mean(rhos)),
                         spearman_sd=float(np.std(rhos, ddof=1)),
                         n_fits=len(rhos)))
        if verbose:
            print(f"  {arm:>16}: rho {rows[-1]['spearman']:.4f} "
                  f"± {rows[-1]['spearman_sd']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    out = pd.DataFrame(rows)
    if verbose:
        print(f"\n  the figure to reproduce is 0.472; our harness reports 0.404")
        print(f"written -> {OUT}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--decompose", action="store_true",
                    help="separate the knobs instead of running the arms")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="restrict --decompose to these arms")
    ap.add_argument("--additivity", action="store_true",
                    help="ridge against boosting on the same selected columns")
    a = ap.parse_args(argv)
    if a.additivity:
        additivity()
    elif a.decompose:
        decompose(only=tuple(a.arms) if a.arms else None)
    else:
        run()
    return 0



# --------------------------------------------------------------------------
# Which knob does the work?
# --------------------------------------------------------------------------
#
# `jacky_400_1000` differs from the champion in three ways at once -- 1,000
# trees instead of 400, max_depth 6 and learning_rate 0.3 instead of 4 and 0.05,
# and no row/column subsampling. The first is capacity, which is what the
# teammate's note describes ("n_estimators kept improving with no clear
# plateau, unlike bacteria") and which would be a statement about the human
# dataset. The second is model shape, which would be a statement about our
# hyperparameters. These predict different things, so they are worth separating.
#
# Each arm below changes exactly one of the three, on the same folds and the
# same 400 columns as `jacky_cols_only`, which is the reference.

KNOBS = config.RESULTS / "human_baseline_knobs.csv"


def _merge(rows, out) -> None:
    """Add these arms to the file without dropping arms already measured."""
    new = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        old = old[~old["arm"].isin(new["arm"])]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out, index=False)


def decompose(seeds=(41, 42, 43), n_splits=5, verbose: bool = True,
              only: tuple[str, ...] | None = None):
    import xgboost as xgb

    from .run_ablation import _impute, champion_params, make_selector

    X, y = load(verbose=verbose)
    arms = {
        "plus_trees": dict(n_estimators=1000),
        "plus_shape": dict(max_depth=6, learning_rate=0.3),
        "plus_nosubsample": dict(subsample=1.0, colsample_bytree=1.0),
        # `plus_shape` moves depth and learning rate together, so it cannot say
        # whether human wants more interaction order or just less shrinkage.
        # These two separate them; only the first is a claim about the signal.
        "plus_depth": dict(max_depth=6),
        "plus_lr": dict(learning_rate=0.3),
    }
    if only:
        arms = {k: v for k, v in arms.items() if k in only}

    rows = []
    for arm, delta in arms.items():
        rhos = []
        for seed in seeds:
            for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(X):
                X_tr, X_va = _impute(X[tr], X[va])
                sel = make_selector(seed)
                sel.fit(X_tr, y[tr])
                top = np.argsort(sel.feature_importances_)[::-1][:400]
                p = dict(champion_params(seed))
                p.update(delta)
                m = xgb.XGBRegressor(**p)
                m.fit(X_tr[:, top], y[tr])
                rhos.append(float(spearmanr(y[va], m.predict(X_va[:, top])).statistic))
        rows.append(dict(arm=arm, n_features=400, changed=";".join(delta),
                         spearman=float(np.mean(rhos)),
                         spearman_sd=float(np.std(rhos, ddof=1)),
                         n_fits=len(rhos)))
        if verbose:
            print(f"  {arm:>16}: rho {rows[-1]['spearman']:.4f} "
                  f"± {rows[-1]['spearman_sd']:.4f}", flush=True)
        _merge(rows, KNOBS)

    if verbose:
        print(f"  reference (400 cols, champion) is 0.4074; all three is 0.5076")
        print(f"written -> {KNOBS}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Is the human signal less additive than E. coli's?
# --------------------------------------------------------------------------
#
# `plus_depth` is the largest single knob (+0.0585 of the +0.1002), and depth is
# the interaction-order knob. That suggests human efficiency carries interaction
# structure that E. coli's does not -- but depth also buys plain capacity, so the
# hyperparameter alone cannot say it.
#
# The direct test is the one Appendix B already ran on E. coli: fit ridge on the
# same selected columns and see how close a straight line gets. There, ridge
# reaches 95% of boosting's rho, which is why "the signal is overwhelmingly
# additive" is in the report. If human's ratio is much lower, the depth result
# has a reading; if it matches, depth is buying capacity and nothing more.

ADDITIVITY = config.RESULTS / "human_additivity.csv"


def additivity(seeds=(41, 42, 43), n_splits=5, n_features=400,
               verbose: bool = True):
    import xgboost as xgb
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    from .run_ablation import _impute, champion_params, make_selector

    X, y = load(verbose=verbose)
    out = {"boosting_defaults": [], "boosting_champion": [], "ridge": []}
    for seed in seeds:
        for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(X):
            X_tr, X_va = _impute(X[tr], X[va])
            sel = make_selector(seed)
            sel.fit(X_tr, y[tr])
            top = np.argsort(sel.feature_importances_)[::-1][:n_features]
            A, B = X_tr[:, top], X_va[:, top]

            p = dict(champion_params(seed))
            out["boosting_champion"].append(
                _rho(xgb.XGBRegressor(**p), A, y[tr], B, y[va]))
            p.update(n_estimators=1000, max_depth=6, learning_rate=0.3,
                     subsample=1.0, colsample_bytree=1.0)
            out["boosting_defaults"].append(
                _rho(xgb.XGBRegressor(**p), A, y[tr], B, y[va]))

            sc = StandardScaler().fit(A)
            out["ridge"].append(
                _rho(RidgeCV(alphas=np.logspace(-2, 4, 25)),
                     sc.transform(A), y[tr], sc.transform(B), y[va]))

    rows = [dict(arm=k, n_features=n_features, spearman=float(np.mean(v)),
                 spearman_sd=float(np.std(v, ddof=1)), n_fits=len(v))
            for k, v in out.items()]
    d = pd.DataFrame(rows)
    d.to_csv(ADDITIVITY, index=False)
    if verbose:
        for r in rows:
            print(f"  {r['arm']:>18}: rho {r['spearman']:.4f} "
                  f"± {r['spearman_sd']:.4f}")
        best = max(r["spearman"] for r in rows if r["arm"].startswith("boost"))
        ridge = next(r["spearman"] for r in rows if r["arm"] == "ridge")
        print(f"  ridge reaches {100 * ridge / best:.1f}% of boosting "
              f"(E. coli: 95.6% on 427 columns)")
        print(f"written -> {ADDITIVITY}")
    return d


def _rho(model, X_tr, y_tr, X_va, y_va) -> float:
    model.fit(X_tr, y_tr)
    return float(spearmanr(y_va, model.predict(X_va)).statistic)

if __name__ == "__main__":
    raise SystemExit(main())
