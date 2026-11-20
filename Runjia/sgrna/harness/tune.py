"""Does the head-to-head margin survive hyper-parameter tuning?

The §14 comparison is untuned against untuned. Our LightGBM runs on
hand-chosen settings; crisprHAL 2 runs on the settings its authors shipped.
Neither was searched. That is defensible as a like-for-like comparison, but it
leaves an obvious objection: the +0.0107 margin might be an accident of which
defaults happened to suit which model.

Tuning both would settle it. Tuning theirs is not practical here -- one
configuration costs ~105 minutes of CPU for five folds, so a twelve-point
search is a day of compute. So this module answers the half that *is*
affordable and reports it as an uncertainty band rather than an improvement:

    how far can tuning move *our* number?

If the answer is "less than the margin", the margin is not an artefact of our
defaults. If it is "more", then the honest statement is that the two models are
indistinguishable until both are tuned, and this module says so.

Protocol, and the reason for it:

  * The search runs **inside each training fold**. A configuration is chosen on
    an inner split of the training rows and then applied once to the untouched
    validation fold. Picking the configuration by validation score -- the usual
    shortcut -- would report the best of twelve draws on the test set, which is
    not a held-out number at all.
  * Feature selection is done **once per outer fold** and shared across
    configurations. It is 31 s of the 35 s fold (`results/timing.csv`) and does
    not depend on the LightGBM settings, so repeating it twelve times would
    measure patience rather than tuning.
  * The grid is a random search over the parameters that actually matter for
    this kind of tabular problem: capacity (`num_leaves`, `max_depth`,
    `min_child_samples`), how fast it learns (`learning_rate`,
    `n_estimators`), how much it subsamples (`feature_fraction`, `bagging`),
    and regularisation (`lambda_l2`).

    python -m sgrna.tune --run --trials 12
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from .. import config

RESULTS_PATH = config.RESULTS / "tuning.csv"
TRACE_PATH = config.RESULTS / "tuning_trace.csv"

# The settings every number in this project so far was produced with.
DEFAULT = dict(
    n_estimators=400, learning_rate=0.05, num_leaves=16, max_depth=4,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
)


def sample_config(rng: np.random.Generator) -> dict:
    """One draw from the search space."""
    return dict(
        n_estimators=int(rng.choice([200, 400, 800, 1600])),
        learning_rate=float(rng.choice([0.01, 0.02, 0.05, 0.1])),
        num_leaves=int(rng.choice([8, 16, 31, 63, 127])),
        max_depth=int(rng.choice([3, 4, 6, 8, -1])),
        min_child_samples=int(rng.choice([5, 20, 50, 100])),
        colsample_bytree=float(rng.choice([0.4, 0.6, 0.8, 1.0])),
        subsample=float(rng.choice([0.6, 0.8, 1.0])),
        subsample_freq=1,
        reg_lambda=float(rng.choice([0.0, 1.0, 10.0, 100.0])),
    )


def _fit_score(params: dict, X_tr, y_tr, X_va, y_va, seed: int):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **params)
    model.fit(X_tr, y_tr)
    pred = np.asarray(model.predict(X_va)).ravel()
    return (float(spearmanr(y_va, pred).statistic), float(r2_score(y_va, pred)),
            pred)


def run(arm: str = "curated_flank", seed: int = 41, n_splits: int = 5,
        trials: int = 12, inner_frac: float = 0.2, patience: int | None = None,
        deadline: float | None = None, verbose: bool = True) -> pd.DataFrame:
    """Default settings vs a tuned-inside-the-fold search, same folds.

    `trials` is now a ceiling rather than a target. With `patience`, a fold's
    search stops once the best inner score has not improved for that many
    consecutive draws, and every draw is written to `tuning_trace.csv` so the
    plateau can be shown rather than asserted -- "we searched until it stopped
    improving" is a claim about a curve, and the curve should be on file.
    """
    import xgboost as xgb
    from sklearn.model_selection import KFold

    from . import evaluate as ev
    from ..compare.headtohead import _assemble
    from .run_ablation import _impute, make_selector

    X, y, _ = _assemble(arm, verbose=verbose)

    prev = pd.DataFrame()
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size:
        prev = pd.read_csv(RESULTS_PATH)
    done = set(zip(prev.get("arm", []), prev.get("fold", []))) if len(prev) else set()

    rows = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        if (arm, fold) in done:
            if verbose:
                print(f"  fold {fold} already recorded")
            continue
        if deadline and time.time() > deadline:
            if verbose:
                print("    time budget reached")
            break

        t0 = time.time()
        X_tr, X_va = _impute(X[tr], X[va])
        sel = make_selector(seed)
        sel.fit(X_tr, y[tr])
        top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
        A, B = X_tr[:, top], X_va[:, top]
        y_tr, y_va = y[tr], y[va]
        select_s = time.time() - t0

        # Inner split of the training rows. The configuration is chosen here
        # and nowhere else; the validation fold is scored exactly once per arm.
        inner = KFold(n_splits=int(round(1 / inner_frac)), shuffle=True,
                      random_state=seed + fold)
        itr, iva = next(iter(inner.split(A)))

        rng = np.random.default_rng(seed * 1000 + fold)
        best, best_params, trace = -np.inf, None, []
        since_improved = 0
        t1 = time.time()
        for t in range(trials):
            params = sample_config(rng)
            rho_inner, _, _ = _fit_score(params, A[itr], y_tr[itr],
                                         A[iva], y_tr[iva], seed)
            if rho_inner > best:
                best, best_params, since_improved = rho_inner, params, 0
            else:
                since_improved += 1
            trace.append(dict(arm=arm, seed=seed, fold=fold, trial=t + 1,
                              inner_rho=rho_inner, best_so_far=best,
                              trials_since_improved=since_improved))
            if patience and since_improved >= patience:
                if verbose:
                    print(f"      fold {fold}: no improvement in {patience} "
                          f"draws, stopping at {t + 1}", flush=True)
                break
            if deadline and time.time() > deadline:
                if verbose:
                    print(f"      fold {fold}: time budget reached at draw "
                          f"{t + 1}", flush=True)
                break
        search_s = time.time() - t1
        tdf = pd.DataFrame(trace)
        if TRACE_PATH.exists() and TRACE_PATH.stat().st_size:
            old = pd.read_csv(TRACE_PATH)
            old = old[~((old["arm"] == arm) & (old["fold"] == fold))]
            tdf = pd.concat([old, tdf], ignore_index=True)
        tdf.to_csv(TRACE_PATH, index=False)

        rho_def, r2_def, _ = _fit_score(DEFAULT, A, y_tr, B, y_va, seed)
        rho_tun, r2_tun, _ = _fit_score(best_params, A, y_tr, B, y_va, seed)

        row = dict(
            arm=arm, seed=seed, fold=fold, trials=len(trace),
            trials_allowed=trials,
            rho_default=rho_def, r2_default=r2_def,
            rho_tuned=rho_tun, r2_tuned=r2_tun,
            delta_rho=rho_tun - rho_def,
            inner_rho_best=best, inner_rho_default=np.nan,
            select_seconds=round(select_s, 1),
            search_seconds=round(search_s, 1),
            best_params=json.dumps(best_params),
        )
        # How much of any gain is just the inner split being easier?
        row["inner_rho_default"] = _fit_score(
            DEFAULT, A[itr], y_tr[itr], A[iva], y_tr[iva], seed)[0]
        rows.append(row)
        pd.concat([prev, pd.DataFrame(rows)]).to_csv(RESULTS_PATH, index=False)
        if verbose:
            print(f"    fold {fold}  default {rho_def:.4f}  tuned {rho_tun:.4f}  "
                  f"delta {row['delta_rho']:+.4f}  ({search_s:.0f}s search)",
                  flush=True)
    return pd.DataFrame(rows)


def summary() -> pd.DataFrame:
    d = pd.read_csv(RESULTS_PATH)
    from scipy.stats import ttest_rel
    out = []
    for arm, g in d.groupby("arm"):
        t = ttest_rel(g["rho_tuned"], g["rho_default"]) if len(g) > 1 else None
        out.append(dict(
            arm=arm, folds=len(g),
            rho_default=round(g["rho_default"].mean(), 4),
            rho_tuned=round(g["rho_tuned"].mean(), 4),
            delta=round(g["delta_rho"].mean(), 4),
            sd=round(g["delta_rho"].std(), 4),
            folds_improved=int((g["delta_rho"] > 0).sum()),
            p=round(float(t.pvalue), 4) if t is not None else np.nan,
        ))
    return pd.DataFrame(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--arm", default="curated_flank")
    ap.add_argument("--trials", type=int, default=12,
                    help="ceiling on draws per fold")
    ap.add_argument("--patience", type=int, default=None,
                    help="stop a fold's search after this many draws with no "
                         "improvement")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--time-budget", type=float, default=None)
    args = ap.parse_args(argv)

    deadline = time.time() + args.time_budget if args.time_budget else None
    if args.run:
        run(arm=args.arm, seed=args.seed, n_splits=args.folds,
            trials=args.trials, patience=args.patience, deadline=deadline)
    print(summary().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
