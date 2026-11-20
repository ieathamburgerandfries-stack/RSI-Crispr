"""Dropping a model's top columns costs nothing. Is that because they are interchangeable?

The existing faithfulness check removes each model's top 20 **columns** and
refits: it costs +0.008 for LightGBM and +0.003 for XGBoost, barely more than
removing 20 columns at random. That was read as "the top of the ranking is not
load-bearing".

There is a competing reading, and it is the more likely one on this matrix. If
the top 20 columns are twenty spellings of the same few quantities, then removing
them removes no *information* -- the model rebuilds the same function from the
correlates left behind. On that reading the test says nothing about whether the
attribution is right; it only re-demonstrates redundancy.

The two readings separate if you remove the *quantity* rather than the columns.
So: take the top-ranked column, remove it together with everything correlated
with it above a threshold -- that whole set is one "thing" -- then the next
surviving column and its correlates, and so on. Removing the first few things
should hurt, if the ranking has found anything real.

Controls, because an arm that removes 900 columns and loses accuracy has not
proved anything on its own:

  matched random   the same number of columns, chosen at random
  matched blocks   the same number of columns, taken as whole correlation
                   clusters built around randomly chosen seeds

The second is the one that matters: it removes the same amount of *structure*,
just not the structure the ranking pointed at.

    python -m sgrna.faithfulness --run
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config

OUT = config.RESULTS / "faithfulness_groups.csv"
OUT_CLUSTERS = config.RESULTS / "faithfulness_clusters.csv"
OUT_REDUCED = config.RESULTS / "faithfulness_groups_reduced.csv"
OUT_CLUSTERS_REDUCED = config.RESULTS / "faithfulness_clusters_reduced.csv"


def reduced_columns(names, family_of) -> np.ndarray:
    """The §10 reduced representation: mono + di indicators, the 63 leftovers,
    and family A. Restricting to it removes most of the restatement, so a
    faithfulness test run here is not protected by redundancy."""
    from .representation import column_sets

    family_of = np.asarray(family_of)
    base = np.where(family_of == "base")[0]
    extra = np.where(family_of != "base")[0]
    sets = column_sets([names[i] for i in base])
    keep = np.concatenate([base[sets["mono_di_364"]],
                           base[sets["hand_named_only"]], extra])
    return np.sort(keep)


def _standardise(X: np.ndarray) -> np.ndarray:
    Z = X.astype(np.float32, copy=True)
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    return (Z - mu) / sd


def clusters_around(Z: np.ndarray, order: np.ndarray, tau: float = 0.7,
                    n_clusters: int = 6) -> list[np.ndarray]:
    """Greedy: the top-ranked surviving column plus everything |r| >= tau with it.

    `Z` is standardised, `order` is column indices best-first. Each returned
    array is one "thing": a quantity the matrix happens to spell several ways.
    """
    n = Z.shape[0]
    taken = np.zeros(Z.shape[1], bool)
    out = []
    for seed_col in order:
        if taken[seed_col]:
            continue
        r = np.abs(Z.T @ Z[:, seed_col]) / n
        members = np.where((r >= tau) & (~taken))[0]
        if seed_col not in members:
            members = np.append(members, seed_col)
        taken[members] = True
        out.append(np.sort(members))
        if len(out) >= n_clusters:
            break
    return out


def run(families=("a_flank",), models=("xgboost", "lightgbm"), seed: int = 41,
        n_splits: int = 3, tau: float = 0.7, depths=(1, 2, 3, 5),
        fixed=(20, 50, 100), reduced: bool = False,
        verbose: bool = True) -> pd.DataFrame:
    import lightgbm as lgb
    import xgboost as xgb
    from scipy.stats import spearmanr
    from sklearn.model_selection import KFold

    from ..build.build_matrix import load_dataset
    from .importance import shap_importance
    from ..harness.run_ablation import _impute, champion_params

    X, y, names, _, family_of = load_dataset(list(families), verbose=verbose)
    if reduced:
        keep = reduced_columns(list(names), family_of)
        X = np.ascontiguousarray(X[:, keep])
        names = [names[i] for i in keep]
        if verbose:
            print(f"  reduced representation: {X.shape[1]:,} columns",
                  flush=True)
    names = np.asarray(names)
    out, out_clusters = ((OUT_REDUCED, OUT_CLUSTERS_REDUCED) if reduced
                         else (OUT, OUT_CLUSTERS))

    def _make(kind):
        if kind == "xgboost":
            return xgb.XGBRegressor(**champion_params(seed))
        return lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=16, max_depth=4,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1)

    rows, cluster_rows = [], []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        X_tr, X_va = _impute(X[tr], X[va])
        y_tr, y_va = y[tr], y[va]
        Z = _standardise(X_tr)
        rng = np.random.default_rng(seed + fold)

        for kind in models:
            full = _make(kind)
            full.fit(X_tr, y_tr)
            imp = shap_importance(full, X_va, seed=seed)
            if imp is None:
                continue
            order = np.argsort(imp)[::-1]
            cl = clusters_around(Z, order[:200], tau=tau,
                                 n_clusters=max(depths))
            if verbose:
                sizes = [len(c) for c in cl]
                print(f"    fold {fold} {kind}: cluster sizes {sizes}",
                      flush=True)
            for i, c in enumerate(cl, start=1):
                cluster_rows.append(dict(
                    fold=fold, model=kind, rank=i, n_columns=int(len(c)),
                    seed_column=str(names[order[0]]) if i == 1 else "",
                    shap_share=float(imp[c].sum() / imp.sum())))

            base_rho = float(spearmanr(
                y_va, _make(kind).fit(X_tr, y_tr).predict(X_va)).statistic)

            def _refit(drop_mask, label, k, n_dropped):
                keep = np.where(~drop_mask)[0]
                m = _make(kind)
                m.fit(X_tr[:, keep], y_tr)
                rho = float(spearmanr(y_va, m.predict(X_va[:, keep])).statistic)
                rows.append(dict(fold=fold, model=kind, arm=label,
                                 things_removed=k, n_columns_removed=n_dropped,
                                 spearman=rho, delta=rho - base_rho))
                if verbose:
                    print(f"      {label:22s} k={k} ({n_dropped:4d} cols) "
                          f"rho {rho:.4f}  delta {rho - base_rho:+.4f}",
                          flush=True)

            rows.append(dict(fold=fold, model=kind, arm="full model",
                             things_removed=0, n_columns_removed=0,
                             spearman=base_rho, delta=0.0))

            for k in depths:
                drop = np.concatenate(cl[:k])
                mask = np.zeros(X.shape[1], bool)
                mask[drop] = True
                _refit(mask, "top things removed", k, int(mask.sum()))

                # control 1: same count, random columns
                mask_r = np.zeros(X.shape[1], bool)
                mask_r[rng.choice(X.shape[1], size=int(mask.sum()),
                                  replace=False)] = True
                _refit(mask_r, "random columns", k, int(mask_r.sum()))

                # control 2: same count, whole clusters around random seeds
                mask_b = np.zeros(X.shape[1], bool)
                pool = rng.permutation(X.shape[1])
                for c in clusters_around(Z, pool, tau=tau, n_clusters=200):
                    mask_b[c] = True
                    if mask_b.sum() >= mask.sum():
                        break
                _refit(mask_b, "random whole clusters", k, int(mask_b.sum()))

                # for comparison with the old test: the same number of
                # individual top columns, ignoring their correlates
                mask_c = np.zeros(X.shape[1], bool)
                mask_c[order[:int(mask.sum())]] = True
                _refit(mask_c, "top columns, no clusters", k, int(mask_c.sum()))

            # The original faithfulness test, at fixed column counts, so the
            # full and reduced representations are compared on the same ask.
            for n_drop in fixed:
                if n_drop >= X.shape[1]:
                    continue
                m1 = np.zeros(X.shape[1], bool)
                m1[order[:n_drop]] = True
                _refit(m1, f"top {n_drop} columns", 0, n_drop)
                m2 = np.zeros(X.shape[1], bool)
                m2[rng.choice(X.shape[1], size=n_drop, replace=False)] = True
                _refit(m2, f"random {n_drop} columns", 0, n_drop)

            pd.DataFrame(rows).to_csv(out, index=False)
            pd.DataFrame(cluster_rows).to_csv(out_clusters, index=False)
    return pd.DataFrame(rows)


def summary(reduced: bool = False):
    out, out_clusters = ((OUT_REDUCED, OUT_CLUSTERS_REDUCED) if reduced
                         else (OUT, OUT_CLUSTERS))
    d = pd.read_csv(out)
    fixed = d[d["arm"].str.contains("columns") & (d["things_removed"] == 0)]
    if len(fixed):
        print("\n== the original test, at fixed column counts (mean delta rho) ==")
        print(fixed.pivot_table(index="model", columns="arm", values="delta")
              .round(4).to_string())
    d = d[~d["arm"].str.match(r"(top|random) \d+ columns$")]
    piv = (d[d["arm"] != "full model"]
           .pivot_table(index=["model", "things_removed"], columns="arm",
                        values="delta"))
    cols = [c for c in ("top things removed", "top columns, no clusters",
                        "random whole clusters", "random columns")
            if c in piv.columns]
    print("\n== change in held-out rho when columns are removed and the model refitted ==")
    print(piv[cols].round(4).to_string())
    n = (d[d["arm"] == "top things removed"]
         .groupby(["model", "things_removed"])["n_columns_removed"].mean())
    print("\ncolumns removed per arm:")
    print(n.round(0).to_string())
    if out_clusters.exists():
        c = pd.read_csv(out_clusters)
        print("\n== the things themselves (mean over folds) ==")
        print(c.groupby(["model", "rank"])[["n_columns", "shap_share"]]
              .mean().round(3).to_string())
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--fixed", type=int, nargs="*", default=None,
                    help="column counts for the original drop-top-N test; "
                         "pass matched *fractions* when comparing "
                         "representations of different widths")
    ap.add_argument("--reduced", action="store_true",
                    help="run on the reduced representation instead")
    a = ap.parse_args(argv)
    if a.run:
        run(n_splits=a.folds, tau=a.tau, reduced=a.reduced,
            fixed=tuple(a.fixed) if a.fixed else (20, 50, 100))
    summary(reduced=a.reduced)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
