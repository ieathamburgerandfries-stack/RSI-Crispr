"""The pipeline selects columns with the one metric the report says never to use.

`run_ablation.make_selector` fits a throwaway XGBoost and takes
`feature_importances_`, which for the sklearn wrapper is **split gain**; the
LightGBM alternative sets `importance_type="gain"` explicitly. Finding 5 says
gain is the attribution that does not survive a change of model -- it agrees
with SHAP at rho 0.05 per column against SHAP's 0.50 -- and, worse for a
selector, that **gain credits a column for being binary**, worth 26 percentage
points of XGBoost's attributed importance.

That matters here more than it would anywhere else, because the curated base is
5,957 strictly binary k-mer indicators out of 6,169, and every feature family is
dense and continuous. A ranker biased toward binary columns is therefore biased
against exactly the columns the project builds, and it decides which 300 of
6,517 the champion ever sees.

Two things follow that are worth separating, because they have different
consequences:

  composition   does the gain-selected 300 actually under-represent the dense
                family columns relative to a SHAP-selected 300? This is a
                statement about the ranking and needs no cross-validation.
  cost          does it cost accuracy? If the two selections score the same,
                the bias is an interpretability problem only -- "45 of 300
                columns carry half the gain" is a claim about a set chosen by a
                biased ranker -- and if they do not, the pipeline has been
                leaving something on the table since September.

    python -m sgrna.selector_audit
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config
from .curated_ablation import _base, _block

OUT = config.RESULTS / "selector_audit.csv"
RANKS_OUT = config.RESULTS / "selector_audit_ranks.csv"
SHAP_ROWS = 3000          # TreeSHAP is O(rows x trees x depth); 3k is ample
                          # for a mean-|SHAP| ranking and fits in a minute


def _frame():
    X, y, names, protos = _base()
    A, anames = _block("a_flank", protos)
    return (np.hstack([X, A]).astype(np.float32), y,
            list(names) + list(anames), len(names))


def _is_binary(X: np.ndarray) -> np.ndarray:
    """A column is binary if every finite value is 0 or 1."""
    out = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        v = X[:, j]
        v = v[np.isfinite(v)]
        out[j] = v.size > 0 and np.isin(v, (0.0, 1.0)).all()
    return out


def rankings(seed: int = 41, verbose: bool = True):
    """Gain and mean-|SHAP| rankings from the same fitted selector."""
    import xgboost as xgb

    from .run_ablation import _impute, selector_params

    X, y, names, n_base = _frame()
    Xi, _ = _impute(X, X[:1])
    m = xgb.XGBRegressor(**selector_params(seed))
    m.fit(Xi, y)
    gain = np.asarray(m.feature_importances_, dtype=float)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xi), size=min(SHAP_ROWS, len(Xi)), replace=False)
    booster = m.get_booster()
    dm = xgb.DMatrix(Xi[idx], feature_names=[f"f{i}" for i in range(Xi.shape[1])])
    sv = booster.predict(dm, pred_contribs=True)[:, :-1]   # last col is the bias
    shap = np.abs(sv).mean(axis=0)

    binary = _is_binary(X)
    fam = np.array(["a_flank" if i >= n_base else "base" for i in range(len(names))])
    d = pd.DataFrame(dict(column=names, family=fam, binary=binary,
                          gain=gain, shap=shap))
    d.to_csv(RANKS_OUT, index=False)
    if verbose:
        print(f"  selector fitted on {X.shape[0]:,} x {X.shape[1]:,}; "
              f"SHAP over {len(idx):,} rows")
    return d, X, y, names, n_base


def run(seed: int = 41, n_features: int = None, seeds=None, n_splits=None,
        verbose: bool = True):
    from .run_ablation import cross_validate

    nf = n_features or config.N_FEATURES
    d, X, y, names, n_base = rankings(seed, verbose=verbose)

    picks = {}
    rows = []
    for metric in ("gain", "shap"):
        top = np.argsort(d[metric].to_numpy())[::-1][:nf]
        picks[metric] = top
        sub = d.iloc[top]
        rows.append(dict(metric=metric, n_selected=len(top),
                         binary_cols=int(sub.binary.sum()),
                         binary_share=float(sub.binary.mean()),
                         a_flank_cols=int((sub.family == "a_flank").sum()),
                         dense_cols=int((~sub.binary).sum())))
        if verbose:
            r = rows[-1]
            print(f"  top {nf} by {metric:4s}: {r['binary_cols']:3d} binary "
                  f"({r['binary_share']:.0%}), {r['a_flank_cols']:3d} from "
                  f"a_flank, {r['dense_cols']:3d} dense", flush=True)

    overlap = len(set(picks["gain"].tolist()) & set(picks["shap"].tolist()))
    if verbose:
        print(f"  the two selections share {overlap} of {nf} columns "
              f"({overlap / nf:.0%})")

    # Does the bias cost accuracy? Each selection is frozen and handed to the
    # champion with selection switched off, so the only difference between the
    # arms is which columns were chosen.
    for metric in ("gain", "shap"):
        top = picks[metric]
        nm = [names[i] for i in top]
        res = cross_validate(X[:, top], y, nm, ["sel"] * len(top),
                             n_features=len(top), seeds=list(seeds or (41, 42, 43)),
                             n_splits=n_splits or config.N_SPLITS, verbose=False,
                             tag=f"selaudit_{metric}", experiment=metric)
        for r in rows:
            if r["metric"] == metric:
                r["spearman"] = res["summary"]["spearman"]["mean"]
                r["spearman_sd"] = res["summary"]["spearman"]["sd"]
        if verbose:
            print(f"  champion on the {metric}-selected {len(top)}: "
                  f"rho {res['summary']['spearman']['mean']:+.4f}", flush=True)

    out = pd.DataFrame(rows)
    out["overlap_cols"] = overlap
    out.to_csv(OUT, index=False)
    if verbose:
        print(f"written -> {OUT}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-features", type=int, default=None)
    a = ap.parse_args(argv)
    run(n_features=a.n_features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------
# The untested axis: the selection *rule*
# --------------------------------------------------------------------------
# The size of the budget is settled -- 300 was inherited, then checked at 600,
# 1,200 and "every column", and it moves nothing (Part 4's register, Finding 1).
# The metric is nearly settled too: gain against SHAP is +0.0007 above. But both
# of those vary the *number* or the *ordering*, and every arm still takes the top
# k of one ranking. On a matrix where 6,095 columns restate about twenty
# variables, the top of any ranking is full of near-duplicates, so the budget can
# be spent several times over on the same quantity.
#
# This is the rule that has never been tried: pick greedily by relevance, but
# penalise each candidate by how correlated it already is with what has been
# chosen -- the classic max-relevance / min-redundancy trade. It is run inside
# each training fold, like the selection it replaces, so it is leakage-safe. The
# correlation is computed incrementally against the selected set rather than as a
# full 6,517-square matrix, which keeps it to a few seconds a fold.
REDUNDANCY_OUT = config.RESULTS / "selector_redundancy.csv"


def _mrmr(Xtr: np.ndarray, rel: np.ndarray, k: int, lam: float = 1.0):
    """Greedy max-relevance / min-redundancy pick of k columns.

    score_j = rel_j / (1 + lam * max_|corr| with the already-chosen set), so a
    column that duplicates something already taken has to be far more relevant
    to earn a place. Returns the chosen indices in pick order.
    """
    Z = Xtr - Xtr.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, ddof=0)
    live = sd > 0
    Z = np.where(live, Z / np.where(sd > 0, sd, 1.0), 0.0).astype(np.float32)
    n = Z.shape[0]

    chosen = [int(np.argmax(np.where(live, rel, -np.inf)))]
    maxabs = np.zeros(Xtr.shape[1], dtype=np.float32)
    for _ in range(k - 1):
        c = np.abs(Z.T @ Z[:, chosen[-1]]) / n          # corr with the last pick
        np.maximum(maxabs, c, out=maxabs)
        score = rel / (1.0 + lam * maxabs)
        score[chosen] = -np.inf
        score[~live] = -np.inf
        nxt = int(np.argmax(score))
        if not np.isfinite(score[nxt]):
            break
        chosen.append(nxt)
    return np.array(chosen)


def redundancy(seeds=(41, 42, 43), n_splits=None, k=None, lam: float = 1.0,
               verbose: bool = True):
    """Does a redundancy-penalised rule beat taking the top k of the ranking?"""
    import xgboost as xgb
    from scipy.stats import spearmanr
    from sklearn.model_selection import KFold

    from .run_ablation import _impute, champion_params, make_selector

    k = k or config.N_FEATURES
    n_splits = n_splits or config.N_SPLITS
    X, y, names, n_base = _frame()
    binary = _is_binary(X)
    rows = []

    for rule in ("gain_top_k", "mrmr"):
        rhos, n_bin, n_flank, red = [], [], [], []
        for seed in seeds:
            for tr, va in KFold(n_splits, shuffle=True,
                                random_state=seed).split(X):
                A, B = _impute(X[tr], X[va])
                sel = make_selector(seed)
                sel.fit(A, y[tr])
                rel = np.asarray(sel.feature_importances_, dtype=np.float64)
                top = (np.argsort(rel)[::-1][:k] if rule == "gain_top_k"
                       else _mrmr(A, rel, k, lam))
                m = xgb.XGBRegressor(**champion_params(seed))
                m.fit(A[:, top], y[tr])
                rhos.append(float(spearmanr(y[va], m.predict(B[:, top])).statistic))
                n_bin.append(int(binary[top].sum()))
                n_flank.append(int((top >= n_base).sum()))
                # How redundant is the chosen set? Mean of each column's largest
                # absolute correlation with any other chosen column -- the number
                # the rule is supposed to move.
                Z = A[:, top] - A[:, top].mean(axis=0, keepdims=True)
                sd = Z.std(axis=0, ddof=0)
                Z = np.where(sd > 0, Z / np.where(sd > 0, sd, 1.0), 0.0)
                C = np.abs(Z.T @ Z) / len(Z)
                np.fill_diagonal(C, 0.0)
                red.append(float(C.max(axis=1).mean()))
        rows.append(dict(rule=rule, k=k, lam=lam if rule == "mrmr" else np.nan,
                         spearman=float(np.mean(rhos)),
                         spearman_sd=float(np.std(rhos, ddof=1)),
                         binary_cols=float(np.mean(n_bin)),
                         a_flank_cols=float(np.mean(n_flank)),
                         mean_max_abs_corr=float(np.mean(red))))
        if verbose:
            r = rows[-1]
            print(f"  {rule:12s}: rho {r['spearman']:+.4f}  "
                  f"redundancy {r['mean_max_abs_corr']:.3f}  "
                  f"{r['binary_cols']:.0f} binary  "
                  f"{r['a_flank_cols']:.0f} flank", flush=True)
        pd.DataFrame(rows).to_csv(REDUNDANCY_OUT, index=False)

    out = pd.DataFrame(rows).set_index("rule")
    if verbose:
        d = out.loc["mrmr", "spearman"] - out.loc["gain_top_k", "spearman"]
        print(f"\n  mrmr against the pipeline's rule: {d:+.4f}")
        print(f"  redundancy {out.loc['gain_top_k', 'mean_max_abs_corr']:.3f} "
              f"-> {out.loc['mrmr', 'mean_max_abs_corr']:.3f}")
        print(f"written -> {REDUNDANCY_OUT}")
    return out
