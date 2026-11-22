"""The full metric panel, for untuned and tuned, on identical folds.

Spearman is the headline because ranking is the task, but it hides things: a
model can rank well and be badly calibrated, or rank well overall and badly
where the decision is made (the top of the list). This builds every metric this
project has argued for, on the same out-of-fold predictions, for whichever arms
have predictions on disk.

Metrics, and why each is here:

  spearman            rank agreement overall -- the headline
  kendall_tau         probability of ordering a random pair correctly; more
                      interpretable than rho and less sensitive to ties
  pearson, r2         scale, not just order. R2 is the one that catches a model
                      shrinking toward the mean (section 12's Huber case:
                      R2 0.192 with rho 0.568)
  calibration_slope   regression of truth on prediction; 1.0 is honest, below 1
                      means over-confident spread
  top_decile_prec     of the guides ranked in the top 10%, how many are truly
                      in the top 10%
  ndcg_at_10          a ranking metric that weights the head of the list, which
                      is where a practitioner actually looks
  pick_percentile     the decision itself: show the model 10 random candidates,
                      where does its pick land in the truth?
  pick_pct_within     the same, but candidates drawn from one gene -- the real
                      task is choosing among sites inside *your* target

    python -m sgrna.compare.panel --run
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from .. import config

OUT = config.RESULTS / "metric_panel.csv"
PRED_DIR = config.RESULTS


def ndcg_at_k(y_true, y_pred, k: int = 10, n_lists: int = 2000,
              seed: int = 41) -> float:
    """Mean NDCG@k over random candidate lists of size 10*k.

    A guide's gain is its true score shifted to be non-negative, so NDCG asks:
    of the value available in this list, how much does following the model's
    order capture in the first k picks?
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=float)
    gain = y - y.min()
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    out = []
    for _ in range(n_lists):
        idx = rng.choice(len(y), size=10 * k, replace=False)
        order = idx[np.argsort(np.asarray(y_pred)[idx])[::-1][:k]]
        ideal = idx[np.argsort(gain[idx])[::-1][:k]]
        d = (gain[ideal] * disc).sum()
        out.append((gain[order] * disc).sum() / d if d > 0 else np.nan)
    return float(np.nanmean(out))


def pick_percentile(y_true, y_pred, k: int = 10, n_draws: int = 20000,
                    seed: int = 41, groups=None) -> float:
    """Show the model k candidates; report where its pick falls in the truth.

    With `groups`, candidates are drawn from a single group (a gene), which is
    the decision a practitioner actually faces.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ranks = pd.Series(y).rank(pct=True).to_numpy() * 100
    if groups is None:
        pools = [np.arange(len(y))]
    else:
        g = pd.Series(groups)
        pools = [v.to_numpy() for _, v in g.groupby(g).groups.items()] \
            if False else [np.asarray(v) for v in g.groupby(g).indices.values()]
        pools = [v for v in pools if len(v) >= k]
        if not pools:
            return float("nan")
    out = []
    for _ in range(n_draws):
        pool = pools[rng.integers(len(pools))]
        idx = pool if len(pool) == k else rng.choice(pool, size=k, replace=False)
        out.append(ranks[idx[np.argmax(p[idx])]])
    return float(np.mean(out))


def top_decile_precision(y_true, y_pred, q: float = 0.1) -> float:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    k = max(1, int(round(q * len(y))))
    true_top = set(np.argsort(y)[::-1][:k].tolist())
    pred_top = np.argsort(p)[::-1][:k]
    return float(np.mean([i in true_top for i in pred_top]))


def calibration_slope(y_true, y_pred) -> float:
    p = np.asarray(y_pred, float)
    if p.std() == 0:
        return float("nan")
    return float(np.polyfit(p, np.asarray(y_true, float), 1)[0])


def metrics(y_true, y_pred, groups=None) -> dict:
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss = ((y - y.mean()) ** 2).sum()
    return dict(
        n=len(y),
        spearman=float(spearmanr(y, p).statistic),
        kendall_tau=float(kendalltau(y, p).statistic),
        pearson=float(pearsonr(y, p).statistic),
        r2=float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else np.nan,
        calibration_slope=calibration_slope(y, p),
        top_decile_prec=top_decile_precision(y, p),
        ndcg_at_10=ndcg_at_k(y, p),
        pick_percentile=pick_percentile(y, p),
        pick_pct_within=(pick_percentile(y, p, groups=groups)
                         if groups is not None else np.nan),
    )


def _slicer_oof(tuned: bool, seed: int = 41, n_splits: int = 5,
                verbose: bool = True):
    """Out-of-fold predictions for SLICER on the curated head-to-head arm."""
    import json
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    from .headtohead import _assemble
    from ..harness.run_ablation import _impute, make_selector
    from ..tuning.tune import DEFAULT, RESULTS_PATH as TUNE_PATH

    X, y, names = _assemble("curated_flank", verbose=verbose)
    best = None
    if tuned:
        t = pd.read_csv(TUNE_PATH)
        best = {f: json.loads(r) for f, r in zip(t["fold"], t["best_params"])}

    pred = np.empty_like(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        X_tr, X_va = _impute(X[tr], X[va])
        sel = make_selector(seed)
        sel.fit(X_tr, y[tr])
        top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
        params = best[fold] if tuned else dict(DEFAULT)
        m = lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **params)
        m.fit(X_tr[:, top], y[tr])
        pred[va] = m.predict(X_va[:, top])
        if verbose:
            print(f"    fold {fold} done", flush=True)
    return y, pred


def _gene_groups(n: int):
    """Gene labels for the curated rows, where we have them (8,617 of 33,567)."""
    import json
    meta = json.loads((config.INTERIM / "expanded_meta.json").read_text())
    protos = np.asarray(meta["protospacers"])
    idx = pd.read_csv(config.INTERIM / "guide_index.csv")
    gene = (idx.drop_duplicates("protospacer").set_index("protospacer")
            ["gene_name"].reindex(protos).to_numpy())
    return gene if len(gene) == n else None


def run(verbose: bool = True) -> pd.DataFrame:
    rows = []
    groups = None
    for label, tuned in (("SLICER, untuned", False), ("SLICER, tuned", True)):
        if verbose:
            print(f"  {label} ...", flush=True)
        y, pred = _slicer_oof(tuned, verbose=verbose)
        groups = groups if groups is not None else _gene_groups(len(y))
        rows.append(dict(model=label, **metrics(y, pred, groups)))
        pd.DataFrame(rows).to_csv(OUT, index=False)

    for label, pat in (("crisprHAL 2, untuned", "preds_crisprhal_default_fold*.csv"),
                       ("crisprHAL 2, tuned", "preds_crisprhal_tuned_fold*.csv")):
        files = sorted(PRED_DIR.glob(pat))
        if not files:
            if verbose:
                print(f"  {label}: no predictions on disk yet, skipped")
            continue
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        groups = groups if groups is not None else _gene_groups(len(d))
        g = groups if (groups is not None and len(groups) == len(d)) else None
        rows.append(dict(model=label, **metrics(d["y"], d["pred"], g)))
        pd.DataFrame(rows).to_csv(OUT, index=False)
    return pd.DataFrame(rows)


def summary():
    d = pd.read_csv(OUT)
    cols = ["model", "n", "spearman", "kendall_tau", "pearson", "r2",
            "calibration_slope", "top_decile_prec", "ndcg_at_10",
            "pick_percentile", "pick_pct_within"]
    print(d[[c for c in cols if c in d]].round(4).to_string(index=False))
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args(argv)
    if a.run:
        run()
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
