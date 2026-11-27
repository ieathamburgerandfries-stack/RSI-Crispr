"""Beyond rank correlation: the other things a guide-efficiency model is judged on, for SLICER and crisprHAL 2 (SLICER's lead is not a Spearman artefact, and the top of the ranking is where it is smallest).

A guide-design tool is used to pick guides, so the question is not only how well the whole ranking correlates with the label but how well the top is identified, how well duds are avoided, and whether the scores mean the
same thing on a new batch. Six parts, each its own panel (the panels were built one after another, so their metric sets differ on purpose; the table says which file holds which). Where crisprHAL 2 is a column it is the
model as shipped, scored on crisprHAL 2's own two test files (eSpCas9: 11,898 guides, SpCas9: 6,737) with SLICER trained on the matching training file (`vs_crisprhal.hal_split`); differences carry paired bootstrap
intervals over test guides.

  part     what it answers                                                                                                   file                 metrics
  folds    untuned and tuned LightGBM against crisprHAL 2 on identical cross-validation folds (nine metrics, the first       metric_panel.csv     rho, tau, r, R2, calibration slope, top-decile precision, NDCG@10, pick percentile
           panel; the report's "nine-metric panel")                                                                                              (pooled and within a gene)
  splits   the full panel on crisprHAL 2's two test files, SLICER default (champion) and tuned (d099x)                       metrics_panel.csv    rank / linear (rho, r, tau, R2), top of the list (precision and recall of the top 10%,
                                                                                                                                                  precision of the top 1%, NDCG, pick percentile of 10 / 50 / 100), classification (AUROC and AUPRC
                                                                                                                                                  for 'efficient' = top quartile, AUROC for 'dud' = bottom decile), calibration slope
  raw      the six-metric panel for the tuned and mixed models, with every member's RAW test prediction saved so blends are   panel_<split>.csv    rho, tau, r, R2, top-decile precision, pick percentile of the 10 best
           averages of raw predictions (a z-scored blend is meaningless for R2 and calibration)                              (+ data/interim/post_freeze/panel/<split>.npz)
  topk     smoother top-end metrics (the 10-guide pick percentile rests on ten guides and spans +-10 points)                  topk_panel.csv       precision of the top 1 / 5 / 10%, pick percentile of 10 / 50 / 100, calibration slope
  seeds    seed stability of the tuned model's lead over crisprHAL 2 (five seeds, selection and trees)                         seed_stability.csv   rho per seed
  blend    averaging the tuned trees with crisprHAL 2 at equal weight adds only +0.0085 on SpCas9 and nothing detectable on  blend_with_hal.csv    rho at weights 0 to 1 on crisprHAL 2, residual correlation
           eSpCas9, so the network adds little beyond the trees (no weight is fitted on test data)

`topk`, `seeds` and `blend` read the raw predictions that `raw` saves, so run `raw` first. Spearman is the headline because ranking is the task, but it hides things: a model can rank well and be badly calibrated
(R2 is the one that catches a model shrinking toward the mean: the Huber case had R2 0.192 with rho 0.568), or rank well overall and badly where the decision is made, the top of the list.

    python -m sgrna.compare.panel --parts splits
    python -m sgrna.compare.panel --parts raw [--splits espcas9 wt] [--mix "d099x,d077x,d021x,d080x,d000x;lgb011,lgb018;cat015,cat023"] [--no-train]
    python -m sgrna.compare.panel --parts topk seeds blend

Report section: Part 5.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

from .. import config
from ..common import INTERIM, OUT
from . import vs_crisprhal as V

PRED_DIR = config.RESULTS
RAW_DIR = INTERIM / "panel"
RAW_DIR.mkdir(parents=True, exist_ok=True)
MIX = "d099x,d077x,d021x,d080x,d000x;lgb011,lgb018;cat015,cat023"
XGB5 = ["d099x", "d077x", "d021x", "d080x", "d000x"]
SPLITS = {"SpCas9": "wt", "eSpCas9": "espcas9"}


def _align(split):
    """crisprHAL 2's split, with the positions (into the test rows) of the guides it also scored, once each: (X_train, y_train, X_test, y_test, test protospacers, crisprHAL's predictions, positions)."""
    Xtr, ytr, Xte, yte, names_te, hal = V.hal_split(split)
    keep = np.where(names_te.isin(hal.index) & ~names_te.duplicated())[0]
    return Xtr, ytr, Xte, yte, names_te, hal, keep


# =====================================================================================================  folds
def ndcg_at_k(y_true, y_pred, k: int = 10, n_lists: int = 2000, seed: int = 41) -> float:
    """Mean NDCG@k over random candidate lists of size 10*k. A guide's gain is its true score shifted to be non-negative, so NDCG asks: of the value available in this list, how much does following the model's
    order capture in the first k picks?"""
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


def pick_percentile(y_true, y_pred, k: int = 10, n_draws: int = 20000, seed: int = 41, groups=None) -> float:
    """Show the model k candidates; report where its pick falls in the truth. With `groups`, candidates are drawn from a single group (a gene), which is the decision a practitioner actually faces."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ranks = pd.Series(y).rank(pct=True).to_numpy() * 100
    if groups is None:
        pools = [np.arange(len(y))]
    else:
        g = pd.Series(groups)
        pools = [np.asarray(v) for v in g.groupby(g).indices.values()]
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


def metrics_folds(y_true, y_pred, groups=None) -> dict:
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
        pick_pct_within=(pick_percentile(y, p, groups=groups) if groups is not None else np.nan),
    )


def _slicer_oof(tuned: bool, seed: int = 41, n_splits: int = 5, verbose: bool = True):
    """Out-of-fold predictions for SLICER on the curated head-to-head arm."""
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    from ..harness.run_ablation import _impute, make_selector
    from ..tuning.tune import DEFAULT, RESULTS_PATH as TUNE_PATH
    from .headtohead import _assemble

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
    meta = json.loads((config.INTERIM / "expanded_meta.json").read_text())
    protos = np.asarray(meta["protospacers"])
    idx = pd.read_csv(config.INTERIM / "guide_index.csv")
    gene = idx.drop_duplicates("protospacer").set_index("protospacer")["gene_name"].reindex(protos).to_numpy()
    return gene if len(gene) == n else None


def folds(verbose: bool = True) -> pd.DataFrame:
    path = OUT / "metric_panel.csv"
    rows = []
    groups = None
    for label, tuned in (("SLICER, untuned", False), ("SLICER, tuned", True)):
        if verbose:
            print(f"  {label} ...", flush=True)
        y, pred = _slicer_oof(tuned, verbose=verbose)
        groups = groups if groups is not None else _gene_groups(len(y))
        rows.append(dict(model=label, **metrics_folds(y, pred, groups)))
        pd.DataFrame(rows).to_csv(path, index=False)

    for label, pat in (("crisprHAL 2, untuned", "preds_crisprhal_default_fold*.csv"), ("crisprHAL 2, tuned", "preds_crisprhal_tuned_fold*.csv")):
        files = sorted(PRED_DIR.glob(pat))
        if not files:
            if verbose:
                print(f"  {label}: no predictions on disk yet, skipped")
            continue
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        groups = groups if groups is not None else _gene_groups(len(d))
        g = groups if (groups is not None and len(groups) == len(d)) else None
        rows.append(dict(model=label, **metrics_folds(d["y"], d["pred"], g)))
        pd.DataFrame(rows).to_csv(path, index=False)
    out = pd.DataFrame(rows)
    cols = ["model", "n", "spearman", "kendall_tau", "pearson", "r2", "calibration_slope", "top_decile_prec", "ndcg_at_10", "pick_percentile", "pick_pct_within"]
    print(out[[c for c in cols if c in out]].round(4).to_string(index=False))
    return out


# =====================================================================================================  splits
def _pct(y):
    return 100.0 * (rankdata(y) - 0.5) / len(y)


def metrics_splits(y, p):
    n = len(y)
    pc = _pct(y)
    top = int(round(0.10 * n))
    order = np.argsort(-p)
    true_top = set(np.argsort(-y)[:top])
    pred_top = set(order[:top])
    hit = len(true_top & pred_top)
    k1 = max(int(round(0.01 * n)), 1)
    top1 = len(set(np.argsort(-y)[:k1]) & set(order[:k1])) / k1
    disc = 1.0 / np.log2(np.arange(2, top + 2))
    ideal = np.sort(pc)[::-1][:top]
    ndcg = float((pc[order[:top]] * disc).sum() / (ideal * disc).sum())
    out = {
        "spearman": float(spearmanr(y, p).statistic), "pearson": float(pearsonr(y, p).statistic), "kendall": float(kendalltau(y, p).statistic),
        "r2": float(r2_score(y, p)), "precision_top10": hit / top, "recall_top10": hit / top, "precision_top1": top1, "ndcg_top10": ndcg,
        "pick10_pctile": float(pc[order[:10]].mean()), "pick50_pctile": float(pc[order[:50]].mean()), "pick100_pctile": float(pc[order[:100]].mean()),
        "auroc_efficient": float(roc_auc_score(y >= np.quantile(y, 0.75), p)), "auprc_efficient": float(average_precision_score(y >= np.quantile(y, 0.75), p)),
        "auroc_dud": float(roc_auc_score(y <= np.quantile(y, 0.10), -p)),
        "calibration_slope": float(np.polyfit(p, y, 1)[0]),
    }
    return out


def _splits_data(cond):
    """(y, {SLICER arm: prediction}, crisprHAL 2 prediction, n_train) on the guides both models scored, in the order `splits` has always used (espcas9: crisprHAL's own order)."""
    Xtr, ytr, Xte, yte, order, hal = V.hal_split(SPLITS[cond])
    preds, _ = V.train_predict(V.member_specs(["champion", "d099x"]), Xtr, ytr, Xte)
    if cond == "eSpCas9":
        both = hal.loc[hal.index.intersection(order)]
        pos = order.get_indexer(both.index)
        return yte[pos], {k: v[pos] for k, v in preds.items()}, both["pred"].to_numpy(), len(ytr)
    keep = order.isin(hal.index) & ~order.duplicated()
    pos = np.where(keep)[0]
    return yte[pos], {k: v[pos] for k, v in preds.items()}, hal.loc[order[pos], "pred"].to_numpy(), len(ytr)


def splits() -> pd.DataFrame:
    rows = []
    for cond in ("SpCas9", "eSpCas9"):
        y, sl, ph, ntr = _splits_data(cond)
        models = {"SLICER default": sl["champion"], "SLICER tuned": sl["d099x"], "crisprHAL 2": ph}
        point = {m: metrics_splits(y, p) for m, p in models.items()}
        rng = np.random.default_rng(0)
        boot = {m: [] for m in models}
        for _ in range(1000):
            k = rng.integers(0, len(y), len(y))
            for m, p in models.items():
                boot[m].append(metrics_splits(y[k], p[k]))
        for m in models:
            b = pd.DataFrame(boot[m])
            for name, v in point[m].items():
                lo, hi = np.percentile(b[name], [2.5, 97.5])
                row = dict(split=cond, model=m, metric=name, value=v, lo=float(lo), hi=float(hi), n_test=len(y), n_train=ntr)
                if m != "crisprHAL 2":
                    d = np.array(pd.DataFrame(boot[m])[name]) - np.array(pd.DataFrame(boot["crisprHAL 2"])[name])
                    row.update(diff_vs_crisprhal=float(v - point["crisprHAL 2"][name]), diff_lo=float(np.percentile(d, 2.5)), diff_hi=float(np.percentile(d, 97.5)))
                rows.append(row)
        print(f"  {cond}: n_test {len(y):,}, n_train {ntr:,}", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "metrics_panel.csv", index=False)
    w = d[d.metric.isin(["spearman", "pearson", "precision_top10", "ndcg_top10", "pick100_pctile", "auroc_efficient", "auroc_dud", "calibration_slope"])]
    print(w.pivot_table(index=["split", "metric"], columns="model", values="value").round(4).to_string())
    return d


# =====================================================================================================  raw
RAW_METRICS = ("spearman", "kendall_tau", "pearson", "r2", "top_decile_prec", "pick_percentile")


def metrics_raw(y, p):
    top = int(len(y) * 0.1)
    order = np.argsort(p)[::-1]
    return dict(spearman=float(spearmanr(y, p).statistic), kendall_tau=float(kendalltau(y, p).statistic),
                pearson=float(np.corrcoef(y, p)[0, 1]), r2=float(r2_score(y, p)),
                top_decile_prec=float(np.isin(order[:top], np.argsort(y)[::-1][:top]).mean()),
                pick_percentile=float(np.mean([(y < y[i]).mean() for i in order[:10]]) * 100))


def collect(split, members):
    """Train the members on the split's training rows; save raw test predictions."""
    f = RAW_DIR / f"{split}.npz"
    Xtr, ytr, Xte, yte, names_te, hal, keep = _align(split)
    preds, secs = V.train_predict(V.member_specs(members), Xtr, ytr, Xte)
    np.savez(f, y=yte[keep], hal=hal.loc[names_te[keep], "pred"].to_numpy(), y_hal=hal.loc[names_te[keep], "y"].to_numpy(), **{k: v[keep] for k, v in preds.items()})
    return f


def raw(split, mix_groups, n_boot=2000, seed=0):
    z = np.load(RAW_DIR / f"{split}.npz")
    y, hal = z["y"], z["hal"]
    assert np.allclose(y, z["y_hal"], atol=1e-9), "label alignment failed"
    avg = lambda ns: np.mean([z[n] for n in ns], axis=0)
    arms = {"crisprHAL 2": hal, "original champion": z["champion"], "tuned XGBoost": z["d099x"], "average of 5 XGBoost": avg(XGB5),
            "mixed ensemble": np.mean([avg(g) for g in mix_groups], axis=0)}
    rows = []
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    mh = metrics_raw(y, hal)
    bh = [metrics_raw(y[b], hal[b]) for b in boots]
    for lab, p in arms.items():
        m = metrics_raw(y, p)
        if lab == "crisprHAL 2":
            rows.append(dict(split=split, arm=lab, n=len(y), **m))
            continue
        bp = [metrics_raw(y[b], p[b]) for b in boots]
        row = dict(split=split, arm=lab, n=len(y), **m)
        for k in RAW_METRICS:
            d = np.array([a[k] - h[k] for a, h in zip(bp, bh)])
            row[f"d_{k}"] = m[k] - mh[k]
            row[f"lo_{k}"] = float(np.percentile(d, 2.5))
            row[f"hi_{k}"] = float(np.percentile(d, 97.5))
        rows.append(row)
        print(f"  {split} {lab:22s} " + " ".join(f"{k[:7]} {m[k]:.4f} ({m[k] - mh[k]:+.4f})" for k in RAW_METRICS), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"panel_{split}.csv", index=False)
    return out


# =====================================================================================================  topk, seeds, blend
def topk_metrics(y, p):
    n = len(y)
    order = np.argsort(p)[::-1]
    pct = rankdata(y) / n * 100
    trueorder = np.argsort(y)[::-1]
    out = {}
    for frac in (0.01, 0.05, 0.10):
        k = int(n * frac)
        out[f"top{int(frac * 100)}pct_precision"] = float(np.isin(order[:k], trueorder[:k]).mean())
    for k in (10, 50, 100):
        out[f"pick{k}_percentile"] = float(pct[order[:k]].mean())
    out["calibration_slope"] = float(np.polyfit(p, y, 1)[0])
    return out


def topk(n_boot=2000, seed=0):
    """Top-end metrics from the raw predictions `raw` saved: precision of the top 1 / 5 / 10%, mean percentile of the 10 / 50 / 100 highest-scoring guides, the calibration slope, and paired bootstrap differences to crisprHAL 2."""
    rows = []
    rng = np.random.default_rng(seed)
    for split in ("espcas9", "wt"):
        z = np.load(RAW_DIR / f"{split}.npz")
        y, hal = z["y"], z["hal"]
        avg = lambda ns: np.mean([z[n] for n in ns], axis=0)
        arms = {"original champion": z["champion"], "tuned XGBoost": z["d099x"], "mixed ensemble": np.mean([avg(XGB5), avg(["lgb011", "lgb018"]), avg(["cat015", "cat023"])], axis=0)}
        boots = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
        mh = topk_metrics(y, hal)
        bh = [topk_metrics(y[b], hal[b]) for b in boots]
        rows.append(dict(split=split, arm="crisprHAL 2", **mh))
        for lab, p in arms.items():
            m = topk_metrics(y, p)
            bp = [topk_metrics(y[b], p[b]) for b in boots]
            row = dict(split=split, arm=lab, **m)
            for k in mh:
                d = np.array([a[k] - h[k] for a, h in zip(bp, bh)])
                row[f"d_{k}"] = m[k] - mh[k]
                row[f"lo_{k}"] = float(np.percentile(d, 2.5))
                row[f"hi_{k}"] = float(np.percentile(d, 97.5))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "topk_panel.csv", index=False)
    for split in ("espcas9", "wt"):
        d = out[out.split == split]
        print(split)
        print(d[["arm"] + [c for c in d.columns if c.startswith("d_")]].round(4).to_string(index=False))
    return out


def seeds(seeds_=(41, 42, 43, 44, 45)):
    """Tuned XGBoost refitted with five seeds on both of crisprHAL's splits."""
    rows = []
    for split in ("espcas9", "wt"):
        z = np.load(RAW_DIR / f"{split}.npz")
        y, hal = z["y"], z["hal"]
        specs = V.member_specs(["d099x"])
        Xtr, ytr, Xte, yte, names_te, hal_df, keep = _align(split)
        for sd in seeds_:
            pr, _ = V.train_predict(specs, Xtr, ytr, Xte, seed=sd)
            p = pr["d099x"][keep]
            rows.append(dict(split=split, seed=sd, rho=float(spearmanr(y, p).statistic), crisprhal=float(spearmanr(y, hal).statistic)))
            print(f"  {split} seed {sd}: rho {rows[-1]['rho']:.4f}  lead {rows[-1]['rho'] - rows[-1]['crisprhal']:+.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "seed_stability.csv", index=False)
    return out


def blend(n_boot=2000, seed=0):
    """If the two models err differently, averaging them beats both; if the trees already capture what the network does, it does not. Z-scored average of the tuned trees and crisprHAL 2 at weights 0 to 1 on
    crisprHAL 2, plus the correlation of the two models' residuals; paired bootstrap over guides of the equal-weight blend against the better single model."""
    z_ = lambda p: (p - p.mean()) / p.std()
    rng = np.random.default_rng(seed)
    rows = []
    for split in ("espcas9", "wt"):
        z = np.load(RAW_DIR / f"{split}.npz")
        y, hal, tuned = z["y"], z["hal"], z["d099x"]
        boots = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
        rho = lambda p, b=slice(None): float(spearmanr(y[b], p[b]).statistic)
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            bl = w * z_(hal) + (1 - w) * z_(tuned)
            r = rho(bl)
            if w == 0.5:
                d = np.array([rho(bl, b) - max(rho(tuned, b), rho(hal, b)) for b in boots])
                lo, hi = np.percentile(d, [2.5, 97.5])
            else:
                lo = hi = np.nan
            rows.append(dict(split=split, weight_on_crisprhal=w, rho=r, vs_best_single=r - max(rho(tuned), rho(hal)) if w == 0.5 else np.nan, lo=lo, hi=hi))
        res_t, res_h = z_(y) - z_(tuned), z_(y) - z_(hal)
        rows.append(dict(split=split, weight_on_crisprhal=np.nan, rho=float(np.corrcoef(res_t, res_h)[0, 1]), vs_best_single=np.nan, lo=np.nan, hi=np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "blend_with_hal.csv", index=False)
    print(out.round(4).to_string(index=False))
    return out


PARTS = ("folds", "splits", "raw", "topk", "seeds", "blend")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", nargs="+", choices=PARTS, required=True)
    ap.add_argument("--splits", nargs="+", choices=["espcas9", "wt"], default=["espcas9", "wt"], help="raw: which of crisprHAL 2's splits")
    ap.add_argument("--mix", default=MIX, help="raw: groups of member names, g1a,g1b;g2a,g2b (each group averaged, then the groups)")
    ap.add_argument("--no-train", action="store_true", help="raw: reuse the saved raw predictions")
    a = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    for part in a.parts:
        if part == "raw":
            groups = [[m for m in g.split(",") if m] for g in a.mix.split(";")]
            members = sorted({"champion", *XGB5} | {m for g in groups for m in g})
            for split in a.splits:
                if not a.no_train:
                    collect(split, members)
                raw(split, groups)
        else:
            {"folds": folds, "splits": splits, "topk": topk, "seeds": seeds, "blend": blend}[part]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
