"""Do objectives aimed at the top of the ranking improve top-end metrics, where crisprHAL 2 still leads on pick percentile?

The tuned model is trained with squared error, which spends most of its effort on the middle of the label distribution. A design
tool uses the top: the guides it would pick. Arms (all with the tuned configuration d099x, 4,000 trees, 450 columns):

  mse           squared error (the tuned model)
  weighted-k    squared error with weight 1 + k * (label percentile)^3 (k = 3, 9): high-activity guides count more
  pairwise      XGBoost `rank:pairwise` over random groups of 256 guides (pairs inside a group; grouping is by chance so that no
                gene-level signal is excluded)
  ndcg          `rank:ndcg` on 31 relevance grades from label quantiles, top-k pair construction
  mse+pairwise  average of z-scored predictions of the first and third

Metrics on the held-out genes: Spearman, top-decile precision, top-1% precision, and the mean percentile (within the test set) of the
10 / 50 / 100 highest-scoring guides -- the last three smooth the 10-guide pick percentile, which is very noisy.

    python -m sgrna.ext.topend
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata, spearmanr

from sgrna.run_ablation import _impute

from . import tune_ecoli as T
from .common import OUT


def topmetrics(y, p):
    n = len(y)
    order = np.argsort(p)[::-1]
    pct = rankdata(y) / n * 100
    out = dict(spearman=float(spearmanr(y, p).statistic))
    for frac, lab in ((0.10, "top10pct_prec"), (0.01, "top1pct_prec")):
        k = max(int(n * frac), 1)
        out[lab] = float(np.isin(order[:k], np.argsort(y)[::-1][:k]).mean())
    for k in (10, 50, 100):
        out[f"pick{k}_pctile"] = float(pct[order[:k]].mean())
    return out


def _qid(n, size=256, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.integers(0, max(n // size, 1), n)), rng


def fit_predict(arm, cfg, Xa, ya, Xb, seed):
    p = T.xgb_params(cfg, seed)
    p.update(importance_type="gain")
    if arm == "mse":
        return xgb.XGBRegressor(**p).fit(Xa, ya).predict(Xb)
    if arm.startswith("weighted"):
        k = float(arm.split("-")[1]); pc = rankdata(ya) / len(ya)
        return xgb.XGBRegressor(**p).fit(Xa, ya, sample_weight=1 + k * pc ** 3).predict(Xb)
    if arm in ("pairwise", "ndcg"):
        n = len(ya); rng = np.random.default_rng(seed); perm = rng.permutation(n)
        qid = np.sort(np.repeat(np.arange(int(np.ceil(n / 256))), 256)[:n])
        Xs, ys = Xa[perm], ya[perm]
        if arm == "pairwise":
            m = xgb.XGBRanker(objective="rank:pairwise", **p).fit(Xs, ys, qid=qid)
        else:
            grades = np.minimum((rankdata(ys) / n * 31).astype(int), 30)
            m = xgb.XGBRanker(objective="rank:ndcg", lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=8, **p).fit(Xs, grades, qid=qid)
        return m.predict(Xb)
    raise ValueError(arm)


def run(co, arms=("mse", "weighted-3", "weighted-9", "pairwise", "ndcg"), seeds=(41, 42, 43), name="d099x"):
    cfgs = json.loads((OUT / "tune_configs.json").read_text()); sel = T.selected()
    cfg = dict(cfgs[name], n_estimators=int(sel.loc[name, "k"]))
    rows, preds = [], {}
    for sd in seeds:
        top = np.load(T.DIR / f"rank_devall_s{sd}.npy"); cols = top[:int(cfg["nf"])]
        Xa, Xb = _impute(co.X[co.dev][:, cols], co.X[co.conf][:, cols])
        for arm in arms:
            p = fit_predict(arm, cfg, Xa, co.y[co.dev], Xb, sd)
            preds[(arm, sd)] = p
            rows.append(dict(arm=arm, seed=sd, **topmetrics(co.y[co.conf], p)))
            print(f"  seed {sd} {arm:12s} " + " ".join(f"{k} {v:.4f}" for k, v in rows[-1].items() if k not in ("arm", "seed")), flush=True)
        if "mse" in arms and "pairwise" in arms:
            z = lambda a: (a - a.mean()) / a.std()
            p = (z(preds[("mse", sd)]) + z(preds[("pairwise", sd)])) / 2
            rows.append(dict(arm="mse+pairwise", seed=sd, **topmetrics(co.y[co.conf], p)))
        pd.DataFrame(rows).to_csv(OUT / "topend_confirm.csv", index=False)
    d = pd.DataFrame(rows)
    return d.groupby("arm").mean(numeric_only=True).drop(columns="seed")


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    print(run(T.Cohort()).round(4).to_string())
