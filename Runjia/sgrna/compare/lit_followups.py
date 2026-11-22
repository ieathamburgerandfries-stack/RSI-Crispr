"""Two ideas taken from the published predictors and tested on our cohort (Part 6 of the report).

The literature pass found two places where other people's design choices had never been tried here.
Neither is a tweak to our model; each is a different answer to a question we had settled one way.

  merf   Yu et al. (2024, Genome Biology) predict bacterial CRISPRi guide efficiency with a
         mixed-effect random forest: guide features as fixed effects, the targeted gene as a random
         effect. We only ever *grouped folds* by gene, which removes gene leakage but not gene-level
         noise from the training signal -- the fixed model still spends capacity fitting
         between-gene level differences that cannot generalise to a gene it has never seen. A random
         intercept absorbs those, so the prediction should be a cleaner estimate of *within-gene*
         guide efficiency. That is the hypothesis, and the within-gene rho on held-out genes is the
         place to look; pooled rho may not move at all.

  binary Orro et al. (2026, Scientific Reports) binarise their fitness label (z < -2) and fit a
         classifier, reporting AUC. Section 15 tried weighting high-activity guides, pairwise
         ranking and NDCG and none helped the top end, but it never tried the plain thing: train a
         classifier on "is this guide in the top X%". Top-decile precision and the pick percentile
         of the ten best-scoring guides are the two metrics where we only tie crisprHAL 2, and they
         are the two a guide designer actually uses, so this is aimed at the known weakness.
         A blend of the regression score and the classifier probability is included, because the
         classifier can be better at the top while being worse overall.

Both run on `tune_ecoli`'s DEV/CONFIRM gene split, with the tuned configuration, so the regression
baseline reproduces the 0.7215 already on record.

    python -m sgrna.compare.lit_followups --test merf
    python -m sgrna.compare.lit_followups --test binary
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from sgrna.harness.run_ablation import _impute, make_selector

from ..tuning import tune_ecoli as T
from ..common import OUT

SEEDS = (41, 42, 43)
MERF_ITERS = 6


# ------------------------------------------------------------------ shared pieces
def columns(co, cfg, seed):
    """The tuned configuration's column slice, from the cached DEV-wide ranking (as `tune_ecoli.confirm`)."""
    rk = T.DIR / f"rank_devall_s{seed}.npy"
    if rk.exists():
        top = np.load(rk)
    else:
        A, _ = _impute(co.X[co.dev], co.X[co.conf][:1])
        sel = make_selector(seed)
        sel.fit(A, co.y[co.dev])
        top = np.argsort(sel.feature_importances_)[::-1][:900]
        np.save(rk, top)
    cols = top[: int(cfg["nf"])]
    return _impute(co.X[co.dev][:, cols], co.X[co.conf][:, cols])


def parts(y, g, p, min_n=5):
    """Pooled, between-gene and within-gene Spearman, the last two over genes with >= min_n guides."""
    ug, inv = np.unique(g, return_inverse=True)
    rows = [np.where(inv == i)[0] for i in range(len(ug))]
    rows = [r for r in rows if len(r) >= min_n]
    ym = np.array([y[r].mean() for r in rows]); pm = np.array([p[r].mean() for r in rows])
    yd = np.concatenate([y[r] - y[r].mean() for r in rows])
    pdd = np.concatenate([p[r] - p[r].mean() for r in rows])
    return (float(spearmanr(y, p).statistic), float(spearmanr(ym, pm).statistic),
            float(spearmanr(yd, pdd).statistic), len(rows))


def _z(v):
    return (v - v.mean()) / (v.std() + 1e-12)


# ------------------------------------------------------------------ Yu et al.: gene as a random effect
def merf(co, cfg, seed):
    """Back-fitting mixed effects: alternate a boosted fixed part with shrunk per-gene intercepts.

    b_g = n_g / (n_g + lam) * mean(residual in gene g), with lam = sigma2_eps / sigma2_b re-estimated
    each round -- the standard random-intercept shrinkage, so a gene with few guides is pulled
    towards zero. Held-out genes get b = 0 by construction, which is the point: the fixed part has
    to carry everything that transfers.
    """
    A, B = columns(co, cfg, seed)
    y_dev, g_dev = co.y[co.dev], co.g[co.dev]
    ug, inv = np.unique(g_dev, return_inverse=True)
    n_g = np.bincount(inv).astype(float)
    b = np.zeros(len(ug))
    traj = []
    for it in range(MERF_ITERS):
        m = xgb.XGBRegressor(**T.xgb_params(cfg, seed)).fit(A, y_dev - b[inv])
        r = y_dev - m.predict(A)
        gm = np.bincount(inv, weights=r) / n_g
        s2b = max(gm.var(), 1e-9)
        s2e = max((r - gm[inv]).var(), 1e-9)
        lam = s2e / s2b
        b = n_g / (n_g + lam) * gm
        p = m.predict(B)
        pooled, betw, within, ngen = parts(co.y[co.conf], co.g[co.conf], p)
        traj.append(dict(iter=it + 1, lam=float(lam), sd_b=float(b.std()),
                         pooled=pooled, between=betw, within=within))
    return m.predict(B), traj, ngen


def run_merf(co):
    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    sel = T.selected()
    name = "d099x"
    cfg = dict(cfgs[name], n_estimators=int(sel.loc[name, "k"]))
    rows, traj_all = [], []
    for seed in SEEDS:
        # plain regression baseline, from the cache tune_ecoli already wrote where possible
        f = T.DIR / f"conf_{name}_k{cfg['n_estimators']}_s{seed}.npy"
        if f.exists():
            p_plain = np.load(f)
        else:
            A, B = columns(co, cfg, seed)
            p_plain = T.fit_predict(cfg, A, co.y[co.dev], B, seed)[cfg["n_estimators"]]
            np.save(f, p_plain)
        pooled, betw, within, ngen = parts(co.y[co.conf], co.g[co.conf], p_plain)
        rows.append(dict(arm="tuned regression (d099x)", seed=seed, pooled=pooled, between=betw,
                         within=within, n_genes=ngen))
        p_merf, traj, ngen = merf(co, cfg, seed)
        for t in traj:
            traj_all.append(dict(seed=seed, **t))
        pooled, betw, within, ngen = parts(co.y[co.conf], co.g[co.conf], p_merf)
        rows.append(dict(arm="mixed effects, gene intercept", seed=seed, pooled=pooled, between=betw,
                         within=within, n_genes=ngen))
        print(f"  seed {seed}: " + " | ".join(f"{r['arm'][:24]} pooled {r['pooled']:.4f} within {r['within']:.4f}"
                                             for r in rows[-2:]), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "lit_merf.csv", index=False)
    pd.DataFrame(traj_all).to_csv(OUT / "lit_merf_trajectory.csv", index=False)
    d = pd.DataFrame(rows).groupby("arm")[["pooled", "between", "within"]].mean()
    print("\n" + d.round(4).to_string())
    base = d.loc["tuned regression (d099x)"]
    for k in ("pooled", "between", "within"):
        print(f"  mixed effects - regression, {k:8s}: {d.loc['mixed effects, gene intercept', k] - base[k]:+.4f}")
    return d


# ------------------------------------------------------------------ Orro et al.: binarise and classify
def top_frac_metrics(y, p, frac=0.10, k=10):
    n = len(y)
    thr_y = np.quantile(y, 1 - frac)
    take = np.argsort(p)[::-1][: int(round(frac * n))]
    prec = float((y[take] >= thr_y).mean())
    best = np.argsort(p)[::-1][:k]
    pct = float(np.mean([(y < y[i]).mean() for i in best]) * 100)
    return prec, pct


def run_binary(co, fracs=(0.10, 0.20, 0.30)):
    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    sel = T.selected()
    cfg = dict(cfgs["d099x"], n_estimators=int(sel.loc["d099x", "k"]))
    y_dev, y_conf, g_conf = co.y[co.dev], co.y[co.conf], co.g[co.conf]
    rows = []
    for seed in SEEDS:
        A, B = columns(co, cfg, seed)
        f = T.DIR / f"conf_d099x_k{cfg['n_estimators']}_s{seed}.npy"
        p_reg = np.load(f) if f.exists() else T.fit_predict(cfg, A, y_dev, B, seed)[cfg["n_estimators"]]
        prec, pct = top_frac_metrics(y_conf, p_reg)
        rows.append(dict(arm="tuned regression", seed=seed, rho=float(spearmanr(y_conf, p_reg).statistic),
                         top_decile_precision=prec, pick10_percentile=pct))
        for fr in fracs:
            thr = np.quantile(y_dev, 1 - fr)
            params = dict(T.xgb_params(cfg, seed))
            params.pop("objective", None)
            clf = xgb.XGBClassifier(**params, objective="binary:logistic",
                                    eval_metric="logloss").fit(A, (y_dev >= thr).astype(int))
            p_clf = clf.predict_proba(B)[:, 1]
            prec, pct = top_frac_metrics(y_conf, p_clf)
            rows.append(dict(arm=f"classifier, top {int(fr*100)}%", seed=seed,
                             rho=float(spearmanr(y_conf, p_clf).statistic),
                             top_decile_precision=prec, pick10_percentile=pct))
            p_mix = _z(p_reg) + _z(p_clf)
            prec, pct = top_frac_metrics(y_conf, p_mix)
            rows.append(dict(arm=f"regression + classifier top {int(fr*100)}%", seed=seed,
                             rho=float(spearmanr(y_conf, p_mix).statistic),
                             top_decile_precision=prec, pick10_percentile=pct))
        print(f"  seed {seed} done", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "lit_binary.csv", index=False)
    d = pd.DataFrame(rows).groupby("arm")[["rho", "top_decile_precision", "pick10_percentile"]].mean()
    d = d.reindex(["tuned regression"] + [a for a in d.index if a != "tuned regression"])
    print("\n" + d.round(4).to_string())
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["merf", "binary", "both"], default="both")
    a = ap.parse_args()
    co = T.Cohort()
    pd.set_option("display.width", 200)
    if a.test in ("merf", "both"):
        print("\n=== Yu et al.: the gene as a random effect ===", flush=True)
        run_merf(co)
    if a.test in ("binary", "both"):
        print("\n=== Orro et al.: binarise the label and classify ===", flush=True)
        run_binary(co)


if __name__ == "__main__":
    main()
