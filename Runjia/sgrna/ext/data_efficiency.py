"""The data-efficiency curve again, with the tuned model against the stored crisprHAL 2 curve.

Same construction as `sample_complexity.py`: fold k of the shared five-fold split (KFold, seed 41), nested training subsets of
4,000 to 26,853 guides drawn by a seeded permutation of the fold's training rows, selection refitted at every size. crisprHAL 2's
stored curve (`results/sample_complexity.csv`, seed 41) is the comparison. SLICER arms: the champion, the tuned XGBoost d099x, and a
size-scaled variant that keeps the tuned settings but sets the tree count by a 10% inner validation split (early stopping), since 4,000
trees at a small n may overfit.

    python -m sgrna.ext.data_efficiency
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from sgrna import config
from sgrna.curated_ablation import _base, _block
from sgrna.run_ablation import _impute, make_selector

from . import tune_ecoli as T
from .common import OUT

SIZES = (4000, 8000, 13880, 20000, 26853)


def run(seed=41):
    X, y, _, protos = _base(); A, _ = _block("a_flank", protos); X = np.hstack([X, A])
    cfgs = json.loads((OUT / "tune_configs.json").read_text()); sel = T.selected()
    rows = []
    splits = list(KFold(5, shuffle=True, random_state=41).split(np.arange(len(y))))
    for fold, (tr_all, va) in enumerate(splits, start=1):
        order = np.random.default_rng(seed).permutation(tr_all)
        for n in SIZES:
            idx = order[:n]
            Xa, Xb = _impute(X[idx], X[va])
            s = make_selector(seed); s.fit(Xa, y[idx]); top = np.argsort(s.feature_importances_)[::-1][:900]
            for arm, name, early in (("champion", "champion", False), ("tuned d099x", "d099x", False), ("tuned d099x, early-stopped", "d099x", True)):
                cfg = dict(cfgs[name], n_estimators=int(sel.loc[name, "k"])); cols = top[:int(cfg["nf"])]
                if early:
                    rng = np.random.default_rng(seed); p = rng.permutation(n); nv = n // 10
                    va_i, tr_i = p[:nv], p[nv:]
                    m = xgb.XGBRegressor(**dict(T.xgb_params(cfg, seed), early_stopping_rounds=200)).fit(
                        Xa[tr_i][:, cols], y[idx][tr_i], eval_set=[(Xa[va_i][:, cols], y[idx][va_i])], verbose=False)
                    pred = m.predict(Xb[:, cols])
                    trees = int(m.best_iteration) + 1
                else:
                    pred = T.fit_predict(cfg, Xa[:, cols], y[idx], Xb[:, cols], seed)[cfg["n_estimators"]]
                    trees = cfg["n_estimators"]
                rows.append(dict(arm=arm, fold=fold, n_train=n, trees=trees, rho=float(spearmanr(y[va], pred).statistic)))
            print(f"  fold {fold} n {n}: " + " ".join(f"{r['arm'][:14]} {r['rho']:.4f}" for r in rows[-3:]), flush=True)
            pd.DataFrame(rows).to_csv(OUT / "data_efficiency.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    d = run()
    hal = pd.read_csv(config.ROOT.parent / "results" / "sample_complexity.csv"); hal = hal[(hal.model == "crisprHAL 2") & (hal.seed == 41)]
    t = d.pivot_table(index="n_train", columns="arm", values="rho")
    t["crisprHAL 2 (stored)"] = hal.groupby("n_train").spearman.mean()
    print(t.round(4))
