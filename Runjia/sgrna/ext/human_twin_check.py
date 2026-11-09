"""Does the project's human baseline survive when repeat measurements of a guide are kept in the same fold?

The human iRF matrix (Noshay et al. 2023, Supplementary Table 3; 17,421 guides) decodes to 13,138 distinct 20-mers: 4,216
sequences appear more than once (4,149 twice, 67 three times; 8,499 rows, 49% of the matrix have a twin), as separate rows
with the same genomic position. Between twins the label agrees at
Spearman 0.80 (59% exactly equal). With `KFold(shuffle=True)` on rows, a test guide's twin is usually in the training
folds, so a model that can memorise sequences (deep trees, fast learning rate, no subsampling) scores well without
generalising. This script re-runs `human_baseline.py`'s two arms with folds that keep identical protospacers together.

    python -m sgrna.ext.human_twin_check
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sgrna import config
from sgrna.human_baseline import MATRIX, TARGET, load
from sgrna.run_ablation import _impute, champion_params, grouped_folds, make_selector

from .common import OUT
from .human_flank import decode

ARMS = {
    "harness_default": dict(n_features=config.N_FEATURES, params=None),
    "jacky_400_1000": dict(n_features=400, params=dict(n_estimators=1000, max_depth=6, learning_rate=0.3, subsample=1.0,
                                                        colsample_bytree=1.0, reg_lambda=1.0, min_child_weight=1)),
}


def main(seeds=(41, 42, 43), n_splits=5):
    import xgboost as xgb
    X, y = load()
    dec = decode()
    yy = pd.to_numeric(dec.y, errors="coerce").to_numpy(float)
    ok = np.isfinite(yy)
    assert ok.sum() == len(y) and np.allclose(yy[ok], y), "decoded table is not aligned with the matrix rows"
    grp = pd.factorize(dec.protospacer[ok])[0]
    print(f"{len(y):,} guides, {grp.max() + 1:,} distinct protospacers")
    rows = []
    for arm, cfg in ARMS.items():
        rhos = []
        for seed in seeds:
            for tr, va in grouped_folds(grp, n_splits, seed):
                X_tr, X_va = _impute(X[tr], X[va])
                sel = make_selector(seed); sel.fit(X_tr, y[tr])
                top = np.argsort(sel.feature_importances_)[::-1][:cfg["n_features"]]
                p = dict(champion_params(seed)); p.update(cfg["params"] or {})
                m = xgb.XGBRegressor(**p).fit(X_tr[:, top], y[tr])
                rhos.append(float(spearmanr(y[va], m.predict(X_va[:, top])).statistic))
        rows.append(dict(arm=arm, folds="twin-grouped", spearman=float(np.mean(rhos)),
                         spearman_sd=float(np.std(rhos, ddof=1)), n_fits=len(rhos)))
        print(f"  {arm:>16}: rho {rows[-1]['spearman']:.4f} ± {rows[-1]['spearman_sd']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "human_twin_check.csv", index=False)


if __name__ == "__main__":
    main()
