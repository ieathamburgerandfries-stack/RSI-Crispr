"""Fit the shipped SLICER models (tuned and default) on all 33,567 curated guides and save them with everything needed to score a new guide.

A model is the column selection (the top `nf` of the 6,517 base + flank columns by a throw-away boosted-tree ranking), the medians that fill missing values, and the
boosted trees (XGBoost). Settings are the paper's: the tuned model (`d099x`: 450 columns, 4,000 trees, depth 5) and the default one (`champion`: 300 columns, 400 trees,
depth 4), seed 41. They are written to `Runjia/models/` as `slicer_<name>.ubj` (the trees) and `slicer_<name>.json` (columns, medians, provenance).

    python -m sgrna.design.train_model                 # both
    python -m sgrna.design.train_model --model tuned
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .design_guide import MODELS, model_path

SEED = 41
NAMES = {"tuned": "d099x", "default": "champion"}
PAPER = {"tuned": "five-fold CV holding whole genes out, 15 folds: rho 0.731 (sd 0.007)", "default": "five-fold CV holding whole genes out, 15 folds: rho 0.707 (sd 0.009)"}


def train(which: str, threads: int = 8, verbose: bool = True) -> Path:
    import xgboost as xgb

    from ..cohorts.curated_ablation import _base, _block
    from ..common import OUT
    from ..harness.run_ablation import _impute, make_selector
    from ..tuning import tune_ecoli as T

    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    cfg = dict(cfgs[NAMES[which]], n_estimators=int(T.selected().loc[NAMES[which], "k"]))
    X, y, names, protos = _base()
    A, an = _block("a_flank", protos)
    XA = np.hstack([X, A])
    allnames = list(names) + list(an)
    t0 = time.time()
    Xi, _ = _impute(XA, XA[:1])
    sel = make_selector(SEED)
    sel.fit(Xi, y)
    top = np.argsort(sel.feature_importances_)[::-1][:int(cfg["nf"])]
    model = xgb.XGBRegressor(**dict(T.xgb_params(cfg, SEED), n_jobs=threads)).fit(Xi[:, top], y)
    med = np.nanmedian(XA[:, top], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    MODELS.mkdir(exist_ok=True)
    model.save_model(str(model_path(which, ".ubj")))
    meta = dict(model=which, settings=cfg, seed=SEED, n_train=int(len(y)), n_columns_offered=int(XA.shape[1]), columns=[allnames[i] for i in top], column_index=[int(i) for i in top],
                medians=[float(m) for m in med], label="cut.score: depletion of cells carrying the guide, higher = more cutting", label_quantiles={str(q): float(np.quantile(y, q / 100)) for q in (10, 25, 50, 75, 90)},
                trained_on="the 33,567 curated guides of the E. coli SpCas9 screen (Guo et al. 2018, curated by Browne et al. 2026), base columns + 348 flank columns",
                accuracy_in_the_paper=PAPER[which])
    model_path(which, ".json").write_text(json.dumps(meta))
    if verbose:
        print(f"  {which}: {len(top)} columns, {cfg['n_estimators']} trees, {time.time() - t0:.0f} s -> {model_path(which, '.ubj')}")
    return model_path(which, ".ubj")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=("tuned", "default", "both"), default="both")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    for w in (("tuned", "default") if a.model == "both" else (a.model,)):
        train(w, a.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
