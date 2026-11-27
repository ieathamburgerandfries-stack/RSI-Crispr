"""Importance is a property of the data only for blocks of columns: the tree families agree on blocks but not on individual columns.

The puzzle this module exists to settle. On this matrix, XGBoost's two standard
importance methods -- split gain and permutation importance -- rank its own top
features at rho -0.02, i.e. independently. LightGBM's two methods agree at
+0.44. The two models score within 0.001 of each other, and they select
substantially different features.

Three explanations are possible and they have different consequences:

  (a) one model is right and the other is wrong. Then we should use the right
      one and read mechanism off it.
  (b) importance is ill-defined for individual columns here, because the matrix
      is redundant: where two columns carry the same information a tree picks
      one essentially arbitrarily, that one accumulates all the gain, and the
      other gets none. Then *neither* per-column ranking is evidence about
      biology, and LightGBM's internal agreement is a property of how it grows
      trees rather than a sign of being correct.
  (c) importance is ill-defined per column but well-defined at some coarser
      level -- the group of columns that encode one idea. Then mechanism should
      be read at that level and nowhere finer.

(b) and (c) are not exclusive, and together they are testable. If the signal is
spread across interchangeable columns, two models should disagree about *which
column* while agreeing about *which block of columns*. That is the measurement
here: the same importances, aggregated to twelve semantic blocks, compared
between models and between methods.

The blocks are chosen to be the granularity at which somebody would actually
make a claim -- "the seed region matters", "flank composition matters" -- not
at the granularity of one indicator variable.

    python -m sgrna.interpret.importance --run

Report section: Finding 5.
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config

AGREEMENT = config.RESULTS / "importance_agreement.csv"
BLOCKS = config.RESULTS / "importance_blocks.csv"

_V_MAP: dict | None = None


def _v_positions() -> dict:
    global _V_MAP
    if _V_MAP is None:
        raw = json.loads((config.INTERIM / "v_column_mapping.json").read_text())
        _V_MAP = {k: v["position"] for k, v in raw["mapping"].items()}
    return _V_MAP


def block_of(name: str) -> str:
    """The semantic group a column belongs to.

    Twelve blocks: the protospacer's one-hot indicators split into three
    regions, the quantum descriptors split by k-mer size, the published
    hand-named columns, and family A split by what kind of thing it measures.
    """
    # Family A, by what it measures rather than where
    if name.startswith("eng.flank."):
        if ".seq." in name:
            return "flank: bases within 10 nt"
        if ".qct." in name:
            return "flank: quantum within 10 nt"
        if ".comp." in name:
            return "flank: window composition"
        if ".win." in name:
            return "flank: window quantum"
        return "flank: other"
    if name.startswith("eng."):
        return "other engineered"

    # Quantum descriptors, by k-mer size
    m = re.match(r"^p(\d+)(monomer|dimer|trimer|tetramer|basepair)\.", name)
    if m:
        return f"protospacer quantum: {m.group(2)}"

    # One-hot indicators, by region of the guide
    if re.match(r"^V\d+", name):
        key = name
        pos = _v_positions().get(key)
        if pos is None:
            return "protospacer indicators: unmapped"
        if pos <= 5:
            return "protospacer indicators: pos 1-5 (PAM-distal)"
        if pos <= 15:
            return "protospacer indicators: pos 6-15 (middle)"
        return "protospacer indicators: pos 16-20 (seed)"

    return "published hand-named"


def shap_importance(model, X, n_rows: int = 1000, seed: int = 0,
                    linear_std: np.ndarray | None = None) -> np.ndarray | None:
    """Mean |SHAP value| per column: the importance this project now reports.

    Measured against gain (`importance_models.py`), SHAP agrees across seven
    model families at rho 0.50 per column and 0.80 per block where gain agrees
    at 0.05, and it removes a 26-percentage-point artefact from XGBoost's
    attribution -- gain over-credits a binary column used in many shallow
    nodes, because it counts fit improvement at split time rather than effect
    on the output. Everything in the docs that reports where importance goes is
    generated through here.

    TreeSHAP for tree ensembles, on a subsample of `n_rows` (exact SHAP is
    linear in rows, so the subsample is for cost, not approximation of the
    method). For a linear model the Shapley value of column j is
    coef_j * (x_j - mean_j), so mean |SHAP| is |coef_j| * mean|x_j - mean_j| --
    proportional to the standardised coefficient and computable without the
    `shap` package; pass the training column standard deviations as
    `linear_std`. Returns None for a model that is neither.
    """
    rng = np.random.default_rng(seed)
    if linear_std is not None and hasattr(model, "coef_"):
        coef = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
        return coef * np.asarray(linear_std, dtype=float)
    try:
        import shap
    except ImportError:
        return None
    idx = rng.choice(X.shape[0], size=min(n_rows, X.shape[0]), replace=False)
    try:
        ex = shap.TreeExplainer(model)
        phi = ex.shap_values(X[idx], check_additivity=False)
    except Exception:
        return None
    return np.abs(np.asarray(phi, dtype=float)).mean(axis=0)


def kuncheva(a: set, b: set, n_total: int) -> float:
    """Overlap of two equal-size selections, corrected for chance."""
    k = len(a)
    if k == 0 or k == n_total:
        return np.nan
    expected = k * k / n_total
    return (len(a & b) - expected) / (k - expected)


def _gain(model, n_cols: int) -> np.ndarray:
    imp = np.asarray(model.feature_importances_, dtype=float)
    if imp.shape[0] != n_cols:
        raise ValueError("importance length mismatch")
    s = imp.sum()
    return imp / s if s > 0 else imp


def _permutation(model, X, y, idx, n_repeats: int = 3, seed: int = 0):
    """Drop in rank correlation when one column is shuffled."""
    rng = np.random.default_rng(seed)
    base = spearmanr(y, model.predict(X)).statistic
    out = np.zeros(len(idx))
    for j, col in enumerate(idx):
        drops = []
        for _ in range(n_repeats):
            saved = X[:, col].copy()
            X[:, col] = rng.permutation(saved)
            drops.append(base - spearmanr(y, model.predict(X)).statistic)
            X[:, col] = saved
        out[j] = float(np.mean(drops))
    return out


def run(families=("a_flank",), seed: int = 41, n_splits: int = 3,
        n_features: int | None = None, verbose: bool = True):
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import KFold

    from ..build.build_matrix import load_dataset
    from ..harness.run_ablation import _impute, champion_params

    n_features = n_features or config.N_FEATURES
    X, y, names, _, _ = load_dataset(list(families), verbose=verbose)
    names = np.asarray(names)
    blocks = np.array([block_of(n) for n in names])
    if verbose:
        print(f"  {X.shape[1]:,} columns in {len(set(blocks))} blocks")

    agree_rows, block_rows = [], []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        X_tr, X_va = _impute(X[tr], X[va])
        y_tr, y_va = y[tr], y[va]

        models = {
            "xgboost": xgb.XGBRegressor(**champion_params(seed)),
            "lightgbm": lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=16,
                max_depth=4, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1),
        }

        gains, tops, perms = {}, {}, {}
        for name, m in models.items():
            m.fit(X_tr, y_tr)
            g = _gain(m, X.shape[1])
            top = np.argsort(g)[::-1][:n_features]
            gains[name] = g
            tops[name] = top
            perms[name] = _permutation(m, X_va.copy(), y_va, top, seed=seed)
            if verbose:
                print(f"    fold {fold} {name}: fitted", flush=True)

        # --- how much do the two models agree about individual columns? -----
        a, b = set(tops["xgboost"].tolist()), set(tops["lightgbm"].tolist())
        union = sorted(a | b)
        agree_rows.append(dict(
            fold=fold, level="column",
            selected_overlap=len(a & b),
            selected_kuncheva=kuncheva(a, b, X.shape[1]),
            gain_vs_gain_rho=float(spearmanr(gains["xgboost"][union],
                                             gains["lightgbm"][union]).statistic),
            xgb_gain_vs_perm_rho=float(spearmanr(
                gains["xgboost"][tops["xgboost"]], perms["xgboost"]).statistic),
            lgb_gain_vs_perm_rho=float(spearmanr(
                gains["lightgbm"][tops["lightgbm"]], perms["lightgbm"]).statistic),
        ))

        # --- and about blocks of columns? -----------------------------------
        labels = sorted(set(blocks))
        agg = {}
        for name in models:
            g = pd.Series(gains[name]).groupby(blocks).sum()
            p = pd.Series(perms[name], index=blocks[tops[name]]).groupby(level=0).sum()
            agg[name] = (g.reindex(labels).fillna(0.0),
                         p.reindex(labels).fillna(0.0))
            for lab in labels:
                block_rows.append(dict(
                    fold=fold, model=name, block=lab,
                    n_cols=int((blocks == lab).sum()),
                    gain_share=float(agg[name][0][lab]),
                    perm_share=float(agg[name][1][lab]),
                ))

        agree_rows.append(dict(
            fold=fold, level="block",
            selected_overlap=np.nan, selected_kuncheva=np.nan,
            gain_vs_gain_rho=float(spearmanr(agg["xgboost"][0],
                                             agg["lightgbm"][0]).statistic),
            xgb_gain_vs_perm_rho=float(spearmanr(agg["xgboost"][0],
                                                 agg["xgboost"][1]).statistic),
            lgb_gain_vs_perm_rho=float(spearmanr(agg["lightgbm"][0],
                                                 agg["lightgbm"][1]).statistic),
        ))

    ag = pd.DataFrame(agree_rows)
    bl = pd.DataFrame(block_rows)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    ag.to_csv(AGREEMENT, index=False)
    bl.to_csv(BLOCKS, index=False)
    return ag, bl


def summary():
    ag = pd.read_csv(AGREEMENT)
    bl = pd.read_csv(BLOCKS)
    print("\n== agreement, by granularity (mean over folds) ==")
    print(ag.groupby("level")[["selected_overlap", "selected_kuncheva",
                               "gain_vs_gain_rho", "xgb_gain_vs_perm_rho",
                               "lgb_gain_vs_perm_rho"]]
          .mean().round(3).to_string())
    print("\n== share of total gain per block ==")
    piv = (bl.groupby(["block", "model"])["gain_share"].mean().unstack()
           * 100).round(1)
    piv["n_cols"] = bl.groupby("block")["n_cols"].first()
    print(piv.sort_values("lightgbm", ascending=False).to_string())
    return ag, bl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--folds", type=int, default=3)
    args = ap.parse_args(argv)
    if args.run:
        run(n_splits=args.folds)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
