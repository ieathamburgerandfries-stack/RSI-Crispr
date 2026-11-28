"""Importance is a property of the data only for blocks of columns, not for single columns: five tree families agree on columns at only 0.496 even with SHAP, but on blocks of columns at 0.800; two models with different features give the same score because the matrix is redundant, and dropping a model's top columns costs nothing for the same reason.

Four parts, in the order the argument was built (all Finding 5), each its own files:

  agreement      XGBoost's two importance methods rank its own top features at rho -0.02 and LightGBM's agree at +0.44, though the models score within 0.001: aggregated to twelve semantic blocks, do they agree?   importance_agreement.csv, importance_blocks.csv
  models         "you only tried two boosted trees, and split gain is crude": five tree families, penalised linear models, native importance, SHAP and permutation    importance_model_pairs.csv, _methods.csv, _blocks.csv
  attribution    two models, different features, same score: are they disagreeing about biology? (low overlap, but a close correlate for every column the other picked)     attribution.csv, attribution_summary.csv
  faithfulness   dropping a model's top columns costs nothing: is that because they are interchangeable? Remove the quantity (a column and its correlates), not the columns     faithfulness_groups.csv, faithfulness_clusters.csv (+ _reduced)

    python -m sgrna.interpret.importance --parts agreement models attribution faithfulness [--summary-only]
    python -m sgrna.interpret.importance --parts agreement --folds 3          # computes, then prints the summary

`shap_importance`, `block_of` and `kuncheva` are the helpers that `harness/evaluate.py`, `mechanism.py`, `tuned_settings.py`, `interp_extra.py` and `vs_crisprhal_interpret.py` import.

------------------------------------------------------------------------------------------------------------------------------
agreement
------------------------------------------------------------------------------------------------------------------------------
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

------------------------------------------------------------------------------------------------------------------------------
models
------------------------------------------------------------------------------------------------------------------------------
`importance.py` compared two models (XGBoost, LightGBM) and two methods (split
gain, permutation) and found the uncomfortable result that they agree about
*blocks* of columns (rho 0.50) while disagreeing about *individual* columns
(rho -0.14), with only 138 of 300 selected columns in common. The explanation
offered was redundancy: where many columns carry the same information, which
one a model credits is close to arbitrary.

Two objections to that, and this module exists to answer both.

**"You only tried two models, and they are both boosted trees."** Boosting
grows trees on residuals, so two boosters might fail the same way for reasons
that have nothing to do with the data. The test is to ask model families that
reach a prediction by genuinely different routes -- bagged trees, which fit
independently and average; and penalised linear models, which have no column
selection step at all and assign every column a coefficient. If the
column-level disagreement is a property of the matrix, it should survive the
change of model class. If it is a property of boosting, it should not.

**"Split gain is a crude heuristic. Use SHAP."** This is the right objection to
raise and it does not survive contact with the problem. SHAP's uniqueness
theorem is about dividing credit for *one prediction* among the inputs that
produced it; it says nothing about which input is mechanistically responsible.
Given two columns carrying identical information, Shapley values split the
credit between them -- by symmetry, roughly evenly, and exactly how evenly
depends on how the particular trees happened to use them. So the prediction
here is explicit and falsifiable: **SHAP should show the same column-level
instability as gain, and the same block-level stability.** If SHAP instead
produces a ranking all models agree on, the redundancy account is wrong and
this project has been reading its own interpretability results incorrectly.

Three importance methods are computed where each applies:

  native       split gain (boosters), PredictionValuesChange (CatBoost),
               impurity decrease (forests), |coefficient| on standardised
               columns (linear). What a practitioner reads off by default.
  shap         TreeSHAP mean |phi| over validation rows, for the five tree
               models. Model-agnostic in principle, tractable here only for
               trees. For a linear model SHAP is proportional to the
               standardised coefficient, so the native column already is it.
  permutation  drop in validation rho when one column is shuffled. The only
               method that is defined identically for every model, and the
               only one measured against held-out data.

------------------------------------------------------------------------------------------------------------------------------
attribution
------------------------------------------------------------------------------------------------------------------------------
The observation that prompts this: XGBoost's own two importance methods rank
its top features at rho -0.02 (i.e. independently), LightGBM's agree at +0.44,
the two models pick substantially different features, and they score the same.
That sounds like a contradiction. It is not -- but saying so is worth nothing
without a test, so this module runs one.

The hypothesis is **redundancy**: the matrix carries each underlying quantity in
many correlated columns, so there are many near-equivalent ways to reach the
same prediction. If that is right, then the two models' chosen columns should
be *different names for the same variables* -- different labels, same
information -- and three things should hold:

  1. Column overlap between the two top sets should be low.
  2. But every column one model picked and the other did not should have a
     close correlate inside the other model's set.
  3. And the *share of importance* each model gives to each kind of feature
     should agree, even though the individual columns do not.

The alternative hypothesis -- that the models have genuinely found different
biology -- predicts low overlap, *no* close correlates, and disagreeing shares.

A permuted-column control is included for scale: two selections that genuinely
share nothing still overlap a little by luck, and the matched-correlate
statistic needs a null to be read against.

------------------------------------------------------------------------------------------------------------------------------
faithfulness
------------------------------------------------------------------------------------------------------------------------------
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

Report section: Finding 5.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..common import OUT

AGREEMENT = OUT / "importance_agreement.csv"
BLOCKS = OUT / "importance_blocks.csv"

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


def agreement(families=("a_flank",), seed: int = 41, n_splits: int = 3,
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
    OUT.mkdir(parents=True, exist_ok=True)
    ag.to_csv(AGREEMENT, index=False)
    bl.to_csv(BLOCKS, index=False)
    return ag, bl


def agreement_summary():
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


# ======================================================================================================  models
PAIRS = OUT / "importance_model_pairs.csv"
METHODS = OUT / "importance_model_methods.csv"
SHARES = OUT / "importance_model_blocks.csv"


def _models(seed: int):
    """Four routes to a prediction: boosting, bagging, ordered boosting, linear."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge

    from ..harness.run_ablation import champion_params

    out = {
        "xgboost": xgb.XGBRegressor(**champion_params(seed)),
        "lightgbm": lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=16, max_depth=4,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            max_features="sqrt", random_state=seed, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            max_features="sqrt", random_state=seed, n_jobs=-1),
        "ridge": Ridge(alpha=10.0, random_state=seed),
        "elastic_net": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=seed,
                                  max_iter=5000),
    }
    try:
        from catboost import CatBoostRegressor
        out["catboost"] = CatBoostRegressor(
            iterations=400, learning_rate=0.05, depth=4, random_seed=seed,
            verbose=0, allow_writing_files=False)
    except ImportError:
        pass
    return out


IS_TREE = {"xgboost", "lightgbm", "catboost", "random_forest", "extra_trees"}
IS_LINEAR = {"ridge", "elastic_net"}


def _native(name, model, X_tr) -> np.ndarray:
    """Whatever this model reports when you ask it what mattered."""
    if name in IS_LINEAR:
        # |coef| on standardised columns -- the scale-free form, and what
        # SHAP reduces to for a linear model with independent inputs.
        sd = X_tr.std(axis=0)
        v = np.abs(np.asarray(model.coef_, dtype=float)) * sd
    else:
        v = np.abs(np.asarray(model.feature_importances_, dtype=float))
    s = v.sum()
    return v / s if s > 0 else v


def _shap(name, model, X_va, n_rows: int, seed: int) -> np.ndarray | None:
    """Mean |phi| per column, TreeSHAP, on a validation subsample."""
    if name not in IS_TREE:
        return None
    import shap
    rng = np.random.default_rng(seed)
    idx = rng.choice(X_va.shape[0], size=min(n_rows, X_va.shape[0]),
                     replace=False)
    ex = shap.TreeExplainer(model)
    phi = ex.shap_values(X_va[idx], check_additivity=False)
    v = np.abs(np.asarray(phi, dtype=float)).mean(axis=0)
    s = v.sum()
    return v / s if s > 0 else v


def _permutation_models(model, X, y, cols, n_repeats: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = spearmanr(y, model.predict(X)).statistic
    out = np.zeros(len(cols))
    for j, c in enumerate(cols):
        drops = []
        for _ in range(n_repeats):
            saved = X[:, c].copy()
            X[:, c] = rng.permutation(saved)
            drops.append(base - spearmanr(y, model.predict(X)).statistic)
            X[:, c] = saved
        out[j] = float(np.mean(drops))
    return out


def _rho(a, b) -> float:
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else np.nan


def models(families=("a_flank",), seed: int = 41, n_splits: int = 3,
        n_features: int | None = None, shap_rows: int = 1000,
        perm_repeats: int = 2, verbose: bool = True):
    from sklearn.model_selection import KFold

    from ..build.build_matrix import load_dataset
    from ..harness.run_ablation import _impute

    n_features = n_features or config.N_FEATURES
    X, y, names, _, _ = load_dataset(list(families), verbose=verbose)
    names = np.asarray(names)
    blocks = np.array([block_of(n) for n in names])
    labels = sorted(set(blocks))
    if verbose:
        print(f"  {X.shape[1]:,} columns in {len(labels)} blocks")

    pair_rows, method_rows, share_rows = [], [], []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        X_tr, X_va = _impute(X[tr], X[va])
        y_tr, y_va = y[tr], y[va]

        imp = {}          # (model, method) -> full-length importance vector
        tops = {}         # model -> indices of its own top n_features
        score = {}
        for name, model in _models(seed).items():
            t0 = time.time()
            model.fit(X_tr, y_tr)
            nat = _native(name, model, X_tr)
            top = np.argsort(nat)[::-1][:n_features]
            imp[(name, "native")] = nat
            tops[name] = top
            score[name] = _rho(y_va, model.predict(X_va))

            sv = _shap(name, model, X_va, shap_rows, seed)
            if sv is not None:
                imp[(name, "shap")] = sv

            perm_full = np.zeros(X.shape[1])
            perm_full[top] = _permutation_models(model, X_va.copy(), y_va, top,
                                          perm_repeats, seed)
            imp[(name, "permutation")] = perm_full
            if verbose:
                print(f"    fold {fold} {name:14s} rho {score[name]:.4f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

        models = sorted(tops)

        # --- within a model: do its own methods agree? ----------------------
        for name in models:
            avail = [m for m in ("native", "shap", "permutation")
                     if (name, m) in imp]
            for m1, m2 in itertools.combinations(avail, 2):
                t = tops[name]
                method_rows.append(dict(
                    fold=fold, model=name, method_a=m1, method_b=m2,
                    rho_column=_rho(imp[(name, m1)][t], imp[(name, m2)][t]),
                    rho_block=_rho(
                        pd.Series(imp[(name, m1)]).groupby(blocks).sum()
                        .reindex(labels).fillna(0.0),
                        pd.Series(imp[(name, m2)]).groupby(blocks).sum()
                        .reindex(labels).fillna(0.0)),
                ))

        # --- between models: same method, different model -------------------
        for method in ("native", "shap", "permutation"):
            present = [m for m in models if (m, method) in imp]
            for a, b in itertools.combinations(present, 2):
                ia, ib = imp[(a, method)], imp[(b, method)]
                sa, sb = set(tops[a].tolist()), set(tops[b].tolist())
                union = sorted(sa | sb)
                ga = pd.Series(ia).groupby(blocks).sum().reindex(labels).fillna(0.0)
                gb = pd.Series(ib).groupby(blocks).sum().reindex(labels).fillna(0.0)
                pair_rows.append(dict(
                    fold=fold, method=method, model_a=a, model_b=b,
                    rho_a=score[a], rho_b=score[b],
                    rho_column=_rho(ia[union], ib[union]),
                    rho_block=_rho(ga, gb),
                    selected_overlap=len(sa & sb),
                    selected_kuncheva=kuncheva(sa, sb, X.shape[1]),
                ))

        # --- where each model puts its weight, by block ---------------------
        for (name, method), v in imp.items():
            g = pd.Series(v).groupby(blocks).sum().reindex(labels).fillna(0.0)
            tot = g.sum()
            for lab in labels:
                share_rows.append(dict(
                    fold=fold, model=name, method=method, block=lab,
                    n_cols=int((blocks == lab).sum()),
                    share=float(g[lab] / tot) if tot else 0.0,
                ))

        pd.DataFrame(pair_rows).to_csv(PAIRS, index=False)
        pd.DataFrame(method_rows).to_csv(METHODS, index=False)
        pd.DataFrame(share_rows).to_csv(SHARES, index=False)

    return (pd.DataFrame(pair_rows), pd.DataFrame(method_rows),
            pd.DataFrame(share_rows))


def models_summary():
    pairs = pd.read_csv(PAIRS)
    meth = pd.read_csv(METHODS)
    sh = pd.read_csv(SHARES)

    print("\n== between models, same method (mean over folds and pairs) ==")
    print(pairs.groupby("method")[["rho_column", "rho_block",
                                   "selected_overlap", "selected_kuncheva"]]
          .mean().round(3).to_string())

    print("\n== between models, per pair (native importance) ==")
    nat = pairs[pairs["method"] == "native"]
    print(nat.groupby(["model_a", "model_b"])[["rho_column", "rho_block",
                                               "selected_overlap"]]
          .mean().round(3).to_string())

    print("\n== within one model, between its methods ==")
    print(meth.groupby(["model", "method_a", "method_b"])[["rho_column", "rho_block"]]
          .mean().round(3).to_string())

    print("\n== share of importance per block, native (%) ==")
    piv = (sh[sh["method"] == "native"]
           .groupby(["block", "model"])["share"].mean().unstack() * 100).round(1)
    piv["n_cols"] = sh.groupby("block")["n_cols"].first()
    print(piv.sort_values("xgboost", ascending=False).to_string())
    return pairs, meth, sh


# ======================================================================================================  attribution
ATTR_OUT = OUT / "attribution.csv"
ATTR_SUMMARY_OUT = OUT / "attribution_summary.csv"


def _paths(importance: str):
    """Keep the gain-based figures on disk beside the SHAP ones."""
    if importance == "native":
        return ATTR_OUT, ATTR_SUMMARY_OUT
    return (OUT / f"attribution_{importance}.csv",
            OUT / f"attribution_summary_{importance}.csv")


def _kind(name: str) -> str:
    """What quantity does this column describe? Coarse, hypothesis-level."""
    n = name
    if n.startswith("eng.flank.comp.") or n.startswith("eng.flank.win."):
        side = "up" if ".up" in n else ("dn" if ".dn" in n else "?")
        return f"flank window ({side}stream)"
    if n.startswith("eng.flank.seq.") or n.startswith("eng.flank.qct."):
        return "flank, nearest 10 nt"
    if n.startswith("V"):
        return "target: position/k-mer indicator"
    if "monomer" in n or "dimer" in n or "trimer" in n or "tetramer" in n:
        return "target: quantum descriptor"
    return "target: other published column"


def attribution(families=("a_flank",), seed: int = 41, n_features: int = 300,
        top: int = 50, verbose: bool = True,
        importance: str = "shap") -> pd.DataFrame:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import KFold

    from ..build.build_matrix import load_dataset
    from ..harness.run_ablation import _impute, champion_params

    if importance not in ("native", "shap"):
        raise ValueError(f"unknown importance {importance!r}")

    X, y, names, _, _ = load_dataset(list(families), verbose=verbose)
    names = np.asarray(names)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    tr, va = next(iter(kf.split(X)))
    X_tr, X_va = _impute(X[tr], X[va])
    y_tr, y_va = y[tr], y[va]

    # Each model selects with its own gain, as in the interpretability run --
    # the question is what each one chooses when left to itself.
    fits = {}
    for tag in ("xgboost", "lightgbm"):
        if tag == "xgboost":
            sel = xgb.XGBRegressor(**champion_params(seed))
        else:
            sel = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                    num_leaves=16, max_depth=4, subsample=0.8,
                                    subsample_freq=1, colsample_bytree=0.8,
                                    random_state=seed, n_jobs=-1, verbose=-1)
        sel.fit(X_tr, y_tr)
        if importance == "shap":
            imp = shap_importance(sel, X_va, seed=seed)
            if imp is None:
                raise RuntimeError("shap is not installed; pass "
                                   "importance='native' for the old figures")
        else:
            imp = np.asarray(sel.feature_importances_, dtype=float)
        order = np.argsort(imp)[::-1][:n_features]

        if tag == "xgboost":
            model = xgb.XGBRegressor(**champion_params(seed))
        else:
            model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                      num_leaves=16, max_depth=4, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.8,
                                      random_state=seed, n_jobs=-1, verbose=-1)
        model.fit(X_tr[:, order], y_tr)
        pred = np.asarray(model.predict(X_va[:, order])).ravel()
        fits[tag] = dict(order=order, imp=imp,
                         rho=float(spearmanr(y_va, pred).statistic))
        if verbose:
            print(f"  {tag:9s} held-out rho {fits[tag]['rho']:.4f}")

    a = fits["xgboost"]["order"][:top]
    b = fits["lightgbm"]["order"][:top]
    shared = np.intersect1d(a, b)

    # --- test 2: does each unshared pick have a close correlate in the other set?
    Xs = X_tr
    def _corr_matrix(idx_a, idx_b):
        A = Xs[:, idx_a]
        B = Xs[:, idx_b]
        A = (A - A.mean(0)) / np.where(A.std(0) == 0, 1, A.std(0))
        B = (B - B.mean(0)) / np.where(B.std(0) == 0, 1, B.std(0))
        return np.abs(A.T @ B) / len(Xs)

    only_a = np.setdiff1d(a, b)
    only_b = np.setdiff1d(b, a)
    M = _corr_matrix(only_a, b)
    best_a = M.max(axis=1) if M.size else np.array([])
    M2 = _corr_matrix(only_b, a)
    best_b = M2.max(axis=1) if M2.size else np.array([])

    # Null: random columns from the same matrix, matched for count.
    rng = np.random.default_rng(seed)
    rand = rng.choice(X.shape[1], size=len(only_a), replace=False)
    Mnull = _corr_matrix(rand, b)
    best_null = Mnull.max(axis=1) if Mnull.size else np.array([])

    rows = []
    for j in only_a:
        k = int(np.argmax(_corr_matrix(np.array([j]), b)[0]))
        rows.append(dict(picked_by="xgboost only", column=names[j],
                         kind=_kind(names[j]),
                         closest_in_other_set=names[b[k]],
                         abs_corr=float(_corr_matrix(np.array([j]), b)[0][k])))
    out, out_summary = _paths(importance)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    # --- test 3: importance share by kind of feature
    share = {}
    for tag in ("xgboost", "lightgbm"):
        imp = fits[tag]["imp"]
        idx = fits[tag]["order"]
        tot = imp[idx].sum()
        s = {}
        for j in idx:
            s[_kind(names[j])] = s.get(_kind(names[j]), 0.0) + imp[j] / tot
        share[tag] = s
    kinds = sorted(set(share["xgboost"]) | set(share["lightgbm"]))
    sh = pd.DataFrame({
        "kind": kinds,
        "xgboost_share": [round(share["xgboost"].get(k, 0.0), 4) for k in kinds],
        "lightgbm_share": [round(share["lightgbm"].get(k, 0.0), 4) for k in kinds],
    })
    sh["difference"] = (sh["xgboost_share"] - sh["lightgbm_share"]).round(4)
    sh.to_csv(out_summary, index=False)

    if verbose:
        print()
        print(f"  top-{top} columns shared between the two models: "
              f"{len(shared)} / {top}")
        print(f"  rank correlation of the two {importance} rankings, all columns: "
              f"{spearmanr(fits['xgboost']['imp'], fits['lightgbm']['imp']).statistic:+.3f}")
        print()
        print("  for columns only one model picked, |correlation| with its "
              "closest counterpart in the other model's set:")
        for lab, arr in (("xgboost-only -> lightgbm set", best_a),
                         ("lightgbm-only -> xgboost set", best_b),
                         ("random columns -> lightgbm set (null)", best_null)):
            if arr.size:
                print(f"    {lab:40s} median {np.median(arr):.3f}   "
                      f"share > 0.7: {np.mean(arr > 0.7):.0%}")
        print()
        print("  importance share by kind of feature:")
        print(sh.to_string(index=False))
    return sh


# ======================================================================================================  faithfulness
FAITH_OUT = OUT / "faithfulness_groups.csv"
FAITH_CLUSTERS_OUT = OUT / "faithfulness_clusters.csv"
FAITH_REDUCED_OUT = OUT / "faithfulness_groups_reduced.csv"
FAITH_CLUSTERS_REDUCED_OUT = OUT / "faithfulness_clusters_reduced.csv"


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


def faithfulness(families=("a_flank",), models=("xgboost", "lightgbm"), seed: int = 41,
        n_splits: int = 3, tau: float = 0.7, depths=(1, 2, 3, 5),
        fixed=(20, 50, 100), reduced: bool = False,
        verbose: bool = True) -> pd.DataFrame:
    import lightgbm as lgb
    import xgboost as xgb
    from scipy.stats import spearmanr
    from sklearn.model_selection import KFold

    from ..build.build_matrix import load_dataset
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
    out, out_clusters = ((FAITH_REDUCED_OUT, FAITH_CLUSTERS_REDUCED_OUT) if reduced
                         else (FAITH_OUT, FAITH_CLUSTERS_OUT))

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


def faithfulness_summary(reduced: bool = False):
    out, out_clusters = ((FAITH_REDUCED_OUT, FAITH_CLUSTERS_REDUCED_OUT) if reduced
                         else (FAITH_OUT, FAITH_CLUSTERS_OUT))
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


PARTS = ("agreement", "models", "attribution", "faithfulness")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", nargs="+", choices=PARTS, required=True)
    ap.add_argument("--summary-only", action="store_true", help="agreement, models, faithfulness: print the stored summary, fit nothing")
    ap.add_argument("--folds", type=int, default=3, help="agreement, models, faithfulness")
    ap.add_argument("--shap-rows", type=int, default=1000, help="models")
    ap.add_argument("--top", type=int, default=50, help="attribution")
    ap.add_argument("--seed", type=int, default=41, help="attribution")
    ap.add_argument("--importance", choices=["native", "shap"], default="shap", help="attribution")
    ap.add_argument("--tau", type=float, default=0.7, help="faithfulness: correlation threshold of a 'quantity'")
    ap.add_argument("--fixed", type=int, nargs="*", default=None,
                    help="faithfulness: column counts for the original drop-top-N test; pass matched *fractions* when comparing representations of different widths")
    ap.add_argument("--reduced", action="store_true", help="faithfulness: run on the reduced representation instead")
    a = ap.parse_args(argv)
    for part in a.parts:
        if part == "agreement":
            if not a.summary_only:
                agreement(n_splits=a.folds)
            agreement_summary()
        elif part == "models":
            if not a.summary_only:
                models(n_splits=a.folds, shap_rows=a.shap_rows)
            models_summary()
        elif part == "attribution":
            attribution(top=a.top, seed=a.seed, importance=a.importance)
        else:
            if not a.summary_only:
                faithfulness(n_splits=a.folds, tau=a.tau, reduced=a.reduced, fixed=tuple(a.fixed) if a.fixed else (20, 50, 100))
            faithfulness_summary(reduced=a.reduced)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
