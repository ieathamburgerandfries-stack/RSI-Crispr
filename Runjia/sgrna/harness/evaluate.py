"""Quantifying the two things a model is judged on: prediction and explanation.

Predictive power is easy to measure badly. R2 answers "how much variance is
explained", which nobody acts on; what this project's model is actually used
for is **ranking candidate guides**, so the metrics here include the ones that
speak to that directly -- rank correlation, precision in the top decile, and
the expected quality of the guide you end up picking.

Interpretability is harder, because "interpretable" usually means "I looked at
the feature list and it seemed reasonable". That is not a measurement. The
five below are, and each answers a question you would otherwise have to take
on faith:

  stability      Re-run with a different random seed. Do you get the same
                 features? Measured as the Kuncheva index over the selected
                 sets -- 1.0 is identical, 0.0 is chance. An unstable
                 explanation is not an explanation, and with 6,580 correlated
                 columns instability is the default, not the exception.

  concentration  How many features carry the story? If 50% of the importance
                 sits in 12 features you can write a mechanism; if it takes
                 400 you cannot. Reported as the effective feature count and
                 the Gini coefficient of the importance distribution.

  agreement      Do two different importance methods rank features the same
                 way? Split gain and permutation importance measure different
                 things, and where they disagree the story depends on which
                 one you happened to plot.

  faithfulness   Remove the features the explanation says matter, refit, and
                 see how much worse the model gets -- against removing the
                 same number at random. If the gap is small, the explanation
                 is not describing what the model uses.

  direction      Does the sign of each top feature's effect match its
                 univariate direction? A feature can be important and still
                 be used in a direction that contradicts the biology, usually
                 because it is proxying for something else.

Model class changes all of these, and not in the same direction. Run
`compare()` for prediction and `interpretability()` for the rest.

    python -m sgrna.harness.evaluate --models xgboost lightgbm ridge --families a_flank
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from .. import config
from ..build.build_matrix import load_dataset
from .run_ablation import (_impute, _safe_pearson, _safe_spearman,
                           champion_params, make_selector)

RESULTS_PREDICTION = config.RESULTS / "model_comparison.csv"
RESULTS_INTERPRET = config.RESULTS / "interpretability.csv"


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
#
# `scale` marks models that need standardised inputs (fitted on the training
# fold only, like everything else here). Tree models do not care.

def _make(name: str, seed: int):
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(**champion_params(seed)), False
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=16, max_depth=4,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1,
        ), False
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=400, learning_rate=0.05, depth=4, random_seed=seed,
            verbose=0, allow_writing_files=False,
        ), False
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=400, max_features="sqrt", min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        ), False
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(
            n_estimators=400, max_features="sqrt", min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        ), False
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=10.0, random_state=seed), True
    if name == "elastic_net":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000,
                          random_state=seed), True
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(
            hidden_layer_sizes=(128, 32), alpha=1e-3, max_iter=300,
            early_stopping=True, random_state=seed,
        ), True

    if name == "hist_gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=16,
            max_depth=4, l2_regularization=1.0, random_state=seed,
        ), False
    if name == "dart":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            boosting_type="dart", n_estimators=400, learning_rate=0.05,
            num_leaves=16, max_depth=4, drop_rate=0.1,
            random_state=seed, n_jobs=-1, verbose=-1,
        ), False
    if name == "lgbm_huber":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            objective="huber", n_estimators=400, learning_rate=0.05,
            num_leaves=16, max_depth=4, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1,
        ), False
    if name == "svr_rbf":
        from sklearn.svm import SVR
        return SVR(kernel="rbf", C=10.0, epsilon=0.5, gamma="scale",
                   cache_size=1000), True
    if name == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=50, weights="distance",
                                   n_jobs=-1), True
    if name == "pls":
        from sklearn.cross_decomposition import PLSRegression
        return PLSRegression(n_components=25, scale=False), True
    if name == "nystroem_ridge":
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        return make_pipeline(
            Nystroem(kernel="rbf", gamma=None, n_components=800,
                     random_state=seed),
            Ridge(alpha=1.0),
        ), True
    if name == "stack":
        return _Stack(seed), True
    raise ValueError(f"Unknown model {name!r}. Choose from {MODELS}")


MODELS = ("xgboost", "lightgbm", "catboost", "random_forest", "extra_trees",
          "ridge", "elastic_net", "mlp")

# A second wave, added once the first showed that the eight above spanned only
# 1.8 percentile points on the guide actually picked. These probe genuinely
# different inductive biases: a different boosting implementation, a dropout
# variant, a robust loss, two kernel methods, a purely local learner, a latent-
# factor linear model, and a blend of the three best families.
MODELS_EXTRA = ("hist_gbm", "dart", "lgbm_huber", "svr_rbf",
                "nystroem_ridge", "knn", "pls", "stack")

ALL_MODELS = MODELS + MODELS_EXTRA


class _Stack:
    """Ridge over the out-of-fold predictions of three different learners.

    Worth testing because the model families disagree in *how* they are wrong:
    boosting captures interactions, ridge captures the additive part, and the
    forest is the most stable. If their errors were independent, blending
    would beat all three. Whether they actually are is the question.

    Standardised inputs are passed through to the members that want them; the
    trees do not care either way.
    """

    def __init__(self, seed: int = 41, n_splits: int = 5):
        self.seed = seed
        self.n_splits = n_splits
        self.members = ("lightgbm", "random_forest", "ridge")

    def fit(self, X, y):
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import KFold

        oof = np.zeros((len(X), len(self.members)))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for tr, va in kf.split(X):
            for j, name in enumerate(self.members):
                m, _ = _make(name, self.seed)
                m.fit(X[tr], y[tr])
                oof[va, j] = np.asarray(m.predict(X[va])).ravel()
        self.meta_ = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0)).fit(oof, y)
        self.fitted_ = []
        for name in self.members:
            m, _ = _make(name, self.seed)
            m.fit(X, y)
            self.fitted_.append(m)
        return self

    def predict(self, X):
        P = np.column_stack(
            [np.asarray(m.predict(X)).ravel() for m in self.fitted_])
        return self.meta_.predict(P)

    @property
    def weights_(self):
        return dict(zip(self.members, np.round(self.meta_.coef_, 3)))


def _standardise(train, test):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd == 0] = 1.0
    return (train - mu) / sd, (test - mu) / sd


# --------------------------------------------------------------------------
# Predictive metrics
# --------------------------------------------------------------------------


def _read_csv_if_any(path) -> pd.DataFrame:
    """Read a results CSV, tolerating a missing or zero-byte file.

    Zero-byte matters: the project folder is a mount where `rm` is refused, so
    "clearing" a results file in practice means truncating it, and pandas
    raises EmptyDataError on that.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _append(df: pd.DataFrame, path):
    if df.empty:
        return
    prev = _read_csv_if_any(path)
    if not prev.empty:
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(path, index=False)


def top_decile_precision(y_true, y_pred, q: float = 0.1) -> float:
    """Of the guides the model ranks in the top q, what fraction really are?

    This is the metric that matches how the model gets used: you take the
    best-looking guides and build them. Chance is q.
    """
    k = max(1, int(len(y_true) * q))
    picked = np.argsort(y_pred)[::-1][:k]
    truly = set(np.argsort(y_true)[::-1][:k].tolist())
    return len(truly.intersection(picked.tolist())) / k


def bootstrap_interval(y_true, y_pred, metric_fn, n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile CI for a metric by resampling test rows with replacement."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    if n < 3:
        v = float(metric_fn(y_true, y_pred))
        return dict(point=v, lo=v, hi=v, n_boot=0, alpha=alpha)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(float(metric_fn(y_true[idx], y_pred[idx])))
    stats = np.asarray(stats)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return dict(
        point=float(metric_fn(y_true, y_pred)),
        lo=lo, hi=hi, n_boot=n_boot, alpha=alpha,
        se=float(stats.std(ddof=1)) if len(stats) > 1 else float("nan"),
    )


def bootstrap_interval_grouped(y_true, y_pred, groups, metric_fn,
                               n_boot: int = 2000, alpha: float = 0.05,
                               seed: int = 0) -> dict:
    """Bootstrap by resampling *groups* (genes, bins), not rows."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    groups = np.asarray(groups, dtype=object)
    uniq = np.unique(groups)
    if len(uniq) < 3:
        return bootstrap_interval(y_true, y_pred, metric_fn, n_boot, alpha, seed)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(groups, picked)
        if mask.sum() < 3:
            continue
        stats.append(float(metric_fn(y_true[mask], y_pred[mask])))
    if not stats:
        return bootstrap_interval(y_true, y_pred, metric_fn, n_boot, alpha, seed)
    stats = np.asarray(stats)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return dict(
        point=float(metric_fn(y_true, y_pred)),
        lo=lo, hi=hi, n_boot=len(stats), alpha=alpha,
        se=float(stats.std(ddof=1)),
        n_groups=int(len(uniq)),
    )


def within_gene_spearman(y_true, y_pred, gene_names, min_guides: int = 10) -> dict:
    """Mean Spearman ρ inside each gene with enough guides."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    gene_names = np.asarray(gene_names, dtype=object)
    df = pd.DataFrame(dict(gene=gene_names, y=y_true, pred=y_pred))
    rhos = []
    sizes = []
    for g, sub in df.groupby("gene"):
        if len(sub) < min_guides:
            continue
        rho, _ = spearmanr(sub["y"], sub["pred"])
        if np.isfinite(rho):
            rhos.append(float(rho))
            sizes.append(len(sub))
    if not rhos:
        return dict(n_genes=0, mean_rho=float("nan"), sd_rho=float("nan"),
                    median_rho=float("nan"), rhos=[], gene_sizes=[])
    rhos = np.asarray(rhos)
    return dict(
        n_genes=int(len(rhos)),
        mean_rho=float(rhos.mean()),
        sd_rho=float(rhos.std(ddof=1)),
        median_rho=float(np.median(rhos)),
        rhos=rhos.tolist(),
        gene_sizes=sizes,
    )


def expected_pick_percentile(y_true, y_pred, k: int = 10,
                             n_draws: int = 20000, seed: int = 0) -> float:
    """Draw k candidates, take the model's favourite, report its true percentile.

    Directly answers "if I have ten candidates for a gene and trust the model,
    how good is the guide I build?". Chance is the 50th percentile.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    ranks = pd.Series(y_true).rank(pct=True).to_numpy()
    idx = rng.integers(0, n, size=(n_draws, k))
    best = idx[np.arange(n_draws), np.argmax(y_pred[idx], axis=1)]
    return float(ranks[best].mean() * 100)


def expected_pick_percentile_within_gene(y_true, y_pred, gene_names, k: int = 10,
                                         n_draws: int = 20000, seed: int = 0) -> float:
    """Pick-percentile simulation with candidates drawn from one gene at a time."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    gene_names = np.asarray(gene_names, dtype=object)
    df = pd.DataFrame(dict(gene=gene_names, y=y_true, pred=y_pred))
    percentiles = []
    for _, sub in df.groupby("gene"):
        if len(sub) < k:
            continue
        y = sub["y"].to_numpy()
        pred = sub["pred"].to_numpy()
        ranks = pd.Series(y).rank(pct=True).to_numpy()
        n = len(y)
        for _ in range(max(1, n_draws // max(1, df["gene"].nunique()))):
            idx = rng.integers(0, n, size=k)
            pick = idx[np.argmax(pred[idx])]
            percentiles.append(float(ranks[pick] * 100))
    if not percentiles:
        return float("nan")
    return float(np.mean(percentiles))


def calibration_slope(y_true, y_pred) -> float:
    """Slope of true on predicted. 1.0 means the scale is right."""
    if np.std(y_pred) == 0:
        return float("nan")
    return float(np.polyfit(y_pred, y_true, 1)[0])


def compare(families=("a_flank",), models=MODELS, seeds=(41, 42, 43),
            n_splits: int = 5, n_features: int | None = None,
            deadline: float | None = None, verbose: bool = True) -> pd.DataFrame:
    """Same folds, same selected features, different predictor.

    Feature selection is deliberately held fixed (XGBoost gain, top 300) so
    the comparison isolates the predictor. A model that selects its own
    features is a different experiment -- see `interpretability`.
    """
    from sklearn.model_selection import KFold
    import xgboost as xgb

    n_features = n_features or config.N_FEATURES
    X, y, names, ids, fam = load_dataset(list(families), verbose=verbose)
    if verbose:
        print(f"  matrix {X.shape[0]:,} x {X.shape[1]:,}, models: {list(models)}")

    # Resume support: every (model, seed, fold) is written as it finishes, and
    # one already on disk is skipped. Eight model classes over fifteen folds is
    # more than fits in one sitting.
    done: set = set()
    prev = _read_csv_if_any(RESULTS_PREDICTION)
    if not prev.empty:
        done = set(zip(prev["model"], prev["seed"], prev["fold"]))
        if verbose and done:
            print(f"  resuming: {len(done)} model-folds already recorded")

    rows = []
    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(kf.split(X), start=1):
            if deadline and time.time() > deadline:
                if verbose:
                    print("    time budget reached")
                return pd.DataFrame(rows)
            todo = [m for m in models if (m, seed, fold) not in done]
            if not todo:
                continue
            X_tr, X_va = _impute(X[tr], X[va])
            y_tr, y_va = y[tr], y[va]

            sel = make_selector(seed)
            sel.fit(X_tr, y_tr)
            top = np.argsort(sel.feature_importances_)[::-1][:n_features]
            A, B = X_tr[:, top], X_va[:, top]

            for name in todo:
                model, needs_scale = _make(name, seed)
                a, b = _standardise(A, B) if needs_scale else (A, B)
                t0 = time.time()
                model.fit(a, y_tr)
                pred = np.asarray(model.predict(b)).ravel()
                row = dict(
                    model=name, seed=seed, fold=fold,
                    r2=r2_score(y_va, pred),
                    spearman=_safe_spearman(y_va, pred),
                    pearson=_safe_pearson(y_va, pred),
                    top_decile_precision=top_decile_precision(y_va, pred),
                    pick_percentile=expected_pick_percentile(y_va, pred),
                    calibration_slope=calibration_slope(y_va, pred),
                    fit_seconds=round(time.time() - t0, 2),
                )
                rows.append(row)
                _append(pd.DataFrame([row]), RESULTS_PREDICTION)
                if verbose:
                    print(f"    {name:14s} seed {seed} fold {fold}  "
                          f"rho {row['spearman']:.4f}  {row['fit_seconds']:.1f}s")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Interpretability metrics
# --------------------------------------------------------------------------


def kuncheva(sets: list[set], n_total: int) -> float:
    """Stability of feature selection, corrected for chance.

    Two selections of k features out of n share k^2/n by luck alone; the
    Kuncheva index subtracts that, so 0 means "no better than random overlap"
    and 1 means identical. Plain Jaccard flatters a large k badly.
    """
    if len(sets) < 2:
        return float("nan")
    k = np.mean([len(s) for s in sets])
    expected = k * k / n_total
    scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = len(sets[i] & sets[j])
            denom = k - expected
            scores.append((inter - expected) / denom if denom else np.nan)
    return float(np.nanmean(scores))


def concentration(importance: np.ndarray) -> dict:
    """How few features carry the story."""
    imp = np.asarray(importance, dtype=float)
    imp = np.where(np.isfinite(imp), np.abs(imp), 0.0)
    total = imp.sum()
    if total <= 0:
        return dict(n_for_50pct=np.nan, n_for_90pct=np.nan, gini=np.nan,
                    effective_features=np.nan)
    share = np.sort(imp)[::-1] / total
    cum = np.cumsum(share)
    # Effective number of features: the inverse participation ratio, i.e. how
    # many equally-weighted features would give the same concentration.
    eff = 1.0 / np.sum(share ** 2)
    s = np.sort(share)
    n = len(s)
    gini = float((2 * np.sum((np.arange(1, n + 1)) * s) / (n * np.sum(s))) - (n + 1) / n)
    return dict(
        n_for_50pct=int(np.searchsorted(cum, 0.5) + 1),
        n_for_90pct=int(np.searchsorted(cum, 0.9) + 1),
        gini=gini,
        effective_features=float(eff),
    )


def native_importance(model, X, y, names) -> np.ndarray | None:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        return np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    return None


def permutation_importance_fast(model, X, y, n_repeats: int = 3,
                                seed: int = 0) -> np.ndarray:
    """Drop in Spearman when each column is shuffled. No refitting."""
    rng = np.random.default_rng(seed)
    base = _safe_spearman(y, np.asarray(model.predict(X)).ravel())
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            drops.append(base - _safe_spearman(y, np.asarray(model.predict(Xp)).ravel()))
        out[j] = np.mean(drops)
    return out


def interpretability(families=("a_flank",), models=("xgboost",), seed: int = 41,
                     n_splits: int = 5, n_features: int | None = None,
                     n_permute: int = 40, verbose: bool = True,
                     importance: str = "native") -> pd.DataFrame:
    """Stability, concentration, method agreement, faithfulness, direction.

    `importance` picks what the explanation is read from. `"native"` is split
    gain / impurity decrease / |coefficient| -- what these numbers were
    originally computed with. `"shap"` is mean |SHAP value|, which is what the
    docs now report: it agrees across model families where gain does not
    (`importance.py`, part `models`), so every metric here that depends on a ranking --
    stability, rank agreement, concentration -- is a different number under it.
    """
    from sklearn.model_selection import KFold

    from ..interpret.importance import shap_importance

    if importance not in ("native", "shap"):
        raise ValueError(f"unknown importance {importance!r}")

    n_features = n_features or config.N_FEATURES
    X, y, names, ids, fam = load_dataset(list(families), verbose=verbose)
    names = np.asarray(names)

    rows = []
    for name in models:
        selected_sets: list[set] = []
        importances: list[np.ndarray] = []
        agree_scores, faith_scores, direction_scores = [], [], []

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(kf.split(X), start=1):
            X_tr, X_va = _impute(X[tr], X[va])
            y_tr, y_va = y[tr], y[va]

            # Each model selects with its OWN importances here -- the question
            # is whether *this* model tells a stable story, not whether
            # XGBoost's selection is stable.
            model, needs_scale = _make(name, seed)
            a, b = _standardise(X_tr, X_va) if needs_scale else (X_tr, X_va)
            model.fit(a, y_tr)
            if importance == "shap":
                imp = shap_importance(
                    model, b, seed=seed,
                    linear_std=a.std(axis=0) if needs_scale else None)
            else:
                imp = native_importance(model, a, y_tr, names)
            if imp is None:
                if verbose:
                    print(f"    {name}: no {importance} importance, skipped")
                continue
            top = np.argsort(imp)[::-1][:n_features]
            selected_sets.append(set(top.tolist()))
            importances.append(imp)

            # --- method agreement, on the top slice only (cost) ------------
            probe = top[:n_permute]
            m2, _ = _make(name, seed)
            m2.fit(a[:, probe], y_tr)
            perm = permutation_importance_fast(m2, b[:, probe], y_va, seed=seed)
            if np.std(perm) > 0 and np.std(imp[probe]) > 0:
                agree_scores.append(spearmanr(imp[probe], perm).statistic)

            # --- faithfulness: drop the top-20 vs 20 random ---------------
            rng = np.random.default_rng(seed + fold)
            keep_top = np.setdiff1d(top, top[:20])
            rand20 = rng.choice(top, size=20, replace=False)
            keep_rand = np.setdiff1d(top, rand20)
            scores = {}
            for label, cols in (("top", keep_top), ("rand", keep_rand),
                                ("all", top)):
                m3, _ = _make(name, seed)
                m3.fit(a[:, cols], y_tr)
                scores[label] = _safe_spearman(
                    y_va, np.asarray(m3.predict(b[:, cols])).ravel())
            drop_top = scores["all"] - scores["top"]
            drop_rand = scores["all"] - scores["rand"]
            faith_scores.append(drop_top - drop_rand)

            # --- direction: does the model use features the way the data does?
            agree = 0
            for j in top[:20]:
                col = X_tr[:, j]
                if np.std(col) == 0:
                    continue
                uni = spearmanr(col, y_tr).statistic
                # local slope of prediction vs feature, as the model sees it
                pred = np.asarray(model.predict(a)).ravel()
                mdl = spearmanr(col, pred).statistic
                if np.isfinite(uni) and np.isfinite(mdl) and uni * mdl > 0:
                    agree += 1
            direction_scores.append(agree / 20)

            if verbose:
                print(f"    {name} fold {fold} done")

        if not importances:
            if verbose:
                print(f"  {name}: exposes no feature importances, skipped")
            continue

        mean_imp = np.mean(importances, axis=0)
        cross = [spearmanr(importances[i], importances[j]).statistic
                 for i in range(len(importances))
                 for j in range(i + 1, len(importances))]
        row = dict(
            model=name,
            stability_kuncheva=kuncheva(selected_sets, X.shape[1]),
            importance_rank_agreement=float(np.nanmean(cross)),
            method_agreement=float(np.nanmean(agree_scores)) if agree_scores else np.nan,
            faithfulness=float(np.nanmean(faith_scores)) if faith_scores else np.nan,
            direction_consistency=float(np.nanmean(direction_scores)),
        )
        row.update(concentration(mean_imp))
        rows.append(row)
        if verbose:
            print(f"  {name}: {row}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=["xgboost"], choices=MODELS)
    ap.add_argument("--families", nargs="*", default=["a_flank"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[41, 42, 43])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--what", choices=["prediction", "interpretability", "both"],
                    default="both")
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--importance", choices=["native", "shap"], default="shap",
                    help="what the interpretability metrics are read from; "
                         "SHAP is the reported default, native is the "
                         "historical gain-based figure")
    args = ap.parse_args(argv)

    deadline = time.time() + args.time_budget if args.time_budget else None

    if args.what in ("prediction", "both"):
        print("=== predictive comparison ===")
        compare(args.families, args.models, tuple(args.seeds), args.folds,
                deadline=deadline)
        df = _read_csv_if_any(RESULTS_PREDICTION)
        if not df.empty:
            print(df.groupby("model")[["r2", "spearman", "top_decile_precision",
                                       "pick_percentile", "fit_seconds"]]
                  .agg(["mean", "count"]).round(4).to_string())
            print(f"\n{RESULTS_PREDICTION}")

    if args.what in ("interpretability", "both"):
        print("\n=== interpretability ===")
        df = interpretability(args.families, args.models, seed=args.seeds[0],
                              n_splits=args.folds, importance=args.importance)
        df.insert(1, "importance", args.importance)
        _append(df, RESULTS_INTERPRET)
        if not df.empty:
            print(df.round(4).to_string(index=False))
            print(f"\nappended to {RESULTS_INTERPRET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
