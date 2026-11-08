"""The label's reliability is per guide, and the model treats it as uniform.

Finding 6 puts the label's ceiling at 0.946, measured from crisprHAL 2's own
per-replicate counts. That is an *average* over a population that is far from
homogeneous: pairing Cas9 replicate 1 with control 1 and replicate 2 with
control 2 gives two independent estimates of every guide's score, and their
disagreement runs from 0.06 to 2.39 log2 units between the 5th and 95th
percentiles. Every one of the 33,567 curated guides has both estimates -- the
join is exact -- and the model currently weights all of them equally.

Two facts make this worth a run rather than a remark.

  * Disagreement is **not** a read-depth effect: rho(disagreement, total counts)
    is -0.053. So this is per-guide information that the obvious proxy misses,
    and a depth-weighted arm is included to show that.
  * It is mildly related to the label itself: rho(|y|, disagreement) = -0.236,
    because large effects are measured more consistently. Inverse-variance
    weighting therefore up-weights extreme guides, which is a change in what
    the fit is pulled toward and not only a change in noise handling. That is
    the reason for the permuted-weight control.

**The weights touch training only.** Every validation row is scored, unweighted,
against the released label -- bit-identical to crisprHAL's `y`. Weighting the
evaluation would be the same mistake the report criticises in their cutoff,
which is applied before the split and so cleans the test set too.

Arms:

    unweighted                  the baseline, measured in this same loop
    inverse_variance            w = 1 / (sigma^2 + s0^2), moderated
    inverse_variance_permuted   the same weights, shuffled across guides
    depth                       w proportional to total read count
    trim_worst_decile           drop the noisiest 10% of training rows
    trim_random_decile          drop a random 10% -- the row-count control

A caveat the result has to be read against: the released label is an ALDEx2
posterior mean, which already shrinks noisy guides toward the prior. Some of
this reliability information is therefore spent before we see it, which predicts
a smaller effect than the spread of disagreement suggests.

    python -m sgrna.replicate_weight
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config
from .curated_ablation import _base, _block

COUNTS = config.EXTERNAL / "crisprHAL" / "rawData" / "E_coli_WTSpCas9_counts.tsv"
OUT = config.RESULTS / "replicate_weight.csv"
EPS = 0.5
ARMS = ("unweighted", "inverse_variance", "inverse_variance_permuted",
        "inverse_variance_flat_in_y", "depth",
        "trim_worst_decile", "trim_random_decile")


def replicate_noise(protos) -> pd.DataFrame:
    """Per-guide disagreement between the two independent log-ratio estimates."""
    if not COUNTS.exists():
        raise FileNotFoundError(f"{COUNTS} not found -- see curated_ceiling.py")
    d = pd.read_csv(COUNTS, sep="\t")
    for c in ("Cas1_S", "Cas2_S", "dCas1_NS", "dCas2_NS"):
        d[c + "_cpm"] = d[c] / d[c].sum() * 1e6
    d["lr1"] = np.log2((d.Cas1_S_cpm + EPS) / (d.dCas1_NS_cpm + EPS))
    d["lr2"] = np.log2((d.Cas2_S_cpm + EPS) / (d.dCas2_NS_cpm + EPS))
    d["depth"] = d[["Cas1_S", "Cas2_S", "dCas1_NS", "dCas2_NS"]].sum(axis=1)
    m = d.set_index("sgRNA").reindex(protos)
    if m["lr1"].isna().any():
        raise ValueError(f"{int(m['lr1'].isna().sum())} guides have no counts")
    # (lr1 - lr2)^2 / 2 is an unbiased 1-df estimate of a single estimate's
    # variance; the mean of two has a quarter of the squared difference. One
    # degree of freedom is a very noisy variance, so it is moderated toward the
    # pooled value -- otherwise a handful of guides whose two estimates happen
    # to agree take almost all the weight.
    sigma2 = (m["lr1"].to_numpy() - m["lr2"].to_numpy()) ** 2 / 4.0
    return pd.DataFrame(dict(sigma2=sigma2, depth=m["depth"].to_numpy()))


def _flatten_in_y(w: np.ndarray, y: np.ndarray, bins: int = 20) -> np.ndarray:
    """Strip the weights' relationship with |y|, keeping their spread within it.

    Reliability is related to the label: rho(|y|, disagreement) = -0.236,
    because large effects are measured more consistently. So inverse-variance
    weighting does two things at once -- it down-weights noisy guides, and it
    tilts the fit toward extreme ones. This renormalises the weights to mean 1
    inside each of 20 equal-count bins of |y|, which leaves the first effect and
    removes the second. If this arm recovers what `inverse_variance` loses, the
    tilt was the problem and the noise information is usable.
    """
    out = w.copy()
    edges = np.quantile(np.abs(y), np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, np.abs(y), side="right") - 1, 0, bins - 1)
    for b in range(bins):
        m = idx == b
        if m.sum() and out[m].mean() > 0:
            out[m] = out[m] / out[m].mean()
    return out


def _weights(arm: str, noise: pd.DataFrame, rng,
             y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(weight per guide, boolean keep mask). Weights are normalised to mean 1."""
    n = len(noise)
    keep = np.ones(n, dtype=bool)
    s2 = noise["sigma2"].to_numpy()
    s0 = float(np.median(s2))                    # the moderating prior scale

    if arm == "unweighted":
        w = np.ones(n)
    elif arm.startswith("inverse_variance"):
        w = 1.0 / (s2 + s0)
        if arm.endswith("permuted"):
            w = w[rng.permutation(n)]
        elif arm.endswith("flat_in_y"):
            w = _flatten_in_y(w, y)
    elif arm == "depth":
        w = noise["depth"].to_numpy().astype(float)
    elif arm == "trim_worst_decile":
        w = np.ones(n)
        keep = s2 <= np.quantile(s2, 0.90)
    elif arm == "trim_random_decile":
        w = np.ones(n)
        keep = rng.random(n) >= 0.10
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return w / w.mean(), keep


def run(seeds=(41, 42, 43), n_splits=None, arms=ARMS, verbose: bool = True):
    import xgboost as xgb
    from scipy.stats import spearmanr
    from sklearn.model_selection import KFold

    from .run_ablation import _impute, champion_params, make_selector

    n_splits = n_splits or config.N_SPLITS
    X, y, names, protos = _base()
    A, _ = _block("a_flank", protos)
    X = np.hstack([X, A]).astype(np.float32)
    noise = replicate_noise(protos)
    if verbose:
        q = np.percentile(np.sqrt(noise.sigma2) * 2, [5, 50, 95])
        print(f"  {X.shape[0]:,} x {X.shape[1]:,}; replicate disagreement "
              f"{q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} (5/50/95 pct)", flush=True)

    rows = []
    for arm in arms:
        per_fold = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            w, keep = _weights(arm, noise, rng, y)
            for tr, va in KFold(n_splits, shuffle=True,
                                random_state=seed).split(X):
                # The mask applies to training rows only; every validation row
                # is scored, and scored against the released label unweighted.
                tr = tr[keep[tr]]
                A_, B_ = _impute(X[tr], X[va])
                sel = make_selector(seed)
                sel.fit(A_, y[tr], sample_weight=w[tr])
                top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
                m = xgb.XGBRegressor(**champion_params(seed))
                m.fit(A_[:, top], y[tr], sample_weight=w[tr])
                per_fold.append(float(spearmanr(y[va],
                                                m.predict(B_[:, top])).statistic))
        rows.append(dict(arm=arm, n_fits=len(per_fold),
                         spearman=float(np.mean(per_fold)),
                         spearman_sd=float(np.std(per_fold, ddof=1)),
                         train_rows=int(keep.sum()),
                         weight_ratio=float(w.max() / w.min())))
        if verbose:
            r = rows[-1]
            print(f"  {arm:28s}: rho {r['spearman']:+.4f} "
                  f"(sd {r['spearman_sd']:.4f}, {r['train_rows']:,} train rows)",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    out = pd.DataFrame(rows).set_index("arm")
    if verbose and "unweighted" in out.index:
        base = out.loc["unweighted", "spearman"]
        print("\n  against the unweighted baseline:")
        for arm in out.index:
            if arm != "unweighted":
                print(f"    {arm:28s} {out.loc[arm, 'spearman'] - base:+.4f}")
        print(f"written -> {OUT}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--raw-label", action="store_true")
    a = ap.parse_args(argv)
    if a.raw_label:
        raw_label()
    else:
        run(arms=tuple(a.arms) if a.arms else ARMS)
    return 0


# --------------------------------------------------------------------------
# The one route the posterior label closes off
# --------------------------------------------------------------------------
# Weighting fails above because ALDEx2's `diff.btw` is a posterior **mean**: it
# has already shrunk noisy guides toward the prior, so the reliability was spent
# before we saw it. That diagnosis makes a prediction. Fit the same model to the
# **raw** CPM log-ratio -- the plain mean of the two independent estimates, which
# correlates |rho| 0.957 with the released label -- and the shrinkage is no
# longer pre-applied, so inverse-variance weighting should help, or at least
# stop hurting.
#
# The comparison is strictly within-label. A model fitted to the raw log-ratio
# cannot be compared with 0.7084 on the released label: the targets differ, the
# raw one is noisier by construction, and a lower rho would say nothing. Only
# weighted-against-unweighted on the same target is meaningful here.
RAW_OUT = config.RESULTS / "replicate_weight_raw.csv"


def _raw_label(protos) -> np.ndarray:
    """The plain CPM log-ratio: the mean of the two independent estimates."""
    d = pd.read_csv(COUNTS, sep="\t")
    for c in ("Cas1_S", "Cas2_S", "dCas1_NS", "dCas2_NS"):
        d[c + "_cpm"] = d[c] / d[c].sum() * 1e6
    lr1 = np.log2((d.Cas1_S_cpm + EPS) / (d.dCas1_NS_cpm + EPS))
    lr2 = np.log2((d.Cas2_S_cpm + EPS) / (d.dCas2_NS_cpm + EPS))
    d["raw"] = (lr1 + lr2) / 2.0
    return d.set_index("sgRNA").reindex(protos)["raw"].to_numpy()


def raw_label(seeds=(41, 42, 43), n_splits=None,
              arms=("unweighted", "inverse_variance",
                    "inverse_variance_permuted"), verbose: bool = True):
    """Does weighting help once the label is not already shrunk?"""
    import xgboost as xgb
    from scipy.stats import spearmanr
    from sklearn.model_selection import KFold

    from .run_ablation import _impute, champion_params, make_selector

    n_splits = n_splits or config.N_SPLITS
    X, y_released, names, protos = _base()
    A, _ = _block("a_flank", protos)
    X = np.hstack([X, A]).astype(np.float32)
    noise = replicate_noise(protos)
    y = _raw_label(protos)
    ok = np.isfinite(y)
    if verbose:
        r = spearmanr(y[ok], y_released[ok]).statistic
        print(f"  raw log-ratio label: {ok.sum():,} guides, sd {np.nanstd(y[ok]):.3f} "
              f"(released sd {np.nanstd(y_released[ok]):.3f}), "
              f"rho against the released label {r:+.4f}", flush=True)

    rows = []
    for arm in arms:
        per_fold = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            w, keep = _weights(arm, noise, rng, y)
            for tr, va in KFold(n_splits, shuffle=True,
                                random_state=seed).split(X):
                tr = tr[keep[tr] & ok[tr]]
                va = va[ok[va]]
                A_, B_ = _impute(X[tr], X[va])
                sel = make_selector(seed)
                sel.fit(A_, y[tr], sample_weight=w[tr])
                top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
                m = xgb.XGBRegressor(**champion_params(seed))
                m.fit(A_[:, top], y[tr], sample_weight=w[tr])
                per_fold.append(float(spearmanr(y[va],
                                                m.predict(B_[:, top])).statistic))
        rows.append(dict(arm=arm, label="raw log-ratio", n_fits=len(per_fold),
                         spearman=float(np.mean(per_fold)),
                         spearman_sd=float(np.std(per_fold, ddof=1))))
        if verbose:
            print(f"  {arm:28s}: rho {rows[-1]['spearman']:+.4f} "
                  f"(sd {rows[-1]['spearman_sd']:.4f})", flush=True)
        pd.DataFrame(rows).to_csv(RAW_OUT, index=False)

    out = pd.DataFrame(rows).set_index("arm")
    if verbose and "unweighted" in out.index:
        b = out.loc["unweighted", "spearman"]
        print("\n  against the unweighted baseline, same label:")
        for arm in out.index:
            if arm != "unweighted":
                print(f"    {arm:28s} {out.loc[arm, 'spearman'] - b:+.4f}")
        print(f"written -> {RAW_OUT}")
    return out

if __name__ == "__main__":
    raise SystemExit(main())
