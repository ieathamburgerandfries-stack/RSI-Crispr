"""On C. rodentium, which no choice in this project ever saw, the tuned model beats the default model by +0.0172, within the range seen inside E. coli. E. coli -> C. rodentium.

The question. The tuned settings were chosen on 70% of the E. coli curated cohort's genes, and the report's own estimate of how much that
adaptive selection inflates a gain is +0.0087 rho, measured on genes sealed *inside* E. coli. A sturdier seal is a different organism.
The C. rodentium TevSpCas9 screen (25,210 guides, a 5.35 Mb genome) was used only to test the frozen champion's transfer; no feature,
setting, column count or model was ever chosen on it, so it is as untouched as anything left in this project. (The other E. coli
screen, eSpCas9, is not: it scores the same guide library at the same loci, so it can be held out as a *label* but not as a *locus*.)

The design. Train on the whole E. coli WT-SpCas9 cohort, score C. rodentium zero-shot, with three arms that share a protocol: the frozen
champion, the tuned XGBoost (d099x), and the XGBoost average of five (the arms of the head-to-head). Only this direction is clean: the
reverse (train on C. rodentium, score E. coli) would score the very genes the settings were chosen on.

Prediction, written before the run: the tuned arm is at least as good as the champion (probability about 0.65), by less than it is
within E. coli (+0.014 on genes it never saw, +0.019 and +0.026 on crisprHAL 2's two splits), because part of that gain is the data
it was chosen on.
Decision rule: the difference (tuned minus champion), paired on guides and bootstrapped over 100 kb blocks (neighbouring guides share
DNA and are not independent), with three seeds averaged. Lower bound above 0: the tuning's benefit survives a sealed organism. Interval
spanning 0: inconclusive. Upper bound below 0: tuning hurt outside the data it was chosen on, which would be evidence of adaptive
overfitting in the settings.

    python -m sgrna.compare.sealed_organism
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sgrna import config

from ..common import OUT
from .vs_crisprhal import member_specs, train_predict
from ..harness.metrics import zscore as _z

SEEDS = (41, 42, 43)
ARMS = {"champion": ["champion"], "tuned XGBoost (d099x)": ["d099x"], "XGBoost average of 5": ["d099x", "d077x", "d021x", "d080x", "d000x"]}
N_BOOT = 1000


def _rho(y, p):
    return float(spearmanr(y, p).statistic)


def _block_boot(y, a, b, block, rng):
    """Spearman(a) - Spearman(b), resampling 100 kb blocks of the target genome."""
    u, inv = np.unique(block, return_inverse=True)
    rows = [np.flatnonzero(inv == k) for k in range(len(u))]
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ix = np.concatenate([rows[k] for k in rng.integers(0, len(u), len(u))])
        out[i] = _rho(y[ix], a[ix]) - _rho(y[ix], b[ix])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    def cached(name):
        """The stored matrix and label of one screen (the cache `transfer.build` writes); the C. rodentium genome is not needed to read it."""
        d = config.INTERIM / ("transfer_%s" % name)
        meta = json.loads((d / "meta.json").read_text())
        return np.asarray(np.load(d / "X.npy", mmap_mode="r"), np.float32), np.asarray(meta["y"], float), list(meta["names"]), meta
    Xs, ys, names, _ = cached("WT-SpCas9")
    Xt, yt, names_t, meta = cached("TevSpCas9")
    assert names == names_t, "feature names differ between the two screens"
    block = (np.asarray(meta["left"], float) // 100_000).astype(int)
    needed = sorted({n for v in ARMS.values() for n in v})
    specs = member_specs(needed)
    print(f"train {Xs.shape[0]:,} E. coli guides, score {Xt.shape[0]:,} C. rodentium guides in {len(set(block))} 100 kb blocks", flush=True)
    per_seed = {s: {} for s in SEEDS}
    for s in SEEDS:
        preds, _ = train_predict(specs, Xs, ys, Xt, seed=s)
        for arm, members in ARMS.items():
            per_seed[s][arm] = np.mean([_z(preds[m]) for m in members], axis=0)
        print("  seed", s, {a: round(_rho(yt, p), 4) for a, p in per_seed[s].items()}, flush=True)
    mean_pred = {arm: np.mean([per_seed[s][arm] for s in SEEDS], axis=0) for arm in ARMS}
    rng = np.random.default_rng(0)
    rows = []
    for arm in ARMS:
        r_seeds = [_rho(yt, per_seed[s][arm]) for s in SEEDS]
        row = dict(arm=arm, rho_seed_mean=float(np.mean(r_seeds)), rho_seed_sd=float(np.std(r_seeds)), rho_of_seed_average=_rho(yt, mean_pred[arm]))
        if arm != "champion":
            d = row["rho_of_seed_average"] - _rho(yt, mean_pred["champion"])
            lo, hi = _block_boot(yt, mean_pred[arm], mean_pred["champion"], block, rng)
            row.update(minus_champion=d, lo=lo, hi=hi)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "sealed_organism.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
