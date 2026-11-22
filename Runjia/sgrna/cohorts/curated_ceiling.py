"""The ceiling for the label we actually train on, from replicate agreement.

Finding 11's 0.937 was derived for Noshay's *published* score and inherited by
the curated arms without re-derivation -- Part 4 item 8. Two routes were tried
and failed: crisprHAL 2's paper reports no replicate statistics, and their label
is an ALDEx2 Bayesian posterior mean, so Spearman-Brown on a two-replicate
geometric mean does not describe it.

The third route works, because the per-replicate counts are published
(`data/rawData/E_coli_WTSpCas9_counts.tsv` in their paper repository):

    curl -sLO https://raw.githubusercontent.com/tbrowne5/Better-data-for-\\
    better-predictions-data-curation-improves-deep-learning-for-sgRNA-Cas9-\\
    prediction/main/data/rawData/E_coli_WTSpCas9_counts.tsv

Four columns: two Cas9 replicates and two dead-nuclease controls. That allows
**two independent estimates of the same quantity** -- pair Cas9 replicate 1 with
control 1, and replicate 2 with control 2 -- whose agreement is the reliability
of a one-pair score. Spearman-Brown then gives the reliability of the
two-replicate score they release, and its square root is the highest Spearman
any predictor can reach against it.

Three things make the number trustworthy, and they are the reason this is worth
more than the inherited figure:

  * **The cohort reproduces.** Their filter is an average control count of 73
    (their own R script, in this directory); applying it gives 33,871 guides
    against the 33,567 they release.
  * **The estimator reproduces.** A plain CPM log-ratio of the two conditions
    correlates **|rho| = 0.957** with their released label, so the replicates
    being correlated here are the replicates behind that label.
  * **It is a lower bound.** ALDEx2's posterior shrinks noisy guides toward the
    prior, which can only *raise* reliability above a plain log-ratio's. So the
    true ceiling is at least this.

    python -m sgrna.cohorts.curated_ceiling
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .. import config

COUNTS = config.EXTERNAL / "crisprHAL" / "rawData" / "E_coli_WTSpCas9_counts.tsv"
OUT = config.RESULTS / "curated_ceiling.json"
CUTOFF = 73          # their 5-fold-CV-chosen average control-read threshold
EPS = 0.5


def _cpm(c: pd.Series) -> pd.Series:
    return c / c.sum() * 1e6


def _spearman_brown(r: float) -> float:
    """Reliability of a mean of two parallel measurements, each of reliability r."""
    return 2 * r / (1 + r)


def run(n_boot: int = 2000, seed: int = 0, verbose: bool = True) -> dict:
    if not COUNTS.exists():
        raise FileNotFoundError(f"{COUNTS} not found -- see this module's docstring")
    d = pd.read_csv(COUNTS, sep="\t")
    for c in ("Cas1_S", "Cas2_S", "dCas1_NS", "dCas2_NS"):
        d[c + "_cpm"] = _cpm(d[c])
    d["avg_control"] = (d["dCas1_NS"] + d["dCas2_NS"]) / 2
    f = d[d["avg_control"] >= CUTOFF].copy()

    def lr(num, den):
        return np.log2((f[num + "_cpm"] + EPS) / (f[den + "_cpm"] + EPS))

    arms = {"1-1 / 2-2": (lr("Cas1_S", "dCas1_NS"), lr("Cas2_S", "dCas2_NS")),
            "1-2 / 2-1": (lr("Cas1_S", "dCas2_NS"), lr("Cas2_S", "dCas1_NS"))}

    rng = np.random.default_rng(seed)
    out = {"n_counts": int(len(d)), "n_after_cutoff": int(len(f)),
            "cutoff": CUTOFF, "arms": {}}
    for name, (u, v) in arms.items():
        u, v = u.to_numpy(), v.to_numpy()
        rs = float(spearmanr(u, v).statistic)
        rp = float(pearsonr(u, v).statistic)
        boot = np.empty(n_boot)
        for i in range(n_boot):
            k = rng.integers(0, len(u), len(u))
            boot[i] = np.sqrt(_spearman_brown(spearmanr(u[k], v[k]).statistic))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["arms"][name] = dict(
            replicate_spearman=rs, replicate_pearson=rp,
            reliability_spearman_brown=_spearman_brown(rs),
            ceiling=float(np.sqrt(_spearman_brown(rs))),
            ceiling_ci=[float(lo), float(hi)],
            ceiling_from_pearson=float(np.sqrt(_spearman_brown(rp))))
        if verbose:
            a = out["arms"][name]
            print(f"  {name}: replicate rho {rs:.4f} -> reliability "
                  f"{a['reliability_spearman_brown']:.4f} -> ceiling "
                  f"**{a['ceiling']:.4f}** [{lo:.4f}, {hi:.4f}]")

    # Does a plain log-ratio recover their released label?
    f["score2"] = (arms["1-1 / 2-2"][0] + arms["1-1 / 2-2"][1]) / 2
    meta = json.loads((config.INTERIM / "expanded_meta.json").read_text())
    lab = pd.DataFrame({"sgRNA": meta["protospacers"], "y": meta["y"]}).merge(
        f[["sgRNA", "score2"]], on="sgRNA")
    rho = float(spearmanr(lab["y"], lab["score2"]).statistic)
    out["reconstruction"] = dict(n_shared=int(len(lab)), spearman=rho)
    if verbose:
        print(f"  plain CPM log-ratio against their released label: "
              f"rho {rho:+.4f} on {len(lab):,} shared guides "
              f"(sign is orientation: theirs scores depletion as high)")
    OUT.write_text(json.dumps(out, indent=1))
    if verbose:
        print(f"written -> {OUT}")
    return out


if __name__ == "__main__":
    run()
