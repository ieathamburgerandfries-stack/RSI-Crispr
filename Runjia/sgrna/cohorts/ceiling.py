"""The ceiling on any predictor of a replicate-averaged label is the square root of its Spearman-Brown reliability, not the replicate agreement the field quotes: 0.946 [0.944, 0.947] for the label SLICER trains on, measured from crisprHAL 2's per-replicate counts, and tabulated here for any agreement and replicate count.

Two parts, both Finding 6 (the field's "a correlation of ~0.7 is an upper limit" restored to a square root, then corrected for the label being an average over replicates):

  measured   the ceiling for the label we actually train on, from the published per-replicate counts          -> curated_ceiling.json
  table      the same correction as arithmetic over the replicate agreement r and the replicate count k          -> ceiling_parametric.csv

    python -m sgrna.cohorts.ceiling --parts measured table

------------------------------------------------------------------------------------------------------------------------------
measured
------------------------------------------------------------------------------------------------------------------------------
The ceiling for the label we actually train on, from replicate agreement.

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

------------------------------------------------------------------------------------------------------------------------------
table
------------------------------------------------------------------------------------------------------------------------------
The open item this closes
-------------------------
Finding 6 corrects the field's ceiling (Konstantakos et al. 2022, *NAR*: replicate agreement 0.71-0.77, therefore "a
correlation of ~0.7 constitutes an upper limit") by restoring the square root and then applying Spearman-Brown for the fact
that a released activity score is an average over replicates. The report could only work the correction for crisprHAL,
because that is the one dataset in the comparison whose per-replicate counts are released, and so it had to end with
*"Not done: corrected ceilings for the specific datasets the review tabulates, which needs their per-replicate counts."*

That is the wrong shape of gap to leave open, because the correction needs no data beyond two numbers a dataset's own paper
states: the replicate agreement *r* and how many replicates *k* the released label averages. So the correction is tabulated
here over the whole plausible range of both, which lets anyone apply it to any dataset in that review -- or to one published
tomorrow -- without re-deriving it. Nothing here is an estimate; it is arithmetic over the two assumptions below.

Which *r* you have
------------------
The chain depends on what the published correlation is *between*, and this is the one place it is easy to go wrong:

  **single replicates** (two individual replicates correlated with each other). Then *r* is the reliability of one
  measurement, the released k-replicate label has reliability kr/(1+(k-1)r) by Spearman-Brown, and the ceiling on
  predicting that label is its square root. This is the crisprHAL case and the one the review's 0.71-0.77 most likely is.

  **k-replicate aggregates** (two averages of k replicates each, correlated). Then *r* already *is* the reliability of the
  released label, no Spearman-Brown step applies, and the ceiling is simply sqrt(r). The single-measurement reliability it
  implies is r/(k - (k-1)r), which `single_from_aggregate` inverts, so a dataset reported either way can be placed on the
  same axis.

What the correction is not
--------------------------
Exact for Pearson correlations of scores with independent, additive noise. Spearman's rho is a correlation of ranks, so
these are close approximations rather than identities, and both steps assume replicate noise is independent. That is
an assumption, here and in the field's own derivation, and nothing in this module tests it: shared batch effects between
replicates inflate *r* and so inflate the ceiling, and a real systematic difference between replicates deflates it. (An
earlier attempt compared the WT and eSpCas9 screens of the same guides and found that 30% of their disagreement is
predictable from sequence -- but those are two different enzymes, so that is expected biology and says nothing about
whether replicates are independent. Finding 6 avoids the question by using crisprHAL's released per-replicate counts.)

Report section: Finding 6.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .. import config
from ..common import OUT

COUNTS = config.EXTERNAL / "crisprHAL" / "rawData" / "E_coli_WTSpCas9_counts.tsv"
MEASURED = OUT / "curated_ceiling.json"
TABLE = OUT / "ceiling_parametric.csv"
CUTOFF = 73          # their 5-fold-CV-chosen average control-read threshold
EPS = 0.5


# ====================================================================================================  measured
def _cpm(c: pd.Series) -> pd.Series:
    return c / c.sum() * 1e6


def _spearman_brown(r: float) -> float:
    """Reliability of a mean of two parallel measurements, each of reliability r."""
    return 2 * r / (1 + r)


def measured(n_boot: int = 2000, seed: int = 0, verbose: bool = True) -> dict:
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
    MEASURED.write_text(json.dumps(out, indent=1))
    if verbose:
        print(f"written -> {MEASURED}")
    return out



# ====================================================================================================  table
# The agreements the field actually quotes, plus ours. 0.71 and 0.77 are the ends of the review's range; 0.697/0.783 are
# the two crisprHAL test-set performances the report is measured against; 0.8097 is our own measured agreement.
R_GRID = (0.50, 0.55, 0.60, 0.65, 0.697, 0.70, 0.71, 0.75, 0.77, 0.783, 0.80, 0.8097, 0.85, 0.90)
K_GRID = (1, 2, 3, 4, 6)


def spearman_brown(r: float | np.ndarray, k: int) -> float | np.ndarray:
    """Reliability of the mean of `k` measurements, each of reliability `r`."""
    return k * np.asarray(r, float) / (1.0 + (k - 1) * np.asarray(r, float))


def ceiling(r: float | np.ndarray, k: int = 2) -> float | np.ndarray:
    """Largest correlation any predictor can reach with a label that averages `k` replicates agreeing pairwise at `r`."""
    return np.sqrt(spearman_brown(r, k))


def single_from_aggregate(r_agg: float | np.ndarray, k: int) -> float | np.ndarray:
    """Inverse of Spearman-Brown: the single-measurement reliability implied by agreement `r_agg` between k-replicate means."""
    r_agg = np.asarray(r_agg, float)
    return r_agg / (k - (k - 1) * r_agg)


def table(r_grid=R_GRID, k_grid=K_GRID) -> pd.DataFrame:
    rows = []
    for r in r_grid:
        for k in k_grid:
            c = float(ceiling(r, k))
            rows.append(dict(replicate_agreement_r=r, k_replicates=k,
                             reliability_of_label=float(spearman_brown(r, k)),
                             ceiling=c,
                             field_ceiling_quoted_as_r=r,
                             understated_by=c - r,
                             share_of_ceiling_at_rho_0p70=0.70 / c,
                             single_reliability_if_r_is_an_aggregate=float(single_from_aggregate(r, k)) if k > 1 else r))
    return pd.DataFrame(rows)



def parametric(k: int = 2) -> pd.DataFrame:
    """Write the table and print the summary for a label averaging `k` replicates."""
    df = table()
    df.to_csv(TABLE, index=False)
    sub = df[df.k_replicates == k]
    print(f"Corrected ceiling for a label averaging k = {k} replicates\n")
    print(sub[["replicate_agreement_r", "reliability_of_label", "ceiling", "understated_by",
               "share_of_ceiling_at_rho_0p70"]].round(4).to_string(index=False))
    print("\nHow much the ceiling moves with k, at the ends of the review's range:")
    for r in (0.71, 0.77):
        line = "  r = %.2f: " % r + ",  ".join(f"k={kk} -> {float(ceiling(r, kk)):.3f}" for kk in K_GRID)
        print(line)
    print(f"\nwritten -> {TABLE}")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", nargs="+", choices=("measured", "table"), required=True)
    ap.add_argument("--k", type=int, default=2, help="table: replicate count for the printed summary")
    a = ap.parse_args(argv)
    if "measured" in a.parts:
        measured()
    if "table" in a.parts:
        parametric(a.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
