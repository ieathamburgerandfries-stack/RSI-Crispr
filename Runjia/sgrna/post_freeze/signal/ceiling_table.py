"""The corrected ceiling as a function of replicate agreement and replicate count -- the parametric form of Finding 6.

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

    ./run_ext.sh sgrna.post_freeze.signal.ceiling_table
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..common import OUT

OUTCSV = OUT / "ceiling_parametric.csv"

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=2, help="replicate count for the printed summary")
    a = ap.parse_args(argv)
    df = table()
    df.to_csv(OUTCSV, index=False)
    sub = df[df.k_replicates == a.k]
    print(f"Corrected ceiling for a label averaging k = {a.k} replicates\n")
    print(sub[["replicate_agreement_r", "reliability_of_label", "ceiling", "understated_by",
               "share_of_ceiling_at_rho_0p70"]].round(4).to_string(index=False))
    print("\nHow much the ceiling moves with k, at the ends of the review's range:")
    for r in (0.71, 0.77):
        line = "  r = %.2f: " % r + ",  ".join(f"k={k} -> {float(ceiling(r, k)):.3f}" for k in K_GRID)
        print(line)
    print(f"\nwritten -> {OUTCSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
