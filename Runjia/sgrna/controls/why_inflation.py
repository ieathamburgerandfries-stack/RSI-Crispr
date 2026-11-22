"""Why did every family's Δρ roughly triple when the cohort moved?

Four things changed between the published 13,880-row matrix and crisprHAL's
33,567 curated guides, and "noise was removed from the label" is only one of
them:

  1. **the label** -- crisprHAL's read-count-filtered score replaces `cut_score`;
  2. **which rows** -- the cohorts are not nested (8,617 shared, 24,950 only
     crisprHAL, 5,263 only Noshay), and the curated release is genome-wide,
     4,013 genes at 7.5 guides each against a 1,222-gene panel at 11.3;
  3. **how many rows** -- 2.4× more;
  4. **the base representation** -- the curated base is the 6,169 regenerated
     columns and carries none of the published base's 63 hand-named summaries,
     so every family is now measured against a weaker comparison.

Only (1) is noise removal, and (1) is already ruled out: at **fixed rows** --
the 8,617 shared guides, published base -- family A is worth +0.0892 against
`cut_score` and +0.0936 against the curated score (`ablation_label_*.csv`). A
label swap moves it by 0.004 while the cohort move takes it from +0.080 to
+0.164.

This module separates (3) from (2) and (4) the only way available: re-run the
curated ablation on a **random 13,880-row subsample of the curated cohort**. The
subsample holds the label, the row population and the base fixed and changes only
the count, so

    Δρ(13,880 curated rows) ≈ Δρ(33,567)   =>  row count is not the cause
    Δρ(13,880 curated rows) ≈ Δρ(published) =>  row count is the cause

Three seeds of subsample, because one draw of 13,880 from 33,567 is itself a
sample.

    python -m sgrna.controls.why_inflation --families a_flank d_mechanics
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config
from ..cohorts.curated_ablation import _base, _block

OUT = config.RESULTS / "why_inflation.csv"
N_SUB = 13_880


def run(families=("a_flank", "d_mechanics"), n_sub: int = N_SUB,
        sub_seeds=(0, 1, 2), n_features=300, verbose: bool = True):
    from ..harness.run_ablation import cross_validate

    X, y, names, protos = _base()
    rows = []
    for sseed in sub_seeds:
        idx = np.random.default_rng(sseed).choice(len(y), n_sub, replace=False)
        Xs, ys, ps = X[idx], y[idx], protos[idx]
        res = cross_validate(Xs, ys, names, ["base"] * len(names),
                             n_features=n_features, verbose=False,
                             tag=f"whyinf{sseed}_baseline", experiment="baseline")
        base = res["summary"]["spearman"]["mean"]
        rows.append(dict(sub_seed=sseed, n_rows=n_sub, family="baseline",
                         spearman=base, d_spearman=0.0))
        if verbose:
            print(f"  subsample {sseed}: baseline rho {base:.4f}", flush=True)
        for fam in families:
            v, fnames = _block(fam, ps)
            Xa = np.hstack([Xs, v])
            res = cross_validate(Xa, ys, names + fnames,
                                 ["base"] * len(names) + [fam] * len(fnames),
                                 n_features=n_features, verbose=False,
                                 tag=f"whyinf{sseed}_{fam}", experiment=fam)
            rho = res["summary"]["spearman"]["mean"]
            rows.append(dict(sub_seed=sseed, n_rows=n_sub, family=fam,
                             spearman=rho, d_spearman=rho - base))
            if verbose:
                print(f"  subsample {sseed}: {fam:>14} rho {rho:.4f}   "
                      f"delta {rho - base:+.4f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    if verbose:
        print()
        print(out.groupby("family")["d_spearman"].agg(["mean", "std"]).round(4)
              .to_string())
        print(f"written -> {OUT}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--families", nargs="*", default=["a_flank", "d_mechanics"])
    ap.add_argument("--n-sub", type=int, default=N_SUB)
    ap.add_argument("--sub-seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args(argv)
    run(families=tuple(a.families), n_sub=a.n_sub, sub_seeds=tuple(a.sub_seeds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
