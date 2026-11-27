"""Per-family permutation floors, paired fold by fold.

A permutation control re-runs a family with its rows shuffled: same columns,
same count, no relationship to the label. Whatever it scores is that family's
noise floor, and a gain inside the floor is not a gain.

Until 2026-10-03 only four families had one, and the project quoted a single
floor measured on `a_flank` as if it applied to all of them. It does not: the
floor depends on how many columns a family contributes and how easily they
survive selection, so a 348-column family and a 13-column family have
different floors.

This module also fixes how the delta was computed. `ablation_*.csv` stores the
mean over whatever folds each experiment happens to have, so a family run on
three seeds was being differenced against a baseline run on five -- mixing a
real effect with the seed-to-seed spread. Here every delta is **paired on
(seed, fold)**, so the two arms always see identical data.

    python -m sgrna.controls.floors

Report section: Appendix A.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from .. import config

OUT = config.RESULTS / "permutation_floors.csv"


def build(tag: str = "main", verbose: bool = True) -> pd.DataFrame:
    folds = pd.read_csv(config.RESULTS / f"folds_{tag}.csv")
    base = folds[folds["experiment"] == "baseline"]

    def paired(name: str):
        arm = folds[folds["experiment"] == name]
        if arm.empty:
            return None
        m = arm.merge(base, on=["seed", "fold"], suffixes=("", "_base"))
        if m.empty:
            return None
        d_r2 = m["r2"] - m["r2_base"]
        d_rho = m["spearman"] - m["spearman_base"]
        t = ttest_rel(m["spearman"], m["spearman_base"]) if len(m) > 1 else None
        return dict(n_folds=len(m), d_r2=d_r2.mean(), d_rho=d_rho.mean(),
                    d_rho_sd=d_rho.std(),
                    folds_positive=int((d_rho > 0).sum()),
                    p=float(t.pvalue) if t is not None else np.nan)

    families = sorted({e for e in folds["experiment"].unique()
                       if "(permuted)" not in e and e != "baseline"})
    rows = []
    for fam in families:
        real = paired(fam)
        floor = paired(f"{fam} (permuted)")
        if real is None:
            continue
        row = dict(family=fam, real_folds=real["n_folds"],
                   real_d_r2=real["d_r2"], real_d_rho=real["d_rho"],
                   real_p=real["p"])
        if floor is None:
            row.update(floor_folds=0, floor_d_r2=np.nan, floor_d_rho=np.nan,
                       ratio=np.nan, verdict="no control run")
        else:
            ratio = (real["d_rho"] / abs(floor["d_rho"])
                     if floor["d_rho"] != 0 else np.inf)
            row.update(floor_folds=floor["n_folds"], floor_d_r2=floor["d_r2"],
                       floor_d_rho=floor["d_rho"], ratio=ratio,
                       verdict=("indistinguishable from noise" if ratio < 1.5
                                else "marginal" if ratio < 5
                                else "clear of the floor"))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("real_d_rho", ascending=False)
    df.to_csv(OUT, index=False)
    if verbose:
        cols = ["family", "real_d_rho", "floor_d_rho", "ratio", "verdict"]
        print(df[cols].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return df


if __name__ == "__main__":
    build()
