"""Decompose `d_mechanics`'s +0.099 into representation and new DNA.

Finding 2 rests on `d_mechanics` as the clean case: nearest-neighbour duplex
thermodynamics is a sum of dinucleotide parameters, so it is a deterministic
function of a 20-mer the base already encodes losslessly, and it still buys
+0.099 ρ. A feature carrying no information, worth a tenth of a correlation.

**Eight of its thirty columns are not functions of the 20-mer.**
`eng.mech.flank.{up,dn}{25,100}.{dG37,dG_per_bp}` are duplex thermodynamics of
the flanking DNA, which the base does not contain -- found by a reconstruction
probe, where those eight columns came back at out-of-fold R² ≈ 0 from the 364
mono- and dinucleotide indicator columns while nine of the guide-only columns
came back at exactly 1.0000. So the headline arm mixes the claim with family
A's mechanism, and the clean number is the guide-only one.

This runs the decomposition: the same base, and three arms over it.

    guide_only   22 columns, every one a function of the protospacer
    flank_only    8 columns, thermodynamics of DNA outside it
    full         all 30, the arm already reported

    python -m sgrna.controls.no_information
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..harness.curated import load_base, load_block

OUT = config.RESULTS / "no_information.csv"
FLANK = ".flank."


def run(seeds=None, n_splits=None, n_features=300, verbose: bool = True):
    from ..harness.run_ablation import cross_validate

    seeds = list(seeds or config.SEEDS)
    n_splits = n_splits or config.N_SPLITS
    X, y, names, protos = load_base()
    v, fnames = load_block("d_mechanics", protos)
    is_flank = np.array([FLANK in f for f in fnames])
    if verbose:
        print(f"  d_mechanics: {len(fnames)} columns, {is_flank.sum()} from the "
              f"flanks, {(~is_flank).sum()} from the protospacer")

    arms = {
        "guide_only": ~is_flank,
        "flank_only": is_flank,
        "full": np.ones(len(fnames), dtype=bool),
    }

    # The baseline is the one already measured for the curated ablation table
    # -- same rows, same protocol, same tag, so `cross_validate` resumes from
    # its fold checkpoint instead of refitting it.
    res = cross_validate(X, y, names, ["base"] * len(names), n_features=n_features,
                         seeds=seeds, n_splits=n_splits, verbose=False,
                         tag="curabl_baseline", experiment="baseline")
    base_rho = res["summary"]["spearman"]["mean"]
    rows = [dict(arm="baseline", n_family_cols=0, spearman=base_rho,
                 spearman_sd=res["summary"]["spearman"]["sd"], d_spearman=0.0)]
    if verbose:
        print(f"  {'baseline':>12}: rho {base_rho:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)

    for arm, mask in arms.items():
        sub = [f for f, m in zip(fnames, mask) if m]
        Xa = np.hstack([X, v[:, mask]])
        res = cross_validate(Xa, y, names + sub,
                             ["base"] * len(names) + ["d_mechanics"] * len(sub),
                             n_features=n_features, seeds=seeds, n_splits=n_splits,
                             verbose=False, tag=f"noinfo_{arm}", experiment=arm)
        rho = res["summary"]["spearman"]["mean"]
        rows.append(dict(arm=arm, n_family_cols=int(mask.sum()), spearman=rho,
                         spearman_sd=res["summary"]["spearman"]["sd"],
                         d_spearman=rho - base_rho))
        if verbose:
            print(f"  {arm:>12}: rho {rho:.4f}   delta {rho - base_rho:+.4f}  "
                  f"({int(mask.sum())} cols)", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    out = pd.DataFrame(rows)
    if verbose:
        print()
        print(out.round(4).to_string(index=False))
        print(f"written -> {OUT}")
    return out


if __name__ == "__main__":
    run()
