"""Joint arms on the curated cohort: how much of the eleven family gains overlap?

Appendix B3 reports eleven single-family increments over one base and warns that
they must not be added up — summing them gives +0.70 ρ, which is not a real
quantity. That warning rests on the *published* cohort's joint arms, where
`i_shape` on top of family A was worth +0.001 against +0.036 alone, and the three
positional families collapsed into one variable. Neither has been checked here,
so the warning is inherited rather than measured.

The arms are chosen to answer specific suspicions rather than to enumerate:

    a+i             `i_shape` is flank bendability and `a_flank` is flank
                    composition over the same DNA -- the suspected duplicate
    a+d_sup         the largest external family on top of the largest flank one
    d_sup+e+g       the three positional families, which collapsed to one
                    variable on the published matrix
    a+d_sup+i       the three biggest effects together
    all_eleven      every family at once: the ceiling on stacking them
    guide_only      `b_energy` + `c_folding` + `f_methylation`, the three that
                    read nothing outside the guide, as a control -- they should
                    be close to additive because they overlap only through the
                    20-mer

Everything else matches `curated_ablation`: same base, same folds, same seeds,
selection fitted inside each training split.

    python -m sgrna.curated_joint
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config
from .curated_ablation import _base, _block

OUT = config.RESULTS / "curated_joint.csv"

ARMS = {
    "a_flank+i_shape": ("a_flank", "i_shape"),
    "a_flank+d_supercoiling": ("a_flank", "d_supercoiling"),
    "d_supercoiling+e_nucleoid+g_transcription": ("d_supercoiling", "e_nucleoid",
                                                  "g_transcription"),
    "a_flank+d_supercoiling+i_shape": ("a_flank", "d_supercoiling", "i_shape"),
    "guide_only_three": ("b_energy", "c_folding", "f_methylation"),
    "all_eleven": ("a_flank", "b_energy", "c_folding", "d_mechanics",
                   "d_supercoiling", "e_nucleoid", "f_methylation",
                   "g_transcription", "h_offtarget", "i_shape", "j_periodicity"),
}


def run(arms=None, seeds=None, n_splits=None, n_features=300,
        verbose: bool = True):
    from ..harness.run_ablation import cross_validate

    arms = arms or ARMS
    seeds = list(seeds or config.SEEDS)
    n_splits = n_splits or config.N_SPLITS
    X, y, names, protos = _base()

    singles = pd.read_csv(config.RESULTS / "ablation_curated.csv").set_index("family")
    base_rho = float(singles.loc["baseline", "spearman"])

    rows = []
    if OUT.exists() and OUT.stat().st_size:
        rows = [r for r in pd.read_csv(OUT).to_dict("records")
                if r["arm"] not in arms]

    for arm, fams in arms.items():
        if any(r["arm"] == arm for r in rows):
            continue
        vals, nm, fam_of = [], [], []
        for f in fams:
            v, fn = _block(f, protos)
            vals.append(v)
            nm += fn
            fam_of += [f] * len(fn)
        Xa = np.hstack([X] + vals)
        res = cross_validate(Xa, y, names + nm,
                             ["base"] * len(names) + fam_of,
                             n_features=n_features, seeds=seeds,
                             n_splits=n_splits, verbose=False,
                             tag=f"joint_{len(fams)}_{fams[0]}",
                             experiment=arm)
        rho = res["summary"]["spearman"]["mean"]
        additive = float(sum(singles.loc[f, "d_spearman"] for f in fams))
        rows.append(dict(arm=arm, n_families=len(fams), n_cols=Xa.shape[1],
                         spearman=rho, spearman_sd=res["summary"]["spearman"]["sd"],
                         d_spearman=rho - base_rho,
                         sum_of_singles=additive,
                         overlap=additive - (rho - base_rho),
                         share_realised=(rho - base_rho) / additive))
        if verbose:
            r = rows[-1]
            print(f"  {arm:<44} rho {rho:.4f}  joint {r['d_spearman']:+.4f}  "
                  f"sum {additive:+.4f}  realised {r['share_realised']:.0%}",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    if verbose:
        print()
        print(out.round(4).to_string(index=False))
        print(f"written -> {OUT}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args(argv)
    run(arms={k: v for k, v in ARMS.items() if k in a.arms} if a.arms else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
