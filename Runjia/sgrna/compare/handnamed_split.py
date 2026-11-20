"""Split the 63 hand-named columns into the computable and the rest, and test both.

Finding 2 says the 63 are "non-sequence" columns and that the 427-column reduced
set (364 sequence indicators plus the 63) is non-inferior to all 6,232. An audit
of what they actually contain (`portability_audit.csv`) shows the label is wrong:
**61 of the 63 are the guide re-described** — whole-guide k-mer counts, GC, Tm,
RNA structure, PAM bases, and the 19 `V` columns reconstructible from mono- and
dinucleotide indicators — and only **2** need anything else, both distances that
require gene annotation. A further 15 `V` columns are undecoded and not
reconstructible, so they are unknown rather than computable.

That makes a clean two-way test of a question the report has been circling: when
the 63 help, is it because they carry information the guide does not, or because
they re-spell the guide in a form a depth-limited tree can use?

    computable_46    20 k-mer counts + GC/Tm/structure + 4 PAM + 19
                     reconstructible V -- all functions of the 20-mer
    noncomputable_17 2 annotation distances + 15 undecoded, non-reconstructible V
    all_63           both, which is the published 427 arm

The interpretations, fixed in advance:

  * if `computable_46` carries the gain and `noncomputable_17` does not, the 427
    result is another instance of re-description paying — the same effect as the
    quantum tables, CRISPRoff and ViennaRNA, which all land at +0.011 to +0.019;
  * if `noncomputable_17` carries it, the hand-named columns are a genuine
    source of outside information and annotation-derived features deserve more
    attention than this project has given them.

Run on the published 13,880-row matrix, because that is the only cohort where
these columns exist.

    python -m sgrna.handnamed_split
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config

OUT = config.RESULTS / "handnamed_split.csv"
AUDIT = config.RESULTS / "portability_audit.csv"


def _groups(names: list[str]):
    """The audit's verdict, mapped onto column indices of the loaded matrix."""
    a = pd.read_csv(AUDIT)
    # `portable` in the audit means "computable from the genome", which for the
    # two gene-distance columns still requires annotation -- so they are split
    # out by `kind` rather than by that flag.
    needs_annotation = set(a.loc[a["kind"].str.contains("annotation"), "column"])
    v_bad = set(a.loc[(a["kind"] == "undecoded V column")
                      & (~a["portable"]), "column"])
    hand = set(a["column"])
    idx = {n: i for i, n in enumerate(names)}
    present = [n for n in hand if n in idx]
    missing = len(hand) - len(present)
    if missing:
        print(f"  ! {missing} audited columns are not in the matrix")
    noncomp = [n for n in present if n in needs_annotation or n in v_bad]
    comp = [n for n in present if n not in needs_annotation and n not in v_bad]
    return ([idx[n] for n in comp], [idx[n] for n in noncomp],
            [idx[n] for n in present])


def run(seeds=None, n_splits=None, verbose: bool = True):
    from ..build.build_matrix import load_dataset
    from ..interpret.representation import column_sets
    from ..harness.run_ablation import cross_validate

    seeds = list(seeds or config.SEEDS)
    n_splits = n_splits or config.N_SPLITS
    X, y, names, ids, fam = load_dataset([], verbose=verbose)
    sets = column_sets(names)
    base = np.sort(sets["mono_di_364"])
    comp, noncomp, allh = _groups(list(names))
    if verbose:
        print(f"  base {len(base)} sequence indicators; hand-named split "
              f"{len(comp)} computable / {len(noncomp)} not")

    rows = []
    arms = {"base_364": base,
            "+computable_46": np.sort(np.concatenate([base, comp])),
            "+noncomputable_17": np.sort(np.concatenate([base, noncomp])),
            "+all_63": np.sort(np.concatenate([base, allh]))}
    for arm, cols in arms.items():
        res = cross_validate(X[:, cols], y, [names[i] for i in cols],
                             [fam[i] for i in cols],
                             n_features=len(cols),      # no selection anywhere
                             seeds=seeds, n_splits=n_splits, verbose=False,
                             tag=f"hns_{arm.strip('+')}", experiment=arm)
        rho = res["summary"]["spearman"]["mean"]
        rows.append(dict(arm=arm, n_columns=len(cols), spearman=rho,
                         spearman_sd=res["summary"]["spearman"]["sd"]))
        if verbose:
            print(f"  {arm:>18}: {len(cols):>4} cols  rho {rho:.4f} "
                  f"± {rows[-1]['spearman_sd']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    out = pd.DataFrame(rows).set_index("arm")
    if verbose:
        b = out.loc["base_364", "spearman"]
        for arm in ("+computable_46", "+noncomputable_17", "+all_63"):
            print(f"  {arm:>18} over the base: "
                  f"{out.loc[arm, 'spearman'] - b:+.4f}")
        print(f"written -> {OUT}")
    return out.reset_index()


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
