"""The cutting-screen figure on the within-gene definition used for the dCas9 screens.

On Cui and Rousset the coding-strand gain of family A was split into a between-gene
part and a within-gene part (strand_within_gene.py). The +0.157 on the cutting screen
was a pooled figure. Here the cutting screen (crisprHAL's curated cohort, the guides
inside an annotated gene) gets the same split: out-of-fold predictions from
gene-grouped CV, ten seeds, rho of gene means (between) and rho of values with each
gene's mean removed (within), genes with at least five guides.

    python -m sgrna.ext.cut_within_gene
"""
from __future__ import annotations

import numpy as np

from sgrna.curated_ablation import _base, _block, _groups
from sgrna.strand_within_gene import run

from .common import OUT


def main() -> int:
    X, y, names, protos = _base()
    groups, ok = _groups("gene", protos)
    X, y, protos, groups = X[ok], y[ok], protos[ok], groups[ok]
    v, fn = _block("a_flank", protos)
    arms = {"base": (X, names), "a_flank": (np.hstack([X, v]), names + fn)}
    print(f"  {X.shape[0]:,} guides in {len(np.unique(groups)):,} genes", flush=True)
    run("Cutting (curated, genic guides)", arms, y, groups, groups, "sfg10_cut",
        OUT / "cut_within_gene.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
