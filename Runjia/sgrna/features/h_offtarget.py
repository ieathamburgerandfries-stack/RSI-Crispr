"""Family H -- off-target burden and target copy number.

This family is not about Cas9 chemistry. It is about the other half of what
`cut.score` measures.

The label is a depletion score: a guide looks efficient when cells carrying it
die. Two things distort that in opposite directions.

  * **Multiple targets.** A guide whose protospacer occurs more than once --
    in an rRNA operon, an IS element, a duplicated gene family -- causes
    several simultaneous double-strand breaks and is far more lethal than its
    sequence alone would predict. Morgens et al. (2017) turned exactly this
    effect into a genome-scale readout in human screens.
  * **Near-cognate sites.** Partial matches titrate Cas9 away from the
    intended target and add their own toxicity. In E. coli, where every
    successful cut is lethal, even a handful of tolerated mismatches matters.

Counting these honestly means comparing every guide against every NGG-flanked
protospacer in the genome -- 541,809 of them across both strands. Done one
guide at a time in Python that is hopeless; done as one vectorised numpy
comparison per guide it is about two minutes for the whole library, which is
cheap enough to just do exactly rather than approximate.

Mismatches are counted twice over: across the whole 20-mer, and restricted to
the PAM-proximal seed, because Cas9 tolerates PAM-distal mismatches far more
readily than seed mismatches.

Columns
-------
eng.offt.copies            exact NGG-flanked copies of the protospacer
eng.offt.mm<k>             sites at exactly k mismatches (k = 1..4)
eng.offt.seed_exact_mm<k>  of those, the ones whose seed matches perfectly
eng.offt.le3               sites within 3 mismatches
eng.offt.min_mismatch      distance to the closest non-self site
eng.offt.weighted_burden   sum of 1/4^k over near-cognate sites
eng.offt.multi_copy        does the protospacer occur more than once
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..build import genome
from ..io_utils import save_block

PREFIX = "eng.offt"

MAX_MISMATCH = 4
SEED_LENGTH = 12     # PAM-proximal nucleotides
REPORT_EVERY = 2048

_BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}


def _encode(seqs, length: int) -> np.ndarray:
    """(n, L) uint8 array; anything not ACGT becomes 255 so it never matches."""
    arr = np.full((len(seqs), length), 255, dtype=np.uint8)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s[:length]):
            arr[i, j] = _BASE_CODE.get(ch, 255)
    return arr


def build(index: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    site_seqs, _, _ = genome.protospacer_sites()
    length = len(site_seqs[0])
    if verbose:
        print(f"  encoding {len(site_seqs):,} genomic protospacer sites ...")
    sites = _encode(site_seqs, length)
    seed_slice = slice(length - SEED_LENGTH, length)

    guides = index["protospacer"].fillna("").tolist()
    ok = np.array(
        [isinstance(s, str) and len(s) == length and "N" not in s for s in guides]
    )
    encoded = _encode(
        [s if len(s) == length else "N" * length for s in guides], length
    )

    n = len(guides)
    counts = np.zeros((n, MAX_MISMATCH + 1), dtype=np.int32)
    seed_counts = np.zeros((n, MAX_MISMATCH + 1), dtype=np.int32)
    min_mm = np.full(n, np.nan)

    for row in range(n):
        if not ok[row]:
            continue
        diff = sites != encoded[row][None, :]
        mm = diff.sum(axis=1, dtype=np.int16)
        seed_mm = diff[:, seed_slice].sum(axis=1, dtype=np.int16)
        for j in range(MAX_MISMATCH + 1):
            counts[row, j] = int(np.count_nonzero(mm == j))
            seed_counts[row, j] = int(np.count_nonzero((mm == j) & (seed_mm == 0)))
        nonself = mm[mm > 0]
        min_mm[row] = float(nonself.min()) if nonself.size else np.nan
        if verbose and row % REPORT_EVERY == 0:
            print(f"    {row:,} / {n:,} guides")

    out = pd.DataFrame({config.ID_COL: index[config.ID_COL].to_numpy()})
    out[f"{PREFIX}.copies"] = np.where(ok, counts[:, 0], np.nan)
    for k in range(1, MAX_MISMATCH + 1):
        out[f"{PREFIX}.mm{k}"] = np.where(ok, counts[:, k], np.nan)
        out[f"{PREFIX}.seed_exact_mm{k}"] = np.where(ok, seed_counts[:, k], np.nan)
    out[f"{PREFIX}.le3"] = np.where(ok, counts[:, 1:4].sum(axis=1), np.nan)
    out[f"{PREFIX}.min_mismatch"] = min_mm

    weights = np.array([0.0] + [4.0 ** -k for k in range(1, MAX_MISMATCH + 1)])
    out[f"{PREFIX}.weighted_burden"] = np.where(
        ok, (counts * weights).sum(axis=1), np.nan
    )
    out[f"{PREFIX}.multi_copy"] = np.where(ok, (counts[:, 0] > 1).astype(int), np.nan)
    return out


def main() -> None:
    from ..io_utils import load_guide_index

    idx = load_guide_index()
    block = build(idx)
    path = save_block(block, "h_offtarget")
    print(f"h_offtarget: {block.shape[0]:,} rows x {block.shape[1] - 1:,} features -> {path}")
    print(f"  multi-copy guides: {int(np.nansum(block[f'{PREFIX}.multi_copy'])):,}")
    print(
        "  median distance to the nearest other NGG site: "
        f"{np.nanmedian(block[f'{PREFIX}.min_mismatch']):.0f} mismatches"
    )


if __name__ == "__main__":
    main()
