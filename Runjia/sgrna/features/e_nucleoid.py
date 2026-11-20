"""Family E -- nucleoid organisation, the bacterial answer to chromatin.

In eukaryotes, target accessibility is one of the strongest non-sequence
predictors of Cas9 activity: nucleosomes measurably impede the enzyme both in
vitro and in vivo, and a 2024 study found that spatial-density features
derived from Hi-C raised model R-squared by 11-12% on human datasets.

Bacteria have no nucleosomes, but they are not naked either. The E. coli
nucleoid is folded into macrodomains and MukBEF-organised loops, and
nucleoid-associated proteins -- H-NS above all -- coat AT-rich regions in
dense filaments. Whether that occludes Cas9 is an open question, and Lioy et
al. (2018, Cell) published exactly the data needed to ask it: Hi-C contact
maps at 5 kb for wild type plus single mutants of H-NS, HU, Fis, MatP, MukB
and ZapB, along with a 3D model of the chromosome.

Two kinds of feature come out of that:

1. **Where the locus sits in 3D.** Directly ported from the human study --
   mean distance to the k nearest bins, and how many bins lie within a radius.
   Their finding was that *less* crowded sites cut better.

2. **Which protein holds it there.** The ratio of contacts a bin makes in the
   wild type versus in a mutant is a proxy for how much that NAP structures
   the locus. A guide that fails inside an H-NS-dependent domain and not
   elsewhere is the signature worth looking for.

Columns
-------
eng.nuc.density.k*      mean 3D distance to the k nearest bins
eng.nuc.radius.r*       number of bins within radius r
eng.nuc.contacts.*      total, short-range and long-range contact counts
eng.nuc.insulation      directional index -- domain boundary strength
eng.nuc.dep.<protein>   log2(wild type / mutant) contact ratio for the bin
eng.nuc.xyz.*           raw 3D coordinates and distance from the centroid
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config
from ..build import genome
from ..io_utils import save_block

PREFIX = "eng.nuc"

K_NEIGHBOURS = (5, 10, 25, 50)
RADII = (5.0, 10.0, 20.0)

# Short range = within this many bins (5 kb each), i.e. 100 kb.
SHORT_RANGE_BINS = 20
INSULATION_BINS = 10

WT_MATRIX = "MAT_WT_LB_37C_rep1_filtered.dat"
MUTANT_MATRICES = {
    "hns": "MAT_HNS_LB_30C_rep1_filtered.txt",
    "hu": "MAT_HU_LB_37C_rep1_filtered.dat",
    "fis": "MAT_Fis_MM_30C_filtered.dat",
    "matp": "MAT_MatP_MM_30C_filtered.txt",
    "mukb": "MAT_MukB_MM_22C__rep1_filtered.txt",
}
# Each mutant was grown in the condition its own wild-type control matches,
# so ratios are taken against the matching WT rather than the LB 37C one.
MUTANT_REFERENCE = {
    "hns": "MAT_WT_LB_30C_rep1_filtered.dat",
    "hu": "MAT_WT_LB_37C_rep1_filtered.dat",
    "fis": "MAT_WT_MM_30C_rep1_filtered.txt",
    "matp": "MAT_WT_MM_30C_rep1_filtered.txt",
    "mukb": "MAT_WT_MM_22C_filtered.txt",
}


@lru_cache(maxsize=16)
def load_matrix(name: str) -> np.ndarray | None:
    path = config.LIOY_MATRIX_DIR / name
    if not path.exists():
        return None
    return np.loadtxt(path, dtype=np.float64)


@lru_cache(maxsize=1)
def load_coordinates() -> np.ndarray | None:
    if not config.LIOY_XYZ.exists():
        return None
    return np.loadtxt(config.LIOY_XYZ, dtype=np.float64)


def _bin_of(position: float, n_bins: int) -> int:
    """5 kb bin index for a 1-based genomic position."""
    return int((position - 1) // config.LIOY_BIN_SIZE) % n_bins


def _coordinate_features(xyz: np.ndarray) -> pd.DataFrame:
    """Per-bin crowding, from the published 3D model."""
    n = len(xyz)
    diff = xyz[:, None, :] - xyz[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    ordered = np.sort(dist, axis=1)

    out = {}
    for k in K_NEIGHBOURS:
        out[f"{PREFIX}.density.k{k}"] = ordered[:, :k].mean(axis=1)
    for r in RADII:
        out[f"{PREFIX}.radius.r{r:g}"] = (dist <= r).sum(axis=1).astype(float)

    centroid = xyz.mean(axis=0)
    out[f"{PREFIX}.xyz.from_centroid"] = np.sqrt(
        ((xyz - centroid) ** 2).sum(axis=1)
    )
    return pd.DataFrame(out, index=np.arange(n))


def _contact_features(mat: np.ndarray) -> pd.DataFrame:
    n = len(mat)
    total = mat.sum(axis=1)

    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    sep = np.minimum(sep, n - sep)  # circular
    short_mask = sep <= SHORT_RANGE_BINS

    short = (mat * short_mask).sum(axis=1)
    long_ = total - short

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log2((short + 1.0) / (long_ + 1.0))

    # Directional index: contacts to the left versus to the right, the usual
    # way of spotting a domain boundary in a contact map.
    di = np.zeros(n)
    for i in range(n):
        lo = [(i - d) % n for d in range(1, INSULATION_BINS + 1)]
        hi = [(i + d) % n for d in range(1, INSULATION_BINS + 1)]
        a, b = mat[i, lo].sum(), mat[i, hi].sum()
        di[i] = (b - a) / (a + b) if (a + b) else 0.0

    return pd.DataFrame(
        {
            f"{PREFIX}.contacts.total": total,
            f"{PREFIX}.contacts.short": short,
            f"{PREFIX}.contacts.long": long_,
            f"{PREFIX}.contacts.short_long_log2": ratio,
            f"{PREFIX}.insulation": di,
        },
        index=np.arange(n),
    )


def _dependency_features() -> pd.DataFrame | None:
    """log2(WT / mutant) marginal contacts -- how NAP-dependent each bin is."""
    cols = {}
    n = None
    for protein, mutant_file in MUTANT_MATRICES.items():
        mut = load_matrix(mutant_file)
        ref = load_matrix(MUTANT_REFERENCE[protein])
        if mut is None or ref is None or mut.shape != ref.shape:
            continue
        n = len(mut)
        # Normalise each map to the same total so the ratio reflects structure
        # rather than sequencing depth.
        m = mut.sum(axis=1) / max(mut.sum(), 1.0)
        r = ref.sum(axis=1) / max(ref.sum(), 1.0)
        cols[f"{PREFIX}.dep.{protein}"] = np.log2((r + 1e-9) / (m + 1e-9))
    if not cols or n is None:
        return None
    return pd.DataFrame(cols, index=np.arange(n))


def build(index: pd.DataFrame) -> pd.DataFrame:
    xyz = load_coordinates()
    wt = load_matrix(WT_MATRIX)
    if xyz is None and wt is None:
        raise FileNotFoundError(
            f"Neither {config.LIOY_XYZ} nor {config.LIOY_MATRIX_DIR / WT_MATRIX} "
            "exists. Clone koszullab/E_coli_analysis into external_data/."
        )

    per_bin = []
    if xyz is not None:
        per_bin.append(_coordinate_features(xyz))
    if wt is not None:
        per_bin.append(_contact_features(wt))
    dep = _dependency_features()
    if dep is not None:
        per_bin.append(dep)

    n_bins = min(len(p) for p in per_bin)
    table = pd.concat([p.iloc[:n_bins] for p in per_bin], axis=1)

    left = index["left"].to_numpy(dtype="float64")
    valid = np.isfinite(left)
    bins = np.where(
        valid, [(int(p) - 1) // config.LIOY_BIN_SIZE % n_bins if np.isfinite(p) else 0
                for p in left], -1
    ).astype(int)

    out = pd.DataFrame({config.ID_COL: index[config.ID_COL].to_numpy()})
    out[f"{PREFIX}.bin"] = np.where(valid, bins, np.nan)
    for col in table.columns:
        vals = table[col].to_numpy()
        out[col] = np.where(valid, vals[np.clip(bins, 0, n_bins - 1)], np.nan)

    # Where within its 5 kb bin does the guide sit? Cheap, and it lets the
    # model tell "near a boundary" from "in the middle of a domain".
    out[f"{PREFIX}.offset_in_bin"] = np.where(
        valid, (left - 1) % config.LIOY_BIN_SIZE / config.LIOY_BIN_SIZE, np.nan
    )
    return out


def main() -> None:
    from ..io_utils import load_guide_index

    idx = load_guide_index()
    block = build(idx)
    path = save_block(block, "e_nucleoid")
    print(f"e_nucleoid: {block.shape[0]:,} rows x {block.shape[1] - 1:,} features -> {path}")


if __name__ == "__main__":
    main()
