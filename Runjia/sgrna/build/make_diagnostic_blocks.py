"""Decompose the positional families, to find out what they are really measuring.

`d_supercoiling` looked like the project's headline result -- +0.047 R2 under
100 kb-blocked CV, clean permutation control. It is not a supercoiling result.
This script builds the blocks that show why, each one a strict subset or
derivative of features already computed, so no new data is involved:

    z_signal_only     the GapR ChIP track alone            (4 columns)
    z_control_only    the UNTAGGED control track alone     (4 columns)
    z_abundance_all   all four tracks, raw density only     (16 columns)
    z_sc_enrichment   only the ratios -- log2(signal/control), log2(signal/rif),
                      and the across-cut-site gradient. This is the part that
                      actually carries supercoiling information, because the
                      copy-number component divides out.                (10 columns)
    z_dosage          an analytic replication-gradient proxy from oriC distance,
                      with no sequencing data in it at all                (4 columns)

Run them against the same baseline and the attribution falls out: raw read
density reproduces essentially the whole effect, the untagged control predicts
about as well as the ChIP, and the supercoiling-specific ratios collapse to the
noise floor. See results/INTERPRETATION.md section 1.

    nb.make_diagnostic_blocks()
    nb.ablate(["z_abundance_all"], group="arc5", tag="dissect")
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .. import config
from . import genome
from ..io_utils import load_block, load_guide_index, save_block

# Column prefixes follow d_supercoiling's post-dissection naming: eng.sc.abund.*
# is raw read density, eng.sc.torsion.* is the normalised residual.
BLOCKS = {
    "z_signal_only": ("eng.sc.abund.signal",),
    "z_control_only": ("eng.sc.abund.control",),
    "z_abundance_all": ("eng.sc.abund.",),
    "z_sc_enrichment": ("eng.sc.torsion.",),
}


def build_dosage(index: pd.DataFrame | None = None) -> pd.DataFrame:
    """Replication position only -- no sequencing data of any kind."""
    idx = load_guide_index() if index is None else index
    lm = genome.replication_landmarks()
    half = lm["genome_length"] / 2

    out = pd.DataFrame({config.ID_COL: idx[config.ID_COL]})
    out["eng.dose.ori_distance"] = idx["ori_distance"].to_numpy()
    out["eng.dose.ori_fraction"] = idx["ori_distance"].to_numpy() / half
    # Copy number falls roughly geometrically with distance from oriC in
    # exponentially growing cells; this is the idealised version of what the
    # measured read-density tracks capture empirically.
    out["eng.dose.copy_proxy"] = 2.0 ** (-2.0 * out["eng.dose.ori_fraction"])
    out["eng.dose.right_replichore"] = (
        idx["replichore"].map({"right": 1, "left": 0}).to_numpy()
    )
    return out


def build_all(verbose: bool = True) -> dict[str, int]:
    made: dict[str, int] = {}
    sc = load_block("d_supercoiling")
    for name, prefixes in BLOCKS.items():
        cols = [c for c in sc.columns if c.startswith(prefixes)]
        save_block(sc[[config.ID_COL] + cols], name)
        made[name] = len(cols)
        if verbose:
            print(f"  {name:18s} {len(cols):3d} columns")

    dose = build_dosage()
    save_block(dose, "z_dosage")
    made["z_dosage"] = dose.shape[1] - 1
    if verbose:
        print(f"  {'z_dosage':18s} {dose.shape[1] - 1:3d} columns")
    return made


def main() -> int:
    print("Building diagnostic blocks in", config.FEATURE_DIR)
    build_all()
    print(
        "\nNow compare them against the same baseline, e.g.\n"
        "  python -m sgrna.run_ablation --families z_abundance_all "
        "--group arc5 --tag dissect"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Item 4 of the progress report: what *is* the DNA-abundance signal?
#
# `z_dosage` showed an analytic replication gradient is worth +0.013 of
# abundance's +0.045, so replication timing is part of the story but not most
# of it. The analytic proxy is a rigid shape, though (2^-2f), so it understates
# how much of abundance is *any* smooth function of chromosomal position.
#
# This split is the non-parametric version. For each track, the large-scale
# component is the mean read density of the 100 kb bin the guide sits in; the
# local component is what is left after subtracting it. Both are computed from
# read density alone with no reference to the label, so neither leaks.
#
#     z_abund_smooth    the 100 kb-binned positional component
#     z_abund_local     the within-bin residual
#
# Ablated separately, these say whether abundance acts through where the guide
# is on the chromosome (smooth) or through something local to it -- mappability,
# repeats, accessibility (local). 100 kb matches the `bin100k` CV grouping, so
# the smooth component is exactly the part that grouping is designed to hold
# out, which makes the strict-CV comparison interpretable.

SMOOTH_BIN = 100_000


def build_abundance_split(index: pd.DataFrame | None = None,
                          bin_size: int = SMOOTH_BIN,
                          verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split every abundance track into binned-positional and local parts."""
    idx = load_guide_index() if index is None else index
    sc = load_block("d_supercoiling")
    df = idx[[config.ID_COL, "left"]].merge(sc, on=config.ID_COL, how="left")
    tracks = [c for c in df.columns if c.startswith("eng.sc.abund.")]

    bin_id = (df["left"].to_numpy(float) // bin_size)
    smooth = pd.DataFrame({config.ID_COL: df[config.ID_COL]})
    local = pd.DataFrame({config.ID_COL: df[config.ID_COL]})
    for c in tracks:
        stem = c.replace("eng.sc.abund.", "")
        binned = pd.Series(df[c].to_numpy(float)).groupby(bin_id).transform("mean")
        smooth[f"eng.absm.{stem}"] = binned.to_numpy()
        local[f"eng.abloc.{stem}"] = df[c].to_numpy(float) - binned.to_numpy()
    if verbose:
        print(f"  split {len(tracks)} tracks at {bin_size:,} bp bins "
              f"({int(np.nanmax(bin_id)) + 1} bins)")
    return smooth, local


# `z_mapgc`: the abundance signal's two sequence-intrinsic candidate causes,
# computed from the reference genome alone -- no sequencing data at all.
#
#   uniq.w*   fraction of positions in the window whose canonical 25-mer is
#             unique in the genome. Low values mark repeats, where short reads
#             map ambiguously and depth is unreliable (a mappability artifact
#             rather than biology).
#   gc.w*     GC fraction of the window. Library prep and sequencing are
#             GC-biased, so depth falls in GC-rich DNA.
#
# Measured against `z_abund_local`, these two explain most of it: local read
# density tracks window-matched GC at r = -0.3 to -0.70, and the untagged
# control's 10 kb density tracks 25-mer uniqueness at -0.69. If this block
# reproduces abundance's ablation gain, the effect needs no ChIP data.

MAPGC_K = 25
MAPGC_WINDOWS = (100, 500, 2000, 10000)


def _uniqueness_track(seq: str, k: int = MAPGC_K) -> np.ndarray:
    """1.0 where the canonical k-mer starting here occurs once in the genome."""
    code = np.full(256, 255, np.uint8)
    for i, b in enumerate("ACGT"):
        code[ord(b)] = i
    arr = code[np.frombuffer(seq.upper().encode(), np.uint8)]
    n = len(arr)
    ok = arr < 4
    packed = np.where(ok, arr, 0).astype(np.uint64)
    comp = np.uint64(3) - packed

    fwd = np.zeros(n - k + 1, np.uint64)
    rc = np.zeros(n - k + 1, np.uint64)
    for i in range(k):
        fwd = (fwd << np.uint64(2)) | packed[i : n - k + 1 + i]
        rc = (rc << np.uint64(2)) | comp[k - 1 - i : n - k + 1 + k - 1 - i]
    canon = np.minimum(fwd, rc)
    valid = np.lib.stride_tricks.sliding_window_view(ok, k).all(axis=1)

    _, inv, cnt = np.unique(canon[valid], return_inverse=True, return_counts=True)
    mult = np.ones(len(canon), np.int32)
    mult[valid] = cnt[inv]
    track = np.zeros(n, float)
    track[: len(mult)] = (mult == 1).astype(float)
    return track


def _window_mean(track: np.ndarray, centres: np.ndarray, w: int) -> np.ndarray:
    cum = np.r_[0.0, np.cumsum(track)]
    lo = np.clip(centres - w // 2, 0, len(track))
    hi = np.clip(centres + w // 2, 0, len(track))
    return (cum[hi] - cum[lo]) / np.maximum(hi - lo, 1)


def build_mapgc(index: pd.DataFrame | None = None, verbose: bool = True) -> pd.DataFrame:
    """Mappability and GC around each guide, from the genome only."""
    idx = load_guide_index() if index is None else index
    seq = genome.load_genome().upper()
    uniq = _uniqueness_track(seq)
    gc = np.isin(
        np.frombuffer(seq.encode(), np.uint8), [ord("G"), ord("C")]
    ).astype(float)
    if verbose:
        print(f"  {MAPGC_K}-mers unique at {uniq.mean() * 100:.2f}% of positions")

    pos = idx["left"].to_numpy(float)
    known = np.isfinite(pos)
    centres = np.where(known, pos, 0).astype(int)

    out = pd.DataFrame({config.ID_COL: idx[config.ID_COL]})
    for w in MAPGC_WINDOWS:
        out[f"eng.mapgc.uniq.w{w}"] = np.where(
            known, _window_mean(uniq, centres, w), np.nan)
        out[f"eng.mapgc.gc.w{w}"] = np.where(
            known, _window_mean(gc, centres, w), np.nan)
    return out
