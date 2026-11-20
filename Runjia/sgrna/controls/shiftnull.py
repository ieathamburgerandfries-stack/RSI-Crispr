"""A spatially preserving null for family A: shift the chromosome, keep its texture.

The permutation control used everywhere else shuffles the flank block's rows.
That destroys **genomic autocorrelation** along with the pairing, so it tests a
weaker null than the report claimed: shuffled flanks are not merely misassigned,
they are no longer smooth, and a model could in principle be rejecting them for
the wrong reason.

The sharper null keeps every statistical property of the flank track and breaks
only the correspondence to the guide. Each guide is re-assigned the flanks of a
genomic position **offset by a fixed circular shift** around the chromosome:
composition, smoothness, long-range autocorrelation and the distribution of every
column are preserved exactly, because the same track is being read -- just from
the wrong place.

**One shift is not enough, and the reason is the thing being tested.** GC content
on this chromosome is autocorrelated over tens of kilobases, so a 1 kb shift
hands each guide almost the correct flanks and would retain most of the gain --
not a null at all. So the shift is swept, which turns a control into a
measurement: **Delta rho against shift distance is the autocorrelation length of
the effect**, independently of the window sizes family A happens to use. A real
local effect decays to zero as the shift exceeds its correlation length; an
artefact of large-scale chromosome structure does not.

Grouped CV throughout (100 kb blocks), because a positional feature under random
KFold can score by locus recognition -- the shifted arms would inherit that too.

    python -m sgrna.shiftnull --run
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config

OUT = config.RESULTS / "shift_null.csv"
SHIFTS = (0, 1_000, 10_000, 100_000, 500_000, 1_000_000, 2_000_000)


def _shifted_block(block: pd.DataFrame, index: pd.DataFrame, shift: int,
                   genome_len: int) -> pd.DataFrame:
    """Re-key the flank block so each guide reads the track `shift` bp away.

    Implemented as a nearest-neighbour lookup in genomic coordinate: for guide at
    position p, take the flank row of whichever guide sits closest to
    (p + shift) mod L. That reuses real rows, so every column keeps its exact
    marginal distribution -- unlike a shuffle, which keeps the marginals but
    destroys the spatial structure, and unlike interpolation, which would invent
    values.
    """
    pos = index["left"].to_numpy(dtype=float)
    order = np.argsort(pos)
    sorted_pos = pos[order]
    target = (pos + shift) % genome_len
    j = np.searchsorted(sorted_pos, target)
    j = np.clip(j, 0, len(sorted_pos) - 1)
    # pick the nearer of the two bracketing guides, wrapping at the ends
    left = np.clip(j - 1, 0, len(sorted_pos) - 1)
    pick = np.where(np.abs(sorted_pos[j] - target)
                    <= np.abs(sorted_pos[left] - target), j, left)
    src = order[pick]
    out = block.iloc[src].reset_index(drop=True)
    out[config.ID_COL] = index[config.ID_COL].to_numpy()
    return out


def run(shifts=SHIFTS, seeds=(41, 42, 43), n_splits: int = 5,
        verbose: bool = True) -> pd.DataFrame:
    from ..build.build_matrix import build_cache, load_dataset
    from ..io_utils import load_block, load_guide_index
    from ..harness.run_ablation import cross_validate, load_groups

    X_base, y, names, ids, fam = load_dataset([], verbose=verbose)
    idx = load_guide_index()
    idx = idx[idx["located"].astype(bool)].drop_duplicates(config.ID_COL)
    idx = idx.set_index(config.ID_COL).reindex(ids).reset_index()
    genome_len = int(idx["genome_length"].iloc[0]) if "genome_length" in idx \
        else 4_639_675
    block = (load_block("a_flank").set_index(config.ID_COL)
             .reindex(ids).reset_index())
    flank_cols = [c for c in block.columns if c != config.ID_COL]
    groups = load_groups(ids, by="bin100k")

    rows = []
    for shift in shifts:
        b = block if shift == 0 else _shifted_block(block, idx, shift, genome_len)
        vals = b[flank_cols].to_numpy(dtype=np.float32)
        keep = ~np.all(np.isnan(vals), axis=0)
        X = np.hstack([X_base, vals[:, keep]])
        fam_of = np.asarray(list(fam) + ["a_flank"] * int(keep.sum()))
        res = cross_validate(X, y, names + [c for c, k in zip(flank_cols, keep) if k],
                             fam_of, seeds=seeds, n_splits=n_splits, groups=groups,
                             verbose=False, tag=f"shift{shift}",
                             experiment=f"shift_{shift}")
        s = res["summary"]
        rows.append(dict(shift_bp=shift, n_flank_cols=int(keep.sum()),
                         spearman=s["spearman"]["mean"],
                         spearman_sd=s["spearman"]["sd"],
                         r2=s["r2"]["mean"]))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        if verbose:
            base = rows[0]["spearman"]
            print(f"  shift {shift:>9,} bp: rho {s['spearman']['mean']:.4f}"
                  + (f"   retained {100*(s['spearman']['mean']-rows[-1]['spearman']*0):.0f}%"
                     if shift == 0 else
                     f"   vs aligned {s['spearman']['mean']-base:+.4f}"),
                  flush=True)
    return pd.DataFrame(rows)


def summary():
    d = pd.read_csv(OUT).sort_values("shift_bp")
    base_row = d[d.shift_bp == 0]
    if len(base_row):
        aligned = float(base_row["spearman"].iloc[0])
        d["delta_vs_aligned"] = (d["spearman"] - aligned).round(4)
    print(d.round(4).to_string(index=False))
    print("\nshift 0 is the real, aligned flanks; every other row keeps the same")
    print("track and reads it from the wrong place.")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.run:
        run(seeds=tuple(a.seeds) if a.seeds else (41, 42, 43))
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
