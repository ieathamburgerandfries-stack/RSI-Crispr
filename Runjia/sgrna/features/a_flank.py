"""Family A -- sequence and quantum descriptors outside the 20 nt protospacer.

Why this one first
------------------
The published matrix stops at the protospacer. Curated re-analysis of this
very screen (Browne et al. 2025, PeerJ) found that widening the input window
raised Spearman by 0.143 -- more than any feature family reported anywhere for
bacterial Cas9 -- with the strongest contribution from roughly the first 11
nucleotides 3' of the PAM, and a broader AT-richness effect extending past
100 nt on both sides. crisprHAL, the strongest bacterial model published,
feeds on 28 nt for the same reason.

What makes this more than one-hot encoding
------------------------------------------
`qct.learn_lookup_tables` recovers the paper's own quantum chemical tensors as
exact k-mer -> value tables (all 4/16/64/256 k-mers are present in the 13,880
guides, so the recovery is complete, not approximate). That means the same
quantum parameterisation Noshay et al. applied to positions 1-20 can be
applied to the PAM, to the nucleotides downstream of it, and to the flanking
DNA -- sequence the original encoding never touched. The claim moves from
"quantum descriptors of the protospacer" to "quantum descriptors of the
target locus", without introducing a single new assumption.

Two changes after the first ablation
------------------------------------
That first pass gave +0.083 R2, of which 62% came from the wide-flank window
summaries -- and those summaries turned out to be **almost exactly flank GC
content** (rho 0.998 between `win.up250.trimer.Hbondenergy.mean` and
`comp.up250.gc`). Averaging a per-position tensor over hundreds of bases
collapses it onto base composition, so emitting all seventeen unit/property
combinations per window was seventeen near-copies of one number.

So: the summary block now carries a small, deliberately spread set of four
tensor series plus explicit composition, and the windows reach **1 kb** rather
than 250 nt, because the effect showed no sign of saturating at the old
boundary. Fewer columns, four times the reach.

Columns
-------
eng.flank.seq.*    one-hot base identity at u10..u1, the PAM's N, d1..d15
eng.flank.qct.*    per-position monomer and dimer tensors at those positions
eng.flank.win.*    mean/sd of four spread tensor series at 50/250/500/1000 nt
eng.flank.comp.*   GC fraction, purine fraction, longest homopolymer run
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..build import qct
from ..io_utils import save_block

PREFIX = "eng.flank"

# Positions encoded one base at a time. u1 is the nucleotide immediately 5' of
# protospacer position 1; d1 is the first nucleotide 3' of the NGG PAM.
N_UPSTREAM = config.UPSTREAM_POSITIONS      # 10
N_DOWNSTREAM = config.DOWNSTREAM_POSITIONS  # 15

# Per-position tensors. Restricted to the two shortest molecular units, which
# keeps the block manageable; the longer units enter through the summaries.
PER_POSITION_UNITS = ("monomer", "dimer")

# Windows (nt) used for the summary statistics, out to 1 kb.
SUMMARY_WINDOWS = config.FLANK_WINDOWS

# Four tensor series chosen to span the descriptor space -- one per molecular
# unit, and not all the same physical property. Averaging any of these over a
# long window partly collapses onto composition, which is why `comp.*` is
# emitted alongside: it makes the redundancy visible instead of hiding it in
# seventeen correlated columns.
SUMMARY_SERIES = (
    ("monomer", "HLgap.eVraw"),
    ("dimer", "Hbond.stackingraw"),
    ("trimer", "Hbond.energyraw"),
    ("tetramer", "Hbond.stackingraw"),
)

BASES = ("A", "C", "G", "T")


def _positions() -> list[str]:
    """Ordered labels of the single-base positions, 5' -> 3'."""
    up = [f"u{i}" for i in range(N_UPSTREAM, 0, -1)]
    pam = ["pamN"]  # positions 2 and 3 of an NGG PAM are constant, so skipped
    down = [f"d{i}" for i in range(1, N_DOWNSTREAM + 1)]
    return up + pam + down


def _extended_sequence(row) -> tuple[str, dict[str, int]]:
    """Concatenate the local context and record where each label sits in it.

    Returned indices are 0-based offsets into the returned string. We keep a
    little extra sequence on both ends so that dimer/trimer/tetramer windows
    starting at the outermost labelled position still have sequence to read.
    """
    pad = 4
    up = str(row["upstream"])
    down = str(row["downstream"])
    proto = str(row["protospacer"])
    pam = str(row["pam"])

    left = up[-(N_UPSTREAM + pad):] if len(up) >= N_UPSTREAM + pad else up
    right = down[: N_DOWNSTREAM + pad]
    ext = left + proto + pam + right

    idx: dict[str, int] = {}
    proto_start = len(left)
    for i in range(N_UPSTREAM, 0, -1):
        idx[f"u{i}"] = proto_start - i
    idx["pamN"] = proto_start + len(proto)
    for i in range(1, N_DOWNSTREAM + 1):
        idx[f"d{i}"] = proto_start + len(proto) + 3 + (i - 1)
    return ext, idx


def _longest_run(seq: str) -> int:
    best = run = 0
    prev = ""
    for ch in seq:
        run = run + 1 if ch == prev else 1
        prev = ch
        best = max(best, run)
    return best


def _composition(seq: str, prefix: str) -> dict[str, float]:
    if not seq:
        return {f"{prefix}.{k}": np.nan for k in ("gc", "purine", "maxrun")}
    n = len(seq)
    gc = sum(seq.count(b) for b in "GC") / n
    pur = sum(seq.count(b) for b in "AG") / n
    return {
        f"{prefix}.gc": gc,
        f"{prefix}.purine": pur,
        f"{prefix}.maxrun": _longest_run(seq) / n,
    }


def _summarise_window(values: np.ndarray, key: str) -> dict[str, float]:
    if values.size == 0 or np.all(np.isnan(values)):
        return {f"{key}.mean": np.nan, f"{key}.sd": np.nan}
    return {
        f"{key}.mean": float(np.nanmean(values)),
        f"{key}.sd": float(np.nanstd(values)),
    }


def build(index: pd.DataFrame, tables: dict | None = None) -> pd.DataFrame:
    tables = tables or qct.load_tables()
    labels = _positions()
    widest = max(SUMMARY_WINDOWS)

    per_pos_pairs = [
        (unit, prop)
        for unit, props in tables.items()
        for prop in props
        if unit in PER_POSITION_UNITS
    ]

    rows = []
    for _, row in index.iterrows():
        rec: dict[str, float] = {config.ID_COL: row[config.ID_COL]}

        if not isinstance(row.get("protospacer"), str) or not row.get("located", False):
            rows.append(rec)  # everything stays NaN; imputed downstream
            continue

        ext, idx = _extended_sequence(row)

        # ---- one-hot base identity -------------------------------------
        for label in labels:
            pos = idx[label]
            base = ext[pos] if 0 <= pos < len(ext) else "N"
            for b in BASES:
                rec[f"{PREFIX}.seq.{label}.{b}"] = int(base == b)

        # ---- per-position quantum tensors ------------------------------
        for unit, prop in per_pos_pairs:
            span = qct.UNIT_SPAN[unit]
            table = tables[unit][prop]
            short = prop.replace(".", "").replace("raw", "")
            for label in labels:
                pos = idx[label]
                kmer = ext[pos : pos + span]
                rec[f"{PREFIX}.qct.{unit}.{short}.{label}"] = (
                    table.get(kmer, np.nan) if len(kmer) == span else np.nan
                )

        # ---- window summaries over the wider flanks --------------------
        #
        # The windows are nested, so each tensor profile is computed once over
        # the widest segment and then sliced. Recomputing it per window would
        # quadruple the cost of the whole family for no new information.
        up_full = str(row["upstream"])[-widest:]
        down_full = str(row["downstream"])[:widest]

        for side, seg in (("up", up_full), ("dn", down_full)):
            profiles = {
                (unit, prop): qct.profile(seg, unit, prop, tables)
                for unit, prop in SUMMARY_SERIES
            }
            for w in SUMMARY_WINDOWS:
                sub = seg[-w:] if side == "up" else seg[:w]
                rec.update(_composition(sub, f"{PREFIX}.comp.{side}{w}"))
                for (unit, prop), prof in profiles.items():
                    short = prop.replace(".", "").replace("raw", "")
                    # upstream windows are suffixes of the segment, downstream
                    # windows are prefixes -- slice the profile the same way
                    part = prof[-w:] if side == "up" else prof[:w]
                    rec.update(
                        _summarise_window(
                            part, f"{PREFIX}.win.{side}{w}.{unit}.{short}"
                        )
                    )
        rows.append(rec)

    return pd.DataFrame(rows)


def main() -> None:
    from ..io_utils import load_guide_index

    idx = load_guide_index()
    block = build(idx)
    path = save_block(block, "a_flank")
    n_seq = sum(c.startswith(f"{PREFIX}.seq") for c in block.columns)
    n_qct = sum(c.startswith(f"{PREFIX}.qct") for c in block.columns)
    n_win = sum(c.startswith((f"{PREFIX}.win", f"{PREFIX}.comp"))
                for c in block.columns)
    print(f"a_flank: {block.shape[0]:,} rows x {block.shape[1] - 1:,} features "
          f"({n_seq} one-hot, {n_qct} per-position tensors, {n_win} window "
          f"summaries out to {max(SUMMARY_WINDOWS)} nt) -> {path}")


if __name__ == "__main__":
    main()
