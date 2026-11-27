"""Reference genome helpers for E. coli K-12 MG1655 (NC_000913.2).

Everything here treats the chromosome as circular, because it is: a guide
near coordinate 1 has a perfectly ordinary upstream flank, it just wraps.

Coordinate convention used throughout the project
-------------------------------------------------
Positions are 1-based and inclusive, the way genome browsers and RegulonDB
report them. Internally we index a Python string, so the translation is
always `genome[start - 1 : end]`.

A guide record means:
    start, end        1-based inclusive span of the 20 nt protospacer
    strand            '+' if the protospacer matches the plus strand as given
    pam               the 3 nt immediately 3' of the protospacer, same strand
"""

from __future__ import annotations

import functools

import numpy as np

from .. import config

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


@functools.lru_cache(maxsize=1)
def load_genome(path=None) -> str:
    """Return the chromosome as one upper-case string."""
    path = path or config.GENOME_FASTA
    chunks = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            chunks.append(line.strip())
    seq = "".join(chunks).upper()
    if not seq:
        raise ValueError(f"No sequence found in {path}")
    return seq


def genome_length() -> int:
    return len(load_genome())


# --------------------------------------------------------------------------
# Circular slicing
# --------------------------------------------------------------------------


def slice_circular(start: int, end: int, genome: str | None = None) -> str:
    """1-based inclusive slice that wraps around the origin if needed."""
    g = genome if genome is not None else load_genome()
    n = len(g)
    start0 = (start - 1) % n
    end0 = end % n
    if start0 < end0:
        return g[start0:end0]
    return g[start0:] + g[:end0]


def circular_distance(a: int, b: int, n: int | None = None) -> int:
    """Shortest distance between two 1-based positions on a circle."""
    n = n or genome_length()
    d = abs(a - b) % n
    return min(d, n - d)


# --------------------------------------------------------------------------
# Locating protospacers
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def protospacer_sites(length: int = 20) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Every NGG-flanked protospacer in the genome, both strands.

    Scanning the genome once and indexing the result is what makes step 0 and
    the off-target counts practical: naive `str.find` per guide would re-scan
    4.6 Mb tens of thousands of times.

    Returns (sequences, left_positions, strands) where `left` is the 1-based
    genomic left edge of the protospacer and `strands` is +1 / -1. Sequences
    are given in guide orientation.
    """
    g = load_genome()
    n = len(g)
    seqs: list[str] = []
    lefts: list[int] = []
    strands: list[int] = []

    # Plus strand: ...[20 nt protospacer][N][G][G]...
    i = g.find("GG")
    while i >= 0:
        pam_start0 = i - 1               # the N of NGG
        proto_start0 = pam_start0 - length
        if proto_start0 >= 0:
            seqs.append(g[proto_start0 : proto_start0 + length])
            lefts.append(proto_start0 + 1)
            strands.append(1)
        i = g.find("GG", i + 1)

    # Minus strand: a CC on the plus strand reads GG on the minus strand, so
    # the protospacer lies immediately 3' of it in plus-strand coordinates.
    i = g.find("CC")
    while i >= 0:
        proto_start0 = i + 3             # skip CCN (= NGG read on the minus strand)
        if proto_start0 + length <= n:
            seqs.append(revcomp(g[proto_start0 : proto_start0 + length]))
            lefts.append(proto_start0 + 1)
            strands.append(-1)
        i = g.find("CC", i + 1)

    return seqs, np.asarray(lefts, dtype=np.int64), np.asarray(strands, dtype=np.int8)


@functools.lru_cache(maxsize=2)
def protospacer_index(length: int = 20) -> dict[str, list[int]]:
    """Map each NGG-flanked protospacer to the row indices of its occurrences."""
    seqs, _, _ = protospacer_sites(length)
    idx: dict[str, list[int]] = {}
    for k, s in enumerate(seqs):
        idx.setdefault(s, []).append(k)
    return idx


def locate_protospacer(proto: str, genome: str | None = None) -> list[dict]:
    """Find every NGG-flanked genomic copy of a protospacer, on either strand.

    Returns one dict per hit. `left`/`right` are plus-strand genomic bounds of
    the 20 nt protospacer; `start`/`end` are the same span in guide
    orientation; `pam` is read on the guide's own strand.
    """
    g = genome if genome is not None else load_genome()
    n = len(g)
    seqs, lefts, strands = protospacer_sites(len(proto))
    index = protospacer_index(len(proto))

    hits: list[dict] = []
    for k in index.get(proto, ()):
        left = int(lefts[k])
        right = left + len(proto) - 1
        if strands[k] == 1:
            pam = slice_circular(right + 1, right + 3, g)
            hits.append(
                dict(strand="+", left=left, right=right, start=left, end=right,
                     pam=pam)
            )
        else:
            pam = revcomp(slice_circular(left - 3, left - 1, g))
            hits.append(
                dict(strand="-", left=left, right=right, start=right, end=left,
                     pam=pam)
            )

    for h in hits:
        h["n_genome_copies"] = len(hits)
        h["pos"] = h["left"]
        h["genome_length"] = n
    return hits


def guide_context(left: int, right: int, strand: str, flank: int,
                  genome: str | None = None) -> dict:
    """Sequence around a protospacer, returned in guide orientation.

    upstream   `flank` nt 5' of protospacer position 1
    protospacer
    pam        3 nt
    downstream `flank` nt 3' of the PAM
    """
    g = genome if genome is not None else load_genome()
    if strand == "+":
        upstream = slice_circular(left - flank, left - 1, g)
        proto = slice_circular(left, right, g)
        pam = slice_circular(right + 1, right + 3, g)
        downstream = slice_circular(right + 4, right + 3 + flank, g)
    else:
        upstream = revcomp(slice_circular(right + 1, right + flank, g))
        proto = revcomp(slice_circular(left, right, g))
        pam = revcomp(slice_circular(left - 3, left - 1, g))
        downstream = revcomp(slice_circular(left - 3 - flank, left - 4, g))
    return dict(upstream=upstream, protospacer=proto, pam=pam, downstream=downstream)


# --------------------------------------------------------------------------
# Replication origin and terminus from cumulative GC skew
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def replication_landmarks() -> dict:
    """Locate oriC and the terminus from the cumulative GC skew.

    The leading strand is G-rich, so (G - C) summed along the chromosome falls
    to a minimum at the origin and rises to a maximum at the terminus. This is
    self-contained -- no annotation file needed -- and for MG1655 it lands
    within a few kb of the annotated oriC (~3.92 Mb), which is the check
    printed by `build_guide_index.py`.
    """
    g = load_genome()
    arr = np.frombuffer(g.encode(), dtype="S1")
    skew = np.where(arr == b"G", 1, np.where(arr == b"C", -1, 0)).cumsum()
    ori = int(np.argmin(skew)) + 1
    ter = int(np.argmax(skew)) + 1
    return {"ori": ori, "ter": ter, "genome_length": len(g)}


def replichore(pos: int) -> str:
    """'right' (ori -> ter clockwise) or 'left' replichore."""
    lm = replication_landmarks()
    n = lm["genome_length"]
    d_ori = (pos - lm["ori"]) % n
    d_ter = (lm["ter"] - lm["ori"]) % n
    return "right" if d_ori < d_ter else "left"


def replication_fork_direction(pos: int) -> int:
    """+1 if replication moves along the plus strand here, -1 otherwise.

    On the replichore that runs clockwise from oriC, the plus strand is the
    leading-strand template; on the other replichore it is the lagging one.
    """
    return 1 if replichore(pos) == "right" else -1
