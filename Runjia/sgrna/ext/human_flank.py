"""Human guides with their genomic flanks, for running Finding 8 (shape / 3D versus family A) on human.

The project's human matrix (Noshay et al. 2023, Supplementary Table 3, 17,422 guides) describes only the 20-nt
protospacer. The protospacer is decoded from the monomer electron-count columns (C 42, T 48, A 50, G 56), then located
on hg38 as an NGG-flanked 20-mer with exactly one NGG-flanked copy in the primary assembly (chr1-22, X, Y, M);
non-NGG copies are ignored, as in the E. coli curation. 1,000 nt are cut on each side in the guide's orientation, giving
the same table layout as `curated_guide_index.csv` (upstream / protospacer / pam / downstream).

The scan encodes each chromosome as 2-bit codes and tests every 20-mer against the wanted set with numpy, so the whole
genome takes minutes rather than a Python loop over 4e8 positions.

    python -m sgrna.ext.human_flank
"""
from __future__ import annotations

import gzip
import json
import time

import numpy as np
import pandas as pd

from sgrna import config, genome

from .common import EXTERNAL, INTERIM

HG38 = EXTERNAL / "hg38" / "hg38.fa.gz"
DIR = INTERIM / "human"
DIR.mkdir(parents=True, exist_ok=True)
FLANK = 1000
ELECTRONS_TO_BASE = {42.0: "C", 48.0: "T", 50.0: "A", 56.0: "G"}
PRIMARY = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}
LUT = np.full(256, 4, np.uint8)
for _i, _c in enumerate("ACGT"):
    LUT[ord(_c)] = _i
    LUT[ord(_c.lower())] = _i
REV = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def revcomp(s: str) -> str:
    return "".join(REV.get(c, "N") for c in reversed(s))


def decode() -> pd.DataFrame:
    path = DIR / "decoded.csv"
    if path.exists():
        return pd.read_csv(path)
    cols = [config.ID_COL, config.TARGET_COL] + [f"p{i}monomer.No.electronsraw" for i in range(1, 21)]
    df = pd.read_csv(config.RAW / "human_feature_matrix.csv", header=config.BASE_MATRIX_HEADER_ROW, usecols=cols,
                     low_memory=False)
    mat = df[[f"p{i}monomer.No.electronsraw" for i in range(1, 21)]].to_numpy(float)
    seqs = ["".join(ELECTRONS_TO_BASE.get(v, "N") for v in row) for row in mat]
    out = pd.DataFrame({config.ID_COL: df[config.ID_COL].astype(str),
                        "y": pd.to_numeric(df[config.TARGET_COL], errors="coerce"), "protospacer": seqs})
    out["decoded_ok"] = ~out.protospacer.str.contains("N")
    out.to_csv(path, index=False)
    return out


def _fasta(path):
    name, chunks = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.strip())
        if name is not None:
            yield name, "".join(chunks)


def _codes(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """20-mer codes (2 bits per base, first base most significant) for every start, and a validity mask."""
    n = len(arr) - 19
    code = np.zeros(n, np.uint64)
    bad = np.zeros(n, bool)
    for k in range(20):
        a = arr[k:k + n]
        bad |= a > 3
        code = (code << np.uint64(2)) | (a & 3).astype(np.uint64)
    return code, ~bad


def encode20(s: str) -> int:
    v = 0
    for ch in s:
        v = (v << 2) | "ACGT".index(ch)
    return v


def slice_linear(seq: str, a: int, b: int) -> str:
    """1-based inclusive slice of the chromosome, padded with N outside it."""
    length, n = b - a + 1, len(seq)
    left = max(0, min(length, 1 - a))
    right = max(0, min(length - left, b - n))
    mid = length - left - right
    start = max(a, 1)
    return "N" * left + seq[start - 1:start - 1 + mid].upper() + "N" * right


def context(seq: str, strand: str, left: int, right: int) -> dict:
    """Upstream / protospacer / PAM / downstream in the guide's orientation (1-based, inclusive)."""
    if strand == "+":
        return dict(upstream=slice_linear(seq, left - FLANK, left - 1), protospacer=slice_linear(seq, left, right),
                    pam=slice_linear(seq, right + 1, right + 3),
                    downstream=slice_linear(seq, right + 4, right + 3 + FLANK))
    return dict(upstream=revcomp(slice_linear(seq, right + 1, right + FLANK)),
                protospacer=revcomp(slice_linear(seq, left, right)),
                pam=revcomp(slice_linear(seq, left - 3, left - 1)),
                downstream=revcomp(slice_linear(seq, left - 3 - FLANK, left - 4)))


CHUNK = 20_000_000


def locate(force=False) -> pd.DataFrame:
    path = DIR / "located.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    dec = decode()
    ok = dec[dec.decoded_ok]
    wanted = set(ok.protospacer)
    fwd = {encode20(p): p for p in wanted}
    rcm = {encode20(revcomp(p)): p for p in wanted}
    fkeys, rkeys = np.array(sorted(fwd), np.uint64), np.array(sorted(rcm), np.uint64)
    hits: dict[str, list[dict]] = {}
    t0 = time.time()
    for chrom, seq in _fasta(HG38):
        if chrom not in PRIMARY:
            continue
        arr = LUT[np.frombuffer(seq.encode("ascii"), np.uint8)]
        found = []                                            # (protospacer, strand, 0-based start)
        for c0 in range(0, len(arr) - 19, CHUNK):
            sub = arr[c0:c0 + CHUNK + 19]
            code, valid = _codes(sub)
            for strand, keys, table in (("+", fkeys, fwd), ("-", rkeys, rcm)):
                for j in np.nonzero(valid & np.isin(code, keys))[0]:
                    g = c0 + int(j)
                    if strand == "+":
                        if g + 22 < len(arr) and arr[g + 21] == 2 and arr[g + 22] == 2:    # PAM N G G
                            found.append((table[int(code[j])], "+", g))
                    elif g >= 3 and arr[g - 3] == 1 and arr[g - 2] == 1:                    # C C N before it
                        found.append((table[int(code[j])], "-", g))
        for proto, strand, g in found:
            left, right = g + 1, g + 20
            hits.setdefault(proto, []).append(dict(chrom=chrom, strand=strand, left=left, right=right,
                                                   **context(seq, strand, left, right)))
        print(f"  {chrom:6s} {len(seq):>11,} bp  {len(found):5d} NGG hits  {time.time() - t0:5.0f}s", flush=True)
        del seq, arr
    rows, n_multi, n_none = [], 0, 0
    for _, r in ok.iterrows():
        hs = hits.get(r.protospacer, [])
        if not hs:
            n_none += 1
        elif len(hs) > 1:
            n_multi += 1
        else:
            rows.append(dict(sgRNAID=r[config.ID_COL], y=r.y, **hs[0]))
    out = pd.DataFrame(rows)
    assert (out.protospacer.str.len() == 20).all() and out.pam.str[1:].eq("GG").all()
    out.to_csv(path, index=False)
    (DIR / "locate_audit.json").write_text(json.dumps(dict(n_guides=len(dec), n_decoded=int(len(ok)),
                                                           n_unique=len(out), n_multi=n_multi, n_none=n_none)))
    return out


if __name__ == "__main__":
    out = locate()
    print(len(out), "located;", (DIR / "locate_audit.json").read_text())
