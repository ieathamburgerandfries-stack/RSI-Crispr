"""Family A for any genome: a FASTA file in, the 348 flank columns of every NGG guide out.

Family A is the block of columns that describe the DNA around a guide's target: one-hot bases and quantum-chemical values at 26 positions (the 10 nt before the
protospacer, the PAM's N, the 15 nt after the PAM), and summaries (GC, purine fraction, longest homopolymer run, mean and spread of four tensor series) of the 1,000 nt on each
side in windows of 50, 250, 500 and 1,000 nt (`sgrna/features/a_flank.py` defines them). The released table of the 33,567 curated *E. coli* guides is one instance; this
makes the same table for any genome, with the same column names and order.

    python -m sgrna.design.family_a_table --genome genome.fasta --out family_A_genome.csv.gz
    python -m sgrna.design.family_a_table --genome genome.fasta --contig NC_000913.2 --start 1000000 --end 1010000 --out locus.csv
    python -m sgrna.design.family_a_table --genome genome.fasta --guides guides.csv --out table.csv     # only the guides listed (columns: left, strand; contig if several)
    python -m sgrna.design.family_a_table --genome hg38.fa.gz --contig chr21 --linear --start 20000000 --end 20050000 --out locus.csv     # a mammalian chromosome: one record, linear, one stretch

Every guide is a 20-nt protospacer followed by NGG, on either strand; `left` is the 1-based plus-strand coordinate of the protospacer's first base and `right = left + 19`,
as in the released `guide_flanks.csv`. Each row also carries the protospacer and PAM read on the guide's own strand and `n_genome_copies`, the number of NGG sites in the
records read (the whole FASTA unless `--contig` is given, both strands) with that exact 20-mer. The 348 columns come after those seven.

Sequences are circular by default, as a bacterial chromosome is: flanks wrap around the origin. With `--linear` a guide within about 1 kb of a record's end uses the DNA that exists
(its windows are cut to what is there) and is computed one guide at a time; every other guide is computed exactly as in the circular case. Records shorter than 2,023 nt are skipped
(their two flanks would overlap the guide). Any character other than A, C, G, T (N, ambiguity codes; lower case is upper-cased) is read as N: allowed in the flanks, where the columns that read it are left empty, and a guide whose
protospacer or PAM's N holds one is not a guide and is dropped. A record is read, scanned and featurised on its own, so a mammalian chromosome (hundreds of Mb) needs a few GB; name it with
`--contig` (the other records are skipped as the file is read) and write a stretch of it with `--start/--end`, because a whole chromosome has tens of millions of sites.

The lookup tables that turn a k-mer into its quantum-chemical value are `models/resources/qct_lookup_tables.json` (the values published by Noshay et al. 2023; the release holds the
same file). Only numpy and pandas are needed. The E. coli genome (4.6 Mb) has 541,810 sites and takes about a minute; its table is 1.25 GB as plain CSV and 246 MB as `.csv.gz`, so write `.csv.gz`.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

from . import design_guide as D          # first: it gives `sgrna.config` the project root it insists on

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..build import a_flank_fast as AF  # noqa: E402
from ..build import flank_locus as FL  # noqa: E402

FLANK = D.FLANK
MIN_LENGTH = 2 * FLANK + 23
TABLES = D.MODELS / "resources" / "qct_lookup_tables.json"
CHUNK = 100_000
ID_COLUMNS = ["contig", "left", "right", "strand", "protospacer", "pam", "n_genome_copies"]
_LETTERS = np.frombuffer(b"ACGT", np.uint8)
_LETTERS_N = np.frombuffer(b"ACGTN", np.uint8)


# ---------------------------------------------------------------------------------------------------------------- sites
def pam_codes(codes: np.ndarray, left0, plus) -> np.ndarray:
    """(n, 3) base codes of each guide's PAM on the guide's own strand (N, then the two G)."""
    left0 = np.asarray(left0, np.int64)[:, None]
    plus = np.asarray(plus, bool)[:, None]
    j = np.arange(3)[None, :]
    pos = np.where(plus, left0 + 20 + j, left0 - 1 - j) % len(codes)
    c = codes[pos].astype(np.int16)
    return np.where(plus | (c > 3), c, 3 - c).astype(np.uint8)


def to_strings(c: np.ndarray) -> np.ndarray:
    """(n, k) base codes -> n strings of k letters."""
    return _LETTERS[c].view(f"S{c.shape[1]}").ravel().astype(str)


def spacer_keys(pc: np.ndarray) -> np.ndarray:
    """One integer per 20-mer (two bits per base) from (n, 20) base codes below 4."""
    k = np.zeros(len(pc), np.uint64)
    for j in range(pc.shape[1]):
        k = (k << np.uint64(2)) | pc[:, j].astype(np.uint64)
    return k


def scan(codes: np.ndarray, circular: bool = True, block: int = 4_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(plus-strand 0-based protospacer starts, +1/-1 strands, 20-mer keys) of every complete NGG guide of one record, in coordinate order.

    Complete: the 20-mer and the PAM lie inside the record (always true when circular), and the protospacer and the PAM's N are A, C, G or T. The record is scanned a
    block at a time, so the working memory is a few hundred MB on a chromosome of hundreds of Mb.
    """
    n = len(codes)
    parts = []
    for lo in range(0, n, block):
        left0, sign = FL.ngg_sites(codes, lo, min(lo + block, n))
        plus = sign > 0
        if not circular:
            keep = np.where(plus, left0 + 23 <= n, (left0 >= 3) & (left0 + 20 <= n))
            left0, sign, plus = left0[keep], sign[keep], plus[keep]
        pc = FL.protospacer_codes(codes, left0, plus)
        ok = (pc < 4).all(axis=1) & (pam_codes(codes, left0, plus)[:, 0] < 4)
        parts.append((left0[ok], sign[ok], spacer_keys(pc[ok])))
    return tuple(np.concatenate(x) for x in zip(*parts))


def sites(codes: np.ndarray, circular: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """`scan` without the keys: (plus-strand 0-based protospacer starts, +1/-1 strands)."""
    left0, sign, _ = scan(codes, circular)
    return left0, sign


def normalise(seq: str) -> tuple[str, np.ndarray]:
    """(the sequence with every character other than A, C, G, T turned into N, its base codes with 255 for N).

    One alphabet for both ways of reading a flank (arrays here, strings in the per-guide builder), so that ambiguity codes and assembly gaps give the same columns in each.
    """
    codes = FL.genome_codes(seq)
    return _LETTERS_N[np.minimum(codes, 4)].tobytes().decode("ascii"), codes


# ---------------------------------------------------------------------------------------------------------------- columns
def flank_matrix(seq: str, codes: np.ndarray, left0: np.ndarray, sign: np.ndarray, tables: dict, circular: bool = True) -> np.ndarray:
    """(n, 348) family A for guides of one record, in the released table's column order (`AF._plan(tables)['names']`).

    Circular: every guide goes through the per-locus tracks of `flank_locus`. Linear: guides with a full margin of DNA on both sides do, and the few near an end are
    featurised one at a time from the flanks that exist (windows cut to the DNA available), by the same code that scores a guide in the design tool.
    """
    n = len(codes)
    names = AF._plan(tables)["names"]
    M = np.full((len(left0), len(names)), np.nan)
    inner = np.ones(len(left0), bool) if circular else (left0 >= FL.MARGIN) & (left0 + 20 + FL.MARGIN <= n)
    if inner.any():
        strand = np.where(sign[inner] > 0, "+", "-")
        M[inner], _ = FL.build_matrix(left0[inner], strand, codes, tables)
    edge = np.flatnonzero(~inner)
    if len(edge):
        ctx = pd.DataFrame([D.site_context(seq, int(s), "+" if g > 0 else "-", circular=False) for s, g in zip(left0[edge], sign[edge])])
        ctx["sgRNAID"] = ctx["protospacer"]
        ctx["located"] = True
        M[edge], _ = AF.build_matrix(ctx, tables)
    return M


def copies_of(keys: np.ndarray, counted: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """How many of the genome's NGG sites carry each 20-mer, given `np.unique(all_keys, return_counts=True)`."""
    uniq, counts = counted
    if len(uniq) == 0:
        return np.zeros(len(keys), np.int64)
    i = np.minimum(np.searchsorted(uniq, keys), len(uniq) - 1)
    return np.where(uniq[i] == keys, counts[i], 0)


# ---------------------------------------------------------------------------------------------------------------- the table
def _read_guides(path, records: dict) -> pd.DataFrame:
    g = pd.read_csv(path)
    for c in ("left", "strand"):
        if c not in g.columns:
            raise ValueError(f"{path} needs a '{c}' column (and 'contig' when the FASTA has several records)")
    if "contig" not in g.columns:
        if len(records) != 1:
            raise ValueError(f"{path} has no 'contig' column and the FASTA holds {len(records)} records")
        g["contig"] = next(iter(records))
    bad = sorted(set(g["contig"]) - set(records))
    if bad:
        raise KeyError(f"contig(s) {bad[:5]} in {path} are not in the FASTA (or are too short: records under {MIN_LENGTH} nt are skipped)")
    if not set(g["strand"]) <= {"+", "-"}:
        raise ValueError("'strand' must be + or -")
    return g


def build_table(genome, out, contigs=None, start=None, end=None, guides=None, circular: bool = True, tables_path=None, chunk: int = CHUNK,
                verbose: bool = True) -> dict:
    """Write the family A table of a FASTA's guides to `out` (.csv or .csv.gz) and return a summary (rows, columns, seconds, records skipped)."""
    log = (lambda *a: print(*a, file=sys.stderr, flush=True)) if verbose else (lambda *a: None)
    t0 = time.perf_counter()
    tables = json.loads(Path(tables_path or TABLES).read_text())
    names = AF._plan(tables)["names"]
    records = D.read_records(genome, only=contigs)
    skipped = [k for k, v in records.items() if len(v) < MIN_LENGTH]
    records = {k: v for k, v in records.items() if len(v) >= MIN_LENGTH}
    if skipped:
        log(f"note: skipped {len(skipped)} record(s) shorter than {MIN_LENGTH} nt: {', '.join(skipped[:5])}{' ...' if len(skipped) > 5 else ''}")
    if not records:
        raise ValueError(f"no record of {genome} is at least {MIN_LENGTH} nt long")
    chosen = list(records)
    if (start is not None or end is not None) and (len(chosen) != 1 or guides is not None):
        raise ValueError("--start/--end need exactly one record (name it with --contig) and no --guides file")

    # every NGG site of the records read, for the copy counts
    codes, found, keys = {}, {}, {}
    for k in list(records):
        records[k], codes[k] = normalise(records[k])
        left0, sign, keys[k] = scan(codes[k], circular)
        found[k] = (left0, sign)
    counted = np.unique(np.concatenate(list(keys.values())), return_counts=True)
    log(f"{len(records)} record(s), {sum(len(v) for v in records.values()):,} bp, {sum(len(f[0]) for f in found.values()):,} NGG sites, "
        f"{'circular' if circular else 'linear'}")
    if circular and max(len(v) for v in records.values()) > 20_000_000:
        log("note: a record this long is usually a linear (eukaryotic) chromosome; pass --linear if it is, or its ends will be joined")

    if guides is not None:
        gd = _read_guides(guides, records)
        todo = []
        for k in chosen if contigs else dict.fromkeys(gd["contig"]):
            sub = gd[gd["contig"] == k]
            left0 = sub["left"].to_numpy(np.int64) - 1
            sign = np.where(sub["strand"].to_numpy() == "+", 1, -1)
            plus = sign > 0
            n = len(codes[k])
            if (left0 < 0).any() or (left0 >= n).any():
                raise ValueError(f"a guide of {k} starts outside 1..{n}")
            if not circular and (np.where(plus, left0 + 23 > n, (left0 < 3) | (left0 + 20 > n))).any():
                raise ValueError(f"a guide of {k} runs past the end of the record (--linear)")
            if (FL.protospacer_codes(codes[k], left0, plus) > 3).any():
                raise ValueError(f"a protospacer of {k} holds a base other than A, C, G, T")
            todo.append((k, left0, sign, spacer_keys(FL.protospacer_codes(codes[k], left0, plus))))
    else:
        todo = []
        for k in chosen:
            left0, sign = found[k]
            sel = np.ones(len(left0), bool) if start is None and end is None else (left0 + 1 >= (start or 1)) & (left0 + 1 <= (end or len(codes[k])))
            todo.append((k, left0[sel], sign[sel], keys[k][sel]))

    gz = str(out).endswith(".gz")
    fh = gzip.open(out, "wt", newline="", compresslevel=6) if gz else open(out, "w", newline="")
    rows, first = 0, True
    try:
        for k, left0, sign, key in todo:
            t1 = time.perf_counter()
            for a in range(0, len(left0), chunk):
                l0, sg, ky = left0[a:a + chunk], sign[a:a + chunk], key[a:a + chunk]
                plus = sg > 0
                ids = pd.DataFrame({
                    "contig": k, "left": l0 + 1, "right": l0 + 20, "strand": np.where(plus, "+", "-"),
                    "protospacer": to_strings(FL.protospacer_codes(codes[k], l0, plus)), "pam": to_strings(pam_codes(codes[k], l0, plus)),
                    "n_genome_copies": copies_of(ky, counted)})
                M = flank_matrix(records[k], codes[k], l0, sg, tables, circular)
                pd.concat([ids, pd.DataFrame(M.astype(np.float32), columns=names)], axis=1).to_csv(fh, index=False, header=first)
                first = False
                rows += len(l0)
            log(f"  {k}: {len(left0):,} guides in {time.perf_counter() - t1:.1f} s")
    finally:
        fh.close()
    secs = time.perf_counter() - t0
    log(f"wrote {rows:,} rows x {len(ID_COLUMNS) + len(names)} columns to {out} in {secs:.0f} s")
    return dict(rows=rows, columns=len(ID_COLUMNS) + len(names), seconds=secs, skipped=skipped)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--genome", required=True, help="FASTA file (plain or .gz); every record of at least 2,023 nt is used")
    ap.add_argument("--out", required=True, help="CSV to write; a name ending in .gz is gzip-compressed")
    ap.add_argument("--contig", nargs="+", help="only these records, which are the only ones read (default: all); n_genome_copies counts within them")
    ap.add_argument("--start", type=int, help="first 1-based `left` to include (needs one --contig)")
    ap.add_argument("--end", type=int, help="last 1-based `left` to include (needs one --contig)")
    ap.add_argument("--guides", help="CSV of the guides to featurise instead of every NGG site: columns left (1-based plus-strand start of the protospacer), strand (+ or -), contig")
    ap.add_argument("--linear", action="store_true", help="treat records as linear, as eukaryotic chromosomes are (default: circular, as a bacterial chromosome)")
    ap.add_argument("--tables", help=f"quantum lookup tables (JSON); default {TABLES}")
    a = ap.parse_args(argv)
    try:
        build_table(a.genome, a.out, a.contig, a.start, a.end, a.guides, not a.linear, a.tables)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
