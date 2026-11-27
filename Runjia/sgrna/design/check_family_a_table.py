"""Check `family_a_table`: it rebuilds the released family A table from the genome, and agrees with the older per-guide code on a genome it was not built for.

A. The 33,567 curated guides, given only as coordinates on the E. coli genome, must come back with the stored protospacer, PAM, genome copy count and 348 columns.
B. Every NGG site of the E. coli genome: how many, how long it takes, how big the table is, and that every curated guide is among them.
C. Citrobacter rodentium (a second genome): every site is written; 400 sites (the first and last 40, so that the origin is crossed, and 320 at random) are featurised again
   one at a time from their flanking sequence by `a_flank_fast`, the code the design tool and the released table were made with, and must agree; copy counts must equal
   the design tool's `_count_sites`.
E. Human chromosome 21 (hg38; linear, soft-masked, with long assembly gaps): 400 sites, up to 120 of them within a kb of a gap, checked the same way as in C.
G. The same for the domestic cat (*Felis catus*) chromosome E3, 41.4 Mb of RefSeq F.catus_Fca126_mat1.0 (downloaded once from NCBI into `external_data/cat/`).
F. A stress genome: E. coli with 450 planted tracts (poly-A, poly-T, (AC)n, N gaps, ambiguity codes), 40 stray N or Y, and a third of its length soft-masked, checked the
   same way at 400 sites within 200 nt of a tract (what a mammalian genome has and E. coli does not).
D. Linear mode on E. coli: sites that would wrap round the origin are dropped, guides with a full margin of DNA give the same columns as when circular, and a guide with fewer
   than 250 nt upstream has the same 250 and 1,000 nt window (the windows are cut to the DNA there is).

Written to `results/family_a_table_check.csv`. Runs for about five minutes (two whole genomes are written to a temporary folder and deleted).

    ./run_ext.sh sgrna.design.check_family_a_table
"""
from __future__ import annotations

import gzip
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..common import EXTERNAL, OUT
from . import design_guide as D
from . import family_a_table as FA

CRODENTIUM = EXTERNAL / "C_rodentium" / "NC_013716.1.fasta"
HG38 = EXTERNAL / "hg38" / "hg38.fa.gz"
CAT = EXTERNAL / "cat" / "NC_058383.1.fasta"        # domestic cat chromosome E3, RefSeq F.catus_Fca126_mat1.0 (efetch of NC_058383.1)


def _compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Two (n, 348) arrays that were both cast to float32: same missing cells, how many cells differ, and by how much."""
    same_nan = bool((np.isnan(a) == np.isnan(b)).all())
    both = ~np.isnan(a) & ~np.isnan(b)
    d = np.where(both, np.abs(a - b), 0.0)
    rel = d / np.maximum(np.abs(np.where(both, b, 0.0)), 1e-12)
    return dict(same_nan=same_nan, differ=int((d > 0).sum()), max_abs=float(d.max()), max_rel=float(rel.max()))


def _sample_check(genome, sample: pd.DataFrame, g: str, tables: dict, tmp: Path, tag: str, contig: str | None = None, circular: bool = True) -> dict:
    """Run the CLI on the sampled guides and compare with the per-guide path on the same guides (`a_flank_fast` from the flanking sequence; `g` is the normalised sequence)."""
    csv = tmp / f"{tag}_guides.csv"
    sample[["left", "strand"]].to_csv(csv, index=False)
    out = tmp / f"{tag}_sample.csv"
    FA.build_table(genome, out, contigs=[contig] if contig else None, guides=csv, circular=circular, verbose=False)
    got = pd.read_csv(out)
    names = [c for c in got.columns if c.startswith("eng.flank.")]
    ctx = pd.DataFrame([D.site_context(g, int(l) - 1, s, circular) for l, s in zip(sample["left"], sample["strand"])])
    ref, ref_names = FA.AF.build_matrix(ctx.assign(sgRNAID=ctx["protospacer"], located=True), tables)
    assert ref_names == names, "column order differs from the per-guide builder"
    cmp = _compare(got[names].to_numpy(np.float32).astype(np.float64), ref.astype(np.float32).astype(np.float64))
    counts = [D._count_sites(g, p, circular)[0] for p in ctx["protospacer"]]
    return dict(n=len(got), protospacer=int((got["protospacer"].to_numpy() == ctx["protospacer"].to_numpy()).sum()),
                pam=int((got["pam"].to_numpy() == ctx["pam"].to_numpy()).sum()),
                copies=int((got["n_genome_copies"].to_numpy() == np.array(counts)).sum()), columns=len(names), **cmp)


def main() -> int:
    rows = []
    add = lambda item, value: rows.append((item, value))
    tables = json.loads(FA.TABLES.read_text())
    name, g = D.read_fasta(config.GENOME_FASTA)
    tmp_dir = tempfile.TemporaryDirectory()
    tmp = Path(tmp_dir.name)

    # ---- A. the released table, from coordinates
    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv", usecols=["sgRNAID", "protospacer", "pam", "strand", "left", "n_genome_copies"])
    pd.DataFrame({"left": idx["left"], "strand": idx["strand"]}).to_csv(tmp / "curated_guides.csv", index=False)
    FA.build_table(config.GENOME_FASTA, tmp / "curated_table.csv", guides=tmp / "curated_guides.csv", verbose=False)
    ours = pd.read_csv(tmp / "curated_table.csv")
    stored = pd.read_csv(config.FEATURE_DIR / "a_flank_curated.csv").set_index(config.ID_COL).loc[idx[config.ID_COL]]
    names = list(stored.columns)
    assert [c for c in ours.columns if c.startswith("eng.flank.")] == names
    cmp = _compare(ours[names].to_numpy(np.float32).astype(np.float64), stored[names].to_numpy(np.float32).astype(np.float64))
    add("A: curated guides given as coordinates", len(ours))
    add("A: protospacer equals the stored one", int((ours["protospacer"].to_numpy() == idx["protospacer"].to_numpy()).sum()))
    add("A: PAM equals the stored one", int((ours["pam"].to_numpy() == idx["pam"].to_numpy()).sum()))
    add("A: genome copy count equals the stored one", int((ours["n_genome_copies"].to_numpy() == idx["n_genome_copies"].to_numpy()).sum()))
    add("A: columns compared", len(names))
    add("A: missing-value pattern equal", float(cmp["same_nan"]))
    add("A: cells that differ after the float32 cast", cmp["differ"])
    add("A: maximum absolute difference in a column", cmp["max_abs"])
    add("A: maximum relative difference in a column", cmp["max_rel"])
    del ours, stored

    # ---- B. every NGG site of the E. coli genome
    t = time.perf_counter()
    big = tmp / "ecoli_all.csv.gz"
    res = FA.build_table(config.GENOME_FASTA, big, verbose=False)
    secs = time.perf_counter() - t
    ids = pd.read_csv(big, usecols=range(len(FA.ID_COLUMNS)))
    plain = 0
    with gzip.open(big, "rb") as fh:
        while chunk := fh.read(1 << 24):
            plain += len(chunk)
    key = set(zip(idx["left"] - 1, idx["strand"]))
    found = sum((l - 1, s) in key for l, s in zip(ids["left"], ids["strand"]))
    add("B: E. coli NGG sites", len(ids))
    add("B: on the plus strand", int((ids["strand"] == "+").sum()))
    add("B: on the minus strand", int((ids["strand"] == "-").sum()))
    add("B: nucleotides per site", len(g) / len(ids))
    add("B: sites whose 20-mer has more than one NGG site", int((ids["n_genome_copies"] > 1).sum()))
    add("B: curated guides found among the sites", found)
    add("B: columns in the table", res["columns"])
    add("B: seconds for the whole genome, written as .csv.gz", secs)
    add("B: milliseconds per guide, featurising and writing", secs / len(ids) * 1000)
    add("B: size of the .csv.gz, MB", big.stat().st_size / 1e6)
    add("B: size as plain CSV, GB", plain / 1e9)
    big.unlink()
    del ids

    # ---- C. a second genome
    cname, cg = D.read_fasta(CRODENTIUM)
    cg = FA.normalise(cg)[0]
    t = time.perf_counter()
    cbig = tmp / "crod_all.csv.gz"
    FA.build_table(CRODENTIUM, cbig, verbose=False)
    csecs = time.perf_counter() - t
    cids = pd.read_csv(cbig, usecols=range(len(FA.ID_COLUMNS)))
    cbig.unlink()
    rng = np.random.default_rng(0)
    pick = np.unique(np.r_[np.arange(40), len(cids) - 40 + np.arange(40), rng.choice(len(cids), 320, replace=False)])
    s = _sample_check(CRODENTIUM, cids.iloc[pick].reset_index(drop=True), cg, tables, tmp, "crod")
    add("C: C. rodentium length, bp", len(cg))
    add("C: NGG sites", len(cids))
    add("C: seconds for the whole genome, written as .csv.gz", csecs)
    add("C: sites sampled (first 40, last 40 and 320 random, repeats removed)", s["n"])
    add("C: protospacer equals the one read from the sequence", s["protospacer"])
    add("C: PAM equals the one read from the sequence", s["pam"])
    add("C: genome copy count equals the design tool's", s["copies"])
    add("C: columns compared", s["columns"])
    add("C: missing-value pattern equal", float(s["same_nan"]))
    add("C: cells that differ after the float32 cast", s["differ"])
    add("C: maximum relative difference in a column", s["max_rel"])
    del cids

    # ---- E, G. a mammalian chromosome: human chr21 and cat chromosome E3 (linear, with assembly gaps)
    for L, species, fasta, contig in (("E", "human chr21", HG38, "chr21"), ("G", "cat chromosome E3", CAT, "NC_058383.1")):
        cseq, ccodes = FA.normalise(D.read_records(fasta, only=[contig])[contig])
        cl, cs, _ = FA.scan(ccodes, circular=False)
        cn = np.concatenate([[0], np.cumsum(ccodes == 255)])
        reach = cn[np.minimum(len(ccodes), cl + 1033)] - cn[np.maximum(0, cl - 1010)]        # N bases within reach of each site's flanks
        near = np.flatnonzero((reach > 0) & (reach < 2000))
        cpick = np.unique(np.r_[np.arange(40), len(cl) - 40 + np.arange(40), rng.choice(near, min(120, len(near)), replace=False), rng.choice(len(cl), 200, replace=False)])
        r_ = _sample_check(fasta, pd.DataFrame({"left": cl[cpick] + 1, "strand": np.where(cs[cpick] > 0, "+", "-")}), cseq, tables, tmp, L, contig=contig, circular=False)
        add(f"{L}: {species}: length, bp", len(ccodes))
        add(f"{L}: {species}: bases that are N (assembly gaps)", int((ccodes == 255).sum()))
        add(f"{L}: {species}: NGG sites (linear)", len(cl))
        add(f"{L}: {species}: sites sampled (first 40, last 40, up to 120 within a kb of a gap and 200 random, repeats removed)", r_["n"])
        add(f"{L}: {species}: of these, with an N within reach of a flank", int((reach[cpick] > 0).sum()))
        add(f"{L}: {species}: protospacer equals the one read from the sequence", r_["protospacer"])
        add(f"{L}: {species}: PAM equals the one read from the sequence", r_["pam"])
        add(f"{L}: {species}: genome copy count equals the design tool's", r_["copies"])
        add(f"{L}: {species}: columns compared", r_["columns"])
        add(f"{L}: {species}: missing-value pattern equal", float(r_["same_nan"]))
        add(f"{L}: {species}: cells that differ after the float32 cast", r_["differ"])
        add(f"{L}: {species}: maximum absolute difference in a column", r_["max_abs"])
        del ccodes, cseq

    # ---- F. a stress genome
    rng = np.random.default_rng(1)
    gl, n_g = list(g), len(g)
    tracts = []
    for unit, count in (("A", 150), ("AC", 100), ("N", 60), ("T", 60), ("R", 40)):
        for _ in range(count):
            p0 = int(rng.integers(2000, n_g - 2000))
            gl[p0:p0 + 60] = list((unit * 60)[:60])
            tracts.append(p0)
    for _ in range(40):
        gl[int(rng.integers(2000, n_g - 2000))] = "N" if rng.random() < 0.5 else "Y"
    mixed = "".join(ch.lower() if (i // 500) % 3 == 0 else ch for i, ch in enumerate("".join(gl)))
    stress = tmp / "stress.fa"
    stress.write_text(">stress\n" + "\n".join(mixed[i:i + 80] for i in range(0, n_g, 80)) + "\n")
    sseq, scodes = FA.normalise(mixed.upper())
    sl, ss, _ = FA.scan(scodes, circular=True)
    near = np.flatnonzero(np.abs(sl[:, None] - np.array(tracts)[None, :]).min(axis=1) < 200)
    spick = np.unique(rng.choice(near, 400, replace=False))
    st_ = _sample_check(stress, pd.DataFrame({"left": sl[spick] + 1, "strand": np.where(ss[spick] > 0, "+", "-")}), sseq, tables, tmp, "stress", circular=True)
    add("F: planted tracts and gaps", len(tracts))
    add("F: bases that are not A, C, G or T", int((scodes == 255).sum()))
    add("F: lower-case bases in the FASTA", sum(ch.islower() for ch in mixed))
    add("F: sites within 200 nt of a planted tract", len(near))
    add("F: sites sampled", st_["n"])
    add("F: protospacer equals the one read from the sequence", st_["protospacer"])
    add("F: PAM equals the one read from the sequence", st_["pam"])
    add("F: genome copy count equals the design tool's", st_["copies"])
    add("F: columns compared", st_["columns"])
    add("F: missing-value pattern equal", float(st_["same_nan"]))
    add("F: cells that differ after the float32 cast", st_["differ"])
    add("F: maximum absolute difference in a column", st_["max_abs"])
    del scodes, sseq, mixed, gl

    # ---- D. linear mode
    codes = FA.FL.genome_codes(g)
    n = len(codes)
    lc, _ = FA.sites(codes, True)
    ll, ls = FA.sites(codes, False)
    M = FA.FL.MARGIN
    inner = (ll >= M) & (ll + 20 + M <= n)
    low = np.flatnonzero((ll < 60_000) | (ll > n - 60_000))
    lin = FA.flank_matrix(g, codes, ll[low], ls[low], tables, circular=False)
    cir = FA.flank_matrix(g, codes, ll[low], ls[low], tables, circular=True)
    ok = inner[low]
    nm = FA.AF._plan(tables)["names"]
    c = {k: nm.index(f"eng.flank.comp.{k}.gc") for k in ("up250", "up1000", "dn250", "dn1000")}
    st, pl = ll[low], ls[low] > 0
    up_len, dn_len = np.where(pl, st, n - st - 20), np.where(pl, n - st - 23, st - 3)       # nt of DNA on each side of the guide
    t_up, t_dn = (up_len >= 50) & (up_len < 250), (dn_len >= 50) & (dn_len < 250)
    add("D: E. coli sites when circular", len(lc))
    add("D: sites when linear (those wrapping the origin are dropped)", len(ll))
    add("D: linear sites within the margin of an end", int((~inner).sum()))
    add("D: guides near the ends compared with circular", int(ok.sum()))
    add("D: maximum absolute difference for guides with a full margin", float(np.nanmax(np.abs(lin[ok] - cir[ok]))))
    add("D: flanks with 50 to 249 nt of DNA", int(t_up.sum() + t_dn.sum()))
    add("D: of these, the 1,000 nt window equals the 250 nt window (GC)",
        int((lin[t_up, c["up1000"]] == lin[t_up, c["up250"]]).sum() + (lin[t_dn, c["dn1000"]] == lin[t_dn, c["dn250"]]).sum()))

    out = pd.DataFrame(rows, columns=["item", "value"])
    out.to_csv(OUT / "family_a_table_check.csv", index=False)
    print(out.to_string(index=False))
    tmp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
