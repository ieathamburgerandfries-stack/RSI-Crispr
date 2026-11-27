"""Check `family_a_table`: it rebuilds the released family A table from the genome, and agrees with the older per-guide code on a genome it was not built for.

A. The 33,567 curated guides, given only as coordinates on the E. coli genome, must come back with the stored protospacer, PAM, genome copy count and 348 columns.
B. Every NGG site of the E. coli genome: how many, how long it takes, how big the table is, and that every curated guide is among them.
C. Citrobacter rodentium (a second genome): every site is written; 400 sites (the first and last 40, so that the origin is crossed, and 320 at random) are featurised again
   one at a time from their flanking sequence by `a_flank_fast`, the code the design tool and the released table were made with, and must agree; copy counts must equal
   the design tool's `_count_sites`.
D. Linear mode on E. coli: sites that would wrap round the origin are dropped, guides with a full margin of DNA give the same columns as when circular, and a guide with fewer
   than 250 nt upstream has the same 250 and 1,000 nt window (the windows are cut to the DNA there is).

Written to `results/post_freeze/family_a_table_check.csv`. Runs for about five minutes (two whole genomes are written to a temporary folder and deleted).

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


def _compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Two (n, 348) arrays that were both cast to float32: same missing cells, how many cells differ, and by how much."""
    same_nan = bool((np.isnan(a) == np.isnan(b)).all())
    both = ~np.isnan(a) & ~np.isnan(b)
    d = np.where(both, np.abs(a - b), 0.0)
    rel = d / np.maximum(np.abs(np.where(both, b, 0.0)), 1e-12)
    return dict(same_nan=same_nan, differ=int((d > 0).sum()), max_abs=float(d.max()), max_rel=float(rel.max()))


def _sample_check(genome: str, sample: pd.DataFrame, g: str, tables: dict, tmp: Path, tag: str) -> dict:
    """Run the CLI on the sampled guides and compare with the per-guide path on the same guides (`a_flank_fast` from the flanking sequences)."""
    csv = tmp / f"{tag}_guides.csv"
    sample[["left", "strand"]].to_csv(csv, index=False)
    out = tmp / f"{tag}_sample.csv"
    FA.build_table(genome, out, guides=csv, tables_path=None, verbose=False)
    got = pd.read_csv(out)
    names = [c for c in got.columns if c.startswith("eng.flank.")]
    ctx = pd.DataFrame([D.site_context(g, int(l) - 1, s) for l, s in zip(sample["left"], sample["strand"])])
    ref, ref_names = FA.AF.build_matrix(ctx.assign(sgRNAID=ctx["protospacer"], located=True), tables)
    assert ref_names == names, "column order differs from the per-guide builder"
    cmp = _compare(got[names].to_numpy(np.float32).astype(np.float64), ref.astype(np.float32).astype(np.float64))
    counts = [D._count_sites(g, p, True)[0] for p in ctx["protospacer"]]
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
