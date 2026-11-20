"""Step 0 -- turn 13,880 anonymous rows into located guides.

The published matrix identifies each guide only by a name like
`aaeAb3241_107_Cas9`. Nothing downstream of family A can be built from that:
supercoiling, nucleoid structure, methylation, transcription and off-target
burden all need to know *where on the chromosome* the guide sits. This script
recovers that, and writes one table everything else reads.

What it does
------------
1. Decodes the 20 nt protospacer from the QCT electron-count columns
   (see `qct.py` for why that works).
2. Finds each protospacer in NC_000913.2 on either strand, and records the
   coordinates, the strand, the real PAM and the flanking sequence.
3. Checks every hit: the PAM must be NGG, and a guide that matches the genome
   more than once is flagged rather than silently averaged.
4. Learns the QCT k-mer lookup tables so family A can extend the published
   quantum descriptors into the flanks.

Run from a notebook cell (the usual way):

    import sgrna.notebook as nb
    nb.setup()
    nb.build_index()

or from a terminal:

    python -m sgrna.build_guide_index
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .. import config
from . import genome
from . import qct
from .qct import parse_sgrna_id


def build(flank: int = config.FLANK, verbose: bool = True) -> pd.DataFrame:
    say = print if verbose else (lambda *a, **k: None)

    say(f"Reading {config.BASE_MATRIX.name} ...")
    frame = qct.load_qct_frame(verbose=verbose)
    decoded = qct.decode_protospacers(frame)
    say(f"  {len(decoded):,} guides, {decoded['decoded_ok'].sum():,} fully decoded")

    g = genome.load_genome()
    say(f"  genome {config.GENOME_FASTA.name}: {len(g):,} bp")
    n_sites = len(genome.protospacer_sites()[0])
    say(f"  indexed {n_sites:,} NGG-flanked protospacer sites")

    ids = decoded[config.ID_COL].map(parse_sgrna_id).apply(pd.Series)
    records = []
    for proto, ok in zip(decoded["protospacer"], decoded["decoded_ok"]):
        if not ok:
            records.append(dict(n_genome_copies=0))
            continue
        hits = genome.locate_protospacer(proto, g)
        if not hits:
            records.append(dict(n_genome_copies=0))
            continue
        h = hits[0]
        ctx = genome.guide_context(h["left"], h["right"], h["strand"], flank, g)
        records.append(
            dict(
                strand=h["strand"],
                left=h["left"],
                right=h["right"],
                pam=h["pam"],
                n_genome_copies=len(hits),
                upstream=ctx["upstream"],
                downstream=ctx["downstream"],
            )
        )

    loc = pd.DataFrame(records)
    out = pd.concat(
        [decoded.reset_index(drop=True), ids.reset_index(drop=True), loc], axis=1
    )

    # ---- quality control -------------------------------------------------
    out["located"] = out["n_genome_copies"] >= 1
    out["unique_in_genome"] = out["n_genome_copies"] == 1
    out["pam_is_ngg"] = out["pam"].fillna("").str.endswith("GG")

    n = len(out)
    say(f"  located in genome      : {out['located'].sum():,} / {n:,}")
    say(f"  unique single copy     : {out['unique_in_genome'].sum():,} / {n:,}")
    say(f"  NGG PAM confirmed      : {out['pam_is_ngg'].sum():,} / {n:,}")
    if out["located"].sum() and not out.loc[out["located"], "pam_is_ngg"].all():
        bad = out.loc[out["located"] & ~out["pam_is_ngg"], config.ID_COL].head(5)
        say(f"  !! non-NGG PAMs, e.g. {list(bad)}")

    # ---- replication landmarks ------------------------------------------
    lm = genome.replication_landmarks()
    say(
        f"  GC-skew origin         : {lm['ori']:,} "
        f"(annotated oriC is near 3,923,800 -- these should agree)"
    )
    say(f"  GC-skew terminus       : {lm['ter']:,}")

    out["ori_distance"] = [
        genome.circular_distance(int(p), lm["ori"]) if pd.notna(p) else None
        for p in out["left"]
    ]
    out["replichore"] = [
        genome.replichore(int(p)) if pd.notna(p) else None for p in out["left"]
    ]
    # Is the guide's own strand the leading-strand template here?
    out["leading_strand"] = [
        None
        if pd.isna(p) or not isinstance(s, str)
        else int((genome.replication_fork_direction(int(p)) == 1) == (s == "+"))
        for p, s in zip(out["left"], out["strand"])
    ]

    out["gene_offset_check"] = out["gene_offset"]
    out = out.rename(columns={config.TARGET_COL: "cut_score"})

    cols = [
        config.ID_COL, "cut_score", "gene_name", "b_number", "gene_offset",
        "nuclease", "protospacer", "pam", "strand", "left", "right",
        "n_genome_copies", "located", "unique_in_genome", "pam_is_ngg",
        "ori_distance", "replichore", "leading_strand", "upstream", "downstream",
    ]
    out = out[[c for c in cols if c in out.columns]]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--flank", type=int, default=config.FLANK,
                    help="nt of flanking sequence to store either side")
    ap.add_argument("--skip-qct-tables", action="store_true",
                    help="do not rebuild the QCT k-mer lookup tables")
    args = ap.parse_args(argv)

    idx = build(flank=args.flank)
    idx.to_csv(config.GUIDE_INDEX, index=False)
    print(f"\nWrote {config.GUIDE_INDEX} ({len(idx):,} rows)")

    if not args.skip_qct_tables:
        print("\nLearning QCT k-mer lookup tables from the published matrix ...")
        tables = qct.learn_lookup_tables(idx["protospacer"], qct.load_qct_frame())
        qct.save_tables(tables)
        cov = qct.table_coverage(tables)
        print(cov.to_string(index=False))
        print(f"\nWrote {config.QCT_TABLES}")
        thin = cov[cov["coverage"] < 0.9]
        if len(thin):
            print(
                "\nNote: the k-mers below were never seen in the 13,880 guides, so "
                "flank positions containing them will be NaN and median-imputed:"
            )
            print(thin.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
