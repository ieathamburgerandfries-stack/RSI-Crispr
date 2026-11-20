"""Is the downstream flank's advantage a transcription effect?

Family A helps more on the PAM side than the other side -- in the sides split,
+0.072 downstream against +0.027 upstream. Two explanations make different
predictions, and Part 12 listed separating them as cheap.

**PAM / R-loop geometry.** "Downstream" is defined relative to the guide: the
PAM sits immediately 3' of the protospacer, and the R-loop opens there and
propagates away from it. On this account the asymmetry belongs to the Cas9
complex and should not care which way the guide faces relative to its gene.

**Transcription.** RNA polymerase travels along a gene in one direction,
unwinding DNA ahead of itself and leaving supercoiling behind. That direction is
set by the gene, not the guide, so on this account the asymmetry should depend
on the guide's orientation within its gene, and should be stronger where there
is more transcription to do.

Those give two tests. One of them **cannot be run on this screen**, and finding
that out is the first thing this module reports; the other can.

    python -m sgrna.asymmetry --audit     # why the orientation test is impossible
    python -m sgrna.asymmetry --run       # the expression-conditioned test
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config

OUT = config.RESULTS / "asymmetry_expression.csv"
OUT_AUDIT = config.RESULTS / "asymmetry_strand_audit.csv"


def gene_strand() -> pd.Series:
    """Gene name -> '+' or '-', from the reference gene table."""
    g = pd.read_csv(config.GENE_TABLE, sep="\t")
    return g.set_index("name")["ori"].astype(str)


def audit(verbose: bool = True) -> pd.DataFrame:
    """Does the library contain guides facing both ways inside their genes?

    It does not, and that kills the orientation test before it starts. Checked
    two independent ways: joining the guide index's mapped strand against the
    reference gene table, and reading the `g_transcription` family's own
    `template_strand` column, which was derived separately.
    """
    from ..io_utils import load_block, load_guide_index

    idx = load_guide_index()
    idx = idx[idx["located"].astype(bool)].copy()
    idx["gene_strand"] = idx["gene_name"].map(gene_strand())
    idx["relative_orientation"] = np.where(
        idx["strand"].astype(str) == idx["gene_strand"],
        "sense strand", "template strand")

    blk = load_block("g_transcription")[[config.ID_COL,
                                         "eng.txn.gene.template_strand"]]
    merged = idx.merge(blk, on=config.ID_COL, how="left")

    rows = []
    for label, counts in (
        ("guide index joined to gene table",
         merged["relative_orientation"].value_counts(dropna=False)),
        ("g_transcription template_strand column",
         merged["eng.txn.gene.template_strand"]
         .map({1.0: "template strand", 0.0: "sense strand"})
         .value_counts(dropna=False)),
    ):
        for k, v in counts.items():
            rows.append(dict(source=label, orientation=str(k), guides=int(v)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_AUDIT, index=False)

    if verbose:
        print(df.to_string(index=False))
        n_sense = int((merged["eng.txn.gene.template_strand"] == 0.0).sum())
        print(f"\n  guides on the gene's sense strand: {n_sense} of "
              f"{len(merged):,} ({n_sense / len(merged):.4%})")
        print("  => the orientation test is impossible on this screen: the "
              "library targets the template strand almost exclusively, so "
              "there is no variation to condition on.")
        print("  => and `eng.txn.gene.template_strand` is a near-constant "
              "column, which is part of why g_transcription could only give "
              "+0.015.")
    return df


def _side_of(col: str) -> str:
    """'up', 'dn' or 'other' for one family-A column.

    Family A names the side in three different places depending on the
    sub-block, so all three are handled explicitly:
      eng.flank.seq.u10.A                  -> 4th component
      eng.flank.qct.dimer.HLgapeVE.u10     -> last component
      eng.flank.comp.up50.gc               -> infix
    The PAM-spanning dinucleotide (`.pamN`) belongs to neither side.
    """
    parts = col.split(".")
    if col.startswith("eng.flank.seq.") and len(parts) > 3:
        tag = parts[3]
    elif col.startswith("eng.flank.qct."):
        tag = parts[-1]
    else:
        tag = ""
    if tag.startswith("u") and tag[1:].isdigit():
        return "up"
    if tag.startswith("d") and tag[1:].isdigit():
        return "dn"
    if ".up" in col:
        return "up"
    if ".dn" in col:
        return "dn"
    return "other"


def run(seed: int = 41, n_splits: int = 5, n_bins: int = 3,
        verbose: bool = True) -> pd.DataFrame:
    """The test this screen can support: does the asymmetry scale with transcription?

    Split guides by their gene's expression and measure the upstream and
    downstream gains inside each band. If polymerase traffic is what makes the
    PAM-side flank matter more, the gap should widen with expression. If it is
    R-loop geometry, the gap should be flat across bands.

    Expression is `eng.txn.expr.median_log_tpm`, which came in with
    `g_transcription`; it is a property of the *gene*, so it is constant across
    the ~20 guides in each gene and cannot be a per-guide artefact of the flank
    features.
    """
    from ..build.build_matrix import load_dataset
    from ..io_utils import load_block
    from ..harness.run_ablation import cross_validate

    X, y, names, ids, family_of = load_dataset(["a_flank"], verbose=verbose)
    family_of = np.asarray(family_of)
    names = np.asarray(names)
    base = np.where(family_of == "base")[0]
    flank = np.where(family_of != "base")[0]
    side = np.array([_side_of(n) for n in names[flank]])
    if verbose:
        print(f"  family A: {(side == 'up').sum()} upstream, "
              f"{(side == 'dn').sum()} downstream columns")

    expr = (load_block("g_transcription")
            .set_index(config.ID_COL)["eng.txn.expr.median_log_tpm"])
    e = pd.Series(ids).map(expr).to_numpy(dtype=float)
    ok = np.isfinite(e)
    qs = np.quantile(e[ok], np.linspace(0, 1, n_bins + 1))
    band = np.full(len(e), -1)
    for b in range(n_bins):
        lo, hi = qs[b], qs[b + 1]
        sel = ok & (e >= lo) & ((e <= hi) if b == n_bins - 1 else (e < hi))
        band[sel] = b
    labels = {0: "low expression", 1: "mid expression", 2: "high expression"}

    # The sides are not matched for size -- family A has 194 downstream columns
    # against 144 upstream, because the windows and the PAM-side k-mers are not
    # mirror images. A third of the downstream advantage could therefore be
    # column count rather than side, so a count-matched downstream arm is run
    # alongside: 144 downstream columns drawn at random, same seed.
    rng = np.random.default_rng(seed)
    dn_idx = flank[side == "dn"]
    up_idx = flank[side == "up"]
    dn_matched = rng.choice(dn_idx, size=len(up_idx), replace=False)

    arms = {
        "baseline": base,
        "upstream_only": np.sort(np.concatenate([base, up_idx])),
        "downstream_only": np.sort(np.concatenate([base, dn_idx])),
        "downstream_matched": np.sort(np.concatenate([base, dn_matched])),
        "both": np.arange(X.shape[1]),
    }

    rows = []
    for b in range(n_bins):
        keep = band == b
        grp = labels.get(b, f"band {b}")
        for arm, cols in arms.items():
            res = cross_validate(
                X[keep][:, cols], y[keep], list(names[cols]), family_of[cols],
                seeds=(seed,), n_splits=n_splits, verbose=False,
                tag=f"asymexpr_b{b}_{arm}", experiment=f"{grp} | {arm}")
            s = res["summary"]
            rows.append(dict(band=b, group=grp, n_guides=int(keep.sum()),
                             median_log_tpm=float(np.nanmedian(e[keep])),
                             arm=arm, spearman=s["spearman"]["mean"],
                             r2=s["r2"]["mean"]))
            if verbose:
                print(f"    {grp:18s} {arm:16s} rho "
                      f"{s['spearman']['mean']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
    return pd.DataFrame(rows)


def summary():
    d = pd.read_csv(OUT)
    piv = d.pivot_table(index=["band", "group"], columns="arm",
                        values="spearman")
    for arm in ("upstream_only", "downstream_only",
                "downstream_matched", "both"):
        if arm in piv:
            piv[f"gain_{arm}"] = (piv[arm] - piv["baseline"]).round(4)
    if {"gain_downstream_only", "gain_upstream_only"} <= set(piv.columns):
        piv["asymmetry"] = (piv["gain_downstream_only"]
                            - piv["gain_upstream_only"]).round(4)
    if {"gain_downstream_matched", "gain_upstream_only"} <= set(piv.columns):
        piv["asymmetry_matched"] = (piv["gain_downstream_matched"]
                                    - piv["gain_upstream_only"]).round(4)
    show = ["baseline"] + [c for c in piv.columns if c.startswith("gain_")]
    show += [c for c in ("asymmetry", "asymmetry_matched")
             if c in piv.columns]
    print(piv[show].round(4).to_string())
    return piv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)
    if args.audit or not args.run:
        audit()
    if args.run:
        run(n_splits=args.folds)
        summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
