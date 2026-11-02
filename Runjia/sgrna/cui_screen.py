"""The last untouched test: does the flanking-context effect appear in a CRISPRi screen?

Every external check so far has been another Cas9 **cutting** screen, and all of
them are now used for something. Cui et al. (2018) is a dCas9 **CRISPRi** screen
in the same organism -- 85,392 guides, two measurement arms -- and this project
has taken exactly one number from it (the two arms agree at ρ 0.711, quoted once
as a ceiling range). Nothing else: no model, no feature, no hyperparameter, no
split has ever been chosen or evaluated against it.

**It can falsify the mechanism, which is why it is worth spending.** Finding 3's
account is a duplex-opening step limited by local DNA composition at the *cut
site*. dCas9 binds and does not cut, and the measured quantity is fitness under
transcriptional silencing rather than survival after cleavage. So:

    family A still predicts  ->  the effect is about binding or accessibility,
                                 not about cutting, and Finding 3's mechanism
                                 sentence is wrong as written
    family A does not        ->  the effect is specific to cleavage, which is
                                 what Finding 3 claims

Either outcome is a result. Neither is available from another cutting screen.

**Gene-grouped CV is not optional here, it is the whole experiment.** CRISPRi
fitness is dominated by whether the silenced gene is essential, so almost all of
the label's variance is *between* genes. Family A's columns are read from a 2 kb
window and identify the locus, so under random KFold a model can score by
recognising which gene it is looking at and be rewarded for it. That route
would produce a large, entirely spurious family A gain. Both arms are therefore
run under whole-gene holdout, with the random-KFold version reported beside it
to show the size of the trap.

    python -m sgrna.cui_screen --build
    python -m sgrna.cui_screen --run
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config, genome

CACHE = config.INTERIM / "cui_crispri"
OUT = config.RESULTS / "cui_screen.csv"
FLANK = 1000


def load_screen(verbose: bool = True) -> pd.DataFrame:
    """One row per uniquely-targeting guide, labelled by the mean of both arms."""
    d = pd.read_csv(config.BADSEED / "screen_data.csv")
    n0 = len(d)
    d = d[d["ntargets"] == 1]                      # multi-target guides are a
    d = d[d["guide"].str.fullmatch(r"[ACGT]{20}")] # different measurement
    # The two arms (18 and 75 generations) agree at rho 0.711, so their mean is
    # the more reliable label: Spearman-Brown gives 2r/(1+r) = 0.83 against 0.71
    # for either arm alone.
    d = d.groupby("guide", as_index=False).agg(
        fit18=("fit18", "mean"), fit75=("fit75", "mean"),
        gene=("gene", "first"), essential=("essential", "first"),
        pos=("pos", "first"), ori=("ori", "first"))
    d["y"] = d[["fit18", "fit75"]].mean(axis=1)
    if verbose:
        print(f"  {n0:,} rows -> {len(d):,} unique single-target guides")
    return d


def build(force: bool = False, verbose: bool = True):
    """Locate every guide in NC_000913.2, cut its flanks, featurise."""
    CACHE.mkdir(parents=True, exist_ok=True)
    x_path, meta_path = CACHE / "X.npy", CACHE / "meta.json"
    if x_path.exists() and meta_path.exists() and not force:
        return (np.load(x_path, mmap_mode="r"),
                json.loads(meta_path.read_text()))

    from . import featurise
    from .features import a_flank

    d = load_screen(verbose=verbose)
    g = genome.load_genome()
    seqs, lefts, strands = genome.protospacer_sites()
    # Guides with more than one NGG site in the genome are dropped: their flanks
    # are ambiguous, and Cui's own ntargets column counts targets rather than
    # genomic copies.
    counts = pd.Series(seqs).value_counts()
    site = {s: (l, t) for s, l, t in zip(seqs, lefts, strands)}
    d["n_sites"] = d["guide"].map(counts).fillna(0).astype(int)
    found = d[d["n_sites"] == 1].reset_index(drop=True)
    if verbose:
        print(f"  {len(found):,} located uniquely in NC_000913.2 "
              f"({(d['n_sites'] == 0).sum():,} not found, "
              f"{(d['n_sites'] > 1).sum():,} multi-copy)")

    # Their coordinates are a free consistency check on the assembly -- but the
    # conventions differ, so a naive |ours - theirs| <= 1 test reports 0% and
    # looks like an assembly mismatch. The offset takes exactly two values,
    # +3 and -21, one per strand: they report the PAM-proximal end rather than
    # the protospacer's left edge. Checking that, every guide agrees exactly,
    # which confirms NC_000913.2 and our placement.
    ours = np.array([site[s][0] for s in found["guide"]])
    strand = np.array([site[s][1] for s in found["guide"]])
    off = ours - found["pos"].to_numpy()
    exact = float(np.mean(np.where(strand == 1, off == 3, off == -21)))
    if verbose:
        print(f"  their `pos` matches ours exactly for {exact:.1%} of guides "
              f"(offset +3 on the plus strand, -21 on the minus: they report "
              f"the PAM-proximal end)")
        if exact < 0.99:
            print(f"  ! offsets seen: "
                  f"{pd.Series(off).value_counts().head(5).to_dict()}")

    rows = []
    for s in found["guide"]:
        left, strand = site[s]
        ctx = genome.guide_context(int(left), int(left) + 19,
                                   "+" if strand == 1 else "-", FLANK, g)
        rows.append((ctx["upstream"], ctx["pam"], ctx["downstream"],
                     "+" if strand == 1 else "-", int(left)))
    found[["upstream", "pam", "downstream", "strand", "left"]] = pd.DataFrame(
        rows, index=found.index)
    found["protospacer"] = found["guide"]
    found[config.ID_COL] = found["guide"]
    found["located"] = True
    found["right"] = found["left"] + 19

    base, base_names = featurise.matrix(found["protospacer"].tolist(),
                                        verbose=verbose)
    block = a_flank.build(found)
    block = block.set_index(config.ID_COL).reindex(found["protospacer"])
    vals = block.to_numpy(dtype=np.float32)
    keep = ~np.all(np.isnan(vals), axis=0)
    X = np.hstack([np.asarray(base, dtype=np.float32), vals[:, keep]])
    names = list(base_names) + [c for c, k in zip(block.columns, keep) if k]

    np.save(x_path, X)
    meta_path.write_text(json.dumps(dict(
        y=found["y"].tolist(), fit18=found["fit18"].tolist(),
        fit75=found["fit75"].tolist(), names=names, n_base=len(base_names),
        # pandas 3 keeps missing values through `astype(str)` rather than
        # writing "nan", so they are filled explicitly -- otherwise the gene
        # column comes back as a mix of str and float and every grouped run
        # dies sorting it.
        gene=found["gene"].fillna("").astype(str).tolist(),
        essential=found["essential"].fillna("").astype(str).tolist(),
        left=found["left"].tolist(), protospacers=found["protospacer"].tolist())))
    if verbose:
        print(f"  built {X.shape[0]:,} x {X.shape[1]:,} "
              f"({len(base_names):,} base + {X.shape[1] - len(base_names)} family A)")
    return np.load(x_path, mmap_mode="r"), json.loads(meta_path.read_text())


def run(seeds=(41, 42, 43), n_splits=None, verbose: bool = True):
    from .run_ablation import cross_validate

    n_splits = n_splits or config.N_SPLITS
    X, meta = build(verbose=verbose)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(meta["y"], dtype=float)
    names = list(meta["names"])
    n_base = int(meta["n_base"])
    gene = np.asarray([g if isinstance(g, str) else "" for g in meta["gene"]],
                      dtype=object)

    # Intergenic guides have no gene to hold out, so they are dropped from the
    # grouped arms rather than each forming its own group.
    has_gene = (gene != "") & (gene != "nan") & (gene != "None")
    if verbose:
        print(f"  {has_gene.sum():,} guides inside a named gene "
              f"({len(np.unique(gene[has_gene])):,} genes), "
              f"{(~has_gene).sum():,} intergenic")

    rows = []
    for scheme in ("gene", "random"):
        if scheme == "gene":
            Xs, ys, groups = X[has_gene], y[has_gene], gene[has_gene]
        else:
            Xs, ys, groups = X[has_gene], y[has_gene], None
        # `a_flank` alone is the arm that makes the null clean. Adding 348
        # columns to a pool of 6,169 under a 300-column budget could dilute
        # selection, so a small negative delta is not by itself evidence of
        # nothing; family A on its own cannot be diluted by anything.
        for arm, cols in (("base", slice(0, n_base)),
                          ("base+a_flank", slice(0, X.shape[1])),
                          ("a_flank_only", slice(n_base, X.shape[1]))):
            Xa = Xs[:, cols]
            nm = names[cols]
            res = cross_validate(Xa, ys, nm, ["arm"] * len(nm),
                                 n_features=config.N_FEATURES, seeds=list(seeds),
                                 n_splits=n_splits, verbose=False, groups=groups,
                                 tag=f"cui_{scheme}_{arm.replace('+', '_')}",
                                 experiment=arm)
            rows.append(dict(cv=scheme, arm=arm, n_rows=int(len(ys)),
                             n_cols=Xa.shape[1],
                             spearman=res["summary"]["spearman"]["mean"],
                             spearman_sd=res["summary"]["spearman"]["sd"]))
            if verbose:
                print(f"  {scheme:>6} / {arm:>12}: rho "
                      f"{rows[-1]['spearman']:.4f} ± {rows[-1]['spearman_sd']:.4f}",
                      flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    if verbose:
        for scheme in ("gene", "random"):
            a = out[(out.cv == scheme)].set_index("arm")["spearman"]
            print(f"\n  family A under {scheme:>6} CV: "
                  f"{a['base+a_flank'] - a['base']:+.4f}")
        print(f"written -> {OUT}")
    return out


def cas9_reference(seeds=(41, 42, 43), n_splits=None, verbose: bool = True):
    """The same family-A-only arm on the curated Cas9 cohort, for comparison.

    "Family A does not predict CRISPRi fitness" means nothing without the number
    it is being compared to, measured the same way: family A alone, gene-grouped,
    same seeds and folds, on the cutting screen where it does work.
    """
    from .curated_ablation import _base, _block, _groups
    from .run_ablation import cross_validate

    n_splits = n_splits or config.N_SPLITS
    X, y, names, protos = _base()
    groups, ok = _groups("gene", protos)
    A, anames = _block("a_flank", protos)
    res = cross_validate(A[ok], y[ok], anames, ["a_flank"] * len(anames),
                         n_features=config.N_FEATURES, seeds=list(seeds),
                         n_splits=n_splits, verbose=False, groups=groups[ok],
                         tag="cui_ref_cas9_flank_only", experiment="a_flank_only")
    row = dict(cv="gene", arm="a_flank_only (curated Cas9 cohort)",
               n_rows=int(ok.sum()), n_cols=A.shape[1],
               spearman=res["summary"]["spearman"]["mean"],
               spearman_sd=res["summary"]["spearman"]["sd"])
    if verbose:
        print(f"  Cas9 reference, family A alone: rho {row['spearman']:.4f} "
              f"± {row['spearman_sd']:.4f}")
    prev = pd.read_csv(OUT) if OUT.exists() and OUT.stat().st_size else pd.DataFrame()
    prev = prev[prev.get("arm", pd.Series(dtype=str)) != row["arm"]] if len(prev) else prev
    pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(OUT, index=False)
    return row




# --------------------------------------------------------------------------
# Is the CRISPRi null a null about *binding*, or about *phenotype*?
# --------------------------------------------------------------------------
# Finding 5 reads the CRISPRi null as localising the flank effect to the
# cleavage step: the same features are worth +0.157 on a cutting screen and
# -0.009 here, so binding- and accessibility-limited accounts are excluded.
#
# That inference assumes the two screens differ only in whether a cut is made.
# They do not. A Cas9 depletion screen scores a guide by whether its cells die,
# which works at any locus. A CRISPRi screen scores a guide by the growth change
# from knocking a gene *down*, which requires that the gene be expressed and
# that its level matter. Where it does not, the guide scores ~0 however well
# dCas9 bound -- and the label says that is most of this cohort: essential-gene
# guides have a label sd of 2.886 against 1.133 for non-essential ones, and
# 85% of the cohort is non-essential.
#
# So the CRISPRi null could be a null about the *phenotypic channel* rather than
# about binding, and Finding 5 cannot tell the two apart as it stands. The test:
# re-run the same arm inside the subset where knockdown definitely has a
# phenotype. If the flank effect appears among essential genes, the localisation
# claim is confounded; if it is still absent where the dynamic range is 2.5x
# larger, the claim survives a test it had not faced.
ESS_OUT = config.RESULTS / "cui_essentiality_split.csv"


def _slug(label: str) -> str:
    """A checkpoint tag that distinguishes every arm.

    `cross_validate` resumes from `results/folds_<tag>.csv` and skips folds it
    has already recorded, so two arms sharing a tag silently return the *first*
    arm's numbers. The first version of this took `label.split()[0]`, which gave
    "essential" for both "essential genes" and "essential genes, 100kb blocks" --
    and the block-grouped arm duly reported the gene-grouped result to four
    decimal places. Identical numbers under two different groupings is the only
    reason it was caught.
    """
    return "".join(c if c.isalnum() else "_" for c in label).strip("_")


def essentiality_split(seeds=(41, 42, 43), n_splits=None, verbose: bool = True):
    from .run_ablation import cross_validate

    n_splits = n_splits or config.N_SPLITS
    X, meta = build(verbose=verbose)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(meta["y"], dtype=float)
    names = list(meta["names"])
    n_base = int(meta["n_base"])
    gene = np.asarray([g if isinstance(g, str) else "" for g in meta["gene"]],
                      dtype=object)
    ess = np.asarray([str(e) for e in meta["essential"]], dtype=object)
    has_gene = (gene != "") & (gene != "nan") & (gene != "None")

    left = np.asarray(meta["left"], dtype=float)
    # Essential genes are spatially clustered in E. coli -- ribosomal operons
    # above all -- and in a CRISPRi screen the locus largely sets the phenotype.
    # Flank composition is autocorrelated over kilobases, so holding out whole
    # genes does not stop a model learning "this GC signature is the ribosomal
    # region, where knockdown is lethal" and carrying it to a held-out gene next
    # door. Grouping by 100 kb block is the control that does, and it is the one
    # the rest of the project uses for exactly this risk.
    block = np.floor_divide(left, 100_000).astype(int).astype(str)

    rows = []
    subsets = (("essential genes", has_gene & (ess == "True"), gene),
               ("essential genes, 100kb blocks", has_gene & (ess == "True"), block),
               ("non-essential genes", has_gene & (ess == "False"), gene),
               ("all genic (reference)", has_gene, gene))
    for label, sel, grouping in subsets:
        if sel.sum() < 500:
            continue
        if verbose:
            print(f"  {label}: {sel.sum():,} guides, label sd "
                  f"{np.nanstd(y[sel]):.3f}, "
                  f"{len(np.unique(grouping[sel])):,} groups", flush=True)
        scores = {}
        for arm, cols in (("base", slice(0, n_base)),
                          ("base+a_flank", slice(0, X.shape[1])),
                          ("a_flank_only", slice(n_base, X.shape[1]))):
            res = cross_validate(
                X[sel][:, cols], y[sel], names[cols],
                ["arm"] * len(names[cols]), n_features=config.N_FEATURES,
                seeds=list(seeds), n_splits=n_splits, verbose=False,
                groups=grouping[sel],
                tag=f"cui_ess_{_slug(label)}_{arm.replace('+', '_')}",
                experiment=arm)
            scores[arm] = res["summary"]["spearman"]["mean"]
            rows.append(dict(subset=label, arm=arm, n_rows=int(sel.sum()),
                             label_sd=float(np.nanstd(y[sel])),
                             spearman=scores[arm],
                             spearman_sd=res["summary"]["spearman"]["sd"]))
            if verbose:
                print(f"    {arm:>14}: rho {scores[arm]:+.4f}", flush=True)
            pd.DataFrame(rows).to_csv(ESS_OUT, index=False)
        if verbose:
            print(f"    -> family A delta {scores['base+a_flank'] - scores['base']:+.4f}",
                  flush=True)

    out = pd.DataFrame(rows)
    if verbose:
        print("\n  family A delta by subset:")
        for label, g in out.groupby("subset"):
            d = g.set_index("arm")["spearman"]
            print(f"    {label:24s}: {d['base+a_flank'] - d['base']:+.4f}  "
                  f"(a_flank alone {d['a_flank_only']:+.4f})")
        print(f"written -> {ESS_OUT}")
    return out


# The essentiality split leaves one question open and it is the decisive one:
# *what* is family A predicting inside those essential genes? Not what it
# predicts on a cutting screen. Flank GC at +-500 nt -- the column Finding 2's
# whole compositional account runs through -- has a marginal Spearman of -0.0004
# with this label there. The column that does carry it is
# `eng.flank.comp.up1000.purine` at +0.4045, and purine counted on the guide's
# own strand flips with the guide's orientation, because coding strands are
# purine-rich. So it is a proxy for which strand the guide targets -- which is
# the dominant covariate of CRISPRi efficacy, with coding-strand guides averaging
# -5.07 here against -1.19 off it.
#
# Two controls follow. `strand_split` hands the model the strand call directly;
# if family A was a lossy re-encoding of it, it has nothing left to add.
# `position_split` rules out the other reading, that 348 columns of local
# composition act as a smooth genomic coordinate, by handing over the coordinate.
STRAND_OUT = config.RESULTS / "cui_essential_strand.csv"
POS_OUT = config.RESULTS / "cui_essential_position.csv"


def _essential_frame():
    """The essential-gene subset, with both groupings and the strand call."""
    X, meta = build(verbose=False)
    X = np.asarray(X, dtype=np.float32)
    names = list(meta["names"])
    y = np.asarray(meta["y"], dtype=float)
    n_base = int(meta["n_base"])
    gene = np.asarray([g if isinstance(g, str) else "" for g in meta["gene"]],
                      dtype=object)
    ess = np.asarray([str(e) for e in meta["essential"]], dtype=object)
    has_gene = (gene != "") & (gene != "nan") & (gene != "None")
    left = np.asarray(meta["left"], dtype=float)
    block = np.floor_divide(left, 100_000).astype(int).astype(str)

    raw = pd.read_csv(config.BADSEED / "screen_data.csv")
    raw = raw[raw["ntargets"] == 1].groupby("guide", as_index=False).agg(
        coding=("coding", "first"))
    coding = raw.set_index("guide").reindex(meta["protospacers"])["coding"]
    # 6,027 guides have no coding call; all of them sit outside genes, so none
    # reaches the essential subset. -1 keeps the column free of NaN regardless.
    cod = np.where(coding.isna(), -1.0, coding.astype(float)).astype(np.float32)

    sel = has_gene & (ess == "True")
    return dict(Xb=X[sel][:, :n_base], Xf=X[sel][:, n_base:], y=y[sel],
                coding=cod[sel].reshape(-1, 1),
                left=left[sel].reshape(-1, 1).astype(np.float32),
                nb=names[:n_base], nf=names[n_base:],
                groupings=(("100kb blocks", block[sel]), ("genes", gene[sel])))


def _arm_sweep(arms, out, slug, seeds, n_splits, verbose):
    from .run_ablation import cross_validate

    f = _essential_frame()
    rows = []
    for gname, grouping in f["groupings"]:
        for arm, Xa, nm in arms(f):
            res = cross_validate(
                Xa, f["y"], nm, ["arm"] * len(nm),
                n_features=config.N_FEATURES, seeds=list(seeds),
                n_splits=n_splits or config.N_SPLITS, verbose=False,
                groups=grouping,
                tag=f"cui_{slug}_{gname.replace(' ', '')}_{arm.replace('+', '_')}",
                experiment=arm)
            m = res["summary"]["spearman"]["mean"]
            rows.append(dict(grouping=gname, arm=arm, spearman=m,
                             spearman_sd=res["summary"]["spearman"]["sd"]))
            if verbose:
                print(f"  {gname:12s} {arm:20s}: rho {m:+.4f}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    if verbose:
        print(f"written -> {out}")
    return pd.DataFrame(rows)


def strand_split(seeds=(41, 42, 43), n_splits=None, verbose: bool = True):
    """Does family A add anything once the strand call is given outright?"""
    def arms(f):
        return (("base+coding", np.hstack([f["Xb"], f["coding"]]),
                 f["nb"] + ["screen.coding_strand"]),
                ("coding_only", f["coding"], ["screen.coding_strand"]),
                ("base+coding+a_flank",
                 np.hstack([f["Xb"], f["coding"], f["Xf"]]),
                 f["nb"] + ["screen.coding_strand"] + f["nf"]))
    return _arm_sweep(arms, STRAND_OUT, "essstr", seeds, n_splits, verbose)


def position_split(seeds=(41, 42, 43), n_splits=None, verbose: bool = True):
    """Were the 348 columns acting as a smooth genomic coordinate?"""
    def arms(f):
        return (("coord_only", f["left"], ["genome.left"]),
                ("base+coord", np.hstack([f["Xb"], f["left"]]),
                 f["nb"] + ["genome.left"]))
    return _arm_sweep(arms, POS_OUT, "esspos", seeds, n_splits, verbose)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--essentiality", action="store_true")
    ap.add_argument("--strand", action="store_true")
    ap.add_argument("--position", action="store_true")
    ap.add_argument("--cas9-reference", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.build or not (a.run or a.cas9_reference or a.essentiality
                       or a.strand or a.position):
        build(force=a.force)
    if a.essentiality:
        essentiality_split()
    if a.strand:
        strand_split()
    if a.position:
        position_split()
    if a.run:
        run()
    if a.cas9_reference:
        cas9_reference()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

