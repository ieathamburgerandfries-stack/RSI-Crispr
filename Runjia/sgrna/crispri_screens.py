"""Three CRISPRi screens instead of one, and what that buys beyond accuracy.

Every dCas9 claim in this report rests on **one** screen. Finding 3's
cross-assay test, the essentiality split that nearly retired it, and the strand
result that rescued it are all Cui et al. The obvious risk is that a single
screen from a single laboratory is being read as a property of CRISPRi.

Barquist et al. (2024) assembled three *E. coli* MG1655 CRISPRi screens for a
mixed-effect guide-efficiency model, with the processed tables in
`BarquistLab/CRISPRi_guide_efficiency_bacteria/0_Datasets`:

    E75_Rousset    59,246 guides, 4,371 genes -- never used in this project
    E18_Cui        59,246 guides, the *same library* under a stronger dCas9
                   promoter, so Cui-vs-Rousset varies dosage and nothing else
    Wang_dataset   52,104 guides, a different library, 6,139 shared with the
                   other two

Their tables carry `coding_strand` and `gene_essentiality`, which is what makes
the strand result testable on data this project did not produce.

Three things follow that are not accuracy numbers.

1. **A transferable-signal ceiling.** The replicate ceiling in Finding 6 (0.946)
   measures noise *inside* one experiment. Agreement *between* screens measures
   how much of the signal is guide-intrinsic rather than specific to a library,
   a promoter, a medium or a laboratory. That is a different and lower ceiling,
   and it bounds what any model trained on one screen can score on another.
2. **A replication of the strand result** on a screen this project has never
   touched, in a table produced by someone else's pipeline.
3. **An untouched external set, which Part 4 says is permanently gone.** Rousset
   and Wang are untouched *by this project*, which restores for the dCas9 claims
   exactly what eSpCas9 and Cui stopped providing.

    python -m sgrna.crispri_screens --agreement
    python -m sgrna.crispri_screens --strand
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config

DIR = config.EXTERNAL / "crispri_screens"
FILES = {"Rousset": "E75_Rousset.tsv", "Cui": "E18_Cui.tsv",
         "Wang": "Wang_dataset.tsv"}
AGREEMENT_OUT = config.RESULTS / "crispri_screen_agreement.csv"
STRAND_OUT = config.RESULTS / "crispri_screen_strand.csv"


def load(name: str) -> pd.DataFrame:
    """One screen, with this project's column names.

    Their `log2FC` is negative for a depleted (effective) guide, which is the
    same sign convention as crisprHAL's curated score and the opposite of
    Noshay's `cut_score`. It is left as published.
    """
    path = DIR / FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch the three tables from "
            "raw.githubusercontent.com/BarquistLab/"
            "CRISPRi_guide_efficiency_bacteria/main/0_Datasets/")
    d = pd.read_csv(path, sep="\t")
    keep = ["sequence", "PAM", "log2FC", "geneid", "genename",
            "gene_essentiality", "guide_strand", "coding_strand",
            "distance_start_codon", "guide_GC_content", "gene_expression_max"]
    d = d[[c for c in keep if c in d.columns]].copy()
    d = d.rename(columns={"sequence": "guide", "log2FC": "y",
                          "geneid": "gene",
                          "gene_essentiality": "essential"})
    d["screen"] = name
    return d


def agreement(verbose: bool = True):
    """How much of a guide's measured efficiency survives a change of screen?"""
    from scipy.stats import spearmanr

    d = {n: load(n) for n in FILES}
    rows = []
    for n, t in d.items():
        rows.append(dict(pair=f"{n} (self)", a=n, b=n, n_shared=len(t),
                         subset="all", spearman=np.nan,
                         label_sd=float(t.y.std()),
                         n_essential=int((t.essential == 1).sum())))
    names = list(FILES)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ia = d[a].drop_duplicates("guide").set_index("guide")
            ib = d[b].drop_duplicates("guide").set_index("guide")
            shared = sorted(set(ia.index) & set(ib.index))
            ia, ib = ia.reindex(shared), ib.reindex(shared)
            ess = (ia.essential == 1).to_numpy()
            for subset, mask in (("all", np.ones(len(shared), bool)),
                                 ("essential genes", ess),
                                 ("non-essential genes", ~ess)):
                if mask.sum() < 50:
                    continue
                r = spearmanr(ia.y.to_numpy()[mask], ib.y.to_numpy()[mask],
                              nan_policy="omit").statistic
                rows.append(dict(pair=f"{a} vs {b}", a=a, b=b,
                                 n_shared=int(mask.sum()), subset=subset,
                                 spearman=float(r),
                                 label_sd=float(np.nanstd(ia.y.to_numpy()[mask])),
                                 n_essential=int(ess.sum())))
                if verbose:
                    print(f"  {a:8s} vs {b:8s} {subset:20s}: "
                          f"{int(mask.sum()):6,} guides, rho {r:+.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(AGREEMENT_OUT, index=False)
    if verbose:
        print(f"written -> {AGREEMENT_OUT}")
    return out


def strand(verbose: bool = True):
    """Does the strand result replicate on screens this project did not make?"""
    from scipy.stats import spearmanr

    rows = []
    for n in FILES:
        t = load(n)
        for subset, sel in (("essential genes", t.essential == 1),
                            ("non-essential genes", t.essential == 0),
                            ("all", pd.Series(True, index=t.index))):
            e = t[sel]
            cs = e.coding_strand.to_numpy(dtype=float)
            if np.nanstd(cs) == 0:
                rho = np.nan           # Wang's library is coding-strand only
            else:
                rho = float(spearmanr(cs, e.y, nan_policy="omit").statistic)
            g = e.groupby("coding_strand").y.agg(["size", "mean"])
            rows.append(dict(
                screen=n, subset=subset, n=int(len(e)),
                spearman_coding_strand=rho,
                n_coding=int(g.loc[1, "size"]) if 1 in g.index else 0,
                mean_y_coding=float(g.loc[1, "mean"]) if 1 in g.index else np.nan,
                n_noncoding=int(g.loc[0, "size"]) if 0 in g.index else 0,
                mean_y_noncoding=float(g.loc[0, "mean"]) if 0 in g.index else np.nan,
                spearman_distance_start=float(
                    spearmanr(e.distance_start_codon, e.y,
                              nan_policy="omit").statistic)))
            if verbose:
                r = rows[-1]
                print(f"  {n:8s} {subset:20s} n={r['n']:6,}  "
                      f"rho(coding_strand) {r['spearman_coding_strand']:+.4f}  "
                      f"mean y {r['mean_y_coding']:+.2f} on strand vs "
                      f"{r['mean_y_noncoding']:+.2f} off", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(STRAND_OUT, index=False)
    if verbose:
        print(f"written -> {STRAND_OUT}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agreement", action="store_true")
    ap.add_argument("--strand", action="store_true")
    ap.add_argument("--family-a", nargs="*", default=None,
                    metavar="SCREEN")
    a = ap.parse_args(argv)
    if a.agreement or not (a.strand or a.family_a is not None):
        agreement()
    if a.strand:
        strand()
    if a.family_a is not None:
        for n in (a.family_a or ["Rousset", "Wang"]):
            family_a(n)
    return 0


# --------------------------------------------------------------------------
# The replication that matters: does family A add anything on a screen this
# project has never touched?
# --------------------------------------------------------------------------
# Finding 3's localisation rests on one screen. On Cui, family A adds -0.009
# across the whole genic cohort and +0.007 inside the essential genes once the
# strand call is supplied. Rousset is the same library under a different dCas9
# promoter and Wang is a different library; neither has ever been featurised
# here. If the null reproduces on them, the claim stops being a property of one
# dataset.
FAMILY_A_OUT = config.RESULTS / "crispri_screen_family_a.csv"
FLANK = 1000


def build(name: str, force: bool = False, verbose: bool = True):
    """Locate this screen's guides in NC_000913.2, cut flanks, featurise.

    Same path as `cui_screen.build`, which is what makes the comparison valid:
    the guides differ, nothing else does.
    """
    import json

    from . import featurise, genome
    from .features import a_flank

    cache = config.INTERIM / "crispri_screens" / name
    cache.mkdir(parents=True, exist_ok=True)
    x_path, meta_path = cache / "X.npy", cache / "meta.json"
    if x_path.exists() and meta_path.exists() and not force:
        return np.load(x_path, mmap_mode="r"), json.loads(meta_path.read_text())

    d = load(name).drop_duplicates("guide").reset_index(drop=True)
    d["guide"] = d["guide"].str.upper()
    g = genome.load_genome()
    seqs, lefts, strands = genome.protospacer_sites()
    counts = pd.Series(seqs).value_counts()
    site = {s: (l, t) for s, l, t in zip(seqs, lefts, strands)}
    d["n_sites"] = d["guide"].map(counts).fillna(0).astype(int)
    found = d[d["n_sites"] == 1].reset_index(drop=True)
    if verbose:
        print(f"  {name}: {len(found):,} of {len(d):,} located uniquely "
              f"({int((d['n_sites'] == 0).sum()):,} not found, "
              f"{int((d['n_sites'] > 1).sum()):,} multi-copy)", flush=True)

    rows = []
    for s in found["guide"]:
        left, st = site[s]
        ctx = genome.guide_context(int(left), int(left) + 19,
                                   "+" if st == 1 else "-", FLANK, g)
        rows.append((ctx["upstream"], ctx["pam"], ctx["downstream"],
                     "+" if st == 1 else "-", int(left)))
    found[["upstream", "pam", "downstream", "strand", "left"]] = pd.DataFrame(
        rows, index=found.index)
    found["protospacer"] = found["guide"]
    found[config.ID_COL] = found["guide"]
    found["located"] = True
    found["right"] = found["left"] + 19

    base, base_names = featurise.matrix(found["protospacer"].tolist(),
                                        verbose=verbose)
    block = a_flank.build(found).set_index(config.ID_COL).reindex(
        found["protospacer"])
    vals = block.to_numpy(dtype=np.float32)
    keep = ~np.all(np.isnan(vals), axis=0)
    X = np.hstack([np.asarray(base, dtype=np.float32), vals[:, keep]])
    names = list(base_names) + [c for c, k in zip(block.columns, keep) if k]

    np.save(x_path, X)
    meta_path.write_text(json.dumps(dict(
        y=found["y"].tolist(), names=names, n_base=len(base_names),
        gene=found["gene"].fillna("").astype(str).tolist(),
        essential=found["essential"].fillna(-1).astype(int).tolist(),
        coding_strand=found["coding_strand"].fillna(-1).astype(float).tolist(),
        left=found["left"].tolist(),
        protospacers=found["protospacer"].tolist())))
    if verbose:
        print(f"  built {X.shape[0]:,} x {X.shape[1]:,}", flush=True)
    return np.load(x_path, mmap_mode="r"), json.loads(meta_path.read_text())


def family_a(name: str, seeds=(41, 42, 43), n_splits=None, verbose: bool = True):
    """The Cui arms, re-run on another screen: with and without the strand call."""
    from .run_ablation import cross_validate

    n_splits = n_splits or config.N_SPLITS
    X, meta = build(name, verbose=verbose)
    X = np.asarray(X, dtype=np.float32)
    names = list(meta["names"])
    y = np.asarray(meta["y"], dtype=float)
    n_base = int(meta["n_base"])
    gene = np.asarray(meta["gene"], dtype=object)
    ess = np.asarray(meta["essential"], dtype=int)
    cod = np.asarray(meta["coding_strand"], dtype=np.float32).reshape(-1, 1)
    left = np.asarray(meta["left"], dtype=float)
    block = np.floor_divide(left, 100_000).astype(int).astype(str)
    has_gene = gene != ""

    rows = []
    subsets = (("all genic", has_gene, gene),
               ("essential genes", has_gene & (ess == 1), gene),
               ("essential genes, 100kb blocks", has_gene & (ess == 1), block),
               ("non-essential genes", has_gene & (ess == 0), gene))
    for label, sel, grouping in subsets:
        if sel.sum() < 500:
            continue
        if verbose:
            print(f"  {name} / {label}: {int(sel.sum()):,} guides, "
                  f"label sd {np.nanstd(y[sel]):.3f}", flush=True)
        arms = (("base", X[sel][:, :n_base], names[:n_base]),
                ("base+a_flank", X[sel], names),
                ("base+strand", np.hstack([X[sel][:, :n_base], cod[sel]]),
                 names[:n_base] + ["screen.coding_strand"]),
                ("base+strand+a_flank",
                 np.hstack([X[sel][:, :n_base], cod[sel], X[sel][:, n_base:]]),
                 names[:n_base] + ["screen.coding_strand"] + names[n_base:]))
        for arm, Xa, nm in arms:
            import zlib
            tag = (f"cris_{name}_{zlib.crc32(label.encode()):08x}_"
                   f"{arm.replace('+', '_')}")
            res = cross_validate(
                Xa, y[sel], nm, ["arm"] * len(nm),
                n_features=config.N_FEATURES, seeds=list(seeds),
                n_splits=n_splits, verbose=False, groups=grouping[sel],
                tag=tag, experiment=arm)
            rows.append(dict(screen=name, subset=label, arm=arm,
                             n_rows=int(sel.sum()), n_cols=Xa.shape[1],
                             spearman=res["summary"]["spearman"]["mean"],
                             spearman_sd=res["summary"]["spearman"]["sd"]))
            if verbose:
                print(f"    {arm:>22}: rho {rows[-1]['spearman']:+.4f}",
                      flush=True)
            pd.DataFrame(rows).to_csv(FAMILY_A_OUT, index=False)

    out = pd.DataFrame(rows)
    if verbose:
        for subset, g in out.groupby("subset"):
            s = g.set_index("arm")["spearman"]
            if {"base", "base+a_flank"} <= set(s.index):
                print(f"  {subset}: family A {s['base+a_flank'] - s['base']:+.4f}"
                      + (f", given strand "
                         f"{s['base+strand+a_flank'] - s['base+strand']:+.4f}"
                         if "base+strand" in s.index else ""))
        print(f"written -> {FAMILY_A_OUT}")
    return out

if __name__ == "__main__":
    raise SystemExit(main())
