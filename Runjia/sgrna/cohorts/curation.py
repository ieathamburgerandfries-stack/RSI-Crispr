"""Curation does not create the flank effect, and the larger cohort's bigger gains are a steeper flank effect in its genome-wide guides (slope 15.85 against 5.35 on shared guides), not a wider spread of the data.

crisprHAL 2's curated cohort (33,567 guides) replaced Noshay's published 13,880-row matrix, every family's gain roughly tripled, and the base went 0.5278 to 0.5445. Two questions a sceptic asks of that, each its own part:

  check      does curation itself create the flank effect? The discarded guides with the most reads still show it (flank GC correlation 0.159, against 0.175 to 0.188 in kept guides)    -> curation_check.csv
  variance   why did the gains rise: a wider spread of the predictor, or a steeper relationship? The genome-wide guides added in the larger cohort show a steeper one                        -> curation_variance.csv

    python -m sgrna.cohorts.curation --parts check variance

------------------------------------------------------------------------------------------------------------------------------
check
------------------------------------------------------------------------------------------------------------------------------
crisprHAL 2's curation (`external_data/crisprHAL/rawData/generate_WTSpCas9_cutoff_datasets.R`) keeps the sgRNAs of Guo et al.'s screen that lie at a
target site in the E. coli genome and whose mean count in the two dead-nuclease (unselected) replicates is at least a cutoff (73 for the 33,567-guide set
used here, from 121,340 in the raw counts), then scores them with ALDEx2. The cutoff looks at the control counts only, not at the Cas9-selected counts or at
any score. This module asks two things a sceptic would ask of that:

  inclusion   does the chance of being kept depend on the guide's flank (so that curation could have enriched or removed the signal)?
  discarded   is the flank association present in the guides that were thrown away, scored without ALDEx2 (minus the plain log2 ratio of Cas9-selected to
              unselected counts, mean of the two replicate pairs, pseudocount 0.5, each pair centred on its median; higher = more cutting as in the released label)? It will be weaker there, because
              low counts make the label noisy, but a flank effect made by the curation would not appear in guides it never touched.

Strata of the mean control count: 1-9, 10-29, 30-72 (discarded) and 73-108, 109-168, 169+ (kept, the thirds used in the depth control). Per stratum: the Spearman
correlation of the plain label with flank GC (the mean of the GC fraction of the 250 nt upstream and 250 nt downstream, on the guide's strand) and the
deficit of the lowest decile of flank GC against the rest in SD units of the stratum's label. Guides are those with exactly one NGG site in the genome.

------------------------------------------------------------------------------------------------------------------------------
variance
------------------------------------------------------------------------------------------------------------------------------
The objection is sharp and has not been answered properly. Removing label noise
should raise the baseline and *shrink* each family's increment, because there is
less headroom left. What happened is the opposite: the base went 0.5278 -> 0.5445
and every family's increment roughly tripled. Two things cannot both be
attenuation.

They are not. Appendix B2a establishes that the label swap is worth +0.004 at
fixed rows, so curation is not what moved the table -- the cohort is a different,
larger draw. But "different draw" is not yet a mechanism. This module supplies
one, and it is testable.

**The claim.** For a relationship y = beta*x + noise, the correlation is
beta * sd(x) / sd(y). A wider spread of x raises the measured correlation
*without any change in the underlying biology*. Noshay's rows are a 1,222-gene
panel; crisprHAL's extra 24,950 guides are genome-wide. A genome-wide draw spans
more of the chromosome, so it spans more flank composition -- more sd(x).

**The test.** Both row sets are scored by the same curated label, so the units
match and beta is comparable. Then:

    beta similar, sd(x) larger  =>  the biology is the same and the new rows
                                    merely expose more of it
    beta larger                 =>  the new rows carry a genuinely stronger
                                    relationship, and "more variance" is wrong

Run on the 8,617 guides both cohorts contain against the 24,950 only crisprHAL
publishes.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .. import config
from ..common import OUT

CHECK_OUT = OUT / "curation_check.csv"
VARIANCE_OUT = OUT / "curation_variance.csv"


# ======================================================================================================  check
STRATA = [("1-9", 1, 10), ("10-29", 10, 30), ("30-72", 30, 73), ("73-108", 73, 109), ("109-168", 109, 169), ("169+", 169, 1e9)]


def _guides():
    from ..build import genome

    d = pd.read_csv(config.EXTERNAL / "crisprHAL" / "rawData" / "E_coli_WTSpCas9_counts.tsv", sep="\t")
    seqs, lefts, strands = genome.protospacer_sites()
    s = pd.DataFrame({"sgRNA": seqs, "left": lefts, "strand": np.where(strands == 1, "+", "-")})
    n = s.groupby("sgRNA").size()
    s = s[s.sgRNA.map(n) == 1]
    m = d.merge(s, on="sgRNA", how="inner")
    m["right"] = m["left"] + 19
    return d, m


def check() -> int:
    from ..interpret.flank_scale import flank_composition

    raw, m = _guides()
    ctl = (m.dCas1_NS + m.dCas2_NS) / 2
    comp = flank_composition(m[["left", "right", "strand"]].reset_index(drop=True), (250,))
    up, dn = comp[250]
    fgc = ((up[:, 1] + up[:, 2]) + (dn[:, 1] + dn[:, 2])) / 2
    lr = []
    for c, n in (("Cas1_S", "dCas1_NS"), ("Cas2_S", "dCas2_NS")):
        v = np.log2((m[c] + 0.5) / (m[n] + 0.5)).to_numpy()
        lr.append(v - np.median(v))
    y = -np.mean(lr, axis=0)                     # the log ratio of selected to unselected counts is high where a guide survives, so its negative is the cutting score (higher = more cutting), the orientation of the released label
    edge = np.quantile(fgc, 0.10)
    low = fgc <= edge
    rows = []
    kept = ctl.to_numpy() >= 73
    # inclusion by decile of flank GC
    dec = np.clip(np.searchsorted(np.quantile(fgc, np.linspace(0, 1, 11)), fgc, side="right") - 1, 0, 9)
    incl = [float(kept[dec == k].mean()) for k in range(10)]
    for name, lo, hi in STRATA:
        mk = (ctl.to_numpy() >= lo) & (ctl.to_numpy() < hi)
        yy, ff, ll = y[mk], fgc[mk], low[mk]
        rows.append(dict(stratum=name, n=int(mk.sum()), kept=bool(lo >= 73), spearman_flank_gc=float(spearmanr(ff, yy).statistic),
                         label_sd=float(yy.std()), lowest_decile_deficit_sd=float((yy[ll].mean() - yy[~ll].mean()) / yy.std()),
                         share_in_lowest_decile=float(ll.mean())))
    out = pd.DataFrame(rows)
    out["inclusion_by_decile_min"] = min(incl)
    out["inclusion_by_decile_max"] = max(incl)
    out["raw_guides"] = len(raw)
    out["single_site_guides"] = len(m)
    out["kept_guides"] = int(kept.sum())
    out.to_csv(CHECK_OUT, index=False)
    print(f"raw {len(raw):,}; single-site {len(m):,}; kept (control >= 73) {int(kept.sum()):,}")
    print("P(kept) by decile of flank GC:", [round(x, 3) for x in incl])
    print(out.drop(columns=["inclusion_by_decile_min", "inclusion_by_decile_max", "raw_guides", "single_site_guides", "kept_guides"]).round(3).to_string(index=False))
    return 0


# ======================================================================================================  variance
PREDICTOR = "eng.flank.up500.gc"


def _curated():
    from ..harness.curated import load_base
    from ..io_utils import load_block
    X, y, names, protos = load_base()
    a = load_block("a_flank_curated")
    key = "protospacer" if "protospacer" in a.columns else a.columns[0]
    a = a.set_index(key)
    a.index = a.index.astype(str)
    a = a.reindex(pd.Index([str(p) for p in protos]))
    col = next((c for c in a.columns if c.endswith("up500.gc")), None)
    if col is None:
        raise SystemExit(f"no up500.gc column in a_flank_curated: {list(a.columns)[:8]}")
    return pd.DataFrame({"protospacer": protos, "x": a[col].to_numpy(float),
                         "y": y})


def _noshay_protospacers() -> set[str]:
    d = pd.read_csv(config.GUIDE_INDEX, usecols=["protospacer"]).dropna()
    return set(d["protospacer"])


def _panel_split(shared, only):
    """Split the crisprHAL-only rows by whether their gene is in Noshay's panel."""
    try:
        from ..build.build_features import load_curated_index
        idx = load_curated_index(verbose=False)
    except Exception:                                  # noqa: BLE001
        return None, None, None
    key = "protospacer"
    if key not in idx.columns or "gene_name" not in idx.columns:
        return None, None, None
    gene = idx.dropna(subset=[key]).set_index(key)["gene_name"]
    gene.index = gene.index.astype(str)
    panel = set(gene.reindex(shared["protospacer"]).dropna())
    g_only = gene.reindex(only["protospacer"])
    in_panel = g_only.isin(panel).to_numpy()
    return panel, only[in_panel], only[~in_panel]


def _gene_counts(protos) -> tuple[int, float]:
    d = pd.read_csv(config.GUIDE_INDEX, usecols=["protospacer", "gene_name"])
    d = d.dropna(subset=["protospacer"])
    sub = d[d["protospacer"].isin(set(protos))]
    g = sub["gene_name"].dropna()
    return int(g.nunique()), round(len(g) / max(g.nunique(), 1), 1)


def variance(verbose: bool = True):
    d = _curated().dropna(subset=["x", "y"])
    nos = _noshay_protospacers()
    shared = d[d["protospacer"].isin(nos)]
    only = d[~d["protospacer"].isin(nos)]

    # Is the weaker relationship a property of Noshay's *genes* or of the
    # particular guides he published inside them? Split the crisprHAL-only rows
    # by whether their host gene is one of the panel's.
    panel_genes, same_gene, other_gene = _panel_split(shared, only)

    rows = []
    for name, sub in (("shared with Noshay", shared),
                      ("crisprHAL only, panel genes", same_gene),
                      ("crisprHAL only, other genes", other_gene),
                      ("crisprHAL only", only),
                      ("whole curated cohort", d)):
        if sub is None or len(sub) < 200:
            continue
        x, y = sub["x"].to_numpy(float), sub["y"].to_numpy(float)
        beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
        rows.append(dict(
            rows=name, n=len(sub),
            sd_x=round(float(np.std(x, ddof=1)), 4),
            sd_y=round(float(np.std(y, ddof=1)), 4),
            slope_beta=round(beta, 4),
            pearson=round(float(pearsonr(x, y).statistic), 4),
            spearman=round(float(spearmanr(x, y).statistic), 4),
            predicted_pearson=round(beta * np.std(x, ddof=1)
                                    / np.std(y, ddof=1), 4)))
        if verbose:
            r = rows[-1]
            print(f"  {name:>22}  n {r['n']:>6,}  sd(x) {r['sd_x']:.4f}  "
                  f"sd(y) {r['sd_y']:.4f}  beta {r['slope_beta']:+.4f}  "
                  f"r {r['pearson']:+.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(VARIANCE_OUT, index=False)

    if verbose:
        s, o = rows[0], rows[1]
        print(f"\n  sd(x) ratio  {o['sd_x'] / s['sd_x']:.2f}x")
        print(f"  sd(y) ratio  {o['sd_y'] / s['sd_y']:.2f}x")
        print(f"  beta  ratio  {o['slope_beta'] / s['slope_beta']:.2f}x")
        print(f"  r     ratio  {o['pearson'] / s['pearson']:.2f}x")
        try:
            for nm, sub in (("shared", shared), ("crisprHAL only", only)):
                g, per = _gene_counts(sub["protospacer"])
                print(f"  {nm}: {g:,} genes located in Noshay's index, "
                      f"{per} guides each")
        except Exception as exc:                       # noqa: BLE001
            print(f"  (gene counts unavailable: {exc})")
        print(f"written -> {VARIANCE_OUT}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", nargs="+", choices=("check", "variance"), required=True)
    a = ap.parse_args(argv)
    if "check" in a.parts:
        check()
    if "variance" in a.parts:
        variance()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
