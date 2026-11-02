"""Why did every family's gain rise when the baseline rose too?

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

    python -m sgrna.curation_variance
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from . import config

OUT = config.RESULTS / "curation_variance.csv"
PREDICTOR = "eng.flank.up500.gc"


def _curated():
    from .curated_ablation import _base
    from .io_utils import load_block
    X, y, names, protos = _base()
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
        from .build_curated_features import load_index
        idx = load_index(verbose=False)
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


def run(verbose: bool = True):
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
    out.to_csv(OUT, index=False)

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
        print(f"written -> {OUT}")
    return out


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
