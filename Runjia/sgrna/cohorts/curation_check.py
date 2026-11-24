"""Curation does not create the flank effect: the discarded guides with the most reads still show it (flank GC correlation 0.159, against 0.175 to 0.188 in kept guides). The flank association in the guides the curation discarded.

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

    python -m sgrna.cohorts.curation_check
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..common import OUT

RESULT = OUT / "curation_check.csv"
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


def main() -> int:
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
    out.to_csv(RESULT, index=False)
    print(f"raw {len(raw):,}; single-site {len(m):,}; kept (control >= 73) {int(kept.sum()):,}")
    print("P(kept) by decile of flank GC:", [round(x, 3) for x in incl])
    print(out.drop(columns=["inclusion_by_decile_min", "inclusion_by_decile_max", "raw_guides", "single_site_guides", "kept_guides"]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
