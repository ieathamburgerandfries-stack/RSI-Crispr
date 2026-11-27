"""Family A raises held-out rho in every organism it was built for: +0.161 in E. coli, +0.104 for eSpCas9, +0.059 in C. rodentium and +0.023 in human, positive in every fold.

One table, assembled from the stored results of the runs that produced them (nothing is re-fitted here). Each row is a screen, how its folds were made, and what the 348 flank
columns added to the columns that read only the 20-mer ("base"): the held-out Spearman rho of each arm and the paired gain, with how many folds (or seeds) the gain was positive in.

    E. coli SpCas9     whole-gene folds, 5 x 3 seeds        the XGBoost champion          ablation_paper_folds_summary.csv
    E. coli eSpCas9    five contiguous genome arcs, 5 x 3   LightGBM                      transfer.csv (cv = arc5)
    C. rodentium Tev   10 kb bins, 5 x 3                    LightGBM                      transfer.csv (cv = bin10k)
    human (hg38)       folds that keep identical 20-mers    the XGBoost champion          human_shape_ablation_twin.csv
                       together, 5 x 3 (the matrix has no gene identifier)

and two zero-shot transfers between organisms (train on one screen, score the other, no refitting), which are sequence-disjoint: E. coli SpCas9 -> C. rodentium TevSpCas9 and
back (`transfer_model.csv`, LightGBM, 3 seeds). The sizes of the gains are not comparable across rows (different enzymes, assays, models and fold designs); the sign is. Mouse and
zebrafish tables, and the dCas9 screens, are not here: the first have a 77 nt window and no genome position to read 1 kb of flank from, the second measure another quantity.

Written to `results/post_freeze/family_a_by_organism.csv`.

    ./run_ext.sh sgrna.compare.family_a_by_organism
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import OUT

RESULTS = OUT.parent


def _folds(df: pd.DataFrame, dataset: str, cv: str) -> tuple[pd.Series, pd.Series]:
    s = df[(df.dataset == dataset) & (df.cv == cv)]
    p = s.pivot_table(index=["seed", "fold"], columns="arm", values="spearman")
    return p["base"], p["base+a_flank"]


def _row(organism, screen, guides, protocol, model, unit, base, withA) -> dict:
    gain = np.asarray(withA, float) - np.asarray(base, float)
    return dict(organism=organism, screen=screen, guides=guides, protocol=protocol, model=model, unit=unit, n=len(gain),
                base_rho=float(np.mean(base)), with_family_A_rho=float(np.mean(withA)), gain=float(gain.mean()),
                gain_sd=float(gain.std(ddof=1)), n_positive=int((gain > 0).sum()), gain_min=float(gain.min()))


def main() -> int:
    rows = []
    # E. coli SpCas9, the paper's whole-gene folds: only the summary (means over the 15 folds, and how many were positive) is stored
    s = pd.read_csv(OUT / "ablation_paper_folds_summary.csv").set_index("family").loc["a_flank"]
    rows.append(dict(organism="E. coli", screen="SpCas9 (curated)", guides=33567, protocol="whole-gene folds", model="XGBoost champion", unit="folds", n=int(s.n_folds),
                     base_rho=float(s.base_selected_rho), with_family_A_rho=float(s.rho), gain=float(s.gain_same_selection), gain_sd=float(s.gain_same_selection_sd),
                     n_positive=int(s.folds_positive_same_selection), gain_min=np.nan))
    t = pd.read_csv(RESULTS / "transfer.csv")
    for screen, ds, cv, guides, protocol in (("eSpCas9", "eSpCas9", "arc5", 59489, "five contiguous genome arcs"), ("TevSpCas9", "TevSpCas9", "bin10k", 25210, "10 kb bins")):
        b, a = _folds(t, ds, cv)
        rows.append(_row("E. coli" if ds == "eSpCas9" else "C. rodentium", screen, guides, protocol, "LightGBM", "folds", b.to_numpy(), a.to_numpy()))
    h = pd.read_csv(OUT / "human_shape_ablation_twin.csv").set_index("arm")
    hb = np.array([float(x) for x in h.loc["h_base", "fold_rho"].split(";")])
    ha = np.array([float(x) for x in h.loc["h_base+A", "fold_rho"].split(";")])
    rows.append(_row("human", "Noshay et al. matrix on hg38", int(h.loc["h_base", "n"]), "folds keeping identical 20-mers together", "XGBoost champion", "folds", hb, ha))
    m = pd.read_csv(RESULTS / "transfer_model.csv")
    for src, dst, org, name in (("WT-SpCas9", "TevSpCas9", "C. rodentium", "E. coli SpCas9 -> C. rodentium TevSpCas9"), ("TevSpCas9", "WT-SpCas9", "E. coli", "C. rodentium TevSpCas9 -> E. coli SpCas9")):
        g = m[(m.source == src) & (m.target == dst) & m.arm.isin(["base", "base+a_flank"])].pivot_table(index="seed", columns="arm", values="spearman")
        rows.append(_row(org, name, int(m[(m.source == src) & (m.target == dst)].n_test.iloc[0]), "zero-shot transfer, sequence-disjoint", "LightGBM", "seeds", g["base"].to_numpy(), g["base+a_flank"].to_numpy()))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "family_a_by_organism.csv", index=False)
    print(out.drop(columns=["protocol", "model"]).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
