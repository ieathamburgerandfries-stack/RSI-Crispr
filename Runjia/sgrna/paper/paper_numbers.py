"""Audit of every number in the paper (point form): each is recomputed from a stored result file, as `verify_numbers` does for the report.

    ./run_ext.sh sgrna.paper.paper_numbers

The checks are in three modules by topic (A: data, features, the flank signal and the hypotheses tested; B: transfer, design rules across
kingdoms, the label's ceiling, interpretability; C: the comparison with crisprHAL 2, cost, adaptive overfitting). Each states, next to
the number, which cohort and protocol it belongs to, because a value often recurs for a different quantity (for example +0.164 is random-row
cross-validation on the curated cohort and +0.157 the gene-grouped figure). This module adds the few checks that were missing after the
text was corrected. `verify_numbers` calls `run` so `./run_audit.sh` covers the paper too.

Not recomputable here, and cited instead: the replicate correlations of 0.71 to 0.77 and the field's "about 0.7" (Haeussler et al., 2016;
Konstantakos et al., 2022), and the 236 kb size of the C. rodentium screen (Ham et al., 2023).
"""
from __future__ import annotations

import json

import pandas as pd

from ..common import OUT
from . import paper_numbers_A, paper_numbers_B, paper_numbers_C

P = "paper: "


def run(chk):
    R = OUT.parent
    paper_numbers_A.run(chk)
    paper_numbers_B.run(chk)
    paper_numbers_C.run(chk)
    # the published matrix's guides lie in a 1,222-gene panel
    cg = pd.read_csv(R / "cohort_groups.csv")
    chk(P + "genes in the published matrix's panel (cohort_groups.csv, gene grouping)",
        float(cg[(cg.cohort == "published") & (cg.grouping == "gene")].n_groups.iloc[0]), 1222, 0)
    # which way the label points: more cutting scores higher, shown by the guides with more than one genomic copy
    lo = pd.read_csv(OUT / "label_orientation.csv").iloc[0]
    chk(P + "guides with more than one genomic copy (label_orientation.csv)", float(lo.n_multi_copy), 118, 0)
    chk(P + "   their median label", float(lo.median_label_multi_copy), 2.11, 2)
    chk(P + "   median label of single-copy guides", float(lo.median_label_single_copy), 0.27, 2)
    # sealed-gene drop with whole genes held out inside the pool as well (first eight sealings; sealed_holdout_gene.py)
    sg = pd.read_csv(OUT / "sealed_holdout_gene.csv")
    pv = sg.pivot(index="seal_seed", columns="arm", values=["gap_gene", "gap_random"])
    rnd = pv["gap_random"]["frozen"] - pv["gap_random"]["base_only"]
    gen = pv["gap_gene"]["frozen"] - pv["gap_gene"]["base_only"]
    chk(P + "sealings with the whole-gene pool control (of the 15)", float(len(rnd)), 8, 0)
    chk(P + "sealed-gene drop, random-row pool folds, those sealings", float(rnd.mean()), 0.0088, 4)
    chk(P + "sealed-gene drop, whole-gene pool folds, those sealings", float(gen.mean()), 0.0081, 4)
    chk(P + "   the part that random-row pool folds account for", float((rnd - gen).mean()), 0.0007, 4)
    # crisprHAL 2 retrained at four input widths (crisprhal_wide_window_folds.csv; Colab fold 1, local CPU folds 2-5), against SLICER on the same folds
    ws = pd.read_csv(OUT.parent / "crisprhal_wide_window_summary.csv").set_index("arm")
    wf = pd.read_csv(OUT.parent / "crisprhal_wide_window_folds.csv")
    nf = wf.groupby("arm").fold.nunique()
    chk(P + "width: folds with the 23 nt arm / the 378 nt arm / the 1,023 nt arm / the 2,023 nt arm",
        float(nf["core_23"] * 1000 + nf["native_378"] * 100 + nf["mid_1023"] * 10 + nf["wide_2023"]), 5311, 0)
    chk(P + "width: crisprHAL 2 at 23 nt, mean over its folds", float(ws.loc["core_23", "crisprhal_mean"]), 0.5556, 4)
    chk(P + "width:   SLICER, 20-mer representation, same folds", float(ws.loc["core_23", "slicer_mean"]), 0.5440, 4)
    chk(P + "width:   crisprHAL 2 lead at 23 nt", -float(ws.loc["core_23", "lead_mean"]), 0.012, 3)
    chk(P + "width:   folds (of 5) in which crisprHAL 2 is ahead at 23 nt", float(5 - ws.loc["core_23", "folds_slicer_ahead"]), 5, 0)
    chk(P + "width: crisprHAL 2 at 378 nt, mean over its three folds", float(ws.loc["native_378", "crisprhal_mean"]), 0.6772, 4)
    chk(P + "width:   its across-fold sd", float(ws.loc["native_378", "crisprhal_sd"]), 0.016, 3)
    chk(P + "width:   SLICER matched to 378 nt, same folds", float(ws.loc["native_378", "slicer_mean"]), 0.6675, 4)
    chk(P + "width:   crisprHAL 2 lead at 378 nt", -float(ws.loc["native_378", "lead_mean"]), 0.010, 3)
    chk(P + "width:   folds (of 3) in which crisprHAL 2 is ahead at 378 nt", float(3 - ws.loc["native_378", "folds_slicer_ahead"]), 2, 0)
    hh = pd.read_csv(OUT.parent / "headtohead.csv")
    hh = hh[hh.seed == 41].pivot_table(index="fold", columns="arm", values="spearman")
    w3 = wf[wf.arm == "native_378"].set_index("fold").spearman
    chk(P + "width:   crisprHAL 2 lead at 378 nt on fold 3", float(w3[3] - hh.loc[3, "curated_matched"]), -0.013, 3)
    chk(P + "width: 23 nt arm minus 378 nt arm, mean over their three common folds", float(ws.loc["core_23", "vs_378_mean"]), -0.122, 3)
    chk(P + "width: 1,023 nt arm minus 378 nt arm (fold 1)", float(ws.loc["mid_1023", "vs_378_mean"]), -0.020, 3)
    chk(P + "width: 2,023 nt arm minus 378 nt arm (fold 1)", float(ws.loc["wide_2023", "vs_378_mean"]), -0.050, 3)
    chk(P + "width:   SLICER with full flank minus crisprHAL 2 at 2,023 nt (fold 1)", float(ws.loc["wide_2023", "lead_mean"]), 0.065, 3)
    chk(P + "width: crisprHAL 2 at 1,023 nt (fold 1)", float(ws.loc["mid_1023", "crisprhal_mean"]), 0.6602, 4)
    chk(P + "width: crisprHAL 2 at 2,023 nt (fold 1)", float(ws.loc["wide_2023", "crisprhal_mean"]), 0.6299, 4)
    # the lead over the best configuration found for crisprHAL 2, on the 1,204 genes held out of the settings search
    cv = pd.read_csv(OUT / "clean_vs_hal.csv")
    r = cv[cv.vs.str.contains("tuned configuration") & (cv.arm == "tuned XGBoost (d099x)")].iloc[0]
    chk(P + "1,204 held-out genes: tuned SLICER minus crisprHAL 2's best configuration found", float(r["diff"]), 0.0240, 4)
    chk(P + "   lower", float(r.lo), 0.0162, 4)
    chk(P + "   upper", float(r.hi), 0.0318, 4)
    # averaging with crisprHAL 2 on its SpCas9 split
    bw = pd.read_csv(OUT / "blend_with_hal.csv")
    b = bw[(bw.split == "wt") & (bw.weight_on_crisprhal == 0.5)].iloc[0]
    chk(P + "equal-weight blend minus the better single model, SpCas9 split", float(b.vs_best_single), 0.0085, 4)
    chk(P + "   lower", float(b.lo), 0.0040, 4)
    chk(P + "   upper", float(b.hi), 0.0127, 4)
    # size of the fitted model: leaf values against leaf values plus each split's column and threshold
    cx = json.loads((R / "complexity.json").read_text())
    chk(P + "tree model size counting each split's column and threshold", float(cx["parameters_incl_structure"]), 18244, 0)
    chk(P + "crisprHAL 2 parameters over that count", cx["crisprhal2"]["WT-SpCas9"] / cx["parameters_incl_structure"], 43, 0)


if __name__ == "__main__":
    import sys
    n, bad = 0, []

    def chk(label, got, claimed, dec):
        global n
        n += 1
        ok = abs(got - claimed) <= 0.5 * 10 ** (-dec) + 1e-9
        if not ok:
            bad.append(label)
        print(f"  {'ok ' if ok else 'BAD'} {label:90s} computed {got:+.5f}  paper {claimed:+.{dec}f}")
    run(chk)
    print(f"\n{n} checks, {len(bad)} mismatches")
    sys.exit(1 if bad else 0)
