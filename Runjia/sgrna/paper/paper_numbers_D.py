"""Audit of the numbers added to the paper (point form) in the second pass of 23 Nov 2026: one protocol, the reach of the flank signal, controls, the transfer matrix,
SLICER against crisprHAL 2 on more metrics, and the data counts behind the Methods. Every check recomputes its number from a stored file by a different route
from the one that wrote the text (the text is generated from the same files by a build script outside the repository, so a stale figure in the Doc cannot survive a
changed file).

    ./run_ext.sh sgrna.paper.paper_numbers_D
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..common import OUT

P = "paper D: "


def run(chk):
    R = OUT.parent
    # ---------------------------------------------------------------------------------------------------- Methods: the data
    raw = pd.read_csv(config.EXTERNAL / "crisprHAL" / "rawData" / "E_coli_WTSpCas9_counts.tsv", sep="\t")
    chk(P + "[D1] raw guides in crisprHAL 2's counts file", float(len(raw)), 121340, 0)
    ctl = (raw.dCas1_NS + raw.dCas2_NS) / 2
    chk(P + "[D1] ... with a mean unselected count of at least 73 (the cutoff; 33,567 are curated, the rest do not locate)", float((ctl >= 73).sum()), 33871, 0)
    from ..build import genome
    chk(P + "[D1] NC_000913.2: genome length, bp", float(genome.genome_length()), 4639675, 0)
    gt = pd.read_csv(config.GENE_TABLE, sep="\t")
    chk(P + "[D1] genes in the reference gene table", float(len(gt)), 4499, 0)
    cg = pd.read_csv(R / "cohort_groups.csv").set_index(["cohort", "grouping"])
    chk(P + "[D1] curated guides in no gene (33,567 minus those in genes)", 33567 - float(cg.loc[("curated", "gene"), "n"]), 3364, 0)
    ck = pd.read_csv(OUT / "curation_check.csv").set_index("stratum")
    chk(P + "[D2] curation check: single-site guides among the raw 121,340", float(ck.single_site_guides.iloc[0]), 60811, 0)
    chk(P + "[D2]   ... of them kept (unselected count at least 73)", float(ck.kept_guides.iloc[0]), 33449, 0)
    chk(P + "[D2]   guides with 30 to 72 unselected counts (discarded)", float(ck.loc["30-72", "n"]), 18733, 0)
    chk(P + "[D2]   their flank-GC correlation with the cutting score", float(ck.loc["30-72", "spearman_flank_gc"]), 0.159, 3)
    chk(P + "[D2]   their lowest-decile deficit, SD", float(abs(ck.loc["30-72", "lowest_decile_deficit_sd"])), 0.94, 2)
    kept = ck.loc[["73-108", "109-168", "169+"]]
    chk(P + "[D2]   kept guides: smallest flank-GC correlation", float(kept.spearman_flank_gc.min()), 0.175, 3)
    chk(P + "[D2]   kept guides: largest flank-GC correlation", float(kept.spearman_flank_gc.max()), 0.188, 3)
    chk(P + "[D2]   kept guides: smallest deficit, SD", float(abs(kept.lowest_decile_deficit_sd).min()), 0.97, 2)
    chk(P + "[D2]   kept guides: largest deficit, SD", float(abs(kept.lowest_decile_deficit_sd).max()), 1.02, 2)
    chk(P + "[D2]   chance of being kept by decile of flank GC: lowest", float(ck.inclusion_by_decile_min.iloc[0]), 0.53, 2)
    chk(P + "[D2]   ... highest", float(ck.inclusion_by_decile_max.iloc[0]), 0.57, 2)
    # the released matrix: an alphabetical block of genes
    ids = pd.read_csv(config.BASE_MATRIX, usecols=[0]).iloc[:, 0].astype(str)
    ids = ids[ids != "sgRNAID"]
    genes = ids.str.split("_").str[0].str.replace(r"b\d{4}$", "", regex=True)
    order = list(dict.fromkeys(genes))
    chk(P + "[D3] released matrix: guide rows", float(len(ids)), 13880, 0)
    chk(P + "[D3]   distinct gene identifiers in its row names", float(len(order)), 1225, 0)
    chk(P + "[D3]   genes are contiguous and in alphabetical order (1 = yes)", float(order == sorted(order, key=str.lower) and (genes != genes.shift()).sum() == len(order)), 1, 0)
    chk(P + "[D3]   first gene is aaeA and last is leuC (1 = yes)", float(order[0] == "aaeA" and order[-1] == "leuC"), 1, 0)

    # ---------------------------------------------------------------------------------------------------- the headline protocol and the reach of the flank
    fr = pd.read_csv(OUT / "flank_scale.csv")
    bb = fr[fr.arm == "base"].set_index(["seed", "fold"]).rho
    ff = fr[fr.arm == "family A shifted 0"].set_index(["seed", "fold"]).rho
    idx = bb.index.intersection(ff.index)
    gg = ff.loc[idx] - bb.loc[idx]
    chk(P + "[D4] folds with both the 20-mer and the aligned family A (3 seeds x 5)", float(len(idx)), 15, 0)
    chk(P + "[D4] rho of the 20-mer alone, mean over folds", float(bb.loc[idx].mean()), 0.547, 3)
    chk(P + "[D4] rho with family A, mean over folds", float(ff.loc[idx].mean()), 0.707, 3)
    chk(P + "[D4] gain from flank features, mean over folds", float(gg.mean()), 0.161, 3)
    chk(P + "[D4]   its sd over folds", float(gg.std(ddof=1)), 0.010, 3)
    chk(P + "[D4]   folds in which it is positive", float((gg > 0).sum()), 15, 0)
    fs = pd.read_csv(OUT / "flank_scale_summary.csv").set_index("arm")
    for W in (250, 5000):
        chk(P + f"[D5] gain from two GC columns at W = {W} nt", float(fs.loc[f"gc W={W}", "gain"]), {250: 0.105, 5000: 0.042}[W], 3)
    chk(P + "[D5] the best window of the first single-window GC sweep (nine windows) is 250 nt (1 = yes)", float(fs[fs.index.str.startswith("gc W=")].gain.idxmax() == "gc W=250"), 1, 0)
    for k, claimed in (("gc", 0.113), ("gcpur", 0.113), ("base4", 0.108)):
        chk(P + f"[D5] gain from the {k} composition of the four windows", float(fs.loc[f"{k} at 50/250/500/1000", "gain"]), claimed, 3)
    sh = fs[fs.index.str.startswith("family A shifted")]
    full = float(sh.loc["family A shifted 0", "gain"])
    for s_, claimed in ((100, 68), (300, 57), (1000, 32), (3000, 6)):
        chk(P + f"[D6] gain retained when family A is read from {s_:,} bp away, % of the aligned gain", 100 * float(sh.loc[f"family A shifted {s_}", "gain"]) / full, claimed, 0)
    for s_, claimed in ((10000, -0.0001), (100000, -0.0001), (1000000, -0.0008)):
        chk(P + f"[D6] gain in rho when family A is read from {s_:,} bp away (none left)", float(sh.loc[f"family A shifted {s_}", "gain"]), claimed, 4)

    # ---------------------------------------------------------------------------------------------------- the step, the slope, the target GC, the gate
    de = pd.read_csv(OUT / "signal_decile.csv")
    chk(P + "[D7] genic guides in the deciles of flank GC", float(de.n.sum()), 30203, 0)
    chk(P + "[D7] mean score in the lowest decile of flank GC", float(de.mean_label.iloc[0]), -3.154, 3)
    chk(P + "[D7]   its gap to the rest, label units", float(abs(de.gap_vs_rest.iloc[0])), 3.16, 2)
    chk(P + "[D7] every |mean score| among deciles 3 to 10 is below 0.15 (1 = yes)", float(de.mean_label.iloc[2:].abs().max() < 0.15), 1, 0)
    sl = pd.read_csv(OUT / "signal_slope.csv").drop_duplicates("base").set_index("base")
    chk(P + "[D7] Spearman of score with flank GC, base A at position 20", float(sl.loc["A", "spearman"]), 0.219, 3)
    chk(P + "[D7]   base G", float(sl.loc["G", "spearman"]), 0.119, 3)
    chk(P + "[D7]   base A, lower end of the interval", float(sl.loc["A", "lo"]), 0.186, 3)
    chk(P + "[D7]   base G, upper end of the interval", float(sl.loc["G", "hi"]), 0.146, 3)
    cr = pd.read_csv(OUT / "signal_corr.csv")
    gcp = cr[(cr.statistic == "GC") & cr.window_nt.isin([50, 250, 500, 1000])].partial_given_target_gc
    chk(P + "[D7] partial correlation of flank GC with the score, smallest of the 8 windows", float(gcp.min()), 0.146, 3)
    chk(P + "[D7]   largest", float(gcp.max()), 0.203, 3)
    pu = cr[(cr.statistic == "purine") & cr.window_nt.isin([50, 250, 500, 1000])].partial_given_target_gc
    chk(P + "[D7] partial correlation of flank purine, smallest of the 8 windows", float(pu.min()), -0.018, 3)
    chk(P + "[D7]   largest", float(pu.max()), 0.000, 3)
    tg = pd.read_csv(OUT / "signal_target_gc.csv")
    chk(P + "[D7] flank gain among the GC-poorest fifth of targets", float(tg.gain.iloc[0]), 0.349, 3)
    chk(P + "[D7]   among the GC-richest fifth", float(tg.gain.iloc[-1]), 0.072, 3)
    chk(P + "[D7]   quintiles in which it is positive", float((tg.gain_lo > 0).sum()), 5, 0)
    gt_ = pd.read_csv(OUT / "gate_test.csv").pivot_table(index=["seed", "fold"], columns="model", values=["rho", "r2"])
    dr2 = gt_["r2"]["with interaction"] - gt_["r2"]["additive"]
    drho = gt_["rho"]["with interaction"] - gt_["rho"]["additive"]
    chk(P + "[D8] gate test: R2 of the interaction model minus the additive one", float(dr2.mean()), 0.0035, 4)
    chk(P + "[D8]   folds in which it is better", float((dr2 > 0).sum()), 15, 0)
    chk(P + "[D8]   rho difference", float(drho.mean()), 0.0007, 4)
    chk(P + "[D8]   folds in which it is better", float((drho > 0).sum()), 14, 0)

    # ---------------------------------------------------------------------------------------------------- controls
    m = pd.read_csv(OUT / "mappability_stratum.csv").set_index("stratum")
    r = lambda p: m[m.index.str.startswith(p)].iloc[0]
    for lab, p, gain, lo, hi, n_ in (("all guides", "all guides", 0.161, 0.149, 0.171, 33567), ("no repeat signal", "strict", 0.157, 0.145, 0.169, 26146),
                                      ("any repeat signal", "the rest", 0.175, 0.150, 0.199, 7421), ("over 10% repeated 25-mers", "more than 10%", 0.261, 0.190, 0.334, 852),
                                      ("a seed duplicate", "a seed duplicate", 0.142, 0.113, 0.173, 1938), ("published block genes", "genes of the published", 0.102, 0.089, 0.115, 10530),
                                      ("all other genes", "all other genes", 0.180, 0.164, 0.197, 19673)):
        x = r(p)
        chk(P + f"[D9] mappability: gain, {lab}: n", float(x.n), n_, 0)
        chk(P + f"[D9]   gain", float(x.gain_A), gain, 3)
        chk(P + f"[D9]   lower end", float(x.gain_A_lo), lo, 3)
        chk(P + f"[D9]   upper end", float(x.gain_A_hi), hi, 3)
    chk(P + "[D9] gain given the uniqueness columns, all guides", float(r("all guides").gain_A_given_U), 0.159, 3)
    dp = pd.read_csv(OUT / "mappability_depth.csv").set_index("stratum")
    for k, claimed in (("control depth, lowest third", 0.166), ("control depth, middle third", 0.161), ("control depth, highest third", 0.155)):
        chk(P + f"[D9] gain in the {k}", float(dp.loc[k, "gain_A"]), claimed, 3)
    chk(P + "[D9] gain given control depth, all guides", float(dp.loc["all guides", "gain_A_given_D"]), 0.160, 3)
    chk(P + "[D9]   rho of 20-mer plus depth minus 20-mer alone (0 = depth adds nothing)", float(dp.loc["all guides", "rho_base+D"] - dp.loc["all guides", "rho_base"]), 0.0, 3)
    cs = pd.read_csv(OUT / "sealed_crossscreen.csv")
    chk(P + "[D10] sealed genes scored with eSpCas9 labels: draws", float(len(cs)), 15, 0)
    chk(P + "[D10]   mean gain", float(cs.gain_esp.mean()), 0.103, 3)
    half = 1.96 * cs.gain_esp.std(ddof=1) / np.sqrt(len(cs))
    chk(P + "[D10]   lower end of mean +- 1.96 se", float(cs.gain_esp.mean() - half), 0.100, 3)
    chk(P + "[D10]   upper end", float(cs.gain_esp.mean() + half), 0.107, 3)
    chk(P + "[D10]   draws with a positive gain", float((cs.gain_esp > 0).sum()), 15, 0)
    chk(P + "[D10]   smallest gain", float(cs.gain_esp.min()), 0.092, 3)
    chk(P + "[D10]   largest gain", float(cs.gain_esp.max()), 0.116, 3)
    chk(P + "[D10]   gain on the WT labels of the same sealed genes", float(cs.gain_wt.mean()), 0.156, 3)
    chk(P + "[D10]   eSpCas9 rho with the 20-mer alone", float(cs.rho_base.mean()), 0.566, 3)
    chk(P + "[D10]   with family A", float(cs.rho_A.mean()), 0.669, 3)

    # ---------------------------------------------------------------------------------------------------- transfer
    tn = pd.read_csv(OUT / "transfer_vs_crisprhal_nuclease.csv")
    ab = tn[tn.subset.str.startswith("guides absent")].set_index("model")
    chk(P + "[D11] eSpCas9 guides absent from the SpCas9 screen", float(ab.n.iloc[0]), 25922, 0)
    chk(P + "[D11]   SLICER default, trained on the 26,830 SpCas9 training guides: rho", float(ab.loc["slicer_default", "rho"]), 0.659, 3)
    chk(P + "[D11]   SLICER tuned", float(ab.loc["slicer_tuned", "rho"]), 0.678, 3)
    chk(P + "[D11]   crisprHAL 2 released WT weights", float(ab.loc["slicer_default", "crisprhal"]), 0.633, 3)
    chk(P + "[D11]   lead of the default model", float(ab.loc["slicer_default", "diff"]), 0.025, 3)
    chk(P + "[D11]   its lower end", float(ab.loc["slicer_default", "lo"]), 0.021, 3)
    chk(P + "[D11]   its upper end", float(ab.loc["slicer_default", "hi"]), 0.030, 3)
    chk(P + "[D11]   lead of the tuned model", float(ab.loc["slicer_tuned", "diff"]), 0.044, 3)
    chk(P + "[D11]   flank features' share of the default model's rho", float(ab.loc["slicer_default", "rho"] - ab.loc["slicer_base_only", "rho"]), 0.110, 3)
    kg = pd.read_csv(OUT / "transfer_vs_crisprhal_kingdom.csv")
    chk(P + "[D11] E. coli to eukaryotes, SLICER 20-mer features: median rho", float(kg.slicer_base.median()), -0.041, 3)
    chk(P + "[D11]   datasets with a positive rho", float((kg.slicer_base > 0).sum()), 2, 0)
    chk(P + "[D11]   largest rho", float(kg.slicer_base.max()), 0.108, 3)
    chk(P + "[D11] E. coli to eukaryotes, crisprHAL 2 architecture on 23 nt: median rho", float(kg.crisprhal_core23.median()), -0.015, 3)
    chk(P + "[D11]   datasets with a positive rho", float((kg.crisprhal_core23 > 0).sum()), 5, 0)
    chk(P + "[D11]   largest rho", float(kg.crisprhal_core23.max()), 0.137, 3)
    t = pd.read_csv(OUT / "eukaryote_transfer.csv")
    t = t[t.model.str.startswith("pipeline")]
    own = t[t.within].set_index("scored_on").rho
    x = t[~t.within].copy()
    x["ret"] = x.rho / x.scored_on.map(own)
    sel = lambda a, b: x[(x.species_trained == a) & (x.species_scored == b) & (~x.same_library)]
    ee = x[(x.species_trained != "E. coli") & (~x.same_library)]
    chk(P + "[D12] eukaryote-to-eukaryote ordered pairs from different libraries", float(len(ee)), 106, 0)
    chk(P + "[D12]   of them with a positive rho", float((ee.rho > 0).sum()), 106, 0)
    chk(P + "[D12] human to human: median rho", float(sel("human", "human").rho.median()), 0.26, 2)
    chk(P + "[D12]   share of the target's own rho kept, %", 100 * float(sel("human", "human").ret.median()), 70, 0)
    chk(P + "[D12] human to mouse: median rho", float(sel("human", "mouse").rho.median()), 0.34, 2)
    chk(P + "[D12]   share kept, %", 100 * float(sel("human", "mouse").ret.median()), 73, 0)
    chk(P + "[D12] mouse to human: share kept, %", 100 * float(sel("mouse", "human").ret.median()), 70, 0)

    te = pd.read_csv(OUT / "transfer_to_ecoli.csv")
    tx = te[~te.within]
    chk(P + "[D12] eukaryotic models scored on E. coli: datasets", float(len(tx)), 11, 0)
    chk(P + "[D12]   with a negative rho", float((tx.rho < 0).sum()), 11, 0)
    chk(P + "[D12]   median rho", float(tx.rho.median()), -0.058, 3)
    chk(P + "[D12]   lowest", float(tx.rho.min()), -0.120, 3)
    chk(P + "[D12]   highest", float(tx.rho.max()), -0.012, 3)
    chk(P + "[D12] E. coli five-fold CV (random folds), 20-mer alone", float(te[te.within].rho.iloc[0]), 0.544, 3)

    td = pd.read_csv(OUT / "transfer_diagonal.csv")
    chk(P + "[D12] diagonal: eukaryotic tables", float(len(td)), 11, 0)
    chk(P + "[D12]   smallest table, guides", float(td.n.min()), 881, 0)
    chk(P + "[D12]   largest table, guides", float(td.n.max()), 4293, 0)
    chk(P + "[D12]   lowest median published-predictor rho", float(td.published_median.min()), 0.10, 2)
    chk(P + "[D12]   highest median published-predictor rho", float(td.published_median.max()), 0.42, 2)
    chk(P + "[D12]   lowest best published-predictor rho", float(td.published_best.min()), 0.17, 2)
    chk(P + "[D12]   highest best published-predictor rho", float(td.published_best.max()), 0.70, 2)
    chk(P + "[D12]   lowest 20-mer cross-validated rho on the table itself", float(td.own_cv_rho.min()), 0.18, 2)
    chk(P + "[D12]   highest", float(td.own_cv_rho.max()), 0.55, 2)
    chk(P + "[D12]   tables where the 20-mer pipeline beats the median published predictor", float((td.own_cv_rho > td.published_median).sum()), 11, 0)
    chk(P + "[D12]   ... and the best published predictor", float((td.own_cv_rho > td.published_best).sum()), 3, 0)
    chk(P + "[D12]   published predictors read per table", float(td.published_tools.min()), 6, 0)

    # ---------------------------------------------------------------------------------------------------- SLICER against crisprHAL 2 on more metrics
    mp = pd.read_csv(OUT / "metrics_panel.csv")
    g = lambda sp, mo, me, col="value": float(mp[(mp.split == sp) & (mp.model == mo) & (mp.metric == me)][col].iloc[0])
    for (sp, me), claimed in {("SpCas9", "auroc_efficient"): 0.011, ("eSpCas9", "auroc_efficient"): 0.016, ("SpCas9", "auroc_dud"): 0.008, ("eSpCas9", "auroc_dud"): 0.011,
                              ("SpCas9", "precision_top10"): 0.019, ("eSpCas9", "precision_top10"): 0.003, ("SpCas9", "ndcg_top10"): 0.012, ("eSpCas9", "ndcg_top10"): 0.004}.items():
        chk(P + f"[D13] tuned SLICER minus crisprHAL 2, {me}, {sp}", g(sp, "SLICER tuned", me, "diff_vs_crisprhal"), claimed, 3)
    chk(P + "[D13] AUROC for duds, SpCas9, interval lower end (0.000: the interval touches zero)", g("SpCas9", "SLICER tuned", "auroc_dud", "diff_lo"), 0.000, 3)
    chk(P + "[D13] top-decile precision SpCas9, interval", g("SpCas9", "SLICER tuned", "precision_top10", "diff_lo"), -0.012, 3)
    chk(P + "[D13]   upper end", g("SpCas9", "SLICER tuned", "precision_top10", "diff_hi"), 0.052, 3)
    chk(P + "[D13] calibration slope, tuned SLICER SpCas9", g("SpCas9", "SLICER tuned", "calibration_slope"), 1.01, 2)
    chk(P + "[D13]   crisprHAL 2", g("SpCas9", "crisprHAL 2", "calibration_slope"), 1.08, 2)
    chk(P + "[D13] calibration slope, tuned SLICER eSpCas9", g("eSpCas9", "SLICER tuned", "calibration_slope"), 1.05, 2)
    chk(P + "[D13]   crisprHAL 2", g("eSpCas9", "crisprHAL 2", "calibration_slope"), 1.05, 2)
    chk(P + "[D13] pick percentile of the 100 best, SpCas9: tuned minus crisprHAL 2", g("SpCas9", "SLICER tuned", "pick100_pctile", "diff_vs_crisprhal"), 1.0, 1)
    chk(P + "[D13]   eSpCas9", g("eSpCas9", "SLICER tuned", "pick100_pctile", "diff_vs_crisprhal"), -0.8, 1)
    for sp_, lo_, hi_ in (("SpCas9", -3.1, 5.2), ("eSpCas9", -3.2, 1.4)):
        chk(P + f"[D13]   pick percentile interval, {sp_}, lower end", g(sp_, "SLICER tuned", "pick100_pctile", "diff_lo"), lo_, 1)
        chk(P + f"[D13]   upper end", g(sp_, "SLICER tuned", "pick100_pctile", "diff_hi"), hi_, 1)
    chk(P + "[D13] tuned SLICER rho on the SpCas9 split (this panel and vs_crisprhal_wt.csv agree)", g("SpCas9", "SLICER tuned", "spearman"),
        float(pd.read_csv(OUT / "vs_crisprhal_wt.csv").query("arm.str.startswith('tuned')").rho.iloc[0]), 3)

    for me, a_, b_ in (("spearman", 0.717, 0.695), ("pearson", 0.738, 0.720), ("kendall", 0.529, 0.507), ("r2", 0.544, 0.515)):
        chk(P + f"[D13] SpCas9 split, {me}: tuned SLICER", g("SpCas9", "SLICER tuned", me), a_, 3)
        chk(P + f"[D13]   crisprHAL 2", g("SpCas9", "crisprHAL 2", me), b_, 3)
    chk(P + "[D13] AUROC for efficient guides, SpCas9 split: tuned SLICER, crisprHAL 2 (0.825, 0.814)", g("SpCas9", "SLICER tuned", "auroc_efficient") - g("SpCas9", "crisprHAL 2", "auroc_efficient"), 0.011, 3)

    # ---------------------------------------------------------------------------------------------------- the position-20 rule and the eukaryotic datasets
    c = pd.read_csv(OUT / "eukaryote_contrast_g_minus_t.csv")
    big = pd.read_csv(OUT / "eukaryote_orientation.csv")
    big = big[big.n >= 300].dataset
    cb = c[c.group.isin(big)]
    chk(P + "[D14] G at position 20 against the mean of A and C: smallest of the 11 eukaryotic datasets", float(cb.g_minus_ac_pos20.min()), 0.082, 3)
    chk(P + "[D14]   largest", float(cb.g_minus_ac_pos20.max()), 0.568, 3)
    chk(P + "[D14]   datasets with a positive value", float((cb.g_minus_ac_pos20 > 0).sum()), 11, 0)
    chk(P + "[D14]   E. coli", float(c[c.group == "E. coli"].g_minus_ac_pos20.iloc[0]), -0.333, 3)
    chk(P + "[D14]   E. coli, as printed in the appendix (two decimals)", float(c[c.group == "E. coli"].g_minus_ac_pos20.iloc[0]), -0.33, 2)

    # ---------------------------------------------------------------------------------------------------- the cost (cost_repeats.py, cost_boostmec.py, cost_boostmec_shipped.py, model_size.py, colab/cost_gpu.py)
    # One quiet laptop session and one Colab T4 session; every claimed value below is typed as the Doc prints it (three significant figures), and its number of decimals is read from it.
    def dec(txt):
        return len(txt.replace(",", "").split(".")[1]) if "." in txt else 0

    def claim(label, got, txt):
        chk(label, float(got), float(txt.replace(",", "").replace("−", "-").rstrip("%")), dec(txt))

    inf = pd.read_csv(OUT / "cost_repeats_inference.csv")
    gi = inf.groupby(["model", "batch"]).ms_per_guide
    cnt, med, sd, mean = gi.count(), gi.median(), gi.std(ddof=1), gi.mean()
    CPU = {("SLICER default", 1): (300, "0.629"), ("SLICER default", 100): (60, "0.0726"), ("SLICER default", 10000): (15, "0.0795"),
           ("SLICER tuned", 1): (300, "0.716"), ("SLICER tuned", 100): (60, "0.0885"), ("SLICER tuned", 10000): (15, "0.0935"),
           ("BoostMEC", 1): (300, "2.94"), ("BoostMEC", 100): (60, "3.07"), ("BoostMEC", 10000): (15, "3.11"),
           ("crisprHAL 2", 1): (200, "15.4"), ("crisprHAL 2", 100): (40, "1.37"), ("crisprHAL 2", 10000): (8, "1.22")}
    for (m, b), (n_, txt) in CPU.items():
        claim(f"[D15] repeats, {m}, batch {b:,}", cnt[(m, b)], str(n_))
        claim(f"[D15] median ms per guide, {m}, batch {b:,} (quiet CPU, one thread)", med[(m, b)], txt)
    claim("[D15] largest coefficient of variation over the twelve configurations, %", (100 * sd / mean).max(), "4")
    claim("[D15] load average while scoring, lowest", min(inf.load1_start.min(), inf.load1_end.min()), "1.8")
    claim("[D15]   highest", max(inf.load1_start.max(), inf.load1_end.max()), "2.4")
    assert inf.session.nunique() == 3, "scoring was timed in the three commands of one chained session"

    def ratio(num, den, b, seed=0, B=4000):
        rng = np.random.default_rng(seed)
        a_ = inf[(inf.model == num) & (inf.batch == b)].ms_per_guide.to_numpy()
        c_ = inf[(inf.model == den) & (inf.batch == b)].ms_per_guide.to_numpy()
        r = [np.median(rng.choice(a_, len(a_))) / np.median(rng.choice(c_, len(c_))) for _ in range(B)]
        return np.median(a_) / np.median(c_), np.percentile(r, 2.5), np.percentile(r, 97.5)
    for num, den, b, (r_, lo_, hi_) in (("crisprHAL 2", "SLICER default", 1, ("24.5", "24.4", "24.5")), ("crisprHAL 2", "SLICER default", 100, ("18.8", "18.8", "18.9")),
                                         ("crisprHAL 2", "SLICER default", 10000, ("15.4", "15.3", "15.4")), ("BoostMEC", "SLICER default", 100, ("42.3", "42.2", "42.5"))):
        e = ratio(num, den, b)
        claim(f"[D15] {num} / {den}, batch {b:,}: ratio of medians", e[0], r_)
        claim("[D15]   lower end of the bootstrap interval", e[1], lo_)
        claim("[D15]   upper end", e[2], hi_)
    # the T4 (colab/cost_gpu.py)
    gp = pd.read_csv(OUT / "cost_gpu.csv")
    gs = gp[gp.unit == "ms/guide"]
    gm = lambda model, device, pl, b: gs[(gs.model == model) & (gs.device == device) & (gs.placement == pl) & (gs.batch == b)].value
    claim("[D18] T4 session: repeats at 1, 100 and 10,000 guides (200, 60, 30)", len(gm("crisprHAL 2", "T4", "raw contexts on host", 100)), "60")
    claim("[D18] crisprHAL 2 on the T4 from raw contexts, one guide, median ms", gm("crisprHAL 2", "T4", "raw contexts on host", 1).median(), "74.5")
    claim("[D18]   10,000 guides, ms per guide", gm("crisprHAL 2", "T4", "raw contexts on host", 10000).median(), "0.311")
    claim("[D18] encoded batch on the card, 10,000 guides, ms per guide", gm("crisprHAL 2", "T4", "encoded batch on the card", 10000).median(), "0.0837")
    ra = np.random.default_rng(0)
    a_ = gm("crisprHAL 2", "T4", "encoded batch on the card", 10000).to_numpy(); c_ = inf[(inf.model == "SLICER default") & (inf.batch == 10000)].ms_per_guide.to_numpy()
    rr_ = [np.median(ra.choice(a_, len(a_))) / np.median(ra.choice(c_, len(c_))) for _ in range(4000)]
    claim("[D18]   ... times SLICER default's whole path on one CPU thread (ratio of medians)", np.median(a_) / np.median(c_), "1.05")
    claim("[D18]     lower end", np.percentile(rr_, 2.5), "1.04")
    claim("[D18]     upper end", np.percentile(rr_, 97.5), "1.07")
    claim("[D18] best batch size on the card at 10,000 guides", gp[(gp.what == "best batch size") & (gp.placement == "encoded batch on the card") & (gp.batch == 10000)].value.iloc[0], "2048")
    for lab, vals in (("SLICER default shape", ("0.00636", "0.00519", "0.000617")), ("SLICER tuned shape", ("0.0690", "0.0194", "0.00803"))):
        for (dev, pl), txt in zip((("Colab CPU", "one CPU thread"), ("T4", "matrix on host"), ("T4", "matrix on the card")), vals):
            claim(f"[D18] {lab}, forward pass at 10,000 guides, {pl}", gm(lab, dev, pl, 10000).median(), txt)
    gf = gp[gp.unit == "s"]
    gfv = lambda model, device: gf[(gf.model == model) & (gf.device == device)].value
    claim("[D18] fits on the T4 of the default shape, s", gfv("SLICER default shape", "T4").median(), "3.20")
    claim("[D18]   on Colab's CPU", gfv("SLICER default shape", "Colab CPU, all threads").median(), "23.8")
    claim("[D18] tuned shape on the T4, s", gfv("SLICER tuned shape", "T4").median(), "29.0")
    claim("[D18]   on Colab's CPU", gfv("SLICER tuned shape", "Colab CPU, all threads").median(), "479")
    claim("[D18] crisprHAL 2 fit on the T4: fits", len(gfv("crisprHAL 2", "T4")), "3")
    claim("[D18]   median seconds", gfv("crisprHAL 2", "T4").median(), "326")
    claim("[D18]   shortest, minutes", gfv("crisprHAL 2", "T4").min() / 60, "5.4")
    claim("[D18]   longest, minutes", gfv("crisprHAL 2", "T4").max() / 60, "5.6")
    for ln in (OUT.parent / "inference_onecore.csv").read_text().splitlines()[1:]:         # device names hold commas, so parse from the right
        q = ln.split(",")
        if q[0].startswith("crisprHAL 2 WT-SpCas9 (shipped)") and "input on card" in ln and q[-4] == "10000":
            claim("[D18] an earlier Colab session: crisprHAL 2 on the card at 10,000 guides, ms per guide", float(q[-3]) / 10000, "0.0762")
    # fits on the laptop
    fit = pd.read_csv(OUT / "cost_repeats_fit.csv")
    FIT = {"SLICER default": (10, "12.8", "81.5"), "SLICER tuned": (5, "24.5", "132"), "BoostMEC, final fit": (5, "13.2", "46.0"), "BoostMEC, with its tree count": (5, "29.2", "181"), "crisprHAL 2": (3, "1,037", "9,617")}
    for m, (n_, w_, c_) in FIT.items():
        q = fit[fit.model == m]
        claim(f"[D15] fits timed, {m}", len(q), str(n_))
        claim("[D15]   median wall-clock seconds per fit", q.wall_seconds.median(), w_)
        claim("[D15]   median process CPU seconds", q.cpu_seconds.median(), c_)
    hq = fit[fit.model == "crisprHAL 2"]
    claim("[D15] crisprHAL 2 fit on the laptop: median minutes", hq.wall_seconds.median() / 60, "17.3")
    claim("[D15]   shortest, minutes", hq.wall_seconds.min() / 60, "17.0")
    claim("[D15]   CPU minutes", hq.cpu_seconds.median() / 60, "160")
    claim("[D15]   cores busy on average", (hq.cpu_seconds / hq.wall_seconds).median(), "9.3")
    wm = lambda m: fit[fit.model == m].wall_seconds.median()
    claim("[D15] crisprHAL 2 on the laptop over SLICER default, fit time", wm("crisprHAL 2") / wm("SLICER default"), "81")
    claim("[D15] crisprHAL 2 on the T4 over SLICER default", gfv("crisprHAL 2", "T4").median() / wm("SLICER default"), "25")
    claim("[D15] computing the training features, SLICER default (26,853 guides, one thread, s)", 26853 * med[("SLICER default", 10000)] / 1000, "2.13")
    claim("[D15]   BoostMEC", 26853 * med[("BoostMEC", 10000)] / 1000, "83.5")
    rr = pd.read_csv(OUT.parent / "crisprhal_rerun.csv")
    hf = rr[(rr.arm == "our_folds") & rr.model.str.startswith("crisprHAL2")].fit_seconds
    claim("[D15] earlier fits of crisprHAL 2 on this CPU (five folds, other sessions): median minutes", hf.median() / 60, "17.0")
    sz = pd.read_csv(OUT / "model_size.csv").set_index("model")
    SIZE = {"SLICER default": (400, "6,338", "0.68", "0.15"), "SLICER tuned": (4000, "113,540", "10.2", "2.71"), "BoostMEC (released)": (1582, "136,142", "11.5", "4.14")}
    for m, (t_, lv_, mb_, gz_) in SIZE.items():
        claim(f"[D15] stored model, {m}: trees", sz.loc[m, "trees"], str(t_))
        claim("[D15]   leaves", sz.loc[m, "leaves"], lv_)
        claim("[D15]   megabytes", sz.loc[m, "bytes_file"] / 1e6, mb_)
        claim("[D15]   megabytes compressed (gzip, level 9)", sz.loc[m, "bytes_gzip"] / 1e6, gz_)
    claim("[D15] crisprHAL 2: parameters", sz.loc["crisprHAL 2", "parameters"], "790,533")
    claim("[D15]   megabytes (the released .keras file)", sz.loc["crisprHAL 2", "bytes_file"] / 1e6, "9.6")
    # BoostMEC as shipped (cost_boostmec_shipped.py): the authors' image, unmodified
    sh = pd.read_csv(OUT / "boostmec_shipped_timings.csv")
    for b, (n_, s_, ms_) in {10: (5, "3.74", None), 100: (5, "4.17", "41.7"), 10000: (3, "53.5", "5.35")}.items():
        q = sh[(sh.batch == b) & sh.seconds.notna()]
        claim(f"[D19] shipped pipeline, {b:,} guides: runs", len(q), str(n_))
        claim("[D19]   median seconds, whole script", q.seconds.median(), s_)
        if ms_:
            claim("[D19]   ms per guide", q.ms_per_guide.median(), ms_)
    q = sh[(sh.batch == 10000) & sh.seconds.notna()]
    claim("[D19]   10,000 guides, fastest run", q.seconds.min(), "52.7")
    claim("[D19]   slowest run", q.seconds.max(), "54.5")
    claim("[D19] ... times SLICER default's whole path at 10,000 guides", q.ms_per_guide.median() / med[("SLICER default", 10000)], "67.3")
    claim("[D19] ... times the Python version of BoostMEC's features", q.ms_per_guide.median() / med[("BoostMEC", 10000)], "1.72")
    assert sh[sh.batch == 1].note.astype(str).str.startswith("fails").all(), "a one-guide input must have failed as shipped"
    st = pd.read_csv(OUT / "boostmec_shipped_stages.csv")
    q = st[st.batch == 10000].reset_index(drop=True)
    rna = q[q.stage.str.startswith("RNAfold")].seconds.tolist(); whole = float(q[q.stage == "whole pipeline"].seconds.iloc[0])
    claim("[D19] stages at 10,000 guides: RNAfold on the guide with its scaffold, s", rna[1], "31.6")
    claim("[D19]   ... % of the whole script", 100 * rna[1] / whole, "60")
    claim("[D19]   whole script in the staged run, s", whole, "52.5")
    claim("[D19]   R feature script, s", q[q.stage == "Rscript featurize-dataset.R"].seconds.iloc[0], "16.8")
    claim("[D19]   R prediction script, s", q[q.stage == "Rscript predict.R"].seconds.iloc[0], "2.18")
    claim("[D19]   RNAfold on the guide, s", rna[0], "0.523")
    rn = pd.read_csv(OUT / "boostmec_shipped_rnafold.csv").set_index("workload")
    claim("[D19] RNAfold on the 101-nt guide with scaffold, natively, ms per fold", 1000 * rn.loc["guide plus scaffold (101 nt)", "native_seconds"] / rn.loc["guide plus scaffold (101 nt)", "sequences"], "2.9")
    claim("[D19]   in the container, relative to native", rn.loc["guide plus scaffold (101 nt)", "ratio"], "1.1")
    ck = pd.read_csv(OUT / "boostmec_shipped_check.csv").set_index("item").value
    claim("[D19] shipped pipeline against the Python features, 100 guides: Spearman between predictions", ck["prediction: Spearman between the shipped pipeline and the Python version"], "0.9990")
    claim("[D19]   mean absolute difference of predictions", ck["prediction: mean absolute difference"], "0.34")
    claim("[D19]   largest", ck["prediction: largest absolute difference"], "3.19")
    for k, txt in (("tm1", "1.95"), ("tm2", "0.91"), ("tm3", "1.46"), ("tm4", "1.10")):
        claim(f"[D19]   {k}, mean absolute difference in degrees", ck[f"feature {k}: mean absolute difference"], txt)
    for item in ("feature grna_energy: mean absolute difference", "feature grna_scaffold_energy: mean absolute difference", "feature gc_count: mean absolute difference", "feature num_poly_t: mean absolute difference",
                 "feature max_poly_t_len: mean absolute difference", "k-mer counts (84 columns): largest absolute difference", "categorical columns (56): cells that differ from the sequence"):
        claim(f"[D19]   {item.split(':')[0]} identical", ck[item], "0")
    orc = pd.read_csv(OUT / "boostmec_oracle.csv")
    claim("[D19] the paper's worked example S2: reproduced prediction", orc.reproduced_prediction.iloc[0], "49.19")
    claim("[D19]   S3", orc.reproduced_prediction.iloc[1], "28.41")
    claim("[D19]   published S2 and S3 (49.53, 28.42): differences", orc.difference.iloc[0], "-0.34")
    # ---------------------------------------------------------------------------------------------------- the reference genome (Appendix A4)
    rg = pd.read_csv(OUT / "reference_genome_versions.csv").set_index("accession_version")
    from .. import config as _cfg
    seq_len = sum(len(l.strip()) for l in open(_cfg.GENOME_FASTA) if not l.startswith(">"))
    claim("[D20] reference genome used (NC_000913.2): length of the fasta file, bp", seq_len, "4,639,675")
    claim("[D20]   NCBI's length for NC_000913.2", rg.loc["NC_000913.2", "length_bp"], "4,639,675")
    claim("[D20]   NCBI's length for the replacing version NC_000913.3", rg.loc["NC_000913.3", "length_bp"], "4,641,652")
    assert rg.loc["NC_000913.2", "status"] == "replaced" and rg.loc["NC_000913.2", "replaced_by"] == "NC_000913.3", "NCBI no longer lists NC_000913.2 as replaced by NC_000913.3"
    # ---------------------------------------------------------------------------------------------------- coverage additions (numbers that were printed in the Doc without a check)
    from .. import config as _c
    dd = _c.EXTERNAL / "crisprHAL" / "data"
    claim("[D21] C. rodentium TevSpCas9 files: rows (training 20,195 and testing 5,049)", len(pd.read_csv(dd / "TevSpCas9_training_data.csv", header=None)) + len(pd.read_csv(dd / "TevSpCas9_testing_data.csv", header=None)), "25,244")
    from ..compare import transfer as _tr
    _x, _y, _ = _tr.build("TevSpCas9", genome_fasta=_c.EXTERNAL / "C_rodentium" / "NC_013716.1.fasta")
    claim("[D21]   unique guides located in the genome", len(_y), "25,210")
    import re as _re
    mt = (_c.EXTERNAL / "BoostMEC" / "repo" / "model" / "model.txt").read_text()
    par = {k: float(v) for k, v in _re.findall(r"\[(\w+): ([-\d.e+]+)\]", mt.split("parameters:")[-1])}
    claim("[D21] BoostMEC released model: learning rate", par["learning_rate"], "0.0099")
    claim("[D21]   leaves per tree (limit)", par["num_leaves"], "701")
    claim("[D21]   maximum depth", par["max_depth"], "12")
    claim("[D21]   rows per leaf", par["min_data_in_leaf"], "203")
    claim("[D21]   share of columns per tree, %", 100 * par["feature_fraction"], "48")
    claim("[D21]   L2 penalty", par["lambda_l2"], "1.48")
    claim("[D21]   histogram bins", par["max_bin"], "343")
    claim("[D21]   features of the model file", int(_re.search(r"max_feature_idx=(\d+)", mt).group(1)) + 1, "149")
    img = (OUT / "boostmec_shipped_image.txt").read_text()
    claim("[D21] BoostMEC Docker image size, GB", float(_re.search(r"amd64 (\d+)", img).group(1)) / 1e9, "1.2")
    for want in ("x86_64", "Ubuntu", "18.04", "R version 4.1.0", "RNAfold 2.4.14", "Python 3.7.1"):
        assert want in img, f"{want} missing from the stored image description"
    orc2 = pd.read_csv(OUT / "boostmec_oracle.csv")
    claim("[D21] the paper's worked examples, published predictions", orc2.paper_prediction.iloc[0], "49.53")
    claim("[D21]   second", orc2.paper_prediction.iloc[1], "28.42")
    # ---------------------------------------------------------------------------------------------------- the head-to-head on whole-gene folds, and the window sweep
    gg = pd.read_csv(OUT / "gene_grouped_vs_crisprhal.csv").set_index(["model", "against"])
    pub = gg.loc[("SLICER default", "crisprHAL 2")]
    claim("[D22] whole-gene folds, crisprHAL 2 at its published batch (1,024): mean rho over the five folds", pub.crisprhal_mean, "0.693")
    claim("[D22]   SLICER default model on the same folds", pub["mean"], "0.707")
    claim("[D22]   lead of the default model", pub.lead_mean, "0.014")
    claim("[D22]     sd of the lead over the folds", pub.lead_sd, "0.006")
    claim("[D22]     folds in which SLICER default is ahead", pub.folds_ahead, "5")
    tu = gg.loc[("SLICER tuned", "crisprHAL 2")]
    claim("[D22]   SLICER tuned model", tu["mean"], "0.723")
    claim("[D22]   lead of the tuned model", tu.lead_mean, "0.031")
    claim("[D22]     folds ahead", tu.folds_ahead, "5")
    b256 = gg.loc[("SLICER default", "crisprHAL 2 (batch 256)")]
    claim("[D22] the earlier run at batch 256: crisprHAL 2 mean", b256.crisprhal_mean, "0.680")
    claim("[D22]   below its published batch by", pub.crisprhal_mean - b256.crisprhal_mean, "0.013")
    claim("[D22]   the default model's lead then", b256.lead_mean, "0.027")
    claim("[D22]   the tuned model's lead then", gg.loc[("SLICER tuned", "crisprHAL 2 (batch 256)")].lead_mean, "0.044")
    rr2 = pd.read_csv(OUT.parent / "crisprhal_rerun.csv")
    claim("[D22] crisprHAL 2 at its published configuration on our five random-row folds (seed 41)", rr2[(rr2.arm == "our_folds") & (rr2.seed == 41) & rr2.model.str.startswith("crisprHAL2")].spearman.mean(), "0.697")
    fsd = pd.read_csv(OUT / "flank_scale.csv")
    fsd = fsd[fsd.seed == 41]
    wg = lambda W: fsd[fsd.arm == f"gc W={W}"].set_index("fold").rho
    bs0 = fsd[fsd.arm == "base"].set_index("fold").rho
    claim("[D23] window sweep (seed 41, five folds), two GC columns: gain at 100 nt", (wg(100) - bs0).mean(), "0.095")
    claim("[D23]   at 250 nt", (wg(250) - bs0).mean(), "0.105")
    claim("[D23]   at 500 nt", (wg(500) - bs0).mean(), "0.097")
    claim("[D23]   folds in which 250 nt beats both 100 and 500 nt", ((wg(250) > wg(100)) & (wg(250) > wg(500))).sum(), "5")
    claim("[D23]   250 nt is within 0.01 of 100 and of 500 nt (1 = yes)", float(max((wg(250) - wg(100)).mean(), (wg(250) - wg(500)).mean()) < 0.01), "1")
    claim("[D23]   windows of the first sweep (10, 25, 50, 100, 250, 500, 1,000, 2,000, 5,000)", fsd[fsd.part == "window"].arm.str.extract(r"gc W=(\d+)").dropna()[0].nunique(), "9")
    # tuning the two windows separately (flank_window_tune.py): the five symmetric windows it adds to the nine above, one-sided scans, a 3 x 3 refinement, a replication on two new fold assignments
    wtn = pd.read_csv(OUT / "flank_window_tune.csv").drop_duplicates(["arm", "seed", "fold"])
    wsm = pd.read_csv(OUT / "flank_window_tune_summary.csv").set_index("arm")
    sym = {W: float((wg(W) - bs0).mean()) for W in (10, 25, 50, 100, 250, 500, 1000, 2000, 5000)}
    sym.update({W: float(wsm.loc[f"gc up={W} down={W}", "gain_41"]) for W in (75, 150, 200, 300, 400)})
    best_w = max(sym, key=sym.get)
    claim("[D24] symmetric windows tried, same window on both sides (seed 41, five folds)", len(sym), "14")
    claim("[D24]   the highest is 200 nt (1 = yes)", float(best_w == 200), "1")
    claim("[D24]   gain at 200 nt", sym[200], "0.106")
    claim("[D24]   windows within 0.01 of the highest are exactly 150, 200, 250, 300, 400 and 500 nt (1 = yes)", float(sorted(W for W in sym if sym[best_w] - sym[W] < 0.01) == [150, 200, 250, 300, 400, 500]), "1")
    cells41 = wsm[wsm.n_folds_41 == 5]
    claim("[D24] combinations of an upstream and a downstream window tried (seed 41; 50 to 1,000 nt)", len(cells41), "33")
    top = cells41.gain_41.idxmax()
    claim("[D24]   the best is 200 nt upstream and 250 nt downstream (1 = yes)", float(top == "gc up=200 down=250"), "1")
    claim("[D24]   its lead over 250/250 on the folds it was chosen on", wsm.loc[top, "minus_250_250_41"], "0.0021")
    claim("[D24]     folds ahead of 250/250 (of 5)", wsm.loc[top, "folds_ahead_41"], "5")
    claim("[D24]   its lead over 250/250 on two new fold assignments (seeds 42, 43)", wsm.loc[top, "minus_250_250_rep"], "0.0001")
    claim("[D24]     folds ahead of 250/250 (of 10)", wsm.loc[top, "folds_ahead_rep"], "5")
    claim("[D24]     folds on the new assignments", wsm.loc[top, "n_folds_rep"], "10")
    near = cells41[(cells41.up.between(150, 300)) & (cells41.down.between(150, 300))]
    claim("[D24] combinations with both windows from 150 to 300 nt", len(near), "13")
    claim("[D24]   the largest shortfall of any of them from the best", cells41.gain_41.max() - near.gain_41.min(), "0.004")
    for side, u_, d_ in (("upstream", 1, 0), ("downstream", 0, 1)):
        for W, txt in ((100, {"upstream": "\u22120.0031", "downstream": "\u22120.0028"}), (50, {"upstream": "\u22120.0107", "downstream": "\u22120.0121"}), (750, {"upstream": "\u22120.0097", "downstream": "\u22120.0056"}), (1000, {"upstream": "\u22120.0114", "downstream": "\u22120.0085"})):
            arm = f"gc up={W} down=250" if u_ else f"gc up=250 down={W}"
            claim(f"[D24] change from 250/250 when only the {side} window is {W} nt", wsm.loc[arm, "minus_250_250_41"], txt[side])
    gaps = [abs(wsm.loc[f"gc up={W} down=250", "minus_250_250_41"] - wsm.loc[f"gc up=250 down={W}", "minus_250_250_41"]) for W in (50, 75, 100, 150, 200, 300, 400, 500, 750, 1000)]
    claim("[D24] largest difference between the upstream and the downstream change at any window", max(gaps), "0.004")
    claim("[D24] gain of 250/250 on the two new fold assignments (mean over 10 folds)", wsm.loc["gc up=250 down=250", "gain_rep"], "0.106")
    claim("[D24] gain of 200/250 on the two new fold assignments", wsm.loc[top, "gain_rep"], "0.106")
    # BoostMEC retrained on the cohort as its authors trained it (boostmec_replica.py), on the features of their own pipeline; released hyperparameters, same folds and seeds as the flank-scale arms
    bm = pd.read_csv(OUT / "boostmec_released.csv").set_index(["seed", "fold"])
    base_ = fr[fr.arm == "base"].set_index(["seed", "fold"]).rho
    al_ = fr[fr.arm == "family A shifted 0"].set_index(["seed", "fold"]).rho
    ix = bm.index.intersection(base_.index).intersection(al_.index)
    claim("[D16] BoostMEC, released hyperparameters: folds (3 seeds x 5)", len(ix), "15")
    claim("[D16]   rho, mean over folds", bm.loc[ix].rho.mean(), "0.620")
    claim("[D16]   sd over folds", bm.loc[ix].rho.std(ddof=1), "0.009")
    claim("[D16]   over the 20-mer alone", (bm.loc[ix].rho - base_.loc[ix]).mean(), "0.074")
    claim("[D16]   against SLICER's default model with the flank features", (bm.loc[ix].rho - al_.loc[ix]).mean(), "-0.087")
    claim("[D16]   folds in which SLICER is ahead", ((al_.loc[ix] - bm.loc[ix].rho) > 0).sum(), "15")
    claim("[D16]   mean trees chosen by its ten-fold rule", bm.trees.mean(), "919")
    claim("[D16]   mean leaves", bm.leaves.mean(), "88,309")
    ss = pd.read_csv(OUT / "boostmec_searched.csv")
    cvx = pd.read_csv(OUT / "boostmec_vs_slicer_confirm.csv").set_index("arm")
    me = cvx.loc["BoostMEC, its search"]
    claim("[D17] BoostMEC after its own search: guides in the genes held out of the search", me.n_guides, "8,943")
    claim("[D17]   rho there (three seeds, mean)", me.rho, "0.623")
    claim("[D17]   SLICER default model on the same genes", cvx.loc["SLICER default", "rho"], "0.707")
    claim("[D17]   SLICER tuned model on the same genes", cvx.loc["SLICER tuned", "rho"], "0.722")
    for arm, (d_, lo_, hi_) in {"SLICER default": ("-0.084", "-0.103", "-0.066"), "SLICER tuned": ("-0.099", "-0.117", "-0.080")}.items():
        claim(f"[D17]   BoostMEC minus {arm}", cvx.loc[arm, "boostmec_minus_arm"], d_)
        claim("[D17]     interval, lower end", cvx.loc[arm, "diff_lo"], lo_)
        claim("[D17]     upper end", cvx.loc[arm, "diff_hi"], hi_)
    claim("[D17]   trees of the searched model", ss.trees.iloc[0], "704")
    claim("[D17]   minutes the search took (ten cores)", ss.search_seconds.max() / 60, "17")
    tr = pd.read_csv(OUT / "boostmec_search_trace.csv")
    claim("[D17]   evaluations in the search (10 initial + 100 Bayesian)", len(tr), "110")
    zc = pd.read_csv(OUT / "boostmec_shipped_cohort.csv")
    claim("[D19] the released BoostMEC model (human cells) on the E. coli cohort: rho", zc.rho_released_model_on_ecoli.iloc[0], "-0.066")
    claim("[D19]   the shipped pipeline on all cohort guides, seconds", zc.pipeline_seconds.iloc[0], "171")
    claim("[D19]   Python loader against its predictions: largest difference (0 to rounding)", zc.max_abs_diff_python_loader_vs_shipped.iloc[0], "0.0")


# ---------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    bad, n = [], 0

    def chk(label, got, claimed, dec):
        """Same rule as verify_numbers: pass when |got - claimed| <= half a unit in the last quoted digit."""
        global n
        n += 1
        ok = abs(got - claimed) <= 0.5 * 10 ** (-dec) + 1e-9
        if not ok:
            bad.append((label, got, claimed))
        print(f"  {'ok ' if ok else 'BAD'} {label:100s} computed {got:+.5f}  paper {claimed:+.{max(dec, 0)}f}")

    run(chk)
    print(f"\n{n} checks, {len(bad)} mismatches")
    for lab, got, claimed in bad:
        print(f"  BAD {lab}: computed {got:+.5f}, paper {claimed}")
    raise SystemExit(1 if bad else 0)
