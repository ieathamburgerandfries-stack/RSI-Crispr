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
    chk(P + "[D5] the best window of the single-window GC sweep is 250 nt (1 = yes)", float(fs[fs.index.str.startswith("gc W=")].gain.idxmax() == "gc W=250"), 1, 0)
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

    # ---------------------------------------------------------------------------------------------------- the cost, with its spread (cost_repeats.py, model_size.py, boostmec_style.py)
    inf = pd.read_csv(OUT / "cost_repeats_inference.csv")
    g = inf.groupby(["model", "batch"]).ms_per_guide
    cnt, med, sd, mean = g.count(), g.median(), g.std(ddof=1), g.mean()
    for (m, b), n_ in {("SLICER default", 1): 300, ("SLICER default", 100): 60, ("SLICER default", 10000): 15, ("SLICER tuned", 1): 300, ("SLICER tuned", 100): 60,
                       ("SLICER tuned", 10000): 15, ("crisprHAL 2", 1): 200, ("crisprHAL 2", 100): 40, ("crisprHAL 2", 10000): 8}.items():
        chk(P + f"[D15] repeats, {m}, batch {b:,}", float(cnt[(m, b)]), n_, 0)
    for m, b, claimed, dec in (("SLICER default", 1, 0.64, 2), ("SLICER default", 100, 0.077, 3), ("SLICER default", 10000, 0.081, 3), ("SLICER tuned", 1, 0.72, 2),
                               ("SLICER tuned", 100, 0.091, 3), ("SLICER tuned", 10000, 0.097, 3), ("crisprHAL 2", 1, 16, 0), ("crisprHAL 2", 100, 1.4, 1), ("crisprHAL 2", 10000, 1.2, 1)):
        chk(P + f"[D15] median ms per guide, {m}, batch {b:,}", float(med[(m, b)]), claimed, dec)
    chk(P + "[D15] largest coefficient of variation over the nine configurations, %", float((100 * sd / mean).max()), 8, 0)

    def ratio(m, b, seed=0, B=4000):
        rng = np.random.default_rng(seed)
        a_ = inf[(inf.model == "crisprHAL 2") & (inf.batch == b)].ms_per_guide.to_numpy()
        c_ = inf[(inf.model == m) & (inf.batch == b)].ms_per_guide.to_numpy()
        r = [np.median(rng.choice(a_, len(a_))) / np.median(rng.choice(c_, len(c_))) for _ in range(B)]
        return np.median(a_) / np.median(c_), np.percentile(r, 2.5), np.percentile(r, 97.5)
    for b, (r_, lo_, hi_) in {1: (24.8, 24.6, 24.9), 100: (18.1, 17.9, 18.3), 10000: (15.1, 14.9, 15.2)}.items():
        e = ratio("SLICER default", b)
        chk(P + f"[D15] crisprHAL 2 / SLICER default, batch {b:,}: ratio of medians", float(e[0]), r_, 1)
        chk(P + f"[D15]   lower end of the bootstrap interval", float(e[1]), lo_, 1)
        chk(P + f"[D15]   upper end", float(e[2]), hi_, 1)
    fit = pd.read_csv(OUT / "cost_repeats_fit.csv")
    for m, n_, w_ in (("SLICER default", 10, 14), ("SLICER tuned", 5, 25)):
        q = fit[fit.model == m]
        chk(P + f"[D15] fits timed, {m}", float(len(q)), n_, 0)
        chk(P + f"[D15]   median wall-clock seconds per fit", float(q.wall_seconds.median()), w_, 0)
    chk(P + "[D15] SLICER default fits: slowest wall-clock seconds (the text says 14 to 15)", float(fit[fit.model == "SLICER default"].wall_seconds.max()), 15, 0)
    rr = pd.read_csv(OUT.parent / "crisprhal_rerun.csv")
    hf = rr[(rr.arm == "our_folds") & rr.model.str.startswith("crisprHAL2")].fit_seconds
    chk(P + "[D15] crisprHAL 2 fits on the CPU (five folds): median minutes", float(hf.median()) / 60, 17, 0)
    t4 = pd.read_csv(OUT.parent / "crisprhal_wide_window_folds.csv")
    t4 = t4[(t4.arm == "native_378") & t4.note.astype(str).str.startswith("Colab")].fit_minutes
    chk(P + "[D15] crisprHAL 2 fits on a Colab T4 (folds): number", float(len(t4)), 3, 0)
    chk(P + "[D15]   median minutes", float(t4.median()), 5.7, 1)
    sz = pd.read_csv(OUT / "model_size.csv").set_index("model")
    chk(P + "[D15] stored model, SLICER default: trees", float(sz.loc["SLICER default", "trees"]), 400, 0)
    chk(P + "[D15]   leaves (one training fold; the all-data model has 6,348, see [54])", float(sz.loc["SLICER default", "leaves"]), 6338, 0)
    chk(P + "[D15]   megabytes", float(sz.loc["SLICER default", "bytes_file"]) / 1e6, 0.68, 2)
    chk(P + "[D15] stored model, SLICER tuned: trees", float(sz.loc["SLICER tuned", "trees"]), 4000, 0)
    chk(P + "[D15]   leaves", float(sz.loc["SLICER tuned", "leaves"]), 113540, 0)
    chk(P + "[D15]   megabytes", float(sz.loc["SLICER tuned", "bytes_file"]) / 1e6, 10.23, 2)
    chk(P + "[D15] crisprHAL 2: parameters", float(sz.loc["crisprHAL 2", "parameters"]), 790533, 0)
    chk(P + "[D15]   megabytes (the released .keras file)", float(sz.loc["crisprHAL 2", "bytes_file"]) / 1e6, 9.6, 1)
    # BoostMEC rebuilt as its authors built it (boostmec_replica.py), released hyperparameters, same folds and seeds as the flank-scale arms
    bm = pd.read_csv(OUT / "boostmec_released.csv").set_index(["seed", "fold"])
    base_ = fr[fr.arm == "base"].set_index(["seed", "fold"]).rho
    al_ = fr[fr.arm == "family A shifted 0"].set_index(["seed", "fold"]).rho
    ix = bm.index.intersection(base_.index).intersection(al_.index)
    chk(P + "[D16] BoostMEC, released hyperparameters: folds (3 seeds x 5)", float(len(ix)), 15, 0)
    chk(P + "[D16]   rho, mean over folds", float(bm.loc[ix].rho.mean()), 0.620, 3)
    chk(P + "[D16]   sd over folds", float(bm.loc[ix].rho.std(ddof=1)), 0.009, 3)
    chk(P + "[D16]   over the 20-mer alone", float((bm.loc[ix].rho - base_.loc[ix]).mean()), 0.073, 3)
    chk(P + "[D16]   against SLICER's default model with the flank features", float((bm.loc[ix].rho - al_.loc[ix]).mean()), -0.087, 3)
    chk(P + "[D16]   folds in which SLICER is ahead", float(((al_.loc[ix] - bm.loc[ix].rho) > 0).sum()), 15, 0)
    chk(P + "[D16]   milliseconds per guide to compute its 149 features", float(bm.features_ms_per_guide.iloc[0]), 3.0, 1)
    chk(P + "[D16]   ... times SLICER default's whole scoring path at a batch of 100", float(bm.features_ms_per_guide.iloc[0] / med[("SLICER default", 100)]), 39, 0)
    chk(P + "[D16]   mean trees chosen by its ten-fold rule", float(bm.trees.mean()), 904, 0)
    chk(P + "[D16]   mean leaves", float(bm.leaves.mean()), 86762, 0)
    chk(P + "[D16]   seconds per model (ten-fold rule on ten cores, then the fit)", float(bm.seconds.mean()), 30, 0)
    ss = pd.read_csv(OUT / "boostmec_searched.csv")
    cvx = pd.read_csv(OUT / "boostmec_vs_slicer_confirm.csv").set_index("arm")
    me = cvx.loc["BoostMEC, its search"]
    chk(P + "[D17] BoostMEC after its own search: guides in the genes held out of the search", float(me.n_guides), 8943, 0)
    chk(P + "[D17]   rho there (three seeds, mean)", float(me.rho), 0.624, 3)
    chk(P + "[D17]   SLICER default model on the same genes", float(cvx.loc["SLICER default", "rho"]), 0.707, 3)
    chk(P + "[D17]   SLICER tuned model on the same genes", float(cvx.loc["SLICER tuned", "rho"]), 0.722, 3)
    chk(P + "[D17]   BoostMEC minus SLICER default", float(cvx.loc["SLICER default", "boostmec_minus_arm"]), -0.084, 3)
    chk(P + "[D17]   BoostMEC minus SLICER tuned", float(cvx.loc["SLICER tuned", "boostmec_minus_arm"]), -0.098, 3)
    chk(P + "[D17]   trees of the searched model", float(ss.trees.iloc[0]), 350, 0)
    chk(P + "[D17]   minutes the search took (ten cores)", float(ss.search_seconds.max() / 60), 16, 0)
    tr = pd.read_csv(OUT / "boostmec_search_trace.csv")
    chk(P + "[D17]   evaluations in the search (10 initial + 100 Bayesian)", float(len(tr)), 110, 0)


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
