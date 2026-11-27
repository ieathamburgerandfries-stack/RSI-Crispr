"""Report checks R7: numbers of PROGRESS_REPORT.md lines 1095-1341 (Appendix A, Controls and nulls; Appendix B, the methods and controls behind
Findings 1-7, up to 'Why rho, and why R2') that had no line in the audit, or whose audit line matched a different quantity, recomputed from a
stored file or from the repository's own code. Same idiom as `paper_numbers_C`: `chk(label, got, claimed, dec)` passes when
|got - claimed| <= 0.5 * 10**(-dec); `got` is always computed, `claimed` is the value as the corrected report prints it. Every label starts
with 'report R7: ' and names the cohort and protocol of the number.

    ./run_ext.sh sgrna.paper.report_checks_R7

Cohorts: *published* = Noshay et al.'s 13,880-row matrix (6,232 columns); *curated* = crisprHAL 2's 33,567 guides (base 6,169 columns; "genic" =
the 30,203 guides inside an annotated gene). Protocols: 5 x 5 = five seeds x five random-row folds unless a label says otherwise.
Files the fork does not track (`ablation_nfeat600.csv`, `ablation_nfeat1200.csv`, `folds_shift_baseline.csv`) are read from the main project's
results/ (`config.RESULTS`); Jacky's orientation tables from `RSI-Crispr/Jacky/orientation-controls/results/`. A check whose raw input is absent
on this machine is skipped with a message, never adjusted.

`run(chk)` is called by `verify_numbers` through `paper_numbers`, and the harness at the bottom runs the module alone.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config
from ..common import OUT

P = "report R7: "


def _boot_ci(v, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    b = [rng.choice(v, v.size, replace=True).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def run(chk):
    """Run every check of this module through `chk(label, got, claimed, dec)`."""
    RES = OUT.parent                                  # the fork's results/
    MAIN = config.RESULTS                             # the main project's results/
    PF = OUT                                          # results/post_freeze
    JACKY = config.ROOT / "RSI-Crispr" / "Jacky" / "orientation-controls" / "results"

    # ------------------------------------------------------------------------------------------------ nulls
    print("report R7: Appendix A, permutation floor (published matrix)")
    am = pd.read_csv(RES / "ablation_main.csv").set_index("experiment")
    chk(P + "permutation floor: family A permuted, rho (ablation_main.csv, 5 x 5)", float(am.loc["a_flank (permuted)", "spearman"]), 0.5264, 4)
    chk(P + "permutation floor: baseline, rho (ablation_main.csv)", float(am.loc["baseline", "spearman"]), 0.5273, 4)
    pf = pd.read_csv(RES / "permutation_floors.csv").set_index("family")
    chk(P + "permutation floor: family A floor, paired by (seed, fold)", float(pf.loc["a_flank", "floor_d_rho"]), -0.0009, 4)
    others = list(pf.drop(index="a_flank").floor_d_rho) + list(pd.read_csv(RES / "ablation_grouped.csv").query("experiment.str.contains('permuted')", engine="python").d_spearman)
    chk(P + "permutation floor: lowest floor of the other families", float(min(others)), -0.0001, 4)
    chk(P + "permutation floor: highest floor of the other families", float(max(others)), 0.0011, 4)
    chk(P + "permutation floor: h_offtarget indistinguishable from its floor (verdict)", float(pf.loc["h_offtarget", "verdict"] == "indistinguishable from noise"), 1, 0)
    chk(P + "permutation floor: f_methylation marginal (verdict)", float(pf.loc["f_methylation", "verdict"] == "marginal"), 1, 0)
    jp = pd.read_csv(RES / "ablation_jper.csv").set_index("experiment")
    chk(P + "j_periodicity real, rho (ablation_jper.csv, published, 5 x 5)", float(jp.loc["j_periodicity", "spearman"]), 0.5264, 4)
    chk(P + "j_periodicity permuted, rho", float(jp.loc["j_periodicity (permuted)", "spearman"]), 0.5273, 4)
    chk(P + "i_shape has no permuted arm in any ablation file", float(not any("i_shape" in e and "permuted" in e for f in ("ablation_main", "ablation_grouped", "ablation_jper", "ablation_jper_joint", "ablation_shape_joint")
                                                                            for e in pd.read_csv(RES / f"{f}.csv").get("experiment", pd.Series([], dtype=str)))), 1, 0)

    print("report R7: Appendix A, texture-preserving null, aligned-column correlation of eng.flank.comp.dn500.gc (published matrix)")
    from .hooks import CACHE_META, CACHE_X
    if CACHE_X.exists() and CACHE_META.exists():
        from .hooks import load_dataset
        from .hooks import _shifted_block
        from ..io_utils import load_block, load_guide_index
        _, _, _, ids, _ = load_dataset([], verbose=False)
        idx = load_guide_index()
        idx = idx[idx["located"].astype(bool)].drop_duplicates(config.ID_COL).set_index(config.ID_COL).reindex(ids).reset_index()
        glen = int(idx["genome_length"].iloc[0]) if "genome_length" in idx else 4_639_675
        block = load_block("a_flank").set_index(config.ID_COL).reindex(ids).reset_index()
        a = block["eng.flank.comp.dn500.gc"].to_numpy(float)
        for s, claimed in ((1_000, 0.41), (100_000, 0.05), (2_000_000, -0.03)):
            b = _shifted_block(block, idx, s, glen)["eng.flank.comp.dn500.gc"].to_numpy(float)
            m = ~(np.isnan(a) | np.isnan(b))
            chk(P + f"shift {s:,} bp: correlation of dn500.gc with the aligned column", float(np.corrcoef(a[m], b[m])[0, 1]), claimed, 2)
    else:
        print("  (skipped: the published-matrix cache is absent)")

    print("report R7: Appendix A, negative control from the source experiment (published matrix, 100 kb blocks)")
    chk(P + "d_supercoiling family (ablation_grouped.csv, 5 x 5)", float(pd.read_csv(RES / "ablation_grouped.csv").set_index("experiment").loc["d_supercoiling", "d_spearman"]), 0.044, 3)
    ds = pd.read_csv(RES / "dissection_supercoiling.csv").set_index("experiment")
    ds = ds[ds.grouping == "bin100k"]
    for key, claimed in (("z_dosage", 0.0112), ("z_signal_only", 0.0111), ("z_control_only", 0.0087), ("z_sc_enrichment", 0.0053)):
        chk(P + f"dissection_supercoiling.csv {key} (3 x 5)", float(ds.loc[key, "d_spearman"]), claimed, 4)
    chk(P + "the untagged control's share of the ChIP track's gain, %", 100 * float(ds.loc["z_control_only", "d_spearman"] / ds.loc["z_signal_only", "d_spearman"]), 78, 0)
    mf = MAIN / "folds_shift_baseline.csv"
    if mf.exists():
        sn = pd.read_csv(RES / "shift_null.csv").set_index("shift_bp")
        base = pd.read_csv(mf).drop_duplicates(["seed", "fold"]).spearman.mean()
        chk(P + "family A under 100 kb blocking, published matrix (shift_null.csv shift 0 minus the de-duplicated baseline)", float(sn.spearman[0] - base), 0.077, 3)

    print("report R7: Appendix A, grouped cross-validation (curated cohort)")
    a_ = pd.read_csv(RES / "ablation_curated.csv").set_index("family").d_spearman
    g_ = pd.read_csv(RES / "ablation_curated_gene.csv").set_index("family").d_spearman
    ret = 100 * g_ / a_
    big = [f for f in a_.index if f != "baseline" and a_[f] > 0.05]
    chk(P + "families with a random-row gain above +0.05: how many", float(len(big)), 6, 0)
    chk(P + "... lowest share kept under gene grouping, %", float(ret[big].min()), 79, 0)
    chk(P + "... highest share kept, %", float(ret[big].max()), 96, 0)
    guide = ["b_energy", "c_folding", "f_methylation"]
    chk(P + "guide-only families, share kept in aggregate, %", 100 * float(g_[guide].sum() / a_[guide].sum()), 94, 0)
    chk(P + "j_periodicity share kept, %", float(ret["j_periodicity"]), 64, 0)
    chk(P + "h_offtarget share kept, %", float(ret["h_offtarget"]), 23, 0)
    cg = pd.read_csv(RES / "cohort_groups.csv").set_index(["cohort", "grouping"]).within_share
    chk(P + "between-gene share of label variance, published, %", 100 * (1 - float(cg[("published", "gene")])), 16, 0)
    chk(P + "between-gene share of label variance, curated, %", 100 * (1 - float(cg[("curated", "gene")])), 29, 0)
    chk(P + "within-gene share, published (their own annotation)", float(cg[("published", "gene (their annotation)")]), 0.844, 3)
    try:
        from .hooks import _b_numbers
        pub = pd.read_csv(config.GUIDE_INDEX, low_memory=False)
        g = _b_numbers(pub)["gene_name"]
        chk(P + "published rows where the coordinate-based gene equals Noshay's annotation, %", 100 * float((g.fillna("X") == pub["gene_name"].fillna("X")).mean()), 97.7, 1)
    except Exception as e:                                           # pragma: no cover - raw inputs absent
        print("  (skipped: gene agreement:", e, ")")

    print("report R7: Appendix A, matching and instruments")
    ww = pd.read_csv(RES / "crisprhal_wide_window.csv")
    w256 = ww[(ww.model == "crisprHAL 2") & (ww.batch == 256)].set_index("input_nt").spearman
    sl = ww[ww.model == "SLICER"].set_index("arm").spearman
    chk(P + "equal DNA, fold 1: crisprHAL 2 minus SLICER at their 378 nt span", float(w256.loc[378] - sl["matched"]), 0.023, 3)
    chk(P + "equal DNA, fold 1: SLICER minus crisprHAL 2 at +-1,000 nt (crisprHAL 2 at 2,023 nt)", float(sl["full_flank"] - w256.loc[2023]), 0.065, 3)
    gc = pd.read_csv(RES / "gc_strata_corr_curated.csv").set_index("stratum")
    chk(P + "flank GC (+-500 nt mean) against the label, curated, marginal", float(gc.loc["all", "rho_flank"]), 0.176, 3)
    chk(P + "... partial, target GC held fixed", float(gc.loc["all", "rho_flank_partial_overall"]), 0.221, 3)
    chk(P + "... positive in all five target-GC strata", float((gc.drop(index="all").rho_flank > 0).all()), 1, 0)
    chk(P + "target GC against the label, curated cohort", float(gc.loc["all", "rho_target"]), -0.147, 3)
    chk(P + "target GC against the label, published matrix (13,879 rows)", float(pd.read_csv(RES / "gc_strata_corr.csv").set_index("stratum").loc["all", "rho_target"]), -0.201, 3)
    sc = pd.read_csv(RES / "selector_control.csv").set_index("arm").spearman
    chk(P + "selector control: d_mechanics selecting 300 (curated)", float(sc["base + mech, select 300"] - sc["base, select 300"]), 0.0988, 4)
    chk(P + "selector control: d_mechanics using all columns", float(sc["base + mech, ALL columns"] - sc["base, ALL columns"]), 0.0993, 4)
    sn_ = pd.read_csv(RES / "interp_sensitivity_n.csv")
    perm = sn_[sn_.attribution == "SHAP, columns permuted"].pearson
    real = sn_[sn_.attribution == "SHAP"].pearson
    chk(P + "sensitivity-n: permuted-column SHAP, lowest correlation", float(perm.min()), -0.023, 3)
    chk(P + "sensitivity-n: permuted-column SHAP, highest correlation", float(perm.max()), 0.047, 3)
    chk(P + "sensitivity-n: SHAP on real columns, lowest", float(real.min()), 0.92, 2)
    chk(P + "sensitivity-n: SHAP on real columns, highest", float(real.max()), 0.98, 2)

    print("report R7: Appendix A, orientation controls (Jacky's tables, 2,000 block resamples over 47 blocks)")
    jc, jd = JACKY / "intergenic_orientation_controls_delta_ci.csv", JACKY / "intergenic_orientation_delta_ci.csv"
    if jc.exists() and jd.exists():
        d = pd.read_csv(jc).set_index("analysis")
        for k, (v, lo, hi) in {"gene_matched_dist_gc": (0.007, -0.024, 0.038), "tss": (0.051, 0.020, 0.082), "tss_matched_dist_gc": (0.025, -0.003, 0.053)}.items():
            chk(P + f"orientation {k}: delta", float(d.loc[k, "delta"]), v, 3)
            chk(P + f"orientation {k}: lower", float(d.loc[k, "delta_lo"]), lo, 3)
            chk(P + f"orientation {k}: upper", float(d.loc[k, "delta_hi"]), hi, 3)
        chk(P + "orientation gene, matched: one-sided P", float(d.loc["gene_matched_dist_gc", "p_delta_le_0"]), 0.34, 2)
        chk(P + "orientation TSS, matched: one-sided P", float(d.loc["tss_matched_dist_gc", "p_delta_le_0"]), 0.04, 2)
        g0 = pd.read_csv(jd).set_index("set").loc["s6_good"]
        chk(P + "orientation gene, unmatched: delta", float(g0.delta), 0.026, 3)
        chk(P + "orientation gene, unmatched: lower", float(g0.delta_lo), -0.006, 3)
        chk(P + "orientation gene, unmatched: upper", float(g0.delta_hi), 0.058, 3)
        chk(P + "orientation: blocks resampled", float(d.loc["tss", "n_groups_pos"]), 47, 0)
    else:
        print("  (skipped: Jacky's orientation tables are absent)")

    print("report R7: Appendix A, what the comparison on their splits holds equal")
    ext = config.EXTERNAL / "crisprHAL" / "data"
    if (ext / "eSpCas9_testing_data.csv").exists():
        cur = pd.read_csv(config.INTERIM / "curated_guide_index.csv", low_memory=False)
        wt = set(cur.protospacer.dropna().str.upper())
        te = pd.read_csv(ext / "eSpCas9_testing_data.csv", header=None)[0]
        tr = pd.read_csv(ext / "eSpCas9_training_data.csv", header=None)[0]
        pte, ptr = te.str[193:213], tr.str[193:213]       # NGG PAM at positions 213-215 of every 406-nt context
        chk(P + "eSpCas9 test file: guides", float(len(te)), 11898, 0)
        chk(P + "eSpCas9 test guides that are WT-cohort guides (by protospacer)", float(pte.isin(wt).sum()), 6778, 0)
        chk(P + "... as a share of the test file, %", 100 * float(pte.isin(wt).mean()), 57, 0)
        chk(P + "WT-cohort guides present in the eSpCas9 screen (train + test), of 33,567", float(ptr.isin(wt).sum() + pte.isin(wt).sum()), 33567, 0)
    rr = pd.read_csv(RES / "crisprhal_rerun.csv").query("arm == 'their_split'").iloc[0]
    chk(P + "re-run of crisprHAL 2 on their WT-SpCas9 files, seed 41, rho", float(rr.spearman), 0.6940, 4)
    chk(P + "... rows of the validation file", float(rr.n_val), 6749, 0)
    chk(P + "frozen SLICER on their WT-SpCas9 split minus our re-run of their model", float(pd.read_csv(RES / "their_split_slicer.csv").spearman.iloc[0] - rr.spearman), 0.006, 3)

    print("report R7: Appendix A, reconstructibility probe and equivalence bounds")
    rc = pd.read_csv(RES / "d_mechanics_reconstructibility.csv").oof_r2_from_mono_di.to_numpy()
    chk(P + "reconstructibility: columns", float(len(rc)), 30, 0)
    chk(P + "reconstructibility: dG/dH/dS sums at R2 1.0000", float((rc > 0.99999).sum()), 9, 0)
    chk(P + "reconstructibility: melting temperatures, lowest R2", float(rc[(rc > 0.99) & (rc < 0.99999)].min()), 0.997, 3)
    chk(P + "reconstructibility: melting temperatures, how many", float(((rc > 0.99) & (rc < 0.99999)).sum()), 4, 0)
    mid = rc[(rc > 0.1) & (rc < 0.99)]
    chk(P + "reconstructibility: stacking and slide summaries, how many", float(len(mid)), 9, 0)
    chk(P + "... lowest R2", float(mid.min()), 0.20, 2)
    chk(P + "... highest R2", float(mid.max()), 0.98, 2)
    flank = rc[rc < 0.1]
    chk(P + "reconstructibility: flank thermodynamics, how many", float(len(flank)), 8, 0)
    chk(P + "... lowest R2", float(flank.min()), -0.006, 3)
    chk(P + "... highest R2", float(flank.max()), 0.042, 3)
    ni = pd.read_csv(RES / "no_information.csv").set_index("arm").d_spearman
    chk(P + "guide-only 22 columns over the curated base (no_information.csv)", float(ni["guide_only"]), 0.011, 3)
    chk(P + "flank-only 8 columns", float(ni["flank_only"]), 0.097, 3)
    fe = pd.read_csv(RES / "final_eval_espcas9.csv").set_index("arm").loc["difference"]
    chk(P + "eSpCas9, frozen SLICER minus crisprHAL 2: rho difference", float(fe.spearman), 0.0021, 4)
    chk(P + "... interval lower end", float(fe.ci_lo), -0.0031, 4)
    chk(P + "... interval upper end (smallest margin)", float(fe.ci_hi), 0.0072, 4)
    mb = pd.read_csv(RES / "final_eval_metric_bootstrap.csv").set_index("metric")
    for m, lo, hi in (("spearman", -0.0028, 0.0071), ("kendall", -0.0024, 0.0073), ("pearson", -0.0066, 0.0030), ("r2", -0.0103, 0.0041), ("top_decile", -0.0244, 0.0202), ("pick_pct", -17.91, 3.98)):
        dec = 2 if m == "pick_pct" else 4
        chk(P + f"eSpCas9 metric bootstrap: {m}, lower", float(mb.loc[m, "ci_lo"]), lo, dec)
        chk(P + f"eSpCas9 metric bootstrap: {m}, upper", float(mb.loc[m, "ci_hi"]), hi, dec)
    chk(P + "eSpCas9 pick-percentile interval width, percentile points", float(mb.loc["pick_pct", "ci_hi"] - mb.loc["pick_pct", "ci_lo"]), 22, 0)
    wt_ = pd.read_csv(PF / "vs_crisprhal_wt.csv").set_index("arm").loc["champion"]
    chk(P + "WT-SpCas9 split (6,737 test guides), frozen SLICER minus crisprHAL 2: rho difference", float(wt_["diff"]), 0.0039, 4)
    chk(P + "... interval lower end", float(wt_.lo), -0.0047, 4)
    chk(P + "... interval upper end (smallest margin, outside +-0.01)", float(wt_.hi), 0.0124, 4)
    chk(P + "... the pre-declared-in-code margin of final_eval_espcas9.csv", float(fe.margin), 0.01, 2)

    print("report R7: Appendix A, attenuation ceiling (curated label)")
    cc = json.loads((RES / "curated_ceiling.json").read_text())
    a1, a2 = cc["arms"]["1-1 / 2-2"], cc["arms"]["1-2 / 2-1"]
    chk(P + "ceiling, first pairing", float(a1["ceiling"]), 0.946, 3)
    chk(P + "... lower end", float(a1["ceiling_ci"][0]), 0.944, 3)
    chk(P + "... upper end", float(a1["ceiling_ci"][1]), 0.947, 3)
    chk(P + "ceiling, second pairing", float(a2["ceiling"]), 0.945, 3)
    chk(P + "cohort under their stated cutoff", float(cc["n_after_cutoff"]), 33871, 0)
    chk(P + "plain log-ratio against their released score, |rho|", abs(float(cc["reconstruction"]["spearman"])), 0.957, 3)
    cej = json.loads((RES / "ceiling.json").read_text())
    chk(P + "WT against eSpCas9 screens of the same guides, Spearman", float(cej["cross_screen_spearman"]), 0.810, 3)
    chk(P + "... naive ceiling", float(cej["implied_ceiling_spearman"]), 0.90, 2)
    for r2, claimed in ((0.771, 0.878), (0.78, 0.883)):
        chk(P + f"Guo et al. 2018 Figure 2: r from R2 = {r2}", float(np.sqrt(r2)), claimed, 3)
    for r, claimed in ((0.878, 0.937), (0.863, 0.929), (0.893, 0.945), (0.938, 0.969)):
        chk(P + f"published label's ceiling, square root of reliability {r}", float(np.sqrt(r)), claimed, 3)
    counts = config.EXTERNAL / "crisprHAL" / "rawData" / "E_coli_WTSpCas9_counts.tsv"
    meta = config.INTERIM / "expanded_meta.json"
    if counts.exists() and meta.exists():
        from scipy.stats import spearmanr
        d = pd.read_csv(counts, sep="\t")
        f = d[(d.dCas1_NS + d.dCas2_NS) / 2 >= 73].reset_index(drop=True)
        C = f[["Cas1_S", "Cas2_S", "dCas1_NS", "dCas2_NS"]].to_numpy(float)
        rng = np.random.default_rng(1)
        S1, S2 = np.zeros((len(f), 128)), np.zeros((len(f), 128))
        for m in range(128):
            clr = np.zeros_like(C)
            for j in range(4):
                gam = rng.gamma(C[:, j] + 0.5)
                l = np.log2(gam / gam.sum())
                clr[:, j] = l - l.mean()
            S1[:, m], S2[:, m] = clr[:, 0] - clr[:, 2], clr[:, 1] - clr[:, 3]
        a1_, a2_ = S1.mean(axis=1), S2.mean(axis=1)
        agree = float(spearmanr(a1_, a2_).statistic)
        mj = json.loads(meta.read_text())
        lab = pd.DataFrame({"sgRNA": mj["protospacers"], "y": mj["y"]}).merge(pd.DataFrame({"sgRNA": f.sgRNA, "a": (a1_ + a2_) / 2}), on="sgRNA")
        chk(P + "Dirichlet(c + 0.5)-CLR re-implementation of the released estimator, 128 draws: replicate agreement", agree, 0.798, 3)
        chk(P + "... ceiling", float(np.sqrt(2 * agree / (1 + agree))), 0.942, 3)
        chk(P + "... |rho| with the released score", abs(float(spearmanr(lab.y, lab.a).statistic)), 0.993, 3)
    else:
        print("  (skipped: the replicate counts are absent)")

    # ------------------------------------------------------------------------------------------------ inventory
    print("report R7: Appendix A, inventory of nulls")
    mc = pd.read_csv(RES / "model_comparison.csv").groupby("model").spearman.mean()
    chk(P + "16 model classes, published matrix + family A, 3 x 5: spread", float(mc.max() - mc.min()), 0.116, 3)
    chk(P + "... number of classes", float(len(mc)), 16, 0)
    cm = pd.read_csv(RES / "curated_models.csv")
    chk(P + "8 model classes, curated + family A: spread", float(cm.spearman.max() - cm.spearman.min()), 0.108, 3)
    chk(P + "... number of classes", float(len(cm)), 8, 0)
    am_a = float(am.loc["a_flank", "spearman"])
    for n, claimed_rho, claimed_gain in ((600, 0.0011, 0.0013), (1200, 0.0014, 0.0036)):
        f = MAIN / f"ablation_nfeat{n}.csv"
        if f.exists():
            x = pd.read_csv(f).set_index("experiment")
            chk(P + f"{n}-column budget, published matrix, 3 x 5: rho with family A over the 300-column value (5 x 5)", float(x.loc["a_flank", "spearman"] - am_a), claimed_rho, 4)
            chk(P + f"{n}-column budget: family A's gain over the {n}-column baseline, over the 300-column gain", float(x.loc["a_flank", "d_spearman"] - am.loc["a_flank", "d_spearman"]), claimed_gain, 4)
        else:
            print(f"  (skipped: {f.name} is not on this machine)")
    cl = pd.read_csv(RES / "curated_ladder.csv").set_index("arm").spearman
    ac = pd.read_csv(RES / "ablation_curated.csv").set_index("family")
    chk(P + "300 selected against every column, curated base", float(cl["base_full"] - ac.loc["baseline", "spearman"]), 0.0004, 4)
    chk(P + "300 selected against every column, curated base + family A", float(ac.loc["a_flank", "spearman"] - cl["base_full+a_flank"]), 0.0004, 4)
    scm = pd.read_csv(RES / "selector_comparison.csv").groupby("selector")[["select_s", "spearman"]].mean()
    x, l = scm.loc["xgboost gain, 300 trees (current)"], scm.loc["lightgbm gain, 300 trees"]
    chk(P + "selector library: LightGBM gain minus XGBoost gain (curated + family A, one seed)", float(l.spearman - x.spearman), 0.0008, 4)
    chk(P + "... speed of the selection step, times", float(x.select_s / l.select_s), 3, 0)
    sa = pd.read_csv(RES / "selector_audit.csv").set_index("metric")
    chk(P + "selector: SHAP minus gain, rho", float(sa.loc["shap", "spearman"] - sa.loc["gain", "spearman"]), 0.0007, 4)
    chk(P + "selector: binary indicators in the top 300 by gain", float(sa.loc["gain", "binary_cols"]), 144, 0)
    chk(P + "selector: ... by SHAP", float(sa.loc["shap", "binary_cols"]), 102, 0)
    sr = pd.read_csv(RES / "selector_redundancy.csv").set_index("rule")
    chk(P + "redundancy-penalised rule minus top-k by gain (3 x 5)", float(sr.loc["mrmr", "spearman"] - sr.loc["gain_top_k", "spearman"]), -0.0015, 4)
    chk(P + "... top-k rho", float(sr.loc["gain_top_k", "spearman"]), 0.7083, 4)
    chk(P + "... penalised rho", float(sr.loc["mrmr", "spearman"]), 0.7068, 4)
    chk(P + "... binary columns, penalised", float(sr.loc["mrmr", "binary_cols"]), 175, 0)
    chk(P + "... binary columns, top-k", float(sr.loc["gain_top_k", "binary_cols"]), 146, 0)
    ko = pd.read_csv(RES / "kmer_order_paired.csv").pivot_table(index=["seed", "fold"], columns="arm", values="spearman")
    chk(P + "guide tetramers on top of trimers, published matrix, paired by fold, mean difference (kmer_order_paired.csv)",
        float((ko[ko.columns[-1]] - ko["mono+di+tri"]).mean()), -0.0028, 4)
    jj = pd.read_csv(RES / "ablation_jper_joint.csv").set_index("experiment")
    chk(P + "periodicity + skew on top of family A, published (rho with minus without)", float(jj.loc["a_flank+j_periodicity", "spearman"] - am_a), 0.0004, 4)
    tx = pd.read_csv(RES / "txn_interactions.csv").set_index("arm").spearman
    chk(P + "interaction terms: both minus marginals (curated)", float(tx["both"] - tx["marginals"]), 0.0004, 4)
    rw = pd.read_csv(RES / "replicate_weight.csv").set_index("arm").spearman
    rwr = pd.read_csv(RES / "replicate_weight_raw.csv").set_index("arm").spearman
    chk(P + "per-guide weights: inverse variance minus unweighted (curated, 3 x 5)", float(rw["inverse_variance"] - rw["unweighted"]), -0.0050, 4)
    chk(P + "... the same weights permuted", float(rw["inverse_variance_permuted"] - rw["unweighted"]), -0.0041, 4)
    chk(P + "... on the raw log-ratio label: inverse variance", float(rwr["inverse_variance"] - rwr["unweighted"]), -0.0040, 4)
    chk(P + "... on the raw log-ratio label: permuted", float(rwr["inverse_variance_permuted"] - rwr["unweighted"]), -0.0043, 4)
    chk(P + "... what the correspondence with measured noise is worth (raw label)", float(rwr["inverse_variance"] - rwr["inverse_variance_permuted"]), 0.0004, 4)
    chk(P + "... the raw label against the released label, same rows, folds and features", float(rw["unweighted"] - rwr["unweighted"]), 0.022, 3)
    tu = pd.read_csv(RES / "tuning.csv")
    chk(P + "627-draw search: draws in total", float(tu.trials.sum()), 627, 0)
    chk(P + "... fewest per fold", float(tu.trials.min()), 97, 0)
    chk(P + "... most per fold", float(tu.trials.max()), 157, 0)
    chk(P + "... gain over the default (curated + family A, seed 41, five folds)", float(tu.delta_rho.mean()), 0.0131, 4)
    wr = pd.read_csv(RES / "window_robustness.csv").iloc[0]
    chk(P + "family A at 75/300/600/1,200 nt (curated, 5 x 5)", float(wr.d_spearman), 0.1634, 4)
    chk(P + "... against the original windows", float(wr.d_published_windows), 0.1638, 4)
    chk(P + "... difference", float(wr.difference), -0.0005, 4)

    # ------------------------------------------------------------------------------------------------ Appendix B
    print("report R7: Appendix B, search effort, knob sweep and the joint search")
    cm_ = pd.read_csv(RES / "crisprhal_matched_folds.csv")
    chk(P + "crisprHAL 2 nested search: fits", float(len(cm_)), 33, 0)
    fin = cm_[cm_.stage == "final"].sort_values("fold")
    chk(P + "... mean of the five outer folds", float(fin.spearman.mean()), 0.6987, 4)
    chk(P + "... folds in which the published configuration (draw 0) won", float(fin.is_published_config.sum()), 4, 0)
    srch = cm_[cm_.stage == "search"].groupby("fold").draw.count()
    chk(P + "... fewest draws before patience stopped a fold", float(srch.min()), 5, 0)
    chk(P + "... most draws", float(srch.max()), 8, 0)
    chk(P + "... outer-fold scores of draw 0, lowest", float(fin[fin.is_published_config].spearman.min()), 0.689, 3)
    chk(P + "... highest", float(fin[fin.is_published_config].spearman.max()), 0.701, 3)
    hr = pd.read_csv(RES / "crisprhal_rerun.csv")
    ww2 = pd.read_csv(RES / "crisprhal_wide_window.csv")
    chk(P + "Colab T4 against local, one fold of the published configuration, times",
        float(hr.query("arm == 'our_folds' and fold == 1").fit_seconds.iloc[0] / 60 / ww2.query("model == 'crisprHAL 2' and arm == 'native_378' and batch == 1024").fit_minutes.iloc[0]), 3.1, 1)
    kn = pd.read_csv(RES / "xgb_knobs.csv")
    chk(P + "knob sweep: rows", float(len(kn)), 25, 0)
    chk(P + "... non-default settings", float((~kn.is_default).sum()), 16, 0)
    k = kn.set_index(["knob", "value"]).d_spearman
    chk(P + "... lossguide with 63 leaves", float(k[("grow_policy", "lossguide, max_leaves=63")]), 0.0069, 4)
    chk(P + "... four parallel trees", float(k[("num_parallel_tree", "4")]), 0.0025, 4)
    chk(P + "... max_delta_step 1", float(k[("max_delta_step", "1")]), -0.0080, 4)
    cs = kn[kn.knob.isin(["colsample_bylevel", "colsample_bynode"]) & ~kn.is_default].d_spearman
    chk(P + "... column subsampling, mildest", float(cs.max()), -0.0004, 4)
    chk(P + "... column subsampling, harshest", float(cs.min()), -0.0014, 4)
    vv = pd.read_csv(RES / "xgb_knobs_verify.csv")
    rng = np.random.default_rng(0)
    for arm, claimed in (("depthwise-d6", (0.0063, 0.0053, 0.0073)), ("lossguide-63", (0.0067, 0.0054, 0.0080)), ("parallel-tree-4", (0.0024, 0.0019, 0.0029))):
        v = vv[vv.arm == arm].d_rho.to_numpy()
        boot = [rng.choice(v, v.size, replace=True).mean() for _ in range(20000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        chk(P + f"knob verification {arm}: mean over 15 folds (seeds 41-43)", float(v.mean()), claimed[0], 4)
        chk(P + f"... lower", float(lo), claimed[1], 4)
        chk(P + f"... upper", float(hi), claimed[2], 4)
        chk(P + f"... folds ahead of 15", float((v > 0).sum()), 15, 0)
    sh = pd.read_csv(RES / "sealed_holdout.csv")
    keys = set()
    for p in sh[sh.arm == "dev_tuned"].params:
        keys |= set(json.loads(p))
    chk(P + "the 8-draw XGBoost search inside the sealed holdout varied this many knobs", float(len(keys)), 6, 0)
    ft = pd.read_csv(PF / "tune_fullcv.csv")
    ex = pd.read_csv(PF / "tune_external.csv")
    def paired(df, arm, ref="champion"):
        a = df[df.arm == arm].set_index(["seed", "fold"]).rho
        b = df[df.arm == ref].set_index(["seed", "fold"]).rho
        return (a - b).dropna().to_numpy()
    for lab, df, arm, claimed in (("whole cohort, gene-grouped 3 x 5: tuned d099x", ft, "tuned d099x", (0.0162, 0.0152, 0.0173)),
                                  ("whole cohort: depth 6", ft, "depth6", (0.0072, 0.0062, 0.0082)),
                                  ("whole cohort: average of five", ft, "ensemble of 5", (0.0183, 0.0175, 0.0191)),
                                  ("eSpCas9 (10 folds): tuned d099x", ex, "d099x", (0.0255, 0.0246, 0.0262)),
                                  ("eSpCas9: depth 6", ex, "depth6", (0.0174, 0.0165, 0.0183))):
        v = paired(df, arm)
        lo, hi = _boot_ci(v)
        chk(P + f"joint search, {lab}: gain", float(v.mean()), claimed[0], 4)
        chk(P + f"... lower", float(lo), claimed[1], 4)
        chk(P + f"... upper", float(hi), claimed[2], 4)
    s2 = pd.read_csv(PF / "tune_stage2.csv")
    chk(P + "fresh partitions (seeds 42-44): champion at 400 trees", float(s2[(s2.draw == "champion") & (s2.k == 400)].rho.mean()), 0.7121, 4)
    chk(P + "... best tuned configuration at 4,000 trees", float(s2[(s2.draw == "d099x") & (s2.k == 4000)].rho.mean()), 0.7248, 4)
    chk(P + "... seeds of those partitions", float(sorted(s2.seed.unique())[0]), 42, 0)
    fa = pd.read_csv(PF / "family_a_at_tuned_summary.csv").set_index("comparison")
    chk(P + "tuning gain without family A, gene-grouped 3 x 5", float(fa.loc["tuning gain (d099x - champion) on base", "delta"]), 0.003, 3)
    chk(P + "tuning gain with family A", float(fa.loc["tuning gain (d099x - champion) on base+A", "delta"]), 0.016, 3)
    dd = pd.read_csv(PF / "interpret_depth.csv").set_index("max_depth").confirm_rho
    chk(P + "interaction order: depth 4 minus depth 3 on the held-out genes", float(dd.loc[4] - dd.loc[3]), 0.0003, 4)
    mr = pd.read_csv(PF / "mix_residual_corr.csv", index_col=0)
    chk(P + "mixing: residual correlation XGBoost-LightGBM", float(mr.loc["xgb", "lgb"]), 0.973, 3)
    chk(P + "mixing: residual correlation XGBoost-CatBoost", float(mr.loc["xgb", "cat"]), 0.959, 3)
    for f_, claimed in (("mix_lgb_stage1", 40), ("mix_cat_stage1", 30)):
        chk(P + f"mixing: draws searched, {f_}", float(pd.read_csv(PF / f"{f_}.csv").draw.nunique()), claimed, 0)

    print("report R7: Appendix B, the human baseline and the knob that leaked")
    ta = pd.read_csv(PF / "human_additivity_twin_summary.csv").set_index(["folds", "arm"])["mean"]
    chk(P + "human, row-wise: boosting defaults minus champion", float(ta[("row-wise", "boosting_defaults")] - ta[("row-wise", "champion")]), 0.100, 3)
    chk(P + "human, twin-grouped: champion", float(ta[("twin-grouped", "champion")]), 0.3377, 4)
    chk(P + "human, twin-grouped: champion is the best of the six boosting arms", float(ta[("twin-grouped", "champion")] == max(ta[("twin-grouped", a)] for a in ("boosting_defaults", "champion", "plus_depth", "plus_lr", "plus_nosubsample", "plus_trees"))), 1, 0)
    chk(P + "E. coli published matrix, champion baseline (comparison for the human 0.3377)", float(am.loc["baseline", "spearman"]), 0.5273, 4)
    ar = pd.read_csv(RES / "additivity_reduced.csv").set_index("arm")
    chk(P + "ridge share of boosting, 427 columns, published matrix, %", 100 * float(ar.iloc[0].ridge / ar.iloc[0].lightgbm), 95.6, 1)
    chk(P + "ridge share of boosting, 775 columns (+ family A), %", 100 * float(ar.iloc[1].ridge / ar.iloc[1].lightgbm), 93.7, 1)
    chk(P + "boosting minus ridge, published matrix + family A, 300 selected, 3 x 5", float(mc["xgboost"] - mc["ridge"]), 0.03, 2)

    print("report R7: Appendix B, recoverability and re-description (curated cohort)")
    tr_ = pd.read_csv(RES / "recoverability_trend.csv").set_index("band")
    chk(P + "recoverability trend: columns with a defined R2", float(tr_.n_cols.sum()), 636, 0)
    chk(P + "... R2 <= 0.01: columns", float(tr_.loc["R2<=0.01", "n_cols"]), 425, 0)
    chk(P + "... R2 <= 0.01: increment", float(tr_.loc["R2<=0.01", "d_spearman"]), 0.180, 3)
    chk(P + "... 0.2 < R2 <= 0.5: increment", float(tr_.loc["0.2<R2<=0.5", "d_spearman"]), 0.018, 3)
    chk(P + "... R2 > 0.99: increment", float(tr_.loc["R2>0.99", "d_spearman"]), 0.008, 3)
    chk(P + "... strongest marginal |rho| in the lowest band", float(tr_.loc["R2<=0.01", "max_abs_rho"]), 0.168, 3)
    chk(P + "... in the 0.2-0.5 band", float(tr_.loc["0.2<R2<=0.5", "max_abs_rho"]), 0.143, 3)
    chk(P + "... in the top band", float(tr_.loc["R2>0.99", "max_abs_rho"]), 0.184, 3)
    chk(P + "... lowest strongest-|rho| over the six bands", float(tr_.max_abs_rho.min()), 0.14, 2)
    chk(P + "... highest strongest-|rho| over the six bands", float(tr_.max_abs_rho.max()), 0.18, 2)
    tx_ = pd.read_csv(RES / "redescription_taxonomy.csv").set_index("arm").d_spearman
    chk(P + "coarse cut: determined by the guide", float(tx_["base + determined by the guide"]), 0.008, 3)
    chk(P + "coarse cut: outside the guide (602 columns)", float(tx_["base + outside the guide"]), 0.201, 3)
    pc = pd.read_csv(RES / "recoverability_percolumn.csv")
    chk(P + "the 18 fully recoverable columns, one at a time: largest", float(pc.d_spearman.max()), 0.0065, 4)
    chk(P + "... median", float(pc.d_spearman.median()), 0.0038, 4)
    rf = pd.read_csv(RES / "redescription_random.csv")
    rm = rf[rf.n_random > 0].groupby("n_random").d_spearman.mean()
    chk(P + "random linear tables, mean of three draws: 4", float(rm[4]), -0.0004, 4)
    chk(P + "... 64", float(rm[64]), -0.0002, 4)
    chk(P + "... 256", float(rm[256]), -0.0015, 4)
    chk(P + "... the most negative single draw (256 tables)", float(rf.d_spearman.min()), -0.0021, 4)
    rn = pd.read_csv(RES / "redescription_nonlinear.csv")
    rn = rn[rn.order > 0]
    means = rn.groupby("order").d_spearman.mean()
    chk(P + "no random arm's mean over its draws is further than 0.0015 from zero (largest |mean| over random linear and non-linear arms)",
        float(max(rm.abs().max(), means.abs().max())), 0.0015, 4)
    rfo = pd.read_csv(RES / "redescription_folds.csv").set_index("arm").d_spearman
    chk(P + "real guide self-fold energy / c_folding block, same run, share carried, %", 100 * float(rfo["real eng.dg.guide_selffold"] / rfo["real c_folding block"]), 76, 0)
    chk(P + "... c_folding block in that run", float(rfo["real c_folding block"]), 0.0153, 4)
    fs = pd.read_csv(RES / "redescription_family_split.csv").set_index(["family", "part"]).d_spearman
    chk(P + "family split: c_folding whole family (a different run)", float(fs[("c_folding", "whole family")]), 0.0147, 4)
    chk(P + "family split: d_mechanics not-cheap 10 columns", float(fs[("d_mechanics", "outside")]), 0.0970, 4)
    try:
        from .hooks import _base, _block
        X, y, names, protos = _base()
        b, bn = _block("b_energy", protos)
        c, cn = _block("c_folding", protos)
        u = b[:, bn.index("eng.dg.guide_selffold")].astype(float)
        v2 = c[:, cn.index("eng.fold.spacer_mfe")].astype(float)
        m = np.isfinite(u) & np.isfinite(v2)
        chk(P + "eng.dg.guide_selffold against eng.fold.spacer_mfe, Pearson (curated, 33,567)", float(np.corrcoef(u[m], v2[m])[0, 1]), 0.9993, 4)
    except Exception as e:                                           # pragma: no cover
        print("  (skipped: guide fold energies:", e, ")")

    print("report R7: Appendix B, Chi sites and why rho")
    ch = pd.read_csv(RES / "chi_sites.csv").set_index(["window", "test"])
    for w, a_, b_, se in ((250, 0.306, -0.103, 0.053), (500, 0.287, -0.047, 0.037), (1000, 0.219, -0.053, 0.026)):
        chk(P + f"Chi total, +-{w} nt: beta alone", float(ch.loc[(w, "chi total"), "beta_alone"]), a_, 3)
        chk(P + f"Chi total, +-{w} nt: beta with flank GC held fixed", float(ch.loc[(w, "chi total"), "beta_gc_adjusted"]), b_, 3)
        chk(P + f"Chi total, +-{w} nt: its standard error", float(ch.loc[(w, "chi total"), "se_gc_adjusted"]), se, 3)
    chk(P + "Chi: guides carrying one within +-1 kb", float(ch.loc[(1000, "chi total"), "n_with_site"]), 11677, 0)
    ctl = ch.loc[(1000, "control octamers (same GC)")]
    chk(P + "Chi control octamers at +-1,000 nt: mean beta", float(ctl.beta_gc_adjusted), -0.022, 3)
    chk(P + "... sd", float(ctl.se_gc_adjusted), 0.201, 3)
    chk(P + "... octamers", float(ctl.n_with_site), 200, 0)
    chk(P + "Chi's beta in units of its standard error at +-1,000 nt (nominal t)", float(ch.loc[(1000, "chi total"), "beta_gc_adjusted"] / ch.loc[(1000, "chi total"), "se_gc_adjusted"]), -2.1, 1)
    chk(P + "GCTGGTGG downstream, +-500 nt, GC fixed", float(ch.loc[(500, "chi fwd_dn"), "beta_gc_adjusted"]), 0.169, 3)
    chk(P + "reverse complement upstream, +-500 nt, GC fixed", float(ch.loc[(500, "chi rev_up"), "beta_gc_adjusted"]), -0.110, 3)
    try:
        from scipy.stats import spearmanr
        ci = pd.read_csv(config.INTERIM / "curated_guide_index.csv", low_memory=False)
        up = ci["upstream"].fillna("").astype(str).str.upper().str[-250:]
        gcf = (up.str.count("G") + up.str.count("C")) / up.str.len().clip(lower=1)
        ok = ci["score"].notna()
        chk(P + "flank GC (250 nt upstream) against the label, Spearman, curated", float(spearmanr(gcf[ok], ci.loc[ok, "score"]).statistic), 0.161, 3)
    except Exception as e:                                           # pragma: no cover
        print("  (skipped: flank GC:", e, ")")
    chk(P + "robust-loss LightGBM, R2 (model_comparison.csv, published + family A, 3 x 5)", float(pd.read_csv(RES / "model_comparison.csv").groupby("model").r2.mean()["lgbm_huber"]), 0.192, 3)
    chk(P + "... its rho", float(mc["lgbm_huber"]), 0.500, 3)
    r2m = pd.read_csv(RES / "model_comparison.csv").groupby("model").r2.mean()
    chk(P + "squared-error LightGBM, R2", float(r2m["lightgbm"]), 0.382, 3)
    chk(P + "... its rho", float(mc["lightgbm"]), 0.609, 3)

    print("report R7: Appendix B, cost spread")
    cp = pd.read_csv(RES / "cost_panel.csv").set_index("model")
    chk(P + "cost: SLICER CPU-seconds per fold", float(cp.loc["SLICER, as published", "cpu_seconds"]), 65.8, 1)
    chk(P + "cost: crisprHAL 2 CPU-seconds per fold (hand-entered in cost.py)", float(cp.loc["crisprHAL 2, published", "cpu_seconds"]), 10000, 0)
    chk(P + "cost: ratio", float(cp.loc["SLICER, as published", "ratio_cpu_seconds"]), 152, 0)
    chk(P + "cost: CPU-seconds per wall-second, SLICER", float(cp.loc["SLICER, as published", "cpu_per_wall"]), 4.8, 1)
    chk(P + "cost: ... crisprHAL 2", float(cp.loc["crisprHAL 2, published", "cpu_per_wall"]), 7.9, 1)
    chk(P + "cost: crisprHAL 2 wall-seconds x CPU per wall-second", float(cp.loc["crisprHAL 2, published", "wall_seconds"] * cp.loc["crisprHAL 2, published", "cpu_per_wall"]), 10000, -1)
    xg = cm.set_index("model")
    chk(P + "wall-clock, 5 folds: XGBoost champion, minutes", float(xg.loc["xgboost", "seconds"] / 60), 1.9, 1)
    chk(P + "... ridge, minutes", float(xg.loc["ridge", "seconds"] / 60), 1.0, 1)
    hal_pub = float(hr.query("arm == 'our_folds'").fit_seconds.sum() / 60)
    hal_best = float(pd.read_csv(RES / "crisprhal_tuned_folds.csv").fit_seconds.sum() / 60)
    chk(P + "... crisprHAL 2 published configuration, minutes", hal_pub, 85, 0)
    chk(P + "... its best configuration, minutes", hal_best, 220, 0)
    chk(P + "... ratio to XGBoost, published configuration", hal_pub / (xg.loc["xgboost", "seconds"] / 60), 44, 0)
    chk(P + "... ratio to XGBoost, best configuration", hal_best / (xg.loc["xgboost", "seconds"] / 60), 113, 0)
    chk(P + "boosting beats the best crisprHAL 2 configuration (curated, family A, seed 41)", float(xg.loc["xgboost", "spearman"] - pd.read_csv(RES / "crisprhal_tuned_folds.csv").spearman.mean()), 0.003, 3)
    chk(P + "ridge within this of the best crisprHAL 2 configuration", float(pd.read_csv(RES / "crisprhal_tuned_folds.csv").spearman.mean() - xg.loc["ridge", "spearman"]), 0.058, 3)

    # ------------------------------------------------------------------------------------------------ split 2
    print("report R7: Appendix B, the second gene split")
    try:
        from .hooks import _base, _groups
        from .hooks import tune_ecoli as T
        X, y, _, protos = _base()
        g, ok = _groups("gene", protos)
        g = np.asarray(g[ok], dtype=object)
        genes = np.array(sorted(set(g)))
        c2 = set(np.random.default_rng(7).choice(genes, size=int(round(0.3 * len(genes))), replace=False))
        c1 = set(np.random.default_rng(T.SPLIT_SEED).choice(genes, size=int(round(0.3 * len(genes))), replace=False))
        chk(P + "genes in the genic curated cohort", float(len(genes)), 4013, 0)
        chk(P + "split 2: new confirmation genes", float(len(c2)), 1204, 0)
        chk(P + "split 2: confirmation genes that fed split 1's selection (were development genes there)", float(len(c2) - len(c1 & c2)), 847, 0)
    except Exception as e:                                           # pragma: no cover
        print("  (skipped: split 2:", e, ")")


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
        print(f"  {'ok ' if ok else 'BAD'} {label:120s} computed {got:+.6f}  report {claimed:+.{max(dec, 0)}f}")

    run(chk)
    print(f"\n{n} checks, {len(bad)} mismatches")
    for lab, got, claimed in bad:
        print(f"  BAD {lab}: computed {got:+.6f}, report {claimed}")
    raise SystemExit(1 if bad else 0)
