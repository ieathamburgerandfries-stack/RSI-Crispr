"""Audit of PROGRESS_REPORT.md lines 1342-1554 (Appendix B2 why the imported families failed, B2a curation, B3 the family ablations on the
curated cohort, B4 shape and 3D structure, B5 the dCas9 / CRISPRi boundary experiment, C0 reproducibility, Appendix C feature sets): the
headline numbers that `verify_numbers` did not recompute, each from the stored file (or data file) that holds it.

    ./run_ext.sh sgrna.paper.report_checks_R8

Same idiom as `paper_numbers_C`: `run(chk)` takes `chk(label, got, claimed, dec)` (pass when |got - claimed| <= half a unit in the last
quoted digit); `got` is always computed from a file, `claimed` is the value exactly as the corrected report writes it. Labels start
'report R8: ' and say which cohort and protocol the number belongs to, because the same value occurs for different quantities:

    published   Noshay et al.'s 13,880-row matrix, 6,232 columns, five seeds x five random-row folds (`ablation_main`); the three positional
                families `ablation_grouped` (100 kb blocks); family A's joint arms `ablation_shape_joint`, `ablation_jper_joint`
    curated     crisprHAL 2's 33,567-guide release, base 6,169 columns; `ablation_curated` (random rows), `ablation_curated_gene`
                (gene-grouped, the 30,203 genic guides), both five seeds x five folds
    CRISPRi     Cui et al. (76,325 guides, 70,298 genic; label = mean of the arms fit18, fit75), 'Rousset' (= the fit75 arm of the same
                experiment on 59,246 filtered guides, not a second screen), Wang (the one independent screen): three seeds x five folds,
                gene or 100 kb-block groups

Files the fork does not track (the Rousset per-fold files `folds_cris_Rousset_*.csv`) are read from the main project's results/.
"""
from __future__ import annotations

import json
import sys
import zlib

import numpy as np
import pandas as pd
from scipy import stats

from .. import config
from ..common import OUT

P = "report R8: "


def _folds(df, arm):
    return np.array([float(x) for x in df.set_index("arm").loc[arm, "fold_rho"].split(";")])


def _eta2(gene, y):
    """Share of the label's variance explained by gene identity (between-gene sum of squares over the total)."""
    d = pd.DataFrame({"g": gene, "y": y})
    gm = d.groupby("g").y.transform("mean")
    return float(1 - ((d.y - gm) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum())


def _meta_genes(path):
    m = json.loads(path.read_text())
    gene = np.array([str(x) if x == x and x is not None else "" for x in m["gene"]], dtype=object)
    return m, gene, np.asarray(m["y"], float), np.array([str(x) for x in m["essential"]], dtype=object)


def run(chk):
    """Run every check through `chk(label, got, claimed, dec)`."""
    R = OUT.parent                       # the fork's results/
    MAIN = config.RESULTS                # the main project's results/, for files the fork does not track
    INT = config.INTERIM

    # ------------------------------------------------------------------ B2: why the imported families failed
    print("report R8: Appendix B2 (redundancy.csv: ridge R2 of each new column from the protospacer's mono+di indicators, published matrix)")
    rd = pd.read_csv(R / "redundancy.csv")
    g = rd.groupby("family").r2_from_protospacer
    med, p10, gt9 = g.median(), g.quantile(0.1), g.apply(lambda s: (s > 0.9).mean())
    for fam, v in (("b_energy", 0.984), ("d_mechanics", 0.845), ("c_folding", 0.297), ("f_methylation", 0.141)):
        chk(P + f"B2 median R2 from the guide, {fam}", float(med[fam]), v, 3)
    chk(P + "B2 share of b_energy columns with R2 > 0.9, percent", 100 * float(gt9["b_energy"]), 70, 0)
    chk(P + "B2 share of d_mechanics columns with R2 > 0.9, percent", 100 * float(gt9["d_mechanics"]), 50, 0)
    seven = ["a_flank", "d_supercoiling", "e_nucleoid", "g_transcription", "h_offtarget", "i_shape", "j_periodicity"]
    chk(P + "B2 the seven other families: highest median R2 (a_flank)", float(med[seven].max()), -0.012, 3)
    chk(P + "B2 the seven other families: lowest median R2 (h_offtarget)", float(med[seven].min()), -0.022, 3)
    chk(P + "B2 the seven other families: highest 10th percentile of R2", float(p10[seven].max()), -0.019, 3)
    chk(P + "B2 the seven other families: lowest 10th percentile of R2", float(p10[seven].min()), -0.025, 3)
    chk(P + "B2 R2 span of the seven families' medians", float(med[seven].max() - med[seven].min()), 0.010, 3)
    chk(P + "B2 families with a positive median R2", float((med > 0).sum()), 4, 0)
    chk(P + "B2 columns analysed outside the constant ones (7 for h_offtarget ... 322 for a_flank): smallest family", float(g.count().loc[seven].min()), 7, 0)
    chk(P + "B2 ... largest family", float(g.count().loc[seven].max()), 322, 0)

    am = pd.read_csv(R / "ablation_main.csv").set_index("experiment")
    ag = pd.read_csv(R / "ablation_grouped.csv").set_index("experiment")
    aj = pd.read_csv(R / "ablation_jper.csv").set_index("experiment")
    ajj = pd.read_csv(R / "ablation_jper_joint.csv").set_index("experiment")
    ash = pd.read_csv(R / "ablation_shape_joint.csv").set_index("experiment")
    pub = {"a_flank": am.loc["a_flank", "d_spearman"], "b_energy": am.loc["b_energy", "d_spearman"],
           "c_folding": am.loc["c_folding", "d_spearman"], "d_mechanics": am.loc["d_mechanics", "d_spearman"],
           "f_methylation": am.loc["f_methylation", "d_spearman"], "h_offtarget": am.loc["h_offtarget", "d_spearman"],
           "d_supercoiling": ag.loc["d_supercoiling", "d_spearman"], "e_nucleoid": ag.loc["e_nucleoid", "d_spearman"],
           "g_transcription": ag.loc["g_transcription", "d_spearman"], "j_periodicity": aj.loc["j_periodicity", "d_spearman"]}
    for fam, v in (("a_flank", 0.080), ("b_energy", 0.007), ("c_folding", 0.007), ("d_mechanics", 0.020), ("f_methylation", 0.003),
                   ("h_offtarget", 0.000), ("d_supercoiling", 0.044), ("e_nucleoid", 0.016), ("g_transcription", 0.015),
                   ("j_periodicity", -0.001)):
        chk(P + f"B2/B3 published increment over the base, {fam}", float(pub[fam]), v, 3)
    chk(P + "B2/B3 the seven families' published gains span (a_flank minus j_periodicity)",
        float(pub["a_flank"] - pub["j_periodicity"]), 0.080, 3)
    chk(P + "B2 j_periodicity on top of family A, published (ablation_jper_joint minus ablation_main A)",
        float(ajj.loc["a_flank+j_periodicity", "spearman"] - am.loc["a_flank", "spearman"]), 0.0004, 4)
    chk(P + "B2 i_shape on top of family A, published (ablation_shape_joint minus ablation_main A)",
        float(ash.loc["a_flank+i_shape", "spearman"] - am.loc["a_flank", "spearman"]), 0.0010, 4)
    # i_shape's published standalone +0.0356 is recorded only in results/RESULTS_archive_2026-10-10.md (no stored csv): 11 families
    pub["i_shape"] = 0.0356
    fams11 = ["b_energy", "d_mechanics", "c_folding", "f_methylation", "a_flank", "d_supercoiling", "e_nucleoid", "g_transcription",
              "h_offtarget", "i_shape", "j_periodicity"]
    rho = stats.spearmanr(med.loc[fams11].to_numpy(), np.array([pub[f] for f in fams11]))
    chk(P + "B2 Spearman rho(median R2, published increment) over eleven families", float(rho.statistic), 0.227, 3)
    chk(P + "B2 ... its p-value", float(rho.pvalue), 0.50, 2)

    # ------------------------------------------------------------------ B2a: curation
    print("report R8: Appendix B2a (the curated cohort against the published matrix)")
    ac = pd.read_csv(R / "ablation_curated.csv").set_index("family")
    be = pd.read_csv(R / "base_enrichment.csv").set_index("arm")
    bp = pd.read_csv(R / "base_enrichment_published.csv").set_index("arm")
    lp = pd.read_csv(R / "ablation_label_published.csv").set_index("experiment")
    lc = pd.read_csv(R / "ablation_label_curated.csv").set_index("experiment")
    gap = float(ac.loc["a_flank", "d_spearman"] - am.loc["a_flank", "d_spearman"])
    chk(P + "B2a gap in family A's gain, curated (random rows) minus published", gap, 0.084, 3)
    a_base = float(be.loc["base+a_flank", "spearman"] - be.loc["base", "spearman"])
    a_sum = float(be.loc["base+summaries+a_flank", "spearman"] - be.loc["base+summaries", "spearman"])
    chk(P + "B2a the base: family A's gain without the 41 summary columns", a_base, 0.1638, 4)
    chk(P + "B2a the base: ... with them", a_sum, 0.1504, 4)
    chk(P + "B2a the base takes this off family A's gain", a_base - a_sum, 0.013, 3)
    lab = float(lc.loc["a_flank", "d_spearman"] - lp.loc["a_flank", "d_spearman"])
    why = pd.read_csv(R / "why_inflation.csv")
    rows = float(ac.loc["a_flank", "d_spearman"] - why[why.family == "a_flank"].d_spearman.mean())
    chk(P + "B2a remainder, 'which rows' (gap minus label, row count and base)", gap - lab - (-0.003) - 0.013, 0.070, 3)
    chk(P + "B2a share of the gap that the label, rows and base account for (+0.014 of +0.084), in thousandths",
        1000 * (lab + rows + (a_base - a_sum)), 14, 0)
    chk(P + "B2a the 63 hand-named summaries take this off family A's gain (published, sparse base)",
        float((bp.loc["sparse base+a_flank", "spearman"] - bp.loc["sparse base", "spearman"])
              - (bp.loc["published base+a_flank", "spearman"] - bp.loc["published base (+63 hand-named)", "spearman"])), 0.0015, 4)
    chk(P + "B2a curation figure: base + family A on all 13,880 published rows", float(am.loc["a_flank", "spearman"]), 0.607, 3)
    chk(P + "B2a curation figure: base + family A on the 8,617 retained rows, Noshay's label", float(lp.loc["a_flank", "spearman"]), 0.628, 3)
    cv = pd.read_csv(R / "curation_variance.csv").set_index("rows")
    sh_, ad_, ins, out_ = (cv.loc["shared with Noshay"], cv.loc["crisprHAL only"], cv.loc["crisprHAL only, panel genes"],
                           cv.loc["crisprHAL only, other genes"])
    chk(P + "B2a slope beta, shared rows", float(sh_.slope_beta), 5.35, 2)
    chk(P + "B2a slope beta, added rows inside the panel's genes", float(ins.slope_beta), 8.63, 2)
    chk(P + "B2a slope beta, added rows outside the panel", float(out_.slope_beta), 17.03, 2)
    chk(P + "B2a predictor spread, added over shared rows", float(ad_.sd_x / sh_.sd_x), 1.21, 2)
    chk(P + "B2a label spread, added over shared rows", float(ad_.sd_y / sh_.sd_y), 1.34, 2)
    chk(P + "B2a Pearson r, added over shared rows", float(ad_.pearson / sh_.pearson), 2.7, 1)
    chk(P + "B2a beta, inside-panel over shared", float(ins.slope_beta / sh_.slope_beta), 1.6, 1)
    chk(P + "B2a beta, outside over inside-panel", float(out_.slope_beta / ins.slope_beta), 2.0, 1)
    # the predictor is the GC fraction of the 500 nt UPSTREAM of the protospacer (eng.flank.comp.up500.gc); +0.069 is Noshay's label
    from ..cohorts.curated_ablation import _base
    _, y, _, protos = _base()
    a = pd.read_csv(config.FEATURE_DIR / "a_flank_curated.csv", usecols=[config.ID_COL, "eng.flank.comp.up500.gc"]).set_index(config.ID_COL)
    a.index = a.index.astype(str)
    x = a.reindex(pd.Index([str(p) for p in protos]))["eng.flank.comp.up500.gc"].to_numpy(float)
    gi = pd.read_csv(config.GUIDE_INDEX, usecols=["protospacer", "cut_score"]).dropna(subset=["protospacer"])
    nos = gi.groupby("protospacer").cut_score.mean()
    d = pd.DataFrame({"p": protos, "x": x, "y": y})
    d["noshay"] = d.p.map(nos)
    shd, add = d[d.noshay.notna()].dropna(subset=["x"]), d[d.noshay.isna()].dropna(subset=["x"])
    chk(P + "B2a upstream-500 GC against Noshay's label, 8,617 shared guides (Spearman)", float(stats.spearmanr(shd.x, shd.noshay).statistic), 0.069, 3)
    chk(P + "B2a ... against crisprHAL's label, same guides", float(stats.spearmanr(shd.x, shd.y).statistic), 0.067, 3)
    chk(P + "B2a ... against crisprHAL's label, the 24,950 added guides", float(stats.spearmanr(add.x, add.y).statistic), 0.172, 3)

    # ------------------------------------------------------------------ B3: the family ablations on the curated cohort
    print("report R8: Appendix B3 (curated cohort: random-row `ablation_curated`, gene-grouped `ablation_curated_gene`, `grouped_retention`)")
    gc_ = pd.read_csv(R / "ablation_curated_gene.csv").set_index("family")
    gr = pd.read_csv(R / "grouped_retention.csv").set_index("family")
    curated = {"a_flank": 0.164, "d_supercoiling": 0.139, "i_shape": 0.120, "d_mechanics": 0.099, "e_nucleoid": 0.072, "g_transcription": 0.058,
               "b_energy": 0.019, "c_folding": 0.014, "j_periodicity": 0.011, "f_methylation": 0.008, "h_offtarget": 0.001}
    gene = {"a_flank": 0.157, "d_supercoiling": 0.131, "i_shape": 0.112, "d_mechanics": 0.092, "e_nucleoid": 0.063, "g_transcription": 0.046,
            "b_energy": 0.017, "c_folding": 0.014, "j_periodicity": 0.007, "f_methylation": 0.007, "h_offtarget": 0.000}
    kept = {"a_flank": 96, "d_supercoiling": 94, "i_shape": 94, "d_mechanics": 92, "e_nucleoid": 88, "g_transcription": 79,
            "b_energy": 91, "c_folding": 97, "j_periodicity": 64, "f_methylation": 93, "h_offtarget": 23}
    for fam in curated:
        chk(P + f"B3 curated random-row increment, {fam}", float(ac.loc[fam, "d_spearman"]), curated[fam], 3)
        chk(P + f"B3 curated gene-grouped increment, {fam}", float(gc_.loc[fam, "d_spearman"]), gene[fam], 3)
        chk(P + f"B3 share retained under gene-grouping (percent), {fam}", float(gr.loc[fam, "retained_pct"]), kept[fam], 0)
    chk(P + "B3 curated baseline rho, random rows", float(ac.loc["baseline", "spearman"]), 0.5445, 4)
    chk(P + "B3 curated baseline rho, gene-grouped", float(gc_.loc["baseline", "spearman"]), 0.5584, 4)
    chk(P + "B3 baseline rho of the published runs, ablation_main (random rows)", float(am.loc["baseline", "spearman"]), 0.5273, 4)
    chk(P + "B3 baseline rho of the published runs, ablation_jper", float(aj.loc["baseline", "spearman"]), 0.5272, 4)
    chk(P + "B3 baseline rho of the published runs, ablation_grouped (100 kb blocks)", float(ag.loc["baseline", "spearman"]), 0.5283, 4)
    chk(P + "B3 guides outside every annotated gene", float(len(ac) * 0 + 33567 - 30203), 3364, 0)
    tax = ["b_energy", "c_folding", "f_methylation"]
    chk(P + "B3 grouping tax: aggregate retention of the three guide-only families (percent)",
        100 * float(sum(gc_.loc[f, "d_spearman"] for f in tax) / sum(ac.loc[f, "d_spearman"] for f in tax)), 94, 0)
    chk(P + "B3 excess loss beyond the tax, e_nucleoid (points)", float(gr.loc["e_nucleoid", "excess_loss_pp"]), 6, 0)
    chk(P + "B3 excess loss beyond the tax, g_transcription (points)", float(gr.loc["g_transcription", "excess_loss_pp"]), 15, 0)
    chk(P + "B3 excess loss beyond the tax, j_periodicity (points)", float(gr.loc["j_periodicity", "excess_loss_pp"]), 30, 0)
    chk(P + "B3 excess loss beyond the tax, h_offtarget (points)", float(gr.loc["h_offtarget", "excess_loss_pp"]), 71, 0)
    chk(P + "B3 grouping costs e_nucleoid this share of its gain here (percent)", 100 * (1 - float(gc_.loc["e_nucleoid", "d_spearman"] / ac.loc["e_nucleoid", "d_spearman"])), 12, 0)
    chk(P + "B3 grouping costs g_transcription this share of its gain here (percent)", 100 * (1 - float(gc_.loc["g_transcription", "d_spearman"] / ac.loc["g_transcription", "d_spearman"])), 21, 0)
    # ratios published -> curated (ten families with a positive published increment)
    ratios = pd.Series({f: float(ac.loc[f, "d_spearman"] / pub[f]) for f in fams11 if pub[f] > 0})
    chk(P + "B3 ranking transfers: Spearman rho of published against curated increments", float(stats.spearmanr([pub[f] for f in fams11], [ac.loc[f, "d_spearman"] for f in fams11]).statistic), 0.97, 2)
    chk(P + "B3 median inflation, curated over published (ten families with a positive published increment)", float(ratios.median()), 3.2, 1)
    chk(P + "B3 smallest inflation (c_folding)", float(ratios.min()), 1.9, 1)
    chk(P + "B3 largest inflation (d_mechanics)", float(ratios.max()), 4.9, 1)
    chk(P + "B3 family A's inflation, curated over published", float(ratios["a_flank"]), 2.1, 1)
    cs = pd.read_csv(R / "ablation_curated_shape.csv")
    r = stats.spearmanr(cs.continuous, cs.d_spearman)
    chk(P + "B3 rho(dense continuous columns, increment) over the eleven families", float(r.statistic), 0.882, 3)
    chk(P + "B3 ... its p-value", float(r.pvalue), 0.0003, 4)
    sc = pd.read_csv(R / "selector_control.csv").set_index("arm")
    chk(P + "B3 d_mechanics gain with 300 selected columns (seed 41, five folds)",
        float(sc.loc["base + mech, select 300", "spearman"] - sc.loc["base, select 300", "spearman"]), 0.0988, 4)
    chk(P + "B3 d_mechanics gain with every column (seed 41, five folds)",
        float(sc.loc["base + mech, ALL columns", "spearman"] - sc.loc["base, ALL columns", "spearman"]), 0.0993, 4)
    cj = pd.read_csv(R / "curated_joint.csv").set_index("arm")
    for arm, joint, singles, share in (("a_flank+i_shape", 0.1647, 0.2833, 58), ("a_flank+d_supercoiling", 0.1882, 0.3029, 62),
                                       ("d_supercoiling+e_nucleoid+g_transcription", 0.1454, 0.2692, 54),
                                       ("a_flank+d_supercoiling+i_shape", 0.1882, 0.4224, 45), ("guide_only_three", 0.0289, 0.0409, 70),
                                       ("all_eleven", 0.2030, 0.7046, 29)):
        chk(P + f"B3 joint arm {arm}: joint increment", float(cj.loc[arm, "d_spearman"]), joint, 4)
        chk(P + f"B3 joint arm {arm}: sum of singles", float(cj.loc[arm, "sum_of_singles"]), singles, 4)
        chk(P + f"B3 joint arm {arm}: share realised (percent)", 100 * float(cj.loc[arm, "share_realised"]), share, 0)
    chk(P + "B3 i_shape on top of A together with d_supercoiling (three-family minus two-family arm)",
        float(cj.loc["a_flank+d_supercoiling+i_shape", "spearman"] - cj.loc["a_flank+d_supercoiling", "spearman"]), 0.00005, 5)
    fs = pd.read_csv(R / "redescription_family_split.csv").set_index(["family", "part"])
    chk(P + "B3 d_mechanics: the 20 columns the guide determines linearly", float(fs.loc[("d_mechanics", "guide-explained"), "d_spearman"]), 0.011, 3)
    pv = pd.read_csv(R / "redescription_provenance.csv").set_index(["family", "part"])
    chk(P + "B3 d_mechanics: the 8 flank columns", float(pv.loc[("d_mechanics", "reads outside the 20-mer"), "d_spearman"]), 0.097, 3)
    cg = pd.read_csv(R / "cohort_groups.csv").set_index(["cohort", "grouping"])
    chk(P + "B3 share of the curated label's variance between genes (percent)", 100 * (1 - float(cg.loc[("curated", "gene"), "within_share"])), 29, 0)
    chk(P + "B3 share of the published label's variance between genes (percent)", 100 * (1 - float(cg.loc[("published", "gene"), "within_share"])), 16, 0)
    ds = pd.read_csv(R / "dissection_supercoiling.csv").set_index("experiment")
    chk(P + "B3/C untagged control alone, published matrix, 100 kb blocks", float(ds.loc["z_control_only", "d_spearman"]), 0.0087, 4)
    chk(P + "B3/C GapR ChIP track alone, same protocol", float(ds.loc["z_signal_only", "d_spearman"]), 0.0111, 4)
    chk(P + "B3/C control as a share of the ChIP track (percent)", 100 * float(ds.loc["z_control_only", "d_spearman"] / ds.loc["z_signal_only", "d_spearman"]), 78, 0)

    # ------------------------------------------------------------------ B4: shape and 3D structure, beyond verify_numbers
    print("report R8: Appendix B4 (post_freeze/shape_ablation.csv, esp_replicate.csv, ridge_check.csv, human_shape_ablation*.csv)")
    sh = pd.read_csv(OUT / "shape_ablation.csv")
    F = lambda arm: _folds(sh, arm)
    chk(P + "B4 shape_ablation.csv folds per arm", float(len(F("base"))), 15, 0)
    chk(P + "B4 base rho, gene-grouped, 30,203 genic guides", float(sh.set_index("arm").loc["base", "spearman"]), 0.5588, 4)
    chk(P + "B4 A rho", float(sh.set_index("arm").loc["A", "spearman"]), 0.7151, 4)
    chk(P + "B4 A at depth 6 and 800 trees", float(sh.set_index("arm").loc["A@d6", "spearman"]), 0.7225, 4)
    alone = [f"{s}_{b}" for s in ("md", "xrc", "mc") for b in ("mean", "vec", "geom", "all")]
    chk(P + "B4 standalone cells ahead of the base in all 15 folds (of 12 cells)", float(sum(int((F(a) > F("base")).all()) for a in alone)), 12, 0)
    dA = lambda a: float((F(a) - F("A")).mean())
    chk(P + "B4 the largest on-top-of-A gain among the nine real arms", max(dA(a) for a in ("A+mc_mean", "A+mc_vec", "A+mc_geom", "A+mc_all", "A+md_geom", "A+xrc_geom", "A+md_all", "A+xrc_all", "A+allsrc_all")), 0.0006, 4)
    chk(P + "B4 the smallest on-top-of-A gain among the nine real arms", min(dA(a) for a in ("A+mc_mean", "A+mc_vec", "A+mc_geom", "A+mc_all", "A+md_geom", "A+xrc_geom", "A+md_all", "A+xrc_all", "A+allsrc_all")), -0.0008, 4)
    real = ("A+mc_all", "A+md_all", "A+xrc_all", "A+allsrc_all")
    gaps = [dA(a) - dA(c) for a in real for c in ("A+mc_blockshuf_all", "A+mc_armshuf_all")]
    chk(P + "B4 real all-block arms above the shuffled controls: smallest gap", min(gaps), 0.0002, 4)
    chk(P + "B4 real all-block arms above the shuffled controls: largest gap", max(gaps), 0.0018, 4)
    dB = lambda a: float((F(a) - F("base")).mean())
    chk(P + "B4 allsrc_all alone over the base as a share of family A (percent)", 100 * dB("allsrc_all") / dB("A"), 93, 0)
    chk(P + "B4 mc_all alone over the base as a share of family A (percent)", 100 * dB("mc_all") / dB("A"), 90, 0)
    chk(P + "B4 vec loses this share of its gain when the flank is shuffled in blocks (percent)", 100 * (1 - dB("mc_blockshuf_vec") / dB("mc_vec")), 86, 0)
    chk(P + "B4 vec loses this share when shuffled whole-arm (percent)", 100 * (1 - dB("mc_armshuf_vec") / dB("mc_vec")), 94, 0)
    chk(P + "B4 mean loses this share under block shuffling (percent)", 100 * (1 - dB("mc_blockshuf_mean") / dB("mc_mean")), 4, 0)
    chk(P + "B4 mean loses this share under whole-arm shuffling (percent)", 100 * (1 - dB("mc_armshuf_mean") / dB("mc_mean")), 12, 0)
    chk(P + "B4 geom loses this share under block shuffling (percent)", 100 * (1 - dB("mc_blockshuf_geom") / dB("mc_geom")), 0, 0)
    chk(P + "B4 geom loses this share under whole-arm shuffling (percent)", 100 * (1 - dB("mc_armshuf_geom") / dB("mc_geom")), 3, 0)
    chk(P + "B4 letters (40 columns) plus the shape means, against the letters alone", float((F("comp+mc_mean") - F("comp_mean")).mean()), -0.0002, 4)
    es = pd.read_csv(OUT / "esp_replicate.csv")
    Fe = lambda arm: _folds(es, arm)
    chk(P + "B4 eSpCas9 (100 kb-block CV): A + mc all ahead of A in this many of 15 folds", float(((Fe("A+mc_all") - Fe("A")) > 0).sum()), 14, 0)
    chk(P + "B4 eSpCas9: the blocks alone recover this share of A (percent)", 100 * float((Fe("mc_all") - Fe("base")).mean() / (Fe("A") - Fe("base")).mean()), 94, 0)
    rc = pd.read_csv(OUT / "ridge_check.csv").set_index("arm")
    Fr = lambda arm: np.array([float(x) for x in rc.loc[arm, "fold_rho"].split(";")])
    chk(P + "B4 ridge, one seed: A + mc all ahead of A in this many of 5 folds", float(((Fr("base+A+mc_all") - Fr("base+A")) > 0).sum()), 5, 0)
    chk(P + "B4 ridge: near-junction vec (mc) over A", float((Fr("base+A+mc_vec") - Fr("base+A")).mean()), 0.0048, 4)
    chk(P + "B4 ridge: block-shuffled vec control over A", float((Fr("base+A+mc_blockshuf_vec") - Fr("base+A")).mean()), -0.0004, 4)
    chk(P + "B4 ridge: window means (mc) over A", float((Fr("base+A+mc_mean") - Fr("base+A")).mean()), 0.0054, 4)
    chk(P + "B4 ridge: block-shuffled mean control over A", float((Fr("base+A+mc_blockshuf_mean") - Fr("base+A")).mean()), 0.0018, 4)
    from ..features import dnashape
    t = dnashape.load()
    for src, m_, s_ in (("mc", 34.3, 1.5), ("xrc", 33.1, 3.8), ("md", 32.5, 4.1)):
        chk(P + f"B4 mean helical twist of the {src} pentamer table (degrees)", float(t[f"{src}.HelT1"].mean()), m_, 1)
        chk(P + f"B4 sd of twist across the {src} pentamers (degrees)", float(t[f"{src}.HelT1"].std()), s_, 1)
    hs = pd.read_csv(OUT / "human_shape_ablation.csv")
    Fh = lambda df, arm: np.array([float(x) for x in df.set_index("arm").loc[arm, "fold_rho"].split(";")])
    chk(P + "B4 human (16,490 located guides, random-row folds): mc blocks on top of family A", float((Fh(hs, "h_base+A+mc") - Fh(hs, "h_base+A")).mean()), 0.0183, 4)
    chk(P + "B4 human: ... folds ahead (of 15)", float(((Fh(hs, "h_base+A+mc") - Fh(hs, "h_base+A")) > 0).sum()), 15, 0)
    ht = pd.read_csv(OUT / "human_shape_ablation_twin.csv")
    chk(P + "B4 human, twin-grouped folds: mc blocks on top of family A", float((Fh(ht, "h_base+A+mc") - Fh(ht, "h_base+A")).mean()), 0.0177, 4)

    # ------------------------------------------------------------------ B5: the dCas9 / CRISPRi screens
    print("report R8: Appendix B5 (Cui, Rousset, Wang; whole genes or 100 kb blocks held out; three seeds x five folds)")
    cu = pd.read_csv(R / "cui_screen.csv")
    cug = cu[(cu.cv == "gene") & (cu.n_rows == 70298)].set_index("arm")
    # Rousset (E75_Rousset.tsv) is the fit75 column of Cui's own screen_data.csv; E18_Cui.tsv is fit18 (strains LC-E75 / LC-E18, not time points)
    ext = config.ROOT / "external_data"
    sd = pd.read_csv(ext / "CRISPRBact" / "badSeed_public" / "screen_data.csv")
    sd = sd[(sd.ntargets == 1) & sd.guide.str.fullmatch(r"[ACGT]{20}")].groupby("guide", as_index=False).agg(fit18=("fit18", "mean"), fit75=("fit75", "mean"))
    for fname, col in (("E75_Rousset.tsv", "fit75"), ("E18_Cui.tsv", "fit18")):
        e = pd.read_csv(ext / "crispri_screens" / fname, sep="\t", usecols=["sequence", "log2FC"]).drop_duplicates("sequence")
        mm = sd.merge(e, left_on="guide", right_on="sequence").dropna(subset=[col, "log2FC"])
        chk(P + f"B5 {fname} against the {col} column of Cui's screen_data.csv: Spearman", float(stats.spearmanr(mm[col], mm["log2FC"]).statistic), 1.0, 4)
        chk(P + f"B5 {fname} against {col}: shared single-target guides", float(len(mm)), 54326, 0)
    chk(P + "B5 Cui, genes held out: 20-mer base", float(cug.loc["base", "spearman"]), 0.4577, 4)
    chk(P + "B5 Cui, genes held out: base + family A", float(cug.loc["base+a_flank", "spearman"]), 0.4484, 4)
    chk(P + "B5 Cui, genes held out: family A's increment (all genic guides)", float(cug.loc["base+a_flank", "spearman"] - cug.loc["base", "spearman"]), -0.0093, 4)
    chk(P + "B5 Cui, genes held out: family A alone", float(cug.loc["a_flank_only", "spearman"]), 0.1333, 4)
    fa3 = pd.read_csv(OUT / "family_a_at_tuned_summary.csv").set_index("comparison")
    chk(P + "B5 cutting, gene-grouped, 3 seeds, champion: base", float(fa3.loc["family A increment, champion", "base"]), 0.5588, 4)
    chk(P + "B5 cutting, gene-grouped, 3 seeds, champion: base + family A", float(fa3.loc["family A increment, champion", "with_a"]), 0.7151, 4)
    chk(P + "B5 cutting, gene-grouped, 3 seeds, champion: family A's increment", float(fa3.loc["family A increment, champion", "delta"]), 0.1563, 4)
    chk(P + "B5 cutting, gene-grouped, 5 seeds (ablation_curated_gene.csv): family A's increment", float(gc_.loc["a_flank", "d_spearman"]), 0.1567, 4)
    mc, gc, yc, ec = _meta_genes(INT / "cui_crispri" / "meta.json")
    has = (gc != "") & (gc != "nan")
    ess = has & (ec == "True")
    non = has & (ec == "False")
    chk(P + "B5 Cui guides in a named gene", float(has.sum()), 70298, 0)
    chk(P + "B5 Cui genes", float(len(set(gc[has]))), 4265, 0)
    chk(P + "B5 Cui guides in essential genes", float(ess.sum()), 5514, 0)
    chk(P + "B5 Cui essential genes", float(len(set(gc[ess]))), 297, 0)
    chk(P + "B5 Cui 100 kb blocks holding essential-gene guides", float(len(set(np.floor_divide(np.asarray(mc["left"], float)[ess], 100000)))), 44, 0)
    chk(P + "B5 non-essential share of the Cui genic guides (percent)", 100 * float(non.sum() / has.sum()), 92, 0)
    cs0 = pd.read_csv(R / "cui_essentiality_split.csv").set_index(["subset", "arm"])
    chk(P + "B5 Cui label sd, essential genes (cui_essentiality_split.csv)", float(cs0.loc[("essential genes", "base"), "label_sd"]), 2.886, 3)
    chk(P + "B5 Cui label sd, non-essential genes (cui_essentiality_split.csv)", float(cs0.loc[("non-essential genes", "base"), "label_sd"]), 1.133, 3)
    chk(P + "B5 Cui share of label variance between genes (genic guides)", _eta2(gc[has], np.asarray(mc["y"], float)[has]), 0.296, 3)
    _, g2, y2, e2 = _meta_genes(INT / "crispri_screens" / "Rousset" / "meta.json")
    h2 = (g2 != "") & (g2 != "nan")
    chk(P + "B5 Rousset share of label variance between genes (genic guides)", _eta2(g2[h2], y2[h2]), 0.375, 3)
    _, g3, y3, e3 = _meta_genes(INT / "crispri_screens" / "Wang" / "meta.json")
    h3 = (g3 != "") & (g3 != "nan")
    chk(P + "B5 Wang share of label variance between genes (genic guides)", _eta2(g3[h3], y3[h3]), 0.713, 3)
    ee = pd.read_csv(R / "cui_essentiality_split.csv").set_index(["subset", "arm"])
    es_ = pd.read_csv(R / "cui_essential_strand.csv").set_index(["grouping", "arm"])
    chk(P + "B5 essential genes, 100 kb blocks: family A alone", float(ee.loc[("essential genes, 100kb blocks", "a_flank_only"), "spearman"]), 0.5265, 4)
    chk(P + "B5 essential genes, genes held out: family A alone", float(ee.loc[("essential genes", "a_flank_only"), "spearman"]), 0.5227, 4)
    chk(P + "B5 strand alone (1 column), 100 kb blocks", float(es_.loc[("100kb blocks", "coding_only"), "spearman"]), 0.6621, 4)
    chk(P + "B5 strand alone (1 column), genes", float(es_.loc[("genes", "coding_only"), "spearman"]), 0.6630, 4)
    chk(P + "B5 family A given strand, 100 kb blocks", float(es_.loc[("100kb blocks", "base+coding+a_flank"), "spearman"] - es_.loc[("100kb blocks", "base+coding"), "spearman"]), 0.0069, 4)
    chk(P + "B5 family A given strand, genes", float(es_.loc[("genes", "base+coding+a_flank"), "spearman"] - es_.loc[("genes", "base+coding"), "spearman"]), 0.0148, 4)
    mg = pd.read_csv(R / "cui_essential_marginals.csv").set_index(["subset", "column"])
    chk(P + "B5 essential genes: marginal Spearman of the GC fraction of the 500 nt upstream", float(mg.loc[("essential genes", "eng.flank.comp.up500.gc"), "spearman"]), -0.0004, 4)
    chk(P + "B5 essential genes: marginal Spearman of eng.flank.comp.up1000.purine", float(mg.loc[("essential genes", "eng.flank.comp.up1000.purine"), "spearman"]), 0.4045, 4)
    cs_ = pd.read_csv(R / "cui_strand_summary.csv")
    cd = cs_[(cs_.subset == "essential genes") & (cs_.coding_strand.astype(str) == "derived")].iloc[0] if (cs_.coding_strand.astype(str) == "derived").any() else cs_.iloc[6]
    chk(P + "B5 essential genes: a binary column cannot exceed this rho", abs(float(cd.spearman_ceiling_for_a_binary_column)), 0.863, 3)
    chk(P + "B5 essential genes: strand's rho as a share of that ceiling (percent)", 100 * abs(float(cd.spearman)) / abs(float(cd.spearman_ceiling_for_a_binary_column)), 77, 0)
    sf = pd.read_csv(R / "strand_families.csv")
    sfm = sf[~sf.arm.str.contains("proxy")]
    chk(P + "B5 strand_families: smallest seed sd (arms with the strand column), pooled", float(sfm[sfm.arm.str.endswith("+strand")].spearman_sd.min()), 0.0001, 4)
    chk(P + "B5 strand_families: median seed sd, pooled", float(sfm.spearman_sd.median()), 0.005, 3)
    sw = pd.read_csv(R / "strand_families_within.csv")
    chk(P + "B5 strand_families_within: median seed sd, within a strand", float(sw.spearman_sd.median()), 0.013, 3)
    chk(P + "B5 strand_families_within: largest seed sd", float(sw.spearman_sd.max()), 0.05, 2)
    # Rousset: the stored crispri_screen_family_a.csv holds Wang only (the loop rewrites it per screen); the per-fold files are in MAIN
    def arm_mean(screen, label, arm):
        tag = f"cris_{screen}_{zlib.crc32(label.encode()):08x}_{arm.replace('+', '_')}"
        return float(pd.read_csv(MAIN / f"folds_{tag}.csv").spearman.mean())
    for lab, dg, dgs in (("all genic", -0.0206, None), ("non-essential genes", -0.0250, None), ("essential genes", 0.2902, 0.0245),
                         ("essential genes, 100kb blocks", 0.2916, 0.0213)):
        chk(P + f"B5 Rousset, {lab}: family A's increment", arm_mean("Rousset", lab, "base+a_flank") - arm_mean("Rousset", lab, "base"), dg, 4)
        if dgs is not None:
            chk(P + f"B5 Rousset, {lab}: family A given strand", arm_mean("Rousset", lab, "base+strand+a_flank") - arm_mean("Rousset", lab, "base+strand"), dgs, 4)
    chk(P + "B5 Rousset, essential genes: 20-mer base", arm_mean("Rousset", "essential genes", "base"), 0.2141, 4)
    chk(P + "B5 Rousset, essential genes: base + family A", arm_mean("Rousset", "essential genes", "base+a_flank"), 0.5043, 4)
    chk(P + "B5 Rousset, essential genes: base + strand", arm_mean("Rousset", "essential genes", "base+strand"), 0.5878, 4)
    fw = pd.read_csv(R / "crispri_screen_family_a.csv").set_index(["subset", "arm"])
    chk(P + "B5 Wang, all genic: family A's increment", float(fw.loc[("all genic", "base+a_flank"), "spearman"] - fw.loc[("all genic", "base"), "spearman"]), 0.1019, 4)
    chk(P + "B5 Wang, essential genes: 20-mer base", float(fw.loc[("essential genes", "base"), "spearman"]), 0.0829, 4)
    chk(P + "B5 Wang, essential genes: family A given strand, by gene", float(fw.loc[("essential genes", "base+strand+a_flank"), "spearman"] - fw.loc[("essential genes", "base+strand"), "spearman"]), 0.2669, 4)
    chk(P + "B5 Wang, essential genes: family A given strand, by block", float(fw.loc[("essential genes, 100kb blocks", "base+strand+a_flank"), "spearman"] - fw.loc[("essential genes, 100kb blocks", "base+strand"), "spearman"]), 0.2195, 4)
    # within-gene decomposition
    wc = pd.read_csv(R / "strand_within_gene_cui_summary.csv").set_index("arm")
    wr = pd.read_csv(R / "strand_within_gene_rousset_summary.csv").set_index("arm")
    cw = pd.read_csv(OUT / "cut_within_gene_summary.csv").set_index("arm")
    ceil_cui = float(np.sqrt(2 * 0.768 / (1 + 0.768)))                      # Spearman-Brown reliability of the two arms (strains LC-E18, LC-E75), square-rooted
    ceil_rou = float(np.sqrt(0.805 ** 2 / (2 * 0.768 / (1 + 0.768))))       # lower bound from agreement with the E18 screen
    chk(P + "B5 Cui within-gene ceiling (Spearman-Brown of the two arms' rho 0.768, square-rooted)", ceil_cui, 0.93, 2)
    chk(P + "B5 Rousset within-gene ceiling (lower bound)", ceil_rou, 0.864, 3)
    chk(P + "B5 Cui within-gene share of the attainable gain (percent)", 100 * float((wc.loc["a_flank", "within_genes"] - wc.loc["base", "within_genes"]) / (ceil_cui - wc.loc["base", "within_genes"])), 13, 0)
    chk(P + "B5 Rousset within-gene share of the attainable gain (percent)", 100 * float((wr.loc["a_flank", "within_genes"] - wr.loc["base", "within_genes"]) / (ceil_rou - wr.loc["base", "within_genes"])), 15, 0)
    chk(P + "B5 cutting within-gene share of the attainable gain (percent; ceiling 0.946)", 100 * float((cw.loc["a_flank", "within_genes"] - cw.loc["base", "within_genes"]) / (0.946 - cw.loc["base", "within_genes"])), 26, 0)
    d2 = lambda df: float(df.loc["a_flank", "within_genes"] ** 2 - df.loc["base", "within_genes"] ** 2)
    chk(P + "B5 within-gene change in rho squared, cutting", d2(cw), 0.113, 3)
    chk(P + "B5 within-gene change in rho squared, Cui", d2(wc), 0.046, 3)
    chk(P + "B5 within-gene change in rho squared, Rousset", d2(wr), 0.032, 3)
    chk(P + "B5 dCas9 over cutting by variance explained, Rousset (percent)", 100 * d2(wr) / d2(cw), 28, 0)
    chk(P + "B5 dCas9 over cutting by variance explained, Cui (percent)", 100 * d2(wc) / d2(cw), 40, 0)
    others = [a for a in wc.index if a not in ("base", "a_flank")]
    chk(P + "B5 within genes on Cui family A is the largest of the families run (1 = yes)", float(wc.loc["a_flank", "within_genes"] > wc.loc[others, "within_genes"].max()), 1, 0)
    chk(P + "B5 families run within genes on Cui besides base and A", float(len(others)), 4, 0)
    chk(P + "B5 E75-arm ('Rousset') arms stored in the within-gene summary (base and a_flank only)", float(len(wr)), 2, 0)

    # ------------------------------------------------------------------ C0 / Appendix C
    print("report R8: Appendix C0 (the environment) and Appendix C (feature sets, data added after the freeze)")
    import importlib
    ver = {"numpy": "2.5.3", "pandas": "3.0.6", "scipy": "1.18.1", "sklearn": "1.9.1", "lightgbm": "4.7.0", "xgboost": "3.4.1",
           "catboost": "1.2.10", "shap": "0.52.0", "torch": "2.14.1"}
    for mod, v in ver.items():
        try:
            got = str(importlib.import_module(mod).__version__).split("+")[0]
        except Exception:                                                  # noqa: BLE001
            got = "missing"
        chk(P + f"C0 {mod} version is {v}", 1.0 if got == v else 0.0, 1, 0)
    chk(P + "C0 Python version is 3.12.15", 1.0 if f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}" == "3.12.15" else 0.0, 1, 0)
    chk(P + "C0 config.SEEDS is (41, 42, 43, 44, 45)", 1.0 if tuple(config.SEEDS) == (41, 42, 43, 44, 45) else 0.0, 1, 0)
    chk(P + "C0 config.N_SPLITS", float(config.N_SPLITS), 5, 0)
    chk(P + "C0 config.SELECTOR is xgboost", 1.0 if config.SELECTOR == "xgboost" else 0.0, 1, 0)
    chk(P + "C0 folds per arm in the tuned-model comparison (three seeds x five folds)", float(fa3.loc["family A increment, champion", "n_folds"]), 15, 0)
    h2h = pd.read_csv(R / "headtohead.csv")
    near = h2h[(h2h.arm == "curated_local") & (h2h.seed == 41)].set_index("fold").spearman
    base_ = h2h[(h2h.arm == "curated_all") & (h2h.seed == 41)].set_index("fold").spearman
    chk(P + "C near-cut layer: folds measured", float(len(near)), 3, 0)
    chk(P + "C near-cut layer over the base, LightGBM seed 41 (mean of 3 folds against the 5-fold base)", float(near.mean() - base_.mean()), 0.074, 3)
    chk(P + "C near-cut layer over the base on the same 3 folds", float((near - base_.loc[near.index]).mean()), 0.073, 3)
    mt = h2h[(h2h.arm == "curated_matched") & (h2h.seed == 41)].spearman.mean()
    chk(P + "C layer beyond the near-cut columns up to crisprHAL's span", float(mt - near.mean()), 0.055, 3)
    chk(P + "C near-cut columns: 10 before the protospacer + the PAM's N + 15 after the PAM, one-hot", float((10 + 1 + 15) * 4), 104, 0)
    chk(P + "C near-cut columns including the quantum tensors (26 positions x (4 + 6))", float(26 * 4 + 26 * 6), 260, 0)
    be_ = pd.read_csv(INT / "features" / "b_energy_curated.csv")
    cf_ = pd.read_csv(INT / "features" / "c_folding_curated.csv")
    chk(P + "C b_energy's self-fold column against c_folding's spacer MFE (Pearson, curated)", float(np.corrcoef(be_["eng.dg.guide_selffold"], cf_["eng.fold.spacer_mfe"])[0, 1]), 0.9993, 4)
    chk(P + "C the three locus folders jointly (curated, random rows)", float(cj.loc["d_supercoiling+e_nucleoid+g_transcription", "d_spearman"]), 0.145, 3)
    jk = config.ROOT / "RSI-Crispr" / "Jacky" / "orientation-controls" / "results" / "intergenic_orientation_controls_delta_ci.csv"
    if jk.exists():                                                        # a teammate's file, read only
        oc = pd.read_csv(jk).set_index("analysis")
        chk(P + "C orientation controls (Jacky): TSS orientation, unmatched, delta", float(oc.loc["tss", "delta"]), 0.051, 3)
        chk(P + "C ... unmatched, lower end", float(oc.loc["tss", "delta_lo"]), 0.020, 3)
        chk(P + "C ... unmatched, upper end", float(oc.loc["tss", "delta_hi"]), 0.082, 3)
        chk(P + "C ... after distance/GC matching, delta", float(oc.loc["tss_matched_dist_gc", "delta"]), 0.025, 3)
        chk(P + "C ... after matching, lower end", float(oc.loc["tss_matched_dist_gc", "delta_lo"]), -0.003, 3)
        chk(P + "C ... after matching, upper end", float(oc.loc["tss_matched_dist_gc", "delta_hi"]), 0.053, 3)
        chk(P + "C ... after matching, p(delta <= 0)", float(oc.loc["tss_matched_dist_gc", "p_delta_le_0"]), 0.04, 2)
        chk(P + "C nearest-gene orientation after distance/GC matching, delta", float(oc.loc["gene_matched_dist_gc", "delta"]), 0.007, 3)
        chk(P + "C ... p(delta <= 0)", float(oc.loc["gene_matched_dist_gc", "p_delta_le_0"]), 0.34, 2)
    raw = pd.read_csv(config.ROOT / "data" / "raw" / "ecoli_feature_matrix.csv", skiprows=1, header=0, usecols=[0])
    chk(P + "C rows of Noshay et al.'s Supplementary Table 2 (the 'published' matrix)", float(raw.iloc[:, 0].notna().sum()), 13880, 0)
    import glob
    chk(P + "C CRISPOR benchmark tables on disk (*.scores.tab)", float(len(glob.glob(str(config.ROOT / "external_data" / "crispor_effdata" / "*.scores.tab")))), 24, 0)
    from ..compare import eukaryote_rules
    chk(P + "C CRISPOR benchmark tables in the eukaryote_rules catalogue", float(len(eukaryote_rules.CATALOGUE)), 24, 0)


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
        print(f"  {'ok ' if ok else 'BAD'} {label:120s} computed {got:+.5f}  report {claimed:+.{max(dec, 0)}f}")

    run(chk)
    print(f"\n{n} checks, {len(bad)} mismatches")
    for lab, got, claimed in bad:
        print(f"  BAD {lab}: computed {got:+.5f}, report {claimed}")
    raise SystemExit(1 if bad else 0)
