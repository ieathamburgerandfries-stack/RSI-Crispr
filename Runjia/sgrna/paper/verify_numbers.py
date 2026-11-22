"""Audit: recompute every headline number of Appendix B4 and the Finding 3 addendum from the result
files and compare with the value typed into the report. A mismatch is a transcription error or a stale
result. Tolerance is half a unit in the last quoted digit.

    python -m sgrna.paper.verify_numbers
"""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pandas as pd

from ..common import INTERIM, OUT
from sgrna import config

RUNJIA = pathlib.Path(OUT).parent.parent      # the fork's Runjia/, which holds the documents these checks read

bad, n = [], 0


def chk(label, got, claimed, dec):
    global n
    n += 1
    ok = abs(got - claimed) <= 0.5 * 10 ** (-dec) + 1e-9
    if not ok:
        bad.append((label, got, claimed))
    print(f"  {'ok ' if ok else 'BAD'} {label:58s} computed {got:+.5f}  report {claimed:+.{dec}f}")


def folds(df, arm):
    return np.array([float(x) for x in df.set_index("arm").loc[arm, "fold_rho"].split(";")])


def main():
    """Run every check and exit non-zero if any quoted number no longer matches its file."""
    sh = pd.read_csv(OUT / "shape_ablation.csv")
    F = lambda a: folds(sh, a)
    d = lambda a, ref="base": float((F(a) - F(ref)).mean())
    print("Appendix B4, alone over base")
    for src, vals in (("md", (0.099, 0.052, 0.031, 0.138)), ("xrc", (0.102, 0.056, 0.078, 0.140)), ("mc", (0.103, 0.053, 0.080, 0.141))):
        for blk, v in zip(("mean", "vec", "geom", "all"), vals):
            chk(f"{src}_{blk} - base", d(f"{src}_{blk}"), v, 3)
    chk("allsrc_all - base", d("allsrc_all"), 0.145, 3)
    chk("A - base", d("A"), 0.156, 3)
    chk("comp_mean - base", d("comp_mean"), 0.105, 3)
    chk("A_only_comp - base", d("A_only_comp"), 0.107, 3)
    print("Appendix B4, on top of A")
    for a, v in (("A+mc_mean", 0.0001), ("A+mc_vec", -0.0005), ("A+mc_geom", -0.0008), ("A+mc_all", -0.0007),
                 ("A+md_geom", -0.0006), ("A+xrc_geom", -0.0004), ("A+md_all", -0.0007), ("A+xrc_all", 0.0006),
                 ("A+allsrc_all", -0.0006), ("A+comp_mean", -0.0001), ("A+mc_blockshuf_all", -0.0012),
                 ("A+mc_armshuf_all", -0.0009)):
        chk(f"{a} - A", d(a, "A"), v, 4)
    chk("A+mc_all@600 - A@600", d("A+mc_all@600", "A@600"), -0.0008, 4)
    chk("A+mc_all@d6 - A@d6", d("A+mc_all@d6", "A@d6"), -0.0012, 4)
    chk("A+armshuf_all@d6 - A@d6", d("A+mc_armshuf_all@d6", "A@d6"), -0.0023, 4)
    chk("A@d6 - A", d("A@d6", "A"), 0.0073, 4)
    chk("A+mc_geom@600 - A@600", d("A+mc_geom@600", "A@600"), -0.0018, 4)
    print("Appendix B4, standalone mechanism")
    chk("mc_blockshuf_geom - base", d("mc_blockshuf_geom"), 0.080, 3)
    chk("mc_armshuf_geom - base", d("mc_armshuf_geom"), 0.077, 3)
    chk("mc_vec - base", d("mc_vec"), 0.053, 3)
    chk("mc_blockshuf_vec - base", d("mc_blockshuf_vec"), 0.008, 3)
    chk("mc_armshuf_vec - base", d("mc_armshuf_vec"), 0.003, 3)
    chk("md_geom_tw rho", float(sh.set_index("arm").loc["md_geom_tw", "spearman"]), 0.5914, 4)
    chk("md_geom rho", float(sh.set_index("arm").loc["md_geom", "spearman"]), 0.5897, 4)
    for a, v in (("A_only_seq", 0.055), ("A_only_qct", 0.070), ("A_only_win", 0.109), ("A_only_comp", 0.107)):
        chk(f"{a} - base", d(a), v, 3)
    for a, v in (("A_minus_seq", 0.0004), ("A_minus_comp", -0.0003), ("A_minus_win", -0.0020), ("A_minus_qct", -0.0054)):
        chk(f"{a} - A", d(a, "A"), v, 4)

    print("fresh seeds 44-46")
    chk("A#s2 - base#s2", d("A#s2", "base#s2"), 0.1570, 4)
    chk("mc_all#s2 - base#s2", d("mc_all#s2", "base#s2"), 0.1411, 4)
    chk("A+mc_all#s2 - A#s2", d("A+mc_all#s2", "A#s2"), -0.0006, 4)
    chk("A+armshuf_all#s2 - A#s2", d("A+mc_armshuf_all#s2", "A#s2"), -0.0010, 4)
    print("eSpCas9 replicate")
    es = pd.read_csv(OUT / "esp_replicate.csv"); E = lambda a: folds(es, a)
    chk("esp A - base", float((E("A") - E("base")).mean()), 0.1034, 4)
    chk("esp mc_all - base", float((E("mc_all") - E("base")).mean()), 0.0969, 4)
    chk("esp A+mc_all - A", float((E("A+mc_all") - E("A")).mean()), 0.0022, 4)
    chk("esp A+mc_geom - A", float((E("A+mc_geom") - E("A")).mean()), 0.0008, 4)
    chk("esp A+armshuf_all - A", float((E("A+mc_armshuf_all") - E("A")).mean()), 0.0016, 4)

    print("Ridge")
    rd = pd.read_csv(OUT / "ridge_check.csv"); R = lambda a: folds(rd, a)
    chk("ridge A+mc_all - A", float((R("base+A+mc_all") - R("base+A")).mean()), 0.0097, 4)
    chk("ridge A+md_all - A", float((R("base+A+md_all") - R("base+A")).mean()), 0.0078, 4)
    chk("ridge A+armshuf - A", float((R("base+A+mc_armshuf_all") - R("base+A")).mean()), 0.0002, 4)

    print("Cutting-screen within-gene")
    cw = pd.read_csv(OUT / "cut_within_gene_summary.csv").set_index("arm")
    chk("cut within base", cw.loc["base", "within_genes"], 0.618, 3)
    chk("cut within A", cw.loc["a_flank", "within_genes"], 0.703, 3)
    chk("cut within gain", cw.loc["a_flank", "within_genes_diff"], 0.085, 3)
    chk("cut between gain", cw.loc["a_flank", "between_genes_diff"], 0.275, 3)
    chk("cut overall gain", cw.loc["a_flank", "overall_diff"], 0.155, 3)
    chk("cut boot lo", cw.loc["a_flank", "within_diff_boot_lo"], 0.078, 3)
    chk("cut boot hi", cw.loc["a_flank", "within_diff_boot_hi"], 0.090, 3)

    print("Section 10: human side result")
    hs = pd.read_csv(OUT / "human_shape_ablation.csv").set_index("arm").spearman
    hl = pd.read_csv(OUT / "human_shape_ablation_locus.csv").set_index("arm").spearman
    h9 = pd.read_csv(OUT / "human_shape_ablation_locus_n900.csv").set_index("arm").spearman
    ht = pd.read_csv(OUT / "human_shape_ablation_twin.csv").set_index("arm").spearman
    hc = pd.read_csv(OUT / "human_twin_check.csv").set_index("arm").spearman
    for lab, v, c in (("random base", hs["h_base"], 0.3807), ("random A", hs["h_base+A"], 0.4287), ("random A+mc", hs["h_base+A+mc"], 0.4470),
                      ("random A+comp", hs["h_base+A+comp"], 0.3942), ("random A+dcomp", hs["h_base+A+dcomp"], 0.4039),
                      ("random A+mc block-shuffled", hs["h_base+A+mc[block-shuffled]"], 0.3779),
                      ("random A+mc arm-shuffled", hs["h_base+A+mc[arm-shuffled]"], 0.3729), ("random base+noise861", hs["h_base+noise861"], 0.3372),
                      ("random base+mc", hs["h_base+mc"], 0.4402), ("locus base", hl["h_base"], 0.2987), ("locus A", hl["h_base+A"], 0.3169),
                      ("locus A+mc", hl["h_base+A+mc"], 0.3319), ("locus900 A", h9["h_base+A"], 0.3193), ("locus900 A+mc", h9["h_base+A+mc"], 0.3311),
                      ("twin base", ht["h_base"], 0.3063), ("twin baseline harness", hc["harness_default"], 0.3356),
                      ("twin baseline library-defaults", hc["jacky_400_1000"], 0.2515)):
        chk(lab, float(v), c, 4)
    hb = pd.read_csv(OUT.parent / "human_baseline.csv").set_index("arm").spearman
    chk("random baseline harness", float(hb["harness_default"]), 0.4036, 4)
    chk("random baseline library-defaults", float(hb["jacky_400_1000"]), 0.5076, 4)
    dec = pd.read_csv(INTERIM / "human" / "decoded.csv", dtype={"sgRNAID": str})
    vc = dec.protospacer.value_counts()
    chk("human rows", len(dec), 17421, 0); chk("distinct 20-mers", len(vc), 13138, 0)
    chk("sequences occurring more than once", int((vc > 1).sum()), 4216, 0); chk("rows with a twin", int(vc[vc > 1].sum()), 8499, 0)
    ci = pd.read_csv(config.INTERIM / "curated_guide_index.csv")
    chk("bacterial rows", len(ci), 33567, 0); chk("bacterial distinct protospacers", ci.protospacer.nunique(), 33567, 0)


    print("Section 11: tuning")
    tc = pd.read_csv(OUT / "tune_confirm.csv").set_index("arm")
    for arm, rho, dd, lo, hi in (("champion", 0.7073, None, None, None), ("depth6", 0.7126, 0.0053, 0.0027, 0.0083),
                                 ("d099x", 0.7215, 0.0143, 0.0111, 0.0174), ("ensemble of top 5", 0.7246, 0.0174, 0.0144, 0.0204)):
        chk(f"confirm {arm} rho", float(tc.loc[arm, "rho"]), rho, 4)
        if dd is not None:
            chk(f"  vs champion", float(tc.loc[arm, "d_vs_champion"]), dd, 4); chk("  lo", float(tc.loc[arm, "d_lo"]), lo, 4)
            chk("  hi", float(tc.loc[arm, "d_hi"]), hi, 4)
    te = pd.read_csv(OUT / "tune_external.csv").pivot_table(index=["seed", "fold"], columns="arm", values="rho")
    chk("eSpCas9 champion mean", float(te["champion"].mean()), 0.7840, 4)
    for arm, v in (("d021x", 0.0227), ("d077x", 0.0257), ("d099x", 0.0255), ("depth6", 0.0174)):
        chk(f"eSpCas9 {arm} - champion", float((te[arm] - te["champion"]).mean()), v, 4)
        chk(f"  folds ahead", float(((te[arm] - te["champion"]) > 0).sum()), 10, 0)
    tb = pd.read_csv(OUT / "tune_budget_check.csv")
    chk("selector columns with nonzero gain", float(tb.n_nonzero_gain.dropna().iloc[0]), 878, 0)
    for nf, v in ((450, 0.7208), (900, 0.7205), (1500, 0.7202), (2500, 0.7199)):
        chk(f"budget {nf}: best-tree rho", float(tb[tb.nf == nf].rho.max()), v, 4)
    from ..tuning.tune_analyze import table as _tt
    tt = _tt(); dr = tt.drop(index=["champion", "depth6"])
    chk("stage-1 champion rho", float(tt.loc["champion", "rho"]), 0.7107, 4)
    for dep, v in ((3, 0.712), (4, 0.714), (5, 0.713), (6, 0.707), (7, 0.706), (8, 0.704), (9, 0.700), (10, 0.698)):
        chk(f"mean rho at depth {dep}", float(dr[dr.max_depth == dep].rho.mean()), v, 3)
    for nf, v in ((150, 0.698), (300, 0.709), (450, 0.711), (600, 0.711), (900, 0.708)):
        chk(f"mean rho at budget {nf}", float(dr[dr.nf == nf].rho.mean()), v, 3)


    tf = pd.read_csv(OUT / "tune_fullcv.csv").pivot_table(index=["seed", "fold"], columns="arm", values="rho")
    for arm, v in (("champion", 0.7151), ("depth6", 0.7223), ("tuned d099x", 0.7314), ("ensemble of 5", 0.7334)):
        chk(f"full cohort {arm}", float(tf[arm].mean()), v, 4)
    for arm, v in (("depth6", 0.0072), ("tuned d099x", 0.0162), ("ensemble of 5", 0.0183)):
        chk(f"full cohort {arm} - champion", float((tf[arm] - tf["champion"]).mean()), v, 4)
        chk("  folds ahead", float(((tf[arm] - tf["champion"]) > 0).sum()), 15, 0)


    print("Section 12: mixing model families (held-out genes)")
    mc = pd.read_csv(OUT / "mix_confirm.csv").set_index("arm")
    for arm, v in (("champion", 0.7073), ("xgb ensemble (5)", 0.7246), ("lgb (best)", 0.7206), ("lgb ensemble (2)", 0.7235), ("cat (best)", 0.7207),
                   ("cat ensemble (2)", 0.7242), ("et (best)", 0.6657), ("mlp (best)", 0.6920), ("ridge (best)", 0.6470),
                   ("equal blend: xgb+lgb", 0.7250), ("equal blend: xgb+cat", 0.7262), ("equal blend: xgb+lgb+cat", 0.7261),
                   ("equal blend: xgb+lgb+cat+et", 0.7224), ("equal blend: xgb+lgb+cat+et+mlp", 0.7234),
                   ("equal blend: xgb+lgb+cat+et+mlp+ridge", 0.7226), ("NNLS blend (weights from DEV OOF)", 0.7264)):
        chk(f"confirm {arm}", float(mc.loc[arm, "rho"]), v, 4)
    for arm, d_, lo_, hi_ in (("equal blend: xgb+cat", 0.0016, 0.0006, 0.0026), ("equal blend: xgb+lgb+cat", 0.0014, 0.0005, 0.0025),
                              ("NNLS blend (weights from DEV OOF)", 0.0017, 0.0005, 0.0030), ("equal blend: xgb+lgb", 0.0004, -0.0003, 0.0010)):
        chk(f"{arm} - xgb ensemble", float(mc.loc[arm, "d_vs_xgb_ens"]), d_, 4); chk("  lo", float(mc.loc[arm, "dx_lo"]), lo_, 4)
        chk("  hi", float(mc.loc[arm, "dx_hi"]), hi_, 4)
    chk("equal blend xgb+cat vs champion", float(mc.loc["equal blend: xgb+cat", "d_vs_champion"]), 0.0190, 4)
    chk("NNLS blend vs champion", float(mc.loc["NNLS blend (weights from DEV OOF)", "d_vs_champion"]), 0.0191, 4)
    mf = pd.read_csv(OUT / "mix_finalists.csv").set_index("name")
    for nm, v in (("lgb011", 0.7233), ("cat015", 0.7237), ("mlp006", 0.6988), ("ridge000", 0.6458), ("et001", 0.6642)):
        chk(f"stage-1 {nm}", float(mf.loc[nm, "stage1_rho"]), v, 4)
    pc = pd.read_csv(OUT / "mix_pred_corr.csv", index_col=0); rc = pd.read_csv(OUT / "mix_residual_corr.csv", index_col=0)
    chk("prediction corr xgb-lgb", float(pc.loc["xgb", "lgb"]), 0.985, 3); chk("prediction corr xgb-cat", float(pc.loc["xgb", "cat"]), 0.977, 3)
    chk("residual corr xgb-cat", float(rc.loc["xgb", "cat"]), 0.959, 3); chk("residual corr xgb-mlp", float(rc.loc["xgb", "mlp"]), 0.882, 3)
    nw = pd.read_csv(OUT / "mix_nnls_weights.csv").set_index("family").weight
    for fam_, v in (("xgb", 0.416), ("cat", 0.215), ("mlp", 0.180), ("lgb", 0.153)):
        chk(f"NNLS weight {fam_}", float(nw[fam_]), v, 3)

    print("Section 13: against crisprHAL 2 again")
    vs = {c: pd.read_csv(OUT / f"vs_crisprhal_{c}.csv") for c in ("espcas9", "wt", "folds")}
    def vsrow(c, arm, cond=None):
        d = vs[c]
        if cond:
            d = d[d.condition.str.contains(cond)]
        return d[d.arm == arm].iloc[0]
    for c, arm, cond, rho, hal, diff, lo, hi in (
            ("espcas9", "champion", None, 0.7842, 0.7821, 0.0021, -0.0031, 0.0072), ("espcas9", "tuned XGBoost (d099x)", None, 0.8107, 0.7821, 0.0286, 0.0239, 0.0331),
            ("espcas9", "XGBoost average of 5", None, 0.8104, 0.7821, 0.0283, 0.0237, 0.0328), ("espcas9", "mixed ensemble", None, 0.8095, 0.7821, 0.0273, 0.0227, 0.0318),
            ("wt", "champion", None, 0.6985, 0.6945, 0.0039, -0.0047, 0.0124), ("wt", "tuned XGBoost (d099x)", None, 0.7172, 0.6945, 0.0226, 0.0139, 0.0314),
            ("wt", "XGBoost average of 5", None, 0.7193, 0.6945, 0.0248, 0.0162, 0.0333), ("wt", "mixed ensemble", None, 0.7210, 0.6945, 0.0264, 0.0180, 0.0348),
            ("folds", "champion", "default", 0.7084, 0.6968, 0.0115, 0.0076, 0.0156), ("folds", "tuned XGBoost (d099x)", "default", 0.7256, 0.6968, 0.0288, 0.0251, 0.0326),
            ("folds", "XGBoost average of 5", "default", 0.7280, 0.6968, 0.0312, 0.0275, 0.0349), ("folds", "mixed ensemble", "default", 0.7293, 0.6968, 0.0324, 0.0287, 0.0362),
            ("folds", "champion", "tuned", 0.7084, 0.7040, 0.0044, 0.0005, 0.0082), ("folds", "tuned XGBoost (d099x)", "tuned", 0.7256, 0.7040, 0.0216, 0.0180, 0.0252),
            ("folds", "XGBoost average of 5", "tuned", 0.7280, 0.7040, 0.0240, 0.0204, 0.0276), ("folds", "mixed ensemble", "tuned", 0.7293, 0.7040, 0.0252, 0.0217, 0.0288)):
        r = vsrow(c, arm, cond)
        tag = f"{c}{'/' + cond if cond else ''} {arm}"
        chk(f"{tag} rho", float(r.rho), rho, 4); chk(f"  crisprHAL", float(r.crisprhal), hal, 4); chk("  diff", float(r["diff"]), diff, 4)
        chk("  lo", float(r.lo), lo, 4); chk("  hi", float(r.hi), hi, 4)
    for c, arm, v in (("espcas9", "champion", 1.9), ("espcas9", "tuned XGBoost (d099x)", 30.0), ("espcas9", "XGBoost average of 5", 131.6), ("espcas9", "mixed ensemble", 202.0)):
        chk(f"fit seconds {c} {arm}", float(vsrow(c, arm).fit_seconds), v, 1 if v < 10 else 0)

    print("Cost")
    cs = pd.read_csv(OUT / "cost_slicer.csv"); ch = pd.read_csv(OUT / "cost_crisprhal.csv"); cf = pd.read_csv(OUT / "cost_featurise.csv").set_index("part")
    cl = lambda m, b: float(cs[(cs.model == m) & (cs.batch == b)].latency_ms.iloc[0])
    for m, vals in (("original champion", (0.04, 0.21, 16.21)), ("tuned XGBoost (d099x)", (0.13, 1.81, 149.4)),
                    ("XGBoost average of 5", (0.54, 6.28, 535.0)), ("mixed ensemble", (1.56, 16.74, 1398.7))):
        for b, v in zip((1, 100, 10000), vals):
            chk(f"{m} batch {b} ms", cl(m, b), v, 2 if v < 20 else 1)
    for b, v in zip((1, 100, 10000), (15.59, 132.13, 11705.2)):
        chk(f"crisprHAL 2 batch {b} ms", float(ch[ch.batch == b].latency_ms.iloc[0]), v, 2 if v < 200 else 1)
    chk("featurisation total ms/guide", float(cf.loc["total", "per_guide_ms"]), 1.41, 2)
    chk("featurisation family A ms/guide", float(cf.loc["family A (348 columns from the 1,000-nt flank windows)", "per_guide_ms"]), 1.33, 2)
    chk("mixed ensemble total trees", float(cs[cs.model == "mixed ensemble"].total_trees.iloc[0]), 24800, 0)

    print("Section 14: what the settings say")
    dc = pd.read_csv(OUT / "interpret_depth.csv").set_index("max_depth")
    for dep, dv, cv in ((1, 0.6949, 0.6905), (2, 0.7203, 0.7177), (3, 0.7232, 0.7206), (4, 0.7253, 0.7210), (5, 0.7240, 0.7201), (6, 0.7219, 0.7188), (8, 0.7176, 0.7158)):
        chk(f"depth {dep} DEV", float(dc.loc[dep, "dev_rho"]), dv, 4); chk(f"depth {dep} CONFIRM", float(dc.loc[dep, "confirm_rho"]), cv, 4)
    wb = pd.read_csv(OUT / "interpret_within_between.csv").set_index(["model", "part"])
    for (m, part), rho, dd, lo_, hi_ in ((("champion", "between-gene"), 0.7115, 0, 0, 0), (("champion", "within-gene"), 0.6942, 0, 0, 0),
                                          (("tuned (d099x)", "between-gene"), 0.7293, 0.0179, 0.0077, 0.0280), (("tuned (d099x)", "within-gene"), 0.7124, 0.0182, 0.0144, 0.0219),
                                          (("average of five", "between-gene"), 0.7302, 0.0187, 0.0090, 0.0282), (("average of five", "within-gene"), 0.7158, 0.0217, 0.0183, 0.0251)):
        chk(f"{m} {part} rho", float(wb.loc[(m, part), "rho"]), rho, 4)
        if dd:
            chk("  gain", float(wb.loc[(m, part), "d_vs_champion"]), dd, 4); chk("  lo", float(wb.loc[(m, part), "d_lo"]), lo_, 4); chk("  hi", float(wb.loc[(m, part), "d_hi"]), hi_, 4)
    pp = pd.read_csv(OUT / "interpret_position_profile.csv")
    gp = pp[pp.region == "guide"]
    for m, a13, a1, p18 in (("original", 0.627, 0.100, 0.138), ("tuned", 0.618, 0.166, 0.144)):
        x = gp[gp.model == m].set_index("position").gain_share
        chk(f"{m} gain share positions 13-20", float(x.loc[13:20].sum()), a13, 3); chk(f"{m} positions 1-12", float(x.loc[1:12].sum()), a1, 3)
        chk(f"{m} position 18", float(x.loc[18]), p18, 3)
    for m, v in (("original", 0.062), ("tuned", 0.050)):
        chk(f"{m} 3 nt after the PAM", float(pp[(pp.model == m) & (pp.region == "downstream") & (pp.position <= 3)].gain_share.sum()), v, 3)
    gg = pd.read_csv(OUT / "interpret_gain_groups.csv")
    for m, ind, q, fl in (("original (depth 4, 400 trees, 300 columns)", 0.353, 0.373, 0.273), ("tuned (d099x)", 0.513, 0.271, 0.216)):
        x = gg[gg.model == m]
        chk(f"{m[:8]} indicator share", float(x[x.block.str.startswith("protospacer indicators")].gain_share.sum()), ind, 3)
        chk(f"{m[:8]} quantum share", float(x[x.block.str.startswith("protospacer quantum")].gain_share.sum()), q, 3)
        chk(f"{m[:8]} flank share", float(x[x.block.str.startswith("flank")].gain_share.sum()), fl, 3)
    ds = pd.read_csv(OUT / "interpret_data_scaling.csv").pivot_table(index=["fraction", "seed"], columns="model", values="rho")
    gain = (ds["d099x"] - ds["champion"]).groupby(level=0).mean()
    for fr, v in ((0.25, 0.0095), (0.5, 0.0061), (1.0, 0.0143)):
        chk(f"data scaling gain at {fr}", float(gain.loc[fr]), v, 4)


    lin = float(mc.loc["ridge (best)", "rho"]); top = float(dc.loc[4, "confirm_rho"]); tot = top - lin
    chk("trees add over linear (depth 4 - ridge)", tot, 0.074, 3)
    chk("share: nonlinearity within a column", (float(dc.loc[1, "confirm_rho"]) - lin) / tot, 0.59, 2)
    chk("share: pairs", (float(dc.loc[2, "confirm_rho"]) - float(dc.loc[1, "confirm_rho"])) / tot, 0.37, 2)
    chk("share: third and fourth order", (top - float(dc.loc[2, "confirm_rho"])) / tot, 0.04, 2)


    print("Section 15: metric by metric, seeds, data efficiency, clean comparison, speed")
    for sp, vals in (("espcas9", dict(hal=(0.7821, 0.5920, 0.7801, 0.6009, 0.4937, 95.4572), tuned=(0.0286, 0.0301, 0.0234, 0.0435, 0.0034, -1.9247))),
                     ("wt", dict(hal=(0.6945, 0.5066, 0.7203, 0.5152, 0.2808, 85.3614), tuned=(0.0226, 0.0221, 0.0173, 0.0285, 0.0193, -0.4408)))):
        pn = pd.read_csv(OUT / f"panel_{sp}.csv").set_index("arm")
        for m, h, d_ in zip(("spearman", "kendall_tau", "pearson", "r2", "top_decile_prec", "pick_percentile"), vals["hal"], vals["tuned"]):
            chk(f"{sp} crisprHAL {m}", float(pn.loc["crisprHAL 2", m]), h, 4 if m != "pick_percentile" else 2)
            chk(f"{sp} tuned - crisprHAL {m}", float(pn.loc["tuned XGBoost", "d_" + m]), d_, 4 if m != "pick_percentile" else 2)
    pn = pd.read_csv(OUT / "panel_espcas9.csv").set_index("arm")
    for m, lo_, hi_ in (("spearman", 0.0241, 0.0331), ("r2", 0.0366, 0.0503), ("top_decile_prec", -0.0151, 0.0286)):
        chk(f"espcas9 {m} lo", float(pn.loc["tuned XGBoost", "lo_" + m]), lo_, 4); chk(f"espcas9 {m} hi", float(pn.loc["tuned XGBoost", "hi_" + m]), hi_, 4)
    pw = pd.read_csv(OUT / "panel_wt.csv").set_index("arm")
    for m, lo_, hi_ in (("spearman", 0.0139, 0.0313), ("pearson", 0.0094, 0.0254), ("r2", 0.0169, 0.0397), ("pick_percentile", -10.3474, 9.7335)):
        chk(f"wt {m} lo", float(pw.loc["tuned XGBoost", "lo_" + m]), lo_, 4); chk(f"wt {m} hi", float(pw.loc["tuned XGBoost", "hi_" + m]), hi_, 4)
    tk = pd.read_csv(OUT / "topk_panel.csv")
    tg = lambda sp, arm, col: float(tk[(tk.split == sp) & (tk.arm == arm)][col].iloc[0])
    for sp, col, v in (("espcas9", "d_top5pct_precision", 0.019), ("espcas9", "d_pick50_percentile", -1.702), ("espcas9", "d_pick100_percentile", -0.806),
                       ("wt", "d_top5pct_precision", 0.048), ("wt", "lo_top5pct_precision", 0.003), ("wt", "d_pick100_percentile", 1.017), ("wt", "d_pick50_percentile", -1.958)):
        chk(f"topk {sp} {col}", tg(sp, "tuned XGBoost", col), v, 3)
    chk("calibration slope wt tuned", tg("wt", "tuned XGBoost", "calibration_slope"), 1.012, 3); chk("calibration slope wt crisprHAL", tg("wt", "crisprHAL 2", "calibration_slope"), 1.082, 3)
    ss = pd.read_csv(OUT / "seed_stability.csv"); ss["lead"] = ss.rho - ss.crisprhal
    for sp, mn, mx, mean in (("espcas9", 0.0280, 0.0286, 0.0284), ("wt", 0.0208, 0.0228, 0.0221)):
        x = ss[ss.split == sp].lead
        chk(f"seed lead {sp} min", float(x.min()), mn, 4); chk(f"seed lead {sp} max", float(x.max()), mx, 4); chk(f"seed lead {sp} mean", float(x.mean()), mean, 4)
    de = pd.read_csv(OUT / "data_efficiency.csv")
    hal_sc = pd.read_csv(OUT.parents[3] / "results" / "sample_complexity.csv"); hal_sc = hal_sc[(hal_sc.model == "crisprHAL 2") & (hal_sc.seed == 41)]
    dm = de.pivot_table(index="n_train", columns="arm", values="rho")
    for nn, ch_, tu_, hh_ in ((4000, 0.6524, 0.6576, 0.5235), (8000, 0.6814, 0.6915, 0.6323), (13880, 0.6988, 0.7106, 0.6733), (20000, 0.7053, 0.7198, 0.6879), (26853, 0.7076, 0.7251, 0.6950)):
        chk(f"n={nn} champion", float(dm.loc[nn, "champion"]), ch_, 4); chk(f"n={nn} tuned", float(dm.loc[nn, "tuned d099x"]), tu_, 4)
        chk(f"n={nn} crisprHAL stored", float(hal_sc[hal_sc.n_train == nn].spearman.mean()), hh_, 4)
    cv = pd.read_csv(OUT / "clean_vs_hal.csv")
    cr = lambda vs, arm, col: float(cv[(cv.vs.str.contains(vs)) & (cv.arm == arm)][col].iloc[0])
    for vs, arm, d_, lo_, hi_ in (("default", "tuned XGBoost (d099x)", 0.0324, 0.0248, 0.0403), ("default", "original champion", 0.0181, 0.0098, 0.0265),
                                  ("tuned", "tuned XGBoost (d099x)", 0.0240, 0.0162, 0.0318), ("tuned", "original champion", 0.0097, 0.0018, 0.0179)):
        chk(f"clean vs {vs} {arm} diff", cr(vs, arm, "diff"), d_, 4); chk("  lo", cr(vs, arm, "lo"), lo_, 4); chk("  hi", cr(vs, arm, "hi"), hi_, 4)
    chk("clean n genes", cr("default", "original champion", "n_genes"), 1204, 0); chk("clean n guides", cr("default", "original champion", "n"), 8943, 0)
    bw = pd.read_csv(OUT / "blend_with_hal.csv")
    bb = lambda sp, w, col: float(bw[(bw.split == sp) & (bw.weight_on_crisprhal == w)][col].iloc[0])
    chk("blend espcas9 equal weights", bb("espcas9", 0.5, "rho"), 0.8099, 4); chk("blend wt equal weights", bb("wt", 0.5, "rho"), 0.7257, 4)
    chk("blend wt vs best single", bb("wt", 0.5, "vs_best_single"), 0.0085, 4); chk("blend wt lo", bb("wt", 0.5, "lo"), 0.0040, 4); chk("blend wt hi", bb("wt", 0.5, "hi"), 0.0127, 4)
    chk("blend espcas9 vs best single", bb("espcas9", 0.5, "vs_best_single"), -0.0008, 4)
    te_ = pd.read_csv(OUT / "topend_confirm.csv").groupby("arm").mean(numeric_only=True)
    for arm, rho_, t10 in (("mse", 0.7215, 0.2912), ("weighted-3", 0.7174, 0.2852), ("weighted-9", 0.7102, 0.2890), ("pairwise", 0.5758, 0.2506), ("ndcg", 0.5119, 0.2289)):
        chk(f"top-end {arm} rho", float(te_.loc[arm, "spearman"]), rho_, 4); chk(f"top-end {arm} top-10% precision", float(te_.loc[arm, "top10pct_prec"]), t10, 4)
    rc_ = pd.read_csv(OUT / "topend_rank_capacity.csv"); chk("pairwise best rho over lr/depth/trees", float(rc_.spearman.max()), 0.6017, 4)
    e2 = pd.read_csv(OUT / "e2e_slicer.csv"); eh = pd.read_csv(OUT / "e2e_crisprhal.csv")
    ee = lambda m, b: float(e2[(e2.model == m) & (e2.batch == b)].ms_per_guide.iloc[0])
    for m, vals in (("original champion", (0.63, 0.072, 0.079)), ("tuned XGBoost (d099x)", (0.72, 0.089, 0.093)), ("XGBoost average of 5", (1.16, 0.139, 0.141)), ("mixed ensemble", (2.16, 0.218, 0.206))):
        for b, v in zip((1, 100, 10000), vals):
            chk(f"e2e {m} batch {b} ms/guide", ee(m, b), v, 2 if v >= 0.6 else 3)
    for b, v in zip((1, 100, 10000), (15.29, 1.365, 1.214)):
        chk(f"e2e crisprHAL batch {b} ms/guide", float(eh[eh.batch == b].ms_per_guide.iloc[0]), v, 3 if v < 2 else 2)
    for b, v in zip((1, 100, 10000), (21, 15, 13)):
        chk(f"e2e speed ratio tuned batch {b}", float(eh[eh.batch == b].ms_per_guide.iloc[0]) / ee("tuned XGBoost (d099x)", b), v, 0)
    fe = pd.read_csv(OUT / "featuriser_equivalence.csv").set_index("item").value
    chk("featurisers: guides compared", float(fe["guides compared"]), 33567, 0); chk("featurisers: base max abs difference", float(fe["base columns: max abs difference"]), 0.0, 6)
    chk("featurisers: family A max abs difference", float(fe["family A columns: max abs difference after float32 cast"]), 0.0, 6)
    chk("featurisers: champion prediction difference", float(fe["champion: prediction max abs difference, stored vs fast features"]), 0.0, 6)
    chk("featurisers: d099x prediction difference", float(fe["d099x: prediction max abs difference, stored vs fast features"]), 0.0, 6)


    em = de[de.arm == "tuned d099x, early-stopped"].groupby("n_train").rho.mean()
    chk("early-stopped at 4,000", float(em.loc[4000]), 0.6550, 4); chk("early-stopped at 26,853", float(em.loc[26853]), 0.7238, 4)
    t8, t13 = float(dm.loc[8000, "tuned d099x"]), float(dm.loc[13880, "tuned d099x"])
    n_match = 8000 + (0.6950 - t8) / (t13 - t8) * (13880 - 8000)
    chk("labels at which the tuned model matches crisprHAL's 26,853-guide figure (rounded to 100)", round(n_match, -2), 9100, 0)
    chk("ratio of labels", 26853 / n_match, 3.0, 0)
    chk("clean crisprHAL default rho", cr("default", "original champion", "crisprhal"), 0.6892, 4); chk("clean crisprHAL tuned-config rho", cr("tuned", "original champion", "crisprhal"), 0.6976, 4)
    chk("clean tuned rho", cr("default", "tuned XGBoost (d099x)", "rho"), 0.7215, 4)
    chk("calibration slope espcas9 tuned", tg("espcas9", "tuned XGBoost", "calibration_slope"), 1.047, 3); chk("calibration slope espcas9 crisprHAL", tg("espcas9", "crisprHAL 2", "calibration_slope"), 1.053, 3)
    chk("residual correlation espcas9", float(bw[(bw.split == "espcas9") & bw.weight_on_crisprhal.isna()].rho.iloc[0]), 0.83, 2)
    chk("residual correlation wt", float(bw[(bw.split == "wt") & bw.weight_on_crisprhal.isna()].rho.iloc[0]), 0.81, 2)
    chk("fast family A ms per guide", float(fe["family A ms per guide (whole cohort, one call)"]), 0.068, 3); chk("fast base ms per guide", float(fe["base ms per guide (whole cohort, one call)"]), 0.019, 3)

    print("16.1 family A at the tuned settings")
    fa = pd.read_csv(OUT / "family_a_at_tuned_summary.csv").set_index("comparison")
    fr = pd.read_csv(OUT / "family_a_at_tuned.csv")
    chk("16.1 folds measured", len(fr) / 8, 15, 0)            # 4 arms x 2 matrices per fold
    chk("16.1 base-only rankings never reach family A", float(fr[fr.matrix == "base"].n_flank_cols.abs().max()), 0, 0)
    for arm, (b, w, d_, lo, hi) in (("champion", (0.5588, 0.7151, 0.1563, 0.1515, 0.1616)),
                                    ("depth6", (0.5569, 0.7223, 0.1655, None, None)),
                                    ("d099x", (0.5617, 0.7314, 0.1697, 0.1659, 0.1738)),
                                    ("d077x", (0.5631, 0.7310, 0.1680, None, None))):
        r = fa.loc[f"family A increment, {arm}"]
        chk(f"16.1 {arm} base only", float(r.base), b, 4)
        chk(f"16.1 {arm} base+A", float(r.with_a), w, 4)
        chk(f"16.1 {arm} family A increment", float(r.delta), d_, 4)
        if lo is not None:
            chk(f"16.1 {arm} increment lo", float(r.lo), lo, 4); chk(f"16.1 {arm} increment hi", float(r.hi), hi, 4)
        chk(f"16.1 {arm} folds positive", float(r.folds_positive), 15, 0)
    bo = fa.loc[[i for i in fa.index if i.startswith("family A increment, best-of-four")][0]]
    chk("16.1 best-of-four increment", float(bo.delta), 0.1683, 4)
    chk("16.1 best-of-four lo", float(bo.lo), 0.1642, 4); chk("16.1 best-of-four hi", float(bo.hi), 0.1728, 4)
    tb = fa.loc["tuning gain (d099x - champion) on base"]
    ta = fa.loc["tuning gain (d099x - champion) on base+A"]
    chk("16.1 tuning gain on base only", float(tb.delta), 0.0028, 4)
    chk("16.1 tuning gain on base, lo", float(tb.lo), 0.0009, 4); chk("16.1 tuning gain on base, hi", float(tb.hi), 0.0050, 4)
    chk("16.1 tuning gain on base, folds positive", float(tb.folds_positive), 9, 0)
    chk("16.1 tuning gain on base+A", float(ta.delta), 0.0162, 4)
    chk("16.1 tuning gain on base+A, lo", float(ta.lo), 0.0152, 4); chk("16.1 tuning gain on base+A, hi", float(ta.hi), 0.0173, 4)
    chk("16.1 tuning gain on base+A, folds positive", float(ta.folds_positive), 15, 0)
    chk("16.1 ratio of tuning gains, with A over without", float(ta.delta / tb.delta), 5.7, 1)
    # the two arms that must reproduce tune_fullcv.csv, or this is a third protocol rather than the project's
    fc = pd.read_csv(OUT / "tune_fullcv.csv").groupby("arm").rho.mean()
    chk("16.1 champion base+A reproduces tune_fullcv", float(fa.loc["family A increment, champion"].with_a), float(fc.loc["champion"]), 3)
    chk("16.1 d099x base+A reproduces tune_fullcv", float(fa.loc["family A increment, d099x"].with_a), float(fc.loc["tuned d099x"]), 3)
    chk("16.1 ceiling gap at the tuned model", 0.946 - float(fa.loc["family A increment, d099x"].with_a), 0.21, 2)
    # how much a base-only search would have to find for the reading to flip
    _p = fr.pivot_table(index="arm", columns="matrix", values="rho")
    _need = float(_p.loc["d099x", "base+A"]) - float(fa.loc["family A increment, champion"].delta)
    _best, _champ = float(_p["base"].max()), float(_p.loc["champion", "base"])
    chk("16.1 base-only rho needed to flip the reading", _need, 0.5751, 4)
    chk("16.1   beyond the best of the four", _need - _best, 0.0120, 4)
    chk("16.1   largest base-only gain observed", _best - _champ, 0.0042, 4)
    chk("16.1   ratio", (_need - _best) / (_best - _champ), 2.9, 1)

    print("16.2 the human additivity claim under twin-grouped folds")
    ht = pd.read_csv(OUT / "human_additivity_twin_summary.csv").set_index(["folds", "arm"])["mean"]
    hc = pd.read_csv(OUT / "human_additivity_twin_claim.csv").set_index("folds")
    # the positive control: every row-wise arm must reproduce the published figure
    for arm, v in (("champion", 0.4074), ("plus_depth", 0.4659), ("plus_lr", 0.4490), ("plus_trees", 0.4435),
                   ("plus_nosubsample", 0.3999), ("boosting_defaults", 0.5076), ("ridge", 0.3232)):
        chk(f"16.2 row-wise {arm} reproduces published", float(ht[("row-wise", arm)]), v, 4)
    for arm, v in (("champion", 0.3377), ("plus_depth", 0.3315), ("plus_lr", 0.2560), ("plus_trees", 0.3226),
                   ("plus_nosubsample", 0.3333), ("boosting_defaults", 0.2515), ("ridge", 0.3001)):
        chk(f"16.2 twin-grouped {arm}", float(ht[("twin-grouped", arm)]), v, 4)
    chk("16.2 row-wise ridge/boosting", float(hc.loc["row-wise", "ridge_over_boosting"]) * 100, 63.7, 1)
    chk("16.2 twin-grouped ridge/boosting", float(hc.loc["twin-grouped", "ridge_over_boosting"]) * 100, 88.8, 1)
    chk("16.2 row-wise depth 4->6", float(hc.loc["row-wise", "depth_4_to_6"]), 0.0585, 4)
    chk("16.2 twin-grouped depth 4->6", float(hc.loc["twin-grouped", "depth_4_to_6"]), -0.0062, 4)
    chk("16.2 twin-grouped depth lo", float(hc.loc["twin-grouped", "depth_lo"]), -0.0093, 4)
    chk("16.2 twin-grouped depth hi", float(hc.loc["twin-grouped", "depth_hi"]), -0.0034, 4)
    chk("16.2 twin-grouped best arm is the champion", 1.0 if hc.loc["twin-grouped", "best_boosting_arm"] == "champion" else 0.0, 1, 0)
    chk("16.2 row-wise best arm is boosting_defaults", 1.0 if hc.loc["row-wise", "best_boosting_arm"] == "boosting_defaults" else 0.0, 1, 0)
    chk("16.2 kingdom gap in the ratio, row-wise (points)", 95.6 - float(hc.loc["row-wise", "ridge_over_boosting"]) * 100, 31.9, 1)
    chk("16.2 kingdom gap in the ratio, twin-grouped (points)", 95.6 - float(hc.loc["twin-grouped", "ridge_over_boosting"]) * 100, 6.8, 1)
    hw = pd.read_csv(OUT / "human_additivity_twin.csv")
    chk("16.2 twin-grouped folds share no protospacer",
        float(hw[hw.folds == "twin-grouped"].shared_protospacers.abs().max()), 0, 0)
    chk("16.2 folds per arm per scheme", len(hw) / (7 * 2), 15, 0)

    print("16.3 a second gene split with its own search")
    s2 = pd.read_csv(OUT / "tune_split2_confirm.csv").set_index("arm")
    win = [a for a in s2.index if a.startswith("split-2 winner")][0]
    chk("16.3 split-2 champion rho", float(s2.loc["champion", "rho"]), 0.7329, 4)
    chk("16.3 split-2 depth6 gain", float(s2.loc["depth6", "d_vs_champion"]), 0.0045, 4)
    chk("16.3 split-2 winner rho", float(s2.loc[win, "rho"]), 0.7450, 4)
    chk("16.3 split-2 winner gain", float(s2.loc[win, "d_vs_champion"]), 0.0121, 4)
    chk("16.3 split-2 winner gain lo", float(s2.loc[win, "d_lo"]), 0.0092, 4)
    chk("16.3 split-2 winner gain hi", float(s2.loc[win, "d_hi"]), 0.0152, 4)
    chk("16.3 split-2 top-3 average rho", float(s2.loc["average of top 3", "rho"]), 0.7488, 4)
    chk("16.3 split-2 top-3 average gain", float(s2.loc["average of top 3", "d_vs_champion"]), 0.0159, 4)
    chk("16.3 split-2 d099x (contaminated) gain", float(s2.loc["d099x (contaminated)", "d_vs_champion"]), 0.0143, 4)
    # the selection effect: contaminated minus honest, against the ~+0.002 section 11 got another way
    _sel = float(s2.loc["d099x (contaminated)", "d_vs_champion"]) - float(s2.loc[win, "d_vs_champion"])
    chk("16.3 selection effect (contaminated - fresh search)", _sel, 0.0022, 4)
    # split-to-split variation in the level, which is the point of running a second split
    _c1 = pd.read_csv(OUT / "tune_confirm.csv").set_index("arm")
    chk("16.3 split-1 champion rho", float(_c1.loc["champion", "rho"]), 0.7073, 4)
    chk("16.3 level swing between splits", float(s2.loc["champion", "rho"]) - float(_c1.loc["champion", "rho"]), 0.026, 3)
    _g1 = float([_c1.loc[a, "d_vs_champion"] for a in _c1.index if a.startswith("tuned best")][0])
    chk("16.3 split-1 tuning gain", _g1, 0.0143, 4)
    chk("16.3 gain agreement between splits", abs(_g1 - float(s2.loc[win, "d_vs_champion"])), 0.0022, 3)
    s2s = pd.read_csv(OUT / "tune_split2_stage1.csv")
    chk("16.3 split-2 draws scored", float(s2s.draw.nunique()), 42, 0)      # 40 fresh + champion + depth6

    print("18 imported design choices")
    mf = pd.read_csv(OUT / "lit_merf.csv").groupby("arm")[["pooled", "between", "within"]].mean()
    R, M = "tuned regression (d099x)", "mixed effects, gene intercept"
    chk("18.1 regression pooled", float(mf.loc[R, "pooled"]), 0.7215, 4)
    chk("18.1 regression between-gene", float(mf.loc[R, "between"]), 0.7293, 4)
    chk("18.1 regression within-gene", float(mf.loc[R, "within"]), 0.7124, 4)
    chk("18.1 mixed-effect pooled", float(mf.loc[M, "pooled"]), 0.7213, 4)
    chk("18.1 mixed-effect between-gene", float(mf.loc[M, "between"]), 0.7213, 4)
    chk("18.1 mixed-effect within-gene", float(mf.loc[M, "within"]), 0.7140, 4)
    chk("18.1 within-gene gain", float(mf.loc[M, "within"] - mf.loc[R, "within"]), 0.0016, 4)
    chk("18.1 between-gene loss", float(mf.loc[M, "between"] - mf.loc[R, "between"]), -0.0080, 4)
    bn = pd.read_csv(OUT / "lit_binary.csv").groupby("arm")[["rho", "top_decile_precision", "pick10_percentile"]].mean()
    chk("18.2 regression rho", float(bn.loc["tuned regression", "rho"]), 0.7215, 4)
    chk("18.2 regression top-decile precision", float(bn.loc["tuned regression", "top_decile_precision"]), 0.2912, 4)
    chk("18.2 regression pick percentile", float(bn.loc["tuned regression", "pick10_percentile"]), 85.10, 2)
    for fr, v in ((10, 0.6135), (20, 0.6679), (30, 0.6913)):
        chk(f"18.2 classifier top {fr}% rho", float(bn.loc[f"classifier, top {fr}%", "rho"]), v, 4)
    chk("18.2 best classifier rho is below regression",
        1.0 if bn.loc[[a for a in bn.index if a.startswith("classifier")], "rho"].max() < bn.loc["tuned regression", "rho"] else 0.0, 1, 0)
    chk("18.2 best blend top-decile precision", float(bn.loc["regression + classifier top 30%", "top_decile_precision"]), 0.2938, 4)

    print("19 the field's ceiling, recomputed")
    import math
    sb = lambda r, k=2: k * r / (1 + (k - 1) * r)
    chk("19 ceiling vs 1 replicate at r=0.71", math.sqrt(0.71), 0.843, 3)
    chk("19 ceiling vs 1 replicate at r=0.77", math.sqrt(0.77), 0.877, 3)
    chk("19 2-replicate reliability at r=0.71", sb(0.71), 0.830, 3)
    chk("19 2-replicate reliability at r=0.77", sb(0.77), 0.870, 3)
    chk("19 ceiling vs 2-rep mean at r=0.71", math.sqrt(sb(0.71)), 0.911, 3)
    chk("19 ceiling vs 2-rep mean at r=0.77", math.sqrt(sb(0.77)), 0.933, 3)
    chk("19 our replicate agreement -> our ceiling", math.sqrt(sb(0.8097)), 0.946, 3)
    chk("19 understatement at r=0.71", math.sqrt(sb(0.71)) - 0.7, 0.211, 3)
    chk("19 understatement at r=0.77", math.sqrt(sb(0.77)) - 0.7, 0.233, 3)
    chk("19 tuned model as % of our ceiling", 0.7314 / 0.946 * 100, 77.3, 1)

    print("20 our own two questions")
    rc = pd.read_csv(OUT / "novel_relative_composition.csv").set_index("columns")
    chk("20.1 absolute, C. rodentium", float(rc.loc["absolute (as shipped)", "crod_rho"]), 0.7142, 4)
    chk("20.1 relative, C. rodentium", float(rc.loc["species-relative percentile", "crod_rho"]), 0.7105, 4)
    chk("20.1 transfer difference", float(rc.loc["species-relative percentile", "crod_rho"] - rc.loc["absolute (as shipped)", "crod_rho"]), -0.0037, 4)
    chk("20.1 home difference (monotone no-op)", float(rc.loc["species-relative percentile", "home_cv_rho"] - rc.loc["absolute (as shipped)", "home_cv_rho"]), -0.0003, 4)
    chk("20.1 C. rodentium guides", float(rc.loc["absolute (as shipped)", "n_crod"]), 25210, 0)
    fd = pd.read_csv(OUT / "fold_distance.csv").set_index("scheme")
    for s, mn, fi in (("curated / gene-grouped", 3, 0.0), ("curated / random rows", 3, 0.0),
                      ("human / twin-grouped", 3, 0.0), ("human / random rows", 0, 0.391)):
        chk(f"20.2 {s}: min Hamming", float(fd.loc[s, "min"]), mn, 0)
        chk(f"20.2 {s}: fraction identical", float(fd.loc[s, "frac_identical"]), fi, 3)
    chk("20.2 twin count predicts the leak fraction (0.49 x 0.8)", 0.49 * 0.8, float(fd.loc["human / random rows", "frac_identical"]), 2)

    print("21 mechanism")
    dp = pd.read_csv(OUT / "mechanism_distance_profile.csv")
    g = dp[dp.statistic == "gc"].set_index(["side", "window_nt"]).partial_given_target_gc
    for side, vals in (("up", {50: 0.172, 250: 0.203, 500: 0.196, 1000: 0.173}),
                       ("dn", {50: 0.146, 250: 0.197, 500: 0.194, 1000: 0.181})):
        for w, v in vals.items():
            chk(f"21.1 GC {side} {w}nt partial", float(g[(side, w)]), v, 3)
    pu = dp[dp.statistic == "purine"].partial_given_target_gc.abs().max()
    chk("21.1 purine: largest |partial| at any window/side", float(pu), 0.018, 3)
    chk("21.1 GC peaks at 250nt upstream", 1.0 if g[("up", 250)] == g[("up",)].max() else 0.0, 1, 0)
    chk("21.1 up/dn largest disagreement", float(max(abs(g[("up", w)] - g[("dn", w)]) for w in (50, 250, 500, 1000))), 0.026, 3)
    pb = pd.read_csv(OUT / "mechanism_position_base.csv").set_index(["position", "base"])
    for pos, b, v in ((20, "T", 0.348), (20, "G", -0.322), (18, "G", -0.266), (18, "T", 0.210),
                      (20, "A", 0.193), (19, "C", -0.161), (19, "T", 0.152), (14, "C", -0.147),
                      (14, "T", 0.141), (19, "G", 0.138), (15, "C", -0.133), (15, "G", 0.103)):
        chk(f"21.2 position {pos} base {b}", float(pb.loc[(pos, b), "effect_sd"]), v, 3)
    chk("21.2 G at 20 interval excludes zero", 1.0 if pb.loc[(20, "G"), "hi"] < 0 else 0.0, 1, 0)
    early = pd.read_csv(OUT / "mechanism_position_base.csv").query("position <= 7").effect_sd.abs().max()
    chk("21.2 largest |effect| at positions 1-7", float(early), 0.039, 3)
    chk("21.2 strongest effect overall is position 20", float(pd.read_csv(OUT / "mechanism_position_base.csv").effect_sd.abs().max()), 0.348, 3)

    print("21.3-21.5 cross-kingdom rules, resolution floor, interaction pairs")
    cmp_ = pd.read_csv(OUT / "mechanism_position_base_compare.csv").set_index(["position", "base"])
    from scipy.stats import spearmanr as _sp
    chk("21.3 rank correlation of the two matrices", float(_sp(cmp_.ecoli, cmp_.human).statistic), -0.052, 3)
    both_clear = cmp_[cmp_.ecoli_clear & cmp_.human_clear]
    chk("21.3 cells clear of zero in both", float(len(both_clear)), 27, 0)
    chk("21.3 of those, opposite in sign", float((np.sign(both_clear.ecoli) != np.sign(both_clear.human)).sum()), 15, 0)
    for pos, b, ec, hu in ((20, "G", -0.322, 0.131), (20, "T", 0.348, -0.170)):
        chk(f"21.3 position {pos} {b}, E. coli", float(cmp_.loc[(pos, b), "ecoli"]), ec, 3)
        chk(f"21.3 position {pos} {b}, human", float(cmp_.loc[(pos, b), "human"]), hu, 3)
    chk("21.3 position 20 G inverts", 1.0 if cmp_.loc[(20, "G"), "ecoli"] < 0 < cmp_.loc[(20, "G"), "human"] else 0.0, 1, 0)

    res = pd.read_csv(OUT / "novel_resolution.csv").set_index("arm")
    base, ref = float(res.loc["base only", "rho"]), float(res.loc["base + family A comp block (24 cols)", "rho"])
    chk("21.4 base only", base, 0.5617, 4)
    chk("21.4 family A comp block reference", ref, 0.6732, 4)
    for cols, rho, share in ((4, 0.6462, 75.7), (8, 0.6616, 89.6), (16, 0.6679, 95.2),
                             (32, 0.6701, 97.2), (80, 0.6718, 98.7), (200, 0.6687, 96.0)):
        lbl = [a for a in res.index if a.startswith(f"base + {cols} binned")][0]
        chk(f"21.4 {cols} binned columns", float(res.loc[lbl, "rho"]), rho, 4)
        chk(f"21.4 {cols} columns, share of composition signal", (float(res.loc[lbl, "rho"]) - base) / (ref - base) * 100, share, 1)
    chk("21.4 the curve turns over at 200 columns",
        1.0 if float(res.loc[[a for a in res.index if a.startswith("base + 200 binned")][0], "rho"])
        < float(res.loc[[a for a in res.index if a.startswith("base + 32 binned")][0], "rho"]) else 0.0, 1, 0)
    chk("21.4 all of family A", float(res.loc["base + all of family A (348 cols)", "rho"]), 0.7314, 4)

    bp = pd.read_csv(OUT / "mechanism_pairs_blocks.csv").set_index("block_pair").gain_share
    chk("21.5 largest block pair share", float(bp.max()), 0.058, 3)
    chk("21.5 flank composition x protospacer basepair quantum",
        float(bp["flank: window composition x protospacer quantum: basepair"]), 0.048, 3)
    cp = pd.read_csv(OUT / "mechanism_pairs_columns.csv").set_index("col_pair").gain_share
    chk("21.5 strongest single pair share", float(cp.max()), 0.0186, 4)
    chk("21.5 strongest pair is up250 GC x position 20",
        1.0 if "up250.gc" in cp.idxmax() and "p20" in cp.idxmax() else 0.0, 1, 0)

    print("21.6 human flank gain under twin-grouped folds")
    tw = pd.read_csv(OUT / "human_shape_ablation_twin.csv").set_index("arm").spearman
    rw = pd.read_csv(OUT / "human_shape_ablation.csv").set_index("arm").spearman
    chk("21.6 twin h_base", float(tw["h_base"]), 0.3063, 4)
    chk("21.6 twin h_base+A", float(tw["h_base+A"]), 0.3289, 4)
    chk("21.6 twin h_base+A+mc", float(tw["h_base+A+mc"]), 0.3466, 4)
    chk("21.6 twin family A increment", float(tw["h_base+A"] - tw["h_base"]), 0.0226, 4)
    chk("21.6 row-wise family A increment", float(rw["h_base+A"] - rw["h_base"]), 0.0480, 4)
    chk("21.6 increment survives (clear of zero)", 1.0 if tw["h_base+A"] - tw["h_base"] > 0.01 else 0.0, 1, 0)
    chk("21.6 shape on A, twin folds", float(tw["h_base+A+mc"] - tw["h_base+A"]), 0.0177, 4)
    chk("21.6 shape on A, row-wise folds", float(rw["h_base+A+mc"] - rw["h_base+A"]), 0.0183, 4)

    print("cross-kingdom transfer, re-quoted at champion")
    kt = pd.read_csv(config.RESULTS / "kingdom_transfer_boot.csv").set_index(["direction", "settings"])
    for d, s, v, lo, hi in (("ecoli -> human", "champion", -0.046, -0.061, -0.031),
                            ("human -> ecoli", "champion", -0.016, -0.032, 0.001),
                            ("ecoli -> human", "human_fitted", -0.039, -0.054, -0.024),
                            ("human -> ecoli", "human_fitted", -0.006, -0.023, 0.011)):
        chk(f"transfer {d} ({s})", float(kt.loc[(d, s), "spearman"]), v, 3)
        chk(f"transfer {d} ({s}) lo", float(kt.loc[(d, s), "ci_lo"]), lo, 3)
        chk(f"transfer {d} ({s}) hi", float(kt.loc[(d, s), "ci_hi"]), hi, 3)
    chk("transfer: ecoli->human excludes zero at champion",
        1.0 if bool(kt.loc[("ecoli -> human", "champion"), "excludes_zero"]) else 0.0, 1, 0)
    chk("transfer: human->ecoli does NOT exclude zero at champion",
        0.0 if bool(kt.loc[("human -> ecoli", "champion"), "excludes_zero"]) else 1.0, 1, 0)
    chk("transfer: the asymmetry holds under both settings",
        1.0 if (bool(kt.loc[("ecoli -> human", "champion"), "excludes_zero"])
                and bool(kt.loc[("ecoli -> human", "human_fitted"), "excludes_zero"])
                and not bool(kt.loc[("human -> ecoli", "champion"), "excludes_zero"])
                and not bool(kt.loc[("human -> ecoli", "human_fitted"), "excludes_zero"])) else 0.0, 1, 0)

    print("figures: ACS geometry and completeness")
    import os
    man = pd.read_csv(OUT / "figure_manifest.csv")
    chk("figures produced", float(len(man)), 7, 0)
    chk("widest figure within the 7.0 in double-column limit", float(man.width_in.max()), 7.0, 2)
    chk("no figure exceeds the 7.0 in limit", 0.0 if (man.width_in > 7.0 + 1e-6).any() else 1.0, 1, 0)
    chk("deepest figure within the 9.167 in limit", 1.0 if (man.height_in <= 9.167).all() else 0.0, 1, 0)
    chk("single-column figure is within 3.33 in", float(man.width_in.min()), 3.33, 2)
    _fd = OUT / "figures"
    chk("every figure has a vector PDF", float(sum((_fd / f"{n_}.pdf").exists() for n_ in man.figure)), 7, 0)
    chk("every figure has a 600 dpi PNG", float(sum((_fd / f"{n_}.png").exists() for n_ in man.figure)), 7, 0)

    print("figure 2: the shift null, against a de-duplicated baseline")
    _sh = pd.read_csv(config.RESULTS / "shift_null.csv").sort_values("shift_bp")
    _bl_raw = pd.read_csv(config.RESULTS / "folds_shift_baseline.csv")
    _bl = _bl_raw.drop_duplicates(["seed", "fold"])
    _g = _sh.spearman.to_numpy() - float(_bl.spearman.mean())
    chk("figure 2: repeated fold rows in the frozen baseline file", float(len(_bl_raw) - len(_bl)), 3, 0)
    chk("figure 2: aligned gain over the de-duplicated no-flank baseline", float(_g[0]), 0.0769, 4)
    chk("figure 2: gain at 1 kb", float(_g[1]), 0.0061, 4)
    chk("figure 2: gain at 10 kb", float(_g[2]), -0.0132, 4)
    chk("figure 2: share of the gain left at 1 kb, percent (report: 8%)", float(_g[1] / _g[0] * 100), 8.0, 0)

    print("8. what the geometry columns are (geom_importance)")
    gi = pd.read_csv(OUT / "geom_importance.csv").set_index("column")
    xt = gi[gi.index.str.startswith("xtwist.")]
    chk("8. the five largest geometry columns are all excess twist", float(gi.head(5).index.str.startswith("xtwist.").sum()), 5, 0)
    chk("8. excess twist, letters: lowest R2", float(xt.r2_from_composition.head(5).min()), 0.43, 2)
    chk("8. excess twist, letters: highest R2", float(xt.r2_from_composition.head(5).max()), 0.67, 2)
    chk("8. excess twist, dinucleotides by ridge: lowest R2 (five largest)", float(xt.r2_from_dinucleotide_ridge.head(5).min()), 0.86, 2)
    chk("8. excess twist, dinucleotides by ridge: highest R2", float(xt.r2_from_dinucleotide_ridge.head(5).max()), 0.97, 2)
    chk("8. excess twist, dinucleotides by boosting: lowest R2 (five largest)", float(xt.r2_from_dinucleotide_gb.head(5).min()), 0.85, 2)
    chk("8. excess twist, dinucleotides by boosting: highest R2", float(xt.r2_from_dinucleotide_gb.head(5).max()), 0.94, 2)
    chk("8. downstream 499-nt window, dinucleotides by ridge", float(gi.loc["xtwist.dn499", "r2_from_dinucleotide_ridge"]), 0.968, 3)
    chk("8. downstream 499-nt window, dinucleotides by boosting", float(gi.loc["xtwist.dn499", "r2_from_dinucleotide_gb"]), 0.937, 3)
    chk("8. downstream 100-nt window, dinucleotides by ridge", float(gi.loc["xtwist.dn100", "r2_from_dinucleotide_ridge"]), 0.865, 3)
    pc = gi[gi.index.str.startswith(("path.", "axis."))]
    chk("8. path/axis columns: highest R2 from letters", float(pc.r2_from_composition.max()), 0.0, 1)
    chk("8. path/axis columns: highest R2 even from dinucleotides", float(pc.r2_from_dinucleotide_ridge.max()), 0.15, 2)

    print("22.1 the per-locus precompute")
    fl = pd.read_csv(OUT / "flank_locus_bench.csv").set_index("width_nt")
    for w, old, new, pre, up in ((1_000, 0.0614, 0.0113, 0.0051, 5.4), (10_000, 0.0604, 0.0044, 0.0009, 13.6),
                                 (100_000, 0.0632, 0.0039, 0.0005, 16.1), (1_000_000, 0.1383, 0.0042, 0.0005, 33.1)):
        chk(f"22.1 a_flank_fast ms/guide at {w:,} nt", float(fl.loc[w, "a_flank_fast_ms_per_guide"]), old, 4)
        chk(f"22.1 flank_locus ms/guide at {w:,} nt", float(fl.loc[w, "flank_locus_ms_per_guide"]), new, 4)
        chk(f"22.1 precompute ms/guide at {w:,} nt", float(fl.loc[w, "precompute_ms_per_guide"]), pre, 4)
        chk(f"22.1 speed-up at {w:,} nt", float(fl.loc[w, "speedup"]), up, 1)
    chk("22.1 NGG site density (nt per site, 100 kb)", float(fl.loc[100_000, "nt_per_guide"]), 8.6, 1)
    el = pd.read_csv(OUT / "e2e_slicer_locus.csv")
    le = lambda m, w: float(el[(el.model == m) & (el.width_nt == w)].ms_per_guide.iloc[0])
    for m, vals in (("original champion", (0.0234, 0.0223, 0.0300)), ("tuned XGBoost (d099x)", (0.0392, 0.0361, 0.0477)),
                    ("XGBoost average of 5", (0.0890, 0.0835, 0.1003))):
        for w, v in zip((1_000, 100_000, 1_000_000), vals):
            chk(f"22.1 end-to-end {m} at {w:,} nt", le(m, w), v, 4)
    hal10k = float(eh[eh.batch == 10000].ms_per_guide.iloc[0])
    chk("22.1 design-path ratio, tuned vs crisprHAL 2", hal10k / le("tuned XGBoost (d099x)", 100_000), 34, 0)
    chk("22.1 design-path ratio, champion vs crisprHAL 2", hal10k / le("original champion", 100_000), 54, 0)
    chk("22.1 crisprHAL 2 forward pass as a share of its cost",
        float(eh[eh.batch == 10000].model_ms.iloc[0]) / float(eh[eh.batch == 10000].ms_per_call.iloc[0]), 0.95, 2)
    chk("22.1 crisprHAL 2 encoding as a share of its cost",
        100 * float(eh[eh.batch == 10000].encode_ms.iloc[0]) / float(eh[eh.batch == 10000].ms_per_call.iloc[0]), 4.7, 1)

    print("22.2 the ceiling correction, parametrically")
    cp = pd.read_csv(OUT / "ceiling_parametric.csv").set_index(["replicate_agreement_r", "k_replicates"])
    for r_, k_, v in ((0.71, 2, 0.9113), (0.77, 2, 0.9328), (0.8097, 2, 0.9460),     # the three rows Finding 6 quotes
                      (0.71, 1, 0.8426), (0.77, 1, 0.8775), (0.50, 2, 0.8165), (0.90, 2, 0.9733)):
        chk(f"22.2 ceiling at r={r_}, k={k_}", float(cp.loc[(r_, k_), "ceiling"]), v, 4)
    for r_, k_, v in ((0.71, 1, 83), (0.77, 1, 80), (0.71, 2, 77), (0.77, 2, 75)):
        chk(f"22.2 share of ceiling at rho 0.70, r={r_}, k={k_}",
            100 * float(cp.loc[(r_, k_), "share_of_ceiling_at_rho_0p70"]), v, 0)

    print("22.3 the bound on any local shape predictor")
    lb = pd.read_csv(OUT / "local_sequence_bound_summary.csv").set_index(["arm", "matrix"])
    lr = pd.read_csv(OUT / "local_sequence_bound.csv")
    chk("22.3 folds measured", float(len(lr)) / 6, 15, 0)                      # 3 matrices x 2 configurations per fold
    chk("22.3 base+A at the tuned configuration reproduces the headline rho",
        float(lr[(lr.matrix == "base+A") & (lr.arm == "d099x")].rho.mean()), 0.7314, 4)
    chk("22.3 base+A at the champion", float(lr[(lr.matrix == "base+A") & (lr.arm == "champion")].rho.mean()), 0.7151, 4)
    for arm, mat, rho_, dl, lo_, hi_, npick in (
            ("d099x", "base+A+loc50", 0.7316, 0.0002, -0.0004, 0.0008, 8.7),
            ("d099x", "base+A+loc50shuf", 0.7309, -0.0004, -0.0012, 0.0003, 5.3),
            ("champion", "base+A+loc50", 0.7144, -0.0008, -0.0015, -0.0001, 2.1),
            ("champion", "base+A+loc50shuf", 0.7152, 0.0001, -0.0006, 0.0008, 0.9)):
        chk(f"22.3 {arm} {mat} rho", float(lb.loc[(arm, mat), "rho"]), rho_, 4)
        chk(f"22.3 {arm} {mat} difference", float(lb.loc[(arm, mat), "delta"]), dl, 4)
        chk(f"22.3 {arm} {mat} lo", float(lb.loc[(arm, mat), "lo"]), lo_, 4)
        chk(f"22.3 {arm} {mat} hi", float(lb.loc[(arm, mat), "hi"]), hi_, 4)
        chk(f"22.3 {arm} {mat} local columns picked", float(lb.loc[(arm, mat), "local_cols_selected"]), npick, 1)
    chk("22.3 the real block is inside the shuffled block's interval at the tuned configuration",
        1.0 if (float(lb.loc[("d099x", "base+A+loc50"), "delta"]) <= float(lb.loc[("d099x", "base+A+loc50shuf"), "hi"])) else 0.0, 1, 0)
    chk("22.3 upper bound on any function of the inner +-50 nt",
        float(lb.loc[("d099x", "base+A+loc50"), "hi"]), 0.0008, 4)

    print("22.4 the package imports, and importing it writes nothing")
    ic = json.loads((OUT / "import_check.json").read_text())
    chk("22.4 modules swept", float(ic["n_modules"]), 143, 0)
    chk("22.4 modules excluded by name", float(ic["n_skipped"]), 1, 0)
    chk("22.4 modules that fail to import", float(len(ic["fails"])), 0, 0)
    chk("22.4 files under results/", float(ic["files_in_results"]), 675, 0)
    chk("22.4 files changed by importing the package", float(ic["files_changed_by_import"]), 0, 0)

    print("4. headroom: does the model leave information on the table? (representation_headroom.csv, restored 22 Nov)")
    rh = pd.read_csv(OUT.parent / "representation_headroom.csv"); rh["a"] = rh.rho_with_residual.abs()
    def _top(arm, grp):
        g = rh[(rh.arm == arm) & (rh.group == grp)].sort_values("a", ascending=False)
        return float(g.a.iloc[0]), g.column.iloc[0]
    for arm, given, held, fam in (("published_full", 0.028, 0.147, "a_flank"), ("subset_427", 0.022, 0.146, "a_flank"),
                                  ("subset_427+a_flank", 0.028, 0.070, None)):
        chk(f"4. {arm}: strongest column it has", _top(arm, "given to the model")[0], given, 3)
        v, col = _top(arm, "dropped from the matrix")
        chk(f"4. {arm}: strongest column held back", v, held, 3)
        if fam:
            chk(f"4. {arm}: the held-back column is family A (the positive control)", 1.0 if col.startswith("eng.flank.") else 0.0, 1, 0)
    chk("4. with family A given, nothing held back exceeds 0.10", float(_top("subset_427+a_flank", "dropped from the matrix")[0] < 0.10), 1, 0)

    # The files below are read from the repository's own results/ (OUT.parent), not config.RESULTS: config.RESULTS is the
    # main project's untracked folder, and a check that read that copy would pass without testing what a clone contains.
    print("the human ladder (folds_human_base_cv.csv, folds_human_base_a_cv.csv; cited by RESULTS.md as `human_ladder`)")
    hb = pd.read_csv(OUT.parent / "folds_human_base_cv.csv").spearman.mean()
    ha = pd.read_csv(OUT.parent / "folds_human_base_a_cv.csv").spearman.mean()
    chk("human ladder: base, random folds", float(hb), 0.382, 3)
    chk("human ladder: base + family A", float(ha), 0.434, 3)
    chk("human ladder: family A increment", float(ha - hb), 0.052, 3)

    print("5. importance across seven model families (importance_model_pairs.csv, restored 22 Nov)")
    ip = pd.read_csv(OUT.parent / "importance_model_pairs.csv")
    trees = {"catboost", "extra_trees", "lightgbm", "random_forest", "xgboost"}
    it = ip[ip.model_a.isin(trees) & ip.model_b.isin(trees)]
    chk("5. model pairs (all families)", float(len(ip[ip.method == "native"]) / ip.fold.nunique()), 21, 0)
    chk("5. model pairs (tree families, the table)", float(len(it[it.method == "native"]) / it.fold.nunique()), 10, 0)
    for meth, col, blk in (("native", 0.050, 0.531), ("permutation", 0.258, 0.581), ("shap", 0.496, 0.800)):
        g = it[it.method == meth]
        chk(f"5. trees, {meth}: agreement per column", float(g.rho_column.mean()), col, 3)
        chk(f"5. trees, {meth}: agreement per block", float(g.rho_block.mean()), blk, 3)
    for meth, col, blk in (("native", -0.057, 0.522), ("permutation", 0.187, 0.512)):
        g = ip[ip.method == meth]
        chk(f"5. all 21 pairs, {meth}: agreement per column", float(g.rho_column.mean()), col, 3)
        chk(f"5. all 21 pairs, {meth}: agreement per block", float(g.rho_block.mean()), blk, 3)
    chk("5. adding the linear models lowers the default agreement",
        1.0 if ip[ip.method == "native"].rho_column.mean() < it[it.method == "native"].rho_column.mean() else 0.0, 1, 0)

    print("4. k-mer order without selection (kmer_order_paired.csv, restored 22 Nov)")
    from scipy import stats as _st
    ko = pd.read_csv(OUT.parent / "kmer_order_paired.csv").pivot_table(index=["seed", "fold"], columns="arm", values="spearman")
    for lab, (x, y), mean_, up, pv in (("+ trimers", ("mono+di", "mono+di+tri"), 0.0035, 12, 0.025),
                                       ("+ tetramers on top", ("mono+di+tri", "mono+di+tri+tetra"), -0.0028, 4, 0.013),
                                       ("net", ("mono+di", "mono+di+tri+tetra"), 0.0007, 8, 0.62)):
        dd = ko[y] - ko[x]
        chk(f"4. {lab}: mean difference", float(dd.mean()), mean_, 4)
        chk(f"4. {lab}: folds ahead of 15", float((dd > 0).sum()), up, 0)
        chk(f"4. {lab}: paired t p-value", float(_st.ttest_1samp(dd, 0).pvalue), pv, 3 if pv < 0.1 else 2)

    print("7. crisprHAL 2 re-run at seed 41 (crisprhal_rerun.csv, restored 22 Nov)")
    cr_ = pd.read_csv(OUT.parent / "crisprhal_rerun.csv")
    of = cr_[cr_.arm == "our_folds"].sort_values("fold")
    for k_, v_ in enumerate((0.6855, 0.6988, 0.6978, 0.7022, 0.7012), start=1):
        chk(f"7. crisprHAL 2 published config, fold {k_}", float(of[of.fold == k_].spearman.iloc[0]), v_, 4)

    print("the layout, and the pointers to it")
    from . import doc_refs
    stale_refs = doc_refs.scan()
    chk("documents and docstrings: references to modules, paths or Colab scripts that do not exist", float(len(stale_refs)), 0, 0)
    for rel, ln, kind, ref in stale_refs[:5]:
        print(f"      stale: {rel}:{ln} {kind} {ref}")
    _sg = pathlib.Path(OUT).parent.parent / "sgrna"
    loose = sorted(f.name for f in _sg.glob("*.py") if f.name not in ("__init__.py", "config.py", "io_utils.py", "common.py"))
    chk("sgrna: modules sitting outside a package (besides config, io_utils, common)", float(len(loose)), 0, 0)
    from . import code_map
    _on_disk = {f.stem for f in _sg.rglob("*.py") if f.name != "__init__.py"}
    _text = (_sg.parent / "CODE_MAP.md").read_text()
    chk("CODE_MAP.md: modules on disk that the map does not list", float(len([m for m in _on_disk if f"| `{m}` |" not in _text])), 0, 0)
    chk("sgrna: packages, by function", float(sum(1 for d in _sg.iterdir() if d.is_dir() and (d / "__init__.py").exists())), 10, 0)
    chk("sgrna: folders left over from the old layouts (post_freeze, strand, screens, ext)",
        float(sum((_sg / n_).exists() for n_ in ("post_freeze", "strand", "screens", "ext"))), 0, 0)

    print("Finding 7, the response curve and the hypotheses (hypotheses.py)")
    h1s = pd.read_csv(OUT / "hypotheses_h1_strata.csv").set_index(["position", "base"])
    for (pos_, b_), (rho_, ols_) in {(20, "A"): (0.219, 21.2), (20, "T"): (0.184, 19.6), (20, "C"): (0.169, 16.4), (20, "G"): (0.118, 12.5),
                                    (18, "A"): (0.214, 20.3), (18, "T"): (0.202, 19.8), (18, "C"): (0.137, 14.7), (18, "G"): (0.115, 13.1)}.items():
        chk(f"H1 rho of label on flank GC, position {pos_} base {b_}", float(h1s.loc[(pos_, b_), "rho"]), rho_, 3)
        chk(f"H1 OLS slope, position {pos_} base {b_}", float(h1s.loc[(pos_, b_), "ols_slope"]), ols_, 1)
    h1c = pd.read_csv(OUT / "hypotheses_h1_contrasts.csv"); h1c = h1c[h1c.steepest == "T minus G"].set_index("position")
    chk("H1 T minus G at position 20", float(h1c.loc[20, "difference"]), 0.066, 3); chk("H1 T minus G at position 20, lower bound", float(h1c.loc[20, "lo"]), 0.029, 3)
    chk("H1 T minus G at position 20, upper bound", float(h1c.loc[20, "hi"]), 0.102, 3); chk("H1 T minus G at position 18", float(h1c.loc[18, "difference"]), 0.087, 3)
    h2 = pd.read_csv(OUT / "hypotheses_h2_thirds.csv").set_index("kind")
    chk("H2 pooled low-third slope", float(h2.loc["pooled", "slope_low_third"]), 41.9, 1); chk("H2 pooled high-third slope", float(h2.loc["pooled", "slope_high_third"]), -5.6, 1)
    chk("H2 pooled difference", float(h2.loc["pooled", "low_minus_high"]), 47.6, 1); chk("H2 pooled difference, lower bound", float(h2.loc["pooled", "lo_diff"]), 43.0, 1)
    chk("H2 within-gene low-third slope", float(h2.loc["within genes", "slope_low_third"]), 19.3, 1); chk("H2 within-gene high-third slope", float(h2.loc["within genes", "slope_high_third"]), 13.8, 1)
    chk("H2 within-gene difference", float(h2.loc["within genes", "low_minus_high"]), 5.4, 1); chk("H2 within-gene difference, lower bound", float(h2.loc["within genes", "lo_diff"]), 0.6, 1)
    cv = pd.read_csv(OUT / "hypotheses_h2_curve.csv"); cv = cv[cv.kind == "pooled"].sort_values("bin")
    for k_, v_ in enumerate((-3.09, -0.40, 0.11, 0.01, 0.05, 0.07, 0.14, 0.14, 0.02, -0.05)):
        chk(f"H2 pooled decile {k_ + 1} mean label", float(cv.mean_label.iloc[k_]), v_, 2)
    hg = pd.read_csv(OUT / "hypotheses_followup_hinge.csv").set_index("quantile")
    chk("follow-up: rho within the lowest 10% of flank GC", float(hg.loc[0.10, "rho_below"]), 0.485, 3); chk("follow-up: rho within the other 90%", float(hg.loc[0.10, "rho_above"]), 0.021, 3)
    chk("follow-up: rho within the lowest 15%", float(hg.loc[0.15, "rho_below"]), 0.488, 3); chk("follow-up: rho within the other 85%", float(hg.loc[0.15, "rho_above"]), 0.003, 3)
    chk("follow-up: rho over the whole range", float(hg.loc[0.10, "rho_full"]), 0.158, 3); chk("follow-up: guides in the lowest decile", float(hg.loc[0.10, "n_below"]), 3086, 0)
    cc = pd.read_csv(OUT / "hypotheses_followup_concentration.csv").iloc[0]
    chk("follow-up: lowest-decile deficit", float(cc.deficit), -3.10, 2); chk("follow-up: deficit lower bound", float(cc.lo), -3.37, 2); chk("follow-up: deficit upper bound", float(cc.hi), -2.78, 2)
    chk("follow-up: genes in the lowest decile", float(cc.genes_in_lowest_decile), 857, 0); chk("follow-up: share held by the ten largest genes", float(cc.top10_genes_share), 0.044, 3)
    chk("follow-up: deficit without those ten genes", float(cc.deficit_without_top10_genes), -2.97, 2)
    rp = pd.read_csv(OUT / "hypotheses_followup_replication.csv")
    for k_, v_, lo_, hi_ in ((0, -3.29, -3.76, -2.84), (1, -2.15, -2.43, -1.86), (2, -0.57, -0.75, 0.14)):
        chk(f"follow-up: replication row {k_} deficit", float(rp.deficit.iloc[k_]), v_, 2); chk(f"follow-up: replication row {k_} lower", float(rp.lo.iloc[k_]), lo_, 2)
        chk(f"follow-up: replication row {k_} upper", float(rp.hi.iloc[k_]), hi_, 2)
    ab = pd.read_csv(OUT / "hypotheses_followup_absolute.csv").iloc[0]
    chk("follow-up: C. rodentium guides below the absolute threshold", float(ab.n_below), 653, 0); chk("follow-up: their deficit", float(ab.deficit), -1.97, 2)
    chk("follow-up: their deficit, upper bound", float(ab.hi), 1.10, 2)
    ex = pd.read_csv(OUT / "hypotheses_followup_expression.csv").set_index("stratum")
    for s_, v_ in (("low expression", -3.79), ("middle", -2.80), ("high expression", -1.77)):
        chk(f"follow-up: deficit in the {s_} stratum", float(ex.loc[s_, "deficit"]), v_, 2)
    ep = pd.read_csv(OUT / "hypotheses_followup_expression_partial.csv").iloc[0]
    chk("follow-up: flank GC against expression", float(ep.rho_gc_expression), -0.003, 3); chk("follow-up: flank GC against label, marginal", float(ep.gc_label_marginal), 0.153, 3)
    chk("follow-up: flank GC against label, given expression", float(ep.gc_label_given_expression), 0.154, 3)
    wi = pd.read_csv(OUT / "hypotheses_followup_within.csv").set_index("excluded_lowest_share")
    chk("follow-up: within-gene rho above the lowest 10%", float(wi.loc[0.10, "rho_within_genes"]), 0.066, 3); chk("follow-up: within-gene rho above the lowest 20%", float(wi.loc[0.20, "rho_within_genes"]), 0.056, 3)
    h3 = pd.read_csv(OUT / "hypotheses_h3_pam_density.csv").iloc[0]
    chk("H3 PAM density given GC", float(h3.pam_given_gc), 0.022, 3); chk("H3 percentile among GC-matched sums", float(h3.percentile_of_pam), 52, 0)
    chk("H3 null mean", float(h3.null_mean), 0.005, 3); chk("H3 null sd", float(h3.null_sd), 0.040, 3); chk("H3 null 97.5th percentile", float(h3.null_p97_5), 0.060, 3)
    chk("H3 GC given PAM density", float(h3.gc_given_pam), 0.092, 3); chk("H3 GC marginal", float(h3.marginal_gc), 0.158, 3); chk("H3 correlation of the two densities", float(h3.corr_pam_gc), 0.733, 3)

    print("the sealed organism (sealed_organism.py)")
    so = pd.read_csv(OUT / "sealed_organism.csv").set_index("arm")
    chk("sealed organism: champion rho, mean of seeds", float(so.loc["champion", "rho_seed_mean"]), 0.6986, 4)
    chk("sealed organism: champion rho, sd of seeds", float(so.loc["champion", "rho_seed_sd"]), 0.0038, 4)
    chk("sealed organism: tuned rho, mean of seeds", float(so.loc["tuned XGBoost (d099x)", "rho_seed_mean"]), 0.7160, 4)
    chk("sealed organism: tuned minus champion", float(so.loc["tuned XGBoost (d099x)", "minus_champion"]), 0.0172, 4)
    chk("sealed organism: tuned minus champion, lower", float(so.loc["tuned XGBoost (d099x)", "lo"]), 0.0162, 4)
    chk("sealed organism: tuned minus champion, upper", float(so.loc["tuned XGBoost (d099x)", "hi"]), 0.0206, 4)
    chk("sealed organism: average of five, mean of seeds", float(so.loc["XGBoost average of 5", "rho_seed_mean"]), 0.7170, 4)
    chk("sealed organism: average of five minus champion", float(so.loc["XGBoost average of 5", "minus_champion"]), 0.0165, 4)

    sc = pd.read_csv(OUT / "hypotheses_audit_scale.csv").set_index("window_nt")
    for w_, dsd_, rb_ in ((50, -0.76, 0.435), (250, -0.95, 0.485), (500, -0.92, 0.416), (1000, -0.78, 0.389)):
        chk(f"audit: lowest-decile gap in SD at {w_} nt", float(sc.loc[w_, "deficit_in_sd"]), dsd_, 2)
        chk(f"audit: rho inside the tail at {w_} nt", float(sc.loc[w_, "rho_below"]), rb_, 3)
    gt = pd.read_csv(OUT / "hypotheses_audit_gate.csv").set_index("base_at_20")
    for b_, lo_sd, share_, rq_ in (("A", 3.12, 0.187, 0.084), ("T", 3.08, 0.151, 0.083), ("C", 3.20, 0.288, 0.060), ("G", 3.15, 0.353, 0.030)):
        chk(f"audit: label SD in the {b_} stratum", float(gt.loc[b_, "label_sd"]), lo_sd, 2)
        chk(f"audit: share in the lowest label quartile, {b_}", float(gt.loc[b_, "share_in_lowest_quartile"]), share_, 3)
        chk(f"audit: flank slope above the lowest quartile, {b_}", float(gt.loc[b_, "rho_above_global_q25"]), rq_, 3)

    print("eukaryotes (eukaryote_rules.py)")
    pr = pd.read_csv(OUT / "eukaryote_map_pairs.csv"); pr = pr[~pr.same_library.astype(bool)].copy()
    pr["kind"] = [" - ".join(sorted([a_, b_])) for a_, b_ in zip(pr.species_a, pr.species_b)]
    for kind_, n_, med_, lo_, hi_, cor_ in (("human - human", 26, 0.456, -0.047, 0.711, 0.59), ("human - mouse", 16, 0.443, 0.101, 0.695, 0.56),
                                             ("human - zebrafish", 8, 0.221, -0.239, 0.381, 0.28), ("mouse - zebrafish", 2, 0.048, -0.123, 0.219, 0.07),
                                             ("E. coli - human", 8, -0.140, -0.276, 0.078, -0.16), ("E. coli - mouse", 2, -0.013, -0.103, 0.078, -0.02)):
        g_ = pr[pr.kind == kind_]
        chk(f"eukaryotes: pairs, {kind_}", float(len(g_)), n_, 0); chk(f"eukaryotes: median map rho, {kind_}", float(g_.rho.median()), med_, 3)
        chk(f"eukaryotes: lowest map rho, {kind_}", float(g_.rho.min()), lo_, 3); chk(f"eukaryotes: highest map rho, {kind_}", float(g_.rho.max()), hi_, 3)
        chk(f"eukaryotes: corrected median, {kind_}", float(g_.rho_corrected.median()), cor_, 2)
    g_ = pr[pr.kind == "mouse - mouse"]; chk("eukaryotes: mouse - mouse map rho", float(g_.rho.iloc[0]), 0.415, 3)
    g_ = pr[pr.kind == "E. coli - zebrafish"]; chk("eukaryotes: E. coli - zebrafish map rho", float(g_.rho.iloc[0]), -0.209, 3)
    ct = pd.read_csv(OUT / "eukaryote_contrast_g_minus_t.csv").set_index("group")
    for grp_, v_, lo_, hi_ in (("E. coli", -0.333, -0.359, -0.308), ("human (pooled)", 0.277, 0.248, 0.306), ("mouse (pooled)", 0.398, 0.308, 0.482),
                               ("zebrafish (pooled)", 0.430, 0.303, 0.548), ("worm (pooled)", 1.048, 0.562, 1.568), ("fly (pooled)", 0.029, -0.377, 0.544)):
        chk(f"eukaryotes: G minus mean(A,C) at position 20, {grp_}", float(ct.loc[grp_, "g_minus_ac_pos20"]), v_, 3)
        chk(f"eukaryotes: ... lower bound, {grp_}", float(ct.loc[grp_, "g_minus_ac_lo"]), lo_, 3); chk(f"eukaryotes: ... upper bound, {grp_}", float(ct.loc[grp_, "g_minus_ac_hi"]), hi_, 3)
    big_ = ["chari2015Train293T", "doench2014-Hs", "doench2016_hg19", "hart2016-HelaLib1Avg", "hart2016-Hct1161lib1Avg", "wang2015_hg19", "xu2015TrainHl60",
            "xu2015TrainKbm7", "doench2014-Mm", "xu2015TrainMEsc", "morenoMateos2015"]
    chk("eukaryotes: datasets with at least 300 guides whose G-minus-A/C contrast has the eukaryotic sign", float((ct.loc[big_, "g_minus_ac_pos20"] > 0).sum()), 11, 0)
    chk("eukaryotes: smallest of them (Wang 2015)", float(ct.loc["wang2015_hg19", "g_minus_ac_pos20"]), 0.082, 3)
    ori = pd.read_csv(OUT / "eukaryote_orientation.csv").set_index("dataset")
    chk("eukaryotes: datasets of the eleven where all six predictors agree with the orientation", float((ori.loc[big_, "tools_agreeing"] == 6).sum()), 11, 0)
    chk("eukaryotes: weakest orientation anchor (Wang 2015)", float(ori.loc["wang2015_hg19", "median_tool_rho"]), -0.103, 3)
    fs = pd.read_csv(OUT / "eukaryote_flank_step.csv").set_index("dataset")
    for d_, g_, lo_, hi_ in (("E. coli", -0.70, -0.73, -0.66), ("chari2015Train293T", -0.27, -0.43, -0.10), ("doench2016_hg19", -0.45, -0.52, -0.38), ("wang2015_hg19", -0.32, -0.45, -0.22),
                             ("xu2015TrainHl60", -0.16, -0.29, -0.02), ("xu2015TrainMEsc", -0.34, -0.58, -0.15)):
        chk(f"eukaryotes: lowest-decile gap at 77 nt, {d_}", float(fs.loc[d_, "gap_sd"]), g_, 2); chk(f"eukaryotes: ... upper bound, {d_}", float(fs.loc[d_, "hi"]), hi_, 2)
    chk("eukaryotes: eukaryotic datasets whose lowest-decile interval excludes zero", float(((fs.drop("E. coli").hi < 0) | (fs.drop("E. coli").lo > 0)).sum()), 5, 0)
    chk("eukaryotes: largest rho inside the tail among them", float(fs.drop("E. coli").rho_below.max()), 0.200, 3); chk("eukaryotes: E. coli rho inside the tail at 77 nt", float(fs.loc["E. coli", "rho_below"]), 0.326, 3)
    tr_ = pd.read_csv(OUT / "eukaryote_transfer.csv")
    within_ = tr_[tr_.within.astype(bool)].set_index(["model", "scored_on"]).rho
    tr_ = tr_[~tr_.within.astype(bool)].copy()
    tr_["ret"] = [r_.rho / within_[(r_.model, r_.scored_on)] for r_ in tr_.itertuples()]
    def _cat(r_):
        a_, b_ = r_.species_trained, r_.species_scored
        if a_ == "E. coli": return "E. coli -> eukaryote"
        if a_ == "human" and b_ == "human": return "human -> human" if not r_.same_library else "same library"
        if (a_, b_) in (("human", "mouse"), ("mouse", "human")): return f"{a_} -> {b_}"
        if "zebrafish" in (a_, b_): return "zebrafish <-> mammal"
        return "other"
    tr_["cat"] = [_cat(r_) for r_ in tr_.itertuples()]
    pm_, rm_ = "pipeline (SLICER base, champion settings)", "ridge on one-hot + dinucleotides"
    for cat_, n_, mp_, mr_, rp_, rr_ in (("human -> human", 52, 0.263, 0.278, 0.70, 0.72), ("human -> mouse", 16, 0.342, 0.283, 0.73, 0.65), ("mouse -> human", 16, 0.261, 0.269, 0.70, 0.66),
                                          ("zebrafish <-> mammal", 20, 0.115, 0.099, 0.34, 0.30), ("E. coli -> eukaryote", 11, -0.041, -0.056, -0.13, -0.14)):
        a_, b_ = tr_[(tr_.cat == cat_) & (tr_.model == pm_)], tr_[(tr_.cat == cat_) & (tr_.model == rm_)]
        chk(f"eukaryotes transfer: pairs, {cat_}", float(len(a_)), n_, 0)
        chk(f"eukaryotes transfer: median rho, {cat_}, pipeline", float(a_.rho.median()), mp_, 3); chk(f"eukaryotes transfer: median rho, {cat_}, ridge", float(b_.rho.median()), mr_, 3)
        chk(f"eukaryotes transfer: share retained, {cat_}, pipeline", float(a_.ret.median()), rp_, 2); chk(f"eukaryotes transfer: share retained, {cat_}, ridge", float(b_.ret.median()), rr_, 2)
    e_ = tr_[(tr_.cat == "E. coli -> eukaryote") & (tr_.model == pm_)]
    chk("eukaryotes transfer: E. coli-trained, lowest rho (zebrafish)", float(e_.rho.min()), -0.238, 3); chk("eukaryotes transfer: E. coli-trained, highest rho", float(e_.rho.max()), 0.108, 3)

    print("results/: every tracked file is in the generated index, and no two are the same file")
    import hashlib, subprocess
    _tracked = [f for f in subprocess.check_output(["git", "ls-files", "results"], text=True, cwd=_sg.parent).split("\n")
                if f and f != "results/INDEX.md" and not pathlib.Path(f).name.startswith(code_map.TEAMMATE_RESULTS)]
    _idx = (_sg.parent / "results" / "INDEX.md").read_text()
    chk("results/INDEX.md: tracked files the index does not list", float(sum(1 for f in _tracked if f"| `{f.removeprefix('results/')}` |" not in _idx)), 0, 0)
    _sha = [hashlib.sha1((_sg.parent / f).read_bytes()).hexdigest() for f in _tracked]
    chk("results/: files whose bytes duplicate another tracked file", float(len(_sha) - len(set(_sha))), 0, 0)

    print("the overview for judges: every number in it appears in the project's own documents")
    _ov = (_sg.parent / "CODE_OVERVIEW.md").read_text()
    _src = "\n".join((_sg.parent / f).read_text() for f in ("PROGRESS_REPORT.md", "results/RESULTS.md", "EXTENSION_LOG.md", "README.md"))
    _norm = lambda t: t.replace("\u2212", "-")
    _toks = sorted({m.lstrip("+-") for m in re.findall(r"[+\-]?\d+\.\d{2,}|\d{1,3}(?:,\d{3})+", _norm(_ov))})
    _missing = [t for t in _toks if not re.search(r"(?<![\d.,])" + re.escape(t) + r"(?![\d])", _norm(_src))]
    chk("CODE_OVERVIEW.md: numbers quoted that appear in no project document", float(len(_missing)), 0, 0)
    for t in _missing[:10]:
        print(f"      not found anywhere: {t}")

    print("documents: no stale claim, and one check count")
    _docs = {f.name: f.read_text() for f in (RUNJIA / "PROGRESS_REPORT.md", RUNJIA / "EXTENSION_LOG.md",
                                             RUNJIA / "PAPER_FLOW.md", RUNJIA / "results" / "README.md")}
    # claims that were true when written and were made false by later work; each was found stale on 20 November
    for phrase, where in (("We have not tried a mixed-effect", "EXTENSION_LOG.md"),
                          ("mixed-effect random forest or boosted equivalent on the curated cohort. Untried", "PROGRESS_REPORT.md"),
                          ("We have not tried it, and on a screen whose label is survival", "PROGRESS_REPORT.md"),
                          ("Not done:* corrected ceilings for the specific datasets", "PROGRESS_REPORT.md")):
        chk(f"stale claim absent: {phrase[:44]}", 0.0 if phrase in _docs[where] else 1.0, 1, 0)
    _quoted = [("PROGRESS_REPORT.md", r"\((\d+) checks as of"), ("results/README.md", r"\((\d+) checks as of"),
               ("EXTENSION_LOG.md", r"\*\*(\d+) checks, 0 mismatches\*\*"), ("PAPER_FLOW.md", r"`run_audit\.sh`, (\d+) checks")]
    _total = n + len(_quoted)      # this loop's own checks included, so the quoted number is the one finally printed
    for fname, pat in _quoted:
        m_ = re.search(pat, _docs[pathlib.Path(fname).name])
        chk(f"{fname} quotes the audit's own check count", float(m_.group(1)) if m_ else -1.0, _total, 0)

    print(f"\n{n} checks, {len(bad)} mismatches")
    for b in bad:
        print("  MISMATCH", b)
    raise SystemExit(1 if bad else 0)       # so `./run_audit.sh` can be used as a check in a pipeline


if __name__ == "__main__":
    main()
