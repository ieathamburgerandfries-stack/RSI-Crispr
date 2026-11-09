"""Audit: recompute every headline number of Findings 8-9 and the Finding 3 addendum from the result
files and compare with the value typed into the report. A mismatch is a transcription error or a stale
result. Tolerance is half a unit in the last quoted digit.

    python -m sgrna.ext.verify_numbers
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import OUT

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


sh = pd.read_csv(OUT / "shape_ablation.csv")
F = lambda a: folds(sh, a)
d = lambda a, ref="base": float((F(a) - F(ref)).mean())
print("Finding 8, alone over base")
for src, vals in (("md", (0.099, 0.052, 0.031, 0.138)), ("xrc", (0.102, 0.056, 0.078, 0.140)), ("mc", (0.103, 0.053, 0.080, 0.141))):
    for blk, v in zip(("mean", "vec", "geom", "all"), vals):
        chk(f"{src}_{blk} - base", d(f"{src}_{blk}"), v, 3)
chk("allsrc_all - base", d("allsrc_all"), 0.145, 3)
chk("A - base", d("A"), 0.156, 3)
chk("comp_mean - base", d("comp_mean"), 0.105, 3)
chk("A_only_comp - base", d("A_only_comp"), 0.107, 3)
print("Finding 8, on top of A")
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
print("Finding 8, standalone mechanism")
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

print("Nuclease")
ind = pd.read_csv(OUT / "nuclease_indist.csv")
pa = ind.groupby("arm")[["rho", "rho_ngg"]].mean()
chk("per-nuclease all rho", pa.loc["per-nuclease [all]", "rho"], 0.881, 3)
chk("per-nuclease all NGG", pa.loc["per-nuclease [all]", "rho_ngg"], 0.793, 3)
chk("-library_trna NGG", pa.loc["per-nuclease [-library_trna]", "rho_ngg"], 0.759, 3)
chk("spacer only rho", pa.loc["per-nuclease [spacer_onehot only]", "rho"], 0.341, 3)
cap = pd.read_csv(OUT / "nuclease_capacity.csv").groupby("arm")[["rho", "rho_ngg"]].mean()
chk("pooled 3000x8 rho", cap.loc["pooled one-hot n_est=3000 depth=8", "rho"], 0.886, 3)
chk("per-nuclease 3000x8 rho", cap.loc["per-nuclease n_est=3000 depth=8", "rho"], 0.883, 3)
chk("pooled 500x6 rho", cap.loc["pooled one-hot n_est=500 depth=6", "rho"], 0.866, 3)
nl = pd.read_csv(OUT / "nuclease_indist_null.csv").groupby(["arm", "perm"])[["rho", "rho_ngg"]].mean()
chk("permuted all-descriptors rho (mean)", nl.loc["pooled [all descriptors] permuted"]["rho"].mean(), 0.869, 3)
chk("permuted all-descriptors NGG (mean)", nl.loc["pooled [all descriptors] permuted"]["rho_ngg"].mean(), 0.745, 3)
lo = pd.read_csv(OUT / "lono_by_pam_class.csv")
for arm, m, v in (("U", "all", 0.795), ("D_label-free", "all", 0.809), ("D_all", "all", 0.813),
                  ("U", "class_order", 0.855), ("D_label-free", "class_order", 0.908), ("U", "within_class", 0.410),
                  ("D_label-free", "within_class", 0.408), ("own", "within_class", 0.545), ("own", "all", 0.881)):
    chk(f"LONO 500x6 {arm} {m}", lo[lo.arm == arm][m].mean(), v, 3)
bc = pd.read_csv(OUT / "lono_bigcap_summary.csv")
for arm, m, field, v in (("D_label-free", "all", "d_vs_U", 0.023), ("D_label-free", "ngg", "d_vs_U", 0.056),
                         ("D_label-free", "class_order", "d_vs_U", 0.047), ("D_label-free", "within_class", "d_vs_U", 0.035),
                         ("D_label-free", "within_class", "lo", -0.021), ("D_label-free", "within_class", "hi", 0.100),
                         ("D_label-free", "ngg", "lo", 0.004), ("U", "all", "mean", 0.791)):
    chk(f"LONO 1500x8 {arm} {m} {field}", float(bc[(bc.arm == arm) & (bc.metric == m)][field].iloc[0]), v, 3)
fs = pd.read_csv(OUT / "nuclease_fewshot.csv")
for k, (a_, b_, c_) in {100: (0.713, 0.820, 0.828), 500: (0.807, 0.843, 0.851), 2000: (0.843, 0.859, 0.865)}.items():
    g = lambda pat: fs[(fs.k == k) & fs.arm.str.contains(pat)].groupby("nuclease").rho.mean().mean()
    chk(f"few-shot k={k} own-only", g("own model"), a_, 3)
    chk(f"few-shot k={k} one-hot", g("one-hot"), b_, 3)
    chk(f"few-shot k={k} descriptors", g("descriptors"), c_, 3)
tn = pd.read_csv(OUT / "nuclease_fewshot_tuned.csv")
for k, v in {100: 0.712, 500: 0.804, 2000: 0.847}.items():
    chk(f"tuned k-only k={k}", tn[tn.k == k].groupby("nuclease").rho.mean().mean(), v, 3)

print("Section 7: 770 variants, leave one campaign out")
ps = pd.read_csv(OUT / "pam_variants" / "summary_source.csv")
pg = ps[(ps.subset == "all") & (ps.metric == "rho_pam")].set_index(["model", "arm"])
for arm, vals in (("id", (0.306, 0.277, 0.277)), ("comp", (0.313, 0.298, 0.312)), ("id+comp", (0.318, 0.292, 0.292)),
                  ("id+comp+reach", (0.316, 0.303, 0.290)), ("reach", (0.213, 0.288, 0.243)), ("env", (0.0, 0.0, 0.0)),
                  ("id_perm", (0.001, 0.010, 0.002))):
    for model, v in zip(("ridge", "et", "mlp"), vals):
        chk(f"rho_pam {arm} {model}", float(pg.loc[(model, arm), "value"]), v, 3)
for model, d_, lo_, hi_ in (("ridge", 0.010, 0.006, 0.014), ("et", 0.026, 0.007, 0.040), ("mlp", 0.013, 0.005, 0.022)):
    r = pg.loc[(model, "id+comp+reach")]
    chk(f"id+comp+reach - id {model}", float(r.d_vs_ref), d_, 3); chk(f"  lo", float(r.d_lo), lo_, 3); chk(f"  hi", float(r.d_hi), hi_, 3)
for model, d_, lo_, hi_ in (("ridge", 0.007, -0.008, 0.021), ("et", 0.021, -0.001, 0.036), ("mlp", 0.036, 0.023, 0.050)):
    r = pg.loc[(model, "comp")]
    chk(f"comp - id {model}", float(r.d_vs_ref), d_, 3); chk(f"  lo", float(r.d_lo), lo_, 3); chk(f"  hi", float(r.d_hi), hi_, 3)
cg = pd.read_csv(OUT / "pam_variants" / "campaign_source.csv")
for model, k in (("et", 7), ("mlp", 5), ("ridge", 7)):
    t = cg[(cg.model == model) & cg.arm.isin(["id", "id+comp+reach"])].pivot_table(index="campaign", columns="arm", values="rho_pam")
    chk(f"campaigns ahead, id+comp+reach vs id, {model}", float((t["id+comp+reach"] > t["id"]).sum()), k, 0)
for model, v in (("et", 0.284), ("mlp", 0.193), ("ridge", 0.154)):
    chk(f"only-volume {model}", float(pg.loc[(model, "comp_only_vol"), "value"]), v, 3)
for model, v in (("et", 0.256), ("mlp", 0.272), ("ridge", 0.250)):
    chk(f"only-charge {model}", float(pg.loc[(model, "comp_only_chg"), "value"]), v, 3)
pc = pd.read_csv(OUT / "protein_composition_l1.csv")
chk("whole-protein composition L1 min (non-reference)", float(pc[pc.n_mut > 0].l1.min()), 0.0044, 4)
chk("whole-protein composition L1 max", float(pc.l1.max()), 0.0102, 4)

print("Section 8: sequence-predicted PAM descriptor (stage 10)")
sp = pd.read_csv(OUT / "lono_pam_pred_summary.csv").set_index(["metric", "arm"])
for (m, arm), field, v in ((("all", "D_pam_pred"), "d_vs_U", 0.0248), (("all", "D_pam_pred"), "lo", 0.0022),
                           (("all", "D_pam_pred"), "hi", 0.0565), (("all", "D_pam"), "d_vs_U", 0.0110),
                           (("all", "D_label-free"), "d_vs_U", 0.0143), (("all", "D_lf_pam_pred"), "d_vs_U", 0.0160),
                           (("class_order", "D_pam_pred"), "d_vs_U", 0.0444), (("class_order", "D_pam_pred"), "lo", 0.0004),
                           (("class_order", "D_pam_pred"), "hi", 0.0911), (("class_order", "D_pam"), "d_vs_U", 0.0453),
                           (("class_order", "D_label-free"), "d_vs_U", 0.0534),
                           (("class_order", "D_pam_pred - D_pam"), "d_vs_U", -0.0009),
                           (("all", "D_pam_pred - D_pam"), "d_vs_U", 0.0139),
                           (("within_class", "D_pam_pred - D_pam"), "d_vs_U", -0.0249),
                           (("within_class", "D_pam_pred"), "d_vs_U", -0.0005), (("within_class", "D_pam"), "d_vs_U", 0.0244)):
    chk(f"stage10 {m} {arm} {field}", float(sp.loc[(m, arm), field]), v, 4)
lo2 = pd.read_csv(OUT / "lono_by_pam_class.csv")
for arm, m, v in (("D_pam_pred", "all", 0.820), ("D_pam", "all", 0.806), ("D_pam_pred", "class_order", 0.899),
                  ("D_pam", "class_order", 0.900), ("D_pam_pred", "within_class", 0.410), ("D_pam", "within_class", 0.435)):
    chk(f"stage10 mean {arm} {m}", lo2[lo2.arm == arm][m].mean(), v, 3)

print("Section 9: position-resolved contact profiles")
cp = pd.read_csv(OUT / "enzyme_contact_pairing.csv")
sel = cp[(cp.profile == "contact") & cp.normalised]
get = lambda sites, en: sel[(sel.sites == sites) & (sel.enzymes == en)].iloc[0]
chk("all eight, protospacer obs", float(get("protospacer", "all eight").obs), 0.264, 3)
chk("all eight, protospacer p", float(get("protospacer", "all eight").p_ge), 0.056, 3)
chk("all eight, +PAM obs", float(get("protospacer+PAM", "all eight").obs), 0.368, 3)
chk("all eight, +PAM p", float(get("protospacer+PAM", "all eight").p_ge), 0.013, 3)
chk("HF five, protospacer obs", float(get("protospacer", "high-fidelity five").obs), 0.417, 3)
chk("HF five, protospacer null mean", float(get("protospacer", "high-fidelity five").null_mean), 0.404, 3)
chk("HF five, protospacer p", float(get("protospacer", "high-fidelity five").p_ge), 0.34, 2)


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
dec = pd.read_csv(OUT.parent.parent.parent / "data" / "interim" / "ext" / "human" / "decoded.csv", dtype={"sgRNAID": str})
vc = dec.protospacer.value_counts()
chk("human rows", len(dec), 17421, 0); chk("distinct 20-mers", len(vc), 13138, 0)
chk("sequences occurring more than once", int((vc > 1).sum()), 4216, 0); chk("rows with a twin", int(vc[vc > 1].sum()), 8499, 0)
ci = pd.read_csv(OUT.parent.parent.parent / "data" / "interim" / "curated_guide_index.csv")
chk("bacterial rows", len(ci), 33567, 0); chk("bacterial distinct protospacers", ci.protospacer.nunique(), 33567, 0)


print("Kim 2020 published vs ours")
kp = pd.read_csv(OUT / "kim2020_published_vs_ours.csv").set_index("nuclease")
chk("paper mean Spearman", float(kp.paper_spearman_R.mean()), 0.896, 3)
chk("ours mean Spearman", float(kp.ours_spearman.mean()), 0.881, 3)
chk("mean difference", float(kp.diff_spearman.mean()), -0.014, 3)
chk("evoCas9 ours", float(kp.loc["evoCas9", "ours_spearman"]), 0.683, 3)
chk("variants where ours is ahead", float((kp.diff_spearman > 0).sum()), 5, 0)
chk("ahead range min", float(kp[kp.diff_spearman > 0].diff_spearman.min()), 0.014, 3)
chk("ahead range max", float(kp[kp.diff_spearman > 0].diff_spearman.max()), 0.037, 3)


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
from .tune_analyze import table as _tt
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

print(f"\n{n} checks, {len(bad)} mismatches")
for b in bad:
    print("  MISMATCH", b)
