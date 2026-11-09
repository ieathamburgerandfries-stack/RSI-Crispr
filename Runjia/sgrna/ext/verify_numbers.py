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

print(f"\n{n} checks, {len(bad)} mismatches")
for b in bad:
    print("  MISMATCH", b)
