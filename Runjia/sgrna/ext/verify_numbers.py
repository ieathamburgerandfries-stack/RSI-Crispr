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
print(f"\n{n} checks, {len(bad)} mismatches")
for b in bad:
    print("  MISMATCH", b)
