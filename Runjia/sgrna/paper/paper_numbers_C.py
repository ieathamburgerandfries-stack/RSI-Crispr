"""Audit of paper claims [29]-[31], [46]-[48], [50]-[55] (group C): every number is recomputed from a stored result file.

Same idiom as `verify_numbers.py`: `chk(label, got, claimed, dec)` passes when |got - claimed| <= 0.5 * 10**(-dec). `got` is always
computed from a file, `claimed` is the value exactly as the paper writes it. A mismatch is reported, not adjusted.

Which cohort and protocol each claim sits on (a value can occur for a different quantity, so this is part of the audit):

    [29]  curated 33,567-guide cohort, our five random-row folds, seed 41, LightGBM on 6,169 base columns + the flank columns that
          fit inside crisprHAL 2's 378 nt input (`headtohead.csv`, arm curated_matched) against crisprHAL 2 at its published
          configuration on the same folds (`crisprhal_rerun.csv`). The 378-nt contexts are the released files in external_data.
    [30], [54]  cost: `cost_panel.csv` (CPU-seconds, hand-entered for crisprHAL 2), `complexity.json`, `sample_complexity.csv`
          (fold-mean rho at nested training sizes, fixed test folds), `cost_slicer.csv` / `cost_crisprhal.csv` (forward pass, batch
          100), `e2e_*.csv` (featurisation + model, one host thread).
    [31], [55]  `sealed_holdout.csv`: 20% of genes sealed, 15 draws, XGBoost champion, 80%-matched training size; and
          `sealed_organism.csv`: trained on all 33,567 E. coli guides, scored zero-shot on the 25,210 C. rodentium guides.
    [46]-[48], [50]  crisprHAL 2's own two splits (eSpCas9 11,898 test; WT-SpCas9 6,737 test), our five folds (33,567 pooled
          out-of-fold predictions, paired on guides), and the 1,204 CONFIRM genes (8,943 guides) never used by the search.
    [51]  gene-grouped five folds x three seeds on the 30,203 genic guides (`family_a_at_tuned*.csv`) for +0.1563 / +0.1697; the
          +0.080 and +0.164 are the published 13,880-row matrix (`ablation_main.csv`) and the curated cohort with random-row folds
          (`ablation_curated.csv`); the cohort decomposition is Appendix B2a.
    [52]  the 30,203 genic guides of the curated cohort, flank GC averaged over 250 nt each side (hypotheses_*.csv).
    [53]  transfer with LightGBM, base + family A, no retraining (`transfer*.csv`, three seeds), and the eukaryote transfer table.

    ./run_ext.sh sgrna.paper.paper_numbers_C
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..common import OUT
from sgrna import config

P = "paper C: "


def run(chk):
    """Run every check of group C through `chk(label, got, claimed, dec)`."""
    RES = OUT.parent                                   # the fork's results/ (the same files sit in the main project's results/)
    MAIN = config.RESULTS                              # the main project's results/, for files the fork does not track
    hal_dir = config.EXTERNAL / "crisprHAL" / "data"
    ci = lambda lo, hi: 1.0 if lo < 0 < hi else 0.0    # 1 when an interval contains zero

    # ------------------------------------------------------------------ [29] reach
    print("paper C [29] reach: inside crisprHAL 2's 378-nt span, and beyond it")
    ctx = pd.read_csv(hal_dir / "WT-SpCas9_training_data.csv", header=None, nrows=2000)[0]
    chk(P + "[29] 378 nt: length of every context in crisprHAL 2's released WT-SpCas9 file", float(ctx.str.len().min()), 378, 0)
    chk(P + "[29] 378 nt: ... and the longest", float(ctx.str.len().max()), 378, 0)
    chk(P + "[29] 378 nt: NGG PAM at positions 209-211 of every context, so 189 nt upstream of the 20-mer",
        float((ctx.str[210:212] == "GG").mean()) * 189, 189, 0)
    chk(P + "[29] 378 nt: 166 nt downstream of the PAM (378 - 212)", float(ctx.str.len().min() - 212), 166, 0)
    ww = pd.read_csv(RES / "crisprhal_wide_window.csv")
    chk(P + "[29] 378 nt: input_nt of crisprHAL 2's 'published' arm in the width experiment",
        float(ww[ww.arm == "native_378"].input_nt.iloc[0]), 378, 0)
    beyond = [w for w in config.FLANK_WINDOWS if w > 189]
    chk(P + "[29] 250 nt: smallest family-A window that reaches past 189 nt upstream / 166 downstream", float(min(beyond)), 250, 0)
    chk(P + "[29] 1,000 nt: largest family-A window", float(max(beyond)), 1000, 0)
    h2h = pd.read_csv(RES / "headtohead.csv")
    hr = pd.read_csv(RES / "crisprhal_rerun.csv")
    matched = float(h2h[(h2h.arm == "curated_matched") & (h2h.seed == 41)].spearman.mean())
    hal5 = float(hr[(hr.arm == "our_folds") & (hr.seed == 41)].spearman.mean())
    chk(P + "[29] 0.025: crisprHAL 2 (published config, our five folds) minus family A cut to its 378-nt span", hal5 - matched, 0.025, 3)
    full = float(h2h[(h2h.arm == "curated_flank") & (h2h.seed == 41)].spearman.mean())
    chk(P + "[29] +0.035: full family A minus the 378-nt-span family A (same model, seed 41, five folds)", full - matched, 0.035, 3)
    # the report's table prints 0.7072 for full family A: that is the mean of all eight stored curated_flank rows (5 folds of seed 41
    # and 3 folds of seed 42), not the seed-41 five-fold figure
    chk(P + "[29] (report's 0.7072 for full family A is the mean over 8 stored rows: seed 41 x5 and seed 42 x3)",
        float(h2h[h2h.arm == "curated_flank"].spearman.mean()), 0.7072, 4)
    w = ww[(ww.model == "crisprHAL 2") & (ww.batch == 256)].set_index("input_nt").spearman
    chk(P + "[29] 'a wider input declines': crisprHAL 2 at 378 nt > 1,023 nt > 2,023 nt (batch 256, fold 1, one seed)",
        1.0 if w.loc[378] > w.loc[1023] > w.loc[2023] else 0.0, 1, 0)

    # ------------------------------------------------------------------ [30] cost
    print("paper C [30] cost")
    cx = json.loads((RES / "complexity.json").read_text())
    hal_params, leaves = cx["crisprhal2"]["WT-SpCas9"], cx["n_leaves"]
    chk(P + "[30] 790,533: crisprHAL 2 WT-SpCas9 trainable parameters (complexity.json)", float(hal_params), 790533, 0)
    chk(P + "[30] 125x: parameters / leaf values", hal_params / leaves, 125, 0)
    cp = pd.read_csv(RES / "cost_panel.csv").set_index("model")
    cpu = float(cp.loc["crisprHAL 2, published", "cpu_seconds"]) / float(cp.loc["SLICER, as published", "cpu_seconds"])
    chk(P + "[30] about 150x: fit CPU-seconds, crisprHAL 2 published / SLICER as published (exact 152, rounded to tens)", round(cpu, -1), 150, 0)
    sc = pd.read_csv(RES / "sample_complexity.csv")
    piv = sc.pivot_table(index="n_train", columns="model", values="spearman")
    hal_full = float(sc[(sc.model == "crisprHAL 2") & (sc.seed == 41) & (sc.n_train == 26853)].spearman.mean())
    chk(P + "[30] 0.6950: crisprHAL 2 at 26,853 training guides, fold-mean rho, seed 41", hal_full, 0.6950, 4)
    chk(P + "[30] 26,853: largest training size in the curve", float(sc.n_train.max()), 26853, 0)
    chk(P + "[30] 26,853: training guides per fold on our folds (33,567 - 6,714)",
        float(hr[(hr.arm == "our_folds") & (hr.fold == 1)].n_train.iloc[0]), 26853, 0)
    de = pd.read_csv(OUT / "data_efficiency.csv").pivot_table(index="n_train", columns="arm", values="rho")
    t8, t13 = float(de.loc[8000, "tuned d099x"]), float(de.loc[13880, "tuned d099x"])
    n_match = 8000 + (hal_full - t8) / (t13 - t8) * (13880 - 8000)
    chk(P + "[30] about 9,100: labels at which the tuned model interpolates to crisprHAL 2's 26,853-guide rho (to 100)", round(n_match, -2), 9100, 0)
    e2 = pd.read_csv(OUT / "e2e_slicer.csv")
    eh = pd.read_csv(OUT / "e2e_crisprhal.csv")
    tuned = lambda b: float(e2[(e2.model == "tuned XGBoost (d099x)") & (e2.batch == b)].ms_per_guide.iloc[0])
    hal_e = lambda b: float(eh[eh.batch == b].ms_per_guide.iloc[0])
    ratios = [hal_e(b) / tuned(b) for b in (1, 100, 10000)]
    chk(P + "[30] 13x: end to end per guide, slowest ratio (10,000 guides), tuned model", min(ratios), 13, 0)
    chk(P + "[30] 21x: end to end per guide, fastest ratio (1 guide), tuned model", max(ratios), 21, 0)
    el = pd.read_csv(OUT / "e2e_slicer_locus.csv")
    loc = float(el[(el.model == "tuned XGBoost (d099x)") & (el.width_nt == 100_000)].ms_per_guide.iloc[0])
    chk(P + "[30] 34x: crisprHAL 2 (10,000-guide batch) / tuned model over every NGG site in a 100 kb locus", hal_e(10000) / loc, 34, 0)
    chk(P + "[30] 100 kb: the locus width of that benchmark, in kb", float(el[el.width_nt == 100_000].width_nt.iloc[0]) / 1000, 100, 0)

    # ------------------------------------------------------------------ [55] and [31] sealed genes
    print("paper C [31], [55] sealed genes (sealed_holdout.csv) and the sealed organism")
    sh = pd.read_csv(RES / "sealed_holdout.csv")
    chk(P + "[55] 15 draws: distinct sealing seeds in the file", float(sh.seal_seed.nunique()), 15, 0)
    chk(P + "[55] 20% of genes sealed (share of guides sealed, percent)", 100 * float(sh.n_sealed.sum() / (sh.n_sealed + sh.n_dev).sum()), 20, 0)
    gap = sh.pivot(index="seal_seed", columns="arm", values="gap")
    adapt = gap["frozen"] - gap["base_only"]
    se = float(adapt.std(ddof=1) / np.sqrt(len(adapt)))
    chk(P + "[55] adaptive component, frozen arm (difference of differences, mean of 15 draws)", float(adapt.mean()), 0.0087, 4)
    chk(P + "[55] adaptive component, lower 95% (mean - 1.96 se over draws)", float(adapt.mean()) - 1.96 * se, 0.0020, 4)
    chk(P + "[55] adaptive component, upper 95% (mean + 1.96 se over draws)", float(adapt.mean()) + 1.96 * se, 0.0153, 4)
    cvp = sh.pivot(index="seal_seed", columns="arm", values="cv_in_pool")
    sld = sh.pivot(index="seal_seed", columns="arm", values="sealed")
    gain_cv = float((cvp["frozen"] - cvp["base_only"]).mean())
    gain_sealed = float((sld["frozen"] - sld["base_only"]).mean())
    chk(P + "[55] flank gain in pooled cross-validation inside the development pool", gain_cv, 0.1647, 4)
    chk(P + "[55] flank gain on sealed genes", gain_sealed, 0.1560, 4)
    chk(P + "[31] sealing lowers the flank gain (arm-to-arm difference) by 0.0087 rho", gain_cv - gain_sealed, 0.0087, 4)
    so = pd.read_csv(OUT / "sealed_organism.csv").set_index("arm")
    chk(P + "[31] +0.0172: tuned minus champion, C. rodentium zero-shot, trained on all E. coli guides",
        float(so.loc["tuned XGBoost (d099x)", "minus_champion"]), 0.0172, 4)

    # ------------------------------------------------------------------ [46] default SLICER against crisprHAL 2
    print("paper C [46] default (frozen champion) against crisprHAL 2 as released")
    vs = {c: pd.read_csv(OUT / f"vs_crisprhal_{c}.csv") for c in ("espcas9", "wt", "folds")}
    def row(c, arm, cond=None):
        d = vs[c][vs[c].condition.str.contains(cond)] if cond else vs[c]
        return d[d.arm == arm].iloc[0]
    for c, lab, d_, lo_, hi_ in (("espcas9", "eSpCas9", 0.0021, -0.0031, 0.0072), ("wt", "SpCas9", 0.0039, -0.0047, 0.0124)):
        r = row(c, "champion")
        chk(P + f"[46] default SLICER minus crisprHAL 2 as released, {lab}", float(r["diff"]), d_, 4)
        chk(P + f"[46]   {lab} lower bound", float(r.lo), lo_, 4)
        chk(P + f"[46]   {lab} upper bound", float(r.hi), hi_, 4)
    # 'equivalent': the largest margin the 95% interval needs is the larger of |lo| and |hi| (no margin was pre-declared)
    chk(P + "[46] smallest margin at which eSpCas9 equivalence holds (max |bound|)", float(max(abs(row("espcas9", "champion").lo), abs(row("espcas9", "champion").hi))), 0.0072, 4)
    chk(P + "[46] smallest margin at which SpCas9 equivalence holds (max |bound|)", float(max(abs(row("wt", "champion").lo), abs(row("wt", "champion").hi))), 0.0124, 4)
    chk(P + "[46] +0.0115: default SLICER minus crisprHAL 2 (published config), our five folds, pooled out-of-fold, paired on guides",
        float(row("folds", "champion", "default")["diff"]), 0.0115, 4)
    hc = pd.read_csv(RES / "heldout_compare.csv").set_index("pairing")
    chk(P + "[46] +0.0115: same quantity in heldout_compare.csv (untuned vs crisprHAL 2 published)", float(hc.loc["untuned", "difference"]), 0.0115, 4)
    lad = pd.read_csv(MAIN / "folds_ladder_base_full+a_flank.csv").query("seed == 41").set_index("fold").spearman.sort_index()
    hf = hr[(hr.arm == "our_folds") & (hr.seed == 41)].set_index("fold").spearman.sort_index()
    chk(P + "[46] ahead in five of five folds (per-fold rho, seed 41, frozen champion vs crisprHAL 2 published config)", float(((lad - hf) > 0).sum()), 5, 0)

    # ------------------------------------------------------------------ [47] top-end metrics
    print("paper C [47] top-end metrics, tuned minus crisprHAL 2 (paired bootstrap over guides)")
    pe = pd.read_csv(OUT / "panel_espcas9.csv").set_index("arm").loc["tuned XGBoost"]
    pw = pd.read_csv(OUT / "panel_wt.csv").set_index("arm").loc["tuned XGBoost"]
    chk(P + "[47] top-decile precision difference, eSpCas9", float(pe.d_top_decile_prec), 0.0034, 4)
    chk(P + "[47]   eSpCas9 lower", float(pe.lo_top_decile_prec), -0.0151, 4)
    chk(P + "[47]   eSpCas9 upper", float(pe.hi_top_decile_prec), 0.0286, 4)
    chk(P + "[47] top-decile precision difference, SpCas9", float(pw.d_top_decile_prec), 0.0193, 4)
    chk(P + "[47]   SpCas9 lower", float(pw.lo_top_decile_prec), -0.0119, 4)
    chk(P + "[47]   SpCas9 upper", float(pw.hi_top_decile_prec), 0.0520, 4)
    chk(P + "[47] pick percentile of the ten best-scoring guides, difference, eSpCas9", float(pe.d_pick_percentile), -1.92, 2)
    chk(P + "[47] pick percentile of the ten best-scoring guides, difference, SpCas9", float(pw.d_pick_percentile), -0.44, 2)
    chk(P + "[47] all four intervals contain zero (top-decile precision and pick percentile, both splits)",
        ci(pe.lo_top_decile_prec, pe.hi_top_decile_prec) + ci(pw.lo_top_decile_prec, pw.hi_top_decile_prec)
        + ci(pe.lo_pick_percentile, pe.hi_pick_percentile) + ci(pw.lo_pick_percentile, pw.hi_pick_percentile), 4, 0)
    te = pd.read_csv(OUT / "topend_confirm.csv").groupby("arm").mean(numeric_only=True)
    chk(P + "[47] objectives aimed at the top: squared error has the highest top-decile precision of the six arms (CONFIRM genes)",
        1.0 if te.top10pct_prec.idxmax() == "mse" else 0.0, 1, 0)

    # ------------------------------------------------------------------ [48] blend
    print("paper C [48] averaging with crisprHAL 2")
    bw = pd.read_csv(OUT / "blend_with_hal.csv")
    b = bw[(bw.split == "espcas9") & (bw.weight_on_crisprhal == 0.5)].iloc[0]
    chk(P + "[48] equal-weight blend minus the better single model, eSpCas9", float(b.vs_best_single), -0.0008, 4)
    chk(P + "[48]   lower", float(b.lo), -0.0030, 4)
    chk(P + "[48]   upper", float(b.hi), 0.0015, 4)

    # ------------------------------------------------------------------ [50] the lead repeats
    print("paper C [50] the tuned lead over crisprHAL 2")
    r = row("folds", "tuned XGBoost (d099x)", "default")
    chk(P + "[50] our five folds vs crisprHAL 2 published configuration", float(r["diff"]), 0.0288, 4)
    chk(P + "[50]   lower", float(r.lo), 0.0251, 4); chk(P + "[50]   upper", float(r.hi), 0.0326, 4)
    r = row("folds", "tuned XGBoost (d099x)", "tuned")
    chk(P + "[50] our five folds vs the best configuration found for crisprHAL 2", float(r["diff"]), 0.0216, 4)
    chk(P + "[50]   lower", float(r.lo), 0.0180, 4); chk(P + "[50]   upper", float(r.hi), 0.0252, 4)
    cv = pd.read_csv(OUT / "clean_vs_hal.csv")
    r = cv[cv.vs.str.contains("default") & (cv.arm == "tuned XGBoost (d099x)")].iloc[0]
    chk(P + "[50] 1,204 CONFIRM genes (never used by the search) vs crisprHAL 2 published configuration", float(r["diff"]), 0.0324, 4)
    chk(P + "[50]   lower", float(r.lo), 0.0248, 4); chk(P + "[50]   upper", float(r.hi), 0.0403, 4)
    chk(P + "[50]   CONFIRM genes", float(r.n_genes), 1204, 0)
    ss = pd.read_csv(OUT / "seed_stability.csv").query("split == 'espcas9'")
    lead = ss.rho - ss.crisprhal
    chk(P + "[50] five seeds, eSpCas9: smallest lead", float(lead.min()), 0.0280, 4)
    chk(P + "[50] five seeds, eSpCas9: largest lead", float(lead.max()), 0.0286, 4)
    chk(P + "[50] five seeds: number of seeds", float(len(ss)), 5, 0)

    # ------------------------------------------------------------------ [51] flank gain, and why the cohorts differ
    print("paper C [51] flank gain at the default and tuned settings, and the two cohorts")
    fa = pd.read_csv(OUT / "family_a_at_tuned_summary.csv").set_index("comparison")
    for arm, d_, lo_, hi_ in (("champion", 0.1563, 0.1515, 0.1616), ("d099x", 0.1697, 0.1659, 0.1738)):
        r = fa.loc[f"family A increment, {arm}"]
        chk(P + f"[51] family A increment at {arm} settings (gene-grouped, 30,203 genic guides)", float(r.delta), d_, 4)
        chk(P + f"[51]   {arm} lower", float(r.lo), lo_, 4); chk(P + f"[51]   {arm} upper", float(r.hi), hi_, 4)
        chk(P + f"[51]   {arm}: folds with a positive increment", float(r.folds_positive), 15, 0)
        chk(P + f"[51]   {arm}: folds measured", float(r.n_folds), 15, 0)
        if arm == "champion":
            chk(P + "[51]   the 20-mer alone under the same 300-column selection (the paper's 0.559, against 0.547 with no selection)", float(r.base), 0.559, 3)
    fr = pd.read_csv(OUT / "family_a_at_tuned.csv")
    chk(P + "[51] genic guides in that protocol (test rows over the five folds of one seed)",
        float(fr[(fr.matrix == "base+A") & (fr.arm == "champion") & (fr.seed == 41)].n_test.sum()), 30203, 0)
    pub = pd.read_csv(RES / "ablation_main.csv").set_index("experiment")
    chk(P + "[51] +0.080: family A on the published 13,880-row matrix (5 seeds x 5 folds, random rows)", float(pub.loc["a_flank", "d_spearman"]), 0.080, 3)
    cur = pd.read_csv(RES / "ablation_curated.csv").set_index("family")
    chk(P + "[51] +0.164: family A on the curated 33,567-guide cohort (5 seeds x 5 folds, random rows)", float(cur.loc["a_flank", "d_spearman"]), 0.164, 3)
    cvr = pd.read_csv(RES / "curation_variance.csv").set_index("rows")
    pub_ix = set(pd.read_csv(config.GUIDE_INDEX, usecols=["protospacer"]).protospacer.dropna())
    cur_ix = set(pd.read_csv(config.INTERIM / "curated_guide_index.csv", usecols=["protospacer"]).protospacer.dropna())
    chk(P + "[51] 13,880 rows in the published matrix's guide index (data file)", float(len(pub_ix)), 13880, 0)
    chk(P + "[51] 33,567 guides in the curated cohort (data file)", float(len(cur_ix)), 33567, 0)
    chk(P + "[51] 24,950 guides found only in the curated set (protospacers absent from the published matrix; data files)", float(len(cur_ix - pub_ix)), 24950, 0)
    chk(P + "[51] (8,617 guides shared by the two cohorts; data files)", float(len(cur_ix & pub_ix)), 8617, 0)
    chk(P + "[51] 24,950 guides found only in the curated set (curation_variance.csv)", float(cvr.loc["crisprHAL only", "n"]), 24950, 0)
    chk(P + "[51] 24,950 = 33,567 curated - 8,617 shared with the published matrix",
        float(cvr.loc["whole curated cohort", "n"] - cvr.loc["shared with Noshay", "n"]), 24950, 0)
    chk(P + "[51] steeper relationship: regression slope of label on flank GC, added rows over shared rows (report: 2.96x)",
        float(cvr.loc["crisprHAL only", "slope_beta"] / cvr.loc["shared with Noshay", "slope_beta"]), 2.96, 2)
    lp = pd.read_csv(RES / "ablation_label_published.csv").set_index("experiment")
    lc = pd.read_csv(RES / "ablation_label_curated.csv").set_index("experiment")
    chk(P + "[51] the label accounts for +0.004 (same 8,617 rows: curated score minus Noshay's cut_score)",
        float(lc.loc["a_flank", "d_spearman"] - lp.loc["a_flank", "d_spearman"]), 0.004, 3)
    wi = pd.read_csv(RES / "why_inflation.csv")
    sub = wi[wi.family == "a_flank"].d_spearman
    chk(P + "[51] curated family A at 13,880 random rows, mean of three draws (report +0.1672)", float(sub.mean()), 0.1672, 4)
    chk(P + "[51] the row count accounts for -0.003 (33,567 rows minus 13,880 rows)", float(cur.loc["a_flank", "d_spearman"] - sub.mean()), -0.003, 3)

    # ------------------------------------------------------------------ [52] the step at the AT-rich end
    print("paper C [52] the step at the AT-rich end")
    hg = pd.read_csv(OUT / "hypotheses_followup_hinge.csv").set_index("quantile")
    chk(P + "[52] +0.485: rho of label with flank GC inside the lowest decile (250 nt each side, genic guides)", float(hg.loc[0.10, "rho_below"]), 0.485, 3)
    chk(P + "[52] 3,086 guides in the lowest decile", float(hg.loc[0.10, "n_below"]), 3086, 0)
    chk(P + "[52] +0.021 outside the lowest decile", float(hg.loc[0.10, "rho_above"]), 0.021, 3)
    chk(P + "[52] (outside the lowest decile: guides)", float(hg.loc[0.10, "n_above"]), 27117, 0)
    wg = pd.read_csv(OUT / "hypotheses_followup_within.csv").set_index("excluded_lowest_share")
    chk(P + "[52] +0.066: within-gene rho with the lowest tenth set aside", float(wg.loc[0.10, "rho_within_genes"]), 0.066, 3)
    hc2 = pd.read_csv(OUT / "hypotheses_h2_curve.csv")
    wc = hc2[hc2.kind == "within genes"].sort_values("bin").mean_label.to_numpy()
    chk(P + "[52] within genes the dependence is graded: lowest decile mean label (SD-free label units)", float(wc[0]), -0.58, 2)
    chk(P + "[52] within genes the dependence is graded: highest decile mean label", float(wc[-1]), 0.50, 2)
    rp = pd.read_csv(OUT / "hypotheses_followup_replication.csv").set_index("dataset")
    es, cr = rp.loc["E. coli, eSpCas9 (same loci)"], rp.loc["C. rodentium, TevSpCas9 (a different genome)"]
    chk(P + "[52] -0.80 SD: lowest-decile deficit, eSpCas9, a second enzyme at the same loci", float(es.deficit_in_sd), -0.80, 2)
    chk(P + "[52] -0.21 SD: lowest-decile deficit, C. rodentium", float(cr.deficit_in_sd), -0.21, 2)
    chk(P + "[52] C. rodentium interval includes zero", ci(cr.lo, cr.hi), 1, 0)
    chk(P + "[52] eSpCas9 interval excludes zero", ci(es.lo, es.hi), 0, 0)

    # ------------------------------------------------------------------ [53] transfer
    print("paper C [53] transfer with the same columns and no retraining")
    tm = pd.read_csv(RES / "transfer_model.csv")
    tl = pd.read_csv(RES / "transfer.csv")
    xfer = lambda s, t: float(tm[(tm.source == s) & (tm.target == t) & (tm.arm == "base+a_flank")].spearman.mean())
    local = lambda d: float(tl[(tl.dataset == d) & (tl.arm == "base+a_flank") & (tl.cv == "random") & (tl.seed == 41)].spearman.mean())
    chk(P + "[53] SpCas9 -> eSpCas9, transferred (mean of 3 seeds)", xfer("WT-SpCas9", "eSpCas9"), 0.689, 3)
    chk(P + "[53] eSpCas9 trained locally (5-fold, random rows)", local("eSpCas9"), 0.784, 3)
    chk(P + "[53] E. coli -> C. rodentium, transferred", xfer("WT-SpCas9", "TevSpCas9"), 0.698, 3)
    chk(P + "[53] C. rodentium trained locally", local("TevSpCas9"), 0.766, 3)
    chk(P + "[53] back: C. rodentium -> E. coli, transferred", xfer("TevSpCas9", "WT-SpCas9"), 0.628, 3)
    chk(P + "[53] E. coli trained locally (same LightGBM arm, seed 41)", full, 0.708, 3)
    tr = pd.read_csv(OUT / "eukaryote_transfer.csv")
    within = tr[tr.within.astype(bool)].set_index(["model", "scored_on"]).rho
    tr = tr[~tr.within.astype(bool)].copy()
    tr["ret"] = [r_.rho / within[(r_.model, r_.scored_on)] for r_ in tr.itertuples()]
    pm = "pipeline (SLICER base, champion settings)"
    def pairs(a, b):
        return tr[(tr.model == pm) & (tr.species_trained == a) & tr.species_scored.isin(b)]
    hm, mh = pairs("human", ["mouse"]), pairs("mouse", ["human"])
    ec = tr[(tr.model == pm) & (tr.species_trained == "E. coli")]
    chk(P + "[53] human -> mouse, zero-shot: median share of the target's own rho kept, percent", 100 * float(hm.ret.median()), 73, 0)
    chk(P + "[53]   human -> mouse: library pairs", float(len(hm)), 16, 0)
    chk(P + "[53] mouse -> human, zero-shot: median share kept, percent", 100 * float(mh.ret.median()), 70, 0)
    chk(P + "[53]   mouse -> human: library pairs", float(len(mh)), 16, 0)
    chk(P + "[53] E. coli -> eukaryotes: median transferred rho", float(ec.rho.median()), -0.041, 3)
    chk(P + "[53]   E. coli -> eukaryotes: pairs", float(len(ec)), 11, 0)

    # ------------------------------------------------------------------ [54] cost appendix
    print("paper C [54] cost: labels, forward pass, end to end, size")
    def reach(model, thr):
        hit = piv.index[piv[model] >= thr]
        return float(hit.min()) if len(hit) else 0.0
    for thr, s_, h_ in ((0.62, 4000, 8000), (0.66, 8000, 13880), (0.68, 8000, 20000), (0.70, 20000, 0)):
        chk(P + f"[54] labels to reach rho {thr:.2f}, default SLICER", reach("SLICER", thr), s_, 0)
        chk(P + f"[54] labels to reach rho {thr:.2f}, crisprHAL 2" + (" (0 = not reached at 26,853)" if h_ == 0 else ""), reach("crisprHAL 2", thr), h_, 0)
    chk(P + "[54] 13,880 is one of the five training sizes of the curve (and the published matrix's row count)", 1.0 if 13880 in piv.index else 0.0, 1, 0)
    chk(P + "[54] crisprHAL 2 best fold-mean rho at 26,853 guides is below 0.70", 1.0 if hal_full < 0.70 else 0.0, 1, 0)
    cs = pd.read_csv(OUT / "cost_slicer.csv")
    ch = pd.read_csv(OUT / "cost_crisprhal.csv")
    fwd = lambda m: float(cs[(cs.model == m) & (cs.batch == 100)].per_guide_ms.iloc[0])
    chk(P + "[54] forward pass per guide, default model (batch 100)", fwd("original champion"), 0.002, 3)
    chk(P + "[54] forward pass per guide, tuned model (batch 100)", fwd("tuned XGBoost (d099x)"), 0.018, 3)
    chk(P + "[54] forward pass per guide, crisprHAL 2 (batch 100)", float(ch[ch.batch == 100].per_guide_ms.iloc[0]), 1.32, 2)
    for b, t_, h_ in ((1, 0.72, 15.29), (100, 0.089, 1.365), (10000, 0.093, 1.214)):
        chk(P + f"[54] end to end per guide, tuned model, {b:,} guides (ms)", tuned(b), t_, 2 if b == 1 else 3)
        chk(P + f"[54] end to end per guide, crisprHAL 2, {b:,} guides (ms)", hal_e(b), h_, 2 if b == 1 else 3)
    chk(P + "[54] 6,348 leaf values", float(leaves), 6348, 0)
    chk(P + "[54] 790,533 parameters", float(hal_params), 790533, 0)
    chk(P + "[54] 6,348 leaf values over 400 trees (complexity.json)", float(cx["n_trees"]), 400, 0)


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
