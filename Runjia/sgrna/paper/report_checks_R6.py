"""Checks for headline numbers of PROGRESS_REPORT.md Part 5 (the head-to-head with crisprHAL 2, cost, other comparisons, what other
predictors read) that had no audit line, computed against the corrected wording of the report. Same idiom as `paper_numbers_C.py`:
`chk(label, got, claimed, dec)` passes when |got - claimed| <= 0.5 * 10**(-dec); `got` is always computed from a stored file.

Which cohort and protocol each number belongs to (several values recur for different quantities):

    equivalence   crisprHAL 2's own two splits, paired bootstrap over guides (`final_eval_espcas9.csv`, `panel_wt.csv`); `final_eval_espcas9.csv`
                  carries a +/-0.01 margin that was committed after the result, so the report quotes the intervals, not a verdict.
    our folds     curated 33,567-guide cohort, five random-row folds, seed 41. Per-fold means (`crisprhal_rerun.csv`, the main project's
                  `folds_ladder_base_full+a_flank.csv`, `tuning.csv`, `matched_search_folds.csv`) differ from the pooled out-of-fold
                  figures (`heldout_compare.csv`) in the fourth decimal: +0.0105 per fold against +0.0115 pooled.
    cost          `complexity.json`, `cost_panel.csv` (crisprHAL 2's 10,000 CPU-s is a literal in `sgrna/cost/cost.py`), the inference files.
    Noshay        `ablation_main.csv` (published 13,880-row matrix, 5 seeds x 5 folds, random rows) and `noshay_replication.csv` (seed 41).

    ./run_ext.sh sgrna.paper.report_checks_R6
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import OUT, MAIN_RESULTS
from sgrna import config

P = "report R6: "


def run(chk):
    """Run every check through `chk(label, got, claimed, dec)`."""
    RES = OUT                                    # the fork's results/
    MAIN = MAIN_RESULTS                                 # the main project's results/ (files the fork does not track)
    ext = config.EXTERNAL / "crisprHAL" / "data"

    # ---------------------------------------------------------------- equivalence: intervals, no pre-declared margin
    print("report R6: the frozen model against crisprHAL 2 on its own splits")
    fe = pd.read_csv(RES / "final_eval_espcas9.csv").set_index("arm")
    chk(P + "eSpCas9 frozen minus crisprHAL 2, paired", float(fe.loc["difference", "spearman"]), 0.0021, 4)
    chk(P + "eSpCas9 interval lower", float(fe.loc["difference", "ci_lo"]), -0.0031, 4)
    chk(P + "eSpCas9 interval upper (smallest margin at which equivalence holds)", float(fe.loc["difference", "ci_hi"]), 0.0072, 4)
    chk(P + "final_eval_espcas9.csv applies a +/-0.01 margin (committed after the result: Appendix A)", float(fe.loc["difference", "margin"]), 0.01, 2)
    pw = pd.read_csv(OUT / "panel_wt.csv").set_index("arm")
    vw0 = pd.read_csv(OUT / "vs_crisprhal_wt.csv").set_index("arm").loc["champion"]       # the bootstrap the report quotes
    chk(P + "WT-SpCas9 frozen minus crisprHAL 2, paired (re-derived frozen 0.6985)", float(vw0["diff"]), 0.0039, 4)
    chk(P + "WT-SpCas9 interval lower", float(vw0["lo"]), -0.0047, 4)
    chk(P + "WT-SpCas9 interval upper (outside the +/-0.01 margin)", float(vw0["hi"]), 0.0124, 4)
    chk(P + "WT-SpCas9 interval upper exceeds 0.01", 1.0 if float(vw0["hi"]) > 0.01 else 0.0, 1, 0)
    ts = pd.read_csv(RES / "their_split_slicer.csv")
    chk(P + "WT-SpCas9 frozen, first run (their_split_slicer.csv)", float(ts.spearman.iloc[0]), 0.6997, 4)
    chk(P + "WT-SpCas9 frozen, re-derived (panel_wt.csv)", float(pw.loc["original champion", "spearman"]), 0.6985, 4)

    # ---------------------------------------------------------------- our five folds: per-fold means against pooled
    print("report R6: our five folds, per-fold and pooled")
    lad = pd.read_csv(MAIN / "folds_ladder_base_full+a_flank.csv").query("seed == 41").set_index("fold").spearman.sort_index()
    hr = pd.read_csv(RES / "crisprhal_rerun.csv")
    hal = hr[(hr.arm == "our_folds") & (hr.seed == 41)].set_index("fold").spearman.sort_index()
    chk(P + "frozen leads in five of five folds (per-fold rho, seed 41)", float((lad - hal > 0).sum()), 5, 0)
    chk(P + "per-fold mean gap, frozen minus crisprHAL 2 published configuration", float((lad - hal).mean()), 0.0105, 4)
    chk(P + "per-fold mean, frozen champion", float(lad.mean()), 0.7076, 4)
    chk(P + "per-fold mean, crisprHAL 2 published configuration", float(hal.mean()), 0.6971, 4)
    hc = pd.read_csv(RES / "heldout_compare.csv").set_index("pairing")
    chk(P + "pooled out-of-fold, frozen minus published configuration", float(hc.loc["untuned", "difference"]), 0.0115, 4)
    chk(P + "pooled, interval lower", float(hc.loc["untuned", "ci_lo"]), 0.0074, 4)
    chk(P + "pooled, interval upper", float(hc.loc["untuned", "ci_hi"]), 0.0157, 4)
    chk(P + "pooled, frozen minus the best configuration found", float(hc.loc["cross", "difference"]), 0.0043, 4)
    chk(P + "pooled crisprHAL 2 published configuration", float(hc.loc["untuned", "crisprhal"]), 0.6968, 4)
    chk(P + "pooled crisprHAL 2 best configuration found", float(hc.loc["tuned", "crisprhal"]), 0.7040, 4)
    ms = pd.read_csv(RES / "matched_search_folds.csv")
    chk(P + "per-fold mean, crisprHAL 2 published configuration (matched_search_folds)", float(ms.hal_default.mean()), 0.6971, 4)
    chk(P + "per-fold mean, crisprHAL 2 best configuration any search found", float(ms.hal_single_split.mean()), 0.7051, 4)
    chk(P + "lift of the best configuration, per-fold mean", float(ms.hal_single_split.mean() - ms.hal_default.mean()), 0.0080, 4)
    chk(P + "lift of the best configuration, pooled", float(hc.loc["tuned", "crisprhal"] - hc.loc["untuned", "crisprhal"]), 0.0072, 4)

    # ---------------------------------------------------------------- the nested search and the 12-draw search
    print("report R6: the nested search of crisprHAL 2 and the small search of SLICER")
    tn = pd.read_csv(RES / "tuning.csv")
    chk(P + "SLICER tuned on our folds, mean of the five per-fold scores (the report's old 0.7218 is not in any file)", float(tn.rho_tuned.mean()), 0.7212, 4)
    chk(P + "per-fold search draws, total", float(tn.trials.sum()), 627, 0)
    chk(P + "per-fold search draws, smallest", float(tn.trials.min()), 97, 0)
    chk(P + "per-fold search draws, largest", float(tn.trials.max()), 157, 0)
    chk(P + "crisprHAL 2 after its nested search (first run), mean of five folds", float(ms.hal_matched.mean()), 0.6987, 4)
    d = (ms.slicer_tuned - ms.hal_matched).to_numpy()
    chk(P + "paired lead, SLICER tuned over crisprHAL 2 nested search", float(d.mean()), 0.0225, 4)
    chk(P + "ahead in five of five folds", float((d > 0).sum()), 5, 0)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    chk(P + "paired t on five folds (the log's 19.69), p < 0.001 needs t > 8.61", 1.0 if t > 8.61 else 0.0, 1, 0)
    mcf = pd.read_csv(RES / "crisprhal_matched_folds.csv")
    fin = mcf[mcf.stage == "final"]
    chk(P + "published configuration won the inner search in four of five folds", float(fin.is_published_config.astype(bool).sum()), 4, 0)
    t12 = pd.read_csv(MAIN / "tuning_trials12.csv")
    chk(P + "12-draw search: draws per fold", float(t12.trials.iloc[0]), 12, 0)
    chk(P + "12-draw search: mean gain over the default (tuning_trials12.csv, main project)", float(t12.delta_rho.mean()), 0.0103, 4)

    # ---------------------------------------------------------------- data efficiency
    print("report R6: data efficiency")
    de = pd.read_csv(OUT / "data_efficiency.csv")
    sc = pd.read_csv(RES / "sample_complexity.csv")
    h = {(int(r.n_train), int(r.fold)): float(r.spearman) for r in sc[(sc.model == "crisprHAL 2") & (sc.seed == 41)].itertuples()}
    tu = {(int(r.n_train), int(r.fold)): float(r.rho) for r in de[de.arm == "tuned d099x"].itertuples()}
    chk(P + "tuned model ahead on every fold at every size: cells compared", float(len([k for k in tu if k in h])), 25, 0)
    chk(P + "... cells ahead", float(sum(tu[k] > h[k] for k in tu if k in h)), 25, 0)
    t8 = float(de[(de.arm == "tuned d099x") & (de.n_train == 8000)].rho.mean())
    h26 = float(sc[(sc.model == "crisprHAL 2") & (sc.seed == 41) & (sc.n_train == 26853)].spearman.mean())
    chk(P + "tuned model at 8,000 guides is within 0.004 of crisprHAL 2 at 26,853 (gap 0.0035)", 1.0 if 0 < h26 - t8 <= 0.004 else 0.0, 1, 0)

    # ---------------------------------------------------------------- top-end objectives (CONFIRM genes, seeds 41-43)
    print("report R6: top-end objectives")
    te = pd.read_csv(OUT / "topend_confirm.csv").groupby("arm").mean(numeric_only=True)
    chk(P + "squared error, top-decile precision", float(te.loc["mse", "top10pct_prec"]), 0.2912, 4)
    chk(P + "weight 3, top-decile precision", float(te.loc["weighted-3", "top10pct_prec"]), 0.2852, 4)
    chk(P + "weight 9, top-decile precision", float(te.loc["weighted-9", "top10pct_prec"]), 0.2890, 4)
    chk(P + "weight 9, pick-10 percentile", float(te.loc["weighted-9", "pick10_pctile"]), 87.2, 1)
    chk(P + "squared error, pick-10 percentile", float(te.loc["mse", "pick10_pctile"]), 85.1, 1)
    chk(P + "weight 9, top-1% precision", float(te.loc["weighted-9", "top1pct_prec"]), 0.0487, 4)
    chk(P + "squared error, top-1% precision", float(te.loc["mse", "top1pct_prec"]), 0.0412, 4)
    chk(P + "pairwise rho", float(te.loc["pairwise", "spearman"]), 0.5758, 4)
    chk(P + "NDCG rho", float(te.loc["ndcg", "spearman"]), 0.5119, 4)

    # ---------------------------------------------------------------- the eSpCas9 and WT-SpCas9 files
    print("report R6: split sizes and the guides the two screens share")
    rd = lambda f: pd.read_csv(ext / f, header=None)[0]
    chk(P + "released WT-SpCas9 training file, rows", float(len(rd("WT-SpCas9_training_data.csv"))), 26994, 0)
    chk(P + "released WT-SpCas9 testing file, rows", float(len(rd("WT-SpCas9_testing_data.csv"))), 6749, 0)
    vw = pd.read_csv(OUT / "vs_crisprhal_wt.csv")
    chk(P + "guides scored here, WT-SpCas9 training", float(vw.n_train.iloc[0]), 26830, 0)
    chk(P + "guides scored here, WT-SpCas9 test", float(vw.n_test.iloc[0]), 6737, 0)
    meta = json.loads((config.INTERIM / "transfer_WT-SpCas9" / "meta.json").read_text())
    wt = set(meta["protospacers"])
    esp = rd("eSpCas9_testing_data.csv")
    ps = esp.str[193:213]
    chk(P + "eSpCas9 test guides that are also WT-SpCas9 cohort guides", float(ps.isin(wt).sum()), 6778, 0)
    chk(P + "... as a share of the 11,898 test guides, percent", 100 * float(ps.isin(wt).mean()), 57, 0)

    # ---------------------------------------------------------------- cost
    print("report R6: cost")
    cx = json.loads((RES / "complexity.json").read_text())
    chk(P + "internal (split) nodes", float(cx["n_internal_nodes"]), 5948, 0)
    chk(P + "leaf values + 2 x split nodes = 18,244", float(cx["n_leaves"] + 2 * cx["n_internal_nodes"]), 18244, 0)
    chk(P + "parameters / leaf values", cx["crisprhal2"]["WT-SpCas9"] / cx["n_leaves"], 125, 0)
    chk(P + "parameters / (leaves + structure)", cx["crisprhal2"]["WT-SpCas9"] / cx["parameters_incl_structure"], 43, 0)
    txt = (Path(__file__).resolve().parents[1] / "cost" / "cost.py").read_text()
    chk(P + "crisprHAL 2's 10,000 CPU-s and 1,266 s are literals in cost.py (hand-entered)",
        1.0 if re.search(r"10000\.0,\s*1266\.0", txt) else 0.0, 1, 0)
    cp = pd.read_csv(RES / "cost_panel.csv").set_index("model")
    chk(P + "crisprHAL 2 CPU-s per wall-second implied by the two literals", 10000.0 / 1266.0, 7.9, 1)
    chk(P + "exact CPU-seconds ratio", float(cp.loc["SLICER, as published", "ratio_cpu_seconds"]), 152, 0)
    cm = pd.read_csv(RES / "curated_models.csv").set_index("model")
    w_slicer = float(cm.loc["xgboost", "seconds"])                       # five folds, champion
    w_pub = float(hr[(hr.arm == "our_folds") & (hr.seed == 41)].fit_seconds.sum())
    w_best = float(pd.read_csv(RES / "crisprhal_tuned_folds.csv").fit_seconds.sum())
    chk(P + "wall-clock ratio against the published configuration (5,100 s / 116.4 s; Appendix B's 45x uses a rounded 1.9 min)", w_pub / w_slicer, 44, 0)
    chk(P + "wall-clock ratio against the best configuration found (13,200 s / 116.4 s)", w_best / w_slicer, 113, 0)
    ol = pd.read_csv(RES / "inference_onecore_local.csv")
    g = lambda m, n: float(ol[(ol.model == m) & (ol.n_guides == n)].latency_ms.iloc[0])
    sl, hl = "SLICER configuration (synthetic data)", "crisprHAL 2 WT-SpCas9 (shipped)"
    chk(P + "laptop forward pass, 1 guide, crisprHAL 2 / frozen", g(hl, 1) / g(sl, 1), 329, 0)
    chk(P + "laptop forward pass, 10,000 guides, crisprHAL 2 / frozen", g(hl, 10000) / g(sl, 10000), 817, 0)
    rows = []                                    # the device names hold unquoted commas, so split from the right
    for ln in (RES / "inference_onecore.csv").read_text().splitlines()[1:]:
        f = ln.split(",")
        if f[-4].strip().isdigit():
            rows.append(dict(model=f[0], device=",".join(f[1:-4]), n_guides=int(f[-4]), latency_ms=float(f[-3])))
    oc = pd.DataFrame(rows)
    r = lambda dev, m, n: float(oc[(oc.device == dev) & (oc.model == m) & (oc.n_guides == n)].latency_ms.iloc[0])
    cpu = "Colab CPU one host thread"
    chk(P + "Colab CPU one thread, 1 guide ratio", r(cpu, hl, 1) / r(cpu, sl, 1), 175, 0)
    chk(P + "Colab CPU one thread, 10,000 guides ratio", r(cpu, hl, 10000) / r(cpu, sl, 10000), 344, 0)
    th, tc = "Colab T4, input on host, one host thread", "Colab T4, input on card, one host thread"
    chk(P + "T4, input on host, smallest ratio (100 guides)", r(th, hl, 100) / r(th, sl, 100), 15, 0)
    chk(P + "T4, input on host, largest ratio (10,000 guides)", r(th, hl, 10000) / r(th, sl, 10000), 20, 0)
    chk(P + "T4, input on card, smallest ratio (100 guides)", r(tc, hl, 100) / r(tc, sl, 100), 12, 0)
    chk(P + "T4, input on card, largest ratio (10,000 guides)", r(tc, hl, 10000) / r(tc, sl, 10000), 110, -1)
    ic = pd.read_csv(RES / "inference_cost.csv")
    gs = lambda m: float(ic[(ic.model.str.startswith(m)) & (ic.n_guides == 10000)].guides_per_s.iloc[0])
    chk(P + "ten laptop cores, throughput ratio at 10,000 guides ('about 1,700')", gs("SLICER") / gs("crisprHAL"), 1700, -2)
    icl = pd.read_csv(RES / "inference_colab.csv").dropna(subset=["cold_start_s"])
    chk(P + "cold start, crisprHAL 2 on CPU", float(icl[(icl.model.str.startswith("crisprHAL")) & (icl.device.str.contains("CPU"))].cold_start_s.iloc[0]), 6.0, 1)
    chk(P + "cold start, crisprHAL 2 on the T4", float(icl[(icl.model.str.startswith("crisprHAL")) & (icl.device == "Colab T4")].cold_start_s.iloc[0]), 9.9, 1)
    el = pd.read_csv(OUT / "e2e_slicer_locus.csv")
    eh = pd.read_csv(OUT / "e2e_crisprhal.csv")
    fro = float(el[(el.model == "original champion") & (el.width_nt == 100_000)].ms_per_guide.iloc[0])
    chk(P + "every NGG site in 100 kb: frozen model ms per guide", fro, 0.0223, 4)
    chk(P + "... crisprHAL 2 (10,000 batch) over frozen model", float(eh[eh.batch == 10000].ms_per_guide.iloc[0]) / fro, 54, 0)
    e10 = eh[eh.batch == 10000].iloc[0]
    chk(P + "share of crisprHAL 2's end-to-end time that is its forward pass, percent", 100 * float(e10.model_ms / e10.ms_per_call), 95, 0)
    chk(P + "feature matrix at 1 Mb, GB (113,201 guides x 6,517 float32 columns)", 113201 * 6517 * 4 / 1e9, 2.95, 2)

    # ---------------------------------------------------------------- other comparisons
    print("report R6: Noshay's iRF, parity by training rows, the Cui screen")
    am = pd.read_csv(RES / "ablation_main.csv").set_index("experiment")
    nr = pd.read_csv(RES / "noshay_replication.csv")
    irf = float(nr.spearman.mean())
    chk(P + "iRF reimplementation, mean rho over five folds (seed 41)", irf, 0.4785, 4)
    chk(P + "iRF Pearson", float(nr.pearson.mean()), 0.488, 3)
    chk(P + "iRF R2 (0.2345 is the rounding boundary; the stored mean is 0.23449)", float(nr.r2.mean()), 0.234, 3)
    base, withA = float(am.loc["baseline", "spearman"]), float(am.loc["a_flank", "spearman"])
    chk(P + "champion XGBoost on the 6,232 published columns (ablation_main.csv)", base, 0.5273, 4)
    chk(P + "... with family A (5 seeds x 5 folds)", withA, 0.6068, 4)
    chk(P + "SLICER minus iRF", withA - irf, 0.128, 3)
    chk(P + "model class: champion minus iRF", base - irf, 0.049, 3)
    chk(P + "family A on the published matrix", withA - base, 0.080, 3)
    da = pd.read_csv(RES / "data_ablation.csv").set_index("arm")
    nm = float(da.loc[["curated_matched_n_0", "curated_matched_n_1", "curated_matched_n_2"], "spearman"].mean())
    a0, a1, a2, a3 = (float(da.loc["noshay_rows_label", "spearman"]), float(da.loc["noshay_rows", "spearman"]), nm,
                      float(da.loc["curated_full", "spearman"]))
    chk(P + "Noshay rows, Noshay's cut_score", a0, 0.6537, 4)
    chk(P + "same rows, curated label", a1, 0.6600, 4)
    chk(P + "count-matched random curated rows, mean of three draws", a2, 0.6683, 4)
    chk(P + "all 26,830 curated rows", a3, 0.6985, 4)
    chk(P + "label +0.0064", a1 - a0, 0.0064, 4)
    chk(P + "which rows +0.0082", a2 - a1, 0.0082, 4)
    chk(P + "how many rows +0.0302", a3 - a2, 0.0302, 4)
    chk(P + "total +0.0448", a3 - a0, 0.0448, 4)
    chk(P + "family A on the same partition at full data", a3 - float(da.loc["curated_full_base", "spearman"]), 0.1554, 4)
    cf = pd.read_csv(RES / "cui_final_exam.csv").set_index("arm")
    chk(P + "Cui zero-shot |rho|, crisprHAL 2", abs(float(cf.loc["crisprhal", "spearman"])), 0.0867, 4)
    chk(P + "Cui zero-shot |rho|, SLICER (frozen champion)", abs(float(cf.loc["slicer", "spearman"])), 0.0609, 4)
    chk(P + "Cui paired difference for SLICER", float(cf.loc["difference", "spearman"]), -0.0258, 4)
    chk(P + "Cui interval lower", float(cf.loc["difference", "ci_lo"]), -0.0289, 4)
    chk(P + "Cui interval upper", float(cf.loc["difference", "ci_hi"]), -0.0227, 4)
    cs = pd.read_csv(RES / "cui_screen.csv")
    chk(P + "trained on the Cui screen: base columns, gene-grouped CV", float(cs[(cs.cv == "gene") & (cs.arm == "base")].spearman.iloc[0]), 0.458, 3)

    # ---------------------------------------------------------------- what other predictors read: SLICER's side of the table
    print("report R6: SLICER's side of the predictor table")
    ac = pd.read_csv(RES / "ablation_curated.csv").set_index("family")
    chk(P + "curated base, random-row five-by-five CV", float(ac.loc["baseline", "spearman"]), 0.545, 3)
    chk(P + "curated base + family A (0.708, not 0.709)", float(ac.loc["a_flank", "spearman"]), 0.708, 3)
    chk(P + "flank gain, curated, random-row", float(ac.loc["a_flank", "d_spearman"]), 0.164, 3)
    ag = pd.read_csv(RES / "ablation_curated_gene.csv").set_index("family")
    chk(P + "flank gain, curated, whole genes held out", float(ag.loc["a_flank", "d_spearman"]), 0.157, 3)
    chk(P + "h_offtarget, curated random-row", float(ac.loc["h_offtarget", "d_spearman"]), 0.0011, 4)
    chk(P + "h_offtarget, curated gene-grouped", float(ag.loc["h_offtarget", "d_spearman"]), 0.0003, 4)
    hh = pd.read_csv(RES / "headtohead.csv").query("seed == 41")
    chk(P + "measured expected pick of ten, base only (curated_all), percentile", float(hh[hh.arm == "curated_all"].pick_percentile.mean()), 69.6, 1)
    chk(P + "measured expected pick of ten, with family A (curated_flank), percentile", float(hh[hh.arm == "curated_flank"].pick_percentile.mean()), 75.9, 1)
    fa = pd.read_csv(OUT / "family_a_at_tuned_summary.csv").set_index("comparison")
    chk(P + "frozen champion with family A, gene-grouped, 30,203 genic guides, 3 seeds x 5 folds", float(fa.loc["family A increment, champion", "with_a"]), 0.7151, 4)
    chk(P + "tuned d099x with family A", float(fa.loc["family A increment, d099x", "with_a"]), 0.7314, 4)
    chk(P + "tuning gain on base + A", float(fa.loc["tuning gain (d099x - champion) on base+A", "delta"]), 0.016, 3)
    chk(P + "tuning gain on the base alone", float(fa.loc["tuning gain (d099x - champion) on base", "delta"]), 0.003, 3)
    fc = pd.read_csv(OUT / "tune_fullcv.csv")
    chk(P + "average of five XGBoost models, same CV", float(fc[fc.arm == "ensemble of 5"].rho.mean()), 0.7334, 4)
    pp = pd.read_csv(OUT / "interpret_position_profile.csv")
    gp = lambda m, lo, hi: float(pp[(pp.model == m) & (pp.region == "guide") & (pp.position.between(lo, hi))].gain_share.sum())
    chk(P + "gain share of the eight PAM-proximal positions, frozen (original)", 100 * gp("original", 13, 20), 63, 0)
    chk(P + "... tuned", 100 * gp("tuned", 13, 20), 62, 0)
    chk(P + "gain share of positions 1-12, frozen", 100 * gp("original", 1, 12), 10, 0)
    chk(P + "... tuned", 100 * gp("tuned", 1, 12), 17, 0)
    tw = pd.read_csv(OUT / "human_additivity_twin_summary.csv").query("folds == 'twin-grouped'").set_index("arm")["mean"]
    chk(P + "twin-grouped folds: default (champion) minus the best-looking row-wise configuration", float(tw["champion"] - tw["boosting_defaults"]), 0.086, 3)
    sh = pd.read_csv(RES / "sealed_holdout.csv")
    gap = sh.pivot(index="seal_seed", columns="arm", values="gap")
    ad = gap["frozen"] - gap["base_only"]
    se = float(ad.std(ddof=1) / np.sqrt(len(ad)))
    chk(P + "adaptive component, mean of 15 draws", float(ad.mean()), 0.0087, 4)
    chk(P + "... mean - 1.96 se", float(ad.mean()) - 1.96 * se, 0.0020, 4)
    chk(P + "... mean + 1.96 se", float(ad.mean()) + 1.96 * se, 0.0153, 4)
    g1 = pd.read_csv(RES / "gc_strata_corr.csv").set_index("stratum").loc["all", "rho_target"]
    g2 = pd.read_csv(RES / "gc_strata_corr_curated.csv").set_index("stratum").loc["all", "rho_target"]
    chk(P + "target GC against activity, published 13,880-row matrix", float(g1), -0.201, 3)
    chk(P + "target GC against activity, curated cohort", float(g2), -0.147, 3)
    vr = pd.read_csv(OUT / "vs_crisprhal_wt.csv")
    chk(P + "binned flank columns: 200 minus 32 (inside the fold SD 0.006-0.008), report 'no better than'", 0.6687 - 0.6701, -0.0014, 4)

    # ---------------------------------------------------------------- the field's ceiling: arithmetic
    print("report R6: the ceiling arithmetic")
    sb = lambda r: float(np.sqrt(2 * r / (1 + r)))
    chk(P + "ceiling against a 2-replicate mean, r = 0.71", sb(0.71), 0.911, 3)
    chk(P + "ceiling against a 2-replicate mean, r = 0.77", sb(0.77), 0.933, 3)
    chk(P + "headroom of a model at rho 0.70 below that ceiling, r = 0.71", sb(0.71) - 0.70, 0.21, 2)
    chk(P + "headroom of a model at rho 0.70 below that ceiling, r = 0.77", sb(0.77) - 0.70, 0.23, 2)
    chk(P + "headroom below the highest replicate figure, 0.77 - 0.70", 0.77 - 0.70, 0.07, 2)


# ---------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from .standalone import run_alone
    run_alone(run, "report")
