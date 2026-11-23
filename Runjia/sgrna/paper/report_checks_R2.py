"""Audit of PROGRESS_REPORT.md lines 121-307 (Finding 2 on the flank effect, Finding 3 on transfer and confound controls, Finding 4 on
the model being out of material): the headline numbers that `verify_numbers` did not recompute, each from the stored file that holds it.

    ./run_ext.sh sgrna.paper.report_checks_R2

Same idiom as `paper_numbers_C`: `run(chk)` takes `chk(label, got, claimed, dec)` (pass when |got - claimed| <= half a unit in the last
quoted digit); `got` is always computed from a file, `claimed` is the value exactly as the corrected report writes it. Labels start
'report R2: ' and say which cohort and protocol the number belongs to, because the same value occurs for different quantities:

    published   Noshay et al.'s 13,880-row matrix (13,879 located), 6,232 columns; family A adds 348 flank columns
    curated     crisprHAL 2's 33,567-guide release, base 6,169 columns; "genic" = the 30,203 guides inside an annotated gene

Files that the fork does not track (`folds_shift_baseline.csv`, `model_inversion.csv`, the champion rows of `kingdom_transfer_boot.csv`) are
read from the main project's results/, as in `paper_numbers_A`.
"""
from __future__ import annotations

import os

import collections
import json
import re

import numpy as np
import pandas as pd

from .. import config
from ..common import OUT

P = "report R2: "


def _main_file(name):
    """A results file of the fork's own results/, else the main project's untracked copy."""
    f = OUT.parent / name
    return f if f.exists() else config.RESULTS / name


def _shift_correlations():
    """Correlation of every family A column with the same column read `shift` bp away (the texture-preserving null's input)."""
    from ..build.build_matrix import load_dataset
    from ..controls.shiftnull import _shifted_block
    from ..io_utils import load_block, load_guide_index

    _, _, _, ids, _ = load_dataset([], verbose=False)
    idx = load_guide_index()
    idx = idx[idx["located"].astype(bool)].drop_duplicates(config.ID_COL).set_index(config.ID_COL).reindex(ids).reset_index()
    genome_len = int(idx["genome_length"].iloc[0]) if "genome_length" in idx else 4_639_675
    block = load_block("a_flank").set_index(config.ID_COL).reindex(ids).reset_index()
    cols = [c for c in block.columns if c != config.ID_COL]

    def cc(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
            return np.nan
        return float(np.corrcoef(a[m], b[m])[0, 1])

    def kind(c):
        m = re.search(r"(up|dn)(\d+)", c)
        return m.group(2) if m else "near"

    out = {}
    for s in (1_000, 10_000):
        shifted = _shifted_block(block, idx, s, genome_len)
        groups = collections.defaultdict(list)
        for c in cols:
            groups[kind(c)].append(cc(block[c], shifted[c]))
        out[s] = {"dn500.gc": cc(block["eng.flank.comp.dn500.gc"], shifted["eng.flank.comp.dn500.gc"]),
                  "all": float(np.nanmedian(sum(groups.values(), []))),
                  **{k: float(np.nanmedian(v)) for k, v in groups.items()}}
    counts = collections.Counter(kind(c) for c in cols)
    return out, counts, len(cols)


def _esp_unseen_guides(seed: int = 41):
    """SpCas9 -> eSpCas9 transfer as `transfer.transfer_model` does it (fit on all 33,567 SpCas9 guides, score all 59,489 eSpCas9
    guides, base and base + family A), split by whether the eSpCas9 guide is also in the SpCas9 screen. About a minute."""
    from scipy.stats import spearmanr

    from ..compare.transfer import build
    from ..harness import evaluate as ev
    from ..harness.run_ablation import _impute, make_selector

    Xs, ys, _ = build("WT-SpCas9", verbose=False)
    Xt, yt, _ = build("eSpCas9", verbose=False)
    ms = json.loads((config.INTERIM / "transfer_WT-SpCas9" / "meta.json").read_text())
    mt = json.loads((config.INTERIM / "transfer_eSpCas9" / "meta.json").read_text())
    wt = set(ms["protospacers"])
    seen = np.array([g in wt for g in mt["protospacers"]])
    out = {"n_unseen": int((~seen).sum())}
    for arm, cols in (("base", np.arange(ms["n_base"])), ("base+a_flank", np.arange(Xs.shape[1]))):
        A, B = _impute(np.asarray(Xs[:, cols], dtype=np.float32), np.asarray(Xt[:, cols], dtype=np.float32))
        sel = make_selector(seed)
        sel.fit(A, ys)
        top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
        m, _ = ev._make("lightgbm", seed)
        m.fit(A[:, top], ys)
        pred = np.asarray(m.predict(B[:, top])).ravel()
        out[arm] = {"all": float(spearmanr(yt, pred).statistic), "seen": float(spearmanr(yt[seen], pred[seen]).statistic),
                    "unseen": float(spearmanr(yt[~seen], pred[~seen]).statistic)}
    return out


HEAVY = bool(os.environ.get("RSI_AUDIT_HEAVY"))


def run(chk):
    """Run every check of this module through `chk(label, got, claimed, dec)`."""
    R = OUT.parent                                      # the fork's results/
    MAIN = config.RESULTS                               # the main project's results/

    # ------------------------------------------------------------------ Finding 2: what family A is made of
    print("report R2: family A's columns and the shift null")
    corr, counts, ncols = _shift_correlations()
    chk(P + "family A columns (a_flank block)", float(ncols), 348, 0)
    chk(P + "near-cut block: one-hot + per-position quantum columns (10 nt 5', the PAM's N, 15 nt 3')", float(counts["near"]), 260, 0)
    chk(P + "columns of the four windows at one width, both sides (comp + win)", float(counts["50"]), 22, 0)
    chk(P + "windows account for 88 columns", float(sum(v for k, v in counts.items() if k != "near")), 88, 0)
    chk(P + "columns that fit inside crisprHAL 2's span: near-cut + 50 nt windows", float(counts["near"] + counts["50"]), 282, 0)
    chk(P + "shift 1 kb, published matrix: eng.flank.comp.dn500.gc correlation with the aligned column", corr[1000]["dn500.gc"], 0.41, 2)
    chk(P + "shift 1 kb: median over near-cut columns", corr[1000]["near"], 0.03, 2)
    chk(P + "shift 1 kb: median over the 50 nt window columns", corr[1000]["50"], 0.11, 2)
    chk(P + "shift 1 kb: median over the 250 nt window columns", corr[1000]["250"], 0.28, 2)
    chk(P + "shift 1 kb: median over the 500 nt window columns", corr[1000]["500"], 0.40, 2)
    chk(P + "shift 1 kb: median over the 1,000 nt window columns", corr[1000]["1000"], 0.58, 2)
    chk(P + "shift 10 kb: largest of the per-window medians", max(corr[10000][k] for k in ("near", "50", "250", "500", "1000")), 0.07, 2)

    sn = pd.read_csv(R / "shift_null.csv").sort_values("shift_bp").set_index("shift_bp")
    base = pd.read_csv(_main_file("folds_shift_baseline.csv")).drop_duplicates(["seed", "fold"]).spearman.mean()
    gain = sn.spearman - float(base)
    chk(P + "shift null, published matrix, 100 kb blocks: rho at shift 0", float(sn.spearman[0]), 0.6060, 4)
    chk(P + "shift null: rho at 1 kb", float(sn.spearman[1000]), 0.5353, 4)
    chk(P + "shift null: rho at 10 kb", float(sn.spearman[10_000]), 0.5160, 4)
    chk(P + "shift null: lowest rho from 100 kb to 2 Mb", float(sn.spearman.loc[100_000:].min()), 0.5194, 4)
    chk(P + "shift null: highest rho from 100 kb to 2 Mb", float(sn.spearman.loc[100_000:].max()), 0.5229, 4)
    chk(P + "shift null: gain at 100 kb-2 Mb, closest to zero", float(gain.loc[100_000:].max()), -0.006, 3)
    chk(P + "shift null: gain at 100 kb-2 Mb, most negative", float(gain.loc[100_000:].min()), -0.010, 3)

    # ------------------------------------------------------------------ Finding 2: target GC
    print("report R2: target GC")
    cur = pd.read_csv(R / "gc_strata_corr_curated.csv")
    cur = cur[cur.stratum != "all"].sort_values("stratum")
    for s, v in zip(cur.stratum, (0.425, 0.278, 0.180, 0.107, 0.068)):
        chk(P + f"flank GC against the label inside curated target-GC stratum {s} (strongest where the target is GC-poor)",
            float(cur.set_index("stratum").loc[s, "rho_flank"]), v, 3)
    pub = pd.read_csv(R / "gc_strata_corr.csv").set_index("stratum")
    chk(P + "target GC against the label, PUBLISHED matrix (gc_strata_corr.csv, n = 13,879)", float(pub.loc["all", "rho_target"]), -0.201, 3)
    ab = pd.read_csv(R / "gc_strata_ablation.csv")
    piv = ab.pivot(index="stratum", columns="arm", values="spearman")
    delta = piv["a_flank"] - piv["baseline"]
    for s, v in zip(range(5), (0.128, 0.075, 0.059, 0.046, 0.058)):
        chk(P + f"family A gain inside published-matrix target-GC stratum {s} (gc_strata_ablation.csv, 100 kb blocks)", float(delta[s]), v, 3)
    chk(P + "the stratified ablation covers 13,879 published guides", float(ab[ab.arm == "baseline"].n.sum()), 13879, 0)
    chk(P + "the GC-poor stratum has the lowest baseline", float(piv["baseline"].idxmin() == 0), 1, 0)
    chk(P + "the gain is not monotone: the top stratum exceeds the one below", float(delta[4] > delta[3]), 1, 0)
    inv = pd.read_csv(MAIN / "model_inversion.csv").set_index("kind")
    chk(P + "model_inversion.csv: GC of the single beam-search optimum (published-matrix champion)", float(inv.loc["model-optimal", "gc"]), 0.35, 2)
    gi = pd.read_csv(config.GUIDE_INDEX)
    gi = gi[gi["located"].astype(bool)] if "located" in gi else gi
    gc = gi.protospacer.astype(str).str.upper().map(lambda s: (s.count("G") + s.count("C")) / len(s))
    chk(P + "mean GC of the published guides (the '51% mean')", float(gc.mean()), 0.51, 2)

    # ------------------------------------------------------------------ Finding 2: the convolutional network and the head-to-head
    print("report R2: the in-house CNN and family A inside crisprHAL 2's span")
    sq = pd.read_csv(R / "seqnet.csv")
    sm = pd.read_csv(R / "seqnet_matched.csv")
    proto = float(sq[sq.model == "seqnet_proto"].spearman.mean())
    byw = sm.groupby("flank").spearman.agg(["mean", "std"])
    chk(P + "CNN, published matrix, protospacer + PAM only (seqnet.csv, seed 41, five folds)", proto, 0.4980, 4)
    chk(P + "CNN at +/-100 nt (seqnet_matched.csv)", float(byw.loc[100, "mean"]), 0.5203, 4)
    chk(P + "CNN at +/-250 nt", float(byw.loc[250, "mean"]), 0.5247, 4)
    chk(P + "CNN at +/-1,000 nt", float(byw.loc[1000, "mean"]), 0.5190, 4)
    chk(P + "CNN gain over the guide-only CNN at +/-1,000 nt (the DNA the windows summarise)", float(byw.loc[1000, "mean"]) - proto, 0.021, 3)
    chk(P + "CNN gain at its best width (+/-250 nt)", float(byw.loc[250, "mean"]) - proto, 0.027, 3)
    chk(P + "CNN: spread between the +/-100, +/-250 and +/-1,000 nt arms", float(byw.loc[[100, 250, 1000], "mean"].max() - byw.loc[[100, 250, 1000], "mean"].min()), 0.006, 3)
    chk(P + "CNN: smallest fold sd over the four widths", float(byw["std"].min()), 0.007, 3)
    chk(P + "CNN: largest fold sd over the four widths", float(byw["std"].max()), 0.012, 3)
    sh = pd.read_csv(R / "sealed_holdout.csv")
    pv = sh.pivot(index="seal_seed", columns="arm", values=["cv_in_pool", "sealed"])
    chk(P + "sealed_holdout.csv: sealings (whole genes holding 20% of the guides)", float(sh.seal_seed.nunique()), 15, 0)
    chk(P + "sealed_holdout.csv: family A gain in pool CV (frozen minus base-only, random-row folds in the 80% pool)",
        float((pv["cv_in_pool"]["frozen"] - pv["cv_in_pool"]["base_only"]).mean()), 0.1647, 4)
    chk(P + "sealed_holdout.csv: family A gain on the sealed genes (mean of 15 sealings)",
        float((pv["sealed"]["frozen"] - pv["sealed"]["base_only"]).mean()), 0.1560, 4)
    chk(P + "sealed_holdout.csv: columns of the frozen arm and of the base-only arm", float(sh[sh.arm == "frozen"].n_cols.iloc[0]), 6517, 0)
    chk(P + "   base-only arm", float(sh[sh.arm == "base_only"].n_cols.iloc[0]), 6169, 0)
    cu_ = pd.read_csv(R / "gc_strata_corr_curated.csv").set_index("stratum")
    chk(P + "flank GC against the label on the curated cohort (marginal)", float(cu_.loc["all", "rho_flank"]), 0.176, 3)
    ab_main = pd.read_csv(R / "ablation_main.csv").set_index("experiment")
    chk(P + "windowed means, published matrix, random-row 5 x 5 (ablation_main.csv a_flank)", float(ab_main.loc["a_flank", "d_spearman"]), 0.080, 3)

    h2h = pd.read_csv(R / "headtohead.csv")
    hr = pd.read_csv(R / "crisprhal_rerun.csv")
    hal = hr[(hr.arm == "our_folds") & (hr.seed == 41)].set_index("fold").spearman
    arm = lambda a, s=41: h2h[(h2h.arm == a) & (h2h.seed == s)].set_index("fold").spearman
    local, matched, full = arm("curated_local"), arm("curated_matched"), arm("curated_flank")
    chk(P + "near-cut-only arm (260 columns) was run on three folds only", float(len(local)), 3, 0)
    chk(P + "near-cut-only arm, mean over folds 1-3", float(local.mean()), 0.6178, 4)
    chk(P + "near-cut-only minus crisprHAL 2 on the same three folds", float(local.mean() - hal.loc[local.index].mean()), -0.076, 3)
    chk(P + "near-cut-only minus crisprHAL 2's five-fold mean (the figure this replaces)", float(local.mean() - hal.mean()), -0.079, 3)
    chk(P + "crisprHAL 2 at its published configuration, our five folds, seed 41", float(hal.mean()), 0.6971, 4)
    chk(P + "matched-span arm (282 columns), five folds", float(matched.mean()), 0.6724, 4)
    chk(P + "matched-span arm, sd over the five folds", float(matched.std()), 0.0094, 4)
    chk(P + "crisprHAL 2 ahead of the matched-span arm in every one of the five folds", float((hal - matched > 0).sum()), 5, 0)
    chk(P + "crisprHAL 2 minus matched-span arm", float(hal.mean() - matched.mean()), 0.025, 3)
    chk(P + "full family A, seed 41, five folds (LightGBM)", float(full.mean()), 0.7078, 4)
    chk(P + "full family A minus crisprHAL 2, seed 41, five folds", float(full.mean() - hal.mean()), 0.011, 3)
    chk(P + "full family A minus matched-span arm, same model, seed 41", float(full.mean() - matched.mean()), 0.035, 3)

    ww = pd.read_csv(R / "crisprhal_wide_window.csv")
    nat = ww[(ww.model == "crisprHAL 2") & (ww.arm == "native_378")].set_index("batch").spearman
    chk(P + "batch cost at 378 nt: batch 256 minus batch 1,024 (fold 1)", float(nat[256] - nat[1024]), -0.011, 3)
    w256 = ww[(ww.model == "crisprHAL 2") & (ww.batch == 256)].set_index("input_nt").spearman
    chk(P + "decline at 2,023 nt against 378 nt as a multiple of crisprHAL 2's five-fold sd at 378 nt", float((w256[378] - w256[2023]) / hal.std()), 7, 0)
    chk(P + "crisprHAL 2's across-fold sd at 378 nt (our five folds, seed 41)", float(hal.std()), 0.0067, 4)
    sl = ww[ww.model == "SLICER"].set_index("arm").spearman
    for nt, v in ((23, 0.5494), (378, 0.6799), (1023, 0.6602), (2023, 0.6299)):
        chk(P + f"width experiment (fold 1, one seed, batch 256): crisprHAL 2 at {nt} nt", float(w256[nt]), v, 4)
    for nt, v in ((23, -0.130), (1023, -0.020), (2023, -0.050)):
        chk(P + f"   crisprHAL 2 at {nt} nt minus its 378 nt arm", float(w256[nt] - w256[378]), v, 3)
    chk(P + "   SLICER with the 20-mer representation only", float(sl["no_flank"]), 0.5473, 4)
    chk(P + "   SLICER, family A cut to 189/166 nt", float(sl["matched"]), 0.6568, 4)
    chk(P + "   SLICER, family A to +/-1,000 nt", float(sl["full_flank"]), 0.6944, 4)
    chk(P + "   crisprHAL 2 minus SLICER at 23 nt (tied)", float(w256[23] - sl["no_flank"]), 0.002, 3)
    chk(P + "   crisprHAL 2 minus SLICER at 378 nt", float(w256[378] - sl["matched"]), 0.023, 3)
    chk(P + "   SLICER at 2,023 nt minus crisprHAL 2", float(sl["full_flank"] - w256[2023]), 0.065, 3)
    chk(P + "   the width experiment is one fold", float(ww.fold.nunique()), 1, 0)

    # ------------------------------------------------------------------ Finding 3: transfer
    print("report R2: transfer")
    tm = pd.read_csv(R / "transfer_model.csv")
    pick = lambda s, t, a: tm[(tm.source == s) & (tm.target == t) & (tm.arm == a)].spearman
    chk(P + "SpCas9 -> eSpCas9, base + family A, mean of 3 seeds", float(pick("WT-SpCas9", "eSpCas9", "base+a_flank").mean()), 0.689, 3)
    chk(P + "SpCas9 -> eSpCas9: seeds in the mean", float(len(pick("WT-SpCas9", "eSpCas9", "base+a_flank"))), 3, 0)
    prot = {d: json.loads((config.INTERIM / f"transfer_{d}" / "meta.json").read_text())["protospacers"] for d in ("WT-SpCas9", "eSpCas9", "TevSpCas9")}
    wt, es, tev = set(prot["WT-SpCas9"]), prot["eSpCas9"], set(prot["TevSpCas9"])
    chk(P + "WT-SpCas9 guides that are also in the eSpCas9 screen (of 33,567)", float(len(wt & set(es))), 33567, 0)
    chk(P + "eSpCas9 test rows whose guide is a SpCas9 training guide, %", 100 * sum(g in wt for g in es) / len(es), 56, 0)
    chk(P + "20-mers shared by the SpCas9 and C. rodentium screens", float(len(wt & tev)), 1, 0)
    ct = pd.read_csv(R / "transfer.csv")
    loc = ct[(ct.dataset == "eSpCas9") & (ct.arm == "base+a_flank") & (ct.cv == "random")]
    chk(P + "eSpCas9 'local' denominator: random-row 5-fold, seed 41", float(loc.spearman.mean()), 0.784, 3)
    chk(P + "eSpCas9 'local' folds (one seed)", float(len(loc)), 5, 0)
    # The refit (about a minute) is stored in results/post_freeze/transfer_esp_unseen_guides.csv; RSI_AUDIT_HEAVY=1 redoes it and
    # overwrites the file, and every run checks the stored values.
    store = R / "post_freeze" / "transfer_esp_unseen_guides.csv"
    if HEAVY:
        es = _esp_unseen_guides()
        pd.DataFrame([{"arm": a_, "scope": s_, "spearman": es[a_][s_]} for a_ in ("base", "base+a_flank") for s_ in ("all", "seen", "unseen")]
                     + [{"arm": "n_unseen_guides", "scope": "unseen", "spearman": float(es["n_unseen"])}]).to_csv(store, index=False)
    sv = pd.read_csv(store)
    es = {"n_unseen": int(sv[sv.arm == "n_unseen_guides"].spearman.iloc[0])}
    for a_ in ("base", "base+a_flank"):
        es[a_] = sv[sv.arm == a_].set_index("scope").spearman.to_dict()
    s41 = tm[(tm.source == "WT-SpCas9") & (tm.target == "eSpCas9") & (tm.seed == 41)].set_index("arm").spearman
    chk(P + "refit of SpCas9 -> eSpCas9 (seed 41) reproduces transfer_model.csv, base + family A, all rows", es["base+a_flank"]["all"], float(s41["base+a_flank"]), 3)
    chk(P + "eSpCas9 guides absent from the SpCas9 screen (25,922 of 59,489)", float(es["n_unseen"]), 25922, 0)
    chk(P + "SpCas9 -> eSpCas9 on those guides, base + family A, seed 41", es["base+a_flank"]["unseen"], 0.662, 3)
    chk(P + "   same, base only", es["base"]["unseen"], 0.5585, 4)
    chk(P + "   family A share on the unseen guides", es["base+a_flank"]["unseen"] - es["base"]["unseen"], 0.103, 3)
    chk(P + "   family A share on all rows, seed 41", es["base+a_flank"]["all"] - es["base"]["all"], 0.098, 3)

    kb = pd.read_csv(MAIN / "kingdom_transfer_boot.csv").set_index(["direction", "settings"])   # the fork's copy holds the human-fitted rows only
    k = kb.loc[("ecoli -> human", "human_fitted")]
    chk(P + "E. coli -> human, shared columns, human-fitted settings (the table's -0.039)", float(k.spearman), -0.039, 3)
    chk(P + "   its bootstrap interval, lower end", float(k.ci_lo), -0.054, 3)
    chk(P + "   its bootstrap interval, upper end", float(k.ci_hi), -0.024, 3)
    kc = kb.loc[("ecoli -> human", "champion")]
    chk(P + "E. coli -> human, champion settings", float(kc.spearman), -0.046, 3)
    chk(P + "   its interval, lower end", float(kc.ci_lo), -0.061, 3)
    chk(P + "   its interval, upper end", float(kc.ci_hi), -0.031, 3)
    kt = pd.read_csv(R / "kingdom_transfer.csv")
    chk(P + "shared columns of the two published matrices (kingdom_transfer.csv)", float(kt.n_shared.iloc[0]), 6216, 0)
    chk(P + "human rows scored by the E. coli-trained model", float(kt[kt.direction == "ecoli -> human"].n_test.iloc[0]), 17421, 0)
    hb = pd.read_csv(R / "human_baseline.csv").set_index("arm")
    chk(P + "human 'local' rho, row-wise folds, human-fitted settings (human_baseline.csv)", float(hb.loc["jacky_400_1000", "spearman"]), 0.508, 3)
    tw = pd.read_csv(OUT / "human_twin_check.csv").set_index("arm")
    chk(P + "the same arm, twin-grouped folds (human_twin_check.csv)", float(tw.loc["jacky_400_1000", "spearman"]), 0.2515, 4)

    cs = pd.read_csv(R / "cui_final_exam.csv").set_index("arm")
    chk(P + "zero-shot on Cui et al.'s CRISPRi screen: SLICER", float(cs.loc["slicer", "spearman"]), 0.061, 3)
    chk(P + "zero-shot on Cui et al.'s CRISPRi screen: crisprHAL 2", float(cs.loc["crisprhal", "spearman"]), 0.087, 3)
    chk(P + "crisprHAL 2's |rho| as a multiple of SLICER's (the report says about 1.4 times)", float(cs.loc["crisprhal", "spearman"] / cs.loc["slicer", "spearman"]), 1.4, 1)
    chk(P + "guides scored zero-shot on Cui's screen", float(cs.loc["slicer", "n_test"]), 76325, 0)
    cu = pd.read_csv(R / "cui_screen.csv")
    chk(P + "trained on Cui's screen, gene-grouped folds, base columns", float(cu[(cu.cv == "gene") & (cu.arm == "base")].spearman.iloc[0]), 0.458, 3)

    # ------------------------------------------------------------------ Finding 4
    print("report R2: Finding 4")
    hd = pd.read_csv(R / "representation_headroom.csv")
    hd = hd[hd.group != "summary"].assign(a=lambda d: d.rho_with_residual.abs())
    top = hd.groupby(["arm", "group"]).a.max()
    chk(P + "residuals of the champion on the published matrix (random-row 5-fold, seed 41): all 6,232, strongest column it has", float(top["published_full", "given to the model"]), 0.028, 3)
    chk(P + "all 6,232 + family A, strongest column it has", float(top["published_full+a_flank", "given to the model"]), 0.031, 3)
    chk(P + "all 6,232 + family A: no column is held back", float(("published_full+a_flank", "dropped from the matrix") not in top.index), 1, 0)
    chk(P + "427 + family A, strongest column it has", float(top["subset_427+a_flank", "given to the model"]), 0.028, 3)
    chk(P + "427 + family A, strongest held-back column", float(top["subset_427+a_flank", "dropped from the matrix"]), 0.070, 3)
    held = hd[(hd.arm == "published_full") & (hd.group == "dropped from the matrix")]
    chk(P + "all 6,232: the strongest held-back column is a family A column (eng.flank.*)", float(held.loc[held.a.idxmax(), "column"].startswith("eng.flank.")), 1, 0)

    rd = pd.read_csv(R / "representation_depth.csv")
    d4 = rd[rd.max_depth == 4].set_index("arm").spearman
    chk(P + "through the top-300 selector, adding trimers to mono+di (champion depth 4, published matrix)", float(d4["mono_di_tri"] - d4["mono_di_364"]), -0.0025, 4)

    cfg = json.loads((OUT / "tune_configs.json").read_text())
    chk(P + "tuned d099x trees over the champion's", cfg["d099x"]["n_estimators"] / cfg["champion"]["n_estimators"], 10, 0)
    chk(P + "tuned d099x learning rate over the champion's (a quarter)", cfg["d099x"]["learning_rate"] / cfg["champion"]["learning_rate"], 0.25, 2)
    chk(P + "tuned d099x depth", float(cfg["d099x"]["max_depth"]), 5, 0)
    chk(P + "tuned d099x selected columns", float(cfg["d099x"]["nf"]), 450, 0)

    fa = pd.read_csv(OUT / "family_a_at_tuned.csv")
    chk(P + "family_a_at_tuned.csv: the five test folds of one seed together hold the 30,203 genic guides",
        float(fa[(fa.seed == 41) & (fa.arm == "champion") & (fa.matrix == "base")].n_test.sum()), 30203, 0)
    cj = pd.read_csv(R / "curated_joint.csv").set_index("arm")
    ac = pd.read_csv(R / "ablation_curated.csv").set_index("family")
    chk(P + "the ten other families add this beyond family A (all eleven minus family A, curated, random-row)",
        float(cj.loc["all_eleven", "d_spearman"] - ac.loc["a_flank", "d_spearman"]), 0.039, 3)


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
        print(f"  {'ok ' if ok else 'BAD'} {label:110s} computed {got:+.5f}  report {claimed:+.{max(dec, 0)}f}")

    run(chk)
    print(f"\n{n} checks, {len(bad)} mismatches")
    for lab, got, claimed in bad:
        print(f"  BAD {lab}: computed {got:+.5f}, report {claimed}")
    raise SystemExit(1 if bad else 0)
