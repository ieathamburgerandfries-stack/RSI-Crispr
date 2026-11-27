"""Audit of the numbers in the draft paper's claims [20]-[28], [32]-[34] and [49] (group B): every number is recomputed from a stored
result file and compared with the value the paper prints. Tolerance is half a unit in the last quoted digit, as in `verify_numbers`.

    ./run_ext.sh sgrna.paper.paper_numbers_B

`run(chk)` takes the audit's own `chk(label, got, claimed, dec)`, so `verify_numbers` can call it; the harness at the bottom defines the same
function so the module also runs alone. Which cohort and protocol each number belongs to is stated beside it, because several values recur
for different quantities (0.8097 is the replicate agreement of crisprHAL 2's counts; 0.8095 in `ceiling.json` is the WT-against-eSpCas9
agreement; -0.20 is the published matrix's target-GC correlation and -0.147 the curated cohort's).

A check that fails here is a number in the paper that its file does not support; the paper value is left as written.
Quantities taken from outside the repository (replicate correlations 0.71 to 0.77 of Haeussler et al., the field's "about 0.7" of
Konstantakos et al.) are not checked; the values derived from them by arithmetic are.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..common import INTERIM, OUT

P = "paper B: "


def _sb(r, k=2):
    """Spearman-Brown reliability of the mean of k replicates whose pairwise agreement is r."""
    return k * r / (1 + (k - 1) * r)


def run(chk):
    R = OUT                      # the fork's own results/ (not the main project's untracked copy); OUT is results

    # ------------------------------------------------------------------------------------------------------------------------------
    # [20] transfer without retraining
    # transferred: transfer_model.csv, fit on all of the source screen, predict all of the target, LightGBM, mean of seeds 41-43.
    #   WT-SpCas9 = crisprHAL 2's curated 33,567 guides; eSpCas9 = their 59,489-row release; TevSpCas9 = 25,210 C. rodentium guides.
    # local: the same pipeline trained on the target screen with random-row 5-fold CV:
    #   eSpCas9 and TevSpCas9 from transfer.csv (cv == "random", base+a_flank); WT-SpCas9 from ablation_curated.csv (curated, a_flank).
    # The kingdom row is the published 13,880-row E. coli matrix against the published 17,421-row human matrix, shared columns only.
    tm = pd.read_csv(R / "transfer_model.csv")
    tl = pd.read_csv(R / "transfer.csv")
    ac = pd.read_csv(R / "ablation_curated.csv").set_index("family")

    def transferred(src, tgt, arm):
        g = tm[(tm.source == src) & (tm.target == tgt) & (tm.arm == arm)]
        assert g.seed.nunique() == 3, (src, tgt, arm, g.seed.unique())
        return float(g.spearman.mean())

    def local_tl(ds):
        return float(tl[(tl.dataset == ds) & (tl.arm == "base+a_flank") & (tl.cv == "random")].spearman.mean())

    chk(P + "[20] SpCas9 -> eSpCas9, share of the locally trained rho kept (%)",
        100 * transferred("WT-SpCas9", "eSpCas9", "base+a_flank") / local_tl("eSpCas9"), 88, 0)
    chk(P + "[20] E. coli -> C. rodentium, share of the locally trained rho kept (%)",
        100 * transferred("WT-SpCas9", "TevSpCas9", "base+a_flank") / local_tl("TevSpCas9"), 91, 0)
    chk(P + "[20] C. rodentium -> E. coli, share of the locally trained rho kept (%)",
        100 * transferred("TevSpCas9", "WT-SpCas9", "base+a_flank") / float(ac.loc["a_flank", "spearman"]), 89, 0)
    chk(P + "[20] C. rodentium -> E. coli, flank features carry (base+A minus base, transferred)",
        transferred("TevSpCas9", "WT-SpCas9", "base+a_flank") - transferred("TevSpCas9", "WT-SpCas9", "base"), 0.149, 3)
    chk(P + "[20] C. rodentium -> E. coli, rho that arrives (base+A, transferred)",
        transferred("TevSpCas9", "WT-SpCas9", "base+a_flank"), 0.628, 3)
    kb = pd.read_csv(R / "kingdom_transfer_boot.csv").set_index(["direction", "settings"])
    hb = pd.read_csv(R / "human_baseline.csv").set_index("arm")
    k_rho = float(kb.loc[("ecoli -> human", "human_fitted"), "spearman"])
    k_loc = float(hb.loc["jacky_400_1000", "spearman"])
    chk(P + "[20] E. coli -> human, rho (shared columns, human-fitted settings, 17,421 human rows)", k_rho, -0.039, 3)
    chk(P + "[20] E. coli -> human, the locally trained human rho (row-wise folds, same settings)", k_loc, 0.508, 3)
    chk(P + "[20] E. coli -> human 'keeps none' (share kept, floored at 0, %)", max(0.0, 100 * k_rho / k_loc), 0, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [21] / [23] the position x base maps, E. coli (curated 33,567 guides) against human (Noshay's matrix, 13,138 distinct guides)
    # mechanism_position_base_compare.csv: E. coli and human effect (SD of the label) of each base at each of the 20 positions.
    cmp_ = pd.read_csv(OUT / "mechanism_position_base_compare.csv").set_index(["position", "base"])
    chk(P + "[21] G at position 20, E. coli (SD of the label)", float(cmp_.loc[(20, "G"), "ecoli"]), -0.32, 2)
    chk(P + "[21] G at position 20, human (SD of the label)", float(cmp_.loc[(20, "G"), "human"]), 0.13, 2)
    chk(P + "[21] cells in each map", float(len(cmp_)), 80, 0)
    chk(P + "[21] rank correlation of the two 80-cell maps", float(spearmanr(cmp_.ecoli, cmp_.human).statistic), -0.052, 3)
    both = cmp_[cmp_.ecoli_clear & cmp_.human_clear]
    chk(P + "[23] cells that clear zero in both maps", float(len(both)), 27, 0)
    chk(P + "[23] ... of which opposite in sign", float((np.sign(both.ecoli) != np.sign(both.human)).sum()), 15, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [22] eukaryotic datasets; CRISPOR benchmark collection, inclusion rule: at least 300 guides (eukaryote_orientation.csv `n`)
    ori = pd.read_csv(OUT / "eukaryote_orientation.csv").set_index("dataset")
    ct = pd.read_csv(OUT / "eukaryote_contrast_g_minus_t.csv").set_index("group")
    big = ori.index[ori.n >= 300]
    chk(P + "[22] eukaryotic datasets with at least 300 guides", float(len(big)), 11, 0)
    chk(P + "[22] ... of which carry the eukaryotic sign (G minus mean of A and C at position 20 above zero)",
        float((ct.loc[big, "g_minus_ac_pos20"] > 0).sum()), 11, 0)
    pr = pd.read_csv(OUT / "eukaryote_map_pairs.csv")
    pr = pr[~pr.same_library.astype(bool)].copy()
    pr["kind"] = [" - ".join(sorted([a, b])) for a, b in zip(pr.species_a, pr.species_b)]
    med = lambda kind, col: float(pr[pr.kind == kind][col].median())
    chk(P + "[22] human - mouse, median map correlation corrected for reliability (16 library pairs)", med("human - mouse", "rho_corrected"), 0.56, 2)
    chk(P + "[22] human - human, median map correlation corrected for reliability (26 library pairs)", med("human - human", "rho_corrected"), 0.59, 2)
    chk(P + "[22] E. coli - human, median map correlation corrected for reliability (8 library pairs)", med("E. coli - human", "rho_corrected"), -0.16, 2)
    chk(P + "[22] E. coli - human, median map correlation, uncorrected (8 pairs; the paper quotes the corrected one beside the other corrected medians)", med("E. coli - human", "rho"), -0.14, 2)
    # target GC against activity. The curated cohort is the one Figure 3's E. coli map is built on (mechanism._cohort); -0.20 is the
    # published 13,880-row matrix's figure (gc_strata_corr.csv, n = 13,879).
    gcc = pd.read_csv(R / "gc_strata_corr_curated.csv").set_index("stratum")
    gcp = pd.read_csv(R / "gc_strata_corr.csv").set_index("stratum")
    chk(P + "[22] target GC vs activity in E. coli, curated 33,567-guide cohort (the cohort of Figure 3)",
        float(gcc.loc["all", "rho_target"]), -0.147, 3)
    chk(P + "[22] target GC vs activity in E. coli, published 13,880-row matrix (-0.20; not the cohort of Figure 3)",
        float(gcp.loc["all", "rho_target"]), -0.20, 2)
    chk(P + "[22] target GC vs activity in human (13,138 distinct guides, labels averaged; recomputed from the decoded matrix)",
        _human_target_gc(), 0.018, 3)
    pb = pd.read_csv(OUT / "mechanism_position_base.csv").set_index(["position", "base"])
    chk(P + "[22] E. coli: G next to the PAM is disfavoured and T favoured, each clear of zero",
        float(pb.loc[(20, "G"), "hi"] < 0 and pb.loc[(20, "T"), "lo"] > 0), 1, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [24] / [27] the label's ceiling, curated cohort, from crisprHAL 2's published per-replicate counts (curated_ceiling.json)
    cc = json.loads((R / "curated_ceiling.json").read_text())["arms"]["1-1 / 2-2"]
    chk(P + "[24] replicate agreement (Spearman, Cas9 replicate 1 vs control 1 against replicate 2 vs control 2)", cc["replicate_spearman"], 0.8097, 4)
    chk(P + "[27] the same, as quoted in the Figure 4 caption", cc["replicate_spearman"], 0.810, 3)
    chk(P + "[24] Spearman-Brown reliability of the two-replicate score (stored)", cc["reliability_spearman_brown"], 0.8948, 4)
    chk(P + "[24] Spearman-Brown reliability recomputed from the stored agreement", _sb(cc["replicate_spearman"]), 0.8948, 4)
    chk(P + "[24] ceiling, the square root of the reliability", cc["ceiling"], 0.946, 3)
    chk(P + "[27] the same, as quoted in the Figure 4 caption", cc["ceiling"], 0.946, 3)
    chk(P + "[24] ceiling, lower end of its interval", cc["ceiling_ci"][0], 0.944, 3)
    chk(P + "[24] ceiling, upper end of its interval", cc["ceiling_ci"][1], 0.947, 3)
    # the best model: tuned d099x, base + family A, gene-grouped 5 folds x 3 seeds on the 30,203 genic curated guides
    fa = pd.read_csv(OUT / "family_a_at_tuned_summary.csv").set_index("comparison")
    chk(P + "[24] best model (tuned d099x, base + family A; gene-grouped CV, curated cohort)",
        float(fa.loc["family A increment, d099x", "with_a"]), 0.7314, 4)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [25] the field's ceiling recomputed: arithmetic on the cited 0.71 and 0.77 (the cited values themselves are external), and the stored table
    chk(P + "[25] attainable ceiling at r = 0.71 (square root of the Spearman-Brown reliability)", math.sqrt(_sb(0.71)), 0.911, 3)
    chk(P + "[25] attainable ceiling at r = 0.77", math.sqrt(_sb(0.77)), 0.933, 3)
    cp = pd.read_csv(OUT / "ceiling_parametric.csv").set_index(["replicate_agreement_r", "k_replicates"])
    chk(P + "[25] attainable ceiling at r = 0.71, stored (ceiling_parametric.csv)", float(cp.loc[(0.71, 2), "ceiling"]), 0.911, 3)
    chk(P + "[25] attainable ceiling at r = 0.77, stored (ceiling_parametric.csv)", float(cp.loc[(0.77, 2), "ceiling"]), 0.933, 3)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [26] out of material, not out of capacity
    # representation_headroom.csv: champion, published 13,880-row matrix, random-row 5-fold CV, seed 41; out-of-fold residuals against every column.
    rh = pd.read_csv(R / "representation_headroom.csv")
    given = lambda arm: float(rh[(rh.arm == arm) & (rh.group == "given to the model")].rho_with_residual.abs().max())
    chk(P + "[26] strongest |rho| of a column the model was given with its residuals (all 6,232 published columns; no family A)",
        given("published_full"), 0.028, 3)
    chk(P + "[26] the same, full matrix with family A given (the figure the paper quotes: 0.031)",
        given("published_full+a_flank"), 0.031, 3)
    chk(P + "[26] the same, 427 columns with family A given", given("subset_427+a_flank"), 0.028, 3)
    chk(P + "[26] tuning gain without flank features (d099x minus champion, base only; gene-grouped, curated)",
        float(fa.loc["tuning gain (d099x - champion) on base", "delta"]), 0.003, 3)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [28] tuned SLICER (XGBoost d099x) against crisprHAL 2's released weights on crisprHAL 2's own train/test splits
    for split, tag, d, lo, hi in (("espcas9", "eSpCas9 (47,591 train / 11,898 test)", 0.0286, 0.0239, 0.0331),
                                  ("wt", "SpCas9 (26,830 train / 6,737 test)", 0.0226, 0.0139, 0.0314)):
        v = pd.read_csv(OUT / f"vs_crisprhal_{split}.csv").set_index("arm").loc["tuned XGBoost (d099x)"]
        chk(P + f"[28] tuned SLICER minus crisprHAL 2, rho, {tag}", float(v["diff"]), d, 4)
        chk(P + f"[28] ... lower bound, {tag}", float(v["lo"]), lo, 4)
        chk(P + f"[28] ... upper bound, {tag}", float(v["hi"]), hi, 4)
        pn = pd.read_csv(OUT / f"panel_{split}.csv").set_index("arm").loc["tuned XGBoost"]
        for metric, col in (("Kendall tau", "kendall_tau"), ("Pearson", "pearson"), ("R2", "r2")):
            chk(P + f"[28] {metric} difference clear of zero in the same direction, {tag}",
                float(pn[f"d_{col}"] > 0 and pn[f"lo_{col}"] > 0), 1, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [32] the full matrix against the 104-column non-redundant version (curated, gene-grouped, 30,203 genic guides, three seeds)
    rp = pd.read_csv(R / "representation.csv").set_index("arm")
    chk(P + "[32] columns of the published matrix that restate the 20-mer (6,232 - 74 mononucleotide - 63 hand-named)",
        float(rp.loc["published_full", "n_columns"] - rp.loc["mono_74", "n_columns"] - rp.loc["hand_named_only", "n_columns"]), 6095, 0)
    chk(P + "[32] columns of the compact representation", float(len(pd.read_csv(OUT / "compact_model_shap_columns.csv"))), 104, 0)
    cs = pd.read_csv(OUT / "compact_model_stability.csv").set_index(["method", "level"])
    chk(P + "[32] SHAP importance agreement per column between the five tree families, 104 columns", float(cs.loc[("shap", "column"), "agreement"]), 0.87, 2)
    ca = pd.read_csv(OUT / "compact_model_accuracy.csv", index_col=0)
    chk(P + "[32] cost of the compact model in rho (1 - share of the full recipe, %)", 100 * (1 - float(ca.loc["compact (104)", "share_of_full"])), 12, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [33] sensitivity-n, 1 to 6 blocks removed; crisprHAL 2's own WT-SpCas9 split, 100 test guides, shipped models
    fz = pd.read_csv(OUT / "interp_compare_faithfulness.csv")
    fz = fz[~fz.shuffled_control].groupby("model").pearson.agg(["min", "max"])
    chk(P + "[33] sensitivity-n, SLICER (shipped), lowest over the block counts", float(fz.loc["SLICER (shipped)", "min"]), 0.96, 2)
    chk(P + "[33] sensitivity-n, SLICER (shipped), highest over the block counts", float(fz.loc["SLICER (shipped)", "max"]), 0.97, 2)
    chk(P + "[33] sensitivity-n, crisprHAL 2, lowest over the block counts", float(fz.loc["crisprHAL 2", "min"]), 0.86, 2)
    chk(P + "[33] sensitivity-n, crisprHAL 2, highest over the block counts", float(fz.loc["crisprHAL 2", "max"]), 0.88, 2)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [34] models near rho 0.70 against the label's ceiling 0.946
    ceil = cc["ceiling"]
    tuned = float(fa.loc["family A increment, d099x", "with_a"])
    chk(P + "[34] the best model's share of the ceiling, % (0.7314 / 0.946)", 100 * tuned / ceil, 77, 0)
    chk(P + "[34] the remaining rho below the ceiling at the best model (0.946 minus 0.7314)", ceil - tuned, 0.21, 2)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [49] the two models' guide maps (position x base substitution maps, WT-SpCas9 split)
    cm = pd.read_csv(OUT / "interp_compare_cross_model.csv").set_index(["a", "b"]).spearman
    chk(P + "[49] crisprHAL 2 against SLICER (shipped), substitution maps", float(cm[("crisprHAL 2", "SLICER (shipped)")]), 0.97, 2)
    chk(P + "[49] crisprHAL 2 against SLICER-104, substitution maps", float(cm[("crisprHAL 2", "SLICER-104")]), 0.97, 2)


def _human_target_gc():
    """Spearman correlation of target GC with the label on the human matrix, as `mechanism.human_bases` prints it but does not store:
    guides deduplicated (17,421 rows decode to 13,138 distinct 20-mers), labels averaged within a guide. Reads the decoded matrix
    cache (data/interim/post_freeze/human/decoded.csv); returns nan if it is absent."""
    path = INTERIM / "human" / "decoded.csv"
    if not path.exists():
        return float("nan")
    dec = pd.read_csv(path)
    y = pd.to_numeric(dec.y, errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & dec.decoded_ok.to_numpy(bool)
    g = pd.DataFrame({"p": dec.protospacer[ok].astype(str).to_numpy(), "y": y[ok]}).groupby("p", sort=False).y.mean()
    gc = np.array([(s.count("G") + s.count("C")) / 20 for s in g.index])
    return float(spearmanr(gc, g.to_numpy()).statistic)


if __name__ == "__main__":
    from .standalone import run_alone
    run_alone(run, "paper")
