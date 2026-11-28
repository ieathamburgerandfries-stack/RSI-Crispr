"""Audit of the numbers in the draft paper's claims [0]-[19], [35]-[45] (group A): data sets and methods, the flank effect, its
mechanism and the appendix constraints. Every number is recomputed from a stored result file (or a data file of the repository)
and compared with the value the paper prints. Tolerance is half a unit in the last quoted digit, as in `verify_numbers`.

    ./run_ext.sh sgrna.paper.paper_numbers_A

`run(chk)` takes the audit's own `chk(label, got, claimed, dec)`, so `verify_numbers` can call it; the harness at the bottom defines
the same function so the module also runs alone.

Which cohort and protocol a number belongs to is stated beside it, because the same value recurs for different quantities
(+0.156 is the flank gain on 15 sealed gene sets and also the gene-grouped champion increment of the 3-seed protocol, +0.1563;
+0.164 is the curated cohort under *random-row* five-fold CV, and the gene-grouped figure is +0.157). Cohorts:
  published   Noshay et al.'s 13,880-row matrix, label `cut.score` (13,879 located), 6,232 columns, base 6,232
  curated     crisprHAL 2's 33,567 guides, ALDEx2 score, base 6,169 columns; "genic" = the 30,203 guides inside an annotated gene
A check that FAILS here is a number in the paper that its file does not support, or a number that matches only on another cohort or
protocol than the one the sentence names; the paper value is left as written and the stored value is in the printed `computed`.

Not checked (facts taken from outside the repository): the 236 kb size of the C. rodentium TevSpCas9 screen (Ham et al., 2023) and
the descriptions of methods and data sets taken from their papers (ALDEx2, XGBoost, the CRISPOR benchmark of Haeussler et al., 2016).
Sentences with a number the repository cannot source are listed in the report that came with this file, not given a check of their own.

Checks whose label says "cohort guard" or "paper says ... here" are deliberate: the paper's number is right for one cohort or protocol
and the sentence names another, so the check is written against the quantity the sentence names and is expected to fail.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd

from .. import config
from ..common import OUT

P = "paper A: "


def _main_file(name):
    """A results file of the fork's own results/, else the main project's untracked copy (a file the fork does not track)."""
    f = OUT / name
    return f if f.exists() else config.RESULTS / name


def run(chk):
    R = OUT                      # the fork's own results/ (OUT is results)
    RAW = config.RAW / "ecoli_feature_matrix.csv"

    # ------------------------------------------------------------------------------------------------------------------------------
    # [0] the published matrix
    bm = json.loads((config.INTERIM / "base_meta.json").read_text())
    gi = pd.read_csv(config.GUIDE_INDEX)
    hdr = pd.read_csv(RAW, header=config.BASE_MATRIX_HEADER_ROW, nrows=0)
    chk(P + "[0] published matrix rows (base_meta.json shape)", float(bm["shape"][0]), 13880, 0)
    chk(P + "[0] published matrix rows (guide_index.csv)", float(len(gi)), 13880, 0)
    chk(P + "[0] published matrix columns (base_meta.json shape)", float(bm["shape"][1]), 6232, 0)
    chk(P + "[0] published matrix feature columns (raw csv header minus id and cut.score)", float(hdr.shape[1] - 2), 6232, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [1] the curated cohort
    ci = pd.read_csv(config.INTERIM / "curated_guide_index.csv")
    cg = pd.read_csv(R / "cohort_groups.csv").set_index(["cohort", "grouping"])
    em = json.loads((config.INTERIM / "expanded_meta.json").read_text())
    cv = pd.read_csv(R / "curation_variance.csv").set_index("rows")
    chk(P + "[1] curated guides (curated_guide_index.csv)", float(len(ci)), 33567, 0)
    chk(P + "[1] curated guides, distinct protospacers", float(ci.protospacer.nunique()), 33567, 0)
    chk(P + "[1] curated guides inside an annotated gene (cohort_groups.csv)", float(cg.loc[("curated", "gene"), "n"]), 30203, 0)
    chk(P + "[1] genes holding those guides (cohort_groups.csv)", float(cg.loc[("curated", "gene"), "n_groups"]), 4013, 0)
    chk(P + "[1] guides shared with the published matrix (expanded_meta.json in_published)", float(np.sum(em["in_published"])), 8617, 0)
    chk(P + "[1] guides shared with the published matrix (protospacer intersection)",
        float(len(set(ci.protospacer) & set(gi.protospacer))), 8617, 0)
    chk(P + "[1] guides shared with the published matrix (curation_variance.csv)", float(cv.loc["shared with Noshay", "n"]), 8617, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [2] crisprHAL 2's own files and the benchmark collection
    hal = config.CRISPRHAL
    wt_tr = pd.read_csv(hal / "WT-SpCas9_training_data.csv", header=None)
    wt_te = pd.read_csv(hal / "WT-SpCas9_testing_data.csv", header=None)
    es_tr = pd.read_csv(hal / "eSpCas9_training_data.csv", header=None)
    es_te = pd.read_csv(hal / "eSpCas9_testing_data.csv", header=None)
    vw = pd.read_csv(OUT / "vs_crisprhal_wt.csv").iloc[0]
    ve = pd.read_csv(OUT / "vs_crisprhal_espcas9.csv").iloc[0]
    # The paper says "crisprHAL 2's own train and test files for SpCas9 (26,830 / 6,737 guides)". The files hold 26,994 / 6,749 rows;
    # 26,830 / 6,737 are the guides of them that locate in NC_000913.2 and so enter the head-to-head (their sum is the 33,567 cohort).
    chk(P + "[2] SpCas9 (WT) training FILE, rows (26,830 of them locate in the genome; checked below)", float(len(wt_tr)), 26994, 0)
    chk(P + "[2] SpCas9 (WT) testing FILE, rows (6,737 of them locate in the genome; checked below)", float(len(wt_te)), 6749, 0)
    chk(P + "[2] SpCas9 (WT) training guides USED in the head-to-head (vs_crisprhal_wt.csv n_train)", float(vw.n_train), 26830, 0)
    chk(P + "[2] SpCas9 (WT) test guides USED in the head-to-head (vs_crisprhal_wt.csv n_test)", float(vw.n_test), 6737, 0)
    chk(P + "[2] SpCas9 used train + test equals the curated cohort", float(vw.n_train + vw.n_test), 33567, 0)
    chk(P + "[2] eSpCas9 training FILE, rows", float(len(es_tr)), 47591, 0)
    chk(P + "[2] eSpCas9 testing FILE, rows", float(len(es_te)), 11898, 0)
    chk(P + "[2] eSpCas9 screen guides (training and testing files together)", float(len(es_tr) + len(es_te)), 59489, 0)
    chk(P + "[2] eSpCas9 train guides used (vs_crisprhal_espcas9.csv n_train)", float(ve.n_train), 47591, 0)
    chk(P + "[2] eSpCas9 test guides used (vs_crisprhal_espcas9.csv n_test)", float(ve.n_test), 11898, 0)
    from .hooks import MIN_BIG
    ori = pd.read_csv(OUT / "eukaryote_orientation.csv")
    big = ori[ori.n >= MIN_BIG]
    chk(P + "[2] inclusion threshold of the eukaryotic datasets, guides (eukaryote_rules.MIN_BIG)", float(MIN_BIG), 300, 0)
    chk(P + "[2] eukaryotic datasets with at least 300 guides (eukaryote_orientation.csv)", float(len(big)), 11, 0)
    chk(P + "[2]   ... of them human / mouse / zebrafish = 8 / 2 / 1",
        float((big.species == "human").sum() * 100 + (big.species == "mouse").sum() * 10 + (big.species == "zebrafish").sum()), 821, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [3] the base and the flank features
    pa = pd.read_csv(R / "portability_audit.csv")                      # the 63 hand-named columns of the published matrix
    vmap = json.loads((config.INTERIM / "v_column_mapping.json").read_text())["mapping"]
    nq = pd.read_csv(config.INTERIM / "qct_columns.csv", nrows=1).shape[1] - 2
    fl = pd.read_csv(config.FEATURE_DIR / "a_flank_curated.csv", nrows=1)
    fcols = [c for c in fl.columns if c != config.ID_COL]
    chk(P + "[3] base columns = published columns minus the 63 hand-named (portability_audit.csv)", float(bm["shape"][1] - len(pa)), 6169, 0)
    chk(P + "[3] base columns = position x k-mer indicators (v_column_mapping.json) + quantum tables (qct_columns.csv)",
        float(len(vmap) + nq), 6169, 0)
    chk(P + "[3] base columns of the curated cohort (expanded_meta.json)", float(len(em["feature_names"])), 6169, 0)
    chk(P + "[3] longest k-mer among the indicator columns (k <= 4)", float(max(len(v["kmer"]) for v in vmap.values())), 4, 0)
    chk(P + "[3] flank feature columns (a_flank_curated.csv)", float(len(fcols)), 348, 0)
    wins = sorted({int(w) for c in fcols for w in re.findall(r"\.(?:up|dn)(\d+)", c)})
    for w in (50, 250, 500, 1000):
        sides = sum(1 for s in ("up", "dn") if any(re.search(rf"\.{s}{w}\b", c) for c in fcols))
        chk(P + f"[3] flank window {w:,} nt present on both sides of the cut", float(sides), 2, 0)
    chk(P + "[3] number of window sizes in the flank columns", float(len(wins)), 4, 0)
    chk(P + "[3] windows in the flank columns equal config.FLANK_WINDOWS", float(wins == sorted(config.FLANK_WINDOWS)), 1, 0)
    # the near-cut block: one-hot bases and per-position quantum tables at u10..u1, the PAM's N and d1..d15
    seq_pos = sorted({re.search(r"\.seq\.(\w+)\.[ACGT]$", c).group(1) for c in fcols if ".seq." in c})
    chk(P + "[3] 'nearest 10 bases': bases encoded 5' of the protospacer (config.UPSTREAM_POSITIONS)", float(config.UPSTREAM_POSITIONS), 10, 0)
    chk(P + "[3] 'nearest 10 bases': bases encoded 3' of the PAM (config.DOWNSTREAM_POSITIONS; the paper says 15 downstream of the PAM)",
        float(config.DOWNSTREAM_POSITIONS), 15, 0)
    chk(P + "[3]   positions encoded one base at a time = 10 + PAM N + 15", float(len(seq_pos)), 26, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [4] SLICER's settings
    tc = json.loads((OUT / "tune_configs.json").read_text())
    chk(P + "[4] default version: columns kept (tune_configs.json champion nf)", float(tc["champion"]["nf"]), 300, 0)
    chk(P + "[4] default version: trees (tune_configs.json champion n_estimators)", float(tc["champion"]["n_estimators"]), 400, 0)
    chk(P + "[4] default version: columns kept (config.N_FEATURES)", float(config.N_FEATURES), 300, 0)
    chk(P + "[4] default version: trees (config.N_ESTIMATORS)", float(config.N_ESTIMATORS), 400, 0)
    chk(P + "[4] tuned version: columns kept (tune_configs.json d099x nf)", float(tc["d099x"]["nf"]), 450, 0)
    chk(P + "[4] tuned version: trees (tune_configs.json d099x n_estimators)", float(tc["d099x"]["n_estimators"]), 4000, 0)
    chk(P + "[4] tuned version: depth (tune_configs.json d099x max_depth)", float(tc["d099x"]["max_depth"]), 5, 0)
    # the 70% / 30% split of the 4,013 genes, rebuilt exactly as `tune_ecoli.Cohort` does (one fixed seed, 30% of genes confirm)
    from .hooks import load_curated_index
    from .hooks import SPLIT_SEED
    g = load_curated_index(verbose=False)["gene_name"]
    genes = np.array(sorted(set(g[g.notna()])))
    conf = set(np.random.default_rng(SPLIT_SEED).choice(genes, size=int(round(0.3 * len(genes))), replace=False))
    chk(P + "[4] genes holding the search: share of genes (the tuning part, DEV)", 100 * (1 - len(conf) / len(genes)), 70, 0)
    chk(P + "[4] genes confirming the search: share of genes (CONFIRM)", 100 * len(conf) / len(genes), 30, 0)
    cl = pd.read_csv(OUT / "clean_vs_hal.csv")
    chk(P + "[4]   the confirming genes, count (clean_vs_hal.csv n_genes; report: 1,204)", float(cl.n_genes.iloc[0]), len(conf), 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [5] crisprHAL 2
    cx = json.loads((R / "complexity.json").read_text())
    chk(P + "[5] crisprHAL 2 trainable parameters, WT-SpCas9 weights (complexity.json)", float(cx["crisprhal2"]["WT-SpCas9"]), 790533, 0)
    chk(P + "[5] input length of the WT-SpCas9 files, nt (training file sequences)", float(wt_tr[0].str.len().median()), 378, 0)
    chk(P + "[5]   all WT-SpCas9 training sequences have that length", float((wt_tr[0].str.len() == 378).mean()), 1, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [6] cross-validation
    from .hooks import BLOCK
    blocks = (pd.to_numeric(ci["left"], errors="coerce") // BLOCK).nunique()
    chk(P + "[6] block width, kb (harness.cohort_groups.BLOCK)", BLOCK / 1000, 100, 0)
    chk(P + "[6] 100 kb blocks the curated guides fall in (recomputed) = cohort_groups.csv", float(blocks), float(cg.loc[("curated", "100kb_block"), "n_groups"]), 0)
    chk(P + "[6] folds per seed (config.N_SPLITS)", float(config.N_SPLITS), 5, 0)
    chk(P + "[6] seeds (config.SEEDS); the tuned-model arms of tune_fullcv.csv use only 3 of them", float(len(config.SEEDS)), 5, 0)
    tf = pd.read_csv(OUT / "tune_fullcv.csv")
    chk(P + "[6]   seeds in tune_fullcv.csv (the +0.0162 and the 0.7314 rest on these)", float(tf.seed.nunique()), 3, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [7] sealing
    sh = pd.read_csv(R / "sealed_holdout.csv")
    share = (sh.n_sealed / (sh.n_sealed + sh.n_dev)).mean() * 100
    chk(P + "[7] share of the GUIDES sealed, % (sealed_holdout.csv n_sealed / total; whole genes are sealed)", float(share), 20, 0)
    chk(P + "[7] share of the guides left to re-derive selection and settings, %", float(100 - share), 80, 0)
    chk(P + "[7] sealings (distinct seal_seed in sealed_holdout.csv)", float(sh.seal_seed.nunique()), 15, 0)
    from .hooks import SEAL_FRACTION
    chk(P + "[7]   fraction of ROWS to seal (controls.sealed.SEAL_FRACTION; the code seals whole genes until it is reached)", SEAL_FRACTION * 100, 20, 0)
    from .hooks import _seal
    protos = ci.protospacer.to_numpy()
    gmap = load_curated_index(verbose=False).set_index("protospacer")["gene_name"].reindex(protos)
    gshare = [gmap[_seal(protos, seed=int(sd), verbose=False)].nunique() / gmap.nunique() * 100 for sd in sorted(sh.seal_seed.unique())]
    chk(P + "[7]   share of the 4,013 genes inside the sealed guides, % mean over the 15 sealings (the paper says whole genes holding 20% of guides)", float(np.mean(gshare)), 22, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [8] head-to-head: paired 95% bootstrap over guides (read from the code that produces every interval of the comparison)
    for mod in ("vs_crisprhal", "final_eval"):
        src = (pathlib_root() / "compare" / f"{mod}.py").read_text()
        m = re.search(r"np\.percentile\(\w+, \[([\d.]+), ([\d.]+)\]\)", src)
        chk(P + f"[8] bootstrap interval level, % ({mod}.py percentiles)", float(m.group(2)) - float(m.group(1)), 95, 0)
        chk(P + f"[8]   {mod}.py resamples guides (rng.integers over the rows)", float("rng.integers(0, " in src), 1, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [9] the 80 position x base cells
    pb = pd.read_csv(OUT / "mechanism_position_base.csv")
    pc = pd.read_csv(OUT / "mechanism_position_base_compare.csv")
    chk(P + "[9] position x base cells of the E. coli map (mechanism_position_base.csv)", float(len(pb)), 80, 0)
    chk(P + "[9] cells compared with the human map (mechanism_position_base_compare.csv)", float(len(pc)), 80, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [10] the published representation is a re-encoding of the guide (published matrix)
    po = pd.read_csv(R / "portability.csv").set_index("arm")
    chk(P + "[10] columns that regenerate from the protospacer (6,232 - 63 hand-named)", float(bm["shape"][1] - len(pa)), 6169, 0)
    chk(P + "[10] ... of the 6,232, percent", 100 * (bm["shape"][1] - len(pa)) / bm["shape"][1], 99.0, 1)
    chk(P + "[10] variables the matrix spells: bases of the protospacer (about twenty)", float(gi.protospacer.str.len().median()), 20, 0)
    chk(P + "[10] columns computable for any guide (portability.csv portable)", float(po.loc["portable", "n_columns"]), 412, 0)
    chk(P + "[10] published matrix, all columns (portability.csv published_full)", float(po.loc["published_full", "n_columns"]), 6232, 0)
    chk(P + "[10] rho lost by the 412 against all 6,232 (published matrix, 5 x 5 random-row folds)",
        float(po.loc["published_full", "spearman"] - po.loc["portable", "spearman"]), 0.00003, 5)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [11] the one imported guide-derived family that gained heavily: d_mechanics, curated cohort, random-row 5 x 5 folds
    ni = pd.read_csv(R / "no_information.csv").set_index("arm")
    ac = pd.read_csv(R / "ablation_curated.csv").set_index("family")
    chk(P + "[11] all 30 d_mechanics columns over the base (no_information.csv full)", float(ni.loc["full", "d_spearman"]), 0.0991, 4)
    chk(P + "[11]   the same arm in ablation_curated.csv (d_mechanics)", float(ac.loc["d_mechanics", "d_spearman"]), 0.0991, 4)
    chk(P + "[11] share of that gain owed to the 8 flank columns, % (flank_only / full)",
        100 * float(ni.loc["flank_only", "d_spearman"] / ni.loc["full", "d_spearman"]), 98, 0)
    chk(P + "[11]   flank columns in that arm", float(ni.loc["flank_only", "n_family_cols"]), 8, 0)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [12] the size of the flank effect: four cohorts and protocols that must not be interchanged
    acg = pd.read_csv(R / "ablation_curated_gene.csv").set_index("family")
    am = pd.read_csv(R / "ablation_main.csv").set_index("experiment")
    chk(P + "[12] flank gain, curated 33,567 guides, RANDOM-ROW 5-fold CV (ablation_curated.csv)", float(ac.loc["a_flank", "d_spearman"]), 0.164, 3)
    chk(P + "[12] flank gain, curated, GENE-GROUPED CV, 30,203 genic guides (ablation_curated_gene.csv)",
        float(acg.loc["a_flank", "d_spearman"]), 0.157, 3)
    chk(P + "[12]   the same gene-grouped gain is +0.157 (grouped_retention.csv, report Appendix B3)", float(acg.loc["a_flank", "d_spearman"]), 0.157, 3)
    chk(P + "[12]   gene-grouped champion increment, 3 seeds (family_a_at_tuned_summary.csv): a different quantity, same value as sealed",
        float(pd.read_csv(OUT / "family_a_at_tuned_summary.csv").set_index("comparison").loc["family A increment, champion", "delta"]), 0.156, 3)
    sp = sh.pivot(index="seal_seed", columns="arm", values="sealed")
    chk(P + "[12] flank gain on sealed genes, mean over 15 sealings (sealed_holdout.csv frozen - base_only)", float((sp.frozen - sp.base_only).mean()), 0.156, 3)
    chk(P + "[12] flank gain, published matrix (ablation_main.csv a_flank)", float(am.loc["a_flank", "d_spearman"]), 0.080, 3)
    cw = pd.read_csv(OUT / "cut_within_gene_summary.csv").set_index("arm")
    # No stored file holds +0.061. The report cites ablation_grouped.csv for it, which has no family A row. The within-gene gain that IS
    # stored for the flank features is the curated one below.
    chk(P + "[12] flank gain inside genes (gene-centred), curated genic guides, 10 seeds (cut_within_gene_summary.csv)",
        float(cw.loc["a_flank", "within_genes_diff"]), 0.085, 3)
    chk(P + "[12]   lower end of its bootstrap interval over genes", float(cw.loc["a_flank", "within_diff_boot_lo"]), 0.078, 3)
    chk(P + "[12]   upper end", float(cw.loc["a_flank", "within_diff_boot_hi"]), 0.090, 3)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [13], [15] the shift null (published matrix, 100 kb-block CV, 3 seeds x 5 folds, 348 flank columns)
    sn = pd.read_csv(R / "shift_null.csv").sort_values("shift_bp")
    base = pd.read_csv(_main_file("folds_shift_baseline.csv")).drop_duplicates(["seed", "fold"])     # untracked in the fork: main results/
    gain = sn.spearman.to_numpy() - float(base.spearman.mean())
    chk(P + "[13] gain when the flanks come from the right place (shift 0), published matrix", float(gain[0]), 0.0769, 4)
    chk(P + "[13] share of the gain left at a 1 kb shift, % (shift_null.csv)", float(gain[1] / gain[0] * 100), 8, 0)
    chk(P + "[13] gain at a 10 kb shift is gone (<= 0)", float(gain[2] <= 0), 1, 0)
    chk(P + "[15] share of the gain left at 1 kb, % (Figure 1 caption; published matrix)", float(gain[1] / gain[0] * 100), 8, 0)
    chk(P + "[15] gain at 10 kb, rho (Figure 1 caption: gone)", float(gain[2]), 0.0, 1)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [14] tuning with and without the flank features (curated genic guides, gene-grouped, 3 seeds x 5 folds)
    fa = pd.read_csv(OUT / "family_a_at_tuned_summary.csv").set_index("comparison")
    chk(P + "[14] tuning gain d099x - champion with family A (family_a_at_tuned_summary.csv)",
        float(fa.loc["tuning gain (d099x - champion) on base+A", "delta"]), 0.0162, 4)
    chk(P + "[14] tuning gain d099x - champion without family A", float(fa.loc["tuning gain (d099x - champion) on base", "delta"]), 0.0028, 4)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [16], [35] the flank effect is GC, not purine; partial Spearman given target GC, 33,567 curated guides
    dp = pd.read_csv(OUT / "mechanism_distance_profile.csv")
    gc = dp[dp.statistic == "gc"].set_index(["side", "window_nt"]).partial_given_target_gc
    pu = dp[dp.statistic == "purine"].partial_given_target_gc
    chk(P + "[16] smallest GC partial over the windows and sides (n = 33,567)", float(gc.min()), 0.15, 2)
    chk(P + "[16] largest GC partial over the windows and sides", float(gc.max()), 0.20, 2)
    chk(P + "[16] cohort of the partials (mechanism_distance_profile.csv n)", float(dp.n.iloc[0]), 33567, 0)
    for side in ("up", "dn"):
        chk(P + f"[16] window of the largest GC partial, {'upstream' if side == 'up' else 'downstream'}, is 250 or 500 nt",
            float(gc.loc[side].idxmax() in (250, 500)), 1, 0)
    chk(P + "[16] largest GC partial sits at 250 nt (upstream)", float(gc.loc["up"].idxmax()), 250, 0)
    chk(P + "[19] largest GC partial sits at 250 nt (downstream)", float(gc.loc["dn"].idxmax()), 250, 0)
    chk(P + "[19] the 500 nt window is within 0.01 of the 250 nt window on both sides (peak 250 to 500)",
        float(max(abs(gc[(s, 250)] - gc[(s, 500)]) for s in ("up", "dn")) < 0.01), 1, 0)
    chk(P + "[35] largest |purine partial| at any window and side", float(pu.abs().max()), 0.018, 3)
    chk(P + "[35] GC partial, smallest", float(gc.min()), 0.15, 2)
    chk(P + "[35] GC partial, largest", float(gc.max()), 0.20, 2)
    chk(P + "[35] upstream and downstream GC partials differ by at most", float(max(abs(gc[("up", w)] - gc[("dn", w)]) for w in (50, 250, 500, 1000))), 0.026, 3)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [17], [43] the step at the AT-rich end (curated genic guides, 250 nt each side averaged; mechanism.py, tests h1 to h3)
    hg = pd.read_csv(OUT / "hypotheses_followup_hinge.csv").set_index("quantile")
    cc = pd.read_csv(OUT / "hypotheses_followup_concentration.csv").iloc[0]
    sc = pd.read_csv(OUT / "hypotheses_audit_scale.csv").set_index("window_nt")
    chk(P + "[17] flank GC at the edge of the lowest decile, % (hypotheses_followup_hinge.csv)", float(hg.loc[0.10, "threshold_gc"]) * 100, 44.4, 1)
    chk(P + "[17] lowest decile lower than the rest, label units (hypotheses_followup_concentration.csv)", float(-cc.deficit), 3.10, 2)
    chk(P + "[17] ... in SD at 250 nt (hypotheses_audit_scale.csv)", float(-sc.loc[250, "deficit_in_sd"]), 0.95, 2)
    chk(P + "[17]   guides behind it (30,203 genic; the 33,567-guide figure is 3.29, report robustness table)", float(hg.loc[0.10, "n_below"] + hg.loc[0.10, "n_above"]), 30203, 0)
    chk(P + "[43] lowest decile lower than the rest, label units", float(-cc.deficit), 3.10, 2)
    src = (pathlib_root() / "interpret" / "mechanism.py").read_text()
    m = re.search(r"ROUGHLY LINEAR \(probability about ([\d.]+)\)", src)
    chk(P + "[43] pre-registered probability of a roughly linear response (mechanism.py docstring)", float(m.group(1)), 0.6, 1)

    # [18] the slope is steepest beside a favourable seed base (curated genic guides, gene bootstrap)
    h1s = pd.read_csv(OUT / "hypotheses_h1_strata.csv").set_index(["position", "base"])
    h1c = pd.read_csv(OUT / "hypotheses_h1_contrasts.csv")
    h1c = h1c[h1c.steepest == "T minus G"].set_index("position")
    chk(P + "[18] flank-GC correlation with A at position 20 (hypotheses_h1_strata.csv)", float(h1s.loc[(20, "A"), "rho"]), 0.219, 3)
    chk(P + "[18] flank-GC correlation with G at position 20", float(h1s.loc[(20, "G"), "rho"]), 0.118, 3)
    chk(P + "[18] T minus G at position 20 (hypotheses_h1_contrasts.csv)", float(h1c.loc[20, "difference"]), 0.066, 3)
    chk(P + "[18] ... lower bound", float(h1c.loc[20, "lo"]), 0.029, 3)
    chk(P + "[18] ... upper bound", float(h1c.loc[20, "hi"]), 0.102, 3)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [36], [45] resolution of the flank: nested bins (curated, gene-grouped 3 x 5 folds; novel_resolution.csv)
    res = pd.read_csv(OUT / "novel_resolution.csv").set_index("arm")
    b0 = float(res.loc["base only", "rho"])
    ref = float(res.loc["base + family A comp block (24 cols)", "rho"])

    def bins(k):
        lab = [a for a in res.index if a.startswith(f"base + {k} binned")][0]
        return lab, float(res.loc[lab, "rho"]), float(res.loc[lab, "sd"])

    l4, r4, _ = bins(4)
    l16, r16, s16 = bins(16)
    l200, r200, _ = bins(200)
    chk(P + "[36] columns of the 1,000 nt arm: GC and purine x two sides", float(res.loc[l4, "n_cols"]), 4, 0)
    chk(P + "[36] share of the composition signal carried by those 4 numbers, %", (r4 - b0) / (ref - b0) * 100, 75.7, 1)
    chk(P + "[36] bin width of the 16-column arm, nt (arm label)", float(re.search(r"\((\d+) nt bins\)", l16).group(1)), 250, 0)
    chk(P + "[36] 16 columns = 2 stats x 2 sides x (1,000 / 250) bins", 4 * 1000 / float(re.search(r"\((\d+) nt bins\)", l16).group(1)), float(res.loc[l16, "n_cols"]), 0)
    chk(P + "[36] share carried by the 16 columns of 250 nt bins, %", (r16 - b0) / (ref - b0) * 100, 95.2, 1)
    chk(P + "[36] columns of the 20 nt arm", float(res.loc[l200, "n_cols"]), 200, 0)
    chk(P + "[36] 200 columns = 2 stats x 2 sides x (1,000 / 20) bins", 4 * 1000 / float(re.search(r"\((\d+) nt bins\)", l200).group(1)), 200, 0)
    chk(P + "[36] 200 columns of 20 nt bins do no better than 16 (gain over 16 columns below one fold sd)", float(r200 - r16 < s16), 1, 0)
    chk(P + "[36]   share of the 200-column arm, % (the table prints 96.0)", (r200 - b0) / (ref - b0) * 100, 96.0, 1)
    l32, r32, _ = bins(32)
    l80, r80, _ = bins(80)
    chk(P + "[36] share carried by the 32 columns of 125 nt bins, %", (r32 - b0) / (ref - b0) * 100, 97.2, 1)
    chk(P + "[36] share carried by the 80 columns of 50 nt bins, %", (r80 - b0) / (ref - b0) * 100, 98.7, 1)
    chk(P + "[45] columns at which the curve saturates (Figure A1c)", float(res.loc[l16, "n_cols"]), 16, 0)
    chk(P + "[45] columns at which it turns over", float(res.loc[l200, "n_cols"]), 200, 0)
    chk(P + "[45] the 200-column arm scores below the 32- and 80-column arms",
        float(r200 < min(bins(32)[1], bins(80)[1])), 1, 0)
    bp = pd.read_csv(OUT / "mechanism_pairs_blocks.csv").set_index("block_pair").gain_share
    chk(P + "[45] largest pairwise interaction share between blocks, % (mechanism_pairs_blocks.csv)", float(bp.max() * 100), 5.8, 1)

    # ------------------------------------------------------------------------------------------------------------------------------
    # [37] helical phasing, GC skew, bendability periodicity (family J): PUBLISHED matrix, random-row 5 x 5 folds
    aj = pd.read_csv(R / "ablation_jper.csv").set_index("experiment")
    ajj = pd.read_csv(R / "ablation_jper_joint.csv").set_index("experiment")
    chk(P + "[37] family J alone over the base (ablation_jper.csv; published matrix, not curated)", float(aj.loc["j_periodicity", "d_spearman"]), -0.001, 3)
    chk(P + "[37] family J on top of family A (ablation_jper_joint.csv A+J minus ablation_main.csv A; published matrix)",
        float(ajj.loc["a_flank+j_periodicity", "spearman"] - am.loc["a_flank", "spearman"]), 0.0004, 4)
    chk(P + "[37]   (for the record) family J alone on the curated cohort, random-row (ablation_curated.csv); the paper's phasing numbers are the published matrix's", float(ac.loc["j_periodicity", "d_spearman"]), 0.011, 3)

    # [38] DNA shape: i_shape alone over the base (curated, random-row) and on top of family A
    cj = pd.read_csv(R / "curated_joint.csv").set_index("arm")
    chk(P + "[38] i_shape alone over the base (ablation_curated.csv; curated, random-row)", float(ac.loc["i_shape", "d_spearman"]), 0.120, 3)
    chk(P + "[38] i_shape on top of family A, curated (curated_joint.csv A+shape minus ablation_curated.csv A)",
        float(cj.loc["a_flank+i_shape", "spearman"] - ac.loc["a_flank", "spearman"]), 0.0009, 4)
    ash = pd.read_csv(R / "ablation_shape_joint.csv").set_index("experiment")
    chk(P + "[38] i_shape on top of family A, PUBLISHED matrix (ablation_shape_joint.csv minus ablation_main.csv A; not quoted in the paper)",
        float(ash.loc["a_flank+i_shape", "spearman"] - am.loc["a_flank", "spearman"]), 0.0010, 4)

    # [39] supercoiling: +0.139 is the curated family; +0.0112 / +0.0111 are the PUBLISHED matrix under 100 kb-block CV
    ds = pd.read_csv(R / "dissection_supercoiling.csv").set_index("experiment")
    ds = ds[ds.grouping == "bin100k"]
    chk(P + "[39] d_supercoiling family over the base (ablation_curated.csv; curated, random-row)", float(ac.loc["d_supercoiling", "d_spearman"]), 0.139, 3)
    chk(P + "[39] four columns of distance to the origin (dissection_supercoiling.csv z_dosage; published, 100 kb blocks)",
        float(ds.loc["z_dosage", "d_spearman"]), 0.0112, 4)
    chk(P + "[39] the GapR ChIP track alone (z_signal_only; published, 100 kb blocks)", float(ds.loc["z_signal_only", "d_spearman"]), 0.0111, 4)
    chk(P + "[39]   columns in the distance-to-origin arm", float(ds.loc["z_dosage", "columns"] - ds.loc["baseline", "columns"]), 4, 0)
    chk(P + "[39]   the whole family on the published matrix is +0.044 (dissection_supercoiling.csv d_supercoiling); the paper's +0.139 is the curated cohort's", float(ds.loc["d_supercoiling", "d_spearman"]), 0.044, 3)
    chk(P + "[39] the UNTAGGED control track alone (z_control_only; published, 100 kb blocks)", float(ds.loc["z_control_only", "d_spearman"]), 0.0087, 4)

    # [40] essentiality and expression (family G on the curated cohort, random-row 5 x 5; txn_interactions.csv)
    tx = pd.read_csv(R / "txn_interactions.csv").set_index("arm")
    ep = pd.read_csv(OUT / "hypotheses_followup_expression_partial.csv").iloc[0]
    chk(P + "[40] gene geometry alone (txn_interactions.csv no_expression: the 10 non-expression, non-interaction columns)", float(tx.loc["no_expression", "d_spearman"]), 0.0393, 4)
    chk(P + "[40] expression alone (expression_only)", float(tx.loc["expression_only", "d_spearman"]), 0.0213, 4)
    chk(P + "[40] essentiality alone (essential_only, one column)", float(tx.loc["essential_only", "d_spearman"]), 0.00004, 5)
    chk(P + "[40] flank GC against expression, Spearman (hypotheses_followup_expression_partial.csv)", float(ep.rho_gc_expression), -0.003, 3)

    # [41] PAM density (curated genic guides)
    h3 = pd.read_csv(OUT / "hypotheses_h3_pam_density.csv").iloc[0]
    chk(P + "[41] NGG + CCN density given flank GC, partial Spearman (hypotheses_h3_pam_density.csv)", float(h3.pam_given_gc), 0.022, 3)
    chk(P + "[41] percentile among 5,000 GC-matched random sums", float(h3.percentile_of_pam), 52, 0)

    # [42] target GC: correlations on the curated 33,567; the per-stratum ablation is on the PUBLISHED matrix
    gcc = pd.read_csv(R / "gc_strata_corr_curated.csv").set_index("stratum")
    gca = pd.read_csv(R / "gc_strata_ablation.csv")
    chk(P + "[42] target GC against the label (gc_strata_corr_curated.csv, n = 33,567)", float(gcc.loc["all", "rho_target"]), -0.147, 3)
    chk(P + "[42] target GC against flank GC", float(gcc.loc["all", "rho_flank_vs_target_gc"]), 0.245, 3)
    chk(P + "[42] flank GC against the label, marginal", float(gcc.loc["all", "rho_flank"]), 0.176, 3)
    chk(P + "[42] flank GC against the label, target GC held fixed", float(gcc.loc["all", "rho_flank_partial_overall"]), 0.221, 3)
    pv = gca.pivot(index="stratum", columns="arm", values="spearman")
    dl = pv.a_flank - pv.baseline
    chk(P + "[42] smallest family A gain inside a target-GC stratum (gc_strata_ablation.csv)", float(dl.min()), 0.046, 3)
    chk(P + "[42] largest family A gain inside a target-GC stratum", float(dl.max()), 0.128, 3)
    chk(P + "[42] target-GC strata", float(len(dl)), 5, 0)
    chk(P + "[42]   guides behind the stratified gains (the 13,879-guide published matrix, 100 kb-block CV; the paper says published matrix)",
        float(gca[gca.arm == "baseline"].n.sum()), 13879, 0)

    # [44] imported guide-derived families, and random functions of matched difficulty (curated, random-row 5 x 5)
    lad = pd.read_csv(R / "curated_ladder.csv").set_index("arm")
    quantum = float(lad.loc["base_full", "spearman"] - lad.loc["all_kmers", "spearman"])
    fams = {"quantum tables (curated_ladder.csv)": quantum, "b_energy": float(ac.loc["b_energy", "d_spearman"]),
            "c_folding": float(ac.loc["c_folding", "d_spearman"]), "d_mechanics, 22 protospacer columns (no_information.csv)": float(ni.loc["guide_only", "d_spearman"])}
    chk(P + "[44] smallest gain among quantum, energy, folding, mechanics-guide-half", min(fams.values()), 0.011, 3)
    chk(P + "[44] largest gain among them", max(fams.values()), 0.019, 3)
    rr = pd.read_csv(R / "redescription_random.csv")
    rr = rr[rr.n_random > 0].groupby("n_random").d_spearman.mean()
    rn = pd.read_csv(R / "redescription_nonlinear.csv")
    rn = rn[rn.arm.str.startswith("random")]
    matched = rn[(rn.oof_r2 > 0.3) & (rn.oof_r2 < 0.5)]
    chk(P + "[44] random LINEAR position-weight tables, most negative mean over 3 draws (256 tables; redescription_random.csv)", float(rr.min()), -0.0015, 4)
    chk(P + "[44] random NON-LINEAR functions matched on recoverability (R2 0.34-0.48), most positive (redescription_nonlinear.csv)", float(matched.d_spearman.max()), 0.0007, 4)
    chk(P + "[44]   ... and most negative (the paper's matched-difficulty range starts at -0.0004; -0.0015 belongs to the linear tables)", float(matched.d_spearman.min()), -0.0004, 4)


def pathlib_root():
    import pathlib
    return pathlib.Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    from .standalone import run_alone
    run_alone(run, "paper")
