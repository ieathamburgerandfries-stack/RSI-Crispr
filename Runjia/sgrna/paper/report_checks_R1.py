"""Report checks R1: numbers of PROGRESS_REPORT.md lines 1-120 (the Summary, 'How to read', 'What earns a place', Finding 1) that had
no line in the audit, recomputed from a stored file or from the repository's own code. Same idiom as `paper_numbers_C`:
`chk(label, got, claimed, dec)` passes when |got - claimed| <= 0.5 * 10**(-dec); `got` is always computed, `claimed` is the value as
the corrected report prints it. Every label starts with 'report R1: ' and names the cohort and protocol of the number.

    ./run_ext.sh sgrna.paper.report_checks_R1

Cohorts: *published* = Noshay et al.'s 13,880-row matrix (6,232 columns); *curated* = crisprHAL 2's 33,567 guides (base 6,169 columns,
"genic" = the 30,203 guides inside an annotated gene). Protocols: 5 x 5 = five seeds x five random-row folds unless a label says
otherwise; "300-column budget" = `cross_validate` resolves n_features=None to config.N_FEATURES = 300, so every arm of
`portability.csv` selects 300 columns from its pool, while `curated_ladder.csv` and `handnamed_split.csv` set n_features to each arm's own
column count (no selection).

`run(chk)` is called by `verify_numbers` through `paper_numbers`, and the harness at the bottom runs the module alone.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

from .. import config
from ..common import OUT

P = "report R1: "


def _paired_seed_interval(RES, arm_file, full_file):
    """Loss of an arm against the full matrix, paired by (seed, fold); returns mean loss and the t(4) interval over the 5 seed means."""
    a = pd.read_csv(RES / arm_file)
    b = pd.read_csv(RES / full_file)
    m = b.merge(a, on=["seed", "fold"], suffixes=("_full", "_arm"))
    loss = (m.spearman_full - m.spearman_arm).groupby(m.seed).mean()          # one number per seed
    n = len(loss)
    mean, se = float(loss.mean()), float(loss.std(ddof=1) / np.sqrt(n))
    t = float(stats.t.ppf(0.975, n - 1))
    return mean, mean - t * se, mean + t * se, mean, se, n, float((loss > 0).sum())


def run(chk):
    """Run every check of this module through `chk(label, got, claimed, dec)`."""
    RES = OUT.parent                              # the fork's results/
    MAIN = config.RESULTS                         # the main project's results/ (files the fork does not track)
    ROOT = pathlib.Path(config.__file__).parent   # sgrna/

    # ------------------------------------------------------------------------------------------------------ regeneration
    print("report R1: Finding 1, regeneration of the published matrix from the 20-mer")
    from ..build import featurise
    from ..build.build_matrix import CACHE_META, CACHE_X
    from ..io_utils import load_guide_index
    if CACHE_X.exists() and CACHE_META.exists():                # never build the cache from here: this audit writes nothing
        X = np.load(CACHE_X, mmap_mode="r")
        meta = json.loads(CACHE_META.read_text())
        idx = load_guide_index().set_index(config.ID_COL).reindex(meta["ids"])
        seqs = idx["protospacer"].fillna("N" * 20).to_numpy()
        rebuilt, rb = featurise.matrix(seqs, verbose=False)
        pos_rb = {c: i for i, c in enumerate(rb)}
        col_of = {c: i for i, c in enumerate(meta["feature_names"])}
        shared = [c for c in rb if c in col_of]
        ia, ib = [pos_rb[c] for c in shared], [col_of[c] for c in shared]
        cmp_n = {500: [0, 0], len(seqs): [0, 0]}                # [values compared, values identical]
        for lo in range(0, len(seqs), 2000):
            hi = min(lo + 2000, len(seqs))
            a = np.asarray(rebuilt[lo:hi][:, ia], dtype=float)
            b = np.asarray(X[lo:hi][:, ib], dtype=float)
            both = ~(np.isnan(a) | np.isnan(b))
            same = np.isclose(a, b, atol=1e-6) & both
            cmp_n[len(seqs)][0] += int(both.sum())
            cmp_n[len(seqs)][1] += int(same.sum())
            if lo < 500:
                k = min(500, hi) - lo
                bk = both[:k]
                cmp_n[500][0] += int(bk.sum())
                cmp_n[500][1] += int(same[:k].sum())
        chk(P + "Finding 1: columns rebuilt and compared (published matrix)", float(len(shared)), 6169, 0)
        chk(P + "Finding 1: values compared for the first 500 guides, millions (verify_against_published(500))", cmp_n[500][0] / 1e6, 3.08, 2)
        chk(P + "Finding 1: ... share identical, % (first 500 guides)", 100 * cmp_n[500][1] / cmp_n[500][0], 100.0, 6)
        chk(P + "Finding 1: values compared over all 13,880 guides, millions (non-missing cells)", cmp_n[len(seqs)][0] / 1e6, 85.6, 1)
        chk(P + "Finding 1: ... share identical, % (all guides)", 100 * cmp_n[len(seqs)][1] / cmp_n[len(seqs)][0], 100.0, 6)
    gi = pd.read_csv(config.GUIDE_INDEX)
    chk(P + "Finding 1: guides located in NC_000913.2 (guide_index.csv)", float(gi["located"].sum()), 13879, 0)
    chk(P + "Finding 1: ... with an NGG PAM in the right place", float(gi["pam_is_ngg"].sum()), 13879, 0)
    chk(P + "Finding 1: ... unique in the genome", float(gi["unique_in_genome"].sum()), 13877, 0)
    vm = json.loads((config.INTERIM / "v_column_mapping.json").read_text())
    chk(P + "Finding 1: anonymous V#### columns (v_column_mapping.json)", float(vm["n_v_columns"]), 5887, 0)
    chk(P + "Finding 1: ... decoded to (position, k-mer) indicators", float(len(vm["mapping"])), 5853, 0)
    chk(P + "Finding 1: ... ambiguous", float(len(vm["ambiguous"])), 0, 0)

    # ------------------------------------------------------------------------------------------------------ the 63 hand-named
    print("report R1: Finding 1, the 63 hand-named columns (published matrix)")
    pa = pd.read_csv(RES / "portability_audit.csv")
    ann = pa.kind.str.contains("annotation")
    vbad = (pa.kind == "undecoded V column") & (~pa.portable)
    vgood = (pa.kind == "undecoded V column") & pa.portable
    chk(P + "Finding 1: hand-named columns", float(len(pa)), 63, 0)
    chk(P + "Finding 1: ... computable from the 20-mer and its PAM (63 - 2 annotation distances - 15 undecoded V)", float((~ann & ~vbad).sum()), 46, 0)
    chk(P + "Finding 1: ... not computable (2 annotation distances + 15 undecoded V)", float((ann | vbad).sum()), 17, 0)
    chk(P + "Finding 1: undecoded V columns the 364 indicators reconstruct (in-sample R2 1.000)", float(vgood.sum()), 19, 0)
    chk(P + "Finding 1: ... their lowest in-sample R2", float(pa.loc[vgood, "r2_from_20mer"].min()), 1.0, 3)
    chk(P + "Finding 1: undecoded V columns that are not reconstructed", float(vbad.sum()), 15, 0)
    chk(P + "Finding 1: ... their lowest in-sample R2", float(pa.loc[vbad, "r2_from_20mer"].min()), 0.06, 2)
    chk(P + "Finding 1: ... their highest in-sample R2", float(pa.loc[vbad, "r2_from_20mer"].max()), 0.65, 2)
    chk(P + "Finding 1: portable columns in portability_audit.csv (46 + the 2 annotation distances, the 412 = 364 + 48)",
        float(pa.portable.sum()), 48, 0)
    hs = pd.read_csv(RES / "handnamed_split.csv").set_index("arm")
    chk(P + "Finding 1: the 364 sequence indicators = columns", float(hs.loc["base_364", "n_columns"]), 364, 0)
    chk(P + "Finding 1: 364 + the 46 computable = columns", float(hs.loc["+computable_46", "n_columns"]), 410, 0)
    chk(P + "Finding 1: 364 + the 17 not computable = columns", float(hs.loc["+noncomputable_17", "n_columns"]), 381, 0)
    chk(P + "Finding 1: 364 + all 63 = columns", float(hs.loc["+all_63", "n_columns"]), 427, 0)
    b0 = hs.loc["base_364", "spearman"]
    chk(P + "Finding 1: 364 indicators, rho (published, no selection, 5 x 5)", float(b0), 0.5189, 4)
    chk(P + "Finding 1: ... sd", float(hs.loc["base_364", "spearman_sd"]), 0.0010, 4)
    chk(P + "Finding 1: + the 46 computable, rho", float(hs.loc["+computable_46", "spearman"]), 0.5254, 4)
    chk(P + "Finding 1: ... gain over the 364", float(hs.loc["+computable_46", "spearman"] - b0), 0.0065, 4)
    chk(P + "Finding 1: + the 17 not computable, rho", float(hs.loc["+noncomputable_17", "spearman"]), 0.5182, 4)
    chk(P + "Finding 1: ... gain over the 364", float(hs.loc["+noncomputable_17", "spearman"] - b0), -0.0007, 4)
    chk(P + "Finding 1: + all 63, rho", float(hs.loc["+all_63", "spearman"]), 0.5256, 4)
    chk(P + "Finding 1: ... gain over the 364", float(hs.loc["+all_63", "spearman"] - b0), 0.0066, 4)

    # ------------------------------------------------------------------------------------------------------ the ladder
    print("report R1: Finding 1, the curated ladder (3 seeds x 5 folds, no selection)")
    lad = pd.read_csv(RES / "curated_ladder.csv").set_index("arm")
    s = lad.spearman
    for arm, rho, sd in (("mono", 0.5198, 0.0021), ("mono_di", 0.5331, 0.0010), ("mono_di_tri", 0.5354, 0.0003),
                         ("all_kmers", 0.5335, 0.0013), ("base_full", 0.5449, 0.0013),
                         ("mono_di+a_flank", 0.6937, 0.0010), ("mono_di_tri+a_flank", 0.7011, 0.0009),
                         ("all_kmers+a_flank", 0.7005, 0.0004), ("base_full+a_flank", 0.7079, 0.0004)):
        chk(P + f"Finding 1 ladder: {arm}, rho", float(s[arm]), rho, 4)
        chk(P + f"Finding 1 ladder: {arm}, sd", float(lad.spearman_sd[arm]), sd, 4)
    for arm, v in (("mono", -0.0251), ("mono_di", -0.0118), ("mono_di_tri", -0.0096), ("all_kmers", -0.0114)):
        chk(P + f"Finding 1 ladder: {arm} against the full base", float(s[arm] - s["base_full"]), v, 4)
    for arm, v in (("mono_di+a_flank", -0.0143), ("mono_di_tri+a_flank", -0.0068), ("all_kmers+a_flank", -0.0075)):
        chk(P + f"Finding 1 ladder: {arm} against the full base with family A", float(s[arm] - s["base_full+a_flank"]), v, 4)
    chk(P + "Finding 1 ladder: trimers over dinucleotides", float(s["mono_di_tri"] - s["mono_di"]), 0.0023, 4)
    chk(P + "Finding 1 ladder: tetramers over trimers (differences of the stored values)", float(s["all_kmers"] - s["mono_di_tri"]), -0.0018, 4)
    chk(P + "Finding 1 ladder: ... with family A", float(s["all_kmers+a_flank"] - s["mono_di_tri+a_flank"]), -0.0007, 4)
    chk(P + "Finding 1 ladder: the 316 quantum columns over all k-mers", float(s["base_full"] - s["all_kmers"]), 0.0114, 4)
    chk(P + "Finding 1 ladder: ... with family A", float(s["base_full+a_flank"] - s["all_kmers+a_flank"]), 0.0075, 4)
    chk(P + "Finding 1 ladder: 74 mononucleotide columns, share of the full curated base, %", 100 * float(s["mono"] / s["base_full"]), 95.4, 1)
    p364 = pd.read_csv(RES / "portability_364.csv").set_index("arm").spearman
    pf = pd.read_csv(RES / "portability.csv").set_index("arm")
    chk(P + "Finding 1: 74 mononucleotide columns, published matrix (300-column budget, 5 x 5), rho", float(p364["mono_74"]), 0.5081, 4)
    chk(P + "Finding 1: all 6,232 columns, published matrix (300-column budget, 5 x 5), rho", float(pf.loc["published_full", "spearman"]), 0.5272, 4)
    chk(P + "Finding 1: ... share recovered by the 74, %", 100 * float(p364["mono_74"] / pf.loc["published_full", "spearman"]), 96.4, 1)
    run_src = (ROOT / "harness" / "run_ablation.py").read_text()
    port_src = (ROOT / "compare" / "portability.py").read_text()
    chk(P + "Finding 1: cross_validate resolves n_features=None to the 300-column budget", float("n_features = n_features or config.N_FEATURES" in run_src), 1, 0)
    chk(P + "Finding 1: ... and portability.py passes n_features=None (so its arms select 300 columns)", float("n_features=None" in port_src), 1, 0)
    chk(P + "Finding 1: config.N_FEATURES", float(config.N_FEATURES), 300, 0)
    k = pd.read_csv(RES / "kmer_order_paired.csv").pivot_table(index=["seed", "fold"], columns="arm", values="spearman")
    chk(P + "Finding 1: published matrix, trimers over dinucleotides (kmer_order_paired.csv, 15 folds)",
        float((k["mono+di+tri"] - k["mono+di"]).mean()), 0.0035, 4)
    chk(P + "Finding 1: ... tetramers on top", float((k["mono+di+tri+tetra"] - k["mono+di+tri"]).mean()), -0.0028, 4)

    # ------------------------------------------------------------------------------------------------------ reduced sets
    print("report R1: Finding 1, reduced sets (published matrix, 300-column budget, paired t over 5 seeds)")
    m427 = _paired_seed_interval(MAIN, "folds_port_reduced_427.csv", "folds_port_published_full.csv")
    chk(P + "Finding 1: 427 columns, loss against all 6,232", m427[0], 0.0023, 4)
    chk(P + "Finding 1: ... interval, lower end", m427[1], -0.0008, 4)
    chk(P + "Finding 1: ... interval, upper end", m427[2], 0.0054, 4)
    chk(P + "Finding 1: ... one-sided p against a 0.005 margin (t, 4 df)", float(stats.t.cdf((m427[3] - 0.005) / m427[4], m427[5] - 1)), 0.035, 3)
    chk(P + "Finding 1: ... rho of the 427 (portability.csv)", float(pf.loc["reduced_427", "spearman"]), 0.5249, 4)
    m412 = _paired_seed_interval(MAIN, "folds_port_portable.csv", "folds_port_published_full.csv")
    chk(P + "Finding 1: 412 portable columns, loss against all 6,232", m412[0], 0.00003, 5)
    chk(P + "Finding 1: ... interval, lower end", m412[1], -0.0029, 4)
    chk(P + "Finding 1: ... interval, upper end", m412[2], 0.0030, 4)
    m364 = _paired_seed_interval(MAIN, "folds_port_mono_di_364.csv", "folds_port_published_full.csv")
    chk(P + "Finding 1: 364 indicators only, loss against all 6,232", m364[0], 0.0083, 4)
    chk(P + "Finding 1: ... interval, lower end", m364[1], 0.0057, 4)
    chk(P + "Finding 1: ... interval, upper end", m364[2], 0.0109, 4)
    chk(P + "Finding 1: ... seeds (of 5) in which the 364 lose", m364[6], 5, 0)

    # ------------------------------------------------------------------------------------------------------ re-description
    print("report R1: Finding 1, re-description (curated base)")
    rr = pd.read_csv(RES / "redescription_random.csv")
    rr = rr[rr.n_random > 0].groupby("n_random").d_spearman.mean()
    for n_, v in ((4, -0.0004), (16, -0.0004), (64, -0.0002), (256, -0.0015)):
        chk(P + f"Finding 1: random linear position-weight tables, {n_} added, mean of three draws", float(rr[n_]), v, 4)
    nl = pd.read_csv(RES / "redescription_nonlinear.csv")
    o2 = nl[nl.order == 2]
    o38 = nl[nl.order >= 3]
    chk(P + "Finding 1: real eng.dg.guide_selffold, out-of-fold R2 from the guide", float(nl.loc[nl.arm.str.startswith("real"), "oof_r2"].iloc[0]), 0.316, 3)
    chk(P + "Finding 1: random non-linear order 2, lowest R2", float(o2.oof_r2.min()), 0.34, 2)
    chk(P + "Finding 1: ... highest R2", float(o2.oof_r2.max()), 0.48, 2)
    chk(P + "Finding 1: ... most negative gain (3 draws)", float(o2.d_spearman.min()), -0.0004, 4)
    chk(P + "Finding 1: ... most positive gain", float(o2.d_spearman.max()), 0.0007, 4)
    chk(P + "Finding 1: orders 3 to 8, highest R2", float(o38.oof_r2.max()), 0.2, 1)
    chk(P + "Finding 1: ... most negative gain (3 draws each)", float(o38.d_spearman.min()), -0.0010, 4)
    chk(P + "Finding 1: ... most positive gain", float(o38.d_spearman.max()), 0.0016, 4)
    ni = pd.read_csv(RES / "no_information.csv").set_index("arm")
    chk(P + "Finding 1: d_mechanics, 22 protospacer columns over the curated base (5 x 5)", float(ni.loc["guide_only", "d_spearman"]), 0.0110, 4)
    chk(P + "Finding 1: ... 8 flank columns", float(ni.loc["flank_only", "d_spearman"]), 0.0973, 4)
    chk(P + "Finding 1: ... all 30 columns", float(ni.loc["full", "d_spearman"]), 0.0991, 4)
    rc = pd.read_csv(RES / "d_mechanics_reconstructibility.csv").oof_r2_from_mono_di
    chk(P + "Finding 1: d_mechanics columns the base reconstructs at out-of-fold R2 1.0000 (>= 0.99995)", float((rc >= 0.99995).sum()), 9, 0)
    chk(P + "Finding 1: d_mechanics columns at out-of-fold R2 near 0 (< 0.05): the eight flank columns", float((rc < 0.05).sum()), 8, 0)
    chk(P + "Finding 1: ... their highest out-of-fold R2", float(rc[rc < 0.05].max()), 0.04, 2)
    sc = pd.read_csv(RES / "selector_control.csv").set_index("arm").spearman
    chk(P + "Finding 1: selector control, curated base, 300 selected (seed 41)", float(sc["base, select 300"]), 0.5443, 4)
    chk(P + "Finding 1: ... every column", float(sc["base, ALL columns"]), 0.5439, 4)
    chk(P + "Finding 1: d_mechanics over the base, every column (seed 41)", float(sc["base + mech, ALL columns"] - sc["base, ALL columns"]), 0.0993, 4)
    chk(P + "Finding 1: ... selecting 300", float(sc["base + mech, select 300"] - sc["base, select 300"]), 0.0988, 4)
    ac, ag = pd.read_csv(RES / "ablation_curated.csv").set_index("family"), pd.read_csv(RES / "ablation_curated_gene.csv").set_index("family")
    chk(P + "Finding 1: curated base, 300 selected, 5 seeds (ablation_curated.csv)", float(ac.loc["baseline", "spearman"]), 0.5445, 4)
    chk(P + "Finding 1: curated base + family A, 300 selected, 5 seeds", float(ac.loc["a_flank", "spearman"]), 0.7084, 4)
    chk(P + "Finding 1: family A over the curated base, random-row 5 x 5", float(ac.loc["a_flank", "d_spearman"]), 0.164, 3)
    chk(P + "Finding 1: ... gene-grouped (ablation_curated_gene.csv, 30,203 genic guides)", float(ag.loc["a_flank", "d_spearman"]), 0.157, 3)
    chk(P + "Finding 1: b_energy over the curated base (5 x 5)", float(ac.loc["b_energy", "d_spearman"]), 0.019, 3)
    chk(P + "Finding 1: c_folding over the curated base (5 x 5)", float(ac.loc["c_folding", "d_spearman"]), 0.014, 3)
    am = pd.read_csv(RES / "ablation_main.csv").set_index("experiment")
    n12 = pd.read_csv(MAIN / "ablation_nfeat1200.csv").set_index("experiment")
    chk(P + "Finding 1: published matrix, family A gain at the 300-column budget (ablation_main.csv)", float(am.loc["a_flank", "d_spearman"]), 0.080, 3)
    chk(P + "Finding 1: ... at 1,200 columns (main-project ablation_nfeat1200.csv, 3 seeds)", float(n12.loc["a_flank", "d_spearman"]), 0.083, 3)
    cj = pd.read_csv(RES / "curated_joint.csv").set_index("arm")
    chk(P + "Finding 1: d_supercoiling + e_nucleoid + g_transcription jointly over the curated base (5 x 5)",
        float(cj.loc["d_supercoiling+e_nucleoid+g_transcription", "d_spearman"]), 0.145, 3)

    mc = pd.read_csv(RES / "model_comparison.csv").groupby("model").spearman.mean()
    chk(P + "Finding 1: model classes on the published matrix (3 seeds x 5 folds)", float(len(mc)), 16, 0)
    chk(P + "Finding 1: ... span of their mean rho", float(mc.max() - mc.min()), 0.116, 3)
    cm = pd.read_csv(RES / "curated_models.csv").set_index("model").spearman
    chk(P + "Finding 1: eight model classes on the curated cohort with family A (one seed), span", float(cm.max() - cm.min()), 0.1076, 4)
    hd = pd.read_csv(RES / "representation_headroom.csv")
    hd = hd[hd.group == "given to the model"].assign(a=lambda d: d.rho_with_residual.abs()).groupby("arm").a.max()
    chk(P + "Finding 1 / Summary 4: strongest |rho| of a column with the champion's residuals, all 6,232 (published, seed 41)", float(hd["published_full"]), 0.028, 3)
    chk(P + "Finding 1 / Summary 4: ... with family A also given", float(hd["published_full+a_flank"]), 0.031, 3)
    chk(P + "Finding 1 / Summary 4: ... 427 columns", float(hd["subset_427"]), 0.022, 3)
    chk(P + "Finding 1 / Summary 4: ... 427 columns with family A given", float(hd["subset_427+a_flank"]), 0.028, 3)
    tx = pd.read_csv(RES / "redescription_taxonomy.csv").set_index("arm")
    for arm, n_, v in (("base + determined by the guide", 18, 0.008), ("base + partly determined", 16, 0.020), ("base + outside the guide", 602, 0.201)):
        chk(P + f"Finding 1: family columns, {arm}: columns", float(tx.loc[arm, "n_cols"]), n_, 0)
        chk(P + f"Finding 1: ... block gain over the curated base (1 seed x 5 folds)", float(tx.loc[arm, "d_spearman"]), v, 3)

    # ------------------------------------------------------------------------------------------------------ Summary 2, 4, 7
    print("report R1: Summary claims 2, 4, 6, 7")
    for nm, a, b in (("eSpCas9", "transfer_eSpCas9", "transfer_WT-SpCas9"), ("C. rodentium", "transfer_TevSpCas9", "transfer_WT-SpCas9")):
        ta = set(json.loads((config.INTERIM / a / "meta.json").read_text())["protospacers"])
        tb = set(json.loads((config.INTERIM / b / "meta.json").read_text())["protospacers"])
        if nm == "eSpCas9":
            chk(P + "Summary 2: WT-SpCas9 guides that are also in the eSpCas9 screen", float(len(ta & tb)), 33567, 0)
            chk(P + "Summary 2: ... share of the eSpCas9 rows, %", 100 * len(ta & tb) / len(ta), 56, 0)
        else:
            chk(P + "Summary 2: 20-mers shared by the C. rodentium and E. coli WT sets", float(len(ta & tb)), 1, 0)
    sh = pd.read_csv(RES / "sealed_holdout.csv")
    pv = sh.pivot(index="seal_seed", columns="arm", values="gap")
    ad = pv["frozen"] - pv["base_only"]                               # the adaptive component, one value per sealing (gap = pool CV - sealed)
    se = float(ad.std(ddof=1) / np.sqrt(len(ad)))
    chk(P + "Summary 7: adaptive component, frozen arm, mean over 15 sealings", float(ad.mean()), 0.0087, 4)
    chk(P + "Summary 7: ... sd across sealings", float(ad.std(ddof=1)), 0.0131, 4)
    chk(P + "Summary 7: ... mean - 1.96 se (the interval the report prints)", float(ad.mean() - 1.96 * se), 0.0020, 4)
    chk(P + "Summary 7: ... mean + 1.96 se", float(ad.mean() + 1.96 * se), 0.0153, 4)
    tq = float(stats.t.ppf(0.975, len(ad) - 1))
    chk(P + "Summary 7: ... t interval, lower end (the interval is the normal one, not this)", float(ad.mean() - tq * se), 0.0014, 4)
    chk(P + "Summary 7: ... t interval, upper end", float(ad.mean() + tq * se), 0.0159, 4)
    cc = pd.read_csv(RES / "cost_panel.csv").set_index("model")
    chk(P + "Summary 6: CPU-seconds to fit per fold, crisprHAL 2 published over SLICER as published (crisprHAL's 10,000 is hand-entered)",
        float(cc.loc["crisprHAL 2, published", "cpu_seconds"] / cc.loc["SLICER, as published", "cpu_seconds"]), 152, 0)

    fa = pd.read_csv(OUT / "interp_compare_faithfulness.csv")
    chk(P + "Summary 4: test guides in the sensitivity-n test (crisprHAL 2, SpCas9 split)", float(fa[fa.model == "crisprHAL 2"].rows.max()), 100, 0)
    vd = pd.read_csv(OUT / "interp_compare_validity_diff.csv").set_index("model")
    chk(P + "Summary 4: substitution map against the model-free map, SLICER-104 minus crisprHAL 2", float(vd.loc["SLICER-104", "spearman_minus_crisprhal"]), 0.070, 3)
    chk(P + "Summary 4: ... lower end", float(vd.loc["SLICER-104", "lo"]), 0.033, 3)
    chk(P + "Summary 4: ... upper end", float(vd.loc["SLICER-104", "hi"]), 0.122, 3)
    chk(P + "Summary 4: ... shipped SLICER minus crisprHAL 2", float(vd.loc["SLICER (shipped)", "spearman_minus_crisprhal"]), 0.027, 3)
    chk(P + "Summary 4: ... lower end (interval contains zero)", float(vd.loc["SLICER (shipped)", "lo"]), -0.010, 3)
    chk(P + "Summary 4: ... upper end", float(vd.loc["SLICER (shipped)", "hi"]), 0.071, 3)
    st = pd.read_csv(OUT / "interp_compare_stability.csv")
    sub = st[st.unit == "substitution map (80)"].set_index(["model", "comparison"])["mean"]
    chk(P + "Summary 4: substitution map across refit seeds, SLICER (shipped)", float(sub[("SLICER (shipped)", "refit seeds")]), 0.998, 3)
    chk(P + "Summary 4: ... crisprHAL 2", float(sub[("crisprHAL 2", "refit seeds")]), 0.989, 3)
    chk(P + "Summary 4: substitution map across random halves, SLICER (shipped)", float(sub[("SLICER (shipped)", "random halves")]), 0.972, 3)
    chk(P + "Summary 4: ... crisprHAL 2", float(sub[("crisprHAL 2", "random halves")]), 0.981, 3)
    rd = pd.read_csv(OUT / "interp_compare_readable.csv")
    r5 = rd[(rd.population == "five curated folds (out-of-fold)") & (rd.surrogate == "boosted trees (champion settings)") & (rd.model == "SLICER")].iloc[0]
    chk(P + "Summary 4: 104-column surrogate, fidelity, SLICER minus crisprHAL 2 (30,203 guides)", float(r5.fidelity_minus_hal), 0.041, 3)
    chk(P + "Summary 4: ... share of the model's accuracy the surrogate keeps, SLICER minus crisprHAL 2", float(r5.retained_minus_hal), -0.015, 3)

    print("report R1: What earns a place")
    gcc = pd.read_csv(RES / "gc_strata_corr_curated.csv").set_index("stratum")
    chk(P + "What earns a place: target GC against the label, curated cohort (n = 33,567)", float(gcc.loc["all", "rho_target"]), -0.15, 2)


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
        print(f"  {'ok ' if ok else 'BAD'} {label:120s} computed {got:+.6f}  report {claimed:+.{max(dec, 0)}f}")

    run(chk)
    print(f"\n{n} checks, {len(bad)} mismatches")
    for lab, got, claimed in bad:
        print(f"  BAD {lab}: computed {got:+.6f}, report {claimed}")
    raise SystemExit(1 if bad else 0)
