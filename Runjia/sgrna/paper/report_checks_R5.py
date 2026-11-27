"""Checks R5: headline numbers of PROGRESS_REPORT.md lines 576-853 (Part 2 contributions, Part 3 limits, Part 4 freeze) that had no audit line.

Same idiom as `paper_numbers_C.py`: `chk(label, got, claimed, dec)` passes when |got - claimed| <= 0.5 * 10**(-dec). `got` is computed from a stored
file or from the stored cohort; `claimed` is the value as the (corrected) report writes it.

Protocols the numbers sit on (a value can occur for a different quantity, so this is part of the check):

    sealed genes   `sealed_holdout.csv`: 15 sealings of whole genes holding 20% of the 33,567 curated guides, random-row 5-fold CV inside the 80% pool,
                   sealed fits on 80% of the pool, frozen XGBoost (300 of 6,517 columns) minus base-only XGBoost (6,169 columns, no selection).
    family A       +0.1638 curated random-row 5 x 5 (`ablation_curated.csv`), +0.1567 gene-grouped 5 x 5 on 30,203 genic guides
                   (`ablation_curated_gene.csv`); the target GC of the report is the GC of the 20-nt protospacer on either cohort.
    human          `human_shape_ablation_twin.csv`: 3 seeds x 5 twin-grouped folds, 16,490 located guides.
    Jacky's files  RSI-Crispr/Jacky/... (shared repository, next to the fork); skipped when absent.

    ./run_ext.sh sgrna.paper.report_checks_R5
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from ..common import OUT
from sgrna import config

P = "report R5: "


def run(chk):
    RES = OUT.parent                      # the fork's results/
    MAIN = config.RESULTS                 # the main project's results/

    def f(name):                          # the fork's copy when it has one, else the main project's
        return (RES / name) if (RES / name).exists() else (MAIN / name)

    # ----------------------------------------------------------------------------------------------- sealed genes (Part 4)
    print("report R5: sealed genes")
    d = pd.read_csv(f("sealed_holdout.csv"))
    base = d[d.arm == "base_only"].set_index("seal_seed")
    fz = d[d.arm == "frozen"].set_index("seal_seed")
    dt = d[d.arm == "dev_tuned"].set_index("seal_seed")
    n = len(base)
    chk(P + "sealed draws", float(n), 15, 0)
    ad = fz.gap - base.gap
    m, sd = float(ad.mean()), float(ad.std(ddof=1))
    se = sd / np.sqrt(n)
    tlo, thi = stats.t.interval(0.95, n - 1, loc=m, scale=se)
    chk(P + "adaptive component, frozen: mean", m, 0.0087, 4)
    chk(P + "adaptive component, frozen: sd across draws", sd, 0.0131, 4)
    chk(P + "adaptive component, frozen: mean - 1.96 se (the table's interval)", m - 1.96 * se, 0.0020, 4)
    chk(P + "adaptive component, frozen: mean + 1.96 se (the table's interval)", m + 1.96 * se, 0.0153, 4)
    chk(P + "adaptive component, frozen: t-interval lower", float(tlo), 0.0014, 4)
    chk(P + "adaptive component, frozen: t-interval upper", float(thi), 0.0159, 4)
    chk(P + "adaptive component, frozen: draws above zero", float((ad > 0).sum()), 12, 0)
    chk(P + "adaptive component, frozen: t", m / se, 2.56, 2)
    chk(P + "adaptive component, frozen: t-test p", float(2 * stats.t.sf(abs(m / se), n - 1)), 0.022, 3)
    chk(P + "adaptive component, frozen: Wilcoxon p", float(stats.wilcoxon(ad).pvalue), 0.015, 3)
    chk(P + "adaptive component, frozen: sign test p", float(stats.binomtest(int((ad > 0).sum()), n, 0.5).pvalue), 0.035, 3)
    loo = [float(ad.drop(ad.index[i]).mean()) for i in range(n)]
    chk(P + "adaptive component, frozen: leave-one-out minimum", min(loo), 0.0065, 4)
    chk(P + "adaptive component, frozen: leave-one-out maximum", max(loo), 0.0106, 4)
    ad2 = dt.gap - base.gap
    m2, sd2 = float(ad2.mean()), float(ad2.std(ddof=1))
    chk(P + "adaptive component, dev-tuned: mean", m2, 0.0082, 4)
    chk(P + "adaptive component, dev-tuned: sd", sd2, 0.0140, 4)
    chk(P + "adaptive component, dev-tuned: mean - 1.96 se", m2 - 1.96 * sd2 / np.sqrt(n), 0.0011, 4)
    chk(P + "adaptive component, dev-tuned: mean + 1.96 se", m2 + 1.96 * sd2 / np.sqrt(n), 0.0153, 4)
    rg = base.gap
    chk(P + "reference gap: mean", float(rg.mean()), -0.0121, 4)
    chk(P + "reference gap: sd", float(rg.std(ddof=1)), 0.0148, 4)
    chk(P + "reference gap: lower (mean - 1.96 se)", float(rg.mean() - 1.96 * rg.std(ddof=1) / np.sqrt(n)), -0.0196, 4)
    chk(P + "reference gap: upper", float(rg.mean() + 1.96 * rg.std(ddof=1) / np.sqrt(n)), -0.0047, 4)
    fg = fz.gap
    chk(P + "frozen gap: mean", float(fg.mean()), -0.0035, 4)
    chk(P + "frozen gap: sd", float(fg.std(ddof=1)), 0.0077, 4)
    chk(P + "frozen gap: lower", float(fg.mean() - 1.96 * fg.std(ddof=1) / np.sqrt(n)), -0.0074, 4)
    chk(P + "frozen gap: upper", float(fg.mean() + 1.96 * fg.std(ddof=1) / np.sqrt(n)), 0.0004, 4)
    chk(P + "frozen arm, pool CV", float(fz.cv_in_pool.mean()), 0.7043, 4)
    chk(P + "frozen arm, sealed genes", float(fz.sealed.mean()), 0.7077, 4)
    chk(P + "base-only arm, pool CV", float(base.cv_in_pool.mean()), 0.5396, 4)
    chk(P + "base-only arm, sealed genes", float(base.sealed.mean()), 0.5517, 4)
    gp, gs = float((fz.cv_in_pool - base.cv_in_pool).mean()), float((fz.sealed - base.sealed).mean())
    chk(P + "family-A gain, pool CV (random-row)", gp, 0.1647, 4)
    chk(P + "family-A gain, sealed genes", gs, 0.1560, 4)
    chk(P + "the drop in family A's gain equals the adaptive component (algebra)", gp - gs, m, 6)
    # share of guides and genes sealed in each draw (the same routine as sealed_holdout.py, on the stored protospacers)
    from .hooks import _seal
    from .hooks import load_curated_index
    meta = json.loads((config.INTERIM / "expanded_meta.json").read_text())
    protos = np.asarray(meta["protospacers"])
    idx = load_curated_index(verbose=False).set_index(config.ID_COL)
    gene = idx["gene_name"].reindex(protos)
    gene = np.array([g if isinstance(g, str) and g else "" for g in gene], dtype=object)
    genes = {g for g in gene if g}
    share = []
    for s in sorted(base.index):
        sealed = _seal(protos, seed=int(s), verbose=False)
        share.append(len({g for g, mk in zip(gene, sealed) if mk and g}) / len(genes))
    chk(P + "genes in the cohort", float(len(genes)), 4013, 0)
    chk(P + "guides outside every gene (stay in the pool)", float((gene == "").sum()), 3364, 0)
    chk(P + "share of genes sealed, smallest of 15 draws (%)", 100 * min(share), 21.5, 1)
    chk(P + "share of genes sealed, largest of 15 draws (%)", 100 * max(share), 23.2, 1)
    chk(P + "share of guides sealed, mean over draws (%)", float(100 * (d.n_sealed / (d.n_sealed + d.n_dev)).mean()), 20.0, 1)
    ac = pd.read_csv(f("ablation_curated.csv")).set_index("family")
    ag = pd.read_csv(f("ablation_curated_gene.csv")).set_index("family")
    chk(P + "champion family-A gain, curated random-row 5 x 5", float(ac.loc["a_flank", "d_spearman"]), 0.1638, 4)
    chk(P + "champion family-A gain, curated gene-grouped 5 x 5 (30,203 genic guides)", float(ag.loc["a_flank", "d_spearman"]), 0.1567, 4)
    chk(P + "dev-tuned arm: sealed gain differs from frozen's by less than 0.001 in component", abs(m2 - m), 0.0005, 3)

    # ----------------------------------------------------------------------------------------------- sealed genes, split by the pool-CV scheme
    print("report R5: sealed genes, whole-gene pool CV (sealed_holdout_gene.py, 8 of the 15 sealings)")
    gd = pd.read_csv(OUT / "sealed_holdout_gene.csv")
    chk(P + "gene-grouped pool CV: rows (8 sealings x 3 arms)", float(len(gd)), 24, 0)
    w = gd.pivot(index="seal_seed", columns="arm", values=["cv_gene", "cv_random", "sealed"])
    chk(P + "gene-grouped pool CV: sealings with all three arms", float(len(w.dropna())), 8, 0)
    ref = d.set_index(["arm", "seal_seed"])
    same = max(abs(float(ref.loc[(r.arm, r.seal_seed), "cv_in_pool"]) - r.cv_random) + abs(float(ref.loc[(r.arm, r.seal_seed), "sealed"]) - r.sealed)
               for r in gd.itertuples())
    chk(P + "gene-grouped pool CV: random-row CV and sealed scores are copies of sealed_holdout.csv (largest difference, 1e-9)", same / 1e-9, 0, 0)

    def comp(arm, kind):
        return (w[("cv_" + kind, arm)] - w[("sealed", arm)]) - (w[("cv_" + kind, "base_only")] - w[("sealed", "base_only")])

    def band(x):
        mm, ss = float(x.mean()), float(x.std(ddof=1)) / np.sqrt(len(x))
        return mm, mm - 1.96 * ss, mm + 1.96 * ss
    for arm, lab in (("frozen", "frozen"), ("dev_tuned", "dev-tuned")):
        r_, g_ = comp(arm, "random"), comp(arm, "gene")
        chk(P + f"8 sealings, {lab}: difference of differences, random-row pool CV", float(r_.mean()), {"frozen": 0.0088, "dev_tuned": 0.0079}[arm], 4)
        chk(P + f"8 sealings, {lab}: difference of differences, whole-gene pool CV", float(g_.mean()), {"frozen": 0.0081, "dev_tuned": 0.0060}[arm], 4)
        m_, lo_, hi_ = band(r_ - g_)
        chk(P + f"8 sealings, {lab}: cost of random-row folds (random minus whole-gene)", m_, {"frozen": 0.0007, "dev_tuned": 0.0019}[arm], 4)
        chk(P + f"8 sealings, {lab}: cost, mean - 1.96 se", lo_, {"frozen": -0.0011, "dev_tuned": 0.0006}[arm], 4)
        chk(P + f"8 sealings, {lab}: cost, mean + 1.96 se", hi_, {"frozen": 0.0024, "dev_tuned": 0.0032}[arm], 4)
    m_, lo_, hi_ = band(comp("frozen", "gene"))
    chk(P + "8 sealings, frozen: whole-gene component, mean - 1.96 se (interval includes zero)", lo_, -0.0037, 4)
    chk(P + "8 sealings, frozen: whole-gene component, mean + 1.96 se", hi_, 0.0199, 4)
    for k, lab, claimed in (("cv_random", "pool CV, random-row", 0.1648), ("cv_gene", "pool CV, whole genes", 0.1641), ("sealed", "sealed genes", 0.1560)):
        chk(P + f"8 sealings: family-A gain, {lab}", float((w[(k, "frozen")] - w[(k, "base_only")]).mean()), claimed, 4)

    # ----------------------------------------------------------------------------------------------- sealed organism
    so = pd.read_csv(OUT / "sealed_organism.csv").set_index("arm")
    chk(P + "sealed organism: tuned minus champion, difference of the two mean-of-seed rhos", float(so.loc["tuned XGBoost (d099x)", "rho_seed_mean"] - so.loc["champion", "rho_seed_mean"]), 0.0174, 4)
    tm = json.loads((config.INTERIM / "transfer_TevSpCas9" / "meta.json").read_text())
    chk(P + "sealed organism: 100 kb blocks of the C. rodentium screen", float(len(np.unique((np.asarray(tm["left"], float) // 100_000).astype(int)))), 16, 0)

    # ----------------------------------------------------------------------------------------------- Part 2
    print("report R5: Part 2")
    from scipy.stats import spearmanr
    y = np.asarray(meta["y"], float)
    gc = np.array([(p.upper().count("G") + p.upper().count("C")) / len(p) for p in protos])
    chk(P + "GC of the 20-nt protospacer against the label, curated cohort", float(spearmanr(gc, y).statistic), -0.147, 3)
    gcp = pd.read_csv(f("gc_strata_corr.csv")).set_index("stratum")
    cu = pd.read_csv(f("gc_strata_corr_curated.csv")).set_index("stratum")
    chk(P + "the same variable on the published matrix", float(gcp.loc["all", "rho_target"]), -0.201, 3)
    chk(P + "target GC against flank GC, curated", float(cu.loc["all", "rho_flank_vs_target_gc"]), 0.245, 3)
    chk(P + "target GC against flank GC, published matrix (the +0.176 the report once paired with it)", float(gcp.loc["all", "rho_flank_vs_target_gc"]), 0.176, 3)
    tc = pd.read_csv(f("recoverability_trend_columns.csv"))
    top = tc.loc[tc.abs_rho.idxmax()]
    chk(P + "strongest of the 636 family columns is eng.dg.hybridisation_weighted (1 = yes)", 1.0 if top["column"] == "eng.dg.hybridisation_weighted" else 0.0, 1, 0)
    chk(P + "  its |rho| with the label", float(top.abs_rho), 0.184, 3)
    chk(P + "family columns in the file", float(len(tc)), 636, 0)
    fm = tc.groupby("family").abs_rho.max()
    chk(P + "families whose strongest column has |rho| in [0.12, 0.18]", float(((fm >= 0.115) & (fm < 0.185)).sum()), 9, 0)
    chk(P + "strongest a_flank column |rho| (eng.flank.comp.up250.gc)", float(fm["a_flank"]), 0.161, 3)
    chk(P + "b_energy family increment, curated random-row", float(ac.loc["b_energy", "d_spearman"]), 0.019, 3)
    chk(P + "h_offtarget family increment, curated random-row", float(ac.loc["h_offtarget", "d_spearman"]), 0.0011, 4)
    chk(P + "e_nucleoid family increment, curated random-row", float(ac.loc["e_nucleoid", "d_spearman"]), 0.072, 3)
    cj = pd.read_csv(f("curated_joint.csv")).set_index("arm")
    chk(P + "the ten other families add beyond family A (all eleven minus A)", float(cj.loc["all_eleven", "d_spearman"] - ac.loc["a_flank", "d_spearman"]), 0.039, 3)
    rf = pd.read_csv(f("redescription_folds.csv")).set_index("arm")
    chk(P + "recomputed spacer MFE increment", float(rf.loc["spacer MFE, recomputed", "d_spearman"]), 0.0119, 4)
    chk(P + "eng.dg.guide_selffold increment", float(rf.loc["real eng.dg.guide_selffold", "d_spearman"]), 0.0117, 4)
    sh = rf[rf.kind == "shuffled fold"].d_spearman
    chk(P + "shuffled-position MFE, smallest of three permutations", float(sh.min()), -0.0007, 4)
    chk(P + "shuffled-position MFE, largest of three permutations", float(sh.max()), 0.0002, 4)
    nl = pd.read_csv(f("redescription_nonlinear.csv"))
    o2 = nl[nl.order == 2]
    chk(P + "random non-linear order 2: smallest increment", float(o2.d_spearman.min()), -0.0004, 4)
    chk(P + "random non-linear order 2: largest increment", float(o2.d_spearman.max()), 0.0007, 4)
    chk(P + "random non-linear order 2: smallest out-of-fold R2", float(o2.oof_r2.min()), 0.34, 2)
    chk(P + "random non-linear order 2: largest out-of-fold R2", float(o2.oof_r2.max()), 0.48, 2)
    hi = nl[nl.order >= 3]
    chk(P + "random non-linear orders 3-8: smallest increment", float(hi.d_spearman.min()), -0.0010, 4)
    chk(P + "random non-linear orders 3-8: largest increment", float(hi.d_spearman.max()), 0.0016, 4)
    chk(P + "random non-linear orders 3-8: largest out-of-fold R2", float(hi.oof_r2.max()), 0.20, 2)

    # supercoiling dissection (published matrix, 100 kb blocks)
    ds = pd.read_csv(f("dissection_supercoiling.csv"))
    ds = ds[ds.grouping == "bin100k"].set_index("experiment")
    fam = float(ds.loc["d_supercoiling", "d_spearman"])
    chk(P + "untagged control alone as a share of the ChIP track alone (%)", 100 * float(ds.loc["z_control_only", "d_spearman"] / ds.loc["z_signal_only", "d_spearman"]), 78, 0)
    chk(P + "raw read density alone as a share of the family (%)", 100 * float(ds.loc["z_abundance_all", "d_spearman"]) / fam, 98, 0)
    chk(P + "four distance-to-origin columns as a share of the family (%)", 100 * float(ds.loc["z_dosage", "d_spearman"]) / fam, 25, 0)
    chk(P + "untagged control alone", float(ds.loc["z_control_only", "d_spearman"]), 0.0087, 4)
    # transcription family
    tx = pd.read_csv(f("txn_interactions.csv")).set_index("arm")
    chk(P + "family G, all 19 columns (curated random-row)", float(tx.loc["both", "d_spearman"]), 0.058, 3)
    chk(P + "family G: the three sets alone sum to more than the family (0.0393 + 0.0213 + 0.00004)",
        float(tx.loc["no_expression", "d_spearman"] + tx.loc["expression_only", "d_spearman"] + tx.loc["essential_only", "d_spearman"]), 0.061, 3)
    # human twin-grouped flank increment: folds
    ht = pd.read_csv(OUT / "human_shape_ablation_twin.csv").set_index("arm")
    a = np.array([float(x) for x in ht.loc["h_base", "fold_rho"].split(";")])
    b = np.array([float(x) for x in ht.loc["h_base+A", "fold_rho"].split(";")])
    chk(P + "human twin-grouped: folds where family A is ahead", float((b > a).sum()), 15, 0)
    chk(P + "human twin-grouped: fold-level sd of the paired increment", float((b - a).std(ddof=1)), 0.0072, 4)
    # eukaryote map pairs
    ep = pd.read_csv(OUT / "eukaryote_map_pairs.csv")
    ep = ep[~ep.same_library.astype(bool)].copy()
    ep["kind"] = [" - ".join(sorted([x, y_])) for x, y_ in zip(ep.species_a, ep.species_b)]
    em = ep[(ep.kind == "E. coli - mouse") | (ep.kind == "E. coli - zebrafish")]
    chk(P + "E. coli against mouse and zebrafish datasets: pairs", float(len(em)), 3, 0)
    chk(P + "E. coli against mouse and zebrafish datasets: smallest rho", float(em.rho.min()), -0.21, 2)
    chk(P + "E. coli against mouse and zebrafish datasets: largest rho", float(em.rho.max()), 0.08, 2)
    chk(P + "human - mouse: median rho between the 80-cell maps", float(ep[ep.kind == "human - mouse"].rho.median()), 0.44, 2)
    chk(P + "E. coli - human: median rho between the 80-cell maps", float(ep[ep.kind == "E. coli - human"].rho.median()), -0.14, 2)
    # strongest base column on the curated cohort (one column read from the matrix)
    names = meta["feature_names"]
    j = names.index("p20basepair.Hbond.energyraw")
    Xm = np.load(config.INTERIM / "expanded_X.npy", mmap_mode="r")
    chk(P + "strongest base column p20basepair.Hbond.energyraw, |rho| with the label", abs(float(spearmanr(np.asarray(Xm[:, j], float), y).statistic)), 0.255, 3)
    # label facts (Part 3)
    lab = pd.read_csv(config.INTERIM / "labels" / "crisprhal_curated.csv").merge(pd.read_csv(config.INTERIM / "labels" / "published_on_curated_rows.csv"), on="sgRNAID", suffixes=("_cur", "_pub"))
    chk(P + "guides shared by the curated and published labels", float(len(lab)), 8617, 0)
    chk(P + "rank agreement of the two labels on them", float(spearmanr(lab.y_cur, lab.y_pub).statistic), 0.945, 3)
    chk(P + "curated label: share negative (%)", float(100 * (y < 0).mean()), 47.5, 1)
    chk(P + "curated label: minimum", float(y.min()), -9.05, 2)
    chk(P + "curated label: maximum", float(y.max()), 6.21, 2)
    from ..io_utils import load_guide_index
    gi = load_guide_index()
    chk(P + "published cut_score: maximum", float(gi.cut_score.max()), 48.05, 2)
    hal = config.EXTERNAL / "crisprHAL" / "data"
    rel = {}
    for fn in ("WT-SpCas9_training_data.csv", "WT-SpCas9_testing_data.csv"):
        df = pd.read_csv(hal / fn, header=None)
        for s_, v in zip(df[0], df[1]):
            try:
                rel.setdefault(s_[189:209].upper(), []).append(float(v))
            except ValueError:
                pass
    worst = max(min(abs(v - yy) for v in rel[p_]) for p_, yy in zip(protos, y))
    chk(P + "curated y against crisprHAL 2's released y, all 33,567 guides: largest difference (units of 1e-12)", worst / 1e-12, 0.0, 0)
    # curation slopes
    cv = pd.read_csv(f("curation_variance.csv")).set_index("rows")
    chk(P + "flank-GC slope, crisprHAL-only rows over shared rows", float(cv.loc["crisprHAL only", "slope_beta"] / cv.loc["shared with Noshay", "slope_beta"]), 2.96, 2)
    chk(P + "flank-GC slope, other genes over shared rows", float(cv.loc["crisprHAL only, other genes", "slope_beta"] / cv.loc["shared with Noshay", "slope_beta"]), 3.2, 1)
    rw = pd.read_csv(f("replicate_weight.csv")).set_index("arm")
    rr = pd.read_csv(f("replicate_weight_raw.csv"))
    chk(P + "curated label against raw log-ratio, same unweighted model", float(rw.loc["unweighted", "spearman"] - rr[rr.arm == "unweighted"].spearman.iloc[0]), 0.022, 3)
    # constant off-target columns
    ot = pd.read_csv(config.FEATURE_DIR / "h_offtarget_curated.csv")
    chk(P + "h_offtarget columns", float(len(ot.columns) - 1), 13, 0)
    chk(P + "h_offtarget columns constant on the curated cohort", float(sum(ot[c].nunique() <= 1 for c in ot.columns[1:])), 6, 0)
    # selector comparison: which model's ranking gave which number
    sc = pd.read_csv(f("selector_comparison.csv")).groupby("selector").spearman.mean()
    chk(P + "selector: LightGBM gain ranking", float(sc["lightgbm gain, 300 trees"]), 0.7033, 4)
    chk(P + "selector: XGBoost gain ranking (the one used)", float(sc["xgboost gain, 300 trees (current)"]), 0.7025, 4)
    # per-fold-tuned figure and the champion on the same protocol
    tu = pd.read_csv(f("tuning.csv"))
    chk(P + "per-fold-tuned LightGBM, mean over the five folds (seed 41)", float(tu.rho_tuned.mean()), 0.7212, 4)
    chk(P + "per-fold search gain over the default, mean over folds", float(tu.delta_rho.mean()), 0.0131, 4)
    st = pd.read_csv(OUT / "family_a_at_tuned_summary.csv")
    st = st.set_index("comparison")
    chk(P + "frozen champion, base + A, gene-grouped 3 x 5", float(st.loc["family A increment, champion", "with_a"]), 0.7151, 4)
    chk(P + "tuned d099x, base + A, gene-grouped 3 x 5", float(st.loc["family A increment, d099x", "with_a"]), 0.7314, 4)
    chk(P + "family-A increment at the champion, gene-grouped 3 x 5", float(st.loc["family A increment, champion", "delta"]), 0.156, 3)
    chk(P + "family-A increment at the tuned settings, gene-grouped 3 x 5", float(st.loc["family A increment, d099x", "delta"]), 0.170, 3)
    # Jacky's files (shared repository)
    jr = config.ROOT / "RSI-Crispr" / "Jacky"
    fk = jr / "human-boundary" / "results" / "family_a_kingdom_transfer.csv"
    if fk.exists():
        k = pd.read_csv(fk).set_index("arm")
        chk(P + "Jacky: base columns, E. coli -> human", float(k.loc["base", "ecoli_to_human"]), -0.049, 3)
        chk(P + "Jacky: base columns, human -> E. coli", float(k.loc["base", "human_to_ecoli"]), -0.010, 3)
        chk(P + "Jacky: base + family A, E. coli -> human", float(k.loc["base+a_flank", "ecoli_to_human"]), 0.022, 3)
        chk(P + "Jacky: base + family A, human -> E. coli", float(k.loc["base+a_flank", "human_to_ecoli"]), 0.017, 3)
        chk(P + "Jacky: flanks alone, E. coli -> human", float(k.loc["a_flank", "ecoli_to_human"]), 0.042, 3)
        chk(P + "Jacky: flanks alone, human -> E. coli", float(k.loc["a_flank", "human_to_ecoli"]), 0.108, 3)
        chk(P + "Jacky: human rows", float(k.loc["base", "n_human"]), 16490, 0)
    gm = jr / "human-boundary" / "results" / "gc_matched_transfer.csv"
    if gm.exists():
        g = pd.read_csv(gm)
        g = g[g.arm == "shared_style_base"].set_index("control").spearman
        chk(P + "Jacky: base columns, residualised GC", float(g["residualise_gc2"]), -0.049, 3)
        chk(P + "Jacky: base columns, GC histogram matched", float(g["train_subsample_match_human_gc_hist"]), -0.049, 3)
        chk(P + "Jacky: base columns, E. coli GC range 0.35-0.70", float(g["gc_in_ecoli_5_95_[0.35,0.70]"]), -0.060, 3)
    od = jr / "orientation-controls" / "results" / "intergenic_orientation_controls_delta_ci.csv"
    if od.exists():
        o = pd.read_csv(od).set_index("analysis")
        chk(P + "Jacky: TSS orientation, unmatched Delta", float(o.loc["tss", "delta"]), 0.051, 3)
        chk(P + "Jacky: TSS orientation, unmatched lower", float(o.loc["tss", "delta_lo"]), 0.020, 3)
        chk(P + "Jacky: TSS orientation, unmatched upper", float(o.loc["tss", "delta_hi"]), 0.082, 3)
        chk(P + "Jacky: TSS orientation, matched on distance and GC", float(o.loc["tss_matched_dist_gc", "delta"]), 0.025, 3)
        chk(P + "Jacky: TSS orientation, matched lower", float(o.loc["tss_matched_dist_gc", "delta_lo"]), -0.003, 3)
        chk(P + "Jacky: TSS orientation, matched upper", float(o.loc["tss_matched_dist_gc", "delta_hi"]), 0.053, 3)
        chk(P + "Jacky: TSS orientation, matched P(Delta <= 0)", float(o.loc["tss_matched_dist_gc", "p_delta_le_0"]), 0.04, 2)
        chk(P + "Jacky: nearest-gene orientation, matched", float(o.loc["gene_matched_dist_gc", "delta"]), 0.007, 3)


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
        print(f"  {'ok ' if ok else 'BAD'} {label:100s} computed {got:+.5f}  report {claimed:+.{max(dec, 0)}f}")

    run(chk)
    print(f"\n{n} checks, {len(bad)} mismatches")
    for lab, got, claimed in bad:
        print(f"  BAD {lab}: computed {got:+.5f}, report {claimed}")
    raise SystemExit(1 if bad else 0)
