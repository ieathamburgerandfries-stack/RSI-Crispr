"""Checks for PROGRESS_REPORT.md Finding 7 (what the signal is made of) that the main audit does not read, or reads under another label.

Same idiom as `paper_numbers_C.py`: `chk(label, got, claimed, dec)` passes when |got - claimed| <= 0.5 * 10**(-dec); `got` is computed from a stored
file (or, where no file stores the quantity, from the data it is a function of), `claimed` is the value exactly as the report writes it.

What is checked here, and on which cohort / protocol (a number can occur for a different quantity, so each group says so):

    (d)   curated 33,567-guide cohort, per-base effects (`mechanism_position_base*.csv`), the E. coli - human map comparison, the champion's and the
          tuned model's gain by guide position (`interpret_position_profile.csv`, DEV genes, seed 41).
    P1-P5 the CRISPOR tables (`eukaryote_*.csv`): pairs of different libraries only; 'corrected' = divided by the split-half reliabilities; the
          transfer shares are per-pair transfer rho / the target's own random-row five-fold rho, pipeline model (SLICER base, champion settings).
    H1-H3 the 30,203 genic guides, gene bootstrap (`hypotheses_*.csv`); the target-GC strata are the curated 33,567 guides with the 500-nt flank GC.
    (c)   `novel_resolution.csv`: curated genic guides, gene-grouped five-fold CV, seeds 41-43 (15 fits), tuned d099x settings, top-450 selector.
    (g)   the position-20 contrast of the two bacterial transfer screens, computed from `data/interim/transfer_*/meta.json` (the only quantity here that
          no result file stores): G minus the mean of A and C at guide position 20, in SD of the screen's label, guide-level bootstrap (300 draws, seed 41).

    ./run_ext.sh sgrna.paper.report_checks_R4
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

from sgrna import config

from ..common import OUT

P = "report R4: "


def _contrast_pos20(name, rng, boot=300):
    meta = json.loads((config.INTERIM / f"transfer_{name}" / "meta.json").read_text())
    p = np.asarray(meta["protospacers"]); y = np.asarray(meta["y"], float)
    ok = np.isfinite(y) & np.array([len(s) == 20 for s in p])
    p, y = p[ok], y[ok]
    y = (y - y.mean()) / y.std()
    c = np.frombuffer("".join(p).encode(), np.uint8).reshape(len(p), 20)[:, 19]
    g, ac = c == ord("G"), (c == ord("A")) | (c == ord("C"))
    f = lambda ix: float(y[ix][g[ix]].mean() - y[ix][ac[ix]].mean())
    bs = [f(rng.integers(0, len(y), len(y))) for _ in range(boot)]
    return f(np.arange(len(y))), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(y)


def run(chk):
    """Run every check through `chk(label, got, claimed, dec)`."""
    RES = OUT

    # ------------------------------------------------------------------ (d) the guide's own base preferences
    print("report R4 (d): per-base effects, the E. coli - human comparison, gain by position")
    e = pd.read_csv(OUT / "mechanism_position_base.csv")
    strong = e[e.clear_of_zero]
    strong = strong.reindex(strong.effect_sd.abs().sort_values(ascending=False).index).head(12)
    chk(P + "(d) the twelve strongest per-base effects: earliest position", float(strong.position.min()), 14, 0)
    chk(P + "(d) ... latest position", float(strong.position.max()), 20, 0)
    chk(P + "(d) largest |per-base effect| at positions 1-7 (SD)", float(e[e.position <= 7].effect_sd.abs().max()), 0.04, 2)
    c = pd.read_csv(OUT / "mechanism_position_base_compare.csv")
    joint = c.ecoli_clear & c.human_clear
    chk(P + "(d) cells clear of zero in both organisms", float(joint.sum()), 27, 0)
    chk(P + "(d) ... of which opposite in sign", float(c.inverts.sum()), 15, 0)
    chk(P + "(d) 15 of 27 against a fair coin: two-sided binomial p", float(binomtest(int(c.inverts.sum()), int(joint.sum()), 0.5).pvalue), 0.70, 2)
    inv = c[c.inverts]
    chk(P + "(d) opposite cells at position 20 (G and T)", float((inv.position == 20).sum()), 2, 0)
    chk(P + "(d) opposite cells at positions 18-19", float(inv.position.between(18, 19).sum()), 5, 0)
    pp = pd.read_csv(OUT / "interpret_position_profile.csv")
    gs = pp[(pp.region == "guide") & (pp.position >= 13)].groupby("model").gain_share.sum()
    chk(P + "(d) share of the champion's split gain in the eight PAM-proximal positions, %", 100 * float(gs["original"]), 63, 0)
    chk(P + "(d) share of the tuned model's split gain in the eight PAM-proximal positions, %", 100 * float(gs["tuned"]), 62, 0)

    # ------------------------------------------------------------------ the CRISPOR tables
    print("report R4 (d): the CRISPOR tables, orientation, P1-P5")
    tabs = sorted((config.ROOT / "external_data" / "crispor_effdata").glob("*.scores.tab"))
    chk(P + "(d) tables of the CRISPOR benchmark collection on disk and in the orientation file", float(len(tabs)), 24, 0)
    o = pd.read_csv(OUT / "eukaryote_orientation.csv")
    chk(P + "(d) ... rows of eukaryote_orientation.csv", float(len(o)), 24, 0)
    big = o[o.n >= 300]
    chk(P + "(d) datasets of at least 300 guides", float(len(big)), 11, 0)
    chk(P + "(d) ... of which flipped by the orientation rule", float((big.sign < 0).sum()), 4, 0)
    chk(P + "(d) ... and the flipped ones are Wang 2015 and the three Xu 2015 tables",
        float(set(big[big.sign < 0].dataset) == {"wang2015_hg19", "xu2015TrainHl60", "xu2015TrainKbm7", "xu2015TrainMEsc"}), 1, 0)
    ct = pd.read_csv(OUT / "eukaryote_contrast_g_minus_t.csv").set_index("group")
    bc = ct.loc[big.dataset]
    chk(P + "(d) of the eleven, contrast G minus mean(A, C) positive", float((bc.g_minus_ac_pos20 > 0).sum()), 11, 0)
    chk(P + "(d) ... with an interval excluding zero", float((bc.g_minus_ac_lo > 0).sum()), 10, 0)
    chk(P + "(d) of the seven that kept their native direction, intervals excluding zero",
        float((ct.loc[big[big.sign > 0].dataset].g_minus_ac_lo > 0).sum()), 7, 0)
    chk(P + "(d) G minus T at position 20 is undefined for exactly the two Hart libraries (no T at position 20)",
        float(ct.loc[big.dataset].g_minus_t_pos20.isna().sum()), 2, 0)

    pr = pd.read_csv(OUT / "eukaryote_map_pairs.csv")
    rel = pd.concat([pr[["a", "reliability_a"]].set_axis(["d", "r"], axis=1), pr[["b", "reliability_b"]].set_axis(["d", "r"], axis=1)]).groupby("d").r.first()
    chk(P + "(d) lowest split-half reliability of an 80-cell map", float(rel.min()), 0.66, 2)
    chk(P + "(d) highest split-half reliability", float(rel.max()), 0.95, 2)
    diff = pr[~pr.same_library]
    mam = diff[diff.species_a.isin(["human", "mouse"]) & diff.species_b.isin(["human", "mouse"])]
    chk(P + "P1: mammal pairs of different libraries", float(len(mam)), 43, 0)
    chk(P + "P1: ... with corrected correlation below 0.4", float((mam.rho_corrected < 0.4).sum()), 9, 0)
    low = mam[mam.rho_corrected < 0.4]
    chk(P + "P1: ... of which involve Wang 2015 or Doench 2016",
        float(low.apply(lambda r: bool({r.a, r.b} & {"wang2015_hg19", "doench2016_hg19"}), axis=1).sum()), 7, 0)
    kind = lambda d: d.apply(lambda r: " - ".join(sorted([r.species_a, r.species_b])), axis=1)
    kd = diff.assign(kind=kind(diff))
    med = kd.groupby("kind").rho_corrected.median()
    chk(P + "P1: corrected median, mouse - human no more than 0.2 below human - human (gap)", float(med["human - human"] - med["human - mouse"]), 0.03, 2)
    ec = diff[(diff.a == "E. coli") | (diff.b == "E. coli")]
    chk(P + "P2: E. coli pairs of different libraries", float(len(ec)), 11, 0)
    chk(P + "P2: ... with |rho| beyond 0.2 (uncorrected)", float((ec.rho.abs() > 0.2).sum()), 2, 0)
    chk(P + "P2: ... lowest rho", float(ec.rho.min()), -0.28, 2)
    chk(P + "P2: ... highest rho", float(ec.rho.max()), 0.08, 2)
    em = ec.assign(sp=ec.apply(lambda r: r.species_b if r.a == "E. coli" else r.species_a, axis=1))
    chk(P + "P2: E. coli - mouse median rho (2 pairs)", float(em[em.sp == "mouse"].rho.median()), -0.01, 2)
    chk(P + "P2: E. coli - zebrafish rho (1 pair)", float(em[em.sp == "zebrafish"].rho.iloc[0]), -0.21, 2)
    chk(P + "P3: human - zebrafish corrected median (predicted band 0.2 to 0.5)", float(med["human - zebrafish"]), 0.28, 2)
    chk(P + "P3: mouse - zebrafish corrected median (2 pairs)", float(med["mouse - zebrafish"]), 0.07, 2)

    t = pd.read_csv(OUT / "eukaryote_transfer.csv")
    g = t[t.model.str.startswith("pipeline")]
    own = g[g.within].set_index("scored_on").rho
    x = g[~g.within & ~g.same_library].copy()
    x["share"] = x.rho / x.scored_on.map(own)
    for a, b, k in (("human", "mouse", 13), ("mouse", "human", 12)):
        h = x[(x.species_trained == a) & (x.species_scored == b)]
        chk(P + f"P4: {a} -> {b} pairs", float(len(h)), 16, 0)
        chk(P + f"P4: {a} -> {b} pairs keeping at least 60% of the target's own rho", float((h.share >= 0.6).sum()), k, 0)
    ex = x[x.trained_on == "E. coli"]
    chk(P + "P4: E. coli -> eukaryote pairs", float(len(ex)), 11, 0)
    chk(P + "P4: ... beyond +-0.05 of zero", float((ex.rho.abs() > 0.05).sum()), 5, 0)
    chk(P + "P4: ... lowest rho (zebrafish)", float(ex.rho.min()), -0.24, 2)

    f = pd.read_csv(OUT / "eukaryote_flank_step.csv")
    ef = f[f.species != "E. coli"]
    chk(P + "P5: eukaryotic datasets with an interval excluding zero", float((ef.hi < 0).sum()), 5, 0)
    chk(P + "P5: ... all of them mammalian (human or mouse)", float(ef[ef.hi < 0].species.isin(["human", "mouse"]).sum()), 5, 0)
    chk(P + "P5: mammalian datasets of at least 300 guides", float(ef.species.isin(["human", "mouse"]).sum()), 10, 0)
    chk(P + "P5: datasets with rho inside the tail at or above 0.15", float((ef.rho_below >= 0.15).sum()), 2, 0)
    chk(P + "P5: largest rho inside the tail (the report: at most 0.20)", float(ef.rho_below.max()), 0.20, 2)

    # ------------------------------------------------------------------ (b) the orientation effect, Jacky's file (teammate's, read only)
    jp = config.ROOT / "RSI-Crispr" / "Jacky" / "orientation-controls" / "results" / "intergenic_orientation_controls_delta_ci.csv"
    if jp.exists():
        print("report R4 (b): orientation effect against a curated TSS, unmatched and matched for distance and GC")
        j = pd.read_csv(jp).set_index("analysis")
        for row, tag, vals in (("tss", "unmatched", (0.051, 0.020, 0.082)), ("tss_matched_dist_gc", "matched", (0.025, -0.003, 0.053))):
            for col, v in zip(("delta", "delta_lo", "delta_hi"), vals):
                chk(P + f"(b) orientation effect against a curated TSS, {tag}: {col}", float(j.loc[row, col]), v, 3)
    else:
        print(f"report R4 (b): {jp} not found, orientation checks skipped")

    # ------------------------------------------------------------------ H1-H3 and the step
    print("report R4 (e)-(f): H1, the expression thirds, the strata, the genes")
    h1 = pd.read_csv(OUT / "hypotheses_h1_contrasts.csv")
    r = h1[(h1.position == 20) & (h1.steepest == "A")].iloc[0]
    chk(P + "H1: steepest (A) minus shallowest (G) slope at position 20, genic guides", float(r["difference"]), 0.101, 3)
    chk(P + "H1: ... lower bound", float(r["lo"]), 0.066, 3)
    chk(P + "H1: ... upper bound", float(r["hi"]), 0.131, 3)
    chk(P + "H1: the steepest base at position 20 is A and the shallowest G", float(r.steepest == "A" and r.shallowest == "G"), 1, 0)
    ex_ = pd.read_csv(OUT / "hypotheses_followup_expression.csv")
    chk(P + "(f) guides with an expression value (three thirds)", float(ex_.n.sum()), 29097, 0)
    chk(P + "(f) deficit in the lowest expression third over the highest (about twice)", float(ex_.deficit.iloc[0] / ex_.deficit.iloc[2]), 2.0, 0)
    tg = pd.read_csv(OUT / "hypotheses_followup_top_genes.csv", index_col=0)
    named = {"wbbJ", "waaL", "glf", "waaO", "alpA", "trkG", "lit", "mcrC", "casA"}
    chk(P + "(g) genes most often in the lowest decile that the report ranks (top fifteen)", float(len(tg)), 15, 0)
    chk(P + "(g) ... of which the nine named ones", float(len(named & set(tg.index))), 9, 0)

    # the lowest decile of flank GC set aside, within each target-GC stratum (curated 33,567 guides, 500-nt flank GC)
    from .hooks import _frame_curated, _strata
    df = _frame_curated()
    s, xg, y = _strata(df).to_numpy(), df.flank_gc.to_numpy(float), df.y.to_numpy(float)
    keep = xg > np.quantile(xg, 0.10)
    full = [float(spearmanr(xg[s == k], y[s == k]).statistic) for k in range(5)]
    rest = [float(spearmanr(xg[(s == k) & keep], y[(s == k) & keep]).statistic) for k in range(5)]
    chk(P + "(f) stratum 0 (GC-poor targets): flank GC against label, all guides", full[0], 0.425, 3)
    chk(P + "(f) stratum 4 (GC-rich targets): flank GC against label, all guides", full[4], 0.068, 3)
    chk(P + "(f) stratum 0 with the lowest 10% of flank GC set aside", rest[0], 0.155, 3)
    chk(P + "(f) stratum 4 with the lowest 10% of flank GC set aside", rest[4], 0.037, 3)
    chk(P + "(f) the five strata, tail set aside, fall monotonically from stratum 0 to 4", float(all(np.diff(rest) < 0)), 1, 0)

    # ------------------------------------------------------------------ family H, resolution, bacteria
    print("report R4 (c), (d): Family H, the resolution experiment, the second bacterial genome")
    ac = pd.read_csv(RES / "ablation_curated.csv").set_index("family")
    chk(P + "(d) Family H (h_offtarget) gain, curated cohort, random-row", float(ac.loc["h_offtarget", "d_spearman"]), 0.001, 3)
    ag = pd.read_csv(RES / "ablation_curated_gene.csv").set_index("family")
    chk(P + "(d) Family H gain, curated cohort, gene-grouped", float(ag.loc["h_offtarget", "d_spearman"]), 0.000, 3)
    rr = pd.read_csv(OUT / "novel_resolution.csv").set_index("arm")
    rho = lambda n: float(rr[rr.index.str.startswith(f"base + {n} binned cols")].rho.iloc[0])
    chk(P + "(c) 32 binned columns minus 200 binned columns", rho(32) - rho(200), 0.0014, 4)
    chk(P + "(c) 80 binned columns minus 200 binned columns", rho(80) - rho(200), 0.003, 3)
    chk(P + "(c) smallest fold-to-fold SD over the binned arms", float(rr.sd.min()), 0.006, 3)
    chk(P + "(c) largest fold-to-fold SD over the binned arms", float(rr.sd.max()), 0.008, 3)
    chk(P + "(c) fits per arm (3 seeds x 5 gene-grouped folds)", float(rr.n_fits.min()), 15, 0)

    rng = np.random.default_rng(41)
    cr = _contrast_pos20("TevSpCas9", rng)
    es = _contrast_pos20("eSpCas9", rng)
    wt = _contrast_pos20("WT-SpCas9", rng)
    chk(P + "(d) position-20 contrast G minus mean(A, C), WT SpCas9 (reproduces the stored -0.333)", wt[0], -0.333, 3)
    chk(P + "(d) ... E. coli eSpCas9, same loci", es[0], -0.21, 2)
    chk(P + "(d) ... C. rodentium TevSpCas9", cr[0], -0.42, 2)
    chk(P + "(d) ... C. rodentium lower bound", cr[1], -0.45, 2)
    chk(P + "(d) ... C. rodentium upper bound", cr[2], -0.39, 2)
    chk(P + "(d) ... C. rodentium guides", float(cr[3]), 25210, 0)


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
        print(f"  {'ok ' if ok else 'BAD'} {label:112s} computed {got:+.5f}  report {claimed:+.{max(dec, 0)}f}")

    run(chk)
    print(f"\n{n} checks, {len(bad)} mismatches")
    for lab, got, claimed in bad:
        print(f"  BAD {lab}: computed {got:+.5f}, report {claimed}")
    raise SystemExit(1 if bad else 0)
