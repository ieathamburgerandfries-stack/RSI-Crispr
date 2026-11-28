"""What the flank effect physically is can be asked of the cohort directly, without a model: flank GC and the seed base act in synergy and not as compensation, the dependence on flank GC is concentrated at the AT-rich end between genes (roughly linear within them), and it is composition and not PAM density.

Finding 5 says that on a matrix where 99% of columns regenerate from the 20-mer, column-level importance is a statement about the selector, so every reading here is a marginal or stratified statistic, a fitted arm or an enumeration of paths, never a model's importance. Twelve tests in three
groups, run with `--test` (several allowed; `all` runs every one):

  where the signal sits (Finding 5 and 7)
    bases        position x base effect on activity, straight from the protospacers and the label                          mechanism_position_base.csv
    distance     signed partial association of each flank window's composition with activity, holding target GC fixed       mechanism_distance_profile.csv
    asymmetry    upstream-only against downstream-only composition, as fitted arms                                          mechanism_asymmetry.csv
    human        the same position x base estimator on the human matrix                                                     mechanism_position_base_human.csv, mechanism_position_base_compare.csv
    pairs        which column pairs carry the pairwise interaction, by gain of depth-2 paths                                mechanism_pairs_columns.csv, mechanism_pairs_blocks.csv
  what it might physically be: three predictions written before the run
    h1 h2 h3     the form of the coupling to the seed (H1), the shape of the dependence on flank GC (H2), GC or PAM density (H3)   hypotheses_h*.csv
    followups    exploratory, designed after H2 failed: how robust the AT-rich step is, and which genes sit in it            hypotheses_followup_*.csv
    audit        the checks on the scale and the gate used by the above                                                        hypotheses_audit_*.csv
  two experiments that are ours rather than borrowed
    resolution   how finely does the flank have to be described? the resolution floor of the flank signal                  novel_resolution.csv
    relative     species-relative composition and cross-organism transfer                                                    novel_relative_composition.csv

    python -m sgrna.interpret.mechanism --test bases distance
    python -m sgrna.interpret.mechanism --test h1 h2 h3        # about half a minute
    python -m sgrna.interpret.mechanism --test all

------------------------------------------------------------------------------------------------------------------------------
bases, distance, asymmetry, human, pairs
------------------------------------------------------------------------------------------------------------------------------
Finding 5 says that on a matrix where 99% of columns regenerate from the 20-mer, column-level
importance is a statement about the selector. So the readings here avoid model importance wherever a
direct one exists:

  bases      position x base effect on activity, computed from the protospacers and the label
             directly -- no model, no selector, nothing to be confounded by redundancy. This is the
             part a bench scientist can use, and it is comparable with published position weights.
  distance   signed partial association of each flank window's composition with activity, holding
             target GC fixed, as a function of distance and side. A marginal statistic, so again no
             model.
  asymmetry  upstream-only against downstream-only composition, as fitted arms. If the effect is
             one-sided it points at a directional process (replication, transcription, resection)
             rather than at a bulk property.
  pairs      which column PAIRS carry the pairwise interaction. Section 14 found 37% of what trees
             add over a linear model is pairwise; a depth-2 model's root-to-leaf paths ARE pairs, so
             enumerating them by gain names the interactions without SHAP's additivity assumptions.

------------------------------------------------------------------------------------------------------------------------------
h1, h2, h3, followups, audit
------------------------------------------------------------------------------------------------------------------------------
Finding 7 says what the signal is made of (GC, symmetric, one length scale of 250-500 nt, coupled to the PAM-proximal seed) and
leaves "why" as hypotheses. These tests turn three of them into something the cohort can answer without a model, so each is a
marginal or stratified statistic on the 30,203 genic guides, bootstrapped over whole genes (4,013 of them) so a gene's guides move
together. Nothing here is a feature added to the model.

H1 -- the FORM of the coupling to the seed. Finding 7 found flank GC x a position-20 property to be the strongest pair but not its
   sign. Two readings predict different slopes. *Compensation* (the flank and the seed feed one rate-limiting step, so a good flank
   rescues a bad seed): the effect of flank GC is steeper where the position-20 base is the unfavourable one (G). *Synergy* (they
   act at different steps, so one gates the other): steeper where it is favourable (T).
   Prediction, written before the run: SYNERGY. The only prior evidence is the target-GC strata, where the flank effect is largest
   in GC-poor targets (+0.425 against +0.068), but that is confounded with headroom, so this is a prior and not a repeat.
   Decision rule: heterogeneity is "present" if the gene-bootstrap 95% interval of (steepest minus shallowest stratum slope)
   excludes 0; direction by which of G and T is steeper.

H2 -- the SHAPE of the dependence on flank GC. A bulk physical property (duplex stability, stiffness) predicts a roughly linear
   dependence over the whole range. A protection or occupancy mechanism (AT-rich DNA bound by nucleoid proteins such as H-NS and so
   shielded from Cas9) predicts that the dependence is concentrated at the AT-rich end and flat above it.
   Prediction, written before the run: ROUGHLY LINEAR (probability about 0.6).
   Decision rule: slope of the label on flank GC within the lowest 30% of the flank-GC range against the highest 30%; "curved" if
   the gene-bootstrap 95% interval of (low-third slope minus high-third slope) excludes 0. Run pooled and within genes (both
   variables centred on the gene mean), because low-GC genes are also horizontally acquired genes.

H3 -- is it GC, or PAM density? GC-rich DNA is dense in NGG/CCN. A Cas9 that samples PAMs and dissociates is more likely to find a
   target that sits in a PAM-rich neighbourhood (local rebinding), and that would look exactly like a GC effect. Composition alone
   predicts that PAM density is unremarkable once GC is held fixed.
   Prediction, written before the run: COMPOSITION ONLY (probability about 0.8).
   Decision rule: partial Spearman of PAM-site density (NGG on the guide strand plus CCN, both flanks, 250 nt) with the label, holding
   flank GC fixed, against 5,000 random sums of four of the 32 strand-collapsed trimer classes (the same kind of GC-matched null as
   the Chi-site test). "Distinct" if it is above the 97.5th percentile of that null AND the partial of GC holding PAM density fixed
   falls below half its marginal value.

RESULT OF THE THREE, recorded here so the prediction record is not rewritten: H1 held (synergy), H3 held (composition only), and
H2 FAILED -- the dependence is not roughly linear but concentrated at the AT-rich end between genes, and roughly linear within
them. `followups()` was designed AFTER seeing that and is exploratory: it asks how robust the AT-rich step is (window width,
location, leave-one-block-out, a different enzyme on the same genome, a different genome) and which genes sit in it. Nothing in it
was predicted in advance, so what it shows is a description to be tested, not a test.

------------------------------------------------------------------------------------------------------------------------------
resolution, relative
------------------------------------------------------------------------------------------------------------------------------
resolution floor of the flank signal.

Neither copies a published design. Each asks a question the literature's framing makes available
but nobody in it has asked.

  relative    Family A's strongest column is `eng.flank.comp.up250.gc`, a RAW GC fraction. K-12 is
              50.79% GC and C. rodentium 54.72%, so a flank at 52% GC is above background in one
              genome and below it in the other: the feature carries the opposite sign of local
              enrichment across the two, and the cross-organism transfer that retains 88-91% does
              so despite this. Expressing each window as a percentile of its OWN genome's
              distribution of that same statistic should fix the sign.
              **Within one genome this cannot change anything** -- a percentile map of a single
              column is monotone and boosted trees are invariant to monotone transforms of
              individual features -- so the home cohort is run only as a check that it is exactly a
              no-op, and the real experiment is the transfer.

  resolution  Browne et al. searched input *length* and found the gains level off. The orthogonal
              question is *resolution*: how finely does the flank have to be described? Family A
              uses four nested windows per side; Finding 2 says the effect is a smooth gradient with
              no motif and no dependence on where the window edges fall. If that is right there
              should be a resolution floor -- a small number of bins beyond which nothing is gained
              -- and finding it bounds the kind of physical process that could produce the effect,
              as well as telling us how cheap the featuriser could be.

Report section: Finding 7.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from sgrna import config
from sgrna.harness.run_ablation import _impute, grouped_folds, make_selector

from ..common import OUT
from ..tuning import tune_ecoli as T

BASES = "ACGT"


def _cohort():
    from sgrna.harness.curated import load_base, load_block
    X, y, names, protos = load_base()
    A, an = load_block("a_flank", protos)
    return X, y, np.asarray(protos, dtype=object), A, list(an)


def _partial(a, b, c):
    """Spearman partial correlation of a and b given c, via ranks and residualisation."""
    ra, rb, rc = (rankdata(v) for v in (a, b, c))
    ra, rb, rc = (v - v.mean() for v in (ra, rb, rc))
    res = lambda v: v - rc * (v @ rc) / (rc @ rc)
    x, z = res(ra), res(rb)
    return float((x @ z) / np.sqrt((x @ x) * (z @ z)))


# ------------------------------------------------------------------ position x base
def bases(seed=0):
    """Effect of each base at each guide position on activity, in SD units of the label."""
    _, y, protos, _, _ = _cohort()
    seqs = np.frombuffer("".join(protos).encode("ascii"), np.uint8).reshape(len(protos), 20)
    gc_target = np.array([(s.count("G") + s.count("C")) / 20 for s in protos])
    sign = np.sign(spearmanr(gc_target, y).statistic)
    print(f"  target GC vs label rho = {spearmanr(gc_target, y).statistic:+.4f} "
          f"(report: -0.20, so higher label = more activity when this is negative)", flush=True)
    sd = y.std()
    rows = []
    for pos in range(20):
        col = seqs[:, pos]
        for b in BASES:
            m = col == ord(b)
            if m.sum() < 50:
                continue
            eff = (y[m].mean() - y.mean()) / sd
            # normal approximation to the SE of a mean over ~thousands of guides; a bootstrap adds nothing here
            se = y[m].std(ddof=1) / np.sqrt(m.sum()) / sd
            rows.append(dict(position=pos + 1, base=b, n=int(m.sum()), effect_sd=float(eff),
                             lo=float(eff - 1.96 * se), hi=float(eff + 1.96 * se)))
    out = pd.DataFrame(rows)
    out["clear_of_zero"] = (out.lo > 0) | (out.hi < 0)
    out.to_csv(OUT / "mechanism_position_base.csv", index=False)
    piv = out.pivot(index="position", columns="base", values="effect_sd")
    pd.set_option("display.width", 200)
    print("\n  effect on the label, SD units (positive = higher label):")
    print(piv.round(3).to_string())
    strong = out[out.clear_of_zero].reindex(out[out.clear_of_zero].effect_sd.abs().sort_values(ascending=False).index)
    print("\n  strongest position-base preferences clear of zero:")
    print(strong.head(12)[["position", "base", "n", "effect_sd", "lo", "hi"]].round(3).to_string(index=False))
    return out


# ------------------------------------------------------------------ signed distance profile
def distance():
    _, y, protos, A, an = _cohort()
    gc_target = np.array([(s.count("G") + s.count("C")) / 20 for s in protos])
    rows = []
    for j, n in enumerate(an):
        p = n.split(".")
        if len(p) == 5 and p[2] == "comp" and p[4] in ("gc", "purine"):
            side, w = p[3][:2], int(p[3][2:])
            v = A[:, j].astype(float)
            ok = np.isfinite(v)
            rows.append(dict(side=side, window_nt=w, statistic=p[4], n=int(ok.sum()),
                             marginal=float(spearmanr(v[ok], y[ok]).statistic),
                             partial_given_target_gc=_partial(v[ok], y[ok], gc_target[ok])))
    out = pd.DataFrame(rows).sort_values(["statistic", "side", "window_nt"])
    out.to_csv(OUT / "mechanism_distance_profile.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    for stat in ("gc", "purine"):
        s = out[out.statistic == stat]
        for side in ("up", "dn"):
            t = s[s.side == side].sort_values("window_nt")
            print(f"  {stat} {side}: " + "  ".join(f"{int(r.window_nt)}nt {r.partial_given_target_gc:+.3f}"
                                                   for r in t.itertuples()))
    return out


# ------------------------------------------------------------------ upstream vs downstream
def asymmetry(seeds=(41, 42, 43)):
    import json
    from sgrna.harness.run_ablation import _impute, grouped_folds, make_selector
    from ..tuning import tune_ecoli as T
    X0, y, protos, A, an = _cohort()
    from sgrna.harness.curated import load_groups
    g, ok = load_groups("gene", protos)
    g = np.asarray(g, dtype=object)
    comp = {s: [j for j, n in enumerate(an) if n.startswith(f"eng.flank.comp.{s}")] for s in ("up", "dn")}
    cfg = dict(json.loads((OUT / "tune_configs.json").read_text())["d099x"],
               n_estimators=int(T.selected().loc["d099x", "k"]))
    arms = {"base only": X0,
            "base + upstream composition": np.hstack([X0, A[:, comp["up"]]]),
            "base + downstream composition": np.hstack([X0, A[:, comp["dn"]]]),
            "base + both sides": np.hstack([X0, A[:, comp["up"] + comp["dn"]]])}
    rows = []
    for label, Xa in arms.items():
        Xa = Xa.astype(np.float32)
        per = []
        for seed in seeds:
            for tr, va in grouped_folds(g[ok], 5, seed):
                i = np.where(ok)[0]
                a0, _ = _impute(Xa[i[tr]], Xa[i[va]][:1])
                sel = make_selector(seed); sel.fit(a0, y[i[tr]])
                top = np.argsort(sel.feature_importances_)[::-1][: int(cfg["nf"])]
                a, b = _impute(Xa[i[tr]][:, top], Xa[i[va]][:, top])
                p = T.fit_predict(cfg, a, y[i[tr]], b, seed)[cfg["n_estimators"]]
                per.append(float(spearmanr(y[i[va]], p).statistic))
        rows.append(dict(arm=label, n_cols=Xa.shape[1] - X0.shape[1], rho=float(np.mean(per)),
                         sd=float(np.std(per, ddof=1))))
        print(f"  {label:32s} rho {rows[-1]['rho']:.4f} +- {rows[-1]['sd']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "mechanism_asymmetry.csv", index=False)
    return pd.DataFrame(rows)






# ------------------------------------------------------------------ the same estimator on human
def human_bases():
    """Position x base effects on the human matrix, by the identical estimator, for a like-for-like
    comparison with 21.2.

    Comparing our statistic against Doench's fitted model weights would be apples to oranges -- their
    weights come from a regularised regression over many correlated features and are not per-position
    marginals. So we run OUR estimator on THEIR data instead. Guides are deduplicated first (17,421
    rows decode to 13,138 distinct 20-mers, and a repeated guide is a repeat measurement, not a second
    observation), labels averaged within a guide.
    """
    from ..compare.human_flank import decode
    dec = decode()
    y = pd.to_numeric(dec.y, errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & dec.decoded_ok.to_numpy(bool)
    d = pd.DataFrame({"p": dec.protospacer[ok].astype(str).to_numpy(), "y": y[ok]})
    g = d.groupby("p", sort=False).y.mean()
    protos, yy = g.index.to_numpy(), g.to_numpy()
    print(f"  human: {int(ok.sum()):,} usable rows -> {len(protos):,} distinct guides (labels averaged)", flush=True)
    seqs = np.frombuffer("".join(protos).encode("ascii"), np.uint8).reshape(len(protos), 20)
    gc = np.array([(s.count("G") + s.count("C")) / 20 for s in protos])
    print(f"  target GC vs label rho = {spearmanr(gc, yy).statistic:+.4f} "
          f"(report: +0.02 in human against -0.20 in E. coli)", flush=True)
    sd = yy.std()
    rows = []
    for pos in range(20):
        col = seqs[:, pos]
        for b in BASES:
            m = col == ord(b)
            if m.sum() < 50:
                continue
            eff = (yy[m].mean() - yy.mean()) / sd
            se = yy[m].std(ddof=1) / np.sqrt(m.sum()) / sd
            rows.append(dict(position=pos + 1, base=b, n=int(m.sum()), effect_sd=float(eff),
                             lo=float(eff - 1.96 * se), hi=float(eff + 1.96 * se)))
    out = pd.DataFrame(rows)
    out["clear_of_zero"] = (out.lo > 0) | (out.hi < 0)
    out.to_csv(OUT / "mechanism_position_base_human.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n  human, effect on the label in SD units:")
    print(out.pivot(index="position", columns="base", values="effect_sd").round(3).to_string())
    # the comparison that matters
    ec = pd.read_csv(OUT / "mechanism_position_base.csv").set_index(["position", "base"])
    hu = out.set_index(["position", "base"])
    both = pd.DataFrame({"ecoli": ec.effect_sd, "human": hu.effect_sd,
                         "ecoli_clear": ec.clear_of_zero, "human_clear": hu.clear_of_zero}).dropna()
    both["inverts"] = (np.sign(both.ecoli) != np.sign(both.human)) & both.ecoli_clear & both.human_clear
    both.to_csv(OUT / "mechanism_position_base_compare.csv")
    print(f"\n  rank correlation of the two matrices: {spearmanr(both.ecoli, both.human).statistic:+.4f}")
    print(f"  position-base cells clear of zero in BOTH and opposite in sign: "
          f"{int(both.inverts.sum())} of {int((both.ecoli_clear & both.human_clear).sum())}")
    key = both.loc[[(20, 'G'), (20, 'T'), (18, 'G'), (18, 'T')]] if (20, 'G') in both.index else both.head(4)
    print("\n  the cells the inversion claim rests on:")
    print(key.round(3).to_string())
    return both


# ------------------------------------------------------------------ which pairs interact
def pairs(seed=41, top=20):
    """Column pairs that co-occur on a depth-2 tree's root-to-leaf path, weighted by gain.

    Section 14 found that 37% of what trees add over a linear model is pairwise. A depth-2 model can
    express exactly pairs, so its (root, child) pairs ARE the interactions, with none of SHAP's
    additivity assumptions. Reported at block level, which is the unit Finding 5 says is readable.
    """
    import json
    import xgboost as xgb
    from sgrna.harness.run_ablation import _impute, make_selector
    from sgrna.interpret.importance import block_of
    from ..tuning import tune_ecoli as T
    from sgrna.harness.curated import load_base, load_block
    X0, y, names, protos = load_base()
    A, an = load_block("a_flank", protos)
    allnames = list(names) + list(an)
    co = T.Cohort()
    cfg = dict(json.loads((OUT / "tune_configs.json").read_text())["d099x"], max_depth=2, n_estimators=4000)
    rk = T.DIR / f"rank_devall_s{seed}.npy"
    top_cols = np.load(rk)[: int(cfg["nf"])]
    Atr, _ = _impute(co.X[co.dev][:, top_cols], co.X[co.conf][:1][:, top_cols])
    m = xgb.XGBRegressor(**T.xgb_params(cfg, seed)).fit(Atr, co.y[co.dev])
    tdf = m.get_booster().trees_to_dataframe()
    tdf = tdf[tdf.Feature != "Leaf"]
    idx = {f"f{i}": allnames[c] for i, c in enumerate(top_cols)}
    rows = []
    for tree, grp in tdf.groupby("Tree"):
        root = grp[grp.ID.str.endswith("-0")]
        if root.empty:
            continue
        rf = root.Feature.iloc[0]
        rg = float(root.Gain.iloc[0])
        for _, ch in grp[~grp.ID.str.endswith("-0")].iterrows():
            rows.append(dict(a=idx.get(rf, rf), b=idx.get(ch.Feature, ch.Feature), gain=rg + float(ch.Gain)))
    d = pd.DataFrame(rows)
    d["block_a"] = d.a.map(block_of); d["block_b"] = d.b.map(block_of)
    d["block_pair"] = [" x ".join(sorted([x, y_])) for x, y_ in zip(d.block_a, d.block_b)]
    d["col_pair"] = [" x ".join(sorted([x, y_])) for x, y_ in zip(d.a, d.b)]
    bl = d.groupby("block_pair").gain.sum().sort_values(ascending=False)
    bl = (bl / bl.sum()).rename("gain_share").reset_index()
    bl.to_csv(OUT / "mechanism_pairs_blocks.csv", index=False)
    cp = d.groupby("col_pair").gain.sum().sort_values(ascending=False)
    cp = (cp / cp.sum()).rename("gain_share").head(top).reset_index()
    cp.to_csv(OUT / "mechanism_pairs_columns.csv", index=False)
    pd.set_option("display.width", 220)
    print("  interaction gain by block pair:")
    print(bl.head(12).round(4).to_string(index=False))
    print("\n  top specific column pairs:")
    print(cp.round(4).to_string(index=False))
    return bl, cp


# ======================================================================================================  h1, h2, h3, followups, audit
N_BOOT = 400
LUT = np.full(256, 255, np.uint8)
for _i, _c in enumerate("ACGT"):
    LUT[ord(_c)] = _i


def _data():
    """Genic guides only (so genes can be resampled): label, the seed base at positions 20 and 18, flank GC, flank sequences, gene."""
    from sgrna.harness.curated import load_groups
    from sgrna.build.build_features import load_curated_index
    from sgrna import config
    X, y, protos, A, an = _cohort()
    g, ok = load_groups("gene", protos)
    idx = load_curated_index(verbose=False).set_index(config.ID_COL).reindex(protos)
    col = {n: A[:, i] for i, n in enumerate(an)}
    gc = {w: (col[f"eng.flank.comp.up{w}.gc"] + col[f"eng.flank.comp.dn{w}.gc"]) / 2 for w in (50, 250, 500, 1000)}
    p = np.frombuffer("".join(protos).encode(), np.uint8).reshape(len(protos), 20)
    seq = {w: (idx["upstream"].str[-w:] + idx["downstream"].str[:w]).to_numpy() for w in (250, 500)}
    return dict(y=y[ok], gc=gc[250][ok], gc500=gc[500][ok], gc50=gc[50][ok], gc1000=gc[1000][ok], b20=p[ok, 19], b18=p[ok, 17], gene=g[ok], seq250=seq[250][ok], protos=protos[ok])


def _boot_index(gene, rng):
    """One resample of whole genes, as a row index."""
    u, inv = np.unique(gene.astype(str), return_inverse=True)
    rows = [np.flatnonzero(inv == k) for k in range(len(u))]
    return lambda: np.concatenate([rows[k] for k in rng.integers(0, len(u), len(u))])


def _spear(x, y):
    a, b = rankdata(x) - (len(x) + 1) / 2, rankdata(y) - (len(y) + 1) / 2
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def _ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ------------------------------------------------------------------------------- H1
def h1(d, rng):
    """Slope of label on flank GC (Spearman) within each base at position 20 and at 18."""
    draw = _boot_index(d["gene"], rng)
    rows, boots = [], {}
    for pos, key in ((20, "b20"), (18, "b18")):
        for b in "ACGT":
            m = d[key] == ord(b)
            rows.append(dict(position=pos, base=b, n=int(m.sum()), mean_label=float(d["y"][m].mean()), rho=_spear(d["gc"][m], d["y"][m]),
                             ols_slope=_slope(d["gc"][m].astype(float), d["y"][m].astype(float))))
            boots[(pos, b)] = []
    for _ in range(N_BOOT):
        ix = draw()
        for pos, key in ((20, "b20"), (18, "b18")):
            for b in "ACGT":
                m = d[key][ix] == ord(b)
                boots[(pos, b)].append(_spear(d["gc"][ix][m], d["y"][ix][m]))
    for r in rows:
        r["lo"], r["hi"] = _ci(boots[(r["position"], r["base"])])
    out = pd.DataFrame(rows)
    diffs = []
    for pos in (20, 18):
        sub = out[out.position == pos]
        hi_b, lo_b = sub.loc[sub.rho.idxmax(), "base"], sub.loc[sub.rho.idxmin(), "base"]
        dv = np.array(boots[(pos, hi_b)]) - np.array(boots[(pos, lo_b)])
        diffs.append(dict(position=pos, steepest=hi_b, shallowest=lo_b, difference=float(sub.rho.max() - sub.rho.min()),
                          lo=_ci(dv)[0], hi=_ci(dv)[1]))
        # the pre-specified contrast, whichever is steepest: T minus G
        dv_tg = np.array(boots[(pos, "T")]) - np.array(boots[(pos, "G")])
        diffs.append(dict(position=pos, steepest="T minus G", shallowest="", difference=float(sub.set_index("base").rho["T"] - sub.set_index("base").rho["G"]),
                          lo=_ci(dv_tg)[0], hi=_ci(dv_tg)[1]))
    out.to_csv(OUT / "hypotheses_h1_strata.csv", index=False)
    pd.DataFrame(diffs).to_csv(OUT / "hypotheses_h1_contrasts.csv", index=False)
    print("\nH1  Spearman of label on flank GC (250 nt, both sides), by the base at the seed position")
    print(out.round(3).to_string(index=False))
    print(pd.DataFrame(diffs).round(3).to_string(index=False))
    return out, pd.DataFrame(diffs)


# ------------------------------------------------------------------------------- H2
def _slope(x, y):
    xc = x - x.mean()
    return float((xc @ (y - y.mean())) / (xc @ xc))


def h2(d, rng):
    """Mean label across the flank-GC range, and the slope in the lowest and highest thirds of that range."""
    draw = _boot_index(d["gene"], rng)
    g = pd.factorize(d["gene"].astype(str))[0]
    res = {}
    for kind in ("pooled", "within genes"):
        x, y = d["gc"].astype(float), d["y"].astype(float)
        if kind == "within genes":
            x = x - pd.Series(x).groupby(g).transform("mean").to_numpy()
            y = y - pd.Series(y).groupby(g).transform("mean").to_numpy()

        def stats(ix):
            xs, ys = x[ix], y[ix]
            lo, hi = np.percentile(xs, [100 / 3, 200 / 3])
            ml, mh = xs <= lo, xs >= hi
            return _slope(xs[ml], ys[ml]), _slope(xs[mh], ys[mh]), _slope(xs, ys)
        point = stats(np.arange(len(y)))
        bs = np.array([stats(draw()) for _ in range(N_BOOT)])
        diff = bs[:, 0] - bs[:, 1]
        res[kind] = dict(kind=kind, slope_low_third=point[0], lo_low=_ci(bs[:, 0])[0], hi_low=_ci(bs[:, 0])[1],
                         slope_high_third=point[1], lo_high=_ci(bs[:, 1])[0], hi_high=_ci(bs[:, 1])[1], slope_all=point[2],
                         low_minus_high=point[0] - point[1], lo_diff=_ci(diff)[0], hi_diff=_ci(diff)[1])
        # the curve itself, in ten bins of the (possibly centred) variable
        q = pd.qcut(x, 10, labels=False, duplicates="drop")
        curve = pd.DataFrame({"bin": q, "x": x, "y": y}).groupby("bin").agg(flank_gc=("x", "mean"), mean_label=("y", "mean"), n=("y", "size"))
        curve.insert(0, "kind", kind)
        res[kind + "_curve"] = curve.reset_index()
    out = pd.DataFrame([res["pooled"], res["within genes"]])
    out.to_csv(OUT / "hypotheses_h2_thirds.csv", index=False)
    pd.concat([res["pooled_curve"], res["within genes_curve"]]).to_csv(OUT / "hypotheses_h2_curve.csv", index=False)
    print("\nH2  slope of the label on flank GC (label units per unit GC fraction)")
    print(out.round(3).to_string(index=False))
    print(pd.concat([res["pooled_curve"], res["within genes_curve"]]).round(3).to_string(index=False))
    return out


# ------------------------------------------------------------------------------- H3
def _trimer_classes(seq):
    """Counts of the 32 strand-collapsed trimer classes in each flank string, and the class index of every trimer."""
    n = len(seq)
    L = len(seq[0])
    codes = LUT[np.frombuffer("".join(seq).encode(), np.uint8)].reshape(n, L)
    idx = codes[:, :-2].astype(np.int32) * 16 + codes[:, 1:-1] * 4 + codes[:, 2:]
    ok = (codes[:, :-2] < 4) & (codes[:, 1:-1] < 4) & (codes[:, 2:] < 4)
    rc = np.zeros(64, int)
    for t in range(64):
        a, b, c = t // 16, (t // 4) % 4, t % 4
        rc[t] = (3 - c) * 16 + (3 - b) * 4 + (3 - a)
    canon = np.minimum(np.arange(64), rc)
    cls = np.unique(canon, return_inverse=True)[1]              # 32 classes
    cnt = np.zeros((n, 32))
    c = cls[np.where(ok, idx, 0)]
    for k in range(32):
        cnt[:, k] = ((c == k) & ok).sum(axis=1)
    names = {k: "/".join(sorted(["ACGT"[t // 16] + "ACGT"[(t // 4) % 4] + "ACGT"[t % 4] for t in range(64) if cls[t] == k])) for k in range(32)}
    return cnt / ok.sum(axis=1, keepdims=True), cls, names


def h3(d, rng):
    dens, cls, names = _trimer_classes(d["seq250"])
    pam_classes = [cls[16 * a + 2 * 4 + 2] for a in range(4)]        # N G G for each N
    assert len(set(pam_classes)) == 4, pam_classes
    pam = dens[:, pam_classes].sum(axis=1)
    y, gc = d["y"], d["gc"]
    marg_gc, marg_pam = _spear(gc, y), _spear(pam, y)
    pam_given_gc, gc_given_pam = _partial(pam, y, gc), _partial(gc, y, pam)
    # null: random sums of four of the 32 classes, each given GC
    null = []
    for _ in range(5000):
        pick = rng.choice(32, 4, replace=False)
        null.append(_partial(dens[:, pick].sum(axis=1), y, gc))
    null = np.array(null)
    pct = float((null < pam_given_gc).mean() * 100)
    per_class = pd.DataFrame({"class": [names[k] for k in range(32)], "partial_given_gc": [_partial(dens[:, k], y, gc) for k in range(32)]})
    per_class = per_class.sort_values("partial_given_gc", ascending=False)
    out = pd.DataFrame([dict(marginal_gc=marg_gc, marginal_pam=marg_pam, pam_given_gc=pam_given_gc, gc_given_pam=gc_given_pam,
                             null_mean=float(null.mean()), null_sd=float(null.std()), null_p2_5=float(np.percentile(null, 2.5)),
                             null_p97_5=float(np.percentile(null, 97.5)), percentile_of_pam=pct, corr_pam_gc=_spear(pam, gc))])
    out.to_csv(OUT / "hypotheses_h3_pam_density.csv", index=False)
    per_class.to_csv(OUT / "hypotheses_h3_trimer_classes.csv", index=False)
    print("\nH3  PAM-site density against flank GC")
    print(out.round(4).T.to_string(header=False))
    print("  top and bottom trimer classes by partial association given GC:")
    print(per_class.head(4).round(4).to_string(index=False)); print(per_class.tail(4).round(4).to_string(index=False))
    return out


# ------------------------------------------------------------------------------- follow-ups (exploratory, after H2)
def _deficit(gc, y, cluster, rng, q=0.10, boot=N_BOOT):
    """Mean label in the lowest `q` of flank GC minus the mean elsewhere, with a cluster bootstrap."""
    def one(ix):
        g, v = gc[ix], y[ix]
        cut = np.quantile(g, q)
        return float(v[g <= cut].mean() - v[g > cut].mean()), float(cut)
    point, cut = one(np.arange(len(y)))
    u, inv = np.unique(cluster.astype(str), return_inverse=True)
    rows = [np.flatnonzero(inv == k) for k in range(len(u))]
    bs = [one(np.concatenate([rows[k] for k in rng.integers(0, len(u), len(u))]))[0] for _ in range(boot)]
    return point, cut, _ci(bs)


def _cohort_gc(name):
    """Flank GC (250 nt, both sides), label and 100 kb block for one crisprHAL dataset, from the stored family-A columns."""
    import json
    from sgrna import config
    meta = json.loads((config.INTERIM / f"transfer_{name}" / "meta.json").read_text())
    a = pd.read_csv(config.INTERIM / "features" / f"a_flank_{name}.csv", usecols=["sgRNAID", "eng.flank.comp.up250.gc", "eng.flank.comp.dn250.gc"])
    if "protospacers" in meta:
        a = a.set_index("sgRNAID").reindex(meta["protospacers"])
    gc = ((a["eng.flank.comp.up250.gc"] + a["eng.flank.comp.dn250.gc"]) / 2).to_numpy()
    y = np.asarray(meta["y"], float)
    block = (np.asarray(meta["left"], float) // 100_000).astype(int)
    ok = np.isfinite(gc) & np.isfinite(y)
    return gc[ok], y[ok], block[ok]


def followups(d, rng):
    y, gc, gene = d["y"], d["gc"], d["gene"]
    out = {}
    # (a) where does the step sit, and does it depend on the window? finer curve, three window widths
    rows = []
    for w, v in (("250", d["gc"]), ("500", d["gc500"])):
        q = pd.qcut(v, 20, labels=False, duplicates="drop")
        for b, grp in pd.DataFrame({"b": q, "x": v, "y": y}).groupby("b"):
            rows.append(dict(window=w, bin=int(b), flank_gc=float(grp.x.mean()), mean_label=float(grp.y.mean()), n=len(grp)))
    pd.DataFrame(rows).to_csv(OUT / "hypotheses_followup_curve20.csv", index=False)
    # (b) where does the association live? Spearman of the label with flank GC *inside* the part of the range below, and above, a threshold
    rows = []
    for qv in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        c = float(np.quantile(gc, qv)); lo_m = gc <= c
        rows.append(dict(quantile=qv, threshold_gc=c, n_below=int(lo_m.sum()), rho_below=_spear(gc[lo_m], y[lo_m]),
                         n_above=int((~lo_m).sum()), rho_above=_spear(gc[~lo_m], y[~lo_m]), rho_full=_spear(gc, y)))
    hinge = pd.DataFrame(rows)
    hinge.to_csv(OUT / "hypotheses_followup_hinge.csv", index=False)
    # (g) two components? within-gene association (both variables centred on the gene mean) among guides ABOVE the AT-rich tail
    gi = pd.factorize(gene.astype(str))[0]
    xc = gc - pd.Series(gc).groupby(gi).transform("mean").to_numpy(); yc = y - pd.Series(y).groupby(gi).transform("mean").to_numpy()
    rows = []
    for qv in (0.0, 0.10, 0.15, 0.20):
        c = float(np.quantile(gc, qv)) if qv > 0 else -1.0
        # a gene's mean is taken over all its guides; the selection is on the guide's own flank GC
        m = gc > c
        rows.append(dict(excluded_lowest_share=qv, n=int(m.sum()), rho_within_genes=_spear(xc[m], yc[m])))
    within = pd.DataFrame(rows)
    within.to_csv(OUT / "hypotheses_followup_within.csv", index=False)
    # (c) robustness: the lowest decile's deficit with each 100 kb block left out, and how concentrated the decile is in genes
    pt, cut, ci = _deficit(gc, y, gene, rng)
    low = gc <= cut
    cnt = pd.Series(gene[low].astype(str)).value_counts()
    conc = dict(deficit=pt, lo=ci[0], hi=ci[1], flank_gc_at_decile_edge=cut, n_lowest_decile=int(low.sum()), genes_in_lowest_decile=int(len(cnt)),
                top10_genes_share=float(cnt.head(10).sum() / low.sum()))
    # leave out the ten genes that contribute most guides to the lowest decile
    keep = ~pd.Series(gene.astype(str)).isin(cnt.head(10).index).to_numpy()
    pt10, cut10, _ = _deficit(gc[keep], y[keep], gene[keep], rng, boot=2)
    conc["deficit_without_top10_genes"] = pt10
    pd.DataFrame([conc]).to_csv(OUT / "hypotheses_followup_concentration.csv", index=False)
    cnt.head(15).rename("guides_in_lowest_decile").to_frame().to_csv(OUT / "hypotheses_followup_top_genes.csv")
    # (d) replication: a different enzyme on the same genome (same loci, so not independent), and a different genome
    rep = []
    for name, label in (("WT-SpCas9", "E. coli, WT SpCas9 (all 33,567 guides)"), ("eSpCas9", "E. coli, eSpCas9 (same loci)"),
                        ("TevSpCas9", "C. rodentium, TevSpCas9 (a different genome)")):
        g_, y_, b_ = _cohort_gc(name)
        pt_, cut_, ci_ = _deficit(g_, y_, b_, rng)
        rep.append(dict(dataset=label, n=len(y_), mean_flank_gc=float(g_.mean()), flank_gc_at_decile_edge=cut_, deficit=pt_, lo=ci_[0], hi=ci_[1],
                        label_sd=float(y_.std()), deficit_in_sd=pt_ / float(y_.std()), rho_all=_spear(g_, y_)))
    rep = pd.DataFrame(rep)
    rep.to_csv(OUT / "hypotheses_followup_replication.csv", index=False)
    # (e) is the AT-rich penalty just low expression? expression of the guide's gene (family G), strata of equal size
    from sgrna.harness.curated import load_groups
    from sgrna import config
    ex = pd.read_csv(config.INTERIM / "features" / "g_transcription_curated.csv", usecols=["sgRNAID", "eng.txn.expr.median_log_tpm"]).set_index("sgRNAID")
    X_, y_all, protos_all, _, _ = _cohort()
    ok_ = load_groups("gene", protos_all)[1]
    e = ex["eng.txn.expr.median_log_tpm"].reindex(protos_all[ok_]).to_numpy(float)
    have = np.isfinite(e)
    rows = []
    cut_all = float(np.quantile(gc, 0.10))
    tert = pd.qcut(e[have], 3, labels=["low expression", "middle", "high expression"])
    for k in ("low expression", "middle", "high expression"):
        m = np.zeros(len(gc), bool); m[np.flatnonzero(have)[(tert == k)]] = True
        low = m & (gc <= cut_all)
        rows.append(dict(stratum=k, n=int(m.sum()), n_lowest_decile=int(low.sum()), share_lowest_decile=float(low.sum() / m.sum()),
                         mean_expression=float(e[m].mean()), deficit=float(y[low].mean() - y[m & ~low].mean())))
    expr = pd.DataFrame(rows)
    expr.to_csv(OUT / "hypotheses_followup_expression.csv", index=False)
    marg = _spear(gc[have], y[have]); part = _partial(gc[have], y[have], e[have])
    expr_summary = dict(rho_gc_expression=_spear(gc[have], e[have]), rho_expression_label=_spear(e[have], y[have]), gc_label_marginal=marg, gc_label_given_expression=part,
                        mean_expression_lowest_decile=float(e[have & (gc <= cut_all)].mean()), mean_expression_rest=float(e[have & (gc > cut_all)].mean()))
    pd.DataFrame([expr_summary]).to_csv(OUT / "hypotheses_followup_expression_partial.csv", index=False)
    # (f) the absolute threshold in the other genome: guides below E. coli's decile edge, wherever they are
    g_t, y_t, b_t = _cohort_gc("TevSpCas9")
    low_t = g_t <= cut_all
    pt_t = float(y_t[low_t].mean() - y_t[~low_t].mean())
    bs = []
    u, inv = np.unique(b_t, return_inverse=True)
    rows_t = [np.flatnonzero(inv == k) for k in range(len(u))]
    for _ in range(N_BOOT):
        ix = np.concatenate([rows_t[k] for k in rng.integers(0, len(u), len(u))])
        m = low_t[ix]
        bs.append(float(y_t[ix][m].mean() - y_t[ix][~m].mean()) if m.sum() > 5 else np.nan)
    absolute = pd.DataFrame([dict(dataset="C. rodentium, TevSpCas9", absolute_threshold_gc=cut_all, n_below=int(low_t.sum()), share_below=float(low_t.mean()),
                                  deficit=pt_t, lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)))])
    absolute.to_csv(OUT / "hypotheses_followup_absolute.csv", index=False)
    print("\nFOLLOW-UP (exploratory)  the AT-rich step")
    print("  (b) Spearman retained when flank GC is clipped at a threshold:"); print(hinge.round(3).to_string(index=False))
    print("  (g) within-gene association once the AT-rich tail is set aside:"); print(within.round(3).to_string(index=False))
    print("  (c) concentration:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in conc.items()})
    print(cnt.head(12).to_string())
    print("  (d) replication of the lowest-decile deficit:"); print(rep.round(3).to_string(index=False))
    print("  (e) expression strata:"); print(expr.round(3).to_string(index=False)); print({k: round(v, 3) for k, v in expr_summary.items()})
    print("  (f) E. coli's absolute threshold applied to C. rodentium:"); print(absolute.round(3).to_string(index=False))
    return rep



# ------------------------------------------------------------------------------- audit of the readings (exploratory)
def audit(d, rng):
    """Two checks made to see how far two readings of H1/H2 are actually supported, designed after the fact.

    (a) the scale of the step: the 'accessibility of AT-rich regions' reading, and in particular a nucleoid-protein reading, is about
        regions, so it predicts the step is at least as sharp when flank GC is measured over a kilobase as over 50 nt. A purely local
        reading predicts the opposite.
    (b) the 'gate' reading of H1: the shallower flank slope beside an unfavourable seed base could be a compressed label (a floor) rather
        than a gate. Compare the spread of the label by seed base, and the slope among guides above the global lower quartile of the label.
    """
    y = d["y"]
    rows = []
    for w, key in ((50, "gc50"), (250, "gc"), (500, "gc500"), (1000, "gc1000")):
        x = d[key]
        c = float(np.quantile(x, 0.10)); lo = x <= c
        rows.append(dict(window_nt=w, flank_gc_at_decile_edge=c, deficit=float(y[lo].mean() - y[~lo].mean()), deficit_in_sd=float((y[lo].mean() - y[~lo].mean()) / y.std()),
                         rho_below=_spear(x[lo], y[lo]), rho_above=_spear(x[~lo], y[~lo]), rho_full=_spear(x, y)))
    scale = pd.DataFrame(rows)
    scale.to_csv(OUT / "hypotheses_audit_scale.csv", index=False)
    q25 = float(np.quantile(y, 0.25))
    rows = []
    for b in "ACGT":
        m = d["b20"] == ord(b)
        up = m & (y > q25)
        rows.append(dict(base_at_20=b, n=int(m.sum()), label_sd=float(y[m].std()), share_in_lowest_quartile=float((y[m] <= q25).mean()),
                         rho_all=_spear(d["gc"][m], y[m]), rho_above_global_q25=_spear(d["gc"][up], y[up]), n_above=int(up.sum())))
    gate = pd.DataFrame(rows)
    gate.to_csv(OUT / "hypotheses_audit_gate.csv", index=False)
    print("\nAUDIT (a) the step at four scales of flank GC"); print(scale.round(3).to_string(index=False))
    print("AUDIT (b) is the shallower slope beside a bad seed base a compressed label?"); print(gate.round(3).to_string(index=False))
    return scale, gate


# ======================================================================================================  resolution, relative
STRIDE = 37            # genome-wide window sampling stride; coprime with everything to avoid phase artefacts
BINS = (1, 2, 4, 8, 20, 50)


# ------------------------------------------------------------------ genome window distributions
def _seq(fasta) -> np.ndarray:
    raw = []
    with open(fasta) as fh:
        for line in fh:
            if not line.startswith(">"):
                raw.append(line.strip().upper())
    return np.frombuffer("".join(raw).encode("ascii"), np.uint8)


def window_dist(fasta, w: int, stat: str) -> np.ndarray:
    """Sorted distribution of `stat` over every w-nt window of the genome (sampled at STRIDE)."""
    a = _seq(fasta)
    if stat == "gc":
        hit = (a == ord("G")) | (a == ord("C"))
    elif stat == "purine":
        hit = (a == ord("A")) | (a == ord("G"))
    else:
        raise ValueError(stat)
    cs = np.concatenate([[0], np.cumsum(hit, dtype=np.int64)])
    starts = np.arange(0, len(a) - w, STRIDE)
    return np.sort((cs[starts + w] - cs[starts]) / w)


def to_percentile(vals: np.ndarray, dist: np.ndarray) -> np.ndarray:
    out = np.searchsorted(dist, vals) / len(dist)
    return np.where(np.isfinite(vals), out, np.nan)


# ------------------------------------------------------------------ experiment 1: species-relative
def _comp_cols(names):
    """(index, window, side, stat) for every raw GC / purine composition column of family A."""
    out = []
    for j, n in enumerate(names):
        p = n.split(".")
        if len(p) == 5 and p[0] == "eng" and p[1] == "flank" and p[2] == "comp" and p[4] in ("gc", "purine"):
            side_w = p[3]
            side, w = side_w[:2], int(side_w[2:])
            out.append((j, w, side, p[4]))
    return out


def relative(seed: int = 41):
    from sgrna.harness.curated import load_base, load_block
    from sgrna.compare import transfer

    X_ec, y_ec, names, protos = load_base()
    A_ec, an = load_block("a_flank", protos)
    X_ec = np.hstack([X_ec, A_ec]).astype(np.float32)
    all_names = list(names) + list(an)

    X_cr, y_cr, cr_names = transfer.build("TevSpCas9", genome_fasta=config.EXTERNAL / "C_rodentium" / "NC_013716.1.fasta")
    X_cr = np.asarray(X_cr, np.float32)
    assert list(cr_names) == all_names, "the two cohorts' column orders differ; cannot transfer column-wise"

    cols = _comp_cols(all_names)
    print(f"{len(cols)} raw composition columns to relativise", flush=True)
    ec_fa, cr_fa = config.GENOME_FASTA, config.EXTERNAL / "C_rodentium" / "NC_013716.1.fasta"
    Xr_ec, Xr_cr = X_ec.copy(), X_cr.copy()
    for j, w, side, stat in cols:
        d_ec = window_dist(ec_fa, w, stat)
        d_cr = window_dist(cr_fa, w, stat)
        Xr_ec[:, j] = to_percentile(X_ec[:, j], d_ec)
        Xr_cr[:, j] = to_percentile(X_cr[:, j], d_cr)
    print(f"genome GC: E. coli {_gc(ec_fa):.4f}, C. rodentium {_gc(cr_fa):.4f}", flush=True)

    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    cfg = dict(cfgs["d099x"], n_estimators=int(T.selected().loc["d099x", "k"]))
    rows = []
    for label, Xe, Xc in (("absolute (as shipped)", X_ec, X_cr), ("species-relative percentile", Xr_ec, Xr_cr)):
        A, _ = _impute(Xe, Xe[:1])
        sel = make_selector(seed); sel.fit(A, y_ec)
        top = np.argsort(sel.feature_importances_)[::-1][: int(cfg["nf"])]
        Atr, Bte = _impute(Xe[:, top], Xc[:, top])
        p = T.fit_predict(cfg, Atr, y_ec, Bte, seed)[cfg["n_estimators"]]
        rho_cr = float(spearmanr(y_cr, p).statistic)
        # home reference, same columns, gene-grouped CV, to confirm the monotone no-op
        home = []
        g, ok = _groups_home(protos)
        for tr, va in grouped_folds(g[ok], 5, seed):
            idx = np.where(ok)[0]
            a, b = _impute(Xe[idx[tr]][:, top], Xe[idx[va]][:, top])
            q = T.fit_predict(cfg, a, y_ec[idx[tr]], b, seed)[cfg["n_estimators"]]
            home.append(float(spearmanr(y_ec[idx[va]], q).statistic))
        rows.append(dict(columns=label, crod_rho=rho_cr, home_cv_rho=float(np.mean(home)),
                         n_crod=len(y_cr), n_home=int(ok.sum())))
        print(f"  {label:28s} C. rodentium {rho_cr:.4f}   home CV {np.mean(home):.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "novel_relative_composition.csv", index=False)
    d = out.set_index("columns")
    print(f"\n  transfer difference (relative - absolute): "
          f"{d.loc['species-relative percentile','crod_rho'] - d.loc['absolute (as shipped)','crod_rho']:+.4f}")
    print(f"  home difference (should be ~0 by monotone invariance): "
          f"{d.loc['species-relative percentile','home_cv_rho'] - d.loc['absolute (as shipped)','home_cv_rho']:+.4f}")
    return out


def _gc(fasta):
    a = _seq(fasta)
    return float(((a == ord("G")) | (a == ord("C"))).mean())


def _groups_home(protos):
    from sgrna.harness.curated import load_groups
    g, ok = load_groups("gene", protos)
    return np.asarray(g, dtype=object), ok


# ------------------------------------------------------------------ experiment 2: resolution floor
def _binned(index, nbins, flank=1000):
    """GC and purine fraction in `nbins` equal bins of the 1,000 nt each side. 4*nbins columns."""
    ups = index["upstream"].astype(str).to_numpy()
    dns = index["downstream"].astype(str).to_numpy()
    n = len(index)
    out = np.full((n, 4 * nbins), np.nan, np.float32)
    w = flank // nbins
    for side, seqs, off in (("up", ups, 0), ("dn", dns, 2 * nbins)):
        keep = np.array([len(s) >= flank for s in seqs])
        if not keep.any():
            continue
        sub = [(s[-flank:] if side == "up" else s[:flank]) for s in seqs[keep]]
        a = np.frombuffer("".join(sub).encode("ascii"), np.uint8).reshape(keep.sum(), flank)
        gc = ((a == ord("G")) | (a == ord("C"))).astype(np.float32)
        pu = ((a == ord("A")) | (a == ord("G"))).astype(np.float32)
        for b in range(nbins):
            sl = slice(b * w, (b + 1) * w)
            out[keep, off + b] = gc[:, sl].mean(1)
            out[keep, off + nbins + b] = pu[:, sl].mean(1)
    return out


def resolution(seeds=(41, 42, 43)):
    from sgrna.harness.curated import load_base, load_block
    X0, y, names, protos = load_base()
    A, an = load_block("a_flank", protos)
    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv").set_index(config.ID_COL)
    idx = idx.reindex(list(protos)).reset_index()
    comp = [j for j, n in enumerate(an) if n.startswith("eng.flank.comp.")]
    co = T.Cohort()
    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    cfg = dict(cfgs["d099x"], n_estimators=int(T.selected().loc["d099x", "k"]))

    g, ok = _groups_home(protos)
    arms = {"base only": X0, "base + family A comp block (24 cols)": np.hstack([X0, A[:, comp]]),
            "base + all of family A (348 cols)": np.hstack([X0, A])}
    for b in BINS:
        arms[f"base + {4*b} binned cols ({1000//b} nt bins)"] = np.hstack([X0, _binned(idx, b)])

    rows = []
    for label, Xa in arms.items():
        Xa = Xa.astype(np.float32)
        per = []
        for seed in seeds:
            for tr, va in grouped_folds(g[ok], 5, seed):
                i = np.where(ok)[0]
                a0, _ = _impute(Xa[i[tr]], Xa[i[va]][:1])
                sel = make_selector(seed); sel.fit(a0, y[i[tr]])
                top = np.argsort(sel.feature_importances_)[::-1][: int(cfg["nf"])]
                a, bb = _impute(Xa[i[tr]][:, top], Xa[i[va]][:, top])
                p = T.fit_predict(cfg, a, y[i[tr]], bb, seed)[cfg["n_estimators"]]
                per.append(float(spearmanr(y[i[va]], p).statistic))
        rows.append(dict(arm=label, n_cols=Xa.shape[1] - X0.shape[1], rho=float(np.mean(per)),
                         sd=float(np.std(per, ddof=1)), n_fits=len(per)))
        print(f"  {label:44s} rho {rows[-1]['rho']:.4f} +- {rows[-1]['sd']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "novel_resolution.csv", index=False)
    return pd.DataFrame(rows)


LOCAL = ("distance", "bases", "asymmetry", "human", "pairs")
HYPOTHESES = ("h1", "h2", "h3", "followups", "audit")
NOVEL = ("resolution", "relative")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--test", nargs="+", default=["all"], choices=["all", *LOCAL, *HYPOTHESES, *NOVEL])
    a = ap.parse_args(argv)
    want = lambda name: "all" in a.test or name in a.test
    titles = {"distance": "signed flank-distance profile", "bases": "position x base preferences, straight from sequence", "asymmetry": "upstream against downstream composition",
              "human": "the same estimator on the human matrix", "pairs": "which column pairs carry the interaction"}
    fns = {"distance": distance, "bases": bases, "asymmetry": asymmetry, "human": human_bases, "pairs": pairs}
    for name in LOCAL:
        if want(name):
            print(f"\n=== {titles[name]} ===", flush=True)
            fns[name]()
    if any(want(n) for n in HYPOTHESES):
        d = _data()
        print(f"{len(d['y']):,} genic guides in {len(set(d['gene'].astype(str))):,} genes; label SD {d['y'].std():.3f}")
        for name, fn in (("h1", h1), ("h2", h2), ("h3", h3)):
            if want(name):
                fn(d, np.random.default_rng(41))
        if want("followups"):
            followups(d, np.random.default_rng(41))
        if want("audit"):
            audit(d, np.random.default_rng(41))
    if any(want(n) for n in NOVEL):
        pd.set_option("display.width", 200)
        if want("resolution"):
            print("\n=== resolution floor of the flank signal ===", flush=True)
            resolution()
        if want("relative"):
            print("\n=== species-relative composition and cross-organism transfer ===", flush=True)
            relative()


if __name__ == "__main__":
    main()
