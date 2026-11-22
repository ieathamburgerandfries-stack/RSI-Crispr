"""Biology out of the model, by routes that Finding 5's redundancy caution does not disqualify.

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

    python -m sgrna.interpret.mechanism --test bases
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from sgrna import config

from ..common import OUT

BASES = "ACGT"


def _cohort():
    from sgrna.cohorts.curated_ablation import _base, _block
    X, y, names, protos = _base()
    A, an = _block("a_flank", protos)
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
    from sgrna.cohorts.curated_ablation import _groups
    g, ok = _groups("gene", protos)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["bases", "distance", "asymmetry", "human", "pairs", "all"], default="all")
    a = ap.parse_args()
    if a.test in ("distance", "all"):
        print("\n=== signed flank-distance profile ===", flush=True)
        distance()
    if a.test in ("bases", "all"):
        print("\n=== position x base preferences, straight from sequence ===", flush=True)
        bases()
    if a.test in ("asymmetry", "all"):
        print("\n=== upstream against downstream composition ===", flush=True)
        asymmetry()
    if a.test in ("human", "all"):
        print("\n=== the same estimator on the human matrix ===", flush=True)
        human_bases()
    if a.test in ("pairs", "all"):
        print("\n=== which column pairs carry the interaction ===", flush=True)
        pairs()




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
    from sgrna.cohorts.curated_ablation import _base, _block
    X0, y, names, protos = _base()
    A, an = _block("a_flank", protos)
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

if __name__ == "__main__":
    main()
