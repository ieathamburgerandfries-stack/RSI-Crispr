"""Checks behind review R3 of PROGRESS_REPORT.md (Finding 5, interpretability; Finding 6, the label's ceiling).

Same idiom as `paper_numbers_C.py`: `chk(label, got, claimed, dec)` passes when |got - claimed| <= 0.5 * 10**(-dec); `got` is always
computed from a stored file or by re-running a short piece of the producing code (about two minutes in all), `claimed` is the value as
the report writes it. The report's claims here sit on four different protocols, so the label of each check says which:

  importance across model families, the drop-and-refit probe   the published 13,880-row matrix + family A (6,580 columns), random-row
                                                               three-fold CV, seed 41 (`importance_model_*.csv`, `faithfulness_*.csv`)
  the three tests that need no refit                           the curated 33,567-guide cohort, one fit of the production recipe on a
                                                               random 80% (`interp_extra.py`)
  the compact model                                            the 30,203 genic guides, gene-grouped folds (`compact_model_*.csv`)
  SLICER against crisprHAL 2                                   crisprHAL 2's WT-SpCas9 split, 100 or 1,000 or 1,500 test guides
                                                               (`interp_compare_*.csv`)
  the ceiling                                                  crisprHAL 2's per-replicate counts (`curated_ceiling.json`)

    ./run_ext.sh sgrna.paper.report_checks_R3
"""
from __future__ import annotations

import os

import json
import math
import zlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..common import OUT
from sgrna import config

P = "report R3: "


def _binary(block):
    """The binary-indicator kinds: the guide's one-hot columns and the flank's bases within 10 nt."""
    return block.startswith("protospacer indicators") or block == "flank: bases within 10 nt"


def _importance(chk, RES, MAIN):
    print("R3 importance: the gain artefact, and what the faithfulness probe reads")
    b = pd.read_csv(RES / "importance_model_blocks.csv")
    b["binary"] = b.block.map(_binary)
    s = (b[b.binary].groupby(["method", "model", "fold"]).share.sum().groupby(["method", "model"]).mean() * 100)
    chk(P + "XGBoost share on binary indicators, gain (published matrix + A, 3 folds)", float(s[("native", "xgboost")]), 57, 0)
    chk(P + "XGBoost share on binary indicators, SHAP", float(s[("shap", "xgboost")]), 31, 0)
    chk(P + "LightGBM share on binary indicators, SHAP", float(s[("shap", "lightgbm")]), 31, 0)
    chk(P + "blocks in the seven-family importance run", float(b.block.nunique()), 14, 0)
    ag = pd.read_csv(MAIN / "attribution_summary.csv").set_index("kind")
    chk(P + "two-model run, gain: guide-indicator share, XGBoost (%)", float(ag.loc["target: position/k-mer indicator", "xgboost_share"] * 100), 49, 0)
    chk(P + "two-model run, gain: guide-indicator share, LightGBM (%)", float(ag.loc["target: position/k-mer indicator", "lightgbm_share"] * 100), 18, 0)
    chk(P + "two-model run, gain: largest gap on any of six kinds (points)", float(ag["difference"].abs().max() * 100), 31, 0)
    ags = pd.read_csv(MAIN / "attribution_summary_shap.csv")
    chk(P + "two-model run, SHAP: kinds of column", float(len(ags)), 6, 0)
    chk(P + "two-model run, SHAP: largest gap on any kind (points)", float(ags["difference"].abs().max() * 100), 0.3, 1)

    # the drop-and-refit probe
    red = pd.read_csv(RES / "faithfulness_groups_reduced.csv")
    top = red[red.arm == "top 20 columns"].delta
    chk(P + "reduced 775-column set: refit without the top 20 SHAP columns, mean change in rho", float(top.mean()), -0.015, 3)
    chk(P + "... the largest of the six model-fold fits", float(top.max()), -0.011, 3)
    chk(P + "... the smallest of the six", float(top.min()), -0.020, 3)
    chk(P + "reduced set: without 20 random columns", float(red[red.arm == "random 20 columns"].delta.mean()), 0.001, 3)
    full = pd.read_csv(RES / "faithfulness_groups.csv")
    thing = full[full.arm == "top things removed"].n_columns_removed
    chk(P + "full matrix: fewest columns in the 'top things removed' arm", float(thing.min()), 3, 0)
    chk(P + "full matrix: most columns in the 'top things removed' arm", float(thing.max()), 12, 0)
    chk(P + "full matrix: 'top things removed', largest |mean change| over the depths", float(full[full.arm == "top things removed"].groupby("things_removed").delta.mean().abs().max()), 0.0, 2)
    sh = pd.read_csv(MAIN / "faithfulness_shap_share.csv")
    g = lambda rep, n: sh[(sh.representation == rep) & (sh.top_n.astype(str) == str(n))].iloc[0]
    chk(P + "share of SHAP in the top 20, full matrix (%)", float(g("full", 20).pct_of_shap), 34.6, 1)
    chk(P + "share of SHAP in the top 20, reduced set (%)", float(g("reduced", 20).pct_of_shap), 37.5, 1)
    half = sh[sh.top_n.astype(str).str.startswith("n for")]
    chk(P + "columns holding half of the SHAP, full matrix", float(half[half.representation == "full"].pct_of_shap.iloc[0]), 44, 0)
    chk(P + "columns holding half of the SHAP, reduced set", float(half[half.representation == "reduced"].pct_of_shap.iloc[0]), 37, 0)
    ft = OUT / "faithfulness_top20_full.csv"
    if ft.exists():
        t = pd.read_csv(ft)
        chk(P + "full matrix: refit without the top 20 SHAP columns, mean change in rho (6 model-fold fits)", float(t.delta.mean()), -0.002, 3)
        chk(P + "... the largest of the six", float(t.delta.max()), 0.002, 3)
        chk(P + "... the smallest of the six", float(t.delta.min()), -0.005, 3)
        chk(P + "model-fold fits behind it", float(len(t)), 6, 0)
    else:
        print(f"  (skipped: {ft.name} is not installed; the report's full-matrix figure rests on it)")
    from ..interpret.faithfulness import reduced_columns
    from ..build.build_matrix import load_dataset
    _, _, names, _, fam = load_dataset(["a_flank"], verbose=False)
    chk(P + "columns of the full matrix (published + family A)", float(len(names)), 6580, 0)
    chk(P + "columns of the reduced set (364 + 63 + 348)", float(len(reduced_columns(list(names), fam))), 775, 0)
    ip = pd.read_csv(RES / "interpretability.csv")
    chk(P + "drop-top-20 probe, four models, largest |faithfulness|", float(ip.faithfulness.abs().max()), 0.02, 2)
    chk(P + "drop-top-20 probe: models", float(ip.model.nunique()), 4, 0)
    pr = pd.read_csv(RES / "importance_model_pairs.csv")
    nat = pr[pr.method == "native"].copy()
    nat["pair"] = nat.model_a + "-" + nat.model_b
    boost = {"xgboost", "lightgbm", "catboost"}
    bb = nat[nat.model_a.isin(boost) & nat.model_b.isin(boost)].rho_column.mean()
    tr = nat[nat.model_a.isin(boost | {"random_forest", "extra_trees"}) & nat.model_b.isin(boost | {"random_forest", "extra_trees"})].rho_column.mean()
    chk(P + "adding the forests lowers the default per-column agreement (boosters only, then five tree families)", float(bb > tr), 1, 0)


def _interp_extra(chk, RES):
    print("R3 the three tests that need no refit (interp_extra.py)")
    from ..interpret import interp_extra as ie
    f = ie._fit(verbose=False)
    chk(P + "production recipe: held-out rho on a random 20% of the curated cohort", f["rho"], 0.712, 3)
    chk(P + "... guides fitted on", float(len(f["tr"])), 26853, 0)
    chk(P + "... guides held out", float(len(f["te"])), 6714, 0)
    chk(P + "columns of the production matrix (6,169 base + 348 family A)", float(f["nbase"] + len(f["an"])), 6517, 0)
    sn = pd.read_csv(RES / "interp_sensitivity_n.csv").set_index(["unit", "n", "attribution"]).pearson
    for (u, n, a), v in (((("column", 1, "SHAP"), 0.928), (("column", 1, "ridge surrogate"), 0.079), (("column", 1, "SHAP, columns permuted"), -0.000),
                          (("column", 100, "SHAP"), 0.957), (("column", 100, "ridge surrogate"), 0.210), (("column", 100, "SHAP, columns permuted"), 0.007),
                          (("block", 1, "SHAP"), 0.956), (("block", 1, "ridge surrogate"), 0.396), (("block", 1, "SHAP, columns permuted"), -0.011),
                          (("block", 6, "SHAP"), 0.969), (("block", 6, "ridge surrogate"), 0.423), (("block", 6, "SHAP, columns permuted"), 0.038))):
        chk(P + f"sensitivity-n, {u}s, n = {n}, {a}", float(sn[(u, n, a)]), v, 3)
    it = pd.read_csv(RES / "interp_interactions.csv").iloc[0]
    chk(P + "interaction test: rows", float(it.rows), 400, 0)
    chk(P + "interaction test: single-column terms, share of the variance (%)", float(it.share_variance_main * 100), 70.2, 1)
    chk(P + "interaction test: pairwise terms, share of the variance (%)", float(it.share_variance_interaction * 100), 14.7, 1)
    chk(P + "interaction test: off-diagonal share of the absolute mass (%)", float(it.share_abs_mass_offdiagonal * 100), 52.7, 1)
    B = pd.read_csv(RES / "interp_interaction_blocks.csv", index_col=0)
    tot, bd = B.values.sum(), float(np.trace(B.values))
    chk(P + "interaction test: between-block share of the absolute mass (%)", float((1 - bd / tot) * 100), 44.2, 1)
    chk(P + "interaction test: within-block off-diagonal share (%)", float((bd / tot - (1 - it.share_abs_mass_offdiagonal)) * 100), 8.5, 1)
    off = B.values.copy(); np.fill_diagonal(off, 0)
    idx = list(B.index)
    pro = [i for i, k in enumerate(idx) if k.startswith("protospacer")]
    fla = [i for i, k in enumerate(idx) if k.startswith("flank")]
    pp = sum(off[i, j] for i in pro for j in pro); ff = sum(off[i, j] for i in fla for j in fla); pf = sum(off[i, j] for i in pro for j in fla) * 2
    chk(P + "between-block interaction mass: two protospacer blocks (%)", float(pp / off.sum() * 100), 46, 0)
    chk(P + "between-block interaction mass: two flank blocks (%)", float(ff / off.sum() * 100), 15, 0)
    chk(P + "between-block interaction mass: a flank and a protospacer block (%)", float(pf / off.sum() * 100), 39, 0)
    pairs = pd.read_csv(RES / "interp_interaction_pairs.csv")
    chk(P + "strongest interaction pair is p18dimer.Hbond.energyraw x p19dimer.Hbond.stackingraw", float(
        {pairs.col_i[0], pairs.col_j[0]} == {"p18dimer.Hbond.energyraw", "p19dimer.Hbond.stackingraw"}), 1, 0)
    chk(P + "... its mean |interaction|", float(pairs.mean_abs_interaction[0]), 0.034, 3)
    chk(P + "second pair is p20basepair.Hbond.energyraw x p19dimer.HLgap.eVEraw, mean |interaction|", float(
        pairs.mean_abs_interaction[1] if {pairs.col_i[1], pairs.col_j[1]} == {"p20basepair.Hbond.energyraw", "p19dimer.HLgap.eVEraw"} else -1), 0.032, 3)
    cf = pd.read_csv(RES / "interp_counterfactual.csv").set_index("edit")
    chk(P + "flank edit, GC +5 points", float(cf.loc["GC +5 points, both flanks", "mean_change"]), 0.46, 2)
    chk(P + "... lower end", float(cf.loc["GC +5 points, both flanks", "ci_lo"]), 0.42, 2)
    chk(P + "... upper end", float(cf.loc["GC +5 points, both flanks", "ci_hi"]), 0.50, 2)
    chk(P + "flank edit, GC -5 points", float(cf.loc["GC -5 points, both flanks", "mean_change"]), -0.88, 2)
    chk(P + "... lower end", float(cf.loc["GC -5 points, both flanks", "ci_lo"]), -0.93, 2)
    chk(P + "... upper end", float(cf.loc["GC -5 points, both flanks", "ci_hi"]), -0.83, 2)
    chk(P + "label sd", float(cf.label_sd.iloc[0]), 3.29, 2)
    chk(P + "flank edit, 5% of bases swapped at fixed GC", float(cf.loc["5% of bases swapped, GC kept", "mean_change"]), 0.003, 3)
    chk(P + "flank edit, 20% of bases swapped at fixed GC", float(cf.loc["20% of bases swapped, GC kept", "mean_change"]), 0.016, 3)
    chk(P + "flank edit, order shuffled", float(cf.loc["order shuffled, composition kept", "mean_change"]), 0.23, 2)
    chk(P + "flank edit, fall over rise (about twice)", float(cf.loc["GC -5 points, both flanks", "mean_change"] / -cf.loc["GC +5 points, both flanks", "mean_change"]), 1.9, 1)
    chk(P + "flank edits: guides", float(cf.guides.iloc[0]), 1500, 0)
    from ..build.build_features import load_curated_index

    ix = load_curated_index(verbose=False).drop_duplicates("protospacer")
    gc = lambda s: (s.count("G") + s.count("C")) / len(s) * 100
    chk(P + "sd of upstream 1 kb flank GC across the curated guides (points)", float(ix.upstream.map(gc).std()), 4.5, 1)
    chk(P + "sd of downstream 1 kb flank GC (points)", float(ix.downstream.map(gc).std()), 4.6, 1)
    return f, ix


def _build_edit(f, ix_of_pick, edit, frac, kind_name, build_x):
    r2 = np.random.default_rng(zlib.crc32(kind_name.encode()))
    from ..interpret.interp_extra import _edit
    e = ix_of_pick.copy()
    e["upstream"] = [_edit(s, edit, r2, frac, "up") for s in ix_of_pick["upstream"]]
    e["downstream"] = [_edit(s, edit, r2, frac, "down") for s in ix_of_pick["downstream"]]
    return build_x(e)


def _group(c):
    if c.startswith("eng.flank.comp."):
        return "comp." + c.split(".")[-1]
    if c.startswith("eng.flank.win."):
        return "win"
    if c.startswith("eng.flank.qct."):
        return "qct"
    if c.startswith("eng.flank.seq."):
        return "seq"
    return "base"


def _shuffle(chk):
    """Which columns carry the response to a shuffle of the flank beyond the near 20 nt: the production recipe, then the shipped model."""
    print("R3 the shuffle edit, by column group")
    from ..build.build_features import load_curated_index
    from ..features import a_flank
    from ..interpret import interp_extra as ie
    from ..interpret import vs_crisprhal_interpret as V

    NAME, EDIT = "order shuffled, composition kept", "shuffle"

    def prep_index(protos_pick):
        idx = load_curated_index(verbose=False).drop_duplicates("protospacer").set_index("protospacer").reindex(protos_pick).reset_index()
        idx = idx.rename(columns={"index": "protospacer"}); idx[config.ID_COL] = idx["protospacer"]; idx["located"] = True
        return idx

    # ---- production recipe (test 3)
    f = ie._fit(verbose=False)
    m = f["m"]
    rng = np.random.default_rng(ie.SEED)
    pick = rng.choice(f["te"], 1500, replace=False)
    idx = prep_index(f["protos"][pick])
    an = f["an"]; pos = {c: i for i, c in enumerate(f["cols"])}; a_cols = [c for c in an if c in pos]
    Xfull = np.asarray(np.load(config.INTERIM / "expanded_X.npy", mmap_mode="r"), dtype=np.float32)[pick]
    med = np.where(np.isnan(f["med"][f["top"]]), 0.0, f["med"][f["top"]])

    def build(index):
        blk = a_flank.build(index).set_index(config.ID_COL).reindex(index[config.ID_COL])
        Xb = np.full((len(index), len(f["cols"])), np.nan, dtype=np.float32)
        for j, i in enumerate(f["top"]):
            if i < f["nbase"]:
                Xb[:, j] = Xfull[:, i]
        for c in a_cols:
            Xb[:, pos[c]] = blk[c].to_numpy(dtype=np.float32)
        r, c = np.where(np.isnan(Xb)); Xb[r, c] = med[c]
        return Xb

    X0 = build(idx); p0 = m.predict(X0)
    chk(P + "unedited rebuild reproduces the stored matrix (mean |difference| in the prediction)", float(np.mean(np.abs(p0 - m.predict(f["Xte"][np.searchsorted(f["te"], pick)])))), 0.0, 4)
    X1 = _build_edit(f, idx, EDIT, 0.0, NAME, build)
    groups = {}
    for c in a_cols:
        groups.setdefault(_group(c), []).append(pos[c])
    chk(P + "production recipe: run-length (maxrun) columns among its 300", float(len(groups.get("comp.maxrun", []))), 0, 0)
    chk(P + "production recipe: shuffle, mean change in the prediction", float((m.predict(X1) - p0).mean()), 0.23, 2)

    def only(v):
        Xo = X0.copy(); Xo[:, v] = X1[:, v]
        return float((m.predict(Xo) - p0).mean())

    chk(P + "production recipe: shuffle, windowed quantum columns edited alone", only(groups["win"]), 0.18, 2)
    chk(P + "production recipe: shuffle, windowed GC columns edited alone", only(groups["comp.gc"]), 0.07, 2)
    cols = f["cols"]
    ch = {c: float(np.abs(X1[:, cols.index(c)] - X0[:, cols.index(c)]).mean() * 100) for c in cols if c.startswith("eng.flank.comp.") and c.endswith(".gc")}
    w = [v for k, v in ch.items() if not k.endswith("1000.gc")]
    chk(P + "shuffle: smallest mean |change| of window GC over the 50, 250 and 500 nt windows (points)", float(min(w)), 2, 0)
    chk(P + "shuffle: largest mean |change| (points)", float(max(w)), 6, 0)
    chk(P + "shuffle: change of the 1 kb GC (points)", float(max(v for k, v in ch.items() if k.endswith("1000.gc"))), 0.0, 1)

    # ---- the shipped tuned model on crisprHAL 2's WT split (the comparison section)
    D = V._wt_split()
    chk(P + "WT split: training guides", float((~D["in_test"]).sum()), 26830, 0)
    chk(P + "WT split: test guides", float(D["in_test"].sum()), 6737, 0)
    tr = V._train_rows(D, "shipped")
    fs = V.fit_full(D, tr, 41)
    names = np.asarray(D["names"]); nb = D["n_base"]
    chk(P + "shipped model: columns", float(len(fs["cols"])), 450, 0)
    chk(P + "shipped model: run-length (maxrun) columns among its 450", float(sum(_group(c) == "comp.maxrun" for c in names[fs["cols"]])), 0, 0)
    X, protos, pk = D["X"], D["protos"], D["pos"]
    idx2 = prep_index(protos[pk])
    a_names = names[nb:]

    def build2(index):
        blk = a_flank.build(index).set_index(config.ID_COL).reindex(index[config.ID_COL])
        Xr = X[pk].copy()
        for j, c in enumerate(a_names):
            if c in blk.columns:
                Xr[:, nb + j] = blk[c].to_numpy(dtype=np.float32)
        return Xr

    Y0 = build2(idx2); q0 = fs["m"].predict(fs["prep"](Y0))
    Y1 = _build_edit(fs, idx2, EDIT, 0.0, NAME, build2)
    chk(P + "shipped model: shuffle, mean change in the prediction", float((fs["m"].predict(fs["prep"](Y1)) - q0).mean()), 0.26, 2)
    grp = {}
    for j, c in enumerate(a_names):
        grp.setdefault(_group(c), []).append(nb + j)

    def only2(v):
        Yo = Y0.copy(); Yo[:, v] = Y1[:, v]
        return float((fs["m"].predict(fs["prep"](Yo)) - q0).mean())

    chk(P + "shipped model: shuffle, windowed quantum columns edited alone", only2(grp["win"]), 0.21, 2)
    chk(P + "shipped model: shuffle, windowed GC columns edited alone", only2(grp["comp.gc"]), 0.07, 2)


def _compare(chk, RES):
    print("R3 SLICER against crisprHAL 2: protocol and interval labels")
    fa = pd.read_csv(OUT / "interp_compare_faithfulness.csv")
    real = fa[~fa.shuffled_control].set_index(["model", "n"])
    chk(P + "sensitivity-n: guides", float(fa.rows.min()), 100, 0)
    chk(P + "sensitivity-n: block subsets per block count", float(fa.subsets.min()), 40, 0)
    d = pd.read_csv(OUT / "interp_compare_faithfulness_diff.csv").set_index(["model", "n"])
    for n in (1, 3, 6):
        for model in ("SLICER (shipped)", "SLICER-104"):
            o, h = real.loc[(model, n)], real.loc[("crisprHAL 2", n)]
            se = math.hypot(o.se, h.se)
            chk(P + f"faithfulness interval is the difference +/- 1.96 se, {model}, {n} blocks: lower end", float(o.pearson - h.pearson - 1.96 * se), float(d.loc[(model, n), "lo"]), 4)
    chk(P + "faithfulness: every lower bound above 0, both SLICER readings", float((d.lo > 0).all()), 1, 0)
    st = pd.read_csv(OUT / "interp_compare_stability.csv").set_index(["model", "unit", "comparison"])
    chk(P + "refit pairs over seeds", float(st.loc[("crisprHAL 2", "substitution map (80)", "refit seeds"), "pairs"]), 3, 0)
    chk(P + "refit pairs over halves", float(st.loc[("crisprHAL 2", "substitution map (80)", "random halves"), "pairs"]), 2, 0)
    a = st.loc[("SLICER-104", "attribution: guide channels (80)", "refit seeds")]; b = st.loc[("crisprHAL 2", "attribution: guide channels (80)", "refit seeds")]
    chk(P + "SLICER-104 and crisprHAL 2 channel agreement over seeds: ranges overlap", float(a["min"] <= b["max"] and b["min"] <= a["max"]), 1, 0)
    a = st.loc[("SLICER (shipped)", "substitution map (80)", "random halves")]; b = st.loc[("crisprHAL 2", "substitution map (80)", "random halves")]
    chk(P + "substitution map over halves: crisprHAL 2's lowest pair above SLICER's highest", float(b["min"] > a["max"]), 1, 0)
    vd = pd.read_csv(OUT / "interp_compare_validity_diff.csv").set_index("model")
    chk(P + "substitution map against the model-free map: SLICER-104's interval excludes 0", float(vd.loc["SLICER-104", "lo"] > 0), 1, 0)
    chk(P + "... the shipped SLICER's interval includes 0", float(vd.loc["SLICER (shipped)", "lo"] < 0 < vd.loc["SLICER (shipped)", "hi"]), 1, 0)
    rd = pd.read_csv(OUT / "interp_compare_readable.csv")
    s = rd[(rd.population == "five curated folds (out-of-fold)") & (rd.surrogate == "boosted trees (champion settings)") & (rd.model == "SLICER")].iloc[0]
    chk(P + "boosted surrogate, five folds: share of accuracy kept, SLICER minus crisprHAL 2", float(s.retained_minus_hal), -0.015, 3)
    chk(P + "... its interval is below 0", float(s.retained_hi < 0), 1, 0)
    fe = pd.read_csv(OUT / "interp_compare_flank_edits.csv")
    chk(P + "flank edits of the comparison: guides per model", float(fe.guides.min()), 1500, 0)
    am = pd.read_csv(OUT / "interp_compare_validity.csv")
    chk(P + "attribution-implied map against the model-free map, crisprHAL 2 minus SLICER-104 (shipped fits)", float(
        am[(am.model == "crisprHAL 2") & (am.fit == "shipped") & (am.readout == "attribution-implied map (80)")].spearman.iloc[0]
        - am[(am.model == "SLICER-104") & (am.fit == "shipped") & (am.readout == "attribution-implied map (80)")].spearman.iloc[0]), 0.005, 3)
    cs = pd.read_csv(OUT / "compact_model_shap_columns.csv")
    chk(P + "compact model: the seven largest columns by mean |SHAP|", float(set(cs.column[:7]) == {"p20.G", "p20.T", "p20.C", "p20.A", "p18.G", "p15.C", "p14.C"}), 1, 0)
    chk(P + "compact model: A at position 20 ranks sixth", float(list(cs.column).index("p20.A")), 5, 0)
    ad = pd.read_csv(RES / "additivity_reduced.csv").set_index("arm")
    r = ad.loc["reduced 427 (mono+di+hand)"]
    chk(P + "427-column reduced set: ridge", float(r.ridge), 0.503, 3)
    chk(P + "427-column reduced set: LightGBM", float(r.lightgbm), 0.526, 3)
    chk(P + "427-column reduced set: ridge over LightGBM", float(r.ridge / r.lightgbm), 0.956, 3)


def _ceiling(chk, RES):
    print("R3 the ceiling")
    from scipy.stats import spearmanr as sp
    cc = json.loads((RES / "curated_ceiling.json").read_text())
    arm = cc["arms"]["1-1 / 2-2"]
    chk(P + "guides passing the filter at an average control count of 73", float(cc["n_after_cutoff"]), 33871, 0)
    chk(P + "replicate agreement, plain CPM log-ratio", arm["replicate_spearman"], 0.8097, 4)
    chk(P + "ceiling, plain CPM log-ratio", arm["ceiling"], 0.946, 3)
    chk(P + "ceiling, other pairing of the replicates", cc["arms"]["1-2 / 2-1"]["ceiling"], 0.9453, 4)
    chk(P + "released score against the plain log-ratio, |rho|", abs(cc["reconstruction"]["spearman"]), 0.957, 3)
    d = pd.read_csv(config.EXTERNAL / "crisprHAL" / "rawData" / "E_coli_WTSpCas9_counts.tsv", sep="\t")
    cols = ["Cas1_S", "Cas2_S", "dCas1_NS", "dCas2_NS"]
    f = d[(d.dCas1_NS + d.dCas2_NS) / 2 >= 73].reset_index(drop=True)
    C = f[cols].to_numpy(float)
    sb = lambda r: math.sqrt(2 * r / (1 + r))
    cpm = C / d[cols].sum().to_numpy(float) * 1e6
    u = np.log2((cpm[:, 0] + .5) / (cpm[:, 2] + .5)); v = np.log2((cpm[:, 1] + .5) / (cpm[:, 3] + .5))
    chk(P + "plain log-ratio re-derived from the counts: replicate agreement", float(sp(u, v).statistic), 0.8097, 4)
    meta = json.loads((config.INTERIM / "expanded_meta.json").read_text())
    lab = pd.DataFrame({"sgRNA": meta["protospacers"], "y": meta["y"]})
    agree, ceil, rho = [], [], []
    for seed in range(1, 6):
        rng = np.random.default_rng(seed)
        D1 = np.zeros((len(f), 128), np.float32); D2 = np.zeros((len(f), 128), np.float32)
        for m_ in range(128):
            clr = np.zeros_like(C)
            for j in range(4):
                gm = rng.gamma(C[:, j] + 0.5); l = np.log2(gm / gm.sum()); clr[:, j] = l - l.mean()
            D1[:, m_] = clr[:, 0] - clr[:, 2]; D2[:, m_] = clr[:, 1] - clr[:, 3]
        a1, a2 = D1.mean(1), D2.mean(1)
        r_ = float(sp(a1, a2).statistic)
        agree.append(r_); ceil.append(sb(r_))
        mm = lab.merge(pd.DataFrame({"sgRNA": f.sgRNA, "a": (a1 + a2) / 2}), on="sgRNA")
        rho.append(abs(float(sp(mm.y, mm.a).statistic)))
    chk(P + "Dirichlet-CLR estimate (prior 0.5, 128 draws, five seeds): replicate agreement", float(np.mean(agree)), 0.798, 3)
    chk(P + "... ceiling", float(np.mean(ceil)), 0.942, 3)
    chk(P + "... spread of the ceiling over the five seeds is below 0.001", float(np.ptp(ceil) < 0.001), 1, 0)
    chk(P + "... |rho| with the released score", float(np.mean(rho)), 0.993, 3)
    chk(P + "... its ceiling is below the plain log-ratio's (so 0.946 is not a lower bound)", float(np.mean(ceil) < arm["ceiling"]), 1, 0)
    tc = pd.read_csv(OUT / "tune_confirm.csv").set_index("arm")
    chk(P + "gap to the ceiling at the tuned model on the CONFIRM genes", float(arm["ceiling"] - tc.loc["d099x", "rho"]), 0.22, 2)
    chk(P + "gap to the ceiling at the frozen champion on the CONFIRM genes", float(arm["ceiling"] - tc.loc["champion", "rho"]), 0.24, 2)
    fa = pd.read_csv(OUT / "family_a_at_tuned_summary.csv").set_index("comparison")
    chk(P + "gap to the ceiling at the best single tuned model (gene-grouped CV)", float(arm["ceiling"] - fa.loc["family A increment, d099x", "with_a"]), 0.215, 3)
    hal = config.EXTERNAL / "crisprHAL" / "data"
    tr = pd.read_csv(hal / "WT-SpCas9_training_data.csv", header=None)[0].astype(str).str.upper().str[189:209]
    te = pd.read_csv(hal / "WT-SpCas9_testing_data.csv", header=None)[0].astype(str).str.upper().str[189:209]
    chk(P + "rows in crisprHAL 2's released WT-SpCas9 files", float(len(tr) + len(te)), 33743, 0)
    chk(P + "distinct guides in them", float(len(set(tr) | set(te))), 33567, 0)
    chk(P + "distinct test guides / training guides not in the test file", float(len(set(te))), 6737, 0)
    chk(P + "... training guides after removing those in the test file", float(len(set(tr) - set(te))), 26830, 0)


def run(chk):
    """Run every check of review R3 through `chk(label, got, claimed, dec)`."""
    RES = OUT.parent                 # the fork's results/
    MAIN = config.RESULTS            # the main project's results/, which holds attribution_summary_shap.csv and faithfulness_shap_share.csv
    _importance(chk, RES, MAIN)
    _interp_extra(chk, RES)
    _shuffle(chk)        # re-runs the flank-edit refits (about a minute)
    _compare(chk, RES)
    _ceiling(chk, RES)   # the ALDEx2 re-implementation (about a minute)


# ---------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    bad, n = [], 0

    def chk(label, got, claimed, dec):
        """Same rule as paper_numbers_C: pass when |got - claimed| <= half a unit in the last quoted digit."""
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
