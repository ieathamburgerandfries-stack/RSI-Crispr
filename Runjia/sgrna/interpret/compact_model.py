"""A compact model of guide base indicators and flank composition columns keeps 88% of the full model's rho and can be read column by column, with SHAP agreeing across tree families at 0.87. A 104-column model, its accuracy and its attributions.

Finding 5 says per-column importance is unstable on the full matrix because 6,095 of its columns restate about twenty variables, so a
column's credit moves to a correlated neighbour when the model is refitted. That is an explanation, and it makes a prediction the
project has not tested: on a representation with no restated columns the instability should go away. This module builds one.

The compact representation (every column named and none a copy of another):
  * 80 indicators for the base at each of the 20 protospacer positions (position x base);
  * 24 flank composition columns: GC fraction, purine fraction and longest homopolymer run, over 50, 250, 500 and 1,000 nt, on each side.
Arms: guide only (80); guide + the four whole-kilobase numbers (84); the 104-column model; and 104 + the 304 adjacent-pair indicators (408).
Reference: the production recipe (base + family A, gain-ranked top 300, champion settings). All arms are gene-grouped five-fold CV on the
30,203 genic guides, three seeds, paired by fold.

Predictions, written before the run:
  P1  The 104-column model keeps at least 90% of the full model's rho (probability about 0.6). Reasoning: the 24 composition columns
      carried most of family A's gain (Finding 7) and the guide's own preferences are sitting in 80 indicators.
  P2  On the 104 columns, SHAP importance agrees between the five tree families at rho >= 0.8 per column (probability about 0.7),
      against 0.496 on the full matrix. If it stays near 0.5, redundancy is not the whole explanation of Finding 5 and that finding's
      account has to be revised.
  P3  The default (gain / impurity) importance also improves, but stays below SHAP.
  P4  SHAP reads out the model-free position x base map (Finding 7): the SHAP-implied effect of each base at each position correlates
      with the model-free effect at rho >= 0.8 over the 80 cells.
  P5  The model's flank contribution is a step at the AT-rich end, as the response curve of Finding 7 is.

    python -m sgrna.interpret.compact_model
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sgrna import config

from ..common import OUT
from .mechanism import _cohort

BASES = "ACGT"
SEEDS = (41, 42, 43)


def _data():
    from sgrna.cohorts.curated_ablation import _base, _block, _groups
    X, y, names, protos = _base()
    A, an = _block("a_flank", protos)
    g, ok = _groups("gene", protos)
    protos = np.asarray(protos, dtype=object)
    codes = np.frombuffer("".join(protos[ok]).encode(), np.uint8).reshape(ok.sum(), 20)
    lut = {ord(b): i for i, b in enumerate(BASES)}
    idx = np.vectorize(lut.get)(codes)
    onehot = np.zeros((len(idx), 80), np.float32)
    for p in range(20):
        for b in range(4):
            onehot[:, p * 4 + b] = idx[:, p] == b
    di = np.zeros((len(idx), 19 * 16), np.float32)
    for p in range(19):
        di[np.arange(len(idx)), p * 16 + idx[:, p] * 4 + idx[:, p + 1]] = 1
    an = list(an)
    comp_cols = [i for i, n in enumerate(an) if n.startswith("eng.flank.comp.")]
    comp = np.asarray(A[ok][:, comp_cols], np.float32)
    comp_names = [an[i] for i in comp_cols]
    four = [comp_names.index(n) for n in ("eng.flank.comp.up1000.gc", "eng.flank.comp.up1000.purine",
                                          "eng.flank.comp.dn1000.gc", "eng.flank.comp.dn1000.purine")]
    onehot_names = [f"p{p + 1}.{BASES[b]}" for p in range(20) for b in range(4)]
    di_names = [f"d{p + 1}.{BASES[a]}{BASES[b]}" for p in range(19) for a in range(4) for b in range(4)]
    return dict(y=np.asarray(y, float)[ok], gene=g[ok], onehot=onehot, di=di, comp=comp, comp_names=comp_names, four=four,
                onehot_names=onehot_names, di_names=di_names, Xfull=np.hstack([np.asarray(X, np.float32)[ok], np.asarray(A, np.float32)[ok]]),
                protos=protos[ok])


def _folds(gene, seed, k=5):
    u = pd.unique(gene.astype(str)); rng = np.random.default_rng(seed); perm = rng.permutation(len(u))
    fold_of = dict(zip(u, perm % k))
    f = np.array([fold_of[x] for x in gene.astype(str)])
    return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]


def _rho(a, b):
    return float(spearmanr(a, b).statistic)


def accuracy(d):
    from ..compare.vs_crisprhal import member_specs, train_predict
    specs = member_specs(["champion"])
    arms = {"guide only (80)": lambda: d["onehot"],
            "guide + four kilobase numbers (84)": lambda: np.hstack([d["onehot"], d["comp"][:, d["four"]]]),
            "compact (104)": lambda: np.hstack([d["onehot"], d["comp"]]),
            "compact + adjacent pairs (408)": lambda: np.hstack([d["onehot"], d["di"], d["comp"]])}
    mats = {k: f() for k, f in arms.items()}
    y = d["y"]
    rows = []
    for seed in SEEDS:
        folds = _folds(d["gene"], seed)
        for fi, (tr, te) in enumerate(folds):
            full, _ = train_predict(specs, d["Xfull"][tr], y[tr], d["Xfull"][te], seed=seed)
            r = {"full recipe (base + family A, top 300)": _rho(y[te], full["champion"])}
            for k, M in mats.items():
                m = _models_xgb(seed).fit(M[tr], y[tr])
                r[k] = _rho(y[te], m.predict(M[te]))
            rows.append(dict(seed=seed, fold=fi + 1, **r))
        print("  accuracy seed", seed, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "compact_model_accuracy_folds.csv", index=False)
    cols = [c for c in df.columns if c not in ("seed", "fold")]
    per_seed = df.groupby("seed")[cols].mean()
    out = pd.DataFrame({"rho_mean_over_seeds": per_seed.mean(), "rho_sd_over_seeds": per_seed.std()})
    ref = out.loc["full recipe (base + family A, top 300)", "rho_mean_over_seeds"]
    out["share_of_full"] = out["rho_mean_over_seeds"] / ref
    out.to_csv(OUT / "compact_model_accuracy.csv")
    print(out.round(4).to_string())
    return out


def _models_xgb(seed):
    from .importance_models import _models
    return _models(seed)["xgboost"]


def _blocks(d):
    """28 blocks: the 20 guide positions and the 8 (side, window) groups of flank composition."""
    names = d["onehot_names"] + d["comp_names"]
    out = []
    for n in names:
        if n.startswith("p"):
            out.append("guide " + n.split(".")[0])
        else:
            parts = n.split(".")                                   # eng.flank.comp.up250.gc
            out.append("flank " + parts[3])
    return np.array(out)


def stability(d):
    from sklearn.model_selection import GroupKFold
    from .importance_models import IS_TREE, _models, _native, _shap
    X = np.hstack([d["onehot"], d["comp"]]); y = d["y"]; names = d["onehot_names"] + d["comp_names"]
    blocks = _blocks(d); ub = list(pd.unique(blocks))
    tree = sorted(IS_TREE)
    agg = {"shap": [], "native": [], "shap_block": [], "native_block": []}
    for split in range(3):
        folds = _folds(d["gene"], 100 + split)
        tr, te = folds[0]
        vec = {"shap": {}, "native": {}}
        for name, m in _models(41 + split).items():
            if name not in IS_TREE:
                continue
            m.fit(X[tr], y[tr])
            vec["native"][name] = _native(name, m, X[tr])
            vec["shap"][name] = _shap(name, m, X[te], 1000, 41 + split)
        for kind in ("shap", "native"):
            col, blk = [], []
            for a, b in [(a, b) for i, a in enumerate(tree) for b in tree[i + 1:]]:
                col.append(_rho(vec[kind][a], vec[kind][b]))
                ba = np.array([vec[kind][a][blocks == u].sum() for u in ub]); bb = np.array([vec[kind][b][blocks == u].sum() for u in ub])
                blk.append(_rho(ba, bb))
            agg[kind].append(np.mean(col)); agg[kind + "_block"].append(np.mean(blk))
        print("  stability split", split, {k: round(v[-1], 3) for k, v in agg.items()}, flush=True)
    out = pd.DataFrame([dict(representation="compact (104 columns)", method=k.replace("_block", ""), level="block" if k.endswith("block") else "column",
                             agreement=float(np.mean(v)), sd_over_splits=float(np.std(v))) for k, v in agg.items()])
    out.to_csv(OUT / "compact_model_stability.csv", index=False)
    print(out.round(3).to_string(index=False))
    return out


def readout(d):
    """SHAP-implied effects of the compact XGBoost model, against the model-free maps of Finding 7."""
    import shap
    X = np.hstack([d["onehot"], d["comp"]]); y = d["y"]
    tr, te = _folds(d["gene"], 7)[0]
    m = _models_xgb(41).fit(X[tr], y[tr])
    phi = shap.TreeExplainer(m).shap_values(X[te], check_additivity=False)
    sd = float(y.std())
    # (i) guide positions: sum of the four columns at a position, averaged over guides carrying each base
    oh = d["onehot"][te]; rows = []
    for p in range(20):
        pos_phi = phi[:, p * 4:(p + 1) * 4].sum(axis=1)
        for b in range(4):
            sel = oh[:, p * 4 + b] == 1
            if sel.sum() >= 20:
                rows.append(dict(position=p + 1, base=BASES[b], n=int(sel.sum()), shap_effect_sd=float(pos_phi[sel].mean() / sd)))
    shap_map = pd.DataFrame(rows)
    free = pd.read_csv(OUT / "mechanism_position_base.csv")[["position", "base", "effect_sd"]]
    mm = shap_map.merge(free, on=["position", "base"])
    r_all = _rho(mm.shap_effect_sd, mm.effect_sd)
    r_pearson = float(np.corrcoef(mm.shap_effect_sd, mm.effect_sd)[0, 1])
    mm.to_csv(OUT / "compact_model_shap_vs_model_free.csv", index=False)
    # (ii) flank: total SHAP of the 24 composition columns by decile of flank GC (250 nt, both sides averaged)
    names = d["comp_names"]
    gc = (d["comp"][te][:, names.index("eng.flank.comp.up250.gc")] + d["comp"][te][:, names.index("eng.flank.comp.dn250.gc")]) / 2
    flank_phi = phi[:, 80:].sum(axis=1)
    q = pd.qcut(gc, 10, labels=False, duplicates="drop")
    dec = pd.DataFrame({"decile": q, "flank_gc": gc, "flank_shap": flank_phi}).groupby("decile").mean().reset_index()
    dec["flank_shap_in_sd"] = dec.flank_shap / sd
    dec.to_csv(OUT / "compact_model_flank_shap_by_decile.csv", index=False)
    print(f"\n  SHAP-implied position x base effects against the model-free map: rho {r_all:.3f} (Pearson {r_pearson:.3f}) over {len(mm)} cells")
    print(dec.round(3).to_string(index=False))
    # largest columns by mean |SHAP|
    mean_abs = np.abs(phi).mean(axis=0)
    top = pd.DataFrame({"column": d["onehot_names"] + names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    top.to_csv(OUT / "compact_model_shap_columns.csv", index=False)
    print(top.head(12).round(4).to_string(index=False))
    return r_all, dec


def main():
    d = _data()
    print(f"{len(d['y']):,} genic guides in {len(set(d['gene'].astype(str))):,} genes")
    acc = accuracy(d)
    stability(d)
    readout(d)


if __name__ == "__main__":
    main()
