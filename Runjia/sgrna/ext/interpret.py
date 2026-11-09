"""What the tuned hyperparameters say about the signal: four tests, each answering a question the choice of settings raises.

The search preferred shallow trees (depth 4-5), slow learning with many trees, 450-900 columns and ignored every
regularisation knob. Read as statements about the data rather than about the learner:

  depth      how much of the signal is additive, and how much needs interactions of what order?   (`depth_curve`)
  columns    which kinds of column carry the gain, and do the tuned and the original models use the same ones? (`gain_groups`)
  where      is the gain in ranking guides within a gene, or in telling genes apart?               (`within_between`)
  data       does the benefit of the slower, larger model grow with the amount of training data?   (`data_scaling`)

All on the DEV / CONFIRM gene split of `tune_ecoli.py`; nothing here is tuned on CONFIRM.

    python -m sgrna.ext.interpret --test depth
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sgrna.curated_ablation import _base, _block
from sgrna.importance import block_of
from sgrna.run_ablation import _impute, make_selector

from . import tune_ecoli as T
from .common import OUT

BASE_CFG = dict(subsample=0.7, colsample_bytree=0.8, min_child_weight=4.0, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
                max_bin=256, nf=450, target="raw")
DEPTHS = {1: (0.1, 8000), 2: (0.05, 6000), 3: (0.03, 4000), 4: (0.015, 4000), 5: (0.015, 4000), 6: (0.015, 3000),
          8: (0.01, 3000)}
KGRID = (100, 200, 400, 800, 1200, 1600, 2000, 3000, 4000, 6000, 8000)


def depth_curve(co):
    """rho against tree depth at slow learning: depth 1 is a purely additive model, depth 2 adds pairwise interactions."""
    rows = []
    for d, (lr, n) in DEPTHS.items():
        cfg = dict(BASE_CFG, max_depth=d, learning_rate=lr, n_estimators=n)
        ks = tuple(k for k in KGRID if k <= n)
        dev = T.score_dev(co, cfg, 41, ks)
        k, r = T.best_k(dev)
        # CONFIRM: train on all DEV at the chosen tree count
        top = np.load(T.DIR / "rank_devall_s41.npy"); cols = top[:cfg["nf"]]
        A, B = _impute(co.X[co.dev][:, cols], co.X[co.conf][:, cols])
        p = T.fit_predict(dict(cfg, n_estimators=k), A, co.y[co.dev], B, 41)[k]
        rows.append(dict(max_depth=d, learning_rate=lr, best_k=k, dev_rho=r, confirm_rho=float(spearmanr(co.y[co.conf], p).statistic)))
        print(f"  depth {d}: best k {k}, DEV {r:.4f}, CONFIRM {rows[-1]['confirm_rho']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "interpret_depth.csv", index=False)
    return pd.DataFrame(rows)


def gain_groups(co):
    """Share of split gain by kind of column: the selector, the original model and the tuned model, all fitted on DEV."""
    import xgboost as xgb
    X0, _, names, protos = _base(); _, an = _block("a_flank", protos)
    allnames = list(names) + list(an)
    groups = np.array([block_of(n) for n in allnames])
    sel_gain = None
    top = np.load(T.DIR / "rank_devall_s41.npy")
    A, _ = _impute(co.X[co.dev], co.X[co.conf][:1])
    s = make_selector(41); s.fit(A, co.y[co.dev]); sel_gain = s.feature_importances_
    rows = []
    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    sel = T.selected()
    for label, name in (("original (depth 4, 400 trees, 300 columns)", "champion"), ("tuned (d099x)", "d099x")):
        cfg = dict(cfgs[name], n_estimators=int(sel.loc[name, "k"]))
        cols = top[:int(cfg["nf"])]
        Xa, _ = _impute(co.X[co.dev][:, cols], co.X[co.conf][:1][:, cols])
        m = xgb.XGBRegressor(**dict(T.xgb_params(cfg, 41), importance_type="gain")).fit(Xa, co.y[co.dev])
        g = np.zeros(len(allnames)); g[cols] = m.feature_importances_
        for b in sorted(set(groups)):
            rows.append(dict(model=label, block=b, n_selected=int(((groups[cols] == b)).sum()), gain_share=float(g[groups == b].sum() / g.sum())))
    for b in sorted(set(groups)):
        rows.append(dict(model="selector (all columns)", block=b, n_selected=int((sel_gain[groups == b] > 0).sum()),
                         gain_share=float(sel_gain[groups == b].sum() / sel_gain.sum())))
    out = pd.DataFrame(rows); out.to_csv(OUT / "interpret_gain_groups.csv", index=False)
    out["n_columns"] = out.block.map(pd.Series(groups).value_counts())
    return out


def position_profile(co):
    """Gain of the original and the tuned model by position along the guide (1 = PAM-distal ... 20 = PAM-proximal, as in the
    matrix) and by distance into the upstream / downstream flank, for every column that names a position."""
    import re
    import xgboost as xgb
    from sgrna.importance import _v_positions
    _, _, names, protos = _base(); _, an = _block("a_flank", protos)
    allnames = list(names) + list(an)
    vpos = _v_positions()

    def where(n):
        if n.startswith("V"):
            q = vpos.get(n); return ("guide", int(q)) if q else None
        m = re.match(r"^p(\d+)(monomer|dimer|trimer|tetramer|basepair)\.", n)
        if m:
            return ("guide", int(m.group(1)))
        m = re.search(r"\.(u|d)(\d+)(?:\.|$)", n) if n.startswith("eng.flank.") and (".seq." in n or ".qct." in n) else None
        if m:
            return ("upstream" if m.group(1) == "u" else "downstream", int(m.group(2)))
        return None

    loc = [where(n) for n in allnames]
    top = np.load(T.DIR / "rank_devall_s41.npy"); cfgs = json.loads((OUT / "tune_configs.json").read_text()); sel = T.selected()
    rows = []
    for label, name in (("original", "champion"), ("tuned", "d099x")):
        cfg = dict(cfgs[name], n_estimators=int(sel.loc[name, "k"])); cols = top[:int(cfg["nf"])]
        Xa, _ = _impute(co.X[co.dev][:, cols], co.X[co.conf][:1][:, cols])
        m = xgb.XGBRegressor(**dict(T.xgb_params(cfg, 41), importance_type="gain")).fit(Xa, co.y[co.dev])
        g = np.zeros(len(allnames)); g[cols] = m.feature_importances_
        tot = g.sum()
        for region in ("guide", "upstream", "downstream"):
            for pos in range(1, 21):
                ii = [i for i, l in enumerate(loc) if l == (region, pos)]
                if ii:
                    rows.append(dict(model=label, region=region, position=pos, gain_share=float(g[ii].sum() / tot), n_columns=len(ii)))
    out = pd.DataFrame(rows); out.to_csv(OUT / "interpret_position_profile.csv", index=False)
    return out


def within_between(co, conf_seeds=(41, 42, 43), min_n=5, B=1000):
    """Split rho on CONFIRM into between-gene (gene means) and within-gene (gene-demeaned) parts, for each model."""
    fin = {"champion": "conf_champion_k400", "tuned (d099x)": "conf_d099x_k4000"}
    P = {lab: np.array([np.load(T.DIR / f"{stem}_s{sd}.npy") for sd in conf_seeds]) for lab, stem in fin.items()}
    ens = [n for n in ("d099x", "d077x", "d021x", "d080x", "d000x")]
    sel = T.selected()
    P["average of five"] = np.array([np.mean([T._z(np.load(T.DIR / f"conf_{n}_k{int(sel.loc[n, 'k'])}_s{sd}.npy")) for n in ens], axis=0)
                                     for sd in conf_seeds])
    y, g = co.y[co.conf], co.g[co.conf]
    ug, inv = np.unique(g, return_inverse=True)
    counts = np.bincount(inv); keep = counts >= min_n
    rows_of = [np.where(inv == i)[0] for i in range(len(ug))]

    def parts(idx_genes, p):
        yy, pp, gg = [], [], []
        for gi in idx_genes:
            r = rows_of[gi]
            if len(r) >= min_n:
                yy.append(y[r]); pp.append(p[r]); gg.append(gi)
        ym = np.array([a.mean() for a in yy]); pm = np.array([a.mean() for a in pp])
        yd = np.concatenate([a - a.mean() for a in yy]); pd_ = np.concatenate([a - a.mean() for a in pp])
        return float(spearmanr(ym, pm).statistic), float(spearmanr(yd, pd_).statistic)

    point = {lab: np.mean([parts(range(len(ug)), p) for p in Ps], axis=0) for lab, Ps in P.items()}
    rng = np.random.default_rng(0)
    bs = {lab: [] for lab in P}
    for _ in range(B):
        gi = rng.integers(0, len(ug), len(ug))
        for lab, Ps in P.items():
            bs[lab].append(np.mean([parts(gi, p) for p in Ps], axis=0))
    rows = []
    for lab in P:
        b = np.array(bs[lab]); d = b - np.array(bs["champion"])
        for j, part in enumerate(("between-gene", "within-gene")):
            rows.append(dict(model=lab, part=part, rho=point[lab][j], lo=np.percentile(b[:, j], 2.5), hi=np.percentile(b[:, j], 97.5),
                             d_vs_champion=point[lab][j] - point["champion"][j], d_lo=np.percentile(d[:, j], 2.5),
                             d_hi=np.percentile(d[:, j], 97.5), n_genes=int(keep.sum())))
    out = pd.DataFrame(rows); out.to_csv(OUT / "interpret_within_between.csv", index=False)
    return out


def data_scaling(co, fracs=(0.25, 0.5, 1.0), seeds=(41, 42, 43)):
    """Original against tuned model, trained on a random fraction of the DEV genes, scored on CONFIRM."""
    cfgs = json.loads((OUT / "tune_configs.json").read_text()); sel = T.selected()
    genes = np.array(sorted(set(co.g[co.dev])))
    rows = []
    for fr in fracs:
        for sd in seeds:
            rng = np.random.default_rng(sd)
            gs = set(rng.choice(genes, size=int(round(fr * len(genes))), replace=False))
            idx = co.dev[np.array([x in gs for x in co.g[co.dev]])]
            A0, _ = _impute(co.X[idx], co.X[co.conf][:1]); s = make_selector(sd); s.fit(A0, co.y[idx])
            top = np.argsort(s.feature_importances_)[::-1][:900]
            for name in ("champion", "d099x"):
                cfg = dict(cfgs[name], n_estimators=int(sel.loc[name, "k"])); cols = top[:int(cfg["nf"])]
                A, B = _impute(co.X[idx][:, cols], co.X[co.conf][:, cols])
                p = T.fit_predict(cfg, A, co.y[idx], B, sd)[cfg["n_estimators"]]
                rows.append(dict(fraction=fr, n_train=len(idx), seed=sd, model=name, rho=float(spearmanr(co.y[co.conf], p).statistic)))
        pd.DataFrame(rows).to_csv(OUT / "interpret_data_scaling.csv", index=False)
        print(f"  fraction {fr} done", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["depth", "groups", "where", "data", "positions"], required=True)
    a = ap.parse_args()
    co = T.Cohort()
    pd.set_option("display.width", 220)
    out = {"depth": depth_curve, "groups": gain_groups, "where": within_between, "data": data_scaling, "positions": position_profile}[a.test](co)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
