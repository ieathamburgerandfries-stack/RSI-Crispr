"""Two controls on the strand result.

`rousset`  Family A inside one strand at a time, on Rousset's essential-gene guides
           (the replication of `strand_families --within`, which used Cui).
`flip`     Tell a trained model the wrong strand at test time, and see what the
           strand column was carrying. Cui, pooled, gene-grouped, three seeds.

    python -m sgrna.strand_strata --rousset
    python -m sgrna.strand_strata --flip
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from . import config

OUT_R = config.RESULTS / "strand_strata_rousset.csv"
OUT_F = config.RESULTS / "strand_flip.csv"


def rousset(name="Rousset", seeds=(41, 42, 43)):
    from .crispri_screens import build
    from .run_ablation import cross_validate

    X, meta = build(name, verbose=False)
    X = np.asarray(X, dtype=np.float32)
    names = list(meta["names"])
    nb = int(meta["n_base"])
    y = np.asarray(meta["y"], dtype=float)
    gene = np.asarray(meta["gene"], dtype=object)
    ess = np.asarray(meta["essential"], dtype=int)
    cod = np.asarray(meta["coding_strand"], dtype=float)
    block = np.floor_divide(np.asarray(meta["left"], dtype=float), 100_000
                            ).astype(int).astype(str)
    sel = (gene != "") & (ess == 1)
    rows, oof = [], {}
    for sname, sval in (("coding", 1.0), ("template", 0.0)):
        m = sel & (cod == sval)
        print(f"  {name} {sname}: {int(m.sum()):,} guides", flush=True)
        for gname, grp in (("genes", gene), ("100kb blocks", block)):
            for arm, cols in (("base", slice(0, nb)), ("a_flank", slice(0, None))):
                Xa, nm = X[m][:, cols], names[cols]
                res = cross_validate(
                    Xa, y[m], nm, ["arm"] * len(nm), n_features=config.N_FEATURES,
                    seeds=list(seeds), n_splits=config.N_SPLITS, verbose=False,
                    groups=grp[m],
                    tag=f"ssr_{name}_{sname}_{gname.replace(' ', '')}_{arm}",
                    experiment=arm, return_oof=True)
                mu = res["summary"]["spearman"]["mean"]
                sd = res["summary"]["spearman"]["sd"]
                row = dict(screen=name, stratum=sname, grouping=gname, arm=arm,
                           n=int(m.sum()), spearman=mu, spearman_sd=sd)
                if sname == "coding" and gname == "genes":
                    oof[arm] = res["oof_pred"]
                rows.append(row)
                print(f"    {gname:12s} {arm:8s}: rho {mu:+.4f} (sd {sd:.4f})", flush=True)
                pd.DataFrame(rows).to_csv(OUT_R, index=False)
    # between genes vs within genes, coding stratum (last seed's out-of-fold)
    m = sel & (cod == 1.0)
    g = gene[m]
    for arm, p in oof.items():
        d = pd.DataFrame(dict(g=g, y=y[m], p=p))
        d = d[d.groupby("g").y.transform("size") >= 5]
        gm = d.groupby("g")[["y", "p"]].mean()
        wy = d.y - d.groupby("g").y.transform("mean")
        wp = d.p - d.groupby("g").p.transform("mean")
        rows.append(dict(screen=name, stratum="coding", grouping="genes", arm=arm,
                         n=len(d), genes=len(gm),
                         overall=spearmanr(d.y, d.p).statistic,
                         between_genes=spearmanr(gm.y, gm.p).statistic,
                         within_genes=spearmanr(wy, wp).statistic))
        print("   ", {k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in rows[-1].items() if k in
                      ("arm", "n", "genes", "overall", "between_genes", "within_genes")})
    pd.DataFrame(rows).to_csv(OUT_R, index=False)


def flip(seeds=(41, 42, 43)):
    """Train with the true strand; score held-out genes with the strand wrong."""
    from .cui_screen import _essential_frame
    from .run_ablation import (_impute, champion_params, grouped_folds,
                               make_selector)

    f = _essential_frame()
    strand = f["coding"]
    rows = []
    for arm, Xb in (("base+strand", f["Xb"]),
                    ("base+strand+a_flank", np.hstack([f["Xb"], f["Xf"]]))):
        X = np.hstack([Xb, strand])
        scol = X.shape[1] - 1
        y = f["y"]
        grp = dict(f["groupings"])["genes"]
        res = {k: [] for k in ("true", "flipped", "shuffled", "constant")}
        rng = np.random.default_rng(0)
        used = []
        for seed in seeds:
            for tr, va in grouped_folds(grp, config.N_SPLITS, seed):
                Xtr, Xva = _impute(X[tr], X[va])
                sel = make_selector(seed)
                sel.fit(Xtr, y[tr])
                top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
                used.append(bool(scol in top))
                mdl = xgb.XGBRegressor(**champion_params(seed))
                mdl.fit(Xtr[:, top], y[tr])
                for k in res:
                    Xv = Xva.copy()
                    if k == "flipped":
                        Xv[:, scol] = 1.0 - Xv[:, scol]
                    elif k == "shuffled":
                        Xv[:, scol] = rng.permutation(Xv[:, scol])
                    elif k == "constant":
                        Xv[:, scol] = float(np.mean(Xtr[:, scol]))
                    res[k].append(spearmanr(y[va], mdl.predict(Xv[:, top])).statistic)
        row = dict(arm=arm, strand_selected_in=f"{sum(used)}/{len(used)} folds",
                   **{k: float(np.mean(v)) for k, v in res.items()},
                   **{f"{k}_sd": float(np.std(v)) for k, v in res.items()})
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
              flush=True)
        pd.DataFrame(rows).to_csv(OUT_F, index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rousset", action="store_true")
    ap.add_argument("--flip", action="store_true")
    a = ap.parse_args(argv)
    if a.flip:
        flip()
    if a.rousset:
        rousset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
