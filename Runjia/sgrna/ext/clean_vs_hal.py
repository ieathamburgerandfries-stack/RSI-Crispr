"""A comparison with crisprHAL 2 that the tuning cannot have touched: the 30% of genes held out from every search.

The tuned settings were chosen on 70% of the curated cohort's genes. crisprHAL's WT-SpCas9 data are the same 33,567 guides, so the
"their split" and "our folds" comparisons of `vs_crisprhal.py` contain guides whose labels fed the settings' selection (not any model's
weights). The CONFIRM genes (1,204 genes, 8,943 guides) did not. Here the SLICER models are those trained on the tuning genes only and
scored on CONFIRM genes (stored by `tune_ecoli --confirm`); crisprHAL 2 is its stored out-of-fold prediction for the same guides
(trained on 80% of all guides by random folds, so on guides from the same genes: an advantage to it, which makes this comparison
conservative for SLICER). Differences are paired and bootstrapped over genes.

    python -m sgrna.ext.clean_vs_hal
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from sgrna.curated_ablation import _base, _groups

from . import tune_ecoli as T
from . import vs_crisprhal as V
from .common import OUT


def run(n_boot=2000, seed=0, seeds=(41, 42, 43)):
    X, y, _, protos = _base()
    g, ok = _groups("gene", protos)
    co = T.Cohort()
    assert np.array_equal(y[ok], co.y), "cohort alignment failed"
    order = np.concatenate([va for _, va in KFold(5, shuffle=True, random_state=41).split(np.arange(len(y)))])
    hal = {}
    for kind in ("default", "tuned"):
        h = pd.concat([pd.read_csv(f) for f in sorted(V.SRC.glob(f"preds_crisprhal_{kind}_fold*.csv"))], ignore_index=True)
        assert np.allclose(y[order], h["y"].to_numpy(), atol=1e-9)
        ph = np.empty(len(y)); ph[order] = h["pred"].to_numpy()
        hal[kind] = ph[ok][co.conf]
    yc, gc = co.y[co.conf], co.g[co.conf]
    sel = T.selected()
    load = lambda name: np.array([np.load(T.DIR / f"conf_{name}_k{int(sel.loc[name, 'k'])}_s{sd}.npy") for sd in seeds])
    members = ["d099x", "d077x", "d021x", "d080x", "d000x"]
    arms = {"original champion": load("champion"), "tuned XGBoost (d099x)": load("d099x"),
            "average of 5 XGBoost": np.array([np.mean([T._z(load(n)[i]) for n in members], axis=0) for i in range(len(seeds))])}
    ug, inv = np.unique(gc, return_inverse=True); rows_of = [np.where(inv == i)[0] for i in range(len(ug))]
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([rows_of[i] for i in rng.integers(0, len(ug), len(ug))]) for _ in range(n_boot)]
    rows = []
    for hk, hp in hal.items():
        rh = float(spearmanr(yc, hp).statistic)
        bh = np.array([spearmanr(yc[b], hp[b]).statistic for b in boots])
        for lab, P in arms.items():
            r = float(np.mean([spearmanr(yc, p).statistic for p in P]))
            bs = np.array([np.mean([spearmanr(yc[b], p[b]).statistic for p in P]) for b in boots]) - bh
            rows.append(dict(vs=f"crisprHAL 2 ({hk} configuration)", arm=lab, n=len(yc), n_genes=len(ug), rho=r, crisprhal=rh, diff=r - rh,
                             lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
            print(f"  {rows[-1]['vs']:36s} {lab:24s} rho {r:.4f} vs {rh:.4f}  diff {r - rh:+.4f} [{rows[-1]['lo']:+.4f}, {rows[-1]['hi']:+.4f}]", flush=True)
    out = pd.DataFrame(rows); out.to_csv(OUT / "clean_vs_hal.csv", index=False)
    return out


if __name__ == "__main__":
    run()
