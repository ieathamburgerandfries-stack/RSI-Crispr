"""Predict a variant's 64-PAM profile from descriptors of its six PI-domain residues (Silverstein 2025 data).

Splits
  random   10-fold x 3 repeats on variants: optimistic, because 5% of variants have a neighbour 1 residue away
  source   leave one selection campaign out (7 campaigns, each selected on a different PAM set or unselected):
           the analogue of leave-one-nuclease-out, and the headline split
Models (capacity varies on purpose; a conclusion has to hold across them)
  ridge    ridge on standardised columns, alpha by inner leave-one-out
  et       extra-trees, 300 trees, min leaf 2
  mlp      two hidden layers (128, 64), early stopping: the PAMmla family
Output: out-of-fold prediction matrices (variants x 64) saved per (split, model, arm), and a metrics table with
bootstrap intervals over variants.

    python -m sgrna.ext.pam_learn --split source --models ridge et
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from . import pam_variants as pv
from .common import OUT

warnings.filterwarnings("ignore")
ARMS = {
    "mean": [],
    "id": ["id"],
    "comp": ["comp"],
    "reach": ["reach"],
    "env": ["env"],
    "id+comp": ["id", "comp"],
    "id+reach": ["id", "reach"],
    "comp+reach": ["comp", "reach"],
    "id+comp+reach": ["id", "comp", "reach"],
    "id+comp+reach+env": ["id", "comp", "reach", "env"],
    "id_perm": ["id"],        # identity columns shuffled across variants in training: must score like 'mean'
}
# composition ablation: drop one property, or keep only one (properties in pv.COMP_NAMES)
for _p in pv.COMP_NAMES:
    ARMS[f"comp_drop_{_p}"] = ["comp=" + "+".join(x for x in pv.COMP_NAMES if x != _p)]
    ARMS[f"comp_only_{_p}"] = [f"comp={_p}"]


def make(model: str, seed: int):
    if model == "ridge":
        return RidgeCV(alphas=np.logspace(-2, 3, 11))
    if model == "et":
        return ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.5, n_jobs=-1,
                                   random_state=seed)
    if model == "mlp":
        return MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, n_iter_no_change=20,
                            max_iter=600, random_state=seed)
    raise ValueError(model)


def folds(split: str, d: pd.DataFrame, seed: int):
    n = len(d)
    if split == "random":
        return list(KFold(10, shuffle=True, random_state=seed).split(np.arange(n)))
    g = d["variant source"].to_numpy()
    return [(np.where(g != u)[0], np.where(g == u)[0]) for u in sorted(set(g))]


def oof(split, model, arm, d, Y, seed):
    groups = ARMS[arm]
    P = np.zeros_like(Y)
    X = pv.features(d, groups)[0] if groups else np.zeros((len(d), 1))
    rng = np.random.default_rng(1000 + seed)
    for tr, te in folds(split, d, seed):
        if not groups:
            P[te] = Y[tr].mean(0)
            continue
        Xtr, Xte = X[tr], X[te]
        if arm == "id_perm":
            Xtr = Xtr[rng.permutation(len(Xtr))]
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        mu = Y[tr].mean(0)
        m = make(model, seed).fit(Xtr, Y[tr] - mu)
        P[te] = m.predict(Xte) + mu
    return P


def fold_ids(split, d, seed):
    f = np.zeros(len(d), int)
    for k, (_, te) in enumerate(folds(split, d, seed)):
        f[te] = k
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="source")
    ap.add_argument("--models", nargs="+", default=["ridge", "et"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    d, Y, pam = pv.load()
    od = OUT / "pam_variants"; od.mkdir(parents=True, exist_ok=True)
    rows = []
    subsets = {"all": np.ones(len(d), bool), "active": d["Active variant?"].to_numpy(bool),
               "random_library": (d["selected/rational"] == "random").to_numpy()}
    for model in a.models:
        for arm in a.arms:
            if arm in ("mean",) and model != a.models[0]:
                continue
            t0 = time.time()
            Ps = []
            for sd in a.seeds:
                P = oof(a.split, model, arm, d, Y, sd)
                np.save(od / f"oof_{a.split}_{model}_{arm}_s{sd}{a.tag}.npy", P)
                Ps.append(P)
            for sub, mask in subsets.items():
                ms = [pv.fold_metrics(Y[mask], P[mask], fold_ids(a.split, d, sd)[mask]) for sd, P in zip(a.seeds, Ps)]
                rows.append(dict(split=a.split, model=model if arm != "mean" else "-", arm=arm, subset=sub,
                                 n=int(mask.sum()), **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]}))
            print(f"{a.split:7s} {model:5s} {arm:20s} {time.time() - t0:5.1f}s  "
                  f"all rho_pam={rows[-3]['rho_pam']:+.3f} rho_var={rows[-3]['rho_variant']:+.3f} rmse={rows[-3]['rmse']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(od / f"metrics_{a.split}_{'-'.join(a.models)}{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
