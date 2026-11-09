"""Summaries of the PAM-profile learning runs: paired intervals over variants, and the campaign-level view.

Two layers of uncertainty are reported because the 770 variants are not independent: variants within a selection
campaign share a lineage and a selection pressure.
  variant bootstrap   variants resampled with replacement (fold membership travels with the variant), same resamples
                      for every arm, so differences are paired
  campaign view       the difference arm - reference computed inside each held-out campaign (7 of them) and the number
                      of campaigns on which the arm is ahead
Skill = 1 - SSE(arm) / SSE(training-mean model).

    python -m sgrna.ext.pam_analyze --split source --models ridge et mlp --seeds 0 1 2
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import pam_variants as pv
from .common import OUT
from .pam_learn import ARMS, fold_ids

SUBSETS = ("all", "active", "random_library")


def _subsets(d):
    return {"all": np.ones(len(d), bool), "active": d["Active variant?"].to_numpy(bool),
            "random_library": (d["selected/rational"] == "random").to_numpy()}


def run(split, models, seeds, B=300, tag="", ref="id"):
    d, Y, _ = pv.load()
    od = OUT / "pam_variants"
    subs = _subsets(d)
    Fs = [fold_ids(split, d, sd) for sd in seeds]
    Pm = [np.load(od / f"oof_{split}_{models[0]}_mean_s{sd}{tag}.npy") for sd in seeds]
    rng = np.random.default_rng(7)
    rows, camp = [], []
    for model in models:
        arms = [a for a in ARMS if a != "mean" and all((od / f"oof_{split}_{model}_{a}_s{sd}{tag}.npy").exists()
                                                       for sd in seeds)]
        Ps = {a: [np.load(od / f"oof_{split}_{model}_{a}_s{sd}{tag}.npy") for sd in seeds] for a in arms}

        def stat(a, s):
            out = []
            for k in range(len(seeds)):
                m = pv.fold_metrics(Y[s], Ps[a][k][s], Fs[k][s]); mm = pv.fold_metrics(Y[s], Pm[k][s], Fs[k][s])
                out.append((m["rho_pam"], m["rho_variant"], 1.0 - m["sse"] / mm["sse"]))
            return np.mean(out, axis=0)

        for sub in SUBSETS:
            idx = np.where(subs[sub])[0]
            boots = [rng.choice(idx, len(idx), replace=True) for _ in range(B)]
            pt = {a: stat(a, idx) for a in arms}
            bs = {a: np.array([stat(a, s) for s in boots]) for a in arms}
            for a in arms:
                diff = bs[a] - bs[ref] if a != ref else np.zeros_like(bs[a])
                for k, nm in enumerate(["rho_pam", "rho_variant", "skill"]):
                    rows.append(dict(split=split, model=model, subset=sub, arm=a, metric=nm, value=pt[a][k],
                                     lo=np.percentile(bs[a][:, k], 2.5), hi=np.percentile(bs[a][:, k], 97.5),
                                     d_vs_ref=pt[a][k] - pt[ref][k], d_lo=np.percentile(diff[:, k], 2.5),
                                     d_hi=np.percentile(diff[:, k], 97.5)))
        # campaign view (subset all): per-campaign rho_pam for each arm, averaged over seeds
        f0 = Fs[0]
        names = sorted(set(d["variant source"])) if split == "source" else None
        if names is not None:
            for a in arms:
                for k, u in enumerate(names):
                    m = (d["variant source"] == u).to_numpy()
                    v = np.mean([pv.fold_metrics(Y[m], Ps[a][j][m], np.zeros(m.sum(), int))["rho_pam"]
                                 for j in range(len(seeds))])
                    camp.append(dict(model=model, arm=a, campaign=u, rho_pam=v))
    out = pd.DataFrame(rows)
    out.to_csv(od / f"summary_{split}{tag}.csv", index=False)
    if camp:
        c = pd.DataFrame(camp); c.to_csv(od / f"campaign_{split}{tag}.csv", index=False)
    return out


def show(split="source", tag="", subs=("all", "random_library"), metrics=("rho_pam", "skill"), arms=None):
    o = pd.read_csv(OUT / "pam_variants" / f"summary_{split}{tag}.csv")
    if arms:
        o = o[o.arm.isin(arms)]
    for sub in subs:
        for met in metrics:
            t = o[(o.subset == sub) & (o.metric == met)].copy()
            t["val"] = t.apply(lambda r: f"{r.value:+.3f}", axis=1)
            t["d_id"] = t.apply(lambda r: f"{r.d_vs_ref:+.3f} [{r.d_lo:+.3f},{r.d_hi:+.3f}]", axis=1)
            print(f"\n{split} {sub} {met}")
            print(t.pivot(index="arm", columns="model", values=["val", "d_id"]).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="source"); ap.add_argument("--models", nargs="+", default=["ridge", "et"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0]); ap.add_argument("--tag", default="")
    ap.add_argument("--B", type=int, default=300)
    a = ap.parse_args()
    run(a.split, a.models, a.seeds, a.B, a.tag)
    show(a.split, a.tag)
