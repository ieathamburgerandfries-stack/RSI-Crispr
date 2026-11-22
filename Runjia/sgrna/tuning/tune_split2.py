"""Does the tuning payoff replicate on a second gene split? A fresh search, judged on a fresh set of held-out genes.

Why a second split is not a formality. Everything the fork claims about tuning rests on one partition of genes
(`tune_ecoli.SPLIT_SEED = 2026`): the +0.0143 on held-out genes, the configurations d099x/d077x/..., the blend weights,
and the untouched-genes comparison against crisprHAL 2. The bootstrap intervals resample genes *within* that split, so
they say nothing about split-to-split variation, which is the one source of error a single partition cannot show.

What must not be done, and why. Simply re-scoring d099x on a different CONFIRM set would be contaminated: d099x was chosen
using the DEV genes of split 1, and a new split reshuffles those genes into both halves, so some of the new CONFIRM genes
contributed labels to the selection. The claim at risk is not "d099x is good" but "a search on 70% of the genes yields a
configuration that beats the champion on the other 30%", so the honest replication re-runs the *procedure*: a fresh random
search on the new DEV, a fresh winner, confirmed on genes that search never saw. The winner may well be a different
configuration -- that is the point, not a problem.

Design mirrors `tune_ecoli` at reduced cost, and is resumable draw by draw so a partial run is still a usable answer:

  split      genes re-divided 70/30 with a different seed; asserted to share no gene across the halves
  stage 1    N fresh configurations (a different stream of the same search space) on 5 gene-grouped folds of the new DEV
  stage 2    the best 3 re-scored on one further gene-grouped partition of the new DEV, with the extended tree grid for
             any whose best tree count sat at the cap
  confirm    champion, the new winner and the top-3 average trained on all of the new DEV, scored on the new CONFIRM,
             three seeds, paired bootstrap over genes

d099x is also scored on the new CONFIRM as a secondary row, flagged `contaminated` because its genes overlap the new
confirmation set. It is reported for completeness and is not the replication.

    python -m sgrna.tuning.tune_split2 --draws 40
    python -m sgrna.tuning.tune_split2 --confirm
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sgrna.cohorts.curated_ablation import _base, _block, _groups
from sgrna.harness.run_ablation import _impute, grouped_folds, make_selector

from . import tune_ecoli as T
from ..common import OUT

SPLIT_SEED = 7           # deliberately not tune_ecoli.SPLIT_SEED (2026)
DRAW_OFFSET = 50_000     # a different stream of the same search space, so the configurations are fresh
DIR = T.DIR / "split2"
DIR.mkdir(parents=True, exist_ok=True)
CFG_F = OUT / "tune_split2_configs.json"


class Cohort:
    def __init__(self):
        X, y, _, protos = _base()
        A, _ = _block("a_flank", protos)
        g, ok = _groups("gene", protos)
        self.X = np.hstack([X, A])[ok]
        self.y = y[ok]
        self.g = np.asarray(g[ok], dtype=object)
        genes = np.array(sorted(set(self.g)))
        rng = np.random.default_rng(SPLIT_SEED)
        conf = set(rng.choice(genes, size=int(round(0.3 * len(genes))), replace=False))
        self.is_conf = np.array([x in conf for x in self.g])
        self.dev = np.where(~self.is_conf)[0]
        self.conf = np.where(self.is_conf)[0]
        assert not (set(self.g[self.dev]) & set(self.g[self.conf])), "the two halves share a gene"
        old = np.random.default_rng(T.SPLIT_SEED)
        old_conf = set(old.choice(genes, size=int(round(0.3 * len(genes))), replace=False))
        shared = len(conf & old_conf)
        print(f"cohort {self.X.shape}; DEV {len(self.dev):,} / CONFIRM {len(self.conf):,} guides "
              f"({len(set(self.g[self.dev])):,} / {len(set(self.g[self.conf])):,} genes); "
              f"{shared:,} of {len(conf):,} CONFIRM genes were also CONFIRM in split 1", flush=True)


def rankings(co, seed, n_splits=5):
    """Per DEV fold: train rows, validation rows, top-900 ranking. Cached under split2/ so it cannot collide with split 1."""
    path = DIR / f"rank_dev_s{seed}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return [(z[f"tr{i}"], z[f"va{i}"], z[f"top{i}"]) for i in range(n_splits)]
    out, save = [], {}
    for i, (tr, va) in enumerate(grouped_folds(co.g[co.dev], n_splits, seed)):
        tr, va = co.dev[tr], co.dev[va]
        A, _ = _impute(co.X[tr], co.X[va][:1])
        sel = make_selector(seed)
        sel.fit(A, co.y[tr])
        top = np.argsort(sel.feature_importances_)[::-1][:900]
        out.append((tr, va, top))
        save.update({f"tr{i}": tr, f"va{i}": va, f"top{i}": top})
    np.savez(path, **save)
    return out


def score_dev(co, cfg, seed, ks=T.KGRID):
    rows = []
    for fold, (tr, va, top) in enumerate(rankings(co, seed)):
        cols = top[:int(cfg["nf"])]
        A, B = _impute(co.X[tr][:, cols], co.X[va][:, cols])
        for k, p in T.fit_predict(cfg, A, co.y[tr], B, seed, ks).items():
            rows.append(dict(seed=seed, fold=fold, k=k, rho=float(spearmanr(co.y[va], p).statistic)))
    return rows


def stage1(co, draws, seed=51):
    cfgs = json.loads(CFG_F.read_text()) if CFG_F.exists() else {}
    f = OUT / "tune_split2_stage1.csv"
    res = pd.read_csv(f) if f.exists() else pd.DataFrame()
    done = set(res.draw.unique()) if len(res) else set()
    todo = [("champion", T.CHAMPION), ("depth6", T.DEPTH6)] + [(f"s2d{i:03d}", T.sample(DRAW_OFFSET + i)) for i in range(draws)]
    t0 = time.time()
    for name, cfg in todo:
        cfgs[name] = cfg
        if name in done:
            continue
        rows = score_dev(co, cfg, seed)
        df = pd.DataFrame(rows)
        df.insert(0, "draw", name)
        res = pd.concat([res, df], ignore_index=True)
        res.to_csv(f, index=False)
        CFG_F.write_text(json.dumps(cfgs, indent=1))
        k, r = T.best_k(rows)
        print(f"  {name}  depth {cfg['max_depth']} lr {cfg['learning_rate']:.3f} nf {cfg['nf']} {cfg['target']:9s} "
              f"best k {k:4d} rho {r:.4f}   ({time.time() - t0:.0f}s)", flush=True)


def stage2(co, top=3, seed=52):
    cfgs = json.loads(CFG_F.read_text())
    s1 = pd.read_csv(OUT / "tune_split2_stage1.csv")
    m = s1.groupby(["draw", "k"]).rho.mean().reset_index().sort_values("rho", ascending=False).drop_duplicates("draw")
    best = m.set_index("draw")
    names = [n for n in best.index if n not in ("champion", "depth6")][:top]
    f = OUT / "tune_split2_stage2.csv"
    res = pd.read_csv(f) if f.exists() else pd.DataFrame()
    done = set(res.draw.unique()) if len(res) else set()
    for name in ["champion", "depth6"] + names:
        cfg, ks, label = cfgs[name], T.KGRID, name
        if name not in ("champion", "depth6") and int(best.loc[name, "k"]) >= int(cfg["n_estimators"]):
            label = name + "x"
            cfg = dict(cfg, n_estimators=4000)
            cfgs[label] = cfg
            ks = T.KGRID_EXT
            CFG_F.write_text(json.dumps(cfgs, indent=1))
        if label in done:
            continue
        rows = score_dev(co, cfg, seed, ks)
        df = pd.DataFrame(rows)
        df.insert(0, "draw", label)
        res = pd.concat([res, df], ignore_index=True)
        res.to_csv(f, index=False)
        k, r = T.best_k(rows)
        print(f"  {label}  stage-2 best k {k} rho {r:.4f}", flush=True)


def selected():
    s2 = pd.read_csv(OUT / "tune_split2_stage2.csv")
    return (s2.groupby(["draw", "k"]).rho.mean().reset_index().sort_values("rho", ascending=False)
            .drop_duplicates("draw").set_index("draw"))


def confirm(co, seeds=(41, 42, 43), B=1000):
    cfgs = json.loads(CFG_F.read_text())
    cfgs.update(json.loads((OUT / "tune_configs.json").read_text()))        # for the contaminated d099x row
    sel = selected()
    finalists = [n for n in sel.index if n not in ("champion", "depth6")]
    arms = {"champion": ("champion", int(sel.loc["champion", "k"])),
            "depth6": ("depth6", int(sel.loc["depth6", "k"]))}
    for n in finalists[:3]:
        arms[f"split-2 winner ({n})" if n == finalists[0] else n] = (n, int(sel.loc[n, "k"]))
    arms["d099x (contaminated)"] = ("d099x", int(T.selected().loc["d099x", "k"]))
    preds = {}
    for arm, (name, k) in arms.items():
        cfg = dict(cfgs[name], n_estimators=k)
        P = []
        for sd in seeds:
            f = DIR / f"conf_{name}_k{k}_s{sd}.npy"
            if f.exists():
                P.append(np.load(f))
                continue
            rk = DIR / f"rank_devall_s{sd}.npy"
            if rk.exists():
                top = np.load(rk)
            else:
                A, _ = _impute(co.X[co.dev], co.X[co.conf][:1])
                selr = make_selector(sd)
                selr.fit(A, co.y[co.dev])
                top = np.argsort(selr.feature_importances_)[::-1][:900]
                np.save(rk, top)
            cols = top[:int(cfg["nf"])]
            A, B_ = _impute(co.X[co.dev][:, cols], co.X[co.conf][:, cols])
            p = T.fit_predict(cfg, A, co.y[co.dev], B_, sd)[k]
            np.save(f, p)
            P.append(p)
            print(f"    {arm} seed {sd} done", flush=True)
        preds[arm] = np.array(P)
    ens = [a for a in arms if a not in ("champion", "depth6", "d099x (contaminated)")]
    preds["average of top 3"] = np.array([np.mean([T._z(preds[n][i]) for n in ens], axis=0) for i in range(len(seeds))])
    y, g = co.y[co.conf], co.g[co.conf]
    point, bs = T._boot_rho(y, g, preds, B=B)
    rows = []
    for a in preds:
        d = bs[a] - bs["champion"]
        rows.append(dict(arm=a, rho=point[a], rho_lo=np.percentile(bs[a], 2.5), rho_hi=np.percentile(bs[a], 97.5),
                         d_vs_champion=point[a] - point["champion"], d_lo=np.percentile(d, 2.5),
                         d_hi=np.percentile(d, 97.5),
                         per_seed=";".join(f"{spearmanr(y, p).statistic:.4f}" for p in preds[a])))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tune_split2_confirm.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=40)
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    co = Cohort()
    pd.set_option("display.width", 220)
    if a.stage1 or not (a.stage2 or a.confirm):
        stage1(co, a.draws)
    if a.stage2 or not (a.stage1 or a.confirm):
        stage2(co)
    if a.confirm or not (a.stage1 or a.stage2):
        print(confirm(co).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
