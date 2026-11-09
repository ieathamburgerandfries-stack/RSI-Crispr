"""Joint hyperparameter search for the E. coli model, with the selection kept apart from the confirmation.

Why this design. An earlier one-knob-at-a-time sweep found that `max_depth` 4 is about +0.006 rho too shallow, and
measured that a 25-arm sweep on one fold partition carries an adaptive component of +0.0087 -- larger than most single
knob effects. So the search here is judged on genes it never saw:

  cohort     curated E. coli cutting cohort, base + family A (6,517 columns), 30,203 guides with a gene label
  split      genes are divided once into DEV (70%) and CONFIRM (30%); nothing about CONFIRM is used to choose anything
  stage 1    N random configurations, 5 gene-grouped folds inside DEV (one seed)
  stage 2    the best few re-scored on three further gene-grouped partitions of DEV (this shrinks the winner's curse)
  confirm    champion, the old depth-6 finding, and the selected configurations are trained on all of DEV and scored on
             CONFIRM; differences are paired and bootstrapped over genes
  external   the same configurations on crisprHAL's eSpCas9 screen with its own folds (`tune_ecoli --external`)

Column selection is model-independent (the project's fixed selector XGBoost, 300 trees, depth 4), so the ranking of the
top 900 columns is computed once per fold and every configuration takes its top `n_features` from it. The number of trees
is a free dimension: each model is fitted with up to 2,000 trees and scored at a grid of tree counts through
`iteration_range`, the best count (by mean rho over the DEV folds) being part of the configuration.

    python -m sgrna.ext.tune_ecoli --stage 1 --draws 200
    python -m sgrna.ext.tune_ecoli --stage 2 --top 15
    python -m sgrna.ext.tune_ecoli --confirm
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm, rankdata, spearmanr

from sgrna import config
from sgrna.curated_ablation import _base, _block, _groups
from sgrna.run_ablation import _impute, champion_params, grouped_folds, make_selector

from .common import INTERIM, OUT

DIR = INTERIM / "tune"
DIR.mkdir(parents=True, exist_ok=True)
SPLIT_SEED = 2026
KGRID = (100, 200, 300, 400, 600, 800, 1200, 1600, 2000)
KGRID_EXT = KGRID + (2400, 3000, 4000)       # for configurations whose best tree count sat at the 2,000 cap
CHAMPION = dict(max_depth=4, learning_rate=0.05, n_estimators=400, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=1, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0, max_bin=256, nf=300, target="raw")
DEPTH6 = dict(CHAMPION, max_depth=6, n_estimators=800)


# ------------------------------------------------------------------ data
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


def rankings(co: Cohort, seed: int, n_splits=5):
    """Per DEV fold: train rows, validation rows, top-900 column ranking (cached)."""
    path = DIR / f"rank_dev_s{seed}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return [(z[f"tr{i}"], z[f"va{i}"], z[f"top{i}"]) for i in range(n_splits)]
    out, save = [], {}
    for i, (tr, va) in enumerate(grouped_folds(co.g[co.dev], n_splits, seed)):
        tr, va = co.dev[tr], co.dev[va]
        A, _ = _impute(co.X[tr], co.X[va][:1])
        sel = make_selector(seed); sel.fit(A, co.y[tr])
        top = np.argsort(sel.feature_importances_)[::-1][:900]
        out.append((tr, va, top)); save.update({f"tr{i}": tr, f"va{i}": va, f"top{i}": top})
    np.savez(path, **save)
    return out


# ------------------------------------------------------------------ search space
def sample(i: int) -> dict:
    r = np.random.default_rng(10_000 + i)
    depth = int(r.choice([3, 4, 5, 6, 7, 8, 9, 10]))
    lr = float(np.exp(r.uniform(np.log(0.01), np.log(0.2))))
    return dict(max_depth=depth, learning_rate=lr, n_estimators=800 if depth >= 9 else 2000,
                subsample=float(r.uniform(0.5, 1.0)), colsample_bytree=float(r.uniform(0.3, 1.0)),
                min_child_weight=float(np.exp(r.uniform(0, np.log(60)))),
                reg_lambda=float(np.exp(r.uniform(np.log(0.1), np.log(50)))),
                reg_alpha=float(r.choice([0, 0.01, 0.1, 1, 5])), gamma=float(r.choice([0, 0, 0.1, 0.5, 2])),
                max_bin=int(r.choice([256, 64])), nf=int(r.choice([150, 300, 450, 600, 900])),
                target=str(r.choice(["raw", "rankgauss"])))


def _target(y, kind):
    return norm.ppf((rankdata(y) - 0.5) / len(y)) if kind == "rankgauss" else y


def xgb_params(cfg, seed):
    p = dict(champion_params(seed))
    p.update(max_depth=int(cfg["max_depth"]), learning_rate=cfg["learning_rate"], n_estimators=int(cfg["n_estimators"]),
             subsample=cfg["subsample"], colsample_bytree=cfg["colsample_bytree"], min_child_weight=cfg["min_child_weight"],
             reg_lambda=cfg["reg_lambda"], reg_alpha=cfg["reg_alpha"], gamma=cfg["gamma"], max_bin=int(cfg["max_bin"]))
    return p


def fit_predict(cfg, X_tr, y_tr, X_te, seed, ks=None):
    """Fit once; return {k: prediction} for every k in `ks` (or the configured n_estimators if None)."""
    m = xgb.XGBRegressor(**xgb_params(cfg, seed)).fit(X_tr, _target(y_tr, cfg["target"]))
    if ks is None:
        return {int(cfg["n_estimators"]): m.predict(X_te)}
    return {k: m.predict(X_te, iteration_range=(0, k)) for k in ks if k <= cfg["n_estimators"]}


def score_dev(co: Cohort, cfg, seed, ks=KGRID):
    """Per-fold rho at each tree count for a configuration, on the DEV folds of `seed`."""
    rows = []
    for fold, (tr, va, top) in enumerate(rankings(co, seed)):
        cols = top[:int(cfg["nf"])]
        A, B = _impute(co.X[tr][:, cols], co.X[va][:, cols])
        preds = fit_predict(cfg, A, co.y[tr], B, seed, ks)
        for k, p in preds.items():
            rows.append(dict(seed=seed, fold=fold, k=k, rho=float(spearmanr(co.y[va], p).statistic)))
    return rows


def best_k(rows):
    d = pd.DataFrame(rows).groupby("k").rho.mean()
    return int(d.idxmax()), float(d.max())


# ------------------------------------------------------------------ stages
def stage1(co, draws, seed=41):
    cfgf, resf = OUT / "tune_configs.json", OUT / "tune_stage1.csv"
    cfgs = json.loads(cfgf.read_text()) if cfgf.exists() else {}
    res = pd.read_csv(resf) if resf.exists() else pd.DataFrame()
    done = set(res.draw.unique()) if len(res) else set()
    todo = [("champion", CHAMPION), ("depth6", DEPTH6)] + [(f"d{i:03d}", sample(i)) for i in range(draws)]
    t0 = time.time()
    for name, cfg in todo:
        if name in done:
            continue
        cfgs[name] = cfg
        rows = score_dev(co, cfg, seed, KGRID if name not in ("champion", "depth6") else KGRID)
        df = pd.DataFrame(rows); df.insert(0, "draw", name)
        res = pd.concat([res, df], ignore_index=True)
        res.to_csv(resf, index=False); cfgf.write_text(json.dumps(cfgs, indent=1))
        k, r = best_k(rows)
        print(f"  {name}  depth {cfg['max_depth']} lr {cfg['learning_rate']:.3f} nf {cfg['nf']} {cfg['target']:9s} "
              f"best k {k:4d} rho {r:.4f}   ({time.time() - t0:.0f}s)", flush=True)


def stage2(co, top, seeds=(42, 43, 44)):
    cfgf = OUT / "tune_configs.json"
    cfgs = json.loads(cfgf.read_text())
    s1 = pd.read_csv(OUT / "tune_stage1.csv")
    m = s1.groupby(["draw", "k"]).rho.mean().reset_index().sort_values("rho", ascending=False)
    best = m.drop_duplicates("draw").set_index("draw")
    names = [n for n in best.index if n not in ("champion", "depth6")][:top]
    res_f = OUT / "tune_stage2.csv"
    res = pd.read_csv(res_f) if res_f.exists() else pd.DataFrame()
    done = set(res.draw.unique()) if len(res) else set()
    for name in ["champion", "depth6"] + names:
        cfg, ks, label = cfgs[name], KGRID, name
        if name in best.index and int(best.loc[name, "k"]) >= int(cfg["n_estimators"]) and name not in ("champion", "depth6"):
            # the best tree count sat at the cap: allow more trees, kept under a new name so the original stays on record
            label = name + "x"; cfg = dict(cfg, n_estimators=4000); cfgs[label] = cfg; ks = KGRID_EXT
            cfgf.write_text(json.dumps(cfgs, indent=1))
        if label in done:
            continue
        rows = []
        for sd in seeds:
            rows += score_dev(co, cfg, sd, ks)
        df = pd.DataFrame(rows); df.insert(0, "draw", label)
        res = pd.concat([res, df], ignore_index=True); res.to_csv(res_f, index=False)
        k, r = best_k(rows)
        print(f"  {label}  stage-1 rho {best.loc[name, 'rho'] if name in best.index else float('nan'):.4f} "
              f"-> stage-2 best k {k} rho {r:.4f}", flush=True)


def selected(top_ens=5):
    """Stage-2 ranking: (name, best tree count, mean rho) for the anchors and the finalists."""
    s2 = pd.read_csv(OUT / "tune_stage2.csv")
    m = s2.groupby(["draw", "k"]).rho.mean().reset_index().sort_values("rho", ascending=False).drop_duplicates("draw")
    return m.set_index("draw")


def _z(p):
    return (p - p.mean()) / (p.std() + 1e-12)


def _boot_rho(y, genes, preds, B=2000, seed=0):
    """preds: {arm: (n_seeds, n) array}. Gene bootstrap of the seed-averaged Spearman and of paired differences."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(genes, return_inverse=True)
    rows_of = [np.where(inv == i)[0] for i in range(len(ug))]
    point = {a: float(np.mean([spearmanr(y, p).statistic for p in P])) for a, P in preds.items()}
    bs = {a: [] for a in preds}
    for _ in range(B):
        idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(ug), len(ug))])
        for a, P in preds.items():
            bs[a].append(np.mean([spearmanr(y[idx], p[idx]).statistic for p in P]))
    return point, {a: np.array(v) for a, v in bs.items()}


def confirm(co, seeds=(41, 42, 43), members=5, B=1000):
    """Train on all of DEV, score CONFIRM genes. Arms: champion, depth6, best finalist, ensemble of the best `members`."""
    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    sel = selected()
    finalists = [n for n in sel.index if n not in ("champion", "depth6")]
    best = finalists[0]
    arms = {"champion": ("champion", int(sel.loc["champion", "k"])), "depth6": ("depth6", int(sel.loc["depth6", "k"])),
            f"tuned best ({best})": (best, int(sel.loc[best, "k"]))}
    for n in finalists[:members]:
        arms.setdefault(n, (n, int(sel.loc[n, "k"])))
    preds = {}
    for arm, (name, k) in arms.items():
        cfg = dict(cfgs[name], n_estimators=k)
        P = []
        for sd in seeds:
            f = DIR / f"conf_{name}_k{k}_s{sd}.npy"
            if f.exists():
                P.append(np.load(f)); continue
            rk = DIR / f"rank_devall_s{sd}.npy"
            if rk.exists():
                top = np.load(rk)
            else:
                A, _ = _impute(co.X[co.dev], co.X[co.conf][:1])
                selr = make_selector(sd); selr.fit(A, co.y[co.dev])
                top = np.argsort(selr.feature_importances_)[::-1][:900]; np.save(rk, top)
            cols = top[:int(cfg["nf"])]
            A, B_ = _impute(co.X[co.dev][:, cols], co.X[co.conf][:, cols])
            p = fit_predict(cfg, A, co.y[co.dev], B_, sd)[k]
            np.save(f, p); P.append(p)
        preds[arm] = np.array(P)
    ens = [n for n in finalists[:members]]
    preds[f"ensemble of top {members}"] = np.array([np.mean([_z(preds[n][i]) for n in ens], axis=0) for i in range(len(seeds))])
    y, g = co.y[co.conf], co.g[co.conf]
    point, bs = _boot_rho(y, g, preds, B=B)
    rows = []
    for a in preds:
        d = bs[a] - bs["champion"]
        rows.append(dict(arm=a, rho=point[a], rho_lo=np.percentile(bs[a], 2.5), rho_hi=np.percentile(bs[a], 97.5),
                         d_vs_champion=point[a] - point["champion"], d_lo=np.percentile(d, 2.5), d_hi=np.percentile(d, 97.5),
                         per_seed=";".join(f"{spearmanr(y, p).statistic:.4f}" for p in preds[a])))
    out = pd.DataFrame(rows); out.to_csv(OUT / "tune_confirm.csv", index=False)
    return out


def external(names, seeds=(41, 42)):
    """Same configurations on crisprHAL's eSpCas9 screen, its own folds (100-kb blocks), paired by fold."""
    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    sel = selected()
    if names == ["auto"]:
        names = ["champion", "depth6"] + [n for n in sel.index if n not in ("champion", "depth6")][:3]
    meta = json.loads((config.INTERIM / "transfer_eSpCas9" / "meta.json").read_text())
    X = np.asarray(np.load(config.INTERIM / "transfer_eSpCas9" / "X.npy", mmap_mode="r"), np.float32)
    y = np.asarray(meta["y"], float); left = np.asarray(meta["left"], float)
    ok = np.isfinite(left) & np.isfinite(y)
    X, y = X[ok], y[ok]; grp = (left[ok] // 100_000).astype(int).astype(str)
    rows = []
    for sd in seeds:
        for fold, (tr, va) in enumerate(grouped_folds(grp, 5, sd)):
            rk = DIR / f"rank_esp_s{sd}_f{fold}.npy"
            if rk.exists():
                top = np.load(rk)
            else:
                A, _ = _impute(X[tr], X[va][:1]); selr = make_selector(sd); selr.fit(A, y[tr])
                top = np.argsort(selr.feature_importances_)[::-1][:900]; np.save(rk, top)
            for name in names:
                k = int(sel.loc[name, "k"]); cfg = dict(cfgs[name], n_estimators=k)
                cols = top[:int(cfg["nf"])]
                A, B_ = _impute(X[tr][:, cols], X[va][:, cols])
                p = fit_predict(cfg, A, y[tr], B_, sd)[k]
                rows.append(dict(arm=name, seed=sd, fold=fold, rho=float(spearmanr(y[va], p).statistic)))
            print(f"  esp seed {sd} fold {fold} done", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "tune_external.csv", index=False)
    return pd.DataFrame(rows)


def full_cv(co, seeds=(41, 42, 43), members=("d099x", "d077x", "d021x", "d080x", "d000x")):
    """The project's standard protocol on the whole cohort (gene-grouped 5 folds x 3 seeds, selection inside folds).

    The tuned configurations were chosen on 70% of these genes, so this is the number to compare with earlier tables, not an
    unbiased estimate of the gain: the held-out-gene run (`confirm`) is that."""
    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    sel = selected()
    arms = {"champion": ["champion"], "depth6": ["depth6"], "tuned d099x": ["d099x"], "ensemble of 5": list(members)}
    rows = []
    for sd in seeds:
        for fold, (tr, va) in enumerate(grouped_folds(co.g, 5, sd)):
            A, _ = _impute(co.X[tr], co.X[va][:1]); sl = make_selector(sd); sl.fit(A, co.y[tr])
            top = np.argsort(sl.feature_importances_)[::-1][:900]
            preds = {}
            for name in {n for v in arms.values() for n in v}:
                k = int(sel.loc[name, "k"]); cfg = dict(cfgs[name], n_estimators=k)
                cols = top[:int(cfg["nf"])]
                A, B = _impute(co.X[tr][:, cols], co.X[va][:, cols])
                preds[name] = fit_predict(cfg, A, co.y[tr], B, sd)[k]
            for arm, names in arms.items():
                p = np.mean([_z(preds[n]) for n in names], axis=0)
                rows.append(dict(arm=arm, seed=sd, fold=fold, rho=float(spearmanr(co.y[va], p).statistic)))
            pd.DataFrame(rows).to_csv(OUT / "tune_fullcv.csv", index=False)
            print(f"  seed {sd} fold {fold}: " + " ".join(f"{r['arm']}={r['rho']:.4f}" for r in rows[-4:]), flush=True)
    return pd.DataFrame(rows)


def budget_check(co, name="d077x", seed=42):
    """One DEV fold: does a wider column budget help at the tuned setting? Also counts the columns the selector uses."""
    cfg = dict(json.loads((OUT / "tune_configs.json").read_text())[name], n_estimators=4000)
    tr, va, _ = rankings(co, seed)[0]
    A, _ = _impute(co.X[tr], co.X[va][:1]); sl = make_selector(seed); sl.fit(A, co.y[tr])
    order = np.argsort(sl.feature_importances_)[::-1]
    rows = [dict(nf=-1, k=0, rho=float("nan"), n_nonzero_gain=int((sl.feature_importances_ > 0).sum()))]
    for nf in (450, 900, 1500, 2500):
        cols = order[:nf]; A, B = _impute(co.X[tr][:, cols], co.X[va][:, cols])
        for k, p in fit_predict(cfg, A, co.y[tr], B, seed, (1200, 2000, 3000, 4000)).items():
            rows.append(dict(nf=nf, k=k, rho=float(spearmanr(co.y[va], p).statistic), n_nonzero_gain=np.nan))
    out = pd.DataFrame(rows); out.to_csv(OUT / "tune_budget_check.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--fullcv", action="store_true")
    ap.add_argument("--budget", action="store_true")
    ap.add_argument("--external", nargs="*", default=None, help="draw names to run on the eSpCas9 screen")
    a = ap.parse_args()
    co = Cohort()
    print(f"cohort {co.X.shape}; DEV {len(co.dev):,} guides / CONFIRM {len(co.conf):,} guides "
          f"({len(set(co.g[co.dev]))} / {len(set(co.g[co.conf]))} genes)", flush=True)
    if a.stage == 1:
        stage1(co, a.draws)
    elif a.stage == 2:
        stage2(co, a.top)
    if a.confirm:
        pd.set_option("display.width", 220)
        print(confirm(co).round(4).to_string(index=False))
    if a.budget:
        print(budget_check(co).round(4).to_string(index=False))
    if a.fullcv:
        full_cv(co)
    if a.external is not None:
        external(a.external)


if __name__ == "__main__":
    main()
