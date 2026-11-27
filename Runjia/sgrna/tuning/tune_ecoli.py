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

    python -m sgrna.tuning.tune_ecoli --stage 1 --draws 200
    python -m sgrna.tuning.tune_ecoli --stage 2 --top 15
    python -m sgrna.tuning.tune_ecoli --confirm
    python -m sgrna.tuning.tune_ecoli --analyze            # what the 200 draws say about each knob (writes nothing; the stored copy is tune_stage1_effects.txt)

**A second gene split** (`--split 2`) replicates the whole procedure, not the winner. Everything the first split claims rests on one partition of genes
(`SPLIT_SEED = 2026`): the +0.0143 on held-out genes, the configurations d099x/d077x/..., the blend weights, and the untouched-genes comparison against
crisprHAL 2. The bootstrap intervals resample genes *within* that split, so they cannot show split-to-split variation. Re-scoring d099x on a different
CONFIRM set would be contaminated (d099x was chosen on genes that a new split reshuffles into both halves), so the claim at risk -- "a search on 70% of
the genes yields a configuration that beats the champion on the other 30%" -- is tested by re-running the search: genes re-divided with seed 7 (asserted
to share no gene across the halves), 40 fresh configurations (a different stream of the same space), the best 3 re-scored on one further DEV partition,
and champion, the new winner and the top-3 average scored on the new CONFIRM genes. d099x is also scored there as a row flagged `contaminated`. The two
splits differ only in the `Plan` they run (seed, result files, cache folder, search size, labels); results go to `tune_split2_*` files.

    python -m sgrna.tuning.tune_ecoli --split 2 --stage 1 --draws 40
    python -m sgrna.tuning.tune_ecoli --split 2 --stage 2
    python -m sgrna.tuning.tune_ecoli --split 2 --confirm

What the 200 draws say (`--analyze`, `tune_stage1_effects.txt`): shallower trees and slower learning with more trees score best (mean rho 0.714 at depth
4, 0.698 at depth 10) and the other settings matter little. For each draw the tree count with the best mean rho is taken (that count is part of the
configuration), so a draw's score is the maximum over a tree grid, a small upward bias shared by all draws. Marginal effects are the mean score of the
draws in each level of a hyperparameter; because draws vary in all dimensions at once, a ridge regression of the score on the standardised
hyperparameters gives the effect of each one holding the others fixed.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm, rankdata, spearmanr

from sgrna import config
from sgrna.cohorts.curated_ablation import _base, _block, _groups
from sgrna.harness.run_ablation import _impute, champion_params, grouped_folds, make_selector

from ..common import INTERIM, OUT

DIR = INTERIM / "tune"
DIR.mkdir(parents=True, exist_ok=True)
SPLIT_SEED = 2026
KGRID = (100, 200, 300, 400, 600, 800, 1200, 1600, 2000)
KGRID_EXT = KGRID + (2400, 3000, 4000)       # for configurations whose best tree count sat at the 2,000 cap
CHAMPION = dict(max_depth=4, learning_rate=0.05, n_estimators=400, subsample=0.8, colsample_bytree=0.8,
                min_child_weight=1, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0, max_bin=256, nf=300, target="raw")
DEPTH6 = dict(CHAMPION, max_depth=6, n_estimators=800)
ANCHORS = ("champion", "depth6")


@dataclass(frozen=True)
class Plan:
    """One run of the search: which genes are held out, where the results and caches go, how large the search is, and how its rows are labelled."""
    split_seed: int            # seeds the 70/30 division of genes
    configs: str               # result files, in `results/post_freeze/`: the configurations drawn, ...
    stage1: str                # ... the stage-1 scores, ...
    stage2: str                # ... the stage-2 scores, ...
    confirm: str               # ... and the held-out confirmation (named in full so that a search for a file finds the code that writes it)
    cache: Path                # fold rankings and held-out predictions, kept apart per split
    draw_label: str            # stage-1 draw names: d000, s2d000
    draw_offset: int           # which stream of the search space (so that split 2's configurations are fresh)
    draws: int                 # default search size
    seed1: int                 # fold seed of stage 1
    seeds2: tuple              # fold seeds of stage 2
    top: int                   # finalists kept by stage 2
    members: int               # finalists in the ensemble
    winner_label: str          # confirm-table row of the best finalist
    winner_by_name: bool       # ... and whether it is also listed under its own name (split 1 does, split 2 does not)
    ensemble_label: str        # confirm-table row of the ensemble
    contaminated: str | None = None      # a configuration from another split, scored as a flagged row


PLANS = {
    1: Plan(SPLIT_SEED, "tune_configs.json", "tune_stage1.csv", "tune_stage2.csv", "tune_confirm.csv", DIR, "d", 0, 200, 41, (42, 43, 44), 15, 5,
            "tuned best ({})", True, "ensemble of top {}"),
    2: Plan(7, "tune_split2_configs.json", "tune_split2_stage1.csv", "tune_split2_stage2.csv", "tune_split2_confirm.csv", DIR / "split2", "s2d", 50_000, 40, 51, (52,),
            3, 3, "split-2 winner ({})", False, "average of top {}", "d099x"),
}


# ------------------------------------------------------------------ data
def _confirm_genes(genes, seed: int) -> set:
    return set(np.random.default_rng(seed).choice(genes, size=int(round(0.3 * len(genes))), replace=False))


class Cohort:
    def __init__(self, plan: Plan | None = None):
        self.plan = plan or PLANS[1]
        self.plan.cache.mkdir(parents=True, exist_ok=True)
        X, y, _, protos = _base()
        A, _ = _block("a_flank", protos)
        g, ok = _groups("gene", protos)
        self.X = np.hstack([X, A])[ok]
        self.y = y[ok]
        self.g = np.asarray(g[ok], dtype=object)
        genes = np.array(sorted(set(self.g)))
        conf = _confirm_genes(genes, self.plan.split_seed)
        self.is_conf = np.array([x in conf for x in self.g])
        self.dev = np.where(~self.is_conf)[0]
        self.conf = np.where(self.is_conf)[0]
        assert not (set(self.g[self.dev]) & set(self.g[self.conf])), "the two halves share a gene"
        self.shared_with_split1 = len(conf & _confirm_genes(genes, SPLIT_SEED))     # CONFIRM genes that were also CONFIRM in split 1


def rankings(co: Cohort, seed: int, n_splits=5):
    """Per DEV fold: train rows, validation rows, top-900 column ranking (cached, under the split's own cache folder)."""
    path = getattr(co, "plan", PLANS[1]).cache / f"rank_dev_s{seed}.npz"
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
def stage1(co, draws=None, seed=None):
    plan = co.plan
    draws = plan.draws if draws is None else draws
    seed = plan.seed1 if seed is None else seed
    cfgf, resf = OUT / plan.configs, OUT / plan.stage1
    cfgs = json.loads(cfgf.read_text()) if cfgf.exists() else {}
    res = pd.read_csv(resf) if resf.exists() else pd.DataFrame()
    done = set(res.draw.unique()) if len(res) else set()
    todo = [("champion", CHAMPION), ("depth6", DEPTH6)] + [(f"{plan.draw_label}{i:03d}", sample(plan.draw_offset + i)) for i in range(draws)]
    t0 = time.time()
    for name, cfg in todo:
        cfgs[name] = cfg
        if name in done:
            continue
        rows = score_dev(co, cfg, seed, KGRID)
        df = pd.DataFrame(rows); df.insert(0, "draw", name)
        res = pd.concat([res, df], ignore_index=True)
        res.to_csv(resf, index=False); cfgf.write_text(json.dumps(cfgs, indent=1))
        k, r = best_k(rows)
        print(f"  {name}  depth {cfg['max_depth']} lr {cfg['learning_rate']:.3f} nf {cfg['nf']} {cfg['target']:9s} "
              f"best k {k:4d} rho {r:.4f}   ({time.time() - t0:.0f}s)", flush=True)


def stage2(co, top=None, seeds=None):
    plan = co.plan
    top = plan.top if top is None else top
    seeds = plan.seeds2 if seeds is None else seeds
    cfgf = OUT / plan.configs
    cfgs = json.loads(cfgf.read_text())
    s1 = pd.read_csv(OUT / plan.stage1)
    m = s1.groupby(["draw", "k"]).rho.mean().reset_index().sort_values("rho", ascending=False)
    best = m.drop_duplicates("draw").set_index("draw")
    names = [n for n in best.index if n not in ANCHORS][:top]
    res_f = OUT / plan.stage2
    res = pd.read_csv(res_f) if res_f.exists() else pd.DataFrame()
    done = set(res.draw.unique()) if len(res) else set()
    for name in list(ANCHORS) + names:
        cfg, ks, label = cfgs[name], KGRID, name
        if name in best.index and int(best.loc[name, "k"]) >= int(cfg["n_estimators"]) and name not in ANCHORS:
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


def selected(plan: Plan | None = None):
    """Stage-2 ranking: (name, best tree count, mean rho) for the anchors and the finalists."""
    s2 = pd.read_csv(OUT / (plan or PLANS[1]).stage2)
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


def confirm(co, seeds=(41, 42, 43), members=None, B=1000):
    """Train on all of DEV, score CONFIRM genes. Arms: champion, depth6, best finalist, the next finalists, and the ensemble of the best `members`
    (on split 2 also a split-1 configuration, flagged contaminated because its genes overlap this CONFIRM set)."""
    plan = co.plan
    members = plan.members if members is None else members
    cfgs = json.loads((OUT / plan.configs).read_text())
    if plan.contaminated:
        cfgs.update(json.loads((OUT / "tune_configs.json").read_text()))
    sel = selected(plan)
    finalists = [n for n in sel.index if n not in ANCHORS]
    best = finalists[0]
    arms = {"champion": ("champion", int(sel.loc["champion", "k"])), "depth6": ("depth6", int(sel.loc["depth6", "k"])),
            plan.winner_label.format(best): (best, int(sel.loc[best, "k"]))}
    for n in finalists[:members]:
        if n != best or plan.winner_by_name:
            arms.setdefault(n, (n, int(sel.loc[n, "k"])))
    if plan.contaminated:
        arms[f"{plan.contaminated} (contaminated)"] = (plan.contaminated, int(selected(PLANS[1]).loc[plan.contaminated, "k"]))
    preds = {}
    for arm, (name, k) in arms.items():
        cfg = dict(cfgs[name], n_estimators=k)
        P = []
        for sd in seeds:
            f = plan.cache / f"conf_{name}_k{k}_s{sd}.npy"
            if f.exists():
                P.append(np.load(f)); continue
            rk = plan.cache / f"rank_devall_s{sd}.npy"
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
    ens = [n if (n != best or plan.winner_by_name) else plan.winner_label.format(best) for n in finalists[:members]]
    preds[plan.ensemble_label.format(members)] = np.array([np.mean([_z(preds[n][i]) for n in ens], axis=0) for i in range(len(seeds))])
    y, g = co.y[co.conf], co.g[co.conf]
    point, bs = _boot_rho(y, g, preds, B=B)
    rows = []
    for a in preds:
        d = bs[a] - bs["champion"]
        rows.append(dict(arm=a, rho=point[a], rho_lo=np.percentile(bs[a], 2.5), rho_hi=np.percentile(bs[a], 97.5),
                         d_vs_champion=point[a] - point["champion"], d_lo=np.percentile(d, 2.5), d_hi=np.percentile(d, 97.5),
                         per_seed=";".join(f"{spearmanr(y, p).statistic:.4f}" for p in preds[a])))
    out = pd.DataFrame(rows); out.to_csv(OUT / plan.confirm, index=False)
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


def stage1_table(plan: Plan | None = None) -> pd.DataFrame:
    """One row per stage-1 draw: its configuration, its best mean rho over the tree grid, and the tree count that gave it."""
    plan = plan or PLANS[1]
    cfgs = json.loads((OUT / plan.configs).read_text())
    s = pd.read_csv(OUT / plan.stage1)
    m = s.groupby(["draw", "k"]).rho.mean().reset_index().sort_values("rho", ascending=False).drop_duplicates("draw")
    m = m.set_index("draw")
    d = pd.DataFrame({n: cfgs[n] for n in m.index}).T
    d["rho"] = m.rho.astype(float); d["best_k"] = m.k.astype(int)
    for c in d.columns:
        if c not in ("target",):
            d[c] = pd.to_numeric(d[c])
    return d


def analyze() -> None:
    """Print what the stage-1 draws say about each knob: marginal means, and a ridge regression of the score on the standardised settings."""
    from sklearn.linear_model import RidgeCV
    pd.set_option("display.width", 200)
    d = stage1_table()
    ref = d.loc["champion", "rho"]
    draws = d.drop(index=list(ANCHORS))
    print(f"{len(draws)} draws; champion {ref:.4f}, depth6 {d.loc['depth6', 'rho']:.4f}; best draw {draws.rho.max():.4f}; "
          f"median draw {draws.rho.median():.4f}; draws above champion: {(draws.rho > ref).sum()}")
    for c in ("max_depth", "nf", "target", "max_bin", "gamma", "reg_alpha"):
        print(f"\nby {c}:"); print(draws.groupby(c).rho.agg(["mean", "max", "count"]).round(4).to_string())
    for c in ("learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda"):
        q = pd.qcut(draws[c], 4, duplicates="drop")
        print(f"\nby {c} (quartiles):"); print(draws.groupby(q, observed=True).rho.agg(["mean", "max", "count"]).round(4).to_string())
    feats = ["max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda", "reg_alpha",
             "gamma", "nf", "best_k"]
    X = draws[feats].astype(float).copy()
    for c in ("learning_rate", "min_child_weight", "reg_lambda", "nf", "best_k"):
        X[c] = np.log(X[c])
    X["rankgauss"] = (draws.target == "rankgauss").astype(float); X["bin64"] = (draws.max_bin == 64).astype(float)
    Z = (X - X.mean()) / X.std()
    r = RidgeCV(alphas=np.logspace(-2, 2, 9)).fit(Z, draws.rho.astype(float))
    eff = pd.Series(r.coef_, index=Z.columns).sort_values(key=abs, ascending=False)
    print("\nridge effect of +1 sd of each hyperparameter on rho (others held fixed):"); print(eff.round(4).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", type=int, choices=sorted(PLANS), default=1,
                    help="1: the gene split of the main search (seed 2026); 2: the replication on a second split (seed 7)")
    ap.add_argument("--stage", type=int)
    ap.add_argument("--draws", type=int, default=None, help="stage 1: configurations to draw (default: 200 on split 1, 40 on split 2)")
    ap.add_argument("--top", type=int, default=None, help="stage 2: finalists kept (default: 15 on split 1, 3 on split 2)")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--analyze", action="store_true", help="print what the split-1 stage-1 draws say about each knob (needs no data)")
    ap.add_argument("--fullcv", action="store_true", help="split 1 only")
    ap.add_argument("--budget", action="store_true", help="split 1 only")
    ap.add_argument("--external", nargs="*", default=None, help="split 1 only: draw names to run on the eSpCas9 screen")
    a = ap.parse_args()
    if a.split != 1 and (a.fullcv or a.budget or a.external is not None or a.analyze):
        ap.error("--fullcv, --budget, --external and --analyze belong to split 1")
    if a.analyze:
        analyze()
        if a.stage is None and not (a.confirm or a.fullcv or a.budget or a.external is not None):
            return
    plan = PLANS[a.split]
    co = Cohort(plan)
    print(f"cohort {co.X.shape}; DEV {len(co.dev):,} guides / CONFIRM {len(co.conf):,} guides "
          f"({len(set(co.g[co.dev]))} / {len(set(co.g[co.conf]))} genes)", flush=True)
    if a.split != 1:
        print(f"{co.shared_with_split1:,} of the {len(set(co.g[co.conf])):,} CONFIRM genes were also CONFIRM in split 1", flush=True)
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
