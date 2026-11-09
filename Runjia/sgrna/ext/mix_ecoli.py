"""Mixing model families for the E. coli model: does an ensemble that is not all XGBoost beat the tuned XGBoost ensemble?

The tuned XGBoost models of `tune_ecoli.py` are close to one another (depth 4-5, slow learning, many trees), so averaging
five of them adds only +0.003 over the best one. Averaging helps in proportion to how differently the members err, so this
module adds model families with different inductive biases, each given its own modest random search on the same DEV genes and
folds, and is judged on the same held-out CONFIRM genes:

  xgb    the tuned finalists (from `tune_ecoli`)
  lgb    LightGBM (leaf-wise trees), 40 draws
  cat    CatBoost (symmetric, oblivious trees), 14 draws
  et     extra-trees (bagged, unboosted), 4 draws
  mlp    a small neural network on the selected columns, 4 draws
  ridge  linear model on all 6,517 columns (the additive baseline)

Selection of columns is the same fixed selector ranking as before (top `nf` of the cached ranking), except ridge, which sees all
columns. Blending is by equal weights on z-scored predictions and by non-negative least squares fitted on DEV out-of-fold
predictions only; the CONFIRM genes are used once, at the end.

    python -m sgrna.ext.mix_ecoli --search lgb --draws 40
    python -m sgrna.ext.mix_ecoli --finalize
    python -m sgrna.ext.mix_ecoli --blend
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr

from sgrna.run_ablation import _impute

from . import tune_ecoli as T
from .common import OUT

THREADS = 6
KG = {"lgb": (200, 400, 800, 1200, 1600, 2000, 3000), "cat": (200, 400, 800, 1200, 1600, 2000)}
def CFGF(kind):
    return OUT / f"mix_configs_{kind}.json"


def all_cfgs():
    out = {}
    for f in sorted(OUT.glob("mix_configs_*.json")):
        out.update(json.loads(f.read_text()))
    return out


# ------------------------------------------------------------------ search spaces
def sample(kind: str, i: int) -> dict:
    r = np.random.default_rng({"lgb": 20_000, "cat": 30_000, "et": 40_000, "mlp": 50_000}[kind] + i)
    if kind == "lgb":
        return dict(num_leaves=int(r.choice([7, 15, 31, 63])), learning_rate=float(np.exp(r.uniform(np.log(0.01), np.log(0.1)))),
                    min_child_samples=int(r.choice([5, 10, 20, 40, 80])), feature_fraction=float(r.uniform(0.3, 1.0)),
                    bagging_fraction=float(r.uniform(0.5, 1.0)), lambda_l2=float(np.exp(r.uniform(np.log(0.1), np.log(50)))),
                    lambda_l1=float(r.choice([0, 0.1, 1])), max_depth=int(r.choice([-1, 4, 6])),
                    nf=int(r.choice([300, 450, 600, 900])), n_estimators=3000)
    if kind == "cat":
        return dict(depth=int(r.choice([4, 5, 6, 7, 8])), learning_rate=float(np.exp(r.uniform(np.log(0.03), np.log(0.15)))),
                    l2_leaf_reg=float(np.exp(r.uniform(0, np.log(30)))), rsm=float(r.uniform(0.3, 1.0)),
                    subsample=float(r.uniform(0.5, 1.0)), border_count=int(r.choice([64, 254])),
                    nf=int(r.choice([300, 450, 600, 900])), n_estimators=2000)
    if kind == "et":
        return dict(max_features=float(r.uniform(0.2, 0.6)), min_samples_leaf=int(r.choice([1, 2, 4])),
                    nf=int(r.choice([300, 600, 900])), n_estimators=600)
    if kind == "mlp":
        return dict(hidden=list([(256, 128), (512, 256), (128, 64)][int(r.integers(0, 3))]),
                    dropout=float(r.uniform(0.1, 0.4)), weight_decay=float(np.exp(r.uniform(np.log(1e-5), np.log(1e-2)))),
                    lr=float(np.exp(r.uniform(np.log(3e-4), np.log(3e-3)))), nf=int(r.choice([450, 900])), epochs=60)
    raise ValueError(kind)


# ------------------------------------------------------------------ fitting
def fit_predict(kind, cfg, X_tr, y_tr, X_te, seed, ks=None):
    """Return {k: predictions}; k is a tree count for the boosted models and 0 for the rest."""
    if kind == "xgb":
        return T.fit_predict(cfg, X_tr, y_tr, X_te, seed, ks)
    if kind == "lgb":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=int(cfg["n_estimators"]), learning_rate=cfg["learning_rate"], num_leaves=int(cfg["num_leaves"]),
                              min_child_samples=int(cfg["min_child_samples"]), colsample_bytree=cfg["feature_fraction"],
                              subsample=cfg["bagging_fraction"], subsample_freq=1, reg_lambda=cfg["lambda_l2"],
                              reg_alpha=cfg["lambda_l1"], max_depth=int(cfg["max_depth"]), random_state=seed, n_jobs=THREADS,
                              verbose=-1).fit(X_tr, y_tr)
        return {k: m.predict(X_te, num_iteration=k) for k in (ks or (cfg["n_estimators"],)) if k <= cfg["n_estimators"]}
    if kind == "cat":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=int(cfg["n_estimators"]), learning_rate=cfg["learning_rate"], depth=int(cfg["depth"]),
                              l2_leaf_reg=cfg["l2_leaf_reg"], rsm=cfg["rsm"], bootstrap_type="Bernoulli", subsample=cfg["subsample"],
                              border_count=int(cfg["border_count"]), thread_count=THREADS, verbose=0, random_seed=seed).fit(X_tr, y_tr)
        return {k: m.predict(X_te, ntree_end=k) for k in (ks or (cfg["n_estimators"],)) if k <= cfg["n_estimators"]}
    if kind == "et":
        from sklearn.ensemble import ExtraTreesRegressor
        m = ExtraTreesRegressor(n_estimators=int(cfg["n_estimators"]), max_features=cfg["max_features"],
                                min_samples_leaf=int(cfg["min_samples_leaf"]), n_jobs=THREADS, random_state=seed).fit(X_tr, y_tr)
        return {0: m.predict(X_te)}
    if kind == "mlp":
        return {0: _mlp(cfg, X_tr, y_tr, X_te, seed)}
    if kind == "ridge":
        from sklearn.linear_model import RidgeCV
        mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
        m = RidgeCV(alphas=np.logspace(1, 5, 9)).fit((X_tr - mu) / sd, y_tr)
        return {0: m.predict((X_te - mu) / sd)}
    raise ValueError(kind)


def _mlp(cfg, X_tr, y_tr, X_te, seed):
    """Run in a separate process: torch next to xgboost / lightgbm in one process stalled silently (see mlp_worker.py)."""
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        np.savez(d / "in.npz", X_tr=X_tr, y_tr=y_tr, X_te=X_te)
        (d / "cfg.json").write_text(json.dumps(cfg))
        subprocess.run([sys.executable, "-m", "sgrna.ext.mlp_worker", str(d / "in.npz"), str(d / "cfg.json"), str(d / "out.npy"),
                        str(seed)], check=True, timeout=1800)
        return np.load(d / "out.npy")


def columns(cfg, top, X_tr_full, X_te_full, kind):
    if kind == "ridge":
        return _impute(X_tr_full, X_te_full)
    cols = top[:int(cfg["nf"])]
    return _impute(X_tr_full[:, cols], X_te_full[:, cols])


# ------------------------------------------------------------------ stage 1 search per family
def search(co, kind, draws, seed=41):
    cfgs = json.loads(CFGF(kind).read_text()) if CFGF(kind).exists() else {}
    resf = OUT / f"mix_{kind}_stage1.csv"
    res = pd.read_csv(resf) if resf.exists() else pd.DataFrame()
    done = set(res.draw.unique()) if len(res) else set()
    ks = KG.get(kind)
    t0 = time.time()
    for i in range(draws if kind != "ridge" else 1):
        name = f"{kind}{i:03d}"
        if name in done:
            continue
        cfg = sample(kind, i) if kind != "ridge" else dict(nf=0)
        cfgs[name] = cfg; CFGF(kind).write_text(json.dumps(cfgs, indent=1))
        rows = []
        for fold, (tr, va, top) in enumerate(T.rankings(co, seed)):
            A, B = columns(cfg, top, co.X[tr], co.X[va], kind)
            for k, p in fit_predict(kind, cfg, A, co.y[tr], B, seed, ks).items():
                rows.append(dict(seed=seed, fold=fold, k=k, rho=float(spearmanr(co.y[va], p).statistic)))
        df = pd.DataFrame(rows); df.insert(0, "draw", name)
        res = pd.concat([res, df], ignore_index=True); res.to_csv(resf, index=False)
        k, r = T.best_k(rows)
        print(f"  {name} {json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in cfg.items() if a != 'n_estimators'})} "
              f"best k {k} rho {r:.4f} ({time.time() - t0:.0f}s)", flush=True)


def best_of(kind, top=1):
    res = pd.read_csv(OUT / f"mix_{kind}_stage1.csv")
    m = res.groupby(["draw", "k"]).rho.mean().reset_index().sort_values("rho", ascending=False).drop_duplicates("draw")
    return [(r.draw, int(r.k), float(r.rho)) for r in m.head(top).itertuples()]


# ------------------------------------------------------------------ finalize: OOF on DEV, CONFIRM predictions
def finalists(top_per_kind=None):
    top_per_kind = top_per_kind or {"lgb": 2, "cat": 2, "et": 1, "mlp": 1, "ridge": 1}
    out = []
    for kind, n in top_per_kind.items():
        try:
            out += [(kind, *b) for b in best_of(kind, n)]
        except FileNotFoundError:
            pass
    sel = T.selected()
    xg = [n for n in sel.index if n not in ("champion", "depth6")][:5]
    out += [("xgb", n, int(sel.loc[n, "k"]), float(sel.loc[n, "rho"])) for n in xg]
    return out


def finalize(co, oof_seed=42, conf_seeds=(41, 42, 43)):
    cfgs = {**json.loads(T.OUT.joinpath("tune_configs.json").read_text()), **all_cfgs()}
    rows = []
    for kind, name, k, s1 in finalists():
        cfg = dict(cfgs[name]);
        if kind in ("xgb", "lgb", "cat"):
            cfg["n_estimators"] = k
        # DEV out-of-fold predictions on the seed-42 partition
        f = T.DIR / f"mixoof_{name}_s{oof_seed}.npy"
        if not f.exists():
            oof = np.full(len(co.y), np.nan)
            for fold, (tr, va, top) in enumerate(T.rankings(co, oof_seed)):
                A, B = columns(cfg, top, co.X[tr], co.X[va], kind)
                oof[va] = list(fit_predict(kind, cfg, A, co.y[tr], B, oof_seed, (k,) if kind in ("xgb", "lgb", "cat") else None).values())[0]
            np.save(f, oof)
        # CONFIRM predictions (trained on all DEV)
        for sd in conf_seeds if kind != "ridge" else conf_seeds[:1]:
            g = T.DIR / f"mixconf_{name}_k{k}_s{sd}.npy"
            if g.exists():
                continue
            top = np.load(T.DIR / f"rank_devall_s{sd}.npy")
            A, B = columns(cfg, top, co.X[co.dev], co.X[co.conf], kind)
            p = list(fit_predict(kind, cfg, A, co.y[co.dev], B, sd, (k,) if kind in ("xgb", "lgb", "cat") else None).values())[0]
            np.save(g, p)
        print(f"  finalized {kind} {name} k={k}", flush=True)
        rows.append(dict(kind=kind, name=name, k=k, stage1_rho=s1))
    pd.DataFrame(rows).to_csv(OUT / "mix_finalists.csv", index=False)


def _z(p):
    return (p - p.mean()) / (p.std() + 1e-12)


def blend(co, conf_seeds=(41, 42, 43), oof_seed=42, B=1000):
    fin = pd.read_csv(OUT / "mix_finalists.csv")
    y_dev, y_conf, g_conf = co.y[co.dev], co.y[co.conf], co.g[co.conf]
    P_conf, P_oof = {}, {}
    for r in fin.itertuples():
        seeds = conf_seeds if r.kind != "ridge" else conf_seeds[:1]
        P_conf[r.name] = np.array([np.load(T.DIR / f"mixconf_{r.name}_k{r.k}_s{sd}.npy") for sd in seeds])
        P_oof[r.name] = np.load(T.DIR / f"mixoof_{r.name}_s{oof_seed}.npy")[co.dev]
    names = list(fin.name)
    # champion for reference
    champ = np.array([np.load(T.DIR / f"conf_champion_k400_s{sd}.npy") for sd in conf_seeds])
    kind_of = dict(zip(fin.name, fin.kind))

    def avg(ns, w=None):
        w = np.ones(len(ns)) / len(ns) if w is None else np.asarray(w) / np.sum(w)
        out = []
        for i in range(len(conf_seeds)):
            out.append(sum(wi * _z(P_conf[n][min(i, len(P_conf[n]) - 1)]) for wi, n in zip(w, ns)))
        return np.array(out)

    xg = [n for n in names if kind_of[n] == "xgb"]
    arms = {"champion": champ, "xgb ensemble (5)": avg(xg)}
    for kind in ("lgb", "cat", "et", "mlp", "ridge"):
        ns = [n for n in names if kind_of[n] == kind]
        if ns:
            arms[f"{kind} (best)"] = avg(ns[:1])
            if len(ns) > 1:
                arms[f"{kind} ensemble ({len(ns)})"] = avg(ns)
    fam = {k: [n for n in names if kind_of[n] == k] for k in ("xgb", "lgb", "cat", "et", "mlp", "ridge")}
    fam = {k: v for k, v in fam.items() if v}
    # family-level equal-weight blends
    def fam_pred(k):
        return avg(fam[k])
    for combo in (("xgb", "lgb"), ("xgb", "cat"), ("xgb", "lgb", "cat"), ("xgb", "lgb", "cat", "et"),
                  ("xgb", "lgb", "cat", "et", "mlp"), tuple(fam)):
        if all(c in fam for c in combo):
            arms["equal blend: " + "+".join(combo)] = np.mean([fam_pred(c) for c in combo], axis=0)
    # NNLS weights from DEV out-of-fold predictions (family level)
    Z = np.column_stack([_z(np.mean([_z(P_oof[n]) for n in fam[k]], axis=0)) for k in fam])
    w, _ = nnls(Z, _z(y_dev))
    print("NNLS family weights (DEV out-of-fold):", dict(zip(fam, np.round(w / w.sum(), 3))))
    pd.DataFrame({"family": list(fam), "weight": w / w.sum()}).to_csv(OUT / "mix_nnls_weights.csv", index=False)
    arms["NNLS blend (weights from DEV OOF)"] = np.sum([wi * fam_pred(k) for wi, k in zip(w / w.sum(), fam)], axis=0)
    point, bs = T._boot_rho(y_conf, g_conf, arms, B=B)
    rows = []
    for a in arms:
        d = bs[a] - bs["champion"]; e = bs[a] - bs["xgb ensemble (5)"]
        rows.append(dict(arm=a, rho=point[a], rho_lo=np.percentile(bs[a], 2.5), rho_hi=np.percentile(bs[a], 97.5),
                         d_vs_champion=point[a] - point["champion"], dc_lo=np.percentile(d, 2.5), dc_hi=np.percentile(d, 97.5),
                         d_vs_xgb_ens=point[a] - point["xgb ensemble (5)"], dx_lo=np.percentile(e, 2.5), dx_hi=np.percentile(e, 97.5)))
    out = pd.DataFrame(rows); out.to_csv(OUT / "mix_confirm.csv", index=False)
    # diversity: correlation of DEV out-of-fold residuals between family bests
    res = {k: _z(P_oof[fam[k][0]]) - _z(y_dev) for k in fam}
    pd.DataFrame(res).corr().round(3).to_csv(OUT / "mix_residual_corr.csv")
    pd.DataFrame({k: _z(P_oof[fam[k][0]]) for k in fam}).corr(method="spearman").round(3).to_csv(OUT / "mix_pred_corr.csv")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", choices=["lgb", "cat", "et", "mlp", "ridge"])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--blend", action="store_true")
    a = ap.parse_args()
    co = T.Cohort()
    if a.search:
        search(co, a.search, a.draws)
    if a.finalize:
        finalize(co)
    if a.blend:
        pd.set_option("display.width", 250)
        print(blend(co).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
