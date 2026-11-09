"""The tuned / mixed E. coli model against crisprHAL 2 again, under the three conditions of the original comparison.

  espcas9   crisprHAL's own shipped eSpCas9 split (47,591 train / 11,898 test), against their shipped weights
            (predictions stored by `final_eval.py`); paired bootstrap on guides
  wt        crisprHAL's own shipped SpCas9 split (26,830 train / 6,737 test), against their shipped WT weights
            (predictions made by `vs_crisprhal_wt_hal.py` in the TensorFlow environment)
  folds     our five curated folds (KFold 5, seed 41, 33,567 guides), against crisprHAL 2's stored out-of-fold predictions at
            its published configuration and at the best configuration found for it

Each SLICER arm trains on the training rows only, with column selection fitted inside them, exactly as in `final_eval.py`.
Arms: the frozen champion (reproduces the earlier figures), the tuned XGBoost (d099x), the XGBoost average of five, and the
mixed ensemble chosen on held-out genes in `mix_ecoli.py` (its members are read from `--members`).

    python -m sgrna.ext.vs_crisprhal --condition espcas9 --members d099x,d077x,...
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sgrna import config
from sgrna.run_ablation import _impute, make_selector

from . import mix_ecoli as M
from . import tune_ecoli as T
from .common import OUT

SRC = config.ROOT.parent / "results"          # the main project's results folder (stored crisprHAL predictions)


def member_specs(names):
    cfgs = {**json.loads((OUT / "tune_configs.json").read_text()), **M.all_cfgs()}
    fin = pd.read_csv(OUT / "mix_finalists.csv").set_index("name")
    sel = T.selected()
    out = []
    for n in names:
        if n in ("champion", "depth6"):
            out.append(("xgb", n, int(sel.loc[n, "k"]), cfgs[n]))
        elif n in fin.index:
            out.append((fin.loc[n, "kind"], n, int(fin.loc[n, "k"]), cfgs[n]))
        else:
            out.append(("xgb", n, int(sel.loc[n, "k"]), cfgs[n]))
    return out


def train_predict(specs, X_tr, y_tr, X_te, seed=41):
    """Selection on the training rows, then every member; returns ({name: prediction}, {name: fit seconds})."""
    A, _ = _impute(X_tr, X_te[:1])
    s = make_selector(seed); s.fit(A, y_tr)
    top = np.argsort(s.feature_importances_)[::-1][:900]
    preds, secs = {}, {}
    for kind, name, k, cfg in specs:
        cfg = dict(cfg)
        if kind in ("xgb", "lgb", "cat"):
            cfg["n_estimators"] = k
        t0 = time.time()
        Xa, Xb = M.columns(cfg, top, X_tr, X_te, kind)
        p = list(M.fit_predict(kind, cfg, Xa, y_tr, Xb, seed, (k,) if kind in ("xgb", "lgb", "cat") else None).values())[0]
        preds[name] = p; secs[name] = time.time() - t0
    return preds, secs


def _z(p):
    return (p - p.mean()) / (p.std() + 1e-12)


def _flat(ns):
    return [m for x in ns for m in (x if isinstance(x, list) else [x])]


def _blend(preds, ns):
    """A flat list is an equal-weight average of z-scored members; a list of lists averages each group first, then the groups."""
    if ns and isinstance(ns[0], list):
        return np.mean([_blend(preds, g) for g in ns], axis=0)
    return np.mean([_z(preds[n]) for n in ns], axis=0)


def arms_from(preds, families):
    """families: {arm label: members or groups of members} -> equal-weight z-averages (see `_blend`)."""
    return {lab: _blend(preds, ns) for lab, ns in families.items()}


def paired(y, ps, ph, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, len(y), len(y))
        d[i] = spearmanr(y[k], ps[k]).statistic - spearmanr(y[k], ph[k]).statistic
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(spearmanr(y, ps).statistic - spearmanr(y, ph).statistic), float(lo), float(hi)


# ------------------------------------------------------------------ the three conditions
def espcas9(families, specs):
    from sgrna.final_eval import CACHE, _split_frames
    tr, te, _ = _split_frames()
    X = np.asarray(np.load(CACHE / "X.npy", mmap_mode="r"), np.float32)
    meta = json.loads((CACHE / "meta.json").read_text())
    protos = pd.Index(meta["protospacers"]); y_all = np.asarray(meta["y"], float)
    i_tr, i_te = np.unique(protos.get_indexer(tr["protospacer"])), np.unique(protos.get_indexer(te["protospacer"]))
    i_tr = np.setdiff1d(i_tr, np.intersect1d(i_tr, i_te))
    preds, secs = train_predict(specs, X[i_tr], y_all[i_tr], X[i_te])
    hal = pd.read_csv(SRC / "preds_final_espcas9_crisprhal.csv").set_index("protospacer")
    order = pd.Index(protos[i_te]); both = hal.loc[hal.index.intersection(order)]
    pos = order.get_indexer(both.index); y = y_all[i_te][pos]
    assert np.allclose(y, both["y"].to_numpy(), atol=1e-9), "label alignment with crisprHAL's predictions failed"
    arms = arms_from({k: v[pos] for k, v in preds.items()}, families)
    return _rows("espcas9", y, arms, both["pred"].to_numpy(), "crisprHAL 2 (shipped ESPCAS9)", secs, len(i_tr), families)


def wt(families, specs):
    from sgrna.data_ablation import load
    X, y, names, protos, in_test, is_noshay, nos, n_base = load(verbose=False)
    te, tr = np.where(in_test)[0], np.where(~in_test)[0]
    preds, secs = train_predict(specs, X[tr], y[tr], X[te])
    hal = pd.read_csv(SRC / "preds_final_wt_crisprhal.csv").set_index("protospacer")
    order = pd.Index(np.asarray(protos)[te]); keep = order.isin(hal.index) & ~order.duplicated()
    pos = np.where(keep)[0]; ph = hal.loc[order[pos], "pred"].to_numpy(); yy = y[te][pos]
    arms = arms_from({k: v[pos] for k, v in preds.items()}, families)
    return _rows("wt", yy, arms, ph, "crisprHAL 2 (shipped WT-SPCAS9)", secs, len(tr), families)


def folds(families, specs, seed=41):
    from sklearn.model_selection import KFold
    from sgrna.curated_ablation import _base, _block
    X, y, _, protos = _base(); A, _ = _block("a_flank", protos); X = np.hstack([X, A])
    oof = {s[1]: np.zeros(len(y)) for s in specs}; secs_all = {s[1]: 0.0 for s in specs}
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(np.arange(len(y))):
        p, secs = train_predict(specs, X[tr], y[tr], X[va], seed)
        for k in p:
            oof[k][va] = p[k]; secs_all[k] += secs[k]
    # per-fold z-scoring would change the pooled ranking; blend inside each fold instead
    arms = {}
    for lab, ns in families.items():
        z = np.zeros(len(y))
        for tr, va in KFold(5, shuffle=True, random_state=seed).split(np.arange(len(y))):
            z[va] = _blend({k: v[va] for k, v in oof.items()}, ns)
        arms[lab] = z
    order = np.concatenate([va for _, va in KFold(5, shuffle=True, random_state=seed).split(np.arange(len(y)))])
    rows = []
    for kind in ("default", "tuned"):
        files = sorted(SRC.glob(f"preds_crisprhal_{kind}_fold*.csv"))
        h = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        assert np.allclose(y[order], h["y"].to_numpy(), atol=1e-9), "fold alignment failed"
        ph = np.empty(len(y)); ph[order] = h["pred"].to_numpy()
        rows += _rows(f"folds vs crisprHAL 2 ({kind} config)", y, arms, ph, f"crisprHAL 2 ({kind})", {k: v / 5 for k, v in secs_all.items()}, int(0.8 * len(y)), families)
    return rows


def _rows(cond, y, arms, ph, hal_label, secs, n_train, families=None):
    out = []
    rho_h = float(spearmanr(y, ph).statistic)
    for lab, p in arms.items():
        d, lo, hi = paired(y, p, ph)
        out.append(dict(condition=cond, arm=lab, n_test=len(y), n_train=n_train, rho=float(spearmanr(y, p).statistic),
                        crisprhal=rho_h, diff=d, lo=lo, hi=hi, crisprhal_model=hal_label,
                        fit_seconds=float(sum(secs[n] for n in _flat(families[lab]))) if families else float('nan')))
        print(f"  {cond:34s} {lab:28s} rho {out[-1]['rho']:.4f}  {hal_label} {rho_h:.4f}  diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["espcas9", "wt", "folds"], required=True)
    ap.add_argument("--mix", default="", help="groups of member names: g1a,g1b;g2a,g2b (each group averaged, then the groups)")
    a = ap.parse_args()
    mix = [[m for m in grp.split(",") if m] for grp in a.mix.split(";") if grp]
    xgb5 = ["d099x", "d077x", "d021x", "d080x", "d000x"]
    families = {"champion": ["champion"], "tuned XGBoost (d099x)": ["d099x"], "XGBoost average of 5": xgb5}
    if mix:
        families["mixed ensemble"] = mix
    needed = sorted({n for v in families.values() for n in _flat(v)})
    specs = member_specs(needed)
    rows = {"espcas9": espcas9, "wt": wt, "folds": folds}[a.condition](families, specs)
    f = OUT / f"vs_crisprhal_{a.condition}.csv"
    pd.DataFrame(rows).to_csv(f, index=False)


if __name__ == "__main__":
    main()
