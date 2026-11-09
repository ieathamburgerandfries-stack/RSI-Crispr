"""The six-metric panel against crisprHAL 2 on its own two splits, for the tuned and mixed models, with raw predictions kept.

`vs_crisprhal.py` compared rho only and z-scored its blends, which is meaningless for R^2 and calibration. Here every member's
raw prediction on the test rows is saved (`data/interim/ext/panel/<split>.npz`), blends are averages of raw predictions, and the
metrics are those of `final_eval._metrics` (Spearman, Kendall tau, Pearson, R^2, top-decile precision, expected pick percentile of the
ten highest-scoring guides). Differences to crisprHAL 2 are paired bootstraps over guides.

    python -m sgrna.ext.panel_vs_hal --split espcas9 --mix "d099x,d077x,d021x,d080x,d000x;lgb011,lgb018;cat015,cat023"
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from sgrna import config

from . import vs_crisprhal as V
from .common import INTERIM, OUT

DIR = INTERIM / "panel"
DIR.mkdir(parents=True, exist_ok=True)
METRICS = ("spearman", "kendall_tau", "pearson", "r2", "top_decile_prec", "pick_percentile")


def metrics(y, p):
    from sklearn.metrics import r2_score
    top = int(len(y) * 0.1)
    order = np.argsort(p)[::-1]
    return dict(spearman=float(spearmanr(y, p).statistic), kendall_tau=float(kendalltau(y, p).statistic),
                pearson=float(np.corrcoef(y, p)[0, 1]), r2=float(r2_score(y, p)),
                top_decile_prec=float(np.isin(order[:top], np.argsort(y)[::-1][:top]).mean()),
                pick_percentile=float(np.mean([(y < y[i]).mean() for i in order[:10]]) * 100))


def collect(split, members):
    """Train the members on the split's training rows; save raw test predictions."""
    f = DIR / f"{split}.npz"
    specs = V.member_specs(members)
    if split == "espcas9":
        from sgrna.final_eval import CACHE, _split_frames
        tr, te, _ = _split_frames()
        X = np.asarray(np.load(CACHE / "X.npy", mmap_mode="r"), np.float32)
        meta = json.loads((CACHE / "meta.json").read_text())
        protos = pd.Index(meta["protospacers"]); y_all = np.asarray(meta["y"], float)
        i_tr, i_te = np.unique(protos.get_indexer(tr["protospacer"])), np.unique(protos.get_indexer(te["protospacer"]))
        i_tr = np.setdiff1d(i_tr, np.intersect1d(i_tr, i_te))
        Xtr, ytr, Xte, yte, names_te = X[i_tr], y_all[i_tr], X[i_te], y_all[i_te], protos[i_te]
        hal = pd.read_csv(V.SRC / "preds_final_espcas9_crisprhal.csv").set_index("protospacer")
    else:
        from sgrna.data_ablation import load
        X, y, names, protos, in_test, *_ = load(verbose=False)
        te, tr = np.where(in_test)[0], np.where(~in_test)[0]
        Xtr, ytr, Xte, yte, names_te = X[tr], y[tr], X[te], y[te], pd.Index(np.asarray(protos)[te])
        hal = pd.read_csv(V.SRC / "preds_final_wt_crisprhal.csv").set_index("protospacer")
    keep = np.where(names_te.isin(hal.index) & ~names_te.duplicated())[0]
    preds, secs = V.train_predict(specs, Xtr, ytr, Xte)
    np.savez(f, y=yte[keep], hal=hal.loc[names_te[keep], "pred"].to_numpy(), y_hal=hal.loc[names_te[keep], "y"].to_numpy(),
             **{k: v[keep] for k, v in preds.items()})
    return f


def panel(split, mix_groups, n_boot=2000, seed=0):
    z = np.load(DIR / f"{split}.npz")
    y, hal = z["y"], z["hal"]
    assert np.allclose(y, z["y_hal"], atol=1e-9), "label alignment failed"
    raw = lambda ns: np.mean([z[n] for n in ns], axis=0)
    arms = {"crisprHAL 2": hal, "original champion": z["champion"], "tuned XGBoost": z["d099x"],
            "average of 5 XGBoost": raw(["d099x", "d077x", "d021x", "d080x", "d000x"]),
            "mixed ensemble": np.mean([raw(g) for g in mix_groups], axis=0)}
    rows = []
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    mh = metrics(y, hal)
    bh = [metrics(y[b], hal[b]) for b in boots]
    for lab, p in arms.items():
        m = metrics(y, p)
        if lab == "crisprHAL 2":
            rows.append(dict(split=split, arm=lab, n=len(y), **m)); continue
        bp = [metrics(y[b], p[b]) for b in boots]
        row = dict(split=split, arm=lab, n=len(y), **m)
        for k in METRICS:
            d = np.array([a[k] - h[k] for a, h in zip(bp, bh)])
            row[f"d_{k}"] = m[k] - mh[k]; row[f"lo_{k}"] = float(np.percentile(d, 2.5)); row[f"hi_{k}"] = float(np.percentile(d, 97.5))
        rows.append(row)
        print(f"  {split} {lab:22s} " + " ".join(f"{k[:7]} {m[k]:.4f} ({m[k] - mh[k]:+.4f})" for k in METRICS), flush=True)
    out = pd.DataFrame(rows); out.to_csv(OUT / f"panel_{split}.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["espcas9", "wt"], required=True)
    ap.add_argument("--mix", default="d099x,d077x,d021x,d080x,d000x;lgb011,lgb018;cat015,cat023")
    ap.add_argument("--no-train", action="store_true")
    a = ap.parse_args()
    groups = [[m for m in g.split(",") if m] for g in a.mix.split(";")]
    members = sorted({"champion", "d099x", "d077x", "d021x", "d080x", "d000x"} | {m for g in groups for m in g})
    if not a.no_train:
        collect(a.split, members)
    pd.set_option("display.width", 250)
    panel(a.split, groups)


if __name__ == "__main__":
    main()
