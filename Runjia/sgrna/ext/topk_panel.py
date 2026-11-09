"""Smoother top-end metrics against crisprHAL 2 on its own splits, and seed stability of the lead.

The 10-guide pick percentile of the original panel rests on ten guides and its interval spans ±10 percentile points; the question
"does the model pick better guides" is answered better by larger top sets. Here, with the raw predictions saved by `panel_vs_hal.py`:
precision of the top 1%, 5% and 10% of predictions against the true top, mean percentile of the 10 / 50 / 100 highest-scoring guides, the
calibration slope of y on the prediction, and paired bootstrap differences to crisprHAL 2. Seed stability: the tuned model refitted
with five seeds (selection and trees), rho lead over crisprHAL 2 per seed.

    python -m sgrna.ext.topk_panel
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import panel_vs_hal as P
from . import vs_crisprhal as V
from .common import OUT


def topk(y, p):
    n = len(y); order = np.argsort(p)[::-1]; pct = rankdata(y) / n * 100; trueorder = np.argsort(y)[::-1]
    out = {}
    for frac in (0.01, 0.05, 0.10):
        k = int(n * frac); out[f"top{int(frac * 100)}pct_precision"] = float(np.isin(order[:k], trueorder[:k]).mean())
    for k in (10, 50, 100):
        out[f"pick{k}_percentile"] = float(pct[order[:k]].mean())
    out["calibration_slope"] = float(np.polyfit(p, y, 1)[0])
    return out


def run(n_boot=2000, seed=0):
    rows = []
    rng = np.random.default_rng(seed)
    for split in ("espcas9", "wt"):
        z = np.load(P.DIR / f"{split}.npz"); y, hal = z["y"], z["hal"]
        raw = lambda ns: np.mean([z[n] for n in ns], axis=0)
        arms = {"original champion": z["champion"], "tuned XGBoost": z["d099x"],
                "mixed ensemble": np.mean([raw(["d099x", "d077x", "d021x", "d080x", "d000x"]), raw(["lgb011", "lgb018"]), raw(["cat015", "cat023"])], axis=0)}
        boots = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
        mh = topk(y, hal); bh = [topk(y[b], hal[b]) for b in boots]
        rows.append(dict(split=split, arm="crisprHAL 2", **mh))
        for lab, p in arms.items():
            m = topk(y, p); bp = [topk(y[b], p[b]) for b in boots]
            row = dict(split=split, arm=lab, **m)
            for k in mh:
                d = np.array([a[k] - h[k] for a, h in zip(bp, bh)])
                row[f"d_{k}"] = m[k] - mh[k]; row[f"lo_{k}"] = float(np.percentile(d, 2.5)); row[f"hi_{k}"] = float(np.percentile(d, 97.5))
            rows.append(row)
    out = pd.DataFrame(rows); out.to_csv(OUT / "topk_panel.csv", index=False)
    return out


def seeds(seeds_=(41, 42, 43, 44, 45)):
    """Tuned XGBoost refitted with five seeds on both of crisprHAL's splits."""
    import json
    from sgrna import config
    rows = []
    for split in ("espcas9", "wt"):
        z = np.load(P.DIR / f"{split}.npz"); y, hal = z["y"], z["hal"]
        specs = V.member_specs(["d099x"])
        # rebuild the training data of the split as in panel_vs_hal.collect, then refit per seed
        if split == "espcas9":
            from sgrna.final_eval import CACHE, _split_frames
            tr, te, _ = _split_frames()
            X = np.asarray(np.load(CACHE / "X.npy", mmap_mode="r"), np.float32)
            meta = json.loads((CACHE / "meta.json").read_text()); protos = pd.Index(meta["protospacers"]); y_all = np.asarray(meta["y"], float)
            i_tr, i_te = np.unique(protos.get_indexer(tr["protospacer"])), np.unique(protos.get_indexer(te["protospacer"]))
            i_tr = np.setdiff1d(i_tr, np.intersect1d(i_tr, i_te))
            Xtr, ytr, Xte, names_te = X[i_tr], y_all[i_tr], X[i_te], protos[i_te]
            hal_df = pd.read_csv(V.SRC / "preds_final_espcas9_crisprhal.csv").set_index("protospacer")
        else:
            from sgrna.data_ablation import load
            X, yy, names, protos, in_test, *_ = load(verbose=False)
            te, tr = np.where(in_test)[0], np.where(~in_test)[0]
            Xtr, ytr, Xte, names_te = X[tr], yy[tr], X[te], pd.Index(np.asarray(protos)[te])
            hal_df = pd.read_csv(V.SRC / "preds_final_wt_crisprhal.csv").set_index("protospacer")
        keep = np.where(names_te.isin(hal_df.index) & ~names_te.duplicated())[0]
        for sd in seeds_:
            pr, _ = V.train_predict(specs, Xtr, ytr, Xte, seed=sd)
            p = pr["d099x"][keep]
            from scipy.stats import spearmanr
            rows.append(dict(split=split, seed=sd, rho=float(spearmanr(y, p).statistic), crisprhal=float(spearmanr(y, hal).statistic)))
            print(f"  {split} seed {sd}: rho {rows[-1]['rho']:.4f}  lead {rows[-1]['rho'] - rows[-1]['crisprhal']:+.4f}", flush=True)
    out = pd.DataFrame(rows); out.to_csv(OUT / "seed_stability.csv", index=False)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    t = run()
    for split in ("espcas9", "wt"):
        d = t[t.split == split]
        print(split); print(d[["arm"] + [c for c in d.columns if c.startswith("d_")] ].round(4).to_string(index=False))
    seeds()
