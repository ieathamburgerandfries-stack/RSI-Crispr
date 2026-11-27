"""Picking the guide SLICER ranks first beats a random pick: among curated guides within 250 bp its measured score sits at the 0.77 percentile, against 0.5 for chance.

Every curated guide gets an out-of-fold prediction (the paper's five whole-gene folds, seed 41: the model never saw the guide's gene). The circular genome is cut into disjoint
windows of 250, 500 and 1,000 bp; in each window that holds at least four curated guides, the guide with the highest prediction is the model's pick. Its quality is the
measured score's percentile within the window (1 = the best guide of the window, 0 = the worst; a random pick averages 0.5), and whether it is the best one. Intervals
resample 100 kb blocks of the genome. Written to `results/design_benchmark.csv`.

The curated library holds about one guide per 138 bp, a subset of the NGG sites, so windows narrower than 250 bp rarely hold several guides to choose between.

    python -m sgrna.design.design_benchmark
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from .. import config
from ..common import INTERIM, OUT

CACHE = INTERIM / "design_oof.npz"
SEED = 41


def out_of_fold(threads: int = 8) -> dict[str, np.ndarray]:
    """Out-of-fold predictions of the tuned and default models on all 33,567 guides."""
    if CACHE.exists():
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    import xgboost as xgb

    from ..cohorts.curated_ablation import _base, _block
    from ..controls.sealed_holdout_gene import _fold_ids
    from ..harness.run_ablation import _impute, make_selector
    from ..interpret.interp_extra import paper_groups
    from ..tuning import tune_ecoli as T

    X, y, names, protos = _base()
    A, an = _block("a_flank", protos)
    XA = np.hstack([X, A])
    fold = _fold_ids(paper_groups(protos), 5, SEED)
    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    sel_table = T.selected()
    preds = {"tuned": np.zeros(len(y)), "default": np.zeros(len(y))}
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        Xtr, Xte = _impute(XA[tr], XA[te])
        sel = make_selector(SEED)
        sel.fit(Xtr, y[tr])
        rank = np.argsort(sel.feature_importances_)[::-1]
        for which, nm in (("tuned", "d099x"), ("default", "champion")):
            cfg = dict(cfgs[nm], n_estimators=int(sel_table.loc[nm, "k"]))
            cols = rank[:int(cfg["nf"])]
            t0 = time.time()
            m = xgb.XGBRegressor(**dict(T.xgb_params(cfg, SEED), n_jobs=threads)).fit(Xtr[:, cols], y[tr])
            preds[which][te] = m.predict(Xte[:, cols])
            print(f"  fold {k + 1} {which}: {time.time() - t0:.0f} s", flush=True)
    np.savez(CACHE, **preds)
    return preds


def main() -> int:
    from ..build.build_features import load_curated_index
    from ..cohorts.curated_ablation import _base
    from .design_guide import cut_after

    X, y, names, protos = _base()
    preds = out_of_fold()
    idx = load_curated_index(verbose=False).set_index(config.ID_COL).reindex(protos)
    cut = np.array([cut_after(int(l) - 1, s) for l, s in zip(idx["left"], idx["strand"])], dtype=np.int64)
    rows = []
    for width in (250, 500, 1000):
        win = cut // width
        for which in ("tuned", "default"):
            df = pd.DataFrame(dict(win=win, y=y, p=preds[which], block=cut // 100_000))
            res = []
            for w, g in df.groupby("win"):
                if len(g) < 4:
                    continue
                n = len(g)
                pick = g["p"].to_numpy().argmax()
                rank_best = (g["y"].to_numpy() > g["y"].to_numpy()[pick]).sum()          # guides measured better than the pick
                ties = (g["y"].to_numpy() == g["y"].to_numpy()[pick]).sum() - 1
                pct = 1 - (rank_best + 0.5 * ties) / (n - 1)
                res.append((g["block"].iloc[0], n, pct, float(rank_best == 0), 1 / n))
            r = pd.DataFrame(res, columns=["block", "n", "percentile", "best", "chance_best"])
            rng = np.random.default_rng(0)
            blocks = r["block"].unique()
            by = {b: r[r.block == b] for b in blocks}
            boots = []
            for _ in range(2000):
                s = pd.concat([by[b] for b in rng.choice(blocks, len(blocks))])
                boots.append((s.percentile.mean(), s.best.mean()))
            boots = np.array(boots)
            rows.append(dict(window_bp=width, model=which, windows=len(r), guides_per_window=float(r.n.mean()), pick_percentile=float(r.percentile.mean()),
                             pick_percentile_lo=float(np.percentile(boots[:, 0], 2.5)), pick_percentile_hi=float(np.percentile(boots[:, 0], 97.5)),
                             random_percentile=0.5, pick_is_best=float(r.best.mean()), pick_is_best_lo=float(np.percentile(boots[:, 1], 2.5)),
                             pick_is_best_hi=float(np.percentile(boots[:, 1], 97.5)), chance_is_best=float(r.chance_best.mean())))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "design_benchmark.csv", index=False)
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
