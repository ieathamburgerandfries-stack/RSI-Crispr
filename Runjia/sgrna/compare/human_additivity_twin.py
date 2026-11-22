"""The human additivity claim and the knob sweep behind it, re-run with folds that keep repeated guides together.

What is at stake. Part 2 of the report carries a contribution-level biology claim -- *"the two kingdoms differ in
interaction order, and two unrelated instruments agree on it"* -- resting on two numbers:

  ridge / boosting   95.6% in E. coli against 63.7% in human (`results/human_additivity.csv`)
  max_depth 4 -> 6   +0.0063 in E. coli against +0.0585 in human (`results/human_baseline_knobs.csv`)

Both were measured on **row-wise random folds** of the human iRF matrix, and `human_twin_check.py` then found that 8,499
of its 17,421 rows (49%) are a repeat measurement of a guide that also appears elsewhere, twin labels agreeing at
Spearman 0.80. Under twin-grouped folds the configuration that scored 0.5076 scores 0.2515, i.e. *below* the harness
default's 0.3356.

The problem this module exists to settle is that the corroboration is not independent of the artefact. Every knob that
"helped" on human is a memorisation knob -- depth 4->6 (+0.0585), learning rate 0.05->0.3 (+0.0416), trees 400->1,000
(+0.0361), while turning subsampling *off* cost -0.0075 -- and memorising a sequence whose twin sits in the training fold
is exactly what deep, fast-learning trees can do. So the second instrument is the first artefact's signature rather than
a separate line of evidence, and the 63.7% ratio has the leaked 0.5076 in its denominator. If ridge memorises twins less
than boosting does (it has no mechanism to), the ratio should rise under grouped folds, possibly above the E. coli figure,
which would reverse the claim.

Design: both fold schemes in one run, same columns, same seeds, same code, so the only thing that differs is the folds and
the row-wise arms reproduce the published numbers as a positive control for the harness.

  row-wise       KFold(shuffle=True) on rows -- what produced the published figures
  twin-grouped   folds grouped on the decoded protospacer, so a guide's twin is never across the split

    python -m sgrna.compare.human_additivity_twin
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from sgrna.compare.human_baseline import load
from sgrna.harness.run_ablation import _impute, champion_params, grouped_folds, make_selector

from ..common import OUT
from .human_flank import decode

N_FEATURES = 400
# the knob sweep of `results/human_baseline_knobs.csv`, plus the configuration all three knobs produce
KNOBS = {
    "champion": {},
    "plus_depth": dict(max_depth=6),
    "plus_lr": dict(learning_rate=0.3),
    "plus_trees": dict(n_estimators=1000),
    "plus_nosubsample": dict(subsample=1.0, colsample_bytree=1.0),
    "boosting_defaults": dict(n_estimators=1000, max_depth=6, learning_rate=0.3, subsample=1.0, colsample_bytree=1.0),
}


def _folds(scheme, X, grp, n_splits, seed):
    if scheme == "row-wise":
        return list(KFold(n_splits, shuffle=True, random_state=seed).split(X))
    return list(grouped_folds(grp, n_splits, seed))


def run(seeds=(41, 42, 43), n_splits=5):
    import xgboost as xgb
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    X, y = load()
    dec = decode()
    yy = pd.to_numeric(dec.y, errors="coerce").to_numpy(float)
    ok = np.isfinite(yy)
    assert ok.sum() == len(y) and np.allclose(yy[ok], y), "decoded table is not aligned with the matrix rows"
    grp = pd.factorize(dec.protospacer[ok])[0]
    n_twin = int(pd.Series(grp).duplicated(keep=False).sum())
    print(f"{len(y):,} rows, {grp.max() + 1:,} distinct protospacers, {n_twin:,} rows ({100 * n_twin / len(y):.0f}%) "
          f"have a twin", flush=True)

    f_out = OUT / "human_additivity_twin.csv"
    rows = pd.read_csv(f_out).to_dict("records") if f_out.exists() else []
    done = {(r["folds"], r["arm"], r["seed"], r["fold"]) for r in rows}
    t0 = time.time()
    for scheme in ("twin-grouped", "row-wise"):          # the one that matters first, in case the night runs out
        for seed in seeds:
            for fold, (tr, va) in enumerate(_folds(scheme, X, grp, n_splits, seed)):
                if all((scheme, a, seed, fold) in done for a in list(KNOBS) + ["ridge"]):
                    continue
                leak = len(set(grp[tr]) & set(grp[va]))
                if scheme == "twin-grouped":
                    assert leak == 0, f"a protospacer appears on both sides of a twin-grouped fold ({leak})"
                X_tr, X_va = _impute(X[tr], X[va])
                sel = make_selector(seed)
                sel.fit(X_tr, y[tr])
                top = np.argsort(sel.feature_importances_)[::-1][:N_FEATURES]
                A, Bm = X_tr[:, top], X_va[:, top]
                for arm, change in KNOBS.items():
                    if (scheme, arm, seed, fold) in done:
                        continue
                    p = dict(champion_params(seed))
                    p.update(change)
                    m = xgb.XGBRegressor(**p).fit(A, y[tr])
                    rows.append(dict(folds=scheme, arm=arm, seed=seed, fold=fold, shared_protospacers=leak,
                                     rho=float(spearmanr(y[va], m.predict(Bm)).statistic)))
                if (scheme, "ridge", seed, fold) not in done:
                    sc = StandardScaler().fit(A)
                    r = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(A), y[tr])
                    rows.append(dict(folds=scheme, arm="ridge", seed=seed, fold=fold, shared_protospacers=leak,
                                     rho=float(spearmanr(y[va], r.predict(sc.transform(Bm))).statistic)))
                pd.DataFrame(rows).to_csv(f_out, index=False)
                print(f"  {scheme:13s} seed {seed} fold {fold}: "
                      + "  ".join(f"{r['arm']}={r['rho']:.4f}" for r in rows[-(len(KNOBS) + 1):])
                      + f"   ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def summarise(B=4000, seed=0):
    """Per arm and scheme: mean rho; then the two numbers the report's claim rests on, under each scheme."""
    d = pd.read_csv(OUT / "human_additivity_twin.csv")
    g = d.groupby(["folds", "arm"]).rho.agg(["mean", "std", "count"]).reset_index()
    g.to_csv(OUT / "human_additivity_twin_summary.csv", index=False)
    piv = d.pivot_table(index=["seed", "fold"], columns=["folds", "arm"], values="rho")
    rng = np.random.default_rng(seed)
    rows = []
    for scheme in [s for s in ("row-wise", "twin-grouped") if s in piv.columns.get_level_values(0)]:
        p = piv[scheme]
        best_boost = p[[a for a in KNOBS]].mean().idxmax()
        ratio = p["ridge"].mean() / p[best_boost].mean()
        dep = (p["plus_depth"] - p["champion"]).to_numpy(float)
        bs = [dep[rng.integers(0, len(dep), len(dep))].mean() for _ in range(B)]
        rows.append(dict(folds=scheme, ridge=p["ridge"].mean(), best_boosting_arm=best_boost,
                         best_boosting=p[best_boost].mean(), ridge_over_boosting=ratio,
                         depth_4_to_6=dep.mean(), depth_lo=np.percentile(bs, 2.5), depth_hi=np.percentile(bs, 97.5),
                         n_folds=len(p)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "human_additivity_twin_claim.csv", index=False)
    return g, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[41, 42, 43])
    ap.add_argument("--summarise-only", action="store_true")
    a = ap.parse_args()
    if not a.summarise_only:
        run(tuple(a.seeds))
    pd.set_option("display.width", 220)
    g, claim = summarise()
    print(g.round(4).to_string(index=False), flush=True)
    print(claim.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
