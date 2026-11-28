"""The human boosting baseline (rho 0.5076 against the harness's 0.4036) is a memorisation artefact: with repeat measurements of a guide kept in one fold it falls to 0.2515, and the "human signal is less additive" claim built on it goes with it.

The question was whether a teammate's human baseline (`Jacky/human-boundary/HUMAN_BASELINE_SETTINGS.md`: 400 selected columns, 1,000 estimators, every other XGBoost knob at library defaults) reproduces on the
human iRF matrix (Noshay et al. 2023, Supplementary Table 3; 17,421 guides; `data/raw/human_feature_matrix.csv`), and what it would mean. Seven parts answer it in order, each writing one file under `results/`:

    arms           the three arms on random row folds                 human_baseline.csv         harness_default 300 columns and the champion; jacky_400_1000; jacky_cols_only (400 columns, champion)
    knobs          which of the three differences does the work       human_baseline_knobs.csv   one knob at a time on the 400 columns: trees, depth + learning rate, no subsampling
    additivity     ridge against boosting on the same columns         human_additivity.csv       the E. coli figure is 95.6%
    twin           the same two arms with twin-grouped folds          human_twin_check.csv       ONE FOLD DESIGN CHANGED; this is where 0.5076 becomes 0.2515
    twin-additivity  knobs and ridge under both fold designs          human_additivity_twin*.csv the claim re-tested, with the row-wise arms as a positive control
    kingdom        E. coli -> human and back on the columns the matrices share, at the champion and at the human-fitted settings   kingdom_transfer.csv
    kingdom-bootstrap  the same with the test rows resampled: does either direction exclude zero?                                   kingdom_transfer_boot.csv

Why the fold design matters. The matrix decodes to 13,138 distinct 20-mers: 4,216 sequences appear more than once (4,149 twice, 67 three times), as separate rows at the same genomic position, so 8,499 rows (49%) have a
twin, and twin labels agree at Spearman 0.80. With `KFold(shuffle=True)` on rows a test guide's twin is usually in the training folds, so a model that can memorise sequences (deep trees, fast learning rate, no
subsampling) scores well without generalising. Every knob that "helped" on human is a memorisation knob (depth 4 to 6 +0.0585, learning rate 0.05 to 0.3 +0.0416, trees 400 to 1,000 +0.0361; turning subsampling
off cost -0.0075), so the depth result that suggested human is "less additive" is the artefact's own signature, and the ridge/boosting ratio (63.7% in human against 95.6% in E. coli) had the leaked 0.5076 in its
denominator. `twin-additivity` runs both fold schemes in one pass with the same columns, seeds and code, so the only thing that differs is the folds; the row-wise arms reproduce the published numbers as the
positive control for the harness.

    python -m sgrna.compare.human_baseline --parts arms knobs additivity twin
    python -m sgrna.compare.human_baseline --parts twin-additivity [--summarise-only]
    python -m sgrna.compare.human_baseline --parts kingdom kingdom-bootstrap

The last two re-run Part 3's cross-kingdom boundary claim (transfer between E. coli and human is negative in both directions) with the human side fitted properly:

Part 3's boundary claim is that transfer between *E. coli* and human is negative
in both directions: rho -0.048 going to human and -0.017 coming back
(`results/human_transfer.csv`). That file was written by a script that is not in
`src/`, and both directions used the champion hyperparameters -- which
the `arms` part has since shown to be mis-specified on the human matrix by
0.10 rho, because the human signal is far less additive than *E. coli*'s
(ridge/boosting 63.7% against 95.6%).

That matters for exactly one direction. **human -> E. coli** trains on human, so
an undertuned human model makes the reported -0.017 pessimistic, and the report
should not rest a boundary claim on a model it knows is the wrong shape for its
training data. **E. coli -> human** trains on *E. coli*, where the champion is
appropriate, so it is a control here rather than a question.

So: both directions, at both settings, on the columns the two matrices share by
name -- no change of representation, which is what makes the comparison a
statement about kingdoms rather than about feature engineering.

`arms`, `knobs` and `additivity` rewrite their file from scratch; `twin-additivity` resumes from `human_additivity_twin.csv` (one row per arm and fold) and then writes the summary and the claim (ridge over
boosting, and the depth effect with its bootstrap interval) under each scheme.

Report section: Part 2.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from .. import config
from ..common import OUT
from ..harness import folds as fh
from ..harness.run_ablation import _impute, grouped_folds
from .human_shape import decode

MATRIX = config.DATA / "raw" / "human_feature_matrix.csv"
TARGET = "cut.score"
SEEDS = (41, 42, 43)

# what each arm changes in the champion (an empty change is the champion itself); the order is the order the files list them in
CHANGES = {
    "champion": {},
    "plus_trees": dict(n_estimators=1000),
    "plus_shape": dict(max_depth=6, learning_rate=0.3),
    "plus_nosubsample": dict(subsample=1.0, colsample_bytree=1.0),
    "plus_depth": dict(max_depth=6),                     # `plus_shape` moves depth and learning rate together; these two separate them
    "plus_lr": dict(learning_rate=0.3),
    "boosting_defaults": dict(n_estimators=1000, max_depth=6, learning_rate=0.3, subsample=1.0, colsample_bytree=1.0),
}
# the teammate's arms: (selected columns, what changes). `jacky_400_1000` is the library's defaults on 400 columns.
ARMS = {
    "harness_default": (config.N_FEATURES, {}),
    "jacky_400_1000": (400, dict(n_estimators=1000, max_depth=6, learning_rate=0.3, subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, min_child_weight=1)),
    "jacky_cols_only": (400, {}),
}


def load(verbose: bool = True):
    """The matrix with a measured target: (X, y), float32 and float."""
    d = pd.read_csv(MATRIX, skiprows=1, low_memory=False)
    y = pd.to_numeric(d[TARGET], errors="coerce").to_numpy(float)
    X = d.drop(columns=[c for c in (config.ID_COL, "sgRNAID", TARGET) if c in d.columns])
    X = X.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    ok = np.isfinite(y)
    if verbose:
        print(f"  human matrix: {ok.sum():,} guides x {X.shape[1]:,} columns")
    return X[ok], y[ok]


def twin_groups(y) -> np.ndarray:
    """One id per distinct protospacer (decoded from the electron-count columns), so repeat measurements of a guide can be kept together."""
    dec = decode()
    yy = pd.to_numeric(dec.y, errors="coerce").to_numpy(float)
    ok = np.isfinite(yy)
    assert ok.sum() == len(y) and np.allclose(yy[ok], y), "decoded table is not aligned with the matrix rows"
    return pd.factorize(dec.protospacer[ok])[0]


def _splits(scheme, X, groups, seed, n_splits=5):
    """'row-wise' is KFold(shuffle) on rows (what produced the published figures); 'twin-grouped' never lets a protospacer be on both sides."""
    if scheme == "row-wise":
        return list(KFold(n_splits, shuffle=True, random_state=seed).split(X))
    return list(grouped_folds(groups, n_splits, seed))


def _rho(y, pred) -> float:
    return float(spearmanr(y, pred).statistic)


def _fold(X, y, tr, va, seed, arms, ridge=None) -> dict:
    """Held-out rho of every arm on one fold. The columns are ranked once and each arm takes its own top n (a shorter list is a prefix of a longer one), so arms share the selection that costs the most.
    `arms` maps a name to (number of columns, change to the champion's settings); `ridge` adds a ridge fitted on that many top columns."""
    X_tr, X_va = _impute(X[tr], X[va])
    order = fh.rank_columns(X_tr, y[tr], seed, n_jobs=None)
    out = {}
    for arm, (n, change) in arms.items():
        top = order[:n]
        out[arm] = _rho(y[va], fh.boost(X_tr[:, top], y[tr], X_va[:, top], seed, change, n_jobs=None))
    if ridge:
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        top = order[:ridge]
        sc = StandardScaler().fit(X_tr[:, top])
        r = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(X_tr[:, top]), y[tr])
        out["ridge"] = _rho(y[va], r.predict(sc.transform(X_va[:, top])))
    return out


def _cv(X, y, scheme, arms, groups=None, ridge=None, seeds=SEEDS, n_splits=5) -> dict:
    """Arm -> list of held-out rhos over seeds x folds."""
    rhos: dict[str, list] = {}
    for seed in seeds:
        for tr, va in _splits(scheme, X, groups, seed, n_splits):
            for arm, r in _fold(X, y, tr, va, seed, arms, ridge).items():
                rhos.setdefault(arm, []).append(r)
    return rhos


def _row(arm, rhos, **extra) -> dict:
    return dict(arm=arm, **extra, spearman=float(np.mean(rhos)), spearman_sd=float(np.std(rhos, ddof=1)), n_fits=len(rhos))


def _merge(rows, path) -> None:
    """Add these arms to the file without dropping arms already measured."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["arm"].isin(new["arm"])], new], ignore_index=True)
    new.to_csv(path, index=False)


def arms(verbose=True):
    """The teammate's configuration against the harness's, on random row folds."""
    X, y = load(verbose)
    rhos = _cv(X, y, "row-wise", ARMS)
    out = pd.DataFrame([_row(a, rhos[a], n_features=ARMS[a][0]) for a in ARMS])
    out.to_csv(OUT / "human_baseline.csv", index=False)
    if verbose:
        print(out.round(4).to_string(index=False), "\n  the figure to reproduce is 0.472; our harness reports 0.404")
    return out


def knobs(only=None, verbose=True):
    """`jacky_400_1000` differs from the champion in three ways at once: capacity (1,000 trees, which the teammate's note reads as a statement about human data), model shape (depth 6, learning rate 0.3,
    a statement about our settings), and no subsampling. Each arm changes one of them on the 400 columns of `jacky_cols_only`, the reference (0.4074; all three together is 0.5076)."""
    X, y = load(verbose)
    names = [n for n in ("plus_trees", "plus_shape", "plus_nosubsample", "plus_depth", "plus_lr") if not only or n in only]
    rhos = _cv(X, y, "row-wise", {n: (400, CHANGES[n]) for n in names})
    rows = [_row(n, rhos[n], n_features=400, changed=";".join(CHANGES[n])) for n in names]
    _merge(rows, OUT / "human_baseline_knobs.csv")
    if verbose:
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    return pd.DataFrame(rows)


def additivity(verbose=True):
    """`plus_depth` is the largest single knob, and depth is the interaction-order knob, but it also buys plain capacity. The direct test is the one Appendix B ran on E. coli: ridge on the same
    selected columns, and how close a straight line gets to boosting (95.6% on 427 columns there)."""
    X, y = load(verbose)
    rhos = _cv(X, y, "row-wise", {"boosting_champion": (400, {}), "boosting_defaults": (400, CHANGES["boosting_defaults"])}, ridge=400)
    out = pd.DataFrame([_row(a, rhos[a], n_features=400) for a in ("boosting_defaults", "boosting_champion", "ridge")])
    out.to_csv(OUT / "human_additivity.csv", index=False)
    if verbose:
        best = out[out.arm.str.startswith("boost")].spearman.max()
        print(out.round(4).to_string(index=False), f"\n  ridge reaches {100 * out.spearman[out.arm == 'ridge'].iloc[0] / best:.1f}% of boosting (E. coli: 95.6% on 427 columns)")
    return out


def twin(verbose=True):
    """The first two arms of `arms` with folds that keep identical protospacers together."""
    X, y = load(verbose)
    groups = twin_groups(y)
    if verbose:
        print(f"{len(y):,} guides, {groups.max() + 1:,} distinct protospacers")
    use = {a: ARMS[a] for a in ("harness_default", "jacky_400_1000")}
    rhos = _cv(X, y, "twin-grouped", use, groups=groups)
    out = pd.DataFrame([dict(arm=a, folds="twin-grouped", spearman=float(np.mean(r)), spearman_sd=float(np.std(r, ddof=1)), n_fits=len(r)) for a, r in rhos.items()])
    out.to_csv(OUT / "human_twin_check.csv", index=False)
    if verbose:
        print(out.round(4).to_string(index=False))
    return out


def twin_additivity(seeds=SEEDS, n_splits=5, summarise_only=False):
    """Knob sweep and ridge, per arm and fold, under both fold schemes (resumable); then the summary."""
    f_out = OUT / "human_additivity_twin.csv"
    if not summarise_only:
        X, y = load(False)
        groups = twin_groups(y)
        print(f"{len(y):,} rows, {groups.max() + 1:,} distinct protospacers, {int(pd.Series(groups).duplicated(keep=False).sum()):,} rows have a twin", flush=True)
        names = ["champion", "plus_depth", "plus_lr", "plus_trees", "plus_nosubsample", "boosting_defaults"]
        use = {n: (400, CHANGES[n]) for n in names}
        rows = pd.read_csv(f_out).to_dict("records") if f_out.exists() else []
        done = {(r["folds"], r["arm"], r["seed"], r["fold"]) for r in rows}
        t0 = time.time()
        for scheme in ("twin-grouped", "row-wise"):          # the one that matters first, in case the night runs out
            for seed in seeds:
                for fold, (tr, va) in enumerate(_splits(scheme, X, groups, seed, n_splits)):
                    if all((scheme, a, seed, fold) in done for a in names + ["ridge"]):
                        continue
                    leak = len(set(groups[tr]) & set(groups[va]))
                    if scheme == "twin-grouped":
                        assert leak == 0, f"a protospacer appears on both sides of a twin-grouped fold ({leak})"
                    scores = _fold(X, y, tr, va, seed, {a: u for a, u in use.items() if (scheme, a, seed, fold) not in done}, ridge=400 if (scheme, "ridge", seed, fold) not in done else None)
                    new = [dict(folds=scheme, arm=a, seed=seed, fold=fold, shared_protospacers=leak, rho=r) for a, r in scores.items()]
                    rows += new
                    pd.DataFrame(rows).to_csv(f_out, index=False)
                    print(f"  {scheme:13s} seed {seed} fold {fold}: " + "  ".join(f"{r['arm']}={r['rho']:.4f}" for r in new) + f"   ({time.time() - t0:.0f}s)", flush=True)
    return _summarise(f_out)


def _summarise(f_out, B=4000, seed=0):
    """Per arm and scheme the mean rho; then the two numbers the claim rests on under each scheme: ridge over the best boosting arm, and depth 4 to 6 with a bootstrap interval over the folds."""
    d = pd.read_csv(f_out)
    g = d.groupby(["folds", "arm"]).rho.agg(["mean", "std", "count"]).reset_index()
    g.to_csv(OUT / "human_additivity_twin_summary.csv", index=False)
    piv = d.pivot_table(index=["seed", "fold"], columns=["folds", "arm"], values="rho")
    rng = np.random.default_rng(seed)
    rows = []
    boosting = ["champion", "plus_depth", "plus_lr", "plus_trees", "plus_nosubsample", "boosting_defaults"]
    for scheme in [s for s in ("row-wise", "twin-grouped") if s in piv.columns.get_level_values(0)]:
        p = piv[scheme]
        best = p[boosting].mean().idxmax()
        dep = (p["plus_depth"] - p["champion"]).to_numpy(float)
        bs = [dep[rng.integers(0, len(dep), len(dep))].mean() for _ in range(B)]
        rows.append(dict(folds=scheme, ridge=p["ridge"].mean(), best_boosting_arm=best, best_boosting=p[best].mean(), ridge_over_boosting=p["ridge"].mean() / p[best].mean(),
                         depth_4_to_6=dep.mean(), depth_lo=np.percentile(bs, 2.5), depth_hi=np.percentile(bs, 97.5), n_folds=len(p)))
    claim = pd.DataFrame(rows)
    claim.to_csv(OUT / "human_additivity_twin_claim.csv", index=False)
    pd.set_option("display.width", 220)
    print(g.round(4).to_string(index=False), flush=True)
    print(claim.round(4).to_string(index=False), flush=True)
    return g, claim


# ======================================================================================================  kingdom, kingdom-bootstrap
KINGDOM_OUT = OUT / "kingdom_transfer.csv"
KINGDOM_BOOT_OUT = OUT / "kingdom_transfer_boot.csv"
SETTINGS = {"champion": {}, "human_fitted": CHANGES["boosting_defaults"]}      # the settings `jacky_400_1000` found, which reach 0.5076 within human against the champion's 0.4074


def _kingdom_matrix(path, skiprows):
    d = pd.read_csv(path, skiprows=skiprows, low_memory=False)
    y = pd.to_numeric(d[config.TARGET_COL], errors="coerce")
    drop = [c for c in (config.ID_COL, "sgRNAID", config.TARGET_COL) if c in d]
    X = d.drop(columns=drop).apply(pd.to_numeric, errors="coerce")
    ok = np.isfinite(y.to_numpy(float))
    return X.loc[ok], y.to_numpy(float)[ok]


def kingdom_load(verbose: bool = True):
    """E. coli's published matrix and the human matrix restricted to the columns they share by name: (X_ecoli, y_ecoli, X_human, y_human, names)."""
    Xe, ye = _kingdom_matrix(config.BASE_MATRIX, config.BASE_MATRIX_HEADER_ROW)
    Xh, yh = _kingdom_matrix(MATRIX, 1)
    shared = [c for c in Xe.columns if c in set(Xh.columns)]
    if verbose:
        print(f"  E. coli {len(ye):,} x {Xe.shape[1]:,}  |  human {len(yh):,} x {Xh.shape[1]:,}  |  shared by name {len(shared):,}")
    return Xe[shared].to_numpy(np.float32), ye, Xh[shared].to_numpy(np.float32), yh, shared


def _transfer_predict(X_tr0, y_tr, X_te0, seed, change):
    """Train on one matrix, score the other: impute, rank the columns on the training rows, keep the champion's budget, fit with `change` on top of the champion's settings."""
    X_tr, X_te = _impute(X_tr0, X_te0)
    top = fh.rank_columns(X_tr, y_tr, seed, n_jobs=None)[:config.N_FEATURES]
    return fh.boost(X_tr[:, top], y_tr, X_te[:, top], seed, change, n_jobs=None)


def kingdom(seeds=SEEDS, verbose: bool = True):
    Xe, ye, Xh, yh, shared = kingdom_load(verbose=verbose)
    directions = {"ecoli -> human": (Xe, ye, Xh, yh), "human -> ecoli": (Xh, yh, Xe, ye)}
    rows = []
    for name, (X_tr0, y_tr, X_te0, y_te) in directions.items():
        for setting, change in SETTINGS.items():
            rhos = [_rho(y_te, _transfer_predict(X_tr0, y_tr, X_te0, seed, change)) for seed in seeds]
            rows.append(dict(direction=name, settings=setting, n_shared=len(shared), n_train=len(y_tr), n_test=len(y_te), spearman=float(np.mean(rhos)),
                             spearman_sd=float(np.std(rhos, ddof=1)), n_fits=len(rhos)))
            if verbose:
                print(f"  {name:>15} / {setting:>12}: rho {rows[-1]['spearman']:+.4f} ± {rows[-1]['spearman_sd']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(KINGDOM_OUT, index=False)
    if verbose:
        print("  the figures in the report are -0.048 and -0.017, both at champion")
        print(f"written -> {KINGDOM_OUT}")
    return pd.DataFrame(rows)


def kingdom_bootstrap(seeds=SEEDS, n_boot=2000, verbose: bool = True, settings=("champion", "human_fitted")):
    """Does either direction actually exclude zero? Three seeds of one train/test split say nothing about how much of the measured rho is the particular test set, and with rho this close to zero that
    is the whole question (-0.006 on 13,880 rows and -0.037 on 17,421 are not the same claim, though both are negative). So resample the test rows."""
    Xe, ye, Xh, yh, shared = kingdom_load(verbose=verbose)
    directions = {"ecoli -> human": (Xe, ye, Xh, yh), "human -> ecoli": (Xh, yh, Xe, ye)}
    rows = []
    for name, (X_tr0, y_tr, X_te0, y_te) in directions.items():
        for setting in settings:
            yhat = np.mean([_transfer_predict(X_tr0, y_tr, X_te0, seed, SETTINGS[setting]) for seed in seeds], axis=0)
            rng = np.random.default_rng(41)
            n = len(y_te)
            draws = np.empty(n_boot)
            for b in range(n_boot):
                i = rng.integers(0, n, n)
                draws[b] = spearmanr(y_te[i], yhat[i]).statistic
            lo, hi = np.percentile(draws, [2.5, 97.5])
            rows.append(dict(direction=name, settings=setting, n_test=n, spearman=float(spearmanr(y_te, yhat).statistic), ci_lo=float(lo), ci_hi=float(hi), n_boot=n_boot,
                             excludes_zero=bool(lo < 0 and hi < 0)))
            if verbose:
                r = rows[-1]
                print(f"  {name:>15}: rho {r['spearman']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  {'excludes zero' if r['excludes_zero'] else 'INCLUDES ZERO'}", flush=True)
            pd.DataFrame(rows).to_csv(KINGDOM_BOOT_OUT, index=False)
    if verbose:
        print(f"written -> {KINGDOM_BOOT_OUT}")
    return pd.DataFrame(rows)


PARTS = ("arms", "knobs", "additivity", "twin", "twin-additivity", "kingdom", "kingdom-bootstrap")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", nargs="+", choices=PARTS, required=True)
    ap.add_argument("--knobs-only", nargs="*", help="restrict `knobs` to these arms")
    ap.add_argument("--summarise-only", action="store_true", help="twin-additivity: write the summary and the claim from the stored rows, fit nothing")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS), help="twin-additivity only")
    a = ap.parse_args(argv)
    for part in a.parts:
        if part == "arms":
            arms()
        elif part == "knobs":
            knobs(only=a.knobs_only)
        elif part == "additivity":
            additivity()
        elif part == "twin":
            twin()
        elif part == "kingdom":
            kingdom()
        elif part == "kingdom-bootstrap":
            kingdom_bootstrap()
        else:
            twin_additivity(tuple(a.seeds), summarise_only=a.summarise_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
