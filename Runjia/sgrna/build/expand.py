"""Lift the 13,880-row cap: featurise every guide in the curated Guo screen.

`featurise.py` reproduces 6,169 of the published matrix's 6,232 columns from a
protospacer alone, verified at 100% on real guides. That means the training
set is no longer limited to the rows Noshay et al. happened to publish.

The nearest source is crisprHAL's curated release of the same Guo screen:
33,567 unique guides with read-count-filtered activity scores, of which
**24,950 are not in the published matrix**. This module featurises all of
them, the same way, and answers the question that decides whether chasing more
data is worth anyone's time:

    does the model keep improving as rows are added, or has it plateaued?

The answer comes from a learning curve, not from a bigger run: a single test
set is held out once, then models are trained on growing subsets of the rest
and scored on that same test set. Comparing R2 across differently-sized runs
would confound "more training data" with "different test distribution"; this
does not.

    python -m sgrna.expand --build      # featurise, ~2 min, cached
    python -m sgrna.expand --curve      # the learning curve
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from .. import config
from . import featurise
from . import genome

CACHE_X = config.INTERIM / "expanded_X.npy"
CACHE_META = config.INTERIM / "expanded_meta.json"
CURVE_PATH = config.RESULTS / "learning_curve.csv"

SPACER_LENGTH = 20


def load_crisprhal() -> pd.DataFrame:
    """Curated WT-SpCas9 guides: protospacer, activity score, 378 nt context."""
    frames = []
    for name in ("WT-SpCas9_training_data.csv", "WT-SpCas9_testing_data.csv"):
        rows = []
        for line in open(config.CRISPRHAL / name):
            seq, _, val = line.rpartition(",")
            rows.append((seq.strip(), float(val)))
        frames.append(pd.DataFrame(rows, columns=["context", "score"]))
    hal = pd.concat(frames, ignore_index=True)
    # The 378-mer is 189 nt upstream + 20 protospacer + 3 PAM + 166 downstream.
    hal["protospacer"] = hal["context"].str[189:209]
    hal["pam"] = hal["context"].str[209:212]
    return hal.drop_duplicates("protospacer").reset_index(drop=True)


def build(force: bool = False, verbose: bool = True):
    """Featurise every curated guide and cache the result."""
    if CACHE_X.exists() and CACHE_META.exists() and not force:
        meta = json.loads(CACHE_META.read_text())
        return np.load(CACHE_X, mmap_mode="r"), meta

    hal = load_crisprhal()
    if verbose:
        print(f"  {len(hal):,} unique curated guides")

    ok = hal["protospacer"].str.fullmatch(r"[ACGT]{20}")
    hal = hal[ok].reset_index(drop=True)
    if verbose and (~ok).sum():
        print(f"  dropped {(~ok).sum()} guides with ambiguous bases")

    X, names = featurise.matrix(hal["protospacer"], verbose=verbose)

    # Which of these are already in the published matrix? Useful for reporting
    # and for holding the overlap out as a sanity check.
    from ..io_utils import load_guide_index

    known = set(load_guide_index()["protospacer"].dropna())
    in_published = hal["protospacer"].isin(known).to_numpy()
    if verbose:
        print(f"  {in_published.sum():,} overlap the published matrix, "
              f"{(~in_published).sum():,} are new")

    np.save(CACHE_X, X.astype(np.float32))
    meta = {
        "protospacers": hal["protospacer"].tolist(),
        "y": hal["score"].tolist(),
        "feature_names": names,
        "in_published": in_published.tolist(),
        "shape": list(X.shape),
    }
    CACHE_META.write_text(json.dumps(meta))
    if verbose:
        print(f"  cached {CACHE_X.name} {X.shape}")
    return np.load(CACHE_X, mmap_mode="r"), meta


def learning_curve(sizes=(4000, 8000, 13880, 20000, 26000), seeds=(41, 42, 43),
                   test_fraction: float = 0.2, n_features: int | None = None,
                   verbose: bool = True) -> pd.DataFrame:
    """Train on growing subsets, score every one on the same held-out guides."""
    import xgboost as xgb
    from ..harness.run_ablation import (_impute, _safe_pearson, _safe_spearman,
                               champion_params, make_selector)
    from sklearn.metrics import r2_score

    X, meta = build(verbose=verbose)
    X = np.asarray(X)
    y = np.asarray(meta["y"], dtype=float)
    n_features = n_features or config.N_FEATURES

    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(y))
        n_test = int(len(y) * test_fraction)
        test, pool = order[:n_test], order[n_test:]

        for size in sizes:
            if size > len(pool):
                continue
            train = pool[:size]
            X_tr, X_te = _impute(X[train], X[test])
            y_tr, y_te = y[train], y[test]

            sel = make_selector(seed)
            sel.fit(X_tr, y_tr)
            top = np.argsort(sel.feature_importances_)[::-1][:n_features]

            model = xgb.XGBRegressor(**champion_params(seed))
            model.fit(X_tr[:, top], y_tr)
            pred = model.predict(X_te[:, top])

            rows.append(dict(seed=seed, train_size=size, test_size=len(test),
                             r2=r2_score(y_te, pred),
                             spearman=_safe_spearman(y_te, pred),
                             pearson=_safe_pearson(y_te, pred)))
            if verbose:
                print(f"    seed {seed}  n={size:6,}  R2 {rows[-1]['r2']:.4f}  "
                      f"rho {rows[-1]['spearman']:.4f}")

    curve = pd.DataFrame(rows)
    summary = (curve.groupby("train_size")[["r2", "spearman", "pearson"]]
               .agg(["mean", "std"]).round(4))
    summary.to_csv(CURVE_PATH)
    if verbose:
        print(f"\n{summary.to_string()}")
        print(f"\nWrote {CURVE_PATH}")
    return curve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args(argv)

    if args.build or not args.curve:
        build(force=args.force)
    if args.curve:
        learning_curve(sizes=tuple(args.sizes) if args.sizes
                       else (4000, 8000, 13880, 20000, 26000),
                       seeds=tuple(args.seeds) if args.seeds else (41, 42, 43))
    return 0


if __name__ == "__main__":
    sys.exit(main())
