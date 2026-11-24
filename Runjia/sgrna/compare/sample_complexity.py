"""crisprHAL 2 needs 1.7 to 2.5 times as many labelled guides as SLICER to reach the same accuracy, at every accuracy level that both models reach. The one architecture-free comparison.

Every cost measure in Part 3 depends on how the computation is organised --
FLOPs on the architecture, CPU-seconds on the hardware, bytes moved on the
access pattern. **Sample complexity does not.** It asks how much labelled data a
method needs to reach a given accuracy, which is a property of the method's
inductive bias and nothing else, and it is the comparison the report flags as
"the fair fight neither of us has run".

Protocol, identical for both models so the curves are comparable:

  * The test set is **fold 1** of `KFold(5, shuffle=True, random_state=41)` over
    `expanded_meta.json`'s order -- the same split every other head-to-head arm
    uses. It is fixed across every training size, so the curves differ only in
    what the model was trained on.
  * Training subsets are drawn from the remaining 26,853 guides, nested so that
    a larger subset contains the smaller ones. Nesting matters: without it, a
    dip in the curve could be a lucky or unlucky subset rather than a sample-size
    effect.
  * Epochs, settings and architecture are held at each model's published
    defaults. Tuning at every size would measure tuning, not sample complexity.

    python -m sgrna.compare.sample_complexity --model slicer
    .venv-crisprhal/bin/python src/sgrna/sample_complexity.py --model crisprhal
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

if __package__ in (None, ""):                 # run as a script from the TF venv
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from sgrna import config
else:
    from .. import config

OUT = config.RESULTS / "sample_complexity.csv"
SIZES = (4000, 8000, 13880, 20000, 26853)


def _fixed_split(n: int, seed: int = 41, fold: int = 1):
    """Fold `fold` of the shared 5-fold split: the same test sets as every other arm.

    Originally this returned fold 1 only, which made the curve a statement about
    one test set. A claim of the form "model A needs half the labelled guides"
    has to survive the choice of test set, so the fold is now a parameter and
    the summary aggregates over it.
    """
    splits = list(KFold(5, shuffle=True, random_state=seed).split(np.arange(n)))
    tr, va = splits[fold - 1]
    return tr, va


def _nested_subsets(train_idx: np.ndarray, sizes, seed: int):
    """Shuffle once, then take prefixes, so each subset contains the smaller ones."""
    order = np.random.default_rng(seed).permutation(train_idx)
    return {s: order[:s] for s in sizes if s <= len(order)}


def _append(row: dict):
    df = pd.DataFrame([row])
    if OUT.exists() and OUT.stat().st_size:
        prev = pd.read_csv(OUT)
        key = ["model", "seed", "fold", "n_train"]
        prev = prev[~(prev[key].astype(str)
                      .apply(tuple, axis=1)
                      .isin([tuple(str(row[k]) for k in key)]))]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(OUT, index=False)


def slicer(sizes=SIZES, seeds=(41, 42), folds=(1, 2, 3, 4, 5), verbose: bool = True):
    """SLICER's curve: selection and champion refitted at every training size."""
    import lightgbm as lgb

    from .headtohead import _assemble
    from ..harness.run_ablation import _impute, make_selector
    from ..tuning.tune import DEFAULT

    X, y, names = _assemble("curated_flank", verbose=verbose)
    done = set()
    if OUT.exists() and OUT.stat().st_size:
        p = pd.read_csv(OUT)
        done = set(zip(p["model"], p["seed"], p.get("fold", 1), p["n_train"]))

    for fold in folds:
      tr_all, va = _fixed_split(len(y), fold=fold)
      for seed in seeds:
        for n, idx in _nested_subsets(tr_all, sizes, seed).items():
            if ("SLICER", seed, fold, n) in done:
                continue
            t0 = time.time()
            X_tr, X_va = _impute(X[idx], X[va])
            sel = make_selector(seed)
            sel.fit(X_tr, y[idx])
            top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
            m = lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1,
                                  **DEFAULT)
            m.fit(X_tr[:, top], y[idx])
            pred = m.predict(X_va[:, top])
            row = dict(model="SLICER", seed=seed, fold=fold, n_train=int(n),
                       n_test=int(len(va)),
                       spearman=float(spearmanr(y[va], pred).statistic),
                       r2=float(r2_score(y[va], pred)),
                       fit_seconds=round(time.time() - t0, 1))
            _append(row)
            if verbose:
                print(f"  SLICER fold {fold} n={n:6d} seed {seed}: rho "
                      f"{row['spearman']:.4f}  ({row['fit_seconds']:.0f}s)",
                      flush=True)


def crisprhal(sizes=SIZES, seeds=(41,), folds=(1, 2, 3, 4, 5), verbose: bool = True):
    """crisprHAL 2's curve: their architecture, their defaults, our split."""
    import tensorflow as tf

    from sgrna.compare.crisprhal_rerun import MODEL, META_PATH, _load_hal, _read

    models, default_epochs, proc = _load_hal()
    meta = json.loads(META_PATH.read_text())
    protos = np.asarray(meta["protospacers"])
    y = np.asarray(meta["y"], dtype=float)
    hal = pd.concat([_read("train"), _read("test")]).drop_duplicates("protospacer")
    seqs = hal.set_index("protospacer")["seq"].reindex(protos).to_numpy()

    done = set()
    if OUT.exists() and OUT.stat().st_size:
        p = pd.read_csv(OUT)
        done = set(zip(p["model"], p["seed"], p.get("fold", 1), p["n_train"]))

    for fold in folds:
      tr_all, va = _fixed_split(len(y), fold=fold)
      X_va = proc.onehotencode(seqs[va])
      for seed in seeds:
        for n, idx in _nested_subsets(tr_all, sizes, seed).items():
            if ("crisprHAL 2", seed, fold, n) in done:
                continue
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            t0 = time.time()
            m = models(MODEL, summary=False)
            m.train(proc.onehotencode(seqs[idx]), y[idx],
                    epochs=default_epochs, batch_size=1024, verbose=0)
            pred = np.asarray(m.predict(X_va, verbose=0)).ravel()
            row = dict(model="crisprHAL 2", seed=seed, fold=fold, n_train=int(n),
                       n_test=int(len(va)),
                       spearman=float(spearmanr(y[va], pred).statistic),
                       r2=float(r2_score(y[va], pred)),
                       fit_seconds=round(time.time() - t0, 1))
            _append(row)
            if verbose:
                print(f"  crisprHAL fold {fold} n={n:6d} seed {seed}: rho "
                      f"{row['spearman']:.4f}  ({row['fit_seconds']/60:.1f} min)",
                      flush=True)


def summary():
    d = pd.read_csv(OUT)
    piv = d.pivot_table(index="n_train", columns="model", values="spearman")
    print("\n== Spearman against training size, fixed test set ==")
    print(piv.round(4).to_string())
    if piv.shape[1] == 2:
        # Absolute thresholds are what the report quotes and the only form that
        # compares the two models on one scale. "Share of each model's own best"
        # is printed second and labelled as diagnostic, because it is a
        # statement about each curve's own shape -- two models can both reach
        # 95% of themselves at the same n while being far apart in rho, and
        # quoting it as a data-efficiency claim invites exactly that confusion.
        print("\n== training guides needed to reach an ABSOLUTE rho (reportable) ==")
        for thr in (0.62, 0.66, 0.68):
            line = {}
            for col in piv.columns:
                hit = piv.index[piv[col] >= thr]
                line[col] = int(hit.min()) if len(hit) else None
            print(f"  rho >= {thr:.2f}: " +
                  "  ".join(f"{k} {v}" for k, v in line.items()))

        full = piv.max()
        print("\n== diagnostic only, NOT a data-efficiency claim: "
              "share of each model's own best ==")
        for frac in (0.90, 0.95, 0.99):
            line = {}
            for col in piv.columns:
                hit = piv.index[piv[col] >= frac * full[col]]
                line[col] = int(hit.min()) if len(hit) else None
            print(f"  {frac:.0%} of own best: " +
                  "  ".join(f"{k} {v}" for k, v in line.items()))
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=["slicer", "crisprhal", "none"],
                    default="none")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    a = ap.parse_args(argv)
    if a.model == "slicer":
        slicer(seeds=tuple(a.seeds) if a.seeds else (41, 42),
               folds=tuple(a.folds) if a.folds else (1, 2, 3, 4, 5))
    elif a.model == "crisprhal":
        crisprhal(seeds=tuple(a.seeds) if a.seeds else (41,),
                  folds=tuple(a.folds) if a.folds else (1, 2, 3, 4, 5))
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
