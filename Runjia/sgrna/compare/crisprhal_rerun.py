"""Re-run crisprHAL 2 ourselves, on our rows and our folds.

`headtohead.py` compares our LightGBM against crisprHAL 2's *published*
Spearman (0.697). A published number is a weaker comparison than a model run
side by side, so this trains their WT-SpCas9 model -- their architecture,
their hyper-parameters, their 48 epochs at batch 1024, imported from their own
code in `external_data/crisprHAL` -- on exactly the rows and folds the
`curated_*` arms use: the 33,567 deduplicated curated guides,
`KFold(5, shuffle=True, random_state=seed)` over the order in
`expanded_meta.json`. Their input is their own 378 nt context
(189 up, 20 nt target, 3 nt PAM, 166 down), looked up by protospacer.

Two arms:

    their_split   their shipped train/test split, as a check that the code
                  reproduces their published hold-out (0.695) here
    our_folds     the like-for-like comparison against `curated_flank`

Needs TensorFlow 2.19 on Python 3.12 (their pinned versions), so it runs from
its own venv rather than the project interpreter:

    /opt/homebrew/bin/python3.12 -m venv .venv-crisprhal
    .venv-crisprhal/bin/pip install "tensorflow==2.19.*" scipy pandas scikit-learn
    .venv-crisprhal/bin/python src/sgrna/crisprhal_rerun.py --arm their_split
    .venv-crisprhal/bin/python src/sgrna/crisprhal_rerun.py --arm our_folds

Resumable: one row per (arm, seed, fold) in `results/crisprhal_rerun.csv`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT = pathlib.Path(__file__).resolve().parents[2]
HAL_DIR = ROOT / "external_data" / "crisprHAL"
META_PATH = ROOT / "data" / "interim" / "expanded_meta.json"
RESULTS_PATH = ROOT / "results" / "crisprhal_rerun.csv"

MODEL = "WT-SPCAS9"
UP = 189                                # protospacer starts here in their 378 nt


def _load_hal():
    """Their model class and encoder, imported from their repo unchanged."""
    sys.path.insert(0, str(HAL_DIR))
    cwd = os.getcwd()
    os.chdir(HAL_DIR)                   # their module paths are repo-relative
    try:
        from models import models, modelVersionDefaultEpochs
        from processing import processing
    finally:
        os.chdir(cwd)
    return models, modelVersionDefaultEpochs[MODEL], processing()


def _read(split: str) -> pd.DataFrame:
    d = pd.read_csv(HAL_DIR / "data" / f"WT-SpCas9_{split}ing_data.csv",
                    header=None, names=["seq", "y"])
    d["protospacer"] = d["seq"].str[UP:UP + 20]
    return d


def _arm_data(arm: str, seed: int, n_splits: int):
    """Yield (fold, seqs_tr, y_tr, seqs_va, y_va)."""
    if arm == "their_split":
        tr, te = _read("train"), _read("test")
        yield 1, tr["seq"].to_numpy(), tr["y"].to_numpy(), te["seq"].to_numpy(), te["y"].to_numpy()
        return

    meta = json.loads(META_PATH.read_text())
    protos = np.asarray(meta["protospacers"])
    y = np.asarray(meta["y"], dtype=float)
    hal = pd.concat([_read("train"), _read("test")]).drop_duplicates("protospacer")
    seqs = hal.set_index("protospacer")["seq"].reindex(protos)
    if seqs.isna().any():
        raise SystemExit(f"{seqs.isna().sum()} curated guides have no crisprHAL context")
    seqs = seqs.to_numpy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(kf.split(seqs), start=1):
        yield fold, seqs[tr], y[tr], seqs[va], y[va]


def run(arm: str, seed: int = 41, n_splits: int = 5, epochs: int | None = None,
        verbose: bool = True) -> pd.DataFrame:
    import tensorflow as tf

    models, default_epochs, proc = _load_hal()
    epochs = epochs or default_epochs

    prev = pd.read_csv(RESULTS_PATH) if RESULTS_PATH.exists() else pd.DataFrame()
    done = (set(zip(prev["arm"], prev["seed"], prev["fold"])) if len(prev) else set())

    rows = []
    for fold, s_tr, y_tr, s_va, y_va in _arm_data(arm, seed, n_splits):
        if (arm, seed, fold) in done:
            if verbose:
                print(f"  {arm} fold {fold}: already done")
            continue
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed + fold)
        X_tr, X_va = proc.onehotencode(s_tr), proc.onehotencode(s_va)
        m = models(MODEL, summary=False)
        t0 = time.time()
        m.train(X_tr, y_tr, epochs=epochs, batch_size=1024, verbose=2 if verbose else 0)
        pred = np.asarray(m.predict(X_va, verbose=0)).ravel()
        if arm != "their_split":
            pd.DataFrame({"y": y_va, "pred": pred, "fold": fold}).to_csv(
                RESULTS_PATH.parent / f"preds_crisprhal_default_fold{fold}.csv",
                index=False)
        row = dict(arm=arm, model="crisprHAL2_WT-SpCas9", seed=seed, fold=fold,
                   epochs=epochs, n_train=len(y_tr), n_val=len(y_va),
                   spearman=float(spearmanr(y_va, pred).statistic),
                   r2=float(r2_score(y_va, pred)),
                   fit_seconds=round(time.time() - t0, 1))
        rows.append(row)
        prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        prev.to_csv(RESULTS_PATH, index=False)
        if verbose:
            print(f"  {arm} fold {fold}  rho {row['spearman']:.4f}  "
                  f"({row['fit_seconds']:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arm", choices=["their_split", "our_folds"], required=True)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--epochs", type=int, default=None,
                   help="default: their model-specific value (48)")
    a = p.parse_args(argv)
    run(a.arm, seed=a.seed, epochs=a.epochs)
    d = pd.read_csv(RESULTS_PATH)
    print(d.groupby("arm")["spearman"].agg(["count", "mean", "std"]).round(4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
