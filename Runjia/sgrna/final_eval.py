"""The designated final evaluation: both models on crisprHAL's eSpCas9 split.

Part 4 reserved the eSpCas9 screen as "the designated untouched final
evaluation". **It is not untouched**, and the record has to say so: four feature
arms were already run on it across three seeds and two CV schemes
(`results/transfer.csv`), and a flank gain of +0.104 under `arc5` is quoted in
`RESULTS.md`. Nothing was *selected* on eSpCas9 -- every configuration was
chosen on the published or curated cohort and then confirmed there -- so the
honest description is a confirmatory dataset, not a holdout, and "evaluated once"
was never available.

What **is** untouched is the comparison this module runs: SLICER against
crisprHAL 2 on **crisprHAL's own shipped eSpCas9 train/test partition**, with
their own published eSpCas9 weights. No split, model, feature set or
hyperparameter here was chosen with reference to eSpCas9.

Two arms, one test set (their 11,898-guide eSpCas9 test file):

    slicer      champion XGBoost, 300 columns selected inside their training
                split only, base (6,169 regenerated) + family A (348) = 6,517
    crisprhal   their shipped `models/ESPCAS9.keras`, loaded and evaluated as
                distributed -- no retraining, so this is their published model
                and not our reconstruction of it

Frozen means frozen: the SLICER arm takes `champion_params`, `make_selector` and
`N_FEATURES` as they stand, and the selector is fitted on their training rows so
no test row touches feature selection.

    python -m sgrna.final_eval --arm slicer
    .venv-crisprhal/bin/python src/sgrna/final_eval.py --arm crisprhal
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
from scipy.stats import kendalltau, spearmanr

if __package__ in (None, ""):                 # run as a script from the TF venv
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from sgrna import config
else:
    from . import config

OUT = config.RESULTS / "final_eval_espcas9.csv"
PRED = config.RESULTS / "preds_final_espcas9_{arm}.csv"
HAL_DIR = config.EXTERNAL / "crisprHAL"
HAL_DATA = HAL_DIR / "data"
MODEL = "ESPCAS9"
PROTO_OFFSET = 193                            # their 406 nt context: 193 up
CACHE = config.INTERIM / "transfer_eSpCas9"


def _split_frames():
    """Their two shipped eSpCas9 files, kept apart, with protospacers attached."""
    out = {}
    for split in ("training", "testing"):
        d = pd.read_csv(HAL_DATA / f"{MODEL.replace('ESPCAS9', 'eSpCas9')}_"
                        f"{split}_data.csv", header=None,
                        names=["context", "score"]).dropna(subset=["context"])
        d["context"] = d["context"].str.upper()
        d = d[d["context"].str.len() == 406]
        d["protospacer"] = d["context"].str[PROTO_OFFSET:PROTO_OFFSET + 20]
        out[split] = d[~d["protospacer"].str.contains("N")].reset_index(drop=True)
    shared = set(out["training"].protospacer) & set(out["testing"].protospacer)
    if shared:
        # Worth checking rather than assuming: a guide on both sides of their
        # own split would make the hold-out partly seen, and any number from it
        # uninterpretable.
        print(f"  ! {len(shared)} protospacers appear in both of their splits")
    return out["training"], out["testing"], shared


def _metrics(y, pred) -> dict:
    from sklearn.metrics import r2_score
    k = 10
    order = np.argsort(pred)[::-1][:k]
    pct = float(np.mean([(y < y[i]).mean() for i in order]) * 100)
    top = int(len(y) * 0.1)
    hit = float(np.isin(np.argsort(pred)[::-1][:top],
                        np.argsort(y)[::-1][:top]).mean())
    return dict(n_test=len(y),
                spearman=float(spearmanr(y, pred).statistic),
                kendall_tau=float(kendalltau(y, pred).statistic),
                pearson=float(np.corrcoef(y, pred)[0, 1]),
                r2=float(r2_score(y, pred)),
                top_decile_prec=hit,
                pick_percentile=pct)


def _append(row: dict):
    df = pd.DataFrame([row])
    if OUT.exists() and OUT.stat().st_size:
        prev = pd.read_csv(OUT)
        prev = prev[prev["arm"] != row["arm"]]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(OUT, index=False)
    print(f"\n{df.round(4).to_string(index=False)}\nwritten -> {OUT}")


def slicer(verbose: bool = True) -> dict:
    """Champion XGBoost, trained on their training rows only."""
    import xgboost as xgb

    from .run_ablation import _impute, champion_params, make_selector

    tr, te, _ = _split_frames()
    X = np.asarray(np.load(CACHE / "X.npy", mmap_mode="r"), dtype=np.float32)
    meta = json.loads((CACHE / "meta.json").read_text())
    protos = pd.Index(meta["protospacers"])
    y_all = np.asarray(meta["y"], dtype=float)

    i_tr = protos.get_indexer(tr["protospacer"])
    i_te = protos.get_indexer(te["protospacer"])
    if (i_tr < 0).any() or (i_te < 0).any():
        raise SystemExit("some of their guides are missing from our features")
    # Their files score the same protospacer more than once; our matrix holds
    # one row per guide, so the indices repeat and are deduplicated here.
    i_tr, i_te = np.unique(i_tr), np.unique(i_te)
    overlap = np.intersect1d(i_tr, i_te)
    if len(overlap):
        i_tr = np.setdiff1d(i_tr, overlap)
        if verbose:
            print(f"  {len(overlap)} guides in both splits -> dropped from training")
    if verbose:
        print(f"  train {len(i_tr):,}  test {len(i_te):,}  columns {X.shape[1]:,}")

    t0 = time.time()
    X_tr, X_te = _impute(X[i_tr], X[i_te])
    y_tr, y_te = y_all[i_tr], y_all[i_te]
    sel = make_selector(41)
    sel.fit(X_tr, y_tr)
    top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
    m = xgb.XGBRegressor(**champion_params(41))
    m.fit(X_tr[:, top], y_tr)
    pred = m.predict(X_te[:, top])

    pd.DataFrame({"protospacer": protos[i_te], "y": y_te, "pred": pred}).to_csv(
        str(PRED).format(arm="slicer"), index=False)
    row = dict(arm="slicer", model="SLICER (champion XGBoost, 300 of 6,517)",
               n_train=int(len(i_tr)), **_metrics(y_te, pred),
               seconds=round(time.time() - t0, 1))
    _append(row)
    return row


def crisprhal(verbose: bool = True) -> dict:
    """Their shipped eSpCas9 weights, evaluated as distributed."""
    import keras

    tr, te, _ = _split_frames()
    sys.path.insert(0, str(HAL_DIR))
    cwd = os.getcwd()
    os.chdir(HAL_DIR)
    try:
        from processing import processing
        proc = processing()
        model = keras.models.load_model("models/ESPCAS9.keras", compile=False)
    finally:
        os.chdir(cwd)

    te = te.groupby("protospacer", as_index=False).agg(
        context=("context", "first"), score=("score", "mean"))
    t0 = time.time()
    X = proc.onehotencode(te["context"].to_numpy())
    pred = np.asarray(model.predict(X, verbose=0)).ravel()
    if not np.isfinite(pred).all() or np.std(pred) == 0:
        raise SystemExit("their model returned degenerate predictions")
    pd.DataFrame({"protospacer": te["protospacer"].to_numpy(),
                  "y": te["score"].to_numpy(float), "pred": pred}).to_csv(
        str(PRED).format(arm="crisprhal"), index=False)
    row = dict(arm="crisprhal", model="crisprHAL 2 (shipped ESPCAS9.keras)",
               n_train=int(len(np.unique(tr["protospacer"]))),
               **_metrics(te["score"].to_numpy(float), pred),
               seconds=round(time.time() - t0, 1))
    _append(row)
    return row


def compare(margin: float = 0.01, n_boot: int = 4000, seed: int = 0,
            verbose: bool = True) -> dict:
    """Is the tie a measured equivalence, or just an unresolved difference?

    "The two models are indistinguishable" cannot be established by failing to
    find a difference -- that is the same error the 427-column claim avoided
    with TOST. So the difference is bootstrapped **paired on guides**, which is
    the right unit because both models predict the same test rows and their
    errors are correlated; an unpaired interval would be far too wide.

    The margin is declared here rather than chosen after seeing the interval:
    **0.01 ρ**, twice the 0.005 used for the column-reduction claim, because
    0.005 is below the fold-to-fold noise of either model on this screen.
    Equivalence means the whole interval falls inside ±margin.
    """
    a = pd.read_csv(str(PRED).format(arm="slicer")).set_index("protospacer")
    b = pd.read_csv(str(PRED).format(arm="crisprhal")).set_index("protospacer")
    both = a.join(b, how="inner", lsuffix="_s", rsuffix="_h")
    if not np.allclose(both["y_s"], both["y_h"]):
        raise SystemExit("the two arms are not scoring the same labels")
    y = both["y_s"].to_numpy()
    ps, ph = both["pred_s"].to_numpy(), both["pred_h"].to_numpy()

    obs = spearmanr(y, ps).statistic - spearmanr(y, ph).statistic
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, n, n)
        diffs[i] = (spearmanr(y[k], ps[k]).statistic
                    - spearmanr(y[k], ph[k]).statistic)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    equivalent = bool(lo > -margin and hi < margin)
    row = dict(arm="difference", model="SLICER - crisprHAL 2, paired on guides",
               n_test=n, spearman=float(obs), ci_lo=float(lo), ci_hi=float(hi),
               margin=margin, equivalent=equivalent,
               p_two_sided=float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    _append(row)
    if verbose:
        print(f"\n  Delta rho = {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"n = {n:,}")
        print(f"  equivalent within +/-{margin}: {equivalent}")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["slicer", "crisprhal", "compare"],
                    required=True)
    ap.add_argument("--margin", type=float, default=0.01)
    a = ap.parse_args(argv)
    if a.arm == "compare":
        compare(margin=a.margin)
    else:
        (slicer if a.arm == "slicer" else crisprhal)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
