"""Both models, neither trained on it, scored on Cui's CRISPRi screen.

This is the "final exam" in the sense asked for: a screen neither model has
trained on, each given its own native input, one shot, both reported. It is
worth being exact about what it can and cannot settle.

**What it measures.** Whether a model of Cas9 *cutting* carries over to dCas9
*silencing fitness* -- a cross-assay transfer test. Both models predict cutting
activity; Cui's label is growth fitness under transcriptional repression. Those
are different constructs, so low numbers are the expectation for both and a
ranking here is a statement about which inductive bias happens to survive a
change of readout.

**What it is not.** A held-out test of the task the models were built for. For
that the screen would have to measure cutting, and no unused cutting screen
exists. Using this as *the* final exam would quietly swap the question.

**One asymmetry to declare.** SLICER's feature family has already been evaluated
against this label once (`cui_screen.py`, the mechanism test), so for our side
this screen is no longer untouched -- nothing was selected on it, and the result
there was a null, but the record should say so. crisprHAL has never seen it.

Sign convention: both models predict *activity*, and effective silencing makes
fitness **more negative**, so transfer appears as a **negative** correlation. The
magnitude is what matters and the sign is reported as measured.

    python -m sgrna.screens.cui_final_exam --arm slicer
    .venv-crisprhal/bin/python src/sgrna/cui_final_exam.py --arm crisprhal
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

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from sgrna import config
    from sgrna.build import genome
else:
    from .. import config
    from ..build import genome

OUT = config.RESULTS / "cui_final_exam.csv"
CACHE = config.INTERIM / "cui_crispri"
HAL_DIR = config.EXTERNAL / "crisprHAL"
UP, DOWN = 189, 166                     # their WT-SpCas9 378 nt window


def _cui():
    meta = json.loads((CACHE / "meta.json").read_text())
    return meta


def _within_gene(y, pred, gene, min_guides: int = 10) -> float:
    """Mean Spearman inside genes: removes the essentiality signal neither model knows."""
    d = pd.DataFrame({"y": y, "p": pred, "g": gene})
    out = [spearmanr(v["y"], v["p"]).statistic
           for _, v in d.groupby("g") if len(v) >= min_guides]
    out = [r for r in out if np.isfinite(r)]
    return float(np.mean(out))


def _append(row: dict):
    df = pd.DataFrame([row])
    if OUT.exists() and OUT.stat().st_size:
        prev = pd.read_csv(OUT)
        prev = prev[prev["arm"] != row["arm"]]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(OUT, index=False)
    print(f"\n{df.round(4).to_string(index=False)}\nwritten -> {OUT}")


def slicer(verbose: bool = True) -> dict:
    """Champion trained on the curated Cas9 cohort, predicting Cui's guides."""
    import xgboost as xgb

    from ..cohorts.curated_ablation import _base, _block
    from ..harness.run_ablation import _impute, champion_params, make_selector

    X, y, names, protos = _base()
    A, anames = _block("a_flank", protos)
    X = np.hstack([X, A])
    names = names + anames

    meta = _cui()
    Xc = np.asarray(np.load(CACHE / "X.npy", mmap_mode="r"), dtype=np.float32)
    cui_names = list(meta["names"])
    # Align by column name: the two cohorts' family A blocks can keep slightly
    # different column sets, and positional alignment would silently mis-map.
    idx = {n: i for i, n in enumerate(cui_names)}
    keep = [i for i, n in enumerate(names) if n in idx]
    take = [idx[names[i]] for i in keep]
    if verbose:
        print(f"  {len(keep):,} of {len(names):,} training columns present in "
              f"the Cui matrix")
    Xtr, Xte = X[:, keep], Xc[:, take]

    t0 = time.time()
    Xtr, Xte = _impute(Xtr, Xte)
    sel = make_selector(41)
    sel.fit(Xtr, y)
    top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
    m = xgb.XGBRegressor(**champion_params(41))
    m.fit(Xtr[:, top], y)
    pred = m.predict(Xte[:, top])

    yc = np.asarray(meta["y"], dtype=float)
    gene = np.asarray([g if isinstance(g, str) else "" for g in meta["gene"]],
                      dtype=object)
    ok = gene != ""
    row = dict(arm="slicer", model="SLICER (trained on curated Cas9, zero-shot)",
               n_test=int(len(yc)),
               spearman=float(spearmanr(yc, pred).statistic),
               spearman_within_gene=_within_gene(yc[ok], pred[ok], gene[ok]),
               seconds=round(time.time() - t0, 1))
    pd.DataFrame({"protospacer": meta["protospacers"], "y": yc,
                  "pred": pred}).to_csv(
        config.RESULTS / "preds_cui_slicer.csv", index=False)
    _append(row)
    return row


def crisprhal(verbose: bool = True) -> dict:
    """Their shipped WT-SpCas9 model, predicting Cui's guides from 378 nt."""
    import keras

    meta = _cui()
    g = genome.load_genome()
    seqs, lefts, strands = genome.protospacer_sites()
    site = {s: (l, t) for s, l, t in zip(seqs, lefts, strands)}
    ctx = []
    for s in meta["protospacers"]:
        left, strand = site[s]
        c = genome.guide_context(int(left), int(left) + 19,
                                 "+" if strand == 1 else "-", 1000, g)
        ctx.append(c["upstream"][-UP:] + c["protospacer"] + c["pam"]
                   + c["downstream"][:DOWN])
    lens = {len(c) for c in ctx}
    if lens != {UP + 23 + DOWN}:
        raise SystemExit(f"context lengths wrong: {lens}")

    sys.path.insert(0, str(HAL_DIR))
    cwd = os.getcwd()
    os.chdir(HAL_DIR)
    try:
        from processing import processing
        proc = processing()
        model = keras.models.load_model("models/WT-SPCAS9.keras", compile=False)
    finally:
        os.chdir(cwd)

    t0 = time.time()
    pred = np.asarray(model.predict(proc.onehotencode(np.asarray(ctx)),
                                    verbose=0)).ravel()
    if not np.isfinite(pred).all() or np.std(pred) == 0:
        raise SystemExit("their model returned degenerate predictions")
    yc = np.asarray(meta["y"], dtype=float)
    gene = np.asarray([x if isinstance(x, str) else "" for x in meta["gene"]],
                      dtype=object)
    ok = gene != ""
    row = dict(arm="crisprhal",
               model="crisprHAL 2 WT-SpCas9 (shipped, zero-shot)",
               n_test=int(len(yc)),
               spearman=float(spearmanr(yc, pred).statistic),
               spearman_within_gene=_within_gene(yc[ok], pred[ok], gene[ok]),
               seconds=round(time.time() - t0, 1))
    pd.DataFrame({"protospacer": meta["protospacers"], "y": yc,
                  "pred": pred}).to_csv(
        config.RESULTS / "preds_cui_crisprhal.csv", index=False)
    _append(row)
    return row


def compare(n_boot: int = 2000, seed: int = 0, verbose: bool = True) -> dict:
    """Paired bootstrap of the difference, on the guides both arms scored."""
    a = pd.read_csv(config.RESULTS / "preds_cui_slicer.csv").set_index("protospacer")
    b = pd.read_csv(config.RESULTS / "preds_cui_crisprhal.csv").set_index("protospacer")
    j = a.join(b, lsuffix="_s", rsuffix="_h", how="inner")
    y = j["y_s"].to_numpy()
    ps, ph = j["pred_s"].to_numpy(), j["pred_h"].to_numpy()
    # Both transfer as negative correlations, so magnitudes are compared.
    obs = abs(spearmanr(y, ps).statistic) - abs(spearmanr(y, ph).statistic)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, len(y), len(y))
        d[i] = (abs(spearmanr(y[k], ps[k]).statistic)
                - abs(spearmanr(y[k], ph[k]).statistic))
    lo, hi = np.percentile(d, [2.5, 97.5])
    row = dict(arm="difference", model="|SLICER| - |crisprHAL|, paired on guides",
               n_test=len(y), spearman=float(obs), ci_lo=float(lo),
               ci_hi=float(hi))
    _append(row)
    if verbose:
        print(f"\n  |rho| difference {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["slicer", "crisprhal", "compare"],
                    required=True)
    a = ap.parse_args(argv)
    {"slicer": slicer, "crisprhal": crisprhal, "compare": compare}[a.arm]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
