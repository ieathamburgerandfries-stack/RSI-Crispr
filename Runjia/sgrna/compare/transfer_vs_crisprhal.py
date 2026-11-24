"""Transferability, SLICER against crisprHAL 2: across nucleases within E. coli, and from E. coli to the eukaryotic tables.

Two comparisons, each with both models given the same training guides and scored on the same target guides.

  nuclease   Train on SpCas9 (crisprHAL 2's WT-SpCas9 weights, trained on their SpCas9 training file; SLICER, default settings, trained on the same
             26,830 guides) and score the eSpCas9 screen's labels, zero-shot, on every eSpCas9 guide: all 59,489, the 25,922 absent from the SpCas9 screen,
             and the 33,567 shared with it. crisprHAL 2's input is cut from each guide's own 378 nt context (189 nt upstream, 20 + 3, 166 downstream);
             SLICER's from the same guides' cached features.
  kingdom    Train on the 33,567 curated E. coli guides and score the eleven eukaryotic tables of the CRISPOR collection (label rank only; the same orientation
             rule as `eukaryote_rules.py`). crisprHAL 2's released weights need 378 nt of context that the eukaryotic tables do not have (they hold 30 nt before
             and 47 after the guide), so its architecture is retrained here on the 23-nt input (guide and PAM) with its published configuration (48 epochs, batch 256),
             the 'core' arm of the width experiment, and compared with SLICER's base representation (the 20-mer) with default settings. Both see the guide only.

    python -m sgrna.compare.transfer_vs_crisprhal prepare                 # main environment: contexts, SLICER predictions, the 23-nt tables
    .venv-crisprhal/bin/python -m sgrna.compare.transfer_vs_crisprhal tf  # TensorFlow environment: crisprHAL 2's predictions
    python -m sgrna.compare.transfer_vs_crisprhal report                  # main environment: the tables
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
RUNJIA = HERE.parents[2]
ROOT = HERE.parents[3]
WORK = ROOT / "data" / "interim" / "post_freeze" / "transfer_vs_crisprhal"
RESULTS = RUNJIA / "results" / "post_freeze"
HAL = ROOT / "external_data" / "crisprHAL"
UP, DOWN = 189, 166


def prepare() -> None:
    from scipy.stats import spearmanr

    from .. import config
    from ..build.base_fast import matrix_fast
    from ..cohorts.curated_ablation import _base
    from . import eukaryote_rules as ER
    from .transfer import build, load_dataset
    from .vs_crisprhal import member_specs, train_predict

    WORK.mkdir(parents=True, exist_ok=True)
    # ---- nuclease: eSpCas9 guides, their 378 nt contexts, and SLICER trained on the SpCas9 training file
    d = load_dataset("eSpCas9", verbose=False)
    assert d.upstream.str.len().min() >= UP and d.downstream.str.len().min() >= DOWN
    ctx = d.upstream.str[-UP:] + d.protospacer + d.pam + d.downstream.str[:DOWN]
    assert (ctx.str.len() == UP + 23 + DOWN).all()
    wt_all = load_dataset("WT-SpCas9", verbose=False)
    wt_tr = pd.read_csv(HAL / "data" / "WT-SpCas9_training_data.csv", header=None, names=["context", "score"]).dropna()
    wt_te = pd.read_csv(HAL / "data" / "WT-SpCas9_testing_data.csv", header=None, names=["context", "score"]).dropna()
    p_tr = set(wt_tr.context.str.upper().str[UP:UP + 20]) - set(wt_te.context.str.upper().str[UP:UP + 20])
    Xw, yw, nw = build("WT-SpCas9", verbose=False)
    Xe, ye, ne = build("eSpCas9", verbose=False)
    assert list(nw) == list(ne)
    import json
    pw = pd.Index(json.loads((config.INTERIM / "transfer_WT-SpCas9" / "meta.json").read_text())["protospacers"])
    pe = pd.Index(json.loads((config.INTERIM / "transfer_eSpCas9" / "meta.json").read_text())["protospacers"])
    tr = np.where(pw.isin(p_tr))[0]
    Xe = np.asarray(Xe, np.float32)
    preds, secs = train_predict(member_specs(["champion", "d099x"]), np.asarray(Xw, np.float32)[tr], np.asarray(yw, float)[tr], Xe)
    nb = int(json.loads((config.INTERIM / "transfer_WT-SpCas9" / "meta.json").read_text())["n_base"])
    pb, _ = train_predict(member_specs(["champion"]), np.asarray(Xw, np.float32)[tr][:, :nb], np.asarray(yw, float)[tr], Xe[:, :nb])      # the 20-mer representation alone
    out = pd.DataFrame({"protospacer": pe, "y": np.asarray(ye, float), "in_spcas9_screen": pe.isin(pw),
                        "slicer_default": preds["champion"], "slicer_tuned": preds["d099x"], "slicer_base_only": pb["champion"]})
    out.to_csv(WORK / "esp_slicer.csv", index=False)
    pd.DataFrame({"protospacer": pe, "context378": ctx.reindex(range(len(d))).to_numpy()[pd.Index(d.protospacer).get_indexer(pe)]}).to_csv(WORK / "esp_contexts.csv", index=False)
    print(f"  eSpCas9: {len(out):,} guides, SLICER trained on {len(tr):,} SpCas9 training-file guides ({secs})", flush=True)
    # ---- kingdom: E. coli 23-mers for training, eukaryotic 23-mers and SLICER base predictions
    D = ER.load_all()
    ec = ER.ecoli()
    from ..build.build_features import load_curated_index
    idx = load_curated_index(verbose=False).set_index(config.ID_COL).reindex(ec["protos"])
    pd.DataFrame({"seq23": (idx.protospacer + idx.pam).to_numpy(), "y": ec["y"]}).to_csv(WORK / "ecoli_23.csv", index=False)
    big = [k for k, v in D.items() if v["n"] >= ER.MIN_BIG]
    rows, slicer_rows = [], []
    Xb = np.asarray(ec["X"], np.float32)
    preds_e, _ = train_predict(member_specs(["champion"]), Xb, ec["y"], np.vstack([np.asarray(matrix_fast(list(D[k]["protos"]))[0], np.float32) for k in big]))
    off = 0
    for k in big:
        n = D[k]["n"]
        pam = [c[50:53] for c in D[k]["ctx"]]
        rows.append(pd.DataFrame({"dataset": k, "seq23": [p + m for p, m in zip(D[k]["protos"], pam)], "y": D[k]["y"]}))
        slicer_rows.append(dict(dataset=k, species=D[k]["species"], n=n, slicer_base=float(spearmanr(D[k]["y"], preds_e["champion"][off:off + n]).statistic)))
        off += n
    pd.concat(rows).to_csv(WORK / "euk_23.csv", index=False)
    pd.DataFrame(slicer_rows).to_csv(WORK / "euk_slicer.csv", index=False)
    print(f"  eukaryotic tables: {len(big)} datasets, {sum(D[k]['n'] for k in big):,} guides", flush=True)


def tf() -> None:
    """crisprHAL 2: WT weights on the eSpCas9 contexts; the architecture retrained on the 23-nt E. coli input and scored on the eukaryotic 23-mers."""
    from scipy.stats import spearmanr

    sys.path.insert(0, str(HAL))
    cwd = os.getcwd()
    os.chdir(HAL)
    import keras
    import models as hal_models
    from models import models
    from processing import processing
    os.chdir(cwd)
    import tensorflow as tf_
    proc = processing()
    c = pd.read_csv(WORK / "esp_contexts.csv")
    os.chdir(HAL)
    wt = keras.models.load_model("models/WT-SPCAS9.keras", compile=False)
    os.chdir(cwd)
    p = np.asarray(wt.predict(proc.onehotencode(c.context378.str.upper().to_numpy()), batch_size=512, verbose=0)).ravel()
    pd.DataFrame({"protospacer": c.protospacer, "crisprhal_wt": p}).to_csv(WORK / "esp_crisprhal.csv", index=False)
    print("  WT weights on eSpCas9 contexts:", len(p), flush=True)
    ec = pd.read_csv(WORK / "ecoli_23.csv")
    euk = pd.read_csv(WORK / "euk_23.csv")
    hal_models.modelVersionInputLength["WT-SPCAS9"][0] = 23
    tf_.keras.backend.clear_session()
    tf_.keras.utils.set_random_seed(41)
    m = models("WT-SPCAS9", summary=False)
    t0 = time.time()
    m.model.fit(proc.onehotencode(ec.seq23.to_numpy()).astype("float32"), ec.y.to_numpy(), epochs=hal_models.modelVersionDefaultEpochs["WT-SPCAS9"], batch_size=256, verbose=0)
    pe = np.asarray(m.model.predict(proc.onehotencode(euk.seq23.to_numpy()).astype("float32"), batch_size=512, verbose=0)).ravel()
    print(f"  23-nt crisprHAL 2 fitted on {len(ec):,} E. coli guides in {(time.time() - t0) / 60:.1f} min", flush=True)
    rows = []
    for k, g in euk.assign(p=pe).groupby("dataset", sort=False):
        rows.append(dict(dataset=k, crisprhal_core23=float(spearmanr(g.y, g.p).statistic)))
    pd.DataFrame(rows).to_csv(WORK / "euk_crisprhal.csv", index=False)


def _boot(y, a, b, B=2000, seed=0):
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    d = np.empty(B)
    for i in range(B):
        k = rng.integers(0, len(y), len(y))
        d[i] = spearmanr(y[k], a[k]).statistic - spearmanr(y[k], b[k]).statistic
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def report() -> None:
    from scipy.stats import spearmanr

    e = pd.read_csv(WORK / "esp_slicer.csv").merge(pd.read_csv(WORK / "esp_crisprhal.csv"), on="protospacer")
    rows = []
    for name, m in (("all eSpCas9 guides", np.ones(len(e), bool)), ("guides absent from the SpCas9 screen", ~e.in_spcas9_screen.to_numpy()),
                    ("guides also in the SpCas9 screen", e.in_spcas9_screen.to_numpy())):
        s = e[m]
        y = s.y.to_numpy()
        for arm in ("slicer_default", "slicer_tuned", "slicer_base_only"):
            lo, hi = _boot(y, s[arm].to_numpy(), s.crisprhal_wt.to_numpy())
            rows.append(dict(part="nuclease", subset=name, n=int(m.sum()), model=arm, rho=float(spearmanr(y, s[arm]).statistic),
                             crisprhal=float(spearmanr(y, s.crisprhal_wt).statistic), diff=float(spearmanr(y, s[arm]).statistic - spearmanr(y, s.crisprhal_wt).statistic),
                             lo=lo, hi=hi))
    k = pd.read_csv(WORK / "euk_slicer.csv").merge(pd.read_csv(WORK / "euk_crisprhal.csv"), on="dataset")
    own = pd.read_csv(RESULTS / "eukaryote_transfer.csv")
    own = own[(own.model.str.startswith("pipeline")) & own.within].set_index("scored_on").rho
    k["own_cv_rho"] = k.dataset.map(own)
    k["slicer_retained"] = k.slicer_base / k.own_cv_rho
    k["crisprhal_retained"] = k.crisprhal_core23 / k.own_cv_rho
    k.to_csv(RESULTS / "transfer_vs_crisprhal_kingdom.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / "transfer_vs_crisprhal_nuclease.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(k.round(3).to_string(index=False))
    print("median rho: SLICER base", round(k.slicer_base.median(), 4), " crisprHAL 2 (23 nt)", round(k.crisprhal_core23.median(), 4))


def main(argv=None) -> int:
    cmd = (argv or sys.argv[1:] or [""])[0]
    {"prepare": prepare, "tf": tf, "report": report}.get(cmd, lambda: print(__doc__))()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
