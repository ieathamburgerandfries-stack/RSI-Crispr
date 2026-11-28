"""SLICER against crisprHAL 2 under every protocol of Part 5: the tuned model leads by +0.0226 on crisprHAL 2's SpCas9 split and +0.0286 on its eSpCas9 split; paired on guides on our folds it leads by +0.0115 untuned and +0.0043 against crisprHAL 2's best settings; and the same comparisons are made on genes the tuning never saw and on whole-gene folds.

Fourteen conditions (`--condition`), one result file each:

  espcas9        crisprHAL's own shipped eSpCas9 split (47,591 train / 11,898 test), against their shipped weights (predictions made by `hal-espcas9`); paired bootstrap on guides        vs_crisprhal_espcas9.csv
  wt             crisprHAL's own shipped SpCas9 split (26,830 train / 6,737 test), against their shipped WT weights (predictions made by `hal-wt` in the TensorFlow environment)          vs_crisprhal_wt.csv
  folds          our five curated folds (KFold 5, seed 41, 33,567 guides), against crisprHAL 2's stored out-of-fold predictions at its published configuration and at the best configuration found  vs_crisprhal_folds.csv
  paired         the same folds with the difference bootstrapped paired on guides: three pairings (untuned, tuned, cross)                                                                 heldout_compare.csv
  confirm        the 30% of genes held out from every search, where nothing in the comparison was tuned                                                                                   clean_vs_hal.csv
  gene-folds     SLICER's default and tuned models on folds that hold whole genes out (also writes the input file of crisprHAL's Colab run)                                              gene_grouped_slicer.csv
  gene-assemble  crisprHAL 2's per-fold rho from that Colab run, paired against SLICER's on the same folds                                                                              gene_grouped_vs_crisprhal.csv
  final-slicer   the designated final evaluation, SLICER's side: the frozen champion, trained on their eSpCas9 training rows only                                                         final_eval_espcas9.csv (arm slicer)
  hal-espcas9    its crisprHAL 2 side: their shipped `ESPCAS9.keras`, loaded and evaluated as distributed, no retraining (TensorFlow environment)                                       final_eval_espcas9.csv (arm crisprhal)
  final-compare  the paired bootstrap of the two, with the equivalence margin of 0.01 declared in code before the interval was seen                                                    final_eval_espcas9.csv (arm difference)
  hal-wt         crisprHAL 2's shipped WT-SpCas9 weights on their own WT test file (TensorFlow environment)                                                                              preds_final_wt_crisprhal.csv
  transfer-prepare, transfer-tf, transfer-report   transferability of the two models across nucleases and across kingdoms: the contexts and SLICER's predictions (main environment), crisprHAL 2's (TensorFlow environment), the tables    transfer_vs_crisprhal_nuclease.csv, transfer_vs_crisprhal_kingdom.csv

Each SLICER arm trains on the training rows only, with column selection fitted inside them, exactly as in `final-slicer`. Arms: the frozen champion (reproduces the earlier figures), the tuned XGBoost (d099x), the XGBoost average of five, a mixed
ensemble chosen on held-out genes in `mix_ecoli.py` (its members are read from `--mix`). Every comparison on crisprHAL 2's two splits starts from `hal_split`. The `hal-*` conditions run in the TensorFlow environment
(`.venv-crisprhal`), which has none of xgboost, lightgbm or catboost, so the tuning modules are imported only by the conditions that need them.

    python -m sgrna.compare.vs_crisprhal --condition espcas9 --mix "d099x,d077x,d021x,d080x,d000x;lgb011,lgb018;cat015,cat023"
    python -m sgrna.compare.vs_crisprhal --condition paired        # or: confirm, gene-folds, gene-assemble, final-slicer, final-compare
    .venv-crisprhal/bin/python -m sgrna.compare.vs_crisprhal --condition hal-espcas9        # or: hal-wt, transfer-tf

------------------------------------------------------------------------------------------------------------------------------
transfer-prepare, transfer-tf, transfer-report
------------------------------------------------------------------------------------------------------------------------------
Two comparisons, each with both models given the same training guides and scored on the same target guides.

  nuclease   Train on SpCas9 (crisprHAL 2's WT-SpCas9 weights, trained on their SpCas9 training file; SLICER, default settings, trained on the same
             26,830 guides) and score the eSpCas9 screen's labels, zero-shot, on every eSpCas9 guide: all 59,489, the 25,922 absent from the SpCas9 screen,
             and the 33,567 shared with it. crisprHAL 2's input is cut from each guide's own 378 nt context (189 nt upstream, 20 + 3, 166 downstream);
             SLICER's from the same guides' cached features.
  kingdom    Train on the 33,567 curated E. coli guides and score the eleven eukaryotic tables of the CRISPOR collection (label rank only; the same orientation
             rule as `eukaryote_rules.py`). crisprHAL 2's released weights need 378 nt of context that the eukaryotic tables do not have (they hold 30 nt before
             and 47 after the guide), so its architecture is retrained here on the 23-nt input (guide and PAM) with its published configuration (48 epochs, batch 256),
             the 'core' arm of the width experiment, and compared with SLICER's base representation (the 20-mer) with default settings. Both see the guide only.

------------------------------------------------------------------------------------------------------------------------------
final-slicer, hal-espcas9, final-compare
------------------------------------------------------------------------------------------------------------------------------
Part 4 reserved the eSpCas9 screen as "the designated untouched final evaluation". **It is not untouched**, and the record has to say so: four feature arms were already run on it across three seeds and two CV schemes
(`results/transfer.csv`), and a flank gain of +0.104 under `arc5` is quoted in `RESULTS.md`. Nothing was *selected* on eSpCas9 -- every configuration was chosen on the published or curated cohort and then confirmed there --
so the honest description is a confirmatory dataset, not a holdout, and "evaluated once" was never available.

What **is** untouched is the comparison: SLICER against crisprHAL 2 on **crisprHAL's own shipped eSpCas9 train/test partition**, with their own published eSpCas9 weights. No split, model, feature set or hyperparameter here was chosen with
reference to eSpCas9. Two arms, one test set (their 11,898-guide eSpCas9 test file): `slicer` is the champion XGBoost, 300 columns selected inside their training split only, base (6,169 regenerated) + family A (348) = 6,517;
`crisprhal` is their shipped `models/ESPCAS9.keras`, loaded and evaluated as distributed. Frozen means frozen: the SLICER arm takes `champion_params`, `make_selector` and `N_FEATURES` as they stand, and the selector is fitted on their
training rows so no test row touches feature selection.

"The two models are indistinguishable" cannot be established by failing to find a difference, so `final-compare` bootstraps the difference **paired on guides** (both models predict the same test rows and their errors are correlated; an
unpaired interval would be far too wide). The margin is declared in the code rather than chosen after seeing the interval: **0.01 rho**, twice the 0.005 used for the column-reduction claim, because 0.005 is below the fold-to-fold noise
of either model on this screen. Equivalence means the whole interval falls inside the margin.

------------------------------------------------------------------------------------------------------------------------------
paired
------------------------------------------------------------------------------------------------------------------------------
The curated five-fold head-to-head is reported two ways elsewhere: as a
fold-level mean (Appendix B) and as a pooled nine-metric panel (`metric_panel`).
Neither puts an interval on the difference. The fold-level version has n = 5, so
it cannot resolve anything below ~0.005; the pooled version has no uncertainty
at all.

This closes that gap. Every guide is predicted exactly once by a model that
never saw it, so the five validation folds together give **33,567 out-of-fold
predictions per model** over the same rows. The difference can then be
bootstrapped **paired on guides**, which is the right unit: both models predict
the same rows and their errors are correlated, so an unpaired interval would be
far too wide.

Three pairings:

    untuned   SLICER at `tune.DEFAULT` against crisprHAL 2 at its published
              defaults (48 epochs, batch 1024)
    tuned     SLICER at its per-fold tuned parameters against crisprHAL 2 at
              the best of its own search
    cross     SLICER **untuned** against crisprHAL 2 **tuned** -- the most
              conservative comparison available to us, and the one that bounds
              how much of our margin is a tuning-effort artefact

The tuned pairing carries a declared asymmetry, and it is larger than this file
said until 16 October: SLICER's search ran **97-157 draws inside each of five
folds** (627 in total, `results/tuning.csv`), while crisprHAL's best arm came
from 10 draws on a single inner split of fold 1, because each of its fits costs
~17 CPU-minutes locally. Neither search ever touches an outer validation fold, so
neither number is leaked -- but theirs is ~20x weaker, so the tuned margin is an
**upper bound**.

Two things bound it from the other side. A nested search of their own on these
folds (16 October, free T4) lands at **0.6987** against their published 0.6971,
with their published configuration winning the inner search in four folds of
five -- so the space contains nothing better than what they shipped. And the
final evaluation against their own *published* weights, on their own splits,
gives +0.002 and +0.005. The conservative margin to quote from this file is
therefore the one against their best known configuration, **+0.0161**.

**Alignment is the thing that can go wrong.** crisprHAL's predictions were saved
one file per fold with no guide key, so the concatenation is in fold order, not
in cohort order. Pairing them with SLICER's out-of-fold vector requires
reconstructing `KFold(5, shuffle=True, random_state=41)` and concatenating its
validation indices in the same order. The `y` column is the check: if the
alignment is right, the labels must match exactly, guide for guide.

------------------------------------------------------------------------------------------------------------------------------
confirm
------------------------------------------------------------------------------------------------------------------------------
The tuned settings were chosen on 70% of the curated cohort's genes. crisprHAL's WT-SpCas9 data are the same 33,567 guides, so the
"their split" and "our folds" comparisons of `vs_crisprhal.py` contain guides whose labels fed the settings' selection (not any model's
weights). The CONFIRM genes (1,204 genes, 8,943 guides) did not. Here the SLICER models are those trained on the tuning genes only and
scored on CONFIRM genes (stored by `tune_ecoli --confirm`); crisprHAL 2 is its stored out-of-fold prediction for the same guides
(trained on 80% of all guides by random folds, so on guides from the same genes: an advantage to it, which makes this comparison
conservative for SLICER). Differences are paired and bootstrapped over genes.

------------------------------------------------------------------------------------------------------------------------------
gene-folds and gene-assemble
------------------------------------------------------------------------------------------------------------------------------
The head-to-head of Part 5 is on crisprHAL 2's own random splits, as it ships. A random split can reward a model for recognising a locus, and a network that reads
378 nt of context can recognise loci better than one that reads a 20-mer and a few composition summaries, so that comparison may favour crisprHAL 2. This module sets
up the same comparison on the gene-grouped folds used everywhere else (whole genes in a fold, guides outside genes grouped by 100 kb block; seed 41):

  folds    writes `data/interim/crisprhal_widewindow_input_genefolds.csv.gz` (the Colab input file with one more column, `gene_fold`, 0-4) and the per-fold rho of SLICER's
           default and tuned models on exactly those folds (`gene_grouped_slicer.csv`)
  assemble reads crisprHAL 2's per-fold rho from `results/crisprhal_gene_grouped_fcolab_b1024.csv` (written by `colab/crisprhal_gene_grouped.py`, about 30 minutes on a T4, at
           crisprHAL 2's published batch size of 1,024) and writes the paired comparison (`gene_grouped_vs_crisprhal.csv`). A first run used batch 256 and is kept in
           `crisprhal_gene_grouped_fcolab_b256.csv`; it handicaps crisprHAL 2 (256 cost it 0.016 on random-row folds) and is reported only as that, beside the published batch

Report section: Part 5.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import KFold

from sgrna import config
from ..common import INTERIM, MAIN_RESULTS, OUT, stored
from ..harness.metrics import zscore as _z

SRC = MAIN_RESULTS                           # the main project's results folder (stored crisprHAL predictions)


def member_specs(names):
    from ..tuning import mix_ecoli as M, tune_ecoli as T            # xgboost, lightgbm, catboost: not in the TensorFlow environment, which runs the `hal-*` conditions
    cfgs = {**json.loads((OUT / "tune_configs.json").read_text()), **M.all_cfgs()}
    fin = pd.read_csv(OUT / "mix_finalists.csv").set_index("name")
    sel = T.selected()
    out = []
    for n in names:
        if n in ("champion", "depth6"):
            out.append(("xgb", n, int(sel.loc[n, "k"]), cfgs[n]))
        elif n in fin.index:
            out.append((fin.loc[n, "kind"], n, int(fin.loc[n, "k"]), cfgs[n]))
        else:
            out.append(("xgb", n, int(sel.loc[n, "k"]), cfgs[n]))
    return out


def train_predict(specs, X_tr, y_tr, X_te, seed=41):
    """Selection on the training rows, then every member; returns ({name: prediction}, {name: fit seconds})."""
    from sgrna.harness.run_ablation import _impute, make_selector
    from ..tuning import mix_ecoli as M
    A, _ = _impute(X_tr, X_te[:1])
    s = make_selector(seed); s.fit(A, y_tr)
    top = np.argsort(s.feature_importances_)[::-1][:900]
    preds, secs = {}, {}
    for kind, name, k, cfg in specs:
        cfg = dict(cfg)
        if kind in ("xgb", "lgb", "cat"):
            cfg["n_estimators"] = k
        t0 = time.time()
        Xa, Xb = M.columns(cfg, top, X_tr, X_te, kind)
        p = list(M.fit_predict(kind, cfg, Xa, y_tr, Xb, seed, (k,) if kind in ("xgb", "lgb", "cat") else None).values())[0]
        preds[name] = p; secs[name] = time.time() - t0
    return preds, secs


def _flat(ns):
    return [m for x in ns for m in (x if isinstance(x, list) else [x])]


def _blend(preds, ns):
    """A flat list is an equal-weight average of z-scored members; a list of lists averages each group first, then the groups."""
    if ns and isinstance(ns[0], list):
        return np.mean([_blend(preds, g) for g in ns], axis=0)
    return np.mean([_z(preds[n]) for n in ns], axis=0)


def arms_from(preds, families):
    """families: {arm label: members or groups of members} -> equal-weight z-averages (see `_blend`)."""
    return {lab: _blend(preds, ns) for lab, ns in families.items()}


def paired_boot(y, ps, ph, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(0, len(y), len(y))
        d[i] = spearmanr(y[k], ps[k]).statistic - spearmanr(y[k], ph[k]).statistic
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(spearmanr(y, ps).statistic - spearmanr(y, ph).statistic), float(lo), float(hi)


# ------------------------------------------------------------------ crisprHAL 2's two test splits
def hal_split(split):
    """crisprHAL 2's own train/test partition of one screen, as SLICER's matrices, with the shipped crisprHAL 2 predictions: `espcas9` (their eSpCas9 files) or `wt` (the WT-SpCas9 cohort's test flag).
    -> (X_train, y_train, X_test, y_test, test protospacers as an Index, crisprHAL's predictions indexed by protospacer with columns y and pred). Every comparison on these splits starts here."""
    if split == "espcas9":
        tr, te, _ = _split_frames()
        X = np.asarray(np.load(CACHE / "X.npy", mmap_mode="r"), np.float32)
        meta = json.loads((CACHE / "meta.json").read_text())
        protos = pd.Index(meta["protospacers"]); y_all = np.asarray(meta["y"], float)
        i_tr, i_te = np.unique(protos.get_indexer(tr["protospacer"])), np.unique(protos.get_indexer(te["protospacer"]))
        i_tr = np.setdiff1d(i_tr, np.intersect1d(i_tr, i_te))
        hal = pd.read_csv(SRC / "preds_final_espcas9_crisprhal.csv").set_index("protospacer")
        return X[i_tr], y_all[i_tr], X[i_te], y_all[i_te], pd.Index(protos[i_te]), hal
    from sgrna.cohorts.data_ablation import load
    X, y, names, protos, in_test, is_noshay, nos, n_base = load(verbose=False)
    te, tr = np.where(in_test)[0], np.where(~in_test)[0]
    hal = pd.read_csv(SRC / "preds_final_wt_crisprhal.csv").set_index("protospacer")
    return X[tr], y[tr], X[te], y[te], pd.Index(np.asarray(protos)[te]), hal


# ------------------------------------------------------------------ the three conditions
def espcas9(families, specs):
    Xtr, ytr, Xte, yte, order, hal = hal_split("espcas9")
    preds, secs = train_predict(specs, Xtr, ytr, Xte)
    both = hal.loc[hal.index.intersection(order)]
    pos = order.get_indexer(both.index); y = yte[pos]
    assert np.allclose(y, both["y"].to_numpy(), atol=1e-9), "label alignment with crisprHAL's predictions failed"
    arms = arms_from({k: v[pos] for k, v in preds.items()}, families)
    return _rows("espcas9", y, arms, both["pred"].to_numpy(), "crisprHAL 2 (shipped ESPCAS9)", secs, len(ytr), families)


def wt(families, specs):
    Xtr, ytr, Xte, yte, order, hal = hal_split("wt")
    preds, secs = train_predict(specs, Xtr, ytr, Xte)
    keep = order.isin(hal.index) & ~order.duplicated()
    pos = np.where(keep)[0]; ph = hal.loc[order[pos], "pred"].to_numpy(); yy = yte[pos]
    arms = arms_from({k: v[pos] for k, v in preds.items()}, families)
    return _rows("wt", yy, arms, ph, "crisprHAL 2 (shipped WT-SPCAS9)", secs, len(ytr), families)


def folds(families, specs, seed=41):
    from sklearn.model_selection import KFold
    from sgrna.harness.curated import load_base, load_block
    X, y, _, protos = load_base(); A, _ = load_block("a_flank", protos); X = np.hstack([X, A])
    oof = {s[1]: np.zeros(len(y)) for s in specs}; secs_all = {s[1]: 0.0 for s in specs}
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(np.arange(len(y))):
        p, secs = train_predict(specs, X[tr], y[tr], X[va], seed)
        for k in p:
            oof[k][va] = p[k]; secs_all[k] += secs[k]
    # per-fold z-scoring would change the pooled ranking; blend inside each fold instead
    arms = {}
    for lab, ns in families.items():
        z = np.zeros(len(y))
        for tr, va in KFold(5, shuffle=True, random_state=seed).split(np.arange(len(y))):
            z[va] = _blend({k: v[va] for k, v in oof.items()}, ns)
        arms[lab] = z
    order = np.concatenate([va for _, va in KFold(5, shuffle=True, random_state=seed).split(np.arange(len(y)))])
    rows = []
    for kind in ("default", "tuned"):
        files = stored(f"preds_crisprhal_{kind}_fold*.csv")
        h = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        assert np.allclose(y[order], h["y"].to_numpy(), atol=1e-9), "fold alignment failed"
        ph = np.empty(len(y)); ph[order] = h["pred"].to_numpy()
        rows += _rows(f"folds vs crisprHAL 2 ({kind} config)", y, arms, ph, f"crisprHAL 2 ({kind})", {k: v / 5 for k, v in secs_all.items()}, int(0.8 * len(y)), families)
    return rows


def _rows(cond, y, arms, ph, hal_label, secs, n_train, families=None):
    out = []
    rho_h = float(spearmanr(y, ph).statistic)
    for lab, p in arms.items():
        d, lo, hi = paired_boot(y, p, ph)
        out.append(dict(condition=cond, arm=lab, n_test=len(y), n_train=n_train, rho=float(spearmanr(y, p).statistic),
                        crisprhal=rho_h, diff=d, lo=lo, hi=hi, crisprhal_model=hal_label,
                        fit_seconds=float(sum(secs[n] for n in _flat(families[lab]))) if families else float('nan')))
        print(f"  {cond:34s} {lab:28s} rho {out[-1]['rho']:.4f}  {hal_label} {rho_h:.4f}  diff {d:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
    return out


# ------------------------------------------------------------------ paired on guides (the curated folds)
PAIRED_OUT = OUT / "heldout_compare.csv"
PAIRED_OOF = config.RESULTS / "preds_slicer_{kind}_oof.csv"
PAIRED_SEED = 41
N_SPLITS = 5


def _fold_order(n: int, seed: int = PAIRED_SEED, n_splits: int = N_SPLITS) -> np.ndarray:
    """Row indices in the order crisprHAL's per-fold files were concatenated."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return np.concatenate([va for _, va in kf.split(np.arange(n))])


def slicer_oof(kind: str, verbose: bool = True) -> pd.DataFrame:
    """Out-of-fold predictions for SLICER, cached to disk."""
    path = str(PAIRED_OOF).format(kind=kind)
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        pass
    from .panel import _slicer_oof

    y, pred = _slicer_oof(tuned=(kind == "tuned"), seed=PAIRED_SEED, n_splits=N_SPLITS, verbose=verbose)
    meta = json.loads((config.INTERIM / "expanded_meta.json").read_text())
    d = pd.DataFrame({"protospacer": meta["protospacers"], "y": y, "pred": pred})
    d.to_csv(path, index=False)
    return d


def _hal_oof(kind: str) -> pd.DataFrame:
    pat = f"preds_crisprhal_{'tuned' if kind == 'tuned' else 'default'}_fold*.csv"
    files = sorted(config.RESULTS.glob(pat))
    if len(files) != N_SPLITS:
        raise FileNotFoundError(f"expected {N_SPLITS} files for {pat}, got {len(files)}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def paired(n_boot: int = 2000, seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for kind in ("untuned", "tuned", "cross"):
        # "cross" pits our defaults against their tuned model.
        s = slicer_oof("untuned" if kind == "cross" else kind, verbose=verbose)
        h = _hal_oof("tuned" if kind == "cross" else kind)
        order = _fold_order(len(s))
        s_ord = s.iloc[order].reset_index(drop=True)
        if len(h) != len(s_ord) or not np.allclose(s_ord["y"], h["y"], atol=1e-9):
            raise SystemExit(f"{kind}: alignment check failed -- the labels do not match guide for guide, so the pairing would be meaningless")
        y = s_ord["y"].to_numpy()
        ps, ph = s_ord["pred"].to_numpy(), h["pred"].to_numpy()

        rho_s = float(spearmanr(y, ps).statistic)
        rho_h = float(spearmanr(y, ph).statistic)
        boot = np.empty(n_boot)
        for i in range(n_boot):
            k = rng.integers(0, len(y), len(y))
            boot[i] = (spearmanr(y[k], ps[k]).statistic - spearmanr(y[k], ph[k]).statistic)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(pairing=kind, n=len(y), slicer=rho_s, crisprhal=rho_h, difference=rho_s - rho_h, ci_lo=float(lo), ci_hi=float(hi),
                         excludes_zero=bool(lo > 0 or hi < 0), p_two_sided=float(2 * min((boot <= 0).mean(), (boot >= 0).mean()))))
        if verbose:
            r = rows[-1]
            print(f"  {kind:>8}: SLICER {rho_s:.4f}  crisprHAL {rho_h:.4f}  diff {r['difference']:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={r['p_two_sided']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(PAIRED_OUT, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(PAIRED_OUT, index=False)
    if verbose:
        print(f"\n{out.round(4).to_string(index=False)}\nwritten -> {PAIRED_OUT}")
    return out


# ------------------------------------------------------------------ the genes the tuning never saw
def confirm(n_boot=2000, seed=0, seeds=(41, 42, 43)):
    from ..tuning import tune_ecoli as T
    from sgrna.harness.curated import load_base, load_groups

    X, y, _, protos = load_base()
    g, ok = load_groups("gene", protos)
    co = T.Cohort()
    assert np.array_equal(y[ok], co.y), "cohort alignment failed"
    order = np.concatenate([va for _, va in KFold(5, shuffle=True, random_state=41).split(np.arange(len(y)))])
    hal = {}
    for kind in ("default", "tuned"):
        h = pd.concat([pd.read_csv(f) for f in stored(f"preds_crisprhal_{kind}_fold*.csv")], ignore_index=True)
        assert np.allclose(y[order], h["y"].to_numpy(), atol=1e-9)
        ph = np.empty(len(y)); ph[order] = h["pred"].to_numpy()
        hal[kind] = ph[ok][co.conf]
    yc, gc = co.y[co.conf], co.g[co.conf]
    sel = T.selected()
    load = lambda name: np.array([np.load(T.DIR / f"conf_{name}_k{int(sel.loc[name, 'k'])}_s{sd}.npy") for sd in seeds])
    members = ["d099x", "d077x", "d021x", "d080x", "d000x"]
    arms = {"original champion": load("champion"), "tuned XGBoost (d099x)": load("d099x"),
            "average of 5 XGBoost": np.array([np.mean([T._z(load(n)[i]) for n in members], axis=0) for i in range(len(seeds))])}
    ug, inv = np.unique(gc, return_inverse=True); rows_of = [np.where(inv == i)[0] for i in range(len(ug))]
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([rows_of[i] for i in rng.integers(0, len(ug), len(ug))]) for _ in range(n_boot)]
    rows = []
    for hk, hp in hal.items():
        rh = float(spearmanr(yc, hp).statistic)
        bh = np.array([spearmanr(yc[b], hp[b]).statistic for b in boots])
        for lab, P in arms.items():
            r = float(np.mean([spearmanr(yc, p).statistic for p in P]))
            bs = np.array([np.mean([spearmanr(yc[b], p[b]).statistic for p in P]) for b in boots]) - bh
            rows.append(dict(vs=f"crisprHAL 2 ({hk} configuration)", arm=lab, n=len(yc), n_genes=len(ug), rho=r, crisprhal=rh, diff=r - rh,
                             lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5))))
            print(f"  {rows[-1]['vs']:36s} {lab:24s} rho {r:.4f} vs {rh:.4f}  diff {r - rh:+.4f} [{rows[-1]['lo']:+.4f}, {rows[-1]['hi']:+.4f}]", flush=True)
    out = pd.DataFrame(rows); out.to_csv(OUT / "clean_vs_hal.csv", index=False)
    return out


# ------------------------------------------------------------------ whole-gene folds
GG_SLICER = OUT / "gene_grouped_slicer.csv"
GG_HAL = OUT / "crisprhal_gene_grouped_fcolab_b1024.csv"
GG_HAL256 = OUT / "crisprhal_gene_grouped_fcolab_b256.csv"
GG_RESULT = OUT / "gene_grouped_vs_crisprhal.csv"
GG_SEED = 41


def gene_folds() -> None:
    from ..controls.mappability_stratum import _groups
    from ..harness import folds as sg
    from ..harness.curated import load_base, load_block

    X, y, names, protos = load_base()
    A, _ = load_block("a_flank", protos)
    X = np.hstack([X, A]).astype(np.float32)
    fold = sg.fold_ids(_groups(protos), 5, GG_SEED)
    src = pd.read_csv(config.INTERIM / "crisprhal_widewindow_input.csv.gz")
    fmap = pd.Series(fold, index=protos)
    assert src.protospacer.isin(fmap.index).all()
    src["gene_fold"] = src.protospacer.map(fmap).astype(int)
    src.to_csv(config.INTERIM / "crisprhal_widewindow_input_genefolds.csv.gz", index=False)
    print("  folds written; sizes", np.bincount(fold).tolist(), flush=True)
    rows = []
    for k in range(5):
        va = fold == k
        preds, _ = train_predict(member_specs(["champion", "d099x"]), X[~va], y[~va], X[va], GG_SEED)
        for arm, name in (("SLICER default", "champion"), ("SLICER tuned", "d099x")):
            rows.append(dict(model=arm, fold=k, n_test=int(va.sum()), spearman=float(spearmanr(y[va], preds[name]).statistic)))
        print(f"  fold {k}: default {rows[-2]['spearman']:.4f}  tuned {rows[-1]['spearman']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(GG_SLICER, index=False)


def gene_assemble() -> None:
    s = pd.read_csv(GG_SLICER)
    parts = [("crisprHAL 2", GG_HAL), ("crisprHAL 2 (batch 256)", GG_HAL256)]
    d = s[["model", "fold", "spearman"]]
    have = []
    for name, path in parts:
        if path.exists():
            d = pd.concat([d, pd.read_csv(path).assign(model=name)[["model", "fold", "spearman"]]])
            have.append(name)
    p = d.pivot_table(index="fold", columns="model", values="spearman")
    rows = []
    for m in ("SLICER default", "SLICER tuned"):
        for h in have:
            diff = p[m] - p[h]
            rows.append(dict(model=m, against=h, mean=float(p[m].mean()), sd=float(p[m].std(ddof=1)), crisprhal_mean=float(p[h].mean()), crisprhal_sd=float(p[h].std(ddof=1)),
                             lead_mean=float(diff.mean()), lead_sd=float(diff.std(ddof=1)), folds_ahead=int((diff > 0).sum())))
    r = pd.DataFrame(rows)
    r.to_csv(GG_RESULT, index=False)
    print(p.round(4).to_string())
    print(r.round(4).to_string(index=False))


# ------------------------------------------------------------------ the designated final evaluation: crisprHAL 2's own eSpCas9 files, both models, one table
FINAL_OUT = OUT / "final_eval_espcas9.csv"
PRED = config.RESULTS / "preds_final_espcas9_{arm}.csv"
HAL_DIR = config.EXTERNAL / "crisprHAL"
HAL_DATA = HAL_DIR / "data"
MODEL = "ESPCAS9"
PROTO_OFFSET = 193                            # their 406 nt context: 193 up
CACHE = config.INTERIM / "transfer_eSpCas9"
FINAL_ORDER = ("slicer", "crisprhal", "difference")


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


def _final_metrics(y, pred) -> dict:
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


def _final_append(row: dict):
    df = pd.DataFrame([row])
    if FINAL_OUT.exists() and FINAL_OUT.stat().st_size:
        prev = pd.read_csv(FINAL_OUT, float_precision="round_trip")          # the other arms' rows keep their exact digits
        prev = prev[prev["arm"] != row["arm"]]
        df = pd.concat([prev, df], ignore_index=True)
    df = df.iloc[np.argsort([FINAL_ORDER.index(a) for a in df['arm']], kind='stable')]
    df.to_csv(FINAL_OUT, index=False)
    print(f"\n{df.round(4).to_string(index=False)}\nwritten -> {FINAL_OUT}")


def final_slicer(verbose: bool = True) -> dict:
    """Champion XGBoost, trained on their training rows only."""
    import xgboost as xgb

    from ..harness.run_ablation import _impute, champion_params, make_selector

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
               n_train=int(len(i_tr)), **_final_metrics(y_te, pred),
               seconds=round(time.time() - t0, 1))
    _final_append(row)
    return row


def hal_espcas9(verbose: bool = True) -> dict:
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
               **_final_metrics(te["score"].to_numpy(float), pred),
               seconds=round(time.time() - t0, 1))
    _final_append(row)
    return row


def final_compare(margin: float = 0.01, n_boot: int = 4000, seed: int = 0,
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
    _final_append(row)
    if verbose:
        print(f"\n  Delta rho = {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"n = {n:,}")
        print(f"  equivalent within +/-{margin}: {equivalent}")
    return row



def hal_wt() -> None:
    """crisprHAL 2's shipped WT-SpCas9 weights on their own WT test file, saved for a paired comparison (TensorFlow environment).

    The WT model takes a 378-nt context (189 nt upstream, 20-nt protospacer, 3-nt PAM, 166 nt downstream); the protospacer starts at offset 189. Guides scored more than once are averaged by protospacer, as in `hal_espcas9` for eSpCas9."""
    HAL = str(HAL_DIR)
    d = pd.read_csv(os.path.join(HAL, "data", "WT-SpCas9_testing_data.csv"), header=None, names=["context", "score"]).dropna(subset=["context"])
    d["context"] = d["context"].str.upper()
    assert (d["context"].str.len() == 378).all(), d["context"].str.len().value_counts()
    d["protospacer"] = d["context"].str[189:209]
    assert d["context"].str[210:212].eq("GG").mean() > 0.99, "protospacer offset is wrong: PAM is not NGG"
    d = d[~d["protospacer"].str.contains("N")]
    d = d.groupby("protospacer", as_index=False).agg(context=("context", "first"), score=("score", "mean"))
    sys.path.insert(0, HAL)
    cwd = os.getcwd()
    os.chdir(HAL)                                       # their module paths are repo-relative
    try:
        import keras
        from processing import processing
        model = keras.models.load_model("models/WT-SPCAS9.keras", compile=False)
    finally:
        os.chdir(cwd)
    X = processing().onehotencode(d["context"].to_numpy())
    pred = np.asarray(model.predict(X, verbose=0)).ravel()
    assert np.isfinite(pred).all() and pred.std() > 0
    out = config.RESULTS / "preds_final_wt_crisprhal.csv"
    pd.DataFrame({"protospacer": d["protospacer"].to_numpy(), "y": d["score"].to_numpy(float), "pred": pred}).to_csv(out, index=False)
    print(len(d), "guides; Spearman of their shipped WT weights on their WT test file:", round(float(spearmanr(d["score"], pred).statistic), 4))


# ------------------------------------------------------------------ transferability: across nucleases and across kingdoms
TRANSFER_WORK = INTERIM / "transfer_vs_crisprhal"
UP, DOWN = 189, 166


def transfer_prepare() -> None:
    from scipy.stats import spearmanr

    from ..build.base_fast import matrix_fast
    from ..harness.curated import load_base
    from . import eukaryote_rules as ER
    from .transfer import build, load_dataset

    TRANSFER_WORK.mkdir(parents=True, exist_ok=True)
    # ---- nuclease: eSpCas9 guides, their 378 nt contexts, and SLICER trained on the SpCas9 training file
    d = load_dataset("eSpCas9", verbose=False)
    assert d.upstream.str.len().min() >= UP and d.downstream.str.len().min() >= DOWN
    ctx = d.upstream.str[-UP:] + d.protospacer + d.pam + d.downstream.str[:DOWN]
    assert (ctx.str.len() == UP + 23 + DOWN).all()
    wt_all = load_dataset("WT-SpCas9", verbose=False)
    wt_tr = pd.read_csv(HAL_DIR / "data" / "WT-SpCas9_training_data.csv", header=None, names=["context", "score"]).dropna()
    wt_te = pd.read_csv(HAL_DIR / "data" / "WT-SpCas9_testing_data.csv", header=None, names=["context", "score"]).dropna()
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
    out.to_csv(TRANSFER_WORK / "esp_slicer.csv", index=False)
    pd.DataFrame({"protospacer": pe, "context378": ctx.reindex(range(len(d))).to_numpy()[pd.Index(d.protospacer).get_indexer(pe)]}).to_csv(TRANSFER_WORK / "esp_contexts.csv", index=False)
    print(f"  eSpCas9: {len(out):,} guides, SLICER trained on {len(tr):,} SpCas9 training-file guides ({secs})", flush=True)
    # ---- kingdom: E. coli 23-mers for training, eukaryotic 23-mers and SLICER base predictions
    D = ER.load_all()
    ec = ER.ecoli()
    from ..build.build_features import load_curated_index
    idx = load_curated_index(verbose=False).set_index(config.ID_COL).reindex(ec["protos"])
    pd.DataFrame({"seq23": (idx.protospacer + idx.pam).to_numpy(), "y": ec["y"]}).to_csv(TRANSFER_WORK / "ecoli_23.csv", index=False)
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
    pd.concat(rows).to_csv(TRANSFER_WORK / "euk_23.csv", index=False)
    pd.DataFrame(slicer_rows).to_csv(TRANSFER_WORK / "euk_slicer.csv", index=False)
    print(f"  eukaryotic tables: {len(big)} datasets, {sum(D[k]['n'] for k in big):,} guides", flush=True)


def transfer_tf() -> None:
    """crisprHAL 2: WT weights on the eSpCas9 contexts; the architecture retrained on the 23-nt E. coli input and scored on the eukaryotic 23-mers."""
    from scipy.stats import spearmanr

    sys.path.insert(0, str(HAL_DIR))
    cwd = os.getcwd()
    os.chdir(HAL_DIR)
    import keras
    import models as hal_models
    from models import models
    from processing import processing
    os.chdir(cwd)
    import tensorflow as tf_
    proc = processing()
    c = pd.read_csv(TRANSFER_WORK / "esp_contexts.csv")
    os.chdir(HAL_DIR)
    wt = keras.models.load_model("models/WT-SPCAS9.keras", compile=False)
    os.chdir(cwd)
    p = np.asarray(wt.predict(proc.onehotencode(c.context378.str.upper().to_numpy()), batch_size=512, verbose=0)).ravel()
    pd.DataFrame({"protospacer": c.protospacer, "crisprhal_wt": p}).to_csv(TRANSFER_WORK / "esp_crisprhal.csv", index=False)
    print("  WT weights on eSpCas9 contexts:", len(p), flush=True)
    ec = pd.read_csv(TRANSFER_WORK / "ecoli_23.csv")
    euk = pd.read_csv(TRANSFER_WORK / "euk_23.csv")
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
    pd.DataFrame(rows).to_csv(TRANSFER_WORK / "euk_crisprhal.csv", index=False)


def _transfer_boot(y, a, b, B=2000, seed=0):
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    d = np.empty(B)
    for i in range(B):
        k = rng.integers(0, len(y), len(y))
        d[i] = spearmanr(y[k], a[k]).statistic - spearmanr(y[k], b[k]).statistic
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def transfer_report() -> None:
    from scipy.stats import spearmanr

    e = pd.read_csv(TRANSFER_WORK / "esp_slicer.csv").merge(pd.read_csv(TRANSFER_WORK / "esp_crisprhal.csv"), on="protospacer")
    rows = []
    for name, m in (("all eSpCas9 guides", np.ones(len(e), bool)), ("guides absent from the SpCas9 screen", ~e.in_spcas9_screen.to_numpy()),
                    ("guides also in the SpCas9 screen", e.in_spcas9_screen.to_numpy())):
        s = e[m]
        y = s.y.to_numpy()
        for arm in ("slicer_default", "slicer_tuned", "slicer_base_only"):
            lo, hi = _transfer_boot(y, s[arm].to_numpy(), s.crisprhal_wt.to_numpy())
            rows.append(dict(part="nuclease", subset=name, n=int(m.sum()), model=arm, rho=float(spearmanr(y, s[arm]).statistic),
                             crisprhal=float(spearmanr(y, s.crisprhal_wt).statistic), diff=float(spearmanr(y, s[arm]).statistic - spearmanr(y, s.crisprhal_wt).statistic),
                             lo=lo, hi=hi))
    k = pd.read_csv(TRANSFER_WORK / "euk_slicer.csv").merge(pd.read_csv(TRANSFER_WORK / "euk_crisprhal.csv"), on="dataset")
    own = pd.read_csv(OUT / "eukaryote_transfer.csv")
    own = own[(own.model.str.startswith("pipeline")) & own.within].set_index("scored_on").rho
    k["own_cv_rho"] = k.dataset.map(own)
    k["slicer_retained"] = k.slicer_base / k.own_cv_rho
    k["crisprhal_retained"] = k.crisprhal_core23 / k.own_cv_rho
    k.to_csv(OUT / "transfer_vs_crisprhal_kingdom.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "transfer_vs_crisprhal_nuclease.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(k.round(3).to_string(index=False))
    print("median rho: SLICER base", round(k.slicer_base.median(), 4), " crisprHAL 2 (23 nt)", round(k.crisprhal_core23.median(), 4))


CONDITIONS = ["espcas9", "wt", "folds", "paired", "confirm", "gene-folds", "gene-assemble", "final-slicer", "hal-espcas9", "final-compare", "hal-wt", "transfer-prepare", "transfer-tf", "transfer-report"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--mix", default="", help="groups of member names: g1a,g1b;g2a,g2b (each group averaged, then the groups)")
    ap.add_argument("--margin", type=float, default=0.01, help="final-compare: the equivalence margin, declared before the interval was seen")
    a = ap.parse_args()
    if a.condition in ("paired", "confirm", "gene-folds", "gene-assemble"):
        {"paired": paired, "confirm": confirm, "gene-folds": gene_folds, "gene-assemble": gene_assemble}[a.condition]()
        return
    if a.condition in ("final-slicer", "hal-espcas9", "final-compare", "hal-wt", "transfer-prepare", "transfer-tf", "transfer-report"):
        {"final-slicer": final_slicer, "hal-espcas9": hal_espcas9, "final-compare": lambda: final_compare(margin=a.margin), "hal-wt": hal_wt,
         "transfer-prepare": transfer_prepare, "transfer-tf": transfer_tf, "transfer-report": transfer_report}[a.condition]()
        return
    mix = [[m for m in grp.split(",") if m] for grp in a.mix.split(";") if grp]
    xgb5 = ["d099x", "d077x", "d021x", "d080x", "d000x"]
    families = {"champion": ["champion"], "tuned XGBoost (d099x)": ["d099x"], "XGBoost average of 5": xgb5}
    if mix:
        families["mixed ensemble"] = mix
    needed = sorted({n for v in families.values() for n in _flat(v)})
    specs = member_specs(needed)
    rows = {"espcas9": espcas9, "wt": wt, "folds": folds}[a.condition](families, specs)
    f = OUT / f"vs_crisprhal_{a.condition}.csv"
    pd.DataFrame(rows).to_csv(f, index=False)


if __name__ == "__main__":
    main()
