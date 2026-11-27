"""SLICER against crisprHAL 2 under every protocol of Part 5: the tuned model leads by +0.0226 on crisprHAL 2's SpCas9 split and +0.0286 on its eSpCas9 split; paired on guides on our folds it leads by +0.0115 untuned and +0.0043 against crisprHAL 2's best settings; and the same comparisons are made on genes the tuning never saw and on whole-gene folds.

Seven conditions (`--condition`), one result file each:

  espcas9        crisprHAL's own shipped eSpCas9 split (47,591 train / 11,898 test), against their shipped weights (predictions stored by `final_eval.py`); paired bootstrap on guides        vs_crisprhal_espcas9.csv
  wt             crisprHAL's own shipped SpCas9 split (26,830 train / 6,737 test), against their shipped WT weights (predictions made by `vs_crisprhal_wt_hal.py` in the TensorFlow environment)  vs_crisprhal_wt.csv
  folds          our five curated folds (KFold 5, seed 41, 33,567 guides), against crisprHAL 2's stored out-of-fold predictions at its published configuration and at the best configuration found  vs_crisprhal_folds.csv
  paired         the same folds with the difference bootstrapped paired on guides: three pairings (untuned, tuned, cross)                                                                 heldout_compare.csv
  confirm        the 30% of genes held out from every search, where nothing in the comparison was tuned                                                                                   clean_vs_hal.csv
  gene-folds     SLICER's default and tuned models on folds that hold whole genes out (also writes the input file of crisprHAL's Colab run)                                                  gene_grouped_slicer.csv
  gene-assemble  crisprHAL 2's per-fold rho from that Colab run, paired against SLICER's on the same folds                                                                              gene_grouped_vs_crisprhal.csv

Each SLICER arm trains on the training rows only, with column selection fitted inside them, exactly as in `final_eval.py`. Arms: the frozen champion (reproduces the earlier figures), the tuned XGBoost (d099x), the XGBoost average of five, and the mixed
ensemble chosen on held-out genes in `mix_ecoli.py` (its members are read from `--mix`). Every comparison on crisprHAL 2's two splits starts from `hal_split`.

    python -m sgrna.compare.vs_crisprhal --condition espcas9 --mix "d099x,d077x,d021x,d080x,d000x;lgb011,lgb018;cat015,cat023"
    python -m sgrna.compare.vs_crisprhal --condition paired        # or: confirm, gene-folds, gene-assemble

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
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from sgrna import config
from sgrna.harness.run_ablation import _impute, make_selector

from ..tuning import mix_ecoli as M
from ..tuning import tune_ecoli as T
from ..common import MAIN_RESULTS, OUT, stored
from ..harness.metrics import zscore as _z

SRC = MAIN_RESULTS                           # the main project's results folder (stored crisprHAL predictions)


def member_specs(names):
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


def paired(y, ps, ph, n_boot=4000, seed=0):
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
        from sgrna.compare.final_eval import CACHE, _split_frames
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
        d, lo, hi = paired(y, p, ph)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--condition", choices=["espcas9", "wt", "folds", "paired", "confirm", "gene-folds", "gene-assemble"], required=True)
    ap.add_argument("--mix", default="", help="groups of member names: g1a,g1b;g2a,g2b (each group averaged, then the groups)")
    a = ap.parse_args()
    if a.condition in ("paired", "confirm", "gene-folds", "gene-assemble"):
        {"paired": paired, "confirm": confirm, "gene-folds": gene_folds, "gene-assemble": gene_assemble}[a.condition]()
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
