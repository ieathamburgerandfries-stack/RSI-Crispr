"""SLICER is more faithful than crisprHAL 2 (sensitivity-n +0.081 to +0.111) but ties or splits on stability, truth and compressibility, so it is not shown to be more interpretable. The same readings, taken the same way, from both models.

"Interpretable" is not one quantity, so this module does not score it. It asks four questions that a person who wants to learn
something from a fitted model needs answered, defines each as a measurement that either model can be put through unchanged, and reports
each separately (Murdoch et al. 2019 call the first three descriptive accuracy and the fourth relevance; Doshi-Velez & Kim 2017 call this
functionally grounded evaluation, because no human is in the loop). Predictions and the decision rule are written here before any of the
crisprHAL 2 numbers exist.

The models, both as released, on crisprHAL 2's own WT-SpCas9 split (their 26,994 training rows, their 6,749 test rows):
  crisprHAL 2   their shipped WT-SPCAS9 weights (TensorFlow, 378 nt of context, 790,533 parameters); refits use their code and 48 epochs.
  SLICER        the tuned model as shipped (d099x: base + family A, 450 gain-ranked columns, 4,000 trees), fitted on the same training rows.
  SLICER-104    the compact readable counterpart (80 position x base indicators + 24 flank composition columns), reported where the question
                is about position x base, because the shipped model's columns are descriptors and not positions.
Attributions are the standard ones for each class: TreeSHAP for the trees, expected gradients (Erion et al. 2021, the estimator behind
SHAP's GradientExplainer) for the network. 1,000 test guides drawn at random, the same for both.

Q1  FAITHFUL. Does the attribution describe the model it was computed from? Sensitivity-n (Ancona et al. 2018) at block level: for random
    subsets of blocks, the summed attribution against the prediction lost when those blocks are replaced by values from background guides
    (the protocol of Finding 5, applied to both). Control: the same attributions with the blocks shuffled.
Q2  STABLE. Does the reading survive refitting? Attributions from independently fitted models of the same class, compared as importance
    vectors (Spearman): three fits on the full training rows (different seeds), and two disjoint random halves of the training rows, twice.
    Unit: the 80 guide position x base channels (the finest unit both models share) and 25 sequence blocks. Control: the same model, two
    Monte Carlo draws of the estimator (a noise floor for expected gradients; TreeSHAP is deterministic).
Q3  TRUE. Does the reading recover what the data show without any model? The model's effect of each base at each guide position
    (attribution read-out, and the substitution map from editing the base in silico) against the model-free position x base map of
    Finding 7 (`mechanism_position_base.csv`), Spearman and Pearson over 80 cells.
Q4  COMPRESSIBLE. How much of what the model does is reproduced by a description a person can hold? A surrogate on the compact readable
    columns (80 position x base + 24 flank composition) is fitted, by gene-grouped cross-validation, to each model's own out-of-fold
    predictions on the 30,203 genic guides; fidelity is Spearman(surrogate, model) and "retained" is the surrogate's Spearman with the label
    divided by the model's. The same surrogate class for both, so the model that is easier to describe scores higher.
Also reported, because it is what the flank claims rest on: the response of each model to the flank-edit tests of Finding 5 (GC +/-5 and
+/-10 points, shuffled order, bases swapped at fixed GC), applied to the DNA each model sees (1 kb for SLICER, 189 nt up and 169 nt down for
crisprHAL 2).

Predictions, written before the run:
  P1  Q1. SLICER's block-level sensitivity-n is at least 0.9 (it was 0.956 on the random-split fit); crisprHAL 2's is lower by at least
      0.03 (probability about 0.5; gradient attributions through a recurrent layer with 20-nt block ablations are the weak case).
  P2  Q2. On the 80 guide channels crisprHAL 2's agreement between refits is 0.8 or more (probability about 0.7), because its input has no
      restated columns. If it is, the instability in Finding 5 is redundancy, not model class, and "SLICER is more stable" is not a claim
      the data will support at that unit. On SLICER's own 450 selected columns it is lower than crisprHAL 2's (probability about 0.8).
  P3  Q3. Both models' read-outs track the model-free map at Spearman 0.8 or more (probability about 0.7), within 0.1 of each other.
  P4  Q4. No difference: the two fidelities are within 0.03 (probability about 0.5); crisprHAL 2 is the better-described (about 0.3).
  P5  Flank. crisprHAL 2's prediction moves in the same direction as SLICER's under GC edits (lower GC, lower prediction; probability
      about 0.8), by less than SLICER's -0.88 at GC -5 (about 0.7), because it sees 189 nt and not 1 kb.

Decision rule. SLICER is called more interpretable than crisprHAL 2 on a question only if the difference excludes zero in SLICER's
favour: a bootstrap over guides for Q1 (the 100 guides are the same for both), over the 80 cells for Q3 and over guides for Q4; for Q2 there
are five refit pairs, too few and too dependent for a bootstrap, so the rule is that the ranges of the pairs do not overlap. The overall
statement is whatever that list supports; there is no composite score, and a question on which crisprHAL 2 wins is reported as such.
Q2 and Q3 are read on two units. The attribution readings are each class's own (TreeSHAP columns; expected-gradient channels) and are not
available at the same unit for the shipped SLICER, whose columns are descriptors; so the unit common to all three models is the
in-silico substitution map: the mean change in a model's prediction when one guide base is set to each base, over the same guides. It needs
no attribution method. For the shipped SLICER the edited 20-mer is re-featurised (`base_fast.matrix_fast`, the generator the 6,169 base
columns came from), so the shipped model is read at the same unit as the network.

    python -m sgrna.interpret.vs_crisprhal_interpret --step <hal_collect|slicer|readable|assemble>
"""
from __future__ import annotations

import argparse
import os
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sgrna import config

from ..common import MAIN_RESULTS, OUT
from .importance import block_of

BASES = "ACGT"
CACHE = Path(os.environ.get("HAL_CACHE") or (OUT / "interp_hal_cache"))
SEEDS = {"shipped": 41, "full1": 101, "full2": 102, "halfA1": 121, "halfB1": 122, "halfA2": 131, "halfB2": 132}
SEED_PAIRS = [("shipped", "full1"), ("shipped", "full2"), ("full1", "full2")]
HALF_PAIRS = [("halfA1", "halfB1"), ("halfA2", "halfB2")]
N_ATTR = 1000                # guides read for every model (the first of the shared permutation)
N_FAITH = 100


def _rho(a, b):
    return float(spearmanr(a, b).statistic)


# ----------------------------------------------------------------------------------------------------------- the data
def _wt_split():
    """The head-to-head's WT condition: curated rows, crisprHAL's own test partition held out. Also the 1,500 shared guides."""
    from sgrna.cohorts.data_ablation import load
    X, y, names, protos, in_test, _, _, n_base = load(verbose=False)
    protos = np.asarray(protos)
    g = pd.read_csv(CACHE / "guides_1500.csv")
    pos = pd.Index(protos).get_indexer(g["protospacer"])
    assert (pos >= 0).all(), "a shared guide is missing from the curated matrix"
    assert in_test[pos].all(), "a shared guide is not in crisprHAL's test partition"
    return dict(X=X, y=y, names=list(names), protos=protos, in_test=np.asarray(in_test), n_base=n_base, pos=pos)


def _train_rows(D, tag):
    """Curated training rows for one fit: all of them, or the guides of one random half of crisprHAL's training file (the split `hal_readout` makes)."""
    tr = np.flatnonzero(~D["in_test"])
    if tag in ("shipped", "full1", "full2"):
        return tr
    f = pd.read_csv(config.CRISPRHAL / "WT-SpCas9_training_data.csv", header=None, names=["context", "y"]).dropna(subset=["context"])
    n = len(f)
    k = int(tag[5:])
    perm = np.random.default_rng(10 + k).permutation(n)
    half = perm[: n // 2] if tag[4] == "A" else perm[n // 2:]
    ps = set(f["context"].str.upper().str[189:209].iloc[half])
    return tr[np.isin(D["protos"][tr], list(ps))]


# --------------------------------------------------------------------------------------- SLICER as shipped (d099x, full matrix)
def fit_full(D, tr, seed):
    import xgboost as xgb
    from sgrna.harness.run_ablation import _impute, make_selector
    from ..compare.vs_crisprhal import member_specs
    from ..tuning import tune_ecoli as T
    kind, _, k, cfg = member_specs(["d099x"])[0]
    cfg = dict(cfg); cfg["n_estimators"] = k
    X, y = D["X"], D["y"]
    A, _ = _impute(X[tr], X[tr][:1])
    sel = make_selector(seed); sel.fit(A, y[tr]); del A
    top = np.argsort(sel.feature_importances_)[::-1][:900]
    cols = top[: int(cfg["nf"])]
    med = np.nanmedian(X[tr][:, cols], axis=0); med = np.where(np.isnan(med), 0.0, med)

    def prep(Xr):
        Z = np.asarray(Xr[:, cols], np.float32).copy()
        r, c = np.where(np.isnan(Z)); Z[r, c] = med[c]
        return Z
    m = xgb.XGBRegressor(**T.xgb_params(cfg, seed)).fit(prep(X[tr]), T._target(y[tr], cfg["target"]))
    return dict(m=m, cols=cols, prep=prep, med=med)


def _ism_full(D, f, rows):
    """In-silico substitution of every guide base, for the shipped model: re-featurise the edited 20-mer, keep family A."""
    from sgrna.build import base_fast
    X, protos, nb = D["X"], D["protos"], D["n_base"]
    f0 = f["m"].predict(f["prep"](X[rows]))
    out = np.zeros((len(rows), 20, 4), np.float32)
    for p in range(20):
        for b in range(4):
            mut = [s[:p] + BASES[b] + s[p + 1:] for s in protos[rows]]
            M, nm = base_fast.matrix_fast(mut)
            if p == 0 and b == 0:
                assert list(nm) == D["names"][:nb], "regenerated column names differ from the matrix"
            Xv = X[rows].copy(); Xv[:, :nb] = M
            out[:, p, b] = f["m"].predict(f["prep"](Xv)) - f0
    same = np.array([[BASES.index(s[p]) for p in range(20)] for s in protos[rows]])
    unchanged = np.stack([out[np.arange(len(rows)), p, same[:, p]] for p in range(20)], 1)
    assert np.abs(unchanged).max() < 1e-4, "re-featurising the unedited 20-mer does not reproduce the stored matrix"
    return out


def read_full(D, tag):
    import shap
    t0 = time.time()
    tr = _train_rows(D, tag)
    f = fit_full(D, tr, SEEDS[tag])
    rows = D["pos"][:N_ATTR]
    te = np.flatnonzero(D["in_test"])
    rho = _rho(D["y"][te], f["m"].predict(f["prep"](D["X"][te])))
    phi = np.asarray(shap.TreeExplainer(f["m"]).shap_values(f["prep"](D["X"][rows]), check_additivity=False))
    imp = np.zeros(D["X"].shape[1]); imp[f["cols"]] = np.abs(phi).mean(0)
    names = np.asarray(D["names"])
    blk = np.array([block_of(n) for n in names])
    ub = sorted(set(blk))
    block_imp = np.array([imp[blk == u].sum() for u in ub])
    delta = _ism_full(D, f, rows)
    np.savez_compressed(CACHE / f"slicer_full_{tag}.npz", imp=imp, block_imp=block_imp, blocks=np.array(ub), ism_delta=delta, ism_map=delta.mean(0),
                        n_train=len(tr), test_rho=rho, cols=f["cols"], pred_test=f["m"].predict(f["prep"](D["X"][te])), test_rows=te)
    print(f"  SLICER full {tag}: {len(tr):,} rows, test rho {rho:.4f}, {time.time() - t0:.0f} s", flush=True)
    return f, phi


def sens_n(predict, X, attr, blocks, bgX, seed=41, n_rows=N_FAITH, n_bg=50, ks=(1, 2, 3, 4, 6), label="TreeSHAP"):
    """Sensitivity-n at block level (`interp_extra.sensitivity_n`): summed attribution against the prediction lost when the blocks are replaced by background rows."""
    rng = np.random.default_rng(seed)
    ub = sorted(set(blocks)); idx = [np.where(blocks == u)[0] for u in ub]
    X = X[:n_rows]; attr = attr[:n_rows]
    bg = bgX[rng.choice(len(bgX), n_bg, replace=False)]
    fx = predict(X)
    battr = np.stack([attr[:, i].sum(1) for i in idx], 1)
    bperm = np.stack([r[rng.permutation(r.size)] for r in battr])
    out = []
    for k in ks:
        subsets = [rng.choice(len(ub), k, replace=False) for _ in range(40)]
        dlt = np.zeros((len(X), len(subsets)))
        for j, s in enumerate(subsets):
            cols = np.concatenate([idx[i] for i in s])
            acc = np.zeros(len(X))
            for b in bg:
                Z = X.copy(); Z[:, cols] = b[cols]
                acc += predict(Z)
            dlt[:, j] = fx - acc / n_bg
        for lab, a_ in ((label, battr), (label + ", blocks shuffled", bperm)):
            a = np.stack([a_[:, s].sum(1) for s in subsets], 1)
            r = [np.corrcoef(a[i], dlt[i])[0, 1] for i in range(len(a)) if a[i].std() > 0 and dlt[i].std() > 0]
            out.append(dict(unit="block", n=k, n_blocks=len(ub), attribution=lab, pearson=float(np.mean(r)), se=float(np.std(r) / np.sqrt(len(r))),
                            subsets=len(subsets), rows=len(X), per_row=r))
        print(f"    sens-n k={k}/{len(ub)} blocks: " + "  ".join(f"{o['attribution'][:22]} {o['pearson']:+.3f}" for o in out[-2:]), flush=True)
    return out


def flank_edits_full(D, f, y_sd):
    """The flank-edit tests of Finding 5 on the same 1,500 guides and this fit (family A rebuilt from the edited DNA)."""
    from ..build.build_features import load_curated_index
    from ..features import a_flank
    from .interp_extra import _edit
    X, protos, names, nb = D["X"], D["protos"], D["names"], D["n_base"]
    pick = D["pos"]
    idx = load_curated_index(verbose=False).drop_duplicates("protospacer").set_index("protospacer").reindex(protos[pick]).reset_index()
    idx = idx.rename(columns={"index": "protospacer"}); idx[config.ID_COL] = idx["protospacer"]; idx["located"] = True
    a_names = names[nb:]

    def predict(index):
        blk = a_flank.build(index).set_index(config.ID_COL).reindex(index[config.ID_COL])
        Xr = X[pick].copy()
        for j, c in enumerate(a_names):
            if c in blk.columns:
                Xr[:, nb + j] = blk[c].to_numpy(dtype=np.float32)
        return f["m"].predict(f["prep"](Xr))
    p0 = predict(idx)
    chk = float(np.mean(np.abs(p0 - f["m"].predict(f["prep"](X[pick])))))
    print(f"    unedited rebuild vs stored matrix: mean |diff| {chk:.5f}", flush=True)
    rows = []
    for name, kind, frac in (("GC +5 points, both flanks", "gc_up", 0.05), ("GC +10 points, both flanks", "gc_up", 0.10),
                             ("GC -5 points, both flanks", "gc_down", 0.05), ("GC -10 points, both flanks", "gc_down", 0.10),
                             ("order shuffled, composition kept", "shuffle", 0.0), ("5% of bases swapped, GC kept", "swap_same_gc", 0.05),
                             ("20% of bases swapped, GC kept", "swap_same_gc", 0.20)):
        r2 = np.random.default_rng(zlib.crc32(name.encode()))
        e = idx.copy()
        e["upstream"] = [_edit(s, kind, r2, frac, "up") for s in idx["upstream"]]
        e["downstream"] = [_edit(s, kind, r2, frac, "down") for s in idx["downstream"]]
        d = predict(e) - p0
        means = d[np.random.default_rng(1).choice(len(d), (2000, len(d)))].mean(1)
        rows.append(dict(model="SLICER (shipped)", edit=name, mean_change=float(d.mean()), ci_lo=float(np.percentile(means, 2.5)),
                         ci_hi=float(np.percentile(means, 97.5)), sd_across_guides=float(d.std()), guides=len(d), label_sd=y_sd, unedited_rebuild_error=chk))
        print(f"    {name:36s} {d.mean():+.4f}  [{rows[-1]['ci_lo']:+.4f}, {rows[-1]['ci_hi']:+.4f}]", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- SLICER-104, the readable counterpart (same split)
def _compact(D):
    """The 104-column representation on the genic guides, with the WT-split membership of each row."""
    from .compact_model import _data
    d = _data()
    prot = pd.Index(np.asarray(d["protos"]))
    test_set = set(D["protos"][D["in_test"]])
    d["in_test"] = np.array([p in test_set for p in prot])
    d["row_of_guide"] = pd.Index(D["protos"]).get_indexer(prot)       # row in the full matrix
    return d


def _compact_xy(d):
    return np.hstack([d["onehot"], d["comp"]]).astype(np.float32)


def read_compact(D, d, tag):
    import shap
    from .compact_model import _blocks, _models_xgb
    t0 = time.time()
    X = _compact_xy(d); y = d["y"]
    tr_all = _train_rows(D, tag)
    tr = np.flatnonzero(np.isin(d["row_of_guide"], tr_all))
    te = np.flatnonzero(d["in_test"])
    m = _models_xgb(SEEDS[tag]).fit(X[tr], y[tr])
    rho = _rho(y[te], m.predict(X[te]))
    sel = np.flatnonzero(np.isin(d["row_of_guide"], D["pos"][:N_ATTR]))
    phi = np.asarray(shap.TreeExplainer(m).shap_values(X[sel], check_additivity=False))
    imp = np.abs(phi).mean(0)
    oh = d["onehot"][sel]
    pos_base = np.full((20, 4), np.nan)
    for p in range(20):
        pp = phi[:, p * 4:(p + 1) * 4].sum(1)
        for b in range(4):
            s = oh[:, p * 4 + b] == 1
            if s.sum() >= 20:
                pos_base[p, b] = pp[s].mean()
    f0 = m.predict(X[sel])
    delta = np.zeros((len(sel), 20, 4), np.float32)
    for p in range(20):
        for b in range(4):
            Z = X[sel].copy(); Z[:, p * 4:(p + 1) * 4] = 0; Z[:, p * 4 + b] = 1
            delta[:, p, b] = m.predict(Z) - f0
    np.savez_compressed(CACHE / f"slicer_compact_{tag}.npz", imp=imp, pos_base=pos_base, ism_delta=delta, ism_map=delta.mean(0), test_rho=rho, n_train=len(tr),
                        sel_rows=sel, pred_test=m.predict(X[te]), test_idx=te, blocks=_blocks(d))
    print(f"  SLICER-104 {tag}: {len(tr):,} rows, test rho {rho:.4f}, {time.time() - t0:.0f} s", flush=True)
    return m, X, phi, sel, _blocks(d)


def step_slicer(tags):
    CACHE.mkdir(parents=True, exist_ok=True)
    D = _wt_split(); d = _compact(D)
    print(f"  WT split: {int((~D['in_test']).sum()):,} training rows, {int(D['in_test'].sum()):,} test rows; compact representation {len(d['y']):,} genic guides", flush=True)
    for tag in tags:
        if not (CACHE / f"slicer_full_{tag}.npz").exists():
            f, phi = read_full(D, tag)
            if tag == "shipped":
                blocks = np.array([block_of(D["names"][c]) for c in f["cols"]])
                rng = np.random.default_rng(0)
                tr = _train_rows(D, tag)
                bg = f["prep"](D["X"][rng.choice(tr, 200, replace=False)])
                rows = D["pos"][:N_ATTR]
                Xr = f["prep"](D["X"][rows])
                res = sens_n(f["m"].predict, Xr, phi, blocks, bg, label="TreeSHAP")
                pd.DataFrame([{k: v for k, v in r.items() if k != "per_row"} for r in res]).assign(model="SLICER (shipped)").to_csv(CACHE / "slicer_full_sensitivity_n.csv", index=False)
                flank_edits_full(D, f, float(D["y"].std())).to_csv(CACHE / "slicer_flank_edits.csv", index=False)
        if not (CACHE / f"slicer_compact_{tag}.npz").exists():
            m, X, phi, sel, blk = read_compact(D, d, tag)
            if tag == "shipped":
                rng = np.random.default_rng(0)
                tr = np.flatnonzero(np.isin(d["row_of_guide"], _train_rows(D, tag)))
                bg = X[rng.choice(tr, 200, replace=False)]
                res = sens_n(m.predict, X[sel], phi, blk, bg, label="TreeSHAP")
                pd.DataFrame([{k: v for k, v in r.items() if k != "per_row"} for r in res]).assign(model="SLICER-104").to_csv(CACHE / "slicer_compact_sensitivity_n.csv", index=False)


# ------------------------------------------------------------------------------------------------------ Q4: compressible
def _readable_inputs(population):
    """(compact columns, label, gene, crisprHAL 2 prediction, SLICER prediction) for one population of guides."""
    from sgrna.compare.vs_crisprhal import member_specs, train_predict
    from sklearn.model_selection import KFold
    from ..common import stored
    D = _wt_split(); d = _compact(D)
    Xc = _compact_xy(d)
    if population == "WT test partition (shipped models)":
        hal = pd.read_csv(MAIN_RESULTS / "preds_final_wt_crisprhal.csv").drop_duplicates("protospacer").set_index("protospacer")
        z = np.load(CACHE / "slicer_full_shipped.npz")
        sl = pd.Series(z["pred_test"], index=D["protos"][z["test_rows"]])
        prot = pd.Index(np.asarray(d["protos"]))
        keep = np.flatnonzero(d["in_test"] & prot.isin(hal.index) & prot.isin(sl.index) & ~prot.duplicated())
        return Xc[keep], d["y"][keep], np.asarray(d["gene"])[keep].astype(str), hal.loc[prot[keep], "pred"].to_numpy(), sl.loc[prot[keep]].to_numpy()
    # five curated folds: out-of-fold predictions of the tuned SLICER (fitted here) and crisprHAL 2's stored out-of-fold predictions (published configuration)
    cache = CACHE / "slicer_oof_folds.npz"
    X, y = D["X"], D["y"]
    folds = list(KFold(5, shuffle=True, random_state=41).split(np.arange(len(y))))
    if cache.exists():
        oof = np.load(cache)["oof"]
    else:
        spec = member_specs(["d099x"]); oof = np.zeros(len(y))
        for tr, va in folds:
            oof[va] = train_predict(spec, X[tr], y[tr], X[va], 41)[0]["d099x"]
        np.savez_compressed(cache, oof=oof)
    order = np.concatenate([va for _, va in folds])
    h = pd.concat([pd.read_csv(f) for f in stored("preds_crisprhal_default_fold*.csv")], ignore_index=True)
    assert np.allclose(y[order], h["y"].to_numpy(), atol=1e-9), "fold alignment failed"
    ph = np.empty(len(y)); ph[order] = h["pred"].to_numpy()
    r = d["row_of_guide"]
    return Xc, d["y"], np.asarray(d["gene"]).astype(str), ph[r], oof[r]


def step_readable(n_boot=1000):
    """Fit one readable surrogate class to each model's own predictions, by gene-grouped cross-validation, in two populations."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from .compact_model import _models_xgb
    allrows = []
    for population in ("WT test partition (shipped models)", "five curated folds (out-of-fold)"):
        X, y, g, ph, ps = _readable_inputs(population)
        print(f"  readable, {population}: {len(y):,} genic guides in {len(set(g)):,} genes; rho with label: crisprHAL 2 {_rho(y, ph):.4f}, SLICER {_rho(y, ps):.4f}", flush=True)
        allrows += _readable_one(population, X, y, g, ph, ps, n_boot)
    out = pd.DataFrame(allrows)
    out.to_csv(OUT / "interp_compare_readable.csv", index=False)
    return out


def _readable_one(population, X, y, g, ph, ps, n_boot):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from .compact_model import _models_xgb
    def oof(target, kind, seed=41):
        out = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, target, g):
            if kind == "additive (ridge)":
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
                out[te] = Ridge(alpha=10.0).fit((X[tr] - mu) / sd, target[tr]).predict((X[te] - mu) / sd)
            else:
                out[te] = _models_xgb(seed).fit(X[tr], target[tr]).predict(X[te])
        return out
    zs = lambda v: (v - v.mean()) / v.std()
    rows, store = [], {}
    for kind in ("additive (ridge)", "boosted trees (champion settings)"):
        for name, p in (("crisprHAL 2", ph), ("SLICER", ps)):
            s_p = oof(zs(p), kind)
            s_y = oof(zs(y), kind)
            store[(kind, name)] = (s_p, s_y, p)
            rows.append(dict(population=population, surrogate=kind, model=name, n=len(y), model_rho=_rho(y, p), fidelity=_rho(s_p, p), surrogate_of_model_rho=_rho(y, s_p),
                             retained=_rho(y, s_p) / _rho(y, p), surrogate_of_label_rho=_rho(y, s_y), retained_when_fitted_to_label=_rho(y, s_y) / _rho(y, p)))
            print(f"    {kind:34s} {name:12s} fidelity {rows[-1]['fidelity']:.4f}  retained {rows[-1]['retained']:.3f}  (fitted to label: {rows[-1]['retained_when_fitted_to_label']:.3f})", flush=True)
    rng = np.random.default_rng(0)
    for kind in ("additive (ridge)", "boosted trees (champion settings)"):
        sH, _, pH = store[(kind, "crisprHAL 2")]; sS, _, pS = store[(kind, "SLICER")]
        d_fid, d_ret = [], []
        for _ in range(n_boot):
            k = rng.integers(0, len(y), len(y))
            d_fid.append(_rho(sS[k], pS[k]) - _rho(sH[k], pH[k]))
            d_ret.append(_rho(y[k], sS[k]) / _rho(y[k], pS[k]) - _rho(y[k], sH[k]) / _rho(y[k], pH[k]))
        for r in rows:
            if r["surrogate"] == kind and r["model"] == "SLICER":
                r.update(fidelity_minus_hal=_rho(sS, pS) - _rho(sH, pH), fidelity_lo=float(np.percentile(d_fid, 2.5)), fidelity_hi=float(np.percentile(d_fid, 97.5)),
                         retained_minus_hal=_rho(y, sS) / _rho(y, pS) - _rho(y, sH) / _rho(y, pH), retained_lo=float(np.percentile(d_ret, 2.5)), retained_hi=float(np.percentile(d_ret, 97.5)))
    return rows

# ------------------------------------------------------------------------------------------------------------ assemble
HAL_BLOCKS = {f"g{p + 1}": np.arange(189 + p, 190 + p) for p in range(20)}
HAL_BLOCKS.update({"PAM": np.arange(209, 212), "up_near": np.arange(169, 189), "up_far": np.arange(0, 169), "dn_near": np.arange(212, 229), "dn_far": np.arange(229, 378)})
MODELS = ("crisprHAL 2", "SLICER (shipped)", "SLICER-104")


def _agree(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    return _rho(a[ok], b[ok])


def _z(model, tag):
    f = {"crisprHAL 2": f"hal_{tag}.npz", "SLICER (shipped)": f"slicer_full_{tag}.npz", "SLICER-104": f"slicer_compact_{tag}.npz"}[model]
    return np.load(CACHE / f, allow_pickle=True)


def _units(model, z):
    """The readings of one fitted model, by unit: name -> vector."""
    ism = z["ism_map"].ravel()
    pos = np.abs(z["ism_delta"]).mean((0, 2))
    if model == "crisprHAL 2":
        ch = z["eg_abs_channel"]
        return {"attribution: guide channels (80)": ch[189:209].ravel(), "attribution: blocks (25)": np.array([ch[i].sum() for i in HAL_BLOCKS.values()]),
                "attribution-implied map (80)": z["eg_pos_base"].ravel(), "substitution map (80)": ism, "substitution: position importance (20)": pos}
    if model == "SLICER (shipped)":
        return {"attribution: blocks (12)": z["block_imp"], "substitution map (80)": ism, "substitution: position importance (20)": pos}
    blk = z["blocks"]; imp = z["imp"]
    return {"attribution: guide channels (80)": imp[:80], "attribution: blocks (28)": np.array([imp[blk == u].sum() for u in pd.unique(blk)]),
            "attribution-implied map (80)": z["pos_base"].ravel(), "substitution map (80)": ism, "substitution: position importance (20)": pos}


def stability():
    rows = []
    for model in MODELS:
        zs = {t: _z(model, t) for t in SEEDS if (CACHE / ({"crisprHAL 2": f"hal_{t}.npz", "SLICER (shipped)": f"slicer_full_{t}.npz", "SLICER-104": f"slicer_compact_{t}.npz"}[model])).exists()}
        us = {t: _units(model, z) for t, z in zs.items()}
        for kind, pairs in (("refit seeds", SEED_PAIRS), ("random halves", HALF_PAIRS)):
            pr = [(a, b) for a, b in pairs if a in zs and b in zs]
            if not pr:
                continue
            for unit in us[pr[0][0]]:
                v = [_agree(us[a][unit], us[b][unit]) for a, b in pr]
                rows.append(dict(model=model, unit=unit, comparison=kind, pairs=len(pr), mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), values=";".join(f"{x:.3f}" for x in v)))
            if model == "SLICER (shipped)":
                v = []
                for a, b in pr:
                    ia, ib = zs[a]["imp"], zs[b]["imp"]
                    u = np.union1d(zs[a]["cols"], zs[b]["cols"])
                    v.append(_agree(ia[u], ib[u]))
                rows.append(dict(model=model, unit="attribution: columns (union of the two fits' selected columns)", comparison=kind, pairs=len(pr), mean=float(np.mean(v)),
                                 min=float(np.min(v)), max=float(np.max(v)), values=";".join(f"{x:.3f}" for x in v)))
    z = _z("crisprHAL 2", "shipped")
    rows.append(dict(model="crisprHAL 2", unit="attribution: guide channels (80)", comparison="same model, second Monte Carlo draw (noise floor)", pairs=1,
                     mean=_agree(z["eg_abs_channel"][189:209], z["eg_abs_channel_mc2"][189:209]), min=np.nan, max=np.nan, values=""))
    rows.append(dict(model="crisprHAL 2", unit="attribution-implied map (80)", comparison="same model, second Monte Carlo draw (noise floor)", pairs=1,
                     mean=_agree(z["eg_pos_base"], z["eg_pos_base_mc2"]), min=np.nan, max=np.nan, values=""))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "interp_compare_stability.csv", index=False)
    return out


def validity(n_boot=2000):
    free = pd.read_csv(OUT / "mechanism_position_base.csv").set_index(["position", "base"])["effect_sd"]
    vec = np.array([free.get((p + 1, BASES[b]), np.nan) for p in range(20) for b in range(4)])
    rows, reads = [], {}
    for model in MODELS:
        for tag in SEEDS:
            try:
                z = _z(model, tag)
            except FileNotFoundError:
                continue
            u = _units(model, z)
            for unit in ("substitution map (80)", "attribution-implied map (80)"):
                if unit in u:
                    a = u[unit]; ok = np.isfinite(a) & np.isfinite(vec)
                    rows.append(dict(model=model, fit=tag, readout=unit, spearman=_rho(a[ok], vec[ok]), pearson=float(np.corrcoef(a[ok], vec[ok])[0, 1]), cells=int(ok.sum())))
                    if tag == "shipped":
                        reads[(model, unit)] = (a, ok)
    out = pd.DataFrame(rows)
    # paired bootstrap over cells, shipped fits: crisprHAL 2 against each SLICER on the substitution map
    rng = np.random.default_rng(0)
    diffs = []
    for other in ("SLICER (shipped)", "SLICER-104"):
        a, oka = reads[("crisprHAL 2", "substitution map (80)")]; b, okb = reads[(other, "substitution map (80)")]
        ok = oka & okb & np.isfinite(vec)
        d = []
        for _ in range(n_boot):
            k = rng.choice(np.flatnonzero(ok), ok.sum())
            d.append(_rho(b[k], vec[k]) - _rho(a[k], vec[k]))
        diffs.append(dict(model=other, readout="substitution map (80)", fit="shipped", spearman_minus_crisprhal=_rho(b[ok], vec[ok]) - _rho(a[ok], vec[ok]),
                          lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5))))
    out.to_csv(OUT / "interp_compare_validity.csv", index=False)
    pd.DataFrame(diffs).to_csv(OUT / "interp_compare_validity_diff.csv", index=False)
    return out, pd.DataFrame(diffs)


def faithfulness():
    h = pd.read_csv(CACHE / "hal_sensitivity_n.csv").assign(model="crisprHAL 2", n_blocks=25)
    f = pd.read_csv(CACHE / "slicer_full_sensitivity_n.csv")
    c = pd.read_csv(CACHE / "slicer_compact_sensitivity_n.csv")
    out = pd.concat([h, f, c], ignore_index=True)
    out["shuffled_control"] = out["attribution"].str.contains("shuffled")
    out = out.drop(columns=[x for x in ("unit",) if x in out])
    out.to_csv(OUT / "interp_compare_faithfulness.csv", index=False)
    real = out[~out["shuffled_control"]].set_index(["model", "n"])
    rows = []
    for k in sorted(set(out["n"])):
        h_, s_, c_ = real.loc[("crisprHAL 2", k)], real.loc[("SLICER (shipped)", k)], real.loc[("SLICER-104", k)]
        for name, o in (("SLICER (shipped)", s_), ("SLICER-104", c_)):
            d = o["pearson"] - h_["pearson"]; se = float(np.hypot(o["se"], h_["se"]))
            rows.append(dict(n=k, model=name, pearson=o["pearson"], crisprhal=h_["pearson"], diff=d, lo=d - 1.96 * se, hi=d + 1.96 * se))
    pd.DataFrame(rows).to_csv(OUT / "interp_compare_faithfulness_diff.csv", index=False)
    return out, pd.DataFrame(rows)


def cross_model():
    """Do the two models say the same thing? Agreement between the substitution maps of the shipped fits, and with the model-free map."""
    free = pd.read_csv(OUT / "mechanism_position_base.csv").set_index(["position", "base"])["effect_sd"]
    vec = np.array([free.get((p + 1, BASES[b]), np.nan) for p in range(20) for b in range(4)])
    m = {k: _z(k, "shipped")["ism_map"].ravel() for k in MODELS}
    rows = [dict(a=a, b=b, spearman=_agree(m[a], m[b])) for i, a in enumerate(MODELS) for b in MODELS[i + 1:]]
    rows += [dict(a=a, b="model-free map (Finding 7)", spearman=_agree(m[a], vec)) for a in MODELS]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "interp_compare_cross_model.csv", index=False)
    return out


def flank():
    h = pd.read_csv(CACHE / "hal_flank_edits.csv").assign(model="crisprHAL 2")
    s = pd.read_csv(CACHE / "slicer_flank_edits.csv")
    out = pd.concat([s, h], ignore_index=True)
    out.to_csv(OUT / "interp_compare_flank_edits.csv", index=False)
    return out


def fits():
    rows = []
    for model in MODELS:
        for tag in SEEDS:
            try:
                z = _z(model, tag)
            except FileNotFoundError:
                continue
            if model == "crisprHAL 2":
                fit = CACHE / f"hal_{tag}_fit.csv"
                r = pd.read_csv(fit).iloc[0].to_dict() if fit.exists() else dict(test_rho=np.nan, n_train=np.nan)
                rows.append(dict(model=model, fit=tag, n_train=r.get("n_train", np.nan), test_rho=r.get("test_rho", np.nan)))
            else:
                rows.append(dict(model=model, fit=tag, n_train=int(z["n_train"]), test_rho=float(z["test_rho"])))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "interp_compare_fits.csv", index=False)
    return out


def step_assemble():
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
    print("\nfits"); print(fits().round(4).to_string(index=False))
    o, d = faithfulness(); print("\nQ1 faithfulness"); print(d.round(3).to_string(index=False))
    print("\nQ2 stability"); print(stability().drop(columns=["values"]).round(3).to_string(index=False))
    v, vd = validity(); print("\nQ3 validity (shipped fits)"); print(v[v.fit == "shipped"].round(3).to_string(index=False)); print(vd.round(3).to_string(index=False))
    print("\nthe models against each other (substitution maps, shipped fits)"); print(cross_model().round(3).to_string(index=False))
    print("\nflank edits"); print(flank()[["model", "edit", "mean_change", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
    r = CACHE.parent / "interp_compare_readable.csv"
    if (OUT / "interp_compare_readable.csv").exists():
        print("\nQ4 compressible"); print(pd.read_csv(OUT / "interp_compare_readable.csv").round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["slicer", "readable", "assemble"], required=True)
    ap.add_argument("--tags", default=",".join(SEEDS), help="slicer step: which fits")
    a = ap.parse_args()
    if a.step == "slicer":
        step_slicer([t for t in a.tags.split(",") if t])
    elif a.step == "readable":
        step_readable()
    else:
        step_assemble()


if __name__ == "__main__":
    main()
