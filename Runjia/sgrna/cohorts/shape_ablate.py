"""Shape and 3D descriptors add nothing to family A under boosted trees (+0.0006 rho at best), although a ridge model on all columns finds +0.0097 in them, so the null belongs to the pipeline and not to the descriptors.

Ablations of the shape / 3D blocks on the curated cutting cohort, gene-grouped CV. Each arm is the base representation (6,169 columns regenerable from the 20-mer) plus a set of extension blocks, optionally plus family A. All arms run on the
same 30,203 genic guides, with the same gene-grouped folds per seed, so paired per-fold differences are valid. Feature selection (top 300 by gain) is fitted inside each training fold, as everywhere in this project.

    python -m sgrna.cohorts.shape_ablate --stage 1     # blocks by source, on base
    python -m sgrna.cohorts.shape_ablate --stage 2     # same, on top of family A
    python -m sgrna.cohorts.shape_ablate --stage 3     # shuffled-flank controls   (stages 4-11: column kinds, composition references, capacity checks, fresh seeds; see `main`)
    python -m sgrna.cohorts.shape_ablate --paired      # every arm against a reference arm, fold by fold                      -> shape_ablation_paired.csv
    python -m sgrna.cohorts.shape_ablate --ridge       # the same comparison with a ridge model instead of boosted trees     -> ridge_check.csv
    python -m sgrna.cohorts.shape_ablate --redundancy  # how much of each shape column family A already holds                  -> shape_redundancy_from_A.csv
    python -m sgrna.cohorts.shape_ablate --esp-replicate   # the same question on a second, independent cohort (eSpCas9)          -> esp_replicate.csv
    python -m sgrna.cohorts.shape_ablate --geom-importance   # which geometry columns the booster uses, and what predicts them -> geom_importance.csv

**--paired.** Folds are identical across arms for a given seed (gene-grouped, deterministic), so the fold-level difference is a paired quantity. Reported: mean difference, the mean per seed (three values), and the number of the 15 folds in which the arm is
ahead. No p-values are quoted: the folds share genes, so they are not independent draws and a standard error over 15 of them would overstate precision.
**--ridge.** A ridge model on all columns (no selector, no trees), gene-grouped 5-fold CV, one seed, alpha chosen on an inner split of the first training fold and then fixed. It answers whether the boosted-tree null is a property of the descriptors or of the
pipeline: the ridge finds +0.0097 beyond family A (the same comparison with block-shuffled and arm-shuffled controls is in the table), the trees +0.0006 at best.
**--redundancy.** For every column of a block, the out-of-fold R^2 of a ridge regression from family A's columns (5-fold, gene-grouped). A column with R^2 near 1 is a re-expression of what A holds; a column with R^2 near 0 carries something A does not. Reported per block and per column kind.
**--geom-importance.** Fits the selector + model once on the whole cohort (base + mc geom), lists the geometry columns
that survive selection with their gain, then asks for each whether base composition in the arm
windows predicts it out of fold (gene-grouped). Three predictors, because they answer different questions:

  mononucleotide   the `comp` block, 40 columns of A/C/G/T fractions, by gradient boosting. A linear ridge from
                   family A could not do this (mean R2 0.07, `--redundancy`); a nonlinear map from letters might.
  dinucleotide, GB the `dcomp` block, 128 columns of dinucleotide fractions, by gradient boosting.
  dinucleotide, ridge  the same 128 columns by a plain ridge. The excess-twist columns are a summed step twist
                   minus a constant per base pair, which is *linear* in dinucleotide counts, so if the finding
                   "excess twist is a weighted dinucleotide count" is what it says, a linear model must recover it.

The dinucleotide figure (R2 0.963, quoted in the report's Appendix B4) was originally computed in a session whose
code was not kept, so it had no producer and no audit check; this module is now that producer.
**--esp-replicate.** 59,489 guides, a different nuclease on the same genome, base + family A already cached by `transfer.py`. Flanks are cut from the genome (500 nt each side, on the guide's strand) because the shipped context is shorter than the window the geometry needs. CV is grouped by 100-kb block (the cohort has no gene labels in its cache). Arms mirror `shape_ablate` stage 2/3.
Absorbed `shape_analyze`, `ridge_check`, `redundancy`, `geom_importance` and `esp_replicate`.

Report section: Appendix B4.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold

from sgrna import config
from sgrna.build import genome
from sgrna.harness.curated import load_base, load_block, load_groups
from sgrna.harness.run_ablation import _impute, champion_params, cross_validate, make_selector

from ..features import geom, shape_build
from ..common import INTERIM, OUT

SEEDS = (41, 42, 43)
OUTCSV = OUT / "shape_ablation.csv"


class Cohort:
    def __init__(self):
        X, y, names, protos = load_base()
        groups, ok = load_groups("gene", protos)
        self.X, self.y, self.protos, self.groups = X[ok], y[ok], protos[ok], groups[ok]
        self.names = names
        self.ok = ok
        self._A = None

    def A(self):
        if self._A is None:
            v, fn = load_block("a_flank", self.protos)
            self._A = (v, fn)
        return self._A

    def ext(self, source, control, block, only=None):
        """`only`: optional predicate on the raw column name, to take a column-kind subset."""
        arr, nm = shape_build.load(source, control, block)
        arr = arr[self.ok]
        keep = np.isfinite(arr).any(0) & (np.nanstd(arr, 0) > 0)
        if only is not None:
            keep &= np.array([bool(only(n)) for n in nm])
        return arr[:, keep], [f"{source}.{control or 'real'}.{block}.{n}" for n, k in zip(nm, keep) if k]


def run_arm(co: Cohort, label: str, parts: list, seeds=SEEDS, tagpref="shp", n_features=None, overrides=None):
    """parts: list of (array, names, family_label)."""
    Xs, names, fams = [co.X], list(co.names), ["base"] * len(co.names)
    for arr, nm, fam in parts:
        Xs.append(arr); names += nm; fams += [fam] * len(nm)
    X = np.hstack(Xs)
    tag = f"{tagpref}_{label}"
    res = cross_validate(X, co.y, names, fams, n_features=n_features or config.N_FEATURES, seeds=list(seeds),
                         n_splits=config.N_SPLITS, verbose=False, groups=co.groups, tag=tag,
                         experiment=label, champion_overrides=overrides)
    folds = pd.read_csv(config.RESULTS / f"folds_{tag}.csv")
    folds = folds[folds.experiment == label].sort_values(["seed", "fold"])
    sel = res.get("selected_by_family") if isinstance(res, dict) else None
    row = dict(arm=label, n_cols=X.shape[1], n_ext=sum(a.shape[1] for a, _, _ in parts),
               spearman=res["summary"]["spearman"]["mean"], spearman_sd=res["summary"]["spearman"]["sd"],
               r2=res["summary"]["r2"]["mean"],
               n_selected_ext=float(folds.n_engineered_selected.mean()),
               fold_rho=";".join(f"{v:.5f}" for v in folds.spearman))
    return row


def append(row):
    df = pd.read_csv(OUTCSV) if OUTCSV.exists() else pd.DataFrame()
    df = pd.concat([df[df.arm != row["arm"]], pd.DataFrame([row])], ignore_index=True) if len(df) else pd.DataFrame([row])
    df.to_csv(OUTCSV, index=False)


def done():
    return set(pd.read_csv(OUTCSV).arm) if OUTCSV.exists() else set()


# ---------------------------------------------------------------------------------------------------------------- --paired (was shape_analyze)
def _load_folds():
    d = pd.read_csv(OUTCSV).set_index("arm")
    return d, {a: np.array([float(x) for x in d.loc[a, "fold_rho"].split(";")]) for a in d.index}


def _paired(f, arm, ref, nseeds=3):
    n = min(len(f[arm]), len(f[ref]))
    diff = f[arm][:n] - f[ref][:n]
    per_seed = diff.reshape(nseeds, -1).mean(1) if n % nseeds == 0 else np.array([np.nan])
    return dict(delta=float(diff.mean()), per_seed=";".join(f"{x:+.4f}" for x in per_seed), folds_ahead=f"{int((diff > 0).sum())}/{n}")


def paired_table(refs=(("base", "base"), ("A", "A"))):
    """Every arm of the ablation against each reference arm: the paired mean difference, its mean per seed and the number of folds it is ahead in."""
    d, f = _load_folds()
    rows = []
    for arm in d.index:
        r = dict(arm=arm, n_ext=int(d.loc[arm, "n_ext"]), rho=round(float(d.loc[arm, "spearman"]), 4), selected=round(float(d.loc[arm, "n_selected_ext"]), 1))
        for name, ref in refs:
            if ref in f and arm != ref:
                p = _paired(f, arm, ref)
                r.update({f"d_vs_{name}": round(p["delta"], 4), f"seeds_vs_{name}": p["per_seed"], f"ahead_vs_{name}": p["folds_ahead"]})
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------- --ridge (was ridge_check)
def _prep(X):
    X = np.asarray(X, np.float64)
    med = np.nanmedian(X, 0); med = np.where(np.isfinite(med), med, 0)
    X = np.where(np.isfinite(X), X, med)
    return X


def ridge(only=None):
    """The ridge comparison on the twelve arms (or just those named in `only`, which must include `base+A`: the paired differences are against it)."""
    from sklearn.linear_model import Ridge

    from ..harness.run_ablation import grouped_folds

    co = Cohort()
    A, nA = co.A()
    parts = {"A": A}
    for blk in ("mean", "vec", "geom"):
        parts[f"mc_{blk}"] = co.ext("mc", None, blk)[0]
        parts[f"mc_blockshuf_{blk}"] = co.ext("mc", "block", blk)[0]
    for k, args in {"mc_all": [("mc", None, b) for b in ("mean", "vec", "geom")],
                    "mc_armshuf_all": [("mc", "arm", b) for b in ("mean", "vec", "geom")],
                    "md_all": [("md", None, b) for b in ("mean", "vec", "geom")]}.items():
        parts[k] = np.hstack([co.ext(*a)[0] for a in args])
    arms = {"base": [], "base+A": ["A"], "base+A+mc_mean": ["A", "mc_mean"], "base+A+mc_vec": ["A", "mc_vec"],
            "base+A+mc_geom": ["A", "mc_geom"], "base+A+mc_blockshuf_mean": ["A", "mc_blockshuf_mean"],
            "base+A+mc_blockshuf_vec": ["A", "mc_blockshuf_vec"], "base+A+mc_blockshuf_geom": ["A", "mc_blockshuf_geom"], "base+mc_all": ["mc_all"], "base+A+mc_all": ["A", "mc_all"],
            "base+A+mc_armshuf_all": ["A", "mc_armshuf_all"], "base+A+md_all": ["A", "md_all"]}
    arms = {k: v for k, v in arms.items() if only is None or k in only}
    Xbase = _prep(co.X)
    folds = list(grouped_folds(co.groups, 5, 41))
    rows = []
    alphas_seen = {}
    for arm, ps in arms.items():
        X = np.hstack([Xbase] + [_prep(parts[p]) for p in ps])
        mu, sd = X[folds[0][0]].mean(0), X[folds[0][0]].std(0) + 1e-9
        X = (X - mu) / sd
        tr0 = folds[0][0]
        inner = np.random.default_rng(0).permutation(tr0)
        a_tr, a_va = inner[: int(0.8 * len(inner))], inner[int(0.8 * len(inner)):]
        best = max((spearmanr(co.y[a_va], Ridge(alpha=al).fit(X[a_tr], co.y[a_tr]).predict(X[a_va])).statistic, al)
                   for al in (3000, 10000, 30000, 100000, 300000, 1000000))
        alpha = best[1]
        alphas_seen[arm] = alpha
        rho = [spearmanr(co.y[te], Ridge(alpha=alpha).fit(X[tr], co.y[tr]).predict(X[te])).statistic for tr, te in folds]
        rows.append(dict(arm=arm, n_cols=X.shape[1], alpha=alpha, rho=float(np.mean(rho)), fold_rho=";".join(f"{r:.5f}" for r in rho)))
        print(f"  {arm:24s} cols {X.shape[1]:6d} alpha {alpha:6.0f}  rho {np.mean(rho):.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "ridge_check.csv", index=False)
    # paired fold differences against base+A, at each arm's own alpha
    f = {r["arm"]: np.array([float(x) for x in r["fold_rho"].split(";")]) for r in rows}
    for arm in f:
        if arm in ("base", "base+A"):
            continue
        d = f[arm] - f["base+A"]
        print(f"  {arm:24s} minus base+A: mean {d.mean():+.4f}  folds {' '.join(f'{x:+.4f}' for x in d)}", flush=True)


# ======================================================================================================  redundancy and geometry importance (--redundancy, --geom-importance)
# Two follow-ups of the null: the shape blocks are partly (the means) or hardly (the 3D geometry) predictable from family A, and the geometry columns the booster does use are excess twist,
# which dinucleotide fractions predict. They need the shape caches (`features/shape_build.py`), which are not in the repository.
REDUNDANCY_OUT = OUT / "shape_redundancy_from_A.csv"
GEOM_IMPORTANCE_OUT = OUT / "geom_importance.csv"


def redundancy():
    co = Cohort()
    A, nA = co.A()
    A = np.nan_to_num((A - np.nanmean(A, 0)) / (np.nanstd(A, 0) + 1e-9))
    rows = []
    for src in ("mc", "md", "xrc"):
        for blk in ("mean", "vec", "geom"):
            arr, nm = co.ext(src, None, blk)
            Y = np.nan_to_num((arr - np.nanmean(arr, 0)) / (np.nanstd(arr, 0) + 1e-9))
            pred = np.zeros_like(Y)
            for tr, te in GroupKFold(5).split(A, groups=co.groups):
                pred[te] = Ridge(alpha=50.0).fit(A[tr], Y[tr]).predict(A[te])
            r2 = 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
            for n, v in zip(nm, r2):
                rows.append(dict(source=src, block=blk, col=n.split(".", 3)[-1], r2_from_A=float(v)))
            print(f"  {src} {blk}: {len(nm)} cols, mean R2 {r2.mean():.3f}, share >0.9: {(r2 > 0.9).mean():.2f}, "
                  f"share <0.2: {(r2 < 0.2).mean():.2f}", flush=True)
    pd.DataFrame(rows).to_csv(REDUNDANCY_OUT, index=False)


def _oof_r2(X, y, groups, model):
    """Gene-grouped 5-fold out-of-fold R2 of `model(Xtr, ytr) -> predict(Xte)`."""
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, groups=groups):
        pred[te] = model(X[tr], y[tr]).predict(X[te])
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def _geom_gb(Xtr, ytr):
    return xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=-1).fit(Xtr, ytr)


def _geom_ridge(Xtr, ytr):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12            # standardise on the training fold only
    m = RidgeCV(alphas=np.logspace(-4, 3, 8)).fit((Xtr - mu) / sd, ytr)
    return type("R", (), {"predict": staticmethod(lambda X: m.predict((X - mu) / sd))})


def geom_importance():
    co = Cohort()
    G, nG = co.ext("mc", None, "geom")
    C, nC = co.ext("comp", None, "mean")
    D, nD = co.ext("dcomp", None, "mean")
    C, D = np.nan_to_num(C), np.nan_to_num(D)
    X = np.hstack([co.X, G]); names = list(co.names) + nG
    Xi, _ = _impute(X, X[:10])
    sel = make_selector(41); sel.fit(Xi, co.y)
    top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
    m = xgb.XGBRegressor(**champion_params(41)).fit(Xi[:, top], co.y)
    imp = pd.Series(m.feature_importances_, index=[names[i] for i in top]).sort_values(ascending=False)
    geo = imp[[i.startswith("mc.real.geom") for i in imp.index]]
    print(f"{len(geo)} geometry columns kept of {len(top)}; share of total importance {geo.sum():.3f}")
    rows = []
    for name, v in geo.head(12).items():
        j = nG.index(name)
        y = G[:, j]
        r2 = _oof_r2(C, y, co.groups, _geom_gb)
        r2_d_gb = _oof_r2(D, y, co.groups, _geom_gb)
        r2_d_ridge = _oof_r2(D, y, co.groups, _geom_ridge)
        rows.append(dict(column=name.replace("mc.real.geom.", ""), importance=v, r2_from_composition=r2,
                         r2_from_dinucleotide_gb=r2_d_gb, r2_from_dinucleotide_ridge=r2_d_ridge))
        print(f"  {rows[-1]['column']:22s} importance {v:.4f}   out-of-fold R2: letters {r2:.3f}, "
              f"dinucleotides GB {r2_d_gb:.3f} / ridge {r2_d_ridge:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(GEOM_IMPORTANCE_OUT, index=False)


# ======================================================================================================  the second cohort (--esp-replicate)
# crisprHAL's E. coli eSpCas9 screen as an independent cohort for the same question. Arms mirror stages 2 and 3 above.
ESP_OUT = OUT / "esp_replicate.csv"


def flank_index():
    meta = json.loads((config.INTERIM / "transfer_eSpCas9" / "meta.json").read_text())
    protos = meta["protospacers"]
    seqs, lefts, strands = genome.protospacer_sites()
    site = {s: (l, t) for s, l, t in zip(seqs, lefts, strands)}
    g = genome.load_genome()
    rows, miss = [], 0
    for p in protos:
        if p not in site:
            miss += 1
            rows.append(dict(protospacer=p, upstream="N" * 500, downstream="N" * 500, pam="NGG"))
            continue
        left, st = site[p]
        ctx = genome.guide_context(int(left), int(left) + 19, "+" if st == 1 else "-", 500, g)
        rows.append(dict(protospacer=p, upstream=ctx["upstream"], downstream=ctx["downstream"], pam=ctx["pam"]))
    print(f"  {len(protos):,} guides, {miss} not uniquely located", flush=True)
    return pd.DataFrame(rows), meta


def esp_build(force=False):
    d = INTERIM / "esp"
    d.mkdir(exist_ok=True)
    if not force and (d / "mc_real_geom.npy").exists():
        return
    idx, _ = flank_index()
    for ctl in (None, "arm"):
        r = geom.compute(idx, "mc", control=ctl, verbose=False)
        for blk, (arr, names) in r.items():
            np.save(d / f"mc_{ctl or 'real'}_{blk}.npy", arr)
            (d / f"mc_{ctl or 'real'}_{blk}.json").write_text(json.dumps(names))
        print(f"  built mc {ctl or 'real'}", flush=True)
    arr, nm = geom.composition(idx)
    np.save(d / "comp_real_mean.npy", arr); (d / "comp_real_mean.json").write_text(json.dumps(nm))


def esp_load_block(src, ctl, blk):
    d = INTERIM / "esp"
    return np.load(d / f"{src}_{ctl}_{blk}.npy"), json.loads((d / f"{src}_{ctl}_{blk}.json").read_text())


def esp_replicate():
    esp_build()
    meta = json.loads((config.INTERIM / "transfer_eSpCas9" / "meta.json").read_text())
    X = np.asarray(np.load(config.INTERIM / "transfer_eSpCas9" / "X.npy", mmap_mode="r"), np.float32)
    y = np.asarray(meta["y"], float); names = list(meta["names"]); nb = int(meta["n_base"])
    left = np.asarray(meta["left"], float)
    ok = np.isfinite(left) & np.isfinite(y)
    groups = (left // 100_000).astype(int).astype(str)
    Xb, XA, nA = X[:, :nb], X[:, nb:], names[nb:]

    def ext(src, ctl, blk):
        a, n = esp_load_block(src, ctl, blk)
        keep = np.isfinite(a).any(0) & (np.nanstd(a, 0) > 0)
        return a[:, keep], [f"{src}.{ctl}.{blk}.{x}" for x, k in zip(n, keep) if k]
    allblk = lambda ctl: [ext("mc", ctl, b) for b in ("mean", "vec", "geom")]
    arms = {"base": [], "A": [(XA, nA)], "mc_all": allblk("real"), "A+mc_all": [(XA, nA)] + allblk("real"),
            "A+mc_geom": [(XA, nA), ext("mc", "real", "geom")],
            "A+mc_armshuf_all": [(XA, nA)] + allblk("arm"), "comp_mean": [ext("comp", "real", "mean")]}
    done = set(pd.read_csv(ESP_OUT).arm) if ESP_OUT.exists() else set()
    rows = pd.read_csv(ESP_OUT).to_dict("records") if ESP_OUT.exists() else []
    for arm, parts in arms.items():
        if arm in done:
            continue
        Xs = np.hstack([Xb[ok]] + [p[0][ok] for p in parts])
        nm = names[:nb] + [n for p in parts for n in p[1]]
        t0 = time.time()
        res = cross_validate(Xs, y[ok], nm, ["arm"] * len(nm), n_features=config.N_FEATURES,
                             seeds=list(SEEDS), n_splits=config.N_SPLITS, verbose=False,
                             groups=groups[ok], tag=f"esp_{arm}", experiment=arm)
        folds = pd.read_csv(config.RESULTS / f"folds_esp_{arm}.csv")
        folds = folds[folds.experiment == arm].sort_values(["seed", "fold"])
        rows.append(dict(arm=arm, n_cols=Xs.shape[1], spearman=res["summary"]["spearman"]["mean"],
                         spearman_sd=res["summary"]["spearman"]["sd"],
                         fold_rho=";".join(f"{v:.5f}" for v in folds.spearman)))
        pd.DataFrame(rows).to_csv(ESP_OUT, index=False)
        print(f"  {arm:20s} rho {rows[-1]['spearman']:.4f} (sd {rows[-1]['spearman_sd']:.4f})  {time.time() - t0:.0f}s",
              flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, help="which arm table to run (1-11)")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    ap.add_argument("--paired", action="store_true", help="every arm against a reference arm, fold by fold (writes shape_ablation_paired.csv)")
    ap.add_argument("--ridge", action="store_true", help="the comparison with a ridge model on all columns (writes ridge_check.csv)")
    ap.add_argument("--esp-replicate", action="store_true", help="the same question on crisprHAL's eSpCas9 screen (writes esp_replicate.csv)")
    ap.add_argument("--redundancy", action="store_true", help="R2 of each shape column from family A (writes shape_redundancy_from_A.csv)")
    ap.add_argument("--geom-importance", action="store_true", help="the geometry columns the booster uses and what predicts them (writes geom_importance.csv)")
    ap.add_argument("--ridge-arms", nargs="*", help="only these ridge arms (include base+A), for a quick check")
    a = ap.parse_args(argv)
    if not (a.stage or a.paired or a.ridge or a.redundancy or a.geom_importance or a.esp_replicate):
        ap.error("give --stage N, --paired, --ridge, --redundancy, --geom-importance or --esp-replicate")
    if a.esp_replicate:
        esp_replicate()
    if a.redundancy:
        redundancy()
    if a.geom_importance:
        geom_importance()
    if a.paired:
        pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
        t = paired_table()
        print(t.to_string(index=False))
        t.to_csv(OUT / "shape_ablation_paired.csv", index=False)
    if a.ridge:
        ridge(a.ridge_arms)
    if not a.stage:
        return 0
    co = Cohort()
    print(f"  {co.X.shape[0]:,} guides, {co.X.shape[1]:,} base columns", flush=True)
    vA, nA = co.A()
    Apart = [(vA, nA, "A")]
    arms = []                       # (label, parts)
    def P(src, ctl, blk, fam=None):
        arr, nm = co.ext(src, ctl, blk); return (arr, nm, fam or f"S_{src}_{ctl or 'real'}_{blk}")

    if a.stage == 1:
        arms.append(("base", []))
        arms.append(("A", Apart))
        for src in ("mc", "md", "xrc"):
            for blk in ("mean", "vec", "geom"):
                arms.append((f"{src}_{blk}", [P(src, None, blk)]))
            arms.append((f"{src}_all", [P(src, None, b) for b in ("mean", "vec", "geom")]))
        arms.append(("allsrc_geom", [P(s, None, "geom") for s in ("mc", "md", "xrc")]))
        arms.append(("allsrc_all", [P(s, None, b) for s in ("mc", "md", "xrc") for b in ("mean", "vec", "geom")]))
    elif a.stage == 2:
        for blk in ("mean", "vec", "geom"):
            arms.append((f"A+mc_{blk}", Apart + [P("mc", None, blk)]))
        arms.append(("A+mc_all", Apart + [P("mc", None, b) for b in ("mean", "vec", "geom")]))
        for src in ("md", "xrc"):
            arms.append((f"A+{src}_geom", Apart + [P(src, None, "geom")]))
            arms.append((f"A+{src}_all", Apart + [P(src, None, b) for b in ("mean", "vec", "geom")]))
        arms.append(("A+allsrc_all", Apart + [P(s_, None, b) for s_ in ("mc", "md", "xrc") for b in ("mean", "vec", "geom")]))
    elif a.stage == 4:
        # column kinds inside the blocks, for the Monte Carlo source, on top of family A
        import re
        INTRA = {"ProT", "Buckle", "Shear", "Opening", "Stretch", "Stagger"}
        STEP = {"Shift", "Slide", "Rise", "Tilt", "Roll", "HelT"}
        feat = lambda n: n.split(".")[-1]
        subsets = {
            "mean": {"intra": lambda n: feat(n) in INTRA, "groove(MGW)": lambda n: feat(n) == "MGW",
                     "step": lambda n: feat(n) in STEP,
                     "up arm": lambda n: n.startswith("up."), "dn arm": lambda n: n.startswith("dn."),
                     "near (<=100 nt)": lambda n: ".w0_25." in n or ".w25_100." in n,
                     "far (>100 nt)": lambda n: ".w100_250." in n or ".w250_" in n},
            "geom": {"path positions": lambda n: n.startswith("path."), "axis tangents": lambda n: n.startswith("axis."),
                     "gyration tensor": lambda n: n.startswith("gyr."), "straightness": lambda n: n.startswith("straight."),
                     "excess twist": lambda n: n.startswith("xtwist."),
                     "near (offset/window <=100)": lambda n: bool(re.search(r"k[+-](12|25|50|100)\.|(dn|up|both)100\.|[a-z]+100$", n)),
                     "far (>100)": lambda n: bool(re.search(r"k[+-](200|400)\.|(dn|up|both)(250|499)\.|[a-z]+(250|499)$", n))},
        }
        for blk, d_ in subsets.items():
            for kind, fn in d_.items():
                arr, nm = co.ext("mc", None, blk, only=fn)
                if arr.shape[1] == 0:
                    print(f"  mc_{blk}[{kind}]: no columns"); continue
                parts = [(arr, nm, f"S_mc_{blk}_{kind}")]
                arms.append((f"mc_{blk}[{kind}]", parts))
                arms.append((f"A+mc_{blk}[{kind}]", Apart + parts))
    elif a.stage == 11:
        # fresh seeds (run with --seeds 44 45 46): do the headline comparisons hold on folds never seen before?
        arms.append(("base#s2", []))
        arms.append(("A#s2", Apart))
        arms.append(("mc_all#s2", [P("mc", None, b) for b in ("mean", "vec", "geom")]))
        arms.append(("A+mc_all#s2", Apart + [P("mc", None, b) for b in ("mean", "vec", "geom")]))
        arms.append(("A+mc_armshuf_all#s2", Apart + [P("mc", "arm", b) for b in ("mean", "vec", "geom")]))
    elif a.stage == 10:
        # capacity check: the champion is depth 4 / 400 trees; give the booster depth 6 / 800 trees
        arms.append(("A@d6", Apart))
        arms.append(("A+mc_all@d6", Apart + [P("mc", None, b) for b in ("mean", "vec", "geom")]))
        arms.append(("A+mc_armshuf_all@d6", Apart + [P("mc", "arm", b) for b in ("mean", "vec", "geom")]))
    elif a.stage == 9:
        for ctl in ("block", "arm"):
            arms.append((f"mc_{ctl}shuf_vec", [P("mc", ctl, "vec")]))
    elif a.stage == 8:
        # family A is itself four kinds of column: take it apart
        vA_, nA_ = co.A()
        def sub(*prefixes, drop=False):
            m = np.array([any(n.startswith(f"eng.flank.{p}.") for p in prefixes) for n in nA_])
            if drop:
                m = ~m
            return [(vA_[:, m], [n for n, k in zip(nA_, m) if k], "A_sub")]
        for kind in ("seq", "qct", "win", "comp"):
            arms.append((f"A_only_{kind}", sub(kind)))
        arms.append(("A_local(seq+qct)", sub("seq", "qct")))
        arms.append(("A_windowed(win+comp)", sub("win", "comp")))
        for kind in ("seq", "qct", "win", "comp"):
            arms.append((f"A_minus_{kind}", sub(kind, drop=True)))
    elif a.stage == 7:
        # displacement check: the same comparison with a 600-column budget instead of 300
        arms.append(("A@600", Apart)); arms.append(("A+mc_all@600", Apart + [P("mc", None, b) for b in ("mean", "vec", "geom")]))
        arms.append(("A+mc_geom@600", Apart + [P("mc", None, "geom")]))
        arms.append(("base@600", []))
    elif a.stage == 6:
        # the composition reference: letters in the same windows, alone and with the shape means
        arms.append(("comp_mean", [P("comp", None, "mean")]))
        arms.append(("comp+mc_mean", [P("comp", None, "mean"), P("mc", None, "mean")]))
        arms.append(("A+comp_mean", Apart + [P("comp", None, "mean")]))
        arms.append(("dcomp_mean", [P("dcomp", None, "mean")]))
        arms.append(("dcomp+mc_mean", [P("dcomp", None, "mean"), P("mc", None, "mean")]))
        arms.append(("A+dcomp_mean", Apart + [P("dcomp", None, "mean")]))
    elif a.stage == 5:
        # does recentring each source's mean twist to 34.3 deg (10.5 bp/turn) explain the md geometry deficit?
        for src in ("md", "xrc", "mc"):
            arms.append((f"{src}_geom_tw", [P(src, "tw", "geom")]))
    elif a.stage == 3:
        # shuffled-flank controls: same pipeline, flank bases shuffled (block = within 25-nt blocks, arm = whole arm)
        for ctl in ("arm", "block"):
            for blk in ("geom", "mean"):
                arms.append((f"mc_{ctl}shuf_{blk}", [P("mc", ctl, blk)]))
            arms.append((f"mc_{ctl}shuf_all", [P("mc", ctl, b) for b in ("mean", "vec", "geom")]))
            arms.append((f"A+mc_{ctl}shuf_all", Apart + [P("mc", ctl, b) for b in ("mean", "vec", "geom")]))
    for label, parts in arms:
        if label in done():
            print(f"  {label}: cached", flush=True); continue
        row = run_arm(co, label, parts, seeds=a.seeds, n_features=600 if label.endswith('@600') else None,
                      overrides=dict(max_depth=6, n_estimators=800) if label.endswith('@d6') else None)
        append(row)
        print(f"  {label:22s} rho {row['spearman']:.4f} (sd {row['spearman_sd']:.4f}) "
              f"ext cols {row['n_ext']} selected {row['n_selected_ext']:.0f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
