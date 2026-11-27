"""On genes sealed away from column selection and settings, family A's gain is 0.0087 lower than in cross-validation of the pool; holding whole genes out of the pool too leaves +0.0081, so unseen genes explain almost none of it, and the gain survives a change of screen (+0.103 on eSpCas9 labels of sealed genes, positive in 15 of 15 draws); on C. rodentium, which no choice in this project ever saw, the tuned model beats the default model by +0.0172, within the range seen inside E. coli.

One question in four steps: how much of family A's cross-validated gain (and of the tuning's) is the pipeline inflating its own estimates, and does it survive genes, a screen and an organism that no step ever saw?

  holdout       Is there adaptive overfitting? Seal whole genes (20% of rows) away from column selection and hyperparameters, re-derive the mechanisable choices inside the development pool, score once on the sealed genes. -> sealed_holdout.csv
  gene          Is that gap just the cost of unseen genes? Re-run only the pool cross-validation with whole genes held out, for the same sealing draws.                                                                   -> sealed_holdout_gene.csv
  crossscreen   Does the gain survive a change of screen? Train on SpCas9 guides in the development genes, score the eSpCas9 labels of guides in the sealed genes (15 draws).                                    -> sealed_crossscreen.csv
  organism      Does the tuning survive a change of organism? Train on all of E. coli WT-SpCas9, score C. rodentium zero-shot with the champion, the tuned XGBoost and the average of five.                  -> sealed_organism.csv

    python -m sgrna.controls.sealed --parts holdout --seal-seeds 20261016 20261017
    OMP_NUM_THREADS=3 python -m sgrna.controls.sealed --parts gene [--max-draws N] [--arms base_only frozen dev_tuned]
    OMP_NUM_THREADS=6 python -m sgrna.controls.sealed --parts crossscreen [--max-draws N] [--threads T]
    python -m sgrna.controls.sealed --parts organism

**holdout.** Part 4 says the curated five-fold figures "carry an unknown adaptive-overfitting component" and leaves it unknown, because no untouched external dataset remains. Sealing rows *now* cannot undo the history: every choice in the pipeline
was made while looking at numbers computed on all 33,567 guides. So this does not measure our actual history; it measures the procedure. Re-derive the mechanisable choices (column selection and hyperparameters) using only a development pool, then score
once on rows no step of that re-derivation ever saw. If a cross-validated estimate computed inside the pool is systematically higher than the sealed-set score, the pipeline inflates its own estimates, and by how much.
*The confound, and the design that removes it.* Sealed rows are a different population: whole genes are sealed, and gene identity explains 29% of this label's variance, so any model scores differently there for reasons that have nothing to do with
adaptivity. The fix is a difference of differences, the CV-minus-sealed gap for a model whose configuration was not chosen on this cohort, subtracted:

    adaptive component  =  (CV - sealed | tuned here)  -  (CV - sealed | not tuned here)

Three arms. `base_only`: 6,169 regenerated columns, champion hyperparameters carried over from the published cohort, no selection, so nothing about it was chosen on these rows. `frozen`: base + family A, champion parameters, 300 columns selected inside each
training split, the configuration the report quotes. `dev_tuned`: the same with an 8-draw search run inside the development pool only, the arm that can overfit the pool. A 5-fold CV inside the pool trains each fold on 80% of it while the sealed-set fit trains
on 100%, which flatters the sealed number by about +0.005; `match_train_size` subsamples the sealed-set fit to the same 80%. *What it cannot capture is the larger part:* family A exists because someone looked at residuals computed on the whole cohort, and
the window sizes, the decision to build it, the choice of `n_features` and five withdrawn claims' worth of exploration are human choices informed by the full data. The number is a lower bound on the adaptive component, and it bounds only the automatable part.

**gene.** The pool CV of `holdout` is a random-row 5-fold CV while the sealed score is on whole genes. For a feature that reads the locus (family A) a random-row fold can score by recognising the neighbourhood of a guide whose neighbours sit in training,
so the difference of differences also contains what unseen genes cost such a feature. This part re-runs only the pool CV with whole genes held out (guides outside any gene are grouped by 100 kb block, so neighbours stay together), with the settings the
original run used (the dev-tuned arm's are read from `sealed_holdout.csv`); the sealed scores are read, not recomputed:

    random-row component   = gap_random(frozen) - gap_random(base)     (the report's +0.0087)
    gene-grouped component = gap_gene(frozen)   - gap_gene(base)       (both CVs and the sealed score hold out whole genes)
    cost of unseen genes   = random-row component - gene-grouped component

Near zero, the whole +0.0087 is the cost of unseen genes to a locus-reading feature and none of it is adaptive; near +0.0087, the reverse. The stored result holds the first 8 of the 15 sealings (about 7 minutes each at 3 threads); their random-row component,
+0.0088, matches the 15-draw +0.0087.

**crossscreen.** Every seal so far held out genes of the screen that supplied the training labels; this one also changes the screen. Fitted on the WT-SpCas9 cohort in the development genes of one of the 15 sealing draws (same seeds, same 20% of guides sealed by
whole gene) and scored on the eSpCas9 labels of every eSpCas9 guide in a sealed gene: a different enzyme, a different sequencing run, labels that no step has fitted. eSpCas9's scores were visible during tuning, so the screen is confirmatory, not untouched (Part 4);
the loci are still the same genome, so a guide in a sealed gene can have training neighbours whose 1 kb flank windows overlap its own (the claim is that the gain survives a change of enzyme and of gene, not independence of loci; that is the C. rodentium check).
Prediction, fixed before the run (23 Nov 2026): the gain is positive in at least 14 of the 15 draws with a mean of at least +0.08 (about half of the WT screen's +0.156 on its own sealed genes). Result: positive in 15 of 15, mean +0.103.

**organism.** The tuned settings were chosen on 70% of the E. coli curated cohort's genes, and the report's estimate of how much that adaptive selection inflates a gain is +0.0087 rho, measured on genes sealed *inside* E. coli. A sturdier seal is a different
organism. The C. rodentium TevSpCas9 screen (25,210 guides, a 5.35 Mb genome) was used only to test the frozen champion's transfer; no feature, setting, column count or model was ever chosen on it. (The other E. coli screen, eSpCas9, is not untouched: it scores the
same guide library at the same loci, so it can be held out as a *label* but not as a *locus*.) Train on the whole E. coli WT-SpCas9 cohort, score C. rodentium zero-shot, with three arms that share a protocol: the frozen champion, the tuned XGBoost (d099x) and the
XGBoost average of five (the arms of the head-to-head). Only this direction is clean: training on C. rodentium and scoring E. coli would score the very genes the settings were chosen on.
Prediction, written before the run: the tuned arm is at least as good as the champion (probability about 0.65), by less than it is within E. coli (+0.014 on genes it never saw, +0.019 and +0.026 on crisprHAL 2's two splits), because part of that gain is the data
it was chosen on. Decision rule: the difference (tuned minus champion), paired on guides and bootstrapped over 100 kb blocks (neighbouring guides share DNA and are not independent), three seeds averaged. Lower bound above 0: the tuning's benefit survives a sealed
organism. Interval spanning 0: inconclusive. Upper bound below 0: tuning hurt outside the data it was chosen on, which would be evidence of adaptive overfitting in the settings.

Replaced `sealed_holdout`, `sealed_holdout_gene`, `sealed_crossscreen` and `compare/sealed_organism`; the fold helpers they shared (and nine other modules imported from `sealed_holdout_gene`) are `harness/folds.py`.

Report section: Part 4.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from .. import config
from ..common import OUT
from ..harness import folds
from ..harness.curated import load_base, load_block
from ..harness.folds import fit_predict, fold_ids

SEALED_CSV = OUT / "sealed_holdout.csv"
SEAL_FRACTION = 0.20
SEAL_SEED = 20261016          # fixed here, once, and not varied afterwards
ARMS = ("base_only", "frozen", "dev_tuned")


def _seal(protos: np.ndarray, seed: int = SEAL_SEED, verbose: bool = True):
    """Seal whole genes until ~20% of rows are sealed; intergenic rows stay in."""
    from ..build.build_features import load_curated_index

    idx = load_curated_index(verbose=False).set_index(config.ID_COL)
    gene = idx["gene_name"].reindex(protos)
    gene = np.array([g if isinstance(g, str) and g else "" for g in gene], dtype=object)
    genes = np.array(sorted({g for g in gene if g}))
    rng = np.random.default_rng(seed)
    rng.shuffle(genes)
    counts = pd.Series(gene).value_counts()
    sealed_genes, n = set(), 0
    target = int(len(protos) * SEAL_FRACTION)
    for g in genes:
        if n >= target:
            break
        sealed_genes.add(g)
        n += int(counts[g])
    sealed = np.array([g in sealed_genes for g in gene])
    if verbose:
        print(f"  sealed {sealed.sum():,} guides in {len(sealed_genes):,} genes ({sealed.mean():.1%}); development pool {(~sealed).sum():,}")
    return sealed


# ---------------------------------------------------------------------------------------------------------------- step 1: holdout
def _cv_in_pool(X, y, params=None, n_features=None, n_splits=5, seed=41):
    """Out-of-fold rho inside the development pool: the estimate we would report."""
    pred = np.empty_like(y)
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        pred[va] = fit_predict(X[tr], y[tr], X[va], params, n_features, seed, n_jobs=None)
    return float(spearmanr(y, pred).statistic)


def _search_in_pool(X, y, draws=8, seed=41, verbose=True):
    """A small hyperparameter search, fitted and scored entirely inside the pool."""
    rng = np.random.default_rng(seed)
    space = dict(max_depth=[3, 4, 6, 8], learning_rate=[0.02, 0.05, 0.1], subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0],
                 reg_lambda=[0.5, 1.0, 5.0], min_child_weight=[1, 5, 20])
    inner = KFold(5, shuffle=True, random_state=seed + 1)
    itr, iva = next(iter(inner.split(X)))
    best, best_p = -np.inf, {}
    for d in range(draws):
        p = {} if d == 0 else {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        pred = fit_predict(X[itr], y[itr], X[iva], p, config.N_FEATURES, seed, n_jobs=None)
        rho = float(spearmanr(y[iva], pred).statistic)
        if rho > best:
            best, best_p = rho, p
        if verbose:
            print(f"    draw {d}: inner rho {rho:.4f}", flush=True)
    return best_p


def holdout(match_train_size: bool = True, seal_seed: int = SEAL_SEED, arms=ARMS, verbose: bool = True) -> pd.DataFrame:
    """`match_train_size` fixes a confound the first version of this had: a 5-fold CV inside the pool trains each fold on 80% of it, while the sealed-set fit trains on 100%, so the sealed number is
    flattered by ~25% more training data (worth roughly +0.005 rho, and more for the arms that benefit most from data, which are the arms under test). Subsampling the sealed-set fit to the same 80% leaves the
    population as the only difference between the two estimates."""
    X, y, names, protos = load_base()
    A, anames = load_block("a_flank", protos)
    XA = np.hstack([X, A])
    sealed = _seal(protos, seed=seal_seed, verbose=verbose)
    dev = ~sealed
    dev_idx = np.where(dev)[0]
    fit_idx = dev_idx
    if match_train_size:
        n = int(len(dev_idx) * (1 - 1 / 5))
        fit_idx = np.random.default_rng(seal_seed).choice(dev_idx, n, replace=False)
        if verbose:
            print(f"  sealed-set fits use {n:,} of {len(dev_idx):,} pool rows, matching what each CV fold trains on")

    spec = {"base_only": dict(X=X, n_features=None, tune=False), "frozen": dict(X=XA, n_features=config.N_FEATURES, tune=False),
            "dev_tuned": dict(X=XA, n_features=config.N_FEATURES, tune=True)}
    rows = []
    for arm in arms:
        cfg = spec[arm]
        Xa = cfg["X"]
        params = None
        if cfg["tune"]:
            if verbose:
                print(f"  {arm}: searching inside the development pool")
            params = _search_in_pool(Xa[dev], y[dev], verbose=verbose)
            if verbose:
                print(f"    chosen: {params}")
        cv = _cv_in_pool(Xa[dev], y[dev], params, cfg["n_features"])
        pred = fit_predict(Xa[fit_idx], y[fit_idx], Xa[sealed], params, cfg["n_features"], n_jobs=None)
        sealed_rho = float(spearmanr(y[sealed], pred).statistic)
        rows.append(dict(arm=arm, seal_seed=seal_seed, n_cols=Xa.shape[1], n_dev=int(dev.sum()), n_sealed=int(sealed.sum()), cv_in_pool=cv,
                         sealed=sealed_rho, gap=cv - sealed_rho, params=json.dumps(params or {})))
        prev = pd.read_csv(SEALED_CSV) if SEALED_CSV.exists() and SEALED_CSV.stat().st_size else pd.DataFrame()
        if len(prev):
            prev = prev[~((prev["arm"] == arm) & (prev.get("seal_seed", -1) == seal_seed))]
        if verbose:
            print(f"  {arm:>10}: pool CV {cv:.4f}   sealed {sealed_rho:.4f}   gap {cv - sealed_rho:+.4f}", flush=True)
        pd.concat([prev, pd.DataFrame([rows[-1]])], ignore_index=True).to_csv(SEALED_CSV, index=False)

    out = pd.DataFrame(rows).set_index("arm")
    if verbose and "base_only" in out.index:
        ref = out.loc["base_only", "gap"]
        print(f"\n  reference gap (nothing chosen on these rows): {ref:+.4f}")
        for arm in ("frozen", "dev_tuned"):
            if arm in out.index:
                print(f"  adaptive component, {arm:>9}: {out.loc[arm, 'gap'] - ref:+.4f}")
        print(f"\nwritten -> {SEALED_CSV}")
    return out.reset_index()


# ---------------------------------------------------------------------------------------------------------------- step 2: gene
def _cv_gene(X, y, groups, params, n_features, seed=41, verbose=True):
    fold = fold_ids(groups, 5, seed)
    pred = np.empty_like(y)
    for k in range(5):
        t0 = time.time()
        va = fold == k
        pred[va] = fit_predict(X[~va], y[~va], X[va], params, n_features, seed)
        if verbose:
            print(f"      fold {k + 1}: {time.time() - t0:5.1f}s", flush=True)
    return float(spearmanr(y, pred).statistic)


def gene(max_draws: int | None = None, arms=ARMS, verbose: bool = True) -> pd.DataFrame:
    from ..build.build_features import load_curated_index

    ref = pd.read_csv(SEALED_CSV)
    out_path = OUT / "sealed_holdout_gene.csv"
    done = pd.read_csv(out_path) if out_path.exists() and out_path.stat().st_size else pd.DataFrame()
    X, y, names, protos = load_base()
    A, anames = load_block("a_flank", protos)
    XA = np.hstack([X, A])
    idx = load_curated_index(verbose=False).set_index(config.ID_COL)
    gene_name = idx["gene_name"].reindex(protos)
    left = idx["left"].reindex(protos).to_numpy(float)
    groups = np.array([g if isinstance(g, str) and g else f"ig{int(l // 100_000)}" for g, l in zip(gene_name, left)], dtype=object)

    seeds = sorted(ref.seal_seed.unique())[:max_draws]
    for sd in seeds:
        sealed = _seal(protos, seed=int(sd), verbose=False)
        dev = np.where(~sealed)[0]
        for arm in arms:
            if len(done) and ((done.arm == arm) & (done.seal_seed == sd)).any():
                continue
            r = ref[(ref.arm == arm) & (ref.seal_seed == sd)].iloc[0]
            params = json.loads(r.params) if isinstance(r.params, str) else {}
            Xa, nf = (X, None) if arm == "base_only" else (XA, config.N_FEATURES)
            if verbose:
                print(f"  seed {sd} {arm}: gene-grouped CV in the pool ({len(dev):,} guides)", flush=True)
            cv = _cv_gene(Xa[dev], y[dev], groups[dev], params or None, nf)
            row = dict(arm=arm, seal_seed=int(sd), cv_gene=cv, cv_random=float(r.cv_in_pool), sealed=float(r.sealed), gap_gene=cv - float(r.sealed),
                       gap_random=float(r.gap), n_dev=int(len(dev)), params=r.params)
            if verbose:
                print(f"    cv_gene {cv:.4f}  cv_random {r.cv_in_pool:.4f}  sealed {r.sealed:.4f}", flush=True)
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(out_path, index=False)
    return done


# ---------------------------------------------------------------------------------------------------------------- step 3: crossscreen
def _esp_matrix():
    """eSpCas9 guides: features (base + family A, 6,517 columns, as cached by `transfer.py`), labels and host genes."""
    from ..build.build_features import _b_numbers
    from ..compare.transfer import build

    X, y, names = build("eSpCas9", verbose=False)
    m = json.loads((config.INTERIM / "transfer_eSpCas9" / "meta.json").read_text())
    left = np.asarray(m["left"], dtype=int)
    gdf = _b_numbers(pd.DataFrame({"left": left, "right": left + 19}))
    gene_name = np.array([g if isinstance(g, str) and g else "" for g in gdf["gene_name"]], dtype=object)
    return np.asarray(X, dtype=np.float32), np.asarray(y, float), gene_name, list(names), list(m["protospacers"])


def crossscreen(max_draws: int | None = None, threads: int = 6, verbose: bool = True) -> pd.DataFrame:
    from ..build.build_features import load_curated_index

    folds.N_JOBS = threads
    out_path = OUT / "sealed_crossscreen.csv"
    done = pd.read_csv(out_path) if out_path.exists() and out_path.stat().st_size else pd.DataFrame()
    ref = pd.read_csv(SEALED_CSV)
    X, y, names, protos = load_base()
    A, anames = load_block("a_flank", protos)
    XA = np.hstack([X, A])
    Xe, ye, gene_e, enames, _ = _esp_matrix()
    assert list(enames) == list(names) + list(anames), "eSpCas9 and curated columns differ"
    nb = X.shape[1]
    seeds = sorted(ref.seal_seed.unique())[:max_draws]
    for sd in seeds:
        if len(done) and (done.seal_seed == sd).any():
            continue
        sealed = _seal(protos, seed=int(sd), verbose=False)
        idx = load_curated_index(verbose=False).set_index(config.ID_COL)
        gw = idx["gene_name"].reindex(protos)
        sealed_genes = {g for g, s in zip(gw, sealed) if s and isinstance(g, str) and g}
        te = np.array([g in sealed_genes for g in gene_e])
        dev = np.where(~sealed)[0]
        t0 = time.time()
        p_base = fit_predict(X[dev], y[dev], Xe[te][:, :nb], None, None, 41)
        p_A = fit_predict(XA[dev], y[dev], Xe[te], None, config.N_FEATURES, 41)
        rb, ra = (float(spearmanr(ye[te], p).statistic) for p in (p_base, p_A))
        # the same arms scored on the WT screen's own sealed genes, from the sealing run
        wt = {a: float(ref[(ref.arm == a) & (ref.seal_seed == sd)].sealed.iloc[0]) for a in ("base_only", "frozen")}
        row = dict(seal_seed=int(sd), n_sealed_genes=len(sealed_genes), n_esp_guides=int(te.sum()), rho_base=rb, rho_A=ra, gain_esp=ra - rb,
                   wt_base=wt["base_only"], wt_A=wt["frozen"], gain_wt=wt["frozen"] - wt["base_only"])
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(out_path, index=False)
        if verbose:
            print(f"  seed {sd}: {te.sum():,} eSpCas9 guides in {len(sealed_genes):,} sealed genes  base {rb:.4f}  +A {ra:.4f}  gain {ra - rb:+.4f} "
                  f"(WT labels, same genes: {row['gain_wt']:+.4f})  {time.time() - t0:.0f}s", flush=True)
    return done


ORGANISM_ARMS = {"champion": ["champion"], "tuned XGBoost (d099x)": ["d099x"], "XGBoost average of 5": ["d099x", "d077x", "d021x", "d080x", "d000x"]}
N_BOOT = 1000


def _block_boot(y, a, b, block, rng):
    """Spearman(a) - Spearman(b), resampling 100 kb blocks of the target genome; 95% interval."""
    u, inv = np.unique(block, return_inverse=True)
    rows = [np.flatnonzero(inv == k) for k in range(len(u))]
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ix = np.concatenate([rows[k] for k in rng.integers(0, len(u), len(u))])
        out[i] = spearmanr(y[ix], a[ix]).statistic - spearmanr(y[ix], b[ix]).statistic
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def organism(seeds=(41, 42, 43), verbose: bool = True) -> pd.DataFrame:
    from ..compare.vs_crisprhal import member_specs, train_predict
    from ..harness.metrics import zscore as _z

    def cached(name):
        """The stored matrix and label of one screen (the cache `compare.transfer.build` writes); the C. rodentium genome is not needed to read it."""
        d = config.INTERIM / ("transfer_%s" % name)
        meta = json.loads((d / "meta.json").read_text())
        return np.asarray(np.load(d / "X.npy", mmap_mode="r"), np.float32), np.asarray(meta["y"], float), list(meta["names"]), meta

    rho = lambda y, p: float(spearmanr(y, p).statistic)
    Xs, ys, names, _ = cached("WT-SpCas9")
    Xt, yt, names_t, meta = cached("TevSpCas9")
    assert names == names_t, "feature names differ between the two screens"
    block = (np.asarray(meta["left"], float) // 100_000).astype(int)
    specs = member_specs(sorted({n for v in ORGANISM_ARMS.values() for n in v}))
    if verbose:
        print(f"train {Xs.shape[0]:,} E. coli guides, score {Xt.shape[0]:,} C. rodentium guides in {len(set(block))} 100 kb blocks", flush=True)
    per_seed = {s: {} for s in seeds}
    for s in seeds:
        preds, _ = train_predict(specs, Xs, ys, Xt, seed=s)
        for arm, members in ORGANISM_ARMS.items():
            per_seed[s][arm] = np.mean([_z(preds[m]) for m in members], axis=0)
        if verbose:
            print("  seed", s, {a: round(rho(yt, p), 4) for a, p in per_seed[s].items()}, flush=True)
    mean_pred = {arm: np.mean([per_seed[s][arm] for s in seeds], axis=0) for arm in ORGANISM_ARMS}
    rng = np.random.default_rng(0)
    rows = []
    for arm in ORGANISM_ARMS:
        r_seeds = [rho(yt, per_seed[s][arm]) for s in seeds]
        row = dict(arm=arm, rho_seed_mean=float(np.mean(r_seeds)), rho_seed_sd=float(np.std(r_seeds)), rho_of_seed_average=rho(yt, mean_pred[arm]))
        if arm != "champion":
            lo, hi = _block_boot(yt, mean_pred[arm], mean_pred["champion"], block, rng)
            row.update(minus_champion=row["rho_of_seed_average"] - rho(yt, mean_pred["champion"]), lo=lo, hi=hi)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "sealed_organism.csv", index=False)
    if verbose:
        print(out.round(4).to_string(index=False))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", nargs="+", choices=("holdout", "gene", "crossscreen", "organism"), required=True, help="which steps to run, in this order")
    ap.add_argument("--seal-seeds", type=int, nargs="*", default=[SEAL_SEED], help="holdout: repeat with several sealing draws (one draw of 872 genes is itself a sample)")
    ap.add_argument("--no-match-train-size", action="store_true", help="holdout: reproduce the first, confounded version")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), help="holdout and gene: which arms")
    ap.add_argument("--max-draws", type=int, default=None, help="gene and crossscreen: use the first N sealing draws of sealed_holdout.csv")
    ap.add_argument("--threads", type=int, default=None, help="gene and crossscreen: XGBoost threads (default: gene 3, crossscreen 6)")
    a = ap.parse_args(argv)
    if a.threads:
        folds.N_JOBS = a.threads
    for part in a.parts:
        if part == "holdout":
            for sd in a.seal_seeds:
                holdout(match_train_size=not a.no_match_train_size, seal_seed=sd, arms=tuple(a.arms))
        elif part == "gene":
            gene(a.max_draws, tuple(a.arms))
        elif part == "organism":
            organism()
        else:
            d = crossscreen(a.max_draws, a.threads or 6)
            print(f"\n{len(d)} draws: mean gain on eSpCas9 labels {d.gain_esp.mean():+.4f} (positive in {int((d.gain_esp > 0).sum())}), "
                  f"on the WT screen's labels of the same sealed genes {d.gain_wt.mean():+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
