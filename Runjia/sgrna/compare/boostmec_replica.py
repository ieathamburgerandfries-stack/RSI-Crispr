"""BoostMEC built the way its authors built it, retrained on the curated E. coli cohort and scored with SLICER's folds.

BoostMEC (Zarate et al., 2022, BMC Bioinformatics 23:446) is the closest published design to SLICER: LightGBM on engineered sequence features, chosen for
interpretability. Its released model was trained on human guides, so it is rebuilt here from what its authors published, and nothing is approximated that they
specify. The sources, all read in full:

  features   the 149 features of `model/featurize-dataset.R` and `model/boostmec-pipeline.py` in github.com/oazarate/BoostMEC, computed from
             the 30-mer (4 nt before the guide, the 20-nt guide, the 3-nt PAM, 3 nt after) BY THE AUTHORS' OWN PIPELINE, run unmodified in their Docker image on every guide of the
             cohort (`sgrna.cost.cost_boostmec_shipped`; `BOOSTMEC_FEATURES=shipped`, the default). `features()` below is a Python reimplementation of the same definitions, kept for the cost
             measurements and as `--features python`; it agrees with the shipped pipeline on every column except the four melting temperatures (Biopython against R's TmCalculator: 0.9
             to 1.9 degrees apart on average), which changes the released model's predictions by 0.34 on average on its 0-100 scale (Spearman 0.999). The features are: two RNAfold minimum free energies (the 20-nt guide, and the guide with the
             81-nt scaffold `GTTTTAGAGC...GCTTTTT` of that file), four nearest-neighbour melting temperatures (Tm_NN of TmCalculator 1.0.1: table DNA_NN4, 50 mM Na,
             25 nM strands, salt correction 5, of the 21-mer made of the guide and the first PAM base, and of its positions 1-4, 5-12 and 16-20), the number of G and C
             in the 30-mer, the number and the longest length of poly-T runs of 3 or more, the 28 position-specific nucleotides and 28 dinucleotides (the PAM's GG removed,
             as categorical features), and the counts (not frequencies) of every 1-, 2- and 3-mer, non-overlapping as `stringr::str_count` counts them. 2 + 4 + 1 + 2 + 28 +
             28 + 4 + 16 + 64 = 149.
  model      LightGBM regression with the hyperparameters of the released model (`model/model.txt`, whose 'parameters' block is quoted in HYPERPARAMETERS: learning rate
             0.00989607, 701 leaves, depth 12, 203 rows per leaf, 48.3% of columns per tree, L2 1.4826, 343 bins; its row subsampling is switched off in that file, bagging
             frequency 0). The released model has 1,582 trees, 136,142 leaves and a size of 11.5 MB (counted from that file).
  trees      the paper: ten-fold cross-validation with the same folds each time, tree limit 7,000, early stopping after 10 rounds on the validation fold, and the final
             model trained on all the training data 'using the average number of trees obtained in cross-validation'.
  search     the paper: Bayesian optimisation (rBayesianOptimization) with 10 initial points and 100 rounds, each hyperparameter set scored by the mean squared error of that ten-fold
             cross-validation, over the learning rate, depth, bins, leaves and the fractions of columns and rows. The search ranges are not published (Supplementary Table S4
             holds only the optimum), so the ranges here are mine, chosen wide around the released optimum; the optimiser is a Gaussian process with a Matern 5/2 kernel and the
             upper-confidence-bound rule at kappa 2.576, the rBayesianOptimization default, written here with scikit-learn.

Two arms, so that each of SLICER's two models has a counterpart that was given the same treatment:

  released   the released hyperparameters, the tree count by their cross-validation rule, on SLICER's default protocol (five-fold cross-validation with whole genes held out,
             three seeds; the 20-mer alone scores 0.547 and SLICER's default model 0.707 on these folds). Nothing is searched on E. coli: the counterpart of SLICER's default.
  searched   their search, run on the development genes of the SpCas9 screen (70% of genes, the same split SLICER's tuned settings were searched on), the final model fitted on
             all of them and scored on the 30% of genes held out of the search: the counterpart of SLICER's tuned model (0.7215 on those genes).

    ./run_ext.sh sgrna.compare.boostmec_replica released [--seeds 41 42 43]
    ./run_ext.sh sgrna.compare.boostmec_replica search
    ./run_ext.sh sgrna.compare.boostmec_replica compare
    ./run_ext.sh sgrna.compare.boostmec_replica report
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..common import INTERIM, OUT

SCAFFOLD = "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTTT"
NUC = ["A", "C", "G", "T"]
DI = [a + b for a in NUC for b in NUC]
TRI = [a + b + c for a in NUC for b in NUC for c in NUC]
HYPERPARAMETERS = dict(learning_rate=0.00989607, num_leaves=701, max_depth=12, min_child_samples=203, colsample_bytree=0.483247, reg_lambda=1.4826, reg_alpha=0.0,
                       max_bin=343, min_split_gain=0.0, subsample=0.134516, subsample_freq=0)          # bagging_fraction is set but bagging_freq is 0: row subsampling is off, as released
TREE_LIMIT, EARLY_STOP, INNER_FOLDS = 7000, 10, 10
RELEASED = dict(trees=1582, leaves=136142, bytes=11455146)       # counted from model/model.txt of the BoostMEC repository
FEATURES = os.environ.get("BOOSTMEC_FEATURES", "shipped")
SHIPPED_CSV = config.EXTERNAL / "BoostMEC" / "work" / "cohort-with-features.csv"
RES_RELEASED, RES_SEARCH, RES_TRACE = OUT / "boostmec_released.csv", OUT / "boostmec_searched.csv", OUT / "boostmec_search_trace.csv"


# ------------------------------------------------------------------------------------------------------------------------ features
def features(s30: str):    # noqa: E301 (also available as features_of)
    """The 149 values of one 30-mer: (continuous block of 93, categorical block of 56)."""
    import RNA
    from Bio.SeqUtils import MeltingTemp as mt

    assert len(s30) == 30 and set(s30) <= set("ACGT"), s30
    spacer = s30[4:24]
    tm = dict(nn_table=mt.DNA_NN4, tmm_table=mt.DNA_TMM1, imm_table=mt.DNA_IMM1, de_table=mt.DNA_DE1, dnac1=25, dnac2=25, Na=50, K=0, Tris=0, Mg=0, dNTPs=0, saltcorr=5)
    x21 = s30[4:25]
    cont = [float(RNA.fold(spacer.replace("T", "U"))[1]), float(RNA.fold((spacer + SCAFFOLD).replace("T", "U"))[1]),
            mt.Tm_NN(x21, **tm), mt.Tm_NN(x21[0:4], **tm), mt.Tm_NN(x21[4:12], **tm), mt.Tm_NN(x21[15:20], **tm),
            float(s30.count("G") + s30.count("C"))]
    runs = [len(r) for r in re.findall(r"TTT+", s30)]
    cont += [float(len(runs)), float(max(runs) if runs else 0)]
    for kmers in (NUC, DI, TRI):
        cont += [float(len(re.findall(k, s30))) for k in kmers]
    code = {b: j for j, b in enumerate(NUC)}
    mono = [code[s30[i]] for i in range(30) if i not in (25, 26)]                           # X26mono and X27mono (the PAM's GG) are dropped
    di = [code[s30[i]] * 4 + code[s30[i + 1]] for i in range(29) if i != 25]                # X26di is dropped, and the 30th 'dinucleotide' is a single base
    assert len(cont) == 93 and len(mono) == 28 and len(di) == 28
    return cont, mono + di


def _from_shipped(thirty, y, protos):
    """The 149 features as the authors' own pipeline wrote them (`cohort-with-features.csv`), in this module's layout: 93 continuous columns, then 28 + 28 category codes (0-based)."""
    d = pd.read_csv(SHIPPED_CSV, dtype={"x30mer": str})
    assert len(d) == len(thirty) and (d.x30mer.to_numpy() == np.asarray(thirty)).all(), "the shipped features are for other sequences or in another order"
    cont = ["grna_energy", "grna_scaffold_energy", "tm1", "tm2", "tm3", "tm4", "gc_count", "num_poly_t", "max_poly_t_len"] + NUC + DI + TRI
    code = {b: j for j, b in enumerate(NUC)}
    mono = [f"X{k}mono" for k in range(1, 31) if k not in (26, 27)]
    di = [f"X{k}di" for k in range(1, 30) if k != 26]
    cat = np.hstack([d[mono].apply(lambda c: c.map(code)).to_numpy(float), d[di].apply(lambda c: c.map(lambda q: 4 * code[q[0]] + code[q[1]])).to_numpy(float)])
    F = np.hstack([d[cont].to_numpy(float), cat])
    ms = float(pd.read_csv(OUT / "boostmec_zeroshot.csv").features_ms_per_guide.iloc[0]) if (OUT / "boostmec_zeroshot.csv").exists() else float("nan")   # the Python version's cost, for the log line
    return F, y, protos, ms


def build(features=None):
    from ..build.build_features import load_curated_index
    from ..cohorts.curated_ablation import _base

    features = features or FEATURES
    X, y, names, protos = _base()
    idx = load_curated_index(verbose=False).set_index(config.ID_COL).reindex(protos)
    thirty = [(up + sp + pam + dn).upper() for up, sp, pam, dn in zip(idx.upstream.str[-4:], idx.protospacer, idx.pam, idx.downstream.str[:3])]
    if features == "shipped":
        return _from_shipped(thirty, np.asarray(y, float), np.asarray(protos, dtype=object))
    t0 = time.time()
    num, cat = [], []
    for s30 in thirty:
        a, c = features_of(s30)
        num.append(a); cat.append(c)
    F = np.hstack([np.asarray(num, np.float64), np.asarray(cat, np.float64)])
    assert F.shape == (len(y), 149)
    return F, np.asarray(y, float), np.asarray(protos, dtype=object), 1000 * (time.time() - t0) / len(y)


features_of = features


# ------------------------------------------------------------------------------------------------------------------------ the model and its tree count
CAT = list(range(93, 149))


def _fit(F, y, params, n_trees, seed, valid=None, threads=1):
    import lightgbm as lgb

    m = lgb.LGBMRegressor(objective="regression", boosting_type="gbdt", n_estimators=int(n_trees), random_state=seed, n_jobs=threads, verbose=-1, **params)
    if valid is None:
        return m.fit(F, y, categorical_feature=CAT)
    return m.fit(F, y, eval_set=[valid], eval_metric="l2", categorical_feature=CAT, callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def _cv_trees(F, y, params, seed, parallel=True):
    """Ten-fold cross-validation as the paper does it: early stopping on the validation fold, tree limit 7,000. Returns (mean validation MSE, mean best iteration)."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold

    def one(tr, va):
        m = _fit(F[tr], y[tr], params, TREE_LIMIT, seed, valid=(F[va], y[va]))
        return float(np.mean((m.predict(F[va]) - y[va]) ** 2)), int(m.best_iteration_)
    folds = list(KFold(INNER_FOLDS, shuffle=True, random_state=2026).split(F))       # the same folds every time, as in the paper
    out = Parallel(n_jobs=INNER_FOLDS if parallel else 1)(delayed(one)(tr, va) for tr, va in folds)
    return float(np.mean([o[0] for o in out])), float(np.mean([o[1] for o in out]))


# ------------------------------------------------------------------------------------------------------------------------ arm 1: the released hyperparameters
def released(seeds=(41, 42, 43)) -> pd.DataFrame:
    from ..controls import sealed_holdout_gene as sg
    from ..controls.mappability_stratum import _groups

    F, y, protos, ms = build()
    print(f"  149 features for {len(y):,} guides: {ms:.2f} ms per guide (one thread, two RNAfold calls included)", flush=True)
    groups = _groups(protos)
    rows = []
    for seed in seeds:
        fold = sg._fold_ids(groups, 5, seed)
        for k in range(5):
            va = fold == k
            t0 = time.time()
            _, best = _cv_trees(F[~va], y[~va], HYPERPARAMETERS, seed)
            n = int(round(best))
            m = _fit(F[~va], y[~va], HYPERPARAMETERS, n, seed, threads=10)
            rho = float(spearmanr(y[va], m.predict(F[va])).statistic)
            leaves = int(sum(t["num_leaves"] for t in m.booster_.dump_model()["tree_info"]))
            rows.append(dict(seed=seed, fold=k + 1, rho=rho, trees=n, leaves=leaves, seconds=time.time() - t0, features_ms_per_guide=ms))
            print(f"  seed {seed} fold {k + 1}: rho {rho:.4f}  {n} trees  {leaves:,} leaves  {time.time() - t0:.0f} s", flush=True)
            pd.DataFrame(rows).to_csv(RES_RELEASED, index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------------------ arm 2: their search, on the development genes
SPACE = {"learning_rate": (0.005, 0.1, "log"), "max_depth": (3, 15, "int"), "max_bin": (32, 512, "int"), "num_leaves": (8, 1024, "int"),
         "feature_fraction": (0.1, 1.0, "lin"), "bagging_fraction": (0.1, 1.0, "lin"), "min_data_in_leaf": (10, 400, "int"), "lambda_l2": (0.0, 10.0, "lin")}


def _decode(u):
    p = {}
    for x, (name, (lo, hi, kind)) in zip(u, SPACE.items()):
        v = np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))) if kind == "log" else lo + x * (hi - lo)
        p[name] = int(round(v)) if kind == "int" else float(v)
    return dict(learning_rate=p["learning_rate"], max_depth=p["max_depth"], max_bin=p["max_bin"], num_leaves=p["num_leaves"], colsample_bytree=p["feature_fraction"],
                subsample=p["bagging_fraction"], subsample_freq=0, min_child_samples=p["min_data_in_leaf"], reg_lambda=p["lambda_l2"], reg_alpha=0.0, min_split_gain=0.0)


def search(n_init=10, n_iter=100, seed=2026) -> None:
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    from ..tuning.tune_ecoli import Cohort
    from ..cohorts.curated_ablation import _base, _groups

    co = Cohort()
    F, y, protos, _ = build()
    g, ok = _groups("gene", protos)
    assert (np.asarray(g[ok], dtype=object) == co.g).all()
    Fd, yd = F[ok][co.dev], y[ok][co.dev]
    print(f"  search on {len(yd):,} development guides, {len(SPACE)} hyperparameters, {n_init} + {n_iter} evaluations of {INNER_FOLDS}-fold cross-validation", flush=True)
    rng = np.random.default_rng(seed)
    U, scores, trace = [], [], []
    done = pd.read_csv(RES_TRACE) if RES_TRACE.exists() else pd.DataFrame()
    for _, r in done.iterrows():
        U.append(np.array([r[f"u_{k}"] for k in SPACE])); scores.append(-r.cv_mse); trace.append(r.to_dict())
    t_all = time.time()
    while len(U) < n_init + n_iter:
        if len(U) < n_init:
            u = rng.random(len(SPACE))
        else:
            gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(len(SPACE)), nu=2.5), alpha=1e-6, normalize_y=True, n_restarts_optimizer=5, random_state=seed)
            gp.fit(np.array(U), np.array(scores))
            cand = rng.random((20000, len(SPACE)))
            mu, sd = gp.predict(cand, return_std=True)
            u = cand[np.argmax(mu + 2.576 * sd)]
        t0 = time.time()
        mse, best = _cv_trees(Fd, yd, _decode(u), seed)
        U.append(u); scores.append(-mse)
        rec = dict(step=len(U), phase="init" if len(U) <= n_init else "bayes", cv_mse=mse, mean_best_iteration=best, seconds=time.time() - t0, **{f"u_{k}": float(v) for k, v in zip(SPACE, u)},
                   **{k: v for k, v in _decode(u).items()})
        trace.append(rec)
        pd.DataFrame(trace).to_csv(RES_TRACE, index=False)
        print(f"  step {len(U):3d}/{n_init + n_iter}  cv mse {mse:.4f} (best so far {-max(scores):.4f})  {best:.0f} trees  {time.time() - t0:.0f} s", flush=True)
    b = int(np.argmax(scores))
    params = _decode(U[b])
    n = int(round(pd.DataFrame(trace).iloc[b].mean_best_iteration))
    Fc, yc, gc = F[ok][co.conf], y[ok][co.conf], co.g[co.conf]
    rows = []
    for sd in (41, 42, 43):
        t0 = time.time()
        m = _fit(Fd, yd, params, n, sd, threads=10)
        p = m.predict(Fc)
        np.save(INTERIM / "tune" / f"conf_boostmec_searched_s{sd}.npy", p)
        leaves = int(sum(t["num_leaves"] for t in m.booster_.dump_model()["tree_info"]))
        rows.append(dict(seed=sd, rho=float(spearmanr(yc, p).statistic), trees=n, leaves=leaves, seconds=time.time() - t0, search_seconds=time.time() - t_all, n_confirm=len(yc),
                         best_step=b + 1, best_cv_mse=-scores[b], **params))
        print(f"  seed {sd}: rho on the held-out genes {rows[-1]['rho']:.4f}  ({n} trees, {leaves:,} leaves)", flush=True)
    pd.DataFrame(rows).to_csv(RES_SEARCH, index=False)


def compare(B=1000) -> None:
    """The searched arm against SLICER's default and tuned models on the same held-out genes (their saved predictions, three seeds), bootstrapped over genes as `tune_ecoli.confirm` does."""
    from ..tuning import tune_ecoli as T

    co = T.Cohort()
    sel = T.selected()
    arms = {"SLICER default": ("champion", int(sel.loc["champion", "k"])), "SLICER tuned": ("d099x", int(sel.loc["d099x", "k"]))}
    preds = {a: np.array([np.load(INTERIM / "tune" / f"conf_{nm}_k{k}_s{sd}.npy") for sd in (41, 42, 43)]) for a, (nm, k) in arms.items()}
    preds["BoostMEC, its search"] = np.array([np.load(INTERIM / "tune" / f"conf_boostmec_searched_s{sd}.npy") for sd in (41, 42, 43)])
    y, g = co.y[co.conf], co.g[co.conf]
    point, bs = T._boot_rho(y, g, preds, B=B)
    rows = []
    for a in preds:
        d = bs["BoostMEC, its search"] - bs[a]
        rows.append(dict(arm=a, n_guides=len(y), rho=point[a], rho_lo=np.percentile(bs[a], 2.5), rho_hi=np.percentile(bs[a], 97.5), boostmec_minus_arm=point["BoostMEC, its search"] - point[a],
                         diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "boostmec_vs_slicer_confirm.csv", index=False)
    print(out.round(4).to_string(index=False))


def report() -> None:
    r = pd.read_csv(RES_RELEASED)
    print(f"released hyperparameters: rho {r.rho.mean():.4f} +- {r.rho.std(ddof=1):.4f} over {len(r)} folds; trees {r.trees.mean():.0f}; leaves {r.leaves.mean():,.0f}; {r.seconds.mean():.0f} s per fold")
    if RES_SEARCH.exists():
        s = pd.read_csv(RES_SEARCH)
        print(s[["seed", "rho", "trees", "leaves", "search_seconds"]].round(4).to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["released", "search", "compare", "report"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[41, 42, 43])
    ap.add_argument("--features", choices=["shipped", "python"], default=None, help="shipped: the authors' own pipeline's features (default); python: this module's reimplementation")
    a = ap.parse_args(argv)
    if a.features:
        globals()["FEATURES"] = a.features
    {"released": lambda: released(tuple(a.seeds)), "search": search, "compare": compare, "report": report}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
