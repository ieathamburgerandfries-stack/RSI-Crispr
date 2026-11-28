"""BoostMEC, the closest published design to SLICER, scores a guide in 3.07 ms at a batch of 100 on one thread (42.3 times SLICER's default model) and, retrained on the E. coli cohort as its authors trained it, reaches rho 0.620 with its released hyperparameters and 0.623 after its own search, 0.087 below SLICER's default model with the released ones.

BoostMEC (Zarate et al., 2022, BMC Bioinformatics 23:446) is LightGBM on 149 engineered sequence features, chosen for interpretability; it reports no run time, memory or model size. Its repository
(github.com/oazarate/BoostMEC, commit cdf190c, cloned to `external_data/BoostMEC/repo`; nothing in it is executed here, only read) holds the released model (`model/model.txt`) and `model/rules.rds`, the
integer code of every category of every categorical feature in training, and the authors recommend their Docker image (`oazarate/boostmec:latest`). It is run here three ways, one `part` at a time, and every
part writes into `results/`:

  its pipeline as shipped, in their image   the Python script that starts RNAfold twice and two R scripts, run unmodified (1.2 GB, Ubuntu 18.04, R 4.1.0, ViennaRNA 2.4.14, Python 3.7.1; linux/amd64 only, so on this
                    Apple M5 it runs under emulation: the times are those of the shipped software on this laptop, not of a native x86 machine)
    inspect         the image, the host and the versions inside                                                                                      boostmec_shipped_image.txt
    check           100 guides: its features and predictions against the Python version below                                                         boostmec_shipped_check.csv
    time            whole-pipeline wall time at batches of 10, 100 and 10,000 (a batch of 1 fails as shipped; the attempt is recorded)               boostmec_shipped_timings.csv
    stages          the same, with an observer that logs how long each child program ran (the script itself is not changed)                          boostmec_shipped_stages.csv
    rnafold         RNAfold natively against inside the container, the one compiled program in the pipeline: the gauge of the emulation               boostmec_shipped_rnafold.csv
    cohort          all 33,567 cohort guides: its time, the released model's rho on E. coli, and the Python loader against its predictions            boostmec_shipped_cohort.csv

  the released model, in Python   its 149 features recomputed (`features`: ViennaRNA's Python binding for the two minimum free energies, Biopython for the four melting temperatures; no R and no RNAfold program
                    needed) and the model run through LightGBM on the released file, with the category codes read out of `rules.rds` by a small read-only parser of R's serialisation (`read_rds`, which executes nothing)
    oracle          the two worked examples of Supplementary Tables S2 and S3 (published predictions 49.53 and 28.42)                                  boostmec_oracle.csv
    size            trees, leaves and bytes of the released file                                                                                      model_size.csv
    inference       raw 30-mer to score on one thread, batches of 1, 100 and 10,000 with 300, 60 and 15 repeats, as `cost_repeats.py`                  cost_repeats_inference.csv
    fit             fits on the 26,853 training guides of fold 1 of SLICER's cost folds, wall and CPU seconds: 'final fit' is the refit alone, 'with its tree count' adds the ten-fold cross-validation that fixes
                    the number of trees (its ten fits run in parallel; the CPU seconds are their sum)                                                  cost_repeats_fit.csv

  retrained as its authors trained it   on the curated E. coli cohort with SLICER's folds (five-fold cross-validation with whole genes held out, three seeds; the 20-mer alone scores 0.547 and SLICER's default model 0.707 on these folds)
    released        the released hyperparameters (learning rate 0.00989607, 701 leaves, depth 12, 203 rows per leaf, 48.3% of columns per tree, L2 1.4826, 343 bins; row subsampling off) and the tree count by their
                    rule: ten-fold cross-validation with the same folds each time, tree limit 7,000, early stopping after 10 rounds, the final model on all the training data with the average tree count          boostmec_released.csv
    search          their Bayesian optimisation (10 initial points and 100 rounds, each set scored by the mean squared error of that ten-fold cross-validation) on the development genes of the SpCas9 screen (70% of
                    genes, the split SLICER's tuned settings were searched on), the final model scored on the 30% of genes held out of every search. The ranges are mine: Supplementary Table S4 holds only the optimum.
                    The optimiser is a Gaussian process with a Matern 5/2 kernel and the upper-confidence-bound rule at kappa 2.576, the rBayesianOptimization default, written with scikit-learn          boostmec_searched.csv, boostmec_search_trace.csv
    compare         the searched arm against SLICER's default and tuned models on those genes, bootstrapped over genes                                 boostmec_vs_slicer_confirm.csv
    report          prints the two arms

The 149 features (`model/featurize-dataset.R` and `model/boostmec-pipeline.py`), from the 30-mer (4 nt before the guide, the 20-nt guide, the 3-nt PAM, 3 nt after): two RNAfold minimum free energies (the 20-nt guide,
and the guide with the 81-nt scaffold `GTTTTAGAGC...GCTTTTT`), four nearest-neighbour melting temperatures (Tm_NN of TmCalculator 1.0.1: table DNA_NN4, 50 mM Na, 25 nM strands, salt correction 5, of the 21-mer made of
the guide and the first PAM base, and of its positions 1-4, 5-12 and 16-20), the number of G and C in the 30-mer, the number and the longest length of poly-T runs of 3 or more, the 28 position-specific nucleotides and 28
dinucleotides (the PAM's GG removed, as categorical features), and the counts (not frequencies) of every 1-, 2- and 3-mer, non-overlapping as `stringr::str_count` counts them: 2 + 4 + 1 + 2 + 28 + 28 + 4 + 16 + 64 = 149.
`released` and `search` use the features of the authors' own pipeline, run on every cohort guide (`cohort`; `BOOSTMEC_FEATURES=shipped`, the default); `--features python` uses `features`, which agrees with it on every
column except the four melting temperatures (Biopython against R's TmCalculator: 0.9 to 1.9 degrees apart on average), which changes the released model's predictions by 0.34 on average on its 0-100 scale (Spearman 0.999).
Timings follow `cost_repeats.py`: one thread, the load average beside every configuration, fits and inference into the same two files as the other models.

    ./run_ext.sh sgrna.compare.boostmec oracle                 # or: size, inference, fit
    ./run_ext.sh sgrna.compare.boostmec check                  # or: inspect, time [--reps 5 5 3], stages, rnafold, cohort
    ./run_ext.sh sgrna.compare.boostmec released [--seeds 41 42 43]            # or: search, compare, report
"""
from __future__ import annotations

import os
import sys

if sys.argv[1:2] == ["inference"]:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"

import argparse
import gzip
import hashlib
import json
import re
import struct
import subprocess
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..common import INTERIM, OUT

REPO = config.EXTERNAL / "BoostMEC" / "repo"
MODEL_DIR = REPO / "model"
WORK = config.EXTERNAL / "BoostMEC" / "work"          # inputs and outputs of the container runs: its own directory, apart from the clone
COMMIT = "cdf190c532ad244c9d451ce22a1cf83a6203c8ef"
IMAGE = "oazarate/boostmec:latest"
REPS = {1: 300, 100: 60, 10_000: 15}
PAPER_EXAMPLES = (("GTCTGCCATCTCTGATGGATGTGATGGGCA", 49.53, "S2, doench2014-Hs"), ("GGGGGGACTGTATCGACGCTGAATTGGGGG", 28.42, "S3, morenoMateos2015"))

SCAFFOLD = "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTTT"
NUC = ["A", "C", "G", "T"]
DI = [a + b for a in NUC for b in NUC]
TRI = [a + b + c for a in NUC for b in NUC for c in NUC]
HYPERPARAMETERS = dict(learning_rate=0.00989607, num_leaves=701, max_depth=12, min_child_samples=203, colsample_bytree=0.483247, reg_lambda=1.4826, reg_alpha=0.0,
                       max_bin=343, min_split_gain=0.0, subsample=0.134516, subsample_freq=0)          # bagging_fraction is set but bagging_freq is 0: row subsampling is off, as released
TREE_LIMIT, EARLY_STOP, INNER_FOLDS = 7000, 10, 10
RELEASED = dict(trees=1582, leaves=136142, bytes=11455146)       # counted from model/model.txt of the BoostMEC repository
FEATURES = os.environ.get("BOOSTMEC_FEATURES", "shipped")
SHIPPED_CSV = WORK / "cohort-with-features.csv"
RES_RELEASED, RES_SEARCH, RES_TRACE = OUT / "boostmec_released.csv", OUT / "boostmec_searched.csv", OUT / "boostmec_search_trace.csv"
BATCHES = (10, 100, 10_000)         # one guide cannot be scored: if a position-specific nucleotide column is T in every sequence, R's CSV reader makes it logical and `predict.R` stops in
                                    # lgb.convert_with_rules (8 such columns for one guide, 1 for two; the attempt is recorded in the timings file), so 10 is the smallest batch timed


# ------------------------------------------------------------------------------------------------------------------------
# the features, the model and its tree count (the retraining)
# ------------------------------------------------------------------------------------------------------------------------

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
    from ..harness.curated import load_base

    features = features or FEATURES
    X, y, names, protos = load_base()
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

# ------------------------------------------------------------------------------------------------------------------------
# arm 1: the released hyperparameters
# ------------------------------------------------------------------------------------------------------------------------

def released(seeds=(41, 42, 43)) -> pd.DataFrame:
    from ..harness import folds as sg
    from ..controls.mappability_stratum import _groups

    F, y, protos, ms = build()
    print(f"  149 features for {len(y):,} guides: {ms:.2f} ms per guide (one thread, two RNAfold calls included)", flush=True)
    groups = _groups(protos)
    rows = []
    for seed in seeds:
        fold = sg.fold_ids(groups, 5, seed)
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

# ------------------------------------------------------------------------------------------------------------------------
# arm 2: their search, on the development genes
# ------------------------------------------------------------------------------------------------------------------------

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
    from ..harness.curated import load_base, load_groups

    co = Cohort()
    F, y, protos, _ = build()
    g, ok = load_groups("gene", protos)
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

# ------------------------------------------------------------------------------------------------------------------------
# the released model, in Python: the parser of R's serialisation, the model with its encoding, the checks, the size
# ------------------------------------------------------------------------------------------------------------------------

NIL, REF = 254, 255
ALTREP = 238


class RdsReader:
    def __init__(self, b):
        self.b, self.p, self.refs = b, 0, []

    def i(self):
        v = struct.unpack(">i", self.b[self.p:self.p + 4])[0]; self.p += 4; return v

    def d(self):
        v = struct.unpack(">d", self.b[self.p:self.p + 8])[0]; self.p += 8; return v

    def item(self):
        flags = self.i()
        t, has_attr, has_tag = flags & 0xFF, bool(flags & 0x200), bool(flags & 0x400)
        if t == NIL or t == 242 or t == 241 or t == 253 or t == 252:       # NILVALUE, EMPTYENV, BASEENV, GLOBALENV, UNBOUNDVALUE
            return None
        if t == REF:
            k = flags >> 8
            if k == 0:
                k = self.i()
            return self.refs[k - 1]
        if t == 1:                                                            # SYMSXP
            s = self.item(); self.refs.append(s); return s
        if t == 2:                                                            # LISTSXP: a pairlist, read as a list of (tag, value)
            out = []
            while True:
                attr = self.item() if has_attr else None
                tag = self.item() if has_tag else None
                car = self.item()
                out.append((tag, car))
                flags = self.i()
                t2, has_attr, has_tag = flags & 0xFF, bool(flags & 0x200), bool(flags & 0x400)
                if t2 == NIL:
                    return out
                assert t2 == 2, f"unexpected cdr type {t2}"
        if t == 9:                                                            # CHARSXP
            n = self.i()
            if n == -1:
                return None
            s = self.b[self.p:self.p + n].decode("utf-8", "replace"); self.p += n; return s
        if t in (10, 13):                                                     # LGLSXP, INTSXP
            n = self.i(); v = [self.i() for _ in range(n)]
        elif t == 14:                                                         # REALSXP
            n = self.i(); v = [self.d() for _ in range(n)]
        elif t == 16:                                                         # STRSXP
            n = self.i(); v = [self.item() for _ in range(n)]
        elif t in (19, 20):                                                   # VECSXP, EXPRSXP
            n = self.i(); v = [self.item() for _ in range(n)]
        elif t == ALTREP:
            info = self.item(); state = self.item(); attr = self.item()
            cls = info[0][1] if isinstance(info, list) and info and isinstance(info[0], tuple) else info[0]
            cls = str(cls)
            if cls == "compact_intseq":
                n, start, step = state; v = [int(start + step * k) for k in range(int(n))]
            elif cls == "compact_realseq":
                n, start, step = state; v = [start + step * k for k in range(int(n))]
            elif cls == "deferred_string":
                arg = state[0] if isinstance(state, list) else state
                v = [str(int(x)) if float(x) == int(x) else str(x) for x in arg]
            else:
                raise ValueError("ALTREP class " + cls)
            return self._attach(v, attr)
        else:
            raise ValueError(f"type {t} at byte {self.p}")
        attr = self.item() if has_attr else None
        return self._attach(v, attr)

    @staticmethod
    def _attach(v, attr):
        names = None
        if isinstance(attr, list):
            for tag, val in attr:
                if tag == "names":
                    names = val
        return {"names": names, "values": v} if names is not None else v


def read_rds(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    assert raw[:2] == b"X\n", raw[:4]
    r = RdsReader(raw); r.p = 2
    version = r.i(); r.i(); r.i()
    if version == 3:
        n = r.i(); r.p += n
    return r.item()

class Released:
    """The released model with the encoding it was trained with."""

    def __init__(self):
        import lightgbm as lgb

        self.bst = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
        self.names = self.bst.feature_name()
        obj = read_rds(str(MODEL_DIR / "rules.rds"))
        rl = obj["values"][obj["names"].index("rules")]
        rules = {n: dict(zip(v["names"], v["values"])) for n, v in zip(rl["names"], rl["values"])}
        self.mono = [f"X{k}mono" for k in range(1, 31) if k not in (26, 27)]
        self.di = [f"X{k}di" for k in range(1, 30) if k != 26]
        assert self.names[9:37] == self.mono and self.names[37:65] == self.di, "column order of model.txt"
        # lookup tables from this repository's 0-based codes (A, C, G, T = 0..3; a dinucleotide = 4 * first + second) to the codes of the released model
        self.t_mono = np.full((28, 4), np.nan)
        self.t_di = np.full((28, 16), np.nan)
        for r, c in enumerate(self.mono):
            for j, b in enumerate(NUC):
                self.t_mono[r, j] = rules[c].get(b, np.nan)
        for r, c in enumerate(self.di):
            for j in range(16):
                self.t_di[r, j] = rules[c].get(NUC[j // 4] + NUC[j % 4], np.nan)

    def matrix(self, s30s) -> np.ndarray:
        """Raw 30-mers to the 149 columns in the order of the released model."""
        feats = [features(s) for s in s30s]
        cont = np.asarray([f[0] for f in feats], float)
        cat = np.asarray([f[1] for f in feats], int)
        mono = self.t_mono[np.arange(28), cat[:, :28]]
        di = self.t_di[np.arange(28), cat[:, 28:]]
        return np.hstack([cont[:, :9], mono, di, cont[:, 9:]])

    def encode(self, cont: np.ndarray, cat: np.ndarray) -> np.ndarray:
        """Already computed features (continuous block and 0-based categorical block, as `build` returns them) to the released layout."""
        return np.hstack([cont[:, :9], self.t_mono[np.arange(28), cat[:, :28]], self.t_di[np.arange(28), cat[:, 28:]], cont[:, 9:]])


def thirty_mers(n=None):
    """The 30-mers of the curated cohort (4 nt before the guide, guide, PAM, 3 nt after), in the cohort's order."""
    from .. import config
    from ..build.build_features import load_curated_index
    from ..harness.curated import load_base

    _, y, _, protos = load_base()
    idx = load_curated_index(verbose=False).set_index(config.ID_COL).reindex(protos)
    s = [(u[-4:] + p + m + d[:3]).upper() for u, p, m, d in zip(idx.upstream, idx.protospacer, idx.pam, idx.downstream)]
    return (s if n is None else s[:n]), np.asarray(y, float)

def oracle() -> None:
    R = Released()
    rows = []
    for s, paper, where in PAPER_EXAMPLES:
        pred = float(R.bst.predict(R.matrix([s]))[0])
        rows.append(dict(sequence=s, table=where, paper_prediction=paper, reproduced_prediction=round(pred, 3), difference=round(pred - paper, 3)))
        print(f"  {where}: reproduced {pred:.3f}, paper {paper}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "boostmec_oracle.csv", index=False)


def size() -> None:
    R = Released()
    d = R.bst.dump_model()
    leaves = int(sum(t["num_leaves"] for t in d["tree_info"]))
    raw = (MODEL_DIR / "model.txt").read_bytes()
    row = dict(model="BoostMEC (released)", trees=len(d["tree_info"]), nodes=np.nan, leaves=leaves, inner_nodes=leaves - len(d["tree_info"]), columns_used=np.nan, columns_selected=149,
               max_depth=12, bytes_file=len(raw), bytes_json=np.nan, parameters=np.nan, n_train=23902,
               bytes_gzip=len(gzip.compress(raw, 9)), sha256=hashlib.sha256(raw).hexdigest(), commit=COMMIT)
    print(f"  {row['trees']} trees, {leaves:,} leaves, {len(raw):,} bytes ({row['bytes_gzip']:,} gzipped)", flush=True)
    path = OUT / "model_size.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old[old.model != row["model"]] if len(old) else old, pd.DataFrame([row])]).to_csv(path, index=False)

# ------------------------------------------------------------------------------------------------------------------------
# timing: raw sequence to score on one thread, and the fits
# ------------------------------------------------------------------------------------------------------------------------

def inference() -> None:
    from ..cost.cost_repeats import INF, SESSION, _timings, load1, machine

    R = Released()
    s_all, _ = thirty_mers(max(REPS))
    machine(dict(model="BoostMEC (released)", lightgbm_threads=1))
    rows = []
    for b, reps in REPS.items():
        sub = s_all[:b]

        def run():
            return R.bst.predict(R.matrix(sub), num_threads=1)
        l0 = load1()
        ts = _timings(run, reps)
        l1 = load1()
        rows += [dict(model="BoostMEC", batch=b, rep=i, ms_per_guide=1000 * t / b, session=SESSION, load1_start=l0, load1_end=l1) for i, t in enumerate(ts)]
        print(f"  BoostMEC batch {b:6d}: median {np.median([r['ms_per_guide'] for r in rows if r['batch'] == b]):.3f} ms/guide", flush=True)
    old = pd.read_csv(INF) if INF.exists() else pd.DataFrame()
    pd.concat([old[old.model != "BoostMEC"] if len(old) else old, pd.DataFrame(rows)]).to_csv(INF, index=False)


def fit(reps: int = 5) -> None:
    import lightgbm as lgb
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold

    from ..cost.cost_repeats import SESSION, load1, machine, save_fit

    F, y, protos, ms = build()
    tr, _ = next(iter(KFold(5, shuffle=True, random_state=41).split(np.arange(len(y)))))
    Ftr, ytr = F[tr], y[tr]
    machine(dict(model="BoostMEC", n_train=len(tr)))

    def one(a, b, seed):
        c0 = time.process_time()
        m = _fit(Ftr[a], ytr[a], HYPERPARAMETERS, TREE_LIMIT, seed, valid=(Ftr[b], ytr[b]))
        return int(m.best_iteration_), time.process_time() - c0

    rows = []
    for i in range(reps):
        folds = list(KFold(INNER_FOLDS, shuffle=True, random_state=2026).split(Ftr))
        l0 = load1()
        w0 = time.perf_counter()
        out = Parallel(n_jobs=INNER_FOLDS)(delayed(one)(a, b, 41 + i) for a, b in folds)
        cv_wall = time.perf_counter() - w0
        n_trees = int(round(np.mean([o[0] for o in out])))
        cv_cpu = float(sum(o[1] for o in out))
        c0, w0 = time.process_time(), time.perf_counter()
        _fit(Ftr, ytr, HYPERPARAMETERS, n_trees, 41 + i, threads=os.cpu_count())
        f_cpu, f_wall = time.process_time() - c0, time.perf_counter() - w0
        l1 = load1()
        common = dict(rep=i, n_train=len(tr), session=SESSION, load1_start=l0, load1_end=l1, trees=n_trees)
        rows.append(dict(model="BoostMEC, final fit", cpu_seconds=f_cpu, wall_seconds=f_wall, **common))
        rows.append(dict(model="BoostMEC, with its tree count", cpu_seconds=cv_cpu + f_cpu, wall_seconds=cv_wall + f_wall, **common))
        print(f"  BoostMEC fit {i + 1}/{reps}: {n_trees} trees; final fit {f_wall:.1f} s wall, {f_cpu:.0f} CPU-s; with the ten-fold tree count {cv_wall + f_wall:.1f} s wall, {cv_cpu + f_cpu:.0f} CPU-s", flush=True)
        save_fit(rows)

# ------------------------------------------------------------------------------------------------------------------------
# their pipeline as shipped, in their Docker image
# ------------------------------------------------------------------------------------------------------------------------

OBSERVER = textwrap.dedent('''
    # sitecustomize.py: logs the duration of every child program of the pipeline; changes nothing it does
    import atexit, json, subprocess, time
    _log = []
    _Popen = subprocess.Popen
    class Popen(_Popen):
        def __init__(self, *a, **k):
            self._t0 = time.time(); self._name = None
            cmd = a[0] if a else k.get("args")
            if isinstance(cmd, (list, tuple)):
                self._name = " ".join(str(c).split("/")[-1] for c in cmd[:2])
            super().__init__(*a, **k)
        def wait(self, *a, **k):
            r = super().wait(*a, **k)
            if self._name and not getattr(self, "_logged", False):
                self._logged = True; _log.append([self._name, time.time() - self._t0])
            return r
    subprocess.Popen = Popen
    atexit.register(lambda: open("/work/stages.json", "w").write(json.dumps(_log)))
''')


def docker(cmd: str, env: dict | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    WORK.mkdir(parents=True, exist_ok=True)
    args = ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{WORK}:/work"]
    for k, v in (env or {}).items():
        args += ["-e", f"{k}={v}"]
    return subprocess.run(args + [IMAGE, "bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)


def repo_dir() -> str:
    p = WORK / "repo_dir.txt"
    if p.exists():
        return p.read_text().strip()
    out = docker("find / -name boostmec-pipeline.py -not -path '/proc/*' 2>/dev/null | head -1").stdout.strip()
    assert out, "boostmec-pipeline.py not found in the image"
    d = str(Path(out).parent.parent)
    p.write_text(d)
    return d


def inspect() -> None:
    info = subprocess.run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}} {{.Architecture}} {{.Size}} {{json .RepoDigests}}"], capture_output=True, text=True).stdout.strip()
    host = subprocess.run(["docker", "info", "--format", "{{.Architecture}} cpus={{.NCPU}} mem={{.MemTotal}} {{.OperatingSystem}}"], capture_output=True, text=True).stdout.strip()
    inside = docker("uname -m; cat /etc/os-release | head -2; R --version | head -1; RNAfold --version; python --version 2>&1; ls /proc/sys/fs/binfmt_misc 2>/dev/null | tr '\\n' ' '").stdout
    d = repo_dir()
    print(f"image: {info}\nhost VM: {host}\nrepo in the image: {d}\n{inside}")
    rec = OUT / "boostmec_shipped_image.txt"
    digest = info.split()[0]
    if rec.exists() and digest in rec.read_text():
        # The record of 24 Nov names this image by its digest and holds the 1.2 GB the report quotes; Docker 29 (containerd store) reports the unpacked 4.13 GB for the same image, so a rewrite would change a number the audit reads.
        print(f"  {rec.name} already records this image (same digest); not rewritten")
        return
    rec.write_text(f"image {info}\nhost VM {host}\nrepo {d}\n{inside}")


def write_input(b: int, name: str) -> Path:
    s, y = thirty_mers(b)
    p = WORK / f"{name}.csv"
    pd.DataFrame({"dataset": "ecoli", "x30mer": s[:b], "efficiency": y[:b]}).to_csv(p, index=False)
    return p


def run_pipeline(name: str, observe: bool = False) -> float:
    d = repo_dir()
    env = {"PYTHONPATH": "/work/obs"} if observe else {}
    if observe:
        (WORK / "obs").mkdir(exist_ok=True)
        (WORK / "obs" / "sitecustomize.py").write_text(OBSERVER)
    cmd = (f"cd {d} && TIMEFORMAT=%R && {{ time python model/boostmec-pipeline.py --file /work/{name}.csv > /work/{name}.log 2>&1 ; }} 2> /work/{name}.elapsed; "
           f"echo exit=$? >> /work/{name}.elapsed")
    r = docker(cmd, env=env)
    t = (WORK / f"{name}.elapsed").read_text().split()
    if r.returncode != 0 or not (WORK / f"{name}-boostmec-predictions.csv").exists():
        raise RuntimeError(f"pipeline failed on {name}: {r.stderr[-800:]} {(WORK / (name + '.log')).read_text()[-1500:]}")
    return float(t[0])


def check() -> None:
    """The shipped pipeline and the Python version on the same 100 E. coli 30-mers: features column by column, and predictions."""
    from scipy.stats import spearmanr

    n = 100
    p = write_input(n, "check_100")
    sec = run_pipeline("check_100")
    feats = pd.read_csv(WORK / "check_100-with-features.csv")
    pred = pd.read_csv(WORK / "check_100-boostmec-predictions.csv")
    R = Released()
    s30 = feats["x30mer"].tolist()
    X = R.matrix(s30)
    mine = R.bst.predict(X)
    theirs = pred["boostmec_predictions"].to_numpy()
    rows = [dict(item="pipeline seconds, 100 guides", value=sec)]
    rows += [dict(item="prediction: Spearman between the shipped pipeline and the Python version", value=float(spearmanr(theirs, mine).statistic)),
             dict(item="prediction: mean absolute difference", value=float(np.mean(np.abs(theirs - mine)))), dict(item="prediction: largest absolute difference", value=float(np.max(np.abs(theirs - mine))))]
    cols = R.names
    ours = pd.DataFrame(X, columns=cols)
    for c in cols[:9]:
        rows.append(dict(item=f"feature {c}: mean absolute difference", value=float(np.mean(np.abs(feats[c].to_numpy(float) - ours[c].to_numpy())))))
    counts = cols[65:]
    rows.append(dict(item="k-mer counts (84 columns): largest absolute difference", value=float(np.max(np.abs(feats[counts].to_numpy(float) - ours[counts].to_numpy())))))
    wrong = 0
    for c in cols[9:65]:                                  # the shipped file holds the letters; the column's position gives what they must be
        k = int(c[1:].replace("mono", "").replace("di", ""))
        want = [q[k - 1] if c.endswith("mono") else q[k - 1:k + 1] for q in s30]
        wrong += int(sum(a != b for a, b in zip(feats[c].astype(str), want)))
    rows.append(dict(item="categorical columns (56): cells that differ from the sequence", value=wrong))
    rows.append(dict(item="rows compared", value=len(s30)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "boostmec_shipped_check.csv", index=False)
    print(d.round(5).to_string(index=False))


def time_runs(reps=(5, 5, 3)) -> None:
    rows = []
    write_input(1, "in_1")
    try:
        run_pipeline("in_1")
        rows.append(dict(batch=1, rep=0, seconds=np.nan, ms_per_guide=np.nan, note="ran"))
    except RuntimeError as e:
        why = (WORK / "in_1.log").read_text().split("Backtrace")[0].strip().replace("\n", " ")
        rows.append(dict(batch=1, rep=0, seconds=np.nan, ms_per_guide=np.nan, note="fails: " + why))
        print("  batch 1 fails as shipped:", why[:160], flush=True)
    for b, r in zip(BATCHES, reps):
        write_input(b, f"in_{b}")
        for i in range(r):
            sec = run_pipeline(f"in_{b}")
            rows.append(dict(batch=b, rep=i, seconds=sec, ms_per_guide=1000 * sec / b))
            print(f"  batch {b:6d} rep {i + 1}/{r}: {sec:7.1f} s  ({1000 * sec / b:.2f} ms per guide)", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "boostmec_shipped_timings.csv", index=False)


def stages(b_list=BATCHES) -> None:
    rows = []
    for b in b_list:
        write_input(b, f"obs_{b}")
        sec = run_pipeline(f"obs_{b}", observe=True)
        st = json.loads((WORK / "stages.json").read_text())
        for name, t in st:
            rows.append(dict(batch=b, stage=name, seconds=t))
        rows.append(dict(batch=b, stage="whole pipeline", seconds=sec))
        print(f"  batch {b}: whole {sec:.1f} s; " + "; ".join(f"{n} {t:.1f}" for n, t in st), flush=True)
        pd.DataFrame(rows).to_csv(OUT / "boostmec_shipped_stages.csv", index=False)


def cohort() -> None:
    """The shipped pipeline on every guide of the cohort (its features feed `released` and `search`), its time, the released model's rho on E. coli, and the Python loader against its predictions."""
    from scipy.stats import spearmanr

    if not (WORK / "cohort-with-features.csv").exists():
        write_input(33_567, "cohort")
        run_pipeline("cohort")
    sec = float((WORK / "cohort.elapsed").read_text().split()[0])
    F, y, protos, ms = build("shipped")
    theirs = pd.read_csv(WORK / "cohort-boostmec-predictions.csv").boostmec_predictions.to_numpy()
    R = Released()
    mine = R.bst.predict(R.encode(F[:, :93], F[:, 93:].astype(int)))
    row = dict(n_guides=len(y), pipeline_seconds=sec, ms_per_guide=1000 * sec / len(y), rho_released_model_on_ecoli=float(spearmanr(y, theirs).statistic),
               spearman_python_loader_vs_shipped=float(spearmanr(mine, theirs).statistic), max_abs_diff_python_loader_vs_shipped=float(np.max(np.abs(mine - theirs))))
    pd.DataFrame([row]).to_csv(OUT / "boostmec_shipped_cohort.csv", index=False)
    print(pd.DataFrame([row]).T.to_string(header=False))


def rnafold(n=2000) -> None:
    """RNAfold on the same n guides (and on the same n guides with the scaffold): ViennaRNA 2.7.2 natively against 2.4.14 inside the container."""
    import RNA

    s, _ = thirty_mers(n)
    guides = [x[4:24] for x in s]
    rows = []
    for name, seqs in (("guide (20 nt)", guides), ("guide plus scaffold (101 nt)", [g + SCAFFOLD for g in guides])):
        t0 = time.perf_counter()
        for q in seqs:
            RNA.fold(q.replace("T", "U"))
        native = time.perf_counter() - t0
        fa = WORK / "rnafold.fa"
        fa.write_text("".join(f">g{i}\n{q}\n" for i, q in enumerate(seqs)))
        out = docker("TIMEFORMAT=%R; { time RNAfold --noPS -i /work/rnafold.fa --outfile=/work/rnafold.out ; } 2> /work/rnafold.elapsed; cat /work/rnafold.elapsed").stdout.strip().split()[-1]
        rows.append(dict(workload=name, sequences=len(seqs), native_seconds=native, container_seconds=float(out), ratio=float(out) / native))
        print(f"  {name}: native {native:.2f} s, in the container {float(out):.2f} s (x{float(out) / native:.2f})", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "boostmec_shipped_rnafold.csv", index=False)

# ------------------------------------------------------------------------------------------------------------------------ the command line
PARTS = dict(oracle=oracle, size=size, inference=inference, fit=fit, inspect=inspect, check=check, stages=stages, rnafold=rnafold, cohort=cohort, search=search, compare=compare, report=report)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=["released", "time", *PARTS])
    ap.add_argument("--seeds", type=int, nargs="*", default=[41, 42, 43], help="released: the seeds")
    ap.add_argument("--features", choices=["shipped", "python"], default=None, help="released, search: shipped = the authors' own pipeline's features (default); python = this module's reimplementation")
    ap.add_argument("--reps", type=int, nargs=3, default=[5, 5, 3], help="time: repeats at batches of 10, 100 and 10,000")
    a = ap.parse_args(argv)
    if a.features:
        globals()["FEATURES"] = a.features
    if a.part == "released":
        released(tuple(a.seeds))
    elif a.part == "time":
        time_runs(tuple(a.reps))
    else:
        PARTS[a.part]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
