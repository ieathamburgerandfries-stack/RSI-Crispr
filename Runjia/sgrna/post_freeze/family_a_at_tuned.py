"""Is family A still worth +0.164 once the model is tuned? The project's headline biology number, re-measured.

Why this exists. `+0.164 rho` (and `+0.156` on sealed genes) is the flanking-context increment, and every version of it was
measured with the *champion* configuration (depth 4, lr 0.05, 400 trees, 300 columns). The fork's search then found +0.016
from hyperparameters alone, so the report's headline accuracy number and its headline biology number now belong to two
different models. Nothing in `results/ext/` carried a base-only arm at the tuned settings: `tune_fullcv.csv` has four arms
and all of them include family A.

Two outcomes are possible and both are informative. If the tuned model extracts more from the 6,169 base columns, the flank
increment shrinks and part of what family A was credited with was really unexploited capacity in the guide representation.
If the extra capacity goes into the flank columns instead, the increment grows.

Design: the project's standard protocol, so the numbers line up with `tune_fullcv.csv` -- gene-grouped 5 folds x 3 seeds on
the 30,203 guides that carry a gene label, column selection fitted inside each training fold. The two matrices differ only
in whether family A's 348 columns are offered to the selector:

  base       the selector ranks the 6,169 base columns only; no arm can see a flank column
  base+A     the selector ranks all 6,517, as in every other run

**The fairness problem, and what is done about it.** The tuned configurations were chosen on DEV genes *with family A
present*, so applying them to a base-only matrix is not a like-for-like tuning effort and could understate the base-only
arm -- which would inflate family A's apparent increment. Two guards: four configurations are run on each matrix and the
best-of-four is reported beside the matched-config differences, and the champion row reproduces the published increment,
which checks the harness. A genuinely matched answer needs a search of its own on base-only columns; the best-of-four is a
bound, not a substitute, and the log says so.

    python -m sgrna.post_freeze.family_a_at_tuned              # 4 configs x 2 matrices x 15 folds
    python -m sgrna.post_freeze.family_a_at_tuned --seeds 41   # one seed, for a smoke test
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sgrna.cohorts.curated_ablation import _base, _block, _groups
from sgrna.harness.run_ablation import _impute, grouped_folds, make_selector

from . import tune_ecoli as T
from .common import OUT

ARMS = ("champion", "depth6", "d099x", "d077x")
MATRICES = ("base", "base+A")


class Cohort:
    """One matrix, base columns first, so a base-only arm is a restriction of the column ranking rather than a second copy."""

    def __init__(self):
        X, y, _, protos = _base()
        A, _ = _block("a_flank", protos)
        g, ok = _groups("gene", protos)
        self.n_base = X.shape[1]
        self.X = np.hstack([X, A])[ok]
        self.y = y[ok]
        self.g = np.asarray(g[ok], dtype=object)
        assert self.X.shape[1] == self.n_base + A.shape[1], "column count does not add up"
        print(f"cohort {self.X.shape}: {self.n_base:,} base + {A.shape[1]} family A columns, "
              f"{len(set(self.g)):,} genes", flush=True)


def _ranking(co, tr, va, seed, matrix):
    """Top-900 column ranking fitted on the training rows of this fold, restricted to the matrix's own columns."""
    hi = co.n_base if matrix == "base" else co.X.shape[1]
    A, _ = _impute(co.X[tr][:, :hi], co.X[va][:1, :hi])
    sel = make_selector(seed)
    sel.fit(A, co.y[tr])
    top = np.argsort(sel.feature_importances_)[::-1][:900]
    assert top.max() < hi, "the base-only ranking reached a family A column"
    return top


def run(seeds=(41, 42, 43), n_splits=5):
    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    sel = T.selected()
    co = Cohort()
    f_out = OUT / "family_a_at_tuned.csv"
    rows = pd.read_csv(f_out).to_dict("records") if f_out.exists() else []
    done = {(r["matrix"], r["arm"], r["seed"], r["fold"]) for r in rows}
    t0 = time.time()
    for seed in seeds:
        for fold, (tr, va) in enumerate(grouped_folds(co.g, n_splits, seed)):
            for matrix in MATRICES:
                if all((matrix, a, seed, fold) in done for a in ARMS):
                    continue
                top = _ranking(co, tr, va, seed, matrix)
                for arm in ARMS:
                    if (matrix, arm, seed, fold) in done:
                        continue
                    k = int(sel.loc[arm, "k"])
                    cfg = dict(cfgs[arm], n_estimators=k)
                    cols = top[:int(cfg["nf"])]
                    A, B = _impute(co.X[tr][:, cols], co.X[va][:, cols])
                    p = T.fit_predict(cfg, A, co.y[tr], B, seed)[k]
                    rows.append(dict(matrix=matrix, arm=arm, seed=seed, fold=fold, n_train=len(tr), n_test=len(va),
                                     n_flank_cols=int((cols >= co.n_base).sum()),
                                     rho=float(spearmanr(co.y[va], p).statistic)))
                    pd.DataFrame(rows).to_csv(f_out, index=False)
                print(f"  seed {seed} fold {fold} {matrix:7s}: "
                      + "  ".join(f"{r['arm']}={r['rho']:.4f}" for r in rows[-len(ARMS):])
                      + f"   ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def summarise(B=4000, seed=0):
    """Family A's increment per configuration, paired over the 15 folds, plus the best-of-four bound."""
    d = pd.read_csv(OUT / "family_a_at_tuned.csv")
    piv = d.pivot_table(index=["seed", "fold"], columns=["matrix", "arm"], values="rho")
    rng = np.random.default_rng(seed)
    rows = []

    def boot(diff):
        v = diff.to_numpy(float)
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
        return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    for arm in ARMS:
        base, both = piv[("base", arm)], piv[("base+A", arm)]
        m, lo, hi = boot(both - base)
        rows.append(dict(comparison=f"family A increment, {arm}", base=base.mean(), with_a=both.mean(),
                         delta=m, lo=lo, hi=hi, folds_positive=int((both > base).sum()), n_folds=len(base)))
    # best-of-four on each side: the bound that does not assume the tuned settings suit a base-only matrix
    b_base = piv["base"][list(ARMS)].mean().idxmax()
    b_both = piv["base+A"][list(ARMS)].mean().idxmax()
    m, lo, hi = boot(piv[("base+A", b_both)] - piv[("base", b_base)])
    rows.append(dict(comparison=f"family A increment, best-of-four ({b_both} over {b_base})",
                     base=piv[("base", b_base)].mean(), with_a=piv[("base+A", b_both)].mean(),
                     delta=m, lo=lo, hi=hi, folds_positive=int((piv[("base+A", b_both)] > piv[("base", b_base)]).sum()),
                     n_folds=len(piv)))
    # and the tuning gain on each matrix, which is the other half of the question
    for matrix in MATRICES:
        m, lo, hi = boot(piv[(matrix, "d099x")] - piv[(matrix, "champion")])
        rows.append(dict(comparison=f"tuning gain (d099x - champion) on {matrix}",
                         base=piv[(matrix, "champion")].mean(), with_a=piv[(matrix, "d099x")].mean(),
                         delta=m, lo=lo, hi=hi,
                         folds_positive=int((piv[(matrix, "d099x")] > piv[(matrix, "champion")]).sum()), n_folds=len(piv)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "family_a_at_tuned_summary.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[41, 42, 43])
    ap.add_argument("--summarise-only", action="store_true")
    a = ap.parse_args()
    if not a.summarise_only:
        run(tuple(a.seeds))
    pd.set_option("display.width", 220)
    print(summarise().round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
