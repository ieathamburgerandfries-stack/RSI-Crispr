"""At the tuned settings family A is worth +0.1697 rho (up from +0.1563 at the champion's), and the inner +-50 nt as explicit one-hot columns add +0.0002 on top of it: the bound on any local shape predictor.

Two questions with one machinery: take the tuned configurations of `tuning.tune_ecoli` and ask what a block of columns adds, by running the same gene-grouped 5 folds x 3 seeds
(the 30,203 guides that carry a gene label), a top-900 column ranking fitted inside each training fold, the arm's own selector budget, and a paired bootstrap over folds. The
matrices differ only in which columns the selector may rank. Resumable row by row (`results/family_a_at_tuned.csv`, `local_sequence_bound.csv`).

    ./run_ext.sh sgrna.cohorts.tuned_ablation --design family_a                    # 4 configs x 2 matrices x 15 folds
    ./run_ext.sh sgrna.cohorts.tuned_ablation --design local_sequence --arms d099x # 2 arms x 3 matrices x 15 folds
    ./run_ext.sh sgrna.cohorts.tuned_ablation --design family_a --seeds 41         # one seed, a smoke test
    ./run_ext.sh sgrna.cohorts.tuned_ablation --design local_sequence --summarise-only

**--design family_a** (`family_a_at_tuned*.csv`). `+0.164 rho` (and `+0.156` on sealed genes) is the flanking-context increment, and every version of it was measured with the
*champion* configuration (depth 4, lr 0.05, 400 trees, 300 columns). The search then found +0.016 from hyperparameters alone, so the headline accuracy and the headline biology number
belonged to two different models; nothing carried a base-only arm at the tuned settings (`tune_fullcv.csv` has four arms and all include family A). Two outcomes are informative: if
the tuned model extracts more from the 6,169 base columns the flank increment shrinks, and if the extra capacity goes into the flank columns it grows. Matrices: `base` (the selector
ranks the 6,169 base columns only; no arm can see a flank column) and `base+A` (all 6,517, as in every other run).
*The fairness problem.* The tuned configurations were chosen on DEV genes with family A present, so applying them to a base-only matrix is not a like-for-like tuning effort and could
understate the base-only arm, which would inflate family A's increment. Two guards: four configurations are run on each matrix and the best-of-four is reported beside the matched
differences, and the champion row reproduces the published increment, which checks the harness. A genuinely matched answer needs a search of its own on base-only columns; the
best-of-four is a bound, not a substitute.

**--design local_sequence** (`local_sequence_bound*.csv`). An upper bound on what any local shape or stiffness predictor can add, the open item Deep DNAshape left behind. Appendix B4
tested tabulated shape (a 5 bp pentamer table) and found nothing on top of family A; Deep DNAshape, which reads up to 7 bp, was cloned but never run (the sandbox refuses to execute
third-party model code). The objection that the next shape predictor might work can be closed without running anyone's model: its output at a position is a deterministic function of
the 7 nucleotides around it, so over the inner flank its entire output is a function of the local sequence, and if the model is given that sequence explicitly and gains nothing, no
function of it can gain anything either. Matrices: `base+A` (the reference: family A already one-hot encodes u10..u1, the PAM's N and d1..d15), `base+A+loc50` (plus the one-hot
identity of every position family A does not reach out to +-50 nt: u50..u11 and d16..d50, 75 positions x 4 = 300 columns) and `base+A+loc50shuf` (the same 300 columns permuted across
guides, the budget control, because 300 new candidates could dilute a fixed selector budget whatever they contain). The bound does not cover the wide windows, where a shape predictor
would enter as a window average; Appendix B4 measured rho 0.998 between window tensor means and flank GC, so averaging collapses them onto composition. Run at the tuned configuration
as well as the champion, because the champion under-uses flank columns and a bound measured at a capacity that cannot use the columns would be worthless.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sgrna import config
from sgrna.cohorts.curated_ablation import _base, _block, _groups
from sgrna.harness.run_ablation import _impute, grouped_folds, make_selector

from ..tuning import tune_ecoli as T
from ..build.a_flank_fast import _codes
from ..common import OUT

REACH = 50                              # how far out the explicit one-hot window goes, each side
DESIGNS = {
    "family_a": dict(matrices=("base", "base+A"), arms=("champion", "depth6", "d099x", "d077x"),
                     rows="family_a_at_tuned.csv", summary="family_a_at_tuned_summary.csv"),
    "local_sequence": dict(matrices=("base+A", "base+A+loc50", "base+A+loc50shuf"), arms=("d099x", "champion"),
                           rows="local_sequence_bound.csv", summary="local_sequence_bound_summary.csv"),
}


def local_block(protos: list[str], reach: int = REACH, seed: int = 0):
    """One-hot identity of the flank positions family A does not reach, and the same columns permuted across guides.

    Positions u{reach}..u11 and d16..d{reach}: the ones family A leaves out, so the two arms together encode the whole
    +-`reach` window exactly once.
    """
    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv").set_index("sgRNAID")
    sub = idx.reindex(list(protos))
    assert sub.upstream.notna().all() and sub.downstream.notna().all(), "a curated guide has no stored flank"
    up = _codes(sub.upstream.astype(str).tolist(), reach, True)      # last `reach` nt, outermost first
    dn = _codes(sub.downstream.astype(str).tolist(), reach, False)   # first `reach` nt, innermost first
    cols, names = [], []
    for i in range(reach, 10, -1):                                   # u{reach} .. u11
        cols.append(up[:, reach - i]); names.append(f"loc.seq.u{i}")
    for i in range(16, reach + 1):                                   # d16 .. d{reach}
        cols.append(dn[:, i - 1]); names.append(f"loc.seq.d{i}")
    code = np.stack(cols, axis=1)
    M = (code[:, :, None] == np.arange(4)).reshape(len(protos), -1).astype(np.float32)
    nm = [f"{n}.{b}" for n in names for b in "ACGT"]
    rng = np.random.default_rng(seed)
    S = M[rng.permutation(len(M))]                                   # whole rows permuted: composition kept, link broken
    return M, S, nm


class Cohort:
    """base | family A, and for the local-sequence design the local one-hot block and its shuffle after them, so each matrix is a prefix plus a slice."""

    def __init__(self, design: str):
        X, y, _, protos = _base()
        A, _ = _block("a_flank", protos)
        blocks = [X, A]
        self.n_loc = 0
        if design == "local_sequence":
            M, S, _ = local_block(list(protos))
            blocks += [M, S]
            self.n_loc = M.shape[1]
        g, ok = _groups("gene", protos)
        self.n_base, self.n_a = X.shape[1], A.shape[1]
        self.X = np.hstack(blocks)[ok]
        self.y = y[ok]
        self.g = np.asarray(g[ok], dtype=object)
        assert self.X.shape[1] == self.n_base + self.n_a + 2 * self.n_loc, "column count does not add up"
        print(f"cohort {self.X.shape}: {self.n_base:,} base + {self.n_a} family A + {self.n_loc} local "
              f"(+{self.n_loc} shuffled), {len(set(self.g)):,} genes", flush=True)

    def columns(self, matrix: str) -> np.ndarray:
        """Column indices the selector may rank for this matrix."""
        if matrix == "base":
            return np.arange(self.n_base)
        core = np.arange(self.n_base + self.n_a)
        if matrix == "base+A":
            return core
        lo = self.n_base + self.n_a + (self.n_loc if matrix.endswith("shuf") else 0)
        return np.concatenate([core, np.arange(lo, lo + self.n_loc)])


def _ranking(co: Cohort, tr, va, seed, matrix):
    """Top-900 column ranking fitted on the training rows of this fold, restricted to the matrix's own columns."""
    cand = co.columns(matrix)
    A, _ = _impute(co.X[tr][:, cand], co.X[va][:1, cand])
    sel = make_selector(seed)
    sel.fit(A, co.y[tr])
    top = cand[np.argsort(sel.feature_importances_)[::-1][:900]]
    if matrix == "base":
        assert top.max() < co.n_base, "the base-only ranking reached a family A column"
    return top


def _boot(v: np.ndarray, rng, B: int) -> np.ndarray:
    return np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)])


def run(design: str, arms=None, seeds=(41, 42, 43), n_splits=5, max_folds=None):
    d = DESIGNS[design]
    arms = tuple(arms or d["arms"])
    out_csv = OUT / d["rows"]
    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    sel = T.selected()
    co = Cohort(design)
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    done = {(r["matrix"], r["arm"], r["seed"], r["fold"]) for r in rows}
    t0 = time.time()
    for seed in seeds:
        for fold, (tr, va) in enumerate(grouped_folds(co.g, n_splits, seed)):
            if max_folds is not None and fold >= max_folds:
                break
            for matrix in d["matrices"]:
                if all((matrix, a, seed, fold) in done for a in arms):
                    continue
                top = _ranking(co, tr, va, seed, matrix)
                for arm in arms:
                    if (matrix, arm, seed, fold) in done:
                        continue
                    k = int(sel.loc[arm, "k"])
                    cfg = dict(cfgs[arm], n_estimators=k)
                    cols = top[:int(cfg["nf"])]
                    A, B = _impute(co.X[tr][:, cols], co.X[va][:, cols])
                    p = T.fit_predict(cfg, A, co.y[tr], B, seed)[k]
                    row = dict(matrix=matrix, arm=arm, seed=seed, fold=fold, n_train=len(tr), n_test=len(va))
                    if design == "family_a":
                        row["n_flank_cols"] = int((cols >= co.n_base).sum())
                    else:
                        row.update(n_sel=len(cols), n_local_selected=int((cols >= co.n_base + co.n_a).sum()))
                    row["rho"] = float(spearmanr(co.y[va], p).statistic)
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(f"  seed {seed} fold {fold} {matrix:20s}: "
                      + "  ".join(f"{r['arm']}={r['rho']:.4f}" + (f" (loc {r['n_local_selected']})" if "n_local_selected" in r else "") for r in rows[-len(arms):])
                      + f"   ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def summarise(design: str, B=4000, seed=0):
    """family_a: family A's increment per configuration, paired over the 15 folds, the best-of-four bound, and the tuning gain on each matrix.
    local_sequence: each matrix against base+A, paired over folds, with a bootstrap interval on the paired difference."""
    d = pd.read_csv(OUT / DESIGNS[design]["rows"])
    piv = d.pivot_table(index=["seed", "fold"], columns=["matrix", "arm"], values="rho")
    rng = np.random.default_rng(seed)
    rows = []
    if design == "family_a":
        arms, matrices = DESIGNS[design]["arms"], DESIGNS[design]["matrices"]

        def boot(diff):
            v = diff.to_numpy(float)
            bs = _boot(v, rng, B)
            return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

        for arm in arms:
            base, both = piv[("base", arm)], piv[("base+A", arm)]
            m, lo, hi = boot(both - base)
            rows.append(dict(comparison=f"family A increment, {arm}", base=base.mean(), with_a=both.mean(),
                             delta=m, lo=lo, hi=hi, folds_positive=int((both > base).sum()), n_folds=len(base)))
        # best-of-four on each side: the bound that does not assume the tuned settings suit a base-only matrix
        b_base = piv["base"][list(arms)].mean().idxmax()
        b_both = piv["base+A"][list(arms)].mean().idxmax()
        m, lo, hi = boot(piv[("base+A", b_both)] - piv[("base", b_base)])
        rows.append(dict(comparison=f"family A increment, best-of-four ({b_both} over {b_base})",
                         base=piv[("base", b_base)].mean(), with_a=piv[("base+A", b_both)].mean(),
                         delta=m, lo=lo, hi=hi, folds_positive=int((piv[("base+A", b_both)] > piv[("base", b_base)]).sum()),
                         n_folds=len(piv)))
        # and the tuning gain on each matrix, which is the other half of the question
        for matrix in matrices:
            m, lo, hi = boot(piv[(matrix, "d099x")] - piv[(matrix, "champion")])
            rows.append(dict(comparison=f"tuning gain (d099x - champion) on {matrix}",
                             base=piv[(matrix, "champion")].mean(), with_a=piv[(matrix, "d099x")].mean(),
                             delta=m, lo=lo, hi=hi,
                             folds_positive=int((piv[(matrix, "d099x")] > piv[(matrix, "champion")]).sum()), n_folds=len(piv)))
    else:
        for arm in sorted({a for _, a in piv.columns}):
            if ("base+A", arm) not in piv:
                continue
            ref = piv[("base+A", arm)]
            for matrix in DESIGNS[design]["matrices"][1:]:
                if (matrix, arm) not in piv:
                    continue
                v = (piv[(matrix, arm)] - ref).dropna().to_numpy(float)
                bs = _boot(v, rng, B)
                sub = d[(d.matrix == matrix) & (d.arm == arm)]
                rows.append(dict(arm=arm, matrix=matrix, base_rho=float(ref.mean()),
                                 rho=float(piv[(matrix, arm)].mean()), delta=float(v.mean()),
                                 lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                                 folds_positive=int((v > 0).sum()), n_folds=len(v),
                                 local_cols_selected=float(sub.n_local_selected.mean())))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / DESIGNS[design]["summary"], index=False)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", choices=sorted(DESIGNS), required=True)
    ap.add_argument("--arms", help="comma-separated configurations of tune_ecoli (default: the design's own)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[41, 42, 43])
    ap.add_argument("--folds", type=int, default=None, help="stop after this many folds per seed (smoke test)")
    ap.add_argument("--summarise-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.summarise_only:
        run(a.design, tuple(x for x in a.arms.split(",") if x) if a.arms else None, tuple(a.seeds), max_folds=a.folds)
    pd.set_option("display.width", 220)
    print(summarise(a.design).round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
