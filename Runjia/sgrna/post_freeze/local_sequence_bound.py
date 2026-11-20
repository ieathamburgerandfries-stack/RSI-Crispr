"""An upper bound on what any local shape or stiffness predictor can add -- the open item Deep DNAshape left behind.

The gap this closes
-------------------
Finding 8 tested tabulated DNA shape (DNAshapeR's pentamer table, 5 bp of context) and found nothing on top of family A.
Deep DNAshape, which reads up to **7 bp** of context, was cloned but never run: the sandbox refuses to execute third-party
model code, and it is not prudent to run a downloaded model anyway. So the report has to say "the wider context is what was
not tested", which leaves the finding open to the obvious objection that the next shape predictor along might work.

That objection can be closed without running anyone's model, because of what a shape predictor *is*. Deep DNAshape's output
at a position is a deterministic function of the 7 nucleotides around it. Over the inner flank, therefore, its entire output
is a function of the local sequence -- so if the model is given that sequence explicitly and gains nothing, no function of it
can gain anything either. That is the bound measured here:

    base+A                 the reference: family A already one-hot encodes u10..u1, the PAM's N, and d1..d15
    base+A+loc50           plus the one-hot identity of every position family A does not reach out to +-50 nt
                           (u50..u11 and d16..d50; 75 positions x 4 = 300 columns)
    base+A+loc50[shuffled] the same 300 columns, each permuted across guides -- the budget control, because 300 new
                           candidates could dilute a fixed selector budget whatever they contain

If `loc50` lands on `base+A`, then every deterministic function of the sequence in +-50 nt -- Deep DNAshape, any k<=7-mer
shape or stiffness table, any local physical model -- is bounded by that same difference. What the bound does *not* cover is
the wide windows, where a shape predictor would enter as a window average; Finding 8 covers that case directly and measured
rho 0.998 between window tensor means and flank GC, so averaging collapses them onto composition.

Protocol is the project's standard, and deliberately identical to `family_a_at_tuned` so the numbers line up with it:
gene-grouped 5 folds x 3 seeds on the guides that carry a gene label, a top-900 column ranking fitted inside each training
fold, the arm's own selector budget. Run at the tuned configuration as well as the champion, because Finding 4 showed the
champion under-uses flank columns and a bound measured at a capacity that cannot use the columns would be worthless.

    ./run_ext.sh sgrna.post_freeze.local_sequence_bound --arms d099x          # the capacity that can use flank columns
    ./run_ext.sh sgrna.post_freeze.local_sequence_bound --seeds 41 --folds 1  # smoke test
    ./run_ext.sh sgrna.post_freeze.local_sequence_bound --summarise-only
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

from . import tune_ecoli as T
from .a_flank_fast import _codes
from .common import OUT

REACH = 50                              # how far out the explicit one-hot window goes, each side
MATRICES = ("base+A", "base+A+loc50", "base+A+loc50shuf")
OUTCSV = OUT / "local_sequence_bound.csv"
SUMCSV = OUT / "local_sequence_bound_summary.csv"


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
    """base | family A | the local one-hot block | its shuffle, in that order, so each matrix is a prefix-plus-slice."""

    def __init__(self):
        X, y, _, protos = _base()
        A, _ = _block("a_flank", protos)
        M, S, nm = local_block(list(protos))
        g, ok = _groups("gene", protos)
        self.n_base, self.n_a, self.n_loc = X.shape[1], A.shape[1], M.shape[1]
        self.X = np.hstack([X, A, M, S])[ok]
        self.y = y[ok]
        self.g = np.asarray(g[ok], dtype=object)
        self.names = nm
        assert self.X.shape[1] == self.n_base + self.n_a + 2 * self.n_loc, "column count does not add up"
        print(f"cohort {self.X.shape}: {self.n_base:,} base + {self.n_a} family A + {self.n_loc} local "
              f"(+{self.n_loc} shuffled), {len(set(self.g)):,} genes", flush=True)

    def columns(self, matrix: str) -> np.ndarray:
        """Column indices the selector may rank for this matrix."""
        core = np.arange(self.n_base + self.n_a)
        if matrix == "base+A":
            return core
        lo = self.n_base + self.n_a + (self.n_loc if matrix.endswith("shuf") else 0)
        return np.concatenate([core, np.arange(lo, lo + self.n_loc)])


def _ranking(co: Cohort, tr, va, seed, matrix):
    cand = co.columns(matrix)
    A, _ = _impute(co.X[tr][:, cand], co.X[va][:1, cand])
    sel = make_selector(seed)
    sel.fit(A, co.y[tr])
    return cand[np.argsort(sel.feature_importances_)[::-1][:900]]


def run(arms=("d099x", "champion"), seeds=(41, 42, 43), n_splits=5, max_folds=None):
    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    sel = T.selected()
    co = Cohort()
    rows = pd.read_csv(OUTCSV).to_dict("records") if OUTCSV.exists() else []
    done = {(r["matrix"], r["arm"], r["seed"], r["fold"]) for r in rows}
    t0 = time.time()
    for seed in seeds:
        for fold, (tr, va) in enumerate(grouped_folds(co.g, n_splits, seed)):
            if max_folds is not None and fold >= max_folds:
                break
            for matrix in MATRICES:
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
                    rows.append(dict(matrix=matrix, arm=arm, seed=seed, fold=fold, n_train=len(tr), n_test=len(va),
                                     n_sel=len(cols),
                                     n_local_selected=int((cols >= co.n_base + co.n_a).sum()),
                                     rho=float(spearmanr(co.y[va], p).statistic)))
                    pd.DataFrame(rows).to_csv(OUTCSV, index=False)
                print(f"  seed {seed} fold {fold} {matrix:20s}: "
                      + "  ".join(f"{r['arm']}={r['rho']:.4f} (loc {r['n_local_selected']})" for r in rows[-len(arms):])
                      + f"   ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def summarise(B=4000, seed=0):
    """Each matrix against base+A, paired over folds, with a bootstrap interval on the paired difference."""
    d = pd.read_csv(OUTCSV)
    piv = d.pivot_table(index=["seed", "fold"], columns=["matrix", "arm"], values="rho")
    rng = np.random.default_rng(seed)
    rows = []
    for arm in sorted({a for _, a in piv.columns}):
        if ("base+A", arm) not in piv:
            continue
        ref = piv[("base+A", arm)]
        for matrix in MATRICES[1:]:
            if (matrix, arm) not in piv:
                continue
            v = (piv[(matrix, arm)] - ref).dropna().to_numpy(float)
            bs = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)])
            sub = d[(d.matrix == matrix) & (d.arm == arm)]
            rows.append(dict(arm=arm, matrix=matrix, base_rho=float(ref.mean()),
                             rho=float(piv[(matrix, arm)].mean()), delta=float(v.mean()),
                             lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                             folds_positive=int((v > 0).sum()), n_folds=len(v),
                             local_cols_selected=float(sub.n_local_selected.mean())))
    out = pd.DataFrame(rows)
    out.to_csv(SUMCSV, index=False)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", default="d099x,champion")
    ap.add_argument("--seeds", type=int, nargs="*", default=[41, 42, 43])
    ap.add_argument("--folds", type=int, default=None, help="stop after this many folds per seed (smoke test)")
    ap.add_argument("--summarise-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.summarise_only:
        run(tuple(x for x in a.arms.split(",") if x), tuple(a.seeds), max_folds=a.folds)
    pd.set_option("display.width", 220)
    print(summarise().round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
