"""Tuning the two flank windows, upstream and downstream, separately.

`flank_scale.py` gave the gain of one symmetric pair of columns (the GC fraction of the W nt upstream of the protospacer and of the W nt downstream of the PAM) at nine
windows from 10 nt to 5 kb: highest at 250 nt, within 0.01 of that from 100 to 500 nt, on a grid with a factor of two or more between neighbours and with the same W on both
sides. This module asks the same question on a finer grid and lets the two sides differ. Same protocol as there (gene-grouped five-fold cross-validation over the 33,567 curated
guides, the default model with nothing searched, seed 41, the 20-mer alone as the base), so every arm is paired by fold with `flank_scale.csv`.

  A  symmetric   W = 75, 150, 200, 300, 400 added to the nine of `flank_scale.py`
  B  one side    the upstream window U over a grid with the downstream window held at 250 nt, and the downstream window D with the upstream held at 250 nt
  C  refinement  the grid cell with the best mean rho after A and B, and its neighbours on the grid in both directions (a 3 x 3 block)
  D  replication the best cell, 250/250 and the base on seeds 42 and 43 (new fold assignments of the same genes): the gain of the selected cell over 250/250 was chosen on the
                 folds of seed 41, so it is an upper bound there, and the replication is the fairer estimate of what tuning buys

    OMP_NUM_THREADS=9 python -m sgrna.interpret.flank_window_tune [--threads 9]

Per-fold results: `results/post_freeze/flank_window_tune.csv`; the summary with paired differences against the symmetric 250/250 arm: `flank_window_tune_summary.csv`.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..common import OUT
from . import flank_scale as FS

RESULT = OUT / "flank_window_tune.csv"
SUMMARY = OUT / "flank_window_tune_summary.csv"
GRID = (50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000)
SYMMETRIC_NEW = (75, 150, 200, 300, 400)
CENTRE = 250


def _arm(u: int, d: int) -> str:
    return f"gc up={u} down={d}"


def _cols(comp, u: int, d: int) -> np.ndarray:
    up, dn = comp[u][0], comp[d][1]
    return np.column_stack([up[:, 1] + up[:, 2], dn[:, 1] + dn[:, 2]])


def run(threads: int = 9, verbose: bool = True) -> None:
    from ..build.build_curated_features import load_index
    from ..cohorts.curated_ablation import _base
    from ..controls import sealed_holdout_gene as sg
    from ..controls.mappability_stratum import _groups

    sg.N_JOBS = threads
    X, y, names, protos = _base()
    idx = load_index(verbose=False).set_index(config.ID_COL).reindex(protos).reset_index()
    groups = _groups(protos)
    comp = FS.flank_composition(idx, GRID)
    done = pd.read_csv(RESULT) if RESULT.exists() and RESULT.stat().st_size else pd.DataFrame(columns=["part", "arm", "up", "down", "seed", "fold", "rho"])
    rows = done.to_dict("records")
    old = pd.read_csv(FS.RESULT)                                           # the symmetric arms already measured by flank_scale.py (same data, folds and model)

    def have(arm, seed):
        return len(done) and ((done.arm == arm) & (done.seed == seed)).any() or any(r["arm"] == arm and r["seed"] == seed for r in rows)

    def do(part, u, d, seed):
        arm = "base" if u is None else _arm(u, d)
        if have(arm, seed):
            return
        if u is not None and u == d and seed == 41:                         # a symmetric arm of flank_scale.py: reuse its per-fold values
            o = old[(old.arm == f"gc W={u}") & (old.seed == 41)]
            if len(o):
                for _, r in o.iterrows():
                    rows.append(dict(part=part, arm=arm, up=u, down=d, seed=41, fold=int(r.fold), rho=float(r.rho)))
                pd.DataFrame(rows).to_csv(RESULT, index=False)
                return
        if u is None and seed == 41:
            o = old[(old.arm == "base") & (old.seed == 41)]
            for _, r in o.iterrows():
                rows.append(dict(part="base", arm="base", up=0, down=0, seed=41, fold=int(r.fold), rho=float(r.rho)))
            pd.DataFrame(rows).to_csv(RESULT, index=False)
            return
        t0 = time.time()
        fold = sg._fold_ids(groups, 5, seed)
        M = X if u is None else np.hstack([X, _cols(comp, u, d)])
        pred = FS._cv(M, y, groups, fold, None, seed)
        for k in range(5):
            va = fold == k
            rows.append(dict(part=part, arm=arm, up=0 if u is None else u, down=0 if u is None else d, seed=seed, fold=k + 1, rho=float(spearmanr(y[va], pred[va]).statistic)))
        pd.DataFrame(rows).to_csv(RESULT, index=False)
        if verbose:
            print(f"  {part:12s} {arm:24s} seed {seed}  rho {spearmanr(y, pred).statistic:.4f}  {time.time() - t0:.0f}s", flush=True)

    def mean_rho(u, d, seed=41):
        r = pd.DataFrame(rows)
        return float(r[(r.arm == _arm(u, d)) & (r.seed == seed)].rho.mean())

    do("base", None, None, 41)
    for W in SYMMETRIC_NEW + (50, 100, 250, 500, 1000):                    # A
        do("symmetric", W, W, 41)
    for W in GRID:                                                          # B
        do("one side", W, CENTRE, 41)
        do("one side", CENTRE, W, 41)
    cells = {(r["up"], r["down"]) for r in rows if r["arm"].startswith("gc") and r["seed"] == 41}
    u0, d0 = max(cells, key=lambda c: mean_rho(*c))
    iu, idn = GRID.index(u0), GRID.index(d0)
    print(f"  best cell so far: up={u0} down={d0} (mean rho {mean_rho(u0, d0):.4f})", flush=True)
    for a in range(max(iu - 1, 0), min(iu + 2, len(GRID))):                 # C
        for b in range(max(idn - 1, 0), min(idn + 2, len(GRID))):
            do("refine", GRID[a], GRID[b], 41)
    cells = {(r["up"], r["down"]) for r in rows if r["arm"].startswith("gc") and r["seed"] == 41}
    u1, d1 = max(cells, key=lambda c: mean_rho(*c))
    print(f"  selected cell: up={u1} down={d1} (mean rho {mean_rho(u1, d1):.4f}); 250/250 {mean_rho(250, 250):.4f}", flush=True)
    for seed in (42, 43):                                                   # D
        do("replicate", None, None, seed)
        do("replicate", CENTRE, CENTRE, seed)
        do("replicate", u1, d1, seed)
    pd.DataFrame(rows).to_csv(RESULT, index=False)


def summary() -> pd.DataFrame:
    """One row per arm. The columns ending `_41` are the selection seed (every arm has them); those ending `_rep` pool the replication seeds 42 and 43 (only the arms of part D have them)."""
    d = pd.read_csv(RESULT).drop_duplicates(["arm", "seed", "fold"])
    base = d[d.arm == "base"].set_index(["seed", "fold"]).rho
    ref = d[d.arm == _arm(CENTRE, CENTRE)].set_index(["seed", "fold"]).rho
    rows = []
    for arm, g in d[d.arm != "base"].groupby("arm", sort=False):
        g = g.set_index(["seed", "fold"])
        gain, vs = g.rho - base.reindex(g.index), g.rho - ref.reindex(g.index)
        row = dict(arm=arm, part=g.part.iloc[0], up=int(g.up.iloc[0]), down=int(g.down.iloc[0]))
        for tag, sel in (("41", g.index.get_level_values(0) == 41), ("rep", g.index.get_level_values(0) != 41)):
            n = int(sel.sum())
            row.update({f"n_folds_{tag}": n, f"gain_{tag}": float(gain[sel].mean()) if n else np.nan,
                        f"gain_sd_{tag}": float(gain[sel].std(ddof=1)) if n > 1 else np.nan,
                        f"minus_250_250_{tag}": float(vs[sel].mean()) if n else np.nan,
                        f"minus_250_250_sd_{tag}": float(vs[sel].std(ddof=1)) if n > 1 else np.nan,
                        f"folds_ahead_{tag}": int((vs[sel] > 0).sum()) if n else 0})
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(SUMMARY, index=False)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=9)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.summary_only:
        run(a.threads)
    print(summary().round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
