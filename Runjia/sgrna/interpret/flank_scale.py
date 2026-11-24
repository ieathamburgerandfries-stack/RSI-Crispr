"""The flank signal is a bulk GC effect over a few hundred bases: two GC columns reach +0.106 at 200 nt on each side, while purine and amino fractions add nothing.

Three questions about the largest finding, all asked under one protocol (gene-grouped five-fold CV over the 33,567 curated guides, whole genes in
a fold and guides outside genes grouped by 100 kb block, the default model with nothing searched, selection inside each training fold):

  displacement   Family A is read from DNA at a different position: each guide is given the family A row of the guide whose protospacer lies
                 `shift` bp away along the chromosome (circular), so the window sizes, the column distributions and the smoothness are unchanged and
                 only the correspondence to this guide's own DNA is broken. This is not a larger window and not a position feature: the model is told
                 nothing about location, it is handed the flank of somewhere else.
  window size    The gain of one symmetric pair of columns (the GC fraction of the `W` nt upstream of the protospacer and of the `W` nt downstream of the
                 PAM, on the guide's strand) over the base, for W from 10 nt to 5 kb. W beyond 1 kb is cut from the genome.
  partition      The same two columns, but the fraction of a different pair of base classes: purine against pyrimidine (A+G) and amino against keto (A+C), at the
                 same 14 windows as GC. GC (strong against weak, C+G) is the third of the three ways to split four bases into two pairs, so this asks whether GC is
                 special or any two-class composition would do.
  asymmetric     The upstream and downstream windows chosen separately (33 combinations, then a replication on two new fold assignments); see the section below.
  resolution     At the four family A window widths (50, 250, 500, 1,000 nt, each side): the GC fraction alone (8 columns), GC and purine fractions (16),
                 the four base fractions A, C, G, T (32). If the gain were more than GC, the four-base arm would beat the GC arm.

    OMP_NUM_THREADS=6 python -m sgrna.interpret.flank_scale [--parts window partition resolution displacement] [--threads 6] [--seeds 41]

Per-fold results: `results/post_freeze/flank_scale.csv` (arm, part, seed, fold, rho); the table of gains over the base, paired by fold, is written by `summary()`.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..common import OUT

RESULT = OUT / "flank_scale.csv"
SUMMARY = OUT / "flank_scale_summary.csv"
WINDOWS = (10, 25, 50, 100, 250, 500, 1000, 2000, 5000)
FAMILY_A_WINDOWS = (50, 250, 500, 1000)
PARTITION_WINDOWS = tuple(sorted(set(WINDOWS) | {75, 150, 200, 300, 400}))       # the 14 symmetric windows of Figure 1b
SHIFTS = (0, 100, 300, 1_000, 3_000, 10_000, 100_000, 1_000_000)
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}


def _prefix_counts(genome: str, pad: int = 6000) -> np.ndarray:
    """Cumulative counts of A, C, G, T along the plus strand, genome plus `pad` nt of its start (circular)."""
    g = (genome + genome[:pad]).upper()
    arr = np.frombuffer(g.encode(), dtype=np.uint8)
    out = np.zeros((len(arr) + 1, 4), dtype=np.int64)
    for j, b in enumerate("ACGT"):
        out[1:, j] = np.cumsum(arr == ord(b))
    return out


def _window_fractions(cs: np.ndarray, start: np.ndarray, length: int, plus: np.ndarray) -> np.ndarray:
    """Base fractions (A, C, G, T on the guide's strand) of `length` nt from plus-strand coordinate `start`, circular."""
    n = len(cs) - 1 - 6000
    s = np.mod(start, n)
    cnt = cs[s + length] - cs[s]                                   # plus-strand A, C, G, T counts
    frac = cnt / float(length)
    minus = frac[:, ::-1]                                          # the guide strand reads the complement: A<->T, C<->G
    return np.where(plus[:, None], frac, minus)


def flank_composition(idx: pd.DataFrame, widths=WINDOWS) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """For each width W: (upstream, downstream) base fractions, each (n_guides, 4), on the guide's strand; upstream is 5' of the protospacer, downstream 3' of the PAM."""
    from ..build import genome

    cs = _prefix_counts(genome.load_genome())
    left = idx["left"].to_numpy(int) - 1                           # the index is 1-based and inclusive; the genome string is 0-based
    right = idx["right"].to_numpy(int) - 1
    plus = (idx["strand"].to_numpy() == "+")
    out = {}
    for W in widths:
        up_start = np.where(plus, left - W, right + 1)             # plus strand: [left - W, left - 1]; minus: [right + 1, right + W]
        dn_start = np.where(plus, right + 4, left - 3 - W)         # plus: [right + 4, right + 3 + W]; minus: [left - 3 - W, left - 4]
        out[W] = (_window_fractions(cs, up_start, W, plus), _window_fractions(cs, dn_start, W, plus))
    return out


def _cols(comp, W, kind):
    up, dn = comp[W]
    if kind == "gc":
        return np.column_stack([up[:, 1] + up[:, 2], dn[:, 1] + dn[:, 2]])
    if kind == "gcpur":
        return np.column_stack([up[:, 1] + up[:, 2], up[:, 0] + up[:, 2], dn[:, 1] + dn[:, 2], dn[:, 0] + dn[:, 2]])
    if kind == "base4":
        return np.column_stack([up, dn])
    if kind == "purine":                                             # A + G, against the pyrimidines C + T
        return np.column_stack([up[:, 0] + up[:, 2], dn[:, 0] + dn[:, 2]])
    if kind == "amino":                                              # A + C, against the keto bases G + T
        return np.column_stack([up[:, 0] + up[:, 1], dn[:, 0] + dn[:, 1]])
    raise ValueError(kind)


def _cv(M, y, groups, fold, n_features, seed):
    from ..controls import sealed_holdout_gene as sg

    pred = np.empty(len(y))
    for k in range(5):
        va = fold == k
        pred[va] = sg._fit_predict(M[~va], y[~va], M[va], None, n_features, seed)
    return pred


def _record(rows, part, arm, seed, fold, y, pred):
    for k in range(5):
        va = fold == k
        rows.append(dict(part=part, arm=arm, seed=seed, fold=k + 1, rho=float(spearmanr(y[va], pred[va]).statistic)))


def run(parts=("window", "resolution", "displacement"), seeds=(41,), threads: int = 6, verbose: bool = True) -> pd.DataFrame:
    from ..build import genome
    from ..build.build_features import load_curated_index
    from ..cohorts.curated_ablation import _base, _block
    from ..controls import sealed_holdout_gene as sg
    from ..controls.mappability_stratum import _groups

    sg.N_JOBS = threads
    X, y, names, protos = _base()
    idx = load_curated_index(verbose=False).set_index(config.ID_COL).reindex(protos).reset_index()
    groups = _groups(protos)
    comp = flank_composition(idx, PARTITION_WINDOWS)
    done = pd.read_csv(RESULT) if RESULT.exists() and RESULT.stat().st_size else pd.DataFrame(columns=["part", "arm", "seed", "fold", "rho"])
    rows = done.to_dict("records")

    def have(part, arm, seed):
        return len(done) and ((done.part == part) & (done.arm == arm) & (done.seed == seed)).any()

    def do(part, arm, M, nf, seed):
        if have(part, arm, seed):
            return
        t0 = time.time()
        fold = sg._fold_ids(groups, 5, seed)
        pred = _cv(M, y, groups, fold, nf, seed)
        _record(rows, part, arm, seed, fold, y, pred)
        pd.DataFrame(rows).to_csv(RESULT, index=False)
        if verbose:
            print(f"  {part:12s} {arm:28s} seed {seed}  rho {spearmanr(y, pred).statistic:.4f}  {time.time() - t0:.0f}s", flush=True)

    for seed in seeds:
        do("base", "base", X, None, seed)
        if "window" in parts:
            for W in WINDOWS:
                do("window", f"gc W={W}", np.hstack([X, _cols(comp, W, "gc")]), None, seed)
            for W in WINDOWS:
                do("window", f"base4 W={W}", np.hstack([X, _cols(comp, W, "base4")]), None, seed)
        if "partition" in parts:
            for kind in ("purine", "amino"):
                for W in PARTITION_WINDOWS:
                    do("partition", f"{kind} W={W}", np.hstack([X, _cols(comp, W, kind)]), None, seed)
        if "resolution" in parts:
            for kind in ("gc", "gcpur", "base4"):
                do("resolution", f"{kind} at 50/250/500/1000", np.hstack([X] + [_cols(comp, W, kind) for W in FAMILY_A_WINDOWS]), None, seed)
        if "full" in parts and "displacement" not in parts:                  # the aligned full family on this seed: the same arm as shift 0 below
            A, _ = _block("a_flank", protos)
            do("displacement", "family A shifted 0", np.hstack([X, A]), config.N_FEATURES, seed)
        if "displacement" in parts:
            A, anames = _block("a_flank", protos)
            pos = idx["left"].to_numpy(float)
            order = np.argsort(pos)
            sp = pos[order]
            glen = float(genome.genome_length())
            for shift in SHIFTS:
                tgt = (pos + shift) % glen
                j = np.clip(np.searchsorted(sp, tgt), 0, len(sp) - 1)
                lft = np.clip(j - 1, 0, len(sp) - 1)
                pick = np.where(np.abs(sp[j] - tgt) <= np.abs(sp[lft] - tgt), j, lft)
                As = A if shift == 0 else A[order[pick]]
                do("displacement", f"family A shifted {shift}", np.hstack([X, As]), config.N_FEATURES, seed)
    return pd.DataFrame(rows)


def summary() -> pd.DataFrame:
    d = pd.read_csv(RESULT)
    base = d[d.arm == "base"].set_index(["seed", "fold"]).rho
    rows = []
    for (part, arm), g in d[d.arm != "base"].groupby(["part", "arm"], sort=False):
        g = g.set_index(["seed", "fold"])
        gain = g.rho - base.reindex(g.index)
        per_seed = gain.groupby(level=0).mean()
        rows.append(dict(part=part, arm=arm, n_folds=len(gain), rho=float(g.rho.mean()), gain=float(gain.mean()), gain_sd_over_folds=float(gain.std(ddof=1)),
                         folds_positive=int((gain > 0).sum()), seeds=int(g.index.get_level_values(0).nunique())))
    s = pd.DataFrame(rows)
    s.insert(2, "base_rho", float(d[d.arm == "base"].rho.mean()))
    s.to_csv(SUMMARY, index=False)
    return s


# ------------------------------------------------------------------------------------------------ the two windows tuned separately
# Part `asymmetric`: the upstream window U and the downstream window D chosen separately. Same engine as the symmetric arms above, so every arm is paired by fold with `flank_scale.csv`.
#   A  five symmetric windows added to the nine of `window` (75, 150, 200, 300, 400 nt);  B  one side varied over the grid with the other held at 250 nt;
#   C  the best cell and its neighbours (3 x 3);  D  the best cell, 250/250 and the base on seeds 42 and 43, because a cell chosen from many on the folds it is scored on overstates its lead.
# Results: `flank_window_tune.csv`; summary with paired differences against 250/250: `flank_window_tune_summary.csv` (columns `_41` for the selection seed, `_rep` for the replication seeds).
TUNE_RESULT = OUT / "flank_window_tune.csv"
TUNE_SUMMARY = OUT / "flank_window_tune_summary.csv"
TUNE_GRID = (50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000)
TUNE_NEW_SYMMETRIC = (75, 150, 200, 300, 400)
TUNE_CENTRE = 250


def _arm(u: int, d: int) -> str:
    return f"gc up={u} down={d}"


def _asym_cols(comp, u: int, d: int) -> np.ndarray:
    up, dn = comp[u][0], comp[d][1]
    return np.column_stack([up[:, 1] + up[:, 2], dn[:, 1] + dn[:, 2]])


def run_asymmetric(threads: int = 9, verbose: bool = True) -> None:
    from ..build.build_features import load_curated_index
    from ..cohorts.curated_ablation import _base
    from ..controls import sealed_holdout_gene as sg
    from ..controls.mappability_stratum import _groups

    sg.N_JOBS = threads
    X, y, names, protos = _base()
    idx = load_curated_index(verbose=False).set_index(config.ID_COL).reindex(protos).reset_index()
    groups = _groups(protos)
    comp = flank_composition(idx, TUNE_GRID)
    done = pd.read_csv(TUNE_RESULT) if TUNE_RESULT.exists() and TUNE_RESULT.stat().st_size else pd.DataFrame(columns=["part", "arm", "up", "down", "seed", "fold", "rho"])
    rows = done.to_dict("records")
    old = pd.read_csv(RESULT)                                           # the symmetric arms already measured by flank_scale.py (same data, folds and model)

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
                pd.DataFrame(rows).to_csv(TUNE_RESULT, index=False)
                return
        if u is None and seed == 41:
            o = old[(old.arm == "base") & (old.seed == 41)]
            for _, r in o.iterrows():
                rows.append(dict(part="base", arm="base", up=0, down=0, seed=41, fold=int(r.fold), rho=float(r.rho)))
            pd.DataFrame(rows).to_csv(TUNE_RESULT, index=False)
            return
        t0 = time.time()
        fold = sg._fold_ids(groups, 5, seed)
        M = X if u is None else np.hstack([X, _asym_cols(comp, u, d)])
        pred = _cv(M, y, groups, fold, None, seed)
        for k in range(5):
            va = fold == k
            rows.append(dict(part=part, arm=arm, up=0 if u is None else u, down=0 if u is None else d, seed=seed, fold=k + 1, rho=float(spearmanr(y[va], pred[va]).statistic)))
        pd.DataFrame(rows).to_csv(TUNE_RESULT, index=False)
        if verbose:
            print(f"  {part:12s} {arm:24s} seed {seed}  rho {spearmanr(y, pred).statistic:.4f}  {time.time() - t0:.0f}s", flush=True)

    def mean_rho(u, d, seed=41):
        r = pd.DataFrame(rows)
        return float(r[(r.arm == _arm(u, d)) & (r.seed == seed)].rho.mean())

    do("base", None, None, 41)
    for W in TUNE_NEW_SYMMETRIC + (50, 100, 250, 500, 1000):                    # A
        do("symmetric", W, W, 41)
    for W in TUNE_GRID:                                                          # B
        do("one side", W, TUNE_CENTRE, 41)
        do("one side", TUNE_CENTRE, W, 41)
    cells = {(r["up"], r["down"]) for r in rows if r["arm"].startswith("gc") and r["seed"] == 41}
    u0, d0 = max(cells, key=lambda c: mean_rho(*c))
    iu, idn = TUNE_GRID.index(u0), TUNE_GRID.index(d0)
    print(f"  best cell so far: up={u0} down={d0} (mean rho {mean_rho(u0, d0):.4f})", flush=True)
    for a in range(max(iu - 1, 0), min(iu + 2, len(TUNE_GRID))):                 # C
        for b in range(max(idn - 1, 0), min(idn + 2, len(TUNE_GRID))):
            do("refine", TUNE_GRID[a], TUNE_GRID[b], 41)
    cells = {(r["up"], r["down"]) for r in rows if r["arm"].startswith("gc") and r["seed"] == 41}
    u1, d1 = max(cells, key=lambda c: mean_rho(*c))
    print(f"  selected cell: up={u1} down={d1} (mean rho {mean_rho(u1, d1):.4f}); 250/250 {mean_rho(250, 250):.4f}", flush=True)
    for seed in (42, 43):                                                   # D
        do("replicate", None, None, seed)
        do("replicate", TUNE_CENTRE, TUNE_CENTRE, seed)
        do("replicate", u1, d1, seed)
    pd.DataFrame(rows).to_csv(TUNE_RESULT, index=False)


def summary_asymmetric() -> pd.DataFrame:
    """One row per arm. The columns ending `_41` are the selection seed (every arm has them); those ending `_rep` pool the replication seeds 42 and 43 (only the arms of part D have them)."""
    d = pd.read_csv(TUNE_RESULT).drop_duplicates(["arm", "seed", "fold"])
    base = d[d.arm == "base"].set_index(["seed", "fold"]).rho
    ref = d[d.arm == _arm(TUNE_CENTRE, TUNE_CENTRE)].set_index(["seed", "fold"]).rho
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
    s.to_csv(TUNE_SUMMARY, index=False)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", nargs="*", default=["window", "resolution", "displacement"], help="window, partition, resolution, displacement, asymmetric")
    ap.add_argument("--seeds", type=int, nargs="*", default=[41])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)
    main_parts = tuple(p for p in a.parts if p != "asymmetric")
    if not a.summary_only:
        if main_parts:
            run(main_parts, tuple(a.seeds), a.threads)
        if "asymmetric" in a.parts:
            run_asymmetric(a.threads)
    if main_parts or a.summary_only:
        print(summary().round(4).to_string(index=False))
    if "asymmetric" in a.parts or a.summary_only:
        print(summary_asymmetric().round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
