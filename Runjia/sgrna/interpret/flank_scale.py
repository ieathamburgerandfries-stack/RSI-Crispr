"""How far does the flank signal reach, and how much of it is composition resolved to the base?

Three questions about the largest finding, all asked under one protocol (gene-grouped five-fold CV over the 33,567 curated guides, whole genes in
a fold and guides outside genes grouped by 100 kb block, the default model with nothing searched, selection inside each training fold):

  displacement   Family A is read from DNA at a different position: each guide is given the family A row of the guide whose protospacer lies
                 `shift` bp away along the chromosome (circular), so the window sizes, the column distributions and the smoothness are unchanged and
                 only the correspondence to this guide's own DNA is broken. This is not a larger window and not a position feature: the model is told
                 nothing about location, it is handed the flank of somewhere else.
  window size    The gain of one symmetric pair of columns (the GC fraction of the `W` nt upstream of the protospacer and of the `W` nt downstream of the
                 PAM, on the guide's strand) over the base, for W from 10 nt to 5 kb. W beyond 1 kb is cut from the genome.
  resolution     At the four family A window widths (50, 250, 500, 1,000 nt, each side): the GC fraction alone (8 columns), GC and purine fractions (16),
                 the four base fractions A, C, G, T (32). If the gain were more than GC, the four-base arm would beat the GC arm.

    OMP_NUM_THREADS=6 python -m sgrna.interpret.flank_scale [--parts window resolution displacement] [--threads 6] [--seeds 41]

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
    from ..build.build_curated_features import load_index
    from ..cohorts.curated_ablation import _base, _block
    from ..controls import sealed_holdout_gene as sg
    from ..controls.mappability_stratum import _groups

    sg.N_JOBS = threads
    X, y, names, protos = _base()
    idx = load_index(verbose=False).set_index(config.ID_COL).reindex(protos).reset_index()
    groups = _groups(protos)
    comp = flank_composition(idx, WINDOWS)
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", nargs="*", default=["window", "resolution", "displacement"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[41])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.summary_only:
        run(tuple(a.parts), tuple(a.seeds), a.threads)
    print(summary().round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
