"""Feature-family ablation on the paper's own folds.

`curated_ablation.py` measured every family's gain over the 20-mer on two fold designs: random rows (the team's shared harness, five seeds of
five-fold `KFold`) and whole genes (`--grouping gene`, five seeds, the 30,203 genic guides only). Neither is the fold design of the paper's other
numbers, which is `controls/sealed_holdout_gene._fold_ids` over all 33,567 guides (every gene whole; the 3,364 guides outside a gene in 100 kb
blocks), seeds 41 to 43, so 15 folds. This runs the same ablation on exactly those folds, with the model of every other number (the default
model: 300 columns chosen inside each training fold, 400 trees), so a Δρ here sits beside the flank gain of +0.161 without a change of protocol.

Arms, per seed (every arm sees the same five folds):

  20-mer, 300 selected   the 6,169 base columns with the same 300-column selection every family arm gets: the like-for-like baseline
  20-mer + one family    the base plus that family's block, 300 selected inside each training fold

Two more arms split `d_mechanics` (`controls/no_information.py` did this on random rows): its 22 columns that are functions of the protospacer
and its 8 that read the flanking DNA (`.flank.` in the column name).

`flank_scale.csv` already holds two arms on these folds and they are read, not rerun: the 20-mer with no selection (the paper's 0.547) and the
20-mer + family A (`family A shifted 0`). Each family's gain is reported against both baselines: against the like-for-like one (what the
family adds at a fixed budget of 300 columns) and against the paper's (what the headline flank gain is measured against). They differ by the
effect of the selection alone.

    OMP_NUM_THREADS=9 python -m sgrna.cohorts.ablation_paper_folds [--threads 9] [--seeds 41 42 43]

Per-fold results: `results/post_freeze/ablation_paper_folds.csv`; summary: `ablation_paper_folds_summary.csv`.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..common import OUT

RESULT = OUT / "ablation_paper_folds.csv"
SUMMARY = OUT / "ablation_paper_folds_summary.csv"
BASE_SEL = "20-mer, 300 selected"
# guide-derived families first (the claim this answers), then the families that read DNA or tracks beyond the guide
ORDER = ("b_energy", "c_folding", "f_methylation", "j_periodicity", "h_offtarget",
         "d_mechanics", "i_shape", "e_nucleoid", "g_transcription", "d_supercoiling")
SPLIT = {"d_mechanics, guide columns only": False, "d_mechanics, flank columns only": True}     # arm -> keep the columns that read the flanks?
FLANK = ".flank."


def run(seeds=(41, 42, 43), threads: int = 9, verbose: bool = True) -> None:
    from ..controls import sealed_holdout_gene as sg
    from ..controls.mappability_stratum import _groups
    from ..interpret import flank_scale as FS
    from .curated_ablation import _base, _block

    sg.N_JOBS = threads
    X, y, names, protos = _base()
    groups = _groups(protos)
    done = pd.read_csv(RESULT) if RESULT.exists() and RESULT.stat().st_size else pd.DataFrame(columns=["arm", "seed", "fold", "rho", "n_cols"])
    rows = done.to_dict("records")
    have = {(r["arm"], r["seed"]) for r in rows}
    blocks = {}
    for seed in seeds:
        fold = sg._fold_ids(groups, 5, seed)
        for arm in (BASE_SEL,) + ORDER + tuple(SPLIT):
            if (arm, seed) in have:
                continue
            if arm == BASE_SEL:
                M = X
            elif arm in SPLIT:
                if "d_mechanics" not in blocks:
                    blocks["d_mechanics"] = _block("d_mechanics", protos)
                v, fnames = blocks["d_mechanics"]
                is_flank = np.array([FLANK in f for f in fnames])
                M = np.hstack([X, v[:, is_flank if SPLIT[arm] else ~is_flank]])
            else:
                if arm not in blocks:
                    blocks[arm] = _block(arm, protos)
                M = np.hstack([X, blocks[arm][0]])
            t0 = time.time()
            pred = FS._cv(M, y, groups, fold, config.N_FEATURES, seed)
            for k in range(5):
                va = fold == k
                rows.append(dict(arm=arm, seed=seed, fold=k + 1, rho=float(spearmanr(y[va], pred[va]).statistic), n_cols=int(M.shape[1])))
            pd.DataFrame(rows).to_csv(RESULT, index=False)
            if verbose:
                print(f"  {arm:24s} seed {seed}  rho {spearmanr(y, pred).statistic:.4f}  {time.time() - t0:.0f}s", flush=True)


def summary() -> pd.DataFrame:
    d = pd.read_csv(RESULT)
    fs = pd.read_csv(OUT / "flank_scale.csv")
    nosel = fs[fs.arm == "base"].set_index(["seed", "fold"]).rho
    fam_a = fs[fs.arm == "family A shifted 0"].set_index(["seed", "fold"]).rho
    sel = d[d.arm == BASE_SEL].set_index(["seed", "fold"]).rho
    arms = {"a_flank": fam_a}
    for arm, g in d[d.arm != BASE_SEL].groupby("arm", sort=False):
        arms[arm] = g.set_index(["seed", "fold"]).rho
    rows = []
    for arm, r in arms.items():
        ix = r.index.intersection(sel.index).intersection(nosel.index)
        a = (r.loc[ix] - sel.loc[ix])                 # like for like: the same 300-column selection on both arms
        b = (r.loc[ix] - nosel.loc[ix])               # against the paper's baseline, the 20-mer with no selection
        rows.append(dict(family=arm, n_folds=len(ix), seeds=int(ix.get_level_values(0).nunique()), rho=float(r.loc[ix].mean()),
                         gain_same_selection=float(a.mean()), gain_same_selection_sd=float(a.std(ddof=1)), folds_positive_same_selection=int((a > 0).sum()),
                         gain_vs_no_selection=float(b.mean()), gain_vs_no_selection_sd=float(b.std(ddof=1)), folds_positive_vs_no_selection=int((b > 0).sum()),
                         base_selected_rho=float(sel.loc[ix].mean()), base_no_selection_rho=float(nosel.loc[ix].mean())))
    s = pd.DataFrame(rows)
    s.to_csv(SUMMARY, index=False)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=9)
    ap.add_argument("--seeds", type=int, nargs="*", default=[41, 42, 43])
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.summary_only:
        run(tuple(a.seeds), a.threads)
    print(summary().round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
