"""How do the two models compare on a held-out fold? Paired on guides, not on folds.

The curated five-fold head-to-head is reported two ways elsewhere: as a
fold-level mean (Appendix B) and as a pooled nine-metric panel (`metric_panel`).
Neither puts an interval on the difference. The fold-level version has n = 5, so
it cannot resolve anything below ~0.005; the pooled version has no uncertainty
at all.

This closes that gap. Every guide is predicted exactly once by a model that
never saw it, so the five validation folds together give **33,567 out-of-fold
predictions per model** over the same rows. The difference can then be
bootstrapped **paired on guides**, which is the right unit: both models predict
the same rows and their errors are correlated, so an unpaired interval would be
far too wide.

Three pairings:

    untuned   SLICER at `tune.DEFAULT` against crisprHAL 2 at its published
              defaults (48 epochs, batch 1024)
    tuned     SLICER at its per-fold tuned parameters against crisprHAL 2 at
              the best of its own search
    cross     SLICER **untuned** against crisprHAL 2 **tuned** -- the most
              conservative comparison available to us, and the one that bounds
              how much of our margin is a tuning-effort artefact

The tuned pairing carries a declared asymmetry, and it is larger than this file
said until 16 October: SLICER's search ran **97-157 draws inside each of five
folds** (627 in total, `results/tuning.csv`), while crisprHAL's best arm came
from 10 draws on a single inner split of fold 1, because each of its fits costs
~17 CPU-minutes locally. Neither search ever touches an outer validation fold, so
neither number is leaked -- but theirs is ~20x weaker, so the tuned margin is an
**upper bound**.

Two things bound it from the other side. A nested search of their own on these
folds (16 October, free T4) lands at **0.6987** against their published 0.6971,
with their published configuration winning the inner search in four folds of
five -- so the space contains nothing better than what they shipped. And the
final evaluation against their own *published* weights, on their own splits,
gives +0.002 and +0.005. The conservative margin to quote from this file is
therefore the one against their best known configuration, **+0.0161**.

**Alignment is the thing that can go wrong.** crisprHAL's predictions were saved
one file per fold with no guide key, so the concatenation is in fold order, not
in cohort order. Pairing them with SLICER's out-of-fold vector requires
reconstructing `KFold(5, shuffle=True, random_state=41)` and concatenating its
validation indices in the same order. The `y` column is the check: if the
alignment is right, the labels must match exactly, guide for guide.

    python -m sgrna.heldout_compare
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from .. import config

OUT = config.RESULTS / "heldout_compare.csv"
PRED = config.RESULTS / "preds_slicer_{kind}_oof.csv"
SEED = 41
N_SPLITS = 5


def _fold_order(n: int, seed: int = SEED, n_splits: int = N_SPLITS) -> np.ndarray:
    """Row indices in the order crisprHAL's per-fold files were concatenated."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return np.concatenate([va for _, va in kf.split(np.arange(n))])


def slicer_oof(kind: str, verbose: bool = True) -> pd.DataFrame:
    """Out-of-fold predictions for SLICER, cached to disk."""
    path = str(PRED).format(kind=kind)
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        pass
    from .panel import _slicer_oof

    y, pred = _slicer_oof(tuned=(kind == "tuned"), seed=SEED,
                          n_splits=N_SPLITS, verbose=verbose)
    meta = json.loads((config.INTERIM / "expanded_meta.json").read_text())
    d = pd.DataFrame({"protospacer": meta["protospacers"], "y": y, "pred": pred})
    d.to_csv(path, index=False)
    return d


def _crisprhal(kind: str) -> pd.DataFrame:
    pat = f"preds_crisprhal_{'tuned' if kind == 'tuned' else 'default'}_fold*.csv"
    files = sorted(config.RESULTS.glob(pat))
    if len(files) != N_SPLITS:
        raise FileNotFoundError(f"expected {N_SPLITS} files for {pat}, got {len(files)}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def run(n_boot: int = 2000, seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for kind in ("untuned", "tuned", "cross"):
        # "cross" pits our defaults against their tuned model.
        s = slicer_oof("untuned" if kind == "cross" else kind, verbose=verbose)
        h = _crisprhal("tuned" if kind == "cross" else kind)
        order = _fold_order(len(s))
        s_ord = s.iloc[order].reset_index(drop=True)
        if len(h) != len(s_ord) or not np.allclose(s_ord["y"], h["y"], atol=1e-9):
            raise SystemExit(
                f"{kind}: alignment check failed -- the labels do not match "
                f"guide for guide, so the pairing would be meaningless")
        y = s_ord["y"].to_numpy()
        ps, ph = s_ord["pred"].to_numpy(), h["pred"].to_numpy()

        rho_s = float(spearmanr(y, ps).statistic)
        rho_h = float(spearmanr(y, ph).statistic)
        boot = np.empty(n_boot)
        for i in range(n_boot):
            k = rng.integers(0, len(y), len(y))
            boot[i] = (spearmanr(y[k], ps[k]).statistic
                       - spearmanr(y[k], ph[k]).statistic)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(pairing=kind, n=len(y), slicer=rho_s, crisprhal=rho_h,
                         difference=rho_s - rho_h, ci_lo=float(lo),
                         ci_hi=float(hi),
                         excludes_zero=bool(lo > 0 or hi < 0),
                         p_two_sided=float(2 * min((boot <= 0).mean(),
                                                   (boot >= 0).mean()))))
        if verbose:
            r = rows[-1]
            print(f"  {kind:>8}: SLICER {rho_s:.4f}  crisprHAL {rho_h:.4f}  "
                  f"diff {r['difference']:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}]  p={r['p_two_sided']:.4f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    if verbose:
        print(f"\n{out.round(4).to_string(index=False)}\nwritten -> {OUT}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    run(n_boot=a.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
