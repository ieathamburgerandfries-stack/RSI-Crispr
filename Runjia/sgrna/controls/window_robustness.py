"""Would a different set of flank windows have given the same answer?

Family A summarises the flanks in windows of **50, 250, 500 and 1,000 nt**
(`config.FLANK_WINDOWS`). Those four numbers were chosen on the published cohort
after a sweep, so with respect to the curated five folds they are an inherited
constant rather than a choice fitted here. That is the weaker form of the
objection. The stronger form survives it: *a sweep was run, and the reported
figure uses the windows that won it*, so some of family A's gain could be the
sweep's own winner's curse.

**The test that settles it is cheap, and it is a robustness check rather than a
re-selection.** Rebuild family A on a window set chosen to share no value with
the original — **75, 300, 600, 1,200 nt** — and re-measure the gain. The
prediction follows from Finding 4: if the effect is a smooth compositional
gradient rather than a feature of particular window edges, any reasonable set of
four nested windows should recover it, and the two gains should agree to within
fold noise. If instead the original set is worth materially more, the reported
+0.164 is partly a selection artefact and should be discounted.

Declared before running, as the interpretation: agreement within **±0.005 ρ**
(the project's non-inferiority margin) counts as the choice not mattering;
a shortfall beyond that is a winner's curse worth quantifying.

    python -m sgrna.window_robustness
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config
from ..cohorts.curated_ablation import _base, _block

OUT = config.RESULTS / "window_robustness.csv"
ALT_WINDOWS = (75, 300, 600, 1200)
BLOCK = config.INTERIM / "features" / "a_flank_altwin_curated.csv"


def build(verbose: bool = True):
    """Family A again, on windows sharing no value with the published set."""
    if BLOCK.exists():
        return pd.read_csv(BLOCK)
    from ..build.build_curated_features import load_index
    from ..features import a_flank

    # The module reads its windows from `config` at import time, so the constant
    # is swapped before the build rather than passed in -- the alternative is an
    # API change to a frozen module.
    original = a_flank.SUMMARY_WINDOWS
    a_flank.SUMMARY_WINDOWS = ALT_WINDOWS
    try:
        idx = load_index(verbose=verbose)
        block = a_flank.build(idx)
    finally:
        a_flank.SUMMARY_WINDOWS = original
    block.to_csv(BLOCK, index=False)
    if verbose:
        print(f"  built {block.shape[0]:,} x {block.shape[1] - 1} columns "
              f"at windows {ALT_WINDOWS} -> {BLOCK.name}")
    return block


def run(seeds=None, n_splits=None, n_features=300, verbose: bool = True):
    from ..harness.run_ablation import cross_validate

    seeds = list(seeds or config.SEEDS)
    n_splits = n_splits or config.N_SPLITS
    X, y, names, protos = _base()
    singles = pd.read_csv(config.RESULTS / "ablation_curated.csv").set_index("family")
    base_rho = float(singles.loc["baseline", "spearman"])
    published = float(singles.loc["a_flank", "d_spearman"])

    block = build(verbose=verbose)
    b = block.set_index(config.ID_COL).reindex(protos)
    v = b.to_numpy(dtype=np.float32)
    keep = ~np.all(np.isnan(v), axis=0)
    v, fnames = v[:, keep], [c for c, k in zip(b.columns, keep) if k]
    if verbose:
        print(f"  alternative family A: {v.shape[1]} columns "
              f"(published set has {int(singles.loc['a_flank', 'n_cols']) - X.shape[1]})")

    res = cross_validate(np.hstack([X, v]), y, names + fnames,
                         ["base"] * len(names) + ["a_flank_alt"] * len(fnames),
                         n_features=n_features, seeds=seeds, n_splits=n_splits,
                         verbose=False, tag="altwin_a_flank",
                         experiment="a_flank_alt")
    rho = res["summary"]["spearman"]["mean"]
    row = dict(arm="a_flank at 75/300/600/1200", n_cols=v.shape[1],
               spearman=rho, spearman_sd=res["summary"]["spearman"]["sd"],
               d_spearman=rho - base_rho, d_published_windows=published,
               difference=(rho - base_rho) - published)
    pd.DataFrame([row]).to_csv(OUT, index=False)
    if verbose:
        print(f"\n  published windows (50/250/500/1000): {published:+.4f}")
        print(f"  alternative windows (75/300/600/1200): {row['d_spearman']:+.4f}")
        print(f"  difference: {row['difference']:+.4f}  "
              f"({'within' if abs(row['difference']) < 0.005 else 'OUTSIDE'} "
              f"the declared +/-0.005)")
        print(f"written -> {OUT}")
    return row


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
