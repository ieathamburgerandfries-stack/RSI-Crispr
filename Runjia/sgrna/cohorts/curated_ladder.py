"""The reduced-representation ladder, re-run on the curated cohort without selection.

Finding 2's ladder — 74 mononucleotide indicators, 364 with dinucleotides, 412
portable, 6,232 full — was measured on the published matrix, and it carries a
protocol caveat: every arm selected 300 columns from its candidate pool, so the
74-column arm used all of its candidates while the 6,232-column arm used 300 of
6,232. The ladder measured candidate pools under a fixed budget, not
representations.

This re-runs it on the curated cohort and removes the caveat: **no selection in
any arm**, `n_features` set to each arm's own column count. Two questions at
once, both open rows in Part 4:

  * does the reduced set still reach the full representation on this cohort,
    where the 63 hand-named columns are absent by construction?
  * is `n_features = 300` costing anything? The arms below use every column
    they have, so the 6,169- and 6,517-column arms answer it directly.

Columns are chosen from the decoded mapping (`decode_v_columns`), so "74
mononucleotide indicators" means exactly the columns that decode to
(position, single base) and nothing else.

Three seeds × five folds rather than five × five: no arm selects, so each fit
is over every column it holds, and the ladder is internally consistent at three.

    python -m sgrna.cohorts.curated_ladder
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config
from .curated_ablation import _base, _block

OUT = config.RESULTS / "curated_ladder.csv"
SEEDS = (41, 42, 43)


def _unit_columns(names, units) -> np.ndarray:
    from ..build import decode_v_columns as dv

    mp = dv.load()["mapping"]
    return np.array([n in mp and mp[n].get("unit") in units
                     and not mp[n].get("ambiguous") for n in names])


def run(seeds=SEEDS, n_splits=None, verbose: bool = True):
    from ..harness.run_ablation import cross_validate

    n_splits = n_splits or config.N_SPLITS
    X, y, names, protos = _base()
    mono = _unit_columns(names, ("monomer",))
    mono_di = _unit_columns(names, ("monomer", "dimer"))
    A, anames = _block("a_flank", protos)

    mono_di_tri = _unit_columns(names, ("monomer", "dimer", "trimer"))
    all_kmers = _unit_columns(names, ("monomer", "dimer", "trimer", "tetramer"))
    sel = lambda m: [n for n, k in zip(names, m) if k]
    arms = {
        "mono": (X[:, mono], sel(mono)),
        "mono_di": (X[:, mono_di], sel(mono_di)),
        # Finding 4 says higher-order k-mers help once selection is removed, and
        # that was measured on the published matrix. These two arms re-ask it
        # here, where no arm selects: trimers add 1,146 columns and tetramers a
        # further 4,343, so if the trimer step pays and the tetramer step does
        # not, the usable order stops at three on this cohort too.
        "mono_di_tri": (X[:, mono_di_tri], sel(mono_di_tri)),
        # The step from mono_di_tri to base_full adds two different things at
        # once -- 4,343 tetramer indicators and 316 quantum-table columns -- so
        # a gain there cannot be attributed to either. This arm is the k-mers
        # without the quantum columns, which is what Finding 4's "the usable
        # order stops at three" is actually a claim about.
        "all_kmers": (X[:, all_kmers], sel(all_kmers)),
        "base_full": (X, names),
        "mono_di+a_flank": (np.hstack([X[:, mono_di], A]), sel(mono_di) + anames),
        "mono_di_tri+a_flank": (np.hstack([X[:, mono_di_tri], A]),
                                sel(mono_di_tri) + anames),
        "all_kmers+a_flank": (np.hstack([X[:, all_kmers], A]),
                              sel(all_kmers) + anames),
        "base_full+a_flank": (np.hstack([X, A]), names + anames),
    }

    rows = []
    for arm, (Xa, nm) in arms.items():
        res = cross_validate(Xa, y, nm, ["arm"] * len(nm),
                             n_features=Xa.shape[1],      # every column, no selection
                             seeds=list(seeds), n_splits=n_splits, verbose=False,
                             tag=f"ladder_{arm}", experiment=arm)
        rows.append(dict(arm=arm, n_cols=Xa.shape[1],
                         spearman=res["summary"]["spearman"]["mean"],
                         spearman_sd=res["summary"]["spearman"]["sd"]))
        if verbose:
            print(f"  {arm:>18} {Xa.shape[1]:>5} cols: rho "
                  f"{rows[-1]['spearman']:.4f} ± {rows[-1]['spearman_sd']:.4f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    out = pd.DataFrame(rows)
    if verbose:
        full = out.set_index("arm").loc["base_full", "spearman"]
        out["vs_base_full"] = out["spearman"] - full
        print()
        print(out.round(4).to_string(index=False))
        print(f"written -> {OUT}")
    out.to_csv(OUT, index=False)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    a = ap.parse_args(argv)
    run(seeds=tuple(a.seeds) if a.seeds else SEEDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
