"""Featurise the 33,567 guides of the E. coli screen (Guo et al., 2018) as crisprHAL 2 curated them, and cache the matrix every model loads.

`featurise.py` reproduces 6,169 of the published matrix's 6,232 columns from a protospacer alone, verified at 100% on real guides,
so the training set is no longer limited to the 13,880 rows Noshay et al. published. The source is crisprHAL's curated release of
the same screen: 33,567 unique guides with read-count-filtered activity scores, **24,950 of them not in the published matrix**.

`build()` writes `expanded_X.npy` (33,567 x 6,169, float32), `expanded_meta.json` (guide order, labels, column names, which guides
are also in the published matrix) under `data/interim/`; `cohorts.curated_ablation._base()` and every experiment on the cohort read
them. The columns come from `base_fast.matrix_fast`, which equals `featurise.matrix` on all 33,567 guides (`controls/equiv_check.py`).

    python -m sgrna.build.curated_matrix --build      # about 1 second with the array featuriser, cached
"""


from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from .. import config
from . import base_fast

CACHE_X = config.INTERIM / "expanded_X.npy"
CACHE_META = config.INTERIM / "expanded_meta.json"

SPACER_LENGTH = 20


def load_crisprhal() -> pd.DataFrame:
    """Curated WT-SpCas9 guides: protospacer, activity score, 378 nt context."""
    frames = []
    for name in ("WT-SpCas9_training_data.csv", "WT-SpCas9_testing_data.csv"):
        rows = []
        for line in open(config.CRISPRHAL / name):
            seq, _, val = line.rpartition(",")
            rows.append((seq.strip(), float(val)))
        frames.append(pd.DataFrame(rows, columns=["context", "score"]))
    hal = pd.concat(frames, ignore_index=True)
    # The 378-mer is 189 nt upstream + 20 protospacer + 3 PAM + 166 downstream.
    hal["protospacer"] = hal["context"].str[189:209]
    hal["pam"] = hal["context"].str[209:212]
    return hal.drop_duplicates("protospacer").reset_index(drop=True)


def build(force: bool = False, verbose: bool = True):
    """Featurise every curated guide and cache the result."""
    if CACHE_X.exists() and CACHE_META.exists() and not force:
        meta = json.loads(CACHE_META.read_text())
        return np.load(CACHE_X, mmap_mode="r"), meta

    hal = load_crisprhal()
    if verbose:
        print(f"  {len(hal):,} unique curated guides")

    ok = hal["protospacer"].str.fullmatch(r"[ACGT]{20}")
    hal = hal[ok].reset_index(drop=True)
    if verbose and (~ok).sum():
        print(f"  dropped {(~ok).sum()} guides with ambiguous bases")

    X, names = base_fast.matrix_fast(hal["protospacer"])

    # Which of these are already in the published matrix? Useful for reporting
    # and for holding the overlap out as a sanity check.
    from ..io_utils import load_guide_index

    known = set(load_guide_index()["protospacer"].dropna())
    in_published = hal["protospacer"].isin(known).to_numpy()
    if verbose:
        print(f"  {in_published.sum():,} overlap the published matrix, "
              f"{(~in_published).sum():,} are new")

    np.save(CACHE_X, X.astype(np.float32))
    meta = {
        "protospacers": hal["protospacer"].tolist(),
        "y": hal["score"].tolist(),
        "feature_names": names,
        "in_published": in_published.tolist(),
        "shape": list(X.shape),
    }
    CACHE_META.write_text(json.dumps(meta))
    if verbose:
        print(f"  cached {CACHE_X.name} {X.shape}")
    return np.load(CACHE_X, mmap_mode="r"), meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", action="store_true", help="featurise and cache (the default)")
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args(argv)
    build(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
