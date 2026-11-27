"""The curated E. coli cohort as the experiments load it: the 6,169 base columns and the label, one feature family aligned to the guides, and the group label of each guide.

These three loaders were private helpers of `cohorts/curated_ablation.py` (`_base`, `_block`, `_groups`) that 53 modules imported across packages; they live here, under public names, because they are the
shared input of nearly every experiment and the audit. The bodies are unchanged.

    X, y, names, protos = load_base()
    block, block_names = load_block("a_flank", protos)
    group, has_group = load_groups("gene", protos)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import config


def load_base():
    """The curated cohort: 6,169 regenerated columns, their label, their order."""
    meta = json.loads((config.INTERIM / "expanded_meta.json").read_text())
    X = np.asarray(np.load(config.INTERIM / "expanded_X.npy", mmap_mode="r"),
                   dtype=np.float32)
    y = np.asarray(meta["y"], dtype=float)
    return X, y, list(meta["feature_names"]), np.asarray(meta["protospacers"])


def load_block(fam: str, protos: np.ndarray):
    """One family, aligned to the curated guides. Returns (values, names)."""
    path = config.INTERIM / "features" / f"{fam}_curated.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not built")
    b = pd.read_csv(path).set_index(config.ID_COL).reindex(protos)
    v = b.to_numpy(dtype=np.float32)
    keep = ~np.all(np.isnan(v), axis=0)          # drop columns empty on this cohort
    if keep.sum() == 0:
        # b_energy built "successfully" with zero feature columns on this cohort,
        # which save_block accepts. An empty block is a failed build, not a null
        # result -- refuse it rather than reporting a delta of exactly zero.
        raise FileNotFoundError(f"{path.name} has no populated feature columns")
    return v[:, keep], [c for c, k in zip(b.columns, keep) if k]


def load_groups(kind: str, protos: np.ndarray):
    """Group label per curated guide, and the rows that have one.

    Grouping by gene is the arm that matters on this cohort: gene identity
    explains 29% of the curated label against 16% of the published one
    (`cohort_groups.py`), so plain KFold has more to leak here, not less.
    Guides that fall outside every annotated gene -- 3,364 of 33,567 -- have no
    gene group, and they are dropped rather than given singleton groups, which
    would leave each of them in its own fold and defeat the point. **Every arm
    then runs on the same subset, baseline included**, so the deltas stay
    internally comparable even though their levels are not comparable to the
    random-row table.
    """
    from ..build.build_features import load_curated_index

    idx = load_curated_index(verbose=False).set_index(config.ID_COL)
    if kind == "gene":
        g = idx["gene_name"].reindex(protos)
    elif kind == "block100k":
        g = (pd.to_numeric(idx["left"], errors="coerce") // 100_000).reindex(protos)
    else:
        raise ValueError(f"unknown grouping {kind!r}")
    ok = g.notna().to_numpy()
    return g.to_numpy(dtype=object), ok
