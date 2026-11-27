"""Shared conventions for reading the base matrix and writing feature blocks.

A *feature block* is one CSV per family in `data/interim/features/`, keyed by
sgRNAID, whose other columns all start with `eng.`. Blocks are built
independently and joined only at the last moment, so a family can be rebuilt
or dropped without touching anything else.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def block_path(family: str) -> Path:
    return config.FEATURE_DIR / f"{family}.csv"


def save_block(df: pd.DataFrame, family: str) -> Path:
    """Write a feature block after checking the naming conventions hold."""
    if config.ID_COL not in df.columns:
        raise ValueError(f"Feature block '{family}' has no {config.ID_COL} column.")
    bad = [
        c
        for c in df.columns
        if c != config.ID_COL and not c.startswith("eng.")
    ]
    if bad:
        raise ValueError(
            f"Feature block '{family}' has columns that do not start with 'eng.': "
            f"{bad[:8]}"
        )
    if df[config.ID_COL].duplicated().any():
        raise ValueError(f"Feature block '{family}' has duplicate sgRNAIDs.")

    path = block_path(family)
    df.to_csv(path, index=False)
    return path


def load_block(family: str) -> pd.DataFrame:
    path = block_path(family)
    if not path.exists():
        raise FileNotFoundError(
            f"No feature block for '{family}'. Build it with "
            f"`python -m sgrna.build.build_features --families {family}`."
        )
    return pd.read_csv(path, low_memory=False)


def available_blocks() -> list[str]:
    return sorted(p.stem for p in config.FEATURE_DIR.glob("*.csv"))


def load_guide_index() -> pd.DataFrame:
    if not config.GUIDE_INDEX.exists():
        raise FileNotFoundError(
            f"{config.GUIDE_INDEX} not found -- run "
            "`python -m sgrna.build.build_guide_index` first."
        )
    return pd.read_csv(config.GUIDE_INDEX, low_memory=False)


def load_base_matrix(usecols=None, nrows=None) -> pd.DataFrame:
    """Read the published matrix, skipping its title row."""
    return pd.read_csv(
        config.BASE_MATRIX,
        header=config.BASE_MATRIX_HEADER_ROW,
        usecols=usecols,
        nrows=nrows,
        low_memory=False,
    )

