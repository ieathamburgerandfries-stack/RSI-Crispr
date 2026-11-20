"""Step 2 -- assemble a model-ready matrix from the base features plus blocks.

The published matrix is 13,880 x 6,232 and lives in a 198 MB CSV that pandas
takes about a minute to tokenise. Re-reading that for every ablation would
waste most of an afternoon, so this module caches it once as a float32 .npy
with its column names beside it, and every later step loads that.

From a notebook cell:

    nb.cache_matrix()
    nb.export_csv(["a_flank"])

or from a terminal:

    python -m sgrna.build_matrix --cache            # build the cache only
    python -m sgrna.build_matrix --families a_flank --export csv

`--export csv` writes a merged CSV in the same shape as the original (title
row included) so it can be dropped straight into the team's existing Colab
scripts by changing DATA_PATH.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from .. import config
from ..features import ALL
from ..io_utils import load_block, load_base_matrix

CACHE_X = config.INTERIM / "base_X.npy"
CACHE_META = config.INTERIM / "base_meta.json"

NON_FEATURE_COLUMNS = [
    "sgRNAID", "sequence", "Sequence", "sgRNA", "sgrna",
    "guide", "Guide", "guide_sequence", "GuideSequence",
    "target_sequence", "TargetSequence",
]


def build_cache(force: bool = False, verbose: bool = True):
    """Cache the published matrix as float32 + column names + target."""
    if CACHE_X.exists() and CACHE_META.exists() and not force:
        meta = json.loads(CACHE_META.read_text())
        return np.load(CACHE_X, mmap_mode="r"), meta

    if verbose:
        print(f"Reading {config.BASE_MATRIX.name} (one-off, ~1 min) ...")
    df = load_base_matrix()
    df = df.dropna(subset=[config.ID_COL, config.TARGET_COL])

    ids = df[config.ID_COL].astype(str).tolist()
    y = pd.to_numeric(df[config.TARGET_COL], errors="coerce").to_numpy(float)

    drop = [config.TARGET_COL] + [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    X_df = df.drop(columns=drop).apply(pd.to_numeric, errors="coerce")

    all_nan = X_df.columns[X_df.isna().all()].tolist()
    if all_nan:
        if verbose:
            print(f"  dropping {len(all_nan)} all-NaN columns")
        X_df = X_df.drop(columns=all_nan)

    X = X_df.to_numpy(dtype=np.float32)
    np.save(CACHE_X, X)
    meta = {
        "ids": ids,
        "y": y.tolist(),
        "feature_names": list(X_df.columns),
        "dropped_all_nan": all_nan,
        "shape": list(X.shape),
    }
    CACHE_META.write_text(json.dumps(meta))
    if verbose:
        print(f"  cached {CACHE_X.name} {X.shape} and {CACHE_META.name}")
    return np.load(CACHE_X, mmap_mode="r"), meta


LABEL_DIR = config.INTERIM / "labels"


def available_labels() -> list[str]:
    return sorted(p.stem for p in LABEL_DIR.glob("*.csv")) if LABEL_DIR.exists() else []


def load_label(name: str) -> pd.Series:
    """An alternative target, keyed by sgRNAID.

    Files live in `data/interim/labels/<name>.csv` with columns sgRNAID,y.
    Swapping the label is not a feature experiment -- it changes what every
    number means -- so it gets its own tag and its own baseline.
    """
    path = LABEL_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Available labels: {available_labels()}"
        )
    df = pd.read_csv(path)
    return df.set_index(config.ID_COL)["y"]


def load_dataset(families=(), verbose: bool = True, label: str | None = None):
    """Base features plus the chosen blocks, aligned on sgRNAID.

    Returns (X, y, feature_names, ids, family_of_column). With `label`, the
    published target is replaced by the named alternative and rows without a
    value for it are dropped -- so the comparison is like-for-like on rows.
    """
    X, meta = build_cache(verbose=verbose)
    ids = meta["ids"]
    y = np.asarray(meta["y"], dtype=float)
    names = list(meta["feature_names"])
    family_of = ["base"] * len(names)

    X = np.asarray(X)
    order = pd.Index(ids)

    for fam in families:
        block = load_block(fam).set_index(config.ID_COL)
        block = block.reindex(order)
        cols = [c for c in block.columns]
        values = block.to_numpy(dtype=np.float32)
        # Columns that are entirely missing carry no information and would
        # only dilute the top-300 selection.
        keep = ~np.all(np.isnan(values), axis=0)
        if verbose and (~keep).sum():
            print(f"  {fam}: dropping {(~keep).sum()} all-NaN columns")
        X = np.hstack([X, values[:, keep]])
        names += [c for c, k in zip(cols, keep) if k]
        family_of += [fam] * int(keep.sum())
        if verbose:
            print(f"  {fam}: +{int(keep.sum()):,} columns -> {X.shape[1]:,} total")

    if label:
        series = load_label(label).reindex(ids)
        keep = series.notna().to_numpy()
        if verbose:
            print(f"  label '{label}': {keep.sum():,} / {len(ids):,} guides have a "
                  f"value; the rest are dropped")
        X = X[keep]
        y = series.to_numpy(dtype=float)[keep]
        ids = [i for i, k in zip(ids, keep) if k]

    return X, y, names, ids, family_of


def export_csv(families, out_path=None, verbose: bool = True):
    """Write a merged CSV shaped like the original, title row and all."""
    out_path = out_path or (
        config.PROCESSED / ("matrix_" + ("_".join(families) or "base") + ".csv")
    )
    df = load_base_matrix()
    for fam in families:
        block = load_block(fam)
        df = df.merge(block, on=config.ID_COL, how="left")
        if verbose:
            print(f"  merged {fam} -> {df.shape[1]:,} columns")

    with open(out_path, "w") as fh:
        fh.write("Engineered feature matrix, built by sgrna.build_matrix"
                 + "," * (df.shape[1] - 1) + "\n")
        df.to_csv(fh, index=False)
    if verbose:
        print(f"Wrote {out_path} ({df.shape[0]:,} x {df.shape[1]:,})")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", nargs="*", default=[], choices=ALL)
    ap.add_argument("--cache", action="store_true", help="build the numpy cache only")
    ap.add_argument("--force", action="store_true", help="rebuild the cache")
    ap.add_argument("--export", choices=["csv"], help="also write a merged CSV")
    args = ap.parse_args(argv)

    if args.cache or not args.families:
        build_cache(force=args.force)
    if args.families:
        X, y, names, ids, fam = load_dataset(args.families)
        print(f"dataset: {X.shape[0]:,} x {X.shape[1]:,}")
        if args.export == "csv":
            export_csv(args.families)
    return 0


if __name__ == "__main__":
    sys.exit(main())
