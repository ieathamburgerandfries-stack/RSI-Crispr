"""Build the shape/3D blocks for the curated cohort, one file per (source, control, block).

    python -m sgrna.ext.shape_build
Output: data/interim/ext/shape/<source>_<control>_<block>.npy (+ .json names), rows in the order
of `curated_ablation._base()`.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from sgrna import config
from sgrna.curated_ablation import _base

from . import geom
from .common import INTERIM

DIR = INTERIM / "shape"
DIR.mkdir(parents=True, exist_ok=True)


def path(source, control, block):
    return DIR / f"{source}_{control or 'real'}_{block}"


def build(sources=("mc", "md", "xrc"), controls=(None,), force=False):
    _, _, _, protos = _base()
    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv").set_index(config.ID_COL)
    idx = idx.reindex(protos).reset_index()
    assert idx.upstream.notna().all()
    for src in sources:
        if src == "dcomp":
            for ctl in controls:
                if not force and path(src, ctl, "mean").with_suffix(".npy").exists():
                    continue
                arr, nm = geom.dinuc_composition(idx, control=ctl if ctl in ("block", "arm") else None)
                np.save(path(src, ctl, "mean").with_suffix(".npy"), arr)
                path(src, ctl, "mean").with_suffix(".json").write_text(json.dumps(nm))
                print(f"  dcomp {ctl or 'real'}: mean={arr.shape[1]}", flush=True)
            continue
        if src == "comp":
            for ctl in controls:
                if not force and path(src, ctl, "mean").with_suffix(".npy").exists():
                    continue
                arr, nm = geom.composition(idx, control=ctl if ctl in ("block", "arm") else None)
                np.save(path(src, ctl, "mean").with_suffix(".npy"), arr)
                path(src, ctl, "mean").with_suffix(".json").write_text(json.dumps(nm))
                print(f"  comp {ctl or 'real'}: mean={arr.shape[1]}", flush=True)
            continue
        for ctl in controls:
            if not force and all(path(src, ctl, b).with_suffix(".npy").exists()
                                 for b in ("mean", "vec", "geom")):
                continue
            t0 = time.time()
            r = geom.compute(idx, src, control=ctl)
            for blk, (arr, names) in r.items():
                np.save(path(src, ctl, blk).with_suffix(".npy"), arr)
                path(src, ctl, blk).with_suffix(".json").write_text(json.dumps(names))
            print(f"  {src} {ctl or 'real'}: {time.time() - t0:.0f}s "
                  + " ".join(f"{k}={v[0].shape[1]}" for k, v in r.items()), flush=True)


def load(source, control, block):
    p = path(source, control, block)
    return (np.load(p.with_suffix(".npy")), json.loads(p.with_suffix(".json").read_text()))


if __name__ == "__main__":
    build(("mc", "md", "xrc"), (None,))
    build(("mc",), ("block", "arm"))
