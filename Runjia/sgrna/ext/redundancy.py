"""How much of each shape / 3D block is already in family A (flank composition)?

For every column of a block, the out-of-fold R^2 of a ridge regression from family A's columns
(5-fold, gene-grouped). A column with R^2 near 1 is a re-expression of what A holds; a column
with R^2 near 0 carries something A does not. Reported per block and per column kind.

    python -m sgrna.ext.redundancy
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from . import shape_build
from .common import OUT
from .shape_ablate import Cohort


def run():
    co = Cohort()
    A, nA = co.A()
    A = np.nan_to_num((A - np.nanmean(A, 0)) / (np.nanstd(A, 0) + 1e-9))
    rows = []
    for src in ("mc", "md", "xrc"):
        for blk in ("mean", "vec", "geom"):
            arr, nm = co.ext(src, None, blk)
            Y = np.nan_to_num((arr - np.nanmean(arr, 0)) / (np.nanstd(arr, 0) + 1e-9))
            pred = np.zeros_like(Y)
            for tr, te in GroupKFold(5).split(A, groups=co.groups):
                pred[te] = Ridge(alpha=50.0).fit(A[tr], Y[tr]).predict(A[te])
            r2 = 1 - ((Y - pred) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
            for n, v in zip(nm, r2):
                rows.append(dict(source=src, block=blk, col=n.split(".", 3)[-1], r2_from_A=float(v)))
            print(f"  {src} {blk}: {len(nm)} cols, mean R2 {r2.mean():.3f}, share >0.9: {(r2 > 0.9).mean():.2f}, "
                  f"share <0.2: {(r2 < 0.2).mean():.2f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "shape_redundancy_from_A.csv", index=False)


if __name__ == "__main__":
    run()
