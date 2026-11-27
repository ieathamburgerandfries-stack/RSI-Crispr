"""The flank and the guide's own score add up rather than gate: an interaction term improves held-out R² by only +0.0035 and rho by +0.0007. Prediction I2 of the report, now tested.

If the flank's effect is larger where the guide itself is competent, a model with a flank x guide-score interaction should fit better than an additive one.
The guide score is the base model's out-of-fold prediction (`mappability_stratum_oof.csv`, the 20-mer representation only); the flank is the GC fraction of
the 250 nt on each side (averaged) and the indicator of its lowest decile (the AT-rich step). Two ordinary least squares fits, scored on whole genes held out
(five folds, the three seeds of the OOF run): y ~ score + flank GC + step against y ~ the same + score x flank GC + score x step. A tree model would find the
interaction itself, so the question is asked of linear models, where an interaction has to be supplied.

    python -m sgrna.interpret.gate_test
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..common import OUT
from .flank_scale import flank_composition

RESULT = OUT / "gate_test.csv"


def main() -> int:
    from ..build.build_features import load_curated_index
    from ..harness import folds as sg

    o = pd.read_csv(OUT / "mappability_stratum_oof.csv")
    idx = load_curated_index(verbose=False).reset_index(drop=True)
    assert (idx[config.ID_COL].to_numpy() == o[config.ID_COL].to_numpy()).all()
    up, dn = flank_composition(idx, (250,))[250]
    f = ((up[:, 1] + up[:, 2]) + (dn[:, 1] + dn[:, 2])) / 2
    y, g = o.y.to_numpy(), o.group.to_numpy()
    s = (o["base"].to_numpy() - o["base"].mean()) / o["base"].std()
    fz = (f - f.mean()) / f.std()
    low = (f <= np.quantile(f, 0.10)).astype(float)
    designs = {"additive": [s, fz, low], "with interaction": [s, fz, low, s * fz, s * low]}
    rows = []
    for seed in (41, 42, 43):
        fold = sg.fold_ids(g, 5, seed)
        for kind, cols in designs.items():
            D = np.column_stack([np.ones(len(y))] + cols)
            pred = np.empty(len(y))
            for k in range(5):
                va = fold == k
                pred[va] = D[va] @ np.linalg.lstsq(D[~va], y[~va], rcond=None)[0]
            for k in range(5):
                va = fold == k
                rows.append(dict(model=kind, seed=seed, fold=k + 1, rho=float(spearmanr(y[va], pred[va]).statistic),
                                 r2=float(1 - ((y[va] - pred[va]) ** 2).sum() / ((y[va] - y[va].mean()) ** 2).sum())))
    r = pd.DataFrame(rows)
    r.to_csv(RESULT, index=False)
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(y))] + designs["with interaction"]), y, rcond=None)[0]
    d = r.pivot_table(index=["seed", "fold"], columns="model", values=["rho", "r2"])
    print(r.groupby("model")[["rho", "r2"]].mean().round(4))
    for m in ("rho", "r2"):
        dd = d[m]["with interaction"] - d[m]["additive"]
        print(f"interaction minus additive, {m}: {dd.mean():+.4f} (sd {dd.std():.4f}), better in {(dd > 0).sum()} of {len(dd)} folds")
    print("coefficients [1, score, flank GC, step, score x flank GC, score x step]:", coef.round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
