"""Paired summaries of the shape ablation: every arm against a reference arm, fold by fold.

Folds are identical across arms for a given seed (gene-grouped, deterministic), so the fold-level
difference is a paired quantity. Reported: mean difference, the mean per seed (three values), and
the number of the 15 folds in which the arm is ahead. No p-values are quoted: the folds share genes,
so they are not independent draws and a standard error over 15 of them would overstate precision.

    python -m sgrna.ext.analyze
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import OUT


def load():
    d = pd.read_csv(OUT / "shape_ablation.csv").set_index("arm")
    return d, {a: np.array([float(x) for x in d.loc[a, "fold_rho"].split(";")]) for a in d.index}


def paired(f, arm, ref, nseeds=3):
    n = min(len(f[arm]), len(f[ref]))
    diff = f[arm][:n] - f[ref][:n]
    per_seed = diff.reshape(nseeds, -1).mean(1) if n % nseeds == 0 else np.array([np.nan])
    return dict(delta=float(diff.mean()), per_seed=";".join(f"{x:+.4f}" for x in per_seed),
                folds_ahead=f"{int((diff > 0).sum())}/{n}")


def table(refs=(("base", "base"), ("A", "A"))):
    d, f = load()
    rows = []
    for arm in d.index:
        r = dict(arm=arm, n_ext=int(d.loc[arm, "n_ext"]), rho=round(float(d.loc[arm, "spearman"]), 4),
                 selected=round(float(d.loc[arm, "n_selected_ext"]), 1))
        for name, ref in refs:
            if ref in f and arm != ref:
                p = paired(f, arm, ref)
                r.update({f"d_vs_{name}": round(p["delta"], 4), f"seeds_vs_{name}": p["per_seed"],
                          f"ahead_vs_{name}": p["folds_ahead"]})
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    t = table()
    print(t.to_string(index=False))
    t.to_csv(OUT / "shape_ablation_paired.csv", index=False)
