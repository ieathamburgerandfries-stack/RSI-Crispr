"""The head-to-head with crisprHAL 2 on folds that hold whole genes out (the lowest-leakage protocol of this project).

The head-to-head of Part 5 is on crisprHAL 2's own random splits, as it ships. A random split can reward a model for recognising a locus, and a network that reads
378 nt of context can recognise loci better than one that reads a 20-mer and a few composition summaries, so that comparison may favour crisprHAL 2. This module sets
up the same comparison on the gene-grouped folds used everywhere else (whole genes in a fold, guides outside genes grouped by 100 kb block; seed 41):

  folds    writes `data/interim/crisprhal_widewindow_input_genefolds.csv.gz` (the Colab input file with one more column, `gene_fold`, 0-4) and the per-fold rho of SLICER's
           default and tuned models on exactly those folds (`gene_grouped_slicer.csv`)
  assemble reads crisprHAL 2's per-fold rho from `results/crisprhal_gene_grouped_fcolab_b1024.csv` (written by `colab/crisprhal_gene_grouped.py`, about 30 minutes on a T4, at
           crisprHAL 2's published batch size of 1,024) and writes the paired comparison (`gene_grouped_vs_crisprhal.csv`). A first run used batch 256 and is kept in
           `crisprhal_gene_grouped_fcolab_b256.csv`; it handicaps crisprHAL 2 (256 cost it 0.016 on random-row folds) and is reported only as that, beside the published batch

    python -m sgrna.compare.gene_grouped_vs_crisprhal folds
    python -m sgrna.compare.gene_grouped_vs_crisprhal assemble
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..common import OUT

SLICER = OUT / "gene_grouped_slicer.csv"
HAL = OUT / "crisprhal_gene_grouped_fcolab_b1024.csv"
HAL256 = OUT / "crisprhal_gene_grouped_fcolab_b256.csv"
RESULT = OUT / "gene_grouped_vs_crisprhal.csv"
SEED = 41


def folds() -> None:
    from ..cohorts.curated_ablation import _base, _block
    from ..controls import sealed_holdout_gene as sg
    from ..controls.mappability_stratum import _groups
    from .vs_crisprhal import member_specs, train_predict

    X, y, names, protos = _base()
    A, _ = _block("a_flank", protos)
    X = np.hstack([X, A]).astype(np.float32)
    fold = sg._fold_ids(_groups(protos), 5, SEED)
    src = pd.read_csv(config.INTERIM / "crisprhal_widewindow_input.csv.gz")
    fmap = pd.Series(fold, index=protos)
    assert src.protospacer.isin(fmap.index).all()
    src["gene_fold"] = src.protospacer.map(fmap).astype(int)
    src.to_csv(config.INTERIM / "crisprhal_widewindow_input_genefolds.csv.gz", index=False)
    print("  folds written; sizes", np.bincount(fold).tolist(), flush=True)
    rows = []
    for k in range(5):
        va = fold == k
        preds, _ = train_predict(member_specs(["champion", "d099x"]), X[~va], y[~va], X[va], SEED)
        for arm, name in (("SLICER default", "champion"), ("SLICER tuned", "d099x")):
            rows.append(dict(model=arm, fold=k, n_test=int(va.sum()), spearman=float(spearmanr(y[va], preds[name]).statistic)))
        print(f"  fold {k}: default {rows[-2]['spearman']:.4f}  tuned {rows[-1]['spearman']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(SLICER, index=False)


def assemble() -> None:
    s = pd.read_csv(SLICER)
    parts = [("crisprHAL 2", HAL), ("crisprHAL 2 (batch 256)", HAL256)]
    d = s[["model", "fold", "spearman"]]
    have = []
    for name, path in parts:
        if path.exists():
            d = pd.concat([d, pd.read_csv(path).assign(model=name)[["model", "fold", "spearman"]]])
            have.append(name)
    p = d.pivot_table(index="fold", columns="model", values="spearman")
    rows = []
    for m in ("SLICER default", "SLICER tuned"):
        for h in have:
            diff = p[m] - p[h]
            rows.append(dict(model=m, against=h, mean=float(p[m].mean()), sd=float(p[m].std(ddof=1)), crisprhal_mean=float(p[h].mean()), crisprhal_sd=float(p[h].std(ddof=1)),
                             lead_mean=float(diff.mean()), lead_sd=float(diff.std(ddof=1)), folds_ahead=int((diff > 0).sum())))
    r = pd.DataFrame(rows)
    r.to_csv(RESULT, index=False)
    print(p.round(4).to_string())
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    {"folds": folds, "assemble": assemble}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
