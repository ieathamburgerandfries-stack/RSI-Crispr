"""Inside the coding-strand guides, is a family's gain between genes or within them?

`strand_families --within` found family A (and transcription context) adding +0.12 to
+0.27 rho among coding-strand guides, where gene-grouped CV holds whole genes out.
A gain can come from ranking genes (flank composition tracks gene-level depletion)
or from ranking guides inside one gene (the local flank). Split the out-of-fold
predictions of one seed into the two: rho of gene means, and rho of the values with
each gene's mean removed.

    python -m sgrna.strand_within_gene
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config
from .cui_screen import _essential_frame
from .run_ablation import cross_validate
from .strand_families import _blocks

OUT = config.RESULTS / "strand_within_gene.csv"


def main() -> int:
    f = _essential_frame()
    meta = json.loads((config.INTERIM / "cui_crispri" / "meta.json").read_text())
    ess = np.asarray([str(e) for e in meta["essential"]], dtype=object) == "True"
    gene = np.asarray([g if isinstance(g, str) else "" for g in meta["gene"]],
                      dtype=object)
    has = (gene != "") & (gene != "nan") & (gene != "None")
    sel = ess & has
    protos = np.asarray(meta["protospacers"])[sel]
    gene = gene[sel]
    blocks = _blocks(protos)
    cod = f["coding"].ravel()
    m = cod == 1
    groups = np.asarray(dict(f["groupings"])["genes"])[m]
    y, g = f["y"][m], gene[m]
    rows = []
    for arm in ("base", "a_flank", "g_transcription", "d_supercoiling", "j_periodicity",
                "i_shape"):
        X = f["Xb"] if arm == "base" else np.hstack([f["Xb"], blocks[arm][0]])
        nm = f["nb"] if arm == "base" else f["nb"] + blocks[arm][1]
        res = cross_validate(X[m], y, nm, ["arm"] * len(nm), n_features=config.N_FEATURES,
                             seeds=[41], n_splits=config.N_SPLITS, verbose=False,
                             groups=groups, tag=f"cui_sfg_coding_{arm}",
                             experiment=arm, return_oof=True)
        p = res["oof_pred"]
        d = pd.DataFrame(dict(g=g, y=y, p=p))
        d = d[d.groupby("g").y.transform("size") >= 5]
        gm = d.groupby("g")[["y", "p"]].mean()
        within = d.y - d.groupby("g").y.transform("mean"), d.p - d.groupby("g").p.transform("mean")
        rows.append(dict(arm=arm, n=len(d), genes=len(gm),
                         overall=spearmanr(d.y, d.p).statistic,
                         between_genes=spearmanr(gm.y, gm.p).statistic,
                         within_genes=spearmanr(*within).statistic))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
