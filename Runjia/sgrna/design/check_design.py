"""Check the design tool against the stored feature matrices: it reads the same DNA, builds the same 6,517 columns and scores the same.

For 600 random curated guides, on either strand, the tool is given only the genome and the guide's coordinates. It must give back the protospacer, PAM and both 1,000 nt
flanks of the curated guide index, the stored base and flank columns, and the same prediction from them; and asking for the cut site of each guide must list that guide at
distance 0. Written to `results/post_freeze/design_check.csv`.

    python -m sgrna.design.check_design
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..common import OUT
from . import design_guide as D


def main() -> int:
    from ..build.build_features import load_curated_index
    from ..cohorts.curated_ablation import _base, _block

    _, g = D.read_fasta(config.GENOME_FASTA)
    X, y, names, protos = _base()
    A, an = _block("a_flank", protos)
    idx = load_curated_index(verbose=False).set_index(config.ID_COL).reindex(protos)
    pick = np.sort(np.random.default_rng(0).choice(len(protos), 600, replace=False))
    sub = idx.iloc[pick]
    ctx = pd.DataFrame([D.site_context(g, int(l) - 1, s) for l, s in zip(sub["left"], sub["strand"])])
    same = {k: int((ctx[k].to_numpy() == sub[k].astype(str).to_numpy()).sum()) for k in ("protospacer", "pam", "upstream", "downstream")}
    sl = D.Slicer("tuned")
    F = sl.features(ctx)
    S = np.hstack([X[pick], A[pick]])
    nan_same = bool((np.isnan(F) == np.isnan(S)).all())
    diff = float(np.nanmax(np.abs(F - S)))
    p_new, p_old = sl.predict_features(F), sl.predict_features(S)
    # the cut site of each guide lists that guide, at distance 0
    found = 0
    for l, s in zip(sub["left"], sub["strand"]):
        s0 = int(l) - 1
        cands = D.find_candidates(g, D.cut_after(s0, s), 0)
        found += any(c["s"] == s0 and c["strand"] == s and c["distance"] == 0 for c in cands)
    rho = float(spearmanr(p_new, y[pick]).statistic)
    rows = [("guides checked", len(pick)), ("guides on the minus strand", int((sub["strand"] == "-").sum())),
            ("protospacer read from the genome equals the index", same["protospacer"]), ("PAM equals the index", same["pam"]),
            ("upstream flank equals the index", same["upstream"]), ("downstream flank equals the index", same["downstream"]),
            ("columns compared", F.shape[1]), ("missing-value pattern equal", float(nan_same)), ("maximum absolute difference in a column", diff),
            ("maximum absolute difference in the prediction", float(np.abs(p_new - p_old).max())),
            ("guides listed at distance 0 by their own cut site", found),
            ("Spearman rho of the prediction against the measured score (these guides were in training)", rho)]
    out = pd.DataFrame(rows, columns=["item", "value"])
    out.to_csv(OUT / "design_check.csv", index=False)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
