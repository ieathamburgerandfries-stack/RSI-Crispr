"""Recomputing the ceiling: is the disagreement between two screens noise, or biology?

The project's ceiling estimate comes from the attenuation argument. Two screens
of the same guide library agree at rho 0.810, and if each observed score is a
shared signal plus *independent noise*, that correlation **is** the reliability,
so no model can correlate better than sqrt(0.810) ~= 0.90 with a single observed
score.

Everything rests on "independent noise", and it is not obviously true. The two
screens use WT-SpCas9 and eSpCas9 -- the same enzyme with three point mutations
-- so part of the 0.19 disagreement could be a **real, systematic** difference
in what the two enzymes do, not measurement error. If so the signal is larger
than 0.810 implies and the ceiling is higher. Pulling the other way, the two
screens share their library, so sequence-specific batch effects cancel and the
agreement is flattered; then the ceiling is lower.

Published read counts would settle it directly, and they do not exist -- the
underlying data is raw reads in SRA (PRJNA450978 and others), so replicate-level
scores would mean re-running the authors' counting pipeline.

So this module settles it a different way, with data we have. **Systematic
effects are predictable; noise is not.** Fit a model to the *difference* between
the two screens' scores for the same guide:

  * if the difference is predictable from sequence, it is enzyme biology, and
    the ceiling is higher than 0.90;
  * if it is not predictable, it is noise, and 0.810 is a reliability estimate.

Reported with the ceiling as a **range under stated assumptions**, which is what
the evidence supports, rather than one number.

    python -m sgrna.ceiling2
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .. import config

OUT = config.RESULTS / "ceiling_recomputed.json"
OUT_FOLDS = config.RESULTS / "ceiling_recomputed_folds.csv"


def _screen(name: str) -> pd.DataFrame:
    """Protospacer and score for one crisprHAL screen, deduplicated."""
    from ..screens import transfer
    d = transfer.load_dataset(name, verbose=False)
    cols = {c.lower(): c for c in d.columns}
    proto = cols.get("protospacer")
    score = cols.get("score") or cols.get("cut_score")
    out = d[[proto, score]].rename(columns={proto: "protospacer", score: "score"})
    return out.groupby("protospacer", as_index=False)["score"].mean()


def run(seed: int = 41, n_splits: int = 5, verbose: bool = True) -> dict:
    import lightgbm as lgb
    from sklearn.model_selection import KFold

    from ..compare.headtohead import _assemble

    wt = _screen("WT-SpCas9")
    es = _screen("eSpCas9")
    both = wt.merge(es, on="protospacer", suffixes=("_wt", "_esp"))
    rho_screens = float(spearmanr(both["score_wt"], both["score_esp"]).statistic)
    if verbose:
        print(f"  {len(both):,} shared protospacers, screens agree at "
              f"rho {rho_screens:.4f}")

    # The two scores are on different scales, so the difference is only
    # meaningful after putting both on a common one. Ranks are the natural
    # choice because rho is a rank statistic.
    n = len(both)
    r_wt = rankdata(both["score_wt"]) / n
    r_es = rankdata(both["score_esp"]) / n
    both["disagreement"] = r_wt - r_es

    # Features for exactly these guides: the same matrix the head-to-head uses.
    X, y_cut, _ = _assemble("curated_flank", verbose=False)
    meta = json.loads((config.INTERIM / "expanded_meta.json").read_text())
    protos = np.asarray(meta["protospacers"])
    order = pd.Index(protos)
    keep = order.isin(set(both["protospacer"]))
    X = X[keep]
    aligned = both.set_index("protospacer").reindex(order[keep])
    y_dis = aligned["disagreement"].to_numpy(dtype=float)
    if verbose:
        print(f"  features for {X.shape[0]:,} of them, {X.shape[1]:,} columns")

    def cv(target, label):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        rows = []
        for fold, (tr, va) in enumerate(kf.split(X), start=1):
            m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                  num_leaves=16, max_depth=4, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8,
                                  random_state=seed, n_jobs=-1, verbose=-1)
            m.fit(X[tr], target[tr])
            pred = np.asarray(m.predict(X[va])).ravel()
            rows.append(dict(target=label, fold=fold,
                             rho=float(spearmanr(target[va], pred).statistic)))
            if verbose:
                print(f"    {label:14s} fold {fold}  rho {rows[-1]['rho']:.4f}",
                      flush=True)
        return pd.DataFrame(rows)

    # Control: the same features predicting the cut score itself. Without it
    # there is no way to tell "the difference is unpredictable" from "these
    # features predict nothing on this subset".
    folds = pd.concat([cv(y_dis, "disagreement"),
                       cv(aligned["score_wt"].to_numpy(float), "cut score")],
                      ignore_index=True)
    folds.to_csv(OUT_FOLDS, index=False)

    rho_dis = float(folds[folds.target == "disagreement"]["rho"].mean())
    rho_cut = float(folds[folds.target == "cut score"]["rho"].mean())

    # Share of the disagreement that is systematic. rho^2 is the usual
    # variance-explained reading; it is a rough conversion, stated as such.
    systematic = max(rho_dis, 0.0) ** 2
    gap = 1.0 - rho_screens
    r_lo = rho_screens                       # all disagreement is noise
    r_mid = rho_screens + gap * systematic   # the predictable part is biology
    r_hi = 1.0                               # all disagreement is biology

    out = dict(
        n_shared=int(len(both)),
        rho_between_screens=rho_screens,
        rho_predicting_disagreement=rho_dis,
        rho_predicting_cut_score=rho_cut,
        systematic_share_of_disagreement=systematic,
        reliability_if_all_noise=r_lo,
        reliability_if_predictable_part_is_biology=r_mid,
        ceiling_if_all_noise=float(np.sqrt(r_lo)),
        ceiling_if_predictable_part_is_biology=float(np.sqrt(r_mid)),
        best_model_rho_for_reference=0.7181,
    )
    OUT.write_text(json.dumps(out, indent=2))
    if verbose:
        print()
        for k, v in out.items():
            print(f"  {k:46s} {v:.4f}" if isinstance(v, float) else
                  f"  {k:46s} {v}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)
    run(seed=args.seed, n_splits=args.folds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
