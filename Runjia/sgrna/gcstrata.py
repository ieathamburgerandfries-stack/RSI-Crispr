"""Is the flanking-context effect just target GC wearing a different hat?

Finding 6 offers a mechanism as a hypothesis: a duplex-opening step limited by
local GC. The evidence is correlational -- target GC correlates -0.201 with the
score, flank GC +0.159 -- and correlational evidence of that shape has an
obvious alternative reading. Target GC and flank GC are themselves correlated
(isochore-like composition runs over kilobases), so the flank block could be
scoring by **restating the target's own composition**, which the model already
holds in its mononucleotide indicators.

The conditional test. Slice the rows into strata of **target GC** and ask the
question again inside each slice. Within a stratum the target composition is
near-constant by construction, so anything flank GC still explains is not
target GC in disguise.

Two arms, cheap then expensive:

  --corr    Spearman of flank GC against the label within each target-GC
            stratum, plus the partial correlation controlling target GC.
            Seconds.
  --run     The full ablation inside each stratum: baseline against
            baseline + family A, grouped CV in 100 kb blocks (the same
            precaution as `shiftnull.py` -- a positional feature under random
            KFold can score by locus recognition). Minutes to an hour.

A flank effect that is a target-GC proxy collapses inside the strata. One that
survives is a property of the surrounding DNA.

    python -m sgrna.gcstrata --corr
    python -m sgrna.gcstrata --run
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config

CORR_OUT = config.RESULTS / "gc_strata_corr.csv"
CORR_CURATED_OUT = config.RESULTS / "gc_strata_corr_curated.csv"
RUN_OUT = config.RESULTS / "gc_strata_ablation.csv"

N_STRATA = 5
FLANK_GC = ["eng.flank.comp.up500.gc", "eng.flank.comp.dn500.gc"]


def _frame() -> pd.DataFrame:
    """Target GC, flank GC and the label, one row per located guide."""
    from .io_utils import load_block, load_guide_index

    idx = load_guide_index()
    idx = idx[idx["located"].astype(bool)] if "located" in idx else idx
    proto = idx["protospacer"].astype(str).str.upper()
    tgc = proto.map(lambda s: (s.count("G") + s.count("C")) / max(len(s), 1))

    block = load_block("a_flank").set_index(config.ID_COL)
    fgc = block[FLANK_GC].mean(axis=1)

    df = pd.DataFrame({
        config.ID_COL: idx[config.ID_COL].values,
        "target_gc": tgc.values,
        "y": idx["cut_score"].values,
    }).set_index(config.ID_COL)
    df["flank_gc"] = fgc.reindex(df.index).values
    return df.dropna()


def _frame_curated() -> pd.DataFrame:
    """The same three columns on the curated 33,567-guide cohort.

    `_frame()` above reads `load_guide_index()`, which is the 13,880-row
    published cohort. Everything it produced was quoted in Finding 3 without a
    cohort label and went stale at the pivot: flank GC's marginal association is
    **3.6x larger** here, because the relationship itself is steeper in the rows
    crisprHAL added (Appendix B2a). The conditioning result is the part that has
    to be re-measured, not just re-labelled -- a suppressor's effect depends on
    the correlations among all three variables, and two of the three moved.
    """
    from .curated_ablation import _base, _block

    X, y, names, protos = _base()
    proto = pd.Series([str(q).upper() for q in protos])
    tgc = proto.map(lambda q: (q.count("G") + q.count("C")) / max(len(q), 1))

    v, fn = _block("a_flank", protos)
    cols = [list(fn).index(c) for c in FLANK_GC if c in list(fn)]
    if not cols:
        raise KeyError(f"none of {FLANK_GC} on the curated cohort")
    fgc = np.nanmean(v[:, cols], axis=1)

    df = pd.DataFrame({"target_gc": tgc.to_numpy(), "y": y, "flank_gc": fgc},
                      index=pd.Index(protos, name=config.ID_COL))
    return df.dropna()


def _strata(df: pd.DataFrame) -> pd.Series:
    """Equal-count bins of target GC. Ties are common at 20 nt, so rank first."""
    r = df["target_gc"].rank(method="first")
    return pd.qcut(r, N_STRATA, labels=False)


def corr(cohort: str = "curated", verbose: bool = True) -> pd.DataFrame:
    from scipy.stats import spearmanr

    df = _frame_curated() if cohort == "curated" else _frame()
    df["stratum"] = _strata(df)

    rows = [dict(stratum="all", n=len(df),
                 gc_lo=df["target_gc"].min(), gc_hi=df["target_gc"].max(),
                 rho_target=spearmanr(df["target_gc"], df["y"]).statistic,
                 rho_flank=spearmanr(df["flank_gc"], df["y"]).statistic)]
    for s, g in df.groupby("stratum"):
        rows.append(dict(stratum=int(s), n=len(g),
                         gc_lo=g["target_gc"].min(), gc_hi=g["target_gc"].max(),
                         rho_target=spearmanr(g["target_gc"], g["y"]).statistic,
                         rho_flank=spearmanr(g["flank_gc"], g["y"]).statistic))
    out = pd.DataFrame(rows)

    # Partial correlation of flank GC with the label, controlling target GC,
    # on ranks -- the Spearman analogue.
    rk = df[["target_gc", "flank_gc", "y"]].rank()
    c = rk.corr().to_numpy()
    r_fy, r_ft, r_ty = c[1, 2], c[1, 0], c[0, 2]
    partial = (r_fy - r_ft * r_ty) / np.sqrt((1 - r_ft**2) * (1 - r_ty**2))

    out.attrs["partial"] = float(partial)
    out["rho_flank_partial_overall"] = round(float(partial), 4)
    out["rho_flank_vs_target_gc"] = round(float(r_ft), 4)
    out["cohort"] = cohort
    out.to_csv(CORR_OUT if cohort == "published" else CORR_CURATED_OUT,
               index=False)
    if verbose:
        print(out.round(4).to_string(index=False))
        print(f"\npartial rho(flank GC, label | target GC) = {partial:+.4f}"
              f"   [rho(flank GC, target GC) = {r_ft:+.4f}]")
        print(f"  cohort: {cohort}, n = {len(df):,}")
        print(f"written -> {CORR_OUT if cohort == 'published' else CORR_CURATED_OUT}")
    return out


def run(seeds=(41, 42, 43), n_splits: int = 5, n_features: int = 300,
        verbose: bool = True) -> pd.DataFrame:
    from .build_matrix import load_dataset
    from .run_ablation import cross_validate, load_groups

    X, y, names, ids, fam = load_dataset(["a_flank"], verbose=verbose)
    groups = load_groups(ids, by="bin100k")
    is_flank = np.array([f == "a_flank" for f in fam])
    if verbose:
        print(f"  {X.shape[0]:,} rows, {is_flank.sum()} flank columns")

    df = _frame()
    stratum = _strata(df).reindex(pd.Index(ids, name=config.ID_COL))
    keep = stratum.notna().to_numpy()

    rows = []
    for s in range(N_STRATA):
        m = keep & (stratum.fillna(-1).to_numpy() == s)
        if m.sum() < 500:
            continue
        band = df.loc[df.index.intersection(np.asarray(ids)[m]), "target_gc"]
        for arm, cols in (("baseline", ~is_flank), ("a_flank", np.ones_like(is_flank))):
            res = cross_validate(
                X[np.ix_(m, cols)], y[m],
                [n for n, c in zip(names, cols) if c],
                [f for f, c in zip(fam, cols) if c],
                n_features=n_features, seeds=list(seeds), n_splits=n_splits,
                groups=groups[m], verbose=False,
                tag=f"gcstrat{s}_{arm}", experiment=f"stratum{s}_{arm}")
            rows.append(dict(stratum=s, arm=arm, n=int(m.sum()),
                             gc_lo=round(float(band.min()), 3),
                             gc_hi=round(float(band.max()), 3),
                             spearman=res["summary"]["spearman"]["mean"],
                             spearman_sd=res["summary"]["spearman"]["sd"]))
            if verbose:
                print(f"  stratum {s} ({band.min():.2f}-{band.max():.2f} GC, "
                      f"n={m.sum():,}) {arm:>9}: rho {res['summary']['spearman']['mean']:.4f}")
            pd.DataFrame(rows).to_csv(RUN_OUT, index=False)

    out = pd.DataFrame(rows)
    if verbose and not out.empty:
        w = out.pivot(index="stratum", columns="arm", values="spearman")
        w["delta"] = (w["a_flank"] - w["baseline"]).round(4)
        print("\nflank gain within each target-GC stratum:")
        print(w.round(4).to_string())
        print(f"written -> {RUN_OUT}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corr", action="store_true")
    ap.add_argument("--cohort", choices=["curated", "published"],
                    default="curated")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    a = ap.parse_args(argv)
    if a.corr or not a.run:
        corr(cohort=a.cohort)
    if a.run:
        run(seeds=tuple(a.seeds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
