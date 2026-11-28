"""The flank effect is a property of the surrounding DNA, not of the guide's own score, its target GC or transcription: the flank and the guide's own score add up rather than gate (an interaction term improves held-out R² by only +0.0035), holding target GC fixed raises flank GC's association with activity from +0.176 to +0.221 (and the flank gain is positive in every GC stratum), and the downstream flank's advantage is the same in low, middle and high expression genes (+0.046 to +0.049 in rho).

Four readings of the same signal, each answering one alternative to "the flank effect is local DNA composition", each a part (`--parts`) with its own files:

  profile       the data behind the figure: the AT-rich step, the slope by the guide's last base, and the correlation by window and by base      signal_decile / signal_slope / signal_corr / signal_target_gc .csv
  gate          does the flank gate on the guide's own competence? A linear interaction test on the base model's out-of-fold score                gate_test.csv
  gc-corr       the target-GC alternative, cheap arm: correlations of flank GC within strata of target GC (seconds)                               gc_strata_corr_curated.csv (and gc_strata_corr.csv for the published cohort)
  gc-ablation   the same, expensive arm: the full ablation inside each stratum, 100 kb grouped CV (minutes to an hour)                             gc_strata_ablation.csv
  asym-audit    why the orientation test of the downstream asymmetry is impossible on this screen                                                asymmetry_strand_audit.csv
  asym-run      the expression-conditioned test of the asymmetry                                                                                    asymmetry_expression.csv

    python -m sgrna.interpret.flank_signal --parts profile gate gc-corr asym-audit
    python -m sgrna.interpret.flank_signal --parts gc-ablation asym-run [--seeds 41 42 43] [--folds 5]

------------------------------------------------------------------------------------------------------------------------------
profile
------------------------------------------------------------------------------------------------------------------------------
All on the 33,567 curated guides (30,203 inside a gene for the first two tables, as in the report's decile table). Flank GC is the mean of the GC
fraction of the 250 nt upstream of the protospacer and the 250 nt downstream of the PAM, on the guide's strand.

  decile    mean label by decile of flank GC, with 95% intervals from resampling genes: the step at the AT-rich end
  slope     the label against flank GC in four strata, by the base at guide position 20 (next to the PAM): Spearman correlation with an interval from
            resampling genes, and the mean label in quintiles of flank GC; the slope is the thing that differs between strata
  corr      for windows of 10 nt to 5 kb on each side: the marginal Spearman correlation of the label with the fraction of each base (A, C, G, T) and of
            GC and purine, and the partial correlation given the target's own GC; the base-resolved version of the report's GC-versus-purine table

------------------------------------------------------------------------------------------------------------------------------
gate
------------------------------------------------------------------------------------------------------------------------------
If the flank's effect is larger where the guide itself is competent, a model with a flank x guide-score interaction should fit better than an additive one.
The guide score is the base model's out-of-fold prediction (`mappability_stratum_oof.csv`, the 20-mer representation only); the flank is the GC fraction of
the 250 nt on each side (averaged) and the indicator of its lowest decile (the AT-rich step). Two ordinary least squares fits, scored on whole genes held out
(five folds, the three seeds of the OOF run): y ~ score + flank GC + step against y ~ the same + score x flank GC + score x step. A tree model would find the
interaction itself, so the question is asked of linear models, where an interaction has to be supplied.

------------------------------------------------------------------------------------------------------------------------------
gc-corr and gc-ablation
------------------------------------------------------------------------------------------------------------------------------
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
            precaution as `family_a_controls.py` (part shift) -- a positional feature under random
            KFold can score by locus recognition). Minutes to an hour.

A flank effect that is a target-GC proxy collapses inside the strata. One that
survives is a property of the surrounding DNA.

------------------------------------------------------------------------------------------------------------------------------
asym-audit and asym-run
------------------------------------------------------------------------------------------------------------------------------
Family A helps more on the PAM side than the other side -- in the sides split,
+0.072 downstream against +0.027 upstream. Two explanations make different
predictions, and Part 12 listed separating them as cheap.

**PAM / R-loop geometry.** "Downstream" is defined relative to the guide: the
PAM sits immediately 3' of the protospacer, and the R-loop opens there and
propagates away from it. On this account the asymmetry belongs to the Cas9
complex and should not care which way the guide faces relative to its gene.

**Transcription.** RNA polymerase travels along a gene in one direction,
unwinding DNA ahead of itself and leaving supercoiling behind. That direction is
set by the gene, not the guide, so on this account the asymmetry should depend
on the guide's orientation within its gene, and should be stronger where there
is more transcription to do.

Those give two tests. One of them **cannot be run on this screen**, and finding
that out is the first thing this module reports; the other can.

Report section: Finding 7.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .. import config
from ..common import OUT
from .flank_scale import WINDOWS, flank_composition

DECILE, SLOPE, CORR = OUT / "signal_decile.csv", OUT / "signal_slope.csv", OUT / "signal_corr.csv"
GATE_OUT = OUT / "gate_test.csv"
CORR_OUT = OUT / "gc_strata_corr.csv"
CORR_CURATED_OUT = OUT / "gc_strata_corr_curated.csv"
RUN_OUT = OUT / "gc_strata_ablation.csv"
ASYM_OUT = OUT / "asymmetry_expression.csv"
ASYM_AUDIT_OUT = OUT / "asymmetry_strand_audit.csv"


# ======================================================================================================  profile
def _profile_load():
    from ..build.build_features import load_curated_index

    idx = load_curated_index(verbose=False).reset_index(drop=True)
    y = idx["score"].to_numpy(float)
    gene = np.array([g if isinstance(g, str) and g else "" for g in idx["gene_name"]], dtype=object)
    comp = flank_composition(idx, WINDOWS)
    proto = idx["protospacer"].astype(str)
    tgc = np.array([(s.count("G") + s.count("C")) / 20 for s in proto])
    last = proto.str[-1].to_numpy()
    return idx, y, gene, comp, tgc, last


def _boot(values_fn, gene, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    gi = pd.Series(np.arange(len(gene))).groupby(gene).apply(lambda s: s.to_numpy())
    keys = [k for k in gi.index if k != ""]
    out = []
    for _ in range(B):
        take = np.concatenate([gi[k] for k in rng.choice(keys, len(keys))])
        out.append(values_fn(take))
    return np.array(out)


def _partial(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def decile(y, gene, fgc):
    g = gene != ""
    yy, ff, gg = y[g], fgc[g], gene[g]
    edges = np.quantile(ff, np.linspace(0, 1, 11))
    d = np.clip(np.searchsorted(edges, ff, side="right") - 1, 0, 9)
    rows = []
    for k in range(10):
        m = d == k
        b = _boot(lambda t, k=k: float(yy[t][d[t] == k].mean()), gg, 400, seed=k)
        rows.append(dict(decile=k + 1, n=int(m.sum()), flank_gc=float(ff[m].mean()), mean_label=float(yy[m].mean()),
                         lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5))))
    rest = d >= 1
    rows[0]["gap_vs_rest"] = float(yy[d == 0].mean() - yy[rest].mean())
    return pd.DataFrame(rows)


def slope(y, gene, fgc, last):
    g = gene != ""
    yy, ff, gg, ll = y[g], fgc[g], gene[g], last[g]
    rows = []
    for b in "ACGT":
        m = ll == b
        r = float(spearmanr(ff[m], yy[m]).statistic)
        bs = _boot(lambda t, b=b: float(spearmanr(ff[t][ll[t] == b], yy[t][ll[t] == b]).statistic), gg, 300, seed=ord(b))
        q = np.clip(np.searchsorted(np.quantile(ff, np.linspace(0, 1, 6)), ff[m], side="right") - 1, 0, 4)
        for k in range(5):
            rows.append(dict(base=b, n=int(m.sum()), spearman=r, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                             quintile=k + 1, flank_gc=float(ff[m][q == k].mean()), mean_label=float(yy[m][q == k].mean())))
    return pd.DataFrame(rows)


def window_corr(y, comp, tgc):
    rows = []
    for W in WINDOWS:
        up, dn = comp[W]
        for side, arr in (("upstream", up), ("downstream", dn)):
            stats = {"A": arr[:, 0], "C": arr[:, 1], "G": arr[:, 2], "T": arr[:, 3], "GC": arr[:, 1] + arr[:, 2], "purine": arr[:, 0] + arr[:, 2]}
            for name, v in stats.items():
                rows.append(dict(window_nt=W, side=side, statistic=name, marginal=float(spearmanr(v, y).statistic), partial_given_target_gc=_partial(v, y, tgc)))
    return pd.DataFrame(rows)


def target_gc_strata(idx, y, gene):
    """Family A's gain inside quintiles of the target's own GC, on the stored gene-grouped out-of-fold predictions (`mappability_stratum_oof.csv`)."""
    from ..controls.mappability_stratum import _boot_gain

    o = pd.read_csv(OUT / "mappability_stratum_oof.csv")
    assert (o[config.ID_COL].to_numpy() == idx[config.ID_COL].to_numpy()).all()
    tgc = np.array([(s.count("G") + s.count("C")) / 20 for s in idx["protospacer"].astype(str)])
    edges = np.quantile(tgc, np.linspace(0, 1, 6))
    q = np.clip(np.searchsorted(edges, tgc, side="right") - 1, 0, 4)
    rows = []
    for k in range(5):
        s = o[q == k]
        lo, hi = _boot_gain(s, "base+A", "base", B=400)
        rows.append(dict(quintile=k + 1, target_gc_mean=float(tgc[q == k].mean()), n=int((q == k).sum()),
                         rho_base=float(spearmanr(s.y, s["base"]).statistic), rho_with_flank=float(spearmanr(s.y, s["base+A"]).statistic),
                         gain=float(spearmanr(s.y, s["base+A"]).statistic - spearmanr(s.y, s["base"]).statistic), gain_lo=lo, gain_hi=hi))
    return pd.DataFrame(rows)

def profile() -> int:
    idx, y, gene, comp, tgc, last = _profile_load()
    fgc = ((comp[250][0][:, 1] + comp[250][0][:, 2]) + (comp[250][1][:, 1] + comp[250][1][:, 2])) / 2
    d = decile(y, gene, fgc); d.to_csv(DECILE, index=False); print(d.round(3).to_string(index=False))
    s = slope(y, gene, fgc, last); s.to_csv(SLOPE, index=False); print(s.drop_duplicates("base")[["base", "n", "spearman", "lo", "hi"]].round(3).to_string(index=False))
    tg = target_gc_strata(idx, y, gene); tg.to_csv(OUT / "signal_target_gc.csv", index=False); print(tg.round(4).to_string(index=False))
    c = window_corr(y, comp, tgc); c.to_csv(CORR, index=False)
    print(c[c.statistic.isin(["GC", "purine"]) & c.window_nt.isin([50, 250, 500, 1000])].round(3).to_string(index=False))
    return 0


# ======================================================================================================  gate
def gate() -> int:
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
    r.to_csv(GATE_OUT, index=False)
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(y))] + designs["with interaction"]), y, rcond=None)[0]
    d = r.pivot_table(index=["seed", "fold"], columns="model", values=["rho", "r2"])
    print(r.groupby("model")[["rho", "r2"]].mean().round(4))
    for m in ("rho", "r2"):
        dd = d[m]["with interaction"] - d[m]["additive"]
        print(f"interaction minus additive, {m}: {dd.mean():+.4f} (sd {dd.std():.4f}), better in {(dd > 0).sum()} of {len(dd)} folds")
    print("coefficients [1, score, flank GC, step, score x flank GC, score x step]:", coef.round(3))
    return 0


# ======================================================================================================  gc-corr, gc-ablation
N_STRATA = 5
FLANK_GC = ["eng.flank.comp.up500.gc", "eng.flank.comp.dn500.gc"]


def _frame() -> pd.DataFrame:
    """Target GC, flank GC and the label, one row per located guide."""
    from ..io_utils import load_block, load_guide_index

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
    from ..harness.curated import load_base, load_block

    X, y, names, protos = load_base()
    proto = pd.Series([str(q).upper() for q in protos])
    tgc = proto.map(lambda q: (q.count("G") + q.count("C")) / max(len(q), 1))

    v, fn = load_block("a_flank", protos)
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


def gc_corr(cohort: str = "curated", verbose: bool = True) -> pd.DataFrame:
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


def gc_ablation(seeds=(41, 42, 43), n_splits: int = 5, n_features: int = 300,
        verbose: bool = True) -> pd.DataFrame:
    from ..build.build_matrix import load_dataset
    from ..harness.run_ablation import cross_validate, load_groups

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


# ======================================================================================================  asym-audit, asym-run
def gene_strand() -> pd.Series:
    """Gene name -> '+' or '-', from the reference gene table."""
    g = pd.read_csv(config.GENE_TABLE, sep="\t")
    return g.set_index("name")["ori"].astype(str)


def asym_audit(verbose: bool = True) -> pd.DataFrame:
    """Does the library contain guides facing both ways inside their genes?

    It does not, and that kills the orientation test before it starts. Checked
    two independent ways: joining the guide index's mapped strand against the
    reference gene table, and reading the `g_transcription` family's own
    `template_strand` column, which was derived separately.
    """
    from ..io_utils import load_block, load_guide_index

    idx = load_guide_index()
    idx = idx[idx["located"].astype(bool)].copy()
    idx["gene_strand"] = idx["gene_name"].map(gene_strand())
    idx["relative_orientation"] = np.where(
        idx["strand"].astype(str) == idx["gene_strand"],
        "sense strand", "template strand")

    blk = load_block("g_transcription")[[config.ID_COL,
                                         "eng.txn.gene.template_strand"]]
    merged = idx.merge(blk, on=config.ID_COL, how="left")

    rows = []
    for label, counts in (
        ("guide index joined to gene table",
         merged["relative_orientation"].value_counts(dropna=False)),
        ("g_transcription template_strand column",
         merged["eng.txn.gene.template_strand"]
         .map({1.0: "template strand", 0.0: "sense strand"})
         .value_counts(dropna=False)),
    ):
        for k, v in counts.items():
            rows.append(dict(source=label, orientation=str(k), guides=int(v)))
    df = pd.DataFrame(rows)
    df.to_csv(ASYM_AUDIT_OUT, index=False)

    if verbose:
        print(df.to_string(index=False))
        n_sense = int((merged["eng.txn.gene.template_strand"] == 0.0).sum())
        print(f"\n  guides on the gene's sense strand: {n_sense} of "
              f"{len(merged):,} ({n_sense / len(merged):.4%})")
        print("  => the orientation test is impossible on this screen: the "
              "library targets the template strand almost exclusively, so "
              "there is no variation to condition on.")
        print("  => and `eng.txn.gene.template_strand` is a near-constant "
              "column, which is part of why g_transcription could only give "
              "+0.015.")
    return df


def _side_of(col: str) -> str:
    """'up', 'dn' or 'other' for one family-A column.

    Family A names the side in three different places depending on the
    sub-block, so all three are handled explicitly:
      eng.flank.seq.u10.A                  -> 4th component
      eng.flank.qct.dimer.HLgapeVE.u10     -> last component
      eng.flank.comp.up50.gc               -> infix
    The PAM-spanning dinucleotide (`.pamN`) belongs to neither side.
    """
    parts = col.split(".")
    if col.startswith("eng.flank.seq.") and len(parts) > 3:
        tag = parts[3]
    elif col.startswith("eng.flank.qct."):
        tag = parts[-1]
    else:
        tag = ""
    if tag.startswith("u") and tag[1:].isdigit():
        return "up"
    if tag.startswith("d") and tag[1:].isdigit():
        return "dn"
    if ".up" in col:
        return "up"
    if ".dn" in col:
        return "dn"
    return "other"


def asym_run(seed: int = 41, n_splits: int = 5, n_bins: int = 3,
        verbose: bool = True) -> pd.DataFrame:
    """The test this screen can support: does the asymmetry scale with transcription?

    Split guides by their gene's expression and measure the upstream and
    downstream gains inside each band. If polymerase traffic is what makes the
    PAM-side flank matter more, the gap should widen with expression. If it is
    R-loop geometry, the gap should be flat across bands.

    Expression is `eng.txn.expr.median_log_tpm`, which came in with
    `g_transcription`; it is a property of the *gene*, so it is constant across
    the ~20 guides in each gene and cannot be a per-guide artefact of the flank
    features.
    """
    from ..build.build_matrix import load_dataset
    from ..io_utils import load_block
    from ..harness.run_ablation import cross_validate

    X, y, names, ids, family_of = load_dataset(["a_flank"], verbose=verbose)
    family_of = np.asarray(family_of)
    names = np.asarray(names)
    base = np.where(family_of == "base")[0]
    flank = np.where(family_of != "base")[0]
    side = np.array([_side_of(n) for n in names[flank]])
    if verbose:
        print(f"  family A: {(side == 'up').sum()} upstream, "
              f"{(side == 'dn').sum()} downstream columns")

    expr = (load_block("g_transcription")
            .set_index(config.ID_COL)["eng.txn.expr.median_log_tpm"])
    e = pd.Series(ids).map(expr).to_numpy(dtype=float)
    ok = np.isfinite(e)
    qs = np.quantile(e[ok], np.linspace(0, 1, n_bins + 1))
    band = np.full(len(e), -1)
    for b in range(n_bins):
        lo, hi = qs[b], qs[b + 1]
        sel = ok & (e >= lo) & ((e <= hi) if b == n_bins - 1 else (e < hi))
        band[sel] = b
    labels = {0: "low expression", 1: "mid expression", 2: "high expression"}

    # The sides are not matched for size -- family A has 194 downstream columns
    # against 144 upstream, because the windows and the PAM-side k-mers are not
    # mirror images. A third of the downstream advantage could therefore be
    # column count rather than side, so a count-matched downstream arm is run
    # alongside: 144 downstream columns drawn at random, same seed.
    rng = np.random.default_rng(seed)
    dn_idx = flank[side == "dn"]
    up_idx = flank[side == "up"]
    dn_matched = rng.choice(dn_idx, size=len(up_idx), replace=False)

    arms = {
        "baseline": base,
        "upstream_only": np.sort(np.concatenate([base, up_idx])),
        "downstream_only": np.sort(np.concatenate([base, dn_idx])),
        "downstream_matched": np.sort(np.concatenate([base, dn_matched])),
        "both": np.arange(X.shape[1]),
    }

    rows = []
    for b in range(n_bins):
        keep = band == b
        grp = labels.get(b, f"band {b}")
        for arm, cols in arms.items():
            res = cross_validate(
                X[keep][:, cols], y[keep], list(names[cols]), family_of[cols],
                seeds=(seed,), n_splits=n_splits, verbose=False,
                tag=f"asymexpr_b{b}_{arm}", experiment=f"{grp} | {arm}")
            s = res["summary"]
            rows.append(dict(band=b, group=grp, n_guides=int(keep.sum()),
                             median_log_tpm=float(np.nanmedian(e[keep])),
                             arm=arm, spearman=s["spearman"]["mean"],
                             r2=s["r2"]["mean"]))
            if verbose:
                print(f"    {grp:18s} {arm:16s} rho "
                      f"{s['spearman']['mean']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(ASYM_OUT, index=False)
    return pd.DataFrame(rows)


def asym_summary():
    d = pd.read_csv(ASYM_OUT)
    piv = d.pivot_table(index=["band", "group"], columns="arm",
                        values="spearman")
    for arm in ("upstream_only", "downstream_only",
                "downstream_matched", "both"):
        if arm in piv:
            piv[f"gain_{arm}"] = (piv[arm] - piv["baseline"]).round(4)
    if {"gain_downstream_only", "gain_upstream_only"} <= set(piv.columns):
        piv["asymmetry"] = (piv["gain_downstream_only"]
                            - piv["gain_upstream_only"]).round(4)
    if {"gain_downstream_matched", "gain_upstream_only"} <= set(piv.columns):
        piv["asymmetry_matched"] = (piv["gain_downstream_matched"]
                                    - piv["gain_upstream_only"]).round(4)
    show = ["baseline"] + [c for c in piv.columns if c.startswith("gain_")]
    show += [c for c in ("asymmetry", "asymmetry_matched")
             if c in piv.columns]
    print(piv[show].round(4).to_string())
    return piv


PARTS = ("profile", "gate", "gc-corr", "gc-ablation", "asym-audit", "asym-run")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", nargs="+", choices=PARTS, required=True)
    ap.add_argument("--cohort", choices=["curated", "published"], default="curated", help="gc-corr: which cohort")
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43], help="gc-ablation: seeds")
    ap.add_argument("--folds", type=int, default=5, help="asym-run: folds")
    a = ap.parse_args(argv)
    for part in a.parts:
        if part == "profile":
            profile()
        elif part == "gate":
            gate()
        elif part == "gc-corr":
            gc_corr(cohort=a.cohort)
        elif part == "gc-ablation":
            gc_ablation(seeds=tuple(a.seeds))
        elif part == "asym-audit":
            asym_audit()
        else:
            asym_run(n_splits=a.folds)
            asym_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
