"""Family A's gain is not the row count (+0.1672 on subsamples of the curated cohort the size of the published matrix, +0.1638 on the whole cohort), not the choice of windows (+0.1634 on a window set sharing no length with the original, against +0.1638), and not Chi sites (coefficient -0.053 with flank GC held fixed, within the spread of GC-matched octamers, mean -0.022); and it falls away as the flanks are read from further along the chromosome.

Four controls on the one effect the report rests on, each answering a different objection to family A's +0.164, each its own part and its own file:

  rows      "the gain rose because there were more rows": re-run on random 13,880-row subsamples of the curated cohort          -> why_inflation.csv      (Appendix B2a)
  windows   "the windows were chosen after a sweep": rebuild family A on 75 / 300 / 600 / 1,200 nt instead of 50 / 250 / 500 / 1,000     -> window_robustness.csv  (Appendix B2)
  shift     "shuffled flanks are no longer smooth, so the shuffle null is weak": a spatially preserving null, the flank track read from a circularly shifted position, swept in distance -> shift_null.csv  (Appendix B2)
  chi       "the flank effect is RecBCD resection and Chi sites": a test built on what separates Chi from composition (orientation, and GC-matched control octamers)  -> chi_sites.csv        (Appendix B2)

    python -m sgrna.controls.family_a_controls --parts rows windows shift chi

------------------------------------------------------------------------------------------------------------------------------
rows
------------------------------------------------------------------------------------------------------------------------------
Four things changed between the published 13,880-row matrix and crisprHAL's
33,567 curated guides, and "noise was removed from the label" is only one of
them:

  1. **the label** -- crisprHAL's read-count-filtered score replaces `cut_score`;
  2. **which rows** -- the cohorts are not nested (8,617 shared, 24,950 only
     crisprHAL, 5,263 only Noshay), and the curated release is genome-wide,
     4,013 genes at 7.5 guides each against a 1,222-gene panel at 11.3;
  3. **how many rows** -- 2.4× more;
  4. **the base representation** -- the curated base is the 6,169 regenerated
     columns and carries none of the published base's 63 hand-named summaries,
     so every family is now measured against a weaker comparison.

Only (1) is noise removal, and (1) is already ruled out: at **fixed rows** --
the 8,617 shared guides, published base -- family A is worth +0.0892 against
`cut_score` and +0.0936 against the curated score (`ablation_label_*.csv`). A
label swap moves it by 0.004 while the cohort move takes it from +0.080 to
+0.164.

This module separates (3) from (2) and (4) the only way available: re-run the
curated ablation on a **random 13,880-row subsample of the curated cohort**. The
subsample holds the label, the row population and the base fixed and changes only
the count, so

    Δρ(13,880 curated rows) ≈ Δρ(33,567)   =>  row count is not the cause
    Δρ(13,880 curated rows) ≈ Δρ(published) =>  row count is the cause

Three seeds of subsample, because one draw of 13,880 from 33,567 is itself a
sample.

------------------------------------------------------------------------------------------------------------------------------
windows
------------------------------------------------------------------------------------------------------------------------------
Family A summarises the flanks in windows of **50, 250, 500 and 1,000 nt**
(`config.FLANK_WINDOWS`). Those four numbers were chosen on the published cohort
after a sweep, so with respect to the curated five folds they are an inherited
constant rather than a choice fitted here. That is the weaker form of the
objection. The stronger form survives it: *a sweep was run, and the reported
figure uses the windows that won it*, so some of family A's gain could be the
sweep's own winner's curse.

**The test that settles it is cheap, and it is a robustness check rather than a
re-selection.** Rebuild family A on a window set chosen to share no value with
the original — **75, 300, 600, 1,200 nt** — and re-measure the gain. The
prediction follows from Finding 4: if the effect is a smooth compositional
gradient rather than a feature of particular window edges, any reasonable set of
four nested windows should recover it, and the two gains should agree to within
fold noise. If instead the original set is worth materially more, the reported
+0.164 is partly a selection artefact and should be discounted.

Declared before running, as the interpretation: agreement within **±0.005 ρ**
(the project's non-inferiority margin) counts as the choice not mattering;
a shortfall beyond that is a winner's curse worth quantifying.

------------------------------------------------------------------------------------------------------------------------------
shift
------------------------------------------------------------------------------------------------------------------------------
The permutation control used everywhere else shuffles the flank block's rows.
That destroys **genomic autocorrelation** along with the pairing, so it tests a
weaker null than the report claimed: shuffled flanks are not merely misassigned,
they are no longer smooth, and a model could in principle be rejecting them for
the wrong reason.

The sharper null keeps every statistical property of the flank track and breaks
only the correspondence to the guide. Each guide is re-assigned the flanks of a
genomic position **offset by a fixed circular shift** around the chromosome:
composition, smoothness, long-range autocorrelation and the distribution of every
column are preserved exactly, because the same track is being read -- just from
the wrong place.

**One shift is not enough, and the reason is the thing being tested.** GC content
on this chromosome is autocorrelated over tens of kilobases, so a 1 kb shift
hands each guide almost the correct flanks and would retain most of the gain --
not a null at all. So the shift is swept, which turns a control into a
measurement: **Delta rho against shift distance is the autocorrelation length of
the effect**, independently of the window sizes family A happens to use. A real
local effect decays to zero as the shift exceeds its correlation length; an
artefact of large-scale chromosome structure does not.

Grouped CV throughout (100 kb blocks), because a positional feature under random
KFold can score by locus recognition -- the shifted arms would inherit that too.

------------------------------------------------------------------------------------------------------------------------------
chi
------------------------------------------------------------------------------------------------------------------------------
Three results in this project point the same way and have never been put
together:

  * flanking composition out to ~500 nt is worth +0.164, the largest effect for
    this phenotype, and it is a smooth gradient with no motif (Findings 3, 4);
  * it is **absent in a dCas9 screen** on the same genome, so it acts at or
    after the cut and not at binding or accessibility (Finding 5);
  * it has a **direction** -- orientation relative to the TSS is worth 0.051
    (Finding 6).

"Downstream of binding" is a strong constraint. Nothing about the HNH
conformational change has any reason to read five hundred nucleotides. One thing
in *E. coli* does: **RecBCD**, which enters a double-strand break and degrades
processively over kilobases until it meets a Chi site (`GCTGGTGG`), whereupon it
is attenuated and hands off to RecA for homologous repair. That predicts exactly
what is observed -- a long, smooth, compositional, orientation-dependent
dependence that exists only when a break is actually made -- and it reframes the
+0.164 as a property of *what happens to the break*, not of cutting.

In a pooled depletion screen a guide scores as efficient when its cells die, and
a cell dies from an unrepaired break. So anything that improves repair lowers
measured efficiency. Chi sites are GC-rich, GC-rich flanks predict *lower*
efficiency here (the association is −0.20), and better repair means less death:
the signs agree.

This module tests it on data already on disk, and the test is built around the
one thing that separates Chi from composition. **GC content is a scalar; Chi
function is orientation-specific** -- RecBCD recognises `GCTGGTGG` only when it
is approached from one side. So:

  1. does Chi count predict the label at all;
  2. does it still predict after flank GC is controlled;
  3. do the two orientations behave *differently*, which composition cannot
     explain;
  4. and -- the control that decides it -- does Chi behave differently from
     **other octamers of the same GC content**? If a random 6-of-8 GC octamer
     buys the same coefficient, "Chi" is a GC proxy and the mechanism is not
     supported.

Report section: Appendix B2a.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from .. import config
from ..common import OUT
from ..harness.curated import load_base, load_block

ROWS_OUT = OUT / "why_inflation.csv"
WINDOWS_OUT = OUT / "window_robustness.csv"
SHIFT_OUT = OUT / "shift_null.csv"
CHI_OUT = OUT / "chi_sites.csv"


# ======================================================================================================  rows
N_SUB = 13_880


def row_count(families=("a_flank", "d_mechanics"), n_sub: int = N_SUB,
        sub_seeds=(0, 1, 2), n_features=300, verbose: bool = True):
    from ..harness.run_ablation import cross_validate

    X, y, names, protos = load_base()
    rows = []
    for sseed in sub_seeds:
        idx = np.random.default_rng(sseed).choice(len(y), n_sub, replace=False)
        Xs, ys, ps = X[idx], y[idx], protos[idx]
        res = cross_validate(Xs, ys, names, ["base"] * len(names),
                             n_features=n_features, verbose=False,
                             tag=f"whyinf{sseed}_baseline", experiment="baseline")
        base = res["summary"]["spearman"]["mean"]
        rows.append(dict(sub_seed=sseed, n_rows=n_sub, family="baseline",
                         spearman=base, d_spearman=0.0))
        if verbose:
            print(f"  subsample {sseed}: baseline rho {base:.4f}", flush=True)
        for fam in families:
            v, fnames = load_block(fam, ps)
            Xa = np.hstack([Xs, v])
            res = cross_validate(Xa, ys, names + fnames,
                                 ["base"] * len(names) + [fam] * len(fnames),
                                 n_features=n_features, verbose=False,
                                 tag=f"whyinf{sseed}_{fam}", experiment=fam)
            rho = res["summary"]["spearman"]["mean"]
            rows.append(dict(sub_seed=sseed, n_rows=n_sub, family=fam,
                             spearman=rho, d_spearman=rho - base))
            if verbose:
                print(f"  subsample {sseed}: {fam:>14} rho {rho:.4f}   "
                      f"delta {rho - base:+.4f}", flush=True)
            pd.DataFrame(rows).to_csv(ROWS_OUT, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(ROWS_OUT, index=False)
    if verbose:
        print()
        print(out.groupby("family")["d_spearman"].agg(["mean", "std"]).round(4)
              .to_string())
        print(f"written -> {ROWS_OUT}")
    return out


# ======================================================================================================  windows
ALT_WINDOWS = (75, 300, 600, 1200)
BLOCK = config.INTERIM / "features" / "a_flank_altwin_curated.csv"


def build_windows(verbose: bool = True):
    """Family A again, on windows sharing no value with the published set."""
    if BLOCK.exists():
        return pd.read_csv(BLOCK)
    from ..build.build_features import load_curated_index
    from ..features import a_flank

    # The module reads its windows from `config` at import time, so the constant
    # is swapped before the build rather than passed in -- the alternative is an
    # API change to a frozen module.
    original = a_flank.SUMMARY_WINDOWS
    a_flank.SUMMARY_WINDOWS = ALT_WINDOWS
    try:
        idx = load_curated_index(verbose=verbose)
        block = a_flank.build(idx)
    finally:
        a_flank.SUMMARY_WINDOWS = original
    block.to_csv(BLOCK, index=False)
    if verbose:
        print(f"  built {block.shape[0]:,} x {block.shape[1] - 1} columns "
              f"at windows {ALT_WINDOWS} -> {BLOCK.name}")
    return block


def windows(seeds=None, n_splits=None, n_features=300, verbose: bool = True):
    from ..harness.run_ablation import cross_validate

    seeds = list(seeds or config.SEEDS)
    n_splits = n_splits or config.N_SPLITS
    X, y, names, protos = load_base()
    singles = pd.read_csv(config.RESULTS / "ablation_curated.csv").set_index("family")
    base_rho = float(singles.loc["baseline", "spearman"])
    published = float(singles.loc["a_flank", "d_spearman"])

    block = build_windows(verbose=verbose)
    b = block.set_index(config.ID_COL).reindex(protos)
    v = b.to_numpy(dtype=np.float32)
    keep = ~np.all(np.isnan(v), axis=0)
    v, fnames = v[:, keep], [c for c, k in zip(b.columns, keep) if k]
    if verbose:
        print(f"  alternative family A: {v.shape[1]} columns "
              f"(published set has {int(singles.loc['a_flank', 'n_cols']) - X.shape[1]})")

    res = cross_validate(np.hstack([X, v]), y, names + fnames,
                         ["base"] * len(names) + ["a_flank_alt"] * len(fnames),
                         n_features=n_features, seeds=seeds, n_splits=n_splits,
                         verbose=False, tag="altwin_a_flank",
                         experiment="a_flank_alt")
    rho = res["summary"]["spearman"]["mean"]
    row = dict(arm="a_flank at 75/300/600/1200", n_cols=v.shape[1],
               spearman=rho, spearman_sd=res["summary"]["spearman"]["sd"],
               d_spearman=rho - base_rho, d_published_windows=published,
               difference=(rho - base_rho) - published)
    pd.DataFrame([row]).to_csv(WINDOWS_OUT, index=False)
    if verbose:
        print(f"\n  published windows (50/250/500/1000): {published:+.4f}")
        print(f"  alternative windows (75/300/600/1200): {row['d_spearman']:+.4f}")
        print(f"  difference: {row['difference']:+.4f}  "
              f"({'within' if abs(row['difference']) < 0.005 else 'OUTSIDE'} "
              f"the declared +/-0.005)")
        print(f"written -> {WINDOWS_OUT}")
    return row


# ======================================================================================================  shift
SHIFTS = (0, 1_000, 10_000, 100_000, 500_000, 1_000_000, 2_000_000)


def _shifted_block(block: pd.DataFrame, index: pd.DataFrame, shift: int,
                   genome_len: int) -> pd.DataFrame:
    """Re-key the flank block so each guide reads the track `shift` bp away.

    Implemented as a nearest-neighbour lookup in genomic coordinate: for guide at
    position p, take the flank row of whichever guide sits closest to
    (p + shift) mod L. That reuses real rows, so every column keeps its exact
    marginal distribution -- unlike a shuffle, which keeps the marginals but
    destroys the spatial structure, and unlike interpolation, which would invent
    values.
    """
    pos = index["left"].to_numpy(dtype=float)
    order = np.argsort(pos)
    sorted_pos = pos[order]
    target = (pos + shift) % genome_len
    j = np.searchsorted(sorted_pos, target)
    j = np.clip(j, 0, len(sorted_pos) - 1)
    # pick the nearer of the two bracketing guides, wrapping at the ends
    left = np.clip(j - 1, 0, len(sorted_pos) - 1)
    pick = np.where(np.abs(sorted_pos[j] - target)
                    <= np.abs(sorted_pos[left] - target), j, left)
    src = order[pick]
    out = block.iloc[src].reset_index(drop=True)
    out[config.ID_COL] = index[config.ID_COL].to_numpy()
    return out


def shift(shifts=SHIFTS, seeds=(41, 42, 43), n_splits: int = 5,
        verbose: bool = True) -> pd.DataFrame:
    from ..build.build_matrix import build_cache, load_dataset
    from ..io_utils import load_block, load_guide_index
    from ..harness.run_ablation import cross_validate, load_groups

    X_base, y, names, ids, fam = load_dataset([], verbose=verbose)
    idx = load_guide_index()
    idx = idx[idx["located"].astype(bool)].drop_duplicates(config.ID_COL)
    idx = idx.set_index(config.ID_COL).reindex(ids).reset_index()
    genome_len = int(idx["genome_length"].iloc[0]) if "genome_length" in idx \
        else 4_639_675
    block = (load_block("a_flank").set_index(config.ID_COL)
             .reindex(ids).reset_index())
    flank_cols = [c for c in block.columns if c != config.ID_COL]
    groups = load_groups(ids, by="bin100k")

    rows = []
    for shift in shifts:
        b = block if shift == 0 else _shifted_block(block, idx, shift, genome_len)
        vals = b[flank_cols].to_numpy(dtype=np.float32)
        keep = ~np.all(np.isnan(vals), axis=0)
        X = np.hstack([X_base, vals[:, keep]])
        fam_of = np.asarray(list(fam) + ["a_flank"] * int(keep.sum()))
        res = cross_validate(X, y, names + [c for c, k in zip(flank_cols, keep) if k],
                             fam_of, seeds=seeds, n_splits=n_splits, groups=groups,
                             verbose=False, tag=f"shift{shift}",
                             experiment=f"shift_{shift}")
        s = res["summary"]
        rows.append(dict(shift_bp=shift, n_flank_cols=int(keep.sum()),
                         spearman=s["spearman"]["mean"],
                         spearman_sd=s["spearman"]["sd"],
                         r2=s["r2"]["mean"]))
        pd.DataFrame(rows).to_csv(SHIFT_OUT, index=False)
        if verbose:
            base = rows[0]["spearman"]
            print(f"  shift {shift:>9,} bp: rho {s['spearman']['mean']:.4f}"
                  + (f"   retained {100*(s['spearman']['mean']-rows[-1]['spearman']*0):.0f}%"
                     if shift == 0 else
                     f"   vs aligned {s['spearman']['mean']-base:+.4f}"),
                  flush=True)
    return pd.DataFrame(rows)


def shift_summary():
    d = pd.read_csv(SHIFT_OUT).sort_values("shift_bp")
    base_row = d[d.shift_bp == 0]
    if len(base_row):
        aligned = float(base_row["spearman"].iloc[0])
        d["delta_vs_aligned"] = (d["spearman"] - aligned).round(4)
    print(d.round(4).to_string(index=False))
    print("\nshift 0 is the real, aligned flanks; every other row keeps the same")
    print("track and reads it from the wrong place.")
    return d


# ======================================================================================================  chi
CHI = "GCTGGTGG"
CHI_RC = "CCACCAGC"
WINDOWS = (250, 500, 1000)


def _revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _load():
    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv")
    idx["upstream"] = idx["upstream"].fillna("").astype(str).str.upper()
    idx["downstream"] = idx["downstream"].fillna("").astype(str).str.upper()
    return idx


def _counts(idx: pd.DataFrame, motif: str, w: int) -> dict[str, np.ndarray]:
    """Occurrences of a motif and its reverse complement, each side separately."""
    rc = _revcomp(motif)
    up, dn = idx["upstream"].str[-w:], idx["downstream"].str[:w]
    return {
        "fwd_up": up.str.count(motif).to_numpy(float),
        "fwd_dn": dn.str.count(motif).to_numpy(float),
        "rev_up": up.str.count(rc).to_numpy(float),
        "rev_dn": dn.str.count(rc).to_numpy(float),
    }


def _gc(idx: pd.DataFrame, w: int) -> np.ndarray:
    out = []
    for side, sl in (("upstream", slice(-w, None)), ("downstream", slice(0, w))):
        s = idx[side].str[sl] if side == "upstream" else idx[side].str[:w]
        g = s.str.count("G") + s.str.count("C")
        out.append((g / s.str.len().clip(lower=1)).to_numpy(float))
    return np.column_stack(out)


def _ols(y: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients and their standard errors, with an intercept added."""
    A = np.column_stack([np.ones(len(Z)), Z])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - A.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    return beta[1:], np.sqrt(np.diag(cov))[1:]


def _control_octamers(gc_count: int = 6, n: int = 200, seed: int = 0) -> list[str]:
    """Octamers with the same GC count as Chi, excluding Chi and its own rc."""
    rng = np.random.default_rng(seed)
    pool = []
    for combo in itertools.product("ACGT", repeat=8):
        s = "".join(combo)
        if sum(c in "GC" for c in s) == gc_count and s not in (CHI, CHI_RC):
            pool.append(s)
    pick = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in pick]


def chi(verbose: bool = True) -> pd.DataFrame:
    from scipy.stats import spearmanr

    idx = _load()
    y = idx["score"].to_numpy(float)
    ok = np.isfinite(y)
    idx, y = idx.loc[ok].reset_index(drop=True), y[ok]
    if verbose:
        print(f"  {len(idx):,} curated guides with a label", flush=True)

    rows = []
    for w in WINDOWS:
        gc = _gc(idx, w)
        c = _counts(idx, CHI, w)
        total = c["fwd_up"] + c["fwd_dn"] + c["rev_up"] + c["rev_dn"]

        rho = float(spearmanr(total, y).statistic)
        b_alone, se_alone = _ols(y, total[:, None])
        b_adj, se_adj = _ols(y, np.column_stack([gc, total]))
        rows.append(dict(window=w, test="chi total", rho=rho,
                         beta_alone=b_alone[0], se_alone=se_alone[0],
                         beta_gc_adjusted=b_adj[-1], se_gc_adjusted=se_adj[-1],
                         n_with_site=int((total > 0).sum())))
        if verbose:
            print(f"  +/-{w:4d} nt  Chi total: rho {rho:+.4f}  "
                  f"beta {b_alone[0]:+.4f}+/-{se_alone[0]:.4f}  "
                  f"GC-adjusted {b_adj[-1]:+.4f}+/-{se_adj[-1]:.4f}  "
                  f"({int((total > 0).sum()):,} guides carry one)", flush=True)

        # Orientation: four counts entered separately, GC held fixed.
        quad = np.column_stack([c["fwd_up"], c["fwd_dn"], c["rev_up"], c["rev_dn"]])
        b4, se4 = _ols(y, np.column_stack([gc, quad]))
        for k, key in enumerate(("fwd_up", "fwd_dn", "rev_up", "rev_dn")):
            rows.append(dict(window=w, test=f"chi {key}", rho=np.nan,
                             beta_alone=np.nan, se_alone=np.nan,
                             beta_gc_adjusted=b4[2 + k], se_gc_adjusted=se4[2 + k],
                             n_with_site=int((quad[:, k] > 0).sum())))
        if verbose:
            print("    orientation, GC held fixed: " + "  ".join(
                f"{key} {b4[2 + k]:+.4f}+/-{se4[2 + k]:.4f}"
                for k, key in enumerate(("fwd_up", "fwd_dn", "rev_up", "rev_dn"))),
                flush=True)
        pd.DataFrame(rows).to_csv(CHI_OUT, index=False)

    # The control that decides it: other octamers of the same GC content.
    w = WINDOWS[-1]
    gc = _gc(idx, w)
    chi_tot = sum(_counts(idx, CHI, w).values())
    chi_beta = _ols(y, np.column_stack([gc, chi_tot]))[0][-1]
    betas = []
    for m in _control_octamers():
        t = sum(_counts(idx, m, w).values())
        if t.sum() < 50:
            continue
        betas.append(_ols(y, np.column_stack([gc, t]))[0][-1])
    betas = np.array(betas)
    pct = float((betas <= chi_beta).mean() * 100)
    rows.append(dict(window=w, test="control octamers (same GC)", rho=np.nan,
                     beta_alone=np.nan, se_alone=np.nan,
                     beta_gc_adjusted=float(betas.mean()),
                     se_gc_adjusted=float(betas.std(ddof=1)),
                     n_with_site=len(betas)))
    pd.DataFrame(rows).to_csv(CHI_OUT, index=False)
    if verbose:
        print(f"\n  Chi beta (GC-adjusted, +/-{w} nt): {chi_beta:+.4f}")
        print(f"  {len(betas)} control octamers of the same GC content: "
              f"mean {betas.mean():+.4f}, sd {betas.std(ddof=1):.4f}, "
              f"range {betas.min():+.4f} to {betas.max():+.4f}")
        print(f"  Chi sits at the {pct:.1f}th percentile of that null")
        print(f"written -> {CHI_OUT}")
    return pd.DataFrame(rows)


PARTS = ("rows", "windows", "shift", "chi")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", nargs="+", choices=PARTS, required=True)
    ap.add_argument("--families", nargs="*", default=["a_flank", "d_mechanics"], help="rows: which families")
    ap.add_argument("--n-sub", type=int, default=N_SUB, help="rows: subsample size")
    ap.add_argument("--sub-seeds", type=int, nargs="*", default=[0, 1, 2], help="rows: subsample draws")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="shift: CV seeds")
    ap.add_argument("--summary-only", action="store_true", help="shift: print the stored table, fit nothing")
    a = ap.parse_args(argv)
    for part in a.parts:
        if part == "rows":
            row_count(families=tuple(a.families), n_sub=a.n_sub, sub_seeds=tuple(a.sub_seeds))
        elif part == "windows":
            windows()
        elif part == "shift":
            if not a.summary_only:
                shift(seeds=tuple(a.seeds) if a.seeds else (41, 42, 43))
            shift_summary()
        else:
            chi()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
