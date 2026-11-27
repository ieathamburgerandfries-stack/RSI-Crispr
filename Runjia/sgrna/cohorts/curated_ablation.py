"""Feature-family ablations on crisprHAL's curated cohort (designs: one family at a time on random rows or whole genes, and `--paper-folds`).

Every Δρ in Appendix C was measured on the published 13,880-row matrix against
the published label. The model layer moved to crisprHAL's 33,567 curated guides
and their curated score, and the family measurements did not follow -- so their
*relative* ranking there is unverified. Family A is the one family that has been
re-measured, and it doubled (+0.080 → +0.164), which is reason enough not to
assume the others carry over.

This runs the same ablation design on the curated cohort: base representation
(the 6,169 columns regenerable from the 20-mer) against base + one family, five
seeds by five folds, feature selection fitted inside each training split.

    python -m sgrna.cohorts.curated_ablation --families b_energy c_folding
    python -m sgrna.cohorts.curated_ablation --all
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .. import config

OUT = config.RESULTS / "ablation_curated.csv"
GROUPED_OUT = {"gene": config.RESULTS / "ablation_curated_gene.csv",
               "block100k": config.RESULTS / "ablation_curated_block.csv"}
FAMILIES = ("a_flank", "b_energy", "c_folding", "d_mechanics", "d_supercoiling",
            "e_nucleoid", "f_methylation", "g_transcription", "h_offtarget",
            "i_shape", "j_periodicity")


def _base():
    """The curated cohort: 6,169 regenerated columns, their label, their order."""
    meta = json.loads((config.INTERIM / "expanded_meta.json").read_text())
    X = np.asarray(np.load(config.INTERIM / "expanded_X.npy", mmap_mode="r"),
                   dtype=np.float32)
    y = np.asarray(meta["y"], dtype=float)
    return X, y, list(meta["feature_names"]), np.asarray(meta["protospacers"])


def _block(fam: str, protos: np.ndarray):
    """One family, aligned to the curated guides. Returns (values, names)."""
    path = config.INTERIM / "features" / f"{fam}_curated.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not built")
    b = pd.read_csv(path).set_index(config.ID_COL).reindex(protos)
    v = b.to_numpy(dtype=np.float32)
    keep = ~np.all(np.isnan(v), axis=0)          # drop columns empty on this cohort
    if keep.sum() == 0:
        # b_energy built "successfully" with zero feature columns on this cohort,
        # which save_block accepts. An empty block is a failed build, not a null
        # result -- refuse it rather than reporting a delta of exactly zero.
        raise FileNotFoundError(f"{path.name} has no populated feature columns")
    return v[:, keep], [c for c, k in zip(b.columns, keep) if k]


ORDER = ("baseline",) + FAMILIES


def _groups(kind: str, protos: np.ndarray):
    """Group label per curated guide, and the rows that have one.

    Grouping by gene is the arm that matters on this cohort: gene identity
    explains 29% of the curated label against 16% of the published one
    (`cohort_groups.py`), so plain KFold has more to leak here, not less.
    Guides that fall outside every annotated gene -- 3,364 of 33,567 -- have no
    gene group, and they are dropped rather than given singleton groups, which
    would leave each of them in its own fold and defeat the point. **Every arm
    then runs on the same subset, baseline included**, so the deltas stay
    internally comparable even though their levels are not comparable to the
    random-row table.
    """
    from ..build.build_features import load_curated_index

    idx = load_curated_index(verbose=False).set_index(config.ID_COL)
    if kind == "gene":
        g = idx["gene_name"].reindex(protos)
    elif kind == "block100k":
        g = (pd.to_numeric(idx["left"], errors="coerce") // 100_000).reindex(protos)
    else:
        raise ValueError(f"unknown grouping {kind!r}")
    ok = g.notna().to_numpy()
    return g.to_numpy(dtype=object), ok


def _write(rows, out=None):
    """Write the table in family order, whatever order the runs happened in."""
    df = pd.DataFrame(rows)
    df["_k"] = df["family"].map({f: i for i, f in enumerate(ORDER)})
    df.sort_values("_k").drop(columns="_k").to_csv(out or OUT, index=False)


def run(families=FAMILIES, seeds=None, n_splits=None, n_features=300,
        force_baseline: bool = False, grouping: str | None = None,
        verbose: bool = True):
    from ..harness.run_ablation import cross_validate

    seeds = list(seeds or config.SEEDS)
    n_splits = n_splits or config.N_SPLITS
    X, y, names, protos = _base()

    OUT_ = OUT if grouping is None else GROUPED_OUT[grouping]
    tag_prefix = "curabl" if grouping is None else f"curabl{grouping[:4]}"
    groups = None
    if grouping is not None:
        groups, ok = _groups(grouping, protos)
        X, y, protos, groups = X[ok], y[ok], protos[ok], groups[ok]
        if verbose:
            print(f"  grouping by {grouping}: {ok.sum():,} of {len(ok):,} guides "
                  f"in {len(np.unique(groups)):,} groups")
    if verbose:
        print(f"  curated cohort: {X.shape[0]:,} guides, {X.shape[1]:,} base columns")

    # Families arrive over days as their blocks get rebuilt on this cohort, so
    # a run of one family must not discard the rest of the table -- the first
    # version of this rewrote OUT from scratch every time, which would have
    # silently deleted eight measured families to add one.
    prev = (pd.read_csv(OUT_) if OUT_.exists() and OUT_.stat().st_size
            else pd.DataFrame(columns=["family"]))
    rows = [r for r in prev.to_dict("records")
            if r["family"] not in ("baseline",) + tuple(families)]

    done_base = prev[prev["family"] == "baseline"]
    if len(done_base) and not force_baseline:
        base = done_base.iloc[0].to_dict()
        base_rho = float(base["spearman"])
        if verbose:
            print(f"  {'baseline':>16}: rho {base_rho:.4f}  (cached)", flush=True)
    else:
        res = cross_validate(X, y, names, ["base"] * len(names),
                             n_features=n_features, seeds=seeds,
                             n_splits=n_splits, verbose=False, groups=groups,
                             tag=f"{tag_prefix}_baseline", experiment="baseline")
        base_rho = res["summary"]["spearman"]["mean"]
        base = dict(family="baseline", n_cols=X.shape[1], spearman=base_rho,
                    spearman_sd=res["summary"]["spearman"]["sd"], d_spearman=0.0)
        if verbose:
            print(f"  {'baseline':>16}: rho {base_rho:.4f}", flush=True)
    rows.insert(0, base)
    _write(rows, OUT_)

    for fam in families:
        try:
            v, fnames = _block(fam, protos)
        except FileNotFoundError as exc:
            if verbose:
                print(f"  {fam:>16}: skipped -- {exc}", flush=True)
            continue
        # `_block` is called with the (already subset) protospacers, so the
        # family block comes back subset too -- masking it again was an
        # off-by-cohort IndexError on the first grouped run.
        Xa = np.hstack([X, v])
        res = cross_validate(Xa, y, names + fnames,
                             ["base"] * len(names) + [fam] * len(fnames),
                             n_features=n_features, seeds=seeds, n_splits=n_splits,
                             verbose=False, groups=groups,
                             tag=f"{tag_prefix}_{fam}", experiment=fam)
        rho = res["summary"]["spearman"]["mean"]
        rows.append(dict(family=fam, n_cols=Xa.shape[1], spearman=rho,
                         spearman_sd=res["summary"]["spearman"]["sd"],
                         d_spearman=rho - base_rho))
        if verbose:
            print(f"  {fam:>16}: rho {rho:.4f}   delta {rho - base_rho:+.4f}  "
                  f"({len(fnames)} cols)", flush=True)
        # Must be OUT_, not OUT: a leftover raw write here sent the
        # gene-grouped results into the random-row table and destroyed it.
        _write(rows, OUT_)

    out = pd.read_csv(OUT_)
    if verbose:
        print()
        print(out.round(4).to_string(index=False))
        print(f"written -> {OUT_}")
    return out


SHAPE_OUT = config.RESULTS / "ablation_curated_shape.csv"


def shape_audit(verbose: bool = True) -> pd.DataFrame:
    """Is a family's gain explained by how many dense continuous columns it supplies?

    The curated base is 6,169 sparse indicators with none of the published
    base's 63 hand-named summaries, so a family that restores dense continuous
    columns may be paid for its *shape* rather than its content. That was an
    eyeball reading of four families; this measures it across all ten.

    A column counts as dense continuous if it takes more than ten distinct
    values on the cohort -- crude, but it separates real-valued summaries from
    motif indicators and small counts without a hand-made list.
    """
    abl = pd.read_csv(OUT).set_index("family")
    rows = []
    for fam in abl.index:
        if fam == "baseline":
            continue
        path = config.INTERIM / "features" / f"{fam}_curated.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path).drop(columns=[config.ID_COL])
        d = d.loc[:, ~d.isna().all()]
        cont = int((d.nunique() > 10).sum())
        rows.append(dict(family=fam, n_cols=d.shape[1], continuous=cont,
                         continuous_frac=round(cont / d.shape[1], 3),
                         d_spearman=float(abl.loc[fam, "d_spearman"])))
    out = pd.DataFrame(rows).sort_values("d_spearman", ascending=False)
    out.to_csv(SHAPE_OUT, index=False)
    if verbose:
        from scipy.stats import spearmanr
        r = spearmanr(out["continuous"], out["d_spearman"])
        print(out.round(4).to_string(index=False))
        print(f"\n  rho(dense continuous columns, delta rho) = {r.statistic:.3f}"
              f"   p = {r.pvalue:.4f}   n = {len(out)}")
        print(f"written -> {SHAPE_OUT}")
    return out

# ------------------------------------------------------------------------------------------------ the paper's own folds
# Design `paper_folds`: the family ablation on exactly the folds of the paper's other numbers (`controls/sealed_holdout_gene._fold_ids` over all 33,567 guides, every gene whole,
# the 3,364 guides outside genes in 100 kb blocks, seeds 41 to 43, so 15 folds) with the default model, so a delta here sits beside the flank gain of +0.161 without a change of protocol.
# Every arm sees the same five folds per seed. Arms: the 20-mer with the same 300-column selection every family arm gets (the like-for-like baseline); the 20-mer plus one family; and the two halves of
# `d_mechanics` (its 22 columns that are functions of the protospacer, its 8 that read the flanking DNA: `.flank.` in the column name). `flank_scale.csv` already holds two arms on these folds and they are
# read, not rerun: the 20-mer with no selection (the paper's 0.547) and the 20-mer + family A. Each gain is reported against both baselines. Results: `ablation_paper_folds.csv` (per fold) and
# `ablation_paper_folds_summary.csv`; fold checkpoints are the rows of the csv, so a rerun resumes.
#     python -m sgrna.cohorts.curated_ablation --paper-folds [--threads 9] [--seeds 41 42 43] [--summary-only]
import time

from scipy.stats import spearmanr

from ..common import OUT as OUT_PF

PF_RESULT = OUT_PF / "ablation_paper_folds.csv"
PF_SUMMARY = OUT_PF / "ablation_paper_folds_summary.csv"
PF_BASE_SEL = "20-mer, 300 selected"
# guide-derived families first (the claim this answers), then the families that read DNA or tracks beyond the guide
PF_ORDER = ("b_energy", "c_folding", "f_methylation", "j_periodicity", "h_offtarget",
         "d_mechanics", "i_shape", "e_nucleoid", "g_transcription", "d_supercoiling")
PF_SPLIT = {"d_mechanics, guide columns only": False, "d_mechanics, flank columns only": True}     # arm -> keep the columns that read the flanks?
PF_FLANK = ".flank."


def run_paper_folds(seeds=(41, 42, 43), threads: int = 9, verbose: bool = True) -> None:
    from ..controls import sealed_holdout_gene as sg
    from ..controls.mappability_stratum import _groups
    from ..interpret import flank_scale as FS
    
    sg.N_JOBS = threads
    X, y, names, protos = _base()
    groups = _groups(protos)
    done = pd.read_csv(PF_RESULT) if PF_RESULT.exists() and PF_RESULT.stat().st_size else pd.DataFrame(columns=["arm", "seed", "fold", "rho", "n_cols"])
    rows = done.to_dict("records")
    have = {(r["arm"], r["seed"]) for r in rows}
    blocks = {}
    for seed in seeds:
        fold = sg._fold_ids(groups, 5, seed)
        for arm in (PF_BASE_SEL,) + PF_ORDER + tuple(PF_SPLIT):
            if (arm, seed) in have:
                continue
            if arm == PF_BASE_SEL:
                M = X
            elif arm in PF_SPLIT:
                if "d_mechanics" not in blocks:
                    blocks["d_mechanics"] = _block("d_mechanics", protos)
                v, fnames = blocks["d_mechanics"]
                is_flank = np.array([PF_FLANK in f for f in fnames])
                M = np.hstack([X, v[:, is_flank if PF_SPLIT[arm] else ~is_flank]])
            else:
                if arm not in blocks:
                    blocks[arm] = _block(arm, protos)
                M = np.hstack([X, blocks[arm][0]])
            t0 = time.time()
            pred = FS._cv(M, y, groups, fold, config.N_FEATURES, seed)
            for k in range(5):
                va = fold == k
                rows.append(dict(arm=arm, seed=seed, fold=k + 1, rho=float(spearmanr(y[va], pred[va]).statistic), n_cols=int(M.shape[1])))
            pd.DataFrame(rows).to_csv(PF_RESULT, index=False)
            if verbose:
                print(f"  {arm:24s} seed {seed}  rho {spearmanr(y, pred).statistic:.4f}  {time.time() - t0:.0f}s", flush=True)


def summary_paper_folds() -> pd.DataFrame:
    d = pd.read_csv(PF_RESULT)
    fs = pd.read_csv(OUT_PF / "flank_scale.csv")
    nosel = fs[fs.arm == "base"].set_index(["seed", "fold"]).rho
    fam_a = fs[fs.arm == "family A shifted 0"].set_index(["seed", "fold"]).rho
    sel = d[d.arm == PF_BASE_SEL].set_index(["seed", "fold"]).rho
    arms = {"a_flank": fam_a}
    for arm, g in d[d.arm != PF_BASE_SEL].groupby("arm", sort=False):
        arms[arm] = g.set_index(["seed", "fold"]).rho
    rows = []
    for arm, r in arms.items():
        ix = r.index.intersection(sel.index).intersection(nosel.index)
        a = (r.loc[ix] - sel.loc[ix])                 # like for like: the same 300-column selection on both arms
        b = (r.loc[ix] - nosel.loc[ix])               # against the paper's baseline, the 20-mer with no selection
        rows.append(dict(family=arm, n_folds=len(ix), seeds=int(ix.get_level_values(0).nunique()), rho=float(r.loc[ix].mean()),
                         gain_same_selection=float(a.mean()), gain_same_selection_sd=float(a.std(ddof=1)), folds_positive_same_selection=int((a > 0).sum()),
                         gain_vs_no_selection=float(b.mean()), gain_vs_no_selection_sd=float(b.std(ddof=1)), folds_positive_vs_no_selection=int((b > 0).sum()),
                         base_selected_rho=float(sel.loc[ix].mean()), base_no_selection_rho=float(nosel.loc[ix].mean())))
    s = pd.DataFrame(rows)
    s.to_csv(PF_SUMMARY, index=False)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--n-features", type=int, default=300)
    ap.add_argument("--grouping", choices=["gene", "block100k"], default=None,
                    help="hold out whole genes (or 100 kb blocks) instead of rows")
    ap.add_argument("--shape-audit", action="store_true",
                    help="only audit column shape against the measured deltas")
    ap.add_argument("--force-baseline", action="store_true",
                    help="re-measure the base representation instead of reusing it")
    ap.add_argument("--paper-folds", action="store_true", help="design paper_folds: the ablation on the paper's own folds (see the section above)")
    ap.add_argument("--threads", type=int, default=9, help="paper_folds only")
    ap.add_argument("--summary-only", action="store_true", help="paper_folds only")
    a = ap.parse_args(argv)
    if a.paper_folds:
        if not a.summary_only:
            run_paper_folds(tuple(a.seeds or (41, 42, 43)), a.threads)
        print(summary_paper_folds().round(4).to_string(index=False))
        return 0
    if a.shape_audit:
        shape_audit()
        return 0
    fams = tuple(a.families) if a.families else FAMILIES
    run(families=fams, seeds=a.seeds, n_features=a.n_features,
        force_baseline=a.force_baseline, grouping=a.grouping)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
