"""Feature-family ablations on crisprHAL's curated cohort (one family at a time on random rows or whole genes, `--paper-folds`), and three questions asked of the same cohort: the family gains overlap heavily (all eleven families together add +0.2030 rho, far less than the +0.7046 their single gains sum to), the 412-column portable representation still reaches the full one with no selection, and model class does not matter more once family A is in (eight classes span 0.108 rho), and the transcription family's interaction columns do not pay (+0.0004 on top of their 15 single-factor parents, far below the +0.0175 they earn alone).

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
    python -m sgrna.cohorts.curated_ablation --joint [--arms a_flank+i_shape ...]
    python -m sgrna.cohorts.curated_ablation --ladder
    python -m sgrna.cohorts.curated_ablation --model-classes [--models xgboost ridge ...]
    python -m sgrna.cohorts.curated_ablation --interactions

--joint
-------
Appendix B3 reports eleven single-family increments over one base and warns that
they must not be added up — summing them gives +0.70 ρ, which is not a real
quantity. That warning rests on the *published* cohort's joint arms, where
`i_shape` on top of family A was worth +0.001 against +0.036 alone, and the three
positional families collapsed into one variable. Neither has been checked here,
so the warning is inherited rather than measured.

The arms are chosen to answer specific suspicions rather than to enumerate:

    a+i             `i_shape` is flank bendability and `a_flank` is flank
                    composition over the same DNA -- the suspected duplicate
    a+d_sup         the largest external family on top of the largest flank one
    d_sup+e+g       the three positional families, which collapsed to one
                    variable on the published matrix
    a+d_sup+i       the three biggest effects together
    all_eleven      every family at once: the ceiling on stacking them
    guide_only      `b_energy` + `c_folding` + `f_methylation`, the three that
                    read nothing outside the guide, as a control -- they should
                    be close to additive because they overlap only through the
                    20-mer

Everything else matches `curated_ablation`: same base, same folds, same seeds,
selection fitted inside each training split.

--ladder
--------
Finding 2's ladder — 74 mononucleotide indicators, 364 with dinucleotides, 412
portable, 6,232 full — was measured on the published matrix, and it carries a
protocol caveat: every arm selected 300 columns from its candidate pool, so the
74-column arm used all of its candidates while the 6,232-column arm used 300 of
6,232. The ladder measured candidate pools under a fixed budget, not
representations.

This re-runs it on the curated cohort and removes the caveat: **no selection in
any arm**, `n_features` set to each arm's own column count. Two questions at
once, both open rows in Part 4:

  * does the reduced set still reach the full representation on this cohort,
    where the 63 hand-named columns are absent by construction?
  * is `n_features = 300` costing anything? The arms below use every column
    they have, so the 6,169- and 6,517-column arms answer it directly.

Columns are chosen from the decoded mapping (`decode_v_columns`), so "74
mononucleotide indicators" means exactly the columns that decode to
(position, single base) and nothing else.

Three seeds × five folds rather than five × five: no arm selects, so each fit
is over every column it holds, and the ladder is internally consistent at three.

--interactions
--------------
`g_transcription` was built with a discipline written into its module docstring:
*claim nothing unless an interaction term beats both of its parents.* The reason
is that Guo et al. already reported the marginals as null — target-gene
expression, chromosome position, leading-versus-lagging strand — so the family
exists to test whether a mechanism needing **both** transcription and strand to
line up shows up where neither marginal does.

That discipline was never applied on the curated cohort. Appendix B3 reports the
family at +0.058 there and flags the omission; this closes it. Three arms over
the same base:

    marginals    the 15 columns that are distance, strand, position or
                 expression on their own
    interactions the 4 `eng.txn.ix.*` columns alone
    both         the full family, which is the +0.058 already reported

The rule resolves as: interactions pay only if `both` beats `marginals` by more
than the interaction columns would be expected to add as four arbitrary dense
columns — so the `interactions`-alone arm is the reference for that, not zero.

--model-classes
---------------
"Sixteen model classes span only 0.116 ρ" is a Finding 2 corollary, and it was
measured on the published matrix **without** family A — i.e. on a representation
with no large effect in it. The reading it supports ("the representation matters
and the model class barely does") is therefore circular unless it survives on a
representation that *does* contain the effect.

Same registry (`evaluate._make`), same folds, curated cohort, every model given
base + family A. One seed × five folds, because the question is the *spread*
across classes and not a per-class estimate to three decimals.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from .. import config
from ..harness.curated import load_base, load_block, load_groups

OUT = config.RESULTS / "ablation_curated.csv"
GROUPED_OUT = {"gene": config.RESULTS / "ablation_curated_gene.csv",
               "block100k": config.RESULTS / "ablation_curated_block.csv"}
FAMILIES = ("a_flank", "b_energy", "c_folding", "d_mechanics", "d_supercoiling",
            "e_nucleoid", "f_methylation", "g_transcription", "h_offtarget",
            "i_shape", "j_periodicity")


ORDER = ("baseline",) + FAMILIES


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
    X, y, names, protos = load_base()

    OUT_ = OUT if grouping is None else GROUPED_OUT[grouping]
    tag_prefix = "curabl" if grouping is None else f"curabl{grouping[:4]}"
    groups = None
    if grouping is not None:
        groups, ok = load_groups(grouping, protos)
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
            v, fnames = load_block(fam, protos)
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
# Design `paper_folds`: the family ablation on exactly the folds of the paper's other numbers (`harness/folds.fold_ids` over all 33,567 guides, every gene whole,
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
    from ..harness import folds as sg
    from ..controls.mappability_stratum import _groups
    from ..interpret import flank_scale as FS
    
    sg.N_JOBS = threads
    X, y, names, protos = load_base()
    groups = load_groups(protos)
    done = pd.read_csv(PF_RESULT) if PF_RESULT.exists() and PF_RESULT.stat().st_size else pd.DataFrame(columns=["arm", "seed", "fold", "rho", "n_cols"])
    rows = done.to_dict("records")
    have = {(r["arm"], r["seed"]) for r in rows}
    blocks = {}
    for seed in seeds:
        fold = sg.fold_ids(groups, 5, seed)
        for arm in (PF_BASE_SEL,) + PF_ORDER + tuple(PF_SPLIT):
            if (arm, seed) in have:
                continue
            if arm == PF_BASE_SEL:
                M = X
            elif arm in PF_SPLIT:
                if "d_mechanics" not in blocks:
                    blocks["d_mechanics"] = load_block("d_mechanics", protos)
                v, fnames = blocks["d_mechanics"]
                is_flank = np.array([PF_FLANK in f for f in fnames])
                M = np.hstack([X, v[:, is_flank if PF_SPLIT[arm] else ~is_flank]])
            else:
                if arm not in blocks:
                    blocks[arm] = load_block(arm, protos)
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


# ======================================================================================================  the cohort's other experiments
# Three more questions on the same curated cohort, base and folds as the designs above. Each writes its own file.
#
#   joint            how the family gains combine -> curated_joint.csv
#   ladder           the reduced-representation ladder without selection -> curated_ladder.csv
#   model-classes    does model class matter more once family A is in? -> curated_models.csv

JOINT_OUT = OUT_PF / "curated_joint.csv"
LADDER_OUT = OUT_PF / "curated_ladder.csv"
MODELS_OUT = OUT_PF / "curated_models.csv"
TXN_OUT = OUT_PF / "txn_interactions.csv"
LADDER_SEEDS = (41, 42, 43)


JOINT_ARMS = {
    "a_flank+i_shape": ("a_flank", "i_shape"),
    "a_flank+d_supercoiling": ("a_flank", "d_supercoiling"),
    "d_supercoiling+e_nucleoid+g_transcription": ("d_supercoiling", "e_nucleoid",
                                                  "g_transcription"),
    "a_flank+d_supercoiling+i_shape": ("a_flank", "d_supercoiling", "i_shape"),
    "guide_only_three": ("b_energy", "c_folding", "f_methylation"),
    "all_eleven": ("a_flank", "b_energy", "c_folding", "d_mechanics",
                   "d_supercoiling", "e_nucleoid", "f_methylation",
                   "g_transcription", "h_offtarget", "i_shape", "j_periodicity"),
}


def joint(arms=None, seeds=None, n_splits=None, n_features=300,
        verbose: bool = True):
    from ..harness.run_ablation import cross_validate

    arms = arms or JOINT_ARMS
    seeds = list(seeds or config.SEEDS)
    n_splits = n_splits or config.N_SPLITS
    X, y, names, protos = load_base()

    singles = pd.read_csv(OUT).set_index("family")
    base_rho = float(singles.loc["baseline", "spearman"])

    rows = []
    if JOINT_OUT.exists() and JOINT_OUT.stat().st_size:
        rows = [r for r in pd.read_csv(JOINT_OUT).to_dict("records")
                if r["arm"] not in arms]

    for arm, fams in arms.items():
        if any(r["arm"] == arm for r in rows):
            continue
        vals, nm, fam_of = [], [], []
        for f in fams:
            v, fn = load_block(f, protos)
            vals.append(v)
            nm += fn
            fam_of += [f] * len(fn)
        Xa = np.hstack([X] + vals)
        res = cross_validate(Xa, y, names + nm,
                             ["base"] * len(names) + fam_of,
                             n_features=n_features, seeds=seeds,
                             n_splits=n_splits, verbose=False,
                             tag=f"joint_{len(fams)}_{fams[0]}",
                             experiment=arm)
        rho = res["summary"]["spearman"]["mean"]
        additive = float(sum(singles.loc[f, "d_spearman"] for f in fams))
        rows.append(dict(arm=arm, n_families=len(fams), n_cols=Xa.shape[1],
                         spearman=rho, spearman_sd=res["summary"]["spearman"]["sd"],
                         d_spearman=rho - base_rho,
                         sum_of_singles=additive,
                         overlap=additive - (rho - base_rho),
                         share_realised=(rho - base_rho) / additive))
        if verbose:
            r = rows[-1]
            print(f"  {arm:<44} rho {rho:.4f}  joint {r['d_spearman']:+.4f}  "
                  f"sum {additive:+.4f}  realised {r['share_realised']:.0%}",
                  flush=True)
        pd.DataFrame(rows).to_csv(JOINT_OUT, index=False)

    out = pd.DataFrame(rows)
    out.to_csv(JOINT_OUT, index=False)
    if verbose:
        print()
        print(out.round(4).to_string(index=False))
        print(f"written -> {JOINT_OUT}")
    return out


def _unit_columns(names, units) -> np.ndarray:
    from ..build import decode_v_columns as dv

    mp = dv.load()["mapping"]
    return np.array([n in mp and mp[n].get("unit") in units
                     and not mp[n].get("ambiguous") for n in names])


def ladder(seeds=LADDER_SEEDS, n_splits=None, verbose: bool = True):
    from ..harness.run_ablation import cross_validate

    n_splits = n_splits or config.N_SPLITS
    X, y, names, protos = load_base()
    mono = _unit_columns(names, ("monomer",))
    mono_di = _unit_columns(names, ("monomer", "dimer"))
    A, anames = load_block("a_flank", protos)

    mono_di_tri = _unit_columns(names, ("monomer", "dimer", "trimer"))
    all_kmers = _unit_columns(names, ("monomer", "dimer", "trimer", "tetramer"))
    sel = lambda m: [n for n, k in zip(names, m) if k]
    arms = {
        "mono": (X[:, mono], sel(mono)),
        "mono_di": (X[:, mono_di], sel(mono_di)),
        # Finding 4 says higher-order k-mers help once selection is removed, and
        # that was measured on the published matrix. These two arms re-ask it
        # here, where no arm selects: trimers add 1,146 columns and tetramers a
        # further 4,343, so if the trimer step pays and the tetramer step does
        # not, the usable order stops at three on this cohort too.
        "mono_di_tri": (X[:, mono_di_tri], sel(mono_di_tri)),
        # The step from mono_di_tri to base_full adds two different things at
        # once -- 4,343 tetramer indicators and 316 quantum-table columns -- so
        # a gain there cannot be attributed to either. This arm is the k-mers
        # without the quantum columns, which is what Finding 4's "the usable
        # order stops at three" is actually a claim about.
        "all_kmers": (X[:, all_kmers], sel(all_kmers)),
        "base_full": (X, names),
        "mono_di+a_flank": (np.hstack([X[:, mono_di], A]), sel(mono_di) + anames),
        "mono_di_tri+a_flank": (np.hstack([X[:, mono_di_tri], A]),
                                sel(mono_di_tri) + anames),
        "all_kmers+a_flank": (np.hstack([X[:, all_kmers], A]),
                              sel(all_kmers) + anames),
        "base_full+a_flank": (np.hstack([X, A]), names + anames),
    }

    rows = []
    for arm, (Xa, nm) in arms.items():
        res = cross_validate(Xa, y, nm, ["arm"] * len(nm),
                             n_features=Xa.shape[1],      # every column, no selection
                             seeds=list(seeds), n_splits=n_splits, verbose=False,
                             tag=f"ladder_{arm}", experiment=arm)
        rows.append(dict(arm=arm, n_cols=Xa.shape[1],
                         spearman=res["summary"]["spearman"]["mean"],
                         spearman_sd=res["summary"]["spearman"]["sd"]))
        if verbose:
            print(f"  {arm:>18} {Xa.shape[1]:>5} cols: rho "
                  f"{rows[-1]['spearman']:.4f} ± {rows[-1]['spearman_sd']:.4f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(LADDER_OUT, index=False)

    out = pd.DataFrame(rows)
    if verbose:
        full = out.set_index("arm").loc["base_full", "spearman"]
        out["vs_base_full"] = out["spearman"] - full
        print()
        print(out.round(4).to_string(index=False))
        print(f"written -> {LADDER_OUT}")
    out.to_csv(LADDER_OUT, index=False)
    return out


CLASSES = ("xgboost", "lightgbm", "random_forest", "extra_trees", "ridge",
          "elastic_net", "knn", "pls")


def model_classes(models=CLASSES, seed: int = 41, n_splits: int = 5, verbose: bool = True):
    from ..harness.evaluate import _make, _standardise
    from ..harness.run_ablation import _impute, make_selector

    X, y, names, protos = load_base()
    A, _ = load_block("a_flank", protos)
    X = np.hstack([X, A])
    if verbose:
        print(f"  curated base + family A: {X.shape[0]:,} x {X.shape[1]:,}")

    done = set()
    if MODELS_OUT.exists() and MODELS_OUT.stat().st_size:
        done = set(pd.read_csv(MODELS_OUT)["model"])

    rows = [] if not done else pd.read_csv(MODELS_OUT).to_dict("records")
    folds = list(KFold(n_splits, shuffle=True, random_state=seed).split(X))
    for name in models:
        if name in done:
            continue
        t0, rhos = time.time(), []
        for tr, va in folds:
            X_tr, X_va = _impute(X[tr], X[va])
            sel = make_selector(seed)
            sel.fit(X_tr, y[tr])
            top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
            X_tr, X_va = X_tr[:, top], X_va[:, top]
            model, scale = _make(name, seed)
            if scale:
                X_tr, X_va = _standardise(X_tr, X_va)
            model.fit(X_tr, y[tr])
            rhos.append(float(spearmanr(y[va], model.predict(X_va)).statistic))
        rows.append(dict(model=name, spearman=float(np.mean(rhos)),
                         spearman_sd=float(np.std(rhos, ddof=1)),
                         seconds=round(time.time() - t0, 1)))
        if verbose:
            print(f"  {name:>14}: rho {rows[-1]['spearman']:.4f} "
                  f"± {rows[-1]['spearman_sd']:.4f}  ({rows[-1]['seconds']:.0f}s)",
                  flush=True)
        pd.DataFrame(rows).to_csv(MODELS_OUT, index=False)

    out = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    out.to_csv(MODELS_OUT, index=False)
    if verbose:
        span = out["spearman"].max() - out["spearman"].min()
        print()
        print(out.round(4).to_string(index=False))
        print(f"\n  spread across {len(out)} classes: {span:.4f} rho")
        print(f"written -> {MODELS_OUT}")
    return out


IX = ".ix."


def interactions(seeds=None, n_splits=None, n_features=300, verbose: bool = True):
    from ..harness.run_ablation import cross_validate

    seeds = list(seeds or config.SEEDS)
    n_splits = n_splits or config.N_SPLITS
    X, y, names, protos = load_base()
    v, fnames = load_block("g_transcription", protos)
    is_ix = np.array([IX in f for f in fnames])
    if verbose:
        print(f"  g_transcription: {len(fnames)} columns, {is_ix.sum()} interactions, "
              f"{(~is_ix).sum()} marginals")

    base = pd.read_csv(OUT).set_index("family")
    base_rho = float(base.loc["baseline", "spearman"])

    # Why these two extra arms. The curated label is *survival after cutting*,
    # so targeting an essential gene depletes the cell whether or not the guide
    # is any good. Any gene-level column -- essentiality above all -- can
    # therefore predict the label through the phenotype rather than through Cas9
    # activity. If the family's gain is carried by `expr.essential` alone, it is
    # measuring the assay, not transcription.
    is_ess = np.array([f.endswith(".expr.essential") for f in fnames])
    is_expr = np.array([".expr." in f for f in fnames])
    rows = []
    for arm, mask in (("marginals", ~is_ix), ("interactions", is_ix),
                      ("both", np.ones(len(fnames), dtype=bool)),
                      ("essential_only", is_ess), ("expression_only", is_expr),
                      ("no_expression", ~is_expr & ~is_ix)):
        sub = [f for f, m in zip(fnames, mask) if m]
        Xa = np.hstack([X, v[:, mask]])
        res = cross_validate(Xa, y, names + sub,
                             ["base"] * len(names) + ["txn"] * len(sub),
                             n_features=n_features, seeds=seeds,
                             n_splits=n_splits, verbose=False,
                             tag=f"txnix_{arm}", experiment=arm)
        rho = res["summary"]["spearman"]["mean"]
        rows.append(dict(arm=arm, n_family_cols=int(mask.sum()), spearman=rho,
                         spearman_sd=res["summary"]["spearman"]["sd"],
                         d_spearman=rho - base_rho))
        if verbose:
            print(f"  {arm:>13}: rho {rho:.4f}   delta {rho - base_rho:+.4f}  "
                  f"({int(mask.sum())} cols)", flush=True)
        pd.DataFrame(rows).to_csv(TXN_OUT, index=False)

    out = pd.DataFrame(rows).set_index("arm")
    if verbose:
        m, i, b = (out.loc[a, "d_spearman"] for a in ("marginals", "interactions", "both"))
        print(f"\n  interactions on top of their parents: {b - m:+.4f}")
        print(f"  the rule is satisfied only if that exceeds what four dense "
              f"columns buy alone ({i:+.4f})")
        if "essential_only" in out.index:
            print(f"  essentiality alone: {out.loc['essential_only', 'd_spearman']:+.4f}"
                  f"   expression block: {out.loc['expression_only', 'd_spearman']:+.4f}"
                  f"   family without it: {out.loc['no_expression', 'd_spearman']:+.4f}")
        print(f"written -> {TXN_OUT}")
    return out.reset_index()


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
    ap.add_argument("--joint", action="store_true", help="how the family gains combine (see the section above)")
    ap.add_argument("--arms", nargs="*", default=None, help="--joint: only these arms")
    ap.add_argument("--ladder", action="store_true", help="the reduced-representation ladder without selection")
    ap.add_argument("--interactions", action="store_true", help="the transcription family: do its interaction columns pay?")
    ap.add_argument("--model-classes", action="store_true", help="model class with family A in the representation")
    ap.add_argument("--models", nargs="*", default=None, help="--model-classes: only these classes")
    a = ap.parse_args(argv)
    if a.joint:
        joint(arms={k: v for k, v in JOINT_ARMS.items() if k in a.arms} if a.arms else None, seeds=a.seeds)
        return 0
    if a.ladder:
        ladder(seeds=tuple(a.seeds) if a.seeds else LADDER_SEEDS)
        return 0
    if a.interactions:
        interactions(seeds=a.seeds)
        return 0
    if a.model_classes:
        model_classes(models=tuple(a.models) if a.models else CLASSES)
        return 0
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
