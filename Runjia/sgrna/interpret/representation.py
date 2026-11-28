"""Seventy-four mononucleotide columns, a lossless spelling of the 20-mer, recover 96.4% of the full published matrix's rho, so the model does not need the other columns; and the published 63 hand-named summary columns, added to the sparse base, take only 0.0015 off family A's gain, so missing dense summaries do not explain the curated table's larger gains (`--base-enrichment`), a model using only the 412 columns that can be computed from any 20-mer plus the genome matches the full published matrix to within 0.00003 in rho (`--portability`), and the 63 hand-named columns split into the computable and the rest (`--handnamed`).

Section 10 of RESULTS.md establishes that 6,169 of the 6,232 published columns
are a deterministic function of the 20 nt protospacer -- the matrix is an
elaborate spelling of a 20-letter word. That is an argument about *information*.
It does not by itself say the model can be given less, because a tree can only
use what it can split on, and a redundant encoding might still be the shape
that makes the signal reachable.

This module settles that empirically. Every arm is the same rows, the same
cross-validation, the same champion -- only the columns the model is allowed to
see change:

  published_full     all 6,232 columns, the baseline
  v_onehot_only      the 5,853 identified (position, k-mer) indicators
  qct_only           the 316 quantum-chemical lookups
  hand_named_only    the ~63 remaining published columns
  mono_74            the mononucleotide indicators alone -- 74 columns, and a
                     *lossless* encoding: every position carries at least three
                     of its four base indicators, so the fourth is one minus
                     the others and the 20-mer is recoverable exactly
  mono_di_364        mononucleotide plus dinucleotide indicators
  mono_74+a_flank    the minimal encoding plus the one family that added
                     information from outside the 20-mer

The prediction, if section 10 is right: mono_74 matches published_full. If it
does not, redundancy is doing real work as a learning aid and the "elaborate
spelling" framing needs qualifying.

Note that with 74 columns the top-300 selection step is a no-op, so that arm
also removes the selector from the pipeline entirely.

    python -m sgrna.interpret.representation --run
    python -m sgrna.interpret.representation --base-enrichment [--published]
    python -m sgrna.interpret.representation --portability [--audit] [--run]
    python -m sgrna.interpret.representation --handnamed

--portability   A model using only the 412 columns that can be computed from any 20-mer plus the genome matches the full published matrix to within 0.00003 in ρ.
Finding 2 says 427 columns do the work of 6,232. That is a claim about
*sufficiency*, and a pivot to crisprHAL's cohort needs a different property:
**portability** -- every column computable from an arbitrary 20-mer plus the
reference genome, with no row in the published matrix to look it up in.

The 427 are `mono_di_364` plus the 63 hand-named columns, and only the second
group is in doubt. They audit as:

    20  guide mono/dinucleotide counts     from the 20-mer
     4  PAM base identity                  from the genome
     3  guide GC, Tm, RNA secondary structure
     2  gene.distance0, pam.distance0      needs gene annotation
    34  undecoded V columns                the open question

The 34 are tested by least squares against the mono+di one-hots. That is an
**in-sample** fit, so it is an upper bound: a column the 20-mer cannot explain
even in sample is definitively not a function of the 20-mer, at least not a
linear one. 19 come back at R2 >= 0.95 (effectively deterministic, so
regenerable by refitting); the rest do not, and those are the blockers.

The test here is the only one that settles the cost: drop the blockers and
measure. If the reduced-minus-blockers arm is within noise of the 427, the
representation is portable at no accuracy cost and the pivot is unobstructed.

--handnamed   Split the 63 hand-named columns into the computable and the rest, and test both.
Finding 2 says the 63 are "non-sequence" columns and that the 427-column reduced
set (364 sequence indicators plus the 63) is non-inferior to all 6,232. An audit
of what they actually contain (`portability_audit.csv`) shows the label is wrong:
**61 of the 63 are the guide re-described** — whole-guide k-mer counts, GC, Tm,
RNA structure, PAM bases, and the 19 `V` columns reconstructible from mono- and
dinucleotide indicators — and only **2** need anything else, both distances that
require gene annotation. A further 15 `V` columns are undecoded and not
reconstructible, so they are unknown rather than computable.

That makes a clean two-way test of a question the report has been circling: when
the 63 help, is it because they carry information the guide does not, or because
they re-spell the guide in a form a depth-limited tree can use?

    computable_46    20 k-mer counts + GC/Tm/structure + 4 PAM + 19
                     reconstructible V -- all functions of the 20-mer
    noncomputable_17 2 annotation distances + 15 undecoded, non-reconstructible V
    all_63           both, which is the published 427 arm

The interpretations, fixed in advance:

  * if `computable_46` carries the gain and `noncomputable_17` does not, the 427
    result is another instance of re-description paying — the same effect as the
    quantum tables, CRISPRoff and ViennaRNA, which all land at +0.011 to +0.019;
  * if `noncomputable_17` carries it, the hand-named columns are a genuine
    source of outside information and annotation-derived features deserve more
    attention than this project has given them.

Run on the published 13,880-row matrix, because that is the only cohort where
these columns exist.

--base-enrichment
-----------------
Adding the published 63 hand-named summary columns to the sparse base takes only 0.0015 off family A's gain, so missing dense summaries do not explain the curated table's larger gains.

Every family's Δρ roughly tripled when the cohort moved, with the ordering
almost unchanged (rank ρ 0.97, Appendix B3). Two of the four candidate causes
are already eliminated: the **label** moves family A by 0.004 at fixed rows
(`ablation_label_*.csv`), and the **row count** by nothing at all --
13,880 curated rows give +0.170 against 33,567 rows' +0.164
(`family_a_controls.py`, part rows). What remains is the row population and the base
representation.

The base is testable directly. The published base carried 63 hand-named columns
including guide Tm and GC; the curated base is 6,169 sparse indicators and
carries none. If that absence is what inflates every family, then giving the
base dense continuous guide-derived summaries should shrink the next family's
gain.

The summaries used are the ones already built and known to carry **no
information** beyond the 20-mer (Finding 2): `b_energy`'s 10 columns,
`c_folding`'s 9, and `d_mechanics`'s 22 protospacer columns -- 41 dense
continuous columns, none of which reads a base outside the guide. That matters
for the logic: if family A's gain shrinks against this base, it shrank because
the base gained *summary shape*, not because it gained *information*.

    base                 0.5445      (cached from the curated ablation)
    base + a_flank       0.7084      Δ +0.164
    base + summaries                 Δ ?
    base + summaries + a_flank       Δ(a_flank | summaries) = ?
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from .. import config
from ..common import OUT as OUT_DIR
from ..harness.curated import load_base, load_block

OUT = config.RESULTS / "representation.csv"
OUT_DEPTH = config.RESULTS / "representation_depth.csv"


def _mapping() -> dict:
    raw = json.loads((config.INTERIM / "v_column_mapping.json").read_text())
    return raw["mapping"]


def column_sets(names: list[str]) -> dict[str, np.ndarray]:
    """Index arrays for each representation, over the published columns only."""
    names = list(names)
    idx = {n: i for i, n in enumerate(names)}
    vmap = _mapping()

    v_cols = [i for n, i in idx.items() if n in vmap]
    mono = [i for n, i in idx.items() if len(vmap.get(n, {}).get("kmer", "")) == 1]
    di = [i for n, i in idx.items() if len(vmap.get(n, {}).get("kmer", "")) == 2]
    qct = [i for n, i in idx.items()
           if n not in vmap and n[:1] == "p" and "." in n
           and any(u in n for u in ("monomer", "dimer", "trimer", "tetramer",
                                    "basepair"))]
    accounted = set(v_cols) | set(qct)
    hand = [i for i in range(len(names)) if i not in accounted]

    return {
        "published_full": np.arange(len(names)),
        "v_onehot_only": np.sort(np.array(v_cols)),
        "qct_only": np.sort(np.array(qct)),
        "hand_named_only": np.sort(np.array(hand)),
        "mono_74": np.sort(np.array(mono)),
        "mono_di_364": np.sort(np.array(mono + di)),
    }


def run(seeds=None, n_splits=None, with_flank: bool = True, verbose: bool = True):
    from ..build.build_matrix import load_dataset
    from ..harness.run_ablation import cross_validate

    seeds = seeds or config.SEEDS
    n_splits = n_splits or config.N_SPLITS

    families = ["a_flank"] if with_flank else []
    X, y, names, ids, family_of = load_dataset(families, verbose=verbose)
    family_of = np.asarray(family_of)
    base = np.where(family_of == "base")[0]
    flank = np.where(family_of != "base")[0]
    base_names = [names[i] for i in base]

    sets = column_sets(base_names)
    arms = {k: base[v] for k, v in sets.items()}
    if with_flank and len(flank):
        arms["mono_74+a_flank"] = np.sort(np.concatenate([base[sets["mono_74"]], flank]))
        arms["published_full+a_flank"] = np.arange(X.shape[1])

    rows = []
    for arm, cols in arms.items():
        if verbose:
            print(f"\n== {arm}: {len(cols):,} columns ==", flush=True)
        res = cross_validate(
            X[:, cols], y, [names[i] for i in cols],
            family_of[cols], seeds=seeds, n_splits=n_splits,
            verbose=verbose, tag=f"repr_{arm}", experiment=arm,
        )
        s = res["summary"]
        rows.append(dict(
            arm=arm, n_columns=int(len(cols)),
            spearman=s["spearman"]["mean"], spearman_sd=s["spearman"]["sd"],
            r2=s["r2"]["mean"], r2_sd=s["r2"]["sd"],
            complete=res["complete"],
        ))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        if verbose:
            print(f"   rho {s['spearman']['mean']:.4f}  R2 {s['r2']['mean']:.4f}",
                  flush=True)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Follow-up: is the redundancy carrying information, or expressivity?
# --------------------------------------------------------------------------


def run_depth(seeds=None, n_splits=None, depths=(4, 6, 8, 12), verbose=True):
    """If 74 columns are lossless, why does the full matrix still score higher?

    The first experiment shows `mono_74` -- a lossless encoding of the 20-mer --
    scoring below `published_full`. That cannot be an information difference,
    because there is no information in the full matrix that is not in the 74
    columns. So it has to be an *expressivity* difference, and there is a
    specific candidate: the champion is `max_depth=4`. A tetramer indicator
    states a four-base interaction in one split; from mononucleotide indicators
    alone the same statement needs four stacked splits, which is the entire
    depth budget, leaving nothing for anything else. The published matrix is
    not adding information -- it is pre-computing interaction terms.

    That is testable. Give the minimal encoding more depth. If the gap closes,
    redundancy was a learning aid and nothing more. If it does not, something
    else is going on and the lossless-encoding argument is incomplete.

    The k-mer ladder is the same question asked from the other side: hand the
    depth-4 model explicit 2-mer and 3-mer indicators and see the gap close in
    proportion to the interaction order supplied.
    """
    from ..build.build_matrix import load_dataset
    from ..harness.run_ablation import cross_validate

    seeds = seeds or config.SEEDS
    n_splits = n_splits or config.N_SPLITS

    X, y, names, _, family_of = load_dataset([], verbose=verbose)
    family_of = np.asarray(family_of)
    sets = column_sets(list(names))
    vmap = _mapping()
    tri = [i for i, n in enumerate(names) if len(vmap.get(n, {}).get("kmer", "")) == 3]
    ladder = {
        "mono_74": sets["mono_74"],
        "mono_di_364": sets["mono_di_364"],
        "mono_di_tri": np.sort(np.concatenate([sets["mono_di_364"], np.array(tri)])),
        "published_full": sets["published_full"],
    }

    rows = []
    for arm, cols in ladder.items():
        for depth in depths:
            if arm == "published_full" and depth != 4:
                continue        # the anchor is only defined at the frozen depth
            tagname = f"reprdepth_{arm}_d{depth}"
            if verbose:
                print(f"\n== {arm}: {len(cols):,} columns, max_depth={depth} ==",
                      flush=True)
            res = cross_validate(
                X[:, cols], y, [names[i] for i in cols], family_of[cols],
                seeds=seeds, n_splits=n_splits, verbose=verbose,
                tag=tagname, experiment=f"{arm}_d{depth}",
                champion_overrides={"max_depth": depth},
            )
            s = res["summary"]
            rows.append(dict(
                arm=arm, n_columns=int(len(cols)), max_depth=depth,
                spearman=s["spearman"]["mean"], spearman_sd=s["spearman"]["sd"],
                r2=s["r2"]["mean"], complete=res["complete"],
            ))
            pd.DataFrame(rows).to_csv(OUT_DEPTH, index=False)
            if verbose:
                print(f"   rho {s['spearman']['mean']:.4f}", flush=True)
    return pd.DataFrame(rows)


def summary_depth():
    d = pd.read_csv(OUT_DEPTH)
    print(d.pivot_table(index="arm", columns="max_depth", values="spearman")
          .round(4).to_string())
    return d


def summary():
    d = pd.read_csv(OUT)
    full = d.loc[d["arm"] == "published_full", "spearman"]
    if len(full):
        d["delta_vs_full"] = (d["spearman"] - float(full.iloc[0])).round(4)
    print(d.round(4).to_string(index=False))
    return d


def run_residual(seeds=None, n_splits=None, verbose=True):
    """What is the full matrix's advantage over 74 columns actually made of?

    `run_depth` tested the obvious explanation -- that a redundant encoding
    pre-computes interaction terms a depth-4 tree cannot build -- and refuted
    it: more depth makes every arm monotonically worse, and adding trimer
    indicators does not help either. So the gap is not expressivity.

    That leaves the one part of the published matrix that section 10 says is
    *not* a function of the protospacer: the 63 columns left over after the
    indicators and the quantum lookups. They include `gene.distance0`, which is
    positional, and base counts that disagree with a direct count of the guide
    by up to 13 -- so they are not redundant with the 20-mer at all. On their
    own they reach rho 0.304.

    If adding them to the minimal encoding closes the gap, the accounting is
    clean: the matrix is 74 columns of sequence, 63 columns of something else,
    and ~6,095 columns of restatement.
    """
    from ..build.build_matrix import load_dataset
    from ..harness.run_ablation import cross_validate

    seeds = seeds or config.SEEDS
    n_splits = n_splits or config.N_SPLITS
    X, y, names, _, family_of = load_dataset([], verbose=verbose)
    family_of = np.asarray(family_of)
    s = column_sets(list(names))
    hand = s["hand_named_only"]

    arms = {
        "mono_74+hand_63": np.sort(np.concatenate([s["mono_74"], hand])),
        "mono_di_364+hand_63": np.sort(np.concatenate([s["mono_di_364"], hand])),
        "qct_316+hand_63": np.sort(np.concatenate([s["qct_only"], hand])),
    }

    rows = []
    for arm, cols in arms.items():
        if verbose:
            print(f"\n== {arm}: {len(cols):,} columns ==", flush=True)
        res = cross_validate(
            X[:, cols], y, [names[i] for i in cols], family_of[cols],
            seeds=seeds, n_splits=n_splits, verbose=verbose,
            tag=f"reprres_{arm}", experiment=arm)
        sm = res["summary"]
        rows.append(dict(arm=arm, n_columns=int(len(cols)),
                         spearman=sm["spearman"]["mean"],
                         spearman_sd=sm["spearman"]["sd"],
                         r2=sm["r2"]["mean"], complete=res["complete"]))
        pd.DataFrame(rows).to_csv(config.RESULTS / "representation_residual.csv",
                                  index=False)
        if verbose:
            print(f"   rho {sm['spearman']['mean']:.4f}", flush=True)
    return pd.DataFrame(rows)



def run_headroom(seeds=(41,), n_splits: int = 5, with_flank: bool = True,
                 verbose: bool = True):
    """Does the 427-column subset leave information on the table?

    The argument that the full matrix is exhausted is a residual test
    (`diagnose.headroom`): fit the champion, take its out-of-fold residuals,
    and correlate every column against them. Nothing above |rho| 0.035 means
    there is no column the model was given and failed to use -- it is out of
    material rather than underfitting.

    Dropping 5,805 columns makes that argument again, and it is not the same
    argument. Those columns are functions of the 20-mer, but the subset only
    carries the *first and second order* of it: a trimer or tetramer indicator
    could in principle track something a mono+di representation cannot express.
    If it did, the subset would be leaving information unused even though its
    headline rho matches. So the test is run on the subset's own residuals,
    with every dropped column scored against them.

    Arms: the 427-column subset, and the same plus family A, each compared
    against the full matrix on the identical folds.
    """
    from scipy.stats import spearmanr
    from sklearn.model_selection import KFold
    import xgboost as xgb

    from ..build.build_matrix import load_dataset
    from ..harness.run_ablation import _impute, champion_params, make_selector

    families = ["a_flank"] if with_flank else []
    X, y, names, _, family_of = load_dataset(families, verbose=verbose)
    names = np.asarray(names)
    family_of = np.asarray(family_of)
    base = np.where(family_of == "base")[0]
    flank = np.where(family_of != "base")[0]
    sets = column_sets([names[i] for i in base])
    subset = np.sort(np.concatenate([base[sets["mono_di_364"]],
                                     base[sets["hand_named_only"]]]))

    arms = {"subset_427": subset, "published_full": base}
    if with_flank and len(flank):
        arms["subset_427+a_flank"] = np.sort(np.concatenate([subset, flank]))
        arms["published_full+a_flank"] = np.arange(X.shape[1])

    rows = []
    seed = seeds[0]
    for arm, cols in arms.items():
        used = np.zeros(X.shape[1], bool)
        used[cols] = True
        pred = np.empty_like(y)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(kf.split(X), start=1):
            X_tr, X_va = _impute(X[tr][:, cols], X[va][:, cols])
            sel = make_selector(seed)
            sel.fit(X_tr, y[tr])
            top = np.argsort(sel.feature_importances_)[::-1][:min(config.N_FEATURES,
                                                                  len(cols))]
            model = xgb.XGBRegressor(**champion_params(seed))
            model.fit(X_tr[:, top], y[tr])
            pred[va] = model.predict(X_va[:, top])
        resid = y - pred
        rho_oof = float(spearmanr(y, pred).statistic)
        if verbose:
            print(f"\n== {arm}: {len(cols):,} columns, out-of-fold rho "
                  f"{rho_oof:.4f} ==", flush=True)

        Xf = np.where(np.isfinite(X), X, np.nan)
        best = {}
        for j in range(X.shape[1]):
            v = Xf[:, j]
            ok = np.isfinite(v)
            if ok.sum() < 200 or np.nanstd(v[ok]) == 0:
                continue
            r = spearmanr(v[ok], resid[ok]).statistic
            if not np.isfinite(r):
                continue
            group = ("given to the model" if used[j]
                     else "dropped from the matrix")
            rows.append(dict(arm=arm, column=names[j], group=group,
                             rho_with_residual=float(r)))
            best[group] = max(best.get(group, 0.0), abs(float(r)))
        rows.append(dict(arm=arm, column="__oof_rho__", group="summary",
                         rho_with_residual=rho_oof))
        pd.DataFrame(rows).to_csv(config.RESULTS / "representation_headroom.csv",
                                  index=False)
        if verbose:
            for g, v in sorted(best.items()):
                print(f"   max |rho| with residual, {g:26s} {v:.3f}", flush=True)
    return pd.DataFrame(rows)


def summary_headroom():
    d = pd.read_csv(config.RESULTS / "representation_headroom.csv")
    oof = d[d["group"] == "summary"].set_index("arm")["rho_with_residual"]
    d = d[d["group"] != "summary"].copy()
    d["abs"] = d["rho_with_residual"].abs()
    t = d.groupby(["arm", "group"])["abs"].agg(["max", "count",
                                               lambda s: (s > 0.10).sum()])
    t.columns = ["max_abs_rho", "n_columns", "n_above_0.10"]
    t["out_of_fold_rho"] = [oof.get(a) for a, _ in t.index]
    print(t.round(4).to_string())
    return t


# ======================================================================================================  base enrichment
ENRICH_OUT = OUT_DIR / "base_enrichment.csv"
ENRICH_PUBLISHED_OUT = OUT_DIR / "base_enrichment_published.csv"

SUMMARIES = ("b_energy", "c_folding", "d_mechanics")
FLANK = ".flank."


def _summary_block(protos):
    """The 41 dense guide-only summary columns, `d_mechanics`'s flanks removed."""
    vals, names = [], []
    for fam in SUMMARIES:
        v, fn = load_block(fam, protos)
        keep = np.array([FLANK not in f for f in fn])
        vals.append(v[:, keep])
        names += [f for f, k in zip(fn, keep) if k]
    return np.hstack(vals), names


def enrichment(seeds=None, n_splits=None, n_features=300, verbose: bool = True):
    from ..harness.run_ablation import cross_validate

    seeds = list(seeds or config.SEEDS)
    n_splits = n_splits or config.N_SPLITS
    X, y, names, protos = load_base()
    S, snames = _summary_block(protos)
    A, anames = load_block("a_flank", protos)
    if verbose:
        print(f"  summaries: {S.shape[1]} dense guide-only columns; "
              f"a_flank: {A.shape[1]}")

    arms = {
        "base": (X, names, ["base"] * len(names), "curabl_baseline", "baseline"),
        "base+a_flank": (np.hstack([X, A]), names + anames,
                         ["base"] * len(names) + ["a_flank"] * len(anames),
                         "curabl_a_flank", "a_flank"),
        "base+summaries": (np.hstack([X, S]), names + snames,
                           ["base"] * len(names) + ["sum"] * len(snames),
                           "enrich_summaries", "summaries"),
        "base+summaries+a_flank": (np.hstack([X, S, A]), names + snames + anames,
                                   ["base"] * len(names) + ["sum"] * len(snames)
                                   + ["a_flank"] * len(anames),
                                   "enrich_sum_flank", "summaries+a_flank"),
    }

    rows = []
    for arm, (Xa, nm, fam, tag, exp) in arms.items():
        res = cross_validate(Xa, y, nm, fam, n_features=n_features, seeds=seeds,
                             n_splits=n_splits, verbose=False, tag=tag,
                             experiment=exp)
        rows.append(dict(arm=arm, n_cols=Xa.shape[1],
                         spearman=res["summary"]["spearman"]["mean"],
                         spearman_sd=res["summary"]["spearman"]["sd"]))
        if verbose:
            print(f"  {arm:>24}: rho {rows[-1]['spearman']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(ENRICH_OUT, index=False)

    out = pd.DataFrame(rows).set_index("arm")
    d_plain = out.loc["base+a_flank", "spearman"] - out.loc["base", "spearman"]
    d_enriched = (out.loc["base+summaries+a_flank", "spearman"]
                  - out.loc["base+summaries", "spearman"])
    if verbose:
        print(f"\n  a_flank over the plain base    : {d_plain:+.4f}")
        print(f"  a_flank over the enriched base : {d_enriched:+.4f}")
        print(f"  shrinkage attributable to base shape: "
              f"{d_plain - d_enriched:+.4f}")
        print(f"written -> {ENRICH_OUT}")
    return out




def enrichment_published(seeds=None, n_splits=None, n_features=300, verbose: bool = True):
    """The same question asked with the real hand-named columns, not a stand-in.

    `run()` above uses 41 dense guide-only columns borrowed from our own
    families, because the published matrix's own 63 hand-named summaries -- guide
    GC, Tm, k-mer counts, PAM indicators -- exist on the 13,880-row cohort and
    nowhere else. So the test belongs on that cohort, where the two bases differ
    by exactly those 63 columns and by nothing else:

        6,169 indicators + quantum tables     -> + a_flank: delta ?
        the same, plus the 63 hand-named      -> + a_flank: delta ?

    If dense summary shape is what inflates a family's gain over a sparse base,
    the second delta is the smaller one, and the difference is the part of the
    published/curated gap that belongs to the base rather than to the rows.
    """
    from ..build.build_matrix import load_dataset
    from ..harness.run_ablation import cross_validate

    seeds = list(seeds or config.SEEDS)
    n_splits = n_splits or config.N_SPLITS
    X, y, names, ids, family_of = load_dataset(families=("a_flank",),
                                               verbose=verbose)
    pub = [i for i, f in enumerate(family_of) if f == "base"]
    flank = [i for i, f in enumerate(family_of) if f == "a_flank"]
    sets = column_sets([names[i] for i in pub])
    hand = set(int(i) for i in sets["hand_named_only"])
    sparse = [pub[i] for i in range(len(pub)) if i not in hand]
    if verbose:
        print(f"  published base {len(pub):,} columns = {len(sparse):,} sparse "
              f"+ {len(hand)} hand-named; a_flank {len(flank)}")

    arms = {
        "sparse base": sparse,
        "sparse base+a_flank": sparse + flank,
        "published base (+63 hand-named)": pub,
        "published base+a_flank": pub + flank,
    }
    rows = []
    for arm, idx in arms.items():
        nm = [names[i] for i in idx]
        fam = [family_of[i] for i in idx]
        res = cross_validate(X[:, idx], y, nm, fam, n_features=n_features,
                             seeds=seeds, n_splits=n_splits, verbose=False,
                             tag="enrpub_" + arm.replace(" ", "_").replace("+", "_"),
                             experiment=arm)
        rows.append(dict(arm=arm, n_cols=len(idx),
                         spearman=res["summary"]["spearman"]["mean"],
                         spearman_sd=res["summary"]["spearman"]["sd"]))
        if verbose:
            print(f"  {arm:>34}: rho {rows[-1]['spearman']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(ENRICH_PUBLISHED_OUT, index=False)

    out = pd.DataFrame(rows).set_index("arm")
    d_sparse = (out.loc["sparse base+a_flank", "spearman"]
                - out.loc["sparse base", "spearman"])
    d_hand = (out.loc["published base+a_flank", "spearman"]
              - out.loc["published base (+63 hand-named)", "spearman"])
    if verbose:
        print(f"\n  a_flank over the sparse base              : {d_sparse:+.4f}")
        print(f"  a_flank over the base with 63 hand-named : {d_hand:+.4f}")
        print(f"  shrinkage attributable to the real summaries: "
              f"{d_sparse - d_hand:+.4f}")
        print(f"written -> {ENRICH_PUBLISHED_OUT}")
    return out


# ======================================================================================================  portability and the hand-named split
PORT_OUT = OUT_DIR / "portability.csv"
PORT_AUDIT_OUT = OUT_DIR / "portability_audit.csv"
HNS_OUT = OUT_DIR / "handnamed_split.csv"

RECONSTRUCTIBLE = 0.95


def _classify(name: str) -> str:
    if re.fullmatch(r"[ACGT]{1,2}sgRNA\.raw", name):
        return "guide k-mer count"
    if name.startswith("PAM."):
        return "PAM base"
    if name == "sgRNA.gcsgRNA.raw":
        return "guide GC"
    if name == "sgRNA.structuresgRNA.raw":
        return "guide RNA structure"
    if name == "sgRNA.tempsgRNA.raw":
        return "guide Tm"
    if name in ("gene.distance0", "pam.distance0"):
        return "positional (needs annotation)"
    if name.startswith("V"):
        return "undecoded V column"
    return "other"


def portability_audit(verbose: bool = True):
    """Classify the 63, and score the undecoded ones against the 20-mer."""
    from ..build.build_matrix import load_dataset

    X, y, names, ids, fam = load_dataset([], verbose=False)
    sets = column_sets(names)
    hand = [names[i] for i in sets["hand_named_only"]]

    A = np.c_[X[:, sets["mono_di_364"]], np.ones(len(X))]
    finite = np.isfinite(A).all(1)

    rows = []
    for h in hand:
        kind = _classify(h)
        r2 = np.nan
        if kind == "undecoded V column":
            v = X[:, names.index(h)]
            m = finite & np.isfinite(v)
            if m.sum() >= 100 and np.std(v[m]) > 0:
                coef, *_ = np.linalg.lstsq(A[m], v[m], rcond=None)
                resid = v[m] - A[m] @ coef
                r2 = 1 - (resid**2).sum() / ((v[m] - v[m].mean())**2).sum()
        rows.append(dict(column=h, kind=kind, r2_from_20mer=r2,
                         portable=(kind != "undecoded V column")
                                  or (np.isfinite(r2) and r2 >= RECONSTRUCTIBLE)))
    out = pd.DataFrame(rows)
    out.to_csv(PORT_AUDIT_OUT, index=False)
    if verbose:
        print(out.groupby(["kind", "portable"]).size().to_string())
        print(f"\nblockers ({(~out['portable']).sum()}):")
        for _, r in out[~out["portable"]].sort_values("r2_from_20mer").iterrows():
            print(f"   R2 {r['r2_from_20mer']:.4f}  {r['column']}")
        print(f"written -> {PORT_AUDIT_OUT}")
    return out


def blockers() -> list[str]:
    a = portability_audit(verbose=False)
    return a.loc[~a["portable"], "column"].tolist()


def portability(seeds=None, n_splits=None, verbose: bool = True):
    """427 columns against 427 minus the non-portable ones."""
    from ..build.build_matrix import load_dataset
    from ..harness.run_ablation import cross_validate

    seeds = list(seeds or config.SEEDS)
    n_splits = n_splits or config.N_SPLITS

    X, y, names, ids, fam = load_dataset([], verbose=verbose)
    sets = column_sets(names)
    reduced = np.sort(np.concatenate([sets["mono_di_364"],
                                      sets["hand_named_only"]]))
    bad = set(blockers())
    keep = np.array([i for i in reduced if names[i] not in bad])
    if verbose:
        print(f"  reduced {len(reduced)} columns, {len(bad)} non-portable, "
              f"portable arm {len(keep)}")

    rows = []
    for arm, cols in (("reduced_427", reduced), ("portable", keep),
                      ("published_full", np.arange(X.shape[1]))):
        res = cross_validate(X[:, cols], y, [names[i] for i in cols],
                             [fam[i] for i in cols], n_features=None,
                             seeds=seeds, n_splits=n_splits, verbose=False,
                             tag=f"port_{arm}", experiment=arm)
        s = res["summary"]
        rows.append(dict(arm=arm, n_columns=len(cols),
                         spearman=s["spearman"]["mean"],
                         spearman_sd=s["spearman"]["sd"],
                         r2=s["r2"]["mean"]))
        if verbose:
            print(f"  {arm:>16}: {len(cols):>4} cols  rho "
                  f"{s['spearman']['mean']:.4f} +- {s['spearman']['sd']:.4f}")
        pd.DataFrame(rows).to_csv(PORT_OUT, index=False)

    out = pd.DataFrame(rows)
    if verbose and len(out) >= 2:
        d = out.loc[1, "spearman"] - out.loc[0, "spearman"]
        print(f"\ncost of dropping the non-portable columns: {d:+.4f} rho")
        print(f"written -> {PORT_OUT}")
    return out


def _handnamed_groups(names: list[str]):
    """The audit's verdict, mapped onto column indices of the loaded matrix."""
    a = pd.read_csv(PORT_AUDIT_OUT)
    # `portable` in the audit means "computable from the genome", which for the
    # two gene-distance columns still requires annotation -- so they are split
    # out by `kind` rather than by that flag.
    needs_annotation = set(a.loc[a["kind"].str.contains("annotation"), "column"])
    v_bad = set(a.loc[(a["kind"] == "undecoded V column")
                      & (~a["portable"]), "column"])
    hand = set(a["column"])
    idx = {n: i for i, n in enumerate(names)}
    present = [n for n in hand if n in idx]
    missing = len(hand) - len(present)
    if missing:
        print(f"  ! {missing} audited columns are not in the matrix")
    noncomp = [n for n in present if n in needs_annotation or n in v_bad]
    comp = [n for n in present if n not in needs_annotation and n not in v_bad]
    return ([idx[n] for n in comp], [idx[n] for n in noncomp],
            [idx[n] for n in present])


def handnamed(seeds=None, n_splits=None, verbose: bool = True):
    from ..build.build_matrix import load_dataset
    from ..harness.run_ablation import cross_validate

    seeds = list(seeds or config.SEEDS)
    n_splits = n_splits or config.N_SPLITS
    X, y, names, ids, fam = load_dataset([], verbose=verbose)
    sets = column_sets(names)
    base = np.sort(sets["mono_di_364"])
    comp, noncomp, allh = _handnamed_groups(list(names))
    if verbose:
        print(f"  base {len(base)} sequence indicators; hand-named split "
              f"{len(comp)} computable / {len(noncomp)} not")

    rows = []
    arms = {"base_364": base,
            "+computable_46": np.sort(np.concatenate([base, comp])),
            "+noncomputable_17": np.sort(np.concatenate([base, noncomp])),
            "+all_63": np.sort(np.concatenate([base, allh]))}
    for arm, cols in arms.items():
        res = cross_validate(X[:, cols], y, [names[i] for i in cols],
                             [fam[i] for i in cols],
                             n_features=len(cols),      # no selection anywhere
                             seeds=seeds, n_splits=n_splits, verbose=False,
                             tag=f"hns_{arm.strip('+')}", experiment=arm)
        rho = res["summary"]["spearman"]["mean"]
        rows.append(dict(arm=arm, n_columns=len(cols), spearman=rho,
                         spearman_sd=res["summary"]["spearman"]["sd"]))
        if verbose:
            print(f"  {arm:>18}: {len(cols):>4} cols  rho {rho:.4f} "
                  f"± {rows[-1]['spearman_sd']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(HNS_OUT, index=False)

    out = pd.DataFrame(rows).set_index("arm")
    if verbose:
        b = out.loc["base_364", "spearman"]
        for arm in ("+computable_46", "+noncomputable_17", "+all_63"):
            print(f"  {arm:>18} over the base: "
                  f"{out.loc[arm, 'spearman'] - b:+.4f}")
        print(f"written -> {HNS_OUT}")
    return out.reset_index()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--no-flank", action="store_true")
    ap.add_argument("--headroom", action="store_true",
                    help="is the reduced column set also out of material?")
    ap.add_argument("--residual", action="store_true",
                    help="run the leftover-columns accounting follow-up")
    ap.add_argument("--base-enrichment", action="store_true", help="does a dense-summary base shrink family A's gain?")
    ap.add_argument("--portability", action="store_true", help="the 412 portable columns against the full published matrix")
    ap.add_argument("--audit", action="store_true", help="--portability: classify the 63 hand-named columns (default if --run is not given)")
    ap.add_argument("--handnamed", action="store_true", help="the 63 hand-named columns split into computable and not")
    ap.add_argument("--published", action="store_true", help="--base-enrichment: with the real 63 hand-named columns on the published cohort")
    ap.add_argument("--depth", action="store_true",
                    help="run the depth / k-mer-ladder follow-up instead")
    args = ap.parse_args(argv)
    if args.portability:
        if args.audit or not args.run:
            portability_audit()
        if args.run:
            portability()
        return 0
    if args.handnamed:
        handnamed()
        return 0
    if args.base_enrichment:
        (enrichment_published if args.published else enrichment)()
        return 0
    if args.headroom:
        run_headroom(n_splits=args.folds or config.N_SPLITS)
        summary_headroom()
        return 0
    if args.residual:
        run_residual(seeds=tuple(args.seeds) if args.seeds else None,
                     n_splits=args.folds)
        return 0
    if args.depth:
        run_depth(seeds=tuple(args.seeds) if args.seeds else None,
                  n_splits=args.folds)
        summary_depth()
        return 0
    if args.run:
        run(seeds=tuple(args.seeds) if args.seeds else None,
            n_splits=args.folds, with_flank=not args.no_flank)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
