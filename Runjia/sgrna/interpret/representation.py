"""Does the model need all 6,232 published columns, or one spelling of the 20-mer?

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

    python -m sgrna.representation --run
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .. import config

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
    ap.add_argument("--depth", action="store_true",
                    help="run the depth / k-mer-ladder follow-up instead")
    args = ap.parse_args(argv)
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
