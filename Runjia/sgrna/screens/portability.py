"""Can the 427-column representation be rebuilt for a guide we have never seen?

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

    python -m sgrna.portability --audit
    python -m sgrna.portability --run
"""

from __future__ import annotations

import argparse
import re

import numpy as np

from .. import config

OUT = config.RESULTS / "portability.csv"
AUDIT_OUT = config.RESULTS / "portability_audit.csv"
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


def audit(verbose: bool = True):
    """Classify the 63, and score the undecoded ones against the 20-mer."""
    import pandas as pd

    from ..build.build_matrix import load_dataset
    from ..interpret.representation import column_sets

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
    out.to_csv(AUDIT_OUT, index=False)
    if verbose:
        print(out.groupby(["kind", "portable"]).size().to_string())
        print(f"\nblockers ({(~out['portable']).sum()}):")
        for _, r in out[~out["portable"]].sort_values("r2_from_20mer").iterrows():
            print(f"   R2 {r['r2_from_20mer']:.4f}  {r['column']}")
        print(f"written -> {AUDIT_OUT}")
    return out


def blockers() -> list[str]:
    a = audit(verbose=False)
    return a.loc[~a["portable"], "column"].tolist()


def run(seeds=None, n_splits=None, verbose: bool = True):
    """427 columns against 427 minus the non-portable ones."""
    import pandas as pd

    from ..build.build_matrix import load_dataset
    from ..interpret.representation import column_sets
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
        pd.DataFrame(rows).to_csv(OUT, index=False)

    out = pd.DataFrame(rows)
    if verbose and len(out) >= 2:
        d = out.loc[1, "spearman"] - out.loc[0, "spearman"]
        print(f"\ncost of dropping the non-portable columns: {d:+.4f} rho")
        print(f"written -> {OUT}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args(argv)
    if a.audit or not a.run:
        audit()
    if a.run:
        run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
