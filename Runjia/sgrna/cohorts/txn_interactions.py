"""The interactions do not pay: adding the four interaction columns to the 15 single-factor columns raises rho by only +0.0004, far below the +0.0175 they earn alone.

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

    python -m sgrna.cohorts.txn_interactions
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import config
from .curated_ablation import _base, _block

OUT = config.RESULTS / "txn_interactions.csv"
IX = ".ix."


def run(seeds=None, n_splits=None, n_features=300, verbose: bool = True):
    from ..harness.run_ablation import cross_validate

    seeds = list(seeds or config.SEEDS)
    n_splits = n_splits or config.N_SPLITS
    X, y, names, protos = _base()
    v, fnames = _block("g_transcription", protos)
    is_ix = np.array([IX in f for f in fnames])
    if verbose:
        print(f"  g_transcription: {len(fnames)} columns, {is_ix.sum()} interactions, "
              f"{(~is_ix).sum()} marginals")

    base = pd.read_csv(config.RESULTS / "ablation_curated.csv").set_index("family")
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
        pd.DataFrame(rows).to_csv(OUT, index=False)

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
        print(f"written -> {OUT}")
    return out.reset_index()


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
