"""What a given Spearman buys when the user has to pick one guide out of k.

rho is the right headline because ranking is the decision, but it is not a scale
anyone has intuitions about, and the gap between 0.72 and the label's 0.946
ceiling does not look like much written that way. This converts both into the
decision they describe: show the model k candidates for a gene, let it choose,
and ask how good the guide it picked actually was.

Two readouts, because they answer different questions:

  population percentile   where the chosen guide sits in the whole cohort.
                          Chance is the 50th; the best of ten averages the 91st.
                          This is `evaluate.expected_pick_percentile`, and this
                          module reproduces its convention exactly.
  P(picks the best)       how often the chosen guide is the actual best of the
                          k offered. Chance is 1/k.

The simulation is a Gaussian copula: draw a true quality and a prediction with
the Pearson correlation that gives the requested Spearman for a bivariate
normal, r = 2 sin(pi rho / 6). That assumes joint normality, which the real
label is not, so these are the idealised translation of a rho rather than a
measurement on this cohort -- the measured pick percentile for the fitted model
is in `panel.py`, and lands in the same place.

    python -m sgrna.decision_scale
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config

OUT = config.RESULTS / "decision_scale.csv"
ANCHORS = (
    ("blind choice", 0.0),
    ("ridge on 300 columns", 0.6473),
    ("crisprHAL 2, published configuration", 0.6971),
    ("crisprHAL 2, best configuration found", 0.7051),
    ("SLICER, our five folds", 0.7218),
    ("the label's measured ceiling", 0.946),
    ("a perfect predictor", 0.9999),
)


def _simulate(rho: float, k: int, n: int, rng) -> tuple[float, float]:
    r = 2.0 * np.sin(np.pi * rho / 6.0)
    t = rng.standard_normal((n, k))
    p = r * t + np.sqrt(max(1.0 - r * r, 0.0)) * rng.standard_normal((n, k))
    chosen = t[np.arange(n), p.argmax(axis=1)]
    return (float(norm.cdf(chosen).mean() * 100),
            float((chosen == t.max(axis=1)).mean() * 100))


def run(ks=(5, 10, 20), n: int = 400_000, seed: int = 41,
        anchors=ANCHORS, verbose: bool = True):
    rng = np.random.default_rng(seed)
    rows = []
    for name, rho in anchors:
        for k in ks:
            pct, best = _simulate(rho, k, n, rng)
            rows.append(dict(arm=name, spearman=rho, k=k,
                             pick_percentile=pct, p_picks_best=best))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    if verbose:
        ten = out[out.k == 10]
        print(f"  choosing 1 of 10, {n:,} draws:")
        print(f"  {'':38s} {'pick %ile':>10s} {'P(best)':>9s}")
        for _, r in ten.iterrows():
            print(f"  {r.arm:38s} {r.pick_percentile:9.1f}% {r.p_picks_best:8.1f}%")
        base = ten[ten.arm == "blind choice"].iloc[0]
        ceil = ten[ten.arm == "the label's measured ceiling"].iloc[0]
        us = ten[ten.arm == "SLICER, our five folds"].iloc[0]
        got = (us.p_picks_best - base.p_picks_best) / (ceil.p_picks_best
                                                       - base.p_picks_best)
        print(f"\n  on P(picks the best of ten), the model captures "
              f"{got:.0%} of the distance from chance to the ceiling")
        print(f"written -> {OUT}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=400_000)
    a = ap.parse_args(argv)
    run(n=a.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
