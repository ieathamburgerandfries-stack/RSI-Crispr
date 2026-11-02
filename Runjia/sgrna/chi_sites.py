"""Is the flank effect a *repair* effect? The Chi-site test.

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

    python -m sgrna.chi_sites
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from . import config

OUT = config.RESULTS / "chi_sites.csv"
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


def run(verbose: bool = True) -> pd.DataFrame:
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
        pd.DataFrame(rows).to_csv(OUT, index=False)

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
    pd.DataFrame(rows).to_csv(OUT, index=False)
    if verbose:
        print(f"\n  Chi beta (GC-adjusted, +/-{w} nt): {chi_beta:+.4f}")
        print(f"  {len(betas)} control octamers of the same GC content: "
              f"mean {betas.mean():+.4f}, sd {betas.std(ddof=1):.4f}, "
              f"range {betas.min():+.4f} to {betas.max():+.4f}")
        print(f"  Chi sits at the {pct:.1f}th percentile of that null")
        print(f"written -> {OUT}")
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
