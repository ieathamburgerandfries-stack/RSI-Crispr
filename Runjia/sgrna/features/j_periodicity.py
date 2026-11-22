"""Family J -- strand asymmetry and helical periodicity in the flanking DNA.

Three features, chosen because each is something GC content provably cannot be.
The report's rule is that a candidate can only help if it is not computable from what
the model already holds; these are all *positional* properties of the genome, so
they pass by construction, and two of them are not even functions of base
composition.

**GC skew**, (G - C) / (G + C) over a window. GC *content* is symmetric under
swapping the two strands; GC skew is not. That matters here because the one
effect this project cannot explain -- the long-range downstream-versus-upstream
asymmetry -- is itself strand-asymmetric, and a symmetric quantity is
the wrong shape to explain an asymmetric effect. In *E. coli* skew also flips
sign at the replication origin and terminus, so it carries replichore
information without the model being told the coordinate.

**A-tract phase power.** Runs of A or T spaced at the helical repeat (~10.5 bp)
produce intrinsic curvature: each tract bends the duplex slightly, and if the
bends are in phase they add instead of cancelling. The quantity is therefore
*periodicity*, not composition -- the same number of A's arranged at 10.5 bp
spacing and arranged at random give the same GC content and very different
curvature. Computed as the share of spectral power of the A/T indicator signal
that falls in the 9.5-11.5 bp band.

**Bendability phase power.** The same spectral measure applied to a
dinucleotide bendability track (`bend_major` from the shape scales) rather than
to the A/T indicator. Where A-tract phasing asks "are the stiff tracts in
phase", this asks "is the deformability itself modulated at the helical repeat",
which is the quantity nucleosome- and nucleoid-positioning models use.

Why spectral power and not an autocorrelation: power at a band is insensitive to
where in the window the periodicity starts, and a guide's flank has no
privileged phase origin.

    python -m sgrna.build.build_features --families j_periodicity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config

NAME = "j_periodicity"
WINDOWS = (50, 250, 500, 1000)
HELICAL_LO, HELICAL_HI = 9.5, 11.5          # bp per turn, B-form DNA
_SHAPE = "bend_major"


def _skew(seq: str) -> float:
    g, c = seq.count("G"), seq.count("C")
    return (g - c) / (g + c) if (g + c) else np.nan


def _band_power(x: np.ndarray) -> float:
    """Share of spectral power in the helical band, mean removed.

    Returns NaN for a window too short to resolve the band at all -- a 50 nt
    window has only ~5 turns, which is thin but usable; anything shorter is not.
    """
    n = len(x)
    if n < 32:
        return np.nan
    x = x - x.mean()
    if not np.any(x):
        return 0.0
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(n)                # cycles per bp
    total = power[1:].sum()                  # drop DC
    if total <= 0:
        return 0.0
    band = (freq >= 1.0 / HELICAL_HI) & (freq <= 1.0 / HELICAL_LO)
    return float(power[band].sum() / total)


def _at_signal(seq: str) -> np.ndarray:
    return np.fromiter((1.0 if ch in "AT" else 0.0 for ch in seq),
                       dtype=float, count=len(seq))


def _shape_signal(seq: str, table: dict) -> np.ndarray:
    return np.fromiter((table.get(seq[i:i + 2], np.nan)
                        for i in range(len(seq) - 1)),
                       dtype=float, count=max(0, len(seq) - 1))


def _load_shape_table() -> dict:
    path = config.REFERENCE / "dinucleotide_shape_scales.tsv"
    df = pd.read_csv(path, sep="\t", comment="#").set_index("scale")
    return df.loc[_SHAPE].astype(float).to_dict()


def build(index: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """One row per located guide; windows on each side, upstream and downstream."""
    table = _load_shape_table()
    rows = []
    kept = index[index["located"].astype(bool)]
    for _, r in kept.iterrows():
        up = str(r.get("upstream") or "").upper()
        dn = str(r.get("downstream") or "").upper()
        out = {config.ID_COL: r[config.ID_COL]}
        for side, seq in (("up", up[::-1]), ("dn", dn)):
            # upstream is reversed so that index 0 is the base nearest the
            # protospacer on both sides -- otherwise a 50 nt upstream window
            # would be the 50 bases furthest away.
            for w in WINDOWS:
                sub = seq[:w]
                if len(sub) < w:
                    skew = atp = bend = np.nan
                else:
                    s = sub[::-1] if side == "up" else sub
                    skew = _skew(s)
                    atp = _band_power(_at_signal(s))
                    sig = _shape_signal(s, table)
                    bend = (_band_power(sig[np.isfinite(sig)])
                            if np.isfinite(sig).sum() >= 32 else np.nan)
                out[f"eng.per.skew.{side}{w}"] = skew
                out[f"eng.per.atphase.{side}{w}"] = atp
                out[f"eng.per.bendphase.{side}{w}"] = bend
        rows.append(out)
    df = pd.DataFrame(rows)
    if verbose:
        n = len([c for c in df.columns if c != config.ID_COL])
        print(f"  {NAME}: {len(df):,} guides x {n} columns "
              f"(GC skew, A-tract phase, bendability phase at "
              f"{HELICAL_LO}-{HELICAL_HI} bp)")
    return df
