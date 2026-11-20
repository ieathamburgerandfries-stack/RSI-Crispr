"""Family D2 -- DNA supercoiling at the target, from GapR-seq.

Guo et al. (2018) could not explain why some chromosomal loci resist cutting
regardless of guide sequence, and pointed at "unknown chromosomal factors".
Supercoiling is the leading candidate: R-loop formation needs the duplex to
unwind locally, so torsional state should change how easily Cas9 invades. A
biophysical model of Cas9 (Farasat & Salis 2016) treats supercoiling as a
first-class term, and recent biochemistry shows it alters the *sequence
dependence* of cleavage, not just the rate.

As far as we can find, nobody has tested this against sgRNA efficiency in
E. coli -- which is what makes it the most novel item in the project, and why
it is worth reporting even if the answer is no.

The data
--------
GSE152880 (Guo, Laub et al., eLife 2021) maps positive supercoiling genome
wide by ChIP of GapR, a protein that binds overtwisted DNA. Seven wiggle
tracks at single-base resolution on NC_000913.2:

    signal      GapR-3xFLAG, exponential phase          (2 replicates)
    control     untagged GapR -- the no-ChIP background (2 replicates)
    rifampicin  GapR-3xFLAG with transcription stopped  (1)
    truncated   GapR(1-76)-3xFLAG                       (2)

Because transcription is what generates most of the supercoiling landscape,
the rifampicin track is the interesting control: signal minus rifampicin
isolates the transcription-driven component, and the rifampicin track alone
carries whatever torsion survives without RNA polymerase.

What the ablation actually found, and why the names changed
-----------------------------------------------------------
This family gives +0.047 R2 under 100 kb-blocked CV -- and almost none of it
is supercoiling. Dissecting it (see results/INTERPRETATION.md section 1):

  * the raw tracks correlate -0.83 to -0.97 with distance from oriC, because
    ChIP read density tracks DNA copy number along the replication gradient;
  * the UNTAGGED control track, which contains no supercoiling information at
    all, predicts cutting about as well as the GapR ChIP does;
  * feeding the model only raw read density, with no ratios, reproduces 97%
    of the family's gain;
  * feeding it only the ratios -- the part where the copy-number component
    divides out, i.e. the actual supercoiling measurement -- leaves +0.004,
    at the edge of the noise floor.

So the columns are now named for what they are. `eng.sc.abund.*` is
sequencing read density, a measured proxy for local DNA abundance and
whatever else makes a locus over- or under-covered. `eng.sc.torsion.*` is the
normalised, supercoiling-specific residual. Report them separately; do not
call the first one supercoiling.

Columns
-------
eng.sc.abund.<track>.w<width>   mean read density in a window at the cut site
eng.sc.torsion.enrichment.w*    log2(signal / control)
eng.sc.torsion.txn_driven.w*    log2(signal / rifampicin)
eng.sc.torsion.gradient.w*      density downstream minus upstream of the cut
"""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from .. import config
from ..build import genome
from ..io_utils import save_block

PREFIX = "eng.sc"

WINDOWS = (100, 500, 2000, 10000)
PSEUDO = 1e-4  # keeps log2 finite where a track is zero

CACHE = config.INTERIM / "gapr_tracks.npz"


def _parse_wig(path) -> np.ndarray:
    """Read a single-base variableStep wiggle into a dense float32 array.

    These are 4.6 million lines each, so the parsing goes through pandas' C
    engine rather than a Python loop -- seconds instead of minutes.
    """
    n = genome.genome_length()
    arr = np.zeros(n, dtype=np.float32)
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["pos", "value"],
        skiprows=2,               # 'track ...' and 'variableStep ...'
        dtype={"pos": np.int64, "value": np.float32},
        compression="gzip" if str(path).endswith(".gz") else None,
        engine="c",
    )
    pos = df["pos"].to_numpy() - 1
    keep = (pos >= 0) & (pos < n)
    arr[pos[keep]] = df["value"].to_numpy()[keep]
    return arr


def load_tracks(force: bool = False, verbose: bool = True) -> dict[str, np.ndarray]:
    """Mean track per sample role, cached as a single .npz.

    Parsing 4.6 million text lines seven times takes a few minutes, so it is
    done once and cached.
    """
    if CACHE.exists() and not force:
        with np.load(CACHE) as z:
            return {k: z[k] for k in z.files}

    if not config.GAPR_WIG_DIR.exists():
        raise FileNotFoundError(
            f"{config.GAPR_WIG_DIR} not found. Extract GSE152880_RAW.tar into it:\n"
            f"  mkdir -p {config.GAPR_WIG_DIR}\n"
            f"  tar -xf {config.GAPR_WIG_DIR.parent / 'GSE152880_RAW.tar'} "
            f"-C {config.GAPR_WIG_DIR}"
        )

    tracks: dict[str, np.ndarray] = {}
    for role, files in config.GAPR_SAMPLES.items():
        stack = []
        for name in files:
            path = config.GAPR_WIG_DIR / name
            if not path.exists():
                if verbose:
                    print(f"  ! missing {name}, skipping")
                continue
            if verbose:
                print(f"  parsing {name} ...")
            stack.append(_parse_wig(path))
        if stack:
            tracks[role] = np.mean(stack, axis=0).astype(np.float32)

    np.savez_compressed(CACHE, **tracks)
    if verbose:
        print(f"  cached {CACHE}")
    return tracks


def _window_means(track: np.ndarray, centres: np.ndarray, width: int) -> np.ndarray:
    """Mean track value in a window of `width` centred on each position.

    Uses a cumulative sum so all 13,880 guides cost one pass, and wraps at the
    origin because the chromosome is circular.
    """
    n = track.size
    cs = np.concatenate([[0.0], np.cumsum(track, dtype=np.float64)])
    half = width // 2
    starts = (centres - half) % n
    ends = starts + width

    straight = ends <= n
    out = np.empty(centres.shape, dtype=np.float64)
    s, e = starts[straight], ends[straight]
    out[straight] = (cs[e] - cs[s]) / width
    s = starts[~straight]
    out[~straight] = ((cs[n] - cs[s]) + cs[(s + width) % n]) / width
    return out


def build(index: pd.DataFrame, tracks: dict | None = None) -> pd.DataFrame:
    tracks = tracks if tracks is not None else load_tracks()
    n = genome.genome_length()

    # Cas9 cuts 3 bp 5' of the PAM. Centre the windows there rather than on
    # the middle of the protospacer.
    left = index["left"].to_numpy(dtype="float64")
    right = index["right"].to_numpy(dtype="float64")
    strand = index["strand"].to_numpy()
    cut = np.where(strand == "+", right - 3, left + 3)
    valid = np.isfinite(cut)
    centres = np.where(valid, cut, 1).astype(np.int64) - 1  # 0-based
    centres %= n

    out = pd.DataFrame({config.ID_COL: index[config.ID_COL].to_numpy()})

    means: dict[tuple[str, int], np.ndarray] = {}
    for role, track in tracks.items():
        for w in WINDOWS:
            vals = _window_means(track, centres, w)
            means[(role, w)] = vals
            col = f"{PREFIX}.abund.{role}.w{w}"
            out[col] = np.where(valid, vals, np.nan)

    for w in WINDOWS:
        if ("signal", w) in means and ("control", w) in means:
            ratio = np.log2(
                (means[("signal", w)] + PSEUDO) / (means[("control", w)] + PSEUDO)
            )
            out[f"{PREFIX}.torsion.enrichment.w{w}"] = np.where(valid, ratio, np.nan)
        if ("signal", w) in means and ("rifampicin", w) in means:
            ratio = np.log2(
                (means[("signal", w)] + PSEUDO) / (means[("rifampicin", w)] + PSEUDO)
            )
            out[f"{PREFIX}.torsion.txn_driven.w{w}"] = np.where(valid, ratio, np.nan)

    # Torsional asymmetry across the cut site: Cas9 sits between a domain that
    # is being overtwisted and one that is being undertwisted, and which side
    # is which depends on the guide's own strand.
    if "signal" in tracks:
        for w in (500, 2000):
            half = w // 2
            downstream = _window_means(
                tracks["signal"], (centres + half) % n, w
            )
            upstream = _window_means(
                tracks["signal"], (centres - half) % n, w
            )
            grad = np.where(strand == "+", downstream - upstream, upstream - downstream)
            out[f"{PREFIX}.torsion.gradient.w{w}"] = np.where(valid, grad, np.nan)
    return out


def main() -> None:
    from ..io_utils import load_guide_index

    idx = load_guide_index()
    print("Loading GapR-seq tracks ...")
    tracks = load_tracks()
    print(f"  roles available: {sorted(tracks)}")
    block = build(idx, tracks)
    path = save_block(block, "d_supercoiling")
    n_ab = sum(c.startswith(f"{PREFIX}.abund") for c in block.columns)
    n_to = sum(c.startswith(f"{PREFIX}.torsion") for c in block.columns)
    print(f"d_supercoiling: {block.shape[0]:,} rows x {block.shape[1] - 1:,} features "
          f"({n_ab} abundance, {n_to} torsion) -> {path}")


if __name__ == "__main__":
    main()
