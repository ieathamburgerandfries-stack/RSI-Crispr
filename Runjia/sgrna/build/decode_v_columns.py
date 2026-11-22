"""Work out what the 5,888 `V####` columns are, so new guides can be featurised.

The problem
-----------
`qct.py` recovers the quantum tensors as exact k-mer lookup tables, which means
those 316 columns can be regenerated for any sequence. But they are only 5% of
the published matrix. The other 5,888 columns are named `V10.x`, `V1076.y`,
`V3878` and so on -- opaque labels with no documentation -- and without them a
guide that is not already in the matrix cannot be scored by a model trained on
it. That caps the project at 13,880 rows forever.

The observation
---------------
Every V column is binary, and 80 + 304 + 1152 + 4352 = 5,888 is exactly the
number of (position, k-mer) indicator variables you get from one-hot encoding
a 20 nt sequence at monomer, dimer, trimer and tetramer resolution. That is
suspiciously precise. So each V column is probably the indicator "position i
holds k-mer s", and since the protospacers are now decoded, every one of those
indicators can be constructed and matched against the real columns.

The method is a dictionary lookup on bit patterns: build the indicator vector
for all 5,888 candidates across the 13,880 guides, hash it, hash every V
column, and pair them up. A match on 13,880 bits is not a coincidence.

Ambiguity is handled explicitly. A k-mer that never occurs at a position gives
an all-zero column, and several candidates can share that pattern; those are
reported rather than guessed at.

What it unlocks
---------------
With this mapping plus the QCT tables, `featurise.py` can build the complete
6,232-column published representation for an arbitrary protospacer. That turns
"13,880 guides with features" into "any bacterial Cas9 screen with features" --
the crisprHAL curated set is 33,567 guides, of which 24,950 are not in this
matrix.

    python -m sgrna.build.decode_v_columns
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from .. import config
from . import qct
from .build_matrix import build_cache
from ..io_utils import load_guide_index

MAPPING_PATH = config.INTERIM / "v_column_mapping.json"

# k-mer width per molecular unit, and how many positions a 20 nt window has.
UNITS = {"monomer": 1, "dimer": 2, "trimer": 3, "tetramer": 4}
SPACER_LENGTH = 20
BASES = "ACGT"


def _hash(vec: np.ndarray) -> str:
    return hashlib.sha1(np.packbits(vec.astype(bool)).tobytes()).hexdigest()


def _kmers(k: int) -> list[str]:
    out = [""]
    for _ in range(k):
        out = [p + b for p in out for b in BASES]
    return out


def build_mapping(verbose: bool = True) -> dict:
    """Match every V column to the (unit, position, k-mer) it encodes."""
    X, meta = build_cache(verbose=verbose)
    names = meta["feature_names"]
    ids = meta["ids"]

    idx = load_guide_index().set_index(config.ID_COL).reindex(ids)
    seqs = idx["protospacer"].fillna("N" * SPACER_LENGTH).to_numpy()

    v_cols = [i for i, n in enumerate(names) if str(n).startswith("V")]
    if verbose:
        print(f"  {len(v_cols):,} V columns to identify, {len(seqs):,} guides")

    # ---- candidate indicators -------------------------------------------
    candidates: dict[str, list[tuple]] = defaultdict(list)
    n_candidates = 0
    for unit, k in UNITS.items():
        for pos in range(1, SPACER_LENGTH - k + 2):
            window = np.array([s[pos - 1 : pos - 1 + k] for s in seqs])
            for kmer in _kmers(k):
                vec = (window == kmer)
                candidates[_hash(vec)].append((unit, pos, kmer, int(vec.sum())))
                n_candidates += 1
    if verbose:
        print(f"  {n_candidates:,} candidate indicators, "
              f"{len(candidates):,} distinct patterns")

    # ---- match ----------------------------------------------------------
    mapping: dict[str, dict] = {}
    unmatched: list[str] = []
    ambiguous: list[str] = []

    Xv = np.asarray(X)
    for i in v_cols:
        col = Xv[:, i]
        binary = np.nan_to_num(col, nan=0.0) > 0.5
        # A non-binary column cannot be a one-hot indicator at all.
        if not np.isin(np.unique(np.nan_to_num(col, nan=0.0)), [0.0, 1.0]).all():
            unmatched.append(names[i])
            continue
        hits = candidates.get(_hash(binary), [])
        if not hits:
            unmatched.append(names[i])
        elif len(hits) > 1:
            ambiguous.append(names[i])
            unit, pos, kmer, n = hits[0]
            mapping[names[i]] = dict(unit=unit, position=pos, kmer=kmer,
                                     count=n, ambiguous=True,
                                     alternatives=len(hits))
        else:
            unit, pos, kmer, n = hits[0]
            mapping[names[i]] = dict(unit=unit, position=pos, kmer=kmer,
                                     count=n, ambiguous=False)

    if verbose:
        print(f"  identified {len(mapping):,}   ambiguous {len(ambiguous):,}   "
              f"unmatched {len(unmatched):,}")
        if unmatched[:5]:
            print(f"  first unmatched: {unmatched[:5]}")
        by_unit = pd.Series(
            [m["unit"] for m in mapping.values()]
        ).value_counts().to_dict()
        print(f"  by molecular unit: {by_unit}")

    return dict(mapping=mapping, unmatched=unmatched, ambiguous=ambiguous,
                n_v_columns=len(v_cols))


def save(result: dict, path=None):
    path = path or MAPPING_PATH
    path.write_text(json.dumps(result))
    return path


def load(path=None) -> dict:
    path = path or MAPPING_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m sgrna.build.decode_v_columns` first."
        )
    return json.loads(path.read_text())


def verify(result: dict, n: int = 200, verbose: bool = True) -> float:
    """Rebuild the V columns for a sample of guides and score the match."""
    X, meta = build_cache(verbose=False)
    names = meta["feature_names"]
    ids = meta["ids"]
    idx = load_guide_index().set_index(config.ID_COL).reindex(ids)
    seqs = idx["protospacer"].fillna("N" * SPACER_LENGTH).to_numpy()[:n]

    col_of = {nm: i for i, nm in enumerate(names)}
    Xv = np.asarray(X)
    agree = total = 0
    for name, spec in result["mapping"].items():
        k = UNITS[spec["unit"]]
        p = spec["position"]
        rebuilt = np.array([s[p - 1 : p - 1 + k] == spec["kmer"] for s in seqs])
        actual = np.nan_to_num(Xv[:n, col_of[name]], nan=0.0) > 0.5
        agree += int((rebuilt == actual).sum())
        total += len(seqs)
    score = agree / total if total else 0.0
    if verbose:
        print(f"  verification: {score:.6%} of {total:,} rebuilt values match")
    return score


def main() -> int:
    print("Identifying the V columns ...")
    result = build_mapping()
    verify(result)
    path = save(result)
    print(f"\nWrote {path}")
    if result["unmatched"]:
        print(f"{len(result['unmatched'])} columns could not be identified; they "
              "are listed in the mapping file and will be left at their median "
              "when featurising new guides.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
