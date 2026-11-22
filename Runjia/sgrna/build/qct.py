"""Quantum chemical tensor (QCT) decoding and extension.

Two jobs.

1. **Decode.** The published feature matrix never stores the guide sequence,
   but it does store per-position QCT descriptors, and those are a pure
   function of the nucleotide at that position. `p{i}monomer.No.electronsraw`
   takes exactly four values across all 13,880 guides -- the valence-electron
   counts of the four bases:

       C4H5N3O   cytosine  16 + 5 + 15 + 6  = 42
       C5H6N2O2  thymine   20 + 6 + 10 + 12 = 48
       C5H5N5    adenine   20 + 5 + 25      = 50
       C5H5N5O   guanine   20 + 5 + 25 + 6  = 56

   so the 20 nt protospacer can be read straight back out of the matrix. Every
   decoded guide is then confirmed by finding it in the reference genome with
   an NGG immediately 3' of it, which is the real check that the mapping is
   right.

2. **Extend.** Once the sequence is known, every QCT column can be inverted
   into a lookup table: k-mer -> descriptor value, learned from the matrix
   itself. That means the *same* quantum parameterisation used in the paper
   can be applied to sequence the paper never encoded -- the flanking DNA, the
   PAM, the nucleotides downstream of it. No new quantum chemistry, no new
   assumptions, just the published tensors read off at new positions.

The k-mer span of each molecular unit follows the column counts in the matrix
(20 monomer positions, 19 dimer, 18 trimer, 17 tetramer over a 20 nt window).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config

ELECTRONS_TO_BASE = {42.0: "C", 48.0: "T", 50.0: "A", 56.0: "G"}
BASES = ("A", "C", "G", "T")

# Molecular unit -> how many consecutive nucleotides it spans.
UNIT_SPAN = {"monomer": 1, "basepair": 1, "dimer": 2, "trimer": 3, "tetramer": 4}

_COL_RE = re.compile(r"^p(\d+)([A-Za-z]+)\.(.+)$")


# --------------------------------------------------------------------------
# Column bookkeeping
# --------------------------------------------------------------------------


def parse_qct_column(col: str) -> tuple[int, str, str] | None:
    """'p18dimer.Hbond.stackingraw' -> (18, 'dimer', 'Hbond.stackingraw')."""
    m = _COL_RE.match(col)
    if not m:
        return None
    pos, unit, prop = int(m.group(1)), m.group(2).lower(), m.group(3)
    if unit not in UNIT_SPAN:
        return None
    return pos, unit, prop


def qct_columns(header: list[str]) -> dict[tuple[str, str], dict[int, str]]:
    """Group QCT columns as {(unit, property): {position: column name}}."""
    out: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    for col in header:
        parsed = parse_qct_column(col)
        if parsed is None:
            continue
        pos, unit, prop = parsed
        out[(unit, prop)][pos] = col
    return dict(out)


def read_header(path=None, header_row=None) -> list[str]:
    path = path or config.BASE_MATRIX
    header_row = config.BASE_MATRIX_HEADER_ROW if header_row is None else header_row
    return list(
        pd.read_csv(path, header=header_row, nrows=0, low_memory=False).columns
    )


# --------------------------------------------------------------------------
# Cached slice of the base matrix
# --------------------------------------------------------------------------

QCT_CACHE = config.INTERIM / "qct_columns.csv"


def load_qct_frame(force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """ID, target and all 316 QCT columns, cached.

    The published matrix is 198 MB across 6,234 columns, and pandas has to
    tokenise all of it however few columns you ask for -- about a minute a
    pass. Everything that needs the quantum columns goes through this cache
    instead, so that cost is paid once.
    """
    if QCT_CACHE.exists() and not force:
        return pd.read_csv(QCT_CACHE, low_memory=False)

    if verbose:
        print(f"  caching QCT columns from {config.BASE_MATRIX.name} (one-off, ~1 min)")
    header = read_header()
    qcols = [c for c in header if parse_qct_column(c) is not None]
    df = pd.read_csv(
        config.BASE_MATRIX,
        header=config.BASE_MATRIX_HEADER_ROW,
        usecols=[config.ID_COL, config.TARGET_COL] + qcols,
        low_memory=False,
    )
    df = df[[config.ID_COL, config.TARGET_COL] + qcols]
    df.to_csv(QCT_CACHE, index=False)
    if verbose:
        print(f"  wrote {QCT_CACHE} ({df.shape[0]:,} x {df.shape[1]:,})")
    return df


# --------------------------------------------------------------------------
# 1. Decoding
# --------------------------------------------------------------------------


def decode_protospacers(frame: pd.DataFrame | None = None,
                        n_positions: int = 20) -> pd.DataFrame:
    """Read the electron-count columns and reconstruct each 20 nt protospacer.

    Returns a frame with sgRNAID, cut.score and `protospacer`. Rows whose
    descriptors are missing get 'N' at that position and are flagged.
    """
    df = load_qct_frame() if frame is None else frame
    mat = df[[f"p{i}monomer.No.electronsraw" for i in range(1, n_positions + 1)]]
    mat = mat.to_numpy(dtype=float)

    unknown = sorted(set(np.unique(mat[~np.isnan(mat)])) - set(ELECTRONS_TO_BASE))
    if unknown:
        raise ValueError(
            "Unexpected electron counts in the monomer columns: "
            f"{unknown}. The decoding table only knows {sorted(ELECTRONS_TO_BASE)}."
        )

    seqs = ["".join(ELECTRONS_TO_BASE.get(v, "N") for v in row) for row in mat]
    out = df[[config.ID_COL, config.TARGET_COL]].copy()
    out["protospacer"] = seqs
    out["decoded_ok"] = ~out["protospacer"].str.contains("N")
    return out


def parse_sgrna_id(sgrna_id: str) -> dict:
    """'aaeAb3241_107_Cas9' -> gene aaeA, b-number b3241, offset 107."""
    m = re.match(r"^(?P<gene>.*?)(?P<bnum>[bB]\d+)_(?P<offset>\d+)_(?P<nuclease>.+)$",
                 str(sgrna_id))
    if not m:
        return dict(gene_name=None, b_number=None, gene_offset=None, nuclease=None)
    return dict(
        gene_name=m.group("gene") or None,
        b_number=m.group("bnum").lower(),
        gene_offset=int(m.group("offset")),
        nuclease=m.group("nuclease"),
    )


# --------------------------------------------------------------------------
# 2. Lookup tables
# --------------------------------------------------------------------------


def learn_lookup_tables(
    protospacers: pd.Series,
    frame: pd.DataFrame | None = None,
) -> dict:
    """Invert the matrix's QCT columns into {unit: {property: {kmer: value}}}.

    Consistency is checked, not assumed: if one k-mer ever maps to two
    different values for the same descriptor the function raises, because that
    would mean the descriptor is not a pure function of sequence and the whole
    extension idea is unsound.
    """
    df = load_qct_frame() if frame is None else frame
    groups = qct_columns(list(df.columns))
    seqs = protospacers.to_numpy()[: len(df)]

    tables: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    conflicts: list[str] = []

    for (unit, prop), pos_map in groups.items():
        span = UNIT_SPAN[unit]
        acc: dict[str, set] = defaultdict(set)
        for pos, col in pos_map.items():
            values = df[col].to_numpy(dtype=float)
            start = pos - 1
            for seq, val in zip(seqs, values):
                if np.isnan(val):
                    continue
                kmer = seq[start : start + span]
                if len(kmer) < span or "N" in kmer:
                    continue
                acc[kmer].add(round(float(val), 6))
        table = {}
        for kmer, vals in acc.items():
            if len(vals) > 1:
                conflicts.append(f"{unit}.{prop}[{kmer}] -> {sorted(vals)}")
            table[kmer] = float(sorted(vals)[0])
        tables[unit][prop] = table

    if conflicts:
        raise ValueError(
            "QCT descriptors are not a pure function of k-mer identity:\n  "
            + "\n  ".join(conflicts[:10])
        )
    return {k: dict(v) for k, v in tables.items()}


def table_coverage(tables: dict) -> pd.DataFrame:
    """How much of each k-mer space the learned tables cover."""
    rows = []
    for unit, props in tables.items():
        span = UNIT_SPAN[unit]
        total = 4 ** span
        for prop, table in props.items():
            rows.append(
                dict(
                    unit=unit,
                    property=prop,
                    span=span,
                    observed=len(table),
                    possible=total,
                    coverage=len(table) / total,
                )
            )
    return pd.DataFrame(rows).sort_values(["unit", "property"]).reset_index(drop=True)


def save_tables(tables: dict, path=None) -> Path:
    path = Path(path or config.QCT_TABLES)
    path.write_text(json.dumps(tables, indent=1, sort_keys=True))
    return path


def load_tables(path=None) -> dict:
    path = Path(path or config.QCT_TABLES)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m sgrna.build.build_guide_index` first."
        )
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# 3. Applying tables to new sequence
# --------------------------------------------------------------------------


def profile(seq: str, unit: str, prop: str, tables: dict) -> np.ndarray:
    """Descriptor value at every position of `seq` for one (unit, property).

    Returns an array of length len(seq) - span + 1. Unknown k-mers (an N, or a
    tetramer the 13,880 guides never contained) come back as NaN, which the
    downstream models already handle by median imputation.
    """
    span = UNIT_SPAN[unit]
    table = tables[unit][prop]
    n = len(seq) - span + 1
    if n <= 0:
        return np.zeros(0)
    out = np.full(n, np.nan)
    for i in range(n):
        out[i] = table.get(seq[i : i + span], np.nan)
    return out


def summarise(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Mean / sd / min / max / sum of a descriptor profile, NaN-safe."""
    if values.size == 0 or np.all(np.isnan(values)):
        return {f"{prefix}.{s}": np.nan for s in ("mean", "sd", "min", "max", "sum")}
    return {
        f"{prefix}.mean": float(np.nanmean(values)),
        f"{prefix}.sd": float(np.nanstd(values)),
        f"{prefix}.min": float(np.nanmin(values)),
        f"{prefix}.max": float(np.nanmax(values)),
        f"{prefix}.sum": float(np.nansum(values)),
    }
