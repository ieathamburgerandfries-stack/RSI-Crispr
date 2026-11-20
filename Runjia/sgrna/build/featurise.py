"""Generate the published feature representation for guides outside the matrix.

Why this exists
---------------
The project was capped at 13,880 rows because that is how many guides Noshay
et al. published features for. Two pieces of reverse engineering lift the cap:

  * `qct.learn_lookup_tables` recovers the 316 quantum tensor columns as exact
    k-mer -> value tables (all 4/16/64/256 k-mers present, no conflicts);
  * `decode_v_columns` identifies 5,853 of the 5,887 `V####` columns as
    (position, k-mer) one-hot indicators, verified at 100% on 1.17 million
    rebuilt values.

Together that is 6,169 of 6,232 columns -- 99.0% of the published matrix --
regenerable from a 20 nt protospacer alone.

What is deliberately left out
-----------------------------
The ~31 hand-named columns (`AAsgRNA.raw`, `sgRNA.gcsgRNA.raw`, `PAM.C0`,
`pam.distance0`, ...) are **not** pure functions of the protospacer: the base
counts disagree with a direct count by up to 13, `sgRNA.temp` takes exactly 14
distinct values spaced 2.05 apart, and the PAM columns take fractions like
0.25 and 0.4. They have been discretised or aggregated somewhere upstream in a
way the published matrix does not document. Rather than guess, this module
omits them, along with the 34 V columns that could not be identified.

That matters for how you use the output: **featurise everything the same way.**
Do not mix rows from the published CSV with rows built here -- rebuild the old
rows through this module too, so every row has the same 6,169 columns computed
by the same code. `build_expanded_dataset` does exactly that.

    from sgrna.build import featurise
    X, names = featurise.matrix(["ACGT...", "TTGC..."])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from . import qct
from .decode_v_columns import UNITS, SPACER_LENGTH
from .decode_v_columns import load as load_v_mapping


def feature_names(v_mapping: dict | None = None, tables: dict | None = None):
    """Ordered column names: the identified V columns, then the QCT columns."""
    v_mapping = v_mapping or load_v_mapping()
    tables = tables or qct.load_tables()

    v_names = sorted(v_mapping["mapping"])
    qct_names = []
    for unit, props in sorted(tables.items()):
        span = qct.UNIT_SPAN[unit]
        for prop in sorted(props):
            for pos in range(1, SPACER_LENGTH - span + 2):
                qct_names.append(f"p{pos}{unit}.{prop}")
    return v_names, qct_names


def matrix(protospacers, v_mapping: dict | None = None,
           tables: dict | None = None, verbose: bool = True):
    """(n, 6169) float32 matrix for a list of 20 nt protospacers."""
    v_mapping = v_mapping or load_v_mapping()
    tables = tables or qct.load_tables()
    v_names, qct_names = feature_names(v_mapping, tables)

    seqs = np.asarray([str(s).upper() for s in protospacers])
    n = len(seqs)

    # ---- V columns: (position, k-mer) indicators ------------------------
    V = np.zeros((n, len(v_names)), dtype=np.float32)
    windows: dict[tuple[str, int], np.ndarray] = {}
    for j, name in enumerate(v_names):
        spec = v_mapping["mapping"][name]
        k = UNITS[spec["unit"]]
        pos = spec["position"]
        key = (spec["unit"], pos)
        if key not in windows:
            windows[key] = np.array([s[pos - 1 : pos - 1 + k] for s in seqs])
        V[:, j] = (windows[key] == spec["kmer"]).astype(np.float32)
    if verbose:
        print(f"  V columns: {V.shape[1]:,}")

    # ---- QCT columns: k-mer -> published tensor value -------------------
    Q = np.full((n, len(qct_names)), np.nan, dtype=np.float32)
    for j, name in enumerate(qct_names):
        pos, unit, prop = qct.parse_qct_column(name)
        span = qct.UNIT_SPAN[unit]
        table = tables[unit][prop]
        kmers = windows.get((unit, pos))
        if kmers is None:
            kmers = np.array([s[pos - 1 : pos - 1 + span] for s in seqs])
        Q[:, j] = [table.get(km, np.nan) for km in kmers]
    if verbose:
        print(f"  QCT columns: {Q.shape[1]:,}")

    return np.hstack([V, Q]), v_names + qct_names


def verify_against_published(n: int = 500, verbose: bool = True) -> dict:
    """Rebuild the published matrix for real guides and compare, cell by cell."""
    from .build_matrix import build_cache
    from ..io_utils import load_guide_index

    X, meta = build_cache(verbose=False)
    names = meta["feature_names"]
    ids = meta["ids"]
    idx = load_guide_index().set_index(config.ID_COL).reindex(ids)
    seqs = idx["protospacer"].fillna("N" * SPACER_LENGTH).to_numpy()[:n]

    rebuilt, rb_names = matrix(seqs, verbose=False)
    col_of = {nm: i for i, nm in enumerate(names)}
    Xv = np.asarray(X)[:n]

    shared = [c for c in rb_names if c in col_of]
    a = rebuilt[:, [rb_names.index(c) for c in shared]]
    b = Xv[:, [col_of[c] for c in shared]]
    both = ~(np.isnan(a) | np.isnan(b))
    match = float(np.isclose(a[both], b[both], atol=1e-6).mean())
    if verbose:
        print(f"  rebuilt {len(shared):,} columns x {n:,} guides: "
              f"{match:.6%} of values identical to the published matrix")
    return dict(columns=len(shared), guides=n, agreement=match)
