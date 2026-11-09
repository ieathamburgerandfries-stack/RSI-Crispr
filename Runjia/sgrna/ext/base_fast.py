"""The 6,169 base columns without a Python loop over columns or guides: the same matrix as `featurise.matrix`, on arrays.

`featurise.matrix` loops over 5,853 indicator columns and 316 tensor columns per call (about 4 ms of fixed cost, so a single guide
costs 4.3 ms) and does a dictionary lookup per guide and column. Here every (unit, position) k-mer is turned into an integer
index once, indicators are an integer comparison, and tensors are a dense-array lookup; windows with a non-ACGT character give
all-zero indicators and NaN tensors, as in the original.

    python -m sgrna.ext.base_fast      # equality against `featurise.matrix` on all curated guides, and timing
"""
from __future__ import annotations

import time

import numpy as np

from sgrna import featurise, qct
from sgrna.decode_v_columns import SPACER_LENGTH, UNITS

from .a_flank_fast import LUT, _dense

_CACHE: dict = {}


def _kmer_code(kmer: str) -> int:
    v = 0
    for ch in kmer:
        v = v * 4 + "ACGT".index(ch)
    return v


def _plan(v_mapping, tables):
    key = (id(v_mapping), id(tables))
    if key in _CACHE:
        return _CACHE[key]
    v_names, q_names = featurise.feature_names(v_mapping, tables)
    by_unit: dict[str, list] = {}
    for j, name in enumerate(v_names):
        spec = v_mapping["mapping"][name]
        by_unit.setdefault(spec["unit"], []).append((j, spec["position"] - 1, _kmer_code(spec["kmer"])))
    v_groups = {u: tuple(np.array(c) for c in zip(*rows)) for u, rows in by_unit.items()}
    q_groups: dict[tuple, list] = {}
    for j, name in enumerate(q_names):
        pos, unit, prop = qct.parse_qct_column(name)
        q_groups.setdefault((unit, prop), []).append((len(v_names) + j, pos - 1))
    q_groups = {k: tuple(np.array(c) for c in zip(*rows)) for k, rows in q_groups.items()}
    dense = {(u, p): _dense(tables[u][p], qct.UNIT_SPAN[u]) for (u, p) in q_groups}
    plan = dict(names=v_names + q_names, n=len(v_names) + len(q_names), v_groups=v_groups, q_groups=q_groups, dense=dense)
    _CACHE[key] = plan
    return plan


def _window_index(codes, k):
    """(n, L-k+1) integer k-mer index at every start; -1 where the window holds a non-ACGT character."""
    n = codes.shape[1] - k + 1
    idx = np.zeros((codes.shape[0], n), np.int64); bad = np.zeros((codes.shape[0], n), bool)
    for j in range(k):
        c = codes[:, j:j + n]
        bad |= c == 255
        idx = idx * 4 + np.where(c == 255, 0, c)
    return np.where(bad, -1, idx)


def matrix_fast(protospacers, v_mapping=None, tables=None):
    v_mapping = v_mapping or featurise.load_v_mapping(); tables = tables or qct.load_tables()
    plan = _plan(v_mapping, tables)
    seqs = [str(s).upper() for s in protospacers]
    codes = LUT[np.frombuffer("".join(seqs).encode("ascii"), np.uint8)].reshape(len(seqs), SPACER_LENGTH)
    M = np.zeros((len(seqs), plan["n"]), np.float32)
    idx_cache = {}

    def widx(k):
        if k not in idx_cache:
            idx_cache[k] = _window_index(codes, k)
        return idx_cache[k]

    for unit, (cols, pos, code) in plan["v_groups"].items():
        M[:, cols] = (widx(UNITS[unit])[:, pos] == code).astype(np.float32)
    for (unit, prop), (cols, pos) in plan["q_groups"].items():
        i = widx(qct.UNIT_SPAN[unit])[:, pos]
        v = plan["dense"][(unit, prop)][np.where(i < 0, 0, i)]
        M[:, cols] = np.where(i < 0, np.nan, v).astype(np.float32)
    return M, plan["names"]


def check(n=None):
    from sgrna.curated_ablation import _base
    _, _, _, protos = _base()
    P = list(protos if n is None else protos[:n])
    vm, tb = featurise.load_v_mapping(), qct.load_tables()
    t = time.perf_counter(); ref, rn = featurise.matrix(P[:3000], v_mapping=vm, tables=tb, verbose=False); t_ref = time.perf_counter() - t
    t = time.perf_counter(); got, gn = matrix_fast(P[:3000], vm, tb); t_fast = time.perf_counter() - t
    assert rn == gn, "column names or order differ"
    full, _ = matrix_fast(P, vm, tb)
    ok = bool((np.isnan(ref) == np.isnan(got)).all()) and float(np.nanmax(np.abs(ref - got))) == 0.0
    print(f"{len(P):,} guides: names equal, values identical on the first 3,000 vs featurise.matrix: {ok}; original {t_ref / 3000 * 1000:.3f} ms/guide, "
          f"fast {t_fast / 3000 * 1000:.4f} ms/guide")
    return ok, full


if __name__ == "__main__":
    check()
