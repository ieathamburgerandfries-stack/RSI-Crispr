"""Family A without a Python loop over nucleotides: the same 348 columns, computed on arrays.

`features.a_flank.build` walks every position of eight 1,000-nt segments per guide in Python (`qct.profile`), which costs about
1.33 ms per guide and dominates the end-to-end cost of scoring a guide (the model itself is tens of microseconds). This module
computes the identical columns with numpy: sequences become 2-bit codes, every k-mer table becomes a dense array indexed by the
k-mer's base-4 value, window means and standard deviations are masked sums over slices, and the longest homopolymer run inside a
prefix or suffix window is a running maximum of run lengths.

Rows that do not fit the fast path -- fewer than 1,000 nt of flank on either side, anything but A/C/G/T in the pieces that are used,
or no `located` flag -- are handed to the original builder, so the output is the original's for every row. `check()` compares the two.

    python -m sgrna.ext.a_flank_fast          # equality check and timing on 300 curated guides
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from sgrna import config, qct
from sgrna.features import a_flank as A

W = max(A.SUMMARY_WINDOWS)
LUT = np.full(256, 255, np.uint8)
for _i, _c in enumerate("ACGT"):
    LUT[ord(_c)] = _i


def _codes(strings, length, tail):
    """(n, length) uint8 codes of the last (tail=True) or first `length` characters of each string; 255 marks a non-ACGT character."""
    raw = np.frombuffer("".join(s[-length:] if tail else s[:length] for s in strings).encode("ascii"), np.uint8)
    return LUT[raw].reshape(len(strings), length)


def _dense(table: dict, k: int) -> np.ndarray:
    out = np.full(4 ** k, np.nan)
    for kmer, v in table.items():
        idx = 0
        for ch in kmer:
            idx = idx * 4 + "ACGT".index(ch)
        out[idx] = v
    return out


def _kmer_index(codes: np.ndarray, k: int) -> np.ndarray:
    n = codes.shape[1] - k + 1
    idx = np.zeros((codes.shape[0], n), np.int64)
    for j in range(k):
        idx = idx * 4 + codes[:, j:j + n]
    return idx


def _run_cummax(codes: np.ndarray) -> np.ndarray:
    """Running maximum of the length of the homopolymer run ending at each position."""
    n, L = codes.shape
    pos = np.arange(L)[None, :]
    new = np.ones((n, L), bool)
    new[:, 1:] = codes[:, 1:] != codes[:, :-1]
    start = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    return np.maximum.accumulate(pos - start + 1, axis=1)


def _window_stats(prof: np.ndarray):
    """NaN-aware mean and (population) standard deviation along axis 1, NaN where nothing is finite."""
    ok = np.isfinite(prof)
    cnt = ok.sum(1)
    v = np.where(ok, prof, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = v.sum(1) / cnt
        sd = np.sqrt((np.where(ok, (prof - mean[:, None]) ** 2, 0.0)).sum(1) / cnt)
    mean[cnt == 0] = np.nan; sd[cnt == 0] = np.nan
    return mean, sd


_CACHE: dict = {}


def _plan(tables):
    """Everything that depends only on the lookup tables: dense arrays, column names, offsets (computed once)."""
    key = id(tables)
    if key in _CACHE:
        return _CACHE[key]
    labels = A._positions()
    pairs = [(u, p) for u, ps in tables.items() for p in ps if u in A.PER_POSITION_UNITS]
    names_seq = [f"{A.PREFIX}.seq.{lab}.{b}" for lab in labels for b in A.BASES]
    names_qct = [f"{A.PREFIX}.qct.{u}.{p.replace('.', '').replace('raw', '')}.{lab}" for u, p in pairs for lab in labels]
    names_win = []
    for side in ("up", "dn"):
        for w in A.SUMMARY_WINDOWS:
            names_win += [f"{A.PREFIX}.comp.{side}{w}.{k}" for k in ("gc", "purine", "maxrun")]
            for u, p in A.SUMMARY_SERIES:
                s = p.replace(".", "").replace("raw", "")
                names_win += [f"{A.PREFIX}.win.{side}{w}.{u}.{s}.mean", f"{A.PREFIX}.win.{side}{w}.{u}.{s}.sd"]
    ps = A.N_UPSTREAM + 4
    offs = {}
    for i in range(A.N_UPSTREAM, 0, -1):
        offs[f"u{i}"] = ps - i
    offs["pamN"] = ps + 20
    for i in range(1, A.N_DOWNSTREAM + 1):
        offs[f"d{i}"] = ps + 20 + 3 + (i - 1)
    plan = dict(labels=labels, pairs=pairs, names=names_seq + names_qct + names_win, n_seq=len(names_seq), n_qct=len(names_qct),
                P=np.array([offs[lab] for lab in labels]),
                dense_pp={(u, p): _dense(tables[u][p], qct.UNIT_SPAN[u]) for u, p in pairs},
                dense_sum={(u, p): _dense(tables[u][p], qct.UNIT_SPAN[u]) for u, p in A.SUMMARY_SERIES})
    _CACHE[key] = plan
    return plan


def build_matrix(index: pd.DataFrame, tables: dict | None = None):
    """(n, 348) float array in the original column order, and the names (without the id column)."""
    tables = tables or qct.load_tables()
    plan = _plan(tables)
    index = index.reset_index(drop=True)
    n_all = len(index)
    M = np.full((n_all, len(plan["names"])), np.nan)
    located = index["located"].fillna(False).astype(bool).to_numpy() if "located" in index else np.zeros(n_all, bool)
    ups, dns = index["upstream"].astype(str).tolist(), index["downstream"].astype(str).tolist()
    protos, pams = index["protospacer"].astype(str).tolist(), index["pam"].astype(str).tolist()
    ok = located.copy()
    for i in np.where(ok)[0]:
        if len(ups[i]) < W or len(dns[i]) < W or len(protos[i]) != 20 or len(pams[i]) != 3:
            ok[i] = False
    fast = np.where(ok)[0]
    if len(fast):
        up = _codes([ups[i] for i in fast], W, True); dn = _codes([dns[i] for i in fast], W, False)
        pr = _codes([protos[i] for i in fast], 20, False); pm = _codes([pams[i] for i in fast], 3, False)
        bad = (up == 255).any(1) | (dn == 255).any(1) | (pr == 255).any(1) | (pm == 255).any(1)
        if bad.any():                                    # a stray character anywhere goes to the original builder
            ok[fast[bad]] = False
            keep = ~bad
            up, dn, pr, pm, fast = up[keep], dn[keep], pr[keep], pm[keep], fast[keep]
    if len(fast):
        ext = np.concatenate([up[:, -(A.N_UPSTREAM + 4):], pr, pm, dn[:, :A.N_DOWNSTREAM + 4]], axis=1)
        P, nl = plan["P"], len(plan["labels"])
        block = np.empty((len(fast), len(plan["names"])))
        block[:, :plan["n_seq"]] = (ext[:, P][:, :, None] == np.arange(4)).reshape(len(fast), -1)
        c = plan["n_seq"]
        for (u, p), dense in plan["dense_pp"].items():
            k = qct.UNIT_SPAN[u]
            idx = np.zeros((len(fast), nl), np.int64)
            for j in range(k):
                idx = idx * 4 + ext[:, P + j]
            block[:, c:c + nl] = dense[idx]; c += nl
        for side, seg in (("up", up), ("dn", dn)):
            rev = seg[:, ::-1] if side == "up" else seg
            runs = _run_cummax(rev)
            gc = np.cumsum((rev == 1) | (rev == 2), axis=1); pur = np.cumsum((rev == 0) | (rev == 2), axis=1)
            profs = [dense[_kmer_index(seg, qct.UNIT_SPAN[u])] for (u, p), dense in plan["dense_sum"].items()]
            for w in A.SUMMARY_WINDOWS:
                cols = [gc[:, w - 1] / w, pur[:, w - 1] / w, runs[:, w - 1] / w]
                for prof in profs:
                    part = prof[:, -w:] if side == "up" else prof[:, :w]
                    mean, sd = _window_stats(part)
                    cols += [mean, sd]
                block[:, c:c + len(cols)] = np.stack(cols, axis=1); c += len(cols)
        M[fast] = block
    slow = np.where(~ok)[0]
    if len(slow):
        ref = A.build(index.iloc[slow], tables).reset_index(drop=True)
        for j, name in enumerate(plan["names"]):
            if name in ref.columns:
                M[slow, j] = ref[name].to_numpy(float)
    return M, plan["names"]


def build_fast(index: pd.DataFrame, tables: dict | None = None) -> pd.DataFrame:
    M, names = build_matrix(index, tables)
    out = pd.DataFrame(M, columns=names)
    out.insert(0, config.ID_COL, index[config.ID_COL].to_numpy())
    return out


def check(n=300, seed=0):
    from sgrna.curated_ablation import _base
    _, _, _, protos = _base()
    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv").set_index(config.ID_COL)
    sub = idx.reindex(list(protos[:n])).reset_index(); sub["located"] = True
    tables = qct.load_tables()
    A.build(sub.iloc[:3], tables); build_fast(sub.iloc[:3], tables)                        # warm
    t = time.perf_counter(); ref = A.build(sub, tables); t_ref = time.perf_counter() - t
    t = time.perf_counter(); fast = build_fast(sub, tables); t_fast = time.perf_counter() - t
    assert list(ref.columns) == list(fast.columns), "column names or order differ"
    r, f = ref.drop(columns=config.ID_COL).to_numpy(float), fast.drop(columns=config.ID_COL).to_numpy(float)
    same_nan = (np.isnan(r) == np.isnan(f)).all()
    err = float(np.nanmax(np.abs(r - f) / (np.abs(r) + 1e-12)))
    print(f"{n} guides, {ref.shape[1] - 1} columns: same columns/order {list(ref.columns) == list(fast.columns)}, same NaN pattern {same_nan}, "
          f"max relative difference {err:.2e}; original {t_ref / n * 1000:.3f} ms/guide, fast {t_fast / n * 1000:.3f} ms/guide "
          f"({t_ref / t_fast:.0f}x)")
    return err, same_nan, t_ref / n * 1000, t_fast / n * 1000


if __name__ == "__main__":
    check()
