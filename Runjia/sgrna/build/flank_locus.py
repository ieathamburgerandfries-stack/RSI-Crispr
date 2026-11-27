"""Family A from a locus instead of from a guide: the window block becomes a table lookup.

Why
---
`a_flank_fast` removed the Python loop but not the work. Each guide still carries its own two 1,000-nt flanks, and for each of
them the four summary tensor series are re-indexed and re-averaged: eight k-mer scans and thirty-two window reductions per guide.
That is 0.068 ms of the 0.093 ms it costs to score one guide end to end -- three quarters of the bill, and it is paid again for
every guide, even though guides in the same gene read almost the same DNA. A gene of 1 kb carries of order 150 NGG sites, so each
base of its flanks is scanned about 150 times.

What this module does instead
-----------------------------
Scan each stretch of chromosome once and store cumulative tracks along it, then answer every guide inside that stretch by
differencing them:

    gc, purine        cumulative base counts                      -> a window count is one subtraction
    tensor mean, sd   cumulative sum and sum of squares, centred   -> mean and population sd from two subtractions
    maxrun            maximal homopolymer runs + a sparse table     -> longest run inside a window in O(1) per guide
    one-hot, per-position tensors   direct gathers at 26 offsets around the cut

Every column is the same column `features.a_flank` emits, in the same order, to within floating-point reassociation: `check()`
compares the two on real guides and reports the largest relative difference.

The longest-run column is the only one that does not decompose into a cumulative sum, because the longest run inside a window is
not a sum of anything. It is still O(1): the maximal runs of the region are listed once, and the longest run inside a window is
the better of the two truncated boundary runs and the longest run strictly between them, which a sparse table over run lengths
answers with two lookups. A window of w nucleotides contains at most w runs, so the table needs only ceil(log2(w)) + 1 levels.

Coordinates
-----------
Guides are given as plus-strand 0-based protospacer bounds (`left0`, `right0 = left0 + 19`) and a strand, which is what
`build.genome.locate_protospacer` already returns (its 1-based `left` minus one). Nothing here reads a stored flanking string, so
this is also the path a design tool takes: coordinates in, columns out, no per-guide sequence extraction.

    ./run_ext.sh sgrna.build.flank_locus            # equality against a_flank_fast, then timings
    ./run_ext.sh sgrna.build.flank_locus --bench    # timings only
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from sgrna import config
from sgrna.build import qct
from sgrna.features import a_flank as A

from . import a_flank_fast as AF
from ..common import OUT

W = max(A.SUMMARY_WINDOWS)              # 1000
SPANS = tuple(sorted({qct.UNIT_SPAN[u] for u, _ in A.SUMMARY_SERIES} |
                     {qct.UNIT_SPAN[u] for u in A.PER_POSITION_UNITS}))
MARGIN = W + 32                         # plus-strand reach of the widest window, with room for the k-mer tails
BLOCK = 100_000                         # a locus is at most this wide; wider sets of guides are cut into several
OUTCSV = OUT / "flank_locus_bench.csv"

# floor(log2(i)) for i up to a window's worth of runs, built by doubling so it is exact at the powers of two.
_LOG2 = np.zeros(W + 2, np.int64)
for _i in range(2, W + 2):
    _LOG2[_i] = _LOG2[_i >> 1] + 1


def _rc_perm(span: int) -> np.ndarray:
    """Permutation p with dense[p][i] = table[revcomp(kmer i)] -- the same lookup read on the other strand."""
    i = np.arange(4 ** span)
    out = np.zeros_like(i)
    for j in range(span):
        out += (3 - ((i >> (2 * (span - 1 - j))) & 3)) * (4 ** j)
    return out


def _series_plan(tables: dict) -> dict:
    """Dense lookup arrays for both strands, and the centring constant, for each series used by the window summaries."""
    plan = {}
    for unit, prop in A.SUMMARY_SERIES:
        span = qct.UNIT_SPAN[unit]
        dense = AF._dense(tables[unit][prop], span)
        plan[(unit, prop)] = dict(span=span, pos=dense, neg=dense[_rc_perm(span)],
                                  mu=float(np.nanmean(dense)))
    for unit, prop in [(u, p) for u, ps in tables.items() for p in ps if u in A.PER_POSITION_UNITS]:
        span = qct.UNIT_SPAN[unit]
        dense = AF._dense(tables[unit][prop], span)
        plan[("pp", unit, prop)] = dict(span=span, pos=dense, neg=dense[_rc_perm(span)])
    return plan


# --------------------------------------------------------------------------
# One stretch of chromosome
# --------------------------------------------------------------------------


class Locus:
    """Cumulative tracks over one stretch of a chromosome. Build once, then every guide inside it is a handful of lookups.

    `codes` holds the stretch as 0-3 base codes (255 for anything else) and `origin` is the plus-strand 0-based coordinate of
    `codes[0]`. Strand-specific tracks are built on first use, so a set of guides that is all one strand pays for one.
    """

    def __init__(self, codes: np.ndarray, origin: int, tables: dict | None = None, plan: dict | None = None):
        self.codes = np.ascontiguousarray(codes, np.uint8)
        self.origin = int(origin)
        self.n = len(self.codes)
        self.tables = tables or qct.load_tables()
        self.plan = plan if plan is not None else _series_plan(self.tables)
        c = self.codes
        good = c < 4
        safe = np.where(good, c, 0)

        # base counts, 1 longer than the stretch so a window is cum[hi] - cum[lo]
        self._gc = np.concatenate([[0], np.cumsum((safe == 1) | (safe == 2))]).astype(np.int64)
        # purines and pyrimidines are counted from the real codes: an N (or any non-ACGT base) is neither, on either strand
        self._pur = np.concatenate([[0], np.cumsum((c == 0) | (c == 2))]).astype(np.int64)
        self._pyr = np.concatenate([[0], np.cumsum((c == 1) | (c == 3))]).astype(np.int64)

        # k-mer indices per span, with the positions whose k-mer is not all-ACGT marked
        self._kidx, self._kok = {}, {}
        for span in SPANS:
            m = self.n - span + 1
            idx = np.zeros(m, np.int64)
            ok = np.ones(m, bool)
            for j in range(span):
                idx = idx * 4 + safe[j:j + m]
                ok &= good[j:j + m]
            self._kidx[span], self._kok[span] = idx, ok

        # maximal homopolymer runs, and a sparse table of run lengths for the interior of a window
        new = np.ones(self.n, bool)
        new[1:] = c[1:] != c[:-1]
        self._run_id = (np.cumsum(new) - 1).astype(np.int64)
        start = np.flatnonzero(new)
        self._run_start = start.astype(np.int64)
        self._run_end = np.append(start[1:] - 1, self.n - 1).astype(np.int64)
        run_len = (self._run_end - self._run_start + 1).astype(np.int64)
        self._st = [run_len]
        m = len(run_len)
        while (1 << (len(self._st) - 1)) < min(W, m):
            k = len(self._st) - 1
            prev = self._st[k]
            shift = np.minimum(np.arange(m) + (1 << k), m - 1)
            self._st.append(np.maximum(prev, prev[shift]))
        self._st = np.stack(self._st)                  # (levels, runs); level k holds the max of 2**k consecutive runs
        self._cum: dict = {}

    # ---- tracks built on demand -----------------------------------------

    def _tracks(self, key, strand_sign: int):
        """Cumulative sum, sum of squares and finite count of one tensor series, read on one strand."""
        ck = (key, strand_sign)
        if ck in self._cum:
            return self._cum[ck]
        spec = self.plan[key]
        span = spec["span"]
        dense = spec["pos"] if strand_sign > 0 else spec["neg"]
        v = dense[self._kidx[span]]
        ok = self._kok[span] & np.isfinite(v)
        vc = np.where(ok, v - spec["mu"], 0.0)
        # where each finite value differs from the finite value before it: a window with no such change after its first finite value is constant
        fidx = np.flatnonzero(ok)
        chg = np.zeros(len(v), np.int64)
        chg[fidx[1:]] = v[fidx[1:]] != v[fidx[:-1]]
        out = (np.concatenate([[0.0], np.cumsum(vc)]),
               np.concatenate([[0.0], np.cumsum(vc * vc)]),
               np.concatenate([[0], np.cumsum(ok)]).astype(np.int64),
               np.concatenate([[0], np.cumsum(chg)]).astype(np.int64), fidx)
        self._cum[ck] = out
        return out

    # ---- queries ---------------------------------------------------------

    def _maxrun(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Longest homopolymer run lying wholly inside each inclusive window [lo, hi] of this stretch."""
        i, j = self._run_id[lo], self._run_id[hi]
        width = hi - lo + 1
        left = self._run_end[i] - lo + 1
        right = hi - self._run_start[j] + 1
        a, b = i + 1, j - 1
        have = b >= a
        length = np.where(have, b - a + 1, 1)
        k = _LOG2[np.minimum(length, W + 1)]
        st = self._st
        m = st.shape[1]
        aa = np.clip(a, 0, m - 1)
        bb = np.clip(b - (1 << k) + 1, 0, m - 1)
        interior = np.where(have, np.maximum(st[k, aa], st[k, bb]), 0)
        return np.where(i == j, width, np.maximum(np.maximum(left, right), interior))

    def _stats(self, key, strand_sign: int, lo: np.ndarray, hi: np.ndarray):
        """Mean and population sd of one tensor series over each half-open k-mer range [lo, hi)."""
        s1, s2, cnt, chg, fidx = self._tracks(key, strand_sign)
        d1, d2 = s1[hi] - s1[lo], s2[hi] - s2[lo]
        c = (cnt[hi] - cnt[lo]).astype(float)
        first = fidx[np.minimum(cnt[lo], max(len(fidx) - 1, 0))] if len(fidx) else lo          # first finite value inside the window (when there is one)
        constant = (c > 0) & (chg[hi] - chg[np.minimum(first + 1, len(chg) - 1)] == 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = d1 / c
            mean = self.plan[key]["mu"] + m
            sd = np.sqrt(np.maximum(d2 / c - m * m, 0.0))
        mean[c == 0] = np.nan
        sd[c == 0] = np.nan
        sd[constant] = 0.0       # a window of one value, or of equal values, has no spread; the cumulative sums would leave up to ~1e-4 of rounding there
        return mean, sd

    def _stats_by_strand(self, key, lo, hi, plus):
        """`_stats` for a mixed batch: each strand reads its own tracks, so a single-strand batch builds only one set."""
        mean = np.empty(len(lo)); sd = np.empty(len(lo))
        for sign, mask in ((1, plus), (-1, ~plus)):
            if not mask.any():
                continue
            mean[mask], sd[mask] = self._stats(key, sign, lo[mask], hi[mask])
        return mean, sd

    def window_block(self, left0: np.ndarray, right0: np.ndarray, plus: np.ndarray) -> np.ndarray:
        """The 88 `comp.*` and `win.*` columns, in `a_flank`'s order."""
        left0 = np.asarray(left0, np.int64) - self.origin
        right0 = np.asarray(right0, np.int64) - self.origin
        plus = np.asarray(plus, bool)
        n = len(left0)
        cols = []
        for side in ("up", "dn"):
            for w in A.SUMMARY_WINDOWS:
                # the w nucleotides nearest the cut on this side, as an inclusive plus-strand window
                if side == "up":
                    lo = np.where(plus, left0 - w, right0 + 1)
                    hi = np.where(plus, left0 - 1, right0 + w)
                else:
                    lo = np.where(plus, right0 + 4, left0 - 3 - w)
                    hi = np.where(plus, right0 + 3 + w, left0 - 4)
                gc = (self._gc[hi + 1] - self._gc[lo]) / w
                pur = np.where(plus, self._pur[hi + 1] - self._pur[lo], self._pyr[hi + 1] - self._pyr[lo]) / w     # a purine on the minus strand reads as a pyrimidine here
                cols += [gc, pur, self._maxrun(lo, hi) / w]
                for unit, prop in A.SUMMARY_SERIES:
                    span = self.plan[(unit, prop)]["span"]
                    L = W - span + 1                            # k-mers in a 1,000-nt segment
                    mw = min(w, L)                              # a 1,000-nt window holds fewer than 1,000 k-mers
                    if side == "up":
                        klo = np.where(plus, left0 - W + L - mw, right0 + 1)
                    else:
                        klo = np.where(plus, right0 + 4, left0 - 2 - span - mw)
                    mean, sd = self._stats_by_strand((unit, prop), klo, klo + mw, plus)
                    cols += [mean, sd]
        return np.stack(cols, axis=1) if n else np.zeros((0, 88))

    def position_block(self, left0: np.ndarray, right0: np.ndarray, plus: np.ndarray) -> np.ndarray:
        """The 104 one-hot and 156 per-position tensor columns, in `a_flank`'s order."""
        left0 = np.asarray(left0, np.int64) - self.origin
        right0 = np.asarray(right0, np.int64) - self.origin
        plus = np.asarray(plus, bool)
        P = AF._plan(self.tables)["P"]                 # offsets of the 26 labels into the extended sequence
        nl = len(P)
        base0 = np.where(plus, left0 - 14, right0 + 14)[:, None]
        step = np.where(plus, 1, -1)[:, None]
        pos = base0 + step * P[None, :]                # plus-strand coordinate of each labelled position
        code = self.codes[pos].astype(np.int16)
        code = np.where(plus[:, None] | (code > 3), code, 3 - code)      # complement on the minus strand
        out = [(code[:, :, None] == np.arange(4)).reshape(len(left0), nl * 4).astype(float)]
        for key, spec in self.plan.items():
            if not (isinstance(key, tuple) and key[0] == "pp"):
                continue
            span = spec["span"]
            kstart = np.where(plus, left0 - 14, right0 + 15 - span)[:, None] + step * P[None, :]
            idx = self._kidx[span][kstart]
            ok = self._kok[span][kstart]
            v = np.where(plus[:, None], spec["pos"][idx], spec["neg"][idx])
            out.append(np.where(ok, v, np.nan))
        return np.hstack(out)

    def matrix(self, left0, right0, plus) -> np.ndarray:
        """The full 348-column family-A block for guides inside this stretch."""
        return np.hstack([self.position_block(left0, right0, plus),
                          self.window_block(left0, right0, plus)])


# --------------------------------------------------------------------------
# Whole-genome driver
# --------------------------------------------------------------------------


def genome_codes(genome: str | None = None) -> np.ndarray:
    from sgrna.build import genome as G
    return AF.LUT[np.frombuffer((genome or G.load_genome()).encode("ascii"), np.uint8)]


def region(codes: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """`codes[lo:hi]` on a **circular** chromosome, which is what a bacterial genome is and what the frozen pipeline assumes.

    A guide within 1 kb of coordinate 0 has an ordinary flank that wraps, and the curated cohort contains such guides
    (`left` runs from 317 to 4,639,509 on a 4,639,675 nt chromosome), so this is exercised rather than hypothetical. A
    linear contig would need the window clipped instead of wrapped; nothing here does that, so do not point it at a draft
    assembly's scaffolds without changing this function.
    """
    if 0 <= lo and hi <= len(codes):
        return codes[lo:hi]
    return codes[np.arange(lo, hi) % len(codes)]


def protospacer_codes(codes: np.ndarray, left0, plus) -> np.ndarray:
    """(n, 20) base codes of each protospacer in guide orientation, gathered straight out of the chromosome array."""
    left0 = np.asarray(left0, np.int64)[:, None]
    plus = np.asarray(plus, bool)[:, None]
    n = len(codes)
    j = np.arange(20)[None, :]
    pos = np.where(plus, left0 + j, left0 + 19 - j) % n
    c = codes[pos].astype(np.int16)
    return np.where(plus | (c > 3), c, 3 - c).astype(np.uint8)


def _groups(left0: np.ndarray, block: int) -> list[np.ndarray]:
    """Row indices grouped into stretches no wider than `block`, in coordinate order."""
    order = np.argsort(left0, kind="stable")
    out, cur, anchor = [], [], None
    for i in order:
        if anchor is None or left0[i] - anchor > block:
            if cur:
                out.append(np.array(cur))
            cur, anchor = [], left0[i]
        cur.append(i)
    if cur:
        out.append(np.array(cur))
    return out


def build_matrix(left0, strand, codes: np.ndarray | None = None, tables: dict | None = None,
                 block: int = BLOCK, names: bool = True):
    """Family A for guides given as plus-strand 0-based protospacer starts and '+'/'-' (or +1/-1) strands.

    Returns the (n, 348) matrix in `a_flank`'s column order, and the column names.
    """
    codes = genome_codes() if codes is None else codes
    tables = tables or qct.load_tables()
    plan = _series_plan(tables)
    left0 = np.asarray(left0, np.int64)
    plus = np.asarray([s in ("+", 1, "1", True) for s in np.asarray(strand).ravel()], bool)
    right0 = left0 + 19
    M = np.full((len(left0), len(AF._plan(tables)["names"])), np.nan)
    for rows in _groups(left0, block):
        lo = int(left0[rows].min()) - MARGIN
        hi = int(right0[rows].max()) + MARGIN + 1
        loc = Locus(region(codes, lo, hi), lo, tables, plan)
        M[rows] = loc.matrix(left0[rows], right0[rows], plus[rows])
    return (M, AF._plan(tables)["names"]) if names else M


# --------------------------------------------------------------------------
# Equality and cost
# --------------------------------------------------------------------------


def _cohort(n: int | None = None):
    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv")
    idx = idx[idx.located.astype(bool) & idx.pam.notna() & idx.upstream.notna()].reset_index(drop=True)
    if n:
        idx = idx.iloc[:n].reset_index(drop=True)
    idx["located"] = True
    return idx


def check(n: int = 4000, tol: float = 1e-9) -> dict:
    """The locus path against `a_flank_fast` on real guides: same columns, same NaN pattern, same values."""
    from sgrna.build import genome as G
    idx = _cohort(n)
    tables = qct.load_tables()
    codes = genome_codes()

    # the stored flanking strings and the genome must agree, or the two sides are not computing the same thing
    g = G.load_genome()
    mism = 0
    for _, r in idx.head(200).iterrows():
        ctx = G.guide_context(int(r.left), int(r.right), r.strand, W, g)
        mism += int(ctx["upstream"] != r.upstream or ctx["downstream"] != r.downstream
                    or ctx["protospacer"] != r.protospacer or ctx["pam"] != r.pam)

    # the protospacers the design path gathers from the chromosome must be the ones the screen reports
    pc = protospacer_codes(codes, idx.left.to_numpy() - 1, (idx.strand.to_numpy() == "+"))
    n_proto_bad = int((pc != AF._codes(idx.protospacer.tolist(), 20, False)).any(1).sum())

    ref, names = AF.build_matrix(idx, tables)
    got, names2 = build_matrix(idx.left.to_numpy() - 1, idx.strand.to_numpy(), codes, tables)
    assert names == names2, "column order differs"
    same_nan = bool((np.isnan(ref) == np.isnan(got)).all())
    denom = np.maximum(np.abs(ref), 1e-12)
    rel = np.abs(ref - got) / denom
    rel[np.isnan(ref) & np.isnan(got)] = 0.0
    worst = float(np.nanmax(rel))
    j = int(np.nanargmax(rel)) % ref.shape[1]
    row = dict(n=len(idx), n_cols=ref.shape[1], index_vs_genome_mismatches=mism,
               protospacer_mismatches=n_proto_bad,
               same_nan=same_nan, max_rel_diff=worst, worst_column=names[j],
               n_cols_over_tol=int((np.nanmax(rel, axis=0) > tol).sum()))
    print(f"  {row['n']:,} guides x {row['n_cols']} columns: stored flanks vs genome, {mism} mismatches in 200 rows; "
          f"same NaN pattern {same_nan}; largest relative difference {worst:.2e} ({row['worst_column']}); "
          f"columns above {tol:g}: {row['n_cols_over_tol']}; protospacers regathered from the genome, "
          f"{n_proto_bad} mismatches")
    assert mism == 0 and same_nan and worst <= tol and n_proto_bad == 0, row
    return row


def ngg_sites(codes: np.ndarray, lo: int = 0, hi: int | None = None):
    """Every NGG-flanked 20-mer whose protospacer starts in [lo, hi): plus-strand 0-based starts and strand signs.

    This is the set a design tool is actually handed -- about one site every 8.6 nt in *E. coli* (541,810 on the genome), against one every 138 nt in the
    curated screen -- so it is the density the per-locus precompute has to be judged at.
    """
    hi = len(codes) if hi is None else hi
    n = len(codes)
    p = np.arange(lo, hi)
    plus = p[(codes[(p + 21) % n] == 2) & (codes[(p + 22) % n] == 2)]          # ...[20 nt][N][G][G]
    minus = p[(codes[(p - 3) % n] == 1) & (codes[(p - 2) % n] == 1)]           # CC on the plus strand reads GG on the minus
    left0 = np.concatenate([plus, minus])
    sign = np.concatenate([np.ones(len(plus), np.int64), -np.ones(len(minus), np.int64)])
    order = np.argsort(left0, kind="stable")
    return left0[order], sign[order]


def bench(widths=(1_000, 10_000, 100_000, 1_000_000), reps: int = 3) -> pd.DataFrame:
    """Cost per guide of the two flank paths at design density, from one gene's worth of chromosome up to a fifth of it.

    Reported separately: the per-locus precompute, which is paid once per stretch of chromosome, and the per-guide queries,
    which are all that scales with the number of guides.
    """
    tables = qct.load_tables()
    codes = genome_codes()
    plan = _series_plan(tables)
    rows = []

    def t(fn, r):
        fn()
        ts = []
        for _ in range(r):
            s = time.perf_counter(); fn(); ts.append(time.perf_counter() - s)
        return float(np.median(ts)) * 1000.0

    for width in widths:
        start = 1_000_000                                     # an arbitrary but fixed stretch, away from the origin
        l0, sg = ngg_sites(codes, start, start + width)
        n = len(l0)
        plus = sg > 0
        sub = pd.DataFrame(dict(sgRNAID=np.arange(n), located=True))
        from sgrna.build import genome as G
        g = G.load_genome()
        ctx = [G.guide_context(int(a) + 1, int(a) + 20, "+" if b else "-", W, g) for a, b in zip(l0, plus)]
        for k in ("upstream", "downstream", "protospacer", "pam"):
            sub[k] = [c[k] for c in ctx]
        r = reps if n <= 20_000 else 1
        ms_old = t(lambda: AF.build_matrix(sub, tables), r)
        lo, hi = int(l0.min()) - MARGIN, int(l0.max()) + 19 + MARGIN + 1
        reg = region(codes, lo, hi)
        ms_pre = t(lambda: Locus(reg, lo, tables, plan).matrix(l0[:1], l0[:1] + 19, plus[:1]), r)
        ms_new = t(lambda: build_matrix(l0, sg, codes, tables), r)
        rows.append(dict(width_nt=width, n_guides=n, nt_per_guide=width / n,
                         a_flank_fast_ms_per_guide=ms_old / n,
                         flank_locus_ms_per_guide=ms_new / n,
                         precompute_ms=ms_pre, precompute_ms_per_guide=ms_pre / n,
                         query_ms_per_guide=max(ms_new - ms_pre, 0.0) / n,
                         speedup=ms_old / ms_new))
        q = rows[-1]
        print(f"  {width:>9,} nt, {n:6,} sites (one per {q['nt_per_guide']:.1f} nt): a_flank_fast "
              f"{q['a_flank_fast_ms_per_guide']:.4f} ms/guide, flank_locus {q['flank_locus_ms_per_guide']:.4f} "
              f"(precompute {q['precompute_ms_per_guide']:.4f} + query {q['query_ms_per_guide']:.4f}) -> {q['speedup']:.0f}x", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTCSV, index=False)
    print(f"written -> {OUTCSV}")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", action="store_true", help="timings only")
    ap.add_argument("--check-n", type=int, default=4000)
    a = ap.parse_args(argv)
    if not a.bench:
        check(a.check_n)
    bench()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
