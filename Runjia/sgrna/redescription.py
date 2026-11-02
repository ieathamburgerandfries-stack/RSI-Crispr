"""How much is re-description worth, and does it have to mean anything?

Finding 2 says a deterministic function of the 20-mer adds no *information* but
does add *performance* -- about +0.015 to a depth-limited tree, agreed across
four families built from physics: CRISPRoff energies, ViennaRNA folding,
nearest-neighbour thermodynamics, quantum-chemical tables.

Two things follow that the report has not tested.

**First: if the mechanism is "the tree cannot cheaply compute this", the function
need not be physical.** Any deterministic function of the sequence that packages
a higher-order effect should do, including a meaningless one. But performance
cannot rise without bound by bolting on arbitrary functions, so the effect has a
size and a saturation point. `random_tables()` measures both, using functions
built to be structurally identical to the physical ones and semantically empty:
a random weight per (position, base), summed over the guide. That is exactly the
shape of a nearest-neighbour energy sum with the physics replaced by noise.

**Second: the family is the wrong unit.** A family is a folder, not a kind -- 8
of `d_mechanics`'s 30 columns read flanking DNA while 22 are the guide restated,
and the family average hides it. `column_taxonomy()` replaces the folder with a
measurement: regress every one of the 668 family columns on the guide's own
one-hot encoding out of fold, and split by how well the guide predicts it.

**Third, and this is the one that settled it: is difficulty the point, or
meaning?** `random_tables()` builds functions that are meaningless *and* linear,
so they lose twice and the null cannot say which loss did it. `random_nonlinear()`
removes the second loss and keeps the first -- random weights over random
position tuples, uninformative by construction and arbitrarily expensive as the
tuple size rises -- with out-of-fold recoverability measured per arm so the ones
genuinely matched to the real fold energy can be named. They are null at every
order. Difficulty without meaning buys nothing.

    python -m sgrna.redescription --arm random
    python -m sgrna.redescription --arm taxonomy
    python -m sgrna.redescription --arm nonlinear
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config
from .curated_ablation import FAMILIES, _base, _block

RANDOM_OUT = config.RESULTS / "redescription_random.csv"
TAXONOMY_OUT = config.RESULTS / "redescription_taxonomy.csv"
FOLDS_OUT = config.RESULTS / "redescription_folds.csv"
COLUMNS_OUT = config.RESULTS / "redescription_columns.csv"

BASES = np.array(list("ACGT"))


def _onehot(protos: np.ndarray, order: int = 2) -> np.ndarray:
    """The guide itself, as indicators: 20x4 mono, plus 19x16 di if order >= 2."""
    seqs = np.array([list(str(p)[:20].upper().ljust(20, "A")) for p in protos])
    mono = np.concatenate(
        [(seqs[:, p:p + 1] == BASES[None, :]).astype(np.float32)
         for p in range(20)], axis=1)
    if order < 2:
        return mono
    pairs = np.array([a + b for a in "ACGT" for b in "ACGT"])
    di = np.concatenate(
        [(np.char.add(seqs[:, p], seqs[:, p + 1])[:, None] == pairs[None, :]
          ).astype(np.float32) for p in range(19)], axis=1)
    return np.hstack([mono, di])


def _oof_r2(Y: np.ndarray, A: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold R^2 of every column of Y predicted by least squares on A.

    One multi-output solve per fold rather than one per column: the design
    matrix is shared, so 668 columns cost no more than one.
    """
    A = np.hstack([A, np.ones((len(A), 1), dtype=np.float32)])
    pred = np.full(Y.shape, np.nan, dtype=np.float64)
    for tr, va in KFold(n_splits, shuffle=True, random_state=41).split(A):
        coef, *_ = np.linalg.lstsq(A[tr], np.nan_to_num(Y[tr]), rcond=None)
        pred[va] = A[va] @ coef
    ss_res = np.nansum((Y - pred) ** 2, axis=0)
    ss_tot = np.nansum((Y - np.nanmean(Y, axis=0)) ** 2, axis=0)
    # A column with no variance has no R^2 to report. Returning NaN here and
    # letting it fall through a chain of comparisons would silently file it
    # under whichever label the last `else` happens to be -- which is how 32
    # constant columns were first labelled "partly determined". They get their
    # own class instead.
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)


def _cv(X, y, n_features=None, seeds=(41, 42, 43), n_splits=5):
    import xgboost as xgb
    from scipy.stats import spearmanr

    from .run_ablation import _impute, champion_params, make_selector

    nf = config.N_FEATURES if n_features is None else n_features
    rhos = []
    for seed in seeds:
        for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(X):
            A, B = _impute(X[tr], X[va])
            if nf < A.shape[1]:
                sel = make_selector(seed)
                sel.fit(A, y[tr])
                top = np.argsort(sel.feature_importances_)[::-1][:nf]
                A, B = A[:, top], B[:, top]
            m = xgb.XGBRegressor(**champion_params(seed))
            m.fit(A, y[tr])
            rhos.append(float(spearmanr(y[va], m.predict(B)).statistic))
    return float(np.mean(rhos)), float(np.std(rhos, ddof=1))


# --------------------------------------------------------------------------

def random_tables(ks=(4, 16, 64, 256), table_seeds=(0, 1, 2),
                  model_seeds=(41, 42), verbose=True):
    """Meaningless functions of the guide, in the shape of physical ones.

    Three random draws per table count, two model seeds each: the spread across
    *draws* is what matters here -- one set of random tables is itself a sample,
    and a saturation curve read off a single draw would be unreadable.
    """
    X, y, names, protos = _base()
    seqs = np.array([list(str(p)[:20].upper().ljust(20, "A")) for p in protos])
    code = np.searchsorted(BASES, seqs)                  # (n, 20) in 0..3

    base_rho, base_sd = _cv(X, y, seeds=model_seeds)
    rows = [dict(arm="base", n_random=0, table_seed=-1, spearman=base_rho,
                 spearman_sd=base_sd, d_spearman=0.0)]
    if verbose:
        print(f"  base: rho {base_rho:.4f}", flush=True)

    for k in ks:
        for ts in table_seeds:
            rng = np.random.default_rng(1000 * k + ts)
            w = rng.normal(size=(k, 20, 4)).astype(np.float32)
            feats = np.empty((len(code), k), dtype=np.float32)
            for j in range(k):
                feats[:, j] = w[j][np.arange(20)[None, :], code].sum(axis=1)
            rho, sd = _cv(np.hstack([X, feats]), y, seeds=model_seeds)
            rows.append(dict(arm=f"base + {k} random tables", n_random=k,
                             table_seed=ts, spearman=rho, spearman_sd=sd,
                             d_spearman=rho - base_rho))
            if verbose:
                print(f"  {k:4d} random tables (seed {ts}): rho {rho:.4f}  "
                      f"d {rho - base_rho:+.4f}", flush=True)
            pd.DataFrame(rows).to_csv(RANDOM_OUT, index=False)

    out = pd.DataFrame(rows)
    if verbose:
        m = out[out.n_random > 0].groupby("n_random")["d_spearman"].mean()
        print("\n  mean gain by table count:")
        for k, v in m.items():
            print(f"    {k:5d}: {v:+.4f}")
        print(f"  for comparison, the physical families buy +0.011 to +0.019")
        print(f"written -> {RANDOM_OUT}")
    return out


def column_taxonomy(hi=0.99, lo=0.50, verbose=True):
    """Split all 668 family columns by how well the guide alone predicts them."""
    X, y, names, protos = _base()
    A = _onehot(protos, order=2)
    if verbose:
        print(f"  guide one-hot basis: {A.shape[1]} columns "
              f"(20x4 mono + 19x16 di)", flush=True)

    blocks, labels = [], []
    for fam in FAMILIES:
        try:
            v, fn = _block(fam, protos)
        except FileNotFoundError as exc:
            if verbose:
                print(f"  {fam}: skipped ({exc})")
            continue
        blocks.append(v)
        labels += [(fam, n) for n in fn]
    V = np.hstack(blocks)
    r2 = _oof_r2(V, A)

    cols = pd.DataFrame(labels, columns=["family", "column"])
    cols["oof_r2_from_guide"] = r2
    kind = np.where(np.isnan(r2), "constant (no variance)",
            np.where(r2 >= hi, "determined by the guide",
             np.where(r2 <= lo, "outside the guide", "partly determined")))
    cols["kind"] = kind
    cols.to_csv(COLUMNS_OUT, index=False)
    if verbose:
        print("\n  columns by kind:")
        print(cols["kind"].value_counts().to_string())
        print("\n  by family:")
        print(pd.crosstab(cols["family"], cols["kind"]).to_string())

    base_rho, base_sd = _cv(X, y)
    rows = [dict(arm="base", n_cols=0, spearman=base_rho, spearman_sd=base_sd,
                 d_spearman=0.0)]
    if verbose:
        print(f"\n  base: rho {base_rho:.4f}", flush=True)
    for kind in ("determined by the guide", "partly determined",
                 "outside the guide", "constant (no variance)"):
        idx = np.where(cols["kind"].to_numpy() == kind)[0]
        if len(idx) == 0:
            continue
        rho, sd = _cv(np.hstack([X, V[:, idx]]), y)
        rows.append(dict(arm=f"base + {kind}", n_cols=len(idx), spearman=rho,
                         spearman_sd=sd, d_spearman=rho - base_rho))
        if verbose:
            print(f"  + {kind:26s} ({len(idx):3d} cols): rho {rho:.4f}  "
                  f"d {rho - base_rho:+.4f}", flush=True)
        pd.DataFrame(rows).to_csv(TAXONOMY_OUT, index=False)

    rho, sd = _cv(np.hstack([X, V]), y)
    rows.append(dict(arm="base + all 668", n_cols=V.shape[1], spearman=rho,
                     spearman_sd=sd, d_spearman=rho - base_rho))
    if verbose:
        print(f"  + all {V.shape[1]} family columns: rho {rho:.4f}  "
              f"d {rho - base_rho:+.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(TAXONOMY_OUT, index=False)
    if verbose:
        print(f"written -> {TAXONOMY_OUT} and {COLUMNS_OUT}")
    return out


# The missing cell of the cheap/expensive table, and the one the report named as
# unrun. `random_tables` built functions that are uninformative *and* cheap
# (linear sums over position weights) and they pay nothing. Real folding is
# uninformative and expensive, and pays +0.0119. The random-energy Nussinov arm
# was meant to be uninformative-and-expensive, but its out-of-fold linear
# recoverability came back at 0.391-0.855 against the real column's 0.316 -- a
# random 4x4 table usually has one dominant pair type, so the minimisation
# collapses to counting it. It was never actually expensive.
#
# This builds the cell properly. A sum of `n_terms` random weights over random
# position *tuples* of a given order is uninformative by construction and gets
# arbitrarily expensive as the order rises, with nothing else changing: same
# inputs, same shape, same single column. Recoverability is measured rather than
# assumed, so the arms that are genuinely matched to 0.316 can be named.
NONLINEAR_OUT = config.RESULTS / "redescription_nonlinear.csv"


def _random_interaction(code: np.ndarray, order: int, rng,
                        n_terms: int = 200) -> np.ndarray:
    """Sum of random weights over random position tuples of one interaction order.

    Order 1 reproduces `random_tables`'s position-weight sum, which is the
    internal check that this instrument agrees with the earlier one.
    """
    n, L = code.shape
    out = np.zeros(n, dtype=np.float32)
    for _ in range(n_terms):
        pos = rng.choice(L, size=min(order, L), replace=False)
        w = rng.normal(size=(4,) * len(pos)).astype(np.float32)
        out += w[tuple(code[:, q] for q in pos)]
    return out


def random_nonlinear(orders=(1, 2, 3, 4, 6, 8), table_seeds=(0, 1, 2),
                     model_seeds=(41,), verbose=True):
    """Does non-linearity alone pay, or does the function have to mean something?"""
    X, y, names, protos = _base()
    seqs20 = np.array([str(q)[:20].upper().ljust(20, "A") for q in protos])
    code = np.searchsorted(BASES, np.array([list(q) for q in seqs20]))
    A = _onehot(protos, order=2)          # the same design matrix as 0.316

    base_rho, base_sd = _cv(X, y, seeds=model_seeds)
    rows = [dict(arm="base", order=-1, table_seed=-1, n_terms=0,
                 oof_r2=np.nan, spearman=base_rho, spearman_sd=base_sd,
                 d_spearman=0.0)]
    if verbose:
        print(f"  base: rho {base_rho:.4f}", flush=True)

    def add(label, order, ts, col):
        col = np.asarray(col, dtype=np.float32).reshape(-1, 1)
        r2 = float(_oof_r2(col.astype(np.float64), A)[0])
        rho, sd = _cv(np.hstack([X, col]), y, seeds=model_seeds)
        rows.append(dict(arm=label, order=order, table_seed=ts,
                         n_terms=0 if order < 0 else 200, oof_r2=r2,
                         spearman=rho, spearman_sd=sd,
                         d_spearman=rho - base_rho))
        if verbose:
            print(f"  {label:34s}: R2 {r2:6.3f}  rho {rho:.4f}  "
                  f"d {rho - base_rho:+.4f}", flush=True)
        pd.DataFrame(rows).to_csv(NONLINEAR_OUT, index=False)

    # The reference the controls have to be matched against, recomputed here so
    # it is measured on the same folds and the same design matrix as the rest.
    vb, fnb = _block("b_energy", protos)
    add("real eng.dg.guide_selffold", -1, -1,
        vb[:, list(fnb).index("eng.dg.guide_selffold")])

    for order in orders:
        for ts in table_seeds:
            add(f"random order {order} (table {ts})", order, ts,
                _random_interaction(code, order, np.random.default_rng(ts)))

    out = pd.DataFrame(rows)
    if verbose:
        print("\n  mean gain by interaction order (and mean recoverability):")
        for order, g in out[out.order > 0].groupby("order"):
            print(f"    order {order}: d {g.d_spearman.mean():+.4f}  "
                  f"R2 {g.oof_r2.mean():.3f}")
        print(f"written -> {NONLINEAR_OUT}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", default="both",
                    choices=["random", "taxonomy", "families", "provenance",
                             "folds", "nonlinear", "both"])
    a = ap.parse_args(argv)
    if a.arm in ("taxonomy", "both"):
        column_taxonomy()
    if a.arm in ("random", "both"):
        random_tables()
    if a.arm == "families":
        family_split()
    if a.arm == "provenance":
        provenance_split()
    if a.arm == "folds":
        random_folds()
    if a.arm == "nonlinear":
        random_nonlinear()
    return 0



# --------------------------------------------------------------------------
# Every family split the way d_mechanics was
# --------------------------------------------------------------------------

PROVENANCE_OUT = config.RESULTS / "redescription_provenance.csv"
FAMILY_SPLIT_OUT = config.RESULTS / "redescription_family_split.csv"


def family_split(families=("a_flank", "b_energy", "c_folding", "d_mechanics",
                           "f_methylation", "h_offtarget"),
                 hi=0.99, lo=0.50, verbose=True):
    """Does the d_mechanics lesson generalise? Split each family and measure.

    `d_mechanics` turned out to be 22 columns restating the guide bundled with 8
    reading flanking DNA, and the family-level number hid it. The column probe
    says three other families are mixtures too -- `a_flank` carries 28 columns
    the guide largely explains, `b_energy` is 9 guide columns and 1 outside,
    `h_offtarget` splits 6 against 7 -- so each is run as two arms here rather
    than one. A family whose halves behave alike was a fair bundle; one whose
    halves diverge was not.
    """
    X, y, names, protos = _base()
    A = _onehot(protos, order=2)
    base_rho, base_sd = _cv(X, y)
    rows = [dict(family="(base)", part="base", n_cols=0, spearman=base_rho,
                 spearman_sd=base_sd, d_spearman=0.0)]
    if verbose:
        print(f"  base: rho {base_rho:.4f}", flush=True)

    for fam in families:
        try:
            v, fn = _block(fam, protos)
        except FileNotFoundError as exc:
            if verbose:
                print(f"  {fam}: skipped ({exc})")
            continue
        r2 = _oof_r2(v, A)
        guide = np.where(r2 > lo)[0]          # determined or partly determined
        outside = np.where(r2 <= lo)[0]       # NaN columns are in neither
        const = np.where(np.isnan(r2))[0]
        for part, idx in (("guide-explained", guide), ("outside", outside),
                          ("constant", const),
                          ("whole family", np.arange(v.shape[1]))):
            if len(idx) == 0:
                continue
            rho, sd = _cv(np.hstack([X, v[:, idx]]), y)
            rows.append(dict(family=fam, part=part, n_cols=len(idx),
                             spearman=rho, spearman_sd=sd,
                             d_spearman=rho - base_rho))
            if verbose:
                print(f"  {fam:15s} {part:16s} ({len(idx):3d} cols): "
                      f"d {rho - base_rho:+.4f}", flush=True)
            pd.DataFrame(rows).to_csv(FAMILY_SPLIT_OUT, index=False)

    out = pd.DataFrame(rows)
    if verbose:
        print(f"written -> {FAMILY_SPLIT_OUT}")
    return out


# Columns whose *definition* reads bases outside the 20-mer. The R-squared probe
# cannot tell these apart from guide-only columns that are merely non-linear --
# both land in "outside the guide" -- so provenance is read off the generator
# code instead. In `d_mechanics` the two kinds are separable by name, which makes
# it the one family where the distinction can be measured rather than argued:
# the eight eng.mech.flank.* columns read 25 and 100 nt either side, while
# slide.strongest_from_pam and stack.min are an argmax and a min over the
# guide's own dinucleotide stack -- deterministic in the 20-mer, and non-linear,
# which is why the linear probe puts them outside.
PROVENANCE = {
    "d_mechanics": {
        "reads outside the 20-mer": ["eng.mech.flank."],
        "guide-only but non-linear": ["eng.mech.slide.strongest_from_pam",
                                      "eng.mech.stack.min"],
    },
}


def provenance_split(family="d_mechanics", verbose=True):
    """Within one family's "outside" half, separate provenance from non-linearity."""
    X, y, names, protos = _base()
    v, fn = _block(family, protos)
    fn = list(fn)
    rule = PROVENANCE[family]

    base_rho, base_sd = _cv(X, y)
    rows = [dict(family="(base)", part="base", n_cols=0, spearman=base_rho,
                 spearman_sd=base_sd, d_spearman=0.0)]
    if verbose:
        print(f"  base: rho {base_rho:.4f}", flush=True)

    groups = {}
    for part, prefixes in rule.items():
        groups[part] = [i for i, n in enumerate(fn)
                        if any(n.startswith(p) or n == p for p in prefixes)]
    groups["both halves together"] = sorted(
        set(i for g in groups.values() for i in g))

    for part, idx in groups.items():
        rho, sd = _cv(np.hstack([X, v[:, idx]]), y)
        rows.append(dict(family=family, part=part, n_cols=len(idx),
                         spearman=rho, spearman_sd=sd,
                         d_spearman=rho - base_rho))
        if verbose:
            print(f"  {part:26s} ({len(idx):2d} cols): d {rho - base_rho:+.4f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(PROVENANCE_OUT, index=False)

    if verbose:
        print(f"written -> {PROVENANCE_OUT}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Is the folding gain the physics, or only the non-linearity?
# --------------------------------------------------------------------------
# `random_tables()` showed that meaningless *linear* functions of the guide buy
# nothing, and the family split showed that the columns which do pay are the
# guide-determined ones a linear probe cannot see -- ViennaRNA folds, and one
# self-folding energy worth +0.0116 alone. Those two results are consistent with
# two very different readings:
#
#   (a) RNA secondary structure is part of the mechanism, and an explicitly
#       computed fold energy delivers a real biological effect the model cannot
#       reach from indicators; or
#   (b) nothing biological is involved -- any non-linear, non-local summary of
#       the guide would pay a depth-limited tree the same hundredth, and folding
#       is simply the one we happened to compute.
#
# The two are separated by replacing the chemistry and keeping the algorithm.
# `_fold_random` runs a Nussinov fold -- the minimum over the exponentially many
# non-crossing pairings of the 20-mer -- with the nearest-neighbour energies
# replaced by a random symmetric 4x4 table. `_sum_random` takes *the same table*
# and sums it over the same position pairs instead of minimising over matchings.
# So the two arms differ in the operator and nothing else:
#
#   sum  over pairs      non-local, linear      <- tree can approximate cheaply
#   min  over matchings  non-local, non-linear  <- tree cannot
#
# If the fold arm pays and the sum arm does not, the gain belongs to the
# combinatorial minimisation and reading (a) is unsupported: a *random* fold
# would be doing the work of a real one. If neither pays, reading (a) survives
# and the physics is load-bearing.
MIN_LOOP = 3


def _random_pair_table(rng) -> np.ndarray:
    """A symmetric 4x4 pairing energy, in the shape of a real one."""
    e = rng.normal(size=(4, 4)).astype(np.float32)
    return ((e + e.T) / 2).astype(np.float32)


def _fold_random(code: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Nussinov minimum-energy fold of every guide, vectorised over guides."""
    n, L = code.shape
    zero = np.zeros(n, dtype=np.float32)
    E = {}
    for i in range(L + 1):
        for j in range(-1, L):
            if j - i < MIN_LOOP + 1:
                E[(i, j)] = zero
    for span in range(MIN_LOOP + 2, L + 1):
        for i in range(0, L - span + 1):
            j = i + span - 1
            best = E[(i + 1, j)].copy()                   # position i unpaired
            for k in range(i + MIN_LOOP + 1, j + 1):      # i pairs with k
                cand = E[(i + 1, k - 1)] + E[(k + 1, j)] + e[code[:, i], code[:, k]]
                np.minimum(best, cand, out=best)
            E[(i, j)] = best
    return E[(0, L - 1)]


def _sum_random(code: np.ndarray, e: np.ndarray) -> np.ndarray:
    """The same energies summed over the same position pairs. No minimisation."""
    n, L = code.shape
    out = np.zeros(n, dtype=np.float32)
    for i in range(L):
        for k in range(i + MIN_LOOP + 1, L):
            out += e[code[:, i], code[:, k]]
    return out


def _mfe(seqs) -> np.ndarray:
    """Real ViennaRNA MFE of each sequence, as RNA."""
    import RNA

    return np.array([RNA.fold(str(q).upper().replace("T", "U"))[1]
                     for q in seqs], dtype=np.float32)


def random_folds(table_seeds=(0, 1, 2), model_seeds=(41,), verbose=True):
    """Three controls for the folding gain, each holding something different fixed.

    `random fold` / `random sum` share one random energy table and differ only
    in the operator, which isolates the combinatorial minimisation. But a random
    table is not a faithful stand-in for a real one: its out-of-fold linear
    recoverability runs 0.39 to 0.86 against the real column's 0.316, because a
    random table usually has one dominant pair type and the minimum collapses to
    counting it. So the load-bearing control is the third.

    `shuffled fold` applies **one fixed permutation of the twenty positions** to
    every guide and folds the result with real ViennaRNA. Same algorithm, same
    energy model, same value distribution, same non-linearity, still a
    deterministic function of the guide -- but the structure it reports is not
    the structure of the sgRNA that was actually transcribed. It is conservative
    on purpose: shuffling preserves base composition, so whatever part of the
    gain is composition-driven foldability survives in the control and only the
    position-specific structure is removed.
    """
    X, y, names, protos = _base()
    seqs20 = np.array([str(p)[:20].upper().ljust(20, "A") for p in protos])
    chars = np.array([list(q) for q in seqs20])
    code = np.searchsorted(BASES, chars)

    base_rho, base_sd = _cv(X, y, seeds=model_seeds)
    rows = [dict(arm="base", kind="base", n_cols=0, table_seed=-1,
                 spearman=base_rho, spearman_sd=base_sd, d_spearman=0.0)]
    if verbose:
        print(f"  base: rho {base_rho:.4f}", flush=True)

    def add(label, kind, cols, ts=-1):
        cols = np.asarray(cols, dtype=np.float32)
        if cols.ndim == 1:
            cols = cols[:, None]
        rho, sd = _cv(np.hstack([X, cols]), y, seeds=model_seeds)
        rows.append(dict(arm=label, kind=kind, n_cols=cols.shape[1],
                         table_seed=ts, spearman=rho, spearman_sd=sd,
                         d_spearman=rho - base_rho))
        if verbose:
            print(f"  {label:38s}: rho {rho:.4f}  d {rho - base_rho:+.4f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(FOLDS_OUT, index=False)

    # The published columns, re-measured on these exact folds.
    vb, fnb = _block("b_energy", protos)
    add("real eng.dg.guide_selffold", "real",
        vb[:, list(fnb).index("eng.dg.guide_selffold")])
    v, _ = _block("c_folding", protos)
    add("real c_folding block", "real", v)

    # The same quantity recomputed here, so the control shares its code path.
    mfe_real = _mfe(seqs20)
    add("spacer MFE, recomputed", "real", mfe_real)

    for ts in table_seeds:
        rng = np.random.default_rng(9000 + ts)
        perm = rng.permutation(20)
        shuf = np.array(["".join(r) for r in chars[:, perm]])
        add(f"shuffled-position MFE (perm {ts})", "shuffled fold",
            _mfe(shuf), ts)

    for ts in table_seeds:
        rng = np.random.default_rng(7000 + ts)
        e = _random_pair_table(rng)
        add(f"random-energy fold (table {ts})", "random fold",
            _fold_random(code, e), ts)
        add(f"random-energy sum (table {ts})", "random sum",
            _sum_random(code, e), ts)

    out = pd.DataFrame(rows)
    if verbose:
        print("\n  mean gain by kind:")
        for kind, g in out[out.kind != "base"].groupby("kind"):
            print(f"    {kind:16s}: {g.d_spearman.mean():+.4f}  "
                  f"({', '.join(f'{v:+.4f}' for v in g.d_spearman)})")
        print(f"written -> {FOLDS_OUT}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
