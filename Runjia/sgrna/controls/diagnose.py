"""Why doesn't adding features help? Three diagnostics that answer it.

The pattern that prompts this module: eight feature families were added, each
motivated by a real mechanism and a real dataset, and six of them moved
Spearman by less than 0.01. The instinct is that something is broken. These
three measurements test that instinct instead of trusting it.

  redundancy()  For each new column, how much of it is already recoverable
                from the 20 nt protospacer? The published matrix was shown to
                be a *lossless* encoding of the protospacer -- every position
                carries at least three of its four mononucleotide indicators,
                so the fourth is 1 minus the others. Any feature that is a
                deterministic function of the 20-mer therefore adds **zero
                information**; whatever gain it shows is the convenience of a
                nonlinear summary the tree would otherwise have to learn. This
                measures which families are in that position.

  headroom()    Take the champion model's out-of-fold residuals and correlate
                every feature -- used or not -- against them. If nothing
                correlates, there is nothing left in this feature set for a
                better model to find. If something does, the model is leaving
                it on the table.

  ceiling()     How reproducible is the label? Two independent screens of the
                same library (WT-SpCas9 and eSpCas9, Guo et al.) share 33,567
                protospacers. Their rank correlation is an estimate of how
                much of a cut score is guide-intrinsic and repeatable, and
                therefore how high any sequence model could ever go.

    python -m sgrna.controls.diagnose --what all
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..build.build_matrix import load_dataset
from ..harness.run_ablation import _impute, _safe_spearman, champion_params, make_selector

RESULTS_REDUNDANCY = config.RESULTS / "redundancy.csv"
RESULTS_HEADROOM = config.RESULTS / "headroom.csv"
RESULTS_CEILING = config.RESULTS / "ceiling.json"

BASES = "ACGT"


# --------------------------------------------------------------------------
# The sequence basis
# --------------------------------------------------------------------------


def onehot(protospacers, k_max: int = 2) -> tuple[np.ndarray, list[str]]:
    """Position-specific k-mer indicators for k = 1..k_max.

    This is the same representation the published `V####` columns turn out to
    be, so a feature well predicted from here is a feature the published
    matrix already contains.
    """
    seqs = [str(s).upper() for s in protospacers]
    L = len(seqs[0])
    cols, names = [], []
    for k in range(1, k_max + 1):
        alphabet = [""]
        for _ in range(k):
            alphabet = [a + b for a in alphabet for b in BASES]
        for pos in range(L - k + 1):
            sub = np.array([s[pos:pos + k] for s in seqs])
            for kmer in alphabet:
                v = (sub == kmer).astype(np.float32)
                if v.any() and not v.all():
                    cols.append(v)
                    names.append(f"p{pos + 1}.{kmer}")
    return np.column_stack(cols), names


def _oof_r2_batch(basis: np.ndarray, targets: np.ndarray, n_splits: int = 5,
                  alpha: float = 1.0, seed: int = 41) -> np.ndarray:
    """Out-of-fold ridge R^2 for many targets against one fixed basis.

    Because the basis never changes, the normal equations are solved once per
    fold for *all* targets at once. That turns several hundred separate ridge
    fits into five matrix solves, which is the difference between ten minutes
    and two seconds.

    Missing values are mean-filled inside the training fold; a column that is
    constant or nearly all missing returns NaN.
    """
    from sklearn.model_selection import KFold

    n, p = basis.shape
    Xb = np.hstack([basis, np.ones((n, 1), dtype=np.float32)])
    Y = np.asarray(targets, dtype=np.float64)
    pred = np.full(Y.shape, np.nan)

    reg = alpha * np.eye(p + 1)
    reg[-1, -1] = 0.0                       # never penalise the intercept

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in kf.split(Xb):
        Ytr = Y[tr].copy()
        mu = np.nanmean(Ytr, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        Ytr = np.where(np.isfinite(Ytr), Ytr, mu)
        A = Xb[tr].T @ Xb[tr] + reg
        B = Xb[tr].T @ Ytr
        coef = np.linalg.solve(A, B)
        pred[va] = Xb[va] @ coef

    out = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        ok = np.isfinite(Y[:, j]) & np.isfinite(pred[:, j])
        if ok.sum() < 200:
            continue
        yj = Y[ok, j]
        ss_tot = float(np.sum((yj - yj.mean()) ** 2))
        if ss_tot <= 0:
            continue
        ss_res = float(np.sum((yj - pred[ok, j]) ** 2))
        out[j] = 1.0 - ss_res / ss_tot
    return out


# --------------------------------------------------------------------------
# 1. Redundancy
# --------------------------------------------------------------------------


def redundancy(families=("a_flank", "b_energy", "c_folding", "d_mechanics",
                         "d_supercoiling", "e_nucleoid", "f_methylation",
                         "g_transcription", "h_offtarget"),
               k_max: int = 2, verbose: bool = True) -> pd.DataFrame:
    """How much of each new feature is already implied by the 20-mer?

    `r2_from_protospacer` near 1.0 means the column is a function of the
    guide sequence, which the published matrix already encodes losslessly --
    so the column carries no new information, only a different shape.
    """
    index = pd.read_csv(config.INTERIM / "guide_index.csv")
    index = index.dropna(subset=["protospacer"])
    basis, _ = onehot(index["protospacer"].tolist(), k_max=k_max)
    if verbose:
        print(f"sequence basis: {basis.shape[1]:,} position-specific indicators "
              f"(k <= {k_max}) over {basis.shape[0]:,} guides")

    ids = index[config.ID_COL].to_numpy()
    rows = []
    for fam in families:
        try:
            block = load_block_safe(fam)
        except Exception as exc:                       # noqa: BLE001
            if verbose:
                print(f"  {fam}: skipped ({exc})")
            continue
        block = block.set_index(config.ID_COL).reindex(ids)
        targets = block.to_numpy(dtype=float)
        r2 = _oof_r2_batch(basis, targets)
        for col, v in zip(block.columns, r2):
            rows.append(dict(family=fam, column=col, r2_from_protospacer=v))
        got = [g for g in r2 if np.isfinite(g)]
        if verbose and got:
            print(f"  {fam:16s} n={len(got):4d}  median R2 {np.median(got):.3f}  "
                  f"share above 0.9: {np.mean(np.array(got) > 0.9):.0%}")

    df = pd.DataFrame(rows)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_REDUNDANCY, index=False)
    return df


def cross_redundancy(target: str, basis_families=("a_flank",),
                     verbose: bool = True) -> pd.DataFrame:
    """Is one feature family computable from another?

    `redundancy` asks whether a family is implied by the 20 nt protospacer.
    This asks the more general question the rule actually needs: whether a
    candidate family is implied by **the features the model already has**.

    Both directions are reported, because they answer different things. If
    `i_shape` is recoverable from `a_flank` then it carries no new information
    and its failure to add anything confirms the rule rather than breaking it.
    If `a_flank` is *also* recoverable from `i_shape`, the two are simply two
    parameterisations of one variable and the question becomes which is the
    better-shaped one.
    """
    from ..build.build_matrix import load_block

    index = pd.read_csv(config.INTERIM / "guide_index.csv")
    ids = index[config.ID_COL].to_numpy()

    def frame(fam):
        b = load_block(fam).set_index(config.ID_COL).reindex(ids)
        v = b.to_numpy(dtype=float)
        keep = ~np.all(~np.isfinite(v), axis=0)
        return b.columns[keep].tolist(), v[:, keep]

    tgt_cols, Y = frame(target)
    basis_cols, B = [], []
    for fam in basis_families:
        c, v = frame(fam)
        basis_cols += c
        B.append(v)
    B = np.hstack(B)

    # Mean-fill the basis; a basis column that is missing for a row cannot be
    # allowed to drop the row, or the two directions would be scored on
    # different subsets.
    mu = np.nanmean(B, axis=0)
    B = np.where(np.isfinite(B), B, np.where(np.isfinite(mu), mu, 0.0))
    # Standardise so one wide-ranged column does not dominate the ridge.
    sd = B.std(axis=0); sd[sd == 0] = 1.0
    B = (B - B.mean(axis=0)) / sd

    if verbose:
        print(f"  basis: {B.shape[1]:,} columns from {list(basis_families)}")
        print(f"  target: {Y.shape[1]:,} columns from {target}")

    r2 = _oof_r2_batch(B.astype(np.float32), Y, alpha=10.0)
    df = pd.DataFrame(dict(target_family=target, column=tgt_cols,
                           r2_from_basis=r2))
    ok = df["r2_from_basis"].dropna()
    if verbose and len(ok):
        print(f"  median R2 {ok.median():.3f}   "
              f"share > 0.9: {(ok > 0.9).mean():.0%}   "
              f"share > 0.5: {(ok > 0.5).mean():.0%}")
    out = config.RESULTS / f"cross_redundancy_{target}_from_{'_'.join(basis_families)}.csv"
    df.to_csv(out, index=False)
    return df


def load_block_safe(fam: str) -> pd.DataFrame:
    from ..build.build_matrix import load_block
    return load_block(fam)


def redundancy_summary(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per family: is this family new information or a re-encoding?"""
    if df is None:
        df = pd.read_csv(RESULTS_REDUNDANCY)
    g = df.dropna(subset=["r2_from_protospacer"]).groupby("family")
    out = g["r2_from_protospacer"].agg(
        n="size", median="median", q90=lambda s: s.quantile(0.9),
        share_determined=lambda s: float((s > 0.9).mean()),
    ).reset_index()
    return out.sort_values("share_determined", ascending=False)


# --------------------------------------------------------------------------
# 2. Headroom
# --------------------------------------------------------------------------


def headroom(families=("a_flank",), extra=("b_energy", "c_folding",
                                           "d_mechanics", "d_supercoiling",
                                           "e_nucleoid", "f_methylation",
                                           "g_transcription", "h_offtarget"),
             seed: int = 41, n_splits: int = 5, n_features: int | None = None,
             verbose: bool = True) -> pd.DataFrame:
    """Correlate every feature against the champion model's residuals.

    `families` is what the model is allowed to use; `extra` is everything else
    we have. A feature in `extra` with a large |rho| against the residual is
    information the current model is failing to use. A table of near-zeros
    means the feature set, not the model, is exhausted.
    """
    from sklearn.model_selection import KFold
    import xgboost as xgb

    n_features = n_features or config.N_FEATURES
    all_fams = list(dict.fromkeys(list(families) + list(extra)))
    X, y, names, ids, fam_of = load_dataset(all_fams, verbose=verbose)
    names = np.asarray(names)
    fam_of = np.asarray(fam_of)

    usable = np.isin(fam_of, ["base"] + list(families))
    if verbose:
        print(f"  model may use {usable.sum():,} columns; "
              f"{(~usable).sum():,} held back for the residual test")

    pred = np.empty_like(y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        X_tr, X_va = _impute(X[tr][:, usable], X[va][:, usable])
        sel = make_selector(seed)
        sel.fit(X_tr, y[tr])
        top = np.argsort(sel.feature_importances_)[::-1][:n_features]
        model = xgb.XGBRegressor(**champion_params(seed))
        model.fit(X_tr[:, top], y[tr])
        pred[va] = model.predict(X_va[:, top])
        if verbose:
            print(f"    fold {fold} rho {_safe_spearman(y[va], pred[va]):.4f}")

    resid = y - pred
    if verbose:
        print(f"  out-of-fold rho {_safe_spearman(y, pred):.4f}; "
              f"residual sd {resid.std():.3f} vs label sd {y.std():.3f}")

    rows = []
    Xf = np.where(np.isfinite(X), X, np.nan)
    for j in range(X.shape[1]):
        v = Xf[:, j]
        ok = np.isfinite(v)
        if ok.sum() < 200 or np.nanstd(v[ok]) == 0:
            continue
        rho = spearmanr(v[ok], resid[ok]).statistic
        rho_y = spearmanr(v[ok], y[ok]).statistic
        rows.append(dict(column=names[j], family=fam_of[j],
                         used=bool(usable[j]),
                         rho_with_residual=float(rho),
                         rho_with_label=float(rho_y)))
    df = pd.DataFrame(rows)
    df["abs_resid"] = df["rho_with_residual"].abs()
    df = df.sort_values("abs_resid", ascending=False)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_HEADROOM, index=False)

    if verbose:
        print("\n  strongest residual correlations among features the model "
              "was NOT given:")
        held = df[~df["used"]].head(10)
        for _, r in held.iterrows():
            print(f"    {r['column'][:44]:44s} {r['family']:16s} "
                  f"rho {r['rho_with_residual']:+.3f}")
    return df


# --------------------------------------------------------------------------
# 3. Ceiling
# --------------------------------------------------------------------------


CRISPRHAL_OFFSETS = {"WT-SpCas9": 189, "eSpCas9": 193}


def _crisprhal(name: str, offset: int) -> pd.DataFrame:
    root = config.EXTERNAL / "crisprHAL" / "data"
    parts = []
    for split in ("training", "testing"):
        p = root / f"{name}_{split}_data.csv"
        if p.exists():
            parts.append(pd.read_csv(p, header=None, names=["seq", "score"]))
    if not parts:
        raise FileNotFoundError(f"no crisprHAL files for {name}")
    d = pd.concat(parts, ignore_index=True)
    d["protospacer"] = d["seq"].str[offset:offset + 20]
    return d.groupby("protospacer", as_index=False)["score"].mean()


def ceiling(verbose: bool = True) -> dict:
    """How much of a cut score is repeatable, and what does that bound?

    Two screens of the same library with different SpCas9 variants give an
    independent replicate of the guide-intrinsic component. Under the usual
    attenuation argument -- each observed score is signal plus independent
    noise -- their rank correlation *is* the reliability, and the highest
    correlation any model could reach against a single observed score is its
    square root.

    The enzyme differs between the two screens, so part of the shortfall is
    real biology rather than noise. That makes this a **conservative** bound:
    the true ceiling is at least this high.
    """
    out: dict = {}
    wt = _crisprhal("WT-SpCas9", CRISPRHAL_OFFSETS["WT-SpCas9"])
    es = _crisprhal("eSpCas9", CRISPRHAL_OFFSETS["eSpCas9"])
    m = wt.merge(es, on="protospacer", suffixes=("_wt", "_esp"))
    rho = float(m["score_wt"].corr(m["score_esp"], method="spearman"))
    out["n_shared_protospacers"] = int(len(m))
    out["cross_screen_spearman"] = rho
    out["implied_ceiling_spearman"] = float(np.sqrt(max(rho, 0.0)))

    index = pd.read_csv(config.INTERIM / "guide_index.csv")
    j = index.merge(m, on="protospacer", how="inner")
    out["n_of_ours_in_both"] = int(len(j))
    out["published_vs_wt_spearman"] = float(
        j["cut_score"].corr(j["score_wt"], method="spearman"))
    out["published_vs_esp_spearman"] = float(
        j["cut_score"].corr(j["score_esp"], method="spearman"))

    # How much of the label is guide-level at all, rather than gene-level?
    sub = index.dropna(subset=["gene_name", "cut_score"])
    counts = sub.groupby("gene_name")["cut_score"].transform("size")
    sub = sub[counts >= 5]
    gene_mean = sub.groupby("gene_name")["cut_score"].transform("mean")
    total = float(sub["cut_score"].var())
    within = float((sub["cut_score"] - gene_mean).var())
    out["genes_with_5plus_guides"] = int(sub["gene_name"].nunique())
    out["within_gene_variance_share"] = within / total if total else np.nan

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    RESULTS_CEILING.write_text(json.dumps(out, indent=2))
    if verbose:
        for k, v in out.items():
            print(f"  {k:32s} {v}")
    return out


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", default="all",
                    choices=["all", "redundancy", "headroom", "ceiling"])
    ap.add_argument("--families", nargs="*", default=["a_flank"])
    ap.add_argument("--k-max", type=int, default=2)
    args = ap.parse_args(argv)

    if args.what in ("all", "ceiling"):
        print("\n== ceiling ==")
        ceiling()
    if args.what in ("all", "redundancy"):
        print("\n== redundancy ==")
        redundancy(k_max=args.k_max)
        print(redundancy_summary().to_string(index=False))
    if args.what in ("all", "headroom"):
        print("\n== headroom ==")
        headroom(families=tuple(args.families))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
