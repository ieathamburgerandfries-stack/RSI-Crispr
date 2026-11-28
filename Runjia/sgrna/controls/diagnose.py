"""Binding energy and duplex stability are mostly functions of the 20-mer (70% and 50% of their columns have R² above 0.9), so they add little, and no other feature family is redundant in this way.

The pattern that prompts this module: eight feature families were added, each
motivated by a real mechanism and a real dataset, and six of them moved
Spearman by less than 0.01. The instinct is that something is broken. These
five measurements test that instinct instead of trusting it.

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

  floors()      What does a family score when its rows are shuffled? A
                permutation control keeps the same columns and the same count
                and removes the relationship to the label, so what it scores
                is that family's noise floor, and a gain inside the floor is
                not a gain. The floor depends on how many columns a family
                contributes and how easily they survive selection, so a
                348-column family and a 13-column family have different
                floors; until 2026-10-03 the project quoted the single floor
                measured on `a_flank` as if it applied to all of them. Every
                delta is **paired on (seed, fold)**, so the two arms always
                see identical data: `ablation_*.csv` stores the mean over
                whatever folds each experiment happens to have, and a family
                run on three seeds was being differenced against a baseline
                run on five. Reads `folds_main.csv`; fits nothing.

  no_information()  Decompose `d_mechanics`'s +0.099 into representation and
                new DNA. Nearest-neighbour duplex thermodynamics is a sum of
                dinucleotide parameters, so it is a deterministic function of
                a 20-mer the base already encodes losslessly, yet the family
                buys +0.099 rho. **Eight of its thirty columns are not
                functions of the 20-mer**: `eng.mech.flank.{up,dn}{25,100}.
                {dG37,dG_per_bp}` are duplex thermodynamics of the flanking
                DNA, which the base does not contain -- found by a
                reconstruction probe, where those eight came back at
                out-of-fold R² ~ 0 from the 364 mono- and dinucleotide
                indicator columns while nine of the guide-only columns came
                back at exactly 1.0000. So the headline arm mixes the claim
                with family A's mechanism, and the clean number is the
                guide-only one. Three arms over the same base: `guide_only`
                (22 columns, every one a function of the protospacer),
                `flank_only` (8, DNA outside it) and `full` (all 30).

    python -m sgrna.controls.diagnose --what all
    python -m sgrna.controls.diagnose --what floors no-information

`--what` takes one or more of redundancy headroom ceiling floors
no-information, or `all`.

Report section: Finding 1, Finding 4, Appendix A.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

from .. import config
from ..common import OUT
from ..build.build_matrix import load_dataset
from ..harness.run_ablation import _impute, _safe_spearman, champion_params, make_selector

RESULTS_REDUNDANCY = OUT / "redundancy.csv"
RESULTS_HEADROOM = OUT / "headroom.csv"
RESULTS_CEILING = OUT / "ceiling.json"
RESULTS_FLOORS = OUT / "permutation_floors.csv"
RESULTS_NO_INFORMATION = OUT / "no_information.csv"

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
                         "g_transcription", "h_offtarget", "i_shape",
                         "j_periodicity"),
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
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_REDUNDANCY, index=False)
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
    OUT.mkdir(parents=True, exist_ok=True)
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

    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS_CEILING.write_text(json.dumps(out, indent=2))
    if verbose:
        for k, v in out.items():
            print(f"  {k:32s} {v}")
    return out


# --------------------------------------------------------------------------
# 4. Permutation floors
# --------------------------------------------------------------------------

FOLDS_MAIN = "folds_main.csv"


def floors(verbose: bool = True) -> pd.DataFrame:
    """Per-family permutation floors, paired fold by fold.

    For each family: the gain over the baseline (Spearman and R^2, paired on seed and fold), the gain of the same
    family with its rows shuffled, their ratio, and a verdict (ratio below 1.5 is "indistinguishable from noise",
    below 5 "marginal", else "clear of the floor"). Reads the fold table of the main ablation.
    """
    folds = pd.read_csv(config.RESULTS / FOLDS_MAIN)
    base = folds[folds["experiment"] == "baseline"]

    def paired(name: str):
        arm = folds[folds["experiment"] == name]
        if arm.empty:
            return None
        m = arm.merge(base, on=["seed", "fold"], suffixes=("", "_base"))
        if m.empty:
            return None
        d_r2 = m["r2"] - m["r2_base"]
        d_rho = m["spearman"] - m["spearman_base"]
        t = ttest_rel(m["spearman"], m["spearman_base"]) if len(m) > 1 else None
        return dict(n_folds=len(m), d_r2=d_r2.mean(), d_rho=d_rho.mean(),
                    d_rho_sd=d_rho.std(),
                    folds_positive=int((d_rho > 0).sum()),
                    p=float(t.pvalue) if t is not None else np.nan)

    families = sorted({e for e in folds["experiment"].unique()
                       if "(permuted)" not in e and e != "baseline"})
    rows = []
    for fam in families:
        real = paired(fam)
        floor = paired(f"{fam} (permuted)")
        if real is None:
            continue
        row = dict(family=fam, real_folds=real["n_folds"],
                   real_d_r2=real["d_r2"], real_d_rho=real["d_rho"],
                   real_p=real["p"])
        if floor is None:
            row.update(floor_folds=0, floor_d_r2=np.nan, floor_d_rho=np.nan,
                       ratio=np.nan, verdict="no control run")
        else:
            ratio = (real["d_rho"] / abs(floor["d_rho"])
                     if floor["d_rho"] != 0 else np.inf)
            row.update(floor_folds=floor["n_folds"], floor_d_r2=floor["d_r2"],
                       floor_d_rho=floor["d_rho"], ratio=ratio,
                       verdict=("indistinguishable from noise" if ratio < 1.5
                                else "marginal" if ratio < 5
                                else "clear of the floor"))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("real_d_rho", ascending=False)
    df.to_csv(RESULTS_FLOORS, index=False)
    if verbose:
        cols = ["family", "real_d_rho", "floor_d_rho", "ratio", "verdict"]
        print(df[cols].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return df


# --------------------------------------------------------------------------
# 5. No information
# --------------------------------------------------------------------------

FLANK = ".flank."


def no_information(seeds=None, n_splits=None, n_features=300, verbose: bool = True):
    """The three arms of the `d_mechanics` decomposition over the curated base: guide_only, flank_only, full."""
    from ..harness.curated import load_base, load_block
    from ..harness.run_ablation import cross_validate

    seeds = list(seeds or config.SEEDS)
    n_splits = n_splits or config.N_SPLITS
    X, y, names, protos = load_base()
    v, fnames = load_block("d_mechanics", protos)
    is_flank = np.array([FLANK in f for f in fnames])
    if verbose:
        print(f"  d_mechanics: {len(fnames)} columns, {is_flank.sum()} from the "
              f"flanks, {(~is_flank).sum()} from the protospacer")

    arms = {
        "guide_only": ~is_flank,
        "flank_only": is_flank,
        "full": np.ones(len(fnames), dtype=bool),
    }

    # The baseline is the one already measured for the curated ablation table
    # -- same rows, same protocol, same tag, so `cross_validate` resumes from
    # its fold checkpoint instead of refitting it.
    res = cross_validate(X, y, names, ["base"] * len(names), n_features=n_features,
                         seeds=seeds, n_splits=n_splits, verbose=False,
                         tag="curabl_baseline", experiment="baseline")
    base_rho = res["summary"]["spearman"]["mean"]
    rows = [dict(arm="baseline", n_family_cols=0, spearman=base_rho,
                 spearman_sd=res["summary"]["spearman"]["sd"], d_spearman=0.0)]
    if verbose:
        print(f"  {'baseline':>12}: rho {base_rho:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS_NO_INFORMATION, index=False)

    for arm, mask in arms.items():
        sub = [f for f, m in zip(fnames, mask) if m]
        Xa = np.hstack([X, v[:, mask]])
        res = cross_validate(Xa, y, names + sub,
                             ["base"] * len(names) + ["d_mechanics"] * len(sub),
                             n_features=n_features, seeds=seeds, n_splits=n_splits,
                             verbose=False, tag=f"noinfo_{arm}", experiment=arm)
        rho = res["summary"]["spearman"]["mean"]
        rows.append(dict(arm=arm, n_family_cols=int(mask.sum()), spearman=rho,
                         spearman_sd=res["summary"]["spearman"]["sd"],
                         d_spearman=rho - base_rho))
        if verbose:
            print(f"  {arm:>12}: rho {rho:.4f}   delta {rho - base_rho:+.4f}  "
                  f"({int(mask.sum())} cols)", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS_NO_INFORMATION, index=False)

    out = pd.DataFrame(rows)
    if verbose:
        print()
        print(out.round(4).to_string(index=False))
        print(f"written -> {RESULTS_NO_INFORMATION}")
    return out


# --------------------------------------------------------------------------


PARTS = ("ceiling", "redundancy", "headroom", "floors", "no-information")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--what", nargs="+", default=["all"], choices=["all", *PARTS],
                    help="the parts to run (default: all five)")
    ap.add_argument("--families", nargs="*", default=["a_flank"])
    ap.add_argument("--k-max", type=int, default=2)
    args = ap.parse_args(argv)
    run = lambda part: "all" in args.what or part in args.what      # noqa: E731

    if run("ceiling"):
        print("\n== ceiling ==")
        ceiling()
    if run("redundancy"):
        print("\n== redundancy ==")
        redundancy(k_max=args.k_max)
        print(redundancy_summary().to_string(index=False))
    if run("headroom"):
        print("\n== headroom ==")
        headroom(families=tuple(args.families))
    if run("floors"):
        print("\n== permutation floors ==")
        floors()
    if run("no-information"):
        print("\n== d_mechanics decomposition ==")
        no_information()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
