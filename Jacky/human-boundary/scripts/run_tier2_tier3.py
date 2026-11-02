#!/usr/bin/env python3
"""Tier-2 / Tier-3 human-boundary experiments.

D1  Chromatin / accessibility at located hg38 loci (ENCODE K562 DNase + ATAC).
E1  Cross-kingdom transfer with Family A on both sides (aligned by column name).
E2  GC-matched / GC²-residualised transfer controls (shared cols + Family A).

Assumes Tier-1 caches already exist under Runjia/data/interim/human_boundary/
and E. coli a_flank at Runjia/data/interim/features/a_flank.csv.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(os.environ.get("RSI09_ROOT", "/workspace/Runjia")).resolve()
os.environ["RSI09_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT))

from sgrna import config, featurise  # noqa: E402
from sgrna.run_ablation import _impute, cross_validate  # noqa: E402

OUT = Path("/workspace/Jacky/human-boundary/results")
CACHE = config.INTERIM / "human_boundary"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 41
N_TOP = 300


def _spearman(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3:
        return float("nan")
    r = spearmanr(y[m], p[m]).statistic
    return float(r) if np.isfinite(r) else float("nan")


def _xgb(seed=SEED, **kw):
    import xgboost as xgb
    p = dict(
        n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, random_state=seed, n_jobs=-1, tree_method="hist",
        importance_type="gain",
    )
    p.update(kw)
    return xgb.XGBRegressor(**p)


# ---------------------------------------------------------------------------
# D1 — chromatin
# ---------------------------------------------------------------------------


def _sample_bigwig(bw_path: Path, chrom: str, start: int, end: int) -> float:
    import pyBigWig
    bw = pyBigWig.open(str(bw_path))
    try:
        # pyBigWig uses 0-based half-open; our left/right are 1-based inclusive
        # on the protospacer. Sample a ±500 nt window around the cut (PAM-proximal).
        mid = (start + end) // 2
        lo = max(0, mid - 500)
        hi = mid + 500
        chroms = set(bw.chroms().keys())
        c = chrom if chrom in chroms else chrom.replace("chr", "")
        if c not in chroms and ("chr" + c) in chroms:
            c = "chr" + c
        if c not in chroms:
            return float("nan")
        vals = bw.values(c, lo, min(hi, bw.chroms()[c]), numpy=True)
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if len(vals) else float("nan")
    finally:
        bw.close()


def run_d1_chromatin():
    print("=== D1 chromatin ===", flush=True)
    loc = pd.read_csv(CACHE / "human_located.csv")
    tracks = {
        "K562_DNase": config.REFERENCE / "chromatin" / "K562_DNase_ENCFF655HFU.bigWig",
        "K562_ATAC": config.REFERENCE / "chromatin" / "K562_ATAC_ENCFF674WKB.bigWig",
    }
    for name, path in tracks.items():
        if not path.exists():
            raise SystemExit(f"missing track {path}")

    # Unique loci only (one row per chrom/left/right) then map back — faster
    key = loc[["chrom", "left", "right"]].drop_duplicates().reset_index(drop=True)
    print(f"  sampling {len(key):,} unique loci x {len(tracks)} tracks ...", flush=True)
    for tname, tpath in tracks.items():
        import pyBigWig
        bw = pyBigWig.open(str(tpath))
        chroms = set(bw.chroms().keys())
        vals = []
        for i, r in key.iterrows():
            chrom = str(r["chrom"])
            c = chrom if chrom in chroms else (
                "chr" + chrom if ("chr" + chrom) in chroms else chrom.replace("chr", "")
            )
            mid = int((int(r["left"]) + int(r["right"])) // 2)
            lo, hi = max(0, mid - 500), mid + 500
            if c not in chroms:
                vals.append(np.nan)
                continue
            hi = min(hi, int(bw.chroms()[c]))
            if hi <= lo:
                vals.append(np.nan)
                continue
            arr = bw.values(c, lo, hi, numpy=True)
            arr = np.asarray(arr, float)
            arr = arr[np.isfinite(arr)]
            vals.append(float(arr.mean()) if len(arr) else np.nan)
            if (i + 1) % 2000 == 0:
                print(f"    {tname}: {i+1}/{len(key)}", flush=True)
        bw.close()
        key[tname] = vals

    merged = loc.merge(key, on=["chrom", "left", "right"], how="left")
    y = merged["y"].to_numpy(float)

    # Load base residuals: y - CV prediction from regenerated base
    Xb = np.load(CACHE / "X_human_base.npy")
    from sklearn.model_selection import KFold
    oof = np.full(len(y), np.nan)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in kf.split(Xb):
        At, Bt = _impute(Xb[tr], Xb[te])
        m0 = _xgb(n_estimators=300)
        m0.fit(At, y[tr])
        top = np.argsort(m0.feature_importances_)[::-1][:N_TOP]
        m = _xgb()
        m.fit(At[:, top], y[tr])
        oof[te] = m.predict(Bt[:, top])
    resid = y - oof

    rows = []
    for tname in tracks:
        s = merged[tname].to_numpy(float)
        rows.append(dict(
            track=tname,
            n=int(np.isfinite(s).sum()),
            spearman_vs_y=_spearman(y, s),
            spearman_vs_base_residual=_spearman(resid, s),
            mean_signal=float(np.nanmean(s)),
            window="±500nt around protospacer midpoint",
            cell_type="K562",
        ))
        # Does adding chromatin help human base CV?
        ok = np.isfinite(s) & np.isfinite(y)
        Xc = np.c_[Xb[ok], s[ok]]
        fam = np.array(["base"] * Xb.shape[1] + ["chromatin"])
        names = [f"b{i}" for i in range(Xb.shape[1])] + [tname]
        res = cross_validate(
            Xc, y[ok], names, fam, seeds=(SEED,),
            tag=f"base_plus_{tname}", experiment=f"base_plus_{tname}", verbose=True,
        )
        rows[-1]["base_cv"] = 0.381879  # from Tier-1 ladder
        rows[-1]["base_plus_track_cv"] = res["summary"]["spearman"]["mean"]
        rows[-1]["delta_cv"] = rows[-1]["base_plus_track_cv"] - rows[-1]["base_cv"]

    # Both tracks together
    ok = np.isfinite(merged["K562_DNase"]) & np.isfinite(merged["K562_ATAC"]) & np.isfinite(y)
    Xc = np.c_[Xb[ok], merged.loc[ok, "K562_DNase"], merged.loc[ok, "K562_ATAC"]]
    fam = np.array(["base"] * Xb.shape[1] + ["chromatin", "chromatin"])
    names = [f"b{i}" for i in range(Xb.shape[1])] + ["K562_DNase", "K562_ATAC"]
    res = cross_validate(
        Xc, y[ok], names, fam, seeds=(SEED,),
        tag="base_plus_dnase_atac", experiment="base_plus_dnase_atac", verbose=True,
    )
    rows.append(dict(
        track="K562_DNase+ATAC",
        n=int(ok.sum()),
        spearman_vs_y=float("nan"),
        spearman_vs_base_residual=float("nan"),
        mean_signal=float("nan"),
        window="±500nt",
        cell_type="K562",
        base_cv=0.381879,
        base_plus_track_cv=res["summary"]["spearman"]["mean"],
        delta_cv=res["summary"]["spearman"]["mean"] - 0.381879,
    ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "human_chromatin.csv", index=False)
    # keep per-guide signals for later
    merged[["sgRNAID", "chrom", "left", "right", "y", "K562_DNase", "K562_ATAC"]].to_csv(
        CACHE / "human_chromatin_signals.csv", index=False
    )
    print(df.to_string(index=False), flush=True)
    return df


# ---------------------------------------------------------------------------
# Shared helpers for E1 / E2
# ---------------------------------------------------------------------------


def _ecoli_base_a():
    """Return (X_full, y, names, family_of, spacer_gc, ids) for E. coli."""
    cache_x = config.INTERIM / "human_boundary" / "X_ecoli_base_a.npy"
    cache_m = config.INTERIM / "human_boundary" / "meta_ecoli_base_a.json"
    if cache_x.exists() and cache_m.exists():
        meta = json.loads(cache_m.read_text())
        print(f"  cached E. coli base+A {meta['n']} x {meta['n_full']}", flush=True)
        return (
            np.load(cache_x), np.asarray(meta["y"], float),
            list(meta["full_names"]), list(meta["family_of"]),
            np.asarray(meta["spacer_gc"], float), list(meta["ids"]),
        )

    idx = pd.read_csv(config.INTERIM / "guide_index.csv")
    has = (
        idx["protospacer"].notna()
        & (idx["protospacer"].astype(str).str.len() == 20)
        & idx.get("located", True).astype(bool)
        & idx.get("unique_in_genome", True).astype(bool)
    )
    idx = idx.loc[has].reset_index(drop=True)
    print(f"  E. coli featurising portable base for {len(idx):,} ...", flush=True)
    base, base_names = featurise.matrix(idx["protospacer"].tolist(), verbose=True)
    base = np.asarray(base, np.float32)
    flank = pd.read_csv(config.FEATURE_DIR / "a_flank.csv")
    flank = flank.set_index(config.ID_COL).reindex(idx[config.ID_COL])
    vals = flank.to_numpy(dtype=np.float32)
    good = ~np.all(np.isnan(vals), axis=0)
    flank_names = [c for c, k in zip(flank.columns, good) if k]
    Xf = vals[:, good]
    X = np.hstack([base, Xf])
    names = list(base_names) + flank_names
    fam = ["base"] * len(base_names) + ["a_flank"] * len(flank_names)
    y = pd.to_numeric(idx["cut_score"], errors="coerce").to_numpy(float)
    gc = np.array([(s.count("G") + s.count("C")) / 20 for s in idx["protospacer"]], float)
    ids = idx[config.ID_COL].astype(str).tolist()
    meta = dict(
        y=y.tolist(), full_names=names, family_of=fam,
        spacer_gc=gc.tolist(), ids=ids, n=len(ids), n_full=int(X.shape[1]),
    )
    np.save(cache_x, X)
    cache_m.write_text(json.dumps(meta))
    return X, y, names, fam, gc, ids


def _human_base_a():
    meta = json.loads((CACHE / "meta_human.json").read_text())
    X = np.load(CACHE / "X_human_base_a.npy")
    y = np.asarray(meta["y"], float)
    names = list(meta["full_names"])
    fam = list(meta["family_of"])
    gc = np.asarray(meta["spacer_gc"], float)
    ids = list(meta["ids"])
    return X, y, names, fam, gc, ids


def _align(Xa, na, Xb, nb):
    shared = [c for c in na if c in set(nb)]
    ia = {c: i for i, c in enumerate(na)}
    ib = {c: i for i, c in enumerate(nb)}
    A = Xa[:, [ia[c] for c in shared]].astype(np.float32)
    B = Xb[:, [ib[c] for c in shared]].astype(np.float32)
    return A, B, shared


def _transfer(Xtr, ytr, Xte, yte, seed=SEED, n_top=N_TOP, **xgb_kw):
    A, B = _impute(Xtr, Xte)
    m0 = _xgb(seed=seed, n_estimators=300, **{k: v for k, v in xgb_kw.items() if k != "n_estimators"})
    m0.fit(A, ytr)
    top = np.argsort(m0.feature_importances_)[::-1][:n_top]
    m = _xgb(seed=seed, **xgb_kw)
    m.fit(A[:, top], ytr)
    pred = np.asarray(m.predict(B[:, top])).ravel()
    return pred, _spearman(yte, pred)


# ---------------------------------------------------------------------------
# E1 — Family A both sides
# ---------------------------------------------------------------------------


def run_e1_family_a_transfer():
    print("=== E1 cross-kingdom with Family A ===", flush=True)
    Xe, ye, ne, fe, gce, _ = _ecoli_base_a()
    Xh, yh, nh, fh, gch, _ = _human_base_a()

    # arms: base-only shared, a_flank-only shared, base+A shared
    def subset(names, fam, want):
        if want == "base":
            return [n for n, f in zip(names, fam) if f == "base"]
        if want == "a_flank":
            return [n for n, f in zip(names, fam) if f == "a_flank"]
        return list(names)

    rows = []
    for arm in ("base", "a_flank", "base+a_flank"):
        ne_s = subset(ne, fe, arm if arm != "base+a_flank" else "all")
        nh_s = subset(nh, fh, arm if arm != "base+a_flank" else "all")
        # Restrict name lists then align
        ie = {c: i for i, c in enumerate(ne)}
        ih = {c: i for i, c in enumerate(nh)}
        Xe_a = Xe[:, [ie[c] for c in ne_s]]
        Xh_a = Xh[:, [ih[c] for c in nh_s]]
        A, B, shared = _align(Xe_a, ne_s, Xh_a, nh_s)
        print(f"  arm {arm}: shared {len(shared)} cols", flush=True)
        ok_e = np.isfinite(ye)
        ok_h = np.isfinite(yh)
        pred_eh, rho_eh = _transfer(A[ok_e], ye[ok_e], B[ok_h], yh[ok_h])
        pred_he, rho_he = _transfer(B[ok_h], yh[ok_h], A[ok_e], ye[ok_e])
        rows.append(dict(
            arm=arm, n_shared=len(shared),
            ecoli_to_human=rho_eh, human_to_ecoli=rho_he,
            n_ecoli=int(ok_e.sum()), n_human=int(ok_h.sum()),
        ))
        print(f"    E→H {rho_eh:+.4f}  H→E {rho_he:+.4f}", flush=True)

    # Also: human-fitted knobs on human→ecoli (n_estimators=1000, max_depth=6)
    ne_s = subset(ne, fe, "all")
    nh_s = subset(nh, fh, "all")
    ie = {c: i for i, c in enumerate(ne)}
    ih = {c: i for i, c in enumerate(nh)}
    A, B, shared = _align(
        Xe[:, [ie[c] for c in ne_s]], ne_s,
        Xh[:, [ih[c] for c in nh_s]], nh_s,
    )
    ok_e, ok_h = np.isfinite(ye), np.isfinite(yh)
    _, rho_he_fit = _transfer(
        B[ok_h], yh[ok_h], A[ok_e], ye[ok_e],
        n_estimators=1000, max_depth=6, learning_rate=0.3, subsample=1.0, colsample_bytree=1.0,
        n_top=400,
    )
    rows.append(dict(
        arm="base+a_flank_human_fitted", n_shared=len(shared),
        ecoli_to_human=float("nan"), human_to_ecoli=rho_he_fit,
        n_ecoli=int(ok_e.sum()), n_human=int(ok_h.sum()),
    ))
    print(f"  H→E human_fitted {rho_he_fit:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "family_a_kingdom_transfer.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return df


# ---------------------------------------------------------------------------
# E2 — GC match / residualise
# ---------------------------------------------------------------------------


def run_e2_gc_controls():
    print("=== E2 GC-matched / residualised transfer ===", flush=True)
    from sklearn.linear_model import LinearRegression

    Xe, ye, ne, fe, gce, _ = _ecoli_base_a()
    Xh, yh, nh, fh, gch, _ = _human_base_a()

    results = []
    for arm, want in (("shared_style_base", "base"), ("base+a_flank", "all")):
        ne_s = [n for n, f in zip(ne, fe) if (want == "all" or f == "base")]
        nh_s = [n for n, f in zip(nh, fh) if (want == "all" or f == "base")]
        ie = {c: i for i, c in enumerate(ne)}
        ih = {c: i for i, c in enumerate(nh)}
        A, B, shared = _align(
            Xe[:, [ie[c] for c in ne_s]], ne_s,
            Xh[:, [ih[c] for c in nh_s]], nh_s,
        )
        ok_e = np.isfinite(ye) & np.isfinite(gce)
        ok_h = np.isfinite(yh) & np.isfinite(gch)
        Ae, ye_, gce_ = A[ok_e], ye[ok_e], gce[ok_e]
        Bh, yh_, gch_ = B[ok_h], yh[ok_h], gch[ok_h]

        pred, rho = _transfer(Ae, ye_, Bh, yh_)
        results.append(dict(arm=arm, control="raw", spearman=rho, n=len(yh_)))

        # Residualise both pred and y against GC+GC²
        Z = np.c_[gch_, gch_ ** 2]
        ry = yh_ - LinearRegression().fit(Z, yh_).predict(Z)
        rp = pred - LinearRegression().fit(Z, pred).predict(Z)
        results.append(dict(
            arm=arm, control="residualise_gc2",
            spearman=_spearman(ry, rp), n=len(yh_),
        ))

        # Matched: keep human guides whose GC is within E. coli GC [5%, 95%]
        lo, hi = np.quantile(gce_, [0.05, 0.95])
        m = (gch_ >= lo) & (gch_ <= hi)
        results.append(dict(
            arm=arm, control=f"gc_in_ecoli_5_95_[{lo:.2f},{hi:.2f}]",
            spearman=_spearman(yh_[m], pred[m]), n=int(m.sum()),
        ))

        # Stratified: mean of per-decile transfer (human GC deciles)
        qs = np.quantile(gch_, np.linspace(0, 1, 11))
        dec_rhos = []
        for i in range(10):
            if i < 9:
                msk = (gch_ >= qs[i]) & (gch_ < qs[i + 1])
            else:
                msk = (gch_ >= qs[i]) & (gch_ <= qs[i + 1])
            if msk.sum() < 30:
                continue
            dec_rhos.append(_spearman(yh_[msk], pred[msk]))
        results.append(dict(
            arm=arm, control="mean_gc_decile_rho",
            spearman=float(np.mean(dec_rhos)) if dec_rhos else float("nan"),
            n=len(yh_),
        ))

        # Matched sampling: for each human guide, weight is not needed —
        # subsample E. coli train to match human GC histogram (quantile bins)
        # Train on GC-reweighted ecoli via stratified subsample equal counts
        bins = np.linspace(0, 1, 11)
        rng = np.random.default_rng(SEED)
        target_per = max(50, len(ye_) // 20)
        keep = []
        for i in range(10):
            m_e = (gce_ >= bins[i]) & (gce_ < bins[i + 1] if i < 9 else gce_ <= bins[i + 1])
            m_h = (gch_ >= bins[i]) & (gch_ < bins[i + 1] if i < 9 else gch_ <= bins[i + 1])
            n_h = int(m_h.sum())
            idx_e = np.where(m_e)[0]
            if len(idx_e) == 0 or n_h == 0:
                continue
            take = min(len(idx_e), max(n_h, target_per))
            keep.append(rng.choice(idx_e, size=take, replace=False))
        if keep:
            sel = np.concatenate(keep)
            pred2, rho2 = _transfer(Ae[sel], ye_[sel], Bh, yh_)
            results.append(dict(
                arm=arm, control="train_subsample_match_human_gc_hist",
                spearman=rho2, n=len(yh_),
            ))

        print(f"  {arm}: raw={rho:+.4f}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT / "gc_matched_transfer.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return df


def main():
    d1 = run_d1_chromatin()
    e1 = run_e1_family_a_transfer()
    e2 = run_e2_gc_controls()
    summary = dict(
        d1=d1.to_dict(orient="records"),
        e1=e1.to_dict(orient="records"),
        e2=e2.to_dict(orient="records"),
    )
    (OUT / "tier2_tier3_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("DONE tier2/tier3", flush=True)


if __name__ == "__main__":
    main()
