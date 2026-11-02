#!/usr/bin/env python3
"""Tier-1 human boundary experiments.

1. Decode protospacers from the Noshay human matrix (QCT electron columns).
2. Locate unique NGG-flanked copies on hg38 (stream chromosomes).
3. Build Family A (a_flank) on located guides.
4. Human-only ladder: shared-cols / regenerated base / base+A.
5. Transfer autopsy: does an E. coli model fail mainly via the GC rule?

Outputs under Jacky/human-boundary/results/.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(os.environ.get("RSI09_ROOT", "/workspace/Runjia")).resolve()
os.environ["RSI09_ROOT"] = str(ROOT)
(ROOT / "external_data").mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from sgrna import config, featurise, genome, qct  # noqa: E402
from sgrna.features import a_flank  # noqa: E402
from sgrna.run_ablation import cross_validate, _impute  # noqa: E402

OUT = Path("/workspace/Jacky/human-boundary/results")
CACHE = config.INTERIM / "human_boundary"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

HUMAN_MATRIX = config.RAW / "human_feature_matrix.csv"
HG38_FA = config.REFERENCE / "hg38.fa"
HG38_GZ = config.REFERENCE / "hg38.fa.gz"
FLANK = 1000
SEED = 41
ELECTRONS_TO_BASE = qct.ELECTRONS_TO_BASE
REVCOMP = genome.revcomp


def _spearman(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3:
        return float("nan")
    r = spearmanr(y[m], p[m]).statistic
    return float(r) if np.isfinite(r) else float("nan")


# ---------------------------------------------------------------------------
# 1. Decode
# ---------------------------------------------------------------------------


def decode_human(force: bool = False) -> pd.DataFrame:
    path = CACHE / "human_decoded.csv"
    if path.exists() and not force:
        print(f"  cached decode {path}", flush=True)
        return pd.read_csv(path)

    print(f"  reading QCT electron columns from {HUMAN_MATRIX.name} ...", flush=True)
    cols = [config.ID_COL, config.TARGET_COL] + [
        f"p{i}monomer.No.electronsraw" for i in range(1, 21)
    ]
    df = pd.read_csv(
        HUMAN_MATRIX, header=config.BASE_MATRIX_HEADER_ROW, usecols=cols, low_memory=False
    )
    mat = df[[f"p{i}monomer.No.electronsraw" for i in range(1, 21)]].to_numpy(float)
    seqs = ["".join(ELECTRONS_TO_BASE.get(v, "N") for v in row) for row in mat]
    out = pd.DataFrame({
        config.ID_COL: df[config.ID_COL].astype(str),
        "y": pd.to_numeric(df[config.TARGET_COL], errors="coerce"),
        "protospacer": seqs,
    })
    out["decoded_ok"] = ~out["protospacer"].str.contains("N")
    out.to_csv(path, index=False)
    print(f"  decoded {len(out):,}  ok={int(out.decoded_ok.sum()):,}", flush=True)
    return out


# ---------------------------------------------------------------------------
# 2. Locate on hg38
# ---------------------------------------------------------------------------


def iter_fasta(path: Path):
    """Yield (header, sequence) from .fa or .fa.gz."""
    opener = gzip.open if str(path).endswith(".gz") else open
    mode = "rt"
    name, chunks = None, []
    with opener(path, mode) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks).upper()
                name = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if name is not None:
            yield name, "".join(chunks).upper()


def slice_linear(g: str, start: int, end: int) -> str:
    """1-based inclusive; clamp to chromosome; pad with N outside."""
    n = len(g)
    out = []
    for pos in range(start, end + 1):
        if 1 <= pos <= n:
            out.append(g[pos - 1])
        else:
            out.append("N")
    return "".join(out)


def linear_guide_context(left: int, right: int, strand: str, flank: int, g: str) -> dict:
    if strand == "+":
        upstream = slice_linear(g, left - flank, left - 1)
        proto = slice_linear(g, left, right)
        pam = slice_linear(g, right + 1, right + 3)
        downstream = slice_linear(g, right + 4, right + 3 + flank)
    else:
        upstream = REVCOMP(slice_linear(g, right + 1, right + flank))
        proto = REVCOMP(slice_linear(g, left, right))
        pam = REVCOMP(slice_linear(g, left - 3, left - 1))
        downstream = REVCOMP(slice_linear(g, left - 3 - flank, left - 4))
    return dict(upstream=upstream, protospacer=proto, pam=pam, downstream=downstream)


def locate_on_hg38(decoded: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Single-pass locate: scan each primary chrom, cut flanks, discard chrom.

    Keeps RAM under ~1× largest chromosome (~250 MB), not full hg38.
    """
    path = CACHE / "human_located.csv"
    if path.exists() and not force:
        print(f"  cached locate {path}", flush=True)
        return pd.read_csv(path)

    fasta = HG38_FA if HG38_FA.exists() else HG38_GZ
    if not fasta.exists():
        raise FileNotFoundError(
            f"Need {HG38_FA} or {HG38_GZ}. Download hg38.fa.gz from UCSC."
        )

    wanted = set(decoded.loc[decoded.decoded_ok, "protospacer"].astype(str))
    # proto -> list of hit dicts (without flanks yet); flanks filled when unique
    hits: dict[str, list[dict]] = {s: [] for s in wanted}
    primary = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}
    # stash flank contexts for the first hit; invalidate if a second appears
    ctx_by_proto: dict[str, dict] = {}
    multi = set()

    print(f"  locating {len(wanted):,} unique protospacers on {fasta.name} "
          f"(single-pass) ...", flush=True)
    t0 = time.time()
    for chrom, seq in iter_fasta(fasta):
        chrom_n = chrom
        if chrom not in primary:
            chrom_n = (f"chr{chrom}" if chrom.isdigit() or chrom in ("X", "Y", "M")
                       else chrom)
        if chrom_n not in primary:
            continue
        n = len(seq)
        print(f"    scanning {chrom_n} ({n:,} bp) ...", flush=True)

        def record(proto, strand, left, right, pam):
            if proto in multi:
                return
            if pam[1:] != "GG":
                return
            h = dict(chrom=chrom_n, strand=strand, left=left, right=right, pam=pam)
            if proto in ctx_by_proto:
                # second genomic copy → ambiguous flanks; drop
                multi.add(proto)
                ctx_by_proto.pop(proto, None)
                hits[proto].append(h)
                return
            hits[proto].append(h)
            ctx = linear_guide_context(left, right, strand, FLANK, seq)
            ctx_by_proto[proto] = dict(h, **ctx)

        i = seq.find("GG")
        while i >= 0:
            pam0 = i - 1
            proto0 = pam0 - 20
            if proto0 >= 0:
                proto = seq[proto0: proto0 + 20]
                if proto in wanted:
                    left = proto0 + 1
                    right = left + 19
                    pam = seq[right: right + 3]
                    if len(pam) == 3:
                        record(proto, "+", left, right, pam)
            i = seq.find("GG", i + 1)

        i = seq.find("CC")
        while i >= 0:
            proto0 = i + 3
            if proto0 + 20 <= n:
                proto = REVCOMP(seq[proto0: proto0 + 20])
                if proto in wanted:
                    left = proto0 + 1
                    right = left + 19
                    pam = REVCOMP(seq[i: i + 3])
                    record(proto, "-", left, right, pam)
            i = seq.find("CC", i + 1)
        print(f"      unique-so-far={len(ctx_by_proto):,}  multi={len(multi):,}  "
              f"elapsed {time.time()-t0:.0f}s", flush=True)
        del seq  # free chromosome

    audit = dict(
        n_wanted=len(wanted),
        n_unique=len(ctx_by_proto),
        n_multi=len(multi),
        n_missing=len(wanted) - len(ctx_by_proto) - len(multi),
    )
    print(f"  locate audit: {json.dumps(audit)}", flush=True)
    (OUT / "human_locate_audit.json").write_text(json.dumps(audit, indent=2))

    rows = []
    for _, r in decoded.iterrows():
        if not r["decoded_ok"]:
            continue
        proto = r["protospacer"]
        ctx = ctx_by_proto.get(proto)
        if ctx is None:
            continue
        rows.append(dict(
            sgRNAID=r[config.ID_COL],
            y=float(r["y"]) if np.isfinite(r["y"]) else np.nan,
            protospacer=ctx["protospacer"],
            pam=ctx["pam"],
            upstream=ctx["upstream"],
            downstream=ctx["downstream"],
            chrom=ctx["chrom"],
            strand=ctx["strand"],
            left=ctx["left"],
            right=ctx["right"],
            located=True,
            unique=True,
        ))

    loc = pd.DataFrame(rows)
    loc.to_csv(path, index=False)
    print(f"  located unique usable: {len(loc):,}", flush=True)
    return loc


# ---------------------------------------------------------------------------
# 3. Build matrices
# ---------------------------------------------------------------------------


def build_human_matrices(loc: pd.DataFrame, force: bool = False):
    x_base = CACHE / "X_human_base.npy"
    x_full = CACHE / "X_human_base_a.npy"
    meta_path = CACHE / "meta_human.json"
    flank_path = CACHE / "a_flank_human.csv"

    if x_full.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        print(f"  cached matrices {meta['n']} x {meta['n_full']}", flush=True)
        return (
            np.load(x_base), np.load(x_full),
            np.asarray(meta["y"], float),
            meta["base_names"], meta["full_names"], meta,
        )

    print(f"  featurising base for {len(loc):,} human guides ...", flush=True)
    base, base_names = featurise.matrix(loc["protospacer"].tolist(), verbose=True)
    base = np.asarray(base, np.float32)
    base_names = list(base_names)

    print("  building a_flank ...", flush=True)
    idx = loc.copy()
    idx[config.ID_COL] = idx["sgRNAID"]
    block = a_flank.build(idx)
    block.to_csv(flank_path, index=False)
    block = block.set_index(config.ID_COL).reindex(idx[config.ID_COL])
    vals = block.to_numpy(dtype=np.float32)
    good = ~np.all(np.isnan(vals), axis=0)
    flank_names = [c for c, k in zip(block.columns, good) if k]
    Xf = vals[:, good]
    Xfull = np.hstack([base, Xf])
    full_names = base_names + flank_names
    family_of = ["base"] * len(base_names) + ["a_flank"] * len(flank_names)

    y = idx["y"].to_numpy(float)
    meta = dict(
        y=y.tolist(),
        base_names=base_names,
        full_names=full_names,
        family_of=family_of,
        n_base=len(base_names),
        n_full=int(Xfull.shape[1]),
        n=int(len(idx)),
        ids=idx[config.ID_COL].astype(str).tolist(),
        spacer_gc=[(s.count("G") + s.count("C")) / 20 for s in idx["protospacer"]],
    )
    np.save(x_base, base)
    np.save(x_full, Xfull)
    meta_path.write_text(json.dumps(meta))
    print(f"  built base {base.shape}  full {Xfull.shape}", flush=True)
    return base, Xfull, y, base_names, full_names, meta


# ---------------------------------------------------------------------------
# 4. Human-only ladder
# ---------------------------------------------------------------------------


def run_ladder(X_base, X_full, y, base_names, full_names, family_of, meta):
    print("=== human-only ladder ===", flush=True)
    # Also shared-column baseline from raw matrix on the SAME ids
    rows = []
    arms = {
        "regenerated_base": (X_base, base_names, ["base"] * len(base_names)),
        "base_plus_a_flank": (X_full, full_names, family_of),
    }
    # a_flank only
    fam = np.asarray(family_of)
    flank_idx = np.where(fam == "a_flank")[0]
    if len(flank_idx):
        arms["a_flank_only"] = (
            X_full[:, flank_idx],
            list(np.asarray(full_names)[flank_idx]),
            ["a_flank"] * len(flank_idx),
        )

    ok = np.isfinite(y)
    for name, (X, names, fof) in arms.items():
        print(f"  CV {name} n={ok.sum()} cols={X.shape[1]}", flush=True)
        res = cross_validate(
            X[ok], y[ok], list(names), np.asarray(fof),
            seeds=(SEED,), groups=None,
            tag=f"human_{name}", experiment=name, verbose=True,
        )
        s = res["summary"]
        rows.append(dict(
            arm=name, n=int(ok.sum()), n_cols=int(X.shape[1]),
            spearman=s["spearman"]["mean"], r2=s["r2"]["mean"],
        ))

    # shared-column arm from published human matrix (same IDs)
    print("  building shared-column matrix for located IDs ...", flush=True)
    eco = pd.read_csv(
        config.BASE_MATRIX, header=config.BASE_MATRIX_HEADER_ROW, nrows=0
    )
    hum = pd.read_csv(HUMAN_MATRIX, header=config.BASE_MATRIX_HEADER_ROW, low_memory=False)
    drop = {config.TARGET_COL, config.ID_COL}
    shared = sorted((set(eco.columns) & set(hum.columns)) - drop)
    ids = list(meta["ids"])
    hum = hum.set_index(config.ID_COL).loc[ids]
    Xs = hum[shared].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    print(f"  CV shared_published_cols n={ok.sum()} cols={Xs.shape[1]}", flush=True)
    res = cross_validate(
        Xs[ok], y[ok], shared, np.array(["shared"] * len(shared)),
        seeds=(SEED,), groups=None,
        tag="human_shared", experiment="shared", verbose=True,
    )
    s = res["summary"]
    rows.append(dict(
        arm="shared_published_cols", n=int(ok.sum()), n_cols=len(shared),
        spearman=s["spearman"]["mean"], r2=s["r2"]["mean"],
    ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "human_ladder.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return df


# ---------------------------------------------------------------------------
# 5. Transfer autopsy
# ---------------------------------------------------------------------------


def transfer_autopsy(meta, X_full, y, full_names, family_of):
    """Train on E. coli (shared cols), predict human; stratify errors by GC."""
    print("=== transfer autopsy ===", flush=True)
    import xgboost as xgb

    eco = pd.read_csv(
        config.BASE_MATRIX, header=config.BASE_MATRIX_HEADER_ROW, low_memory=False
    )
    hum = pd.read_csv(HUMAN_MATRIX, header=config.BASE_MATRIX_HEADER_ROW, low_memory=False)
    drop = {config.TARGET_COL, config.ID_COL}
    shared = sorted((set(eco.columns) & set(hum.columns)) - drop)

    ye = pd.to_numeric(eco[config.TARGET_COL], errors="coerce").to_numpy(float)
    Xe = eco[shared].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    ok_e = np.isfinite(ye)
    Xe, ye = Xe[ok_e], ye[ok_e]

    ids = list(meta["ids"])
    hum_i = hum.set_index(config.ID_COL).loc[ids]
    Xh = hum_i[shared].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    yh = np.asarray(y, float)
    gc = np.asarray(meta["spacer_gc"], float)

    A, B = _impute(Xe, Xh)
    # select 300
    m = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED, n_jobs=-1, tree_method="hist",
        importance_type="gain",
    )
    m.fit(A, ye)
    top = np.argsort(m.feature_importances_)[::-1][:300]
    m2 = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, random_state=SEED, n_jobs=-1, tree_method="hist",
    )
    m2.fit(A[:, top], ye)
    pred = np.asarray(m2.predict(B[:, top])).ravel()

    overall = _spearman(yh, pred)
    # GC deciles
    rows = []
    qs = np.quantile(gc[np.isfinite(gc)], np.linspace(0, 1, 11))
    for i in range(10):
        lo, hi = qs[i], qs[i + 1]
        if i < 9:
            msk = (gc >= lo) & (gc < hi) & np.isfinite(yh)
        else:
            msk = (gc >= lo) & (gc <= hi) & np.isfinite(yh)
        rows.append(dict(
            gc_decile=i + 1,
            gc_lo=float(lo), gc_hi=float(hi),
            n=int(msk.sum()),
            mean_gc=float(np.mean(gc[msk])) if msk.any() else np.nan,
            transfer_rho=_spearman(yh[msk], pred[msk]),
            mean_y=float(np.mean(yh[msk])) if msk.any() else np.nan,
            mean_pred=float(np.mean(pred[msk])) if msk.any() else np.nan,
        ))

    # Residualise human y and pred against GC / GC^2 — does transfer improve?
    ok = np.isfinite(yh) & np.isfinite(pred) & np.isfinite(gc)
    Z = np.c_[gc[ok], gc[ok] ** 2]
    # project out GC from y and pred
    from sklearn.linear_model import LinearRegression
    ry = yh[ok] - LinearRegression().fit(Z, yh[ok]).predict(Z)
    rp = pred[ok] - LinearRegression().fit(Z, pred[ok]).predict(Z)
    rho_resid = _spearman(ry, rp)

    # Match on GC: bin human and ecoli? Simpler: within each human GC decile,
    # report rho — already above. Also: correlation of (pred error) with GC.
    err = pred - yh
    rho_err_gc = _spearman(gc, err)

    out = dict(
        overall_transfer_rho=overall,
        transfer_rho_after_residualising_gc2=rho_resid,
        spearman_error_vs_spacer_gc=rho_err_gc,
        n_human=int(len(yh)),
        n_ecoli_train=int(len(ye)),
        n_shared_cols=len(shared),
        note=("If residualising GC² raises transfer rho toward zero/positive, "
              "the E. coli GC rule is a load-bearing failure mode."),
    )
    (OUT / "transfer_autopsy.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(OUT / "transfer_autopsy_by_gc_decile.csv", index=False)
    print(json.dumps(out, indent=2), flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    # Optional: does Family A help *within* human enough to matter for the story?
    if X_full is not None:
        fam = np.asarray(family_of)
        base_i = np.where(fam == "base")[0]
        both_i = np.arange(X_full.shape[1])
        for label, cols in (("human_base_cv", base_i), ("human_base_a_cv", both_i)):
            res = cross_validate(
                X_full[:, cols][ok], yh[ok],
                list(np.asarray(full_names)[cols]), fam[cols],
                seeds=(SEED,), tag=label, experiment=label, verbose=True,
            )
            out[label] = res["summary"]["spearman"]["mean"]
        (OUT / "transfer_autopsy.json").write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# GC shape refresh on spacer GC (authoritative)
# ---------------------------------------------------------------------------


def gc_shape_report(decoded: pd.DataFrame):
    from sklearn.linear_model import LinearRegression
    rows = []
    for label, path, decode_fn in (
        ("human", None, None),
    ):
        pass
    # human from decoded
    d = decoded[decoded.decoded_ok].copy()
    d["gc"] = d["protospacer"].map(lambda s: (s.count("G") + s.count("C")) / 20)
    y = d["y"].to_numpy(float)
    g = d["gc"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(g)
    g, y = g[ok], y[ok]
    Z1, Z2 = g.reshape(-1, 1), np.c_[g, g ** 2]
    r1 = LinearRegression().fit(Z1, y)
    r2 = LinearRegression().fit(Z2, y)
    a, b = r2.coef_[1], r2.coef_[0]
    turn = float(-b / (2 * a)) if a != 0 else float("nan")
    pred1, pred2 = r1.predict(Z1), r2.predict(Z2)
    out = dict(
        label="human_spacer_gc",
        n=int(len(g)),
        linear_rho=_spearman(y, pred1),
        linear_r2=float(1 - np.mean((y - pred1) ** 2) / np.var(y)),
        quadratic_r2=float(1 - np.mean((y - pred2) ** 2) / np.var(y)),
        turning_point=turn,
        mean_gc=float(np.mean(g)),
        turning_point_data_percentile=float(np.mean(g <= turn)),
    )
    (OUT / "human_gc_shape.json").write_text(json.dumps(out, indent=2))
    print("GC shape:", json.dumps(out, indent=2), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-locate", action="store_true",
                    help="decode + autopsy only (no hg38)")
    args = ap.parse_args()

    if not HUMAN_MATRIX.exists():
        raise SystemExit(f"missing {HUMAN_MATRIX}")
    if not config.QCT_TABLES.exists():
        raise SystemExit("missing qct_lookup_tables.json — run build_guide_index")
    if not (config.INTERIM / "v_column_mapping.json").exists():
        raise SystemExit("missing v_column_mapping.json — run decode_v_columns")

    print("=== decode human ===", flush=True)
    decoded = decode_human(force=args.force)
    gc_shape_report(decoded)

    if args.skip_locate:
        print("skip-locate: building meta from decode for autopsy only", flush=True)
        # autopsy doesn't need locate — use all decoded rows
        # fabricate meta from human matrix order
        print("=== transfer autopsy (all decoded human rows) ===", flush=True)
        # lightweight path
        import xgboost as xgb
        from sklearn.linear_model import LinearRegression
        eco = pd.read_csv(config.BASE_MATRIX, header=config.BASE_MATRIX_HEADER_ROW, low_memory=False)
        hum = pd.read_csv(HUMAN_MATRIX, header=config.BASE_MATRIX_HEADER_ROW, low_memory=False)
        drop = {config.TARGET_COL, config.ID_COL}
        shared = sorted((set(eco.columns) & set(hum.columns)) - drop)
        ye = pd.to_numeric(eco[config.TARGET_COL], errors="coerce").to_numpy(float)
        Xe = eco[shared].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        ok_e = np.isfinite(ye)
        Xe, ye = Xe[ok_e], ye[ok_e]
        yh = pd.to_numeric(hum[config.TARGET_COL], errors="coerce").to_numpy(float)
        Xh = hum[shared].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        # spacer GC from decode aligned by ID
        dec = decoded.set_index(config.ID_COL)
        hum_ids = hum[config.ID_COL].astype(str)
        gc = []
        for gid in hum_ids:
            if gid in dec.index and bool(dec.loc[gid, "decoded_ok"]):
                s = str(dec.loc[gid, "protospacer"])
                gc.append((s.count("G") + s.count("C")) / 20)
            else:
                gc.append(np.nan)
        gc = np.asarray(gc, float)
        ok = np.isfinite(yh) & np.isfinite(Xh).any(axis=1) & np.isfinite(gc)
        Xh, yh, gc = Xh[ok], yh[ok], gc[ok]
        A, B = _impute(Xe, Xh)
        m = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED, n_jobs=-1, tree_method="hist",
            importance_type="gain")
        m.fit(A, ye)
        top = np.argsort(m.feature_importances_)[::-1][:300]
        m2 = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, random_state=SEED, n_jobs=-1, tree_method="hist")
        m2.fit(A[:, top], ye)
        pred = np.asarray(m2.predict(B[:, top])).ravel()
        Z = np.c_[gc, gc ** 2]
        ry = yh - LinearRegression().fit(Z, yh).predict(Z)
        rp = pred - LinearRegression().fit(Z, pred).predict(Z)
        out = dict(
            overall_transfer_rho=_spearman(yh, pred),
            transfer_rho_after_residualising_gc2=_spearman(ry, rp),
            spearman_error_vs_spacer_gc=_spearman(gc, pred - yh),
            n_human=int(len(yh)), n_ecoli_train=int(len(ye)),
            n_shared_cols=len(shared),
        )
        rows = []
        qs = np.quantile(gc, np.linspace(0, 1, 11))
        for i in range(10):
            lo, hi = qs[i], qs[i + 1]
            msk = (gc >= lo) & (gc < hi) if i < 9 else (gc >= lo) & (gc <= hi)
            rows.append(dict(
                gc_decile=i + 1, gc_lo=float(lo), gc_hi=float(hi),
                n=int(msk.sum()), mean_gc=float(gc[msk].mean()),
                transfer_rho=_spearman(yh[msk], pred[msk]),
            ))
        (OUT / "transfer_autopsy.json").write_text(json.dumps(out, indent=2))
        pd.DataFrame(rows).to_csv(OUT / "transfer_autopsy_by_gc_decile.csv", index=False)
        print(json.dumps(out, indent=2), flush=True)
        print("DONE (skip-locate)", flush=True)
        return

    print("=== locate on hg38 ===", flush=True)
    loc = locate_on_hg38(decoded, force=args.force)
    if len(loc) < 500:
        raise SystemExit(f"too few located guides: {len(loc)}")

    print("=== build matrices ===", flush=True)
    Xb, Xf, y, bn, fn, meta = build_human_matrices(loc, force=args.force)
    family_of = meta["family_of"]
    run_ladder(Xb, Xf, y, bn, fn, family_of, meta)
    transfer_autopsy(meta, Xf, y, fn, family_of)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
