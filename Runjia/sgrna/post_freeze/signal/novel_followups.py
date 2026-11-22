"""Two experiments that are ours rather than borrowed: species-relative composition, and the
resolution floor of the flank signal.

Neither copies a published design. Each asks a question the literature's framing makes available
but nobody in it has asked.

  relative    Family A's strongest column is `eng.flank.comp.up250.gc`, a RAW GC fraction. K-12 is
              50.79% GC and C. rodentium 54.72%, so a flank at 52% GC is above background in one
              genome and below it in the other: the feature carries the opposite sign of local
              enrichment across the two, and the cross-organism transfer that retains 88-91% does
              so despite this. Expressing each window as a percentile of its OWN genome's
              distribution of that same statistic should fix the sign.
              **Within one genome this cannot change anything** -- a percentile map of a single
              column is monotone and boosted trees are invariant to monotone transforms of
              individual features -- so the home cohort is run only as a check that it is exactly a
              no-op, and the real experiment is the transfer.

  resolution  Browne et al. searched input *length* and found the gains level off. The orthogonal
              question is *resolution*: how finely does the flank have to be described? Family A
              uses four nested windows per side; Finding 2 says the effect is a smooth gradient with
              no motif and no dependence on where the window edges fall. If that is right there
              should be a resolution floor -- a small number of bins beyond which nothing is gained
              -- and finding it bounds the kind of physical process that could produce the effect,
              as well as telling us how cheap the featuriser could be.

    python -m sgrna.post_freeze.signal.novel_followups --test resolution
    python -m sgrna.post_freeze.signal.novel_followups --test relative
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sgrna import config
from sgrna.harness.run_ablation import _impute, grouped_folds, make_selector

from ..tuning import tune_ecoli as T
from ..common import OUT

STRIDE = 37            # genome-wide window sampling stride; coprime with everything to avoid phase artefacts
BINS = (1, 2, 4, 8, 20, 50)


# ------------------------------------------------------------------ genome window distributions
def _seq(fasta) -> np.ndarray:
    raw = []
    with open(fasta) as fh:
        for line in fh:
            if not line.startswith(">"):
                raw.append(line.strip().upper())
    return np.frombuffer("".join(raw).encode("ascii"), np.uint8)


def window_dist(fasta, w: int, stat: str) -> np.ndarray:
    """Sorted distribution of `stat` over every w-nt window of the genome (sampled at STRIDE)."""
    a = _seq(fasta)
    if stat == "gc":
        hit = (a == ord("G")) | (a == ord("C"))
    elif stat == "purine":
        hit = (a == ord("A")) | (a == ord("G"))
    else:
        raise ValueError(stat)
    cs = np.concatenate([[0], np.cumsum(hit, dtype=np.int64)])
    starts = np.arange(0, len(a) - w, STRIDE)
    return np.sort((cs[starts + w] - cs[starts]) / w)


def to_percentile(vals: np.ndarray, dist: np.ndarray) -> np.ndarray:
    out = np.searchsorted(dist, vals) / len(dist)
    return np.where(np.isfinite(vals), out, np.nan)


# ------------------------------------------------------------------ experiment 1: species-relative
def _comp_cols(names):
    """(index, window, side, stat) for every raw GC / purine composition column of family A."""
    out = []
    for j, n in enumerate(names):
        p = n.split(".")
        if len(p) == 5 and p[0] == "eng" and p[1] == "flank" and p[2] == "comp" and p[4] in ("gc", "purine"):
            side_w = p[3]
            side, w = side_w[:2], int(side_w[2:])
            out.append((j, w, side, p[4]))
    return out


def relative(seed: int = 41):
    from sgrna.cohorts.curated_ablation import _base, _block
    from sgrna.screens import transfer

    X_ec, y_ec, names, protos = _base()
    A_ec, an = _block("a_flank", protos)
    X_ec = np.hstack([X_ec, A_ec]).astype(np.float32)
    all_names = list(names) + list(an)

    X_cr, y_cr, cr_names = transfer.build("TevSpCas9", genome_fasta=config.EXTERNAL / "C_rodentium" / "NC_013716.1.fasta")
    X_cr = np.asarray(X_cr, np.float32)
    assert list(cr_names) == all_names, "the two cohorts' column orders differ; cannot transfer column-wise"

    cols = _comp_cols(all_names)
    print(f"{len(cols)} raw composition columns to relativise", flush=True)
    ec_fa, cr_fa = config.GENOME_FASTA, config.EXTERNAL / "C_rodentium" / "NC_013716.1.fasta"
    Xr_ec, Xr_cr = X_ec.copy(), X_cr.copy()
    for j, w, side, stat in cols:
        d_ec = window_dist(ec_fa, w, stat)
        d_cr = window_dist(cr_fa, w, stat)
        Xr_ec[:, j] = to_percentile(X_ec[:, j], d_ec)
        Xr_cr[:, j] = to_percentile(X_cr[:, j], d_cr)
    print(f"genome GC: E. coli {_gc(ec_fa):.4f}, C. rodentium {_gc(cr_fa):.4f}", flush=True)

    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    cfg = dict(cfgs["d099x"], n_estimators=int(T.selected().loc["d099x", "k"]))
    rows = []
    for label, Xe, Xc in (("absolute (as shipped)", X_ec, X_cr), ("species-relative percentile", Xr_ec, Xr_cr)):
        A, _ = _impute(Xe, Xe[:1])
        sel = make_selector(seed); sel.fit(A, y_ec)
        top = np.argsort(sel.feature_importances_)[::-1][: int(cfg["nf"])]
        Atr, Bte = _impute(Xe[:, top], Xc[:, top])
        p = T.fit_predict(cfg, Atr, y_ec, Bte, seed)[cfg["n_estimators"]]
        rho_cr = float(spearmanr(y_cr, p).statistic)
        # home reference, same columns, gene-grouped CV, to confirm the monotone no-op
        home = []
        g, ok = _groups_home(protos)
        for tr, va in grouped_folds(g[ok], 5, seed):
            idx = np.where(ok)[0]
            a, b = _impute(Xe[idx[tr]][:, top], Xe[idx[va]][:, top])
            q = T.fit_predict(cfg, a, y_ec[idx[tr]], b, seed)[cfg["n_estimators"]]
            home.append(float(spearmanr(y_ec[idx[va]], q).statistic))
        rows.append(dict(columns=label, crod_rho=rho_cr, home_cv_rho=float(np.mean(home)),
                         n_crod=len(y_cr), n_home=int(ok.sum())))
        print(f"  {label:28s} C. rodentium {rho_cr:.4f}   home CV {np.mean(home):.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "novel_relative_composition.csv", index=False)
    d = out.set_index("columns")
    print(f"\n  transfer difference (relative - absolute): "
          f"{d.loc['species-relative percentile','crod_rho'] - d.loc['absolute (as shipped)','crod_rho']:+.4f}")
    print(f"  home difference (should be ~0 by monotone invariance): "
          f"{d.loc['species-relative percentile','home_cv_rho'] - d.loc['absolute (as shipped)','home_cv_rho']:+.4f}")
    return out


def _gc(fasta):
    a = _seq(fasta)
    return float(((a == ord("G")) | (a == ord("C"))).mean())


def _groups_home(protos):
    from sgrna.cohorts.curated_ablation import _groups
    g, ok = _groups("gene", protos)
    return np.asarray(g, dtype=object), ok


# ------------------------------------------------------------------ experiment 2: resolution floor
def _binned(index, nbins, flank=1000):
    """GC and purine fraction in `nbins` equal bins of the 1,000 nt each side. 4*nbins columns."""
    ups = index["upstream"].astype(str).to_numpy()
    dns = index["downstream"].astype(str).to_numpy()
    n = len(index)
    out = np.full((n, 4 * nbins), np.nan, np.float32)
    w = flank // nbins
    for side, seqs, off in (("up", ups, 0), ("dn", dns, 2 * nbins)):
        keep = np.array([len(s) >= flank for s in seqs])
        if not keep.any():
            continue
        sub = [(s[-flank:] if side == "up" else s[:flank]) for s in seqs[keep]]
        a = np.frombuffer("".join(sub).encode("ascii"), np.uint8).reshape(keep.sum(), flank)
        gc = ((a == ord("G")) | (a == ord("C"))).astype(np.float32)
        pu = ((a == ord("A")) | (a == ord("G"))).astype(np.float32)
        for b in range(nbins):
            sl = slice(b * w, (b + 1) * w)
            out[keep, off + b] = gc[:, sl].mean(1)
            out[keep, off + nbins + b] = pu[:, sl].mean(1)
    return out


def resolution(seeds=(41, 42, 43)):
    from sgrna.cohorts.curated_ablation import _base, _block
    X0, y, names, protos = _base()
    A, an = _block("a_flank", protos)
    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv").set_index(config.ID_COL)
    idx = idx.reindex(list(protos)).reset_index()
    comp = [j for j, n in enumerate(an) if n.startswith("eng.flank.comp.")]
    co = T.Cohort()
    cfgs = json.loads((OUT / "tune_configs.json").read_text())
    cfg = dict(cfgs["d099x"], n_estimators=int(T.selected().loc["d099x", "k"]))

    g, ok = _groups_home(protos)
    arms = {"base only": X0, "base + family A comp block (24 cols)": np.hstack([X0, A[:, comp]]),
            "base + all of family A (348 cols)": np.hstack([X0, A])}
    for b in BINS:
        arms[f"base + {4*b} binned cols ({1000//b} nt bins)"] = np.hstack([X0, _binned(idx, b)])

    rows = []
    for label, Xa in arms.items():
        Xa = Xa.astype(np.float32)
        per = []
        for seed in seeds:
            for tr, va in grouped_folds(g[ok], 5, seed):
                i = np.where(ok)[0]
                a0, _ = _impute(Xa[i[tr]], Xa[i[va]][:1])
                sel = make_selector(seed); sel.fit(a0, y[i[tr]])
                top = np.argsort(sel.feature_importances_)[::-1][: int(cfg["nf"])]
                a, bb = _impute(Xa[i[tr]][:, top], Xa[i[va]][:, top])
                p = T.fit_predict(cfg, a, y[i[tr]], bb, seed)[cfg["n_estimators"]]
                per.append(float(spearmanr(y[i[va]], p).statistic))
        rows.append(dict(arm=label, n_cols=Xa.shape[1] - X0.shape[1], rho=float(np.mean(per)),
                         sd=float(np.std(per, ddof=1)), n_fits=len(per)))
        print(f"  {label:44s} rho {rows[-1]['rho']:.4f} +- {rows[-1]['sd']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "novel_resolution.csv", index=False)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["relative", "resolution", "both"], default="both")
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    if a.test in ("resolution", "both"):
        print("\n=== resolution floor of the flank signal ===", flush=True)
        resolution()
    if a.test in ("relative", "both"):
        print("\n=== species-relative composition and cross-organism transfer ===", flush=True)
        relative()


if __name__ == "__main__":
    main()
