"""Inside the coding-strand guides, is a family's gain between genes or within them?

`strand_families --within` found family A (and transcription context) adding +0.12 to
+0.27 rho among coding-strand guides, where gene-grouped CV holds whole genes out.
A gain can come from ranking genes (flank composition tracks gene-level depletion) or
from ranking guides inside one gene (the local flank). The out-of-fold predictions are
split into the rho of gene means (between) and the rho of values with each gene's mean
removed (within), for genes with at least five coding-strand guides.

Ten seeds (41-50), one cross-validation per seed so each seed has its own out-of-fold
predictions. Two kinds of uncertainty are reported, and they are different:
  * seed-to-seed: how much the split moves the figure (mean +- sd of the paired
    difference over seeds);
  * sampling: a bootstrap over genes of the within-gene residuals of the seed-averaged
    predictions (2,000 resamples), for the difference arm - base.

    python -m sgrna.strand_within_gene --cui
    python -m sgrna.strand_within_gene --rousset
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config
from .run_ablation import cross_validate

SEEDS = tuple(range(41, 51))
MIN_G = 5


def decompose(y, g, p):
    d = pd.DataFrame(dict(g=g, y=y, p=p))
    d = d[d.groupby("g").y.transform("size") >= MIN_G]
    gm = d.groupby("g")[["y", "p"]].mean()
    wy = d.y - d.groupby("g").y.transform("mean")
    wp = d.p - d.groupby("g").p.transform("mean")
    return (spearmanr(d.y, d.p).statistic, spearmanr(gm.y, gm.p).statistic,
            spearmanr(wy, wp).statistic)


def _within_resid(y, g, p):
    d = pd.DataFrame(dict(g=g, y=y, p=p))
    d = d[d.groupby("g").y.transform("size") >= MIN_G]
    d["wy"] = d.y - d.groupby("g").y.transform("mean")
    d["wp"] = d.p - d.groupby("g").p.transform("mean")
    return d[["g", "wy", "wp"]]


def run(label, arms, y, gene, groups, tagpref, out_csv, seeds=SEEDS, n_boot=2000):
    """arms: dict name -> (X, feature_names). Rows are the coding-strand guides."""
    per_seed, preds = [], {a: [] for a in arms}
    for seed in seeds:
        for arm, (X, nm) in arms.items():
            res = cross_validate(X, y, nm, ["arm"] * len(nm), n_features=config.N_FEATURES,
                                 seeds=[seed], n_splits=config.N_SPLITS, verbose=False,
                                 groups=groups, tag=f"{tagpref}_{arm}_s{seed}",
                                 experiment=arm, return_oof=True)
            p = res["oof_pred"]; preds[arm].append(p)
            o, b, w = decompose(y, gene, p)
            per_seed.append(dict(screen=label, arm=arm, seed=seed, overall=o,
                                 between_genes=b, within_genes=w))
            print(f"  {label} seed {seed} {arm:16s} overall {o:+.3f} between {b:+.3f} "
                  f"within {w:+.3f}", flush=True)
        pd.DataFrame(per_seed).to_csv(out_csv, index=False)
    ps = pd.DataFrame(per_seed)

    rng = np.random.default_rng(0)
    avg = {a: np.mean(v, axis=0) for a, v in preds.items()}
    rows = []
    base = ps[ps.arm == "base"].set_index("seed")
    resid = {a: _within_resid(y, gene, avg[a]) for a in arms}
    genes = resid["base"].g.unique()
    idx = {g: np.where(resid["base"].g.values == g)[0] for g in genes}
    for arm in arms:
        a = ps[ps.arm == arm].set_index("seed")
        row = dict(screen=label, arm=arm, n_seeds=len(seeds))
        for k in ("overall", "between_genes", "within_genes"):
            row[k] = a[k].mean()
            row[k + "_sd"] = a[k].std(ddof=1)
            if arm != "base":
                d = (a[k] - base[k])
                row[k + "_diff"] = d.mean(); row[k + "_diff_sd"] = d.std(ddof=1)
        if arm != "base":
            diffs = []
            for _ in range(n_boot):
                gs = rng.choice(genes, len(genes))
                ii = np.concatenate([idx[g] for g in gs])
                r1, r0 = resid[arm].iloc[ii], resid["base"].iloc[ii]
                diffs.append(spearmanr(r1.wy, r1.wp).statistic
                             - spearmanr(r0.wy, r0.wp).statistic)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            row.update(within_diff_boot_lo=lo, within_diff_boot_hi=hi)
        rows.append(row)
    sm = pd.DataFrame(rows)
    sm.to_csv(out_csv.with_name(out_csv.stem + "_summary.csv"), index=False)
    print(sm.round(3).to_string(), flush=True)
    return sm


def cui():
    from .cui_screen import _essential_frame
    from .strand_families import _blocks

    f = _essential_frame()
    meta = json.loads((config.INTERIM / "cui_crispri" / "meta.json").read_text())
    ess = np.asarray([str(e) for e in meta["essential"]], dtype=object) == "True"
    gene = np.asarray([g if isinstance(g, str) else "" for g in meta["gene"]], dtype=object)
    sel = ess & (gene != "") & (gene != "nan") & (gene != "None")
    protos = np.asarray(meta["protospacers"])[sel]; gene = gene[sel]
    blocks = _blocks(protos)
    m = f["coding"].ravel() == 1
    groups = np.asarray(dict(f["groupings"])["genes"])[m]
    arms = {"base": (f["Xb"][m], f["nb"])}
    for fam in ("a_flank", "g_transcription", "d_supercoiling", "d_mechanics", "i_shape"):
        arms[fam] = (np.hstack([f["Xb"], blocks[fam][0]])[m], f["nb"] + blocks[fam][1])
    return run("Cui coding", arms, f["y"][m], gene[m], groups, "sfg10_cui",
               config.RESULTS / "strand_within_gene_cui.csv")


def cui_joint():
    """Does family A add within genes once transcription context (which carries the
    guide's relative position in its gene) is already in the model?"""
    from .cui_screen import _essential_frame
    from .strand_families import _blocks

    f = _essential_frame()
    meta = json.loads((config.INTERIM / "cui_crispri" / "meta.json").read_text())
    ess = np.asarray([str(e) for e in meta["essential"]], dtype=object) == "True"
    gene = np.asarray([g if isinstance(g, str) else "" for g in meta["gene"]], dtype=object)
    sel = ess & (gene != "") & (gene != "nan") & (gene != "None")
    protos = np.asarray(meta["protospacers"])[sel]; gene = gene[sel]
    blocks = _blocks(protos)
    m = f["coding"].ravel() == 1
    groups = np.asarray(dict(f["groupings"])["genes"])[m]
    g, a = blocks["g_transcription"], blocks["a_flank"]
    arms = {"base": (f["Xb"][m], f["nb"]),
            "g_transcription": (np.hstack([f["Xb"], g[0]])[m], f["nb"] + g[1]),
            "g_transcription+a_flank": (np.hstack([f["Xb"], g[0], a[0]])[m],
                                        f["nb"] + g[1] + a[1])}
    return run("Cui coding joint", arms, f["y"][m], gene[m], groups, "sfg10_cuij",
               config.RESULTS / "strand_within_gene_cui_joint.csv")


def rousset():
    from .crispri_screens import build

    X, meta = build("Rousset", verbose=False)
    X = np.asarray(X, dtype=np.float32); names = list(meta["names"]); nb = int(meta["n_base"])
    y = np.asarray(meta["y"], dtype=float)
    gene = np.asarray(meta["gene"], dtype=object)
    ess = np.asarray(meta["essential"], dtype=int)
    cod = np.asarray(meta["coding_strand"], dtype=float)
    m = (gene != "") & (ess == 1) & (cod == 1.0)
    arms = {"base": (X[m][:, :nb], names[:nb]), "a_flank": (X[m], names)}
    return run("Rousset coding", arms, y[m], gene[m], gene[m], "sfg10_rou",
               config.RESULTS / "strand_within_gene_rousset.csv")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cui", action="store_true")
    ap.add_argument("--rousset", action="store_true")
    ap.add_argument("--joint", action="store_true")
    a = ap.parse_args(argv)
    if a.joint:
        cui_joint()
    if a.rousset:
        rousset()
    if a.cui:
        cui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
