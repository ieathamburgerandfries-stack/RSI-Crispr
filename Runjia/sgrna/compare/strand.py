"""With the strand call given to every family, flank family A adds only +0.007 on the dCas9 screen, but inside the coding strand its gain is not only gene ranking (+0.102 within genes on Cui's screen).

One question, asked in pieces: on the dCas9 CRISPRi screens, is family A's gain the strand rule arriving through a lossy copy of "which strand the guide targets"? Family A's +0.27 on Cui's
essential-gene guides was the dCas9 strand rule arriving through `eng.flank.comp.up1000.purine` (Finding 3), and the control then was family A against the strand call. The parts below ask the
questions that left open. Each is a `--parts` word; result files are written under `results/` by name.

  build              the eleven feature families for Cui's 5,514 essential-gene guides, cached under `data/interim/cui_crispri/families/`
  families           is A still the strongest family once the strand call is given to everyone? base, base + strand, base + strand + one family, base + one family, and one family without its proxy
                     columns (|rho| with the strand above 0.30), 100 kb-block and gene grouping, three seeds, selection inside each training fold   -> strand_families.csv, strand_families_proxies.csv
  families-within    strand held constant by construction: fit and score inside the coding-strand and template-strand guides separately            -> strand_families_within.csv
  recoverable        after dropping the proxy columns, how well does each family still recover the strand call (gene-grouped AUC)                -> strand_families_recoverable.csv
  rousset            the same within-strand comparison on Rousset's essential-gene guides, with a between-genes / within-genes split              -> strand_strata_rousset.csv
  flip               tell a trained model the wrong strand at test time, and see what the strand column was carrying (Cui, pooled, gene-grouped)    -> strand_flip.csv
  within-cui         among coding-strand guides, is A's gain only gene ranking? Out-of-fold predictions split into the rho of gene means (between) and of values with each gene's mean removed
                     (within), genes with at least five guides, ten seeds; seed-to-seed spread and a bootstrap over genes                           -> strand_within_gene_cui(_summary).csv
  within-joint       does A add within genes once transcription context, which carries the guide's relative position in its gene, is in the model   -> strand_within_gene_cui_joint(_summary).csv
  within-rousset     the within-gene split on Rousset's coding-strand guides                                                                      -> strand_within_gene_rousset(_summary).csv
  cutting            the same split on the cutting screen (the curated cohort's genic guides), where +0.157 had been a pooled figure               -> cut_within_gene(_summary).csv

    python -m sgrna.compare.strand --parts build families
    python -m sgrna.compare.strand --parts within-cui within-joint within-rousset

Resumable: `cross_validate` reads the fold checkpoints `results/folds_<tag>.csv`, so a part whose folds are stored reruns in seconds and reproduces its table to floating-point rounding. The tags are interfaces and
must not be renamed. This module replaced `strand_families`, `strand_strata`, `strand_within_gene`, `cut_within_gene` and `strand_recoverable`.

Report section: Appendix B5.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

from .. import config
from ..build import genome
from ..common import OUT
from ..features import ALL, REGISTRY

FAM_DIR = config.INTERIM / "cui_crispri" / "families"
FLANK = 1000
PROXY = 0.30            # |rho| with the strand call above which a column is a proxy
SEEDS10 = tuple(range(41, 51))
MIN_G = 5               # genes with fewer guides are left out of the between/within split


# ---------------------------------------------------------------------------------------------------------------- the essential-gene frame
def _essential():
    """Cui's meta.json, the mask of its essential-gene guides that have a gene name, and the gene name of every guide."""
    meta = json.loads((config.INTERIM / "cui_crispri" / "meta.json").read_text())
    ess = np.asarray([str(e) for e in meta["essential"]], dtype=object) == "True"
    gene = np.asarray([g if isinstance(g, str) else "" for g in meta["gene"]], dtype=object)
    has_gene = (gene != "") & (gene != "nan") & (gene != "None")
    return meta, ess & has_gene, gene


def index() -> pd.DataFrame:
    """Cui's essential-gene guides as the family builders expect them."""
    from ..build.build_features import augment

    meta, mask, gene = _essential()
    sel = np.where(mask)[0]
    seqs, lefts, strands = genome.protospacer_sites()
    site = {s: (l, t) for s, l, t in zip(seqs, lefts, strands)}
    g = genome.load_genome()
    rows = []
    for i in sel:
        p = meta["protospacers"][i]
        left, st = site[p]
        ctx = genome.guide_context(int(left), int(left) + 19, "+" if st == 1 else "-", FLANK, g)
        rows.append(dict(**{config.ID_COL: p}, protospacer=p, located=True, n_genome_copies=1, strand="+" if st == 1 else "-",
                         left=int(left), right=int(left) + 19, pam=ctx["pam"], upstream=ctx["upstream"], downstream=ctx["downstream"],
                         score=meta["y"][i], gene_name=gene[i]))
    return augment(pd.DataFrame(rows), verbose=False)


def build(families=ALL, verbose: bool = True):
    FAM_DIR.mkdir(parents=True, exist_ok=True)
    idx = index()
    if verbose:
        print(f"  {len(idx):,} essential-gene guides")
    for fam in families:
        path = FAM_DIR / f"{fam}.csv"
        if path.exists():
            continue
        t0 = time.time()
        try:
            blk = REGISTRY[fam].build(idx)
            feats = [c for c in blk.columns if c != config.ID_COL]
            filled = int(blk[feats].notna().any(axis=1).sum()) if feats else 0
            if not feats or not filled:
                raise RuntimeError("empty block")
            blk.to_csv(path, index=False)
            print(f"  {fam}: {len(feats)} columns, {filled:,} populated, {time.time() - t0:.0f}s", flush=True)
        except Exception as exc:
            print(f"  {fam}: FAILED {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=2)


def _blocks(protos):
    """Every built family, aligned to the essential frame's guide order."""
    out = {}
    for fam in ALL:
        path = FAM_DIR / f"{fam}.csv"
        if not path.exists():
            continue
        b = pd.read_csv(path).set_index(config.ID_COL).reindex(protos)
        v = b.to_numpy(dtype=np.float32)
        keep = ~np.all(np.isnan(v), axis=0)
        out[fam] = (v[:, keep], [c for c, k in zip(b.columns, keep) if k])
    return out


def _frame():
    """Cui's essential frame, its guides' protospacers and the family blocks aligned to them."""
    from .cui_screen import _essential_frame

    f = _essential_frame()
    meta, mask, gene = _essential()
    protos = np.asarray(meta["protospacers"])[mask]
    return f, protos, gene[mask]


def _strand_assoc(v, cod):
    """|Spearman rho| of each column of a block with the strand call, over the guides that have one (0 for a constant column)."""
    ok = cod >= 0
    r = np.array([abs(spearmanr(v[ok, j], cod[ok], nan_policy="omit").statistic) if np.nanstd(v[ok, j]) > 0 else 0.0 for j in range(v.shape[1])])
    return np.nan_to_num(r)


# ---------------------------------------------------------------------------------------------------------------- families, families-within, recoverable
def families(seeds=(41, 42, 43), n_splits=None, verbose: bool = True):
    """Three arms per family: with strand, without, and without its proxy columns."""
    from ..harness.run_ablation import cross_validate

    f, protos, _ = _frame()
    assert len(protos) == len(f["y"]), (len(protos), len(f["y"]))
    blocks = _blocks(protos)
    strand = f["coding"]
    cod = strand.ravel()
    out = OUT / "strand_families.csv"

    rows, proxy_rows = [], []
    for gname, grouping in f["groupings"]:
        def cv(Xa, nm, arm):
            res = cross_validate(Xa, f["y"], nm, ["arm"] * len(nm), n_features=config.N_FEATURES, seeds=list(seeds), n_splits=n_splits or config.N_SPLITS,
                                 verbose=False, groups=grouping, tag=f"cui_sf_{gname.replace(' ', '')}_{arm}", experiment=arm)
            return res["summary"]["spearman"]["mean"], res["summary"]["spearman"]["sd"]

        base_names = f["nb"]
        arms = {"base": (f["Xb"], base_names),
                "base+strand": (np.hstack([f["Xb"], strand]), base_names + ["screen.coding_strand"])}
        for fam, (v, fn) in blocks.items():
            r = _strand_assoc(v, cod)
            drop = r > PROXY
            if gname == "genes":
                proxy_rows.append(dict(family=fam, n_cols=v.shape[1], max_abs_rho_with_strand=float(r.max()), n_proxy_cols=int(drop.sum()),
                                       strongest=fn[int(r.argmax())]))
            arms[f"{fam}"] = (np.hstack([f["Xb"], v]), base_names + fn)
            arms[f"{fam}+strand"] = (np.hstack([f["Xb"], strand, v]), base_names + ["screen.coding_strand"] + fn)
            if drop.any():
                keep = ~drop
                arms[f"{fam}-proxy"] = (np.hstack([f["Xb"], v[:, keep]]), base_names + [c for c, k in zip(fn, keep) if k])
        for arm, (Xa, nm) in arms.items():
            m, sd = cv(Xa, nm, arm.replace("+", "_"))
            rows.append(dict(grouping=gname, arm=arm, spearman=m, spearman_sd=sd))
            if verbose:
                print(f"  {gname:12s} {arm:28s}: rho {m:+.4f}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(proxy_rows).to_csv(OUT / "strand_families_proxies.csv", index=False)
    return pd.DataFrame(rows)


def families_within(seeds=(41, 42, 43), verbose: bool = True):
    """Strand held constant by construction: fit and score inside one strand at a time.

    Removing columns whose |rho| with the strand call exceeds PROXY does not make an arm strand-free (the remaining columns still read the strand back, see `strand_families_recoverable.csv`). Here the strand
    cannot enter at all: the 2,5xx coding-strand and 2,9xx template-strand guides are modelled separately and the family increment is the family arm minus `base` within the same stratum.
    """
    from ..harness.run_ablation import cross_validate

    f, protos, _ = _frame()
    blocks = _blocks(protos)
    cod = f["coding"].ravel()
    rows = []
    for sname, sval in (("coding", 1), ("template", 0)):
        m = cod == sval
        for gname, grouping in f["groupings"]:
            gsub = np.asarray(grouping)[m]
            arms = {"base": (f["Xb"][m], f["nb"])}
            for fam, (v, fn) in blocks.items():
                arms[fam] = (np.hstack([f["Xb"], v])[m], f["nb"] + fn)
            for arm, (Xa, nm) in arms.items():
                res = cross_validate(Xa, f["y"][m], nm, ["arm"] * len(nm), n_features=config.N_FEATURES, seeds=list(seeds), n_splits=config.N_SPLITS, verbose=False,
                                     groups=gsub, tag=f"cui_sfw_{sname}_{gname.replace(' ', '')}_{arm}", experiment=arm)
                mu, sd = res["summary"]["spearman"]["mean"], res["summary"]["spearman"]["sd"]
                rows.append(dict(stratum=sname, grouping=gname, arm=arm, n=int(m.sum()), spearman=mu, spearman_sd=sd))
                if verbose:
                    print(f"  {sname:8s} {gname:12s} {arm:16s}: rho {mu:+.4f}", flush=True)
                pd.DataFrame(rows).to_csv(OUT / "strand_families_within.csv", index=False)
    return pd.DataFrame(rows)


def recoverable():
    """Measure how well each family recovers the strand call, and write the table."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    f, protos, _ = _frame()
    blocks = _blocks(protos)
    cod = f["coding"].ravel()
    ok = cod >= 0
    y = cod[ok].astype(int)
    grp = np.asarray(dict(f["groupings"])["genes"])[ok]
    print("guides with strand call:", ok.sum(), "coding share", y.mean().round(3))

    def auc(X):
        X = X[ok]
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, grp):
            m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbosity=0)
            m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        return roc_auc_score(y, p)

    rows = []
    for fam, (v, fn) in blocks.items():
        keep = _strand_assoc(v, cod) <= PROXY
        a_all = auc(v)
        a_keep = auc(v[:, keep]) if keep.any() else np.nan
        rows.append(dict(family=fam, n_cols=v.shape[1], n_proxy=int((~keep).sum()), auc_all=a_all, auc_after_removal=a_keep))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "strand_families_recoverable.csv", index=False)
    v, fn = blocks["g_transcription"]
    r = [(abs(spearmanr(v[ok, j], cod[ok], nan_policy="omit").statistic), fn[j]) for j in range(v.shape[1])]
    for a, b in sorted(r, reverse=True)[:8]:
        print(f"  {a:.3f} {b}")


# ---------------------------------------------------------------------------------------------------------------- the between / within split
def _decompose_parts(y, g, p):
    d = pd.DataFrame(dict(g=g, y=y, p=p))
    d = d[d.groupby("g").y.transform("size") >= MIN_G]
    gm = d.groupby("g")[["y", "p"]].mean()
    wy = d.y - d.groupby("g").y.transform("mean")
    wp = d.p - d.groupby("g").p.transform("mean")
    return d, gm, wy, wp


def decompose(y, g, p):
    """(overall, between-genes, within-genes) Spearman rho of predictions p against y, for genes with at least MIN_G guides."""
    d, gm, wy, wp = _decompose_parts(y, g, p)
    return spearmanr(d.y, d.p).statistic, spearmanr(gm.y, gm.p).statistic, spearmanr(wy, wp).statistic


def _within_resid(y, g, p):
    d = pd.DataFrame(dict(g=g, y=y, p=p))
    d = d[d.groupby("g").y.transform("size") >= MIN_G]
    d["wy"] = d.y - d.groupby("g").y.transform("mean")
    d["wp"] = d.p - d.groupby("g").p.transform("mean")
    return d[["g", "wy", "wp"]]


def within_gene(label, arms, y, gene, groups, tagpref, out_name, seeds=SEEDS10, n_boot=2000):
    """arms: dict name -> (X, feature_names). Rows are the guides of one stratum (coding strand, or the genic cutting guides).

    Ten seeds, one cross-validation per seed so each seed has its own out-of-fold predictions. Two kinds of uncertainty, and they are different: seed-to-seed (mean and sd of the paired difference over
    seeds) and sampling (a bootstrap over genes of the within-gene residuals of the seed-averaged predictions, 2,000 resamples, for the difference arm - base).
    """
    from ..harness.run_ablation import cross_validate

    out_csv = OUT / out_name
    per_seed, preds = [], {a: [] for a in arms}
    for seed in seeds:
        for arm, (X, nm) in arms.items():
            res = cross_validate(X, y, nm, ["arm"] * len(nm), n_features=config.N_FEATURES, seeds=[seed], n_splits=config.N_SPLITS, verbose=False,
                                 groups=groups, tag=f"{tagpref}_{arm}_s{seed}", experiment=arm, return_oof=True)
            p = res["oof_pred"]
            if np.isnan(p).any():
                raise RuntimeError(f"{tagpref}_{arm}_s{seed}: the out-of-fold predictions are missing. This fold set was resumed from its checkpoints, which hold the fold scores but not "
                                   f"the predictions, and writing the between/within split from them would replace a good result file with NaN. Delete the fold checkpoints of this tag (results/folds_<tag>.csv) to refit.")
            preds[arm].append(p)
            o, b, w = decompose(y, gene, p)
            per_seed.append(dict(screen=label, arm=arm, seed=seed, overall=o, between_genes=b, within_genes=w))
            print(f"  {label} seed {seed} {arm:16s} overall {o:+.3f} between {b:+.3f} within {w:+.3f}", flush=True)
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
                row[k + "_diff"] = d.mean()
                row[k + "_diff_sd"] = d.std(ddof=1)
        if arm != "base":
            diffs = []
            for _ in range(n_boot):
                gs = rng.choice(genes, len(genes))
                ii = np.concatenate([idx[g] for g in gs])
                r1, r0 = resid[arm].iloc[ii], resid["base"].iloc[ii]
                diffs.append(spearmanr(r1.wy, r1.wp).statistic - spearmanr(r0.wy, r0.wp).statistic)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            row.update(within_diff_boot_lo=lo, within_diff_boot_hi=hi)
        rows.append(row)
    sm = pd.DataFrame(rows)
    sm.to_csv(out_csv.with_name(out_csv.stem + "_summary.csv"), index=False)
    print(sm.round(3).to_string(), flush=True)
    return sm


def within_cui():
    f, protos, gene = _frame()
    blocks = _blocks(protos)
    m = f["coding"].ravel() == 1
    groups = np.asarray(dict(f["groupings"])["genes"])[m]
    arms = {"base": (f["Xb"][m], f["nb"])}
    for fam in ("a_flank", "g_transcription", "d_supercoiling", "d_mechanics", "i_shape"):
        arms[fam] = (np.hstack([f["Xb"], blocks[fam][0]])[m], f["nb"] + blocks[fam][1])
    return within_gene("Cui coding", arms, f["y"][m], gene[m], groups, "sfg10_cui", "strand_within_gene_cui.csv")


def within_joint():
    """Does family A add within genes once transcription context (which carries the guide's relative position in its gene) is already in the model?"""
    f, protos, gene = _frame()
    blocks = _blocks(protos)
    m = f["coding"].ravel() == 1
    groups = np.asarray(dict(f["groupings"])["genes"])[m]
    g, a = blocks["g_transcription"], blocks["a_flank"]
    arms = {"base": (f["Xb"][m], f["nb"]),
            "g_transcription": (np.hstack([f["Xb"], g[0]])[m], f["nb"] + g[1]),
            "g_transcription+a_flank": (np.hstack([f["Xb"], g[0], a[0]])[m], f["nb"] + g[1] + a[1])}
    return within_gene("Cui coding joint", arms, f["y"][m], gene[m], groups, "sfg10_cuij", "strand_within_gene_cui_joint.csv")


def _rousset_frame(name="Rousset"):
    from .crispri_screens import build as build_screen

    X, meta = build_screen(name, verbose=False)
    X = np.asarray(X, dtype=np.float32)
    return X, meta, list(meta["names"]), int(meta["n_base"])


def within_rousset():
    X, meta, names, nb = _rousset_frame()
    y = np.asarray(meta["y"], dtype=float)
    gene = np.asarray(meta["gene"], dtype=object)
    ess = np.asarray(meta["essential"], dtype=int)
    cod = np.asarray(meta["coding_strand"], dtype=float)
    m = (gene != "") & (ess == 1) & (cod == 1.0)
    arms = {"base": (X[m][:, :nb], names[:nb]), "a_flank": (X[m], names)}
    return within_gene("Rousset coding", arms, y[m], gene[m], gene[m], "sfg10_rou", "strand_within_gene_rousset.csv")


def cutting():
    """The cutting-screen figure on the within-gene definition used for the dCas9 screens (crisprHAL's curated cohort, the guides inside an annotated gene)."""
    from ..harness.curated import load_base, load_block, load_groups

    X, y, names, protos = load_base()
    groups, ok = load_groups("gene", protos)
    X, y, protos, groups = X[ok], y[ok], protos[ok], groups[ok]
    v, fn = load_block("a_flank", protos)
    arms = {"base": (X, names), "a_flank": (np.hstack([X, v]), names + fn)}
    print(f"  {X.shape[0]:,} guides in {len(np.unique(groups)):,} genes", flush=True)
    return within_gene("Cutting (curated, genic guides)", arms, y, groups, groups, "sfg10_cut", "cut_within_gene.csv")


# ---------------------------------------------------------------------------------------------------------------- the two controls on the strand result
def rousset(name="Rousset", seeds=(41, 42, 43)):
    """Family A inside one strand at a time, on Rousset's essential-gene guides (the replication of `families-within`, which used Cui)."""
    from ..harness.run_ablation import cross_validate

    X, meta, names, nb = _rousset_frame(name)
    y = np.asarray(meta["y"], dtype=float)
    gene = np.asarray(meta["gene"], dtype=object)
    ess = np.asarray(meta["essential"], dtype=int)
    cod = np.asarray(meta["coding_strand"], dtype=float)
    block = np.floor_divide(np.asarray(meta["left"], dtype=float), 100_000).astype(int).astype(str)
    sel = (gene != "") & (ess == 1)
    out = OUT / "strand_strata_rousset.csv"
    rows, oof = [], {}
    for sname, sval in (("coding", 1.0), ("template", 0.0)):
        m = sel & (cod == sval)
        print(f"  {name} {sname}: {int(m.sum()):,} guides", flush=True)
        for gname, grp in (("genes", gene), ("100kb blocks", block)):
            for arm, cols in (("base", slice(0, nb)), ("a_flank", slice(0, None))):
                Xa, nm = X[m][:, cols], names[cols]
                res = cross_validate(Xa, y[m], nm, ["arm"] * len(nm), n_features=config.N_FEATURES, seeds=list(seeds), n_splits=config.N_SPLITS, verbose=False,
                                     groups=grp[m], tag=f"ssr_{name}_{sname}_{gname.replace(' ', '')}_{arm}", experiment=arm, return_oof=True)
                mu = res["summary"]["spearman"]["mean"]
                sd = res["summary"]["spearman"]["sd"]
                if sname == "coding" and gname == "genes":
                    oof[arm] = res["oof_pred"]
                    if np.isnan(oof[arm]).any():
                        raise RuntimeError(f"ssr_{name}_{sname}_{gname}_{arm}: the out-of-fold predictions are missing (the fold set was resumed from its checkpoints, which do not hold "
                                           f"them); the between/within rows would be NaN. Delete the fold checkpoints of these tags (results/folds_<tag>.csv) to refit.")
                rows.append(dict(screen=name, stratum=sname, grouping=gname, arm=arm, n=int(m.sum()), spearman=mu, spearman_sd=sd))
                print(f"    {gname:12s} {arm:8s}: rho {mu:+.4f} (sd {sd:.4f})", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    # between genes vs within genes, coding stratum (last seed's out-of-fold)
    m = sel & (cod == 1.0)
    for arm, p in oof.items():
        d, gm, _, _ = _decompose_parts(y[m], gene[m], p)
        overall, between, within = decompose(y[m], gene[m], p)
        rows.append(dict(screen=name, stratum="coding", grouping="genes", arm=arm, n=len(d), genes=len(gm), overall=overall, between_genes=between, within_genes=within))
        print("   ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items() if k in ("arm", "n", "genes", "overall", "between_genes", "within_genes")})
    pd.DataFrame(rows).to_csv(out, index=False)


def flip(seeds=(41, 42, 43)):
    """Train with the true strand; score held-out genes with the strand wrong."""
    from ..harness.run_ablation import _impute, champion_params, grouped_folds, make_selector

    f, _, _ = _frame()
    strand = f["coding"]
    rows = []
    for arm, Xb in (("base+strand", f["Xb"]), ("base+strand+a_flank", np.hstack([f["Xb"], f["Xf"]]))):
        X = np.hstack([Xb, strand])
        scol = X.shape[1] - 1
        y = f["y"]
        grp = dict(f["groupings"])["genes"]
        res = {k: [] for k in ("true", "flipped", "shuffled", "constant")}
        rng = np.random.default_rng(0)
        used = []
        for seed in seeds:
            for tr, va in grouped_folds(grp, config.N_SPLITS, seed):
                Xtr, Xva = _impute(X[tr], X[va])
                sel = make_selector(seed)
                sel.fit(Xtr, y[tr])
                top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
                used.append(bool(scol in top))
                mdl = xgb.XGBRegressor(**champion_params(seed))
                mdl.fit(Xtr[:, top], y[tr])
                for k in res:
                    Xv = Xva.copy()
                    if k == "flipped":
                        Xv[:, scol] = 1.0 - Xv[:, scol]
                    elif k == "shuffled":
                        Xv[:, scol] = rng.permutation(Xv[:, scol])
                    elif k == "constant":
                        Xv[:, scol] = float(np.mean(Xtr[:, scol]))
                    res[k].append(spearmanr(y[va], mdl.predict(Xv[:, top])).statistic)
        row = dict(arm=arm, strand_selected_in=f"{sum(used)}/{len(used)} folds", **{k: float(np.mean(v)) for k, v in res.items()},
                   **{f"{k}_sd": float(np.std(v)) for k, v in res.items()})
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "strand_flip.csv", index=False)


PARTS = {"build": build, "families": families, "families-within": families_within, "recoverable": recoverable, "rousset": rousset, "flip": flip,
         "within-cui": within_cui, "within-joint": within_joint, "within-rousset": within_rousset, "cutting": cutting}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", nargs="+", choices=list(PARTS), required=True, help="which questions to run, in this order")
    a = ap.parse_args(argv)
    for part in a.parts:
        PARTS[part]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
