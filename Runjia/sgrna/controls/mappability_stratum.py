"""Family A's flank gain does not come from read depth or mappability: it is +0.157 among guides in uniquely mappable sequence and +0.160 with control read depth given to the model.

The objection: a guide in a repeat, an IS element or an rRNA operon has counts that can be wrong in a way that tracks local DNA, so a flank
composition feature could be predicting read misassignment and not cutting. The label comes from counts of the guide's own spacer (an
amplicon of the library, not of the genome), which makes the mechanism indirect, but it is testable: if the gain is an artefact of ambiguous
loci it should shrink where the genome is unique and sit in the repeated loci.

Three uniqueness measures per guide, all computed from the genome (NC_000913.2) and the guide index:

  u250, u1000   the fraction of 25-mers (canonical, both strands) in the 501 and 2,001 nt centred on the protospacer that occur exactly once in
                the genome: a read-mappability proxy of the kind a genome aligner sees (the measure of Jacky's stratum, rebuilt here for the curated cohort)
  seed_dup      other genomic NGG sites that share the guide's 12 PAM-proximal nucleotides: where Cas9 can also cut, so the label mixes loci
  copies        genomic copies of the 20-mer itself (`n_genome_copies` of the guide index)
  ctl_depth     log10 of the mean count of the guide in the two dead-nuclease (unselected) replicates, from crisprHAL's released counts: how well the
                guide is sequenced at all, the "read depth" of the objection (every curated guide has at least 73)
                (The Cas9-selected counts are not used: the label is the log-ratio of them to these, and as a feature they give rho 0.998 with it.)

Out-of-fold predictions of base and base + family A come from gene-grouped five-fold CV (guides outside any gene grouped by 100 kb block,
three seeds, models as in `sealed.py --parts gene`: base without column selection, base + family A through the 300-column selector). The
gain is then read inside each stratum on those predictions (the training set is the same for every stratum, so a smaller stratum is not a
smaller training set), with an interval from resampling genes. A third arm adds the uniqueness measures to the base as columns: if the gain were
carried by uniqueness, family A would add little on top of them.

    OMP_NUM_THREADS=3 python -m sgrna.controls.mappability_stratum [--features-only] [--seeds 41 42 43]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..common import OUT

K = 25
FEATURES = OUT / "mappability_stratum_guides.csv"
RESULT = OUT / "mappability_stratum.csv"
DEPTH_RESULT = OUT / "mappability_depth.csv"


def _uniqueness_track() -> np.ndarray:
    """1.0 where the canonical 25-mer starting at a position occurs exactly once in the genome (both strands), else 0.0."""
    from ..build import genome

    g = genome.load_genome().upper()
    code = np.full(len(g), 255, dtype=np.uint8)
    for i, b in enumerate("ACGT"):
        code[np.frombuffer(g.encode(), dtype=np.uint8) == ord(b)] = i
    n = len(g) - K + 1
    pw = (4 ** np.arange(K - 1, -1, -1, dtype=np.uint64))
    win = np.lib.stride_tricks.sliding_window_view(code, K)
    fwd = np.zeros(n, dtype=np.uint64)
    rc = np.zeros(n, dtype=np.uint64)
    for j in range(K):                         # position j of the window: forward weight 4^(K-1-j); reverse complement reads it back to front
        c = win[:, j].astype(np.uint64)
        fwd += c * pw[j]
        rc += (3 - c) * (4 ** np.uint64(j))
    bad = (win == 255).any(axis=1)
    canon = np.minimum(fwd, rc)
    canon[bad] = np.iinfo(np.uint64).max - np.arange(bad.sum(), dtype=np.uint64)       # N-containing windows count as unique, not as repeats
    _, inv, cnt = np.unique(canon, return_inverse=True, return_counts=True)
    track = (cnt[inv] == 1).astype(np.float64)
    return track


def features(force: bool = False) -> pd.DataFrame:
    if FEATURES.exists() and not force:
        return pd.read_csv(FEATURES)
    from ..build import genome
    from ..build.build_features import load_curated_index

    t0 = time.time()
    idx = load_curated_index(verbose=False)
    track = _uniqueness_track()
    cs = np.concatenate([[0.0], np.cumsum(track)])
    n = len(track)
    mid = ((idx["left"].to_numpy(int) + idx["right"].to_numpy(int)) // 2)

    def mean_unique(half: int) -> np.ndarray:
        a = np.clip(mid - half, 0, n - 1)
        b = np.clip(mid + half + 1, 1, n)
        return (cs[b] - cs[a]) / np.maximum(b - a, 1)

    seqs, lefts, strands = genome.protospacer_sites()
    seed = pd.Series([s[8:] for s in seqs])                 # the 12 nt next to the PAM (the guide is written 5' to 3', PAM at its 3' end)
    cnt = seed.map(seed.value_counts())
    seed_map = dict(zip(seed, cnt))
    out = pd.DataFrame({
        config.ID_COL: idx[config.ID_COL].to_numpy(),
        "u250": mean_unique(250),
        "u1000": mean_unique(1000),
        "seed_dup": [max(seed_map.get(p[8:], 1) - 1, 0) for p in idx["protospacer"]],
        "copies": idx["n_genome_copies"].to_numpy(int),
    })
    counts = pd.read_csv(config.EXTERNAL / "crisprHAL" / "rawData" / "E_coli_WTSpCas9_counts.tsv", sep="\t").set_index("sgRNA")
    c = counts.reindex(idx["protospacer"])
    out["ctl_depth"] = np.log10(((c["dCas1_NS"] + c["dCas2_NS"]) / 2).to_numpy())

    out.to_csv(FEATURES, index=False)
    print(f"  uniqueness features for {len(out):,} guides in {time.time() - t0:.0f}s -> {FEATURES}", flush=True)
    return out


OOF = OUT / "mappability_stratum_oof.csv"
ARMS = ("base", "base+U", "base+A", "base+U+A")


def _groups(protos):
    from ..build.build_features import load_curated_index

    idx = load_curated_index(verbose=False).set_index(config.ID_COL)
    gene = idx["gene_name"].reindex(protos)
    left = idx["left"].reindex(protos).to_numpy(float)
    return np.array([g if isinstance(g, str) and g else f"ig{int(l // 100_000)}" for g, l in zip(gene, left)], dtype=object)


def oof(seeds=(41, 42, 43), threads: int = 6, verbose: bool = True) -> pd.DataFrame:
    """Out-of-fold predictions of the four arms, averaged over seeds (gene-grouped five-fold CV over all 33,567 guides)."""
    from ..harness.curated import load_base, load_block
    from ..harness import folds as sg

    sg.N_JOBS = threads
    cache = config.INTERIM / "post_freeze" / "mappability"
    cache.mkdir(parents=True, exist_ok=True)
    X, y, names, protos = load_base()
    A, _ = load_block("a_flank", protos)
    f = features().set_index(config.ID_COL).reindex(protos)
    U = f[["u250", "u1000", "seed_dup", "copies"]].to_numpy(np.float32)
    groups = _groups(protos)
    mats = {"base": (X, None), "base+U": (np.hstack([X, U]), None),
            "base+A": (np.hstack([X, A]), config.N_FEATURES), "base+U+A": (np.hstack([X, U, A]), config.N_FEATURES)}
    per_seed = []
    for sd in seeds:
        path = cache / f"oof_seed{sd}.npz"
        if path.exists():
            per_seed.append(dict(np.load(path)))
            continue
        fold = sg.fold_ids(groups, 5, sd)
        pred = {a: np.empty(len(y)) for a in ARMS}
        for k in range(5):
            va = fold == k
            for a in ARMS:
                M, nf = mats[a]
                t0 = time.time()
                pred[a][va] = sg.fit_predict(M[~va], y[~va], M[va], None, nf, sd)
                if verbose:
                    print(f"  seed {sd} fold {k + 1} {a:9s} {time.time() - t0:5.1f}s", flush=True)
        np.savez(path, **pred)
        per_seed.append(pred)
    out = pd.DataFrame({config.ID_COL: protos, "y": y, "group": groups})
    for a in ARMS:
        out[a] = np.mean([p[a] for p in per_seed], axis=0)
    out = out.merge(features(), on=config.ID_COL, how="left")
    out.to_csv(OOF, index=False, float_format="%.5f")
    return out


def oof_depth(seeds=(41, 42, 43), threads: int = 6, verbose: bool = True) -> pd.DataFrame:
    """Two more arms: base + the control depth, and base + control depth + family A (same folds and seeds as `oof`)."""
    from ..harness.curated import load_base, load_block
    from ..harness import folds as sg

    sg.N_JOBS = threads
    cache = config.INTERIM / "post_freeze" / "mappability"
    X, y, names, protos = load_base()
    A, _ = load_block("a_flank", protos)
    f = features().set_index(config.ID_COL).reindex(protos)
    D = f[["ctl_depth"]].to_numpy(np.float32)
    groups = _groups(protos)
    mats = {"base+D": (np.hstack([X, D]), None), "base+D+A": (np.hstack([X, D, A]), config.N_FEATURES)}
    per_seed = []
    for sd in seeds:
        path = cache / f"oof_depth_seed{sd}.npz"
        if path.exists():
            per_seed.append(dict(np.load(path)))
            continue
        fold = sg.fold_ids(groups, 5, sd)
        pred = {a: np.empty(len(y)) for a in mats}
        for k in range(5):
            va = fold == k
            for a, (M, nf) in mats.items():
                t0 = time.time()
                pred[a][va] = sg.fit_predict(M[~va], y[~va], M[va], None, nf, sd)
                if verbose:
                    print(f"  seed {sd} fold {k + 1} {a:9s} {time.time() - t0:5.1f}s", flush=True)
        np.savez(path, **pred)
        per_seed.append(pred)
    out = pd.read_csv(OOF)
    for a in mats:
        out[a] = np.mean([p[a] for p in per_seed], axis=0)
    out.to_csv(OOF, index=False, float_format="%.5f")
    return out


def depth_strata(d: pd.DataFrame) -> pd.DataFrame:
    """Family A's gain by tertile of control depth, and its gain on top of the depth columns."""
    rows = []
    cut = d.ctl_depth.quantile([1 / 3, 2 / 3]).to_numpy()
    defs = {"control depth, lowest third": d.ctl_depth <= cut[0], "control depth, middle third": (d.ctl_depth > cut[0]) & (d.ctl_depth <= cut[1]),
            "control depth, highest third": d.ctl_depth > cut[1], "all guides": d.ctl_depth.notna()}
    for name, m in defs.items():
        s = d[m.to_numpy()]
        r = dict(stratum=name, n=int(len(s)), depth_min=float(10 ** s.ctl_depth.min()), depth_max=float(10 ** s.ctl_depth.max()))
        for a in ("base", "base+A", "base+D", "base+D+A"):
            r["rho_" + a] = float(spearmanr(s.y, s[a]).statistic)
        r["gain_A"] = r["rho_base+A"] - r["rho_base"]
        r["gain_A_lo"], r["gain_A_hi"] = _boot_gain(s, "base+A", "base")
        r["gain_A_given_D"] = r["rho_base+D+A"] - r["rho_base+D"]
        r["gain_A_given_D_lo"], r["gain_A_given_D_hi"] = _boot_gain(s, "base+D+A", "base+D")
        rows.append(r)
    return pd.DataFrame(rows)


def _boot_gain(d, a, b, B=1000, seed=0):
    """Spearman(d.a) - Spearman(d.b) against y, resampling whole groups."""
    rng = np.random.default_rng(seed)
    gi = d.groupby("group").indices
    keys = list(gi)
    pa, pb, y = d[a].to_numpy(), d[b].to_numpy(), d["y"].to_numpy()
    rows = [gi[k] for k in keys]
    vals = []
    for _ in range(B):
        take = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        vals.append(spearmanr(y[take], pa[take]).statistic - spearmanr(y[take], pb[take]).statistic)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def published_block_genes() -> set[str]:
    """Genes of the released 13,880-row matrix: row names are `<gene><b-number>_<position>_Cas9`; the file is an alphabetical block (aaeA to leuC)."""
    ids = pd.read_csv(config.BASE_MATRIX, usecols=[0]).iloc[:, 0].astype(str)
    ids = ids[ids != "sgRNAID"]
    return set(ids.str.split("_").str[0].str.replace(r"b\d{4}$", "", regex=True))


def strata(d: pd.DataFrame) -> pd.DataFrame:
    block = published_block_genes()
    genic = ~d.group.astype(str).str.startswith("ig")
    strict = (d.u1000 == 1) & (d.seed_dup == 0) & (d.copies == 1)
    defs = {
        "all guides": np.ones(len(d), bool),
        "strict: every 25-mer within 1 kb unique, no seed duplicate, one copy": strict.to_numpy(),
        "the rest (any repeat signal)": (~strict).to_numpy(),
        "every 25-mer within 1 kb unique": (d.u1000 == 1).to_numpy(),
        "some repeated 25-mer within 1 kb": (d.u1000 < 1).to_numpy(),
        "more than 10% of 25-mers within 1 kb repeated": (d.u1000 < 0.9).to_numpy(),
        "every 25-mer within 250 nt unique": (d.u250 == 1).to_numpy(),
        "a seed duplicate elsewhere in the genome": (d.seed_dup > 0).to_numpy(),
        "more than one genomic copy": (d.copies > 1).to_numpy(),
        "genes of the published matrix (aaeA to leuC), genic guides": (genic & d.group.isin(block)).to_numpy(),
        "all other genes, genic guides": (genic & ~d.group.isin(block)).to_numpy(),
    }
    rows = []
    for name, m in defs.items():
        s = d[m]
        r = dict(stratum=name, n=int(m.sum()), genes=int(s.group.nunique()))
        for a in ARMS:
            r["rho_" + a] = float(spearmanr(s.y, s[a]).statistic)
        r["gain_A"] = r["rho_base+A"] - r["rho_base"]
        lo, hi = _boot_gain(s, "base+A", "base") if len(s) >= 300 else (np.nan, np.nan)
        r["gain_A_lo"], r["gain_A_hi"] = lo, hi
        r["gain_A_given_U"] = r["rho_base+U+A"] - r["rho_base+U"]
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=[41, 42, 43])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--depth", action="store_true", help="the depth control: two more arms, then the tertile table (needs the stored predictions of the main run)")
    ap.add_argument("--strata-only", action="store_true", help="read the stored out-of-fold predictions and recompute the strata table")
    a = ap.parse_args(argv)
    f = features(force=a.force)
    if a.features_only:
        print(f.describe().round(4).to_string())
        return 0
    if a.depth:
        d = oof_depth(tuple(a.seeds), a.threads)
        d = d.drop(columns=[c for c in ("u250", "u1000", "seed_dup", "copies", "ctl_depth", "cas_depth") if c in d]).merge(f, on=config.ID_COL, how="left")
        r = depth_strata(d)
        r.to_csv(DEPTH_RESULT, index=False)
        print(r.round(4).to_string(index=False))
        return 0
    d = pd.read_csv(OOF) if a.strata_only else oof(tuple(a.seeds), a.threads)
    d = d.drop(columns=[c for c in ("u250", "u1000", "seed_dup", "copies", "ctl_depth", "cas_depth") if c in d]).merge(f, on=config.ID_COL, how="left")
    r = strata(d)
    r.to_csv(RESULT, index=False)
    print(r.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
