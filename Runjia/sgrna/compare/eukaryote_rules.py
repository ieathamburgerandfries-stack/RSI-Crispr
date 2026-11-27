"""Eukaryotes share guide rules (corrected correlation +0.59 between human tables) that are unrelated to E. coli's: G at position 20 helps in every eukaryotic table and hurts in E. coli.

Finding 7 found that the guide's own position x base rules are unrelated between E. coli and one human matrix (rank correlation
-0.052, with G at position 20 disfavoured in E. coli and favoured in human). That is one comparison against one cohort, so it cannot say
whether the inversion is bacteria against eukaryotes or E. coli against human in particular. This module puts the same estimator on
public efficiency tables for human, mouse and zebrafish (and, with the caveat that they are tiny, rat, fly and worm).

Data: the CRISPOR benchmark collection (github.com/maximilianh/crisporPaper, `effData/*.scores.tab`), downloaded as plain text to
`external_data/crispor_effdata/`: each row is a guide (20 or 23 nt), an efficiency score, the genome and 100 nt of context with the guide
at offset 30. Inclusion rule, fixed before any result was looked at: a dataset enters the map and transfer comparisons if it has at least
300 guides; smaller ones enter only the position-20 contrast and are marked as underpowered.

Orientation. Scores differ in direction (a fitness screen reports log fold change, where lower means more active). Each dataset is oriented
so that higher means more active by the sign of the median Spearman correlation of its label with six published activity predictors
(Azimuth, CRISPRscan, Rule Set 1, Wang, SSC, WU-CRISPR). The predictors are used for the sign only, never fitted. Their disagreement is
printed so a doubtful orientation is visible. E. coli's label is oriented as everywhere in this report.

Estimators, all the same as Finding 7's: the effect of base b at guide position p is the mean label of guides with b at p minus the grand
mean, in SD units (positions 1-20, 20 PAM-proximal); two maps are compared by the Spearman correlation across the 80 cells. Each map's
split-half reliability (Spearman-Brown corrected, 200 random splits) says how much correlation a perfectly identical rule could show, and
the attenuation-corrected correlation divides by it. Intervals are bootstrapped over guides.

Predictions, written before the run:
  P1  Mammalian maps agree with one another: corrected correlation at least 0.4 for every pair of different libraries, and mouse-human
      no lower than human-human by more than 0.2.
  P2  E. coli's map is unrelated to every eukaryotic map (|rho| <= 0.2), and the position-20 contrast (G minus T) has the opposite sign to
      E. coli's in every eukaryotic dataset with at least 300 guides: the inversion is bacteria against eukaryotes, not human-specific.
  P3  Zebrafish agrees with mammals less than mammals agree with each other (0.2 to 0.5).
  P4  Transfer (model trained on one dataset, scored on another, label rank only): mammal to mammal retains at least 60% of the target's own
      cross-validated rho; E. coli to any eukaryote is within 0.05 of zero.
  P5  No AT-rich step in the +-(30/47) nt context of mammalian datasets (lowest-decile gap with an interval including 0, rho inside the
      tail under 0.15), probability about 0.7; E. coli at the same scale has gap -0.76 SD and rho 0.435 inside the tail. A step in
      mammals would mean the cause is generic accessibility and not a bacterial protein.

    python -m sgrna.compare.eukaryote_rules            # maps, reliability, contrast, flank step, about a minute
    python -m sgrna.compare.eukaryote_rules --transfer # adds the transfer matrix (a few minutes)
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from sgrna import config

from ..common import OUT

DATA = config.ROOT / "external_data" / "crispor_effdata"
TOOLS = ["fusi", "crisprScan", "doench", "wang", "ssc", "wuCrispr"]
MIN_BIG = 300
BASES = "ACGT"

#            file stem                    species      library (datasets sharing a library score the same guides)
CATALOGUE = {
    "chari2015Train293T":      ("human", "chari"),     "doench2014-Hs":          ("human", "doench2014-hs"),
    "doench2016_hg19":         ("human", "doench2016"), "hart2016-HelaLib1Avg":   ("human", "hart-lib1"),
    "hart2016-Hct1161lib1Avg": ("human", "hart-lib1"), "wang2015_hg19":          ("human", "wang2015"),
    "xu2015TrainHl60":         ("human", "xu2015"),    "xu2015TrainKbm7":        ("human", "xu2015"),
    "doench2014-Mm":           ("mouse", "doench2014-mm"), "xu2015TrainMEsc":     ("mouse", "xu2015-mesc"),
    "morenoMateos2015":        ("zebrafish", "moreno"),
    # below the 300-guide line
    "concordet2-Hs": ("human", "concordet"), "liu2016_mm9": ("mouse", "liu"), "teboulVitro_mm9": ("mouse", "teboul-vitro"),
    "teboulVivo_mm9": ("mouse", "teboul-vivo"), "concordet2-Mm": ("mouse", "concordet"), "schoenigMm": ("mouse", "schoenig"),
    "concordet2-Rn": ("rat", "concordet"), "schoenigRn": ("rat", "schoenig"), "varshney2015": ("zebrafish", "varshney"),
    "gagnon2014": ("zebrafish", "gagnon"), "housden2015": ("fly", "housden"), "ren2015": ("fly", "ren"), "farboud2015": ("worm", "farboud"),
}


def load_all():
    out = {}
    for stem, (species, library) in CATALOGUE.items():
        t = pd.read_csv(DATA / f"{stem}.scores.tab", sep="\t")
        y = pd.to_numeric(t["modFreq"], errors="coerce")
        t = t[y.notna()].copy(); y = y[y.notna()].to_numpy(float)
        votes = []
        for c in TOOLS:
            s = pd.to_numeric(t[c], errors="coerce")
            m = s.notna().to_numpy()
            votes.append(float(spearmanr(y[m], s[m]).statistic) if m.sum() > 10 and s[m].nunique() > 2 else np.nan)
        sign = 1.0 if np.nanmedian(votes) >= 0 else -1.0
        z = sign * (y - y.mean()) / y.std()
        protos = t["seq"].str.upper().str[:20].to_numpy()
        ctx = t["longSeq100Bp"].astype(str).str.upper().to_numpy()
        out[stem] = dict(species=species, library=library, n=len(t), y=z, protos=protos, ctx=ctx, sign=sign, votes=votes,
                         anchor=float(np.nanmedian(votes)), tools_agree=int(np.sum(np.sign(np.nan_to_num(votes)) == sign)))
    return out


def ecoli():
    from sgrna.harness.curated import load_base
    from sgrna.build.build_features import load_curated_index
    X, y, names, protos = load_base()
    idx = load_curated_index(verbose=False).set_index(config.ID_COL).reindex(protos)
    ctx = (idx["upstream"].str[-30:] + "N" * 23 + idx["downstream"].str[:47]).to_numpy()       # same geometry as the other tables
    return dict(species="E. coli", library="curated", n=len(y), y=(y - y.mean()) / y.std(), protos=np.asarray(protos, dtype=object), ctx=ctx, X=X,
                sign=1.0, votes=[], anchor=np.nan, tools_agree=0, names=list(names))


def _codes(protos):
    a = np.frombuffer("".join(protos).encode(), np.uint8).reshape(len(protos), 20)
    lut = np.full(256, 255, np.uint8)
    for i, c in enumerate(BASES):
        lut[ord(c)] = i
    return lut[a]


def position_base_map(codes, y, min_n=20):
    """effect_sd[20, 4]: mean label of guides with base b at position p, minus the grand mean, in SD units (label already has SD 1)."""
    m = np.full((20, 4), np.nan)
    mu = y.mean()
    for p in range(20):
        for b in range(4):
            sel = codes[:, p] == b
            if sel.sum() >= min_n:
                m[p, b] = y[sel].mean() - mu
    return m.ravel()


def _rho_maps(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() > 10 else np.nan


def split_half(codes, y, rng, reps=200):
    r = []
    for _ in range(reps):
        perm = rng.permutation(len(y)); h = len(y) // 2
        r.append(_rho_maps(position_base_map(codes[perm[:h]], y[perm[:h]], 10), position_base_map(codes[perm[h:]], y[perm[h:]], 10)))
    r = float(np.nanmean(r))
    return max(2 * r / (1 + r), 0.0) if np.isfinite(r) else np.nan


def maps_and_pairs(D, rng, boot=200):
    big = [k for k, v in D.items() if v["n"] >= MIN_BIG]
    codes = {k: _codes(D[k]["protos"]) for k in D}
    maps = {k: position_base_map(codes[k], D[k]["y"]) for k in D}
    rel = {k: split_half(codes[k], D[k]["y"], rng) for k in big}
    rows = []
    for a, b in itertools.combinations(big, 2):
        r = _rho_maps(maps[a], maps[b])
        bs = []
        for _ in range(boot):
            ia = rng.integers(0, D[a]["n"], D[a]["n"]); ib = rng.integers(0, D[b]["n"], D[b]["n"])
            bs.append(_rho_maps(position_base_map(codes[a][ia], D[a]["y"][ia]), position_base_map(codes[b][ib], D[b]["y"][ib])))
        den = np.sqrt(rel[a] * rel[b])
        rows.append(dict(a=a, b=b, species_a=D[a]["species"], species_b=D[b]["species"], same_library=D[a]["library"] == D[b]["library"],
                         rho=r, lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)), reliability_a=rel[a], reliability_b=rel[b],
                         rho_corrected=float(np.clip(r / den, -1, 1)) if den > 0.05 else np.nan))
    pairs = pd.DataFrame(rows)
    mp = pd.DataFrame([dict(dataset=k, position=p // 4 + 1, base=BASES[p % 4], effect_sd=maps[k][p]) for k in D for p in range(80)])
    mp.to_csv(OUT / "eukaryote_maps.csv", index=False)
    pairs.to_csv(OUT / "eukaryote_map_pairs.csv", index=False)
    return maps, rel, pairs, codes


def pair_kind(r):
    sa, sb = sorted([r["species_a"], r["species_b"]])
    return f"{sa} - {sb}"


def contrast(D, codes, rng, boot=400):
    """Effect of G minus effect of T at position 20 (and at 18), in SD units, with a bootstrap interval, for every dataset and pooled by species."""
    def one(c, y, pos, other=(3,)):
        g, t = c[:, pos] == 2, np.isin(c[:, pos], other)
        return float(y[g].mean() - y[t].mean()) if g.sum() > 3 and t.sum() > 3 else np.nan
    rows = []
    groups = {k: [k] for k in D}
    for sp in sorted({v["species"] for v in D.values() if v["species"] != "E. coli"}):
        groups[f"{sp} (pooled)"] = [k for k, v in D.items() if v["species"] == sp]
    for name, members in groups.items():
        c = np.vstack([codes[k] for k in members]); y = np.concatenate([D[k]["y"] for k in members])
        row = dict(group=name, n=len(y), species=D[members[0]]["species"], underpowered=len(y) < MIN_BIG)
        for pos, tag in ((19, "pos20"), (17, "pos18")):
            pt = one(c, y, pos)
            bs = [one(c[ix], y[ix], pos) for ix in (rng.integers(0, len(y), len(y)) for _ in range(boot))]
            row[f"g_minus_t_{tag}"] = pt; row[f"{tag}_lo"] = float(np.nanpercentile(bs, 2.5)); row[f"{tag}_hi"] = float(np.nanpercentile(bs, 97.5))
        # defined even where a library has no T at position 20 (the Hart libraries): G minus the mean of A and C
        c_, y_ = c, y
        ac = lambda cc, yy: float(yy[cc[:, 19] == 2].mean() - yy[np.isin(cc[:, 19], (0, 1))].mean())
        bs = [ac(c_[ix], y_[ix]) for ix in (rng.integers(0, len(y_), len(y_)) for _ in range(boot))]
        row["g_minus_ac_pos20"] = ac(c_, y_); row["g_minus_ac_lo"] = float(np.nanpercentile(bs, 2.5)); row["g_minus_ac_hi"] = float(np.nanpercentile(bs, 97.5))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "eukaryote_contrast_g_minus_t.csv", index=False)
    return out


def _gc(s):
    return sum(ch in "GC" for ch in s) / max(sum(ch in "ACGT" for ch in s), 1)


def flank_step(D, rng, boot=300):
    """The AT-rich step at the scale the context allows: 30 nt before the guide and 47 nt after the PAM (77 nt)."""
    from .. interpret.mechanism import _partial
    rows = []
    for k, v in D.items():
        if v["n"] < MIN_BIG:
            continue
        x = np.array([_gc(c[:30] + c[53:]) for c in v["ctx"]])
        tgt = np.array([_gc(p) for p in v["protos"]]); y = v["y"]
        cut = float(np.quantile(x, 0.10)); lo = x <= cut
        def gap(ix):
            m = lo[ix]; return float(y[ix][m].mean() - y[ix][~m].mean()) if m.sum() > 5 else np.nan
        bs = [gap(rng.integers(0, len(y), len(y))) for _ in range(boot)]
        rows.append(dict(dataset=k, species=v["species"], n=len(y), flank_gc_at_decile_edge=cut, gap_sd=gap(np.arange(len(y))),
                         lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)),
                         rho_below=float(spearmanr(x[lo], y[lo]).statistic), rho_above=float(spearmanr(x[~lo], y[~lo]).statistic),
                         rho_full=float(spearmanr(x, y).statistic), rho_given_target_gc=_partial(x, y, tgt)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "eukaryote_flank_step.csv", index=False)
    return out


# ------------------------------------------------------------------------------- transfer
def _onehot_dinuc(codes):
    codes = codes.astype(np.int64)
    n = len(codes)
    oh = np.zeros((n, 80), np.float32)
    for p in range(20):
        for b in range(4):
            oh[:, p * 4 + b] = codes[:, p] == b
    di = np.zeros((n, 19 * 16), np.float32)
    for p in range(19):
        ok = (codes[:, p] < 4) & (codes[:, p + 1] < 4)
        di[np.flatnonzero(ok), p * 16 + codes[ok, p] * 4 + codes[ok, p + 1]] = 1
    return np.hstack([oh, di])


def transfer(D, ec, rng):
    """Zero-shot rank correlation of a model trained on one dataset and scored on another, with within-dataset 5-fold CV on the diagonal."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from ..build.base_fast import matrix_fast
    from .vs_crisprhal import member_specs, train_predict
    big = [k for k, v in D.items() if v["n"] >= MIN_BIG]
    Xb = {k: np.asarray(matrix_fast(list(D[k]["protos"]))[0], np.float32) for k in big}
    Xb["E. coli"] = np.asarray(ec["X"], np.float32)
    Xr = {k: _onehot_dinuc(_codes(D[k]["protos"])) for k in big}; Xr["E. coli"] = _onehot_dinuc(_codes(ec["protos"]))
    Y = {k: D[k]["y"] for k in big}; Y["E. coli"] = ec["y"]
    specs = member_specs(["champion"])
    models = {"pipeline (SLICER base, champion settings)": lambda Xa, ya, Xt: train_predict(specs, Xa, ya, Xt, seed=41)[0]["champion"],
              "ridge on one-hot + dinucleotides": lambda Xa, ya, Xt: RidgeCV(alphas=np.logspace(-1, 3, 9)).fit(Xa, ya).predict(Xt)}
    feats = {"pipeline (SLICER base, champion settings)": Xb, "ridge on one-hot + dinucleotides": Xr}
    rows = []
    trainers = ["E. coli"] + big
    for mname, fit in models.items():
        F = feats[mname]
        for a in trainers:
            targets = [b for b in big if b != a]
            if mname.startswith("pipeline") and a == "E. coli":
                pass
            Xt = np.vstack([F[b] for b in targets]); sizes = np.cumsum([0] + [len(Y[b]) for b in targets])
            p = fit(F[a], Y[a], Xt)
            for i, b in enumerate(targets):
                pb = p[sizes[i]:sizes[i + 1]]
                rows.append(dict(model=mname, trained_on=a, scored_on=b, rho=float(spearmanr(Y[b], pb).statistic), within=False))
            print(f"  {mname[:8]} trained on {a:26s} done", flush=True)
        for b in big:
            pred = np.zeros(len(Y[b]))
            for tr, te in KFold(5, shuffle=True, random_state=41).split(np.arange(len(Y[b]))):
                pred[te] = fit(F[b][tr], Y[b][tr], F[b][te])
            rows.append(dict(model=mname, trained_on=b, scored_on=b, rho=float(spearmanr(Y[b], pred).statistic), within=True))
    out = pd.DataFrame(rows)
    sp = {**{k: D[k]["species"] for k in big}, "E. coli": "E. coli"}
    lib = {**{k: D[k]["library"] for k in big}, "E. coli": "E. coli"}
    out["species_trained"] = out.trained_on.map(sp); out["species_scored"] = out.scored_on.map(sp)
    out["same_library"] = out.trained_on.map(lib) == out.scored_on.map(lib)
    out.to_csv(OUT / "eukaryote_transfer.csv", index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", action="store_true")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(41)
    D = load_all()
    print(pd.DataFrame([dict(dataset=k, species=v["species"], n=v["n"], sign=v["sign"], median_tool_rho=round(v["anchor"], 3),
                             tools_agreeing_with_sign=f"{v['tools_agree']}/6") for k, v in D.items()]).to_string(index=False))
    pd.DataFrame([dict(dataset=k, species=v["species"], library=v["library"], n=v["n"], sign=v["sign"], median_tool_rho=v["anchor"],
                       tools_agreeing=v["tools_agree"], **{f"rho_{t}": r for t, r in zip(TOOLS, v["votes"])}) for k, v in D.items()]).to_csv(
        OUT / "eukaryote_orientation.csv", index=False)
    ec = ecoli()
    D["E. coli"] = ec
    maps, rel, pairs, codes = maps_and_pairs(D, rng)
    pairs["kind"] = pairs.apply(pair_kind, axis=1)
    print("\nsplit-half reliability of each map:", {k: round(v, 2) for k, v in rel.items()})
    print("\nmap correlations by kind of pair (different libraries only):")
    diff = pairs[~pairs.same_library]
    print(diff.groupby("kind").agg(pairs=("rho", "size"), rho_median=("rho", "median"), rho_min=("rho", "min"), rho_max=("rho", "max"),
                                    corrected_median=("rho_corrected", "median")).round(3).to_string())
    print("\nthe weakest eukaryote-eukaryote pairs (different libraries):")
    print(diff[diff.species_a.ne("E. coli") & diff.species_b.ne("E. coli")].sort_values("rho_corrected").head(8)[["a", "b", "rho", "rho_corrected", "reliability_a", "reliability_b"]].round(3).to_string(index=False))
    print("\nE. coli against each eukaryote:"); print(pairs[(pairs.a == "E. coli") | (pairs.b == "E. coli")][["a", "b", "rho", "lo", "hi", "rho_corrected"]].round(3).to_string(index=False))
    c = contrast(D, codes, rng)
    print("\nG minus T at position 20 (and 18), SD units:"); print(c.round(3).to_string(index=False))
    f = flank_step(D, rng)
    print("\nAT-rich step in the 77-nt context:"); print(f.round(3).to_string(index=False))
    if a.transfer:
        t = transfer({k: v for k, v in D.items() if k != "E. coli"}, ec, rng)
        print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
