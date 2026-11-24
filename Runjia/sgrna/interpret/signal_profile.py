"""The data behind the figure of what the flank signal looks like: the step, the slope by the guide's last base, and the correlation by window and by base.

All on the 33,567 curated guides (30,203 inside a gene for the first two tables, as in the report's decile table). Flank GC is the mean of the GC
fraction of the 250 nt upstream of the protospacer and the 250 nt downstream of the PAM, on the guide's strand.

  decile    mean label by decile of flank GC, with 95% intervals from resampling genes: the step at the AT-rich end
  slope     the label against flank GC in four strata, by the base at guide position 20 (next to the PAM): Spearman correlation with an interval from
            resampling genes, and the mean label in quintiles of flank GC; the slope is the thing that differs between strata
  corr      for windows of 10 nt to 5 kb on each side: the marginal Spearman correlation of the label with the fraction of each base (A, C, G, T) and of
            GC and purine, and the partial correlation given the target's own GC; the base-resolved version of the report's GC-versus-purine table

    python -m sgrna.interpret.signal_profile
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .. import config
from ..common import OUT
from .flank_scale import WINDOWS, flank_composition

DECILE, SLOPE, CORR = OUT / "signal_decile.csv", OUT / "signal_slope.csv", OUT / "signal_corr.csv"


def _load():
    from ..build.build_features import load_curated_index

    idx = load_curated_index(verbose=False).reset_index(drop=True)
    y = idx["score"].to_numpy(float)
    gene = np.array([g if isinstance(g, str) and g else "" for g in idx["gene_name"]], dtype=object)
    comp = flank_composition(idx, WINDOWS)
    proto = idx["protospacer"].astype(str)
    tgc = np.array([(s.count("G") + s.count("C")) / 20 for s in proto])
    last = proto.str[-1].to_numpy()
    return idx, y, gene, comp, tgc, last


def _boot(values_fn, gene, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    gi = pd.Series(np.arange(len(gene))).groupby(gene).apply(lambda s: s.to_numpy())
    keys = [k for k in gi.index if k != ""]
    out = []
    for _ in range(B):
        take = np.concatenate([gi[k] for k in rng.choice(keys, len(keys))])
        out.append(values_fn(take))
    return np.array(out)


def _partial(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])


def decile(y, gene, fgc):
    g = gene != ""
    yy, ff, gg = y[g], fgc[g], gene[g]
    edges = np.quantile(ff, np.linspace(0, 1, 11))
    d = np.clip(np.searchsorted(edges, ff, side="right") - 1, 0, 9)
    rows = []
    for k in range(10):
        m = d == k
        b = _boot(lambda t, k=k: float(yy[t][d[t] == k].mean()), gg, 400, seed=k)
        rows.append(dict(decile=k + 1, n=int(m.sum()), flank_gc=float(ff[m].mean()), mean_label=float(yy[m].mean()),
                         lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5))))
    rest = d >= 1
    rows[0]["gap_vs_rest"] = float(yy[d == 0].mean() - yy[rest].mean())
    return pd.DataFrame(rows)


def slope(y, gene, fgc, last):
    g = gene != ""
    yy, ff, gg, ll = y[g], fgc[g], gene[g], last[g]
    rows = []
    for b in "ACGT":
        m = ll == b
        r = float(spearmanr(ff[m], yy[m]).statistic)
        bs = _boot(lambda t, b=b: float(spearmanr(ff[t][ll[t] == b], yy[t][ll[t] == b]).statistic), gg, 300, seed=ord(b))
        q = np.clip(np.searchsorted(np.quantile(ff, np.linspace(0, 1, 6)), ff[m], side="right") - 1, 0, 4)
        for k in range(5):
            rows.append(dict(base=b, n=int(m.sum()), spearman=r, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                             quintile=k + 1, flank_gc=float(ff[m][q == k].mean()), mean_label=float(yy[m][q == k].mean())))
    return pd.DataFrame(rows)


def corr(y, comp, tgc):
    rows = []
    for W in WINDOWS:
        up, dn = comp[W]
        for side, arr in (("upstream", up), ("downstream", dn)):
            stats = {"A": arr[:, 0], "C": arr[:, 1], "G": arr[:, 2], "T": arr[:, 3], "GC": arr[:, 1] + arr[:, 2], "purine": arr[:, 0] + arr[:, 2]}
            for name, v in stats.items():
                rows.append(dict(window_nt=W, side=side, statistic=name, marginal=float(spearmanr(v, y).statistic), partial_given_target_gc=_partial(v, y, tgc)))
    return pd.DataFrame(rows)


def target_gc_strata(idx, y, gene):
    """Family A's gain inside quintiles of the target's own GC, on the stored gene-grouped out-of-fold predictions (`mappability_stratum_oof.csv`)."""
    from ..controls.mappability_stratum import _boot_gain

    o = pd.read_csv(OUT / "mappability_stratum_oof.csv")
    assert (o[config.ID_COL].to_numpy() == idx[config.ID_COL].to_numpy()).all()
    tgc = np.array([(s.count("G") + s.count("C")) / 20 for s in idx["protospacer"].astype(str)])
    edges = np.quantile(tgc, np.linspace(0, 1, 6))
    q = np.clip(np.searchsorted(edges, tgc, side="right") - 1, 0, 4)
    rows = []
    for k in range(5):
        s = o[q == k]
        lo, hi = _boot_gain(s, "base+A", "base", B=400)
        rows.append(dict(quintile=k + 1, target_gc_mean=float(tgc[q == k].mean()), n=int((q == k).sum()),
                         rho_base=float(spearmanr(s.y, s["base"]).statistic), rho_with_flank=float(spearmanr(s.y, s["base+A"]).statistic),
                         gain=float(spearmanr(s.y, s["base+A"]).statistic - spearmanr(s.y, s["base"]).statistic), gain_lo=lo, gain_hi=hi))
    return pd.DataFrame(rows)


def main() -> int:
    idx, y, gene, comp, tgc, last = _load()
    fgc = ((comp[250][0][:, 1] + comp[250][0][:, 2]) + (comp[250][1][:, 1] + comp[250][1][:, 2])) / 2
    d = decile(y, gene, fgc); d.to_csv(DECILE, index=False); print(d.round(3).to_string(index=False))
    s = slope(y, gene, fgc, last); s.to_csv(SLOPE, index=False); print(s.drop_duplicates("base")[["base", "n", "spearman", "lo", "hi"]].round(3).to_string(index=False))
    tg = target_gc_strata(idx, y, gene); tg.to_csv(OUT / "signal_target_gc.csv", index=False); print(tg.round(4).to_string(index=False))
    c = corr(y, comp, tgc); c.to_csv(CORR, index=False)
    print(c[c.statistic.isin(["GC", "purine"]) & c.window_nt.isin([50, 250, 500, 1000])].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
