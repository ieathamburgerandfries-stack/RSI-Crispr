"""Summaries of the replicated leave-one-nuclease-out run, with paired differences.

For each (held-out nuclease, replicate) the score of an arm is compared with the no-information
arm U trained on the same rows; the unit of analysis is the nuclease (n = 9), replicates are
averaged within it. Shuffled-descriptor arms are averaged over their permutations first.
Reported: mean difference over nuclease, the per-nuclease values, how many of 9 are ahead, and a
bootstrap over nuclease (2,000 resamples). No claim is made from the 9 nucleases beyond what the
spread shows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import OUT


def load():
    d = pd.read_csv(OUT / "nuclease_lono_rep.csv")
    d["base"] = d.arm.str.replace(r"_perm\d+$", "", regex=True)
    d["is_perm"] = d.arm.str.contains("_perm")
    return d


def per_nuclease(d, metric):
    g = d.groupby(["base", "is_perm", "nuclease", "rep"])[metric].mean().reset_index()
    g["arm"] = np.where(g.is_perm, g.base + " [shuffled]", g.base)
    return g.groupby(["arm", "nuclease"])[metric].mean().unstack("nuclease")


def table(metric="rho", ref="U"):
    p = per_nuclease(load(), metric)
    rng = np.random.default_rng(0)
    rows = []
    for arm in p.index:
        diff = (p.loc[arm] - p.loc[ref]).to_numpy()
        boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        rows.append(dict(arm=arm, mean=p.loc[arm].mean(), d_vs_ref=diff.mean(),
                         lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                         ahead=f"{int((diff > 0).sum())}/{len(diff)}"))
    return pd.DataFrame(rows).set_index("arm").round(4), p


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for m in ("rho", "rho_ngg"):
        t, p = table(m)
        print(f"\n{m} (reference U, no nuclease information)"); print(t.to_string())
        t.to_csv(OUT / f"lono_summary_{m}.csv"); p.round(4).to_csv(OUT / f"lono_by_nuclease_{m}.csv")
    print("\nper nuclease (rho):"); print(per_nuclease(load(), "rho").round(3).to_string())


def by_pam_class(min_n=100):
    """Recompute the held-out scores from the saved predictions, split by PAM class: where does a
    descriptor help -- on the canonical NGG targets, on the others, or in the ordering of PAM classes?"""
    from scipy.stats import spearmanr

    from . import nuclease_desc as nd
    from . import nuclease_model as nm
    d = nm.load()
    pam = d.seq.str[25:27].to_numpy()
    rows = []
    pred_dir = OUT / "lono_preds"
    for hk, hn in zip(nd.SHORT, nd.NUCLEASES):
        te = nm.rows_for(d, hk, "Test")
        y = d[hk].to_numpy()[te]; pm = pam[te]
        ngg = pm == "GG"
        classes = [c for c in np.unique(pm) if (pm == c).sum() >= min_n]
        for arm in ("U", "D_mutations", "D_structure", "D_label-free", "D_pam", "D_scale", "D_all", "ensemble-of-sources", "own"):
            r = []
            for rep in range(3):
                p = np.load(pred_dir / f"{arm}__{hn}__r{rep}.npy")
                within = np.nanmean([spearmanr(y[pm == c], p[pm == c]).statistic for c in classes])
                # ordering of PAM classes: rho between class means
                cm = np.array([[y[pm == c].mean(), p[pm == c].mean()] for c in classes])
                r.append(dict(all=spearmanr(y, p).statistic, ngg=spearmanr(y[ngg], p[ngg]).statistic,
                              non_ngg=spearmanr(y[~ngg], p[~ngg]).statistic, within_class=within,
                              class_order=spearmanr(cm[:, 0], cm[:, 1]).statistic))
            rows.append(dict(nuclease=hn, arm=arm, **pd.DataFrame(r).mean().to_dict()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t = by_pam_class()
    t.to_csv(OUT / "lono_by_pam_class.csv", index=False)
    for m in ("all", "ngg", "non_ngg", "within_class", "class_order"):
        p = t.pivot(index="arm", columns="nuclease", values=m)
        p["mean"] = p.mean(axis=1)
        print(f"\n{m}"); print(p.round(3).to_string())


def bigcap():
    """High-capacity (1500 trees, depth 8) leave-one-nuclease-out: three replicates, PAM-class split."""
    from scipy.stats import spearmanr

    from . import nuclease_desc as nd
    from . import nuclease_model as nm
    d = nm.load(); pam = d.seq.str[25:27].to_numpy()
    rows = []
    for hk, hn in zip(nd.SHORT, nd.NUCLEASES):
        te = nm.rows_for(d, hk, "Test"); y = d[hk].to_numpy()[te]; pm = pam[te]
        classes = [c for c in np.unique(pm) if (pm == c).sum() >= 100]
        for arm in ("U", "D_label-free", "D_all"):
            for rep in range(3):
                f = OUT / "lono_preds_bigcap" / (f"{arm}__{hn}.npy" if rep == 0 else f"{arm}__{hn}__r{rep}.npy")
                p = np.load(f)
                cm = np.array([[y[pm == c].mean(), p[pm == c].mean()] for c in classes])
                rows.append(dict(nuclease=hn, arm=arm, rep=rep, all=spearmanr(y, p).statistic,
                                 ngg=spearmanr(y[pm == "GG"], p[pm == "GG"]).statistic,
                                 class_order=spearmanr(cm[:, 0], cm[:, 1]).statistic,
                                 within_class=np.nanmean([spearmanr(y[pm == c], p[pm == c]).statistic for c in classes])))
    t = pd.DataFrame(rows).groupby(["arm", "nuclease"]).mean(numeric_only=True).drop(columns="rep").reset_index()
    t.to_csv(OUT / "lono_bigcap_by_pam_class.csv", index=False)
    rng = np.random.default_rng(0)
    out = []
    for m in ("all", "ngg", "class_order", "within_class"):
        q = t.pivot(index="arm", columns="nuclease", values=m)
        for arm in ("U", "D_label-free", "D_all"):
            diff = (q.loc[arm] - q.loc["U"]).to_numpy()
            bs = [diff[rng.integers(0, 9, 9)].mean() for _ in range(2000)]
            out.append(dict(metric=m, arm=arm, mean=q.loc[arm].mean(), d_vs_U=diff.mean(),
                            lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), ahead=f"{int((diff > 0).sum())}/9"))
    r = pd.DataFrame(out); r.to_csv(OUT / "lono_bigcap_summary.csv", index=False)
    return r, t
