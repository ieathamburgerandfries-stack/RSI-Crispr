"""A sequence-only PAM-profile descriptor for the nine Kim 2020 nucleases.

The PAM profile is the descriptor group that carried the nuclease effect in the leave-one-nuclease-out experiments, but
it was *measured* for each enzyme. Here it is *predicted* from the enzyme's residues at the six PI-domain positions
(D1135 S1136 G1218 E1219 R1335 T1337) by models trained on the 770 variants of Silverstein et al. 2025 -- composition
(and structure x composition) descriptors only, no Kim measurement of the enzyme involved.

What is representable: the six positions are the only ones the PAMmla library varies. Of the nine Kim enzymes,
  SpCas9, Sniper-Cas9, eSpCas9(1.1), SpCas9-HF1, HypaCas9, evoCas9   -> wild type at all six (DSGERT)
  SpCas9-NG    V1135, R1218, F1219, V1335, R1337 (and L1111R, A1322R, outside the six)   -> VSRFVR
  VRQR         V1135, R1218, Q1335, R1337                                                -> VSREQR
  xCas9        V1219 (and six substitutions elsewhere)                                    -> DSGVRT
so the descriptor has only four distinct values over nine enzymes; the five high-fidelity enzymes and SpCas9 are
indistinguishable by construction, which is correct for PAM usage (they all read NGG) and says nothing about the
enzyme's activity scale or sequence preferences.
Variants whose six-residue string is identical to the query are removed from training for that query.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from . import nuclease_desc as nd
from . import pam_variants as pv

SIX = {"SpCas9": "DSGERT", "SpCas9-NG": "VSRFVR", "VRQR": "VSREQR", "xCas9": "DSGVRT", "Sniper-Cas9": "DSGERT",
       "eSpCas9(1.1)": "DSGERT", "SpCas9-HF1": "DSGERT", "HypaCas9": "DSGERT", "evoCas9": "DSGERT"}


def six_from_mutations() -> dict[str, str]:
    """Derive the six-position string from the mutation lists in `nuclease_desc` (checked against SIX)."""
    out = {}
    for v, muts in nd.MUTATIONS.items():
        s = dict(zip(pv.POSNUM, "DSGERT"))
        for m in muts:
            if int(m[1:-1]) in s:
                assert s[int(m[1:-1])] == m[0], (v, m)
                s[int(m[1:-1])] = m[-1]
        out[v] = "".join(s[p] for p in pv.POSNUM)
    return out


def _predict(query: pd.DataFrame, exclude: str | None, seeds=(0, 1, 2)) -> np.ndarray:
    d, Y, _ = pv.load()
    keep = np.ones(len(d), bool)
    if exclude is not None:
        keep = (d[pv.POS].agg("".join, axis=1) != exclude).to_numpy()
    out = []
    for groups, model in ((["id", "comp"], "ridge"), (["id", "comp", "reach"], "et")):
        Xtr = pv.features(d[keep], groups)[0]
        Xq = pv.features(query, groups)[0]
        sc = StandardScaler().fit(Xtr)
        mu = Y[keep].mean(0)
        if model == "ridge":
            out.append(RidgeCV(alphas=np.logspace(-2, 3, 11)).fit(sc.transform(Xtr), Y[keep] - mu)
                       .predict(sc.transform(Xq)) + mu)
        else:
            for sd in seeds:
                out.append(ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.5, n_jobs=-1,
                                               random_state=sd).fit(sc.transform(Xtr), Y[keep] - mu)
                           .predict(sc.transform(Xq)) + mu)
    # ridge counts as one member, the three extra-trees seeds as one: equal weight to the two model families
    return 0.5 * out[0] + 0.5 * np.mean(out[1:], axis=0)


def kim_profiles() -> pd.DataFrame:
    """Predicted log10 rate constants for the 64 PAMs (N + 3 nt), one row per Kim nuclease."""
    six = six_from_mutations()
    assert all(six[k] == SIX[k] for k in SIX), (six, SIX)
    _, _, pam = pv.load()
    rows = {}
    for nuc in nd.NUCLEASES:
        q = pd.DataFrame([list(six[nuc])], columns=pv.POS)
        rows[nuc] = _predict(q, exclude=six[nuc])[0]
    return pd.DataFrame(rows, index=pam).T


def pam_group_pred() -> pd.DataFrame:
    """The same ten NxxN class shares and mean as `nuclease_desc.pam_group`, from the predicted profiles."""
    P = kim_profiles()
    lin = 10.0 ** P
    out = {}
    for v in nd.NUCLEASES:
        s = lin.loc[v]
        tot = s.sum()
        cls = s.groupby(s.index.str[1:3]).sum() / tot
        out[v] = {f"pam_{k}": cls.get(k, 0.0) for k in ("GG", "GA", "GT", "GC", "AG", "AA", "AT", "AC", "TG", "CG")}
        out[v]["pam_total"] = tot / len(s)
    return pd.DataFrame(out).T.loc[nd.NUCLEASES]


def compare_with_measured() -> pd.DataFrame:
    """Predicted against measured class shares for each nuclease (Spearman over the ten classes) and overall."""
    from scipy.stats import spearmanr
    pred, meas = pam_group_pred().drop(columns="pam_total"), nd.pam_group().drop(columns="pam_total")
    rows = [dict(nuclease=v, rho_classes=float(spearmanr(pred.loc[v], meas.loc[v]).statistic),
                 top_pred=pred.loc[v].idxmax(), top_meas=meas.loc[v].idxmax(),
                 share_GG_pred=pred.loc[v, "pam_GG"], share_GG_meas=meas.loc[v, "pam_GG"]) for v in nd.NUCLEASES]
    allr = float(spearmanr(pred.to_numpy().ravel(), meas.to_numpy().ravel()).statistic)
    t = pd.DataFrame(rows); t.attrs["overall"] = allr
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(pd.Series(six_from_mutations()))
    t = compare_with_measured()
    print(t.round(3).to_string(index=False)); print("overall rho over 90 entries:", round(t.attrs["overall"], 3))
    print(pam_group_pred().round(3).to_string())
