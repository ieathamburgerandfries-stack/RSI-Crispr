"""A position-resolved description of each enzyme: where along the target its substitutions sit, and what they change.

Instead of summarising an enzyme with a few global numbers, each substituted residue is mapped, through the R-loop
complex 5F9R, onto the target positions it neighbours, giving one profile per enzyme over the target:

  P_c(e, i) = sum over substituted residues r of  w(r, i) * delta_c(r)        c = charge, volume, H-bond capacity
  C(e, i)   = sum over substituted residues r of  w(r, i)                      (where the changes are, whatever they do)
  w(r, i)   = exp(-d(r, i) / 4 A), d = nearest heavy-atom distance from the wild-type residue to target position i

Target positions i = 1..20 are protospacer positions counted from the PAM (template-strand residue 10 + i in 5F9R);
PAM positions 1..4 are the four PAM base pairs (both strands). The profile has 24 entries per chemistry.

Test of whether the profile means anything: the sequence dependence of an enzyme at each target position is read from
the data (spread of the one-hot ridge coefficients at that position), and the change in sequence dependence relative to
SpCas9 is compared with where the enzyme's substitutions sit. The null shuffles which enzyme's contact profile is paired
with which enzyme's effect profile (all 8! pairings of the eight variants), so any shared structure between positions
(PAM-proximal positions matter more for everyone) is kept and only the enzyme-specific pairing is broken.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import nuclease_desc as nd
from . import nuclease_model as nm
from . import pam_variants as pv

VOL, HBD, HBA = pv.VOL, pv.HBD, pv.HBA


def _atoms():
    prot, ts, nts = {}, {}, {}
    for l in open(nd.PDB):
        if not l.startswith(("ATOM", "HETATM")) or l[76:78].strip() == "H":
            continue
        ch = l[21]
        xyz = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        num = int(l[22:26])
        if ch == "B":
            prot.setdefault(num, []).append(xyz)
        elif ch == "C" and l[17:20].strip().startswith("D"):
            ts.setdefault(num, []).append(xyz)
        elif ch == "D":
            nts.setdefault(num, []).append(xyz)
    return {k: np.array(v) for k, v in prot.items()}, {k: np.array(v) for k, v in ts.items()}, \
        {k: np.array(v) for k, v in nts.items()}


def target_sites() -> list[tuple[str, np.ndarray]]:
    """24 target sites: 20 protospacer positions (template strand), 4 PAM base pairs (both strands)."""
    _, ts, nts = _atoms()
    sites = [(f"P{i}", ts[10 + i]) for i in range(1, 21)]
    for j in range(4):
        sites.append((f"PAM{j + 1}", np.vstack([nts[21 + j], ts[10 - j]])))
    return sites


def weights(positions: list[int]) -> pd.DataFrame:
    prot, _, _ = _atoms()
    sites = target_sites()
    out = {}
    for p in positions:
        out[p] = [float(np.exp(-np.min(np.linalg.norm(prot[p][:, None] - s[None], axis=2)) / 4.0)) for _, s in sites]
    return pd.DataFrame(out, index=[n for n, _ in sites]).T


def profiles() -> dict[str, pd.DataFrame]:
    """Rows: the nine nucleases; columns: the 24 target sites."""
    allpos = sorted({int(m[1:-1]) for ms in nd.MUTATIONS.values() for m in ms})
    W = weights(allpos)
    cols = list(W.columns)
    out = {k: pd.DataFrame(0.0, index=nd.NUCLEASES, columns=cols) for k in ("contact", "charge", "volume", "hbond")}
    for v, muts in nd.MUTATIONS.items():
        for m in muts:
            wt, pos, to = m[0], int(m[1:-1]), m[-1]
            w = W.loc[pos].to_numpy()
            out["contact"].loc[v] += w
            out["charge"].loc[v] += w * (nd.CHARGE.get(to, 0) - nd.CHARGE.get(wt, 0))
            out["volume"].loc[v] += w * (VOL[to] - VOL[wt]) / 100.0
            out["hbond"].loc[v] += w * ((HBD.get(to, 0) + HBA.get(to, 0)) - (HBD.get(wt, 0) + HBA.get(wt, 0)))
    return out


def protein_composition_l1() -> pd.DataFrame:
    """L1 distance of each enzyme's 20-amino-acid composition (whole 1,368-residue protein) from SpCas9's.

    The mutation lists are applied to the SEQRES of 5F9R chain B; every wild-type residue must match (this also checks the
    numbering of the lists)."""
    seq = []
    for l in open(nd.PDB):
        if l.startswith("SEQRES") and l[11] == "B":
            seq += [nd.AA3.get(x, "X") for x in l[19:].split()]
    assert len(seq) == 1368 and "X" not in seq
    aas = sorted(set(seq))
    rows, ref = [], None
    for v, muts in nd.MUTATIONS.items():
        s = list(seq)
        for m in muts:
            assert s[int(m[1:-1]) - 1] == m[0], (v, m)
            s[int(m[1:-1]) - 1] = m[-1]
        comp = np.array([s.count(a) / len(s) for a in aas])
        ref = comp if v == "SpCas9" else ref
        rows.append(dict(nuclease=v, n_mut=len(muts), l1=float(np.abs(comp - ref).sum()),
                         max_single=float(np.abs(comp - ref).max())))
    return pd.DataFrame(rows)


def sequence_dependence(alpha=30.0) -> pd.DataFrame:
    """Per enzyme and 30-mer position: spread (sd over the four bases) of one-hot ridge coefficients.

    Fitted on the enzyme's training-split targets only. Returned for the 20 protospacer positions in PAM-proximal order
    (P1 = the base next to the PAM) plus the three PAM bases."""
    from sklearn.linear_model import Ridge
    d = nm.load()
    out = {}
    for hk, hn in zip(nd.SHORT, nd.NUCLEASES):
        r = nm.rows_for(d, hk, "Training")
        c = nm.CODE[np.frombuffer("".join(d.seq.iloc[r]).encode(), dtype=np.uint8)].reshape(len(r), 30).astype(int)
        X = np.zeros((len(r), 30, 4), np.float32)
        X[np.arange(len(r))[:, None], np.arange(30)[None], c] = 1
        y = d[hk].to_numpy()[r]
        m = Ridge(alpha=alpha).fit(X.reshape(len(r), -1), y - y.mean())
        beta = m.coef_.reshape(30, 4)
        sd = beta.std(1)                                                    # sequence dependence per position
        prox = {f"P{i}": sd[24 - i] for i in range(1, 21)}                 # proximal position i <-> 1-based index 25-i
        prox.update({f"PAM{j}": sd[24 + j - 1] for j in (1, 2, 3)})
        out[hn] = prox
    return pd.DataFrame(out).T


def pairing_test(prof: pd.DataFrame, dep: pd.DataFrame, sites: list[str], ref="SpCas9", enzymes=None,
                 normalise=True, return_each=False) -> dict:
    """Exact permutation test over the non-reference enzymes in `enzymes` (default: all eight).

    Statistic: mean over enzymes of the Spearman correlation across sites between the enzyme's profile and the change in
    its sequence dependence relative to `ref`. With `normalise`, each enzyme's sequence-dependence profile is first
    scaled to sum to one over `sites`, so a global change in activity spread (which rescales every position) cannot
    masquerade as a position-specific change."""
    others = [v for v in (enzymes or nd.NUCLEASES) if v != ref]
    d = dep[sites].copy()
    if normalise:
        d = d.div(d.sum(axis=1), axis=0)
    delta = (d.loc[others] - d.loc[ref]).to_numpy()
    P = prof.loc[others, sites].to_numpy()

    def each(perm):
        return [0.0 if np.ptp(P[perm[k]]) < 1e-12 else float(spearmanr(P[perm[k]], delta[k]).statistic)
                for k in range(len(others))]

    ident = list(range(len(others)))
    obs_each = each(ident)
    obs = float(np.mean(obs_each))
    null = np.array([np.mean(each(p)) for p in itertools.permutations(ident)])
    out = dict(n_enzymes=len(others), obs=obs, p_ge=float((null >= obs - 1e-12).mean()),
               null_mean=float(null.mean()), null_sd=float(null.std()), n_perm=len(null))
    if return_each:
        out["each"] = dict(zip(others, [round(x, 3) for x in obs_each]))
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    protein_composition_l1().to_csv(nm.OUT / "protein_composition_l1.csv", index=False)
    pr = profiles()
    print(pr["contact"].round(2).to_string())
    dep = sequence_dependence()
    dep.to_csv(nm.OUT / "enzyme_sequence_dependence.csv")
    sites = [f"P{i}" for i in range(1, 21)] + ["PAM1", "PAM2", "PAM3"]
    rows = []
    sets = {"all eight": None, "high-fidelity five": ["Sniper-Cas9", "eSpCas9(1.1)", "SpCas9-HF1", "HypaCas9", "evoCas9"],
            "PAM variants three": ["SpCas9-NG", "VRQR", "xCas9"]}
    for c, df in pr.items():
        for sub, ss in (("protospacer", sites[:20]), ("protospacer+PAM", sites)):
            for setname, en in sets.items():
                for norm in (True, False):
                    r = pairing_test(df, dep, ss, enzymes=en, normalise=norm, return_each=(c == "contact" and norm))
                    rows.append(dict(profile=c, sites=sub, enzymes=setname, normalised=norm, **r))
    t = pd.DataFrame(rows); t.to_csv(nm.OUT / "enzyme_contact_pairing.csv", index=False)
    show = t[(t.normalised) ].drop(columns=["each", "n_perm"], errors="ignore")
    print(show.round(4).to_string(index=False))
    print(t[(t.profile == "contact") & t.normalised][["sites", "enzymes", "each"]].to_string(index=False))
