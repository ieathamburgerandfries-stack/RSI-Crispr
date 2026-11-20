"""Many-nuclease test: can composition and structure descriptors of a Cas9 variant predict its PAM profile?

Data: Silverstein et al. 2025 (Nature 643, "Custom CRISPR-Cas9 PAM variants via scalable engineering and machine
learning"), `training_data/220924_6pos_selected_random_only.csv` from github.com/RachelSilverstein/PAMmla (BSD-3):
770 SpCas9 variants that differ at six PI-domain positions (D1135 S1136 G1218 E1219 R1335 T1337), each with the
log10 rate constant of the HT-PAMDA assay for 64 PAMs (N + three PAM nucleotides).

The sheet lumps four kinds of column and they are separated here:
  identifiers   sample, prep, name                       (never used)
  provenance    selected/rational, variant source, source library, PAM selected on, shuffled?, extra mutations
                (used only to define held-out groups and to filter; never as inputs)
  derived       'Active variant?'  -- computed from the outcome columns, so it is leakage if used as an input
  inputs        the six residues
  outcomes      64 PAM rate constants

Descriptor groups (all per position, so they can be added and removed independently):
  id      one-hot residue identity                           (what PAMmla uses)
  comp    composition: hydropathy, side-chain volume, charge, H-bond donors/acceptors, aromatic, Gly/Pro flags
  reach   structure x composition: how far the side chain can reach from the C-alpha of each position, set against the
          distance of that C-alpha to the nearest atom of each PAM base pair in 5F9R (a function of BOTH the residue and the site)
  env     structure only: the position's distances to the PAM base pairs (constant for a position, hence constant
          across variants when the positions are fixed -- included as a by-construction null)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import EXTERNAL

CSV = EXTERNAL / "pammla_silverstein2025" / "220924_6pos_selected_random_only.csv"
PDB = EXTERNAL / "pdb_cas9" / "5F9R.pdb"
POS = ["D1135", "S1136", "G1218", "E1219", "R1335", "T1337"]
POSNUM = [1135, 1136, 1218, 1219, 1335, 1337]
AAS = list("ACDEFGHIKLMNPQRSTVWY")

KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9,
          F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7, L=166.7,
           K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
CHG = dict(K=1, R=1, D=-1, E=-1, H=0.1)
HBD = dict(S=1, T=1, Y=1, C=1, N=1, Q=1, K=1, R=3, H=2, W=1)
HBA = dict(S=1, T=1, Y=1, N=1, Q=1, D=2, E=2, H=2)
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
       "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V"}


def load() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    d = pd.read_csv(CSV, encoding="utf-8-sig")
    pam = list(d.columns[17:])
    return d, d[pam].to_numpy(float), pam


def _pdb():
    prot, nts, ts = {}, {}, {}
    for l in open(PDB):
        if not l.startswith(("ATOM", "HETATM")):
            continue
        ch, el = l[21], l[76:78].strip()
        if el == "H":
            continue
        xyz = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        num, nm, an = int(l[22:26]), l[17:20].strip(), l[12:16].strip()
        if ch == "B":
            prot.setdefault(num, {})[an] = (nm, xyz)
        elif ch == "D":
            nts.setdefault(num, {})[an] = xyz
        elif ch == "C" and nm.startswith("D"):
            ts.setdefault(num, {})[an] = xyz
    return prot, nts, ts


def reach_by_aa() -> dict[str, float]:
    """Longest C-alpha to side-chain heavy-atom distance for each residue type, taken from the structure itself."""
    prot, _, _ = _pdb()
    r = {a: [] for a in AAS}
    for num, atoms in prot.items():
        nm = next(iter(atoms.values()))[0]
        a = AA3.get(nm)
        if a is None or "CA" not in atoms:
            continue
        ca = atoms["CA"][1]
        sc = [np.linalg.norm(x - ca) for an, (_, x) in atoms.items() if an not in ("N", "C", "O", "CA", "OXT")]
        r[a].append(max(sc) if sc else 0.0)
    # 90th percentile over instances: robust to the rare extended conformers, glycine has no side chain
    return {a: (float(np.percentile(v, 90)) if v else 0.0) for a, v in r.items()}


def pam_bp_atoms() -> list[np.ndarray]:
    """Heavy-atom coordinates of base pair j (both strands) for PAM positions 1..4: non-target residues 21..24 and their template partners.

    In 5F9R the non-target strand carries 5'-...GACGC|TGG|CGATTAG-3' so PAM nucleotides 1-3 (N, G, G) are residues
    21-23 and nucleotide 4 (the base after the PAM) is residue 24. The template partner of non-target residue r is
    template residue 31 - r (antiparallel); the C1'-C1' distance is asserted to be that of a Watson-Crick pair.
    Check: R1333 contacts non-target residue 22 (PAM G2) and R1335 residue 23 (PAM G3), as known for NGG readout."""
    _, nts, ts = _pdb()
    out = []
    for j in range(4):
        r = 21 + j
        partner = 31 - r            # antiparallel pairing: 21-10, 22-9, 23-8, 24-7
        d = float(np.linalg.norm(ts[partner]["C1'"] - nts[r]["C1'"]))
        assert 9.5 < d < 11.5, (r, partner, d)   # a Watson-Crick pair has C1'-C1' about 10.4 A
        out.append(np.array(list(nts[r].values()) + list(ts[partner].values())))
    return out


def env_table() -> pd.DataFrame:
    """Per position: distance from the C-alpha to the nearest heavy atom of each PAM base pair (position constants)."""
    prot, nts, ts = _pdb()
    bp = pam_bp_atoms()
    rows = []
    for p in POSNUM:
        ca = prot[p]["CA"][1]
        rows.append({f"d_bp{j + 1}": float(np.min(np.linalg.norm(bp[j] - ca, axis=1))) for j in range(4)})
    return pd.DataFrame(rows, index=POS)


def comp_vec(a: str) -> list[float]:
    return [KD[a], VOL[a], CHG.get(a, 0.0), HBD.get(a, 0), HBA.get(a, 0), float(a in "FYW"), float(a == "G"),
            float(a == "P")]


COMP_NAMES = ["kd", "vol", "chg", "hbd", "hba", "arom", "gly", "pro"]


def features(d: pd.DataFrame, groups: list[str]) -> tuple[np.ndarray, list[str]]:
    reach = reach_by_aa()
    env = env_table()
    cols, names = [], []
    for pi, p in enumerate(POS):
        aa = d[p].to_numpy()
        if "id" in groups:
            for a in AAS:
                cols.append((aa == a).astype(float)); names.append(f"id:{p}:{a}")
        for g in groups:
            if g == "comp" or g.startswith("comp="):
                keep = COMP_NAMES if g == "comp" else g[5:].split("+")      # "comp=kd+vol" keeps a subset
                M = np.array([comp_vec(a) for a in aa])
                for k, nm in enumerate(COMP_NAMES):
                    if nm in keep:
                        cols.append(M[:, k]); names.append(f"comp:{p}:{nm}")
        if "reach" in groups:
            rr = np.array([reach[a] for a in aa])
            cols.append(rr); names.append(f"reach:{p}:len")
            for j in range(4):
                cols.append(np.maximum(env.iloc[pi][f"d_bp{j + 1}"] - rr, 0.0)); names.append(f"reach:{p}:gap{j + 1}")
        if "env" in groups:
            for j in range(4):
                cols.append(np.full(len(d), env.iloc[pi][f"d_bp{j + 1}"])); names.append(f"env:{p}:d{j + 1}")
    return np.column_stack(cols), names


# ---------------------------------------------------------------- metrics
def _sp_axis(A: np.ndarray, B: np.ndarray, axis: int) -> np.ndarray:
    """Spearman correlation along an axis (average ranks for ties); 0 where either side is constant."""
    from scipy.stats import rankdata
    RA, RB = rankdata(A, axis=axis), rankdata(B, axis=axis)
    RA = RA - RA.mean(axis=axis, keepdims=True); RB = RB - RB.mean(axis=axis, keepdims=True)
    den = np.sqrt((RA ** 2).sum(axis=axis) * (RB ** 2).sum(axis=axis))
    num = (RA * RB).sum(axis=axis)
    return np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), 0.0)


def fold_metrics(Y: np.ndarray, P: np.ndarray, fold: np.ndarray, min_sd: float = 0.3) -> dict:
    """Metrics computed inside each held-out fold, then averaged with weight n_fold.

    rho_variant  mean over variants of Spearman across the 64 PAMs (profile shape)
    rho_pam      mean over PAMs (sd > min_sd inside the fold) of Spearman across the fold's variants (does the
                 model order the variants correctly for a given PAM?)
    sse          summed squared error (for the skill score against the training mean)
    Pooling predictions across folds instead would let the fold-to-fold shift of the training mean masquerade as
    signal (or, for leave-one-campaign-out, as anti-signal: the mean model scores about -0.4 pooled)."""
    out = dict(rho_variant=0.0, rho_pam=0.0, sse=0.0)
    n = len(Y)
    for f in np.unique(fold):
        m = fold == f
        w = m.sum() / n
        out["rho_variant"] += w * float(np.mean(_sp_axis(Y[m], P[m], axis=1)))
        sd = Y[m].std(0)
        cols = [j for j in range(Y.shape[1]) if sd[j] > min_sd]
        out["rho_pam"] += w * (float(np.mean(_sp_axis(Y[m][:, cols], P[m][:, cols], axis=0))) if cols else 0.0)
        out["sse"] += float(((Y[m] - P[m]) ** 2).sum())
    out["rmse"] = float(np.sqrt(out["sse"] / Y.size))
    return out
