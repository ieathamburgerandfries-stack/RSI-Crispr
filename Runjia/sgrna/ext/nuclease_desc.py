"""Descriptors of the nine nucleases in Kim et al. 2020, in separate column groups.

The descriptors are built from three different kinds of source, kept apart because they answer
different questions and the ablation in `nuclease_model` removes them group by group:

  mutations   the substitutions that define each variant relative to SpCas9 (Kim 2020; Kleinstiver
              2016 eSpCas9/VQR/VRER, Kleinstiver 2015 VRQR; Chen 2017 HypaCas9; Kleinstiver 2016
              HF1; Casini 2018 evoCas9; Lee 2018 Sniper; Hu 2018 xCas9; Nishimasu 2018 NG),
              summarised as counts by domain and by chemistry
  structure   where those residues sit in the R-loop complex PDB 5F9R (Jiang 2016): distance of
              each substituted residue's atoms to the target strand, the non-target strand and
              the sgRNA, and the charge removed or added near the DNA
  pam         the PAM profile each enzyme was measured to have, from the PAM-determination
              library of the same paper (Supplementary Table 7: mean indel frequency for every
              four-nucleotide PAM), as the share of the enzyme's activity on each NGN/NAN class
  scale       how active the enzyme is overall on the training targets (mean, 90th percentile):
              a target-encoding of the nuclease, included to show what identity alone carries

Rows are the enzymes, in the order of the activity columns of the Supplementary Table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import openpyxl

from .common import EXTERNAL, INTERIM

PDB = EXTERNAL / "pdb_cas9" / "5F9R.pdb"
KIM = EXTERNAL / "kim2020_cas9_variants"

NUCLEASES = ["SpCas9", "SpCas9-NG", "VRQR", "xCas9", "Sniper-Cas9", "eSpCas9(1.1)", "SpCas9-HF1",
             "HypaCas9", "evoCas9"]
SHORT = ["SpCas9", "NG", "VRQR", "xCas9", "Sniper", "eSp", "HF1", "Hypa", "evo"]
# MOESM7 sheet name for each enzyme
PAM_SHEET = {"SpCas9": "SpCas9 (Day 4)", "SpCas9-NG": "SpCas9-NG (Day 4)", "VRQR": "VRQR (Day 4)",
             "xCas9": "xCas9 (Day 4)", "Sniper-Cas9": "Sniper-Cas9 (Day 4)",
             "eSpCas9(1.1)": "eSpCas9(1.1) (Day 4)", "SpCas9-HF1": "SpCas9-HF1 (Day 4)",
             "HypaCas9": "HypaCas9 (Day 4)", "evoCas9": "evoCas9 (Day4)"}
MUTATIONS = {
    "SpCas9": [],
    "SpCas9-NG": ["R1335V", "L1111R", "D1135V", "G1218R", "E1219F", "A1322R", "T1337R"],
    "VRQR": ["D1135V", "G1218R", "R1335Q", "T1337R"],
    "xCas9": ["A262T", "R324L", "S409I", "E480K", "E543D", "M694I", "E1219V"],
    "Sniper-Cas9": ["F539S", "M763I", "K890N"],
    "eSpCas9(1.1)": ["K848A", "K1003A", "R1060A"],
    "SpCas9-HF1": ["N497A", "R661A", "Q695A", "Q926A"],
    "HypaCas9": ["N692A", "M694A", "Q695A", "H698A"],
    "evoCas9": ["M495V", "Y515N", "K526E", "R661Q"],
}
CHARGE = {"K": 1, "R": 1, "D": -1, "E": -1}
DOMAINS = {"REC": [(94, 717)], "HNH": [(776, 908)], "RuvC": [(1, 93), (718, 775), (909, 1098)],
           "PI": [(1099, 1368)]}
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def _domain(pos: int) -> str:
    for d, rng in DOMAINS.items():
        if any(a <= pos <= b for a, b in rng):
            return d
    return "other"


def _atoms():
    prot, strands = {}, {"A": [], "C": [], "D": []}
    for l in open(PDB):
        if not l.startswith(("ATOM", "HETATM")):
            continue
        ch = l[21]
        xyz = (float(l[30:38]), float(l[38:46]), float(l[46:54]))
        if l[76:78].strip() == "H":
            continue
        if ch == "B":
            prot.setdefault(int(l[22:26]), []).append((AA3.get(l[17:20].strip(), "?"), xyz))
        elif ch in strands:
            strands[ch].append(xyz)
    return prot, {k: np.array(v) for k, v in strands.items()}


def structure_table() -> pd.DataFrame:
    """Per substituted residue: identity, domain, distances (Å) to TS (C), NTS (D), sgRNA (A)."""
    prot, st = _atoms()
    rows = []
    for var, muts in MUTATIONS.items():
        for m in muts:
            wt, pos, mt = m[0], int(m[1:-1]), m[-1]
            xyz = np.array([a[1] for a in prot[pos]])
            assert prot[pos][0][0] == wt, (m, prot[pos][0][0])
            d = {k: float(np.min(np.linalg.norm(xyz[:, None] - v[None], axis=2))) for k, v in st.items()}
            rows.append(dict(variant=var, mut=m, pos=pos, wt=wt, to=mt, domain=_domain(pos),
                             dq=CHARGE.get(mt, 0) - CHARGE.get(wt, 0), d_ts=d["C"], d_nts=d["D"],
                             d_rna=d["A"], d_dna=min(d["C"], d["D"])))
    return pd.DataFrame(rows)


def mutation_group() -> pd.DataFrame:
    st = structure_table()
    out = []
    for v in NUCLEASES:
        s = st[st.variant == v]
        out.append({"n_mut": len(s), "n_to_ala": int((s.to == "A").sum()),
                    **{f"n_{d}": int((s.domain == d).sum()) for d in ("REC", "HNH", "RuvC", "PI")},
                    "dq_total": float(s.dq.sum()), "n_charge_removed": int((s.dq < 0).sum() + (s.dq > 0).sum() * 0)})
    return pd.DataFrame(out, index=NUCLEASES)


def structure_group() -> pd.DataFrame:
    st = structure_table()
    out = []
    for v in NUCLEASES:
        s = st[st.variant == v]
        if s.empty:
            out.append(dict(min_d_nts=40.0, min_d_ts=40.0, min_d_rna=40.0, mean_d_dna=40.0, n_contact=0,
                            prox=0.0, dq_near_dna=0.0, dq_near_nts=0.0))
            continue
        out.append(dict(min_d_nts=s.d_nts.min(), min_d_ts=s.d_ts.min(), min_d_rna=s.d_rna.min(),
                        mean_d_dna=s.d_dna.mean(), n_contact=int((s.d_dna < 4.5).sum()),
                        prox=float(np.exp(-s.d_dna / 4.0).sum()),
                        dq_near_dna=float(s.dq[s.d_dna < 6].sum()), dq_near_nts=float(s.dq[s.d_nts < 6].sum())))
    return pd.DataFrame(out, index=NUCLEASES)


def pam_group() -> pd.DataFrame:
    wb = openpyxl.load_workbook(KIM / "MOESM7.xlsx", read_only=True)
    out = {}
    for v in NUCLEASES:
        ws = wb[PAM_SHEET[v]]
        d = {}
        for r in ws.iter_rows(min_row=2, values_only=True):
            if r[0] and isinstance(r[2], (int, float)):
                d[r[0]] = max(float(r[2]), 0.0)
        s = pd.Series(d)
        tot = s.sum()
        n23 = s.groupby(s.index.str[1:3]).sum() / tot        # share on each NxxN class by positions 2-3
        out[v] = {f"pam_{k}": n23.get(k, 0.0) for k in ("GG", "GA", "GT", "GC", "AG", "AA", "AT", "AC",
                                                         "TG", "CG")}
        out[v]["pam_total"] = tot / len(s)
    return pd.DataFrame(out).T.loc[NUCLEASES]


def scale_group() -> pd.DataFrame:
    d = pd.read_csv(KIM / "kim2020_training_test.csv").iloc[:, :12]
    d.columns = ["seq", "trna", "split"] + SHORT
    tr = d[d.split == "Training"]
    return pd.DataFrame({"scale_mean": [tr[s].mean() for s in SHORT],
                         "scale_q90": [tr[s].quantile(0.9) for s in SHORT]}, index=NUCLEASES)


def all_groups() -> dict[str, pd.DataFrame]:
    return {"mutations": mutation_group(), "structure": structure_group(), "pam": pam_group(),
            "scale": scale_group()}


if __name__ == "__main__":
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(structure_table().round(1).to_string(index=False))
    for k, v in all_groups().items():
        print("\n==", k); print(v.round(2))
