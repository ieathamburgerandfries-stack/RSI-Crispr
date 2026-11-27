"""Checks on the report's citation-related numbers (tag CIT): each `got` is computed from a stored file.

Same idiom as `paper_numbers_C.py`: `chk(label, got, claimed, dec)` passes when |got - claimed| <= 0.5 * 10**(-dec). The claimed values
are the ones PROGRESS_REPORT.md carries after the citation corrections.

    published matrix   `data/raw/ecoli_feature_matrix.csv` is the supplementary table of Noshay et al. 2023 as saved here (its title row
                       reads 'Suppemental Table 2: E.coli iRF feature matrix'): 13,880 data rows, 6,232 feature columns, no guide
                       sequences. The paper's own Table 1 lists 40,468 guides (32,374 train); that number is a literature figure and
                       cannot be recomputed from a file here.
    iRF reimplementation   `results/noshay_replication.csv`, five folds on those 13,880 rows (seed 41).
    crisprHAL files    `external_data/crisprHAL/data`: the released WT-SpCas9 and eSpCas9 files and the input-length line of its README.

    ./run_ext.sh sgrna.paper.report_checks_CIT
"""
from __future__ import annotations

import re

import pandas as pd

from ..common import OUT
from sgrna import config

P = "report CIT: "


def run(chk):
    """Run every CIT check through `chk(label, got, claimed, dec)`."""
    RES = OUT
    hal = config.EXTERNAL / "crisprHAL"

    # ------------------------------------------------ the published matrix (Noshay et al. 2023 supplementary tables)
    print("report CIT: the published matrices")
    cols = pd.read_csv(config.BASE_MATRIX, header=config.BASE_MATRIX_HEADER_ROW, nrows=0).columns
    ids = pd.read_csv(config.BASE_MATRIX, header=config.BASE_MATRIX_HEADER_ROW, usecols=[0]).iloc[:, 0]
    chk(P + "published E. coli matrix: data rows", float(len(ids)), 13880, 0)
    chk(P + "published E. coli matrix: distinct sgRNA identifiers", float(ids.nunique()), 13880, 0)
    chk(P + "published E. coli matrix: feature columns (all columns but sgRNAID and cut.score)", float(len(cols) - 2), 6232, 0)
    hum = config.RAW / "human_feature_matrix.csv"
    hcols = pd.read_csv(hum, header=config.BASE_MATRIX_HEADER_ROW, nrows=0).columns
    hids = pd.read_csv(hum, header=config.BASE_MATRIX_HEADER_ROW, usecols=[0], dtype=str).iloc[:, 0]
    chk(P + "published human matrix: data rows", float(len(hids)), 17421, 0)
    chk(P + "published human matrix: feature columns", float(len(hcols) - 2), 6233, 0)

    # ------------------------------------------------ the iRF reimplementation against Noshay et al.'s Table 1
    print("report CIT: iRF reimplementation (results/noshay_replication.csv)")
    nr = pd.read_csv(RES / "noshay_replication.csv")
    chk(P + "iRF reimplementation: rows scored (n_train + n_test of fold 1)", float(nr.n_train.iloc[0] + nr.n_test.iloc[0]), 13880, 0)
    chk(P + "iRF reimplementation: folds", float(len(nr)), 5, 0)
    chk(P + "iRF reimplementation: mean Pearson over five folds", float(nr.pearson.mean()), 0.4881, 4)
    chk(P + "iRF reimplementation: mean R2 over five folds (0.234, not 0.235)", float(nr.r2.mean()), 0.234, 3)
    chk(P + "iRF reimplementation: mean Spearman over five folds", float(nr.spearman.mean()), 0.4785, 4)

    # ------------------------------------------------ crisprHAL: released files and input lengths
    print("report CIT: crisprHAL released files")
    n = lambda f: float(len(pd.read_csv(hal / "data" / f, header=None, usecols=[0])))
    chk(P + "released WT-SpCas9 training file: rows (the paper's Table 2 says 26,795)", n("WT-SpCas9_training_data.csv"), 26994, 0)
    chk(P + "released WT-SpCas9 testing file: rows (the paper's Table 2 says 6,700)", n("WT-SpCas9_testing_data.csv"), 6749, 0)
    chk(P + "released eSpCas9 training file: rows", n("eSpCas9_training_data.csv"), 47591, 0)
    chk(P + "released eSpCas9 testing file: rows", n("eSpCas9_testing_data.csv"), 11898, 0)
    readme = (hal / "README.md").read_text(encoding="utf-8")
    for tag, total in (("SpCas9/TevSpCas9", 37), ("eSpCas9", 406), ("WT-SpCas9", 378)):
        m = re.search(rf"{re.escape(tag)} prediction: ```(\d+) nucleotides = (\d+) upstream, 20 sgRNA, (\d+) downstream", readme)
        chk(P + f"crisprHAL README: {tag} input length", float(m.group(1)), total, 0)
        chk(P + f"crisprHAL README: {tag} upstream + 20 + downstream adds up to the length",
            float(int(m.group(2)) + 20 + int(m.group(3))), total, 0)
    m = re.search(r"SpCas9/TevSpCas9 prediction: ```\d+ nucleotides = (\d+) upstream, 20 sgRNA, (\d+) downstream", readme)
    chk(P + "crisprHAL Tev input: 3 nt upstream", float(m.group(1)), 3, 0)
    chk(P + "crisprHAL Tev input: 14 nt downstream (the NGG PAM plus 11)", float(m.group(2)), 14, 0)


# ---------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from .standalone import run_alone
    run_alone(run, "report")
