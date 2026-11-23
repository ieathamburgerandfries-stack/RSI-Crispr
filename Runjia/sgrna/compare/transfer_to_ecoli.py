"""The E. coli column of the transfer matrix: models trained on each eukaryotic table, scored on the 33,567 curated E. coli guides.

`eukaryote_rules.transfer` fills the 12 x 11 matrix (E. coli as a source only). This adds the column that is missing, so that every dataset is both a source and
a target: the same pipeline (SLICER's 20-mer representation, default settings, seed 41) trained on one eukaryotic table, scored zero-shot on the E. coli
cutting score (higher = more cutting; the eukaryotic labels are oriented by the same rule as in the matrix). The E. coli diagonal is the five-fold CV of the
same model on the 20-mer alone with random folds, as for the other diagonals, which have no gene information to hold out. It is written to its own file so that the
audited matrix `eukaryote_transfer.csv` is unchanged.

    ./run_ext.sh sgrna.compare.transfer_to_ecoli
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..common import OUT
from . import eukaryote_rules as ER


def main() -> int:
    from sklearn.model_selection import KFold

    from ..build.base_fast import matrix_fast
    from .vs_crisprhal import member_specs, train_predict
    D = ER.load_all()
    ec = ER.ecoli()
    big = [k for k, v in D.items() if v["n"] >= ER.MIN_BIG]
    specs = member_specs(["champion"])
    Xe = np.asarray(ec["X"], np.float32)
    rows = []
    for k in big:
        Xk = np.asarray(matrix_fast(list(D[k]["protos"]))[0], np.float32)
        p = train_predict(specs, Xk, D[k]["y"], Xe, seed=41)[0]["champion"]
        rows.append(dict(trained_on=k, scored_on="E. coli", rho=float(spearmanr(ec["y"], p).statistic), within=False, species_trained=D[k]["species"]))
        print(f"  trained on {k:26s} scored on E. coli: rho {rows[-1]['rho']:+.3f}", flush=True)
    pred = np.zeros(len(ec["y"]))
    for tr, te in KFold(5, shuffle=True, random_state=41).split(np.arange(len(pred))):
        pred[te] = train_predict(specs, Xe[tr], ec["y"][tr], Xe[te], seed=41)[0]["champion"]
    rows.append(dict(trained_on="E. coli", scored_on="E. coli", rho=float(spearmanr(ec["y"], pred).statistic), within=True, species_trained="E. coli"))
    print(f"  E. coli five-fold CV (random folds, 20-mer alone): rho {rows[-1]['rho']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "transfer_to_ecoli.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
