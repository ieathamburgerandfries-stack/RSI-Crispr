"""Why the diagonal of the transfer table is low: how well the six published predictors do on the same tables.

The diagonal of `eukaryote_transfer.csv` is five-fold cross-validation of SLICER's 20-mer pipeline on each eukaryotic table itself, 0.18 to 0.55. To see whether that is a weak
model or hard data, each table is scored by the six published predictors that the orientation rule of `eukaryote_rules.py` already reads from the CRISPOR files (Fusi, CRISPRscan,
Doench, Wang, SSC, WU-CRISPR; some of them were trained on these very tables), and the 20-mer pipeline's own ρ is set beside their median and their best. The size of each table is
recorded as well.

    ./run_ext.sh sgrna.compare.transfer_diagonal
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import OUT
from . import eukaryote_rules as ER


def main() -> int:
    D = ER.load_all()
    t = pd.read_csv(OUT / "eukaryote_transfer.csv")
    t = t[t.model.str.startswith("pipeline") & t.within].set_index("scored_on").rho
    rows = []
    for k, v in D.items():
        if v["n"] < ER.MIN_BIG:
            continue
        r = np.abs(np.asarray(v["votes"], float))
        rows.append(dict(dataset=k, species=v["species"], n=v["n"], own_cv_rho=float(t[k]), published_median=float(np.nanmedian(r)), published_best=float(np.nanmax(r)),
                         published_tools=int(np.isfinite(r).sum())))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "transfer_diagonal.csv", index=False)
    print(d.round(3).to_string(index=False))
    print(f"20-mer CV above the median published predictor on {int((d.own_cv_rho > d.published_median).sum())} of {len(d)} tables, above the best on {int((d.own_cv_rho > d.published_best).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
