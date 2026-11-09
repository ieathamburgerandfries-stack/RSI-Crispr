"""Recorded equivalence of the array-based featurisers with the originals, and of the predictions built on them.

Whole curated cohort (33,567 guides): `base_fast.matrix_fast` against the stored base matrix, `a_flank_fast.build_matrix` against the
stored family A block; then a model trained on the stored matrices predicts 3,000 held-out guides from the stored features and from
the fast ones; the largest absolute difference is recorded.

    python -m sgrna.ext.equiv_check
"""
import json
import time

import numpy as np
import pandas as pd

from sgrna import config, featurise, qct
from sgrna.curated_ablation import _base, _block
from sgrna.run_ablation import _impute, make_selector

from . import a_flank_fast as AF
from . import base_fast as BF
from . import tune_ecoli as T
from .common import OUT

X, y, names, protos = _base(); A, An = _block("a_flank", protos)
tables, vm = qct.load_tables(), featurise.load_v_mapping()
idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv").set_index("sgRNAID").reindex(protos).reset_index()
idx = idx.rename(columns={"index": "sgRNAID"}); idx["located"] = True
t = time.perf_counter(); Mb, bn = BF.matrix_fast(list(protos), vm, tables); tb = time.perf_counter() - t
t = time.perf_counter(); Ma, an = AF.build_matrix(idx, tables); ta = time.perf_counter() - t
rows = [dict(item="base columns: names equal", value=float(list(names) == list(bn))),
        dict(item="base columns: NaN pattern equal", value=float((np.isnan(X) == np.isnan(Mb)).all())),
        dict(item="base columns: max abs difference", value=float(np.nanmax(np.abs(X - Mb)))),
        dict(item="family A columns: names equal", value=float(list(An) == list(an))),
        dict(item="family A columns: NaN pattern equal", value=float((np.isnan(A) == np.isnan(Ma)).all())),
        dict(item="family A columns: max abs difference after float32 cast", value=float(np.nanmax(np.abs(A - Ma.astype(np.float32))))),
        dict(item="guides compared", value=float(len(protos))),
        dict(item="base ms per guide (whole cohort, one call)", value=tb / len(protos) * 1000),
        dict(item="family A ms per guide (whole cohort, one call)", value=ta / len(protos) * 1000)]
Xs, Xf = np.hstack([X, A]), np.hstack([Mb, Ma.astype(np.float32)])
tr, te = np.arange(0, 24000), np.arange(24000, 27000)
Xa, Xb = _impute(Xs[tr], Xs[te]); _, Xc = _impute(Xs[tr], Xf[te])
s = make_selector(41); s.fit(Xa, y[tr]); top = np.argsort(s.feature_importances_)[::-1]
cfgs = json.loads((OUT / "tune_configs.json").read_text()); sel = T.selected()
for name in ("champion", "d099x"):
    cfg = dict(cfgs[name], n_estimators=int(sel.loc[name, "k"])); cols = top[:int(cfg["nf"])]; k = cfg["n_estimators"]
    p1 = T.fit_predict(cfg, Xa[:, cols], y[tr], Xb[:, cols], 41)[k]; p2 = T.fit_predict(cfg, Xa[:, cols], y[tr], Xc[:, cols], 41)[k]
    rows.append(dict(item=f"{name}: prediction max abs difference, stored vs fast features", value=float(np.abs(p1 - p2).max())))
pd.DataFrame(rows).to_csv(OUT / "featuriser_equivalence.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
