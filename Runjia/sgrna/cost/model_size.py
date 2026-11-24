"""What has to be stored to run each model: trees, leaves and bytes for SLICER's default and tuned models, parameters and bytes for crisprHAL 2.

SLICER's deployed model is the boosted trees alone (the 300-tree selector that picks the columns is not needed at prediction time, only the list of selected
columns). Both are refitted here on one training fold of the curated cohort exactly as `vs_crisprhal.train_predict` fits them, and the booster is counted:
trees, total nodes, leaves (the numbers a model has to remember: one value per leaf, a split per inner node) and the size of the saved file (XGBoost's binary UBJ format, and its JSON form). crisprHAL 2 is
counted from its released Keras file (parameters) and the file on disk; each file is also gzipped (level 9), so that LightGBM's text format (BoostMEC, `cost_boostmec.py size`),
XGBoost's binary one and Keras' zip can be compared on the bytes that carry information.

    OMP_NUM_THREADS=2 ./run_ext.sh sgrna.cost.model_size                    # SLICER, main environment
    .venv-crisprhal/bin/python -m sgrna.cost.model_size hal               # crisprHAL 2, TensorFlow environment
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
RUNJIA = HERE.parents[2]
ROOT = HERE.parents[3]
OUTF = RUNJIA / "results" / "post_freeze" / "model_size.csv"


def slicer() -> None:
    sys.path.insert(0, str(RUNJIA))
    import tempfile

    import xgboost as xgb
    from sklearn.model_selection import KFold

    from sgrna.cohorts.curated_ablation import _base, _block
    from sgrna.compare.vs_crisprhal import member_specs
    from sgrna.harness.run_ablation import _impute, make_selector
    from sgrna.tuning import tune_ecoli as T

    X, y, names, protos = _base()
    A, _ = _block("a_flank", protos)
    X = np.hstack([X, A]).astype(np.float32)
    tr, va = next(iter(KFold(5, shuffle=True, random_state=41).split(np.arange(len(y)))))
    Xtr, ytr = X[tr], y[tr]
    Atr, _ = _impute(Xtr, Xtr[:1])
    s = make_selector(41)
    s.fit(Atr, ytr)
    top = np.argsort(s.feature_importances_)[::-1][:900]
    rows = []
    for lab, member in (("SLICER default", "champion"), ("SLICER tuned", "d099x")):
        kind, name, k, cfg = member_specs([member])[0]
        cfg = dict(cfg, n_estimators=k)
        cols = top[:int(cfg["nf"])]
        Xa, _ = _impute(Xtr[:, cols], Xtr[:1, cols])
        m = xgb.XGBRegressor(**T.xgb_params(cfg, 41)).fit(Xa, T._target(ytr, cfg["target"]))
        df = m.get_booster().trees_to_dataframe()
        leaves = int((df.Feature == "Leaf").sum())
        with tempfile.TemporaryDirectory() as d:
            m.save_model(f"{d}/m.ubj")
            ubj = Path(f"{d}/m.ubj").stat().st_size
            ubj_gz = len(gzip.compress(Path(f"{d}/m.ubj").read_bytes(), 9))
            m.save_model(f"{d}/m.json")
            js = Path(f"{d}/m.json").stat().st_size
        rows.append(dict(model=lab, trees=int(df.Tree.nunique()), nodes=int(len(df)), leaves=leaves, inner_nodes=int(len(df) - leaves), columns_used=int(len(set(df.Feature) - {"Leaf"})),
                         columns_selected=int(cfg["nf"]), max_depth=int(cfg["max_depth"]), bytes_file=ubj, bytes_json=js, bytes_gzip=ubj_gz, parameters=np.nan, n_train=len(tr)))
        print(f"  {lab}: {rows[-1]}", flush=True)
    old = pd.read_csv(OUTF) if OUTF.exists() else pd.DataFrame()
    pd.concat([old[~old.model.isin([r['model'] for r in rows])] if len(old) else old, pd.DataFrame(rows)]).to_csv(OUTF, index=False)


def hal() -> None:
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import keras

    hal_dir = ROOT / "external_data" / "crisprHAL"
    sys.path.insert(0, str(hal_dir))
    cwd = os.getcwd()
    os.chdir(hal_dir)
    m = keras.models.load_model("models/WT-SPCAS9.keras", compile=False)
    os.chdir(cwd)
    row = dict(model="crisprHAL 2", parameters=int(m.count_params()), bytes_file=int((hal_dir / "models" / "WT-SPCAS9.keras").stat().st_size),
               bytes_gzip=len(gzip.compress((hal_dir / "models" / "WT-SPCAS9.keras").read_bytes(), 9)))
    print(f"  crisprHAL 2: {row}", flush=True)
    old = pd.read_csv(OUTF) if OUTF.exists() else pd.DataFrame()
    pd.concat([old[old.model != "crisprHAL 2"] if len(old) else old, pd.DataFrame([row])]).to_csv(OUTF, index=False)


if __name__ == "__main__":
    {"hal": hal}.get(sys.argv[1] if len(sys.argv) > 1 else "", slicer)()
