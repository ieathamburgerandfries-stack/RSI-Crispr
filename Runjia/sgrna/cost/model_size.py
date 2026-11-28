"""SLICER's default model stores 6,338 leaf values against 790,533 parameters for crisprHAL 2, but the tuned SLICER model is larger on disk than crisprHAL 2's file.

Three parts (a positional argument; no argument runs `slicer`): `slicer` and `hal` write `model_size.csv`, `complexity` writes `complexity.json` and counts the same objects on a model fitted on the whole cohort.

SLICER's deployed model is the boosted trees alone (the 300-tree selector that picks the columns is not needed at prediction time, only the list of selected
columns). Both are refitted here on one training fold of the curated cohort exactly as `vs_crisprhal.train_predict` fits them, and the booster is counted:
trees, total nodes, leaves (the numbers a model has to remember: one value per leaf, a split per inner node) and the size of the saved file (XGBoost's binary UBJ format, and its JSON form). crisprHAL 2 is
counted from its released Keras file (parameters) and the file on disk; each file is also gzipped (level 9), so that LightGBM's text format (BoostMEC, `boostmec.py size`),
XGBoost's binary one and Keras' zip can be compared on the bytes that carry information.

    OMP_NUM_THREADS=2 ./run_ext.sh sgrna.cost.model_size                    # SLICER, main environment
    .venv-crisprhal/bin/python -m sgrna.cost.model_size hal               # crisprHAL 2, TensorFlow environment
    ./run_ext.sh sgrna.cost.model_size complexity                          # the default model fitted on the whole cohort: how much of it is used

------------------------------------------------------------------------------------------------------------------------------
complexity
------------------------------------------------------------------------------------------------------------------------------
A convolutional network is quoted by its trainable parameter count because that
is what is fitted and what has to be stored: crisprHAL 2's WT-SpCas9 model holds
**790,533** of them across 37 layers (823,301 for eSpCas9). A gradient-boosted
tree ensemble has no weight matrices, so the honest analogue is what the fit
actually stores:

  * one **leaf value** per leaf -- these are the fitted real numbers, the closest
    thing to a weight;
  * one (feature index, threshold) pair per **internal node** -- the structure,
    chosen rather than optimised by gradient, but still fitted from the data.

Both are reported below, so the comparison can be made on either definition.

Three numbers matter more than the total for interpretability, and have no CNN
equivalent at all:

  * **`max_depth`** caps the interaction order. At 4, no term in the model
    involves more than four features -- a hard structural bound, not an
    empirical observation.
  * **how many of the 6,517 columns the fit ever looks at**, and how
    concentrated the gain is across them.
  * **how much of the model a straight line reproduces** -- 95.6% of rho on a
    decorrelated basis (Appendix B), which says the fitted function is mostly a
    sum of one-dimensional effects and can be read as one.
"""
from __future__ import annotations

import gzip
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
RUNJIA = HERE.parents[2]
ROOT = HERE.parents[3]
OUTF = Path(os.environ.get("EXT_OUT") or RUNJIA / "results") / "model_size.csv"
COMPLEXITY_OUT = OUTF.parent / "complexity.json"
HAL_PARAMS = {"WT-SpCas9": 790_533, "eSpCas9": 823_301, "layers": 37}


def slicer() -> None:
    sys.path.insert(0, str(RUNJIA))
    import tempfile

    import xgboost as xgb
    from sklearn.model_selection import KFold

    from sgrna.harness.curated import load_base, load_block
    from sgrna.compare.vs_crisprhal import member_specs
    from sgrna.harness.run_ablation import _impute, make_selector
    from sgrna.tuning import tune_ecoli as T

    X, y, names, protos = load_base()
    A, _ = load_block("a_flank", protos)
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


def complexity(seed: int = 41, verbose: bool = True):
    import json

    import xgboost as xgb

    sys.path.insert(0, str(RUNJIA))
    from sgrna import config
    from sgrna.harness.curated import load_base, load_block
    from sgrna.harness.run_ablation import _impute, champion_params, make_selector

    X, y, names, protos = load_base()
    v, fnames = load_block("a_flank", protos)
    X = np.hstack([X, v])
    names = list(names) + list(fnames)
    X, _ = _impute(X, X[:1])

    sel = make_selector(seed)
    sel.fit(X, y)
    top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
    m = xgb.XGBRegressor(**champion_params(seed))
    m.fit(X[:, top], y)

    trees = m.get_booster().trees_to_dataframe()
    leaves = trees[trees["Feature"] == "Leaf"]
    splits = trees[trees["Feature"] != "Leaf"]
    used = splits["Feature"].nunique()
    depth = trees.groupby("Tree")["ID"].count()

    gain = (pd.Series(m.feature_importances_, index=[names[i] for i in top])
            .sort_values(ascending=False))
    share = gain / gain.sum()
    top10 = float(share.head(10).sum())
    n_half = int((share.cumsum() <= 0.5).sum() + 1)

    out = dict(
        n_trees=int(trees["Tree"].nunique()),
        max_depth=int(champion_params(seed)["max_depth"]),
        n_leaves=int(len(leaves)),
        n_internal_nodes=int(len(splits)),
        parameters_leaf_values_only=int(len(leaves)),
        parameters_incl_structure=int(len(leaves) + 2 * len(splits)),
        columns_available=int(X.shape[1]),
        columns_offered_to_the_fit=int(config.N_FEATURES),
        columns_ever_split_on=int(used),
        columns_carrying_half_the_gain=n_half,
        top10_gain_share=round(top10, 4),
        mean_nodes_per_tree=round(float(depth.mean()), 1),
        crisprhal2=HAL_PARAMS,
        ridge_share_of_boosting_rho=0.956,
        top_columns=list(gain.head(15).round(4).items()),
    )
    COMPLEXITY_OUT.write_text(json.dumps(out, indent=2))
    if verbose:
        r = out["parameters_incl_structure"]
        print(f"  {out['n_trees']} trees, depth <= {out['max_depth']}, "
              f"{out['n_leaves']:,} leaves, {out['n_internal_nodes']:,} splits")
        print(f"  fitted numbers: {out['n_leaves']:,} leaf values; "
              f"{r:,} including structure")
        print(f"  crisprHAL 2 (WT-SpCas9): {HAL_PARAMS['WT-SpCas9']:,} "
              f"trainable parameters, {HAL_PARAMS['layers']} layers")
        print(f"  ratio: {HAL_PARAMS['WT-SpCas9'] / r:.0f}x on the inclusive "
              f"count, {HAL_PARAMS['WT-SpCas9'] / out['n_leaves']:.0f}x on leaves")
        print(f"  columns: {out['columns_ever_split_on']} of "
              f"{out['columns_offered_to_the_fit']} offered ever split on; "
              f"{n_half} carry half the gain; top 10 hold {top10:.1%}")
        print(f"written -> {COMPLEXITY_OUT}")
    return out


if __name__ == "__main__":
    {"hal": hal, "complexity": complexity}.get(sys.argv[1] if len(sys.argv) > 1 else "", slicer)()
