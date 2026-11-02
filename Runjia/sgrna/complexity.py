"""What is SLICER's equivalent of a CNN's parameter count?

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

    python -m sgrna.complexity
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config
from .curated_ablation import _base, _block

OUT = config.RESULTS / "complexity.json"
HAL_PARAMS = {"WT-SpCas9": 790_533, "eSpCas9": 823_301, "layers": 37}


def run(seed: int = 41, verbose: bool = True):
    import xgboost as xgb

    from .run_ablation import _impute, champion_params, make_selector

    X, y, names, protos = _base()
    v, fnames = _block("a_flank", protos)
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
    OUT.write_text(json.dumps(out, indent=2))
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
        print(f"written -> {OUT}")
    return out


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
