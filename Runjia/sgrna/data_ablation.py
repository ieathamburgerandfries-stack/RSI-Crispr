"""Hold SLICER fixed and change only the training data.

Finding 12 compares SLICER against crisprHAL 2 and reports parity. It does not
say *where* the parity comes from -- the model, the features, or the 2.4x larger
curated cohort the model is now trained on. A reviewer is entitled to ask, and
the question has a clean form: freeze the architecture, the feature set and the
hyperparameters, change only which rows and which label the model sees, and score
every arm on one held-out set that nothing was fitted on.

**The yardstick** is crisprHAL 2's own shipped WT-SpCas9 test partition -- 6,749
guides at a 378 nt context (protospacer at offset 189; all of them locate in our
curated feature build). Browne et al. report rho **0.695** on it, and our re-run
of their shipped weights gives 0.6940 (`crisprhal_rerun.csv`). No split,
feature or hyperparameter in this project was chosen with reference to it.

**The arms**, all champion XGBoost on 300 columns selected inside the training
rows only, base (6,169 regenerated) + family A (348):

    noshay_rows          the guides Noshay published, curated label
    noshay_rows_label    the same rows, trained against Noshay's own cut_score
    curated_matched_n    random curated rows, n matched to noshay_rows, 3 draws
    curated_full         every curated training row
    curated_full_base    every curated training row, base only -- the feature
                         contrast at fixed data, for scale

Reading them in order separates four things that the single number 0.6997
conflates: how much is the *label*, how much is *which* rows, how much is *how
many* rows, and how much is family A.

    python -m sgrna.data_ablation
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config
from .curated_ablation import _base, _block

OUT = config.RESULTS / "data_ablation.csv"
TEST = config.EXTERNAL / "crisprHAL" / "data" / "WT-SpCas9_testing_data.csv"
PROTO_OFFSET = 189          # verified: 100% of their test guides match at 189
HAL_PUBLISHED = 0.695       # Browne et al.'s reported rho on THIS partition.
                            # Not 0.6971, which is their published *configuration*
                            # scored on our five curated folds -- a different
                            # number for a different split, and easy to confuse.


def _test_protospacers() -> set[str]:
    d = pd.read_csv(TEST, header=None, names=["context", "score"])
    p = d["context"].str.upper().str[PROTO_OFFSET:PROTO_OFFSET + 20]
    return set(p[~p.str.contains("N")])


def _noshay_labels() -> pd.Series:
    """Noshay's own cut_score, keyed by protospacer."""
    d = pd.read_csv(config.GUIDE_INDEX, usecols=["protospacer", "cut_score"])
    d = d.dropna(subset=["protospacer", "cut_score"])
    return d.groupby("protospacer")["cut_score"].mean()


def load(verbose: bool = True):
    X, y, names, protos = _base()
    n_base = X.shape[1]                       # family A is appended after these
    v, fnames = _block("a_flank", protos)
    X = np.hstack([X, v])
    names = list(names) + list(fnames)

    protos = pd.Index(protos)
    in_test = protos.isin(_test_protospacers())
    nos = _noshay_labels()
    is_noshay = protos.isin(nos.index)
    if verbose:
        print(f"  curated cohort {len(protos):,} x {X.shape[1]:,} columns")
        print(f"  their test partition inside it: {in_test.sum():,}")
        print(f"  also published by Noshay: {is_noshay.sum():,} "
              f"({(is_noshay & ~in_test).sum():,} of them trainable)")
    return X, y, names, protos, in_test, is_noshay, nos, n_base


def _fit_score(X, y_train_target, train, test, y_test, seed=41,
               n_features=config.N_FEATURES):
    import xgboost as xgb

    from .run_ablation import _impute, champion_params, make_selector

    X_tr, X_te = _impute(X[train], X[test])
    sel = make_selector(seed)
    sel.fit(X_tr, y_train_target)
    top = np.argsort(sel.feature_importances_)[::-1][:n_features]
    m = xgb.XGBRegressor(**champion_params(seed))
    m.fit(X_tr[:, top], y_train_target)
    return float(spearmanr(y_test, m.predict(X_te[:, top])).statistic)


def run(verbose: bool = True):
    X, y, names, protos, in_test, is_noshay, nos, n_base = load(verbose=verbose)

    test = np.where(in_test)[0]
    y_test = y[test]
    pool = np.where(~in_test)[0]                      # everything trainable
    noshay_tr = np.where(is_noshay & ~in_test)[0]

    rows = []

    def record(arm, what, idx, target=None, seed=41, cols=None):
        Xa = X if cols is None else X[:, cols]
        rho = _fit_score(Xa, y[idx] if target is None else target,
                         idx, test, y_test, seed=seed)
        rows.append(dict(arm=arm, training_data=what, n_train=len(idx),
                         n_cols=Xa.shape[1], label="curated" if target is None
                         else "noshay", spearman=rho,
                         vs_crisprhal=round(rho - HAL_PUBLISHED, 4)))
        if verbose:
            print(f"  {arm:>20}: rho {rho:.4f}  (n_train {len(idx):,})",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    record("noshay_rows", "the 1,222-gene panel Noshay published", noshay_tr)
    record("noshay_rows_label", "same rows, Noshay's own cut_score", noshay_tr,
           target=nos.reindex(protos[noshay_tr]).to_numpy(float))

    rng = np.random.default_rng(41)
    for k in range(3):
        idx = rng.choice(pool, len(noshay_tr), replace=False)
        record(f"curated_matched_n_{k}", "random curated rows, n matched", idx,
               seed=41 + k)

    record("curated_full", "every curated training row", pool)
    record("curated_full_base", "every curated training row, base only", pool,
           cols=np.arange(n_base))

    out = pd.DataFrame(rows)
    if verbose:
        print(f"\n  crisprHAL 2 published on this partition: {HAL_PUBLISHED}")
        print(f"written -> {OUT}")
    return out


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
