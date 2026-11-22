"""Step 3 -- the ablation harness. One family at a time, leakage-safe.

This reproduces the frozen protocol the project already trusts -- five seeds
(41-45), five folds, top-300 features by XGBoost gain, 400-tree champion --
with one rule that matters more than any of the settings: **every fit happens
inside the training split**. Median imputation, feature ranking and selection
all see the training fold only. A validation fold that has influenced which
features were chosen is not a validation fold.

From a notebook cell (the usual way):

    nb.ablate()                                        # baseline only
    nb.ablate(["a_flank"])
    nb.ablate(["d_supercoiling"], group="bin100k", permute=True)
    nb.ablate(["a_flank", "b_energy"], together=True)

or from a terminal:

    python -m sgrna.harness.run_ablation --families a_flank
    python -m sgrna.harness.run_ablation --families a_flank b_energy --together
    python -m sgrna.harness.run_ablation --families d_supercoiling --group bin100k --permute
    python -m sgrna.harness.run_ablation --baseline-only

Reading the output
------------------
Four things decide whether a family earned its place:

  d_spearman   change against the baseline in the same run
  permuted     the same family with its rows shuffled -- the noise floor
  group        hold out whole genes or 100 kb blocks instead of random rows;
               mandatory for any positional family (D2, E, G)
  selected     how many of the family's columns survived into the top 300

A family whose delta is inside the permuted band did not work, however many
of its columns got selected. Run with --permute before claiming anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from .. import config
from ..build.build_matrix import load_dataset
from ..features import ALL


# --------------------------------------------------------------------------
# Model settings -- identical to the frozen baseline script
# --------------------------------------------------------------------------


def selector_params(seed: int) -> dict:
    return dict(
        n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0, min_child_weight=1,
        random_state=seed, n_jobs=-1, tree_method="hist", importance_type="gain",
    )


def make_selector(seed: int, kind: str | None = None):
    """The throwaway model whose only job is to rank columns.

    Its predictions are never used, so the choice is about speed and ordering.
    `selector_params` is kept as the XGBoost settings of the frozen baseline.
    """
    kind = kind or config.SELECTOR
    if kind == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(**selector_params(seed))
    if kind == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1,
            importance_type="gain",
        )
    raise ValueError(f"unknown selector {kind!r}")


def champion_params(seed: int) -> dict:
    return dict(
        n_estimators=config.N_ESTIMATORS, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0,
        min_child_weight=1, random_state=seed, n_jobs=-1, tree_method="hist",
    )


def _impute(X_train: np.ndarray, X_val: np.ndarray):
    """Median impute with statistics from the training fold only."""
    X_train = X_train.astype(np.float32, copy=True)
    X_val = X_val.astype(np.float32, copy=True)
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    r, c = np.where(np.isnan(X_train))
    X_train[r, c] = med[c]
    r, c = np.where(np.isnan(X_val))
    X_val[r, c] = med[c]
    return X_train, X_val


def _safe_spearman(a, b) -> float:
    if len(a) < 2:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return 0.0 if np.isnan(rho) else float(rho)


def _safe_pearson(a, b) -> float:
    if len(a) < 2:
        return float("nan")
    rho, _ = pearsonr(a, b)
    return 0.0 if np.isnan(rho) else float(rho)


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------


def grouped_folds(groups, n_splits: int, seed: int):
    """KFold over *groups* rather than rows.

    Why this matters here: the library puts ~20 guides in every gene, and
    guides in the same gene share a locus. Plain KFold therefore puts guides
    from the same gene in both the training and the validation split. For
    sequence features that is harmless -- the sequences differ. For anything
    positional (supercoiling, nucleoid bin, distance to oriC or to a TSS) it
    is not: those columns identify the locus, so the model can learn "guides
    around here score about this much" and be rewarded for it at validation
    time. The gain is real memorisation, not real prediction.

    Grouping by gene removes that route. Any positional family should be
    reported under `--group gene`; a family whose gain survives grouping is
    the one worth writing up.
    """
    groups = np.asarray(groups, dtype=object)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    chunks = np.array_split(unique, n_splits)
    for chunk in chunks:
        mask = np.isin(groups, chunk)
        yield np.where(~mask)[0], np.where(mask)[0]


def fold_checkpoint_path(tag: str):
    return config.RESULTS / f"folds_{tag}.csv"


def _load_checkpoint(tag: str | None) -> pd.DataFrame:
    if not tag:
        return pd.DataFrame()
    path = fold_checkpoint_path(tag)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _append_checkpoint(tag: str | None, row: dict) -> None:
    if not tag:
        return
    path = fold_checkpoint_path(tag)
    header = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)


def cross_validate(X, y, feature_names, family_of, n_features=None,
                   seeds=None, n_splits=None, verbose=True, groups=None,
                   tag: str | None = None, experiment: str = "unnamed",
                   deadline: float | None = None,
                   champion_overrides: dict | None = None,
                   return_oof: bool = False) -> dict:
    """Leakage-safe CV, resumable one fold at a time.

    Every fold is written to `results/folds_<tag>.csv` as soon as it finishes,
    and a fold already recorded there is skipped. That matters for two
    reasons: a 5-seed run over the 408-column flank family takes about fifteen
    minutes, longer than a dropped Colab session tends to survive; and it lets
    a long experiment be built up over several sittings without ever
    recomputing work. Pass `deadline` (an absolute time.time() value) to stop
    cleanly partway and resume later -- the partial result is still
    summarised, marked incomplete.

    `champion_overrides` replaces named champion hyperparameters for this call
    only -- used by `representation.py` to ask whether a deeper tree can
    recover what a redundant encoding provides. It is deliberately not a config
    setting: the frozen champion is an anchor, not a default to drift from.
    """
    import xgboost as xgb

    n_features = n_features or config.N_FEATURES
    seeds = seeds or config.SEEDS
    n_splits = n_splits or config.N_SPLITS

    family_of = np.asarray(family_of)
    done = _load_checkpoint(tag)
    if len(done):
        done = done[done["experiment"] == experiment]
    seen = set(zip(done.get("seed", []), done.get("fold", [])))
    fold_rows = done.to_dict("records") if len(done) else []
    if seen and verbose:
        print(f"    resuming: {len(seen)} of {len(seeds) * n_splits} folds already done")

    oof_pred = np.full(len(y), np.nan, dtype=float) if return_oof else None
    oof_y = np.full(len(y), np.nan, dtype=float) if return_oof else None

    stopped_early = False
    for seed in seeds:
        if stopped_early:
            break
        if groups is None:
            splitter = KFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            ).split(X)
        else:
            splitter = grouped_folds(groups, n_splits, seed)
        for fold, (tr, va) in enumerate(splitter, start=1):
            if (seed, fold) in seen:
                continue
            if deadline and time.time() > deadline:
                stopped_early = True
                break

            X_tr, X_va = _impute(X[tr], X[va])
            y_tr, y_va = y[tr], y[va]

            selector = make_selector(seed)
            selector.fit(X_tr, y_tr)
            gains = selector.feature_importances_
            top = np.argsort(gains)[::-1][:n_features]
            n_engineered = int((family_of[top] != "base").sum())

            params = champion_params(seed)
            if champion_overrides:
                params.update(champion_overrides)
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr[:, top], y_tr)
            pred = model.predict(X_va[:, top])
            if return_oof:
                oof_pred[va] = np.asarray(pred).ravel()
                oof_y[va] = y_va

            row = dict(
                experiment=experiment, seed=seed, fold=fold,
                r2=r2_score(y_va, pred),
                spearman=_safe_spearman(y_va, pred),
                pearson=_safe_pearson(y_va, pred),
                mse=mean_squared_error(y_va, pred),
                n_engineered_selected=n_engineered,
            )
            fold_rows.append(row)
            _append_checkpoint(tag, row)
        if verbose and not stopped_early:
            last = [r for r in fold_rows if r["seed"] == seed]
            if last:
                print(
                    f"    seed {seed}: R2 {np.mean([r['r2'] for r in last]):.4f}  "
                    f"rho {np.mean([r['spearman'] for r in last]):.4f}"
                )

    folds = pd.DataFrame(fold_rows)
    complete = len(folds) == len(seeds) * n_splits
    if not complete and verbose:
        print(f"    INCOMPLETE: {len(folds)} / {len(seeds) * n_splits} folds. "
              f"Re-run the same call to continue.")
    per_seed = folds.groupby("seed")[["r2", "spearman", "pearson", "mse"]].mean()
    out = dict(
        folds=folds,
        per_seed=per_seed,
        complete=complete,
        n_folds_done=len(folds),
        summary={
            m: dict(mean=float(per_seed[m].mean()), sd=float(per_seed[m].std()))
            for m in ("r2", "spearman", "pearson", "mse")
        },
        selected={"engineered": float(folds["n_engineered_selected"].mean())
                  if len(folds) else 0.0},
    )
    if return_oof:
        out["oof_pred"] = oof_pred
        out["oof_y"] = oof_y
    return out


def _permute_families(X, family_of, families, seed=0):
    """Shuffle the rows of the engineered columns -- the noise floor."""
    rng = np.random.default_rng(seed)
    X = X.copy()
    fam = np.asarray(family_of)
    idx = np.where(np.isin(fam, families))[0]
    if idx.size:
        X[:, idx] = X[rng.permutation(X.shape[0])][:, idx]
    return X


# --------------------------------------------------------------------------
# Experiment driver
# --------------------------------------------------------------------------


GROUP_CHOICES = {"gene": None, "bin5k": 5_000, "bin100k": 100_000,
                 "bin500k": 500_000, "arc5": "arc"}


def load_groups(ids, by: str = "gene") -> np.ndarray:
    """Group label per guide, for grouped cross-validation.

    `gene` holds out whole genes. `bin100k` / `bin500k` hold out contiguous
    chromosome blocks, which is the stricter test for anything positional:
    neighbouring genes sit in the same supercoiling domain and the same Hi-C
    neighbourhood, so gene-level grouping alone does not separate them.

    `arc5` is the strictest: it cuts the chromosome into five contiguous arcs,
    so with `n_splits=5` each fold holds out one whole arc. Blocked schemes
    still let the model *interpolate* a smooth ori-to-ter trend from the
    blocks either side; holding out a contiguous fifth forces extrapolation
    instead. Use it to check any claim that a positional feature generalises.
    """
    from ..io_utils import load_guide_index

    idx = load_guide_index().set_index(config.ID_COL)
    if by == "gene":
        series = idx.reindex(ids)["b_number"]
    elif by == "arc5":
        from ..build import genome

        size = genome.genome_length() / 5
        # clamp so the final base does not spill into a sixth, one-guide arc
        series = (((idx.reindex(ids)["left"] - 1) // size)
                  .clip(upper=4).astype("Int64"))
    else:
        size = GROUP_CHOICES.get(by)
        if size is None:
            raise ValueError(f"Unknown grouping {by!r}; choose from {list(GROUP_CHOICES)}")
        series = (idx.reindex(ids)["left"] // size).astype("Int64")
    # One guide has a missing QCT descriptor and so no genomic position. Left
    # as its own group it becomes a one-row validation fold, which crashes the
    # correlation metrics; fold it into the largest group instead.
    labels = series.astype(str).replace({"<NA>": None})
    if labels.isna().any():
        labels = labels.fillna(labels.mode().iloc[0])
    return labels.to_numpy()


def run(families, together=False, permute=False, n_features=None,
        seeds=None, n_splits=None, tag=None, force=False,
        group_by=None, time_budget=None, label=None) -> pd.DataFrame:
    """Run the requested experiments, appending to results/ablation_<tag>.csv.

    Results accumulate across invocations: an experiment already present in
    the file is skipped unless --force. That makes a long sweep restartable,
    which matters on Colab (sessions drop) and lets you add a family later
    without re-running everything.
    """
    experiments: list[tuple[str, list[str]]] = [("baseline", [])]
    if families:
        if together:
            experiments.append(("+".join(families), list(families)))
        else:
            experiments += [(f, [f]) for f in families]

    tag = tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = config.RESULTS / f"ablation_{tag}.csv"
    # One deadline for the whole call, shared by every experiment in it, so a
    # sweep of ten families still returns inside the budget instead of
    # spending the budget ten times over.
    deadline = (time.time() + time_budget) if time_budget else None

    rows: list[dict] = []
    if out.exists() and not force:
        rows = pd.read_csv(out).to_dict("records")

    # What counts as done is decided by the FOLD checkpoint, not by the summary
    # file. A row can land in the summary while its experiment is still short of
    # folds (that used to happen to permuted controls), and trusting the summary
    # then blocked it from ever finishing.
    expected = len(seeds or config.SEEDS) * (n_splits or config.N_SPLITS)
    done: set[str] = set()
    if not force:
        checkpoint = _load_checkpoint(tag)
        if len(checkpoint):
            counts = checkpoint.groupby("experiment").size()
            done = set(counts[counts >= expected].index)
    if done:
        print(f"Resuming {out.name}: {len(done)} experiment(s) complete "
              f"({expected} folds each)")

    for exp_label, fams in experiments:
        if deadline and time.time() > deadline:
            print("\n=== time budget reached; re-run the same call to continue")
            break
        need_main = exp_label not in done
        need_perm = permute and bool(fams) and (exp_label + " (permuted)") not in done
        if not need_main and not need_perm:
            print(f"\n=== {exp_label}: already in {out.name}, skipping")
            continue
        print(f"\n=== {exp_label} " + "=" * max(0, 58 - len(exp_label)))
        X, y, names, ids, family_of = load_dataset(fams, verbose=True, label=label)
        print(f"  matrix {X.shape[0]:,} x {X.shape[1]:,}")

        groups = load_groups(ids, group_by) if group_by else None
        if groups is not None:
            print(f"  grouped CV by {group_by}: {len(set(groups)):,} groups")

        if need_main:
            t0 = time.time()
            res = cross_validate(X, y, names, family_of, n_features=n_features,
                                 seeds=seeds, n_splits=n_splits, groups=groups,
                                 tag=tag, experiment=exp_label, deadline=deadline)
            s = res["summary"]
            row = dict(
                experiment=exp_label,
                n_columns=X.shape[1],
                r2=s["r2"]["mean"], r2_sd=s["r2"]["sd"],
                spearman=s["spearman"]["mean"], spearman_sd=s["spearman"]["sd"],
                pearson=s["pearson"]["mean"], mse=s["mse"]["mean"],
                selected_from_family=res["selected"]["engineered"],
                folds_done=res["n_folds_done"],
                complete=res["complete"],
                seconds=round(time.time() - t0),
                permuted=False,
            )
            if res["complete"]:
                rows = [r for r in rows if r.get("experiment") != exp_label]
                rows.append(row)
            else:
                print("    not recorded in the summary until all folds finish")
            print(
                f"  R2 {row['r2']:.4f} +/- {row['r2_sd']:.4f}   "
                f"rho {row['spearman']:.4f} +/- {row['spearman_sd']:.4f}   "
                f"({row['selected_from_family']:.1f} engineered columns in the top "
                f"{n_features or config.N_FEATURES})"
            )

        if need_perm:
            print(f"  permuted control for {exp_label} ...")
            Xp = _permute_families(X, family_of, fams)
            resp = cross_validate(Xp, y, names, family_of, n_features=n_features,
                                  seeds=seeds, n_splits=n_splits, verbose=False,
                                  groups=groups, tag=tag,
                                  experiment=exp_label + " (permuted)",
                                  deadline=deadline)
            sp = resp["summary"]
            rows = [r for r in rows
                    if r.get("experiment") != exp_label + " (permuted)"]
            rows.append(
                dict(
                    experiment=exp_label + " (permuted)",
                    n_columns=X.shape[1],
                    r2=sp["r2"]["mean"], r2_sd=sp["r2"]["sd"],
                    spearman=sp["spearman"]["mean"],
                    spearman_sd=sp["spearman"]["sd"],
                    pearson=sp["pearson"]["mean"], mse=sp["mse"]["mean"],
                    selected_from_family=resp["selected"]["engineered"],
                    folds_done=resp["n_folds_done"],
                    complete=resp["complete"],
                    seconds=0, permuted=True,
                )
            )
            if not resp["complete"]:
                rows = [r for r in rows
                        if r.get("experiment") != exp_label + " (permuted)"]
                print("    permuted control incomplete; re-run to finish it")
            print(
                f"  permuted: R2 {sp['r2']['mean']:.4f}   "
                f"rho {sp['spearman']['mean']:.4f}"
            )

    table = pd.DataFrame(rows)
    if "baseline" in set(table["experiment"]):
        base = table.loc[table["experiment"] == "baseline"].iloc[0]
        table["d_r2"] = table["r2"] - base["r2"]
        table["d_spearman"] = table["spearman"] - base["spearman"]
    else:
        table["d_r2"] = np.nan
        table["d_spearman"] = np.nan

    table.to_csv(out, index=False)
    (config.RESULTS / f"ablation_{tag}_settings.json").write_text(
        json.dumps(
            dict(
                families=list(families), together=together, permute=permute,
                n_features=n_features or config.N_FEATURES,
                seeds=list(seeds or config.SEEDS),
                n_splits=n_splits or config.N_SPLITS,
                group_by=group_by,
                label=label,
                frozen_baseline=config.BASELINE,
            ),
            indent=1,
        )
    )

    print("\n" + "=" * 70)
    cols = ["experiment", "n_columns", "r2", "d_r2", "spearman", "d_spearman",
            "selected_from_family"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print(f"\nWrote {out}")
    print(
        f"Frozen reference: R2 {config.BASELINE['r2']:.4f}, "
        f"rho {config.BASELINE['spearman']:.4f}"
    )
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", nargs="*", default=[], choices=ALL)
    ap.add_argument("--together", action="store_true",
                    help="stack the families instead of testing each alone")
    ap.add_argument("--permute", action="store_true",
                    help="also run a row-shuffled control for each family")
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--n-features", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--tag", default=None,
                    help="results/ablation_<tag>.csv; reruns append to it")
    ap.add_argument("--force", action="store_true",
                    help="recompute experiments already in the results file")
    ap.add_argument("--label", default=None,
                    help="use an alternative target from data/interim/labels/")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="stop cleanly after this many seconds; re-run to resume")
    ap.add_argument("--group", dest="group_by", choices=list(GROUP_CHOICES),
                    default=None,
                    help="hold out whole genes (or 5 kb bins) instead of rows -- "
                         "required before believing any positional family")
    args = ap.parse_args(argv)

    families = [] if args.baseline_only else args.families
    run(
        families,
        together=args.together,
        permute=args.permute,
        n_features=args.n_features,
        seeds=tuple(args.seeds) if args.seeds else None,
        n_splits=args.folds,
        tag=args.tag,
        force=args.force,
        group_by=args.group_by,
        time_budget=args.time_budget,
        label=args.label,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
