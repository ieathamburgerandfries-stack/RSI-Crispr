"""Give crisprHAL 2 the same courtesy we gave ourselves: a hyperparameter search.

Why this exists. The head-to-head margin is +0.0107 in SLICER's favour, and
tuning SLICER alone is worth +0.0103 -- the whole margin. So the comparison as it
stands is tuned-against-untuned in our favour, and the report refuses to call it
a win for that reason. The fix is to search their settings too and then say
whichever thing is true.

Design, and where it is deliberately unequal. Their model costs ~17 min of CPU
per fit here, so a nested search matching SLICER's (12 draws inside each of 5
outer folds = 60 fits) would be about 17 hours. Instead:

  stage 1   12 random draws, evaluated on ONE inner split carved out of outer
            fold 1's training data (80/20). Nothing here ever sees an outer
            validation fold.
  stage 2   the winning configuration re-fitted on all 5 outer folds, which is
            the number comparable with `crisprhal_rerun.py --arm our_folds`.

That is a weaker search than SLICER received, in two ways worth stating in the
paper: fewer effective evaluations, and a single inner split rather than one per
fold. **If their tuned score closes the gap, parity is the honest conclusion and
the weakness of the search only strengthens it.** If it does not close the gap,
the asymmetry is a live caveat and must be quoted alongside the result.

The search space is their own constructor's arguments -- learning rate, CNN
width and window, recurrent width, the three dropout rates, dense widths -- plus
epochs and batch size from `train()`. Their published defaults are included as
draw 0, so the search cannot do worse than their paper configuration.

Needs their TensorFlow venv, not the project interpreter:

    .venv-crisprhal/bin/python src/sgrna/crisprhal_tune.py --stage search
    .venv-crisprhal/bin/python src/sgrna/crisprhal_tune.py --stage final

Resumable: one row per draw in `results/crisprhal_tuning.csv`, one row per fold
in `results/crisprhal_tuned_folds.csv`. Re-running skips finished work, so a
`--deadline-hours` stop is safe.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sgrna.compare.crisprhal_rerun import (MODEL, META_PATH, _load_hal,  # noqa: E402
                                   _read)

SEARCH_PATH = ROOT / "results" / "crisprhal_tuning.csv"
FINAL_PATH = ROOT / "results" / "crisprhal_tuned_folds.csv"

# Their published configuration, from models.__init__ defaults plus the
# model-specific epoch count. Draw 0 is exactly this, so the search is anchored.
DEFAULTS = dict(learning_rate=0.0005, drop_rate=0.3, CNN_filters=128,
                window_size=3, CNN_drop=0.3, CNN_dense1=128, CNN_dense2=64,
                RNN_size=128, RNN_dense1=128, RNN_dense2=64, CNN_RNN_drop=0.3)
DEFAULT_BATCH = 1024

SPACE = dict(
    learning_rate=[1e-4, 2.5e-4, 5e-4, 1e-3],
    drop_rate=[0.2, 0.3, 0.4],
    CNN_filters=[64, 128, 192],
    window_size=[3, 5],
    CNN_drop=[0.2, 0.3, 0.4],
    CNN_dense1=[64, 128, 256],
    CNN_dense2=[32, 64, 128],
    RNN_size=[64, 128, 192],
    RNN_dense1=[64, 128, 256],
    RNN_dense2=[32, 64, 128],
    CNN_RNN_drop=[0.2, 0.3, 0.4],
)
EPOCH_CHOICES = [32, 48, 64]
BATCH_CHOICES = [512, 1024]


def draws(n: int, seed: int = 41) -> list[dict]:
    """Draw 0 is their published configuration; the rest are random."""
    rng = np.random.default_rng(seed)
    out = [dict(DEFAULTS, epochs=None, batch_size=DEFAULT_BATCH)]
    for _ in range(n - 1):
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        cfg["epochs"] = int(EPOCH_CHOICES[int(rng.integers(len(EPOCH_CHOICES)))])
        cfg["batch_size"] = int(BATCH_CHOICES[int(rng.integers(len(BATCH_CHOICES)))])
        out.append(cfg)
    return out


def _curated():
    """The 33,567 curated guides in `expanded_meta.json` order, with their context."""
    meta = json.loads(META_PATH.read_text())
    protos = np.asarray(meta["protospacers"])
    y = np.asarray(meta["y"], dtype=float)
    hal = pd.concat([_read("train"), _read("test")]).drop_duplicates("protospacer")
    seqs = hal.set_index("protospacer")["seq"].reindex(protos)
    if seqs.isna().any():
        raise SystemExit(f"{seqs.isna().sum()} curated guides lack crisprHAL context")
    return seqs.to_numpy(), y


def _fit_eval(models, proc, cfg, s_tr, y_tr, s_va, y_va, seed, default_epochs,
              verbose=True, save_preds=None):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    kw = {k: v for k, v in cfg.items() if k not in ("epochs", "batch_size")}
    epochs = cfg["epochs"] or default_epochs
    m = models(MODEL, summary=False, **kw)
    t0 = time.time()
    m.train(proc.onehotencode(s_tr), y_tr, epochs=epochs,
            batch_size=cfg["batch_size"], verbose=0)
    pred = np.asarray(m.predict(proc.onehotencode(s_va), verbose=0)).ravel()
    if save_preds:
        pd.DataFrame({"y": y_va, "pred": pred}).to_csv(save_preds, index=False)
    return dict(spearman=float(spearmanr(y_va, pred).statistic),
                r2=float(r2_score(y_va, pred)),
                epochs=epochs,
                fit_seconds=round(time.time() - t0, 1))


def search(n_draws: int = 12, seed: int = 41, n_splits: int = 5,
           patience: int | None = None, deadline_hours: float | None = None,
           verbose: bool = True):
    """Stage 1: score each draw on one inner split of outer fold 1's training set.

    `n_draws` is a ceiling. With `patience` the search stops once the best inner
    score has not improved for that many consecutive draws, so "searched until
    it stopped improving" is a statement about the recorded curve rather than
    about running out of patience -- every draw is already one row of
    `crisprhal_tuning.csv`, which is the trace.
    """
    models, default_epochs, proc = _load_hal()
    seqs, y = _curated()

    outer = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, _ = next(iter(outer.split(seqs)))
    inner = KFold(n_splits=5, shuffle=True, random_state=seed + 1)
    i_tr, i_va = next(iter(inner.split(tr_idx)))
    s_tr, y_tr = seqs[tr_idx[i_tr]], y[tr_idx[i_tr]]
    s_va, y_va = seqs[tr_idx[i_va]], y[tr_idx[i_va]]
    if verbose:
        print(f"  inner split: {len(y_tr):,} train / {len(y_va):,} validation "
              f"(both inside outer fold 1's training rows)", flush=True)

    prev = pd.read_csv(SEARCH_PATH) if SEARCH_PATH.exists() else pd.DataFrame()
    done = set(prev["draw"]) if len(prev) else set()
    stop = time.time() + deadline_hours * 3600 if deadline_hours else None

    best = float(prev["spearman"].max()) if len(prev) else -np.inf
    since_improved = 0
    for i, cfg in enumerate(draws(n_draws, seed)):
        if i in done:
            if verbose:
                print(f"  draw {i}: already done", flush=True)
            continue
        if stop and time.time() > stop:
            print("  deadline reached; re-run to continue", flush=True)
            break
        if patience and since_improved >= patience:
            print(f"  no improvement in {patience} draws; search stopped at "
                  f"draw {i}", flush=True)
            break
        res = _fit_eval(models, proc, cfg, s_tr, y_tr, s_va, y_va,
                        seed=seed, default_epochs=default_epochs,
                        verbose=verbose)
        # `cfg` carries the requested epochs (possibly None) and `res` the
        # resolved one, so they must not be splatted into one dict literal.
        row = {"draw": i, "is_published_default": (i == 0)}
        row.update(cfg)
        row.update(res)
        prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        prev.to_csv(SEARCH_PATH, index=False)
        if res["spearman"] > best:
            best, since_improved = res["spearman"], 0
        else:
            since_improved += 1
        if verbose:
            tag = " (their published default)" if i == 0 else ""
            print(f"  draw {i:2d}{tag}: inner rho {res['spearman']:.4f}  "
                  f"best {best:.4f}  (+{since_improved} since improved, "
                  f"{res['fit_seconds'] / 60:.1f} min)", flush=True)
    return prev


def best_config() -> dict:
    d = pd.read_csv(SEARCH_PATH)
    row = d.loc[d["spearman"].idxmax()]
    keys = list(SPACE) + ["epochs", "batch_size"]
    cfg = {k: row[k] for k in keys}
    for k in ("CNN_filters", "window_size", "CNN_dense1", "CNN_dense2",
              "RNN_size", "RNN_dense1", "RNN_dense2", "epochs", "batch_size"):
        cfg[k] = int(cfg[k])
    return cfg, int(row["draw"]), float(row["spearman"])


def final(seed: int = 41, n_splits: int = 5, deadline_hours: float | None = None,
          verbose: bool = True):
    """Stage 2: the winning configuration on all 5 outer folds."""
    cfg, draw, inner_rho = best_config()
    if verbose:
        print(f"  winning draw {draw} (inner rho {inner_rho:.4f}): {cfg}",
              flush=True)
    models, default_epochs, proc = _load_hal()
    seqs, y = _curated()

    prev = pd.read_csv(FINAL_PATH) if FINAL_PATH.exists() else pd.DataFrame()
    done = set(prev["fold"]) if len(prev) else set()
    stop = time.time() + deadline_hours * 3600 if deadline_hours else None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(kf.split(seqs), start=1):
        if fold in done:
            continue
        if stop and time.time() > stop:
            print("  deadline reached; re-run to continue", flush=True)
            break
        res = _fit_eval(models, proc, cfg, seqs[tr], y[tr], seqs[va], y[va],
                        seed=seed + fold, default_epochs=default_epochs,
                        verbose=verbose,
                        save_preds=FINAL_PATH.parent /
                        f"preds_crisprhal_tuned_fold{fold}.csv")
        row = {"model": "crisprHAL2_WT-SpCas9_tuned", "draw": draw,
               "fold": fold, "seed": seed, "n_train": len(tr),
               "n_val": len(va)}
        row.update(cfg)
        row.update(res)
        prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        prev.to_csv(FINAL_PATH, index=False)
        if verbose:
            print(f"  fold {fold}: rho {res['spearman']:.4f} "
                  f"({res['fit_seconds'] / 60:.1f} min)", flush=True)
    return prev


def summary():
    if SEARCH_PATH.exists():
        d = pd.read_csv(SEARCH_PATH).sort_values("spearman", ascending=False)
        print("\n== search, best first ==")
        print(d[["draw", "is_published_default", "learning_rate", "CNN_filters",
                 "RNN_size", "epochs", "batch_size", "spearman",
                 "fit_seconds"]].round(4).to_string(index=False))
    if FINAL_PATH.exists():
        f = pd.read_csv(FINAL_PATH)
        print("\n== tuned, 5 outer folds ==")
        print(f[["fold", "spearman", "r2", "fit_seconds"]].round(4)
              .to_string(index=False))
        print(f"\n  mean rho {f['spearman'].mean():.4f} "
              f"± {f['spearman'].std():.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["search", "final", "both", "summary"],
                    default="both")
    ap.add_argument("--draws", type=int, default=12,
                    help="ceiling on draws")
    ap.add_argument("--patience", type=int, default=None,
                    help="stop after this many draws with no improvement")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--deadline-hours", type=float, default=None)
    a = ap.parse_args(argv)
    if a.stage in ("search", "both"):
        search(n_draws=a.draws, seed=a.seed, patience=a.patience,
               deadline_hours=a.deadline_hours)
    if a.stage in ("final", "both"):
        final(seed=a.seed, deadline_hours=a.deadline_hours)
    summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
