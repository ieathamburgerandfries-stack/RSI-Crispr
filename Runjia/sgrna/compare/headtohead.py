"""The like-for-like comparison against crisprHAL 2.

Every performance number this project has quoted so far sits on a different
row set from the model it is being compared to: 13,880 guides here against
33,495 curated guides there. That makes the 0.641-vs-0.697 gap
uninterpretable -- it could be the features, the model, or simply that their
rows are cleaner.

This module removes the excuse. `featurise.py` reproduces 6,169 of the
published matrix's 6,232 columns from a protospacer alone, so the *same*
representation can be computed for crisprHAL's curated guides. Train on those
rows, with their label, under 5-fold cross-validation, and the comparison is
against a published number on matching data.

Three arms, so the result decomposes rather than just landing somewhere:

    published_rows    the 13,880 rows the paper shipped, curated label
    curated_all       all ~33.5k curated rows, base features only
    curated_flank     all ~33.5k curated rows, base features + family A

The first two isolate what the extra rows buy; the second and third isolate
what family A buys once rows are equalised. The comparison target is
crisprHAL 2's reported *E. coli* SpCas9 5-fold CV Spearman of 0.697
(PeerJ 2026), which uses a 378 nt input: 189 nt upstream, the 20 nt target,
3 nt PAM, 166 nt downstream.

    python -m sgrna.compare.headtohead --index     # locate + flank, cached, resumable
    python -m sgrna.compare.headtohead --run       # the three arms
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from .. import config
from ..build import curated_matrix
from ..build import genome

INDEX_PATH = config.INTERIM / "curated_guide_index.csv"
FLANK_PATH = config.INTERIM / "features" / "a_flank_curated.csv"
RESULTS_PATH = config.RESULTS / "headtohead.csv"

# crisprHAL 2, Better data for better predictions, PeerJ 2026.
CRISPRHAL2_SPEARMAN = 0.697
CRISPRHAL2_HELDOUT = 0.695
CRISPRHAL2_ROWS = 33495


# --------------------------------------------------------------------------
# Step 1 -- locate the curated guides and cut their flanks
# --------------------------------------------------------------------------


def build_index(chunk: int = 12000, deadline: float | None = None,
                verbose: bool = True) -> pd.DataFrame:
    """Genomic coordinates and ±1 kb of context for every curated guide.

    Written incrementally and resumable: locating 33.5k guides takes longer
    than one shell call allows, and re-doing the finished ones is wasted time.
    """
    hal = curated_matrix.load_crisprhal()
    hal[config.ID_COL] = hal["protospacer"]

    done = pd.DataFrame()
    if INDEX_PATH.exists() and INDEX_PATH.stat().st_size:
        done = pd.read_csv(INDEX_PATH)
        have = set(done[config.ID_COL])
        todo = hal[~hal[config.ID_COL].isin(have)]
    else:
        todo = hal
    if verbose:
        print(f"  {len(hal):,} curated guides, {len(done):,} already located, "
              f"{len(todo):,} to do")
    if todo.empty:
        return done

    g = genome.load_genome()
    genome.protospacer_sites()            # warm the index once

    records = []
    for n, (_, row) in enumerate(todo.iterrows(), start=1):
        proto = row["protospacer"]
        rec = dict({config.ID_COL: proto}, protospacer=proto,
                   score=row["score"], located=False, n_genome_copies=0)
        hits = genome.locate_protospacer(proto, g)
        if hits:
            h = hits[0]
            ctx = genome.guide_context(h["left"], h["right"], h["strand"],
                                       config.FLANK, g)
            rec.update(located=True, strand=h["strand"], left=h["left"],
                       right=h["right"], pam=h["pam"],
                       n_genome_copies=len(hits),
                       upstream=ctx["upstream"], downstream=ctx["downstream"])
        else:
            # Not in NC_000913.2 -- crisprHAL ships the 378 nt context, so the
            # flanks are still available, just capped at 189/166 nt.
            rec.update(pam=row["pam"],
                       upstream=row["context"][:189],
                       downstream=row["context"][212:])
        records.append(rec)
        if n % chunk == 0 or (deadline and time.time() > deadline):
            break

    out = pd.concat([done, pd.DataFrame(records)], ignore_index=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(INDEX_PATH, index=False)
    if verbose:
        print(f"  located {int(out['located'].sum()):,} / {len(out):,}; "
              f"wrote {INDEX_PATH.name}")
    return out


def build_flank(deadline: float | None = None, verbose: bool = True) -> pd.DataFrame:
    """Family A for the curated guides, using the same builder as the 13,880."""
    from ..features import a_flank

    index = pd.read_csv(INDEX_PATH)
    index["upstream"] = index["upstream"].fillna("").astype(str)
    index["downstream"] = index["downstream"].fillna("").astype(str)
    index["pam"] = index["pam"].fillna("NGG").astype(str)

    done = pd.DataFrame()
    if FLANK_PATH.exists() and FLANK_PATH.stat().st_size:
        done = pd.read_csv(FLANK_PATH)
        index = index[~index[config.ID_COL].isin(set(done[config.ID_COL]))]
    if verbose:
        print(f"  family A: {len(done):,} done, {len(index):,} to build")
    if index.empty:
        return done

    step = 4000
    frames = []
    for start in range(0, len(index), step):
        frames.append(a_flank.build(index.iloc[start:start + step]))
        if verbose:
            print(f"    +{len(frames[-1]):,}", flush=True)
        if deadline and time.time() > deadline:
            break
    out = pd.concat([done] + frames, ignore_index=True)
    FLANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FLANK_PATH, index=False)
    return out


# --------------------------------------------------------------------------
# Step 2 -- the three arms
# --------------------------------------------------------------------------


def _assemble(arm: str, verbose: bool = True):
    """(X, y, names) for one arm, aligned on protospacer."""
    import json

    meta = json.loads((config.INTERIM / "expanded_meta.json").read_text())
    protos = np.asarray(meta["protospacers"])
    y = np.asarray(meta["y"], dtype=float)
    in_pub = np.asarray(meta["in_published"], dtype=bool)
    names = list(meta["feature_names"])
    X = np.load(config.INTERIM / "expanded_X.npy", mmap_mode="r")

    keep = in_pub if arm == "published_rows" else np.ones(len(protos), bool)

    # crisprHAL 2 already feeds on 189 nt upstream + 166 downstream, so the
    # question is not whether long context matters -- it is whether encoding it
    # as windowed *composition* extracts anything their convolutions miss.
    # These two arms split family A by distance to answer that on their rows.
    SUBSET = {
        "curated_local": lambda c: (".seq." in c) or (".qct." in c),
        "curated_longrange": lambda c: (".comp." in c) or (".win." in c),
        # The matched-width arm. Their input spans 189 nt upstream and 166
        # downstream, so their network already *sees* this DNA. Keeping only
        # flank columns that fit inside that span asks the sharper question:
        # does encoding the same bases as windowed composition extract what
        # their convolutions miss, with neither model holding sequence the
        # other lacks? The 250/500/1000 nt windows reach past their input and
        # are dropped; the 50 nt windows and the near-cut columns remain.
        "curated_matched": lambda c: (".seq." in c) or (".qct." in c)
                                     or (".up50." in c) or (".dn50." in c),
    }

    if arm in ("curated_flank",) or arm in SUBSET:
        block = pd.read_csv(FLANK_PATH).set_index(config.ID_COL)
        if arm in SUBSET:
            block = block[[c for c in block.columns if SUBSET[arm](c)]]
        block = block.reindex(protos[keep])
        vals = block.to_numpy(dtype=np.float32)
        good = ~np.all(np.isnan(vals), axis=0)
        Xa = np.asarray(X[keep], dtype=np.float32)
        Xa = np.hstack([Xa, vals[:, good]])
        names = names + [c for c, k in zip(block.columns, good) if k]
    else:
        Xa = np.asarray(X[keep], dtype=np.float32)

    if verbose:
        print(f"  {arm}: {Xa.shape[0]:,} rows x {Xa.shape[1]:,} columns")
    return Xa, y[keep], names


def run(arms=("published_rows", "curated_all", "curated_flank"),
        seed: int = 41, n_splits: int = 5, n_features: int | None = None,
        model: str = "lightgbm", deadline: float | None = None,
        verbose: bool = True) -> pd.DataFrame:
    """Five-fold CV on the curated label, one row per fold. Resumable."""
    import xgboost as xgb
    from sklearn.model_selection import KFold

    from ..harness import evaluate as ev
    from ..harness.run_ablation import _impute, make_selector

    n_features = n_features or config.N_FEATURES
    prev = pd.DataFrame()
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size:
        prev = pd.read_csv(RESULTS_PATH)
    done = (set(zip(prev.get("arm", []), prev.get("seed", []), prev.get("fold", [])))
            if len(prev) else set())

    rows = []
    for arm in arms:
        if all((arm, seed, k) in done for k in range(1, n_splits + 1)):
            if verbose:
                print(f"  {arm}: already complete")
            continue
        X, y, names = _assemble(arm, verbose=verbose)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(kf.split(X), start=1):
            if (arm, seed, fold) in done:
                continue
            if deadline and time.time() > deadline:
                if verbose:
                    print("    time budget reached")
                return pd.DataFrame(rows)
            X_tr, X_va = _impute(X[tr], X[va])
            sel = make_selector(seed)
            sel.fit(X_tr, y[tr])
            top = np.argsort(sel.feature_importances_)[::-1][:n_features]
            m, needs_scale = ev._make(model, seed)
            A, B = X_tr[:, top], X_va[:, top]
            if needs_scale:
                A, B = ev._standardise(A, B)
            m.fit(A, y[tr])
            pred = np.asarray(m.predict(B)).ravel()
            row = dict(arm=arm, model=model, seed=seed, fold=fold,
                       n_rows=int(X.shape[0]), n_cols=int(X.shape[1]),
                       spearman=float(spearmanr(y[va], pred).statistic),
                       r2=float(r2_score(y[va], pred)),
                       pick_percentile=ev.expected_pick_percentile(y[va], pred))
            rows.append(row)
            pd.concat([prev, pd.DataFrame(rows)]).to_csv(RESULTS_PATH, index=False)
            if verbose:
                print(f"    {arm:16s} fold {fold}  rho {row['spearman']:.4f}",
                      flush=True)
    return pd.DataFrame(rows)


def summary() -> pd.DataFrame:
    """The table to put in the paper, with the published target alongside."""
    d = pd.read_csv(RESULTS_PATH)
    g = d.groupby("arm").agg(
        folds=("fold", "size"), seeds=("seed", "nunique"),
        rows=("n_rows", "first"),
        cols=("n_cols", "first"), spearman=("spearman", "mean"),
        sd=("spearman", "std"), r2=("r2", "mean"),
        pick=("pick_percentile", "mean"),
    ).reset_index()
    g["vs_crisprHAL2"] = (g["spearman"] - CRISPRHAL2_SPEARMAN).round(4)
    return g.round(4).sort_values("spearman")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--flank", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--time-budget", type=float, default=None)
    args = ap.parse_args(argv)

    deadline = time.time() + args.time_budget if args.time_budget else None
    if args.index:
        build_index(deadline=deadline)
    if args.flank:
        build_flank(deadline=deadline)
    if args.run:
        run(arms=tuple(args.arms) if args.arms else
            ("published_rows", "curated_all", "curated_flank"),
            model=args.model, seed=args.seed, n_splits=args.folds,
            deadline=deadline)
        print(summary().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
