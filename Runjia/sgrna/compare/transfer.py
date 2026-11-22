"""Does the flank effect transfer to another nuclease? (Part 10, item 2.)

Family A is this project's one large effect, established on *E. coli*
WT-SpCas9. The obvious question is whether it is a property of the DNA or a
property of that one screen, and crisprHAL ships three further datasets to ask
with. Only one of them can answer without new data:

    WT-SpCas9    E. coli,       SpCas9       378 nt context   -- done (headtohead)
    eSpCas9      E. coli,       eSpCas9      406 nt context   -- this module
    TevSpCas9    C. rodentium,  TevSpCas9     37 nt context   -- not possible
    TevSaCas9    C. rodentium,  TevSaCas9     29 nt context   -- not possible

The two *C. rodentium* sets ship only the target and a few flanking bases, so
testing the cross-species question needs the C. rodentium genome to recover
flanks -- a download, and therefore out of scope here. What is available is the
cleaner half of the comparison anyway: eSpCas9 is a different nuclease in the
*same* organism, so a result there isolates the enzyme from the genome. If
windowed flank composition helps for eSpCas9 too, the effect belongs to the
DNA rather than to SpCas9's particular behaviour.

eSpCas9 is an engineered high-fidelity SpCas9 variant: same PAM, same target
length, deliberately reduced tolerance of mismatches. It is also the screen
used for the reproducibility ceiling in `diagnose.ceiling`, so its label is
already known to agree with the WT screen at rho 0.810.

Both arms use the protocol of `headtohead.run` so the numbers are directly
comparable: base features regenerated from the protospacer by `featurise`,
family A by the same builder as everywhere else, top-300 selection by XGBoost
gain fitted inside the training split, LightGBM, 5-fold CV.

    python -m sgrna.compare.transfer --dataset eSpCas9 --build
    python -m sgrna.compare.transfer --dataset eSpCas9 --run
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from .. import config
from ..io_utils import save_block

HAL_DATA = config.EXTERNAL / "crisprHAL" / "data"
RESULTS_PATH = config.RESULTS / "transfer.csv"

PROTO_LEN = 20
PAM_LEN = 3

# Minimum flank we insist on before calling a dataset usable here. Family A's
# windows run to 1 kb but degrade gracefully; below ~100 nt there is nothing
# for the long-range half to measure.
MIN_FLANK = 100


def load_dataset(name: str, genome_fasta=None, verbose: bool = True) -> pd.DataFrame:
    """Context + score for one crisprHAL dataset, with the layout detected.

    Their files are headerless `(context, score)` pairs, and the protospacer
    offset differs between datasets (189 for WT-SpCas9, 193 for eSpCas9), so
    it is found rather than assumed: the offset whose following 3 nt are an
    NGG PAM for every row.
    """
    frames = [
        pd.read_csv(HAL_DATA / f"{name}_{split}_data.csv",
                    header=None, names=["context", "score"])
        for split in ("training", "testing")
    ]
    d = pd.concat(frames, ignore_index=True).dropna(subset=["context"])
    d["context"] = d["context"].str.upper()

    length = int(d["context"].str.len().mode().iloc[0])
    d = d[d["context"].str.len() == length]

    # NGG occupies off+20 .. off+22, so the two Gs are at off+21 and off+22.
    # Checking off+20..off+21 instead silently shifts every guide by one base
    # and still scores 1.000, so this is written out explicitly: the criterion
    # must return 189 for WT-SpCas9, which `crisprhal_rerun` independently
    # hard-codes.
    best_off, best_frac = None, -1.0
    for off in range(0, length - PROTO_LEN - PAM_LEN):
        frac = float((d["context"].str[off + PROTO_LEN + 1:off + PROTO_LEN + 3] == "GG").mean())
        if frac > best_frac:
            best_off, best_frac = off, frac
    if best_frac < 0.99:
        raise ValueError(
            f"{name}: no offset gives a consistent NGG PAM (best {best_frac:.3f} "
            f"at {best_off}); this dataset may use a different PAM")

    up, down = best_off, length - best_off - PROTO_LEN - PAM_LEN
    short = min(up, down) < MIN_FLANK
    if short and genome_fasta is None:
        raise ValueError(
            f"{name}: only {up} nt upstream / {down} nt downstream of context. "
            f"Family A needs at least {MIN_FLANK} on each side; pass "
            f"genome_fasta= for this organism to recover the flanks.")

    # `a_flank.build` skips any row whose `located` flag is unset, because in
    # the main pipeline that flag means "we found this guide in NC_000913.2 and
    # cut its flanks from the genome". Here the flanks come from crisprHAL's own
    # shipped context, which is just as real, so the flag is set directly. (Note
    # the same guard silently leaves family A all-NaN for the handful of curated
    # guides `headtohead` could not locate, even though their context was
    # available -- worth a look if those rows ever matter.)
    d["located"] = True
    d["protospacer"] = d["context"].str[best_off:best_off + PROTO_LEN]
    if short:
        d = _flanks_from_genome(d, genome_fasta, verbose=verbose)
    else:
        d["pam"] = d["context"].str[best_off + PROTO_LEN:best_off + PROTO_LEN + PAM_LEN]
        d["upstream"] = d["context"].str[:best_off]
        d["downstream"] = d["context"].str[best_off + PROTO_LEN + PAM_LEN:]

    # One score per guide: duplicated protospacers are the same target measured
    # twice, so average rather than letting a fold boundary split them.
    # Keep every column the caller needs: an agg silently drops the rest, and
    # losing `left` here is what breaks arc grouping downstream.
    agg = dict(score=("score", "mean"), pam=("pam", "first"),
               upstream=("upstream", "first"), downstream=("downstream", "first"),
               located=("located", "first"))
    for extra in ("left", "genome_length"):
        if extra in d:
            agg[extra] = (extra, "first")
    d = d.groupby("protospacer", as_index=False).agg(**agg)
    d = d[~d["protospacer"].str.contains("N")]
    d[config.ID_COL] = d["protospacer"]

    if verbose:
        print(f"  {name}: {len(d):,} unique guides, {length} nt context "
              f"({up} up / {down} down), PAM at offset {best_off + PROTO_LEN}")
    return d.reset_index(drop=True)


def _flanks_from_genome(d: pd.DataFrame, fasta, flank: int = 1000,
                        verbose: bool = True) -> pd.DataFrame:
    """Recover flanks by locating each protospacer in another organism's genome.

    The crisprHAL *C. rodentium* sets ship only ~37 nt of context, so family A's
    windows have to come from the genome instead. Each guide is searched on both
    strands; slicing the reverse-complement strand directly keeps the guide's own
    orientation, so upstream/downstream mean the same thing as everywhere else.
    """
    seq = "".join(l.strip() for l in open(fasta) if not l.startswith(">")).upper()
    rev = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    ups, downs, pams, ok, lefts = [], [], [], [], []
    for proto in d["protospacer"]:
        i = seq.find(proto)
        strand_seq, pos = (seq, i) if i >= 0 else (rev, rev.find(proto))
        if pos < 0 or seq.find(proto, i + 1) >= 0:
            # Missing, or multi-copy: a guide with two genomic sites has no
            # single flank, so drop it rather than pick one arbitrarily.
            ups.append(None); downs.append(None); pams.append(None)
            ok.append(False); lefts.append(np.nan)
            continue
        end = pos + PROTO_LEN
        ups.append(strand_seq[max(0, pos - flank):pos])
        pams.append(strand_seq[end:end + PAM_LEN])
        downs.append(strand_seq[end + PAM_LEN:end + PAM_LEN + flank])
        ok.append(True)
        # Position on the forward strand, for contiguous-arc grouping. A
        # reverse-strand hit at `pos` in the reversed sequence sits at
        # len - pos - 20 on the forward one.
        lefts.append(pos if strand_seq is seq else len(seq) - pos - PROTO_LEN)

    d = d.assign(upstream=ups, downstream=downs, pam=pams, located=ok,
                 left=lefts, genome_length=len(seq))
    if verbose:
        print(f"  located {sum(ok):,} / {len(ok):,} guides in the supplied genome")
    return d[d["located"]].reset_index(drop=True)


def _positions_from_ecoli(protospacers, verbose: bool = True) -> np.ndarray:
    """Genomic left edge in NC_000913.2 for each guide, or NaN if not found.

    Used for the *E. coli* screens that ship their own context: the flanks come
    from their file, but grouped cross-validation still needs to know where on
    the chromosome each guide sits. `genome.protospacer_sites` indexes every
    NGG-flanked site once, which is far cheaper than a scan per guide.
    """
    from ..build import genome

    seqs, lefts, _ = genome.protospacer_sites()
    pos = dict(zip(seqs, lefts))
    out = np.asarray([pos.get(p, np.nan) for p in protospacers], dtype=float)
    if verbose:
        print(f"  located {int(np.isfinite(out).sum()):,} / {len(out):,} "
              f"guides in NC_000913.2 for grouping")
    return out


def build(name: str, genome_fasta=None, verbose: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Base features + family A for one dataset, cached to disk."""
    from ..build import featurise
    from ..features import a_flank

    idx = load_dataset(name, genome_fasta=genome_fasta, verbose=verbose)
    cache = config.INTERIM / f"transfer_{name}"
    cache.mkdir(parents=True, exist_ok=True)

    x_path, meta_path = cache / "X.npy", cache / "meta.json"
    flank_path = config.INTERIM / "features" / f"a_flank_{name}.csv"

    if x_path.exists() and meta_path.exists():
        import json
        meta = json.loads(meta_path.read_text())
        if "protospacers" not in meta:
            meta["protospacers"] = idx["protospacer"].astype(str).tolist()
            meta_path.write_text(json.dumps(meta))
        X = np.load(x_path, mmap_mode="r")
        if verbose:
            print(f"  cached: {X.shape[0]:,} x {X.shape[1]:,}")
        return X, np.asarray(meta["y"], float), list(meta["names"]),
    if verbose:
        print("  regenerating the published representation from protospacers")
    # `matrix` returns (X, names) -- unpack it, like expand.py does.
    base, base_names = featurise.matrix(idx["protospacer"].tolist(), verbose=verbose)
    base_names = list(base_names)

    if flank_path.exists():
        block = pd.read_csv(flank_path)
    else:
        if verbose:
            print("  building family A from their shipped context")
        block = a_flank.build(idx)
        block.to_csv(flank_path, index=False)
    block = block.set_index(config.ID_COL).reindex(idx["protospacer"])

    vals = block.to_numpy(dtype=np.float32)
    good = ~np.all(np.isnan(vals), axis=0)
    X = np.hstack([np.asarray(base, dtype=np.float32), vals[:, good]])
    names = base_names + [c for c, k in zip(block.columns, good) if k]

    import json
    if "left" not in idx:
        # E. coli screens ship context but no coordinates; grouped CV needs them.
        idx = idx.assign(left=_positions_from_ecoli(idx["protospacer"], verbose),
                         genome_length=4_641_652)
    np.save(x_path, X)
    meta_path.write_text(json.dumps(
        {"y": idx["score"].tolist(), "names": names,
         "n_base": len(base_names), "dataset": name,
         "protospacers": idx["protospacer"].astype(str).tolist(),
         "left": (idx["left"].tolist() if "left" in idx else None),
         "genome_length": (int(idx["genome_length"].iloc[0])
                           if "genome_length" in idx else None)}))
    if verbose:
        print(f"  built {X.shape[0]:,} x {X.shape[1]:,} "
              f"({len(base_names):,} base + {X.shape[1] - len(base_names)} family A)")
    return X, idx["score"].to_numpy(float), names


def run(name: str = "eSpCas9", seed: int = 41, n_splits: int = 5,
        n_features: int | None = None, model: str = "lightgbm",
        genome_fasta=None, group_by: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """base vs base + family A, same protocol as `headtohead.run`. Resumable."""
    import json

    import xgboost as xgb
    from sklearn.model_selection import KFold

    from ..harness import evaluate as ev
    from ..harness.run_ablation import _impute, make_selector

    n_features = n_features or config.N_FEATURES
    X, y, names = build(name, genome_fasta=genome_fasta, verbose=verbose)
    n_base = json.loads((config.INTERIM / f"transfer_{name}" / "meta.json").read_text())["n_base"]

    prev = pd.DataFrame()
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size:
        prev = pd.read_csv(RESULTS_PATH)
    done = (set(zip(prev.get("dataset", []), prev.get("arm", []),
                    prev.get("seed", []), prev.get("fold", []),
                    prev.get("cv", ["random"] * len(prev))))
            if len(prev) else set())

    # Family A splits the same way §14 splits it in E. coli, so the *shape* of
    # the effect can be compared across organisms and not just its size:
    #   local      the immediate +/-10 nt, which published bacterial tools read
    #   longrange  composition of 50-1000 nt windows, which they do not
    names_arr = np.asarray(names)
    is_flank = np.arange(len(names)) >= n_base
    is_local = is_flank & np.array([(".seq." in c) or (".qct." in c) for c in names])
    is_long = is_flank & np.array([(".comp." in c) or (".win." in c) for c in names])

    base_cols = np.arange(n_base)
    arms = {
        "base": base_cols,
        "base+local": np.r_[base_cols, np.flatnonzero(is_local)],
        "base+longrange": np.r_[base_cols, np.flatnonzero(is_long)],
        "base+a_flank": np.arange(X.shape[1]),
    }

    # Random-row CV is not a safe default here. These screens tile the
    # chromosome densely, so neighbouring guides share flanking sequence and a
    # model given windowed composition can score well by recognising a locus
    # rather than by generalising -- the trap §8 identified in E. coli. `arc5`
    # cuts the chromosome into five contiguous arcs and holds out one per fold,
    # which forces extrapolation to unseen sequence.
    groups = None
    if group_by:
        meta_all = json.loads((config.INTERIM / f"transfer_{name}" / "meta.json").read_text())
        left = meta_all.get("left")
        if left is None:
            raise ValueError(
                f"{name}: no genomic positions cached, so arc grouping is not "
                f"possible. Rebuild with genome_fasta= to record them.")
        left = np.asarray(left, float)
        if group_by == "arc5":
            size = meta_all["genome_length"] / 5
            groups = np.clip(np.nan_to_num(left, nan=0.0) // size, 0, 4).astype(int)
        elif group_by.startswith("bin"):
            kb = int(group_by[3:].rstrip("k"))
            groups = (np.nan_to_num(left, nan=0.0) // (kb * 1000)).astype(int)
        else:
            raise ValueError(f"unknown grouping {group_by!r}")
        n_groups = len(set(groups.tolist()))
        if verbose:
            print(f"  {group_by}: {n_groups} occupied groups over "
                  f"{len(groups):,} guides")
        if n_groups < n_splits:
            raise ValueError(
                f"{name}: {group_by} yields only {n_groups} groups for "
                f"{n_splits} folds. This screen does not span enough of the "
                f"chromosome for that scheme -- use finer blocks.")
    if verbose:
        print(f"  arms: local {int(is_local.sum())} cols, "
              f"longrange {int(is_long.sum())} cols, "
              f"family A {int(is_flank.sum())} cols")
    rows = []
    for arm, cols in arms.items():
        if groups is None:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = splitter.split(X)
        else:
            from sklearn.model_selection import GroupKFold
            splits = GroupKFold(n_splits=n_splits).split(X, groups=groups)
        for fold, (tr, va) in enumerate(splits, start=1):
            if (name, arm, seed, fold, group_by or "random") in done:
                continue
            t0 = time.time()
            Xc = np.asarray(X[:, cols], dtype=np.float32)
            X_tr, X_va = _impute(Xc[tr], Xc[va])
            sel = make_selector(seed)
            sel.fit(X_tr, y[tr])
            top = np.argsort(sel.feature_importances_)[::-1][:n_features]
            m, needs_scale = ev._make(model, seed)
            A, B = X_tr[:, top], X_va[:, top]
            if needs_scale:
                A, B = ev._standardise(A, B)
            m.fit(A, y[tr])
            pred = np.asarray(m.predict(B)).ravel()
            rows.append(dict(
                dataset=name, arm=arm, model=model, seed=seed, fold=fold,
                cv=(group_by or "random"),
                n_rows=int(Xc.shape[0]), n_cols=int(Xc.shape[1]),
                spearman=float(spearmanr(y[va], pred).statistic),
                r2=float(r2_score(y[va], pred)),
                pick_percentile=ev.expected_pick_percentile(y[va], pred),
                seconds=round(time.time() - t0, 1)))
            pd.concat([prev, pd.DataFrame(rows)]).to_csv(RESULTS_PATH, index=False)
            if verbose:
                print(f"    {arm:14s} fold {fold}  rho {rows[-1]['spearman']:.4f}",
                      flush=True)
    return pd.DataFrame(rows)


def summary(name: str | None = None) -> pd.DataFrame:
    d = pd.read_csv(RESULTS_PATH)
    if name:
        d = d[d.dataset == name]
    return (d.groupby(["dataset", "arm"])
             .agg(rho=("spearman", "mean"), rho_sd=("spearman", "std"),
                  r2=("r2", "mean"), pick=("pick_percentile", "mean"),
                  folds=("fold", "count"))
             .round(4))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="eSpCas9",
                    help="crisprHAL dataset name, e.g. eSpCas9")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--genome", default=None,
                    help="FASTA for datasets that ship too little context, "
                         "e.g. external_data/C_rodentium/NC_013716.1.fasta")
    a = ap.parse_args(argv)
    if a.build:
        build(a.dataset, genome_fasta=a.genome)
    if a.run:
        run(a.dataset, seed=a.seed, genome_fasta=a.genome)
    if a.summary or not (a.build or a.run):
        print(summary().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Model transfer: train on one screen, rank another's guides.
#
# Every arm above trains and tests inside one screen, so it answers "are these
# features informative here?" and not "does a model built there work here?".
# This is the second question, and it is the one a reader assumes.

TRANSFER_PATH = config.RESULTS / "transfer_model.csv"


def transfer_model(source: str, target: str, source_genome=None,
                   target_genome=None, model: str = "lightgbm",
                   n_features: int | None = None, seed: int = 41,
                   verbose: bool = True) -> pd.DataFrame:
    """Fit on all of `source`, predict all of `target`, with and without family A.

    Both arms are reported because they answer different things: `base` asks
    whether the published representation transfers at all, and `base+a_flank`
    asks whether the flank features help or hurt when the target screen is a
    different enzyme or organism.

    Guide overlap between the two screens is measured and reported rather than
    removed, because for the two *E. coli* screens it is near-total -- the same
    library scored under two nucleases -- and that makes the comparison a test
    of label transfer, not of sequence generalisation. The disjoint arms are
    the ones that test generalisation.
    """
    import json

    import xgboost as xgb

    from ..harness import evaluate as ev
    from ..harness.run_ablation import _impute, make_selector

    n_features = n_features or config.N_FEATURES
    Xs, ys, names = build(source, genome_fasta=source_genome, verbose=verbose)
    Xt, yt, names_t = build(target, genome_fasta=target_genome, verbose=verbose)
    if names != names_t:
        raise ValueError("feature names differ between screens")
    n_base = json.loads(
        (config.INTERIM / f"transfer_{source}" / "meta.json").read_text())["n_base"]

    src_meta = json.loads(
        (config.INTERIM / f"transfer_{source}" / "meta.json").read_text())
    tgt_meta = json.loads(
        (config.INTERIM / f"transfer_{target}" / "meta.json").read_text())
    src_p = set(src_meta.get("protospacers") or [])
    tgt_p = set(tgt_meta.get("protospacers") or [])
    if src_p and tgt_p:
        overlap = len(src_p & tgt_p) / len(tgt_p)
    else:
        overlap = float("nan")

    # Split family A the same way the within-screen arms do, so the transfer
    # can be attributed: which half of the flank encoding actually carries
    # across an organism boundary?
    is_flank = np.arange(len(names)) >= n_base
    is_local = is_flank & np.array([(".seq." in c) or (".qct." in c) for c in names])
    is_long = is_flank & np.array([(".comp." in c) or (".win." in c) for c in names])
    base_cols = np.arange(n_base)

    rows = []
    for arm, cols in (("base", base_cols),
                      ("base+local", np.r_[base_cols, np.flatnonzero(is_local)]),
                      ("base+longrange", np.r_[base_cols, np.flatnonzero(is_long)]),
                      ("base+a_flank", np.arange(Xs.shape[1]))):
        A = np.asarray(Xs[:, cols], dtype=np.float32)
        B = np.asarray(Xt[:, cols], dtype=np.float32)
        A, B = _impute(A, B)
        sel = make_selector(seed)
        sel.fit(A, ys)
        top = np.argsort(sel.feature_importances_)[::-1][:n_features]
        m, needs_scale = ev._make(model, seed)
        A2, B2 = A[:, top], B[:, top]
        if needs_scale:
            A2, B2 = ev._standardise(A2, B2)
        m.fit(A2, ys)
        pred = np.asarray(m.predict(B2)).ravel()
        rows.append(dict(source=source, target=target, arm=arm, model=model,
                         seed=seed, n_train=len(ys), n_test=len(yt),
                         guide_overlap=overlap,
                         spearman=float(spearmanr(yt, pred).statistic),
                         r2=float(r2_score(yt, pred)),
                         pick_percentile=ev.expected_pick_percentile(yt, pred)))
        if verbose:
            print(f"    {source} -> {target:10s} {arm:14s} rho {rows[-1]['spearman']:.4f}",
                  flush=True)
    out = pd.DataFrame(rows)
    prev = pd.read_csv(TRANSFER_PATH) if TRANSFER_PATH.exists() else pd.DataFrame()
    pd.concat([prev, out]).to_csv(TRANSFER_PATH, index=False)
    return out


# ---------------------------------------------------------------------------
# The hybrid. The decomposition said the flank-to-efficiency relationship is
# better learned from a genome-wide screen in the wrong organism than from a
# narrow screen in the right one, while the base features are better learned
# locally. So: fit the flank part where the variation is, the rest where the
# labels are.

def hybrid(source: str, target: str, source_genome=None, target_genome=None,
           seed: int = 41, n_splits: int = 5, group_by: str = "bin10k",
           n_features: int | None = None, model: str = "lightgbm",
           verbose: bool = True) -> pd.DataFrame:
    """Import a flank prior from `source`, fit everything else on `target`.

    Stage 1 trains on `source` using **family A alone** -- no base columns, so
    the prior carries the flank relationship and nothing about that screen's
    label scale or nuclease -- and scores every `target` guide with it.
    Stage 2 runs ordinary grouped CV on `target` with that single extra column.

    No target label is used in stage 1, so the prior leaks nothing across the
    fold boundary.
    """
    import json

    import xgboost as xgb
    from sklearn.model_selection import GroupKFold

    from ..harness import evaluate as ev
    from ..harness.run_ablation import _impute, make_selector

    n_features = n_features or config.N_FEATURES
    Xs, ys, names = build(source, genome_fasta=source_genome, verbose=verbose)
    Xt, yt, _ = build(target, genome_fasta=target_genome, verbose=verbose)
    meta_s = json.loads((config.INTERIM / f"transfer_{source}" / "meta.json").read_text())
    meta_t = json.loads((config.INTERIM / f"transfer_{target}" / "meta.json").read_text())
    n_base = meta_s["n_base"]

    flank = np.arange(n_base, Xs.shape[1])
    A, B = _impute(np.asarray(Xs[:, flank], np.float32),
                   np.asarray(Xt[:, flank], np.float32))
    prior_model, needs_scale = ev._make(model, seed)
    A2, B2 = (ev._standardise(A, B) if needs_scale else (A, B))
    prior_model.fit(A2, ys)
    prior = np.asarray(prior_model.predict(B2), dtype=np.float32).reshape(-1, 1)
    if verbose:
        print(f"  flank prior from {source}: rho vs {target} label "
              f"{spearmanr(yt, prior.ravel()).statistic:.4f}")

    left = np.asarray(meta_t["left"], float)
    kb = int(group_by[3:].rstrip("k"))
    groups = (np.nan_to_num(left, nan=0.0) // (kb * 1000)).astype(int)

    base_cols = np.arange(n_base)
    arms = {
        "base": (base_cols, False),
        "base+prior": (base_cols, True),
        "base+a_flank": (np.arange(Xt.shape[1]), False),
        "base+a_flank+prior": (np.arange(Xt.shape[1]), True),
    }
    rows = []
    for arm, (cols, use_prior) in arms.items():
        X = np.asarray(Xt[:, cols], np.float32)
        if use_prior:
            X = np.hstack([X, prior])
        for fold, (tr, va) in enumerate(
                GroupKFold(n_splits=n_splits).split(X, groups=groups), start=1):
            X_tr, X_va = _impute(X[tr], X[va])
            sel = make_selector(seed)
            sel.fit(X_tr, yt[tr])
            top = np.argsort(sel.feature_importances_)[::-1][:n_features]
            m, scale = ev._make(model, seed)
            P, Q = X_tr[:, top], X_va[:, top]
            if scale:
                P, Q = ev._standardise(P, Q)
            m.fit(P, yt[tr])
            pred = np.asarray(m.predict(Q)).ravel()
            rows.append(dict(source=source, target=target, arm=arm, fold=fold,
                             cv=group_by, seed=seed,
                             spearman=float(spearmanr(yt[va], pred).statistic),
                             r2=float(r2_score(yt[va], pred))))
        if verbose:
            s = np.mean([r["spearman"] for r in rows if r["arm"] == arm])
            print(f"    {arm:22s} rho {s:.4f}", flush=True)
    out = pd.DataFrame(rows)
    path = config.RESULTS / "hybrid.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([prev, out]).to_csv(path, index=False)
    return out
