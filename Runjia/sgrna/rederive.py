"""Re-derive the result files whose producer was lost, and check they agree.

The claim -> evidence index found thirteen files cited by the report with no
producer left in `src/` or `colab/` -- written by ad-hoc scripts from earlier
sessions and changed or discarded since. A number nobody can regenerate is a
number nobody can argue with, which is the gap `human_transfer.csv` had when it
cost a cross-kingdom claim.

This module re-derives them. It writes into `results/rederived/` and **compares**
against the stored file rather than overwriting it: re-derivation is a check, not
a replacement, and a disagreement is a finding. The stored file stays
authoritative until the comparison passes.

They live together here rather than in eleven natural homes because what they
have in common is provenance, not subject -- and one module is one thing to keep
working.

    python -m sgrna.rederive                 # everything cheap
    python -m sgrna.rederive --only grouped_retention selector_control
    python -m sgrna.rederive --list

One of the thirteen is not here. `crisprhal_matched_folds.csv` is a GPU
artefact: `colab/crisprhal_tune_matched.py` now writes it under that exact name
with the `is_published_config` flag, so a rerun reproduces it, but the rerun
needs a T4 and cannot happen on this machine. `matched_search_folds.csv`, which
was transcribed alongside it, is only a join of four local files and *is* here.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config

OUTDIR = config.RESULTS / "rederived"

# The reach classification used in the report's family table. It is a judgement
# about what each family can see, not a measurement, so it is recorded here
# rather than inferred.
REACH = {"a_flank": "flank", "b_energy": "guide", "c_folding": "guide",
         "d_mechanics": "guide+flank", "d_supercoiling": "external",
         "e_nucleoid": "external", "f_methylation": "guide",
         "g_transcription": "external", "h_offtarget": "genome",
         "i_shape": "flank", "j_periodicity": "position"}

# The grouping tax is the *aggregate* retention of the three families that are
# functions of the guide alone, so grouping cannot invalidate them: whatever
# they lose is what holding out whole genes costs any feature.
TAX_FAMILIES = ("b_energy", "c_folding", "f_methylation")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _published():
    from .build_matrix import load_dataset
    X, y, names, ids, fam = load_dataset([], verbose=False)
    return X, y, list(names)


def _cv_rho(X, y, cols, n_features=None, seeds=(41,), n_splits=5,
            model="xgboost"):
    """Mean rho over folds on a fixed column subset, with or without selection."""
    import xgboost as xgb
    from sklearn.model_selection import KFold

    from .run_ablation import _impute, champion_params, make_selector

    rhos = []
    for seed in seeds:
        for tr, va in KFold(n_splits, shuffle=True,
                            random_state=seed).split(X):
            A, B = _impute(X[np.ix_(tr, cols)], X[np.ix_(va, cols)])
            if n_features is not None and n_features < A.shape[1]:
                sel = make_selector(seed)
                sel.fit(A, y[tr])
                top = np.argsort(sel.feature_importances_)[::-1][:n_features]
                A, B = A[:, top], B[:, top]
            if model == "ridge":
                from sklearn.linear_model import RidgeCV
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler().fit(A)
                m = RidgeCV(alphas=np.logspace(-2, 4, 25))
                A, B = sc.transform(A), sc.transform(B)
            elif model == "lightgbm":
                import lightgbm as lgb
                m = lgb.LGBMRegressor(n_estimators=config.N_ESTIMATORS,
                                      learning_rate=0.05, max_depth=4,
                                      verbose=-1, random_state=seed)
            else:
                m = xgb.XGBRegressor(**champion_params(seed))
            m.fit(A, y[tr])
            rhos.append(float(spearmanr(y[va], m.predict(B)).statistic))
    return float(np.mean(rhos)), float(np.std(rhos, ddof=1)) if len(rhos) > 1 else 0.0


# --------------------------------------------------------------------------
# the re-derivations
# --------------------------------------------------------------------------

def grouped_retention() -> pd.DataFrame:
    """Pure post-processing of the two curated ablation tables."""
    rnd = pd.read_csv(config.RESULTS / "ablation_curated.csv")
    gene = pd.read_csv(config.RESULTS / "ablation_curated_gene.csv")
    key = "family"
    r = rnd.set_index(key)["d_spearman"]
    g = gene.set_index(key)["d_spearman"]
    fams = [f for f in r.index if f in g.index and f in REACH]
    rows = []
    for f in fams:
        rows.append(dict(family=f, random=float(r[f]), gene=float(g[f]),
                         retained_pct=float(round(100 * g[f] / r[f])),
                         reach=REACH[f]))
    d = pd.DataFrame(rows).sort_values("random", ascending=False)
    tax = round(100 * sum(g[f] for f in TAX_FAMILIES)
                / sum(r[f] for f in TAX_FAMILIES))
    d["excess_loss_pp"] = (tax - d["retained_pct"]).round(0)
    return d.reset_index(drop=True)


def selector_control() -> pd.DataFrame:
    """Is the 300-column budget costing anything on the curated cohort?"""
    from .curated_ablation import _base, _block
    X, y, names, protos = _base()
    v, _ = _block("d_mechanics", protos)
    Xm = np.hstack([X, v])
    rows = []
    for tag, M in (("base", X), ("base + mech", Xm)):
        for sel, nf in (("select 300", config.N_FEATURES), ("all", None)):
            rho, sd = _cv_rho(M, y, np.arange(M.shape[1]), n_features=nf)
            rows.append(dict(arm=f"{tag}, {sel}", n_cols=M.shape[1],
                             selected=nf or M.shape[1], spearman=rho, sd=sd))
    return pd.DataFrame(rows)


def kmer_order_noselect() -> pd.DataFrame:
    """Reduced k-mer sets on the published matrix, no selection."""
    from .representation import column_sets
    X, y, names = _published()
    s = column_sets(names)
    rows = []
    for arm, cols in (("mono (1-mer)", s["mono_74"]),
                      ("mono+di", s["mono_di_364"])):
        rho, sd = _cv_rho(X, y, cols, n_features=None, seeds=config.SEEDS)
        rows.append(dict(arm=arm, n_columns=len(cols), spearman=rho, sd=sd))
    return pd.DataFrame(rows)


def portability_364() -> pd.DataFrame:
    """The same two sets, reported the way the portability argument uses them."""
    from .representation import column_sets
    X, y, names = _published()
    s = column_sets(names)
    rows = []
    for arm, cols in (("mono_di_364", s["mono_di_364"]),
                      ("mono_74", s["mono_74"])):
        rho, sd = _cv_rho(X, y, cols, n_features=None, seeds=(41, 42, 43))
        rows.append(dict(arm=arm, n_columns=len(cols), spearman=rho,
                         spearman_sd=sd))
    return pd.DataFrame(rows)


def additivity_reduced() -> pd.DataFrame:
    """Ridge against boosting on the decorrelated reduced sets."""
    from .build_matrix import load_dataset
    from .representation import column_sets
    X, y, names, ids, fam = load_dataset(["a_flank"], verbose=False)
    fam = np.asarray(fam)
    base = np.where(fam == "base")[0]
    flank = np.where(fam != "base")[0]
    s = column_sets([names[i] for i in base])
    red = np.sort(np.concatenate([base[s["mono_di_364"]],
                                  base[s["hand_named_only"]]]))
    arms = {"reduced 427 (mono+di+hand)": red,
            "reduced 775 (+a_flank)": np.sort(np.concatenate([red, flank])),
            "full 6,232 (base only)": base,
            "full 6,580 (+a_flank)": np.sort(np.concatenate([base, flank]))}
    rows = []
    for arm, cols in arms.items():
        out = {m: _cv_rho(X, y, cols, n_features=None, model=m)[0]
               for m in ("ridge", "lightgbm", "xgboost")}
        rows.append(dict(arm=arm, n_columns=len(cols), **out,
                         boosting_minus_ridge=out["lightgbm"] - out["ridge"]))
    return pd.DataFrame(rows)


def d_mechanics_reconstructibility() -> pd.DataFrame:
    """How well each d_mechanics column is predicted by the mono+di one-hots."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_predict

    from .build_matrix import load_dataset
    from .representation import column_sets
    X, y, names, ids, fam = load_dataset(["d_mechanics"], verbose=False)
    fam = np.asarray(fam)
    base = np.where(fam == "base")[0]
    mech = np.where(fam != "base")[0]
    s = column_sets([names[i] for i in base])
    A = np.nan_to_num(X[:, base[s["mono_di_364"]]])
    rows = []
    for j in mech:
        t = X[:, j]
        ok = np.isfinite(t)
        if ok.sum() < 100 or np.std(t[ok]) == 0:
            continue
        pred = cross_val_predict(LinearRegression(), A[ok], t[ok], cv=5)
        ss = 1 - np.sum((t[ok] - pred) ** 2) / np.sum((t[ok] - t[ok].mean()) ** 2)
        rows.append(dict(oof_r2_from_mono_di=round(float(ss), 6)))
    return pd.DataFrame(rows)


def abund_local_vs_mappability() -> pd.DataFrame:
    """Are the GapR abundance tracks reading mappability, or local GC?

    Each abundance track is correlated against two genome-derived quantities
    computed in the same four windows: **20-mer uniqueness**, the fraction of
    positions in the window whose 20-mer occurs exactly once in the genome, and
    **GC**. If a track were a mappability artefact it would track uniqueness.
    """
    from .genome import load_genome
    from .io_utils import load_block

    g = load_genome()
    n = len(g)
    arr = np.frombuffer(g.encode(), dtype=np.uint8)
    gc = np.isin(arr, [ord("G"), ord("C"), ord("g"), ord("c")]).astype(np.float32)

    seen = {}
    for i in range(n - 20):
        k = g[i:i + 20]
        seen[k] = seen.get(k, 0) + 1
    uniq = np.zeros(n, dtype=np.float32)
    for i in range(n - 20):
        uniq[i] = 1.0 if seen[g[i:i + 20]] == 1 else 0.0

    def window_mean(track, centre, w):
        half = w // 2
        cs = np.concatenate([[0.0], np.cumsum(track, dtype=np.float64)])
        lo = np.clip(centre - half, 0, len(track))
        hi = np.clip(centre + half, 0, len(track))
        return (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)

    idx = pd.read_csv(config.GUIDE_INDEX)
    sc = load_block("d_supercoiling").set_index("sgRNAID").reindex(idx["sgRNAID"])
    cf = (idx["left"].to_numpy(float) + idx["right"].to_numpy(float)) / 2
    ok = np.isfinite(cf) & (cf > 0) & (cf < n)        # mask BEFORE casting
    centre = cf[ok].astype(int)
    rows = []
    for t in [c for c in sc.columns if ".abund." in c]:
        v = pd.to_numeric(sc[t], errors="coerce").to_numpy(float)[ok]
        row = {"track": t.replace("eng.sc.abund.", "")}
        for w in (100, 500, 2000, 10000):
            for nm, src in (("uniq", uniq), ("gc", gc)):
                x = window_mean(src, centre, w)
                row[f"{nm}.w{w}"] = round(float(
                    spearmanr(v, x, nan_policy="omit").statistic), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def dissection_supercoiling() -> pd.DataFrame:
    """Signal against the no-antibody control, one ablation arm each.

    The stored file's arm names (`z_signal_only`, `z_dosage`, ...) belong to a
    script that is gone, and its grouping was `bin100k`. These arms are defined
    here from the column names and run under random-row CV, so this re-derives
    **the claim** -- that the untagged no-antibody control predicts at least as
    well as the GapR ChIP -- rather than the file row for row.
    """
    from .build_matrix import load_dataset
    from .io_utils import load_block
    sc = load_block("d_supercoiling")
    X, y, names, ids, fam = load_dataset([], verbose=False)
    sc = sc.set_index("sgRNAID").reindex(ids)
    groups = {
        "full family": [c for c in sc.columns],
        "signal tracks only": [c for c in sc.columns if ".signal." in c],
        "control tracks only": [c for c in sc.columns if ".control." in c],
        "rifampicin only": [c for c in sc.columns if ".rifampicin." in c],
        "truncated only": [c for c in sc.columns if ".truncated." in c],
        "torsion only": [c for c in sc.columns if ".torsion." in c],
    }
    base, _ = _cv_rho(X, y, np.arange(X.shape[1]),
                      n_features=config.N_FEATURES, seeds=(41, 42, 43))
    rows = [dict(experiment="baseline", columns=X.shape[1], spearman=base,
                 d_spearman=0.0, note="reference")]
    for arm, cols in groups.items():
        if not cols:
            continue
        M = np.hstack([X, sc[cols].to_numpy(float)])
        rho, _ = _cv_rho(M, y, np.arange(M.shape[1]),
                         n_features=config.N_FEATURES, seeds=(41, 42, 43))
        rows.append(dict(experiment=arm, columns=M.shape[1], spearman=rho,
                         d_spearman=rho - base, note=f"{len(cols)} columns"))
    return pd.DataFrame(rows)


def final_eval_metric_bootstrap() -> pd.DataFrame:
    """Six metrics, paired on guides, from the stored eSpCas9 predictions."""
    from scipy.stats import kendalltau
    from sklearn.metrics import r2_score

    a = pd.read_csv(config.RESULTS / "preds_final_espcas9_slicer.csv")
    b = pd.read_csv(config.RESULTS / "preds_final_espcas9_crisprhal.csv")
    m = a.merge(b, on="protospacer", suffixes=("_s", "_h"))
    y = m["y_s"].to_numpy(float)
    ps, ph = m["pred_s"].to_numpy(float), m["pred_h"].to_numpy(float)

    def metrics(y, p):
        top = int(len(y) * 0.1)
        hit = float(np.isin(np.argsort(p)[::-1][:top],
                            np.argsort(y)[::-1][:top]).mean())
        k = 10
        order = np.argsort(p)[::-1][:k]
        pct = float(np.mean([(y < y[i]).mean() for i in order]) * 100)
        return dict(spearman=spearmanr(y, p).statistic,
                    kendall=kendalltau(y, p).statistic,
                    pearson=float(np.corrcoef(y, p)[0, 1]),
                    r2=r2_score(y, p), top_decile=hit, pick_percentile=pct)

    rng = np.random.default_rng(0)
    names = list(metrics(y, ps))
    draws = {n: [] for n in names}
    for _ in range(4000):
        i = rng.integers(0, len(y), len(y))
        ms, mh = metrics(y[i], ps[i]), metrics(y[i], ph[i])
        for n in names:
            draws[n].append(ms[n] - mh[n])
    obs = {n: metrics(y, ps)[n] - metrics(y, ph)[n] for n in names}
    rows = []
    for n in names:
        lo, hi = np.percentile(draws[n], [2.5, 97.5])
        rows.append(dict(metric=n, diff=obs[n], ci_lo=float(lo),
                         ci_hi=float(hi),
                         excludes_zero=bool(lo > 0 or hi < 0),
                         favours="SLICER" if obs[n] > 0 else "crisprHAL 2"))
    return pd.DataFrame(rows)


def their_split_slicer() -> pd.DataFrame:
    """SLICER on crisprHAL's shipped SpCas9 split, full metric panel."""
    import time

    import xgboost as xgb
    from scipy.stats import kendalltau
    from sklearn.metrics import r2_score

    from .data_ablation import load
    from .run_ablation import _impute, champion_params, make_selector

    X, y, names, protos, in_test, is_noshay, nos, n_base = load(verbose=False)
    te = np.where(in_test)[0]
    tr = np.where(~in_test)[0]
    t0 = time.time()
    A, B = _impute(X[tr], X[te])
    sel = make_selector(41)
    sel.fit(A, y[tr])
    top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
    m = xgb.XGBRegressor(**champion_params(41))
    m.fit(A[:, top], y[tr])
    p = m.predict(B[:, top])
    yt = y[te]
    k, tp = 10, int(len(yt) * 0.1)
    order = np.argsort(p)[::-1]
    return pd.DataFrame([dict(
        model="SLICER, default settings",
        split="crisprHAL 2's shipped split", n_train=len(tr), n=len(te),
        spearman=float(spearmanr(yt, p).statistic),
        kendall_tau=float(kendalltau(yt, p).statistic),
        pearson=float(np.corrcoef(yt, p)[0, 1]), r2=float(r2_score(yt, p)),
        calibration_slope=float(np.polyfit(p, yt, 1)[0]),
        top_decile_prec=float(np.isin(order[:tp],
                                      np.argsort(yt)[::-1][:tp]).mean()),
        pick_percentile=float(np.mean([(yt < yt[i]).mean()
                                       for i in order[:k]]) * 100),
        fit_seconds=round(time.time() - t0, 1))])


def matched_search_folds() -> pd.DataFrame:
    """The per-fold comparison table, assembled from its four sources.

    This one is a join, not a model run: SLICER's tuned and untuned folds from
    the local search, crisprHAL 2 at its published configuration, the same model
    given a single-split search, and the Colab nested search's outer scores.
    """
    t = pd.read_csv(config.RESULTS / "tuning.csv").sort_values("fold")
    hal = pd.read_csv(config.RESULTS / "crisprhal_rerun.csv")
    hal = hal[hal["arm"] == "our_folds"].groupby("fold")["spearman"].mean()
    single = (pd.read_csv(config.RESULTS / "crisprhal_tuned_folds.csv")
              .groupby("fold")["spearman"].max())
    m = pd.read_csv(config.RESULTS / "crisprhal_matched_folds.csv")
    matched = m[m["stage"] == "final"].set_index("fold")["spearman"]
    d = pd.DataFrame({
        "fold": t["fold"].to_numpy(),
        "slicer_tuned": t["rho_tuned"].to_numpy(),
        "slicer_untuned": t["rho_default"].to_numpy(),
        "hal_default": hal.reindex(t["fold"]).to_numpy(),
        "hal_single_split": single.reindex(t["fold"]).to_numpy(),
        "hal_matched": matched.reindex(t["fold"]).to_numpy()})
    d["d_matched"] = d["slicer_tuned"] - d["hal_matched"]
    return d


CHEAP = {
    "grouped_retention": grouped_retention,
    "matched_search_folds": matched_search_folds,
    "d_mechanics_reconstructibility": d_mechanics_reconstructibility,
    "abund_local_vs_mappability": abund_local_vs_mappability,
    "final_eval_metric_bootstrap": final_eval_metric_bootstrap,
}
COSTLY = {
    "selector_control": selector_control,
    "kmer_order_noselect": kmer_order_noselect,
    "portability_364": portability_364,
    "additivity_reduced": additivity_reduced,
    "dissection_supercoiling": dissection_supercoiling,
    "their_split_slicer": their_split_slicer,
}
ALL = {**CHEAP, **COSTLY}


# Columns that are themselves resampling intervals: they are expected to move
# between runs by bootstrap noise, so they are reported but not used to judge.
INTERVAL_COLS = ("ci_lo", "ci_hi", "sd", "spearman_sd", "fit_seconds", "seconds")


def _compare(name: str, new: pd.DataFrame) -> dict:
    """Agreement with the stored file, aligned on a key and scaled by column."""
    old_path = config.RESULTS / f"{name}.csv"
    if not old_path.exists():
        return dict(file=name, status="no stored file", worst_column="",
                    max_rel_diff=np.nan)
    old = pd.read_csv(old_path)
    key = next((c for c in ("family", "arm", "metric", "track", "experiment",
                            "model") if c in old.columns and c in new.columns),
               None)
    if key:
        old = old.drop_duplicates(subset=[key], keep="last").set_index(key)
        new2 = new.drop_duplicates(subset=[key], keep="last").set_index(key)
        shared_rows = [i for i in old.index if i in new2.index]
        old, new2 = old.loc[shared_rows], new2.loc[shared_rows]
    else:
        new2 = new
        if len(old) != len(new2):
            return dict(file=name, status=f"shape {old.shape} vs {new2.shape}",
                        worst_column="", max_rel_diff=np.nan)
        # No key to align on, so row order carries no meaning: compare the
        # sorted distributions, which is what a keyless table actually asserts.
        for c in new2.select_dtypes("number").columns:
            if c in old.columns:
                old[c] = np.sort(pd.to_numeric(old[c], errors="coerce"))
                new2[c] = np.sort(pd.to_numeric(new2[c], errors="coerce"))
    cols = [c for c in new2.select_dtypes("number").columns
            if c in old.columns and c not in INTERVAL_COLS]
    if not cols:
        return dict(file=name, status="no comparable columns", worst_column="",
                    max_rel_diff=np.nan)
    worst, worst_col = 0.0, ""
    for c in cols:
        a = pd.to_numeric(old[c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(new2[c], errors="coerce").to_numpy(float)
        scale = max(np.nanmax(np.abs(a)), 1e-9)
        d = float(np.nanmax(np.abs(a - b)) / scale)
        if d > worst:
            worst, worst_col = d, c
    return dict(file=name,
                status="agrees" if worst < 0.02 else "DIFFERS",
                worst_column=worst_col, max_rel_diff=round(worst, 4))


def compare_only(verbose: bool = True):
    """Re-compare whatever is already in results/rederived/ without recomputing."""
    rows = []
    for name in ALL:
        f = OUTDIR / f"{name}.csv"
        if f.exists():
            rows.append(_compare(name, pd.read_csv(f)))
    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS / "rederivation_check.csv", index=False)
    if verbose:
        print(out.to_string(index=False))
    return out


def run(only=None, verbose: bool = True):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    chosen = {k: v for k, v in ALL.items() if not only or k in only}
    rows = []
    for name, fn in chosen.items():
        if verbose:
            print(f"  {name} ...", flush=True)
        try:
            new = fn()
        except Exception as exc:                       # noqa: BLE001
            rows.append(dict(file=name, status=f"FAILED: {exc}",
                             worst_column="", max_rel_diff=np.nan))
            if verbose:
                print(f"    failed: {exc}", flush=True)
            continue
        new.to_csv(OUTDIR / f"{name}.csv", index=False)
        rows.append(_compare(name, new))
        if verbose:
            print(f"    {rows[-1]['status']} "
                  f"(worst column {rows[-1].get('worst_column') or '-'}, "
                  f"rel {rows[-1].get('max_rel_diff')})", flush=True)
    done = {r["file"] for r in rows}
    for name in ALL:                               # keep earlier verdicts
        f = OUTDIR / f"{name}.csv"
        if name not in done and f.exists():
            rows.append(_compare(name, pd.read_csv(f)))
    out = pd.DataFrame(rows).sort_values("file")
    out.to_csv(config.RESULTS / "rederivation_check.csv", index=False)
    if verbose:
        print(f"\n{out.to_string(index=False)}")
        print(f"written -> {config.RESULTS / 'rederivation_check.csv'}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args(argv)
    if a.compare_only:
        compare_only()
        return 0
    if a.list:
        for k in ALL:
            print(k)
        return 0
    run(only=a.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
