"""Three interpretability tests the report listed as not yet run.

All three use the production recipe -- base + family A on the curated cohort,
gain-ranked top-300 columns, the champion XGBoost -- fitted once on 80% of
guides and probed on held-out ones. None refits the model.

  --sens    Sensitivity-n (Ancona et al. 2018). For random subsets S of n columns,
            does the sum of attributions over S equal what the prediction loses
            when S is replaced by background values? This is the faithfulness
            check that needs no refit, which is what the drop-and-refit probe
            could not give on a redundant matrix. Run at column level and at
            block level, because Finding 5 says blocks are the only unit at
            which attributions are stable. Controls: SHAP columns permuted, and a
            ridge surrogate's attributions.
  --inter   SHAP interaction values on the same model: how much of what the model
            predicts is a sum of single-column effects, and which blocks interact.
  --cf      Counterfactual flank edits: change the DNA >20 nt from the cut
            (raise or lower GC, shuffle the order, swap bases without changing GC),
            recompute family A and ask what the model's prediction does. The
            edits that preserve composition are the controls.

    python -m sgrna.interp_extra --sens --inter --cf
"""

from __future__ import annotations

import argparse
import zlib

import numpy as np
import pandas as pd

from . import config

SEED = 41


def _fit(verbose=True):
    """The production recipe on a random 80% of the curated cohort."""
    import xgboost as xgb

    from .curated_ablation import _base, _block
    from .run_ablation import _impute, champion_params, make_selector

    X, y, names, protos = _base()
    A, an = _block("a_flank", protos)
    X = np.hstack([X, A])
    names = list(names) + list(an)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = np.sort(perm[:cut]), np.sort(perm[cut:])
    Xtr, Xte = _impute(X[tr], X[te])
    sel = make_selector(SEED)
    sel.fit(Xtr, y[tr])
    top = np.argsort(sel.feature_importances_)[::-1][:config.N_FEATURES]
    m = xgb.XGBRegressor(**champion_params(SEED))
    m.fit(Xtr[:, top], y[tr])
    from scipy.stats import spearmanr
    rho = spearmanr(m.predict(Xte[:, top]), y[te]).statistic
    if verbose:
        print(f"  fitted on {len(tr):,}, held-out rho {rho:.4f}", flush=True)
    med = np.nanmedian(X[tr], axis=0)
    return dict(m=m, top=top, names=names, Xtr=Xtr[:, top], Xte=Xte[:, top],
                y=y, tr=tr, te=te, protos=protos, med=med,
                cols=[names[i] for i in top], rho=float(rho), an=list(an),
                nbase=X.shape[1] - A.shape[1])


def sensitivity_n(n_rows=100, n_bg=50, verbose=True):
    import shap
    from sklearn.linear_model import Ridge

    from .importance import block_of

    f = _fit(verbose)
    m, Xtr, Xte = f["m"], f["Xtr"], f["Xte"]
    rng = np.random.default_rng(SEED)
    rows = rng.choice(len(Xte), n_rows, replace=False)
    bg = Xtr[rng.choice(len(Xtr), n_bg, replace=False)]
    X = Xte[rows]
    # path-dependent TreeSHAP, the variant the report's SHAP figures use; the
    # interventional variant refuses this XGBoost build ("categorical split").
    phi = np.asarray(shap.TreeExplainer(m).shap_values(X, check_additivity=False))
    sc = Xtr.std(0); sc[sc == 0] = 1
    rg = Ridge(alpha=10.0).fit((Xtr - Xtr.mean(0)) / sc, f["y"][f["tr"]])
    phi_ridge = rg.coef_ * (X - Xtr.mean(0)) / sc
    phi_perm = np.stack([row[rng.permutation(row.size)] for row in phi])

    blk = np.array([block_of(c) for c in f["cols"]])
    blocks = sorted(set(blk))
    fx = m.predict(X)

    def removed(cols):          # f(x) - E_b f(x with `cols` taken from b)
        out = np.zeros(len(X))
        for b in bg:
            Z = X.copy(); Z[:, cols] = b[cols]
            out += m.predict(Z)
        return fx - out / len(bg)

    def corr(attr, dlt, idx):   # mean over rows of corr across subsets
        a = np.stack([attr[:, c].sum(1) for c in idx], 1)   # rows x subsets
        d = np.stack(dlt, 1)
        r = [np.corrcoef(a[i], d[i])[0, 1] for i in range(len(a))
             if a[i].std() > 0 and d[i].std() > 0]
        return float(np.mean(r)), float(np.std(r) / np.sqrt(len(r)))

    rows_out = []
    plans = [("column", n, [rng.choice(len(f["cols"]), n, replace=False)
                            for _ in range(60)]) for n in (1, 3, 10, 30, 100)]
    for k in (1, 2, 3, 4, 6):
        sub = []
        for _ in range(40):
            bs = rng.choice(len(blocks), k, replace=False)
            sub.append(np.where(np.isin(blk, [blocks[i] for i in bs]))[0])
        plans.append(("block", k, sub))
    for unit, n, idx in plans:
        dlt = [removed(c) for c in idx]
        for name, attr in (("SHAP", phi), ("ridge surrogate", phi_ridge),
                           ("SHAP, columns permuted", phi_perm)):
            r, se = corr(attr, dlt, idx)
            rows_out.append(dict(unit=unit, n=n, attribution=name,
                                 pearson=r, se=se, subsets=len(idx), rows=n_rows))
        if verbose:
            g = [r for r in rows_out if r["unit"] == unit and r["n"] == n]
            print("  " + f"{unit:6s} n={n:<3d} " + "  ".join(
                f"{r['attribution'][:12]} {r['pearson']:+.3f}" for r in g), flush=True)
    out = pd.DataFrame(rows_out)
    out.to_csv(config.RESULTS / "interp_sensitivity_n.csv", index=False)
    return out


def interactions(n_rows=400, verbose=True):
    import shap

    from .importance import block_of

    f = _fit(verbose)
    m, Xte = f["m"], f["Xte"]
    rng = np.random.default_rng(SEED)
    X = Xte[rng.choice(len(Xte), n_rows, replace=False)]
    ex = shap.TreeExplainer(m)
    Phi = np.asarray(ex.shap_interaction_values(X))          # rows x p x p
    main = np.einsum("rii->r", Phi)
    total = Phi.sum((1, 2))
    inter = total - main
    off = np.abs(Phi).sum(0); np.fill_diagonal(off, 0)
    allmass = np.abs(Phi).sum()
    diagmass = np.abs(np.einsum("rii->ri", Phi)).sum()
    blk = np.array([block_of(c) for c in f["cols"]])
    blocks = sorted(set(blk))
    B = np.zeros((len(blocks), len(blocks)))
    ix = {b: np.where(blk == b)[0] for b in blocks}
    for i, bi in enumerate(blocks):
        for j, bj in enumerate(blocks):
            B[i, j] = np.abs(Phi[:, ix[bi]][:, :, ix[bj]]).sum() / len(X)
    pd.DataFrame(B, index=blocks, columns=blocks).to_csv(
        config.RESULTS / "interp_interaction_blocks.csv")
    summ = dict(
        rows=len(X), columns=Phi.shape[1],
        var_total=float(total.var()), var_main=float(main.var()),
        var_interaction=float(inter.var()),
        share_variance_interaction=float(inter.var() / total.var()),
        share_variance_main=float(main.var() / total.var()),
        share_abs_mass_offdiagonal=float(1 - diagmass / allmass),
        corr_main_vs_total=float(np.corrcoef(main, total)[0, 1]),
    )
    pd.DataFrame([summ]).to_csv(config.RESULTS / "interp_interactions.csv", index=False)
    top = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:40], off.shape))[0]
    pairs = [(f["cols"][i], f["cols"][j], float(off[i, j] / len(X)))
             for i, j in top if i < j][:15]
    pd.DataFrame(pairs, columns=["col_i", "col_j", "mean_abs_interaction"]).to_csv(
        config.RESULTS / "interp_interaction_pairs.csv", index=False)
    if verbose:
        for k, v in summ.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        off_b = B.copy(); np.fill_diagonal(off_b, 0)
        k = np.dstack(np.unravel_index(np.argsort(-off_b, axis=None)[:12],
                                       off_b.shape))[0]
        for i, j in k:
            if i < j:
                print(f"  {blocks[i]} x {blocks[j]}: {off_b[i, j]:.4f}")
    return summ


# ---- counterfactual flank edits -------------------------------------------

NEAR = 20      # nucleotides next to the cut left untouched


def _edit(seq: str, kind: str, rng, frac: float = 0.05, side: str = "up") -> str:
    """Edit everything except the NEAR nt adjacent to the protospacer."""
    if side == "up":
        far, near = seq[:-NEAR], seq[-NEAR:]
    else:
        near, far = seq[:NEAR], seq[NEAR:]
    s = list(far)
    n = len(s)
    k = int(round(frac * n))
    if kind in ("gc_up", "gc_down"):
        pool = [i for i, c in enumerate(s) if (c in "AT") == (kind == "gc_up")]
        for i in rng.choice(pool, min(k, len(pool)), replace=False):
            s[i] = rng.choice(list("GC")) if kind == "gc_up" else rng.choice(list("AT"))
    elif kind == "shuffle":
        rng.shuffle(s)
    elif kind == "swap_same_gc":                  # A<->T, G<->C: GC unchanged
        comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
        for i in rng.choice(n, k, replace=False):
            s[i] = comp.get(s[i], s[i])
    far = "".join(s)
    return far + near if side == "up" else near + far


def counterfactual(n_guides=1500, verbose=True):
    from .build_curated_features import load_index
    from .features import a_flank

    f = _fit(verbose)
    m = f["m"]
    protos, te = f["protos"], f["te"]
    rng = np.random.default_rng(SEED)
    pick = rng.choice(te, min(n_guides, len(te)), replace=False)
    idx = load_index(verbose=False).drop_duplicates("protospacer") \
        .set_index("protospacer").reindex(protos[pick]).reset_index()
    idx = idx.rename(columns={"index": "protospacer"})
    idx[config.ID_COL] = idx["protospacer"]
    idx["located"] = True

    an = f["an"]
    pos_in_top = {c: i for i, c in enumerate(f["cols"])}
    a_cols = [c for c in an if c in pos_in_top]

    def predict(index):
        blk = a_flank.build(index).set_index(config.ID_COL).reindex(index[config.ID_COL])
        # unedited base columns come from the stored matrix; family A from the rebuild
        Xfull = np.asarray(np.load(config.INTERIM / "expanded_X.npy", mmap_mode="r"),
                           dtype=np.float32)[pick]
        sel = [i for i in f["top"] if i < f["nbase"]]
        Xb = np.full((len(index), len(f["cols"])), np.nan, dtype=np.float32)
        for j, i in enumerate(f["top"]):
            if i < f["nbase"]:
                Xb[:, j] = Xfull[:, i]
        for c in a_cols:
            Xb[:, pos_in_top[c]] = blk[c].to_numpy(dtype=np.float32)
        med = np.where(np.isnan(f["med"][f["top"]]), 0.0, f["med"][f["top"]])
        r, c = np.where(np.isnan(Xb)); Xb[r, c] = med[c]
        return m.predict(Xb)

    p0 = predict(idx)
    rebuilt_ok = float(np.mean(np.abs(p0 - m.predict(f["Xte"][np.searchsorted(te, pick)]))))
    if verbose:
        print(f"  unedited rebuild vs stored matrix: mean |diff| {rebuilt_ok:.5f}",
              flush=True)

    rows = []
    for name, kind, frac in (("GC +5 points, both flanks", "gc_up", 0.05),
                             ("GC +10 points, both flanks", "gc_up", 0.10),
                             ("GC -5 points, both flanks", "gc_down", 0.05),
                             ("GC -10 points, both flanks", "gc_down", 0.10),
                             ("order shuffled, composition kept", "shuffle", 0.0),
                             ("5% of bases swapped, GC kept", "swap_same_gc", 0.05),
                             ("20% of bases swapped, GC kept", "swap_same_gc", 0.20)):
        seed = zlib.crc32(name.encode())
        r2 = np.random.default_rng(seed)
        e = idx.copy()
        e["upstream"] = [_edit(s, kind, r2, frac, "up") for s in idx["upstream"]]
        e["downstream"] = [_edit(s, kind, r2, frac, "down") for s in idx["downstream"]]
        d = predict(e) - p0
        bs = np.random.default_rng(1).choice(len(d), (2000, len(d)))
        means = d[bs].mean(1)
        rows.append(dict(edit=name, mean_change=float(d.mean()),
                         ci_lo=float(np.percentile(means, 2.5)),
                         ci_hi=float(np.percentile(means, 97.5)),
                         sd_across_guides=float(d.std()), guides=len(d),
                         label_sd=float(f["y"].std())))
        if verbose:
            print(f"  {name:36s} {d.mean():+.4f}  [{rows[-1]['ci_lo']:+.4f}, "
                  f"{rows[-1]['ci_hi']:+.4f}]  sd {d.std():.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS / "interp_counterfactual.csv", index=False)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sens", action="store_true")
    ap.add_argument("--inter", action="store_true")
    ap.add_argument("--cf", action="store_true")
    a = ap.parse_args(argv)
    if a.sens:
        sensitivity_n()
    if a.inter:
        interactions()
    if a.cf:
        counterfactual()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
