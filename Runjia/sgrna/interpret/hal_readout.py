"""crisprHAL 2's side of the interpretability comparison (`vs_crisprhal_interpret.py` has the design and the predictions).

Runs in the TensorFlow environment, beside crisprHAL's own code, and writes small files that the main environment reads:

    .venv-crisprhal/bin/python sgrna/interpret/hal_readout.py <project root> shipped          # shipped WT weights: readings, sensitivity-n, flank edits
    .venv-crisprhal/bin/python sgrna/interpret/hal_readout.py <project root> refit <tag>      # refit by their code and 48 epochs, same readings
          tags: full1 full2 (all training rows, other seeds); halfA1 halfB1 halfA2 halfB2 (disjoint random halves of the training rows)

What a reading is (all on the same 1,000 test guides, drawn once):
  eg_abs_channel   mean |expected-gradient attribution| of each of the 378 x 4 input channels (Erion et al. 2021; 50 draws of
                   (background guide, interpolation point) per guide, 100 background contexts from the training file)
  eg_pos_base      the attribution of the base a guide carries at each of the 20 positions, averaged over the guides that carry it
  ism_delta        the prediction change when one guide base is set to each other base (in-silico substitution), guide x position x base
The 378 nt are split into 25 blocks: the 20 guide positions, the PAM, the 20 nt before the protospacer, the rest upstream, the 17 nt after
the PAM, the rest downstream. The flank edits keep the three blocks beside the cut and rewrite the two far ones, as `interp_extra._edit` does.
"""
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

root, step = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else None
HAL = os.path.join(root, "external_data", "crisprHAL")
CACHE = os.environ.get("HAL_CACHE") or os.path.join(root, "RSI-Crispr-fork", "Runjia", "results", "post_freeze", "interp_hal_cache")
os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, HAL)
os.chdir(HAL)
import tensorflow as tf                                    # noqa: E402
from models import models                                  # noqa: E402
from processing import processing                          # noqa: E402

UP, L = 189, 378
PROC = processing()
BASES = "ACGT"
BLOCKS = {f"g{p + 1}": np.arange(UP + p, UP + p + 1) for p in range(20)}
BLOCKS.update({"PAM": np.arange(209, 212), "up_near": np.arange(169, 189), "up_far": np.arange(0, 169),
               "dn_near": np.arange(212, 229), "dn_far": np.arange(229, 378)})
assert sum(len(v) for v in BLOCKS.values()) == L and len(BLOCKS) == 25


def _read(split):
    d = pd.read_csv(f"data/WT-SpCas9_{split}ing_data.csv", header=None, names=["context", "y"]).dropna(subset=["context"])
    d["context"] = d["context"].str.upper()
    assert (d["context"].str.len() == L).all()
    return d


def _enc(ctx):
    return PROC.onehotencode(list(ctx)).astype(np.float32)


def data():
    train, test = _read("train"), _read("test")
    test["protospacer"] = test["context"].str[UP:UP + 20]
    test = test[~test["protospacer"].str.contains("N")]
    test = test.groupby("protospacer", as_index=False).agg(context=("context", "first"), y=("y", "mean")).sort_values("protospacer").reset_index(drop=True)
    perm = np.random.default_rng(0).permutation(len(test))
    return train.reset_index(drop=True), test, perm


def _predict(model, X):
    return np.asarray(model.predict(X, batch_size=2048, verbose=0)).ravel()


def expected_gradients(model, X, bg, n_samples=50, seed=0, chunk=250):
    """Attribution of every input channel: E over (background b, alpha ~ U(0,1)) of (x - b) * d f(b + alpha (x - b)) / dx."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(X, np.float32)
    for lo in range(0, len(X), chunk):
        x = X[lo:lo + chunk]
        acc = np.zeros_like(x)
        for _ in range(n_samples):
            b = bg[rng.integers(0, len(bg), len(x))]
            a = rng.random((len(x), 1, 1)).astype(np.float32)
            z = tf.convert_to_tensor(b + a * (x - b))
            with tf.GradientTape() as t:
                t.watch(z)
                y = model(z, training=False)
            acc += (x - b) * t.gradient(y, z).numpy()
        out[lo:lo + chunk] = acc / n_samples
    return out


def readings(model, X, ctx, bg, tagname):
    """The three readings for one fitted model on the 1,000 guides, saved as one small file."""
    t0 = time.time()
    f0 = _predict(model, X)
    egs = [expected_gradients(model, X, bg, 50, seed=s) for s in ((0, 1) if tagname == "shipped" else (0,))]   # shipped: two Monte Carlo draws, the estimator's own noise floor
    if len(egs) == 1:
        egs.append(np.full_like(egs[0], np.nan))
    print(f"  {tagname}: expected gradients {time.time() - t0:.0f} s", flush=True)
    eg = egs[0]
    # completeness: sum of attributions vs f(x) - E_b f(b)
    comp_err = float(np.mean(np.abs(eg.sum(axis=(1, 2)) - (f0 - _predict(model, bg).mean()))))
    gi = np.stack([np.argmax(X[:, UP + p, :], axis=1) for p in range(20)], 1)  # base index at each position (0 where N)
    pos_attr = np.stack([eg[:, UP + p, :].sum(axis=1) for p in range(20)], 1)  # guides x 20
    pos_base = np.full((20, 4), np.nan)
    for p in range(20):
        for b in range(4):
            sel = gi[:, p] == b
            if sel.sum() >= 20:
                pos_base[p, b] = pos_attr[sel, p].mean()
    # in-silico substitution of every guide base
    delta = np.zeros((len(X), 20, 4), np.float32)
    for p in range(20):
        for b in range(4):
            Z = X.copy()
            Z[:, UP + p, :] = 0
            Z[:, UP + p, b] = 1
            delta[:, p, b] = _predict(model, Z) - f0
    ism_map = np.zeros((20, 4))                                             # mean over all guides of f(x with base b at p) - f(x)
    for p in range(20):
        ism_map[p] = delta[:, p, :].mean(axis=0)
    np.savez_compressed(os.path.join(CACHE, f"hal_{tagname}.npz"), eg_abs_channel=np.abs(eg).mean(0), eg_abs_channel_mc2=np.abs(egs[1]).mean(0),
                        eg_pos_base=pos_base, eg_pos_base_mc2=_pos_base(egs[1], X), ism_delta=delta, ism_map=ism_map, f0=f0, completeness_error=comp_err,
                        eg_blocks=np.stack([eg[:, v, :].sum(axis=(1, 2)) for v in BLOCKS.values()], 1), guide_base=gi)
    print(f"  {tagname}: saved readings ({time.time() - t0:.0f} s)", flush=True)
    return eg


def _pos_base(eg, X):
    gi = np.stack([np.argmax(X[:, UP + p, :], axis=1) for p in range(20)], 1)
    pa = np.stack([eg[:, UP + p, :].sum(axis=1) for p in range(20)], 1)
    out = np.full((20, 4), np.nan)
    for p in range(20):
        for b in range(4):
            s = gi[:, p] == b
            if s.sum() >= 20:
                out[p, b] = pa[s, p].mean()
    return out


def sensitivity_n(model, X, eg, bgX, n_rows=100, n_bg=50, seed=0):
    """Sensitivity-n at block level, as `interp_extra.sensitivity_n` does for SLICER (same subsets sizes, 40 subsets each)."""
    rng = np.random.default_rng(seed)
    rows = np.arange(n_rows)                                                   # the first guides of the shared permutation
    names = list(BLOCKS)
    Xr = X[rows]
    fx = _predict(model, Xr)
    bgs = bgX[rng.choice(len(bgX), n_bg, replace=False)]
    battr = np.stack([eg[rows][:, BLOCKS[n], :].sum(axis=(1, 2)) for n in names], 1)
    battr_perm = np.stack([r[rng.permutation(r.size)] for r in battr])
    out = []
    for k in (1, 2, 3, 4, 6):
        subsets = [rng.choice(len(names), k, replace=False) for _ in range(40)]
        dlt = np.zeros((len(rows), len(subsets)))
        for j, s in enumerate(subsets):
            pos = np.concatenate([BLOCKS[names[i]] for i in s])
            Z = np.repeat(Xr[:, None], n_bg, 1)
            Z[:, :, pos, :] = bgs[None, :, pos, :]
            dlt[:, j] = fx - _predict(model, Z.reshape(-1, L, 4)).reshape(len(rows), n_bg).mean(1)
        for lab, attr in (("expected gradients", battr), ("expected gradients, blocks shuffled", battr_perm)):
            a = np.stack([attr[:, s].sum(1) for s in subsets], 1)
            r = [np.corrcoef(a[i], dlt[i])[0, 1] for i in range(len(a)) if a[i].std() > 0 and dlt[i].std() > 0]
            out.append(dict(unit="block", n=k, attribution=lab, pearson=float(np.mean(r)), se=float(np.std(r) / np.sqrt(len(r))), subsets=len(subsets), rows=len(rows)))
        print("  sens-n blocks k=%d  " % k + "  ".join(f"{o['attribution'][:24]} {o['pearson']:+.3f}" for o in out[-2:]), flush=True)
    return pd.DataFrame(out)


def eg_draws_control(model, X, bgX, n_rows=100, n_bg=50, seed=0, draws=(50, 200, 800)):
    """Is the gap in sensitivity-n the estimator's Monte Carlo noise? Same 100 guides, same ablations, more expected-gradient draws."""
    rng = np.random.default_rng(seed)
    names = list(BLOCKS)
    rows = np.arange(n_rows)
    Xr = X[rows]
    fx = _predict(model, Xr)
    bgs = bgX[rng.choice(len(bgX), n_bg, replace=False)]
    egs = {n: expected_gradients(model, Xr, bgX, n, seed=7) for n in draws}
    out = []
    for k in (1, 3, 6):
        subsets = [rng.choice(len(names), k, replace=False) for _ in range(40)]
        dlt = np.zeros((len(rows), len(subsets)))
        for j, s in enumerate(subsets):
            pos = np.concatenate([BLOCKS[names[i]] for i in s])
            Z = np.repeat(Xr[:, None], n_bg, 1)
            Z[:, :, pos, :] = bgs[None, :, pos, :]
            dlt[:, j] = fx - _predict(model, Z.reshape(-1, L, 4)).reshape(len(rows), n_bg).mean(1)
        for n, eg in egs.items():
            battr = np.stack([eg[:, BLOCKS[nm], :].sum(axis=(1, 2)) for nm in names], 1)
            a = np.stack([battr[:, s].sum(1) for s in subsets], 1)
            r = [np.corrcoef(a[i], dlt[i])[0, 1] for i in range(len(a)) if a[i].std() > 0 and dlt[i].std() > 0]
            out.append(dict(n_blocks_removed=k, draws=n, pearson=float(np.mean(r)), se=float(np.std(r) / np.sqrt(len(r)))))
            print(f"  k={k} draws={n}: sensitivity-n {out[-1]['pearson']:+.3f}", flush=True)
    return pd.DataFrame(out)


def _edit(seq, kind, rng, frac, side):
    """interp_extra._edit: rewrite everything except the 20 nt beside the cut."""
    far, near = (seq[:-20], seq[-20:]) if side == "up" else (seq[20:], seq[:20])
    s = list(far)
    n = len(s)
    k = int(round(frac * n))
    if kind in ("gc_up", "gc_down"):
        pool = [i for i, c in enumerate(s) if (c in "AT") == (kind == "gc_up")]
        for i in rng.choice(pool, min(k, len(pool)), replace=False):
            s[i] = rng.choice(list("GC")) if kind == "gc_up" else rng.choice(list("AT"))
    elif kind == "shuffle":
        rng.shuffle(s)
    elif kind == "swap_same_gc":
        comp = {"A": "T", "T": "A", "G": "C", "C": "G"}
        for i in rng.choice(n, k, replace=False):
            s[i] = comp.get(s[i], s[i])
    far = "".join(s)
    return far + near if side == "up" else near + far


def flank_edits(model, ctx, y_sd):
    base = _predict(model, _enc(ctx))
    rows = []
    for name, kind, frac in (("GC +5 points, both flanks", "gc_up", 0.05), ("GC +10 points, both flanks", "gc_up", 0.10),
                             ("GC -5 points, both flanks", "gc_down", 0.05), ("GC -10 points, both flanks", "gc_down", 0.10),
                             ("order shuffled, composition kept", "shuffle", 0.0), ("5% of bases swapped, GC kept", "swap_same_gc", 0.05),
                             ("20% of bases swapped, GC kept", "swap_same_gc", 0.20)):
        r2 = np.random.default_rng(zlib.crc32(name.encode()))
        new = [_edit(c[:UP], kind, r2, frac, "up") + c[UP:UP + 20] + _edit(c[UP + 20:], kind, r2, frac, "dn") for c in ctx]
        assert all(len(c) == L for c in new)
        d = _predict(model, _enc(new)) - base
        means = d[np.random.default_rng(1).choice(len(d), (2000, len(d)))].mean(1)
        rows.append(dict(edit=name, mean_change=float(d.mean()), ci_lo=float(np.percentile(means, 2.5)), ci_hi=float(np.percentile(means, 97.5)),
                         sd_across_guides=float(d.std()), guides=len(d), label_sd=y_sd))
        print(f"  {name:36s} {d.mean():+.4f}  [{rows[-1]['ci_lo']:+.4f}, {rows[-1]['ci_hi']:+.4f}]", flush=True)
    return pd.DataFrame(rows)


def main():
    train, test, perm = data()
    sel1000, sel1500 = perm[:1000], perm[:1500]
    pd.DataFrame({"protospacer": test.loc[sel1500, "protospacer"].to_numpy(), "y": test.loc[sel1500, "y"].to_numpy()}).to_csv(os.path.join(CACHE, "guides_1500.csv"), index=False)
    bg = _enc(train["context"].iloc[np.random.default_rng(1).choice(len(train), 100, replace=False)])
    X = _enc(test.loc[sel1000, "context"])
    Xtest = _enc(test["context"])
    from scipy.stats import spearmanr
    if step == "shipped":
        import keras
        model = keras.models.load_model("models/WT-SPCAS9.keras", compile=False)
        print("  shipped weights, test rho", round(float(spearmanr(test["y"], _predict(model, Xtest)).statistic), 4), flush=True)
        eg = readings(model, X, test.loc[sel1000, "context"], bg, "shipped")
        sensitivity_n(model, X, eg, bg).to_csv(os.path.join(CACHE, "hal_sensitivity_n.csv"), index=False)
        flank_edits(model, test.loc[sel1500, "context"].tolist(), float(train["y"].std())).to_csv(os.path.join(CACHE, "hal_flank_edits.csv"), index=False)
        return
    if step == "egdraws":
        import keras
        model = keras.models.load_model("models/WT-SPCAS9.keras", compile=False)
        eg_draws_control(model, X, bg).to_csv(os.path.join(CACHE, "hal_eg_draws_control.csv"), index=False)
        return
    assert step == "refit" and tag
    if os.path.exists(os.path.join(CACHE, f"hal_{tag}.npz")):
        print("  already done:", tag)
        return
    n = len(train)
    if tag.startswith("full"):
        idx, seed = np.arange(n), 100 + int(tag[4:])
    else:
        part, k = tag[4], int(tag[5:])
        p = np.random.default_rng(10 + k).permutation(n)
        idx, seed = (p[: n // 2] if part == "A" else p[n // 2:]), 110 + 10 * k + (0 if part == "A" else 1)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    m = models("WT-SPCAS9", summary=False)
    t0 = time.time()
    m.train(_enc(train["context"].iloc[idx]), train["y"].to_numpy()[idx], epochs=48, batch_size=1024, verbose=0)
    rho = float(spearmanr(test["y"], _predict(m.model, Xtest)).statistic)
    print(f"  {tag}: {len(idx):,} training rows, 48 epochs, {time.time() - t0:.0f} s, test rho {rho:.4f}", flush=True)
    readings(m.model, X, test.loc[sel1000, "context"], bg, tag)
    pd.DataFrame([dict(tag=tag, n_train=len(idx), fit_seconds=time.time() - t0, test_rho=rho, seed=seed)]).to_csv(os.path.join(CACHE, f"hal_{tag}_fit.csv"), index=False)


main()
