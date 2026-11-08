"""Second, independent cohort for the shape / 3D question: crisprHAL's E. coli eSpCas9 screen.

59,489 guides, a different nuclease on the same genome, base + family A already cached by
`transfer.py`. Flanks are cut from the genome (500 nt each side, on the guide's strand) because the
shipped context is shorter than the window the geometry needs. CV is grouped by 100-kb block (the
cohort has no gene labels in its cache). Arms mirror `shape_ablate` stage 2/3.

    python -m sgrna.ext.esp_replicate
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from sgrna import config, genome
from sgrna.run_ablation import cross_validate

from . import geom
from .common import INTERIM, OUT

SEEDS = (41, 42, 43)
OUTCSV = OUT / "esp_replicate.csv"


def flank_index():
    meta = json.loads((config.INTERIM / "transfer_eSpCas9" / "meta.json").read_text())
    protos = meta["protospacers"]
    seqs, lefts, strands = genome.protospacer_sites()
    site = {s: (l, t) for s, l, t in zip(seqs, lefts, strands)}
    g = genome.load_genome()
    rows, miss = [], 0
    for p in protos:
        if p not in site:
            miss += 1
            rows.append(dict(protospacer=p, upstream="N" * 500, downstream="N" * 500, pam="NGG"))
            continue
        left, st = site[p]
        ctx = genome.guide_context(int(left), int(left) + 19, "+" if st == 1 else "-", 500, g)
        rows.append(dict(protospacer=p, upstream=ctx["upstream"], downstream=ctx["downstream"], pam=ctx["pam"]))
    print(f"  {len(protos):,} guides, {miss} not uniquely located", flush=True)
    return pd.DataFrame(rows), meta


def build(force=False):
    d = INTERIM / "esp"
    d.mkdir(exist_ok=True)
    if not force and (d / "mc_real_geom.npy").exists():
        return
    idx, _ = flank_index()
    for ctl in (None, "arm"):
        r = geom.compute(idx, "mc", control=ctl, verbose=False)
        for blk, (arr, names) in r.items():
            np.save(d / f"mc_{ctl or 'real'}_{blk}.npy", arr)
            (d / f"mc_{ctl or 'real'}_{blk}.json").write_text(json.dumps(names))
        print(f"  built mc {ctl or 'real'}", flush=True)
    arr, nm = geom.composition(idx)
    np.save(d / "comp_real_mean.npy", arr); (d / "comp_real_mean.json").write_text(json.dumps(nm))


def load_block(src, ctl, blk):
    d = INTERIM / "esp"
    return np.load(d / f"{src}_{ctl}_{blk}.npy"), json.loads((d / f"{src}_{ctl}_{blk}.json").read_text())


def main():
    build()
    meta = json.loads((config.INTERIM / "transfer_eSpCas9" / "meta.json").read_text())
    X = np.asarray(np.load(config.INTERIM / "transfer_eSpCas9" / "X.npy", mmap_mode="r"), np.float32)
    y = np.asarray(meta["y"], float); names = list(meta["names"]); nb = int(meta["n_base"])
    left = np.asarray(meta["left"], float)
    ok = np.isfinite(left) & np.isfinite(y)
    groups = (left // 100_000).astype(int).astype(str)
    Xb, XA, nA = X[:, :nb], X[:, nb:], names[nb:]

    def ext(src, ctl, blk):
        a, n = load_block(src, ctl, blk)
        keep = np.isfinite(a).any(0) & (np.nanstd(a, 0) > 0)
        return a[:, keep], [f"{src}.{ctl}.{blk}.{x}" for x, k in zip(n, keep) if k]
    allblk = lambda ctl: [ext("mc", ctl, b) for b in ("mean", "vec", "geom")]
    arms = {"base": [], "A": [(XA, nA)], "mc_all": allblk("real"), "A+mc_all": [(XA, nA)] + allblk("real"),
            "A+mc_geom": [(XA, nA), ext("mc", "real", "geom")],
            "A+mc_armshuf_all": [(XA, nA)] + allblk("arm"), "comp_mean": [ext("comp", "real", "mean")]}
    done = set(pd.read_csv(OUTCSV).arm) if OUTCSV.exists() else set()
    rows = pd.read_csv(OUTCSV).to_dict("records") if OUTCSV.exists() else []
    for arm, parts in arms.items():
        if arm in done:
            continue
        Xs = np.hstack([Xb[ok]] + [p[0][ok] for p in parts])
        nm = names[:nb] + [n for p in parts for n in p[1]]
        t0 = time.time()
        res = cross_validate(Xs, y[ok], nm, ["arm"] * len(nm), n_features=config.N_FEATURES,
                             seeds=list(SEEDS), n_splits=config.N_SPLITS, verbose=False,
                             groups=groups[ok], tag=f"esp_{arm}", experiment=arm)
        folds = pd.read_csv(config.RESULTS / f"folds_esp_{arm}.csv")
        folds = folds[folds.experiment == arm].sort_values(["seed", "fold"])
        rows.append(dict(arm=arm, n_cols=Xs.shape[1], spearman=res["summary"]["spearman"]["mean"],
                         spearman_sd=res["summary"]["spearman"]["sd"],
                         fold_rho=";".join(f"{v:.5f}" for v in folds.spearman)))
        pd.DataFrame(rows).to_csv(OUTCSV, index=False)
        print(f"  {arm:20s} rho {rows[-1]['spearman']:.4f} (sd {rows[-1]['spearman_sd']:.4f})  {time.time() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
