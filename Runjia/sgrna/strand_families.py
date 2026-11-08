"""With the strand call supplied, how do all eleven families compare on the dCas9 screen?

Family A's +0.27 on Cui's essential-gene guides was the dCas9 strand rule arriving
through `eng.flank.comp.up1000.purine`, a lossy copy of "which strand the guide
targets" (Finding 3). The control run then was family A against the strand call.
It leaves a fair question open: *is A still the strongest family once the
strand call is given to everyone, or was every family measured against a
baseline that lacked it?*

Design, on the 5,514 essential-gene guides, 100 kb-block and gene grouping, three
seeds, selection fitted inside each training fold (the pipeline's own CV):

  base + strand                      the new baseline, strand supplied outright
  base + strand + one family         for each of the eleven families
  base + one family (no strand)      the same eleven, for the comparison

The increment of a family is its arm minus `base + strand`. All eleven families
are built here for Cui's guides, because only A existed for this screen.

    python -m sgrna.strand_families --build
    python -m sgrna.strand_families --run
"""

from __future__ import annotations

import argparse
import json
import time
import traceback

import numpy as np
import pandas as pd

from . import config, genome
from .features import ALL, REGISTRY

OUT_DIR = config.INTERIM / "cui_crispri" / "families"
OUT = config.RESULTS / "strand_families.csv"
FLANK = 1000


def index() -> pd.DataFrame:
    """Cui's essential-gene guides as the family builders expect them."""
    from .build_curated_features import augment
    from .cui_screen import _essential_frame  # noqa: F401  (ensures cache exists)

    meta = json.loads((config.INTERIM / "cui_crispri" / "meta.json").read_text())
    ess = np.asarray([str(e) for e in meta["essential"]], dtype=object) == "True"
    gene = np.asarray([g if isinstance(g, str) else "" for g in meta["gene"]],
                      dtype=object)
    has_gene = (gene != "") & (gene != "nan") & (gene != "None")
    sel = np.where(ess & has_gene)[0]

    seqs, lefts, strands = genome.protospacer_sites()
    site = {s: (l, t) for s, l, t in zip(seqs, lefts, strands)}
    g = genome.load_genome()
    rows = []
    for i in sel:
        p = meta["protospacers"][i]
        left, st = site[p]
        ctx = genome.guide_context(int(left), int(left) + 19,
                                   "+" if st == 1 else "-", FLANK, g)
        rows.append(dict(**{config.ID_COL: p}, protospacer=p, located=True,
                         n_genome_copies=1, strand="+" if st == 1 else "-",
                         left=int(left), right=int(left) + 19, pam=ctx["pam"],
                         upstream=ctx["upstream"], downstream=ctx["downstream"],
                         score=meta["y"][i], gene_name=gene[i]))
    return augment(pd.DataFrame(rows), verbose=False)


def build(families=ALL, verbose: bool = True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = index()
    if verbose:
        print(f"  {len(idx):,} essential-gene guides")
    for fam in families:
        path = OUT_DIR / f"{fam}.csv"
        if path.exists():
            continue
        t0 = time.time()
        try:
            blk = REGISTRY[fam].build(idx)
            feats = [c for c in blk.columns if c != config.ID_COL]
            filled = int(blk[feats].notna().any(axis=1).sum()) if feats else 0
            if not feats or not filled:
                raise RuntimeError("empty block")
            blk.to_csv(path, index=False)
            print(f"  {fam}: {len(feats)} columns, {filled:,} populated, "
                  f"{time.time() - t0:.0f}s", flush=True)
        except Exception as exc:
            print(f"  {fam}: FAILED {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=2)


def _blocks(protos):
    """Every built family, aligned to the essential frame's guide order."""
    out = {}
    for fam in ALL:
        path = OUT_DIR / f"{fam}.csv"
        if not path.exists():
            continue
        b = pd.read_csv(path).set_index(config.ID_COL).reindex(protos)
        v = b.to_numpy(dtype=np.float32)
        keep = ~np.all(np.isnan(v), axis=0)
        out[fam] = (v[:, keep], [c for c, k in zip(b.columns, keep) if k])
    return out


PROXY = 0.30      # |rho| with the strand call above which a column is a proxy


def run(seeds=(41, 42, 43), n_splits=None, verbose: bool = True):
    """Three arms per family: with strand, without, and without its proxy columns."""
    from scipy.stats import spearmanr

    from .cui_screen import _essential_frame
    from .run_ablation import cross_validate

    f = _essential_frame()
    meta = json.loads((config.INTERIM / "cui_crispri" / "meta.json").read_text())
    ess = np.asarray([str(e) for e in meta["essential"]], dtype=object) == "True"
    gene = np.asarray([g if isinstance(g, str) else "" for g in meta["gene"]],
                      dtype=object)
    has_gene = (gene != "") & (gene != "nan") & (gene != "None")
    protos = np.asarray(meta["protospacers"])[ess & has_gene]
    assert len(protos) == len(f["y"]), (len(protos), len(f["y"]))
    blocks = _blocks(protos)
    strand = f["coding"]
    cod = strand.ravel()

    rows, proxy_rows = [], []
    for gname, grouping in f["groupings"]:
        def cv(Xa, nm, arm):
            res = cross_validate(
                Xa, f["y"], nm, ["arm"] * len(nm), n_features=config.N_FEATURES,
                seeds=list(seeds), n_splits=n_splits or config.N_SPLITS,
                verbose=False, groups=grouping,
                tag=f"cui_sf_{gname.replace(' ', '')}_{arm}", experiment=arm)
            return (res["summary"]["spearman"]["mean"],
                    res["summary"]["spearman"]["sd"])

        base_names = f["nb"]
        arms = {"base": (f["Xb"], base_names),
                "base+strand": (np.hstack([f["Xb"], strand]),
                                base_names + ["screen.coding_strand"])}
        for fam, (v, fn) in blocks.items():
            # each column's association with the strand call, over guides that have one
            ok = cod >= 0
            r = np.array([abs(spearmanr(v[ok, j], cod[ok], nan_policy="omit").statistic)
                          if np.nanstd(v[ok, j]) > 0 else 0.0
                          for j in range(v.shape[1])])
            r = np.nan_to_num(r)
            drop = r > PROXY
            if gname == "genes":
                proxy_rows.append(dict(family=fam, n_cols=v.shape[1],
                                       max_abs_rho_with_strand=float(r.max()),
                                       n_proxy_cols=int(drop.sum()),
                                       strongest=fn[int(r.argmax())]))
            arms[f"{fam}"] = (np.hstack([f["Xb"], v]), base_names + fn)
            arms[f"{fam}+strand"] = (np.hstack([f["Xb"], strand, v]),
                                     base_names + ["screen.coding_strand"] + fn)
            if drop.any():
                keep = ~drop
                arms[f"{fam}-proxy"] = (np.hstack([f["Xb"], v[:, keep]]),
                                        base_names + [c for c, k in zip(fn, keep) if k])
        for arm, (Xa, nm) in arms.items():
            m, sd = cv(Xa, nm, arm.replace("+", "_"))
            rows.append(dict(grouping=gname, arm=arm, spearman=m, spearman_sd=sd))
            if verbose:
                print(f"  {gname:12s} {arm:28s}: rho {m:+.4f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(proxy_rows).to_csv(config.RESULTS / "strand_families_proxies.csv",
                                    index=False)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args(argv)
    if a.build:
        build()
    if a.run:
        run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
