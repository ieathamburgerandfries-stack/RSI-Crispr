"""Finding 8 on human: do the shape / 3D descriptors add to family A on the human cutting matrix?

Cohort: the guides of the Noshay et al. 2023 human matrix that locate to a single NGG-flanked copy on hg38
(`human_flank.py`). Representation, selector, model and folds are those of the E. coli runs: the regenerated base
(everything computable from the 20-mer), family A (`a_flank`, built from the 1,000 nt either side of the cut), top-300
selection inside folds, the frozen XGBoost champion, 5 folds x 3 seeds. The human matrix has no gene identifier, so
folds are random over guides (as in the project's human ladder), which is the more permissive design; the comparison
between arms is paired on identical folds, so it is not affected by that choice, but absolute rho values are not
comparable with the gene-grouped E. coli numbers.

    python -m sgrna.ext.human_shape --stage 1     # base, base+A, and shape on top of A
    python -m sgrna.ext.human_shape --stage 2     # shape without A, composition references
    python -m sgrna.ext.human_shape --stage 3     # shuffled-flank controls
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from sgrna import config, featurise
from sgrna.features import a_flank
from sgrna.run_ablation import cross_validate

from . import geom
from .common import INTERIM, OUT
from .human_flank import DIR as HDIR, locate

SEEDS = (41, 42, 43)
OUTCSV = OUT / "human_shape_ablation.csv"
OUTCSV_G = OUT / "human_shape_ablation_locus.csv"
GAP = 10_000            # guides closer than this on a chromosome belong to one locus (gene-scale)


def locus_groups(loc: pd.DataFrame, gap: int = GAP) -> np.ndarray:
    """Single-linkage clusters of guides along each chromosome: the human analogue of the gene groups used for E. coli."""
    g = np.zeros(len(loc), int)
    gid, pc, pl = 0, None, None
    for i in loc.sort_values(["chrom", "left"]).index:
        c, l = loc.chrom[i], loc.left[i]
        if pc is None or c != pc or l - pl > gap:
            gid += 1
        g[i] = gid
        pc, pl = c, l
    return g
BLK = INTERIM / "shape_human"
BLK.mkdir(parents=True, exist_ok=True)


class HumanCohort:
    def __init__(self):
        loc = locate()
        loc = loc[np.isfinite(loc.y)].reset_index(drop=True)
        self.idx = loc.assign(**{config.ID_COL: loc.sgRNAID, "located": True})      # a_flank.build skips rows without it
        self.y = loc.y.to_numpy(float)
        self.protos = loc.protospacer.tolist()
        self.groups = None                           # random folds (the project's human ladder)
        self.locus = locus_groups(loc)               # locus-grouped folds
        self.twin = pd.factorize(loc.protospacer)[0]  # identical protospacers (repeat measurements) kept together
        xb = HDIR / "X_base.npy"
        if xb.exists():
            self.X = np.load(xb); self.names = json.loads((HDIR / "base_names.json").read_text())
        else:
            X, names = featurise.matrix(self.protos, verbose=False)
            self.X = np.asarray(X, np.float32); self.names = list(names)
            np.save(xb, self.X); (HDIR / "base_names.json").write_text(json.dumps(self.names))
        self._A = None

    def A(self):
        if self._A is None:
            pa = HDIR / "a_flank.csv"
            if pa.exists():
                blk = pd.read_csv(pa)
            else:
                blk = a_flank.build(self.idx); blk.to_csv(pa, index=False)
            blk = blk.set_index(config.ID_COL).reindex(self.idx[config.ID_COL])
            v = blk.to_numpy(np.float32)
            good = ~np.all(np.isnan(v), axis=0)
            assert good.sum() > 100, f"family A came out empty ({good.sum()} columns)"
            self._A = (v[:, good], [c for c, k in zip(blk.columns, good) if k])
        return self._A

    def ext(self, source, control, block):
        key = BLK / f"{source}_{control or 'real'}_{block}"
        if not key.with_suffix(".npy").exists():
            self._build(source, control)
        arr, nm = np.load(key.with_suffix(".npy")), json.loads(key.with_suffix(".json").read_text())
        keep = np.isfinite(arr).any(0) & (np.nanstd(arr, 0) > 0)
        return arr[:, keep], [f"{source}.{control or 'real'}.{block}.{n}" for n, k in zip(nm, keep) if k]

    def _build(self, source, control):
        if source == "comp":
            r = {"mean": geom.composition(self.idx, control=control)}
        elif source == "dcomp":
            r = {"mean": geom.dinuc_composition(self.idx, control=control)}
        else:
            r = geom.compute(self.idx, source, control=control)
        for blk, (arr, names) in r.items():
            key = BLK / f"{source}_{control or 'real'}_{blk}"
            np.save(key.with_suffix(".npy"), arr); key.with_suffix(".json").write_text(json.dumps(names))


def run_arm(co: HumanCohort, label: str, parts: list, seeds=SEEDS, grouped=False, n_features=None):
    # grouped: False (random folds) | True or 'locus' (locus clusters) | 'twin' (identical protospacers together)
    Xs, names, fams = [co.X], list(co.names), ["base"] * len(co.names)
    for arr, nm, fam in parts:
        Xs.append(arr); names += nm; fams += [fam] * len(nm)
    X = np.hstack(Xs)
    kind = "locus" if grouped is True else grouped
    tag = f"hshp{'g' if kind == 'locus' else ('t' if kind == 'twin' else '')}{'_n' + str(n_features) if n_features else ''}_{label}"
    res = cross_validate(X, co.y, names, fams, n_features=n_features or config.N_FEATURES, seeds=list(seeds),
                         n_splits=config.N_SPLITS, verbose=False, groups={'locus': co.locus, 'twin': co.twin}.get(kind), tag=tag,
                         experiment=label)
    folds = pd.read_csv(config.RESULTS / f"folds_{tag}.csv")
    folds = folds[folds.experiment == label].sort_values(["seed", "fold"])
    return dict(arm=label, n=len(co.y), n_cols=X.shape[1], n_ext=sum(a.shape[1] for a, _, _ in parts),
                spearman=res["summary"]["spearman"]["mean"], spearman_sd=res["summary"]["spearman"]["sd"],
                r2=res["summary"]["r2"]["mean"], n_selected_ext=float(folds.n_engineered_selected.mean()),
                fold_rho=";".join(f"{v:.5f}" for v in folds.spearman))


def append(row, path=OUTCSV):
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df = pd.concat([df[df.arm != row["arm"]], pd.DataFrame([row])], ignore_index=True) if len(df) else pd.DataFrame([row])
    df.to_csv(path, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--grouped", action="store_true", help="locus-grouped folds instead of random folds")
    ap.add_argument("--twin", action="store_true", help="folds that keep identical protospacers together")
    ap.add_argument("--nfeat", type=int, default=None, help="selector budget (default: the project's 300)")
    a = ap.parse_args(argv)
    co = HumanCohort()
    path = OUTCSV_G if a.grouped else OUTCSV
    if a.twin:
        path = OUT / "human_shape_ablation_twin.csv"
    if a.nfeat:
        path = path.with_name(path.stem + f"_n{a.nfeat}.csv")
    done = set(pd.read_csv(path).arm) if path.exists() else set()
    A_, An = co.A()
    A = [(A_, An, "a_flank")]

    def blocks(src, ctl=None, which=("mean", "vec", "geom")):
        return [(*co.ext(src, ctl, b), f"ext_{src}") for b in which]

    arms: dict[str, list] = {}
    if a.stage == 1:
        arms["h_base"] = []
        arms["h_base+A"] = A
        for src in ("mc", "xrc", "md"):
            arms[f"h_base+A+{src}"] = A + blocks(src)
        arms["h_base+A+mc+xrc+md"] = A + sum((blocks(s) for s in ("mc", "xrc", "md")), [])
    elif a.stage == 2:
        arms["h_base+comp"] = [(*co.ext("comp", None, "mean"), "ext_comp")]
        arms["h_base+dcomp"] = [(*co.ext("dcomp", None, "mean"), "ext_dcomp")]
        arms["h_base+A+comp"] = A + [(*co.ext("comp", None, "mean"), "ext_comp")]
        arms["h_base+A+dcomp"] = A + [(*co.ext("dcomp", None, "mean"), "ext_dcomp")]
        for src in ("mc", "xrc", "md"):
            arms[f"h_base+{src}"] = blocks(src)
    elif a.stage == 3:
        for ctl in ("block", "arm"):
            arms[f"h_base+A+mc[{ctl}-shuffled]"] = A + blocks("mc", ctl)
    elif a.stage == 7:      # twin-grouped folds: removes the repeat-measurement leak only (run with --twin)
        arms["h_base"] = []
        arms["h_base+A"] = A
        arms["h_base+A+mc"] = A + blocks("mc")
    elif a.stage == 6:      # junk-column control: as many N(0,1) columns as the shape blocks add
        noise = np.random.default_rng(0).standard_normal((len(co.y), 861)).astype(np.float32)
        nm_ = [f"noise.{i}" for i in range(noise.shape[1])]
        arms["h_base+noise861"] = [(noise, nm_, "ext_noise")]
        arms["h_base+A+noise861"] = A + [(noise, nm_, "ext_noise")]
    elif a.stage == 5:      # budget check: same arms as stage 4 without the shuffles (run with --nfeat)
        arms["h_base+A"] = A
        arms["h_base+A+mc"] = A + blocks("mc")
        arms["h_base+A+dcomp"] = A + [(*co.ext("dcomp", None, "mean"), "ext_dcomp")]
        arms["h_base+A+comp"] = A + [(*co.ext("comp", None, "mean"), "ext_comp")]
    elif a.stage == 4:      # the compact set that matters, under locus-grouped folds (run with --grouped)
        arms["h_base"] = []
        arms["h_base+A"] = A
        arms["h_base+A+mc"] = A + blocks("mc")
        arms["h_base+A+dcomp"] = A + [(*co.ext("dcomp", None, "mean"), "ext_dcomp")]
        arms["h_base+A+comp"] = A + [(*co.ext("comp", None, "mean"), "ext_comp")]
        for ctl in ("block", "arm"):
            arms[f"h_base+A+mc[{ctl}-shuffled]"] = A + blocks("mc", ctl)
    for label, parts in arms.items():
        if label in done:
            continue
        row = run_arm(co, label, parts, grouped=('twin' if a.twin else a.grouped), n_features=a.nfeat)
        append(row, path)
        print(f"{label:28s} cols={row['n_cols']:6d} ext={row['n_ext']:4d} rho={row['spearman']:.4f}"
              f" ±{row['spearman_sd']:.4f}", flush=True)


if __name__ == "__main__":
    main()
