"""Ablations of the shape / 3D blocks on the curated cutting cohort, gene-grouped CV.

Each arm is the base representation (6,169 columns regenerable from the 20-mer) plus a set of
extension blocks, optionally plus family A. All arms run on the same 30,203 genic guides, with the
same gene-grouped folds per seed, so paired per-fold differences are valid. Feature selection
(top 300 by gain) is fitted inside each training fold, as everywhere in this project.

    python -m sgrna.ext.shape_ablate --stage 1     # blocks by source, on base
    python -m sgrna.ext.shape_ablate --stage 2     # same, on top of family A
    python -m sgrna.ext.shape_ablate --stage 3     # shuffled-flank controls
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from sgrna import config
from sgrna.curated_ablation import _base, _block, _groups
from sgrna.run_ablation import cross_validate

from . import shape_build
from .common import OUT

SEEDS = (41, 42, 43)
OUTCSV = OUT / "shape_ablation.csv"


class Cohort:
    def __init__(self):
        X, y, names, protos = _base()
        groups, ok = _groups("gene", protos)
        self.X, self.y, self.protos, self.groups = X[ok], y[ok], protos[ok], groups[ok]
        self.names = names
        self.ok = ok
        self._A = None

    def A(self):
        if self._A is None:
            v, fn = _block("a_flank", self.protos)
            self._A = (v, fn)
        return self._A

    def ext(self, source, control, block, only=None):
        """`only`: optional predicate on the raw column name, to take a column-kind subset."""
        arr, nm = shape_build.load(source, control, block)
        arr = arr[self.ok]
        keep = np.isfinite(arr).any(0) & (np.nanstd(arr, 0) > 0)
        if only is not None:
            keep &= np.array([bool(only(n)) for n in nm])
        return arr[:, keep], [f"{source}.{control or 'real'}.{block}.{n}" for n, k in zip(nm, keep) if k]


def run_arm(co: Cohort, label: str, parts: list, seeds=SEEDS, tagpref="shp", n_features=None, overrides=None):
    """parts: list of (array, names, family_label)."""
    Xs, names, fams = [co.X], list(co.names), ["base"] * len(co.names)
    for arr, nm, fam in parts:
        Xs.append(arr); names += nm; fams += [fam] * len(nm)
    X = np.hstack(Xs)
    tag = f"{tagpref}_{label}"
    res = cross_validate(X, co.y, names, fams, n_features=n_features or config.N_FEATURES, seeds=list(seeds),
                         n_splits=config.N_SPLITS, verbose=False, groups=co.groups, tag=tag,
                         experiment=label, champion_overrides=overrides)
    folds = pd.read_csv(config.RESULTS / f"folds_{tag}.csv")
    folds = folds[folds.experiment == label].sort_values(["seed", "fold"])
    sel = res.get("selected_by_family") if isinstance(res, dict) else None
    row = dict(arm=label, n_cols=X.shape[1], n_ext=sum(a.shape[1] for a, _, _ in parts),
               spearman=res["summary"]["spearman"]["mean"], spearman_sd=res["summary"]["spearman"]["sd"],
               r2=res["summary"]["r2"]["mean"],
               n_selected_ext=float(folds.n_engineered_selected.mean()),
               fold_rho=";".join(f"{v:.5f}" for v in folds.spearman))
    return row


def append(row):
    df = pd.read_csv(OUTCSV) if OUTCSV.exists() else pd.DataFrame()
    df = pd.concat([df[df.arm != row["arm"]], pd.DataFrame([row])], ignore_index=True) if len(df) else pd.DataFrame([row])
    df.to_csv(OUTCSV, index=False)


def done():
    return set(pd.read_csv(OUTCSV).arm) if OUTCSV.exists() else set()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    a = ap.parse_args(argv)
    co = Cohort()
    print(f"  {co.X.shape[0]:,} guides, {co.X.shape[1]:,} base columns", flush=True)
    vA, nA = co.A()
    Apart = [(vA, nA, "A")]
    arms = []                       # (label, parts)
    def P(src, ctl, blk, fam=None):
        arr, nm = co.ext(src, ctl, blk); return (arr, nm, fam or f"S_{src}_{ctl or 'real'}_{blk}")

    if a.stage == 1:
        arms.append(("base", []))
        arms.append(("A", Apart))
        for src in ("mc", "md", "xrc"):
            for blk in ("mean", "vec", "geom"):
                arms.append((f"{src}_{blk}", [P(src, None, blk)]))
            arms.append((f"{src}_all", [P(src, None, b) for b in ("mean", "vec", "geom")]))
        arms.append(("allsrc_geom", [P(s, None, "geom") for s in ("mc", "md", "xrc")]))
        arms.append(("allsrc_all", [P(s, None, b) for s in ("mc", "md", "xrc") for b in ("mean", "vec", "geom")]))
    elif a.stage == 2:
        for blk in ("mean", "vec", "geom"):
            arms.append((f"A+mc_{blk}", Apart + [P("mc", None, blk)]))
        arms.append(("A+mc_all", Apart + [P("mc", None, b) for b in ("mean", "vec", "geom")]))
        for src in ("md", "xrc"):
            arms.append((f"A+{src}_geom", Apart + [P(src, None, "geom")]))
            arms.append((f"A+{src}_all", Apart + [P(src, None, b) for b in ("mean", "vec", "geom")]))
        arms.append(("A+allsrc_all", Apart + [P(s_, None, b) for s_ in ("mc", "md", "xrc") for b in ("mean", "vec", "geom")]))
    elif a.stage == 4:
        # column kinds inside the blocks, for the Monte Carlo source, on top of family A
        import re
        INTRA = {"ProT", "Buckle", "Shear", "Opening", "Stretch", "Stagger"}
        STEP = {"Shift", "Slide", "Rise", "Tilt", "Roll", "HelT"}
        feat = lambda n: n.split(".")[-1]
        subsets = {
            "mean": {"intra": lambda n: feat(n) in INTRA, "groove(MGW)": lambda n: feat(n) == "MGW",
                     "step": lambda n: feat(n) in STEP,
                     "up arm": lambda n: n.startswith("up."), "dn arm": lambda n: n.startswith("dn."),
                     "near (<=100 nt)": lambda n: ".w0_25." in n or ".w25_100." in n,
                     "far (>100 nt)": lambda n: ".w100_250." in n or ".w250_" in n},
            "geom": {"path positions": lambda n: n.startswith("path."), "axis tangents": lambda n: n.startswith("axis."),
                     "gyration tensor": lambda n: n.startswith("gyr."), "straightness": lambda n: n.startswith("straight."),
                     "excess twist": lambda n: n.startswith("xtwist."),
                     "near (offset/window <=100)": lambda n: bool(re.search(r"k[+-](12|25|50|100)\.|(dn|up|both)100\.|[a-z]+100$", n)),
                     "far (>100)": lambda n: bool(re.search(r"k[+-](200|400)\.|(dn|up|both)(250|499)\.|[a-z]+(250|499)$", n))},
        }
        for blk, d_ in subsets.items():
            for kind, fn in d_.items():
                arr, nm = co.ext("mc", None, blk, only=fn)
                if arr.shape[1] == 0:
                    print(f"  mc_{blk}[{kind}]: no columns"); continue
                parts = [(arr, nm, f"S_mc_{blk}_{kind}")]
                arms.append((f"mc_{blk}[{kind}]", parts))
                arms.append((f"A+mc_{blk}[{kind}]", Apart + parts))
    elif a.stage == 11:
        # fresh seeds (run with --seeds 44 45 46): do the headline comparisons hold on folds never seen before?
        arms.append(("base#s2", []))
        arms.append(("A#s2", Apart))
        arms.append(("mc_all#s2", [P("mc", None, b) for b in ("mean", "vec", "geom")]))
        arms.append(("A+mc_all#s2", Apart + [P("mc", None, b) for b in ("mean", "vec", "geom")]))
        arms.append(("A+mc_armshuf_all#s2", Apart + [P("mc", "arm", b) for b in ("mean", "vec", "geom")]))
    elif a.stage == 10:
        # capacity check: the champion is depth 4 / 400 trees; give the booster depth 6 / 800 trees
        arms.append(("A@d6", Apart))
        arms.append(("A+mc_all@d6", Apart + [P("mc", None, b) for b in ("mean", "vec", "geom")]))
        arms.append(("A+mc_armshuf_all@d6", Apart + [P("mc", "arm", b) for b in ("mean", "vec", "geom")]))
    elif a.stage == 9:
        for ctl in ("block", "arm"):
            arms.append((f"mc_{ctl}shuf_vec", [P("mc", ctl, "vec")]))
    elif a.stage == 8:
        # family A is itself four kinds of column: take it apart
        vA_, nA_ = co.A()
        def sub(*prefixes, drop=False):
            m = np.array([any(n.startswith(f"eng.flank.{p}.") for p in prefixes) for n in nA_])
            if drop:
                m = ~m
            return [(vA_[:, m], [n for n, k in zip(nA_, m) if k], "A_sub")]
        for kind in ("seq", "qct", "win", "comp"):
            arms.append((f"A_only_{kind}", sub(kind)))
        arms.append(("A_local(seq+qct)", sub("seq", "qct")))
        arms.append(("A_windowed(win+comp)", sub("win", "comp")))
        for kind in ("seq", "qct", "win", "comp"):
            arms.append((f"A_minus_{kind}", sub(kind, drop=True)))
    elif a.stage == 7:
        # displacement check: the same comparison with a 600-column budget instead of 300
        arms.append(("A@600", Apart)); arms.append(("A+mc_all@600", Apart + [P("mc", None, b) for b in ("mean", "vec", "geom")]))
        arms.append(("A+mc_geom@600", Apart + [P("mc", None, "geom")]))
        arms.append(("base@600", []))
    elif a.stage == 6:
        # the composition reference: letters in the same windows, alone and with the shape means
        arms.append(("comp_mean", [P("comp", None, "mean")]))
        arms.append(("comp+mc_mean", [P("comp", None, "mean"), P("mc", None, "mean")]))
        arms.append(("A+comp_mean", Apart + [P("comp", None, "mean")]))
        arms.append(("dcomp_mean", [P("dcomp", None, "mean")]))
        arms.append(("dcomp+mc_mean", [P("dcomp", None, "mean"), P("mc", None, "mean")]))
        arms.append(("A+dcomp_mean", Apart + [P("dcomp", None, "mean")]))
    elif a.stage == 5:
        # does recentring each source's mean twist to 34.3 deg (10.5 bp/turn) explain the md geometry deficit?
        for src in ("md", "xrc", "mc"):
            arms.append((f"{src}_geom_tw", [P(src, "tw", "geom")]))
    elif a.stage == 3:
        # shuffled-flank controls: same pipeline, flank bases shuffled (block = within 25-nt blocks, arm = whole arm)
        for ctl in ("arm", "block"):
            for blk in ("geom", "mean"):
                arms.append((f"mc_{ctl}shuf_{blk}", [P("mc", ctl, blk)]))
            arms.append((f"mc_{ctl}shuf_all", [P("mc", ctl, b) for b in ("mean", "vec", "geom")]))
            arms.append((f"A+mc_{ctl}shuf_all", Apart + [P("mc", ctl, b) for b in ("mean", "vec", "geom")]))
    for label, parts in arms:
        if label in done():
            print(f"  {label}: cached", flush=True); continue
        row = run_arm(co, label, parts, seeds=a.seeds, n_features=600 if label.endswith('@600') else None,
                      overrides=dict(max_depth=6, n_estimators=800) if label.endswith('@d6') else None)
        append(row)
        print(f"  {label:22s} rho {row['spearman']:.4f} (sd {row['spearman_sd']:.4f}) "
              f"ext cols {row['n_ext']} selected {row['n_selected_ext']:.0f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
