"""On the human matrix the shape and 3D-path columns add +0.0183 to family A, where E. coli gives −0.0007, a difference that is not explained.

Cohort: the guides of the Noshay et al. 2023 human matrix that locate to a single NGG-flanked copy on hg38
(`--locate`, below). Representation, selector, model and folds are those of the E. coli runs: the regenerated base
(everything computable from the 20-mer), family A (`a_flank`, built from the 1,000 nt either side of the cut), top-300
selection inside folds, the frozen XGBoost champion, 5 folds x 3 seeds. The human matrix has no gene identifier, so
folds are random over guides (as in the project's human ladder), which is the more permissive design; the comparison
between arms is paired on identical folds, so it is not affected by that choice, but absolute rho values are not
comparable with the gene-grouped E. coli numbers.

    python -m sgrna.compare.human_shape --locate        # the guides with their flanks (cached); see `locate` below
    python -m sgrna.compare.human_shape --stage 1     # base, base+A, and shape on top of A
    python -m sgrna.compare.human_shape --stage 2     # shape without A, composition references
    python -m sgrna.compare.human_shape --stage 3     # shuffled-flank controls

--locate
--------
The project's human matrix (Noshay et al. 2023, Supplementary Table 3, 17,422 guides) describes only the 20-nt
protospacer. The protospacer is decoded from the monomer electron-count columns (C 42, T 48, A 50, G 56), then located
on hg38 as an NGG-flanked 20-mer with exactly one NGG-flanked copy in the primary assembly (chr1-22, X, Y, M);
non-NGG copies are ignored, as in the E. coli curation. 1,000 nt are cut on each side in the guide's orientation, giving
the same table layout as `curated_guide_index.csv` (upstream / protospacer / pam / downstream).

The scan encodes each chromosome as 2-bit codes and tests every 20-mer against the wanted set with numpy, so the whole
genome takes minutes rather than a Python loop over 4e8 positions.

Report section: Part 2.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time

import numpy as np
import pandas as pd

from sgrna import config
from sgrna.build import featurise, genome
from sgrna.features import a_flank
from sgrna.harness.run_ablation import cross_validate

from ..features import geom
from ..common import EXTERNAL, INTERIM, OUT


# ======================================================================================================  human guides with their flanks (--locate)
HG38 = EXTERNAL / "hg38" / "hg38.fa.gz"
HDIR = INTERIM / "human"
HDIR.mkdir(parents=True, exist_ok=True)
FLANK = 1000
ELECTRONS_TO_BASE = {42.0: "C", 48.0: "T", 50.0: "A", 56.0: "G"}
PRIMARY = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}
LUT = np.full(256, 4, np.uint8)
for _i, _c in enumerate("ACGT"):
    LUT[ord(_c)] = _i
    LUT[ord(_c.lower())] = _i
REV = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def revcomp(s: str) -> str:
    return "".join(REV.get(c, "N") for c in reversed(s))


def decode() -> pd.DataFrame:
    path = HDIR / "decoded.csv"
    if path.exists():
        return pd.read_csv(path)
    cols = [config.ID_COL, config.TARGET_COL] + [f"p{i}monomer.No.electronsraw" for i in range(1, 21)]
    df = pd.read_csv(config.RAW / "human_feature_matrix.csv", header=config.BASE_MATRIX_HEADER_ROW, usecols=cols,
                     low_memory=False)
    mat = df[[f"p{i}monomer.No.electronsraw" for i in range(1, 21)]].to_numpy(float)
    seqs = ["".join(ELECTRONS_TO_BASE.get(v, "N") for v in row) for row in mat]
    out = pd.DataFrame({config.ID_COL: df[config.ID_COL].astype(str),
                        "y": pd.to_numeric(df[config.TARGET_COL], errors="coerce"), "protospacer": seqs})
    out["decoded_ok"] = ~out.protospacer.str.contains("N")
    out.to_csv(path, index=False)
    return out


def _fasta(path):
    name, chunks = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.strip())
        if name is not None:
            yield name, "".join(chunks)


def _codes(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """20-mer codes (2 bits per base, first base most significant) for every start, and a validity mask."""
    n = len(arr) - 19
    code = np.zeros(n, np.uint64)
    bad = np.zeros(n, bool)
    for k in range(20):
        a = arr[k:k + n]
        bad |= a > 3
        code = (code << np.uint64(2)) | (a & 3).astype(np.uint64)
    return code, ~bad


def encode20(s: str) -> int:
    v = 0
    for ch in s:
        v = (v << 2) | "ACGT".index(ch)
    return v


def slice_linear(seq: str, a: int, b: int) -> str:
    """1-based inclusive slice of the chromosome, padded with N outside it."""
    length, n = b - a + 1, len(seq)
    left = max(0, min(length, 1 - a))
    right = max(0, min(length - left, b - n))
    mid = length - left - right
    start = max(a, 1)
    return "N" * left + seq[start - 1:start - 1 + mid].upper() + "N" * right


def context(seq: str, strand: str, left: int, right: int) -> dict:
    """Upstream / protospacer / PAM / downstream in the guide's orientation (1-based, inclusive)."""
    if strand == "+":
        return dict(upstream=slice_linear(seq, left - FLANK, left - 1), protospacer=slice_linear(seq, left, right),
                    pam=slice_linear(seq, right + 1, right + 3),
                    downstream=slice_linear(seq, right + 4, right + 3 + FLANK))
    return dict(upstream=revcomp(slice_linear(seq, right + 1, right + FLANK)),
                protospacer=revcomp(slice_linear(seq, left, right)),
                pam=revcomp(slice_linear(seq, left - 3, left - 1)),
                downstream=revcomp(slice_linear(seq, left - 3 - FLANK, left - 4)))


CHUNK = 20_000_000


def locate(force=False) -> pd.DataFrame:
    path = HDIR / "located.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    dec = decode()
    ok = dec[dec.decoded_ok]
    wanted = set(ok.protospacer)
    fwd = {encode20(p): p for p in wanted}
    rcm = {encode20(revcomp(p)): p for p in wanted}
    fkeys, rkeys = np.array(sorted(fwd), np.uint64), np.array(sorted(rcm), np.uint64)
    hits: dict[str, list[dict]] = {}
    t0 = time.time()
    for chrom, seq in _fasta(HG38):
        if chrom not in PRIMARY:
            continue
        arr = LUT[np.frombuffer(seq.encode("ascii"), np.uint8)]
        found = []                                            # (protospacer, strand, 0-based start)
        for c0 in range(0, len(arr) - 19, CHUNK):
            sub = arr[c0:c0 + CHUNK + 19]
            code, valid = _codes(sub)
            for strand, keys, table in (("+", fkeys, fwd), ("-", rkeys, rcm)):
                for j in np.nonzero(valid & np.isin(code, keys))[0]:
                    g = c0 + int(j)
                    if strand == "+":
                        if g + 22 < len(arr) and arr[g + 21] == 2 and arr[g + 22] == 2:    # PAM N G G
                            found.append((table[int(code[j])], "+", g))
                    elif g >= 3 and arr[g - 3] == 1 and arr[g - 2] == 1:                    # C C N before it
                        found.append((table[int(code[j])], "-", g))
        for proto, strand, g in found:
            left, right = g + 1, g + 20
            hits.setdefault(proto, []).append(dict(chrom=chrom, strand=strand, left=left, right=right,
                                                   **context(seq, strand, left, right)))
        print(f"  {chrom:6s} {len(seq):>11,} bp  {len(found):5d} NGG hits  {time.time() - t0:5.0f}s", flush=True)
        del seq, arr
    rows, n_multi, n_none = [], 0, 0
    for _, r in ok.iterrows():
        hs = hits.get(r.protospacer, [])
        if not hs:
            n_none += 1
        elif len(hs) > 1:
            n_multi += 1
        else:
            rows.append(dict(sgRNAID=r[config.ID_COL], y=r.y, **hs[0]))
    out = pd.DataFrame(rows)
    assert (out.protospacer.str.len() == 20).all() and out.pam.str[1:].eq("GG").all()
    out.to_csv(path, index=False)
    (HDIR / "locate_audit.json").write_text(json.dumps(dict(n_guides=len(dec), n_decoded=int(len(ok)),
                                                           n_unique=len(out), n_multi=n_multi, n_none=n_none)))
    return out


# ======================================================================================================  the shape ablation on the human matrix (--stage)
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
    ap.add_argument("--stage", type=int)
    ap.add_argument("--locate", action="store_true", help="decode and locate the human guides on hg38 and cut their flanks (cached)")
    ap.add_argument("--grouped", action="store_true", help="locus-grouped folds instead of random folds")
    ap.add_argument("--twin", action="store_true", help="folds that keep identical protospacers together")
    ap.add_argument("--nfeat", type=int, default=None, help="selector budget (default: the project's 300)")
    a = ap.parse_args(argv)
    if a.locate:
        out = locate()
        print(len(out), "located;", (HDIR / "locate_audit.json").read_text())
        if not a.stage:
            return
    if not a.stage:
        ap.error("give --stage N or --locate")
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
