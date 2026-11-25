"""Pick the SpCas9 guide SLICER predicts will cut best at a site in a genome.

Give it a genome (FASTA) and a cut site; it lists every NGG guide, on either strand, that cuts at that site (or within `--window` bp of it), reads the DNA around
each (the 1,000 nt on either side, as the model was trained), scores them with SLICER and prints the best 20-nt spacer.

    python -m sgrna.design.design_guide --genome genome.fasta --cut-site 1234567
    python -m sgrna.design.design_guide --genome genome.fasta --contig NC_000913.2 --cut-site 1234567 --window 15 --top 5 --out guides.csv

The cut site is the 1-based coordinate of the base just before the cut on the plus strand: the cut falls between `cut-site` and `cut-site + 1`. SpCas9 cuts 3 bp from
the PAM, between positions 17 and 18 of the protospacer, so a guide on the plus strand whose protospacer ends at `right` cuts after `right - 3`, and a guide on the
minus strand whose protospacer starts at `left` cuts after `left + 2`.

What the score is: SLICER's prediction of the E. coli screen's cutting score (higher cuts more; the median guide of the screen scores about 0 and the top tenth about 4).
It ranks guides within a site (the paper's held-out correlation is about 0.73); it is not a probability of cutting. Guides whose 20-mer occurs at more than one NGG site
of the genome are set aside unless nothing else is left, because the screen's score rises with the number of cuts. `seed_sites` counts the NGG sites that share the
12 PAM-proximal nt: a coarse off-target flag, not an off-target model.

Validated for SpCas9 in E. coli (and one other bacterium, C. rodentium). On eukaryotic data the guide rules differ (Appendix A5 of the paper), so do not use it there.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

# The featurisers import `sgrna.config`, which wants a project root. A user of the tool has none, so give it an empty one.
_RUNJIA = Path(__file__).resolve().parents[2]
if "RSI09_ROOT" not in os.environ:
    _root = _RUNJIA / ".slicer_root"
    (_root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (_root / "external_data").mkdir(parents=True, exist_ok=True)
    os.environ["RSI09_ROOT"] = str(_root)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODELS = _RUNJIA / "models"


def model_path(which: str, ext: str) -> Path:
    """`models/slicer_<which><ext>`; built from pieces so the code map does not mistake the prefix for a result file."""
    return MODELS / ("slicer" + "_" + which + ext)
FLANK = 1000
COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


# ---------------------------------------------------------------------------------------------------------------- genome
def read_fasta(path, contig: str | None = None) -> tuple[str, str]:
    """(name, upper-case sequence) of one record of a FASTA file (plain or .gz); the first record unless `contig` names another."""
    opener = gzip.open if str(path).endswith(".gz") else open
    name, chunks, found = None, [], {}
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    found[name] = "".join(chunks).upper()
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line.strip())
    if name is not None:
        found[name] = "".join(chunks).upper()
    if not found:
        raise ValueError(f"no sequence in {path}")
    if contig is None:
        if len(found) > 1:
            print(f"note: {path} holds {len(found)} records; using the first, {next(iter(found))} (name another with --contig)", file=sys.stderr)
        return next(iter(found.items()))
    if contig not in found:
        raise KeyError(f"{contig!r} is not in {path}; records: {', '.join(list(found)[:10])}")
    return contig, found[contig]


def _slice(g: str, a: int, b: int, circular: bool) -> str:
    """g[a:b] (0-based, half-open); wraps around the origin on a circular genome and is cut at the ends otherwise."""
    n = len(g)
    if circular:
        if b - a >= n:
            return g
        a0, b0 = a % n, b % n
        return g[a0:b0] if a0 < b0 else g[a0:] + g[:b0]
    return g[max(a, 0):max(min(b, n), 0)]


# ---------------------------------------------------------------------------------------------------------------- candidates
def site_context(g: str, s: int, strand: str, circular: bool = True, flank: int = FLANK) -> dict:
    """The DNA of the guide whose 20 nt protospacer occupies g[s:s+20] on the plus strand, read on the guide's own strand."""
    if strand == "+":
        up, proto, pam, dn = _slice(g, s - flank, s, circular), _slice(g, s, s + 20, circular), _slice(g, s + 20, s + 23, circular), _slice(g, s + 23, s + 23 + flank, circular)
    else:
        up = revcomp(_slice(g, s + 20, s + 20 + flank, circular))
        proto = revcomp(_slice(g, s, s + 20, circular))
        pam = revcomp(_slice(g, s - 3, s, circular))
        dn = revcomp(_slice(g, s - 3 - flank, s - 3, circular))
    return dict(upstream=up, protospacer=proto, pam=pam, downstream=dn)


def cut_after(s: int, strand: str) -> int:
    """1-based plus-strand coordinate of the base before the cut, for a protospacer at g[s:s+20] (0-based s)."""
    return s + 17 if strand == "+" else s + 3


def find_candidates(g: str, cut_site: int, window: int = 0, circular: bool = True) -> list[dict]:
    """Every NGG guide, on either strand, that cuts within `window` bp of `cut_site` (1-based, the base before the cut)."""
    n = len(g)
    out = []
    for strand, shift in (("+", 17), ("-", 3)):
        for d in range(-window, window + 1):
            s = cut_site + d - shift
            if circular:
                s %= n
            elif s < (0 if strand == "+" else 3) or s + (23 if strand == "+" else 20) > n:
                continue
            ctx = site_context(g, s, strand, circular)
            if len(ctx["protospacer"]) != 20 or len(ctx["pam"]) != 3 or ctx["pam"][1:] != "GG":
                continue
            if set(ctx["protospacer"] + ctx["pam"]) - set("ACGT"):
                continue
            out.append(dict(strand=strand, s=s, distance=d, left=s + 1, right=s + 20, cut_after=cut_after(s, strand), **ctx))
    return out


def _count_sites(g: str, spacer: str, circular: bool) -> tuple[int, int]:
    """(NGG-flanked exact copies of the 20-mer on both strands, NGG sites sharing its 12 PAM-proximal nt), counting the site itself."""
    pad = 3
    ext = g[-pad:] + g + g[:23] if circular else "N" * pad + g
    n = len(g)
    exact = seed = 0
    for k, motif in ((20, spacer), (12, spacer[-12:])):
        for m, plus in ((motif, True), (revcomp(motif), False)):          # plus: motif then NGG;  minus: CCN then the reverse complement
            i = ext.find(m)
            while i != -1:
                if pad <= i < pad + n:
                    ok = ext[i + k + 1:i + k + 3] == "GG" if plus else ext[i - 3:i - 1] == "CC"
                    if ok:
                        if k == 20:
                            exact += 1
                        else:
                            seed += 1
                i = ext.find(m, i + 1)
    return exact, seed


# ---------------------------------------------------------------------------------------------------------------- scoring
class Slicer:
    """A shipped SLICER model with the resources to featurise a guide."""

    def __init__(self, which: str = "tuned"):
        import xgboost as xgb

        meta = json.loads(model_path(which, ".json").read_text())
        self.meta, self.which = meta, which
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path(which, ".ubj")))
        self.tables = json.loads((MODELS / "resources" / "qct_lookup_tables.json").read_text())
        self.v_mapping = json.loads((MODELS / "resources" / "v_column_mapping.json").read_text())
        self.index = np.asarray(meta["column_index"], dtype=int)
        self.median = np.asarray(meta["medians"], dtype=np.float32)

    def features(self, ctx: pd.DataFrame) -> np.ndarray:
        """(n, 6,517) float32: the 6,169 base columns from the 20-mer, then the 348 flank columns."""
        from ..build import a_flank_fast as AF
        from ..build import base_fast as BF

        Mb, _ = BF.matrix_fast(ctx["protospacer"].tolist(), self.v_mapping, self.tables)
        idx = ctx.assign(sgRNAID=ctx["protospacer"], located=True)
        Ma, _ = AF.build_matrix(idx, self.tables)
        return np.hstack([Mb, Ma.astype(np.float32)])

    def predict_features(self, X: np.ndarray) -> np.ndarray:
        Xs = np.array(X[:, self.index], dtype=np.float32)
        r, c = np.where(np.isnan(Xs))
        Xs[r, c] = self.median[c]
        return self.model.predict(Xs)

    def predict(self, ctx: pd.DataFrame) -> np.ndarray:
        return self.predict_features(self.features(ctx))


# ---------------------------------------------------------------------------------------------------------------- the tool
def design(genome: str, cut_site: int, window: int = 0, model: str = "tuned", circular: bool = True, allow_widen: bool = True,
           keep_repeats: bool = False, slicer: Slicer | None = None, verbose: bool = True) -> pd.DataFrame:
    """Candidate guides for a cut site, best first. Columns: spacer, pam, strand, left, right, cut_after, distance, score, copies, seed_sites, flank_5, flank_3, unique, rank."""
    g = genome
    if not 1 <= cut_site <= len(g):
        raise ValueError(f"cut site {cut_site} is outside the sequence (1 to {len(g)})")
    used = window
    cands = find_candidates(g, cut_site, window, circular)
    while not cands and allow_widen and used < 200:
        used += 1
        cands = find_candidates(g, cut_site, used, circular)
    if not cands:
        raise LookupError(f"no NGG guide cuts within {used} bp of {cut_site}")
    if used != window and verbose:
        print(f"note: no NGG guide cuts exactly at {cut_site}; widened the window to {used} bp, the smallest that has one (use --exact to refuse)", file=sys.stderr)
    ctx = pd.DataFrame(cands)
    sl = slicer or Slicer(model)
    ctx["score"] = sl.predict(ctx)
    counts = [_count_sites(g, p, circular) for p in ctx["protospacer"]]
    ctx["copies"] = [c[0] for c in counts]
    ctx["seed_sites"] = [c[1] for c in counts]
    ctx["flank_5"] = ctx["upstream"].str.len()
    ctx["flank_3"] = ctx["downstream"].str.len()
    ctx["unique"] = ctx["copies"] == 1
    pool = ctx if keep_repeats or not ctx["unique"].any() else ctx[ctx["unique"]]
    if verbose and len(pool) < len(ctx):
        print(f"note: set aside {len(ctx) - len(pool)} guide(s) whose 20-mer occurs at more than one NGG site", file=sys.stderr)
    out = pool.sort_values(["score", "distance"], ascending=[False, True], key=lambda c: c if c.name == "score" else c.abs()).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out = out.rename(columns={"protospacer": "spacer"})
    short = (out["flank_5"] < FLANK) | (out["flank_3"] < FLANK)
    if short.any() and verbose:
        print(f"note: {int(short.sum())} candidate(s) have fewer than {FLANK} nt of genome on one side (a linear sequence ends nearby); their flank columns use what exists", file=sys.stderr)
    return out[["rank", "spacer", "pam", "strand", "left", "right", "cut_after", "distance", "score", "copies", "seed_sites", "flank_5", "flank_3", "unique"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--genome", required=True, help="FASTA file (plain or .gz)")
    ap.add_argument("--contig", help="record to use when the FASTA has several (default: the first)")
    ap.add_argument("--cut-site", type=int, required=True, help="1-based coordinate of the base just before the cut, on the plus strand")
    ap.add_argument("--window", type=int, default=0, help="accept guides whose cut is within this many bp of the cut site (default 0: exactly there)")
    ap.add_argument("--exact", action="store_true", help="do not widen the window when no guide cuts exactly at the site")
    ap.add_argument("--top", type=int, default=5, help="how many guides to print")
    ap.add_argument("--model", choices=("tuned", "default"), default="tuned")
    ap.add_argument("--linear", action="store_true", help="treat the sequence as linear (default: circular, as a bacterial chromosome)")
    ap.add_argument("--keep-repeats", action="store_true", help="do not set aside guides whose 20-mer has several NGG sites")
    ap.add_argument("--out", help="write every candidate to this CSV")
    a = ap.parse_args(argv)
    name, g = read_fasta(a.genome, a.contig)
    try:
        res = design(g, a.cut_site, a.window, a.model, circular=not a.linear, allow_widen=not a.exact, keep_repeats=a.keep_repeats)
    except (LookupError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if a.out:
        res.to_csv(a.out, index=False)
    b = res.iloc[0]
    print(f"{name}  cut between {a.cut_site} and {a.cut_site + 1}   ({len(g):,} bp, {'linear' if a.linear else 'circular'})")
    print(f"best guide (SLICER {a.model}):  5'-{b.spacer}-3'   PAM {b.pam}   strand {b.strand}")
    print(f"  protospacer {b.left}-{b.right}; cuts after {b.cut_after} ({b.distance:+d} bp from the requested site); predicted score {b.score:+.2f}; NGG copies of the 20-mer: {b.copies}; sites sharing the seed: {b.seed_sites}")
    if b.copies > 1:
        print(f"  WARNING: this 20-mer has {b.copies} NGG sites in the sequence and will cut at all of them; no candidate here is unique")
    q = json.loads(model_path(a.model, ".json").read_text())["label_quantiles"]
    print(f"  for scale: in the E. coli screen the median measured score is {q['50']:+.1f} and the top tenth starts at {q['90']:+.1f}")
    if len(res) > 1:
        print(f"\ntop {min(a.top, len(res))} of {len(res)} candidates:")
        print(res.head(a.top).drop(columns=["unique"]).to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
