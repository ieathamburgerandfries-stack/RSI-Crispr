"""Build feature families for the published rows or for the 33,567-guide cohort, one CSV per family.

Two cohorts, one loop (`--cohort`):

  published   the 13,880 rows Noshay et al. released; each family writes `features/<family>.csv`
  curated     crisprHAL 2's 33,567 guides, each located on NC_000913.2 with its flanks; each family writes `features/<family>_curated.csv`

From a notebook cell (the usual way, published rows):

    nb.build_features("all")
    nb.build_features(["a_flank", "c_folding"])
    nb.build_features("self-contained")     # no external data needed

or from a terminal:

    python -m sgrna.build.build_features --all
    python -m sgrna.build.build_features --families a_flank c_folding
    python -m sgrna.build.build_features --self-contained
    python -m sgrna.build.build_features --cohort curated --all

Families are independent: rebuilding one never touches the others, and a family whose external dataset is missing fails on its
own without taking the run down.

**A block with no populated feature column is a failure, not a null.** `b_energy` catches its exceptions per guide and drops the
error column, so a missing ViennaRNA once produced a file holding nothing but 33,567 sgRNAIDs and reported success; fed to an
ablation that scores a gain of exactly 0.000 and reads as a clean negative result. The check is cheap and the failure it catches
is silent, so it applies to both cohorts.

Family A is built by `a_flank_fast`, which gives the same columns as `features.a_flank` (maximum absolute difference 0.0 on all
33,567 guides; `controls/equiv_check.py`) about twenty times faster. The curated index also gets the columns the published one
carries and crisprHAL's files do not (`augment`): family G needs `ori_distance`, `replichore`, `leading_strand` and `b_number`.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import numpy as np
import pandas as pd

from .. import config
from ..features import ALL, REGISTRY, SELF_CONTAINED
from ..io_utils import load_guide_index, save_block
from . import genome

CURATED_INDEX = config.INTERIM / "curated_guide_index.csv"


def _b_numbers(index: pd.DataFrame) -> pd.DataFrame:
    """Host gene and its b-number for each curated guide.

    `g_transcription` needs a b-number to join PRECISE-1K expression, and the curated index was built from crisprHAL's CSVs,
    which carry none. The reference gene table (`genes_NC_000913.2.tsv`, from the Bikard screen) gives coordinates and gene
    *names* but no locus tags, and PRECISE-1K's `gene_info.csv` gives locus tags keyed by gene name. So the gene is assigned by
    coordinate on NC_000913.2 and the b-number by name: annotation-free where the coordinates matter, and a name lookup only
    for the expression join. 4,130 of the 4,499 reference genes have a PRECISE-1K name; guides inside the rest get a gene with
    no b-number, which the expression columns then leave as NaN rather than silently as zero.
    """
    gt = pd.read_csv(config.GENE_TABLE, sep="\t").rename(columns={"ori": "strand"})
    gt = gt.sort_values("left").reset_index(drop=True)
    starts, ends = gt["left"].to_numpy(), gt["right"].to_numpy()
    names = gt["name"].to_numpy()

    gi = pd.read_csv(config.PRECISE1K / "annotation" / "gene_info.csv")
    to_b = dict(zip(gi["gene_name"].astype(str), gi["locus_tag"].astype(str).str.lower()))

    pos = pd.to_numeric(index["left"], errors="coerce").to_numpy()
    gene = np.full(len(index), None, dtype=object)
    slot = np.searchsorted(starts, np.nan_to_num(pos, nan=-1), side="right") - 1
    for i, (p, sl) in enumerate(zip(pos, slot)):
        if not np.isfinite(p) or sl < 0:
            continue
        for k in range(sl, max(-1, sl - 5), -1):     # overlapping genes
            if starts[k] <= p <= ends[k]:
                gene[i] = names[k]
                break
    return pd.DataFrame({
        "gene_name": gene,
        "b_number": [to_b.get(g) if g is not None else None for g in gene],
    }, index=index.index)


def augment(index: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Add the columns the published index carried and crisprHAL's CSVs do not.

    Families B, C, D, E, F, H, I and J read only sequence and coordinates, so they built on the curated index as it stood. Family G
    reads four columns that were never there -- `ori_distance`, `replichore`, `leading_strand` and `b_number` -- and `ori_distance`
    in particular is a plain KeyError, which is why the first curated build dropped the family instead of measuring it.
    """
    out = index.copy()
    lm = genome.replication_landmarks()
    pos = pd.to_numeric(out["left"], errors="coerce")

    if "ori_distance" not in out:
        out["ori_distance"] = [genome.circular_distance(int(p), lm["ori"])
                               if pd.notna(p) else np.nan for p in pos]
    if "replichore" not in out:
        out["replichore"] = [genome.replichore(int(p)) if pd.notna(p) else None
                             for p in pos]
    if "leading_strand" not in out:
        out["leading_strand"] = [
            np.nan if pd.isna(p) or not isinstance(s, str)
            else int((genome.replication_fork_direction(int(p)) == 1) == (s == "+"))
            for p, s in zip(pos, out["strand"])
        ]
    if "b_number" not in out:
        g = _b_numbers(out)
        out["gene_name"] = out.get("gene_name", g["gene_name"])
        out["b_number"] = g["b_number"]
        if verbose:
            print(f"  host gene assigned for {g['gene_name'].notna().sum():,} "
                  f"of {len(out):,} guides, b-number for "
                  f"{g['b_number'].notna().sum():,}")
    return out


def load_curated_index(verbose: bool = True) -> pd.DataFrame:
    """The 33,567 curated guides located on the genome with their flanks, plus the columns family G needs."""
    if not CURATED_INDEX.exists():
        raise FileNotFoundError(
            f"{CURATED_INDEX} not found -- build it from curated_matrix.load_crisprhal() and "
            f"genome.locate() first."
        )
    return augment(pd.read_csv(CURATED_INDEX, low_memory=False), verbose=verbose)


def load_index(cohort: str = "published", verbose: bool = True) -> pd.DataFrame:
    """The guide index of a cohort: `published` (Noshay et al.'s 13,880 rows) or `curated` (the 33,567 guides)."""
    if cohort == "curated":
        return load_curated_index(verbose=verbose)
    if cohort == "published":
        return load_guide_index()
    raise ValueError(f"unknown cohort {cohort!r}; use 'published' or 'curated'")


def build_block(name: str, index: pd.DataFrame, tables=None) -> pd.DataFrame:
    """One family on an index, refusing a block that has no populated feature column."""
    if name == "a_flank":
        from . import a_flank_fast
        block = a_flank_fast.build_fast(index, tables)
    else:
        block = REGISTRY[name].build(index)
    feats = [c for c in block.columns if c != config.ID_COL]
    filled = int(block[feats].notna().any(axis=1).sum()) if feats else 0
    if not feats or filled == 0:
        raise RuntimeError(
            f"built {len(block):,} rows with {len(feats)} feature columns and {filled:,} populated guides -- "
            f"an empty block is a failed build, not a null result"
        )
    return block


def run(families, index=None, cohort: str = "published") -> dict[str, str]:
    index = load_index(cohort) if index is None else index
    suffix = "_curated" if cohort == "curated" else ""
    status: dict[str, str] = {}

    for name in families:
        print(f"\n=== {name} ({cohort}) " + "=" * max(0, 52 - len(name) - len(cohort)))
        t0 = time.time()
        try:
            block = build_block(name, index)
            path = save_block(block, name + suffix)
            n_feat = block.shape[1] - 1
            filled = int(block.drop(columns=[config.ID_COL]).notna().any(axis=1).sum())
            print(f"{name}: {n_feat:,} features, {filled:,}/{len(block):,} guides "
                  f"populated, {time.time() - t0:.0f}s -> {path.name}")
            status[name] = f"ok ({n_feat} features)"
        except FileNotFoundError as exc:
            print(f"{name}: SKIPPED -- {exc}")
            status[name] = f"skipped: {exc}"
        except Exception as exc:  # keep going; report at the end
            print(f"{name}: FAILED -- {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
            status[name] = f"failed: {type(exc).__name__}: {exc}"
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", choices=("published", "curated"), default="published")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="build every family")
    group.add_argument("--self-contained", action="store_true",
                       help="only families that need nothing but the genome")
    group.add_argument("--families", nargs="+", choices=ALL,
                       help="build just these")
    args = ap.parse_args(argv)

    if args.all:
        families = list(ALL)
    elif args.self_contained:
        families = list(SELF_CONTAINED)
    else:
        families = args.families

    status = run(families, cohort=args.cohort)

    print("\n" + "=" * 68)
    for name in families:
        print(f"  {name:18s} {status.get(name, 'not run')}")
    failed = [k for k, v in status.items() if v.startswith("failed")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
