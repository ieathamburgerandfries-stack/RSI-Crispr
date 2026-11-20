"""Build feature families on crisprHAL's curated cohort, not the published rows.

The model layer moved to crisprHAL 2's 33,567 curated guides on 13 October; the
feature blocks were built against the published 13,880-row index and had to
follow. This is that build, kept as a module rather than an inline script so
the curated blocks are reproducible from the repo.

Two differences from `build_features`:

  * the index is `data/interim/curated_guide_index.csv` (every curated guide
    located on NC_000913.2, with its flanks), and each family writes
    `features/<family>_curated.csv`;
  * **a block with no populated feature column is a failure, not a null.**
    `b_energy` catches its exceptions per guide and drops the error column, so a
    missing ViennaRNA produced a file containing nothing but 33,567 sgRNAIDs and
    reported success. Fed to an ablation that would have scored Δρ = 0.000 and
    read as a clean negative result. The check below is cheap; the failure it
    catches is silent and would have been reported.

    python -m sgrna.build_curated_features --families b_energy c_folding
    python -m sgrna.build_curated_features --all
"""

from __future__ import annotations

import argparse
import time
import traceback

import numpy as np
import pandas as pd

from .. import config
from . import genome
from ..features import ALL, REGISTRY
from ..io_utils import save_block

INDEX = config.INTERIM / "curated_guide_index.csv"


def _b_numbers(index: pd.DataFrame) -> pd.DataFrame:
    """Host gene and its b-number for each curated guide.

    `g_transcription` needs a b-number to join PRECISE-1K expression, and the
    curated index was built from crisprHAL's CSVs, which carry none -- that is
    what blocked family G on this cohort. The published index got b-numbers
    from Noshay's own annotation, which covers only the rows they published.

    The reference gene table (`genes_NC_000913.2.tsv`, from the Bikard screen)
    gives coordinates and gene *names* but no locus tags, and PRECISE-1K's
    `gene_info.csv` gives locus tags keyed by gene name. So the gene is
    assigned by coordinate on NC_000913.2 and the b-number by name:
    annotation-free where the coordinates matter, and a name lookup only for
    the expression join. 4,130 of the 4,499 reference genes have a PRECISE-1K
    name; guides inside the rest get a gene with no b-number, which the
    expression columns then leave as NaN rather than silently as zero.
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

    Families B, C, D, E, F, H, I and J read only sequence and coordinates, so
    they built on the curated index as it stood. Family G reads four columns
    that were never there -- `ori_distance`, `replichore`, `leading_strand` and
    `b_number` -- and `ori_distance` in particular is a plain KeyError, which is
    why the first curated build dropped the family instead of measuring it.
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


def load_index(verbose: bool = True) -> pd.DataFrame:
    if not INDEX.exists():
        raise FileNotFoundError(
            f"{INDEX} not found -- build it from expand.load_crisprhal() and "
            f"genome.locate() first."
        )
    return augment(pd.read_csv(INDEX, low_memory=False), verbose=verbose)


def run(families, index: pd.DataFrame | None = None) -> dict[str, str]:
    index = load_index() if index is None else index
    status: dict[str, str] = {}

    for name in families:
        print(f"\n=== {name} (curated) " + "=" * max(0, 48 - len(name)))
        t0 = time.time()
        try:
            block = REGISTRY[name].build(index)
            feats = [c for c in block.columns if c != config.ID_COL]
            filled = int(block[feats].notna().any(axis=1).sum()) if feats else 0
            if not feats or filled == 0:
                raise RuntimeError(
                    f"built {len(block):,} rows with {len(feats)} feature "
                    f"columns and {filled:,} populated guides -- an empty block "
                    f"is a failed build, not a null result"
                )
            path = save_block(block, f"{name}_curated")
            print(f"{name}: {len(feats):,} features, {filled:,}/{len(block):,} "
                  f"guides populated, {time.time() - t0:.0f}s -> {path.name}")
            status[name] = f"ok ({len(feats)} features)"
        except FileNotFoundError as exc:
            print(f"{name}: SKIPPED -- {exc}")
            status[name] = f"skipped: {exc}"
        except Exception as exc:
            print(f"{name}: FAILED -- {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
            status[name] = f"failed: {type(exc).__name__}: {exc}"
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--families", nargs="+", choices=ALL)
    a = ap.parse_args(argv)

    status = run(list(ALL) if a.all else a.families)
    print("\n" + "=" * 68)
    for k, v in status.items():
        print(f"  {k:<16} {v}")
    return 0 if all(v.startswith("ok") for v in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
