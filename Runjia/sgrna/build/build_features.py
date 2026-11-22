"""Step 1 -- build one or more feature families.

From a notebook cell (the usual way):

    nb.build_features("all")
    nb.build_features(["a_flank", "c_folding"])
    nb.build_features("self-contained")     # no external data needed

or from a terminal:

    python -m sgrna.build.build_features --all
    python -m sgrna.build.build_features --families a_flank c_folding
    python -m sgrna.build.build_features --self-contained

Each family writes its own CSV into data/interim/features/. They are
independent: rebuilding one never touches the others, and a family whose
external dataset is missing fails on its own without taking the run down.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from .. import config
from ..features import ALL, REGISTRY, SELF_CONTAINED
from ..io_utils import load_guide_index, save_block


def run(families, index=None) -> dict[str, str]:
    index = load_guide_index() if index is None else index
    status: dict[str, str] = {}

    for name in families:
        module = REGISTRY[name]
        print(f"\n=== {name} " + "=" * (60 - len(name)))
        t0 = time.time()
        try:
            block = module.build(index)
            path = save_block(block, name)
            dt = time.time() - t0
            n_feat = block.shape[1] - 1
            filled = block.drop(columns=[config.ID_COL]).notna().any(axis=1).sum()
            print(
                f"{name}: {n_feat:,} features, {filled:,}/{len(block):,} guides "
                f"populated, {dt:.0f}s -> {path.name}"
            )
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

    status = run(families)

    print("\n" + "=" * 68)
    for name in families:
        print(f"  {name:18s} {status.get(name, 'not run')}")
    failed = [k for k, v in status.items() if v.startswith("failed")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
