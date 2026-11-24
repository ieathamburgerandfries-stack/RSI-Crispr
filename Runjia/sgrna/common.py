"""Output folders for the experiment modules, and the resolver for the main project's results that this repository does not track.

The experiment modules read the pipeline and write only under `results/post_freeze/` and
`data/interim/post_freeze/` (the folder names record when the work was done; the code is organised by function).
Run with:  RSI09_ROOT=<project root with data/>  PYTHONPATH=<fork>/Runjia
"""
from __future__ import annotations

import os
from pathlib import Path

from sgrna import config

# results of the extension work go into the fork, not the shared results folder
OUT = Path(os.environ.get("EXT_OUT", config.RESULTS / "post_freeze"))
OUT.mkdir(parents=True, exist_ok=True)
INTERIM = config.INTERIM / "post_freeze"
INTERIM.mkdir(parents=True, exist_ok=True)

def _main_results() -> Path:
    """The main project's results folder, which holds files the fork does not track (crisprHAL's stored predictions).

    Two layouts exist and the old code assumed only one. In the original checkout `config.ROOT` is a subfolder of the main
    project, so the folder is `ROOT.parent / "results"`; under `run_ext.sh`, `RSI09_ROOT` names the main project itself,
    so it is `ROOT / "results"`. Taking `ROOT.parent` unconditionally pointed at `Projects/results`, which does not exist,
    and made every module that reads those predictions fail with "No objects to concatenate".
    """
    for cand in (config.ROOT / "results", config.ROOT.parent / "results"):
        if (cand / "sample_complexity.csv").exists() or any(cand.glob("preds_crisprhal_*")):
            return cand
    return config.ROOT / "results"       # not found: do NOT raise here -- this runs at import, and a clone without the
                                         # untracked prediction files must still import every module. `stored()` raises.


def stored(pattern: str) -> list[Path]:
    """Files of the main project's results folder matching `pattern`, with a message that says what is missing."""
    files = sorted(MAIN_RESULTS.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no file matching {pattern!r} in {MAIN_RESULTS}. These are crisprHAL 2's stored predictions "
                                f"and the main project's sample-complexity table; the fork does not track them. Point "
                                f"RSI09_ROOT at the main project (the folder that holds results/ and data/).")
    return files


MAIN_RESULTS = _main_results()
EXTERNAL = config.ROOT / "external_data"
