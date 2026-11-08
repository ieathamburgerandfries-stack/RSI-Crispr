"""Shared paths for the extension work (the private-fork experiments).

Nothing here edits the frozen pipeline: new feature families live in `sgrna.ext`,
read the frozen modules, and write only under `results/ext/` and `data/interim/ext/`.
Run with:  RSI09_ROOT=<project root with data/>  PYTHONPATH=<fork>/Runjia
"""
from __future__ import annotations

import os
from pathlib import Path

from sgrna import config

# results of the extension work go into the fork, not the shared results folder
OUT = Path(os.environ.get("EXT_OUT", config.RESULTS / "ext"))
OUT.mkdir(parents=True, exist_ok=True)
INTERIM = config.INTERIM / "ext"
INTERIM.mkdir(parents=True, exist_ok=True)
EXTERNAL = config.ROOT / "external_data"
KIM_CSV = EXTERNAL / "kim2020_cas9_variants" / "kim2020_training_test.csv"
