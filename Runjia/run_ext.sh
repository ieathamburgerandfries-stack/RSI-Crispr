#!/usr/bin/env bash
# Run any module of `sgrna.ext` with the environment it needs.
#
# The extension modules need three things set together, and getting one wrong fails with a FileNotFoundError that does not
# say which: `sgrna/config.py` finds the project root by walking up from `src/sgrna/`, which guesses wrong from this fork's
# `Runjia/sgrna/` layout, so RSI09_ROOT has to name the folder that actually holds `data/` and `external_data/`; EXT_OUT has
# to name this fork's `results/ext`, or results are written beside the wrong tree; and PYTHONPATH has to include `Runjia`.
# The interpreter also lives outside both repositories. Override it with PY=... if yours is elsewhere.
#
#   ./run_ext.sh sgrna.ext.verify_numbers
#   ./run_ext.sh sgrna.ext.family_a_at_tuned --seeds 41
#   PY=/path/to/python ./run_ext.sh sgrna.ext.tune_split2 --draws 40
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # .../RSI-Crispr-fork/Runjia
FORK="$(dirname "$HERE")"
PROJECT="$(dirname "$FORK")"

PY="${PY:-$PROJECT/.venv/bin/python}"
if [ ! -x "$PY" ]; then
  echo "run_ext.sh: no usable interpreter at $PY" >&2
  echo "            set PY=/path/to/python and try again" >&2
  exit 1
fi
if [ ! -d "$PROJECT/data" ]; then
  echo "run_ext.sh: no data/ under $PROJECT -- RSI09_ROOT must name the folder holding data/ and external_data/" >&2
  exit 1
fi
if [ $# -eq 0 ]; then
  echo "usage: ./run_ext.sh <module> [args...]        e.g. ./run_ext.sh sgrna.ext.verify_numbers" >&2
  exit 2
fi

export RSI09_ROOT="$PROJECT"
export EXT_OUT="$HERE/results/ext"
export PYTHONPATH="$HERE"
exec "$PY" -m "$@"
