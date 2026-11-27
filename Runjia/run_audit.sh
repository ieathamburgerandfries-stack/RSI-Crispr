#!/usr/bin/env bash
# Recompute every number quoted in PROGRESS_REPORT.md for the extension work from the files that produced it.
# Prints one line per check and a count of mismatches; a non-zero exit means a quoted number no longer matches its file.
#
#   ./run_audit.sh              # all checks
#   ./run_audit.sh | tail -5    # just the total
#   ./run_audit.sh --reproduce  # do the modules still write the tracked result files, byte for byte? (see sgrna/paper/reproduce.py)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${1:-}" = "--reproduce" ]; then       # the counterpart for the code: re-run the resumable modules, fail if a tracked result changes
  shift
  exec "$HERE/run_ext.sh" sgrna.paper.reproduce "$@"
fi
exec "$HERE/run_ext.sh" sgrna.paper.verify_numbers "$@"
