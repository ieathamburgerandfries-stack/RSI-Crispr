#!/usr/bin/env bash
# Recompute every number quoted in PROGRESS_REPORT.md for the extension work from the files that produced it.
# Prints one line per check and a count of mismatches; a non-zero exit means a quoted number no longer matches its file.
#
#   ./run_audit.sh              # all checks
#   ./run_audit.sh | tail -5    # just the total
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_ext.sh" sgrna.paper.verify_numbers "$@"
