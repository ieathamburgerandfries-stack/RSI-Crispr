"""Audit of the numbers added to the report by the line-by-line review of 23 Nov 2026 (modules R1 to R8 and CIT).

    ./run_ext.sh sgrna.paper.report_numbers

Each `report_checks_*` module recomputes the quoted numbers of one part of PROGRESS_REPORT.md from stored result files (or, where a
result is derived in a few seconds from a stored file, from that file) and states which cohort and protocol the number belongs to.
`verify_numbers` calls `run`, so `./run_audit.sh` covers them. R3 refits the flank-edit models and the ALDEx2 re-implementation (about two minutes together); R2's SpCas9 to eSpCas9 transfer refit
(about a minute) is stored in results/transfer_esp_unseen_guides.csv and re-run, overwriting it, only when the environment
variable RSI_AUDIT_HEAVY is set:

    RSI_AUDIT_HEAVY=1 ./run_ext.sh sgrna.paper.report_numbers
"""
from __future__ import annotations

from . import (report_checks_CIT, report_checks_R1, report_checks_R2, report_checks_R3, report_checks_R4,
               report_checks_R5, report_checks_R6, report_checks_R7, report_checks_R8)

MODULES = (report_checks_CIT, report_checks_R1, report_checks_R2, report_checks_R3, report_checks_R4,
           report_checks_R5, report_checks_R6, report_checks_R7, report_checks_R8)


def run(chk):
    for m in MODULES:
        m.run(chk)


if __name__ == "__main__":
    from .standalone import run_alone
    run_alone(run, "report")
