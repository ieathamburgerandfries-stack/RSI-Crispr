"""Run one audit module on its own, with the same pass rule as the whole audit.

`./run_audit.sh` calls every sub-audit's `run(chk)` with the one `chk` of `verify_numbers`. To run a single one (`./run_ext.sh sgrna.paper.report_checks_R4`) its `__main__` block calls
`run_alone(run, "report")`, which supplies an equivalent `chk` (a check passes when |computed - claimed| is at most half a unit in the last quoted digit) and exits non-zero on a mismatch. This
replaced fifteen copies of the same driver, each with its own print width.

Report section: Part 4.
"""
from __future__ import annotations

import sys


def run_alone(run, noun: str = "report") -> None:
    """Call `run(chk)`, print one line per check and the totals, and exit 1 if any check failed."""
    n, bad = 0, []

    def chk(label, got, claimed, dec):
        nonlocal n
        n += 1
        ok = abs(got - claimed) <= 0.5 * 10 ** (-dec) + 1e-9
        if not ok:
            bad.append((label, got, claimed))
        print(f"  {'ok ' if ok else 'BAD'} {label:100s} computed {got:+.5f}  {noun} {claimed:+.{max(dec, 0)}f}")

    run(chk)
    print(f"\n{n} checks, {len(bad)} mismatches")
    for b in bad:
        print("  MISMATCH", b)
    sys.exit(1 if bad else 0)
