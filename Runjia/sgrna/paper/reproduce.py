"""Do the modules still write the files the audit reads? Re-run each resumable module from its stored rows and fail if any tracked result changes.

The audit (`verify_numbers`) checks that the numbers in the documents equal the numbers in the result files. This is its counterpart for the *code*: that the module which wrote a result file still writes
that file, byte for byte, when run again. It is the check that proves a merge, a rename or a refactor changed no result, and before this module it was done by hand.

Each case in `CASES` is one command (a module and its arguments). It is run against a scratch copy of `results/`, never against the tracked files, so a failing case cannot damage them. A case passes when
(1) it exits 0, (2) every file it is listed to write was actually rewritten (a command that quietly does nothing must not pass), and (3) no file of the scratch tree differs from the tracked one. A case whose output
is printed rather than written names the file the output must equal (`stdout`). Only cases that finish in minutes from stored rows and caches belong here; a module that times something, or refits a model from
scratch, cannot reproduce bytes and does not.

    ./run_audit.sh --reproduce                  # every case
    ./run_audit.sh --reproduce --only tune      # the cases whose name contains `tune`
    ./run_audit.sh --reproduce --list

To add a case: append a dict to `CASES` with the command, the result files it writes (relative to `results/`, written out in full), and run it; if it differs, it does not belong.

Report section: Part 4.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import common

RUNJIA = Path(common.OUT).parent            # the folder that holds sgrna/ and results/
RESULTS = RUNJIA / "results"

CASES = [
    dict(name="tune-analysis", cmd=["sgrna.tuning.tune_ecoli", "--analyze"], stdout="tune_stage1_effects.txt"),
    dict(name="tune-confirm-split1", cmd=["sgrna.tuning.tune_ecoli", "--confirm"], writes=["tune_confirm.csv"]),
    dict(name="tune-confirm-split2", cmd=["sgrna.tuning.tune_ecoli", "--split", "2", "--confirm"], writes=["tune_split2_confirm.csv"]),
    dict(name="ablation-family-a-at-tuned", cmd=["sgrna.cohorts.tuned_ablation", "--design", "family_a", "--summarise-only"], writes=["family_a_at_tuned_summary.csv"]),
    dict(name="ablation-local-sequence", cmd=["sgrna.cohorts.tuned_ablation", "--design", "local_sequence", "--summarise-only"], writes=["local_sequence_bound_summary.csv"]),
    dict(name="family-a-by-organism", cmd=["sgrna.compare.family_a_by_organism"], writes=["family_a_by_organism.csv"]),
    dict(name="ablation-paper-folds", cmd=["sgrna.cohorts.curated_ablation", "--paper-folds", "--summary-only"], writes=["ablation_paper_folds_summary.csv"]),
    dict(name="flank-scale-asymmetric", cmd=["sgrna.interpret.flank_scale", "--parts", "asymmetric", "--summary-only"], writes=["flank_window_tune_summary.csv"]),
    dict(name="flank-scale-windows", cmd=["sgrna.interpret.flank_scale", "--summary-only"], writes=["flank_scale_summary.csv"]),
    dict(name="human-additivity-twin", cmd=["sgrna.compare.human_additivity_twin", "--summarise-only"], writes=["human_additivity_twin_summary.csv"]),
    dict(name="design-check", cmd=["sgrna.design.check_design"], writes=["design_check.csv"]),
    dict(name="design-benchmark", cmd=["sgrna.design.design_benchmark"], writes=["design_benchmark.csv"]),
]


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tree(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in sorted(root.rglob("*")) if p.is_file()}


def run_case(case: dict) -> tuple[bool, str]:
    """(passed, one-line detail) for one case, run against a scratch copy of `results/`."""
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "results"
        shutil.copytree(RESULTS, scratch, copy_function=shutil.copy2)          # copy2 keeps mtimes, so "was it rewritten" is a comparison
        before = {k: p.stat().st_mtime_ns for k, p in _tree(scratch).items()}
        env = dict(os.environ, EXT_OUT=str(scratch))
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, "-m", *case["cmd"]], cwd=RUNJIA, env=env, capture_output=True, text=True)
        secs = time.perf_counter() - t0
        if proc.returncode != 0:
            return False, f"exit {proc.returncode} after {secs:.0f} s: {(proc.stderr.strip().splitlines() or ['no stderr'])[-1][:160]}"
        problems = []
        if "stdout" in case and proc.stdout.encode() != (RESULTS / case["stdout"]).read_bytes():
            problems.append(f"its output differs from {case['stdout']}")
        for rel in case.get("writes", []):
            f = scratch / rel
            if not f.exists() or f.stat().st_mtime_ns == before.get(rel):
                problems.append(f"{rel} was not rewritten")
        changed = []
        for rel, p in _tree(scratch).items():
            tracked = RESULTS / rel
            if tracked.exists() and _digest(p) != _digest(tracked):
                changed.append(rel)
        if changed:
            problems.append("differs from the tracked file: " + ", ".join(changed[:5]) + (" ..." if len(changed) > 5 else ""))
        return (not problems), (f"{secs:.0f} s" if not problems else "; ".join(problems))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", help="run the cases whose name contains one of these words")
    ap.add_argument("--list", action="store_true", help="list the cases and exit")
    a = ap.parse_args(argv)
    cases = [c for c in CASES if not a.only or any(w in c["name"] for w in a.only)]
    if a.list:
        for c in cases:
            print(f"{c['name']:30s} python -m {' '.join(c['cmd'])}")
        return 0
    failed = 0
    for c in cases:
        ok, detail = run_case(c)
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {c['name']:30s} {detail}", flush=True)
    print(f"\n{len(cases)} cases, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
