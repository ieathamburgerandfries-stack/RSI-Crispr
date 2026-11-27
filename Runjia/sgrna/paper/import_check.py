"""Every module in the package imports, and importing it writes nothing. A guard on a hazard that bit this project once.

Why this exists
---------------
Several modules here were written as scripts with their whole body at the top level. That is convenient to run and quietly
dangerous to import: `import sgrna.controls.equiv_check` refitted two models and **overwrote its own result file** with
timings measured on a loaded machine, and `import sgrna.paper.verify_numbers` ran the entire audit and then raised
`SystemExit` in the importing process. Both happened, on 20 November 2026, to a script that was only trying to check that
the package loads. The three offenders now keep their bodies in `main()`; this module is what stops the pattern coming back.

What it asserts
---------------
1. Every module under `sgrna/` imports without raising -- which is also the only check in this project that the code
   *loads*, as distinct from the audit's check that the numbers in the documents match their files.
2. The import sweep leaves `results/` byte-identical. Every file under it is hashed before and after.

The sweep runs in a subprocess, because the whole point is that an unsafe module may call `SystemExit` or write a file, and
neither should be able to reach the caller. `EXCLUDE` names the modules that are legitimately not importable, with the
reason; anything else that fails is a defect.

    ./run_ext.sh sgrna.paper.import_check

Report section: Part 4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from sgrna import config

from ..common import OUT

OUTJSON = OUT / "import_check.json"

# Modules that cannot be imported, and why. It is a standalone script rather than library code.
EXCLUDE = {
    "sgrna.compare.vs_crisprhal_wt_hal": "reads sys.argv[1] at the top level and needs the TensorFlow venv",
    "sgrna.interpret.hal_readout": "reads sys.argv at the top level and needs the TensorFlow venv",
}

_SWEEP = r"""
import importlib, json, pkgutil, sys, traceback
import sgrna
exclude = set(json.loads(sys.argv[1]))
names, fails = [], []
for m in pkgutil.walk_packages(sgrna.__path__, "sgrna."):
    names.append(m.name)
for name in sorted(names):
    if name in exclude:
        continue
    try:
        importlib.import_module(name)
    except BaseException as e:                      # SystemExit is a BaseException, and one module used to raise it
        fails.append([name, f"{type(e).__name__}: {e}".splitlines()[0][:200]])
print("@@" + json.dumps(dict(n_modules=len(names), n_skipped=len(exclude), fails=fails)))
"""


def _hash_tree(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def run(verbose: bool = True, write: bool = True) -> dict:
    """Import every module in a fresh interpreter and report what failed and which result files changed. `write=False` leaves the stored receipt (`import_check.json`) alone, which is how the audit calls it."""
    results = Path(config.RESULTS)
    before = _hash_tree(results)
    proc = subprocess.run([sys.executable, "-c", _SWEEP, json.dumps(sorted(EXCLUDE))],
                          capture_output=True, text=True)
    after = _hash_tree(results)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("@@")), None)
    if line is None:
        raise RuntimeError(f"sweep produced no result\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}")
    info = json.loads(line[2:])
    changed = sorted(set(before) ^ set(after)) + sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    info.update(files_in_results=len(before), files_changed_by_import=len(changed), changed=changed,
                excluded={k: v for k, v in EXCLUDE.items()})
    if write:
        OUTJSON.write_text(json.dumps(info, indent=1))
    if verbose:
        print(f"  {info['n_modules']} modules found, {info['n_skipped']} excluded by name, "
              f"{len(info['fails'])} failed to import")
        for n, e in info["fails"]:
            print(f"    FAIL {n}: {e}")
        print(f"  {info['files_in_results']} files under results/: "
              f"{info['files_changed_by_import']} changed by the sweep")
        for c in changed:
            print(f"    CHANGED {c}")
        if write:
            print(f"written -> {OUTJSON}")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    info = run(not a.quiet)
    return 1 if info["fails"] or info["files_changed_by_import"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
