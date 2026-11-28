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
3. Three static scans of the source, with the standard library alone (`static_scan`): no name is read that is bound in no enclosing scope, in the module or in the builtins; no call of a function of the same module, or of one it
   imports from another module of the package, has arguments that do not fit the signature; no function or class is defined twice in a module. Importing a module cannot find these, because a name used only inside a function
   is looked up when the function runs: on 28 November 2026 two parts that had not run since a consolidation (`released` of the BoostMEC retraining, `--paper-folds` of the curated ablation) called a loader with the wrong number of
   arguments, and a merged module defined `paired` twice, the second hiding the first. None of them was found by the audit, which reads result files, or by the import sweep.

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


def static_scan(root: Path | None = None) -> dict:
    """{'undefined': [...], 'bad_calls': [...], 'duplicates': [...]} for every module under `sgrna/` (see the module docstring); each entry is a line of text naming file, line and the name."""
    import ast
    import builtins
    import symtable

    root = Path(root or Path(__file__).resolve().parents[1])
    builtin = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__builtins__", "__class__"}
    files = {p: p.read_text() for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts}
    trees = {p: ast.parse(src) for p, src in files.items()}

    def modname(p: Path) -> str:
        rel = ".".join(p.relative_to(root.parent).with_suffix("").parts)
        return rel[: -len(".__init__")] if rel.endswith(".__init__") else rel

    def signature(n):
        a = n.args
        return [x.arg for x in a.posonlyargs + a.args], len(a.defaults), [x.arg for x in a.kwonlyargs], a.vararg is not None, a.kwarg is not None

    sigs, names = {}, {}
    for p, t in trees.items():
        d = {}
        for n in t.body:
            if isinstance(n, ast.FunctionDef):
                d[n.name] = signature(n)
            elif isinstance(n, ast.ClassDef):
                init = next((m for m in n.body if isinstance(m, ast.FunctionDef) and m.name == "__init__"), None)
                if init:
                    pos, nd, kwo, va, kw = signature(init)
                    d[n.name] = (pos[1:], nd, kwo, va, kw)
        sigs[modname(p)] = d
        names[modname(p)] = p

    undefined, bad, dup = [], [], []
    for p, src in files.items():
        rel = p.relative_to(root.parent)
        # 1. names bound nowhere
        top = symtable.symtable(src, str(p), "exec")
        defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace() or s.is_declared_global()}

        def walk(t):
            for s in t.get_symbols():
                if t.get_type() != "module" and s.is_referenced() and s.is_global() and not s.is_declared_global() and s.get_name() not in defined and s.get_name() not in builtin:
                    undefined.append(f"{rel}: {s.get_name()} read in {t.get_name()}")
            for c in t.get_children():
                walk(c)
        walk(top)
        # 3. functions and classes defined twice at the top of a module
        seen = {}
        for n in trees[p].body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                if n.name in seen:
                    dup.append(f"{rel}: {n.name} defined at lines {seen[n.name]} and {n.lineno}")
                seen[n.name] = n.lineno
        # 2. calls that do not fit the callee: same-module functions, and functions imported from another module of the package
        cur = modname(p)
        imported, aliased, sources = {}, {}, {}
        for n in trees[p].body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                sources.setdefault(n.name, set()).add((cur, n.name))
        for n in ast.walk(trees[p]):
            if isinstance(n, ast.ImportFrom):
                if n.level:
                    base = cur.split(".") if p.name == "__init__.py" else cur.split(".")[:-1]
                    base = base[: len(base) - (n.level - 1)]
                    tgt = ".".join(base + ([n.module] if n.module else []))
                else:
                    tgt = n.module or ""
                for a in n.names:
                    al = a.asname or a.name
                    sources.setdefault(al, set()).add((tgt, a.name))
                    if tgt in sigs and a.name in sigs[tgt]:
                        imported.setdefault(al, set()).add((tgt, a.name))
                    if f"{tgt}.{a.name}" in sigs:
                        aliased.setdefault(al, set()).add(f"{tgt}.{a.name}")
        for fn in ast.walk(trees[p]):
            if not isinstance(fn, ast.FunctionDef):
                continue
            local = {x.arg for x in ast.walk(fn.args) if isinstance(x, ast.arg)} | {y.id for y in ast.walk(fn) if isinstance(y, ast.Name) and isinstance(y.ctx, ast.Store)}
            for c in ast.walk(fn):
                if not isinstance(c, ast.Call):
                    continue
                targets = []
                if isinstance(c.func, ast.Name) and c.func.id not in local and len(sources.get(c.func.id, ())) == 1:        # a name bound in two places (two imports of `load_block`, say) is ambiguous: not judged
                    if c.func.id in sigs[cur] and c.func.id not in imported:
                        targets = [(cur, c.func.id)]
                    elif c.func.id in imported:
                        targets = list(imported[c.func.id])
                elif isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name) and c.func.value.id in aliased and len(aliased[c.func.value.id]) == 1:
                    m = next(iter(aliased[c.func.value.id]))
                    if c.func.attr in sigs[m]:
                        targets = [(m, c.func.attr)]
                if not targets or any(isinstance(x, ast.Starred) for x in c.args) or any(k.arg is None for k in c.keywords):
                    continue
                (tm, tn), = targets
                pos, nd, kwo, va, kw = sigs[tm][tn]
                given, kws, why = len(c.args), [k.arg for k in c.keywords], []
                if given > len(pos) and not va:
                    why.append(f"{given} positional for {len(pos)}")
                why += [f"unknown keyword {k}" for k in kws if k not in pos and k not in kwo and not kw]
                missing = [r for i, r in enumerate(pos[: len(pos) - nd]) if i >= given and r not in kws]
                if missing:
                    why.append(f"missing {missing}")
                if why:
                    bad.append(f"{rel}:{c.lineno}: {tn}(...) of {tm}: {'; '.join(why)}")
    return dict(undefined=undefined, bad_calls=bad, duplicates=dup)


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
                excluded={k: v for k, v in EXCLUDE.items()}, static=static_scan())
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
        for k, label in (("undefined", "name bound nowhere"), ("bad_calls", "call that does not fit"), ("duplicates", "defined twice")):
            print(f"  static scan: {len(info['static'][k])} {label}")
            for line in info["static"][k]:
                print(f"    {line}")
        if write:
            print(f"written -> {OUTJSON}")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    info = run(not a.quiet)
    return 1 if info["fails"] or info["files_changed_by_import"] or any(info["static"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
