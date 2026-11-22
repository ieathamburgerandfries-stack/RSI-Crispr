"""Every module, path and Colab script that a document or a docstring names still exists.

Why this exists
---------------
A reorganisation moves files and does not move the sentences that name them. When the package was split into themed
subpackages, 221 references in 77 files went stale at once -- every frozen module's own "how to run this" line
(`python -m sgrna.curated_ablation`, now `sgrna.cohorts.curated_ablation`), the main results index, the README -- and
nothing noticed, because nothing checked.
The audit verifies that the *numbers* in the documents match their files; this is its counterpart for the *pointers*.

What counts as a reference
--------------------------
* a dotted module name beginning `sgrna.` (`python -m sgrna.cohorts.curated_ablation`, `import sgrna.harness.notebook`);
* a path `sgrna/.../x.py`, with or without a leading `Runjia/`;
* a Colab script `colab/x.py`.

A dotted name resolves if some prefix of it is a module or package and the remainder is an attribute of that module
(`sgrna.harness.notebook.setup`). Names starting with an underscore (`sgrna.__path__`) are attributes, not modules.

What is exempt, and why
-----------------------
* `REMOVED.md`, which exists to name things that are gone, and this file, which discusses references by example;
* `results/RESULTS_archive_*.md`, a dated snapshot of what was believed at the time, whose references are historically
  accurate and should not be rewritten to flatter the present;
* any reference that `REMOVED.md` itself lists as removed, so a docstring may say "folded in from colab/cell3_v3.py";
* text between `<!-- docrefs:ignore:begin -->` and `<!-- docrefs:ignore:end -->`, for a table that names old paths on purpose.

    ./run_ext.sh sgrna.paper.doc_refs            # list stale references, exit 1 if any
    ./run_ext.sh sgrna.paper.doc_refs --fix      # repair the ones whose target is unambiguous
"""
from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

from .. import common

RUNJIA = Path(common.OUT).parent.parent            # the folder that holds sgrna/, colab/ and the documents
PKG = RUNJIA / "sgrna"
EXEMPT_FILES = ("REMOVED.md", "doc_refs.py")          # this file discusses references by example, so it is not scanned
EXEMPT_GLOBS = ("results/RESULTS_archive_*.md",)

_IGNORE = re.compile(r"<!-- docrefs:ignore:begin -->.*?<!-- docrefs:ignore:end -->", re.S)
_PATH = re.compile(r"(?<![\w/.])((?:Runjia/)?sgrna/[A-Za-z0-9_/]+\.py)")
_DOT = re.compile(r"(?<![\w.])(sgrna(?:\.[A-Za-z_][A-Za-z_0-9]*)+)")
_COLAB = re.compile(r"(?<![\w/])(colab/[A-Za-z0-9_]+\.py)")


def _files() -> list[Path]:
    skip = {f for g in EXEMPT_GLOBS for f in RUNJIA.glob(g)}
    return [p for p in sorted(RUNJIA.rglob("*"))
            if p.is_file() and p.suffix in (".md", ".py", ".sh") and "__pycache__" not in p.parts
            and ".git" not in p.parts and p.name not in EXEMPT_FILES and p not in skip]


def _is_module(parts: list[str]) -> bool:
    p = PKG.joinpath(*parts)
    return p.with_suffix(".py").exists() or (p / "__init__.py").exists()


def _dotted_ok(name: str) -> bool:
    parts = name.split(".")[1:]
    if not parts or any(x.startswith("_") for x in parts):
        return True
    for k in range(len(parts), 0, -1):
        if _is_module(parts[:k]):
            return k == len(parts) or PKG.joinpath(*parts[:k]).with_suffix(".py").exists()   # remainder: attributes
    return False


def _removed() -> set[str]:
    f = RUNJIA / "REMOVED.md"
    return set(re.findall(r"[A-Za-z0-9_./]+\.py", f.read_text())) if f.exists() else set()


def stems() -> dict[str, list[str]]:
    """module stem -> every dotted path it has in the package (to repair `sgrna.stem` after a move)."""
    out = collections.defaultdict(list)
    for f in PKG.rglob("*.py"):
        if f.stem != "__init__":
            out[f.stem].append(".".join(("sgrna",) + f.relative_to(PKG).with_suffix("").parts))
    return out


def scan() -> list[tuple[str, int, str, str]]:
    """(file, line, kind, reference) for every reference that does not resolve."""
    gone = _removed()
    bad = []
    for f in _files():
        rel = str(f.relative_to(RUNJIA))
        text = _IGNORE.sub(lambda m: "\n" * m.group(0).count("\n"), f.read_text())      # blank the ignored span, keep line numbers
        for i, line in enumerate(text.split("\n"), 1):
            for m in _PATH.finditer(line):
                p = m.group(1).replace("Runjia/", "", 1)
                if not (RUNJIA / p).exists() and p not in gone and Path(p).name not in {Path(g).name for g in gone}:
                    bad.append((rel, i, "path", p))
            for m in _COLAB.finditer(line):
                if not (RUNJIA / m.group(1)).exists() and m.group(1) not in gone:
                    bad.append((rel, i, "colab", m.group(1)))
            for m in _DOT.finditer(line):
                if not _dotted_ok(m.group(1)):
                    bad.append((rel, i, "dotted", m.group(1)))
    return bad


def fix(bad) -> tuple[int, list]:
    """Repair `sgrna.stem[.attr]` and `sgrna/stem.py` where `stem` is exactly one module; return (n repaired, unresolved)."""
    idx = stems()
    by_file = collections.defaultdict(list)
    left = []
    for rel, ln, kind, ref in bad:
        by_file[rel].append((kind, ref))
    n = 0
    for rel, refs in by_file.items():
        f = RUNJIA / rel
        text = f.read_text()
        for kind, ref in sorted(set(refs), key=lambda r: -len(r[1])):
            new = None
            if kind == "dotted":
                parts = ref.split(".")
                if parts[1] == "ext":                                         # sgrna.ext.<stem>[.attr]
                    cand = idx.get(parts[2]) if len(parts) > 2 else None
                    if cand and len(cand) == 1:
                        new = ".".join([cand[0]] + parts[3:])
                else:
                    cand = idx.get(parts[1])
                    if cand and len(cand) == 1:
                        new = ".".join([cand[0]] + parts[2:])
            elif kind == "path":
                cand = idx.get(Path(ref).stem)
                if cand and len(cand) == 1:
                    new = ("Runjia/" if ref.startswith("Runjia/") else "") + cand[0].replace(".", "/") + ".py"
            if new and new != ref:
                text, k = re.subn(rf"(?<![\w.]){re.escape(ref)}(?![A-Za-z0-9_])" if kind == "dotted"
                                  else rf"(?<![\w/.]){re.escape(ref)}(?![A-Za-z0-9_])", new, text)
                n += k
            else:
                left.append((rel, kind, ref))
        f.write_text(text)
    return n, left


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fix", action="store_true", help="repair references whose target is exactly one module")
    a = ap.parse_args(argv)
    bad = scan()
    print(f"{len(bad)} stale references in {len({b[0] for b in bad})} files")
    if a.fix and bad:
        n, left = fix(bad)
        print(f"repaired {n} occurrences; {len(set(left))} distinct references have no single target")
        for rel, kind, ref in sorted(set(left)):
            print(f"   {rel:44s} {kind:6s} {ref}")
        bad = scan()
        print(f"after repair: {len(bad)} stale references")
    for rel, ln, kind, ref in bad:
        print(f"   {rel}:{ln}  {kind:6s} {ref}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
