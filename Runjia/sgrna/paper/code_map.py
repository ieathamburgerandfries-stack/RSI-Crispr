"""Generate the per-module tables of CODE_MAP.md from the code itself, so the map cannot drift from the files.

For every module under `sgrna/` it reports: its size, what its own docstring says it does, the result files it writes
(string literals that name an existing file in `results/`, including the literal prefix of an f-string), and the section
of `PROGRESS_REPORT.md` that first names the module or one of those files. The narrative parts of CODE_MAP.md -- the
shape of the repository, the finding-to-code index, what trimming the report would free up -- are written by hand
between the markers and are never touched.

    ./run_ext.sh sgrna.paper.code_map            # rewrite the generated tables in CODE_MAP.md
    ./run_ext.sh sgrna.paper.code_map --check    # exit 1 if a module is missing from the map (the audit runs this)

Report section: Part 4.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
from pathlib import Path

from .. import common

RUNJIA = Path(common.OUT).parent.parent
PKG = RUNJIA / "sgrna"
MAP = RUNJIA / "CODE_MAP.md"
BEGIN, END = "<!-- generated:begin -->", "<!-- generated:end -->"

PACKAGES = [
    ("", "Three single files at the root of the package, not folders: `config.py` (project paths and shared constants), `io_utils.py` (reading the base matrix and writing feature blocks) and `common.py` (output folders and the resolver for results this repository does not track). Nearly every other module imports them."),
    ("build", "Decode the published matrix, locate every guide on the genome, rebuild features for guides outside it, and the array featurisers that score a guide in tens of microseconds."),
    ("features", "The eleven feature families A-J (family A, the flanking-DNA composition, is the one that matters) and the shape / 3D descriptors that Appendix B4 tested against it."),
    ("harness", "The leakage-safe cross-validation harness every experiment runs through, plus the notebook front end."),
    ("tuning", None),
    ("cohorts", "Ablations on the curated cohort and the arguments about what its numbers mean (ceilings, inflation, label noise), including the shape ablations of Appendix B4."),
    ("controls", "The adversarial phase: nulls, shuffles, floors, selector and leakage audits, and the check that the fast featurisers equal the originals."),
    ("interpret", "What the model uses and what the signal is made of: importance, attribution, redundancy, the mechanism work of Finding 7, and what the tuned settings say."),
    ("compare", "Every comparison with something else (Part 5 of the report): crisprHAL 2, the inherited baseline, other organisms and the human matrix, and other assays (the dCas9 boundary experiment of Appendix B5)."),
    ("cost", "What each model costs to train and to run, measured rather than estimated."),
    ("design", "The tool the project is for: give it a genome and a cut site and it returns the SpCas9 guide SLICER predicts will cut best there, with the shipped models, the check that it reads the same DNA and builds the same columns as the training matrices, and a benchmark of picking by score. It also holds the generator of the family A table for any genome (FASTA in, the 348 flank columns of every NGG guide out) and its check against the released table."),
    ("paper", None),
]


# A module whose conclusion the report states without naming the module or its files carries its section in its own docstring, on a line of its own:
#
#     Report section: Part 5.
#
# so the tag moves with the file when it is renamed, merged or moved. It is read only when the automatic lookup finds nothing, so a name in the report always wins.
_TAG = re.compile(r"^Report section: (.+?)\.?\s*$", re.M)


def _tag(tree: ast.Module) -> str | None:
    m = _TAG.search(ast.get_docstring(tree) or "")
    return m.group(1) if m else None


# Modules that read result files rather than write them, or that are not experiments at all.
TOOLING = {"verify_numbers", "figures", "import_check", "doc_refs", "code_map", "hooks", "reproduce"}
LIBRARY_PACKAGES = {"", "build", "features", "harness"}      # used by the experiments; not claims in themselves


def _doc(tree: ast.Module) -> str:
    """The first sentence of a module's docstring, whole. Only a very long sentence is shortened, and then at a clause boundary
    with an explicit ellipsis, never in the middle of a word."""
    d = (ast.get_docstring(tree) or "").strip()
    para = " ".join(d.split("\n\n")[0].split())
    head = para
    # end of the first sentence, skipping abbreviations and initials ("E. coli", "et al.", "e.g.") that are not sentence ends
    for m in re.finditer(r"[.?!](?=\s|$)", para):
        h = para[:m.end()]
        last = re.findall(r"[A-Za-z.]+[.?!]$", h)
        word = last[0].rstrip(".?!") if last else ""
        if m.end() < 20 or word in {"E", "e.g", "i.e", "al", "vs", "Fig", "et", "cf", "no", "approx"} or (len(word) == 1 and word.isupper()):
            continue
        head = h
        break
    head = head.replace("|", "/")
    LIMIT = 230
    if len(head) <= LIMIT:
        return head
    cut = max(head.rfind(sep, 0, LIMIT) for sep in (", ", "; ", " -- ", " — ", ": ", " ("))
    if cut < 60:
        cut = head.rfind(" ", 0, LIMIT)
    return head[:cut].rstrip(" ,;:-—(") + " …"


def _results(tree: ast.Module, existing: set[str]) -> list[str]:
    out = set()
    # a string that is a folder in a path (`INTERIM / f"transfer_{name}" / "meta.json"`) names no result file; its pieces would otherwise
    # prefix-match every file that begins `transfer_`
    folder = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Div):
            mid = node.left.right
            folder.update(id(x) for x in ast.walk(mid) if isinstance(x, (ast.Constant, ast.JoinedStr)))
    for node in ast.walk(tree):
        lits = []
        if id(node) in folder:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lits.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            lits.append("".join(v.value for v in node.values if isinstance(v, ast.Constant)))
        for s in lits:
            base = s.split("/")[-1]
            if len(base) < 5 or base in (".csv", ".json"):
                continue
            pre = base.split("{")[0]
            for f in existing:
                # a literal with a `{` in it is a template (`f"transfer_{name}.csv"`): its text before the brace must be long enough to name something, or a Docker
                # `--format '{{.Id}}'` literal (empty prefix) would match every result file
                if f == base or (len(base) >= 6 and (base.endswith("_") or "{" in s) and len(pre) >= 4 and f.startswith(pre)):
                    out.add(f)
    return sorted(out)


TITLES: dict[str, str] = {}


BULK = ("preds_", "folds_", "ablation_")        # prediction dumps and per-fold records: intentionally not all tracked


def untracked_summaries() -> list[tuple[str, list[str], int]]:
    """Small result files a module writes that exist in the main project's `results/` but not in this repository.

    Exact file names only (a prefix like `folds_` would match hundreds of unrelated records), and not the bulk families
    above. These are the summaries whose conclusions the report states but which a clone of the repository cannot verify.
    """
    main = Path(common.MAIN_RESULTS)
    if not main.is_dir():
        return []
    sizes = {p.name: p.stat().st_size for p in main.rglob("*") if p.is_file() and p.suffix in (".csv", ".json")}
    tracked = {p.name for p in (RUNJIA / "results").rglob("*") if p.is_file()}        # on disk here, whether or not yet committed
    out = []
    for f in sorted(PKG.rglob("*.py")):
        if f.name == "__init__.py" or f.stem in TOOLING | {"rederive"}:        # readers, and a duplicate writer of one file
            continue
        lits = {n.value for n in ast.walk(ast.parse(f.read_text())) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        miss = sorted(l.split("/")[-1] for l in lits
                      if l.split("/")[-1] in sizes and l.split("/")[-1] not in tracked and not l.split("/")[-1].startswith(BULK))
        if miss:
            out.append((f.stem, miss, sum(sizes[m] for m in miss)))
    return out


def _headings(report: str) -> list[tuple[int, str]]:
    hs = []
    for i, ln in enumerate(report.split("\n"), 1):
        m = re.match(r"^### (\d+)\. (.+)$", ln)
        if m:
            hs.append((i, f"Finding {m.group(1)}")); TITLES[f"Finding {m.group(1)}"] = m.group(2).strip().rstrip("."); continue
        m = re.match(r"^## ((?:Part \d+|Appendix [A-Z0-9a-z]+)) ?[—-]? ?(.*)$", ln)
        if m:
            hs.append((i, m.group(1))); TITLES.setdefault(m.group(1), m.group(2).strip())
    return hs


def _section(report: str, hs, names: list[str]) -> str:
    """The report section that cites this module (or its result files) most often; ties go to the earliest section."""
    import collections
    count: collections.Counter = collections.Counter()
    first: dict[str, int] = {}
    for n in names:
        # a stem that is also an ordinary English word ("mechanism", "importance", "transfer") only counts when it is
        # written as code; a stem with an underscore or a digit is distinctive enough to count anywhere
        if "_" in n or any(c.isdigit() for c in n) or n.endswith((".csv", ".json")):
            pat = r"(?<![A-Za-z0-9_])" + re.escape(n) + r"(?![A-Za-z0-9_])"
        else:
            pat = r"`[^`\n]*(?<![A-Za-z0-9_])" + re.escape(n) + r"(?:\.py)?(?![A-Za-z0-9_])[^`\n]*`"
        for m in re.finditer(pat, report):
            ln = report.count("\n", 0, m.start()) + 1
            label = None
            for hl, h in hs:
                if hl <= ln:
                    label = h
            if label is None:                       # the summary above Part 1 belongs to no section
                continue
            count[label] += 1
            first[label] = min(first.get(label, ln), ln)
    if not count:
        return "—"
    return sorted(count, key=lambda k: (-count[k], first[k]))[0]


def rows() -> dict[str, list[dict]]:
    existing = {Path(f).name for f in subprocess.check_output(["git", "ls-files", "results"], text=True, cwd=RUNJIA).split("\n")
                if f.endswith((".csv", ".json"))}
    report = (RUNJIA / "PROGRESS_REPORT.md").read_text()
    hs = _headings(report)
    out: dict[str, list[dict]] = {}
    for pkg, _ in PACKAGES:
        d = PKG / pkg
        items = []
        for f in sorted(d.glob("*.py")):
            if f.name == "__init__.py":
                continue
            tree = ast.parse(f.read_text())
            res = _results(tree, existing)
            names = [f.stem] + res                  # full file names (representation.csv) are distinctive; bare stems are not
            where = _section(report, hs, names)
            if f.stem in TOOLING:
                where, res = "tooling", []
            elif where == "—":
                where = _tag(tree) or ("library" if pkg in LIBRARY_PACKAGES else "—")
            items.append(dict(module=f.stem, lines=f.read_text().count("\n") + 1, doc=_doc(tree), results=res, where=where))
        out[pkg] = items
    return out


def _sort_key(label: str):
    m = re.match(r"Finding (\d+)", label)
    if m: return (0, int(m.group(1)), "")
    m = re.match(r"Part (\d+)", label)
    if m: return (1, int(m.group(1)), "")
    if label.startswith("Appendix"): return (2, 0, label)
    return (3 if label == "tooling" else 4 if label == "library" else 5, 0, label)


def by_section(data: dict[str, list[dict]]) -> str:
    """What each section of the report costs in code: the lever for deciding what to cut."""
    groups: dict[str, list[tuple[str, dict]]] = {}
    for pkg, items in data.items():
        for i in items:
            groups.setdefault(i["where"], []).append((pkg, i))
    lines = ["| report section | what it is | modules | lines | result files | where the code lives |", "|---|---|---:|---:|---:|---|"]
    for label in sorted(groups, key=_sort_key):
        L = groups[label]
        files = {r for _, i in L for r in i["results"]}
        pk = sorted({p or "(top)" for p, _ in L})
        title = TITLES.get(label) or {"—": "no section of the report points at it", "tooling": "the audit, the figures, the guards and this map",
                                       "library": "shared code the experiments are built on"}.get(label, "")
        lines.append(f"| {label} | {title.replace('|', '/')} | {len(L)} | {sum(i['lines'] for _, i in L):,} | {len(files)} | {', '.join(pk)} |")
    return "\n".join(lines)


def _untracked_table() -> str:
    rows_ = untracked_summaries()
    if not rows_:
        return "None: every result file a module writes by name is tracked, apart from the bulk prediction and per-fold records."
    total = sum(kb for _, _, kb in rows_)
    lines = [f"{len(rows_)} modules write small summary files that exist in the main project's untracked `results/` folder and "
             f"not here ({sum(len(f) for _, f, _ in rows_)} files, {total / 1024:.0f} KB). The report states these modules' conclusions, "
             "but a clone of this repository cannot verify them. Six such files were checked against the report and brought in on "
             "22 Nov 2026 (see `REMOVED.md`); the rest have not been, and copying them in unchecked would only make them look "
             "like evidence. Listed so the gap is visible; it is row 10 of the open-questions register in the report's Part 4.", "",
             "| module | files | KB |", "|---|---|---:|"]
    for m, fs, kb in rows_:
        lines.append(f"| `{m}` | {', '.join(f'`{x}`' for x in fs[:4])}{' +' + str(len(fs) - 4) if len(fs) > 4 else ''} | {kb / 1024:.1f} |")
    return "\n".join(lines)


def render(data: dict[str, list[dict]]) -> str:
    n_mod = sum(len(v) for v in data.values()); n_lines = sum(i["lines"] for v in data.values() for i in v)
    lines = [f"**{n_mod} modules, {n_lines:,} lines** of Python in {len(PACKAGES)} folders (not counting the `__init__.py` markers).", "",
             "#### By the report section each module supports", "", by_section(data), "",
             "Read the table above as the price list for the report: each claim carries its code and its result files with it. "
             "A module is placed in the section that cites it (or the result files it writes) most often. The `—` row, if "
             "there is one, is completed work that no section of the report points at. Each such module is listed in "
             "`REMOVED.md` under \"Considered for removal and kept\", with what it found; they were kept by decision, not by oversight.", "",
             "#### Result files that exist only outside this repository", "", _untracked_table(), "",
             "#### Package by package", ""]
    for pkg, blurb in PACKAGES:
        items = data[pkg]
        if blurb is None:
            init = (PKG / pkg / "__init__.py").read_text().strip().strip('"')
            blurb = init
        n, total = len(items), sum(i["lines"] for i in items)
        head = f"### `sgrna/{pkg}/`  —  {n} modules, {total:,} lines" if pkg else f"### `sgrna/` (the package root: {n} files, {total:,} lines)"
        lines += [head, "", blurb, "",
                  "| module | lines | what it does | reported in | writes |", "|---|---:|---|---|---|"]
        for i in items:
            res = ", ".join(f"`{r}`" for r in i["results"][:2]) + (f" +{len(i['results']) - 2}" if len(i["results"]) > 2 else "")
            name = i["module"] + (".py" if not pkg else "")
            lines.append(f"| `{name}` | {i['lines']} | {i['doc']} | {i['where']} | {res or '—'} |")
        lines.append("")
    return "\n".join(lines)


INDEX = RUNJIA / "results" / "INDEX.md"
# Result files that teammates added to the shared repository under `Runjia/results/`. They are not this project's and do not exist in the fork,
# so the index and the audit leave them alone.
TEAMMATE_RESULTS = ("folds_h6_", "folds_hstrand_", "folds_hdisc_")
INDEX_HEAD = """# Results, by what they are evidence for

**Generated** by `./run_ext.sh sgrna.paper.code_map` from the code and the report; do not edit. The audit fails if a tracked file in
`results/` is missing from this page.

The folders are named for *when* a file was made, not what it shows, so this page is the conceptual view. Files are grouped by the
section of `PROGRESS_REPORT.md` that cites them most (or, for a file the report does not name, that cites the module that writes it).
Within a group the order is alphabetical.

* **`results/`** holds the files made before the 11 October 2026 freeze, **`results/post_freeze/`** those made after it (including
  the seven figures in `figures/`), and **`results/rederived/`** the re-derivations of seven result files whose producing code had
  been lost, kept as evidence that they can be rebuilt (`sgrna/controls/rederive.py`). A file is named after the module that writes it.
* Why the folders are not simply reorganised by concept: the code reads and writes these paths by name (about 500 references in 98 modules),
  and the same names also exist in the untracked main project's `results/`, so a move here would have to be mirrored there. This index is
  the cheaper fix, and it is checked.
* **Role.** *audited*: a number from the file is recomputed by `./run_audit.sh`. *cited*: a document names the file. *receipt*: the
  file is written by a module whose conclusion the report states, without naming the file; it is the evidence for that sentence and
  can be regenerated by re-running the module. *figure* and *document* are what they say.
* All files are distinct: no two have identical bytes. (Files that teammates added to the shared repository, `folds_h6_*`, `folds_hstrand_*` and
  `folds_hdisc_*`, are not this project's and are not indexed.)
"""


def _csv_cols(f: Path) -> str:
    if f.suffix != ".csv":
        return ""
    try:
        head = f.read_text(errors="ignore").split("\n", 1)[0]
    except OSError:
        return ""
    cols = [c.strip().strip('"') for c in head.split(",")]
    return ", ".join(cols[:7]) + (" …" if len(cols) > 7 else "")


def results_index(data: dict[str, list[dict]]) -> str:
    report = (RUNJIA / "PROGRESS_REPORT.md").read_text()
    hs = _headings(report)
    docs = {f: (RUNJIA / f).read_text() for f in ("PROGRESS_REPORT.md", "results/RESULTS.md", "EXTENSION_LOG.md", "README.md",
                                                  "results/README.md", "PAPER_FLOW.md", "CODE_OVERVIEW.md")}
    audit = (PKG / "paper" / "verify_numbers.py").read_text()
    made_by: dict[str, list[dict]] = {}
    for items in data.values():
        for i in items:
            for r in i["results"]:
                made_by.setdefault(r, []).append(i)
    files = sorted(f for f in subprocess.check_output(["git", "ls-files", "results"], text=True, cwd=RUNJIA).split("\n")
                   if f and f != "results/INDEX.md" and not Path(f).name.startswith(TEAMMATE_RESULTS))
    groups: dict[str, list[tuple]] = {}
    for f in files:
        path = RUNJIA / f
        name = path.name
        mods = made_by.get(name, [])
        sec = _section(report, hs, [name])
        if sec == "—" and mods:
            sec = next((m["where"] for m in mods if m["where"] not in ("—", "library", "tooling")), "—")
        if f.startswith("results/rederived/"):
            role, sec = "audited" if name in audit else "cited", "re-derived result files (kept as evidence they can be rebuilt)"
        elif "/figures/" in f or name == "figure_manifest.csv":
            role, sec = "figure", "figures"
        elif path.suffix == ".md":
            role, sec = "document", "documents"
        elif name in audit:
            role = "audited"
        elif any(name in t for d, t in docs.items()):
            role = "cited"
        else:
            role = "receipt"
        if sec == "—":
            sec = "not placed in a section (kept by decision; see REMOVED.md)"
        groups.setdefault(sec, []).append((f, path.stat().st_size, role, ", ".join(f"`{m['module']}`" for m in mods) or "—", _csv_cols(path)))
    def order(label):
        if label in ("figures", "documents") or label.startswith(("re-derived", "not placed")):
            return (9, 0, label)
        return _sort_key(label)
    out = [INDEX_HEAD, f"**{len(files)} files** in {len(groups)} groups.", ""]
    for label in sorted(groups, key=order):
        title = TITLES.get(label, "")
        out += [f"### {label}" + (f" — {title}" if title else ""), "", "| file | KB | role | written by | columns |", "|---|---:|---|---|---|"]
        for f, size, role, mods, cols in groups[label]:
            out.append(f"| `{f.removeprefix('results/')}` | {size / 1024:.1f} | {role} | {mods} | {cols} |")
        out.append("")
    return "\n".join(out)



def moved() -> list[tuple[str, str]]:
    """(old path, current path) for every module that has lived at a flat path in git history and has since moved.

    Reorganisations produced these: the split of the flat `sgrna/` folder into themed subpackages, and the later move of
    `sgrna/post_freeze/` (earlier `sgrna/ext/`) and `sgrna/strand/` into those packages by function. Teammates' documents
    cite the flat paths, which is what this table is for.
    """
    log = subprocess.check_output(["git", "log", "--name-only", "--format=", "--", "Runjia/sgrna", "sgrna"], text=True, cwd=RUNJIA)
    old = sorted({re.sub(r"^Runjia/", "", ln) for ln in log.split("\n")
                  if re.match(r"^(Runjia/)?sgrna/(ext/|post_freeze/)?[A-Za-z0-9_]+\.py$", ln)})
    now: dict[str, list[str]] = {}
    for f in PKG.rglob("*.py"):
        if f.name != "__init__.py":
            now.setdefault(f.stem, []).append(str(f.relative_to(RUNJIA)))
    out = []
    for o in old:
        cand = now.get(Path(o).stem, [])
        if len(cand) == 1 and cand[0] != o and not (RUNJIA / o).exists():
            out.append((o, cand[0]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if a module is missing from CODE_MAP.md")
    a = ap.parse_args(argv)
    data = rows()
    all_mods = {i["module"] for items in data.values() for i in items}
    on_disk = {f.stem for f in PKG.rglob("*.py") if f.name != "__init__.py"}
    text = MAP.read_text() if MAP.exists() else ""
    missing = sorted(m for m in on_disk if f"| `{m}` |" not in text and f"| `{m}.py` |" not in text)
    if a.check:
        print(f"{len(on_disk)} modules on disk; {len(missing)} missing from CODE_MAP.md" + (f": {missing}" if missing else ""))
        return 1 if missing else 0
    assert on_disk == all_mods, f"module sets differ: {sorted(on_disk ^ all_mods)}"
    body = render(data)
    mv = moved()
    body += ("\n#### Old paths, for documents written before the reorganisations\n\n"
             f"<!-- docrefs:ignore:begin -->\n<details><summary>{len(mv)} modules have moved; documents that cite the left-hand paths (for example "
             "`Jacky/EXPERIMENT_CATALOG.md`) can be updated from this table</summary>\n\n"
             "| was | is now |\n|---|---|\n" + "\n".join(f"| `{o}` | `{n}` |" for o, n in mv) + "\n\n</details>\n<!-- docrefs:ignore:end -->\n")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{MAP} has no {BEGIN} ... {END} markers to write between")
    pre, rest = text.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    MAP.write_text(pre + BEGIN + "\n\n" + body + "\n" + END + post)
    INDEX.write_text(results_index(data) + "\n")
    print(f"wrote {len(all_mods)} modules in {len(PACKAGES)} packages -> {MAP}; results index -> {INDEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
