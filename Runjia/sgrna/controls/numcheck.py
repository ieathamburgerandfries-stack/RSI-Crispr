"""Check every number in the documents against the result files.

Two layers, because the first alone is too weak:

  coverage    -- does this number exist anywhere in `results/`?  Catches
                 invented, mistyped or stale figures.
  attribution -- when a line cites a specific file, do that line's numbers
                 appear in THAT file?  Catches a correctly-copied number quoted
                 against the wrong experiment.

An earlier version checked literals only and flagged 95 attribution misses,
almost all of them Delta-rho claims -- a difference between two rows of the
cited file, which is not itself a literal in it. A checker that cannot see a
difference cannot check this report, since most of its claims ARE differences.

So each cited file contributes:

    its literals, every pairwise difference of its literals, and every ratio.

That is a wide net and will pass some numbers by coincidence, which is the right
trade: the output is a shortlist to read, not a verdict. It does not compute
means, so a figure that averages several rows will be flagged and is usually
fine; what it is for is the figure that belongs to a different experiment.

    python -m sgrna.numcheck
"""
from __future__ import annotations
import pathlib, re
from collections import defaultdict

ROOT = pathlib.Path("/Users/runjia/Claude/Projects/RSI09")
RES = ROOT / "results"
# The findings log is deliberately absent: it records what was believed when,
# so its superseded numbers are *supposed* to disagree with today's results.
DOCS = [ROOT / "docs/PROGRESS_REPORT.md",
        ROOT / "README.md", RES / "RESULTS.md", ROOT / "RSI-Crispr/README.md"]

# The report is one file but four audiences: Part 1 is the claims, the
# appendices are the evidence behind them, and a miss in each means something
# different. Split the report at its own headings so the counts are readable.
SECTIONS = {"## Part 1": "report Part 1", "## Part 2": "report Parts 2-5",
            "## Appendix A": "report appendices"}


def section_of(doc, line_no, offsets):
    if doc != "PROGRESS_REPORT.md":
        return doc
    name = "report front matter"
    for start, label in offsets:
        if line_no >= start:
            name = label
    return name

# A digit run followed by a letter is a commit hash or an identifier, not a
# measurement, so `136755f` must not read as 136755.
NUM = re.compile(r"(?<![\w.])([+-]?\d+(?:,\d{3})*(?:\.\d+)?)(?![\d\w])")
FILE = re.compile(r"`([A-Za-z0-9_.\-]+\.(?:csv|json))`")
# lines that are references, URLs or code, not claims
SKIP_LINE = re.compile(r"https?://|doi\.org|^\s*[|]?\s*`?python |^\s{4}")

def floats_in(path):
    try: txt = path.read_text(errors="ignore")
    except Exception: return []
    out = []
    for m in re.finditer(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt):
        try: out.append(float(m.group()))
        except ValueError: pass
    return out

def table(vals, derive=False):
    """Rounded lookup at 0..5 dp, optionally including differences and ratios."""
    d = defaultdict(set)
    uniq = sorted({round(v, 6) for v in vals})
    pool = list(uniq)
    if derive and len(uniq) <= 1200:
        small = [v for v in uniq if abs(v) < 1e4]
        for i, a in enumerate(small):
            for b in small[i + 1:]:
                pool.append(a - b)
                if abs(b) > 1e-9: pool.append(a / b)
                if abs(a) > 1e-9: pool.append(b / a)
    for v in pool:
        if abs(v) > 1e9: continue
        for k in range(6):
            d[k].add(round(v, k)); d[k].add(round(-v, k))
            d[k].add(round(100 * v, k))          # percentages
    return d

FILES = {p.name: floats_in(p) for p in RES.iterdir()
         if p.is_file() and p.suffix in (".csv", ".json")}
ALL_R = table([v for vs in FILES.values() for v in vs])
FILE_R = {k: table(v, derive=True) for k, v in FILES.items()}

def parse(tok):
    t = tok.replace(",", "").replace("−", "-")
    try: return float(t), (len(t.split(".")[1]) if "." in t else 0)
    except ValueError: return None, 0

def present(val, dec, tab): return round(val, dec) in tab[min(dec, 5)]

cov, attr = [], []
for doc in DOCS:
    lines = doc.read_text().splitlines()
    offsets = [(i, lbl) for i, l in enumerate(lines, 1)
               for h, lbl in SECTIONS.items() if l.startswith(h)]
    for ln, line in enumerate(lines, 1):
        if SKIP_LINE.search(line): continue
        cited = [f for f in FILE.findall(line) if f in FILE_R]
        for tok in NUM.findall(line):
            val, dec = parse(tok)
            if val is None: continue
            if dec == 0 and abs(val) < 1900: continue
            if dec == 1: continue
            where = section_of(doc.name, ln, offsets)
            if not present(val, dec, ALL_R):
                cov.append((where, doc.name, ln, tok, line.strip()[:95]))
            if cited and not any(present(val, dec, FILE_R[f]) for f in cited):
                attr.append((where, doc.name, ln, tok, ",".join(cited),
                             line.strip()[:95]))

from collections import Counter
print(f"result files indexed: {len(FILES)}")
print(f"documents: {len(DOCS)}")
print(f"\nCOVERAGE misses: {len(cov)}")
for k, v in Counter(r[0] for r in cov).most_common():
    print(f"   {v:4d}  {k}")
for r in cov: print(f"   {r[1]}:{r[2]}  {r[3]:>12}   {r[4]}")
print(f"\nATTRIBUTION misses: {len(attr)}")
for k, v in Counter(r[0] for r in attr).most_common():
    print(f"   {v:4d}  {k}")
for r in attr: print(f"   {r[1]}:{r[2]}  {r[3]:>12}  not in {r[4]}")


# --------------------------------------------------------------------------
# Scoped attribution for the report
# --------------------------------------------------------------------------
# The coverage pass above asks "does this number appear in ANY result file".
# Across 387 files that is 470,912 distinct values, so for anything with three
# significant figures or fewer the answer is yes by chance, and the check gives
# false assurance. It is how five hand-computed statistics reached the report
# with nothing on disk behind them on 8 November.
#
# This pass is stricter. It tracks which result files the surrounding prose
# actually cites, and checks each number only against those. A number that no
# nearby file can account for is an attribution miss; a number in a passage that
# names no file at all is *unscoped* -- not wrong, but not checkable either, and
# the count is worth watching.
SCOPE_RESET = re.compile(r"^#{2,4} ")


def report_attribution(path=None, window: int = 10, verbose: bool = True):
    path = path or (ROOT / "docs/PROGRESS_REPORT.md")
    lines = path.read_text(errors="ignore").splitlines()

    cache: dict[str, set] = {}
    def values_of(name):
        if name not in cache:
            hits = list(RES.rglob(name))
            vals = []
            for h in hits:
                vals += floats_in(h)
            cache[name] = {round(v, 6) for v in vals}
        return cache[name]

    scope: list[str] = []
    seen_at: dict[str, int] = {}
    misses, unscoped, checked = [], 0, 0
    # The report sets minus signs as U+2212, which `NUM` does not read as a
    # sign -- so "-0.633" was being captured as "0.633" and then failing to
    # match the stored -0.6333. Normalise before matching.
    lines = [ln.replace("−", "-") for ln in lines]
    for i, line in enumerate(lines, start=1):
        if SCOPE_RESET.match(line):
            scope, seen_at = [], {}
        for f in FILE.findall(line):
            if f not in seen_at:
                scope.append(f)
            seen_at[f] = i
        scope = [f for f in scope if i - seen_at[f] <= window]
        if SKIP_LINE.search(line) or line.lstrip().startswith(">"):
            continue
        nums = [n.replace(",", "") for n in NUM.findall(line)]
        if not nums:
            continue
        if not scope:
            unscoped += len(nums)
            continue
        pool = set()
        for f in scope:
            pool |= values_of(f)
        for raw in nums:
            try:
                v = float(raw)
            except ValueError:
                continue
            # Only measurements, not the prose around them. A measurement in
            # this report carries a decimal point or an explicit sign; bare
            # integers are column counts, window widths, "20-mer", "10 kb",
            # percentages and years, and checking them produces noise rather
            # than findings.
            if "." not in raw and raw[0] not in "+-":
                continue
            if abs(v) >= 1900 and float(v).is_integer():
                continue
            checked += 1
            if not any(round(p, d) == round(v, d)
                       for d in range(0, 6) for p in pool):
                misses.append((i, raw, ",".join(scope[:3]), line.strip()[:80]))

    if verbose:
        print(f"scoped attribution over {path.name}")
        print(f"  numbers checked against a cited file : {checked:,}")
        print(f"  numbers in passages citing no file   : {unscoped:,}")
        print(f"  MISSES (no cited file accounts for it): {len(misses)}")
        for ln, raw, sc, txt in misses[:40]:
            print(f"   {ln:5d}  {raw:>10s}  [{sc[:46]}]  {txt}")
    return misses, unscoped, checked
