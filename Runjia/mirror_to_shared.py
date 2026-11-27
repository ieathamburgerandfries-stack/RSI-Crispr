#!/usr/bin/env python3
"""Mirror the committed changes of Runjia/ from the fork into the shared repository, checking every file first.

The same `Runjia/` folder lives in two repositories: the fork (where the work is done and committed first) and the shared one (which also holds teammates' folders and a few of their files
inside `Runjia/results/`). A plain copy is unsafe in both directions: it would write over a file someone changed in the shared repository, it never removes what the fork deleted, and it would
touch the teammates' files. This tool applies exactly the fork's change since the last mirror, and only if the shared repository still holds the fork's previous version of each file.

    python mirror_to_shared.py                      # dry run: the change set, and whether every precondition holds
    python mirror_to_shared.py --apply              # apply it and stage it in the shared repository; commit it yourself
    python mirror_to_shared.py --shared ../RSI-Crispr --base 767a8bf

The base (the fork commit that was last mirrored) is read from the shared repository's own history: the newest commit whose message says "mirror of fork <hash>". Name it with --base if that
is not the one you mean. The change set is `git diff BASE HEAD` of the fork under `Runjia/` and the top-level `README.md`; it uses committed state only, so commit in the fork first.

Preconditions, each a reason to stop with nothing changed: every file the fork modified or deleted is byte-identical (git blob) in the shared repository to the fork's version at BASE; every file
the fork added does not exist there yet; `Runjia/` in the shared repository has no uncommitted change. After `--apply` it compares the two trees: nothing may be missing or different, and the files
only the shared repository has (the teammates' `folds_*` files) are listed and never touched. Never use `sync_to_shared.sh`-style copying; that script was removed because it did none of this.
"""
from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import tarfile
from pathlib import Path

FORK = Path(__file__).resolve().parents[1]
SCOPE = ("Runjia", "README.md")
TEAMMATES = re.compile(r"^Runjia/results/folds_(h6|hstrand|hdisc)_")      # files only the shared repository holds; this tool never touches them


def git(repo: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check).stdout


def last_mirrored(shared: Path) -> str | None:
    for line in git(shared, "log", "-n", "40", "--format=%s%n%b%x00").split("\x00"):
        m = re.search(r"mirror of fork ([0-9a-f]{7,40})", line)
        if m:
            return m.group(1)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shared", type=Path, default=FORK.parent / "RSI-Crispr", help="the shared repository's checkout (default: ../RSI-Crispr next to the fork)")
    ap.add_argument("--base", help="the fork commit last mirrored (default: read from the shared repository's history)")
    ap.add_argument("--apply", action="store_true", help="make the change (stages it; does not commit)")
    a = ap.parse_args(argv)
    shared = a.shared.resolve()
    if not (shared / ".git").exists():
        print(f"error: {shared} is not a git checkout (use --shared)", file=sys.stderr)
        return 2
    base = a.base or last_mirrored(shared)
    if not base:
        print("error: no commit of the shared repository says 'mirror of fork <hash>'; name the base with --base", file=sys.stderr)
        return 2
    head = git(FORK, "rev-parse", "--short", "HEAD").strip()
    changes = [l.split("\t") for l in git(FORK, "diff", "--no-renames", "--name-status", base, "HEAD", "--", *SCOPE).splitlines() if l]
    modified = [p for s, p in changes if s == "M"]
    deleted = [p for s, p in changes if s == "D"]
    added = [p for s, p in changes if s == "A"]
    print(f"fork {head} against its last mirrored commit {base}: {len(modified)} modified, {len(deleted)} deleted, {len(added)} added")

    blocked = []
    dirty = git(shared, "status", "--porcelain", "--", "Runjia").strip()
    if dirty:
        blocked.append(("the shared repository has uncommitted changes under Runjia/", dirty.splitlines()[0]))
    for p in modified + deleted:
        f = shared / p
        if not f.exists():
            blocked.append(("missing in the shared repository", p))
        elif git(shared, "hash-object", str(f)).strip() != git(FORK, "rev-parse", f"{base}:{p}").strip():
            blocked.append(("the shared repository's file differs from the fork's version at the base", p))
    for p in added:
        if (shared / p).exists():
            blocked.append(("already exists in the shared repository", p))
    if blocked:
        for why, p in blocked[:20]:
            print(f"  BLOCKED: {why}: {p}")
        return 1
    print("all preconditions hold")
    if not a.apply:
        print("(dry run; --apply to make the change)")
        return 0

    for p in deleted:
        git(shared, "rm", "-q", "--", p)
    paths = modified + added
    if paths:
        tar = subprocess.run(["git", "-C", str(FORK), "archive", "HEAD", "--", *paths], capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
            tf.extractall(shared, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    git(shared, "add", "-A", "Runjia")
    if "README.md" in modified:
        git(shared, "add", "README.md")

    def blobs(repo: Path) -> dict[str, str]:
        return {l.split("\t")[1]: l.split()[1] for l in git(repo, "ls-files", "-s", "Runjia").splitlines() if l}

    f, s = blobs(FORK), blobs(shared)
    missing = [p for p in f if p not in s]
    differ = [p for p in f if p in s and f[p] != s[p]]
    extra = [p for p in s if p not in f]
    theirs = [p for p in extra if TEAMMATES.match(p)]
    unexpected = [p for p in extra if not TEAMMATES.match(p)]
    print(f"fork {len(f)} files, shared {len(s)}: missing in shared {len(missing)}, differing {len(differ)}, only in shared {len(extra)} "
          f"({len(theirs)} are the teammates' folds_* files, never touched; {len(unexpected)} other)")
    for p in (missing + differ)[:10]:
        print("  MISMATCH:", p)
    for p in unexpected[:10]:
        print("  only in the shared repository:", p)
    if missing or differ:
        return 1
    subject = git(FORK, "log", "-1", "--format=%s", "HEAD").strip()
    print(f'staged. Commit it in {shared}:\n  git -C {shared} commit -m "Runjia: {subject[:90]} (mirror of fork {head})"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
