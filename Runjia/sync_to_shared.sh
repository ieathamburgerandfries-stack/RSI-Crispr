#!/usr/bin/env bash
# Publish the working folder's documents, code and cited results into the
# shared repository at RSI-Crispr/Runjia/.
#
# Why this exists: the shared repo is nested inside the working folder, so every
# document teammates read has to exist twice -- once here, once under Runjia/.
# Hand-copying is how the two drifted apart before. This makes the Runjia/ copy
# a *build artefact*: never edit it, edit the source and re-run this.
#
# It also fixes the one thing a plain copy cannot. README.md links to
# `docs/PROGRESS_REPORT.md`, which is right here and wrong in the mirror, where
# the report is a sibling. The link is rewritten on the way across.
#
#     ./sync_to_shared.sh            # copy, then show what changed
#     ./sync_to_shared.sh --check    # report drift, change nothing
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$ROOT/RSI-Crispr/Runjia"
CHECK=${1:-}

copy() {  # copy() <source, relative to ROOT> <dest, relative to DEST>
  local src="$ROOT/$1" dst="$DEST/$2"
  [ -e "$src" ] || { echo "  missing: $1" >&2; return 0; }
  mkdir -p "$(dirname "$dst")"
  if [ "$CHECK" = "--check" ]; then
    cmp -s "$src" "$dst" || echo "  drifted: $2"
  else
    cp "$src" "$dst"
  fi
}

# Documents. The report has no relative links; the README has four, and only
# the report one needs rewriting -- the other three are mirrored below.
copy docs/PROGRESS_REPORT.md PROGRESS_REPORT.md
copy results/RESULTS.md results/RESULTS.md
copy results/RESULTS_archive_2026-10-10.md results/RESULTS_archive_2026-10-10.md
copy lit_review/beyond_QCT_literature_review.md lit_review/beyond_QCT_literature_review.md

if [ "$CHECK" = "--check" ]; then
  sed 's#](docs/PROGRESS_REPORT.md)#](PROGRESS_REPORT.md)#g' "$ROOT/README.md" \
    | cmp -s - "$DEST/README.md" || echo "  drifted: README.md"
else
  sed 's#](docs/PROGRESS_REPORT.md)#](PROGRESS_REPORT.md)#g' "$ROOT/README.md" \
    > "$DEST/README.md"
fi

# Code. Every module, so a teammate can run anything the report cites.
for f in "$ROOT"/src/sgrna/*.py "$ROOT"/src/sgrna/README.md; do
  [ -e "$f" ] && copy "src/sgrna/$(basename "$f")" "sgrna/$(basename "$f")"
done
for f in "$ROOT"/src/sgrna/features/*.py; do
  [ -e "$f" ] && copy "src/sgrna/features/$(basename "$f")" "sgrna/features/$(basename "$f")"
done
# Scripts only -- colab/ also holds a 19 MB gzipped input that has no business
# in a git repository.
for f in "$ROOT"/colab/*.py "$ROOT"/colab/*.ipynb "$ROOT"/colab/*.md; do
  [ -f "$f" ] && copy "colab/$(basename "$f")" "colab/$(basename "$f")"
done

# Results: only the files RESULTS.md actually cites, not all 335.
cited=$(grep -o '`[A-Za-z0-9_.-]*\.csv`' "$ROOT/results/RESULTS.md" | tr -d '`' | sort -u)
for f in $cited; do
  [ -e "$ROOT/results/$f" ] && copy "results/$f" "results/$f"
done

# docs/qna_archive.md is deliberately NOT published: it is the local Q&A
# archive, and the shared history already holds every section it contains.

if [ "$CHECK" = "--check" ]; then
  echo "check complete (nothing above = in sync)"
else
  echo "synced -> RSI-Crispr/Runjia"
  git -C "$ROOT/RSI-Crispr" status --short Runjia | head -30
fi
