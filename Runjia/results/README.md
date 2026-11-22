# `results/`

**Start with [`INDEX.md`](INDEX.md)**: every file here, grouped by the section of the report it is evidence for, with the module
that wrote it and its columns (generated and checked by the audit). [`RESULTS.md`](RESULTS.md) is the other entry point: it goes
from each *claim* to the file and the command behind it.

The folders are named for when a file was made, not for what it shows:

* **`results/*.csv`**: the analysis frozen on 11 October 2026 (Findings 1–6, Appendices A–C). Not regenerated since.
* **`results/post_freeze/`**: everything measured after the freeze (Finding 7, Appendix B4, the head-to-head in Part 5, the
  hyperparameter search, the re-comparison against crisprHAL 2, the mechanism work, the publication figures in `figures/`).
  `EXTENSION_LOG.md` sections 7–22 are the record.
* **`results/rederived/`**: re-derivations of seven result files whose producing code had been lost, kept as the evidence that
  they can be rebuilt (`sgrna/controls/rederive.py`; see `REMOVED.md`).

Files are named after the module that writes them. The folders are not reorganised by concept because the code addresses these
paths by name (about 500 references in 98 modules) and the same names exist in the untracked main project's `results/`; the index
gives the conceptual view instead. `.npy` prediction caches are gitignored and regenerable by re-running the module that made them.

`../run_audit.sh` recomputes every post-freeze number quoted in the report from the files here and
exits non-zero on any mismatch (1052 checks as of 22 November 2026). It also checks that the count in
this sentence is the one it actually ran, in this file and in three others, so the four cannot drift
apart again. `sgrna/paper/import_check.py`, which it runs, asserts separately that every module
in the package imports and that importing one writes nothing.

Files are named after the module that writes them. `.npy` prediction caches are gitignored and
regenerable by re-running the module that produced them.
