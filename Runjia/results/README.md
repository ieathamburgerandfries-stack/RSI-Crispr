# `results/`

Every `.csv` the report cites, split by whether it predates the freeze.

* **`results/*.csv`** — the frozen analysis, 11 October 2026. Findings 1–6 and Appendices A–C.
  Nothing in here is regenerated; the numbers in `PROGRESS_REPORT.md` Parts 1–5 come from these files.
* **`results/post_freeze/`** — everything measured after the freeze: Finding 7, Appendix B4, the head-to-head in Part 6, the hyperparameter
  search, the re-comparison against crisprHAL 2, the family-A-at-tuned-settings result, the human
  twin-grouped re-run, the second gene split, the literature follow-ups, the mechanism work, the
  publication figures, and the per-locus featuriser. `EXTENSION_LOG.md` sections 7–22 are the record.

`../run_audit.sh` recomputes every post-freeze number quoted in the report from the files here and
exits non-zero on any mismatch (817 checks as of 20 November 2026). It also checks that the count in
this sentence is the one it actually ran, in this file and in three others, so the four cannot drift
apart again. `sgrna/post_freeze/paper/import_check.py`, which it runs, asserts separately that every module
in the package imports and that importing one writes nothing.

Files are named after the module that writes them. `.npy` prediction caches are gitignored and
regenerable by re-running the module that produced them.
