# `results/`

Every `.csv` the report cites, split by whether it predates the freeze.

* **`results/*.csv`** — the frozen analysis, 11 October 2026. Findings 1–7 and Appendices A–C.
  Nothing in here is regenerated; the numbers in `PROGRESS_REPORT.md` Parts 1–5 come from these files.
* **`results/post_freeze/`** — everything measured after the freeze: Findings 8–9, the hyperparameter
  search, the re-comparison against crisprHAL 2, the family-A-at-tuned-settings result, the human
  twin-grouped re-run, and the second gene split. `EXTENSION_LOG.md` sections 7–17 are the record.

`../run_audit.sh` recomputes every post-freeze number quoted in the report from the files here and
exits non-zero on any mismatch (682 checks as of 20 November 2026).

Files are named after the module that writes them. `.npy` prediction caches are gitignored and
regenerable by re-running the module that produced them.
