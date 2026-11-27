# From the report to the paper

The paper is the point-form write-up in the project's Google Doc (tabs "Paper (point form)" and "Appendix (point form)"); `PROGRESS_REPORT.md` is the full report. They state the same results in different
words: the paper is concise, uses no internal vocabulary (no freeze, DEV or CONFIRM; "default" and "tuned" instead), and keeps one protocol for every effect, while the report keeps the
history. `PROGRESS_REPORT.md` Appendix E repeats the paper's Results, Discussion and appendix bullets as the Doc states them, so every number the paper states is also in the report. The paper is at most 7 pages including its introduction: Methods are two data bullets and one protocol sentence, and every protocol with the controls run on it is in Appendix A3.

**One protocol.** Five-fold cross-validation that holds whole genes out (all 33,567 curated guides; guides outside genes grouped in 100 kb blocks), three seeds, the default model (400 trees, 300 columns
chosen inside each training fold), the 20-mer alone as the baseline. The team's shared harness assigns rows at random (five seeds of five-fold `KFold`); only the fold assignment differs. Appendix A3 of the paper
lists the protocols (paper folds, the team's harness, whole genes of the genic guides, sealed genes, development and confirmation genes, crisprHAL 2's own splits, an unseen genome), says which control ran on which, and
compares the two fold designs; the family ablation was run on both (`ablation_curated.csv`, `ablation_paper_folds.csv`).

**Where each part of the paper comes from.**

| paper | report | result files | audit |
|---|---|---|---|
| Methods: data, curation, genes, protocol | Part 4, Appendices C0, C, B2a | `curated_ceiling.json`, `cohort_groups.csv`, `reference_genome_versions.csv` | `paper_numbers_A`, `paper_numbers_D` |
| Results: the guide-derived families | Finding 1, Appendix B3 | `portability.csv`, `ablation_paper_folds_summary.csv`, `ablation_curated.csv` | `paper_numbers_A` [10], [11]; `paper_numbers_D` [D25] |
| Results: flank features add 0.161, windows, displacement, controls | Findings 2 and 3 | `flank_scale.csv`, `flank_scale_summary.csv`, `sealed_holdout*.csv`, `mappability_*.csv` | `paper_numbers_D` [D4] to [D6], [D24], [D27] |
| Results: the shape of the flank effect, SHAP | Findings 5 and 7 | `signal_*.csv`, `compact_model_*.csv`, `gate_test.csv` | `paper_numbers_D` [D7], [D8], [D26] |
| Results: the replicate ceiling | Finding 6 | `curated_ceiling.json`, `ceiling_parametric.csv` | `paper_numbers_A` |
| Results: SLICER, crisprHAL 2 and BoostMEC | Part 5, Appendix B | `vs_crisprhal_*.csv`, `gene_grouped_vs_crisprhal.csv`, `boostmec_*.csv`, `crisprhal_matched_folds.csv`, `tuning.csv` | `paper_numbers_D` [D13] to [D22], [D26] |
| Results: cost by hardware | Part 5 (cost), Appendix B | `cost_repeats_*.csv`, `cost_gpu.csv`, `model_size.csv`, `boostmec_shipped_*.csv` | `paper_numbers_D` [D15] to [D21], [D26] |
| Appendices A1 to A6 | Findings 2, 5, 7, Appendices A to C | as above, plus `transfer*.csv`, `eukaryote_*.csv` | `paper_numbers_A` to `D` |

**Figures.** The paper has four main figures (1 reach, 2 shape, 3 replicate ceiling and accuracy, 4 cost) and four in the appendix (A1 composition, A2 input width, A3 transfer matrix, A4 position-20 rule) and one table (Table A2: tuned SLICER against crisprHAL 2 on every accuracy measure, the two splits side by side), all drawn from stored result
files by `sgrna/paper/figures_paper.py`. The report's own figures are drawn by `sgrna/paper/figures.py`.

**Reproducibility.** `run_audit.sh` (its last line is the number of checks it ran); `import_check.py` for the code itself. Rounding and coverage: every number in the paper's text is matched to an audit claim or is a setting, a seed or a published constant.
