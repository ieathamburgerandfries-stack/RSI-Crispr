# What was removed, and why

Everything here is recoverable. The state before the 22 November 2026 consolidation is the git tag
`pre-cleanup-2026-11-22`; restore any item with

```bash
git checkout pre-cleanup-2026-11-22 -- Runjia/<path>
```

This file exists so that nobody has to read history to find out what is missing: each entry says what the
item was, what it found, and why it is gone.

## The test that was applied

A result or a module stays if **any one** of these holds, and goes only if **none** does:

1. `PROGRESS_REPORT.md`, `results/RESULTS.md` or a README names it, or names a result file it writes;
2. `sgrna/paper/verify_numbers.py` (the audit) or `figures.py` reads it;
3. a module that stays imports it, launches it, or reads a file it writes — including files whose names are
   built at run time (`mix_configs_{kind}.json`, `folds_<tag>.csv`), which a plain text search cannot see;
4. a teammate's document cites it as evidence (`Jacky/EXPERIMENT_CATALOG.md` indexes much of `results/`).

**The result was much smaller than expected, and that is itself a finding.** Of 151 modules and 253 tracked
result files, only the entries below failed every test. The code base is not padded with dead experiments:
it is roughly a hundred one-question scripts, each of which is the provenance of a claim the report makes,
and the problem was navigability rather than bulk. That is why the rest of the consolidation is a
reorganisation (see `CODE_MAP.md`) rather than a deletion. One further check confirmed it: across the whole
package only one 8-line function is duplicated, so there is no copy-paste to merge away. (Corrected on 25 November: thirteen helpers are copied and 26 clusters of modules differ only by an arm table, a fold scheme or a split; see the last section of this file and `EXTENSION_LOG.md` section 33.)

If you want the repository smaller than this, the lever is not more deletion of orphans: it is deciding
which *claims* leave the report, because each claim carries its code and results with it. See "What
trimming the report would free up" at the end of `CODE_MAP.md`.

## Removed

| item | what it was | why it went | recover |
|---|---|---|---|
| `catboost_info/` (4 files, 155 KB) | CatBoost's own training log, written into the working directory whenever a CatBoost model is fitted there | tracked by mistake; contains no result. Now in `Runjia/.gitignore` | `Runjia/catboost_info` |
| `sgrna/interpret/decision_scale.py` | A Gaussian-copula translation of ρ into "how good is the guide you pick out of *k*" (draw a true quality and a prediction with Pearson `r = 2 sin(πρ/6)`, let the model choose among *k*, read off the percentile of the choice; chance is the 50th, an oracle that always picks the best of ten averages the 91st). **Never run**: it has no result file, and no document, module or audit check refers to it | an unfinished idea, not a result. The measured version of the same readout is already in the report (`panel.py`'s expected pick percentile), so the idealised one added nothing. If a ρ-to-decision translation is wanted for the paper, the formula above is the whole method | `Runjia/sgrna/interpret/decision_scale.py` |
| `colab/cell3_revised.py`, `colab/cell3_v3.py` | Two successive drafts of cell 3 of `colab/crisprhal_wide_window.py`, each a "paste over the previous cell 3" patch | **folded into `crisprhal_wide_window.py`**, whose cell 3 is now v3's. v1 trained through crisprHAL's own `train()`/`predict()`, which swallow their exceptions, so a failed 2,023 nt fit scored an untrained network (ρ 0.0434 in 0.36 min); v2 added failure checks and a batch fallback but still trained through those wrappers, so the swallowed exception hid the failure from its checks; v3 calls Keras directly. The report's wide-window table (batch 256) is v3's, and `results/RESULTS.md` named v1 as its producer, which was wrong and is now right. Keeping the two superseded drafts next to the working one invited running the one that scores garbage | `Runjia/colab/cell3_revised.py`, `Runjia/colab/cell3_v3.py` |
| `sgrna/cohorts/ceiling2.py` | An early ceiling estimate (4 Oct): fit a model to the *difference* between the WT-SpCas9 and eSpCas9 screens of the same 33,567 guides, to ask whether their disagreement is noise or biology. Its result (`ceiling_recomputed.json`, never tracked, still in the main project's untracked `results/`): ρ between screens 0.8095; a model predicts the disagreement at ρ 0.546, so **30% of it is systematic**; ceiling 0.900 if all of it is noise, 0.931 if the predictable part is biology | **superseded by Finding 6.** Its premise was that per-replicate counts do not exist; crisprHAL releases them, and the direct replicate-based ceiling (0.946) replaces this indirect one. The result is not in the report, and it could not have tested replicate independence anyway: WT and eSpCas9 are two enzymes, so a predictable disagreement is expected biology. (Two sentences of mine, in `EXTENSION_LOG.md` §22.2 and in `ceiling_table.py`, had said this module tests that assumption; both are corrected) | `Runjia/sgrna/cohorts/ceiling2.py` |
| `sgrna/controls/numcheck.py` | A document-coverage control from the frozen phase: does each number in the documents exist anywhere in `results/`, and does a cited file contain the numbers on the line that cites it | **superseded by the audit** (`paper/verify_numbers.py`), which checks each quoted number against a named file with a stated tolerance. `numcheck` had absolute paths pinned to one machine, ran its whole body at import, and its attribution layer cannot see a *difference* between two rows (its own docstring says so), so it flags derived numbers as misses. Its last run (22 Nov) listed 18 coverage and 30 attribution "misses", nearly all derived values or column counts from decoding the matrix; **none was investigated, and none is known to be wrong**. `sgrna/README.md`'s instruction to run it before committing a document change now points at the audit | `Runjia/sgrna/controls/numcheck.py` |
| `sgrna/post_freeze/nuclease/` (9 files, 1,567 lines) and its 26 result files (438 KB, `results/post_freeze/{nuclease_*,lono_*,pam_variants/,enzyme_*,kim2020_*,protein_composition_l1}.csv`), plus the audit's checks on them | **Finding 9, "Nuclease information".** Does telling a model which nuclease it is scoring help, and does it help for an unseen one? Data: Kim et al. 2020, nine nucleases on 40,128 targets (a human-cell sheet, not *E. coli*). Found: in distribution it gains nothing (the descriptors act as a label, and the same descriptors permuted across enzymes do as well); for an **unseen** enzyme, descriptors built without measuring it recover **which PAM classes it can use** (class-ordering ρ 0.857 to 0.903, interval clear of zero at both capacities tried), a smaller capacity-dependent gain on ranking targets *within* a PAM class (−0.002 at 500 × 6, +0.035 at 1,500 × 8, interval including zero), and pooling enzymes is worth about a fourfold saving in labels for a new one. A PAM profile predicted from an enzyme's six PI-domain residues (trained on 770 public variants) matched the measured one (difference −0.001 [−0.010, +0.008]). Also: the library covariate moves canonical-PAM ρ more than any sequence column, and our per-nuclease models reach a mean Spearman of 0.881 against the paper's 0.896 (ahead on 5 of 9) | **no substance for this project, by your call on 22 Nov.** The one reliable result, which PAM classes an unseen enzyme can use, is close to what PAM biology already tells you, it is measured on human-cell data, and nothing in the final model uses it. Findings 8 and 9 had already been marked supplementary in `PAPER_FLOW.md` | `Runjia/sgrna/post_freeze/nuclease/`, `Runjia/results/post_freeze/pam_variants/` and the files above |
| `sgrna/post_freeze/README.md`, the six `__init__.py` markers of the dissolved packages (`post_freeze`, `speed`, `shape`, `human`, `signal`, `strand`), and the "What is where" and "Notes from the November audit" sections of `sgrna/README.md` | A hand-written README for the `post_freeze/` folder, and tables in `sgrna/README.md` that described 66 modules by their old flat names | **folded, because the layout they described no longer exists.** The layout and how-to-run text now live in `sgrna/README.md`; per-module descriptions come from each module's docstring in the generated `CODE_MAP.md`. The deleted tables had drifted from the 130 modules on disk (they still listed `ceiling2.py` and `numcheck.py`, both removed above) | `git show pre-layout-2026-11-22:Runjia/sgrna/post_freeze/README.md` and `...:Runjia/sgrna/README.md` |
| `NOTE_human_twin_leak.md` | A note to a teammate (19 Nov): the human iRF matrix repeats 49% of its guides, so row-wise folds leak, and the 0.5076 baseline reverses under twin-grouped folds | **addressed, by the user's account (22 Nov), so the hand-off has done its job.** The finding itself is not lost: the report records it (Part 3, the struck-through human-baseline subsection of Appendix B, and the withdrawn claims listed at its head) and `sgrna/compare/human_twin_check.py` reproduces it. I could not confirm the fix from the repository: `Jacky/` has no commit after 18 Nov, and nothing in it groups folds by guide | `git show pre-note-removal-2026-11-22:Runjia/NOTE_human_twin_leak.md` |

## Considered for removal and kept (your decision, 22 Nov)

These two were **not** removed. Everything above failed on objective grounds (never run, superseded, duplicated, or judged without
substance). These are different: completed, verified work that no section of the report mentions and that nothing imports. They were
put to you with what each found, and you chose to keep both. They stay, and they are the reason the generated map has a `—` row.

| candidate | what it asked and found | why it is a candidate | size |
|---|---|---|---|
| `sgrna/interpret/asymmetry.py`, `results/asymmetry_strand_audit.csv` (`asymmetry_expression.csv` is outside the repo) | *Is the PAM-side flank's advantage a transcription effect?* On the published matrix family A gains +0.072 from the downstream flank and +0.027 from the upstream one. The orientation test **cannot be run** on that screen (all 13,879 guides joined to the gene table are on the template strand). The expression-conditioned test **can**: the downstream-minus-upstream gap is flat across expression terciles (0.049 / 0.046 / 0.048), so the advantage is not a transcription effect | the report does not state either result, and its premise was overtaken by Finding 10: on the curated cohort the two sides agree to within 0.026. Counter-argument for keeping it: it is a clean control for "separate from the orientation effect" | 237 lines |
| `sgrna/cost/bench.py`, `results/memory_comparison.csv` (`memory_comparison_runs.csv`, `net_cost.csv` are outside the repo) | *What does one fold cost in RAM?* Each configuration in its own process, peak RSS read from the child. One fold peaks at 3.33 GB (all 6,517 columns, XGBoost selector) and at **0.65 GB** with the stored reduced matrix, at ρ 0.682 against 0.694. A dense network on the reduced columns trains in 1.4 s against 6.1 s, at ρ 0.640 against 0.641 | no section of the report gives a memory figure (the README quotes older ones from `cost.py`: 3.76 GB against 4.55 GB). Counter-argument for keeping it: a 0.65 GB figure is the kind of line a "usable tool" section would want | 319 lines |

## Kept although they look removable

Recorded so the next sweep does not repeat the reasoning.

| item | why it looked removable | why it stays |
|---|---|---|
| `sgrna/interpret/geom_importance.py` and `geom_importance.csv` | no document names either | the report's Appendix B4 states its conclusion (item 4) without naming it — and in checking this I found that the figure the report quoted, R² 0.963, had **no producer and does not reproduce** (the same analysis gives 0.968 by ridge, 0.937 by boosting for that column, and 0.86–0.97 across the five). The module now computes it and the audit checks it; the report was corrected |
| `baseline_ecoli_sgRNA_model_swappable.py`, and the fork's top-level `USE THIS!! FINAL_baseline_…py` | referenced by no code | they are the teammates' original baseline, and the report names it as where `N_FEATURES = 300` and the champion settings were *inherited from* (Part 4's decision register). Provenance, not clutter |
| `results/rederived/` and `results/rederivation_check.csv` | duplicates of seven `results/*.csv` files | the originals' producers were lost; `controls/rederive.py` rebuilt them and this is the evidence that they agree |
| `results/RESULTS_archive_2026-10-10.md` | superseded by `RESULTS.md` | the report cites it by name as the record of what was believed on 10 October, including the contaminated "+0.104 under arc5" figure |
| `results/folds_human_*.csv`, `folds_humangc_*.csv`, `folds_base_plus_{K562_ATAC,K562_DNase,dnase_atac}.csv` (12 files, each under 1 KB) | no document of mine names them, no module reads them, and for half of them even the per-arm means do not appear in the report | they are the harness's per-fold records for the human experiments that `Jacky/EXPERIMENT_CATALOG.md` lists as load-bearing (the human ladder, the chromatin tier, the GC+GC² ablation). Not part of this report's claims, but a teammate's evidence and not mine to remove; each is tiny |
| `mix_*_stage1.csv`, `mix_configs_*.json`, `tune_stage1.csv` | 100–270 KB search logs nobody cites | `mix_ecoli`/`tune_ecoli` read them by names built at run time, and `e2e_bench` times the "mixed ensemble" row of the report's cost table from them |
| `sgrna/tuning/mlp_worker.py`, `sgrna/compare/vs_crisprhal_wt_hal.py` | imported by nothing | launched as subprocesses — the first to keep PyTorch out of the XGBoost process (they hang together on macOS), the second inside the TensorFlow environment |
| `sgrna/interpret/importance_models.py` | no document names it, and its three result files were **not in the repository** | Finding 5's headline table is its output: agreement between models' importances, 0.050 / 0.258 / 0.496 per column for the default, permutation and SHAP. The files existed only in the untracked main-project folder; they are now in `results/` and audited. Checking them showed the table is the five *tree* families (SHAP is undefined for the linear models), not all seven as its caption said — adding the linear models makes the default agreement −0.057, which strengthens the claim. Caption corrected |
| `sgrna/controls/why_inflation.py` | no document names it | its result exists and `RESULTS.md` cites it; the report states its decomposition (the label alone is worth +0.004) in Appendix B2a |
| `sgrna/compare/crisprhal_tune.py` | never produced a result | it is the starting point for open question 1 in the report's Part 4 register (re-tune crisprHAL 2 on its own splits, ~40 GPU-hours) |

## Brought in (the opposite of removal)

Checking which result files the documents cite by name against which files the repository holds found that the report's
evidence was not all in the repository: some of it lived only in the main project's untracked `results/` folder. Each of
these was **checked against the report text before being copied in**, and is now audited.

| file | what the report rests on it | check |
|---|---|---|
| `results/importance_model_pairs.csv`, `_methods.csv`, `_blocks.csv` | Finding 5's table of agreement between models | reproduces exactly for the five tree families (10 pairs); the caption said seven families and was corrected |
| `results/kmer_order_paired.csv` | Finding 4's trimers-help / tetramers-hurt table | reproduces exactly, including all three p-values |
| `results/crisprhal_rerun.csv` | crisprHAL 2's per-fold figures at seed 41 | reproduces exactly |
| `results/representation_headroom.csv` (2 MB) | Finding 4's headroom table: the model is out of material, with family A flagged at 0.147 *before* family A existed | reproduces all three rows. The table had cited `headroom.csv`, which is a *different* file — a later 6,769-column run whose strongest held-back column is 0.155, not 0.147 — so the citation was corrected and `headroom.csv` was **not** copied, since it would have looked like evidence for a table it does not back. Row 1 was also re-derived from code (`diagnose.headroom(families=(), extra=("a_flank",))`): 0.028 / 0.147, identical |

**A second batch, brought in without being checked one by one (22 Nov, at your direction).** 30 small summary files that existed only in
the main project's untracked `results/` folder, 773 KB in all, written by 17 frozen-phase modules: `attribution*`, `faithfulness_*`,
`importance_agreement`/`_blocks`, `interpretability`, `model_comparison`, `representation*`, `redundancy`, `headtohead`, `metric_panel`,
`permutation_floors`, `strand_*`, `tuning_trace`, `hybrid`, `seqnet`, `net_cost`, `memory_comparison_runs`, `asymmetry_expression`,
`crisprhal_tuning`, `ceiling.json` and `headroom.csv`. They are now in `results/`, so a clone of the repository holds the evidence the report points at. **One caveat
that matters:** `headroom.csv` is a *later* 6,769-column run, **not** the source of Finding 4's headroom table (that is
`representation_headroom.csv`, brought in earlier and audited). It is in the repository because it was asked for, and it is not evidence for that table.

Two citations pointed at nothing and were corrected: `human_ladder.csv` in `results/RESULTS.md` was an aggregate that was never
saved — the figures it summarised (human base 0.382, with family A 0.434, +0.052) are the means of
`folds_human_base_cv.csv` and `folds_human_base_a_cv.csv`, which are now cited and audited; and `final_eval.csv` in the report
was `final_eval_espcas9.csv`.

## 24 November 2026: four Colab benchmarks folded into one

`colab/inference_bench_t4.py`, `colab/inference_bench_onecore.py`, `colab/slicer_cpu_check.py` and `colab/crisprhal_inference_bench.py` timed
crisprHAL 2 and SLICER-shaped trees on a Colab T4 (and, in the last two, a CPU), each a slightly different version of the same measurement
(medians of a few calls, one with a hard-coded laptop path). They are replaced by one paste-into-one-cell file, `colab/cost_gpu.py`, which keeps every repeat,
adds the fits and the tuned tree shape, and needs no upload. Their stored medians stay in `results/inference_colab.csv`, `inference_t4.csv`,
`inference_onecore.csv` and `inference_onecore_local.csv` (their `source` columns name the scripts that wrote them); restore a script with
`git checkout b6fc21a -- Runjia/colab/<file>`.

## 25 November 2026: `flank_window_tune` folded into `flank_scale`

`sgrna/interpret/flank_window_tune.py` (the upstream and downstream windows chosen separately) ran the same engine as `flank_scale.py` (same folds, same model, its symmetric
arms read from `flank_scale.csv`) and differed by the table of window pairs it scored. It is now the part `asymmetric` of `flank_scale.py`
(`python -m sgrna.interpret.flank_scale --parts asymmetric`), and the merged module reproduces its stored summary byte for byte
(`results/post_freeze/flank_window_tune_summary.csv` is unchanged). Restore the old module with `git checkout pre-merge-2026-11-25 -- Runjia/sgrna/interpret/flank_window_tune.py`.

`sgrna/cohorts/ablation_paper_folds.py` (the family ablation on the paper's own folds, written on 24 November) is likewise folded into `curated_ablation.py` as `--paper-folds`; its two result files are unchanged
(`results/post_freeze/ablation_paper_folds.csv`, `ablation_paper_folds_summary.csv`) and the merged design resumes from them and reproduces the summary byte for byte. Restore with
`git checkout pre-merge-2026-11-25 -- Runjia/sgrna/cohorts/ablation_paper_folds.py` (the tag was set before the first merge, so use `git log --diff-filter=D -- Runjia/sgrna/cohorts/ablation_paper_folds.py` to find the commit that holds it).


## 26 November 2026: `build_curated_features` merged into `build_features`, `expand` renamed `curated_matrix` and trimmed

`sgrna/build/build_curated_features.py` and `sgrna/build/build_features.py` were the same loop over the same families, differing in the index (curated or published), the
output suffix (`_curated`) and one guard (a block with no populated column is a failure). They are now one module, `build_features.py`, with `--cohort published|curated`; the guard
applies to both, `load_curated_index` and `augment` moved with it, and family A is built by `a_flank_fast` (equal to the original after the float32 cast on all 33,567 curated and
all 13,880 published guides; the float64 difference is 1.4e-14, so a regenerated CSV differs from the stored one in the fourteenth digit). Restore the old module with
`git checkout 1c61887 -- Runjia/sgrna/build/build_curated_features.py`.

`sgrna/build/expand.py` is renamed `curated_matrix.py`, because the name said nothing about what the module does: it featurises the 33,567 guides of the Guo et al. screen as
crisprHAL 2 curated them and caches the matrix every experiment loads. Its columns now come from `base_fast.matrix_fast` (identical to the cache on all 33,567 guides, maximum
absolute difference 0.0). Its learning curve (`--curve`, `results/learning_curve.csv`: the default recipe at 4,000 to 26,000 training guides, three seeds) is removed: no
paper table, report claim or audit check reads it, and the data-efficiency comparison is `compare/sample_complexity.py`. Restore with `git checkout 1c61887 -- Runjia/sgrna/build/expand.py Runjia/results/learning_curve.csv`.

## 27 November 2026: `tune_split2`, `tune_analyze`, `family_a_at_tuned` and `local_sequence_bound` merged

Four modules ran an existing pipeline with one thing changed, and are now flags or a table row of a module that stays. Safety tag `pre-merge-2026-11-27`; restore any of them with
`git checkout pre-merge-2026-11-27 -- Runjia/sgrna/tuning/tune_split2.py` (and likewise the other three paths below).

* `sgrna/tuning/tune_split2.py` (the tuning replication on a second gene split) is `tune_ecoli.py --split 2`. A `Plan` object in `tune_ecoli.py` holds what differed: the gene-split seed (7 instead of 2026),
  the result files (`tune_split2_*`), a cache folder of its own, 40 fresh draws from another stream of the search space, one stage-2 partition, the top-3 average and the flagged split-1 row. It had
  re-implemented six functions of `tune_ecoli` (`Cohort`, `rankings`, `score_dev`, `stage1`, `stage2`, `confirm`) and imported twelve names from it.
* `sgrna/tuning/tune_analyze.py` (what the 200 stage-1 draws say about each knob) is `tune_ecoli.py --analyze`; the audit imports `stage1_table` from `tune_ecoli` instead of `table` from `tune_analyze`.
* `sgrna/tuning/family_a_at_tuned.py` and `sgrna/cohorts/local_sequence_bound.py` were the same loop (a gene-grouped 5 x 3 ablation of a block of columns at the tuned configurations, with a paired bootstrap over
  folds) on two blocks of columns. They are `cohorts/tuned_ablation.py --design family_a` and `--design local_sequence`; the first moved out of `tuning/`, which it never belonged in (it tunes nothing).

How each was checked: every stored output is reproduced byte for byte by re-running the merged command from the stored rows and caches (`git status` shows no change under `results/`): `tune_stage1_effects.txt`, `tune_confirm.csv`,
`tune_split2_confirm.csv`, `family_a_at_tuned_summary.csv`, `local_sequence_bound_summary.csv`; and fresh fits written to a scratch folder equal the stored rows exactly (largest difference 0.0 in rho and in every count):
both ablation designs (seed 41, fold 0: 8 and 6 rows), stage 1 on split 1 (champion, depth6 and two draws: 140 rows), stage 1 on split 2 (the same: 125 rows) and stage 2 on split 2 (champion, depth6 and the best finalist: 95 rows).
Not merged, with reasons: `tune.py` (the original LightGBM search: another model and protocol), `mix_ecoli.py` (other model families and blending) and `mlp_worker.py` (PyTorch, XGBoost and LightGBM each load an OpenMP
runtime, and loading them in one process stalled at 0% CPU on macOS, so the neural net must run in its own process); `xgb_knobs.py` (one knob at a time under the older random-fold design) is a candidate for a later pass.

## 27 November 2026 (later): `sync_to_shared.sh` and nine unreferenced functions removed; the mirror procedure is now a tool

* `sync_to_shared.sh` is removed. Its premise (the shared repository nested inside the working folder, so every document had to exist twice) ended when the fork became a separate checkout; it never deleted what the
  fork deleted and wrote stale mirrors, and it was not used. The procedure that is used, a check of every file against the fork's last mirrored version before it is changed, is now `mirror_to_shared.py` (a dry run unless
  `--apply`; it reads the last mirrored commit from the shared repository's own history and stages without committing). Restore the old script with `git checkout pre-merge-2026-11-27 -- Runjia/sync_to_shared.sh`.
* Nine functions that no file anywhere mentions beyond their own definition are removed (109 lines): `genome.signed_circular_offset`, `genome.find_all`, `genome.iter_kmers`, `io_utils.base_matrix_columns`, `io_utils.one_hot`,
  `make_diagnostic_blocks.build_abundance_split`, `e_nucleoid._bin_of`, `diagnose.cross_redundancy` (58 lines), `topend._qid`, and the two imports they left unused. The search covered both repositories, the teammates'
  folders and the two notebooks of the working copy. It also showed that five functions the 25 November analysis had listed as unreferenced are used outside `Runjia/` and stay: `evaluate.bootstrap_interval_grouped`,
  `within_gene_spearman` and `expected_pick_percentile_within_gene` and `make_diagnostic_blocks.build_mapgc` (called by `Jacky/agent-notes/scripts/workplan.py`), and `notebook.feature_block` (called by `notebooks/RSI09_pipeline.ipynb`).

What was *not* deleted, against the review of the same day, after reading the reasons recorded above: `baseline_ecoli_sgRNA_model_swappable.py` (the report names it as where `N_FEATURES = 300` and the champion settings were inherited from),
`results/RESULTS_archive_2026-10-10.md` (the report cites it, and the audit's i_shape +0.0356 is recorded only there) and `harness/notebook.py` (the front end of the working copy's `RSI09_pipeline.ipynb` and `RSI09_quickstart.ipynb`).

## 27 November 2026 (evening): `results/post_freeze/` flattened into `results/`; the loaders and the audit's standalone driver have one home

* `results/post_freeze/` no longer exists: its 179 entries (177 files and the subfolders `figures/` and `interp_hal_cache/`) moved to `results/` with `git mv`; no file name occurred in both folders. Paths in this file's earlier entries and in the
  restore commands (`git checkout pre-cleanup-2026-11-22 -- Runjia/results/post_freeze/...`) name the old layout on purpose, because they restore from tags made before the move. `data/interim/post_freeze/` (caches in the project's data folder)
  is untouched.
* `cohorts/curated_ablation._base`, `_block` and `_groups`, which 53 modules and the audit imported by their underscore-private names, are `harness/curated.load_base`, `load_block` and `load_groups` (bodies unchanged); five copies of the
  z-score are `harness/metrics.zscore`.
* The fifteen copies of the audit's standalone `__main__` driver are `paper/standalone.run_alone`.

## 27 November 2026 (night): five strand modules merged into `compare/strand.py`

`strand_families.py`, `strand_strata.py`, `strand_within_gene.py`, `cut_within_gene.py` and `strand_recoverable.py` asked one question in pieces (is family A's gain on the dCas9 screens the strand rule arriving through a lossy copy of
"which strand the guide targets?") and shared the loading of Cui's essential-gene frame (seven copies) and the between/within-genes split (two). They are now `compare/strand.py --parts build|families|families-within|recoverable|rousset|flip|within-cui|within-joint|within-rousset|cutting`,
with the frame loaded once and the split defined once. Every result file keeps its name; all are now written under `results/` (the old modules wrote most of them to the main project's `results/` and the tracked copies were made by hand).
Restore the old modules with `git checkout pre-merge-2026-11-27 -- Runjia/sgrna/compare/strand_families.py` (and the other four).

How it was checked: `families`, `families-within`, `recoverable` and `flip` reproduce their stored tables from the fold checkpoints (to 3e-14, 3e-14, exactly and exactly; `./run_audit.sh --reproduce`, four new cases that run against a scratch copy of the project
root so the checkpoints cannot be touched); the within-gene routine gave identical files in the old and the new code on a real refit (Rousset coding strand, two seeds, 100 bootstrap draws), and the decomposition is identical on random data. The remaining
parts (`rousset`, `within-*`, `cutting`) cannot be reproduced from checkpoints, see below, so they rest on that A/B test and on the unchanged logic.

A hazard found and closed: `cross_validate(return_oof=True)` fills the out-of-fold predictions only for folds it computes in that call, so resuming from complete checkpoints gives NaN predictions, and the old modules then **overwrote their good result files with NaN rows**
(a rerun of `strand_within_gene --cui` from the stored checkpoints did exactly that in a scratch copy). `strand.py` raises instead, naming the checkpoints to delete if a refit is wanted.
