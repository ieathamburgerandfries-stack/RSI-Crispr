# Code map

What the code in this folder is, how it is organised, and which part of the report each module supports.
The tables between the two markers are **generated from the code** (`./run_ext.sh sgrna.post_freeze.paper.code_map`) and
the audit fails if a module is missing from them, so they cannot drift from the files. Everything outside the markers is
hand-written. For a short tour of only the code the report's claims depend on, read [`CODE_OVERVIEW.md`](CODE_OVERVIEW.md) first. What was deleted, and why, is in [`REMOVED.md`](REMOVED.md).

## The shape of it

Roughly 29,000 lines of Python in three layers (the exact totals are the first line of the generated section below).

| layer | where | what it is |
|---|---|---|
| **the pipeline** | `sgrna/build`, `features`, `harness` | The library. Decode the published matrix into a 20-mer, locate every guide on the genome, build eleven feature families, and run a leakage-safe cross-validation harness. Nothing here answers a scientific question by itself. |
| **the frozen experiments** | `sgrna/cohorts`, `compare`, `controls`, `cost`, `strand`, `interpret` | One script per question, run before the freeze of 11 October 2026. They back Findings 1–7 and the appendices. |
| **the extension** | `sgrna/post_freeze/<theme>/` | Work done after the freeze: the hyperparameter search, the shape null, the human arm, Finding 7, the fast featurisers, the figures and the audit. Six themed folders, listed below. The crisprHAL 2 comparison lives in `sgrna/compare/` with the rest of Part 6. |

Plus `colab/` (the GPU runs that had to happen on Colab), `results/` (every number the report cites, as CSV) and the three
documents: `PROGRESS_REPORT.md` (what is claimed), `EXTENSION_LOG.md` (the working record) and `PAPER_FLOW.md` (a skeleton).

## Why there are so many files

Because the report makes many claims and each was tested by its own script, with its own controls. A claim, its null, its
shuffled control and its robustness check are four short scripts rather than one long one, and every one of them is the
provenance of a sentence in the report. That is a structure choice, not padding: an audit of the whole folder (recorded in
`REMOVED.md`) found **no duplicated code** — one 8-line function in the whole package — and almost **no dead code**, because the
report names each experiment's result and the audit checks the number. The size is the number of claims times the number of
ways each was attacked. If the folder should be smaller, the lever is the report's scope, not tidying: see the by-section
table below.

## How to read a module

Every experiment module has the same four parts, so any of them can be read the same way.

1. **The docstring is the question** — usually phrased as one ("Does X survive Y?") — followed by what was expected before
   the run and what the result means.
2. **`run()`** does the experiment through the shared harness (`harness/run_ablation.py`, or `post_freeze/tuning/tune_ecoli.py`
   for the extension), with folds written to `results/` as they finish so a crash loses one fold, not a run.
3. **One CSV** lands in `results/` (frozen work) or `results/post_freeze/` (extension), named after the module.
4. **A `__main__` guard**, so importing a module does nothing. `paper/import_check.py` asserts this for all of them.

## How to run something

```bash
./run_ext.sh sgrna.post_freeze.speed.a_flank_fast              # any module, by dotted path; sets the three variables it needs
./run_ext.sh sgrna.cohorts.curated_ablation --all              # the frozen modules run the same way
./run_audit.sh                                                  # recompute every number the report quotes from its file
```

`run_ext.sh` exists because the modules need `RSI09_ROOT` (the folder holding `data/`), `EXT_OUT` (where results go) and
`PYTHONPATH` set together, and getting one wrong fails with a `FileNotFoundError` that does not say which. **Never run
`sync_to_shared.sh`.**

<!-- generated:begin -->

**130 modules, 27,417 lines** of Python in 17 folders (not counting the `__init__.py` markers).

#### By the report section each module supports

| report section | what it is | modules | lines | result files | where the code lives |
|---|---|---:|---:|---:|---|
| Finding 1 | The published representation is a re-encoding of the guide, and holds about twenty variables | 4 | 506 | 5 | cohorts, compare, controls |
| Finding 2 | The DNA around the cut site predicts cutting — the one large effect, and it is a gradient | 3 | 632 | 6 | controls, harness, interpret |
| Finding 3 | The effect belongs to the DNA, and transfers | 14 | 3,862 | 48 | cohorts, compare, controls, features, harness, pf/signal, strand |
| Finding 4 | The model is out of *material* — and the post-freeze tuning gain is the sharpest evidence for it | 2 | 516 | 9 | pf/tuning |
| Finding 5 | On a redundant matrix, importance and faithfulness are both uninterpretable | 6 | 1,574 | 22 | interpret, pf/tuning |
| Finding 6 | The label's ceiling is 0.946, measured from crisprHAL's own replicates | 1 | 122 | 1 | cohorts |
| Finding 7 | What the signal is made of: GC over a few hundred bases, a seed region that inverts between kingdoms, and the two coupled | 2 | 516 | 9 | pf/signal |
| Part 2 | Contributions | 3 | 434 | 16 | pf/human |
| Part 3 | Where this stops | 1 | 179 | 2 | compare |
| Part 4 | The freeze, and what remains | 4 | 1,185 | 8 | build, controls, interpret |
| Part 6 | Where this sits in the field | 32 | 5,279 | 51 | cohorts, compare, cost, harness, interpret, pf/human, pf/signal, pf/speed, pf/tuning |
| Appendix A | Controls and nulls | 1 | 86 | 13 | controls |
| Appendix B | The methods and controls behind the findings | 13 | 3,551 | 38 | cohorts, compare, controls, harness, interpret, pf/human, pf/tuning |
| Appendix B2 | Why the imported families failed, and why that is not a screen | 1 | 255 | 0 | build |
| Appendix B2a | Curation: what it is, and why family A doubled between cohorts | 2 | 263 | 2 | cohorts, controls |
| Appendix B3 | The family ablations on the curated cohort, and what they measure | 12 | 2,253 | 5 | cohorts, features, interpret |
| Appendix B4 | The flank's shape and 3D structure add nothing that its composition does not already hold | 11 | 1,584 | 21 | build, pf/shape |
| Appendix C0 | Reproducibility: versions, seeds and protocol | 1 | 199 | 0 |  |
| tooling | the audit, the figures, the guards and this map | 5 | 1,805 | 0 | pf/paper |
| library | shared code the experiments are built on | 10 | 2,060 | 4 | , build, harness, post_freeze |
| — | no section of the report points at it | 2 | 556 | 5 | cost, interpret |

Read the table above as the price list for the report: each claim carries its code and its result files with it. A module is placed in the section that cites it (or the result files it writes) most often. The `—` row, if there is one, is completed work that no section of the report points at. Each such module is listed in `REMOVED.md` under "Considered for removal and kept", with what it found; they were kept by decision, not by oversight.

#### Result files that exist only outside this repository

None: every result file a module writes by name is tracked, apart from the bulk prediction and per-fold records.

#### Package by package

### `sgrna/`  —  2 modules, 287 lines

Paths and constants every module shares (`config.py`), and the conventions for reading the base matrix and writing feature blocks (`io_utils.py`).

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `config` | 199 | Project paths and shared constants. | Appendix C0 | — |
| `io_utils` | 88 | Shared conventions for reading the base matrix and writing feature blocks. | library | — |

### `sgrna/build/`  —  10 modules, 1,886 lines

Decode the published matrix, locate every guide on the genome, and rebuild features for guides outside it.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `build_curated_features` | 174 | Build feature families on crisprHAL's curated cohort, not the published rows. | library | — |
| `build_features` | 89 | Step 1 -- build one or more feature families. | library | — |
| `build_guide_index` | 166 | Step 0 -- turn 13,880 anonymous rows into located guides. | library | — |
| `build_matrix` | 193 | Step 2 -- assemble a model-ready matrix from the base features plus blocks. | library | — |
| `decode_v_columns` | 196 | Work out what the 5,888 `V####` columns are, so new guides can be featurised. | library | — |
| `expand` | 176 | Lift the 13,880-row cap: featurise every guide in the curated Guo screen. | library | `learning_curve.csv` |
| `featurise` | 126 | Generate the published feature representation for guides outside the matrix. | Part 4 | — |
| `genome` | 255 | Reference genome helpers for E. coli K-12 MG1655 (NC_000913.2). | Appendix B2 | — |
| `make_diagnostic_blocks` | 223 | Decompose the positional families, to find out what they are really measuring. | library | — |
| `qct` | 288 | Quantum chemical tensor (QCT) decoding and extension. | Appendix B4 | — |

### `sgrna/features/`  —  11 modules, 2,097 lines

The eleven feature families A-J (family A, the flanking-DNA composition, is the one that matters).

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `a_flank` | 240 | Family A -- sequence and quantum descriptors outside the 20 nt protospacer. | Appendix B3 | — |
| `b_energy` | 164 | Family B -- the R-loop free-energy decomposition (CRISPRoff / CRISPRspec). | Appendix B3 | — |
| `c_folding` | 155 | Family C -- sgRNA folding, done on the transcript that is actually made. | Appendix B3 | — |
| `d_mechanics` | 201 | Family D1 -- how hard the target duplex is to open, position by position. | Appendix B3 | — |
| `d_supercoiling` | 233 | Family D2 -- DNA supercoiling at the target, from GapR-seq. | Appendix B3 | — |
| `e_nucleoid` | 226 | Family E -- nucleoid organisation, the bacterial answer to chromatin. | Appendix B3 | — |
| `f_methylation` | 133 | Family F -- Dam and Dcm methylation at the target site. | Appendix B3 | — |
| `g_transcription` | 275 | Family G -- transcription and replication context. | Finding 3 | — |
| `h_offtarget` | 133 | Family H -- off-target burden and target copy number. | Appendix B3 | — |
| `i_shape` | 208 | Family I -- the physical shape of the DNA around the target, not its letters. | Appendix B3 | — |
| `j_periodicity` | 129 | Family J -- strand asymmetry and helical periodicity in the flanking DNA. | Appendix B3 | — |

### `sgrna/harness/`  —  7 modules, 2,640 lines

The leakage-safe cross-validation harness every experiment runs through, plus the notebook front end.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `cohort_groups` | 74 | Does grouped cross-validation still mean the same thing on the curated cohort? | Appendix B | `cohort_groups.csv` |
| `evaluate` | 708 | Quantifying the two things a model is judged on: prediction and explanation. | library | `interpretability.csv`, `model_comparison.csv` |
| `notebook` | 532 | Notebook front end -- everything the pipeline does, callable from a cell. | Finding 3 | `ablation_abund_split.csv`, `ablation_curated.csv` +9 |
| `run_ablation` | 560 | Step 3 -- the ablation harness. | Finding 3 | `ablation_abund_split.csv`, `ablation_curated.csv` +21 |
| `seqnet` | 282 | Does a convolutional net on raw sequence beat gradient boosting on 6,580 engineered columns? | Finding 2 | `seqnet.csv`, `seqnet_matched.csv` |
| `tune` | 241 | Does the head-to-head margin survive hyper-parameter tuning? | Part 6 | `tuning.csv`, `tuning_trace.csv` |
| `xgb_knobs` | 243 | Which XGBoost knobs has this project never touched, and do they matter? | Appendix B | `xgb_knobs.csv`, `xgb_knobs_verify.csv` |

### `sgrna/cohorts/`  —  8 modules, 1,314 lines

Ablations on the curated cohort and the arguments about what its numbers mean (ceilings, inflation, label noise).

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `curated_ablation` | 246 | Feature-family ablations on crisprHAL's curated cohort. | Appendix B3 | `ablation_curated.csv`, `ablation_curated_gene.csv` +1 |
| `curated_ceiling` | 122 | The ceiling for the label we actually train on, from replicate agreement. | Finding 6 | `curated_ceiling.json` |
| `curated_joint` | 122 | Joint arms on the curated cohort: how much of the eleven family gains overlap? | Finding 3 | `ablation_curated.csv`, `curated_joint.csv` |
| `curated_ladder` | 123 | The reduced-representation ladder, re-run on the curated cohort without selection. | Finding 1 | `curated_ladder.csv` |
| `curated_models` | 95 | Does model class matter more once the representation has family A in it? | Appendix B | `curated_models.csv` |
| `curation_variance` | 159 | Why did every family's gain rise when the baseline rose too? | Appendix B2a | `curation_variance.csv` |
| `data_ablation` | 152 | Hold SLICER fixed and change only the training data. | Part 6 | `data_ablation.csv` |
| `replicate_weight` | 295 | The label's reliability is per guide, and the model treats it as uniform. | Appendix B | `replicate_weight.csv`, `replicate_weight_raw.csv` |

### `sgrna/compare/`  —  24 modules, 4,977 lines

Every comparison with something else (Part 6 of the report): crisprHAL 2, the inherited baseline, and other screens, assays and kingdoms.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `blend_with_hal` | 47 | Is crisprHAL 2's network complementary to the tuned trees? | Part 6 | `blend_with_hal.csv` |
| `clean_vs_hal` | 65 | A comparison with crisprHAL 2 that the tuning cannot have touched: the 30% of genes held out from every search. | Part 6 | `clean_vs_hal.csv` |
| `crisprhal_rerun` | 151 | Re-run crisprHAL 2 ourselves, on our rows and our folds. | Appendix B | `crisprhal_rerun.csv` |
| `crisprhal_tune` | 284 | Give crisprHAL 2 the same courtesy we gave ourselves: a hyperparameter search. | Appendix B | `crisprhal_tuned_folds.csv`, `crisprhal_tuning.csv` |
| `crispri_screens` | 324 | Three CRISPRi screens instead of one, and what that buys beyond accuracy. | Finding 3 | `crispri_screen_agreement.csv`, `crispri_screen_family_a.csv` +1 |
| `cui_final_exam` | 219 | Both models, neither trained on it, scored on Cui's CRISPRi screen. | Part 6 | `cui_final_exam.csv` |
| `cui_screen` | 593 | The last untouched test: does the flanking-context effect appear in a CRISPRi screen? | Finding 3 | `cui_essential_marginals.csv`, `cui_essential_position.csv` +4 |
| `data_efficiency` | 67 | The data-efficiency curve again, with the tuned model against the stored crisprHAL 2 curve. | Part 6 | `data_efficiency.csv`, `sample_complexity.csv` +1 |
| `final_eval` | 247 | The designated final evaluation: both models on crisprHAL's eSpCas9 split. | Part 6 | `final_eval_espcas9.csv` |
| `handnamed_split` | 122 | Split the 63 hand-named columns into the computable and the rest, and test both. | Finding 1 | `handnamed_split.csv`, `portability_audit.csv` |
| `headtohead` | 303 | The like-for-like comparison against crisprHAL 2. | Part 6 | `headtohead.csv` |
| `heldout_compare` | 156 | How do the two models compare on a held-out fold? | Part 6 | `heldout_compare.csv` |
| `human_baseline` | 270 | Does the human baseline reproduce ρ ≈ 0.472 under this harness? | Appendix B | `human_additivity.csv`, `human_baseline.csv` +1 |
| `kingdom_transfer` | 179 | Cross-kingdom transfer, with the human side fitted properly. | Part 3 | `kingdom_transfer.csv`, `kingdom_transfer_boot.csv` |
| `noshay_replicate` | 232 | Reproduce the inherited paper's own model, as its methods section describes it. | Part 6 | `noshay_replication.csv` |
| `panel` | 219 | The full metric panel, for untuned and tuned, on identical folds. | Part 6 | `metric_panel.csv` |
| `panel_vs_hal` | 110 | The six-metric panel against crisprHAL 2 on its own two splits, for the tuned and mixed models, with raw predictions kept. | Part 6 | `panel_espcas9.csv`, `panel_wt.csv` |
| `portability` | 167 | Can the 427-column representation be rebuilt for a guide we have never seen? | Finding 1 | `portability.csv`, `portability_audit.csv` |
| `sample_complexity` | 220 | How many labelled guides does each method need? | Part 6 | `sample_complexity.csv` |
| `topend` | 96 | Do objectives aimed at the top of the ranking improve top-end metrics, where crisprHAL 2 still leads on pick percentile? | Part 6 | `topend_confirm.csv`, `tune_configs.json` |
| `topk_panel` | 97 | Smoother top-end metrics against crisprHAL 2 on its own splits, and seed stability of the lead. | Part 6 | `seed_stability.csv`, `topk_panel.csv` |
| `transfer` | 588 | Does the flank effect transfer to another nuclease? | Finding 3 | `hybrid.csv`, `transfer.csv` +1 |
| `vs_crisprhal` | 186 | The tuned / mixed E. coli model against crisprHAL 2 again, under the three conditions of the original comparison. | Part 6 | `mix_finalists.csv`, `tune_configs.json` +3 |
| `vs_crisprhal_wt_hal` | 35 | crisprHAL 2's shipped WT-SpCas9 weights on their own WT test file, saved for a paired comparison (TensorFlow environment). | Part 6 | — |

### `sgrna/controls/`  —  10 modules, 2,204 lines

The adversarial phase: nulls, shuffles, floors, selector and leakage audits.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `chi_sites` | 198 | Is the flank effect a *repair* effect? | Appendix B | `chi_sites.csv` |
| `diagnose` | 429 | Why doesn't adding features help? | Part 4 | `ceiling.json`, `headroom.csv` +1 |
| `floors` | 86 | Per-family permutation floors, paired fold by fold. | Appendix A | `folds_base_plus_K562_ATAC.csv`, `folds_base_plus_K562_DNase.csv` +11 |
| `no_information` | 94 | Decompose `d_mechanics`'s +0.099 into representation and new DNA. | Finding 1 | `no_information.csv` |
| `rederive` | 543 | Re-derive the result files whose producer was lost, and check they agree. | Appendix B | `ablation_curated.csv`, `ablation_curated_gene.csv` +5 |
| `sealed_holdout` | 224 | Can the adaptive-overfitting caveat be measured by sealing part of the cohort? | Part 4 | `sealed_holdout.csv` |
| `selector_audit` | 272 | The pipeline selects columns with the one metric the report says never to use. | Appendix B | `selector_audit.csv`, `selector_audit_ranks.csv` +1 |
| `shiftnull` | 140 | A spatially preserving null for family A: shift the chromosome, keep its texture. | Finding 2 | `shift_null.csv` |
| `why_inflation` | 104 | Why did every family's Δρ roughly triple when the cohort moved? | Appendix B2a | `why_inflation.csv` |
| `window_robustness` | 114 | Would a different set of flank windows have given the same answer? | Finding 3 | `ablation_curated.csv`, `window_robustness.csv` |

### `sgrna/cost/`  —  3 modules, 832 lines

What each model costs to train and to run, measured rather than estimated.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `bench` | 319 | Peak memory and training cost, measured rather than estimated. | — | `memory_comparison.csv`, `memory_comparison_runs.csv` +1 |
| `cost` | 372 | How much computation does each model actually cost, and how light can SLICER get? | Part 6 | `cost_cpu_seconds.csv`, `cost_flops.csv` +1 |
| `inference_cost` | 141 | Training cost is the wrong clock for a guide-design tool. | Part 6 | `inference_cost.csv` |

### `sgrna/strand/`  —  5 modules, 718 lines

The strand-of-transcription confound in the dCas9 screens, and the controls that remove it.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `strand_families` | 241 | With the strand call supplied, how do all eleven families compare on the dCas9 screen? | Finding 3 | `strand_families.csv`, `strand_families_proxies.csv` +1 |
| `strand_recoverable` | 47 | How well can the strand call be read back from each family's columns, after proxy removal? | Finding 3 | `strand_families_recoverable.csv` |
| `strand_strata` | 143 | Two controls on the strand result. | Finding 3 | `strand_flip.csv`, `strand_strata_rousset.csv` |
| `strand_within_gene` | 181 | Inside the coding-strand guides, is a family's gain between genes or within them? | Finding 3 | `strand_within_gene_cui.csv`, `strand_within_gene_cui_joint.csv` +1 |
| `txn_interactions` | 106 | Family G's own rule, applied on the curated cohort: do the interactions pay? | Finding 3 | `ablation_curated.csv`, `txn_interactions.csv` |

### `sgrna/interpret/`  —  11 modules, 3,272 lines

What the model uses and whether that is the data's property or the model's.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `asymmetry` | 237 | Is the downstream flank's advantage a transcription effect? | — | `asymmetry_expression.csv`, `asymmetry_strand_audit.csv` |
| `attribution` | 214 | Two models, different features, same score. | Finding 5 | `attribution.csv`, `attribution_summary.csv` |
| `base_enrichment` | 185 | Is the curated table inflated because the base has no dense summaries? | Appendix B3 | `base_enrichment.csv`, `base_enrichment_published.csv` |
| `complexity` | 118 | What is SLICER's equivalent of a CNN's parameter count? | Part 6 | `complexity.json` |
| `faithfulness` | 270 | Dropping a model's top columns costs nothing. | Finding 5 | `faithfulness_clusters.csv`, `faithfulness_clusters_reduced.csv` +2 |
| `gcstrata` | 210 | Is the flanking-context effect just target GC wearing a different hat? | Finding 2 | `gc_strata_ablation.csv`, `gc_strata_corr.csv` +1 |
| `importance` | 293 | At what granularity is "feature importance" actually a property of the data? | Finding 5 | `importance_agreement.csv`, `importance_blocks.csv` |
| `importance_models` | 297 | Is "feature importance" a property of the data, or of the model asked? | Finding 5 | `importance_model_blocks.csv`, `importance_model_methods.csv` +1 |
| `interp_extra` | 301 | Three interpretability tests the report listed as not yet run. | Finding 5 | `interp_counterfactual.csv`, `interp_interaction_blocks.csv` +3 |
| `redescription` | 741 | How much is re-description worth, and does it have to mean anything? | Appendix B | `recoverability_percolumn.csv`, `recoverability_trend.csv` +7 |
| `representation` | 406 | Does the model need all 6,232 published columns, or one spelling of the 20-mer? | Part 4 | `representation.csv`, `representation_depth.csv` +2 |

### `sgrna/post_freeze/`  —  1 modules, 47 lines

Shared paths for the extension work, and the resolver for the main project's untracked results (`common.py`). Everything else is in the themed folders below.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `common` | 47 | Shared paths for the extension work (the private-fork experiments). | library | `sample_complexity.csv` |

### `sgrna/post_freeze/tuning/`  —  7 modules, 1,348 lines

Hyperparameter search, model mixing, and what the tuned settings say about the signal (Part 6, EXTENSION_LOG s11-14, s16).

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `family_a_at_tuned` | 159 | Is family A still worth +0.164 once the model is tuned? | Finding 4 | `family_a_at_tuned.csv`, `family_a_at_tuned_summary.csv` +1 |
| `interpret` | 199 | What the tuned hyperparameters say about the signal: four tests, each answering a question the choice of settings raises. | Finding 5 | `interpret_data_scaling.csv`, `interpret_depth.csv` +4 |
| `mix_ecoli` | 289 | Mixing model families for the E. coli model: does an ensemble that is not all XGBoost beat the tuned XGBoost ensemble? | Part 6 | `mix_configs_cat.json`, `mix_configs_et.json` +9 |
| `mlp_worker` | 53 | Neural-network member of the E. coli mix, run in its own process. | Part 6 | — |
| `tune_analyze` | 60 | What the stage-1 search says about each hyperparameter (marginal effects on the DEV-fold rho). | Part 6 | `tune_configs.json`, `tune_stage1.csv` |
| `tune_ecoli` | 357 | Joint hyperparameter search for the E. coli model, with the selection kept apart from the confirmation. | Finding 4 | `tune_budget_check.csv`, `tune_configs.json` +5 |
| `tune_split2` | 231 | Does the tuning payoff replicate on a second gene split? | Appendix B | `tune_configs.json`, `tune_split2_configs.json` +3 |

### `sgrna/post_freeze/speed/`  —  6 modules, 1,140 lines

Fast featurisers, the per-locus precompute, and the cost benchmarks (Part 6's cost half; EXTENSION_LOG s13, s15, s22.1).

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `a_flank_fast` | 196 | Family A without a Python loop over nucleotides: the same 348 columns, computed on arrays. | Part 6 | — |
| `base_fast` | 111 | The 6,169 base columns without a Python loop over columns or guides: the same matrix as `featurise.matrix`, on arrays. | Part 6 | — |
| `cost_bench` | 129 | What the extra accuracy costs: inference latency, model size, and featurisation, on one host thread, same machine. | Part 6 | `cost_crisprhal.csv`, `cost_featurise.csv` +3 |
| `e2e_bench` | 175 | End-to-end scoring cost per guide, one thread: featurisation + column selection + imputation + model, against crisprHAL 2 with its input encoding. | Part 6 | `e2e_crisprhal.csv`, `e2e_slicer.csv` +3 |
| `equiv_check` | 57 | Recorded equivalence of the array-based featurisers with the originals, and of the predictions built on them. | Part 6 | `featuriser_equivalence.csv`, `tune_configs.json` |
| `flank_locus` | 472 | Family A from a locus instead of from a guide: the window block becomes a table lookup. | Part 6 | `flank_locus_bench.csv` |

### `sgrna/post_freeze/shape/`  —  10 modules, 1,296 lines

DNA shape and 3D descriptors of the flank, and the bound on any local predictor (Appendix B4; EXTENSION_LOG s2-3, s22.3).

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `analyze` | 51 | Paired summaries of the shape ablation: every arm against a reference arm, fold by fold. | Appendix B4 | `shape_ablation.csv`, `shape_ablation_paired.csv` |
| `dnashape` | 184 | The DNAshapeR pentamer query table, separated into the kinds of column it lumps together. | Appendix B4 | — |
| `esp_replicate` | 110 | Second, independent cohort for the shape / 3D question: crisprHAL's E. coli eSpCas9 screen. | Appendix B4 | `esp_replicate.csv` |
| `geom` | 280 | DNA shape and 3D path descriptors of the locus around a guide, by data source. | Appendix B4 | — |
| `geom_importance` | 81 | Which geometry columns does the model use, and are they re-expressions of composition? | Appendix B4 | `geom_importance.csv` |
| `local_sequence_bound` | 193 | An upper bound on what any local shape or stiffness predictor can add -- the open item Deep DNAshape left behind. | Appendix B4 | `local_sequence_bound.csv`, `local_sequence_bound_summary.csv` +1 |
| `redundancy` | 43 | How much of each shape / 3D block is already in family A (flank composition)? | Appendix B4 | `shape_redundancy_from_A.csv` |
| `ridge_check` | 73 | Is "shape / 3D adds nothing on top of family A" a property of the data or of the boosted-tree pipeline with its 300-column selector? | Appendix B4 | `ridge_check.csv` |
| `shape_ablate` | 207 | Ablations of the shape / 3D blocks on the curated cutting cohort, gene-grouped CV. | Appendix B4 | `folds_base_plus_K562_ATAC.csv`, `folds_base_plus_K562_DNase.csv` +11 |
| `shape_build` | 74 | Build the shape/3D blocks for the curated cohort, one file per (source, control, block). | Appendix B4 | — |

### `sgrna/post_freeze/human/`  —  5 modules, 682 lines

The human matrix with its genomic flanks: Appendix B4 on human, the twin-leak, fold distance (EXTENSION_LOG s10, s16.2, s20.2, s21.6).

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `fold_distance` | 94 | Sequence distance between training and validation guides, the leakage statistic our comparators report. | Part 6 | `fold_distance.csv` |
| `human_additivity_twin` | 154 | The human additivity claim and the knob sweep behind it, re-run with folds that keep repeated guides together. | Appendix B | `human_additivity_twin.csv`, `human_additivity_twin_claim.csv` +1 |
| `human_flank` | 173 | Human guides with their genomic flanks, for running Appendix B4 (shape / 3D versus family A) on human. | Part 2 | — |
| `human_shape` | 201 | Appendix B4 on human: do the shape / 3D descriptors add to family A on the human cutting matrix? | Part 2 | `folds_base_plus_K562_ATAC.csv`, `folds_base_plus_K562_DNase.csv` +13 |
| `human_twin_check` | 60 | Does the project's human baseline survive when repeat measurements of a guide are kept in the same fold? | Part 2 | `human_twin_check.csv` |

### `sgrna/post_freeze/signal/`  —  5 modules, 872 lines

What the flank signal is, the field's ceiling, and published ideas tried on our cohort (Findings 6 and 10; EXTENSION_LOG s18-21).

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `ceiling_table` | 109 | The corrected ceiling as a function of replicate agreement and replicate count -- the parametric form of Finding 6. | Part 6 | `ceiling_parametric.csv` |
| `cut_within_gene` | 36 | The cutting-screen figure on the within-gene definition used for the dCas9 screens. | Finding 3 | `cut_within_gene.csv` |
| `lit_followups` | 211 | Two ideas taken from the published predictors and tested on our cohort (Part 6 of the report). | Part 6 | `lit_binary.csv`, `lit_merf.csv` +2 |
| `mechanism` | 285 | Biology out of the model, by routes that Finding 5's redundancy caution does not disqualify. | Finding 7 | `mechanism_distance_profile.csv`, `mechanism_pairs_blocks.csv` +5 |
| `novel_followups` | 231 | Two experiments that are ours rather than borrowed: species-relative composition, and the resolution floor of the flank signal. | Finding 7 | `novel_relative_composition.csv`, `novel_resolution.csv` +1 |

### `sgrna/post_freeze/paper/`  —  5 modules, 1,805 lines

The tooling the paper rests on: the audit, the figures, and the import guard.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `code_map` | 322 | Generate the per-module tables of CODE_MAP.md from the code itself, so the map cannot drift from the files. | tooling | — |
| `doc_refs` | 165 | Every module, path and Colab script that a document or a docstring names still exists. | tooling | — |
| `figures` | 453 | Publication figures, to ACS graphics specifications, generated from the result files. | tooling | — |
| `import_check` | 106 | Every module in the package imports, and importing it writes nothing. | tooling | — |
| `verify_numbers` | 759 | Audit: recompute every headline number of Appendix B4 and the Finding 3 addendum from the result files and compare with the value typed into the report. | tooling | — |

#### Old paths, for documents written before the reorganisations

<!-- docrefs:ignore:begin -->
<details><summary>151 modules have moved; documents that cite the left-hand paths (for example `Jacky/EXPERIMENT_CATALOG.md`) can be updated from this table</summary>

| was | is now |
|---|---|
| `sgrna/asymmetry.py` | `sgrna/interpret/asymmetry.py` |
| `sgrna/attribution.py` | `sgrna/interpret/attribution.py` |
| `sgrna/base_enrichment.py` | `sgrna/interpret/base_enrichment.py` |
| `sgrna/bench.py` | `sgrna/cost/bench.py` |
| `sgrna/build_curated_features.py` | `sgrna/build/build_curated_features.py` |
| `sgrna/build_features.py` | `sgrna/build/build_features.py` |
| `sgrna/build_guide_index.py` | `sgrna/build/build_guide_index.py` |
| `sgrna/build_matrix.py` | `sgrna/build/build_matrix.py` |
| `sgrna/chi_sites.py` | `sgrna/controls/chi_sites.py` |
| `sgrna/cohort_groups.py` | `sgrna/harness/cohort_groups.py` |
| `sgrna/complexity.py` | `sgrna/interpret/complexity.py` |
| `sgrna/cost.py` | `sgrna/cost/cost.py` |
| `sgrna/crisprhal_rerun.py` | `sgrna/compare/crisprhal_rerun.py` |
| `sgrna/crisprhal_tune.py` | `sgrna/compare/crisprhal_tune.py` |
| `sgrna/crispri_screens.py` | `sgrna/compare/crispri_screens.py` |
| `sgrna/cui_final_exam.py` | `sgrna/compare/cui_final_exam.py` |
| `sgrna/cui_screen.py` | `sgrna/compare/cui_screen.py` |
| `sgrna/curated_ablation.py` | `sgrna/cohorts/curated_ablation.py` |
| `sgrna/curated_ceiling.py` | `sgrna/cohorts/curated_ceiling.py` |
| `sgrna/curated_joint.py` | `sgrna/cohorts/curated_joint.py` |
| `sgrna/curated_ladder.py` | `sgrna/cohorts/curated_ladder.py` |
| `sgrna/curated_models.py` | `sgrna/cohorts/curated_models.py` |
| `sgrna/curation_variance.py` | `sgrna/cohorts/curation_variance.py` |
| `sgrna/data_ablation.py` | `sgrna/cohorts/data_ablation.py` |
| `sgrna/decode_v_columns.py` | `sgrna/build/decode_v_columns.py` |
| `sgrna/diagnose.py` | `sgrna/controls/diagnose.py` |
| `sgrna/evaluate.py` | `sgrna/harness/evaluate.py` |
| `sgrna/expand.py` | `sgrna/build/expand.py` |
| `sgrna/ext/a_flank_fast.py` | `sgrna/post_freeze/speed/a_flank_fast.py` |
| `sgrna/ext/analyze.py` | `sgrna/post_freeze/shape/analyze.py` |
| `sgrna/ext/base_fast.py` | `sgrna/post_freeze/speed/base_fast.py` |
| `sgrna/ext/blend_with_hal.py` | `sgrna/compare/blend_with_hal.py` |
| `sgrna/ext/clean_vs_hal.py` | `sgrna/compare/clean_vs_hal.py` |
| `sgrna/ext/common.py` | `sgrna/post_freeze/common.py` |
| `sgrna/ext/cost_bench.py` | `sgrna/post_freeze/speed/cost_bench.py` |
| `sgrna/ext/cut_within_gene.py` | `sgrna/post_freeze/signal/cut_within_gene.py` |
| `sgrna/ext/data_efficiency.py` | `sgrna/compare/data_efficiency.py` |
| `sgrna/ext/dnashape.py` | `sgrna/post_freeze/shape/dnashape.py` |
| `sgrna/ext/e2e_bench.py` | `sgrna/post_freeze/speed/e2e_bench.py` |
| `sgrna/ext/equiv_check.py` | `sgrna/post_freeze/speed/equiv_check.py` |
| `sgrna/ext/esp_replicate.py` | `sgrna/post_freeze/shape/esp_replicate.py` |
| `sgrna/ext/family_a_at_tuned.py` | `sgrna/post_freeze/tuning/family_a_at_tuned.py` |
| `sgrna/ext/geom.py` | `sgrna/post_freeze/shape/geom.py` |
| `sgrna/ext/geom_importance.py` | `sgrna/post_freeze/shape/geom_importance.py` |
| `sgrna/ext/human_additivity_twin.py` | `sgrna/post_freeze/human/human_additivity_twin.py` |
| `sgrna/ext/human_flank.py` | `sgrna/post_freeze/human/human_flank.py` |
| `sgrna/ext/human_shape.py` | `sgrna/post_freeze/human/human_shape.py` |
| `sgrna/ext/human_twin_check.py` | `sgrna/post_freeze/human/human_twin_check.py` |
| `sgrna/ext/interpret.py` | `sgrna/post_freeze/tuning/interpret.py` |
| `sgrna/ext/mix_ecoli.py` | `sgrna/post_freeze/tuning/mix_ecoli.py` |
| `sgrna/ext/mlp_worker.py` | `sgrna/post_freeze/tuning/mlp_worker.py` |
| `sgrna/ext/panel_vs_hal.py` | `sgrna/compare/panel_vs_hal.py` |
| `sgrna/ext/redundancy.py` | `sgrna/post_freeze/shape/redundancy.py` |
| `sgrna/ext/ridge_check.py` | `sgrna/post_freeze/shape/ridge_check.py` |
| `sgrna/ext/shape_ablate.py` | `sgrna/post_freeze/shape/shape_ablate.py` |
| `sgrna/ext/shape_build.py` | `sgrna/post_freeze/shape/shape_build.py` |
| `sgrna/ext/topend.py` | `sgrna/compare/topend.py` |
| `sgrna/ext/topk_panel.py` | `sgrna/compare/topk_panel.py` |
| `sgrna/ext/tune_analyze.py` | `sgrna/post_freeze/tuning/tune_analyze.py` |
| `sgrna/ext/tune_ecoli.py` | `sgrna/post_freeze/tuning/tune_ecoli.py` |
| `sgrna/ext/tune_split2.py` | `sgrna/post_freeze/tuning/tune_split2.py` |
| `sgrna/ext/verify_numbers.py` | `sgrna/post_freeze/paper/verify_numbers.py` |
| `sgrna/ext/vs_crisprhal.py` | `sgrna/compare/vs_crisprhal.py` |
| `sgrna/ext/vs_crisprhal_wt_hal.py` | `sgrna/compare/vs_crisprhal_wt_hal.py` |
| `sgrna/faithfulness.py` | `sgrna/interpret/faithfulness.py` |
| `sgrna/featurise.py` | `sgrna/build/featurise.py` |
| `sgrna/final_eval.py` | `sgrna/compare/final_eval.py` |
| `sgrna/floors.py` | `sgrna/controls/floors.py` |
| `sgrna/gcstrata.py` | `sgrna/interpret/gcstrata.py` |
| `sgrna/genome.py` | `sgrna/build/genome.py` |
| `sgrna/handnamed_split.py` | `sgrna/compare/handnamed_split.py` |
| `sgrna/headtohead.py` | `sgrna/compare/headtohead.py` |
| `sgrna/heldout_compare.py` | `sgrna/compare/heldout_compare.py` |
| `sgrna/human_baseline.py` | `sgrna/compare/human_baseline.py` |
| `sgrna/importance.py` | `sgrna/interpret/importance.py` |
| `sgrna/importance_models.py` | `sgrna/interpret/importance_models.py` |
| `sgrna/inference_cost.py` | `sgrna/cost/inference_cost.py` |
| `sgrna/interp_extra.py` | `sgrna/interpret/interp_extra.py` |
| `sgrna/j_periodicity.py` | `sgrna/features/j_periodicity.py` |
| `sgrna/kingdom_transfer.py` | `sgrna/compare/kingdom_transfer.py` |
| `sgrna/make_diagnostic_blocks.py` | `sgrna/build/make_diagnostic_blocks.py` |
| `sgrna/no_information.py` | `sgrna/controls/no_information.py` |
| `sgrna/noshay_replicate.py` | `sgrna/compare/noshay_replicate.py` |
| `sgrna/notebook.py` | `sgrna/harness/notebook.py` |
| `sgrna/panel.py` | `sgrna/compare/panel.py` |
| `sgrna/portability.py` | `sgrna/compare/portability.py` |
| `sgrna/post_freeze/a_flank_fast.py` | `sgrna/post_freeze/speed/a_flank_fast.py` |
| `sgrna/post_freeze/analyze.py` | `sgrna/post_freeze/shape/analyze.py` |
| `sgrna/post_freeze/base_fast.py` | `sgrna/post_freeze/speed/base_fast.py` |
| `sgrna/post_freeze/blend_with_hal.py` | `sgrna/compare/blend_with_hal.py` |
| `sgrna/post_freeze/ceiling_table.py` | `sgrna/post_freeze/signal/ceiling_table.py` |
| `sgrna/post_freeze/clean_vs_hal.py` | `sgrna/compare/clean_vs_hal.py` |
| `sgrna/post_freeze/cost_bench.py` | `sgrna/post_freeze/speed/cost_bench.py` |
| `sgrna/post_freeze/cut_within_gene.py` | `sgrna/post_freeze/signal/cut_within_gene.py` |
| `sgrna/post_freeze/data_efficiency.py` | `sgrna/compare/data_efficiency.py` |
| `sgrna/post_freeze/dnashape.py` | `sgrna/post_freeze/shape/dnashape.py` |
| `sgrna/post_freeze/e2e_bench.py` | `sgrna/post_freeze/speed/e2e_bench.py` |
| `sgrna/post_freeze/equiv_check.py` | `sgrna/post_freeze/speed/equiv_check.py` |
| `sgrna/post_freeze/esp_replicate.py` | `sgrna/post_freeze/shape/esp_replicate.py` |
| `sgrna/post_freeze/family_a_at_tuned.py` | `sgrna/post_freeze/tuning/family_a_at_tuned.py` |
| `sgrna/post_freeze/figures.py` | `sgrna/post_freeze/paper/figures.py` |
| `sgrna/post_freeze/flank_locus.py` | `sgrna/post_freeze/speed/flank_locus.py` |
| `sgrna/post_freeze/fold_distance.py` | `sgrna/post_freeze/human/fold_distance.py` |
| `sgrna/post_freeze/geom.py` | `sgrna/post_freeze/shape/geom.py` |
| `sgrna/post_freeze/geom_importance.py` | `sgrna/post_freeze/shape/geom_importance.py` |
| `sgrna/post_freeze/human_additivity_twin.py` | `sgrna/post_freeze/human/human_additivity_twin.py` |
| `sgrna/post_freeze/human_flank.py` | `sgrna/post_freeze/human/human_flank.py` |
| `sgrna/post_freeze/human_shape.py` | `sgrna/post_freeze/human/human_shape.py` |
| `sgrna/post_freeze/human_twin_check.py` | `sgrna/post_freeze/human/human_twin_check.py` |
| `sgrna/post_freeze/import_check.py` | `sgrna/post_freeze/paper/import_check.py` |
| `sgrna/post_freeze/interpret.py` | `sgrna/post_freeze/tuning/interpret.py` |
| `sgrna/post_freeze/lit_followups.py` | `sgrna/post_freeze/signal/lit_followups.py` |
| `sgrna/post_freeze/local_sequence_bound.py` | `sgrna/post_freeze/shape/local_sequence_bound.py` |
| `sgrna/post_freeze/mechanism.py` | `sgrna/post_freeze/signal/mechanism.py` |
| `sgrna/post_freeze/mix_ecoli.py` | `sgrna/post_freeze/tuning/mix_ecoli.py` |
| `sgrna/post_freeze/mlp_worker.py` | `sgrna/post_freeze/tuning/mlp_worker.py` |
| `sgrna/post_freeze/novel_followups.py` | `sgrna/post_freeze/signal/novel_followups.py` |
| `sgrna/post_freeze/panel_vs_hal.py` | `sgrna/compare/panel_vs_hal.py` |
| `sgrna/post_freeze/redundancy.py` | `sgrna/post_freeze/shape/redundancy.py` |
| `sgrna/post_freeze/ridge_check.py` | `sgrna/post_freeze/shape/ridge_check.py` |
| `sgrna/post_freeze/shape_ablate.py` | `sgrna/post_freeze/shape/shape_ablate.py` |
| `sgrna/post_freeze/shape_build.py` | `sgrna/post_freeze/shape/shape_build.py` |
| `sgrna/post_freeze/topend.py` | `sgrna/compare/topend.py` |
| `sgrna/post_freeze/topk_panel.py` | `sgrna/compare/topk_panel.py` |
| `sgrna/post_freeze/tune_analyze.py` | `sgrna/post_freeze/tuning/tune_analyze.py` |
| `sgrna/post_freeze/tune_ecoli.py` | `sgrna/post_freeze/tuning/tune_ecoli.py` |
| `sgrna/post_freeze/tune_split2.py` | `sgrna/post_freeze/tuning/tune_split2.py` |
| `sgrna/post_freeze/verify_numbers.py` | `sgrna/post_freeze/paper/verify_numbers.py` |
| `sgrna/post_freeze/vs_crisprhal.py` | `sgrna/compare/vs_crisprhal.py` |
| `sgrna/post_freeze/vs_crisprhal_wt_hal.py` | `sgrna/compare/vs_crisprhal_wt_hal.py` |
| `sgrna/qct.py` | `sgrna/build/qct.py` |
| `sgrna/rederive.py` | `sgrna/controls/rederive.py` |
| `sgrna/redescription.py` | `sgrna/interpret/redescription.py` |
| `sgrna/replicate_weight.py` | `sgrna/cohorts/replicate_weight.py` |
| `sgrna/representation.py` | `sgrna/interpret/representation.py` |
| `sgrna/run_ablation.py` | `sgrna/harness/run_ablation.py` |
| `sgrna/sample_complexity.py` | `sgrna/compare/sample_complexity.py` |
| `sgrna/sealed_holdout.py` | `sgrna/controls/sealed_holdout.py` |
| `sgrna/selector_audit.py` | `sgrna/controls/selector_audit.py` |
| `sgrna/seqnet.py` | `sgrna/harness/seqnet.py` |
| `sgrna/shiftnull.py` | `sgrna/controls/shiftnull.py` |
| `sgrna/strand_families.py` | `sgrna/strand/strand_families.py` |
| `sgrna/strand_recoverable.py` | `sgrna/strand/strand_recoverable.py` |
| `sgrna/strand_strata.py` | `sgrna/strand/strand_strata.py` |
| `sgrna/strand_within_gene.py` | `sgrna/strand/strand_within_gene.py` |
| `sgrna/transfer.py` | `sgrna/compare/transfer.py` |
| `sgrna/tune.py` | `sgrna/harness/tune.py` |
| `sgrna/txn_interactions.py` | `sgrna/strand/txn_interactions.py` |
| `sgrna/why_inflation.py` | `sgrna/controls/why_inflation.py` |
| `sgrna/window_robustness.py` | `sgrna/controls/window_robustness.py` |
| `sgrna/xgb_knobs.py` | `sgrna/harness/xgb_knobs.py` |

</details>
<!-- docrefs:ignore:end -->

<!-- generated:end -->
