# Code map

What the code in this folder is, how it is organised, and which part of the report each module supports.
The tables between the two markers are **generated from the code** (`./run_ext.sh sgrna.paper.code_map`) and
the audit fails if a module is missing from them, so they cannot drift from the files. Everything outside the markers is
hand-written. For a short tour of only the code the report's claims depend on, read [`CODE_OVERVIEW.md`](CODE_OVERVIEW.md) first. What was deleted, and why, is in [`REMOVED.md`](REMOVED.md).

## The shape of it

Roughly 27,000 lines of Python in three layers, organised by what each module is *for* and not by when it was written
(the exact totals are the first line of the generated section below). Results are the one place the date shows: `results/`
holds numbers from before the 11 October freeze and `results/post_freeze/` those from after it.

| layer | where | what it is |
|---|---|---|
| **the pipeline** | `sgrna/build`, `features`, `harness` | The library. Decode the published matrix into a 20-mer, locate every guide on the genome, build the feature families (and the array featurisers that make scoring fast), and run a leakage-safe cross-validation harness. Nothing here answers a scientific question by itself. |
| **the experiments** | `sgrna/tuning`, `cohorts`, `controls`, `interpret`, `compare`, `cost` | One script per question. `tuning` is the hyperparameter search; `cohorts` the ablations and ceilings; `controls` the nulls and audits; `interpret` what the model uses and what the signal is made of; `compare` everything set against something else (crisprHAL 2, other organisms, other assays); `cost` what it takes to run. They back Findings 1–7, Part 6 and the appendices. |
| **the tooling** | `sgrna/paper` | The audit, the figures, the import and reference guards, and the generator of this map. |

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
2. **`run()`** does the experiment through the shared harness (`harness/run_ablation.py`, or `tuning/tune_ecoli.py`
   for the tuned model), with folds written to `results/` as they finish so a crash loses one fold, not a run.
3. **One CSV** lands in `results/` (before the freeze) or `results/post_freeze/` (after it), named after the module.
4. **A `__main__` guard**, so importing a module does nothing. `paper/import_check.py` asserts this for all of them.

## How to run something

```bash
./run_ext.sh sgrna.build.a_flank_fast              # any module, by dotted path; sets the three variables it needs
./run_ext.sh sgrna.cohorts.curated_ablation --all              # the frozen modules run the same way
./run_audit.sh                                                  # recompute every number the report quotes from its file
```

`run_ext.sh` exists because the modules need `RSI09_ROOT` (the folder holding `data/`), `EXT_OUT` (where results go) and
`PYTHONPATH` set together, and getting one wrong fails with a `FileNotFoundError` that does not say which. **Never run
`sync_to_shared.sh`.**

<!-- generated:begin -->

**174 modules, 37,838 lines** of Python in 11 folders (not counting the `__init__.py` markers).

#### By the report section each module supports

| report section | what it is | modules | lines | result files | where the code lives |
|---|---|---:|---:|---:|---|
| Finding 1 | The published representation is a re-encoding of the guide, and holds about twenty variables | 5 | 770 | 27 | build, cohorts, compare, controls, paper |
| Finding 2 | The DNA around the cut site predicts cutting — the one large effect, and it is local and compositional | 5 | 1,162 | 30 | compare, controls, harness, interpret, paper |
| Finding 3 | The effect belongs to the DNA, and transfers | 4 | 1,134 | 34 | build, compare, paper |
| Finding 4 | The model is out of *material*, and the tuning gain is evidence for it | 3 | 922 | 13 | interpret, tuning |
| Finding 5 | On a redundant matrix, per-column importance is unstable across fits, but SHAP at block level and edit tests are faithful; SLICER is interpretable, more faithful than crisprHAL 2, and ties or splits with it on stability, truth and compressibility depending on the reading used | 10 | 3,054 | 54 | interpret, paper |
| Finding 6 | The label's ceiling is about 0.946 (0.942 to 0.946 by estimator), measured from crisprHAL's own replicates | 1 | 122 | 1 | cohorts |
| Finding 7 | What the signal is made of: GC over a few hundred bases, a seed region that inverts between kingdoms, and the two coupled | 5 | 1,450 | 35 | compare, interpret, paper |
| Part 2 | Contributions | 6 | 722 | 7 | cohorts, compare, controls, interpret |
| Part 3 | Where this stops | 1 | 179 | 2 | compare |
| Part 4 | The freeze, and what remains | 5 | 792 | 5 | compare, controls |
| Part 5 | Where this sits in the field | 37 | 6,515 | 117 | build, cohorts, compare, controls, cost, interpret, paper, tuning |
| Appendix A | Controls and nulls | 4 | 1,609 | 64 | controls, harness, paper |
| Appendix B | The methods and controls behind the findings | 18 | 5,880 | 365 | cohorts, compare, controls, cost, harness, interpret, paper, tuning |
| Appendix B2 | Why the imported families failed, and why that is not a screen | 2 | 472 | 4 | controls, interpret |
| Appendix B2a | Curation: what it is, and why family A doubled between cohorts | 3 | 562 | 6 | cohorts, controls |
| Appendix B3 | The family ablations on the curated cohort, and what they measure | 13 | 2,528 | 5 | cohorts, features, interpret |
| Appendix B4 | The flank's shape and 3D structure add nothing that its composition does not already hold | 10 | 1,535 | 25 | build, cohorts, compare, features, interpret |
| Appendix B5 | A boundary experiment, demoted: the dCas9 / CRISPRi screens | 8 | 2,014 | 65 | compare, paper |
| Appendix C0 | Reproducibility: versions, seeds and protocol | 1 | 199 | 0 | (top) |
| tooling | the audit, the figures, the guards and this map | 5 | 2,120 | 0 | paper |
| library | shared code the experiments are built on | 9 | 1,352 | 2 | (top), build |
| — | no section of the report points at it | 19 | 2,745 | 35 | cohorts, compare, cost, interpret, paper |

Read the table above as the price list for the report: each claim carries its code and its result files with it. A module is placed in the section that cites it (or the result files it writes) most often. The `—` row, if there is one, is completed work that no section of the report points at. Each such module is listed in `REMOVED.md` under "Considered for removal and kept", with what it found; they were kept by decision, not by oversight.

#### Result files that exist only outside this repository

None: every result file a module writes by name is tracked, apart from the bulk prediction and per-fold records.

#### Package by package

### `sgrna/`  —  3 modules, 334 lines

Paths and constants every module shares (`config.py`), the conventions for reading the base matrix and writing feature blocks (`io_utils.py`), and the resolver for the main project's untracked results (`common.py`).

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `common` | 47 | Shared paths for the work done after the freeze (the private-fork experiments). | library | `sample_complexity.csv` |
| `config` | 199 | Project paths and shared constants. | Appendix C0 | — |
| `io_utils` | 88 | Shared conventions for reading the base matrix and writing feature blocks. | library | — |

### `sgrna/build/`  —  13 modules, 2,665 lines

Decode the published matrix, locate every guide on the genome, rebuild features for guides outside it, and the array featurisers that score a guide in tens of microseconds.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `a_flank_fast` | 196 | Family A without a Python loop over nucleotides: the same 348 columns, computed on arrays. | Part 5 | — |
| `base_fast` | 111 | The 6,169 base columns without a Python loop over columns or guides: the same matrix as `featurise.matrix`, on arrays. | Part 5 | — |
| `build_curated_features` | 174 | Build feature families on crisprHAL's curated cohort, not the published rows. | library | — |
| `build_features` | 89 | Step 1 -- build one or more feature families. | library | — |
| `build_guide_index` | 166 | Step 0 -- turn 13,880 anonymous rows into located guides. | library | — |
| `build_matrix` | 193 | Step 2 -- assemble a model-ready matrix from the base features plus blocks. | library | — |
| `decode_v_columns` | 196 | Work out what the 5,888 `V####` columns are, so new guides can be featurised. | library | — |
| `expand` | 176 | Lift the 13,880-row cap: featurise every guide in the curated Guo screen. | library | `learning_curve.csv` |
| `featurise` | 126 | Generate the published feature representation for guides outside the matrix. | Finding 1 | — |
| `flank_locus` | 472 | Family A from a locus instead of from a guide: the window block becomes a table lookup. | Part 5 | `flank_locus_bench.csv` |
| `genome` | 255 | Reference genome helpers for E. coli K-12 MG1655 (NC_000913.2). | Finding 3 | — |
| `make_diagnostic_blocks` | 223 | Decompose the positional families, to find out what they are really measuring. | library | — |
| `qct` | 288 | Quantum chemical tensor (QCT) decoding and extension. | Appendix B4 | — |

### `sgrna/features/`  —  14 modules, 2,635 lines

The eleven feature families A-J (family A, the flanking-DNA composition, is the one that matters) and the shape / 3D descriptors that Appendix B4 tested against it.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `a_flank` | 240 | Family A -- sequence and quantum descriptors outside the 20 nt protospacer. | Appendix B3 | — |
| `b_energy` | 164 | Family B -- the R-loop free-energy decomposition (CRISPRoff / CRISPRspec). | Appendix B3 | — |
| `c_folding` | 155 | Family C -- sgRNA folding, done on the transcript that is actually made. | Appendix B3 | — |
| `d_mechanics` | 201 | Family D1 -- how hard the target duplex is to open, position by position. | Appendix B3 | — |
| `d_supercoiling` | 233 | Family D2 -- DNA supercoiling at the target, from GapR-seq. | Appendix B3 | — |
| `dnashape` | 184 | The DNAshapeR pentamer query table, separated into the kinds of column it lumps together. | Appendix B4 | — |
| `e_nucleoid` | 226 | Family E -- nucleoid organisation, the bacterial answer to chromatin. | Appendix B3 | — |
| `f_methylation` | 133 | Family F -- Dam and Dcm methylation at the target site. | Appendix B3 | — |
| `g_transcription` | 275 | Family G -- transcription and replication context. | Appendix B3 | — |
| `geom` | 280 | DNA shape and 3D path descriptors of the locus around a guide, by data source. | Appendix B4 | — |
| `h_offtarget` | 133 | Family H -- off-target burden and target copy number. | Appendix B3 | — |
| `i_shape` | 208 | Family I -- the physical shape of the DNA around the target, not its letters. | Appendix B3 | — |
| `j_periodicity` | 129 | Family J -- strand asymmetry and helical periodicity in the flanking DNA. | Appendix B3 | — |
| `shape_build` | 74 | Build the shape/3D blocks for the curated cohort, one file per (source, control, block). | Appendix B4 | — |

### `sgrna/harness/`  —  5 modules, 2,156 lines

The leakage-safe cross-validation harness every experiment runs through, plus the notebook front end.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `cohort_groups` | 74 | Does grouped cross-validation still mean the same thing on the curated cohort? | Appendix B | `cohort_groups.csv` |
| `evaluate` | 708 | Quantifying the two things a model is judged on: prediction and explanation. | Appendix B | `interpretability.csv`, `model_comparison.csv` |
| `notebook` | 532 | Notebook front end -- everything the pipeline does, callable from a cell. | Appendix A | `ablation_abund_split.csv`, `ablation_curated.csv` +11 |
| `run_ablation` | 560 | Step 3 -- the ablation harness. | Appendix A | `ablation_abund_split.csv`, `ablation_curated.csv` +25 |
| `seqnet` | 282 | Does a convolutional net on raw sequence beat gradient boosting on 6,580 engineered columns? | Finding 2 | `seqnet.csv`, `seqnet_matched.csv` |

### `sgrna/tuning/`  —  8 modules, 1,633 lines

Hyperparameter search and model mixing: the joint search on development genes, its replication on a second split, the blend of model families, and the knobs no search had touched.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `family_a_at_tuned` | 159 | Is family A still worth +0.164 once the model is tuned? | Finding 4 | `family_a_at_tuned.csv`, `family_a_at_tuned_summary.csv` +1 |
| `mix_ecoli` | 289 | Mixing model families for the E. coli model: does an ensemble that is not all XGBoost beat the tuned XGBoost ensemble? | Part 5 | `mix_configs_cat.json`, `mix_configs_et.json` +9 |
| `mlp_worker` | 53 | Neural-network member of the E. coli mix, run in its own process. | Part 5 | — |
| `tune` | 241 | Does the head-to-head margin survive hyper-parameter tuning? | Part 5 | `tuning.csv`, `tuning_trace.csv` |
| `tune_analyze` | 60 | What the stage-1 search says about each hyperparameter (marginal effects on the DEV-fold rho). | Part 5 | `tune_configs.json`, `tune_stage1.csv` |
| `tune_ecoli` | 357 | Joint hyperparameter search for the E. coli model, with the selection kept apart from the confirmation. | Finding 4 | `tune_budget_check.csv`, `tune_configs.json` +5 |
| `tune_split2` | 231 | Does the tuning payoff replicate on a second gene split? | Appendix B | `tune_configs.json`, `tune_split2_configs.json` +3 |
| `xgb_knobs` | 243 | Which XGBoost knobs has this project never touched, and do they matter? | Appendix B | `xgb_knobs.csv`, `xgb_knobs_verify.csv` |

### `sgrna/cohorts/`  —  17 modules, 2,374 lines

Ablations on the curated cohort and the arguments about what its numbers mean (ceilings, inflation, label noise), including the shape ablations of Appendix B4.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `ablation_paper_folds` | 125 | Feature-family ablation on the paper's own folds. | — | `flank_scale.csv` |
| `ceiling_table` | 109 | The corrected ceiling as a function of replicate agreement and replicate count -- the parametric form of Finding 6. | Part 5 | `ceiling_parametric.csv` |
| `curated_ablation` | 246 | Feature-family ablations on crisprHAL's curated cohort. | Appendix B3 | `ablation_curated.csv`, `ablation_curated_gene.csv` +1 |
| `curated_ceiling` | 122 | The ceiling for the label we actually train on, from replicate agreement. | Finding 6 | `curated_ceiling.json` |
| `curated_joint` | 122 | Joint arms on the curated cohort: how much of the eleven family gains overlap? | Part 2 | `ablation_curated.csv`, `curated_joint.csv` |
| `curated_ladder` | 123 | The reduced-representation ladder, re-run on the curated cohort without selection. | Finding 1 | `curated_ladder.csv` |
| `curated_models` | 95 | Does model class matter more once the representation has family A in it? | Appendix B | `curated_models.csv` |
| `curation_check` | 86 | Does the curation manufacture the flank effect? | — | `curation_check.csv` |
| `curation_variance` | 159 | Why did every family's gain rise when the baseline rose too? | Appendix B2a | `curation_variance.csv` |
| `data_ablation` | 152 | Hold SLICER fixed and change only the training data. | Part 5 | `data_ablation.csv` |
| `esp_replicate` | 110 | Second, independent cohort for the shape / 3D question: crisprHAL's E. coli eSpCas9 screen. | Appendix B4 | `esp_replicate.csv` |
| `local_sequence_bound` | 193 | An upper bound on what any local shape or stiffness predictor can add -- the open item Deep DNAshape left behind. | Appendix B4 | `local_sequence_bound.csv`, `local_sequence_bound_summary.csv` +1 |
| `replicate_weight` | 295 | The label's reliability is per guide, and the model treats it as uniform. | Appendix B | `replicate_weight.csv`, `replicate_weight_raw.csv` |
| `ridge_check` | 73 | Is "shape / 3D adds nothing on top of family A" a property of the data or of the boosted-tree pipeline with its 300-column selector? | Appendix B4 | `ridge_check.csv` |
| `shape_ablate` | 207 | Ablations of the shape / 3D blocks on the curated cutting cohort, gene-grouped CV. | Part 5 | `folds_base_plus_K562_ATAC.csv`, `folds_base_plus_K562_DNase.csv` +13 |
| `shape_analyze` | 51 | Paired summaries of the shape ablation: every arm against a reference arm, fold by fold. | Appendix B4 | `shape_ablation.csv`, `shape_ablation_paired.csv` |
| `txn_interactions` | 106 | Family G's own rule, applied on the curated cohort: do the interactions pay? | Part 2 | `ablation_curated.csv`, `txn_interactions.csv` |

### `sgrna/controls/`  —  15 modules, 2,906 lines

The adversarial phase: nulls, shuffles, floors, selector and leakage audits, and the check that the fast featurisers equal the originals.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `chi_sites` | 198 | Is the flank effect a *repair* effect? | Appendix B | `chi_sites.csv` |
| `diagnose` | 429 | Why doesn't adding features help? | Appendix B2 | `ceiling.json`, `headroom.csv` +1 |
| `equiv_check` | 57 | Recorded equivalence of the array-based featurisers with the originals, and of the predictions built on them. | Part 5 | `featuriser_equivalence.csv`, `tune_configs.json` |
| `floors` | 86 | Per-family permutation floors, paired fold by fold. | Appendix A | `folds_base_plus_K562_ATAC.csv`, `folds_base_plus_K562_DNase.csv` +13 |
| `fold_distance` | 94 | Sequence distance between training and validation guides, the leakage statistic our comparators report. | Part 5 | `fold_distance.csv` |
| `mappability_stratum` | 299 | Is the flank effect read depth or mappability? | Appendix B2a | `mappability_depth.csv`, `mappability_stratum.csv` +2 |
| `no_information` | 94 | Decompose `d_mechanics`'s +0.099 into representation and new DNA. | Finding 1 | `no_information.csv` |
| `rederive` | 543 | Re-derive the result files whose producer was lost, and check they agree. | Appendix B | `ablation_curated.csv`, `ablation_curated_gene.csv` +5 |
| `sealed_crossscreen` | 105 | A second screen as the sealed set: train on SpCas9 guides in development genes, score eSpCas9 labels in the sealed genes. | Part 4 | `sealed_crossscreen.csv` |
| `sealed_holdout` | 224 | Can the adaptive-overfitting caveat be measured by sealing part of the cohort? | Part 4 | `sealed_holdout.csv` |
| `sealed_holdout_gene` | 147 | Splitting the sealed-gene drop: adaptive overfitting, or what unseen genes cost a feature that reads the locus? | Part 4 | `sealed_holdout_gene.csv` |
| `selector_audit` | 272 | The pipeline selects columns with the one metric the report says never to use. | Appendix B | `selector_audit.csv`, `selector_audit_ranks.csv` +1 |
| `shiftnull` | 140 | A spatially preserving null for family A: shift the chromosome, keep its texture. | Finding 2 | `shift_null.csv` |
| `why_inflation` | 104 | Why did every family's Δρ roughly triple when the cohort moved? | Appendix B2a | `why_inflation.csv` |
| `window_robustness` | 114 | Would a different set of flank windows have given the same answer? | Part 2 | `ablation_curated.csv`, `window_robustness.csv` |

### `sgrna/interpret/`  —  25 modules, 6,174 lines

What the model uses and what the signal is made of: importance, attribution, redundancy, the mechanism work of Finding 7, and what the tuned settings say.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `asymmetry` | 237 | Is the downstream flank's advantage a transcription effect? | — | `asymmetry_expression.csv`, `asymmetry_strand_audit.csv` |
| `attribution` | 214 | Two models, different features, same score. | Finding 5 | `attribution.csv`, `attribution_shap.csv` +2 |
| `base_enrichment` | 185 | Is the curated table inflated because the base has no dense summaries? | Appendix B3 | `base_enrichment.csv`, `base_enrichment_published.csv` |
| `compact_model` | 214 | Is there a representation on which SLICER can be read column by column? | Finding 5 | `compact_model_accuracy.csv`, `compact_model_accuracy_folds.csv` +5 |
| `complexity` | 118 | What is SLICER's equivalent of a CNN's parameter count? | Part 5 | `complexity.json` |
| `faithfulness` | 270 | Dropping a model's top columns costs nothing. | Finding 5 | `faithfulness_clusters.csv`, `faithfulness_clusters_reduced.csv` +2 |
| `flank_scale` | 191 | How far does the flank signal reach, and how much of it is composition resolved to the base? | — | `flank_scale.csv`, `flank_scale_summary.csv` |
| `flank_window_tune` | 156 | Tuning the two flank windows, upstream and downstream, separately. | — | `flank_window_tune.csv`, `flank_window_tune_summary.csv` |
| `gate_test` | 65 | Do the flank and the guide's own score act as a gate (multiplicatively) or add up? | — | `gate_test.csv`, `mappability_stratum_oof.csv` |
| `gcstrata` | 210 | Is the flanking-context effect just target GC wearing a different hat? | Finding 2 | `gc_strata_ablation.csv`, `gc_strata_corr.csv` +1 |
| `geom_importance` | 81 | Which geometry columns does the model use, and are they re-expressions of composition? | Appendix B4 | `geom_importance.csv` |
| `hal_readout` | 273 | crisprHAL 2's side of the interpretability comparison (`vs_crisprhal_interpret.py` has the design and the predictions). | Finding 5 | `guides_1500.csv`, `hal_eg_draws_control.csv` +2 |
| `hypotheses` | 401 | Three tests of what the flank effect might physically be, each with its prediction written down before it was run. | Finding 7 | `hypotheses_audit_gate.csv`, `hypotheses_audit_scale.csv` +15 |
| `importance` | 293 | At what granularity is "feature importance" actually a property of the data? | Finding 5 | `importance_agreement.csv`, `importance_blocks.csv` |
| `importance_models` | 297 | Is "feature importance" a property of the data, or of the model asked? | Finding 5 | `importance_model_blocks.csv`, `importance_model_methods.csv` +1 |
| `interp_extra` | 301 | Three interpretability tests the report listed as not yet run. | Finding 5 | `interp_counterfactual.csv`, `interp_interaction_blocks.csv` +3 |
| `label_orientation` | 40 | Is a higher curated label more cutting? | Part 2 | `label_orientation.csv` |
| `mechanism` | 285 | Biology out of the model, by routes that Finding 5's redundancy caution does not disqualify. | Finding 7 | `mechanism_distance_profile.csv`, `mechanism_pairs_blocks.csv` +5 |
| `novel_followups` | 231 | Two experiments that are ours rather than borrowed: species-relative composition, and the resolution floor of the flank signal. | Finding 7 | `novel_relative_composition.csv`, `novel_resolution.csv` +1 |
| `redescription` | 741 | How much is re-description worth, and does it have to mean anything? | Appendix B | `recoverability_percolumn.csv`, `recoverability_trend.csv` +7 |
| `redundancy` | 43 | How much of each shape / 3D block is already in family A (flank composition)? | Appendix B2 | `shape_redundancy_from_A.csv` |
| `representation` | 406 | Does the model need all 6,232 published columns, or one spelling of the 20-mer? | Finding 4 | `representation.csv`, `representation_depth.csv` +2 |
| `signal_profile` | 132 | The data behind the figure of what the flank signal looks like: the step, the slope by the guide's last base, and the correlation by window and by base. | — | `mappability_stratum_oof.csv`, `signal_corr.csv` +3 |
| `tuned_settings` | 199 | What the tuned hyperparameters say about the signal: four tests, each answering a question the choice of settings raises. | Finding 5 | `interpret_data_scaling.csv`, `interpret_depth.csv` +4 |
| `vs_crisprhal_interpret` | 591 | Is SLICER more interpretable than crisprHAL 2? | Finding 5 | `guides_1500.csv`, `hal_flank_edits.csv` +14 |

### `sgrna/compare/`  —  43 modules, 7,786 lines

Every comparison with something else (Part 5 of the report): crisprHAL 2, the inherited baseline, other organisms and the human matrix, and other assays (the dCas9 boundary experiment of Appendix B5).

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `blend_with_hal` | 47 | Is crisprHAL 2's network complementary to the tuned trees? | Part 5 | `blend_with_hal.csv` |
| `boostmec_replica` | 291 | BoostMEC built the way its authors built it, retrained on the curated E. coli cohort and scored with SLICER's folds. | — | `boostmec_released.csv`, `boostmec_search_trace.csv` +2 |
| `clean_vs_hal` | 65 | A comparison with crisprHAL 2 that the tuning cannot have touched: the 30% of genes held out from every search. | Part 5 | `clean_vs_hal.csv` |
| `crisprhal_rerun` | 151 | Re-run crisprHAL 2 ourselves, on our rows and our folds. | Appendix B | `crisprhal_rerun.csv` |
| `crisprhal_tune` | 284 | Give crisprHAL 2 the same courtesy we gave ourselves: a hyperparameter search. | Appendix B | `crisprhal_tuned_folds.csv`, `crisprhal_tuning.csv` |
| `crisprhal_wide_window_folds` | 191 | crisprHAL 2 at wider input, on the folds the first run did not cover (local CPU; TensorFlow env `.venv-crisprhal`, not the main env). | Finding 2 | `crisprhal_wide_window.csv`, `crisprhal_wide_window_folds.csv` +2 |
| `crispri_screens` | 327 | Three CRISPRi screens instead of one, and what that buys beyond accuracy. | Appendix B5 | `crispri_screen_agreement.csv`, `crispri_screen_family_a.csv` +1 |
| `cui_final_exam` | 219 | Both models, neither trained on it, scored on Cui's CRISPRi screen. | Part 5 | `cui_final_exam.csv` |
| `cui_screen` | 594 | The last untouched test: does the flanking-context effect appear in a CRISPRi screen? | Appendix B5 | `cui_essential_marginals.csv`, `cui_essential_position.csv` +4 |
| `cut_within_gene` | 36 | The cutting-screen figure on the within-gene definition used for the dCas9 screens. | Appendix B5 | `cut_within_gene.csv` |
| `data_efficiency` | 67 | The data-efficiency curve again, with the tuned model against the stored crisprHAL 2 curve. | Part 5 | `data_efficiency.csv`, `sample_complexity.csv` +1 |
| `eukaryote_rules` | 309 | Do the guide's own design rules, and the flank effect, carry between eukaryotes, and how do they differ from E. coli's? | Finding 7 | `eukaryote_contrast_g_minus_t.csv`, `eukaryote_flank_step.csv` +4 |
| `final_eval` | 247 | The designated final evaluation: both models on crisprHAL's eSpCas9 split. | Part 5 | `final_eval_espcas9.csv` |
| `gene_grouped_vs_crisprhal` | 84 | The head-to-head with crisprHAL 2 on folds that hold whole genes out (the lowest-leakage protocol of this project). | — | `crisprhal_gene_grouped_fcolab_b1024.csv`, `crisprhal_gene_grouped_fcolab_b256.csv` +2 |
| `handnamed_split` | 122 | Split the 63 hand-named columns into the computable and the rest, and test both. | Finding 1 | `handnamed_split.csv`, `portability_audit.csv` |
| `headtohead` | 303 | The like-for-like comparison against crisprHAL 2. | Part 5 | `headtohead.csv` |
| `heldout_compare` | 156 | How do the two models compare on a held-out fold? | Part 5 | `heldout_compare.csv` |
| `human_additivity_twin` | 154 | The human additivity claim and the knob sweep behind it, re-run with folds that keep repeated guides together. | Appendix B | `human_additivity_twin.csv`, `human_additivity_twin_claim.csv` +1 |
| `human_baseline` | 270 | Does the human baseline reproduce ρ ≈ 0.472 under this harness? | Appendix B | `human_additivity.csv`, `human_baseline.csv` +1 |
| `human_flank` | 173 | Human guides with their genomic flanks, for running Appendix B4 (shape / 3D versus family A) on human. | Part 2 | — |
| `human_shape` | 201 | Appendix B4 on human: do the shape / 3D descriptors add to family A on the human cutting matrix? | Appendix B4 | `folds_base_plus_K562_ATAC.csv`, `folds_base_plus_K562_DNase.csv` +15 |
| `human_twin_check` | 60 | Does the project's human baseline survive when repeat measurements of a guide are kept in the same fold? | Finding 3 | `human_twin_check.csv` |
| `kingdom_transfer` | 179 | Cross-kingdom transfer, with the human side fitted properly. | Part 3 | `kingdom_transfer.csv`, `kingdom_transfer_boot.csv` |
| `lit_followups` | 211 | Two ideas taken from the published predictors and tested on our cohort (Part 5 of the report). | Part 5 | `lit_binary.csv`, `lit_merf.csv` +2 |
| `metrics_panel` | 122 | Beyond rank correlation: the other things a guide-efficiency model is judged on, for SLICER and crisprHAL 2 on crisprHAL 2's own test files. | — | `metrics_panel.csv` |
| `noshay_replicate` | 232 | Reproduce the inherited paper's own model, as its methods section describes it. | Part 5 | `noshay_replication.csv` |
| `panel` | 219 | The full metric panel, for untuned and tuned, on identical folds. | Part 4 | `metric_panel.csv` |
| `panel_vs_hal` | 110 | The six-metric panel against crisprHAL 2 on its own two splits, for the tuned and mixed models, with raw predictions kept. | Part 5 | `panel_espcas9.csv`, `panel_wt.csv` |
| `portability` | 167 | Can the 427-column representation be rebuilt for a guide we have never seen? | Part 2 | `portability.csv`, `portability_audit.csv` |
| `sample_complexity` | 220 | How many labelled guides does each method need? | Part 5 | `sample_complexity.csv` |
| `sealed_organism` | 97 | Does the tuned model's advantage survive a dataset that no choice in this project ever saw? | Part 4 | `sealed_organism.csv` |
| `strand_families` | 241 | With the strand call supplied, how do all eleven families compare on the dCas9 screen? | Appendix B5 | `strand_families.csv`, `strand_families_proxies.csv` +1 |
| `strand_recoverable` | 47 | How well can the strand call be read back from each family's columns, after proxy removal? | Appendix B5 | `strand_families_recoverable.csv` |
| `strand_strata` | 142 | Two controls on the strand result. | Appendix B5 | `strand_flip.csv`, `strand_strata_rousset.csv` |
| `strand_within_gene` | 181 | Inside the coding-strand guides, is a family's gain between genes or within them? | Appendix B5 | `strand_within_gene_cui.csv`, `strand_within_gene_cui_joint.csv` +1 |
| `topend` | 96 | Do objectives aimed at the top of the ranking improve top-end metrics, where crisprHAL 2 still leads on pick percentile? | Part 5 | `topend_confirm.csv`, `tune_configs.json` |
| `topk_panel` | 97 | Smoother top-end metrics against crisprHAL 2 on its own splits, and seed stability of the lead. | Part 5 | `seed_stability.csv`, `topk_panel.csv` |
| `transfer` | 588 | Does the flank effect transfer to another nuclease? | Finding 3 | `hybrid.csv`, `transfer.csv` +6 |
| `transfer_diagonal` | 39 | Why the diagonal of the transfer table is low: how well the six published predictors do on the same tables. | — | `eukaryote_transfer.csv`, `transfer_diagonal.csv` |
| `transfer_to_ecoli` | 48 | The E. coli column of the transfer matrix: models trained on each eukaryotic table, scored on the 33,567 curated E. coli guides. | — | `transfer_to_ecoli.csv` |
| `transfer_vs_crisprhal` | 178 | Transferability, SLICER against crisprHAL 2: across nucleases within E. coli, and from E. coli to the eukaryotic tables. | — | `eukaryote_transfer.csv`, `transfer_vs_crisprhal_kingdom.csv` +1 |
| `vs_crisprhal` | 186 | The tuned / mixed E. coli model against crisprHAL 2 again, under the three conditions of the original comparison. | Part 5 | `mix_finalists.csv`, `tune_configs.json` +3 |
| `vs_crisprhal_wt_hal` | 35 | crisprHAL 2's shipped WT-SpCas9 weights on their own WT test file, saved for a paired comparison (TensorFlow environment). | Part 5 | — |

### `sgrna/cost/`  —  10 modules, 2,007 lines

What each model costs to train and to run, measured rather than estimated.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `bench` | 319 | Peak memory and training cost, measured rather than estimated. | — | `memory_comparison.csv`, `memory_comparison_runs.csv` +1 |
| `cost` | 372 | How much computation does each model actually cost, and how light can SLICER get? | Part 5 | `cost_cpu_seconds.csv`, `cost_flops.csv` +1 |
| `cost_bench` | 129 | What the extra accuracy costs: inference latency, model size, and featurisation, on one host thread, same machine. | Part 5 | `cost_crisprhal.csv`, `cost_featurise.csv` +3 |
| `cost_boostmec` | 188 | What BoostMEC costs, measured with its released model, its own feature definitions and the same machine and protocol as SLICER and crisprHAL 2. | — | `boostmec_oracle.csv`, `model_size.csv` |
| `cost_boostmec_shipped` | 240 | BoostMEC as shipped: the authors' own pipeline, run unmodified in their Docker image, timed and compared with the Python version of `cost_boostmec.py`. | Appendix B | `ablation_abund_split.csv`, `ablation_curated.csv` +363 |
| `cost_repeats` | 246 | Cost with its spread: every timing repeated, every repeat kept, so a ratio comes with a distribution and not a bare range. | — | `cost_repeats_fit.csv`, `cost_repeats_inference.csv` +1 |
| `e2e_bench` | 175 | End-to-end scoring cost per guide, one thread: featurisation + column selection + imputation + model, against crisprHAL 2 with its input encoding. | Part 5 | `e2e_crisprhal.csv`, `e2e_slicer.csv` +3 |
| `inference_cost` | 141 | Training cost is the wrong clock for a guide-design tool. | Part 5 | `inference_cost.csv` |
| `model_size` | 91 | What has to be stored to run each model: trees, leaves and bytes for SLICER's default and tuned models, parameters and bytes for crisprHAL 2. | — | `model_size.csv` |
| `rds_read` | 106 | Minimal read-only parser for R's XDR serialisation (.rds), enough for plain data (lists, vectors, attributes). | — | — |

### `sgrna/paper/`  —  21 modules, 7,168 lines

The tooling the paper rests on: the audit, the figures, the import and reference guards, and the code map.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `code_map` | 420 | Generate the per-module tables of CODE_MAP.md from the code itself, so the map cannot drift from the files. | tooling | — |
| `doc_refs` | 165 | Every module, path and Colab script that a document or a docstring names still exists. | tooling | — |
| `figures` | 427 | Publication figures, to ACS graphics specifications, generated from the result files. | tooling | — |
| `figures_paper` | 405 | Figures for the point-form paper, every one drawn from stored result files (no schematics). | Part 5 | `cost_gpu.csv`, `cost_repeats_fit.csv` +13 |
| `import_check` | 107 | Every module in the package imports, and importing it writes nothing. | tooling | — |
| `paper_numbers` | 130 | Audit of every number in the paper (point form): each is recomputed from a stored result file, as `verify_numbers` does for the report. | Part 5 | `blend_with_hal.csv`, `clean_vs_hal.csv` +7 |
| `paper_numbers_A` | 431 | Audit of the numbers in the draft paper's claims [0]-[19], [35]-[45] (group A): data sets and methods, the flank effect, its mechanism and the appendix constraints. | Appendix A | `ablation_curated.csv`, `ablation_curated_gene.csv` +41 |
| `paper_numbers_B` | 231 | Audit of the numbers in the draft paper's claims [20]-[28], [32]-[34] and [49] (group B): every number is recomputed from a stored result file and compared with the value the paper prints. | Finding 3 | `ablation_curated.csv`, `ceiling_parametric.csv` +25 |
| `paper_numbers_C` | 326 | Audit of paper claims [29]-[31], [46]-[48], [50]-[55] (group C): every number is recomputed from a stored result file. | Part 5 | `ablation_curated.csv`, `ablation_label_curated.csv` +38 |
| `paper_numbers_D` | 536 | Audit of the numbers added to the paper (point form) in the second pass of 23 Nov 2026: one protocol, the reach of the flank signal, controls, the transfer matrix, SLICER against crisprHAL 2 on more metrics … | Appendix B | `ablation_curated.csv`, `boostmec_oracle.csv` +40 |
| `report_checks_CIT` | 90 | Checks on the report's citation-related numbers (tag CIT): each `got` is computed from a stored file. | Part 5 | `noshay_replication.csv` |
| `report_checks_R1` | 305 | Report checks R1: numbers of PROGRESS_REPORT.md lines 1-120 (the Summary, 'How to read', 'What earns a place', Finding 1) that had no line in the audit, recomputed from a stored file or from the repository's own code. | Finding 1 | `ablation_curated.csv`, `ablation_curated_gene.csv` +25 |
| `report_checks_R2` | 339 | Audit of PROGRESS_REPORT.md lines 121-307 (Finding 2 on the flank effect, Finding 3 on transfer and confound controls, Finding 4 on the model being out of material): the headline numbers that `verify_numbers` did not recompute … | Finding 2 | `ablation_curated.csv`, `ablation_main.csv` +26 |
| `report_checks_R3` | 402 | Checks behind review R3 of PROGRESS_REPORT.md (Finding 5, interpretability; Finding 6, the label's ceiling). | Finding 5 | `additivity_reduced.csv`, `attribution_summary.csv` +24 |
| `report_checks_R4` | 224 | Checks for PROGRESS_REPORT.md Finding 7 (what the signal is made of) that the main audit does not read, or reads under another label. | Finding 7 | `ablation_curated.csv`, `ablation_curated_gene.csv` +12 |
| `report_checks_R5` | 320 | Checks R5: headline numbers of PROGRESS_REPORT.md lines 576-853 (Part 2 contributions, Part 3 limits, Part 4 freeze) that had no audit line. | Appendix B | `ablation_curated.csv`, `ablation_curated_gene.csv` +19 |
| `report_checks_R6` | 297 | Checks for headline numbers of PROGRESS_REPORT.md Part 5 (the head-to-head with crisprHAL 2, cost, other comparisons, what other predictors read) that had no audit line, computed against the corrected wording of the report. | Part 5 | `ablation_curated.csv`, `ablation_curated_gene.csv` +37 |
| `report_checks_R7` | 525 | Report checks R7: numbers of PROGRESS_REPORT.md lines 1095-1341 (Appendix A, Controls and nulls; Appendix B, the methods and controls behind Findings 1-7, up to 'Why rho, and why R2') that had no line in the audit … | Appendix B | `ablation_curated.csv`, `ablation_curated_gene.csv` +57 |
| `report_checks_R8` | 446 | Audit of PROGRESS_REPORT.md lines 1342-1554 (Appendix B2 why the imported families failed, B2a curation, B3 the family ablations on the curated cohort, B4 shape and 3D structure, B5 the dCas9 / CRISPRi boundary experiment … | Appendix B5 | `ablation_curated.csv`, `ablation_curated_gene.csv` +52 |
| `report_numbers` | 41 | Audit of the numbers added to the report by the line-by-line review of 23 Nov 2026 (modules R1 to R8 and CIT). | — | — |
| `verify_numbers` | 1001 | Audit: recompute every headline number of Appendix B4 and the Finding 3 addendum from the result files and compare with the value typed into the report. | tooling | — |

#### Old paths, for documents written before the reorganisations

<!-- docrefs:ignore:begin -->
<details><summary>148 modules have moved; documents that cite the left-hand paths (for example `Jacky/EXPERIMENT_CATALOG.md`) can be updated from this table</summary>

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
| `sgrna/ext/a_flank_fast.py` | `sgrna/build/a_flank_fast.py` |
| `sgrna/ext/base_fast.py` | `sgrna/build/base_fast.py` |
| `sgrna/ext/blend_with_hal.py` | `sgrna/compare/blend_with_hal.py` |
| `sgrna/ext/clean_vs_hal.py` | `sgrna/compare/clean_vs_hal.py` |
| `sgrna/ext/common.py` | `sgrna/common.py` |
| `sgrna/ext/cost_bench.py` | `sgrna/cost/cost_bench.py` |
| `sgrna/ext/cut_within_gene.py` | `sgrna/compare/cut_within_gene.py` |
| `sgrna/ext/data_efficiency.py` | `sgrna/compare/data_efficiency.py` |
| `sgrna/ext/dnashape.py` | `sgrna/features/dnashape.py` |
| `sgrna/ext/e2e_bench.py` | `sgrna/cost/e2e_bench.py` |
| `sgrna/ext/equiv_check.py` | `sgrna/controls/equiv_check.py` |
| `sgrna/ext/esp_replicate.py` | `sgrna/cohorts/esp_replicate.py` |
| `sgrna/ext/family_a_at_tuned.py` | `sgrna/tuning/family_a_at_tuned.py` |
| `sgrna/ext/geom.py` | `sgrna/features/geom.py` |
| `sgrna/ext/geom_importance.py` | `sgrna/interpret/geom_importance.py` |
| `sgrna/ext/human_additivity_twin.py` | `sgrna/compare/human_additivity_twin.py` |
| `sgrna/ext/human_flank.py` | `sgrna/compare/human_flank.py` |
| `sgrna/ext/human_shape.py` | `sgrna/compare/human_shape.py` |
| `sgrna/ext/human_twin_check.py` | `sgrna/compare/human_twin_check.py` |
| `sgrna/ext/mix_ecoli.py` | `sgrna/tuning/mix_ecoli.py` |
| `sgrna/ext/mlp_worker.py` | `sgrna/tuning/mlp_worker.py` |
| `sgrna/ext/panel_vs_hal.py` | `sgrna/compare/panel_vs_hal.py` |
| `sgrna/ext/redundancy.py` | `sgrna/interpret/redundancy.py` |
| `sgrna/ext/ridge_check.py` | `sgrna/cohorts/ridge_check.py` |
| `sgrna/ext/shape_ablate.py` | `sgrna/cohorts/shape_ablate.py` |
| `sgrna/ext/shape_build.py` | `sgrna/features/shape_build.py` |
| `sgrna/ext/topend.py` | `sgrna/compare/topend.py` |
| `sgrna/ext/topk_panel.py` | `sgrna/compare/topk_panel.py` |
| `sgrna/ext/tune_analyze.py` | `sgrna/tuning/tune_analyze.py` |
| `sgrna/ext/tune_ecoli.py` | `sgrna/tuning/tune_ecoli.py` |
| `sgrna/ext/tune_split2.py` | `sgrna/tuning/tune_split2.py` |
| `sgrna/ext/verify_numbers.py` | `sgrna/paper/verify_numbers.py` |
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
| `sgrna/post_freeze/a_flank_fast.py` | `sgrna/build/a_flank_fast.py` |
| `sgrna/post_freeze/base_fast.py` | `sgrna/build/base_fast.py` |
| `sgrna/post_freeze/blend_with_hal.py` | `sgrna/compare/blend_with_hal.py` |
| `sgrna/post_freeze/ceiling_table.py` | `sgrna/cohorts/ceiling_table.py` |
| `sgrna/post_freeze/clean_vs_hal.py` | `sgrna/compare/clean_vs_hal.py` |
| `sgrna/post_freeze/common.py` | `sgrna/common.py` |
| `sgrna/post_freeze/cost_bench.py` | `sgrna/cost/cost_bench.py` |
| `sgrna/post_freeze/cut_within_gene.py` | `sgrna/compare/cut_within_gene.py` |
| `sgrna/post_freeze/data_efficiency.py` | `sgrna/compare/data_efficiency.py` |
| `sgrna/post_freeze/dnashape.py` | `sgrna/features/dnashape.py` |
| `sgrna/post_freeze/e2e_bench.py` | `sgrna/cost/e2e_bench.py` |
| `sgrna/post_freeze/equiv_check.py` | `sgrna/controls/equiv_check.py` |
| `sgrna/post_freeze/esp_replicate.py` | `sgrna/cohorts/esp_replicate.py` |
| `sgrna/post_freeze/family_a_at_tuned.py` | `sgrna/tuning/family_a_at_tuned.py` |
| `sgrna/post_freeze/figures.py` | `sgrna/paper/figures.py` |
| `sgrna/post_freeze/flank_locus.py` | `sgrna/build/flank_locus.py` |
| `sgrna/post_freeze/fold_distance.py` | `sgrna/controls/fold_distance.py` |
| `sgrna/post_freeze/geom.py` | `sgrna/features/geom.py` |
| `sgrna/post_freeze/geom_importance.py` | `sgrna/interpret/geom_importance.py` |
| `sgrna/post_freeze/human_additivity_twin.py` | `sgrna/compare/human_additivity_twin.py` |
| `sgrna/post_freeze/human_flank.py` | `sgrna/compare/human_flank.py` |
| `sgrna/post_freeze/human_shape.py` | `sgrna/compare/human_shape.py` |
| `sgrna/post_freeze/human_twin_check.py` | `sgrna/compare/human_twin_check.py` |
| `sgrna/post_freeze/import_check.py` | `sgrna/paper/import_check.py` |
| `sgrna/post_freeze/lit_followups.py` | `sgrna/compare/lit_followups.py` |
| `sgrna/post_freeze/local_sequence_bound.py` | `sgrna/cohorts/local_sequence_bound.py` |
| `sgrna/post_freeze/mechanism.py` | `sgrna/interpret/mechanism.py` |
| `sgrna/post_freeze/mix_ecoli.py` | `sgrna/tuning/mix_ecoli.py` |
| `sgrna/post_freeze/mlp_worker.py` | `sgrna/tuning/mlp_worker.py` |
| `sgrna/post_freeze/novel_followups.py` | `sgrna/interpret/novel_followups.py` |
| `sgrna/post_freeze/panel_vs_hal.py` | `sgrna/compare/panel_vs_hal.py` |
| `sgrna/post_freeze/redundancy.py` | `sgrna/interpret/redundancy.py` |
| `sgrna/post_freeze/ridge_check.py` | `sgrna/cohorts/ridge_check.py` |
| `sgrna/post_freeze/shape_ablate.py` | `sgrna/cohorts/shape_ablate.py` |
| `sgrna/post_freeze/shape_build.py` | `sgrna/features/shape_build.py` |
| `sgrna/post_freeze/topend.py` | `sgrna/compare/topend.py` |
| `sgrna/post_freeze/topk_panel.py` | `sgrna/compare/topk_panel.py` |
| `sgrna/post_freeze/tune_analyze.py` | `sgrna/tuning/tune_analyze.py` |
| `sgrna/post_freeze/tune_ecoli.py` | `sgrna/tuning/tune_ecoli.py` |
| `sgrna/post_freeze/tune_split2.py` | `sgrna/tuning/tune_split2.py` |
| `sgrna/post_freeze/verify_numbers.py` | `sgrna/paper/verify_numbers.py` |
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
| `sgrna/strand_families.py` | `sgrna/compare/strand_families.py` |
| `sgrna/strand_recoverable.py` | `sgrna/compare/strand_recoverable.py` |
| `sgrna/strand_strata.py` | `sgrna/compare/strand_strata.py` |
| `sgrna/strand_within_gene.py` | `sgrna/compare/strand_within_gene.py` |
| `sgrna/transfer.py` | `sgrna/compare/transfer.py` |
| `sgrna/tune.py` | `sgrna/tuning/tune.py` |
| `sgrna/txn_interactions.py` | `sgrna/cohorts/txn_interactions.py` |
| `sgrna/why_inflation.py` | `sgrna/controls/why_inflation.py` |
| `sgrna/window_robustness.py` | `sgrna/controls/window_robustness.py` |
| `sgrna/xgb_knobs.py` | `sgrna/tuning/xgb_knobs.py` |

</details>
<!-- docrefs:ignore:end -->

<!-- generated:end -->
