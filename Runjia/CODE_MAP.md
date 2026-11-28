# Code map

What the code in this folder is, how it is organised, and which part of the report each module supports.
The tables between the two markers are **generated from the code** (`./run_ext.sh sgrna.paper.code_map`) and
the audit fails if a module is missing from them, so they cannot drift from the files. Everything outside the markers is
hand-written. For a short tour of only the code the report's claims depend on, read [`CODE_OVERVIEW.md`](CODE_OVERVIEW.md) first. What was deleted, and why, is in [`REMOVED.md`](REMOVED.md).

## The shape of it

Roughly 27,000 lines of Python in three layers, organised by what each module is *for* and not by when it was written
(the exact totals are the first line of the generated section below). Results are the one place the date shows: `results/`
holds numbers from before the 11 October freeze and `results/` those from after it.

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
`REMOVED.md`) found almost **no dead code**, because the report names each experiment's result and the audit checks the number. It also reported **no duplicated code**, and that did not hold: a second audit on 25 November (`EXTENSION_LOG.md` section 33) found 26 clusters of modules that run one pipeline with a different arm table, fold scheme, split seed or model, and thirteen small helpers copied between modules. The rule going forward is one module per question and one flag per variation; three sets of modules are merged so far (`flank_scale`, which now has the parts `window`, `partition`, `resolution`, `displacement` and `asymmetric`; `curated_ablation`, which absorbed the paper-folds ablation; and `build_features`, which builds both the published and the curated cohort), and the others are listed with their merge plan. The size is the number of claims times the number of
ways each was attacked. If the folder should be smaller, the lever is the report's scope, not tidying: see the by-section
table below.

## How to read a module

Every experiment module has the same four parts, so any of them can be read the same way.

1. **The docstring is the question** — usually phrased as one ("Does X survive Y?") — followed by what was expected before
   the run and what the result means.
2. **`run()`** does the experiment through the shared harness (`harness/run_ablation.py`, or `tuning/tune_ecoli.py`
   for the tuned model), with folds written to `results/` as they finish so a crash loses one fold, not a run.
3. **One CSV** lands in `results/` (before the freeze) or `results/` (after it), named after the module.
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

**120 modules, 37,514 lines** of Python in 12 folders (not counting the `__init__.py` markers).

#### By the report section each module supports

| report section | what it is | modules | lines | result files | where the code lives |
|---|---|---:|---:|---:|---|
| Finding 1 | The published representation is a re-encoding of the guide, and holds about twenty variables | 4 | 1,796 | 36 | build, controls, interpret, paper |
| Finding 2 | The DNA around the cut site predicts cutting — the one large effect, and it is local and compositional | 4 | 1,742 | 39 | controls, harness, interpret, paper |
| Finding 3 | The effect belongs to the DNA, and transfers | 6 | 2,144 | 85 | build, compare, controls, paper |
| Finding 4 | The model is out of *material*, and the tuning gain is evidence for it | 2 | 725 | 15 | cohorts, tuning |
| Finding 5 | On a redundant matrix, per-column importance is unstable across fits, but SHAP at block level and edit tests are faithful; SLICER is interpretable, more faithful than crisprHAL 2, and ties or splits with it on stability, truth and compressibility depending on the reading used | 7 | 3,008 | 54 | interpret, paper |
| Finding 7 | What the signal is made of: GC over a few hundred bases, a seed region that inverts between kingdoms, and the two coupled | 5 | 2,227 | 53 | compare, interpret, paper |
| Part 2 | Contributions | 1 | 280 | 3 | cohorts |
| Part 4 | The freeze, and what remains | 2 | 815 | 11 | compare, controls |
| Part 5 | Where this sits in the field | 24 | 6,121 | 116 | build, cohorts, compare, controls, cost, paper, tuning |
| Appendix A | Controls and nulls | 5 | 1,592 | 65 | harness, paper |
| Appendix B | The methods and controls behind the findings | 12 | 4,718 | 83 | cohorts, compare, controls, harness, interpret, paper, tuning |
| Appendix B3 | The family ablations on the curated cohort, and what they measure | 13 | 2,791 | 10 | cohorts, features, paper |
| Appendix B4 | The flank's shape and 3D structure add nothing that its composition does not already hold | 5 | 1,199 | 17 | build, compare, features |
| Appendix B5 | A boundary experiment, demoted: the dCas9 / CRISPRi screens | 4 | 2,031 | 66 | compare, paper |
| Appendix C0 | Reproducibility: versions, seeds and protocol | 1 | 199 | 0 | (top) |
| Appendix E | The paper's Results, Discussion and appendices, bullet by bullet | 1 | 747 | 11 | compare |
| tooling | the audit, the figures, the guards and this map | 7 | 2,533 | 0 | paper |
| library | shared code the experiments are built on | 8 | 1,191 | 1 | (top), build |
| — | no section of the report points at it | 9 | 1,655 | 11 | cost, design, paper |

Read the table above as the price list for the report: each claim carries its code and its result files with it. A module is placed in the section that cites it (or the result files it writes) most often. The `—` row, if there is one, is completed work that no section of the report points at. Each such module is listed in `REMOVED.md` under "Considered for removal and kept", with what it found; they were kept by decision, not by oversight.

#### Result files that exist only outside this repository

None: every result file a module writes by name is tracked, apart from the bulk prediction and per-fold records.

#### Package by package

### `sgrna/` (the package root: 3 files, 323 lines)

Three single files at the root of the package, not folders: `config.py` (project paths and shared constants), `io_utils.py` (reading the base matrix and writing feature blocks) and `common.py` (output folders and the resolver for results this repository does not track). Nearly every other module imports them.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `common.py` | 47 | Output folders for the experiment modules, and the resolver for the main project's results that this repository does not track. | library | `sample_complexity.csv` |
| `config.py` | 199 | Project paths and shared constants. | Appendix C0 | — |
| `io_utils.py` | 77 | Shared conventions for reading the base matrix and writing feature blocks. | library | — |

### `sgrna/build/`  —  12 modules, 2,507 lines

Decode the published matrix, locate every guide on the genome, rebuild features for guides outside it, and the array featurisers that score a guide in tens of microseconds.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `a_flank_fast` | 198 | Family A without a Python loop over nucleotides: the same 348 columns, computed on arrays. | Part 5 | — |
| `base_fast` | 113 | The 6,169 base columns without a Python loop over columns or guides: the same matrix as `featurise.matrix`, on arrays. | Part 5 | — |
| `build_features` | 211 | Build feature families for the published rows or for the 33,567-guide cohort, one CSV per family. | library | — |
| `build_guide_index` | 166 | Step 0 -- turn 13,880 anonymous rows into located guides. | library | — |
| `build_matrix` | 193 | Step 2 -- assemble a model-ready matrix from the base features plus blocks. | library | — |
| `curated_matrix` | 101 | Featurise the 33,567 guides of the E. coli screen (Guo et al., 2018) as crisprHAL 2 curated them, and cache the matrix every model loads. | library | — |
| `decode_v_columns` | 196 | Work out what the 5,888 `V####` columns are, so new guides can be featurised. | library | — |
| `featurise` | 126 | Generate the published feature representation for guides outside the matrix. | Finding 1 | — |
| `flank_locus` | 483 | Family A from a locus instead of from a guide: the window block becomes a table lookup. | Part 5 | `flank_locus_bench.csv` |
| `genome` | 232 | Reference genome helpers for E. coli K-12 MG1655 (NC_000913.2). | Finding 3 | — |
| `make_diagnostic_blocks` | 200 | Decompose the positional families, to find out what they are really measuring. | library | — |
| `qct` | 288 | Quantum chemical tensor (QCT) decoding and extension. | Appendix B4 | — |

### `sgrna/features/`  —  14 modules, 2,633 lines

The eleven feature families A-J (family A, the flanking-DNA composition, is the one that matters) and the shape / 3D descriptors that Appendix B4 tested against it.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `a_flank` | 240 | Family A -- sequence and quantum descriptors outside the 20 nt protospacer. | Appendix B3 | — |
| `b_energy` | 164 | Family B -- the R-loop free-energy decomposition (CRISPRoff / CRISPRspec). | Appendix B3 | — |
| `c_folding` | 155 | Family C -- sgRNA folding, done on the transcript that is actually made. | Appendix B3 | — |
| `d_mechanics` | 201 | Family D1 -- how hard the target duplex is to open, position by position. | Appendix B3 | — |
| `d_supercoiling` | 233 | Family D2 -- DNA supercoiling at the target, from GapR-seq. | Appendix B3 | — |
| `dnashape` | 186 | The DNAshapeR pentamer query table, separated into the kinds of column it lumps together. | Appendix B4 | — |
| `e_nucleoid` | 220 | Family E -- nucleoid organisation, the bacterial answer to chromatin. | Appendix B3 | — |
| `f_methylation` | 133 | Family F -- Dam and Dcm methylation at the target site. | Appendix B3 | — |
| `g_transcription` | 275 | Family G -- transcription and replication context. | Appendix B3 | — |
| `geom` | 280 | DNA shape and 3D path descriptors of the locus around a guide, by data source. | Appendix B4 | — |
| `h_offtarget` | 133 | Family H -- off-target burden and target copy number. | Appendix B3 | — |
| `i_shape` | 208 | Family I -- the physical shape of the DNA around the target, not its letters. | Appendix B3 | — |
| `j_periodicity` | 129 | Family J -- strand asymmetry and helical periodicity in the flanking DNA. | Appendix B3 | — |
| `shape_build` | 76 | Build the shape/3D blocks for the curated cohort, one file per (source, control, block). | Appendix B4 | — |

### `sgrna/harness/`  —  8 modules, 2,307 lines

The leakage-safe cross-validation harness every experiment runs through, plus the notebook front end.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `cohort_groups` | 74 | Grouped cross-validation means something different on the curated cohort: gene identity explains 29% of its label variance against 16% on the published cohort. | Appendix B | `cohort_groups.csv` |
| `curated` | 69 | The curated E. coli cohort as the experiments load it: the 6,169 base columns and the label, one feature family aligned to the guides, and the group label of each guide. | Appendix A | — |
| `evaluate` | 708 | Quantifying the two things a model is judged on: prediction and explanation. | Appendix B | `interpretability.csv`, `model_comparison.csv` |
| `folds` | 72 | Fold assignment by whole groups, and the fit-and-score step that the held-out-gene experiments share. | Appendix B | — |
| `metrics` | 11 | Small numeric helpers that several experiments need and that used to be copied into each of them. | Appendix A | — |
| `notebook` | 532 | Notebook front end -- everything the pipeline does, callable from a cell. | Appendix A | `ablation_abund_split.csv`, `ablation_curated.csv` +13 |
| `run_ablation` | 560 | Step 3 -- the ablation harness. | Appendix A | `ablation_abund_split.csv`, `ablation_curated.csv` +27 |
| `seqnet` | 281 | A convolutional net on raw DNA does not beat gradient boosting: it reaches rho 0.525 at best, against 0.607 for boosting on the 6,580 engineered columns, on the same rows and folds. | Finding 2 | `seqnet.csv`, `seqnet_matched.csv` |

### `sgrna/tuning/`  —  5 modules, 1,308 lines

Hyperparameter search and model mixing: the joint search on development genes (`tune_ecoli`, on two gene splits), the blend of model families, and the knobs no search had touched.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `mix_ecoli` | 288 | Mixing model families barely helps: no blend beats the tuned XGBoost ensemble by more than +0.0017 rho on held-out genes, and adding weaker families at equal weight lowers it. | Part 5 | `mix_configs_cat.json`, `mix_configs_et.json` +9 |
| `mlp_worker` | 55 | Neural-network member of the E. coli mix, run in its own process. | Part 5 | — |
| `tune` | 241 | The margin survives tuning: a per-fold search adds +0.0131 rho to SLICER's original model, so its lead over crisprHAL 2's best known settings is +0.0161. | Part 5 | `tuning.csv`, `tuning_trace.csv` |
| `tune_ecoli` | 481 | Joint hyperparameter search for the E. coli model, with the selection kept apart from the confirmation. | Finding 4 | `tune_budget_check.csv`, `tune_configs.json` +9 |
| `xgb_knobs` | 243 | Of eight XGBoost knobs never tuned before, only tree size matters: depth 6 adds +0.0063 rho, while the rest add at most +0.0024 or lower it. | Appendix B | `xgb_knobs.csv`, `xgb_knobs_verify.csv` |

### `sgrna/cohorts/`  —  7 modules, 2,388 lines

Ablations on the curated cohort and the arguments about what its numbers mean (ceilings, inflation, label noise), including the shape ablations of Appendix B4.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `ceiling` | 244 | The ceiling on any predictor of a replicate-averaged label is the square root of its Spearman-Brown reliability, not the replicate agreement the field quotes: 0.946 [0.944, 0.947] for the label SLICER trains on … | Part 5 | `ceiling_parametric.csv`, `curated_ceiling.json` |
| `curated_ablation` | 669 | Feature-family ablations on crisprHAL's curated cohort (one family at a time on random rows or whole genes, `--paper-folds`), and three questions asked of the same cohort: the family gains overlap heavily … | Appendix B3 | `ablation_curated.csv`, `ablation_curated_gene.csv` +8 |
| `curation` | 280 | Curation does not create the flank effect, and the larger cohort's bigger gains are a steeper flank effect in its genome-wide guides (slope 15.85 against 5.35 on shared guides), not a wider spread of the data. | Part 2 | `curation_check.csv`, `curation_variance.csv` +1 |
| `data_ablation` | 152 | Hold SLICER fixed and change only the training data. | Part 5 | `data_ablation.csv` |
| `replicate_weight` | 295 | The label's reliability is per guide, and the model treats it as uniform. | Appendix B | `replicate_weight.csv`, `replicate_weight_raw.csv` |
| `shape_ablate` | 504 | Shape and 3D descriptors add nothing to family A under boosted trees (+0.0006 rho at best), although a ridge model on all columns finds +0.0097 in them, so the null belongs to the pipeline and not to the descriptors. | Part 5 | `esp_replicate.csv`, `folds_base_plus_K562_ATAC.csv` +18 |
| `tuned_ablation` | 244 | At the tuned settings family A is worth +0.1697 rho (up from +0.1563 at the champion's), and the inner +-50 nt as explicit one-hot columns add +0.0002 on top of it: the bound on any local shape predictor. | Finding 4 | `family_a_at_tuned.csv`, `family_a_at_tuned_summary.csv` +3 |

### `sgrna/controls/`  —  8 modules, 2,712 lines

The adversarial phase: nulls, shuffles, floors, selector and leakage audits, and the check that the fast featurisers equal the originals.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `diagnose` | 549 | Binding energy and duplex stability are mostly functions of the 20-mer (70% and 50% of their columns have R² above 0.9), so they add little, and no other feature family is redundant in this way. | Finding 1 | `ceiling.json`, `headroom.csv` +3 |
| `equiv_check` | 59 | Recorded equivalence of the array-based featurisers with the originals, and of the predictions built on them. | Part 5 | `featuriser_equivalence.csv`, `tune_configs.json` |
| `family_a_controls` | 523 | Family A's gain is not the row count (+0.1672 on subsamples of the curated cohort the size of the published matrix, +0.1638 on the whole cohort), not the choice of windows … | Finding 2 | `ablation_curated.csv`, `chi_sites.csv` +3 |
| `fold_distance` | 96 | Sequence distance between training and validation guides, the leakage statistic our comparators report. | Part 5 | `fold_distance.csv` |
| `mappability_stratum` | 299 | Family A's flank gain does not come from read depth or mappability: it is +0.157 among guides in uniquely mappable sequence and +0.160 with control read depth given to the model. | Finding 3 | `mappability_depth.csv`, `mappability_stratum.csv` +2 |
| `rederive` | 543 | Re-derive the result files whose producer was lost, and check they agree. | Appendix B | `ablation_curated.csv`, `ablation_curated_gene.csv` +5 |
| `sealed` | 371 | On genes sealed away from column selection and settings, family A's gain is 0.0087 lower than in cross-validation of the pool; holding whole genes out of the pool too leaves +0.0081, so unseen genes explain almost none of it … | Part 4 | `sealed_crossscreen.csv`, `sealed_holdout.csv` +2 |
| `selector_audit` | 272 | The pipeline selects columns with the one metric the report says never to use. | Appendix B | `selector_audit.csv`, `selector_audit_ranks.csv` +1 |

### `sgrna/interpret/`  —  11 modules, 5,989 lines

What the model uses and what the signal is made of: importance, attribution, redundancy, the mechanism work of Finding 7, and what the tuned settings say.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `compact_model` | 214 | A compact model of guide base indicators and flank composition columns keeps 88% of the full model's rho and can be read column by column, with SHAP agreeing across tree families at 0.87. | Finding 5 | `compact_model_accuracy.csv`, `compact_model_accuracy_folds.csv` +5 |
| `flank_scale` | 328 | The flank signal is a bulk GC effect over a few hundred bases: two GC columns reach +0.106 at 200 nt on each side, while purine and amino fractions add nothing. | Finding 7 | `flank_scale.csv`, `flank_scale_summary.csv` +2 |
| `flank_signal` | 613 | The flank effect is a property of the surrounding DNA, not of the guide's own score, its target GC or transcription: the flank and the guide's own score add up rather than gate … | Finding 2 | `asymmetry_expression.csv`, `asymmetry_strand_audit.csv` +9 |
| `hal_readout` | 273 | crisprHAL 2's side of the interpretability comparison (`vs_crisprhal_interpret.py` has the design and the predictions). | Finding 5 | `guides_1500.csv`, `hal_eg_draws_control.csv` +2 |
| `importance` | 1028 | Importance is a property of the data only for blocks of columns, not for single columns: five tree families agree on columns at only 0.496 even with SHAP, but on blocks of columns at 0.800 … | Finding 5 | `attribution.csv`, `attribution_shap.csv` +11 |
| `interp_extra` | 313 | Three interpretability probes on one fixed model: sensitivity-n of SHAP, SHAP interaction values, and counterfactual flank edits. | Finding 5 | `interp_counterfactual.csv`, `interp_interaction_blocks.csv` +3 |
| `mechanism` | 858 | What the flank effect physically is can be asked of the cohort directly, without a model: flank GC and the seed base act in synergy and not as compensation … | Finding 7 | `hypotheses_audit_gate.csv`, `hypotheses_audit_scale.csv` +24 |
| `redescription` | 742 | Re-description helps only when it means something: random linear or non-linear functions of the guide add nothing, while the guide's real self-folding energy adds +0.0117. | Appendix B | `recoverability_percolumn.csv`, `recoverability_trend.csv` +7 |
| `representation` | 828 | Seventy-four mononucleotide columns, a lossless spelling of the 20-mer, recover 96.4% of the full published matrix's rho, so the model does not need the other columns; and the published 63 hand-named summary columns … | Finding 1 | `base_enrichment.csv`, `base_enrichment_published.csv` +7 |
| `tuned_settings` | 201 | The tuned settings show a mostly additive signal: single-split stumps reach held-out rho 0.6905 against 0.7210 for shallow trees, and deeper trees do no better. | Finding 5 | `interpret_data_scaling.csv`, `interpret_depth.csv` +4 |
| `vs_crisprhal_interpret` | 591 | SLICER is more faithful than crisprHAL 2 (sensitivity-n +0.081 to +0.111) but ties or splits on stability, truth and compressibility, so it is not shown to be more interpretable. | Finding 5 | `guides_1500.csv`, `hal_flank_edits.csv` +14 |

### `sgrna/compare/`  —  17 modules, 7,101 lines

Every comparison with something else (Part 5 of the report): crisprHAL 2, the inherited baseline, other organisms and the human matrix, and other assays (the dCas9 boundary experiment of Appendix B5).

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `boostmec` | 747 | BoostMEC, the closest published design to SLICER, scores a guide in 3.07 ms at a batch of 100 on one thread (42.3 times SLICER's default model) and, retrained on the E. coli cohort as its authors trained it … | Appendix E | `boostmec_oracle.csv`, `boostmec_released.csv` +9 |
| `crisprhal_runs` | 580 | crisprHAL 2 retrained by us with its own code reproduces its published hold-out (rho 0.694 against 0.695), and is then run on our rows and folds, searched over its own constructor's arguments … | Appendix B | `crisprhal_rerun.csv`, `crisprhal_tuned_folds.csv` +5 |
| `crispri_screens` | 327 | Three CRISPRi screens instead of one, and what that buys beyond accuracy. | Appendix B5 | `crispri_screen_agreement.csv`, `crispri_screen_family_a.csv` +1 |
| `cui_screen` | 792 | Family A adds nothing on Cui's dCas9 CRISPRi screen (−0.0093, whole genes held out) against +0.1563 on a cutting screen, although the screens differ in assay, library and laboratory. | Appendix B5 | `cui_essential_marginals.csv`, `cui_essential_position.csv` +5 |
| `eukaryote_rules` | 372 | Eukaryotes share guide rules (corrected correlation +0.59 between human tables) that are unrelated to E. coli's: G at position 20 helps in every eukaryotic table and hurts in E. coli. | Finding 7 | `eukaryote_contrast_g_minus_t.csv`, `eukaryote_flank_step.csv` +6 |
| `family_a_by_organism` | 70 | Family A raises held-out rho in every organism it was built for: +0.161 in E. coli, +0.104 for eSpCas9, +0.059 in C. rodentium and +0.023 in human, positive in every fold. | Finding 3 | `ablation_paper_folds_summary.csv`, `family_a_by_organism.csv` +3 |
| `headtohead` | 305 | The like-for-like comparison against crisprHAL 2. | Part 5 | `headtohead.csv` |
| `human_baseline` | 373 | The human boosting baseline (rho 0.5076 against the harness's 0.4036) is a memorisation artefact: with repeat measurements of a guide kept in one fold it falls to 0.2515 … | Appendix B | `human_additivity.csv`, `human_additivity_twin.csv` +7 |
| `human_shape` | 369 | On the human matrix the shape and 3D-path columns add +0.0183 to family A, where E. coli gives −0.0007, a difference that is not explained. | Appendix B4 | `folds_base_plus_K562_ATAC.csv`, `folds_base_plus_K562_DNase.csv` +15 |
| `lit_followups` | 210 | Two ideas taken from the published predictors and tested on our cohort (Part 5 of the report). | Part 5 | `lit_binary.csv`, `lit_merf.csv` +2 |
| `noshay_replicate` | 232 | Reproduce the inherited paper's own model, as its methods section describes it. | Part 5 | `noshay_replication.csv` |
| `panel` | 444 | Beyond rank correlation: the other things a guide-efficiency model is judged on, for SLICER and crisprHAL 2 (SLICER's lead is not a Spearman artefact, and the top of the ranking is where it is smallest). | Part 4 | `blend_with_hal.csv`, `metric_panel.csv` +5 |
| `sample_complexity` | 284 | crisprHAL 2 needs 1.7 to 2.5 times as many labelled guides as SLICER to reach the same accuracy, at every accuracy level that both models reach. | Part 5 | `data_efficiency.csv`, `sample_complexity.csv` +1 |
| `strand` | 480 | With the strand call given to every family, flank family A adds only +0.007 on the dCas9 screen, but inside the coding strand its gain is not only gene ranking (+0.102 within genes on Cui's screen). | Appendix B5 | `cut_within_gene.csv`, `strand_families.csv` +8 |
| `topend` | 93 | No objective aimed at the top of the ranking beat plain squared error on top-decile precision (0.291 for squared error, at most 0.289 for the others), and the rank objectives lose Spearman. | Part 5 | `topend_confirm.csv`, `tune_configs.json` |
| `transfer` | 588 | A model trained on SpCas9 keeps 88% of its own accuracy on eSpCas9, a different nuclease in the same genome, and flank features supply +0.098 of its ρ. | Finding 3 | `hybrid.csv`, `transfer.csv` +6 |
| `vs_crisprhal` | 835 | SLICER against crisprHAL 2 under every protocol of Part 5: the tuned model leads by +0.0226 on crisprHAL 2's SpCas9 split and +0.0286 on its eSpCas9 split … | Part 5 | `clean_vs_hal.csv`, `crisprhal_gene_grouped_fcolab_b1024.csv` +13 |

### `sgrna/cost/`  —  5 modules, 1,553 lines

What each model costs to train and to run, measured rather than estimated.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `bench` | 319 | Peak memory and training cost, measured rather than estimated. | — | `memory_comparison.csv`, `memory_comparison_runs.csv` +1 |
| `cost` | 372 | Measures CPU-seconds and floating-point operations to fit SLICER, a cut-down SLICER and crisprHAL 2; superseded by cost_repeats, as its crisprHAL 2 CPU-seconds were a hand-entered estimate. | Part 5 | `cost_cpu_seconds.csv`, `cost_flops.csv` +1 |
| `cost_repeats` | 295 | Cost with its spread: every timing repeated, every repeat kept, so a ratio comes with a distribution and not a bare range. | — | `cost_repeats_featurisers.csv`, `cost_repeats_fit.csv` +2 |
| `latency` | 373 | Scoring latency of SLICER against crisprHAL 2, on one thread, in the three forms it has been measured: the model alone (0.002 ms per guide for SLICER's default model against 1.32 ms for crisprHAL 2 at a batch of 100) … | Part 5 | `cost_crisprhal.csv`, `cost_featurise.csv` +7 |
| `model_size` | 194 | SLICER's default model stores 6,338 leaf values against 790,533 parameters for crisprHAL 2, but the tuned SLICER model is larger on disk than crisprHAL 2's file. | Part 5 | `complexity.json`, `model_size.csv` |

### `sgrna/design/`  —  6 modules, 1,011 lines

The tool the project is for: give it a genome and a cut site and it returns the SpCas9 guide SLICER predicts will cut best there, with the shipped models, the check that it reads the same DNA and builds the same columns as the training matrices, and a benchmark of picking by score. It also holds the generator of the family A table for any genome (FASTA in, the 348 flank columns of every NGG guide out) and its check against the released table.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `check_design` | 60 | Check the design tool against the stored feature matrices: it reads the same DNA, builds the same 6,517 columns and scores the same. | — | `design_check.csv` |
| `check_family_a_table` | 236 | Check `family_a_table`: it rebuilds the released family A table from the genome, and agrees with the older per-guide code on a genome it was not built for. | — | `family_a_table_check.csv` |
| `design_benchmark` | 109 | Picking the guide SLICER ranks first beats a random pick: among curated guides within 250 bp its measured score sits at the 0.77 percentile, against 0.5 for chance. | — | `design_benchmark.csv`, `tune_configs.json` |
| `design_guide` | 274 | Pick the SpCas9 guide SLICER predicts will cut best at a site in a genome. | — | — |
| `family_a_table` | 260 | Family A for any genome: a FASTA file in, the 348 flank columns of every NGG guide out. | — | — |
| `train_model` | 72 | Fit the shipped SLICER models (tuned and default) on all 33,567 curated guides and save them with everything needed to score a new guide. | — | `tune_configs.json` |

### `sgrna/paper/`  —  24 modules, 7,682 lines

The tooling the paper rests on: the audit, the figures, the import and reference guards, and the code map.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `code_map` | 421 | Generate the per-module tables of CODE_MAP.md from the code itself, so the map cannot drift from the files. | tooling | — |
| `doc_refs` | 167 | Every module, path and Colab script that a document or a docstring names still exists. | tooling | — |
| `figures` | 427 | Publication figures, to ACS graphics specifications, generated from the result files. | tooling | — |
| `figures_paper` | 459 | Figures for the point-form paper, every one drawn from stored result files (no schematics). | Finding 7 | `boostmec_released.csv`, `cost_gpu.csv` +16 |
| `hooks` | 86 | The audit's whole coupling to experiment code, in one table: every module, function or constant of `sgrna/` that a check imports to recompute a number. | tooling | — |
| `import_check` | 227 | Every module in the package imports, and importing it writes nothing. | tooling | — |
| `paper_numbers` | 119 | Audit of every number in the paper (point form): each is recomputed from a stored result file, as `verify_numbers` does for the report. | Part 5 | `blend_with_hal.csv`, `clean_vs_hal.csv` +7 |
| `paper_numbers_A` | 420 | Audit of the numbers in the draft paper's claims [0]-[19], [35]-[45] (group A): data sets and methods, the flank effect, its mechanism and the appendix constraints. | Appendix A | `ablation_curated.csv`, `ablation_curated_gene.csv` +41 |
| `paper_numbers_B` | 218 | Audit of the numbers in the draft paper's claims [20]-[28], [32]-[34] and [49] (group B): every number is recomputed from a stored result file and compared with the value the paper prints. | Finding 3 | `ablation_curated.csv`, `ceiling_parametric.csv` +25 |
| `paper_numbers_C` | 312 | Audit of paper claims [29]-[31], [46]-[48], [50]-[55] (group C): every number is recomputed from a stored result file. | Part 5 | `ablation_curated.csv`, `ablation_label_curated.csv` +38 |
| `paper_numbers_D` | 737 | Audit of the numbers added to the paper (point form) in the second pass of 23 Nov 2026: one protocol, the reach of the flank signal, controls, the transfer matrix, SLICER against crisprHAL 2 on more metrics … | Finding 3 | `ablation_curated.csv`, `ablation_paper_folds_summary.csv` +58 |
| `report_checks_CIT` | 76 | Checks on the report's citation-related numbers (tag CIT): each `got` is computed from a stored file. | Part 5 | `noshay_replication.csv` |
| `report_checks_R1` | 293 | Report checks R1: numbers of PROGRESS_REPORT.md in the Summary, 'How to read', 'What earns a place' and Finding 1 that had no line in the audit, recomputed from a stored file or from the repository's own code. | Finding 1 | `ablation_curated.csv`, `ablation_curated_gene.csv` +25 |
| `report_checks_R2` | 325 | Audit of Findings 2 to 4 of PROGRESS_REPORT.md (Finding 2 on the flank effect, Finding 3 on transfer and confound controls, Finding 4 on the model being out of material) … | Finding 2 | `ablation_curated.csv`, `ablation_main.csv` +26 |
| `report_checks_R3` | 388 | Checks behind review R3 of PROGRESS_REPORT.md (Finding 5, interpretability; Finding 6, the label's ceiling). | Finding 5 | `additivity_reduced.csv`, `attribution_summary.csv` +24 |
| `report_checks_R4` | 210 | Checks for PROGRESS_REPORT.md Finding 7 (what the signal is made of) that the main audit does not read, or reads under another label. | Finding 7 | `ablation_curated.csv`, `ablation_curated_gene.csv` +12 |
| `report_checks_R5` | 306 | Checks R5: headline numbers of Parts 2 to 4 of PROGRESS_REPORT.md (Part 2 contributions, Part 3 limits, Part 4 freeze) that had no audit line. | Appendix B | `ablation_curated.csv`, `ablation_curated_gene.csv` +19 |
| `report_checks_R6` | 283 | Checks for headline numbers of PROGRESS_REPORT.md Part 5 (the head-to-head with crisprHAL 2, cost, other comparisons, what other predictors read) that had no audit line, computed against the corrected wording of the report. | Part 5 | `ablation_curated.csv`, `ablation_curated_gene.csv` +37 |
| `report_checks_R7` | 510 | Report checks R7: numbers of Appendix A and the first part of Appendix B of PROGRESS_REPORT.md (Controls and nulls; the methods and controls behind Findings 1-7, up to 'Why rho, and why R2') that had no line in the audit … | Appendix B | `ablation_curated.csv`, `ablation_curated_gene.csv` +57 |
| `report_checks_R8` | 432 | Audit of the rest of the appendices of PROGRESS_REPORT.md (Appendix B2 why the imported families failed, B2a curation, B3 the family ablations on the curated cohort, B4 shape and 3D structure … | Appendix B5 | `ablation_curated.csv`, `ablation_curated_gene.csv` +52 |
| `report_numbers` | 30 | Audit of the numbers added to the report by the line-by-line review of 23 Nov 2026 (modules R1 to R8 and CIT). | — | — |
| `reproduce` | 196 | Do the modules still write the files the audit reads? | tooling | — |
| `standalone` | 31 | Run one audit module on its own, with the same pass rule as the whole audit. | Appendix B3 | — |
| `verify_numbers` | 1009 | Audit: recompute every headline number of Appendix B4 and the Finding 3 addendum from the result files and compare with the value typed into the report. | tooling | — |

#### Old paths, for documents written before the reorganisations

<!-- docrefs:ignore:begin -->
<details><summary>72 modules have moved; documents that cite the left-hand paths (for example `Jacky/EXPERIMENT_CATALOG.md`) can be updated from this table</summary>

| was | is now |
|---|---|
| `sgrna/bench.py` | `sgrna/cost/bench.py` |
| `sgrna/build_features.py` | `sgrna/build/build_features.py` |
| `sgrna/build_guide_index.py` | `sgrna/build/build_guide_index.py` |
| `sgrna/build_matrix.py` | `sgrna/build/build_matrix.py` |
| `sgrna/cohort_groups.py` | `sgrna/harness/cohort_groups.py` |
| `sgrna/cost.py` | `sgrna/cost/cost.py` |
| `sgrna/crispri_screens.py` | `sgrna/compare/crispri_screens.py` |
| `sgrna/cui_screen.py` | `sgrna/compare/cui_screen.py` |
| `sgrna/curated_ablation.py` | `sgrna/cohorts/curated_ablation.py` |
| `sgrna/data_ablation.py` | `sgrna/cohorts/data_ablation.py` |
| `sgrna/decode_v_columns.py` | `sgrna/build/decode_v_columns.py` |
| `sgrna/diagnose.py` | `sgrna/controls/diagnose.py` |
| `sgrna/evaluate.py` | `sgrna/harness/evaluate.py` |
| `sgrna/ext/a_flank_fast.py` | `sgrna/build/a_flank_fast.py` |
| `sgrna/ext/base_fast.py` | `sgrna/build/base_fast.py` |
| `sgrna/ext/common.py` | `sgrna/common.py` |
| `sgrna/ext/dnashape.py` | `sgrna/features/dnashape.py` |
| `sgrna/ext/equiv_check.py` | `sgrna/controls/equiv_check.py` |
| `sgrna/ext/geom.py` | `sgrna/features/geom.py` |
| `sgrna/ext/human_shape.py` | `sgrna/compare/human_shape.py` |
| `sgrna/ext/mix_ecoli.py` | `sgrna/tuning/mix_ecoli.py` |
| `sgrna/ext/mlp_worker.py` | `sgrna/tuning/mlp_worker.py` |
| `sgrna/ext/shape_ablate.py` | `sgrna/cohorts/shape_ablate.py` |
| `sgrna/ext/shape_build.py` | `sgrna/features/shape_build.py` |
| `sgrna/ext/topend.py` | `sgrna/compare/topend.py` |
| `sgrna/ext/tune_ecoli.py` | `sgrna/tuning/tune_ecoli.py` |
| `sgrna/ext/verify_numbers.py` | `sgrna/paper/verify_numbers.py` |
| `sgrna/ext/vs_crisprhal.py` | `sgrna/compare/vs_crisprhal.py` |
| `sgrna/featurise.py` | `sgrna/build/featurise.py` |
| `sgrna/genome.py` | `sgrna/build/genome.py` |
| `sgrna/headtohead.py` | `sgrna/compare/headtohead.py` |
| `sgrna/human_baseline.py` | `sgrna/compare/human_baseline.py` |
| `sgrna/importance.py` | `sgrna/interpret/importance.py` |
| `sgrna/interp_extra.py` | `sgrna/interpret/interp_extra.py` |
| `sgrna/j_periodicity.py` | `sgrna/features/j_periodicity.py` |
| `sgrna/make_diagnostic_blocks.py` | `sgrna/build/make_diagnostic_blocks.py` |
| `sgrna/noshay_replicate.py` | `sgrna/compare/noshay_replicate.py` |
| `sgrna/notebook.py` | `sgrna/harness/notebook.py` |
| `sgrna/panel.py` | `sgrna/compare/panel.py` |
| `sgrna/post_freeze/a_flank_fast.py` | `sgrna/build/a_flank_fast.py` |
| `sgrna/post_freeze/base_fast.py` | `sgrna/build/base_fast.py` |
| `sgrna/post_freeze/common.py` | `sgrna/common.py` |
| `sgrna/post_freeze/dnashape.py` | `sgrna/features/dnashape.py` |
| `sgrna/post_freeze/equiv_check.py` | `sgrna/controls/equiv_check.py` |
| `sgrna/post_freeze/figures.py` | `sgrna/paper/figures.py` |
| `sgrna/post_freeze/flank_locus.py` | `sgrna/build/flank_locus.py` |
| `sgrna/post_freeze/fold_distance.py` | `sgrna/controls/fold_distance.py` |
| `sgrna/post_freeze/geom.py` | `sgrna/features/geom.py` |
| `sgrna/post_freeze/human_shape.py` | `sgrna/compare/human_shape.py` |
| `sgrna/post_freeze/import_check.py` | `sgrna/paper/import_check.py` |
| `sgrna/post_freeze/lit_followups.py` | `sgrna/compare/lit_followups.py` |
| `sgrna/post_freeze/mechanism.py` | `sgrna/interpret/mechanism.py` |
| `sgrna/post_freeze/mix_ecoli.py` | `sgrna/tuning/mix_ecoli.py` |
| `sgrna/post_freeze/mlp_worker.py` | `sgrna/tuning/mlp_worker.py` |
| `sgrna/post_freeze/shape_ablate.py` | `sgrna/cohorts/shape_ablate.py` |
| `sgrna/post_freeze/shape_build.py` | `sgrna/features/shape_build.py` |
| `sgrna/post_freeze/topend.py` | `sgrna/compare/topend.py` |
| `sgrna/post_freeze/tune_ecoli.py` | `sgrna/tuning/tune_ecoli.py` |
| `sgrna/post_freeze/verify_numbers.py` | `sgrna/paper/verify_numbers.py` |
| `sgrna/post_freeze/vs_crisprhal.py` | `sgrna/compare/vs_crisprhal.py` |
| `sgrna/qct.py` | `sgrna/build/qct.py` |
| `sgrna/rederive.py` | `sgrna/controls/rederive.py` |
| `sgrna/redescription.py` | `sgrna/interpret/redescription.py` |
| `sgrna/replicate_weight.py` | `sgrna/cohorts/replicate_weight.py` |
| `sgrna/representation.py` | `sgrna/interpret/representation.py` |
| `sgrna/run_ablation.py` | `sgrna/harness/run_ablation.py` |
| `sgrna/sample_complexity.py` | `sgrna/compare/sample_complexity.py` |
| `sgrna/selector_audit.py` | `sgrna/controls/selector_audit.py` |
| `sgrna/seqnet.py` | `sgrna/harness/seqnet.py` |
| `sgrna/transfer.py` | `sgrna/compare/transfer.py` |
| `sgrna/tune.py` | `sgrna/tuning/tune.py` |
| `sgrna/xgb_knobs.py` | `sgrna/tuning/xgb_knobs.py` |

</details>
<!-- docrefs:ignore:end -->

<!-- generated:end -->
