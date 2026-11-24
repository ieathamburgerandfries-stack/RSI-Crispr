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
`REMOVED.md`) found almost **no dead code**, because the report names each experiment's result and the audit checks the number. It also reported **no duplicated code**, and that did not hold: a second audit on 25 November (`EXTENSION_LOG.md` section 33) found 26 clusters of modules that run one pipeline with a different arm table, fold scheme, split seed or model, and thirteen small helpers copied between modules. The rule going forward is one module per question and one flag per variation; three sets of modules are merged so far (`flank_scale`, which now has the parts `window`, `partition`, `resolution`, `displacement` and `asymmetric`; `curated_ablation`, which absorbed the paper-folds ablation; and `build_features`, which builds both the published and the curated cohort), and the others are listed with their merge plan. The size is the number of claims times the number of
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

**171 modules, 37,921 lines** of Python in 11 folders (not counting the `__init__.py` markers).

#### By the report section each module supports

| report section | what it is | modules | lines | result files | where the code lives |
|---|---|---:|---:|---:|---|
| Finding 1 | The published representation is a re-encoding of the guide, and holds about twenty variables | 5 | 770 | 27 | build, cohorts, compare, controls, paper |
| Finding 2 | The DNA around the cut site predicts cutting — the one large effect, and it is local and compositional | 5 | 1,161 | 30 | compare, controls, harness, interpret, paper |
| Finding 3 | The effect belongs to the DNA, and transfers | 6 | 2,112 | 81 | build, compare, controls, paper |
| Finding 4 | The model is out of *material*, and the tuning gain is evidence for it | 3 | 922 | 13 | interpret, tuning |
| Finding 5 | On a redundant matrix, per-column importance is unstable across fits, but SHAP at block level and edit tests are faithful; SLICER is interpretable, more faithful than crisprHAL 2, and ties or splits with it on stability, truth and compressibility depending on the reading used | 10 | 3,066 | 54 | interpret, paper |
| Finding 6 | The label's ceiling is about 0.946 (0.942 to 0.946 by estimator), measured from crisprHAL's own replicates | 1 | 122 | 1 | cohorts |
| Finding 7 | What the signal is made of: GC over a few hundred bases, a seed region that inverts between kingdoms, and the two coupled | 7 | 2,237 | 52 | compare, interpret, paper |
| Part 2 | Contributions | 6 | 722 | 7 | cohorts, compare, controls, interpret |
| Part 3 | Where this stops | 1 | 179 | 2 | compare |
| Part 4 | The freeze, and what remains | 5 | 792 | 5 | compare, controls |
| Part 5 | Where this sits in the field | 36 | 6,082 | 107 | build, cohorts, compare, controls, cost, interpret, paper, tuning |
| Appendix A | Controls and nulls | 4 | 1,609 | 66 | controls, harness, paper |
| Appendix B | The methods and controls behind the findings | 16 | 5,104 | 82 | cohorts, compare, controls, harness, interpret, paper, tuning |
| Appendix B2 | Why the imported families failed, and why that is not a screen | 2 | 472 | 4 | controls, interpret |
| Appendix B2a | Curation: what it is, and why family A doubled between cohorts | 2 | 263 | 2 | cohorts, controls |
| Appendix B3 | The family ablations on the curated cohort, and what they measure | 13 | 2,621 | 8 | cohorts, features, interpret |
| Appendix B4 | The flank's shape and 3D structure add nothing that its composition does not already hold | 10 | 1,534 | 25 | build, cohorts, compare, features, interpret |
| Appendix B5 | A boundary experiment, demoted: the dCas9 / CRISPRi screens | 8 | 2,014 | 65 | compare, paper |
| Appendix C0 | Reproducibility: versions, seeds and protocol | 1 | 199 | 0 | (top) |
| Appendix E | The paper's Results, Discussion and appendices, bullet by bullet | 5 | 910 | 4 | build, compare |
| tooling | the audit, the figures, the guards and this map | 5 | 2,125 | 0 | paper |
| library | shared code the experiments are built on | 6 | 913 | 1 | (top), build |
| — | no section of the report points at it | 14 | 1,992 | 31 | cohorts, compare, cost, interpret, paper |

Read the table above as the price list for the report: each claim carries its code and its result files with it. A module is placed in the section that cites it (or the result files it writes) most often. The `—` row, if there is one, is completed work that no section of the report points at. Each such module is listed in `REMOVED.md` under "Considered for removal and kept", with what it found; they were kept by decision, not by oversight.

#### Result files that exist only outside this repository

None: every result file a module writes by name is tracked, apart from the bulk prediction and per-fold records.

#### Package by package

### `sgrna/` (the package root: 3 files, 334 lines)

Three single files at the root of the package, not folders: `config.py` (project paths and shared constants), `io_utils.py` (reading the base matrix and writing feature blocks) and `common.py` (output folders and the resolver for results this repository does not track). Nearly every other module imports them.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `common.py` | 47 | Output folders for the experiment modules, and the resolver for the main project's results that this repository does not track. | library | `sample_complexity.csv` |
| `config.py` | 199 | Project paths and shared constants. | Appendix C0 | — |
| `io_utils.py` | 88 | Shared conventions for reading the base matrix and writing feature blocks. | library | — |

### `sgrna/build/`  —  12 modules, 2,538 lines

Decode the published matrix, locate every guide on the genome, rebuild features for guides outside it, and the array featurisers that score a guide in tens of microseconds.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `a_flank_fast` | 196 | Family A without a Python loop over nucleotides: the same 348 columns, computed on arrays. | Appendix E | — |
| `base_fast` | 111 | The 6,169 base columns without a Python loop over columns or guides: the same matrix as `featurise.matrix`, on arrays. | Appendix E | — |
| `build_features` | 211 | Build feature families for the published rows or for the 33,567-guide cohort, one CSV per family. | Appendix E | — |
| `build_guide_index` | 166 | Step 0 -- turn 13,880 anonymous rows into located guides. | library | — |
| `build_matrix` | 193 | Step 2 -- assemble a model-ready matrix from the base features plus blocks. | library | — |
| `curated_matrix` | 101 | Featurise the 33,567 guides of the E. coli screen (Guo et al., 2018) as crisprHAL 2 curated them, and cache the matrix every model loads. | Appendix E | — |
| `decode_v_columns` | 196 | Work out what the 5,888 `V####` columns are, so new guides can be featurised. | library | — |
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

### `sgrna/harness/`  —  5 modules, 2,155 lines

The leakage-safe cross-validation harness every experiment runs through, plus the notebook front end.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `cohort_groups` | 74 | Grouped cross-validation means something different on the curated cohort: gene identity explains 29% of its label variance against 16% on the published cohort. | Appendix B | `cohort_groups.csv` |
| `evaluate` | 708 | Quantifying the two things a model is judged on: prediction and explanation. | Appendix B | `interpretability.csv`, `model_comparison.csv` |
| `notebook` | 532 | Notebook front end -- everything the pipeline does, callable from a cell. | Appendix A | `ablation_abund_split.csv`, `ablation_curated.csv` +13 |
| `run_ablation` | 560 | Step 3 -- the ablation harness. | Appendix A | `ablation_abund_split.csv`, `ablation_curated.csv` +27 |
| `seqnet` | 281 | A convolutional net on raw DNA does not beat gradient boosting: it reaches rho 0.525 at best, against 0.607 for boosting on the 6,580 engineered columns, on the same rows and folds. | Finding 2 | `seqnet.csv`, `seqnet_matched.csv` |

### `sgrna/tuning/`  —  8 modules, 1,633 lines

Hyperparameter search and model mixing: the joint search on development genes, its replication on a second split, the blend of model families, and the knobs no search had touched.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `family_a_at_tuned` | 159 | Family A's gain survives tuning and grows: +0.1697 rho at the tuned settings, up from +0.1563 at the original settings. | Finding 4 | `family_a_at_tuned.csv`, `family_a_at_tuned_summary.csv` +1 |
| `mix_ecoli` | 289 | Mixing model families barely helps: no blend beats the tuned XGBoost ensemble by more than +0.0017 rho on held-out genes, and adding weaker families at equal weight lowers it. | Part 5 | `mix_configs_cat.json`, `mix_configs_et.json` +9 |
| `mlp_worker` | 53 | Neural-network member of the E. coli mix, run in its own process. | Part 5 | — |
| `tune` | 241 | The margin survives tuning: a per-fold search adds +0.0131 rho to SLICER's original model, so its lead over crisprHAL 2's best known settings is +0.0161. | Part 5 | `tuning.csv`, `tuning_trace.csv` |
| `tune_analyze` | 60 | Shallower trees and slower learning with more trees score best across 200 random settings (mean rho 0.714 at depth 4, 0.698 at depth 10), and the other settings matter little. | Part 5 | `tune_configs.json`, `tune_stage1.csv` |
| `tune_ecoli` | 357 | Joint hyperparameter search for the E. coli model, with the selection kept apart from the confirmation. | Finding 4 | `tune_budget_check.csv`, `tune_configs.json` +5 |
| `tune_split2` | 231 | The tuning payoff replicates on a second gene split: a fresh search gains +0.0121 rho on genes it never saw, against +0.0143 on the first split. | Appendix B | `tune_configs.json`, `tune_split2_configs.json` +3 |
| `xgb_knobs` | 243 | Of eight XGBoost knobs never tuned before, only tree size matters: depth 6 adds +0.0063 rho, while the rest add at most +0.0024 or lower it. | Appendix B | `xgb_knobs.csv`, `xgb_knobs_verify.csv` |

### `sgrna/cohorts/`  —  16 modules, 2,341 lines

Ablations on the curated cohort and the arguments about what its numbers mean (ceilings, inflation, label noise), including the shape ablations of Appendix B4.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `ceiling_table` | 109 | The corrected ceiling as a function of replicate agreement and replicate count -- the parametric form of Finding 6. | Part 5 | `ceiling_parametric.csv` |
| `curated_ablation` | 339 | Feature-family ablations on crisprHAL's curated cohort (designs: one family at a time on random rows or whole genes, and `--paper-folds`). | Appendix B3 | `ablation_curated.csv`, `ablation_curated_gene.csv` +4 |
| `curated_ceiling` | 122 | The ceiling for the label we actually train on, from replicate agreement. | Finding 6 | `curated_ceiling.json` |
| `curated_joint` | 122 | The family gains overlap heavily: all eleven families together add +0.2030 rho, far less than the +0.7046 their single gains sum to. | Part 2 | `ablation_curated.csv`, `curated_joint.csv` |
| `curated_ladder` | 123 | The reduced-representation ladder, re-run on the curated cohort without selection. | Finding 1 | `curated_ladder.csv` |
| `curated_models` | 95 | Including family A does not make model class matter more: eight classes span 0.108 rho on the curated cohort, no wider than on the published matrix. | Appendix B | `curated_models.csv` |
| `curation_check` | 86 | Curation does not create the flank effect: the discarded guides with the most reads still show it (flank GC correlation 0.159, against 0.175 to 0.188 in kept guides). | — | `curation_check.csv` |
| `curation_variance` | 159 | The gains rose mainly because the genome-wide guides added in the larger cohort show a steeper flank effect (slope 15.85 against 5.35 on shared guides), not a wider spread. | Appendix B2a | `curation_variance.csv` |
| `data_ablation` | 152 | Hold SLICER fixed and change only the training data. | Part 5 | `data_ablation.csv` |
| `esp_replicate` | 110 | Second, independent cohort for the shape / 3D question: crisprHAL's E. coli eSpCas9 screen. | Appendix B4 | `esp_replicate.csv` |
| `local_sequence_bound` | 193 | An upper bound on what any local shape or stiffness predictor can add -- the open item Deep DNAshape left behind. | Appendix B4 | `local_sequence_bound.csv`, `local_sequence_bound_summary.csv` +1 |
| `replicate_weight` | 295 | The label's reliability is per guide, and the model treats it as uniform. | Appendix B | `replicate_weight.csv`, `replicate_weight_raw.csv` |
| `ridge_check` | 72 | It is a property of the pipeline: a ridge model on all columns gains +0.0097 rho from shape descriptors beyond family A, while boosted trees gain +0.0006 at best. | Appendix B4 | `ridge_check.csv` |
| `shape_ablate` | 207 | Ablations of the shape / 3D blocks on the curated cutting cohort, gene-grouped CV. | Part 5 | `folds_base_plus_K562_ATAC.csv`, `folds_base_plus_K562_DNase.csv` +13 |
| `shape_analyze` | 51 | Paired summaries of the shape ablation: every arm against a reference arm, fold by fold. | Appendix B4 | `shape_ablation.csv`, `shape_ablation_paired.csv` |
| `txn_interactions` | 106 | The interactions do not pay: adding the four interaction columns to the 15 single-factor columns raises rho by only +0.0004, far below the +0.0175 they earn alone. | Part 2 | `ablation_curated.csv`, `txn_interactions.csv` |

### `sgrna/controls/`  —  15 modules, 2,906 lines

The adversarial phase: nulls, shuffles, floors, selector and leakage audits, and the check that the fast featurisers equal the originals.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `chi_sites` | 198 | Chi sites do not explain the flank effect: with flank GC held fixed their coefficient is -0.053, within the spread of other GC-matched octamers (mean -0.022). | Appendix B | `chi_sites.csv` |
| `diagnose` | 429 | Binding energy and duplex stability are mostly functions of the 20-mer (70% and 50% of their columns have R² above 0.9), so they add little, and no other feature family is redundant in this way. | Appendix B2 | `ceiling.json`, `headroom.csv` +1 |
| `equiv_check` | 57 | Recorded equivalence of the array-based featurisers with the originals, and of the predictions built on them. | Part 5 | `featuriser_equivalence.csv`, `tune_configs.json` |
| `floors` | 86 | Per-family permutation floors, paired fold by fold. | Appendix A | `folds_base_plus_K562_ATAC.csv`, `folds_base_plus_K562_DNase.csv` +13 |
| `fold_distance` | 94 | Sequence distance between training and validation guides, the leakage statistic our comparators report. | Part 5 | `fold_distance.csv` |
| `mappability_stratum` | 299 | Family A's flank gain does not come from read depth or mappability: it is +0.157 among guides in uniquely mappable sequence and +0.160 with control read depth given to the model. | Finding 3 | `mappability_depth.csv`, `mappability_stratum.csv` +2 |
| `no_information` | 94 | Decompose `d_mechanics`'s +0.099 into representation and new DNA. | Finding 1 | `no_information.csv` |
| `rederive` | 543 | Re-derive the result files whose producer was lost, and check they agree. | Appendix B | `ablation_curated.csv`, `ablation_curated_gene.csv` +5 |
| `sealed_crossscreen` | 105 | A second screen as the sealed set: train on SpCas9 guides in development genes, score eSpCas9 labels in the sealed genes. | Part 4 | `sealed_crossscreen.csv` |
| `sealed_holdout` | 224 | On genes sealed away from column selection and settings, family A's gain is 0.0087 lower than in cross-validation of the pool, and a settings search inside the pool does not enlarge the drop. | Part 4 | `sealed_holdout.csv` |
| `sealed_holdout_gene` | 147 | Holding out whole genes in the pool cross-validation leaves the sealed-gene drop at +0.0081 against +0.0088 with random rows, so unseen genes explain almost none of it. | Part 4 | `sealed_holdout_gene.csv` |
| `selector_audit` | 272 | The pipeline selects columns with the one metric the report says never to use. | Appendix B | `selector_audit.csv`, `selector_audit_ranks.csv` +1 |
| `shiftnull` | 140 | A spatially preserving null for family A: shift the chromosome, keep its texture. | Finding 2 | `shift_null.csv` |
| `why_inflation` | 104 | Family A's gain is +0.1672 on subsamples of the curated cohort the size of the published matrix and +0.1638 on the whole cohort, so the larger gain is not due to the row count. | Appendix B2a | `why_inflation.csv` |
| `window_robustness` | 114 | Rebuilding family A on a window set sharing no length with the original gives a gain of +0.1634 against +0.1638, so the choice of windows does not matter. | Part 2 | `ablation_curated.csv`, `window_robustness.csv` |

### `sgrna/interpret/`  —  24 modules, 6,167 lines

What the model uses and what the signal is made of: importance, attribution, redundancy, the mechanism work of Finding 7, and what the tuned settings say.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `asymmetry` | 237 | The downstream flank's advantage is the same in low, middle and high expression genes (+0.046 to +0.049 in rho), so it is not a transcription effect. | — | `asymmetry_expression.csv`, `asymmetry_strand_audit.csv` |
| `attribution` | 214 | Two models, different features, same score. | Finding 5 | `attribution.csv`, `attribution_shap.csv` +2 |
| `base_enrichment` | 185 | Adding the published 63 hand-named summary columns to the sparse base takes only 0.0015 off family A's gain, so missing dense summaries do not explain the curated table's larger gains. | Appendix B3 | `base_enrichment.csv`, `base_enrichment_published.csv` |
| `compact_model` | 214 | A compact model of guide base indicators and flank composition columns keeps 88% of the full model's rho and can be read column by column, with SHAP agreeing across tree families at 0.87. | Finding 5 | `compact_model_accuracy.csv`, `compact_model_accuracy_folds.csv` +5 |
| `complexity` | 118 | The default SLICER model stores about 6,300 fitted leaf values, against 790,533 parameters in crisprHAL 2. | Part 5 | `complexity.json` |
| `faithfulness` | 270 | Dropping a model's top columns costs nothing. | Finding 5 | `faithfulness_clusters.csv`, `faithfulness_clusters_reduced.csv` +2 |
| `flank_scale` | 328 | The flank signal is a bulk GC effect over a few hundred bases: two GC columns reach +0.106 at 200 nt on each side, while purine and amino fractions add nothing. | Finding 7 | `flank_scale.csv`, `flank_scale_summary.csv` +2 |
| `gate_test` | 65 | The flank and the guide's own score add up rather than gate: an interaction term improves held-out R² by only +0.0035 and rho by +0.0007. | — | `gate_test.csv`, `mappability_stratum_oof.csv` |
| `gcstrata` | 210 | The flank effect is not target GC in disguise: holding target GC fixed raises flank GC's association with activity from +0.176 to +0.221, and the flank gain is positive in every GC stratum. | Finding 2 | `gc_strata_ablation.csv`, `gc_strata_corr.csv` +1 |
| `geom_importance` | 81 | The five most used geometry columns are excess twist over the flanks, which dinucleotide fractions predict out of fold at R² 0.86 to 0.97, so they re-express composition. | Appendix B4 | `geom_importance.csv` |
| `hal_readout` | 273 | crisprHAL 2's side of the interpretability comparison (`vs_crisprhal_interpret.py` has the design and the predictions). | Finding 5 | `guides_1500.csv`, `hal_eg_draws_control.csv` +2 |
| `hypotheses` | 401 | Three tests of what the flank effect might physically be, each with its prediction written down before it was run. | Finding 7 | `hypotheses_audit_gate.csv`, `hypotheses_audit_scale.csv` +15 |
| `importance` | 293 | Importance is a property of the data only for blocks of columns: the tree families agree on blocks but not on individual columns. | Finding 5 | `importance_agreement.csv`, `importance_blocks.csv` |
| `importance_models` | 297 | Per-column importance depends on the model asked: five tree families agree on columns at only 0.496 even with SHAP, but on blocks of columns at 0.800. | Finding 5 | `importance_model_blocks.csv`, `importance_model_methods.csv` +1 |
| `interp_extra` | 313 | Three interpretability probes on one fixed model: sensitivity-n of SHAP, SHAP interaction values, and counterfactual flank edits. | Finding 5 | `interp_counterfactual.csv`, `interp_interaction_blocks.csv` +3 |
| `label_orientation` | 40 | A higher curated label means more cutting: guides with several genomic copies have a median label of 2.11 against 0.27 for single-copy guides. | Part 2 | `label_orientation.csv` |
| `mechanism` | 285 | Biology out of the model, by routes that Finding 5's redundancy caution does not disqualify. | Finding 7 | `mechanism_distance_profile.csv`, `mechanism_pairs_blocks.csv` +5 |
| `novel_followups` | 231 | Two experiments that are ours rather than borrowed: species-relative composition, and the resolution floor of the flank signal. | Finding 7 | `novel_relative_composition.csv`, `novel_resolution.csv` +1 |
| `redescription` | 741 | Re-description helps only when it means something: random linear or non-linear functions of the guide add nothing, while the guide's real self-folding energy adds +0.0117. | Appendix B | `recoverability_percolumn.csv`, `recoverability_trend.csv` +7 |
| `redundancy` | 43 | Family A explains the shape blocks only partly (mean out-of-fold R² up to 0.74) and the 3D geometry columns hardly at all (about 0.07). | Appendix B2 | `shape_redundancy_from_A.csv` |
| `representation` | 406 | Seventy-four mononucleotide columns, a lossless spelling of the 20-mer, recover 96.4% of the full published matrix's rho, so the model does not need the other columns. | Finding 4 | `representation.csv`, `representation_depth.csv` +2 |
| `signal_profile` | 132 | The data behind the figure of what the flank signal looks like: the step, the slope by the guide's last base, and the correlation by window and by base. | — | `mappability_stratum_oof.csv`, `signal_corr.csv` +3 |
| `tuned_settings` | 199 | The tuned settings show a mostly additive signal: single-split stumps reach held-out rho 0.6905 against 0.7210 for shallow trees, and deeper trees do no better. | Finding 5 | `interpret_data_scaling.csv`, `interpret_depth.csv` +4 |
| `vs_crisprhal_interpret` | 591 | SLICER is more faithful than crisprHAL 2 (sensitivity-n +0.081 to +0.111) but ties or splits on stability, truth and compressibility, so it is not shown to be more interpretable. | Finding 5 | `guides_1500.csv`, `hal_flank_edits.csv` +14 |

### `sgrna/compare/`  —  43 modules, 7,786 lines

Every comparison with something else (Part 5 of the report): crisprHAL 2, the inherited baseline, other organisms and the human matrix, and other assays (the dCas9 boundary experiment of Appendix B5).

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `blend_with_hal` | 47 | Averaging the tuned trees with crisprHAL 2 at equal weight adds only +0.0085 on SpCas9 and nothing detectable on eSpCas9, so the network adds little beyond the trees. | Part 5 | `blend_with_hal.csv` |
| `boostmec_replica` | 291 | BoostMEC built the way its authors built it, retrained on the curated E. coli cohort and scored with SLICER's folds. | Appendix E | `boostmec_released.csv`, `boostmec_search_trace.csv` +2 |
| `clean_vs_hal` | 65 | A comparison with crisprHAL 2 that the tuning cannot have touched: the 30% of genes held out from every search. | Part 5 | `clean_vs_hal.csv` |
| `crisprhal_rerun` | 151 | Re-run crisprHAL 2 ourselves, on our rows and our folds. | Appendix B | `crisprhal_rerun.csv` |
| `crisprhal_tune` | 284 | Give crisprHAL 2 the same courtesy we gave ourselves: a hyperparameter search. | Appendix B | `crisprhal_tuned_folds.csv`, `crisprhal_tuning.csv` |
| `crisprhal_wide_window_folds` | 191 | crisprHAL 2 at wider input, on the folds the first run did not cover (local CPU; TensorFlow env `.venv-crisprhal`, not the main env). | Finding 2 | `crisprhal_wide_window.csv`, `crisprhal_wide_window_folds.csv` +2 |
| `crispri_screens` | 327 | Three CRISPRi screens instead of one, and what that buys beyond accuracy. | Appendix B5 | `crispri_screen_agreement.csv`, `crispri_screen_family_a.csv` +1 |
| `cui_final_exam` | 219 | Both models, neither trained on it, scored on Cui's CRISPRi screen. | Part 5 | `cui_final_exam.csv` |
| `cui_screen` | 594 | Family A adds nothing on Cui's dCas9 CRISPRi screen (−0.0093, whole genes held out) against +0.1563 on a cutting screen, although the screens differ in assay, library and laboratory. | Appendix B5 | `cui_essential_marginals.csv`, `cui_essential_position.csv` +4 |
| `cut_within_gene` | 36 | The cutting-screen figure on the within-gene definition used for the dCas9 screens. | Appendix B5 | `cut_within_gene.csv` |
| `data_efficiency` | 67 | The data-efficiency curve again, with the tuned model against the stored crisprHAL 2 curve. | Part 5 | `data_efficiency.csv`, `sample_complexity.csv` +1 |
| `eukaryote_rules` | 309 | Eukaryotes share guide rules (corrected correlation +0.59 between human tables) that are unrelated to E. coli's: G at position 20 helps in every eukaryotic table and hurts in E. coli. | Finding 7 | `eukaryote_contrast_g_minus_t.csv`, `eukaryote_flank_step.csv` +4 |
| `final_eval` | 247 | The designated final evaluation: both models on crisprHAL's eSpCas9 split. | Part 5 | `final_eval_espcas9.csv` |
| `gene_grouped_vs_crisprhal` | 84 | The head-to-head with crisprHAL 2 on folds that hold whole genes out (the lowest-leakage protocol of this project). | — | `crisprhal_gene_grouped_fcolab_b1024.csv`, `crisprhal_gene_grouped_fcolab_b256.csv` +2 |
| `handnamed_split` | 122 | Split the 63 hand-named columns into the computable and the rest, and test both. | Finding 1 | `handnamed_split.csv`, `portability_audit.csv` |
| `headtohead` | 303 | The like-for-like comparison against crisprHAL 2. | Part 5 | `headtohead.csv` |
| `heldout_compare` | 156 | Paired on the same held-out guides, SLICER's default model beats crisprHAL 2 by +0.0115 with its published settings and by +0.0043 with the best settings found for it. | Part 5 | `heldout_compare.csv` |
| `human_additivity_twin` | 154 | The human additivity claim and the knob sweep behind it, re-run with folds that keep repeated guides together. | Appendix B | `human_additivity_twin.csv`, `human_additivity_twin_claim.csv` +1 |
| `human_baseline` | 270 | The teammate's human baseline settings give ρ 0.5076 on this matrix against 0.4036 under the harness, and the gain comes from XGBoost's default tree settings, not the column count. | Appendix B | `human_additivity.csv`, `human_baseline.csv` +1 |
| `human_flank` | 173 | Human guides with their genomic flanks, for running Appendix B4 (shape / 3D versus family A) on human. | Part 2 | — |
| `human_shape` | 201 | On the human matrix the shape and 3D-path columns add +0.0183 to family A, where E. coli gives −0.0007, a difference that is not explained. | Appendix B4 | `folds_base_plus_K562_ATAC.csv`, `folds_base_plus_K562_DNase.csv` +15 |
| `human_twin_check` | 60 | Keeping repeat measurements of a guide in one fold drops the teammate's human baseline from 0.508 to 0.2515, below the harness model, so its lead came from memorising repeated guides. | Finding 3 | `human_twin_check.csv` |
| `kingdom_transfer` | 179 | Cross-kingdom transfer, with the human side fitted properly. | Part 3 | `kingdom_transfer.csv`, `kingdom_transfer_boot.csv` |
| `lit_followups` | 211 | Two ideas taken from the published predictors and tested on our cohort (Part 5 of the report). | Part 5 | `lit_binary.csv`, `lit_merf.csv` +2 |
| `metrics_panel` | 122 | Beyond rank correlation: the other things a guide-efficiency model is judged on, for SLICER and crisprHAL 2 on crisprHAL 2's own test files. | — | `metrics_panel.csv` |
| `noshay_replicate` | 232 | Reproduce the inherited paper's own model, as its methods section describes it. | Part 5 | `noshay_replication.csv` |
| `panel` | 219 | The full metric panel, for untuned and tuned, on identical folds. | Part 4 | `metric_panel.csv` |
| `panel_vs_hal` | 110 | The six-metric panel against crisprHAL 2 on its own two splits, for the tuned and mixed models, with raw predictions kept. | Part 5 | `panel_espcas9.csv`, `panel_wt.csv` |
| `portability` | 167 | A model using only the 412 columns that can be computed from any 20-mer plus the genome matches the full published matrix to within 0.00003 in ρ. | Part 2 | `portability.csv`, `portability_audit.csv` |
| `sample_complexity` | 220 | crisprHAL 2 needs 1.7 to 2.5 times as many labelled guides as SLICER to reach the same accuracy, at every accuracy level that both models reach. | Part 5 | `sample_complexity.csv` |
| `sealed_organism` | 97 | On C. rodentium, which no choice in this project ever saw, the tuned model beats the default model by +0.0172, within the range seen inside E. coli. | Part 4 | `sealed_organism.csv` |
| `strand_families` | 241 | With the strand call given to all families, flank family A adds only +0.007 on the dCas9 screen and only transcription context (+0.028) gains clearly in both groupings. | Appendix B5 | `strand_families.csv`, `strand_families_proxies.csv` +1 |
| `strand_recoverable` | 47 | After dropping the columns most correlated with strand, a classifier still reads the strand call at AUC 0.79 from flank family A and 0.9997 from transcription context. | Appendix B5 | `strand_families_recoverable.csv` |
| `strand_strata` | 142 | Two controls on the strand result. | Appendix B5 | `strand_flip.csv`, `strand_strata_rousset.csv` |
| `strand_within_gene` | 181 | Among coding-strand guides, family A's gain is not only gene ranking: it adds +0.102 inside genes on Cui's screen and still +0.096 on top of transcription context. | Appendix B5 | `strand_within_gene_cui.csv`, `strand_within_gene_cui_joint.csv` +1 |
| `topend` | 96 | No objective aimed at the top of the ranking beat plain squared error on top-decile precision (0.291 for squared error, at most 0.289 for the others), and the rank objectives lose Spearman. | Part 5 | `topend_confirm.csv`, `tune_configs.json` |
| `topk_panel` | 97 | Smoother top-end metrics against crisprHAL 2 on its own splits, and seed stability of the lead. | Part 5 | `seed_stability.csv`, `topk_panel.csv` |
| `transfer` | 588 | A model trained on SpCas9 keeps 88% of its own accuracy on eSpCas9, a different nuclease in the same genome, and flank features supply +0.098 of its ρ. | Finding 3 | `hybrid.csv`, `transfer.csv` +6 |
| `transfer_diagonal` | 39 | The low diagonal reflects hard data, not a weak model: the 20-mer pipeline's own cross-validation beats the median of six published predictors on all 11 tables and the best of them on 3. | — | `eukaryote_transfer.csv`, `transfer_diagonal.csv` |
| `transfer_to_ecoli` | 48 | The E. coli column of the transfer matrix: models trained on each eukaryotic table, scored on the 33,567 curated E. coli guides. | — | `transfer_to_ecoli.csv` |
| `transfer_vs_crisprhal` | 178 | Transferability, SLICER against crisprHAL 2: across nucleases within E. coli, and from E. coli to the eukaryotic tables. | — | `eukaryote_transfer.csv`, `transfer_vs_crisprhal_kingdom.csv` +1 |
| `vs_crisprhal` | 186 | The tuned / mixed E. coli model against crisprHAL 2 again, under the three conditions of the original comparison. | Part 5 | `mix_finalists.csv`, `tune_configs.json` +3 |
| `vs_crisprhal_wt_hal` | 35 | crisprHAL 2's shipped WT-SpCas9 weights on their own WT test file, saved for a paired comparison (TensorFlow environment). | Part 5 | — |

### `sgrna/cost/`  —  10 modules, 2,056 lines

What each model costs to train and to run, measured rather than estimated.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `bench` | 319 | Peak memory and training cost, measured rather than estimated. | — | `memory_comparison.csv`, `memory_comparison_runs.csv` +1 |
| `cost` | 372 | Measures CPU-seconds and floating-point operations to fit SLICER, a cut-down SLICER and crisprHAL 2; superseded by cost_repeats, as its crisprHAL 2 CPU-seconds were a hand-entered estimate. | Part 5 | `cost_cpu_seconds.csv`, `cost_flops.csv` +1 |
| `cost_bench` | 129 | Times the model alone on one thread: 0.002 ms per guide for SLICER's default model against 1.32 ms for crisprHAL 2 at a batch of 100; its featurisation timing is superseded by cost_repeats. | Part 5 | `cost_crisprhal.csv`, `cost_featurise.csv` +3 |
| `cost_boostmec` | 188 | BoostMEC scores a guide in 3.07 ms at a batch of 100 on one thread, 42.3 times SLICER's default model, using its released model and the same protocol as SLICER and crisprHAL 2. | Part 5 | `boostmec_oracle.csv`, `model_size.csv` |
| `cost_boostmec_shipped` | 240 | BoostMEC as shipped: the authors' own pipeline, run unmodified in their Docker image, timed and compared with the Python version of `cost_boostmec.py`. | — | `boostmec_shipped_check.csv`, `boostmec_shipped_cohort.csv` +3 |
| `cost_repeats` | 295 | Cost with its spread: every timing repeated, every repeat kept, so a ratio comes with a distribution and not a bare range. | — | `cost_repeats_featurisers.csv`, `cost_repeats_fit.csv` +2 |
| `e2e_bench` | 175 | End-to-end scoring cost per guide, one thread: featurisation + column selection + imputation + model, against crisprHAL 2 with its input encoding. | Part 5 | `e2e_crisprhal.csv`, `e2e_slicer.csv` +3 |
| `inference_cost` | 141 | Training cost is the wrong clock for a guide-design tool. | Part 5 | `inference_cost.csv` |
| `model_size` | 91 | SLICER's default model stores 6,338 leaf values against 790,533 parameters for crisprHAL 2, but the tuned SLICER model is larger on disk than crisprHAL 2's file. | Part 5 | `model_size.csv` |
| `rds_read` | 106 | Minimal read-only parser for R's XDR serialisation (.rds), enough for plain data (lists, vectors, attributes). | — | — |

### `sgrna/paper/`  —  21 modules, 7,370 lines

The tooling the paper rests on: the audit, the figures, the import and reference guards, and the code map.

| module | lines | what it does | reported in | writes |
|---|---:|---|---|---|
| `code_map` | 425 | Generate the per-module tables of CODE_MAP.md from the code itself, so the map cannot drift from the files. | tooling | — |
| `doc_refs` | 165 | Every module, path and Colab script that a document or a docstring names still exists. | tooling | — |
| `figures` | 427 | Publication figures, to ACS graphics specifications, generated from the result files. | tooling | — |
| `figures_paper` | 459 | Figures for the point-form paper, every one drawn from stored result files (no schematics). | Finding 7 | `boostmec_released.csv`, `cost_gpu.csv` +16 |
| `import_check` | 107 | Every module in the package imports, and importing it writes nothing. | tooling | — |
| `paper_numbers` | 130 | Audit of every number in the paper (point form): each is recomputed from a stored result file, as `verify_numbers` does for the report. | Part 5 | `blend_with_hal.csv`, `clean_vs_hal.csv` +7 |
| `paper_numbers_A` | 431 | Audit of the numbers in the draft paper's claims [0]-[19], [35]-[45] (group A): data sets and methods, the flank effect, its mechanism and the appendix constraints. | Appendix A | `ablation_curated.csv`, `ablation_curated_gene.csv` +41 |
| `paper_numbers_B` | 231 | Audit of the numbers in the draft paper's claims [20]-[28], [32]-[34] and [49] (group B): every number is recomputed from a stored result file and compared with the value the paper prints. | Finding 3 | `ablation_curated.csv`, `ceiling_parametric.csv` +25 |
| `paper_numbers_C` | 326 | Audit of paper claims [29]-[31], [46]-[48], [50]-[55] (group C): every number is recomputed from a stored result file. | Part 5 | `ablation_curated.csv`, `ablation_label_curated.csv` +38 |
| `paper_numbers_D` | 679 | Audit of the numbers added to the paper (point form) in the second pass of 23 Nov 2026: one protocol, the reach of the flank signal, controls, the transfer matrix, SLICER against crisprHAL 2 on more metrics … | Finding 3 | `ablation_curated.csv`, `ablation_paper_folds_summary.csv` +54 |
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
<details><summary>146 modules have moved; documents that cite the left-hand paths (for example `Jacky/EXPERIMENT_CATALOG.md`) can be updated from this table</summary>

| was | is now |
|---|---|
| `sgrna/asymmetry.py` | `sgrna/interpret/asymmetry.py` |
| `sgrna/attribution.py` | `sgrna/interpret/attribution.py` |
| `sgrna/base_enrichment.py` | `sgrna/interpret/base_enrichment.py` |
| `sgrna/bench.py` | `sgrna/cost/bench.py` |
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
