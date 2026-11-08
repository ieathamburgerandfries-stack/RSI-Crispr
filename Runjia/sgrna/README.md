# `sgrna` — the analysis pipeline behind `docs/PROGRESS_REPORT.md`

Every number in that report comes from this package. Committed as one unit
because the modules import each other; individual files will not run on their
own.

## It needs a data layout that is not in this repo

The code resolves paths relative to a working folder three levels up from
`config.py`, and expects:

```
<project>/
├── data/raw/            ecoli_feature_matrix.csv, human_feature_matrix.csv
├── data/reference/      NC_000913.2.fasta, gene table, dinucleotide_shape_scales.tsv
├── data/interim/        caches this code writes (several GB)
├── external_data/       crisprHAL, CRISPRoff, GapR-seq, Hi-C, RegulonDB, C. rodentium genome
├── results/             every .csv the report cites
└── src/sgrna/           this package
```

Those inputs total several GB and are other people's published data, so they
are deliberately not committed. `external_data/README.md` in the working folder
records where each came from. **Clone this and it will import but not run** —
it is here so the analysis is readable and reviewable, not so it is one command
away from reproducing.

`features/dinucleotide_shape_scales.tsv` is the one exception: a small
parameter table that belongs with the code. In the working folder it lives at
`data/reference/`, which is where `i_shape.py` reads it from.

## What is where

Sixty-six modules, grouped by what they are for. Audited on 8 November 2026
against the maintained documents, the import graph and the result files each
one writes: **every module is reachable from a live claim**, so nothing here is
dead code. Two notes from that audit are recorded at the bottom.

### The pipeline proper — run in this order

| module | what it does |
|---|---|
| `config.py` | all paths and model settings in one place, including `SELECTOR` |
| `genome.py`, `io_utils.py`, `qct.py` | genome access, caching conventions, the recovered quantum tables |
| `decode_v_columns.py` | works out what the 5,888 anonymous `V####` columns are |
| `featurise.py`, `expand.py` | regenerate 6,169 of the 6,232 published columns from any 20-mer, and lift the 13,880-row cap onto the curated cohort |
| `build_guide_index.py` | locate each guide in NC_000913.2 with the right PAM |
| `build_features.py`, `build_curated_features.py` | build feature families, on the published rows and on the curated ones |
| `build_matrix.py` | assemble a model-ready matrix from base + blocks |
| `run_ablation.py` | the ablation harness: grouped CV, permutation controls, the selector, checkpoint-resume |
| `notebook.py` | thin wrappers so a Colab cell stays short |

### Feature families

| module | what it does |
|---|---|
| `features/` | the eleven feature families, one module each. Appendix C of the report describes what each actually captures; `a_flank` carries the one large effect |
| `make_diagnostic_blocks.py` | decompose the positional families into their parts — this is what caught `d_supercoiling` |
| `no_information.py` | split `d_mechanics` into the 22 columns that restate the guide and the 8 that read flanking DNA |

### The effect, and the controls on it

| module | what it does |
|---|---|
| `curated_ablation.py`, `curated_joint.py` | family ablations on the curated cohort, and how much the eleven gains overlap |
| `shiftnull.py` | the texture-preserving null: shift the chromosome, keep its statistics, break the correspondence |
| `gcstrata.py` | is the flank effect target GC wearing a different hat? (no — conditioning raises it) |
| `window_robustness.py` | would a different set of flank windows have answered differently? (no) |
| `asymmetry.py` | is the downstream flank's advantage a transcription effect? |
| `chi_sites.py` | is it a *repair* effect? RecBCD attenuation at Chi, against a GC-matched octamer null |
| `txn_interactions.py` | family G's own rule — an interaction must beat both its parents |
| `cui_screen.py` | the dCas9 CRISPRi screen: the cross-assay test, its essentiality split, and the strand and position controls that say what that split measures |

### Is the representation redundant, and does re-description pay?

| module | what it does |
|---|---|
| `diagnose.py` | why doesn't adding features help — redundancy, headroom, residual correlation |
| `representation.py` | how few published columns the model needs; **412** are portable and land 0.00003 ρ from all 6,232 on the published matrix, though no reduced *representation* holds on the curated cohort (Finding 1) |
| `handnamed_split.py` | the 63 hand-named columns, split into the computable and the rest |
| `curated_ladder.py` | the same ladder on the curated cohort, with no selection in any arm |
| `portability.py` | can the 427-column representation be rebuilt for a guide never seen before? |
| `redescription.py` | does a deterministic function of the guide have to *mean* something? Random tables, random non-linear functions, real and shuffled folds |
| `base_enrichment.py` | is the curated family table inflated because its base carries no dense summaries? |

### Interpretability

| module | what it does |
|---|---|
| `importance.py` | at what granularity is "importance" a property of the data rather than of the fit |
| `importance_models.py` | seven model families × three importance methods — why SHAP survives a change of model and split gain does not |
| `attribution.py` | two models pick different columns and score the same: are they disagreeing about biology, or is the matrix redundant? This is the test behind Finding 5's explanation |
| `faithfulness.py` | drop what the ranking points at and refit — and why that is uninterpretable on a redundant matrix |

### The comparison against the state of the art

| module | what it does |
|---|---|
| `headtohead.py`, `crisprhal_rerun.py` | the like-for-like comparison, including re-running their model on our rows and folds |
| `crisprhal_tune.py` | a hyperparameter search for crisprHAL 2, so it is not tuned-against-untuned |
| `heldout_compare.py` | paired on guides rather than on folds |
| `final_eval.py` | both models on crisprHAL's own eSpCas9 split, against their shipped weights |
| `data_ablation.py` | hold the architecture fixed and change only the training data |
| `sample_complexity.py` | guides needed to reach a given accuracy — the one cost measure independent of architecture |
| `panel.py` | the nine-metric comparison: ranking, calibration, NDCG, pick-percentile, within-gene |
| `noshay_replicate.py` | the inherited paper's own iRF model, run here |
| `seqnet.py` | a convolutional net on raw sequence, as the in-house comparison |
| `evaluate.py`, `bench.py`, `cost.py`, `complexity.py` | scoring, peak memory, CPU-seconds, FLOPs, and what a boosted ensemble's "parameter count" even means |

### Ceilings, transfer and the honesty checks

| module | what it does |
|---|---|
| `curated_ceiling.py`, `ceiling2.py` | the label's reliability ceiling from replicate agreement, and whether two screens' disagreement is noise or biology |
| `transfer.py` | cross-enzyme, cross-organism and hybrid experiments |
| `kingdom_transfer.py`, `human_baseline.py` | the human arm, fitted properly rather than at *E. coli* settings |
| `cui_final_exam.py` | both models on a screen neither trained on |
| `sealed_holdout.py` | seal 20% of genes by gene and measure the adaptive-overfitting component |
| `cohort_groups.py` | does grouped CV still mean the same thing after the cohort moved? |
| `curation_variance.py`, `why_inflation.py` | why every family's gain rose when the baseline rose too — the four candidate causes, then the mechanism |
| `floors.py` | per-family permutation floors, paired fold by fold |
| `curated_models.py`, `tune.py`, `xgb_knobs.py` | does model class matter once the representation is right; the search; and the eleven knobs no search ever touched |
| `numcheck.py` | checks every number in the maintained documents against the result files. Run it before committing a document change |
| `rederive.py` | re-derives the result files whose producer was lost, and checks they agree |

## Two settings worth knowing before changing anything

**Importance is reported from SHAP, not split gain.** `importance.shap_importance`
is what `attribution.py`, `evaluate.py` and `importance_models.py` all call, and
both of the latter take `--importance native` if you want the historical
gain-based figures. This is not a stylistic choice: across seven model families
gain agrees at ρ 0.05 per column against SHAP's 0.50, and gain additionally
credits a column for being binary — worth 26 percentage points of XGBoost's
attribution. Do not put a gain plot in a paper from this repo.

**`config.SELECTOR`** picks the model that ranks columns before the champion is
fitted. It defaults to `"xgboost"` because that is what the frozen baseline
used, and switching to `"lightgbm"` — which is 3× faster and ranks just as well
— moves the baseline off the published 0.2937 / 0.5278 figure the project
reproduces as a harness check. Fast sweeps can use it; headline numbers should
not.

**Two configurations, two libraries.** The baseline on the 13,880 published rows
uses **XGBoost** as predictor, because that is what reproduces Noshay et al.'s
figure and anchors the harness. The curated head-to-head on 33,567 rows uses
**LightGBM** as predictor with XGBoost still selecting columns. They agree to
within 0.001 wherever both were run, but a number quoted from one is not a number
from the other: ρ 0.5278 is XGBoost, ρ 0.7078 is LightGBM.

**Grouped cross-validation is not optional for positional features.** The
screen puts ~20 guides in every gene, so plain `KFold` lets a model score well
by recognising a locus. Anything positional is reported under `arc5` or a
blocked scheme; see `run_ablation.load_groups`.

**`cross_validate` resumes from `results/folds_<tag>.csv`.** A fold already
recorded there is skipped, which is what lets a long run survive a dropped
session — and is also how a cached result once got replayed into a control arm
that had never run, because two different arms were given the same tag. If you
add an arm, make sure its tag is unique for every distinguishing parameter, not
just its first word. `cui_screen._slug` is there for exactly this reason.

## Notes from the November audit

**Nothing here is dead, but one module was invisible.** `attribution.py` is
cited by no maintained document and imported by nothing, which is how a module
usually earns deletion. It is in fact the test behind Finding 5's central
explanation — that the instability of feature importance on this matrix is
redundancy rather than two models finding different biology — and it writes four
result files. It is now listed above and named in the report.

**This file had no source.** It existed only in the published mirror at
`RSI-Crispr/Runjia/sgrna/README.md`, which `sync_to_shared.sh` writes and never
reads, so it was unmaintained by construction and had gone stale. Restored here,
where it is the source of truth; the sync copies it across. If you find another
file that exists only in the mirror, that is the same bug — the sync never
deletes, so a source file removed here leaves its mirror copy behind for ever.
