# `sgrna` — the analysis pipeline behind `Runjia/PROGRESS_REPORT.md`

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

| module | what it does |
|---|---|
| `config.py` | all paths and model settings in one place, including `SELECTOR` |
| `genome.py`, `io_utils.py`, `qct.py` | genome access, caching, the recovered quantum tables |
| `decode_v_columns.py`, `featurise.py`, `expand.py` | the decoding work — regenerate 6,169 of the 6,232 published columns from a 20-mer |
| `build_guide_index.py`, `build_features.py`, `build_matrix.py` | locate guides, build feature families, assemble the matrix |
| `features/` | the **eleven** feature families, one module each — `a_flank` is the one that worked, `j_periodicity` the most recent null |
| `run_ablation.py` | the ablation harness: grouped CV, permutation controls, the selector |
| `evaluate.py`, `importance.py`, `attribution.py` | model comparison and the interpretability measurements |
| `diagnose.py`, `ceiling2.py` | the redundancy / headroom / ceiling diagnostics |
| `representation.py` | how few of the 6,232 published columns the model needs — **412** are portable and land 0.00003 ρ from all of them on the published matrix, though no reduced *representation* holds on the curated cohort (Finding 2) |
| `importance_models.py` | seven model families × three importance methods — why SHAP survives a change of model and split gain does not |
| `asymmetry.py` | is the downstream flank advantage a transcription effect? (no — and half the test the data cannot support) |
| `bench.py`, `cost.py` | peak memory, CPU-seconds, FLOPs and bytes moved — one fresh process per repetition |
| `panel.py` | the nine-metric comparison: ranking, calibration, NDCG, pick-percentile, within-gene |
| `sample_complexity.py` | guides needed to reach a given accuracy — the one cost measure independent of architecture |
| `faithfulness.py` | drop what the ranking points at and refit — and why that is uninterpretable on a redundant matrix |
| `noshay_replicate.py` | the inherited paper's own iRF model, run here |
| `crisprhal_tune.py` | a hyperparameter search for crisprHAL 2, so the head-to-head is not tuned-against-untuned |
| `make_diagnostic_blocks.py` | the blocks that decomposed the "supercoiling" result |
| `headtohead.py`, `crisprhal_rerun.py` | the comparison against crisprHAL 2, including re-running their model |
| `transfer.py` | cross-enzyme, cross-organism and hybrid experiments |
| `tune.py`, `seqnet.py`, `floors.py` | hyperparameter search, the CNN baseline, permutation floors |
| `notebook.py` | thin wrappers so the Colab notebook stays short |

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
