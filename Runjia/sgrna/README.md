# `sgrna` — the analysis pipeline behind `PROGRESS_REPORT.md`

Every number in that report comes from this package. Committed as one unit
because the modules import each other; individual files will not run on their
own.

## Layout

Organised **by function, not by date**. Until 22 November 2026 the modules written after the 11 October freeze lived in a
`post_freeze/` folder of their own, and the strand work in `strand/`; the same kind of module (a tuner, a benchmark, a
shape descriptor) sat in two places depending on when it was written. They were moved into the package that says what they
are for, with `git mv`, so history follows each file. `config.py`, `io_utils.py` and `common.py` stay at the top because
almost every module imports them. **[`../CODE_MAP.md`](../CODE_MAP.md) is the map of every module, generated from the code
and checked by the audit; [`../CODE_OVERVIEW.md`](../CODE_OVERVIEW.md) is the short tour of the code the claims depend on.**

| folder | what is in it | modules |
|---|---|---:|
| `build/` | turning sequence into columns: the feature matrix, the guide index, the genome and quantum-chemical tables, the published-matrix decoder, and the array featurisers that score a guide in tens of microseconds | 13 |
| `features/` | the eleven feature families, one module each, and the shape / 3D descriptors that Appendix B4 tested against family A | 14 |
| `harness/` | the shared fit-and-score machinery every result goes through: `run_ablation`, folds, `evaluate`, the CNN baseline | 5 |
| `tuning/` | hyperparameter search and model mixing: the joint search on development genes, its replication on a second split, the blend of model families, and the knobs no search had touched | 8 |
| `cohorts/` | the curated and published cohorts, their ablations, ceilings and label variance, and the shape ablations | 15 |
| `controls/` | nulls and audits: shifted flank, no-information, Chi sites, the sealed holdout, the selector audit, leakage distance, and the check that the fast featurisers equal the originals | 12 |
| `interpret/` | importance, attribution, faithfulness, redescription, redundancy, the mechanism work of Finding 7, and what the tuned settings say | 16 |
| `compare/` | every comparison with something else (Part 6 of the report): crisprHAL 2, the inherited baseline, other organisms and the human matrix, and other assays, including the dCas9 boundary experiment of Appendix B5 | 34 |
| `cost/` | wall-clock, CPU, memory and inference benchmarks | 5 |
| `paper/` | the audit, the figures, the import and reference guards, the code-map generator | 5 |

Results are the one place the date shows: `results/` holds the numbers from before the freeze and `results/post_freeze/` those
from after it. `./run_ext.sh <module>` runs any module with the three environment variables it needs (below), and
`./run_audit.sh` recomputes every post-freeze number the report quotes.

**Use `../run_ext.sh`, which sets all of this for you** — `./run_ext.sh sgrna.paper.verify_numbers`. Three variables have to be
right together, and getting one wrong fails with a `FileNotFoundError` that does not say which: `sgrna/config.py` finds the
root by walking up from `src/sgrna/`, which guesses wrong from this fork's `Runjia/sgrna/` layout, so `RSI09_ROOT` must name
the folder that actually holds `data/` and `external_data/`; `EXT_OUT` must name `<fork>/Runjia/results/post_freeze`; and
`PYTHONPATH` must include `<fork>/Runjia`. The interpreter also lives outside both repositories (override with `PY=...`).
Done by hand:

    RSI09_ROOT=<project with data/> EXT_OUT=<fork>/Runjia/results/post_freeze PYTHONPATH=<fork>/Runjia python -m sgrna.<package>.<module>

**If you move a module,** use `git mv`, then run the three guards: `./run_ext.sh sgrna.paper.import_check` (every module still
imports, and importing writes nothing), `./run_ext.sh sgrna.paper.doc_refs` (every document and docstring that names a module or
path still resolves) and `./run_audit.sh`. Regenerate the map with `./run_ext.sh sgrna.paper.code_map`.

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
