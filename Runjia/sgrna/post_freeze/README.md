# `sgrna.post_freeze` — the extension, by theme

Everything done after the freeze of 11 October 2026: Finding 7, Appendix B4 and the head-to-head (Part 6) of `PROGRESS_REPORT.md`, the hyperparameter search, the comparison with crisprHAL 2,
the figures and the audit, recorded in `EXTENSION_LOG.md`. Nothing in the frozen pipeline is edited; these modules import it. Results go to
`results/post_freeze/`. **`../../CODE_MAP.md` is the map of the whole package, generated from the code and checked by the audit; this file adds the
hand-written context for the extension.**

Six folders, each a theme (the crisprHAL 2 comparison moved to `sgrna/compare/` on 22 Nov, with Part 6 of the report). The import graph clusters this way on its own: the shape modules import each other, the human modules
import the same human loader, and `tuning/tune_ecoli.py` (the cohort and model loader) is imported from most of the rest.


**Use `../run_ext.sh`, which sets all of this for you** — `./run_ext.sh sgrna.post_freeze.paper.verify_numbers`, or `./run_audit.sh`
for the audit. Three variables have to be right together and getting one wrong fails with a `FileNotFoundError` that
does not say which: `sgrna/config.py` finds the root by walking up from `src/sgrna/`, which guesses wrong from this
fork's `Runjia/sgrna/` layout, so `RSI09_ROOT` must name the folder that actually holds `data/` and `external_data/`;
`EXT_OUT` must name `<fork>/Runjia/results/post_freeze`; and `PYTHONPATH` must include `<fork>/Runjia`. The interpreter also
lives outside both repositories (override with `PY=...`). Done by hand:

    RSI09_ROOT=<project with data/> EXT_OUT=<fork>/Runjia/results/post_freeze PYTHONPATH=<fork>/Runjia python -m sgrna.post_freeze.<module>

## `tuning/` — Hyperparameter search, model mixing, and what the tuned settings say about the signal.

| module | what it does |
|---|---|
| `tune_ecoli.py`, `tune_analyze.py` | joint hyperparameter search for the E. coli model judged on held-out genes: 200 random draws on 70% of genes, re-scoring on fresh partitions, confirmation on the other 30%, the eSpCas9 screen as external check, and the whole-cohort standard protocol (`EXTENSION_LOG.md` section 11) |
| `mix_ecoli.py`, `mlp_worker.py` | LightGBM / CatBoost / extra-trees / neural network / ridge searches on the same DEV genes, blends on the held-out genes (the network runs in its own process: PyTorch next to XGBoost stalled silently) |
| `interpret.py` | interaction-order curve, gain by column kind and by position, within- versus between-gene gain, data scaling |
| `family_a_at_tuned.py` | family A's increment measured at the *tuned* settings as well as the champion's, base-only against base+A under the standard gene-grouped protocol — the project's headline biology number against its headline model (`EXTENSION_LOG.md` section 16) |
| `tune_split2.py` | a fresh random search on a *second* 70/30 gene split, confirmed on its own held-out genes, so the tuning payoff is replicated rather than resting on one partition (section 16) |

## `speed/` — Fast featurisers, the per-locus precompute, and the cost benchmarks.

| module | what it does |
|---|---|
| `a_flank_fast.py`, `base_fast.py`, `e2e_bench.py`, `equiv_check.py` | array-based featurisers that reproduce the original columns exactly (checked on all 33,567 curated guides), the end-to-end timing against crisprHAL 2, and the recorded equivalence |
| `flank_locus.py` | family A from a *locus* instead of from a guide: cumulative base-count and tensor tracks along the chromosome make every window statistic a subtraction and the longest-homopolymer-run column two sparse-table lookups; identical to `a_flank_fast` to 4.6e-12, 0.093 to 0.036 ms per guide end to end (section 22.1) |

## `shape/` — DNA shape and 3D descriptors of the flank, and the bound on any local predictor.

| module | what it does |
|---|---|
| `dnashape.py` | parses the DNAshapeR pentamer table (`external_data/dnashaper_tables/TableCompiler.cpp`) into named columns by source (mc / xrc / md) and kind; verification checks |
| `geom.py` | per-guide shape tracks, rigid-body 3D path (Lu & Olson mid-step convention), the `mean` / `vec` / `geom` blocks; shuffled-flank and twist-recentred controls; plain and dinucleotide composition references |
| `shape_build.py` | builds and caches the blocks for the curated cohort |
| `shape_ablate.py` | gene-grouped ablations (`--stage 1` … `9`; see the docstring and the log for what each stage is) |
| `analyze.py` | fold-paired differences between arms |
| `redundancy.py`, `geom_importance.py` | how much of each block family A predicts; which geometry columns the model uses, and how well letter and dinucleotide composition predict each (R² 0.86-0.97 for the excess-twist columns by ridge on dinucleotides) |
| `ridge_check.py` | the shape-on-A comparison with ridge regression instead of boosted trees |
| `esp_replicate.py` | the same comparison on crisprHAL's eSpCas9 screen (independent cohort) |
| `local_sequence_bound.py` | an upper bound on what any local shape or stiffness predictor (Deep DNAshape included) can add: hand the model the inner +-50 nt one-hot and see whether it buys anything; +0.0002 rho (section 22.3) |

## `human/` — The human matrix with its genomic flanks: Appendix B4 on human, the twin-leak, fold distance.

| module | what it does |
|---|---|
| `human_flank.py`, `human_shape.py`, `human_twin_check.py` | human side result (stopped; see `EXTENSION_LOG.md` section 10): hg38 location of the human guides, the shape ablation on human, and the twin-grouped re-run of the human baseline |
| `human_additivity_twin.py` | the ridge-versus-boosting ratio and the whole knob sweep behind "the human signal is less additive", re-run under twin-grouped folds with row-wise folds kept in the same run as a positive control (section 16) |
| `fold_distance.py` | train-to-validation minimum Hamming distance, the leakage statistic the comparators report; its positive control recovers the human leak at 39.1% |

## `signal/` — What the flank signal is, the field's ceiling, and published ideas tried on our cohort.

| module | what it does |
|---|---|
| `cut_within_gene.py` | the cutting-screen within-gene decomposition (uses `sgrna.strand.strand_within_gene.run`) |
| `mechanism.py` | biology out of the model by estimators that do not depend on it: GC not purine, side-symmetry, the 250-500 nt length scale, position x base design rules and their inversion between kingdoms (Finding 7) |
| `novel_followups.py` | species-relative composition (a null) and the resolution curve of the flank (Finding 7) |
| `lit_followups.py` | two ideas taken from the published predictors, both nulls: a back-fitted mixed-effect model and a binarised classifier (section 18) |
| `ceiling_table.py` | the corrected ceiling as a function of replicate agreement and replicate count (Finding 6, section 22.2) |

## `paper/` — The tooling the paper rests on: the audit, the figures, the guards and the map.

| module | what it does |
|---|---|
| `figures.py` | the seven publication figures, to ACS specification, from the result files |
| `verify_numbers.py` | the audit: recomputes every number the report quotes from its file (`../../run_audit.sh`) |
| `import_check.py` | asserts that every module imports and that importing one writes nothing |
| `doc_refs.py` | asserts that every module, path and Colab script a document or docstring names still exists |
| `code_map.py` | generates the tables in `../../CODE_MAP.md` from the code |

Data downloaded for this work (publisher / RCSB / GitHub text, nothing executed): `external_data/kim2020_cas9_variants/`
(Supplementary Tables of Kim et al. 2020), `external_data/pdb_cas9/5F9R.pdb`, `external_data/dnashaper_tables/`,
`external_data/pammla_silverstein2025/` (one CSV and the licence, 729 KB), `external_data/hg38/hg38.fa.gz` (UCSC, 983,659,424 bytes; used only for the human side result).
`external_data/deepDNAshape_src/` was cloned and **not run**.
