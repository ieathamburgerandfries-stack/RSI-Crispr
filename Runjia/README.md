# Runjia/: folder guide

SLICER predicts how efficiently a Cas9 guide cuts in *E. coli*. It extends the quantum chemical tensor (QCT) feature set of
[Noshay et al. 2023, NAR 51:10147](https://doi.org/10.1093/nar/gkad736), which describes the 20-nt guide, with descriptors of the DNA
around the target, and fits a small gradient-boosted model to them. This folder holds the pipeline (`sgrna/`), the GPU scripts
(`colab/`), every result file the pipeline wrote (`results/`) and the documents that state and defend what was found.

**The current headline is not repeated here.** The final SLICER model as shipped, against crisprHAL 2 as shipped, is stated in the
top-level [`README.md`](../README.md) of the repository and in [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md) (Summary and Part 5).
Where this guide and either of them differ, this guide is stale.

## Design a guide

The point of the project: give it a genome and a cut site, get the SpCas9 guide SLICER predicts will cut best there.

```bash
python -m sgrna.design.design_guide --genome genome.fasta --cut-site 1234567            # exactly at the site; widens if no NGG guide cuts there
python -m sgrna.design.design_guide --genome genome.fasta --cut-site 1234567 --window 15 --top 5 --out guides.csv
```

From a fresh clone, with no project data (checked on a clean export of the tracked files; tested with xgboost 3.4, scikit-learn 1.9, numpy 2.5, pandas 3.0, scipy 1.18, the versions the models were saved with):

```bash
git clone https://github.com/ieathamburgerandfries-stack/RSI-Crispr && cd RSI-Crispr/Runjia
pip install -r requirements-design.txt
PYTHONPATH=. python -m sgrna.design.design_guide --genome genome.fasta --cut-site 1234567
```

You supply the genome FASTA; everything else (the models and the lookup tables) is in `models/`. The family A release (`family_A_v1.zip`) is data only and is not needed to design a guide. The cut site is the 1-based coordinate of the base before the cut on the plus strand.
It lists every NGG guide on either strand that cuts within `--window` bp of the site, reads the 1,000 nt on each side of each (the model's input), scores them with the shipped model
(`models/slicer_tuned.ubj`, 10 MB; `--model default` uses the 0.7 MB default one; `python -m sgrna.design.train_model` refits both) and prints the best 20-nt spacer with its PAM,
coordinates, predicted score and the number of NGG sites its 20-mer has in the genome (guides with more than one are set aside, or flagged when none is left).
`--linear` treats the sequence as linear, `--contig` picks a record of a multi-record FASTA, `.gz` is read directly.

What it has been checked for (`python -m sgrna.design.check_design`, `results/post_freeze/design_check.csv`): given only the genome and the coordinates of 600 curated guides (294 on the minus strand),
it reads the same protospacer, PAM and both flanks as the curated index in 600 of 600, builds the same 6,517 columns (largest difference 0.0) and the same predictions, and asking for each guide's cut site lists that guide at distance 0.
What picking by score buys (`python -m sgrna.design.design_benchmark`, `design_benchmark.csv`; out-of-fold predictions on the paper's whole-gene folds, the genome cut into disjoint windows):
in the 3,044 windows of 250 bp that hold at least four curated guides (5.07 on average) the tuned model's pick sits at the 0.771 percentile of the measured scores [0.760, 0.782] (a random pick: 0.5) and is the best guide of its window in 48.7% of them (chance 20.9%);
in 1,000 bp windows, 0.784 and 35.5% (chance 13.0%); the default model is 0.764 and 0.780.

Third-party terms for the files in `models/` (the lookup tables are Noshay et al.'s, MIT; the models were trained on CC BY-NC data) are in [`models/NOTICE.md`](models/NOTICE.md).

Limits: the score is SLICER's prediction of the *E. coli* screen's depletion score, a ranking aid (held-out rho about 0.73 in the paper) and not a probability of cutting; validated for SpCas9 in *E. coli* and one other bacterium, not for eukaryotes;
there is no off-target model (`seed_sites` is a coarse count of NGG sites sharing the 12 PAM-proximal nt); the benchmark compares guides of the curated library only.

## The family A table for any genome

The release (`family_A_v1.zip`) is the 348 flank columns (family A) of the 33,567 curated *E. coli* guides. `sgrna/design/family_a_table.py` makes the same table for any genome:
a FASTA in, one row per NGG guide (both strands, every record of at least 2,023 nt) out, with the release's 348 columns in the release's order after seven identifying ones
(`contig`, `left`, `right`, `strand`, `protospacer`, `pam`, `n_genome_copies`; `left` is the 1-based plus-strand start of the 20-mer and `protospacer` is the release's `sgRNAID`).
It needs only numpy and pandas and no project data (checked on a clean export of the tracked files; the lookup tables are `models/resources/qct_lookup_tables.json`, the file the release also holds):

```bash
git clone https://github.com/ieathamburgerandfries-stack/RSI-Crispr && cd RSI-Crispr/Runjia
pip install numpy pandas
PYTHONPATH=. python -m sgrna.design.family_a_table --genome genome.fasta --out family_A_genome.csv.gz
```

`--contig NAME [NAME ...]` keeps some records, `--start N --end M` one stretch of one record (a locus takes a fraction of a second), `--guides guides.csv` featurises only the guides listed
(columns `left`, `strand` and, for a multi-record FASTA, `contig`), `--linear` treats records as linear (default: circular, as a bacterial chromosome), `--tables` points at other lookup tables.
Flanks wrap around the origin when circular; when linear, the 223 guides of the *E. coli* genome within about 1 kb of an end use the DNA that exists (their windows are cut to it) and are computed one at a time.
`n_genome_copies` counts the NGG sites in the whole FASTA (all records, both strands) with the same 20-mer. The columns describe the DNA around a target, they are not a score: to score a guide, use the design tool above.

Checks (`python -m sgrna.design.check_family_a_table`, `results/post_freeze/family_a_table_check.csv`):
* Given only the coordinates of the 33,567 curated guides, it rebuilds the stored table: protospacer, PAM and genome copy count equal in 33,567 of 33,567, the same missing-value pattern, and 3 of 11,681,316 cells differ, each by one float32 step (at most 1.2e-7; the two programs add the same numbers in a different order).
* The *E. coli* genome has 541,810 NGG sites (270,137 on the plus strand, 271,673 on the minus, one per 8.6 nt), and every curated guide is among them; 3.2% share their 20-mer with another site. The whole table takes 58 s (0.11 ms per guide, featurising and writing) and is 246 MB as `.csv.gz`, 1.25 GB as plain CSV.
* A second genome, *Citrobacter rodentium* (5,346,659 bp, 707,896 sites, 76 s): 400 sites (the first 40 and last 40, which cross the origin, and 320 at random) are featurised again one at a time from their flanking sequence by `a_flank_fast`, the code the release was made with, and agree in all 348 columns (0 cells differ); the copy counts equal the design tool's.
* Linear mode on *E. coli*: 541,809 sites (one wraps the origin and is dropped); guides with a full margin of DNA give the same columns as when circular (largest difference 2.6e-11, floating point); the 33 flanks checked that have 50 to 249 nt of DNA all have the same 1,000 nt and 250 nt windows, as they must.

Validated for SpCas9 in *E. coli* (and the one other bacterium the paper tests); the columns say nothing about eukaryotic guides.

## Which document to read for what

| file | read it for |
|---|---|
| [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md) | **the claims**: seven findings with the evidence and limit of each; contributions (Part 2); limits (3); validation and the open-questions register (4); the field (5); controls and nulls (Appendix A); methods (B); data sources (C); versions, seeds, protocol (C0) |
| [`CODE_OVERVIEW.md`](CODE_OVERVIEW.md) | a tour of the ten blocks of code the claims depend on, for a reader who has not seen the project |
| [`CODE_MAP.md`](CODE_MAP.md) | every module: what it is for and which section of the report it supports (generated from the code, checked by the audit) |
| [`results/INDEX.md`](results/INDEX.md) | every result file grouped by the section it is evidence for, with the module that wrote it (generated) |
| [`results/RESULTS.md`](results/RESULTS.md) | claim to evidence: for each claim, the file that backs it and the command that produced it |
| [`results/RESULTS_archive_2026-10-10.md`](results/RESULTS_archive_2026-10-10.md) | the long-form commentary as it stood at the freeze, kept verbatim, nulls and tangents included |
| [`REMOVED.md`](REMOVED.md) | what was deleted or moved, what was brought in, why, and how to restore each item from git |
| [`EXTENSION_LOG.md`](EXTENSION_LOG.md) | the working record of the post-freeze experiments: what was tried, in what order, with what expectation |
| [`PAPER_FLOW.md`](PAPER_FLOW.md) | the paper outline: what each section must establish and which result and figure goes where; a flow, not a draft |
| `./run_audit.sh` | recomputes every post-freeze number the report quotes from the file that produced it; exits non-zero on any mismatch; also checks that every module imports and that importing writes nothing |
| `./run_ext.sh <module> [args]` | runs any module of `sgrna` with the environment it needs (see "Running it") |

## Two cohorts, one anchor

* **Published matrix:** Noshay et al.'s 13,880 guides by 6,232 columns, label `cut.score` (0 to 48, no negatives). It is the
  provenance of the representation (Finding 1) and of the harness check below; its iRF is the inherited model, not the comparator.
* **Curated cohort:** crisprHAL 2's curated 33,567 guides from the same screen, label the ALDEx2 score (−9.05 to +6.21, 47.5% negative).
  It is the working cohort, with crisprHAL 2 as the baseline to beat. The labels agree in rank at ρ 0.945 on the 8,617 guides both
  contain, but they are different quantities.

**Every number belongs to one of these frames and is not comparable across them.** The report and the result files say which.

**Harness anchor (published matrix):** XGBoost, top-300 features, 5 seeds × 5 folds: R² 0.2937 ± 0.0015, Spearman 0.5278, Pearson 0.5424 (`config.BASELINE`). Reproducing it to four decimals is how a change to the pipeline is checked. It is not a headline.

## Folder map

```
Runjia/                 the documents above, run_audit.sh, run_ext.sh
├── baseline_ecoli_sgRNA_model_swappable.py   five-seed baseline, MODEL_NAME switch (xgboost, lightgbm, catboost, random_forest, ridge);
│                                             selector stays XGBoost; for a local Colab runtime, edit DATA_PATH
├── sync_to_shared.sh   copies documents from the author's working folder into a nested shared repository; not used in this fork
├── sgrna/              the pipeline, organised by function (sgrna/README.md has the layout notes)
│   ├── config.py  io_utils.py  common.py   paths, constants, shared helpers; almost every module imports them
│   ├── build/       sequence to columns: published-matrix decoder, guide index, genome and QCT tables, feature matrix, array featurisers
│   ├── features/    the eleven families a_flank to j_periodicity, one module each, and the shape / 3D descriptors
│   ├── harness/     fit-and-score machinery every result goes through: run_ablation, folds, evaluate, notebook, CNN baseline
│   ├── tuning/      hyperparameter search, its replication on a second gene split, model mixing, untuned knobs
│   ├── cohorts/     ablations, ceilings and label variance on the curated and published cohorts
│   ├── controls/    nulls and audits: shifted flank, no-information, sealed holdout, selector audit, fast-featuriser equality
│   ├── interpret/   importance, attribution, faithfulness, redescription, the mechanism work
│   ├── compare/     every comparison with something else: crisprHAL 2, Noshay's iRF, other organisms, human, other assays
│   ├── cost/        wall-clock, CPU, memory and inference benchmarks
│   └── paper/       the audit (verify_numbers), figures, import and reference guards, the code-map generator
├── colab/           scripts that need a GPU or a fresh Colab instance: crisprHAL 2 re-tuning, wide window, sample complexity, inference timing
└── results/         named for when a file was made, not what it shows (INDEX.md gives that); the code addresses files by name; .npy caches are gitignored
    ├── *.csv        the analysis frozen on 11 October 2026 (Findings 1 to 6, Appendices A to C); not regenerated since
    ├── post_freeze/ everything measured after the freeze; figures/ holds the seven publication figures (PDF and 600 dpi PNG)
    └── rederived/   re-derivations of seven result files whose producing code was lost (sgrna/controls/rederive.py)
```

## Running it

**The data layout that is not in this repository.** `sgrna/config.py` resolves every path from a project folder holding:

```
<project>/
├── data/raw/        ecoli_feature_matrix.csv, human_feature_matrix.csv
├── data/reference/  NC_000913.2.fasta, genes_NC_000913.2.tsv, dinucleotide_shape_scales.tsv
├── data/interim/    caches the code writes (several GB): guide index, QCT tables, one CSV per family in features/
├── external_data/   crisprHAL, crisproff, GapR-seq, E_coli_analysis (Hi-C), RegulonDB, precise1k, CRISPRBact, C_rodentium, ...
└── results/         the pre-freeze result files
```

These inputs are several GB of other people's published data and are not committed (`external_data/README.md` in the project folder
records where each came from), so a clone imports but does not run end to end. `features/dinucleotide_shape_scales.tsv` is the one input that
ships with the code; `i_shape.py` reads it from `data/reference/`.

**From a terminal, with `./run_ext.sh`.** Three variables must be right together, and a wrong one fails with a `FileNotFoundError` that does
not say which: `RSI09_ROOT` (the project folder above; `config.py` guesses wrong from this fork's layout), `EXT_OUT` (`Runjia/results/post_freeze`)
and `PYTHONPATH` (includes `Runjia`). `run_ext.sh` sets all three, taking the folder above the fork as the project and
`<project>/.venv/bin/python` as interpreter unless `PY=...` is set:

```bash
./run_ext.sh sgrna.build.build_guide_index                              # step 0
./run_ext.sh sgrna.build.build_features --all                           # step 1
./run_ext.sh sgrna.harness.run_ablation --families a_flank --permute    # step 3
./run_ext.sh sgrna.build.curated_matrix --build                                 # featurise all 33,567 curated guides; any module, same pattern
```

**Colab on a local runtime.** The pipeline reads a 198 MB matrix and several GB of external data from local disk, so it runs on a local
runtime: Colab's interface on the computer's own CPU and filesystem, nothing uploaded.

```bash
pip install jupyter
jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 \
  --NotebookApp.port_retries=0 --NotebookApp.allow_credentials=True
```

Copy the printed `http://localhost:8888/?token=...` URL; in Colab choose **Connect**, **Connect to a local runtime** and paste it. Open a notebook
through that server from inside the project folder. The walkthrough notebooks are in the author's working copy, not in this
repository; against this layout a notebook starts with the cells below. `nb.setup()` puts `<project>/src` on the path, so if the package is
in `Runjia/` instead, start Jupyter with `PYTHONPATH` set to it. On hosted Colab, `nb.setup(root='/content/drive/MyDrive/RSI09')` works, but
the external data must be in Drive too, which is slow.

```python
import sgrna.harness.notebook as nb
nb.setup()                    # finds the project; installs pandas numpy scipy scikit-learn xgboost viennarna biopython if missing
nb.build_index()              # step 0, ~3 min once
nb.build_features("all")      # step 1, ~25 min once
nb.cache_matrix()             # step 2, ~1 min once (float32 .npy); nb.export_csv([...]) writes a merged CSV for notebooks that read DATA_PATH
nb.ablate(["c_folding"])      # step 3, ~2 min per family
```

`nb.status()` shows what exists; every step skips work already done. Package versions used for the report: Appendix C0.
* **Step 0, locate the guides.** The published matrix names guides by ID and stores no sequence. QCT descriptors are a pure function of the
  nucleotide at each position, so the protospacer is decoded from the columns and confirmed against NC_000913.2; the same step inverts the QCT
  columns into k-mer lookup tables. Writes `data/interim/guide_index.csv` and `qct_lookup_tables.json`. Evidence: Finding 1.
* **Step 1, build the families.** One module and one CSV per family, independent of each other; a family whose external data is missing is
  skipped with a message. No external dataset needed (`nb.build_features("self-contained")`): `a_flank`, `c_folding` (ViennaRNA), `d_mechanics`,
  `f_methylation`, `h_offtarget`, `i_shape`, `j_periodicity`. One needed: `b_energy` (CRISPRoff repository), `d_supercoiling` (GapR-seq,
  GSE152880), `e_nucleoid` (Hi-C, Lioy et al. 2018), `g_transcription` (RegulonDB, PRECISE-1K).
* **Step 3, ablate.** The frozen protocol: seeds 41 to 45, 5 folds, top-300 columns, 400-tree champion, with imputation, ranking and selection
  fitted inside the training split. Read `d_spearman` against the baseline in the same run, not a published number; a gain inside the permuted
  floor (`permute=True`) is not a gain; use grouped folds (`group=`). `selected_from_family` is context only, since permuted columns still get
  selected. Runs append to `results/ablation_<tag>.csv` and skip what is recorded, so a sweep survives a dropped session.

## Settings and caveats for anyone changing code

* **Importance is SHAP, not split gain.** `interpret/importance.shap_importance` is what `attribution.py` and `evaluate.py` call; both accept
  `--importance native` for the historical gain figures. Across seven model families gain-based rankings agree at ρ 0.05 per column against
  0.50 for SHAP, and gain credits a column for being binary. No gain plot belongs in a paper from here.
* **`config.SELECTOR`** picks the model that ranks columns before the champion is fitted. The default `"xgboost"` is the frozen baseline's
  choice; `"lightgbm"` is 3x faster and ranks as well, but moves the anchor from R² 0.2937 / ρ 0.5278 to 0.2897 / 0.5251. Exploratory sweeps only.
* **Two configurations, two libraries.** The 13,880-row anchor uses XGBoost as predictor, which reproduces Noshay et al.'s figure. The curated
  head-to-head modules default to LightGBM as predictor (`--model lightgbm`) with XGBoost still selecting, and the main hyperparameter search in
  `tuning/` ran on LightGBM. They agree within 0.001 wherever both were run, but a number from one is not a number from the other.
* **Grouped cross-validation is not optional for positional features.** About 20 guides sit in every gene, so plain `KFold` lets a model score by
  recognising a locus. Report anything positional (D2, E, G) under `arc5` or a blocked scheme (`run_ablation.GROUP_CHOICES`, `load_groups`).
* **A family's gain is an increment over a base, not a contribution.** Test each family alone, stacked and against permuted columns, or a gain
  may re-parameterise something already in the matrix (Appendix B3).
* **`cross_validate` resumes from `results/folds_<tag>.csv`.** A recorded fold is skipped, which lets a long run survive a dropped session and
  is also how a cached result was once replayed into a control arm that never ran. Give every arm a tag unique in every distinguishing parameter,
  not just its first word (`cui_screen._slug`).
* **Coordinates.** RegulonDB reports positions on U00096.3; the genome here, the GapR-seq tracks and the Bikard screen use U00096.2, up to ~2 kb
  apart. `g_transcription` re-locates each promoter by its 80-nt RegulonDB sequence instead of lifting over. Do not mix the two.
* **After moving or editing a module:** `git mv`, then `./run_ext.sh sgrna.paper.import_check`, `./run_ext.sh sgrna.paper.doc_refs` and
  `./run_audit.sh`, and regenerate the map with `./run_ext.sh sgrna.paper.code_map`. `CODE_MAP.md` and `results/INDEX.md` are generated; the
  audit fails if they omit a module or result file.

## Where the rest is

* **Family provenance** (why each was tried, what it carries and needs): report Appendix C, and B2 for why the imported families failed.
* **Known nulls** (essentiality, `d_supercoiling`, `i_shape`, `j_periodicity`, `f_methylation`, `h_offtarget`): Appendix A and C. Read why before
  spending another iteration on one.
* **Ablation tables** per family on the curated cohort, with their caveats: Appendix B3. **The sealed-gene drop** (not shown to be adaptive overfitting): Part 4.
