# RSI-Crispr

**Where the project stands (25 November 2026):** SLICER, the model in `Runjia/`, leads crisprHAL 2 — the published state of the art — on crisprHAL 2's **own two published splits**, both models as released:
Δρ **+0.0286** [+0.0239, +0.0331] on eSpCas9 and **+0.0226** [+0.0139, +0.0314] on SpCas9. Before the settings were tuned, no difference was detected (+0.0021 [−0.0031, +0.0072] on eSpCas9).
On whole-gene folds, with crisprHAL 2 retrained at its published settings, it reaches ρ 0.693 against 0.707 for SLICER's default model and 0.723 for the tuned model; the lead of the default model is
explained by the DNA beyond crisprHAL 2's 378-nt input, the tuned model's is not. Both were tuned with comparable compute, not equal draws (627 draws in 2.4 hours against 28 draws in about 3.0 GPU-hours).
On a quiet laptop CPU SLICER's default model scores a guide 15 to 25 times faster than crisprHAL 2 and 42 times faster than BoostMEC, fits 81 times faster than crisprHAL 2 on the same laptop (25 times
against a Colab T4), and is 0.68 MB; on a Colab T4 crisprHAL 2 costs 3.9 times SLICER's laptop path from raw contexts (two machines).

The result underneath it is not the score: sgRNA efficiency in *E. coli* depends on the **DNA surrounding the target**. Flank features add **+0.161** ρ (0.547 to 0.707; five-fold cross-validation that holds
whole genes out, three seeds, the default model), +0.156 on genes sealed before any choice was made. The signal is GC specifically (the purine and amino fractions give no gain), highest at a window
of 200 nt of 14 tried and within 0.01 of it from 150 to 500 nt, side-symmetric, and a step at the AT-rich end: the lowest decile of flank GC sits 3.16 label units below the rest. Repeats, mappability
and read depth do not account for it. Whether the effect needs a double-strand break was tried against dCas9 CRISPRi screens and **not settled** (the report's Appendix B5).
[`Runjia/PROGRESS_REPORT.md`](Runjia/PROGRESS_REPORT.md) is the full write-up, including the claims withdrawn on the way.

Predicting sgRNA cutting efficiency in *E. coli* CRISPR-Cas9 screens, extending
the quantum chemical tensor (QCT) feature set of
[Noshay et al. 2023, NAR 51:10147](https://doi.org/10.1093/nar/gkad736).

**Note on which data the numbers refer to.** The project pivoted on 13 October
2026 from Noshay's published 13,880-row matrix and its `cut.score` label to
crisprHAL 2's **curated 33,567-guide** release of the same Guo screen and its
ALDEx2 score, with **crisprHAL 2 as the baseline to beat** rather than Noshay's
iRF. The two labels agree in rank (ρ 0.945 on the 8,617 guides both contain) but
are different quantities, so any figure in this repo belongs to one frame or the
other. The shared baseline script below is still the harness check — reproducing
its five-seed CV to four decimal places is how a pipeline change is validated —
but it is no longer the comparator.

## Where to start

**`USE THIS!! FINAL_baseline_ecoli_sgRNA_model_5seeds.py`** is the shared
baseline every experiment is compared against: XGBoost, five seeds (41–45),
with feature selection fit on each training fold only so validation folds never
leak in. Trust the five-seed CV scores it reports; the final all-data model it
trains at the end is for inference, not a validation score.

It is written for Google Colab and reads the feature matrix from Google Drive.
Edit `DATA_PATH` near the top if your copy lives elsewhere.

## Layout

Each team member keeps their experiments in their own folder. Most experiment
files are named after what they add on top of the baseline, e.g.
`XGBoost + regional QCT summary feature set + 300 features`.

| folder | what's in it |
|---|---|
| `Jacky/` | gene-essentiality and features A–D experiments, grid search |
| `Joshua/` | QCT-derived feature experiments (regional, PAM-weighted, profile shape, coupling), seed-control tests, SHAP analysis |
| [`Runjia/`](Runjia) | [`PROGRESS_REPORT.md`](Runjia/PROGRESS_REPORT.md) — the whole project written up for an outside reader (mirrored from the author's working copy); [`sgrna/`](Runjia/sgrna) — the 130-module pipeline behind every number in it, organised by function (needs a data layout not in this repo, see its README); [`CODE_OVERVIEW.md`](Runjia/CODE_OVERVIEW.md) for a short tour of the code that matters and [`CODE_MAP.md`](Runjia/CODE_MAP.md) for every module; [`results/INDEX.md`](Runjia/results/INDEX.md) for the result files by topic; `colab/` for the GPU runs; and a swappable-model baseline for trying XGBoost / LightGBM / CatBoost / random forest / Ridge in Colab. [Folder guide](Runjia/README.md) |

## Not versioned

Model training outputs (`*_ecoli/`, `five_seed_champion_ecoli_*/`, `*.pkl`,
`*.npy`) are ignored — rerun the baseline to regenerate them. **Reference
genomes** (`*.gb`, `*.fasta`, `*.fa`) are ignored too: they are large, they are
other people's published data, and they are a download away. See
[`.gitignore`](.gitignore).
