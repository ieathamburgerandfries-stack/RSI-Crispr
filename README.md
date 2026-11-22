# RSI-Crispr

**Where the project stands (22 November 2026):** SLICER, the model in `Runjia/`, now **leads** crisprHAL 2 — the published
state of the art — on crisprHAL 2's **own two published splits**, scored against its released weights: Δρ **+0.0286**
[+0.0239, +0.0331] on eSpCas9 and **+0.0226** [+0.0139, +0.0314] on SpCas9, after a joint hyperparameter search worth
+0.014 ρ on genes it never saw. Before that tuning the same model was statistically equivalent to crisprHAL 2 (Δρ
+0.0021, 95% CI [−0.0031, +0.0072] on eSpCas9, about 150× less CPU time to fit). It scores a guide 13–21× faster end to
end (34× when designing across a locus) and reaches crisprHAL 2's accuracy on about a third of the labelled guides.

**Both models are compared as released**, which is how they are used in practice: SLICER as shipped, with settings chosen on
held-out genes, against crisprHAL 2's published weights. That is a claim about the two models, not about their architectures:
SLICER's settings were searched and crisprHAL 2's were not, and a Colab script that gives crisprHAL 2 a search of its own is
written and not yet run.
On our own curated five folds the margin is larger (0.7218 against 0.7051, paired +0.0161, five of five folds) but the two
models were not searched with equal effort there, so that is not the headline; and on a screen neither model trained on
(a dCas9 CRISPRi screen) **theirs** transfers better.

The result underneath it is not the score: sgRNA efficiency in *E. coli* depends on the **DNA surrounding the target**,
out to roughly 500 nt, as a smooth compositional gradient rather than a motif. It is carried by GC specifically, is
side-symmetric, has one length scale of 250–500 nt, and is coupled to the PAM-proximal seed — which is why a
gradient-boosted model on hand-built summaries keeps up with a CNN–BiGRU. Whether the effect needs a double-strand break
was tried against dCas9 CRISPRi screens and **not settled** (the report's Appendix B5). [`Runjia/PROGRESS_REPORT.md`](Runjia/PROGRESS_REPORT.md)
is the full write-up, including the claims withdrawn on the way.

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
