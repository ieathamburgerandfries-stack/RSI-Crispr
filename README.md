# RSI-Crispr

**Where the project stands:** SLICER, the model in `Runjia/`, reaches **parity**
with crisprHAL 2 — the published state of the art — on crisprHAL 2's **own two
published splits**, scored against their own released weights: Δρ **+0.0021**,
95% CI [−0.0031, +0.0072] on eSpCas9, inside a pre-declared ±0.01 equivalence
margin, and +0.005 on SpCas9. It gets there for about **150× less CPU time** and
on about **half the labelled guides**.

On *our* curated five folds the margin is larger — **0.7218** against **0.7051**,
paired +0.0161 over 33,567 out-of-fold predictions, five of five folds — but the
two models were not searched with equal effort there, so that figure is not the
headline. A nested search of their own found nothing better than their published
configuration, and on a screen neither model trained on, **theirs** transfers
better. **The claim that survives every condition is parity plus cost and data
efficiency, not a better model.**

The result underneath it is not the score: sgRNA efficiency in *E. coli* depends
on the **DNA surrounding the target**, out to roughly 500 nt, as a smooth
compositional gradient rather than a motif — which is both why a gradient-boosted
model on hand-built summaries keeps up with a CNN–BiGRU, and why it belongs to
the cleavage step rather than to Cas9 binding (it disappears in a dCas9 CRISPRi
screen). [`Runjia/PROGRESS_REPORT.md`](Runjia/PROGRESS_REPORT.md) is the full
write-up, including the ten claims withdrawn on the way to these.

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
| [`Runjia/`](Runjia) | [`PROGRESS_REPORT.md`](Runjia/PROGRESS_REPORT.md) — the whole project written up for an outside reader (synced copy; edit it at the source); [`sgrna/`](Runjia/sgrna) — the 58-module pipeline behind every number in it (needs a data layout not in this repo, see its README); and a swappable-model baseline for trying XGBoost / LightGBM / CatBoost / random forest / Ridge in Colab. [Folder guide](Runjia/README.md) |

## Not versioned

Model training outputs (`*_ecoli/`, `five_seed_champion_ecoli_*/`, `*.pkl`,
`*.npy`) are ignored — rerun the baseline to regenerate them. **Reference
genomes** (`*.gb`, `*.fasta`, `*.fa`) are ignored too: they are large, they are
other people's published data, and they are a download away. See
[`.gitignore`](.gitignore).
