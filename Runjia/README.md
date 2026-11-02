# RSI09 — Enhancing sgRNA efficiency in CRISPR-Cas9 genome editing

Feature engineering for predicting sgRNA cutting efficiency in *E. coli*,
extending the quantum chemical tensor (QCT) feature set of
[Noshay et al. 2023, NAR 51:10147](https://doi.org/10.1093/nar/gkad736).

**The project pivoted cohorts on 13 October 2026.** Everything is now measured on
crisprHAL 2's **curated 33,567-guide** release of the same Guo screen, with
**crisprHAL 2 as the baseline to beat**; Noshay et al.'s 13,880-row published
matrix and its iRF are provenance, not the comparator. Both frames appear below
and **every number belongs to one of them** — they are not interchangeable, least
of all the labels (`cut.score` runs 0 to 48 with no negatives; the curated score
runs −9.05 to +6.21 and is 47.5% negative, agreeing with it in rank at ρ 0.945 on
the 8,617 guides both contain). What the pivot re-opened, and what is closed
again, is tabulated in Part 4 of the progress report.

**Harness anchor (published frame):** XGBoost, top-300 features, 5 seeds × 5
folds → R² 0.2937 ± 0.0015, Spearman 0.5278, Pearson 0.5424 on Noshay's 13,880
guides — kept because reproducing it to four decimal places is how this pipeline
is checked, not because it is the current baseline.

**Working figure (curated frame):** base + family A, ρ **0.7218** on the curated
five folds; **0.6985** on crisprHAL 2's own held-out partition against the
**0.695** Browne et al. report for it.

**Against the state of the art: parity at a fraction of the cost.** On
crisprHAL 2's **own two published splits**, evaluated against their own published
weights, SLICER is statistically equivalent to it — Δρ **+0.0021**, 95% CI
[−0.0031, +0.0072] on eSpCas9 — equivalence at any margin from 0.0072 up, with no
margin needing to be declared — and +0.005 on
SpCas9 — while reaching a given accuracy on about **half the labelled guides**.
Our five folds take **69 s of wall-clock** against **105 min** for their
CNN–BiGRU on the same machine — **92×**, or **~150×** counted in CPU-seconds
(65.8 s per fold against ~10,000), which is the figure to lead with because they
parallelise better than we do and wall-clock understates it. Comparable memory
either way (3.76 GB against 4.55 GB peak), and ~190× with a LightGBM selector
(`results/RESULTS_archive_2026-10-10.md` §15a).

**And against the model this project extends, on its own data, it is not close.**
Our reimplementation of Noshay et al.'s iRF scores ρ **0.4785** on their 13,880
guides (reproducing their published Pearson 0.502 / R² 0.249 at 0.488 / 0.235);
SLICER on the identical rows and label reaches **0.6060** — **+0.128**, of which
+0.049 is the model class and **+0.078 is the flank features**. So the two
comparisons say different things and both are real: Noshay's model is beaten
decisively, crisprHAL 2 is matched.

**On size**, crisprHAL 2 holds **790,533** trainable parameters across 37 layers;
SLICER stores **6,348 leaf values** over 400 trees, or 18,244 numbers counting
each node's feature index and threshold — 125× fewer fitted values, the same
order as the compute ratio.

On **our own** curated five folds we lead by more — base + family A reach
**0.7218** against their **0.7051** at their best known configuration, paired
**+0.0161** over 33,567 out-of-fold predictions, five of five folds — but that
comparison is not symmetric in search effort and the report says so. A nested
search of their own (free Colab T4, 16 October) found **nothing better than their
published configuration**, which won four of five inner searches. And on a screen
neither model trained on, **their** model transfers better (\|ρ\| 0.0867 against
0.0609). The claim that survives every condition is parity plus data efficiency,
not a better model. Details and all four conditions: Finding 7 of
[`docs/PROGRESS_REPORT.md`](PROGRESS_REPORT.md).

(The source paper this extends reports Pearson 0.502 / R² 0.249 on the same
feature set.)

**The effect is specific to cleavage, and the label's ceiling is now measured.**
In a dCas9 **CRISPRi** screen — the last external dataset never used for
anything here — family A adds **−0.009**, against **+0.157** on the cutting
screen under the same whole-gene-holdout protocol. That screen-wide figure is an
average over a cohort 85% of which is non-essential genes, where knockdown barely
moves growth; restricted to its essential genes family A reads **+0.268**, and
that +0.268 is the screen's own strand rule — a single strand indicator scores
ρ 0.662 there on its own, and given it family A adds **+0.007**. So the
flanking-context effect belongs to a step that requires a break, not to Cas9
binding or local accessibility, and the strand result is the positive control the
comparison previously lacked. Separately, the ceiling
for the label we train on is **0.946 [0.944, 0.948]**, derived from crisprHAL's
own published per-replicate counts rather than inherited from a different label,
leaving ~0.22 ρ unaccounted for.

**It is the DNA, not the enzyme — and it is what makes a model portable.**
Transfer, one variable at a time: *transferred* is an imported model with no
retraining, *local* is the same architecture trained on the target, and the last
column is how much of the transferred ρ the flank features supply.

| changed | from → to | transferred | local | retained | of which family A |
|---|---|---:|---:|---:|---:|
| nuclease | SpCas9 → eSpCas9 | 0.689 | 0.784 | 88% | +0.098 |
| nuclease + organism | *E. coli* → *C. rodentium* | 0.698 | 0.766 | 91% | +0.072 |
| the same, reversed | *C. rodentium* → *E. coli* | 0.628 | 0.708 | 89% | **+0.150** |
| kingdom | *E. coli* → human | −0.039 | 0.508 | 0% | — |
| assay | cleavage → dCas9 silencing | 0.061 | 0.458 | 13% | — |

**Changing the nuclease or the host costs about a tenth of the performance;
changing the kingdom or the assay costs all of it.** In the hardest bacterial
direction — a model imported from a 236 kb *C. rodentium* screen into the
*E. coli* genome — the flanks carry +0.150 of the 0.628 that arrives. Decomposed
further, it is the **long-range windows** that do it (+0.107 of the +0.150), so
the gradient holds in both organisms even though *C. rodentium*'s own screen is
too narrow for a correlation to reveal it
(`results/RESULTS_archive_2026-10-10.md` §15).

**And where it stops.** Carried to Noshay et al.'s human matrix — 6,216 shared
columns, so no change of representation — transfer is **−0.039
[−0.054, −0.024]** going to human and **−0.006 [−0.023, +0.011]**, which is zero,
coming back: useless in both directions, with the human model fitted to the
human matrix rather than to *E. coli*'s hyperparameters. That distinction is
worth the extra run — the human signal is far less additive than *E. coli*'s
(ridge reaches 64% of boosting's ρ against 96%), so our champion is the wrong
shape for it by 0.10 ρ. The strand-invasion mechanism behind the effect also
inverts (GC vs cut score: −0.20 in *E. coli*, +0.02 in human). Every claim here
is a bacterial one, and the boundary is measured rather than assumed.

The contribution is not the prediction, though. It is decoding the published
matrix so it can be computed for any guide, showing which feature families
*could not* have helped and why, and finding that the surrounding-DNA effect is
a **smooth compositional gradient rather than a motif** — which is why the cheap
model keeps up. Full assessment, written for someone outside the project, in
[`docs/PROGRESS_REPORT.md`](PROGRESS_REPORT.md).

Everything in `src/` exists to answer one question: *what moves `cut.score`
that the quantum descriptors do not already capture?* The reasoning and the
literature behind each family is in
[`lit_review/beyond_QCT_literature_review.md`](lit_review/beyond_QCT_literature_review.md);
the first-pass answer is in [`results/RESULTS.md`](results/RESULTS.md).

**Ablation, 5 seeds** (each family alone against a baseline reproduced to four
decimal places). `arc` holds out one of five contiguous chromosome arcs, which
is the honest test for anything positional. Full tables and caveats in
[`results/RESULTS.md`](results/RESULTS.md).

| family | ΔR² random-row | ΔR² arc CV |
|---|---:|---:|
| A `a_flank` — sequence + QCT outside the 20-mer | **+0.087** | **+0.087** |
| D2 `d_supercoiling` — chromosomal read density | — | +0.046 |
| D1 `d_mechanics` — duplex thermodynamics | +0.020 | — |
| C `c_folding` | +0.007 | — |
| B `b_energy` — CRISPRoff ΔG | +0.007 | — |
| F `f_methylation` | +0.004 | — |
| H `h_offtarget` | +0.000 | — |
| I `i_shape` — flank bendability / stiffness | +0.038* | — |
| **A + D2 + D1 together** | — | **+0.105** |

Under 100 kb-blocked CV: `e_nucleoid` +0.017, `g_transcription` +0.016.
Permutation floors run −0.002 to +0.001.

**All eleven families have now been re-measured on the 33,567-row curated
cohort** (`python -m sgrna.curated_ablation --all`,
[`results/ablation_curated.csv`](results/ablation_curated.csv)). The ranking
transfers almost exactly — rank ρ 0.97 against the published table — while every
family inflates, median 3.3×. **The inflation is not noise removal and not the
base representation**: measured directly, the per-unit effect of flank GC on the
score is **2.96× steeper** in the guides crisprHAL adds, and splitting those by
host gene shows the weak relationship belongs to Noshay's 1,222-gene panel rather
than to the particular guides in it (β +5.4 inside the panel, +17.0 outside).
Guo's panel is a part of the genome where the flank effect is weak, so +0.080 is
the right figure for that panel and +0.164 the right one genome-wide — neither
is an attenuated version of the other (`python -m sgrna.curation_variance`).

Family A +0.164 (**+0.156 on sealed genes** — see the adaptive-overfitting
caveat below), `d_supercoiling` +0.139, `i_shape`
+0.120, `d_mechanics` +0.099, `g_transcription` +0.058 ρ. Appendix B3 of the
progress report has the caveats, which matter: these are single-family
increments over a base starved of dense continuous columns, not estimates of
biological contribution.

*Family I is starred because the standalone number misleads: added on top of
family A it is worth **+0.0007**, so the physical shape of the flanks is a
reparameterisation of their letter composition rather than new information.

Two findings that change how these read: the `d_supercoiling` gain is **DNA
read density, not supercoiling** (the untagged no-antibody control predicts as
well as the ChIP), and the positional families are **one variable measured
three ways**. [`results/RESULTS.md`](results/RESULTS.md) has the evidence, and the shared
repository's git history is the record of what was believed when.

**The 13,880-row cap is lifted.** 5,853 of the 5,887 opaque `V####` columns
were identified as (position, k-mer) indicators (100% verification), so with
the recovered QCT tables **99% of the published feature matrix regenerates from
a 20 nt protospacer** — any bacterial Cas9 screen can now be featurised the
same way. `python -m sgrna.expand --build` does it for all 33,567 curated Guo
guides.

**Four improvement levers, measured not assumed:** a cleaner label +0.003,
double the rows +0.010 (saturating at ~20k), a larger feature cap +0.001,
wider flanks saturate at 250–500 nt. Only sequence context outside the 20-mer
paid, at +0.080.

**All eleven families together on the curated cohort: ρ 0.7475, Δ +0.2030**
(`python -m sgrna.curated_joint`,
[`results/curated_joint.csv`](results/curated_joint.csv)). Adding up the eleven
single-family gains would claim **+0.7046** — so the families realise **29%** of
their own sum and the family table must never be read as a budget. The overlap is
not uniform: `a_flank+d_supercoiling` realises 62% of its pair sum, and the three
guide-only families are the most additive at **70%**, because they meet only
through the 20-mer. On the older 13,880-row cohort the comparable figure was nine
families at ρ 0.641, +0.114 over the published baseline (random-row CV; the
honest contiguous-arc figure for A + D2 + D1 is ρ 0.615).

**Why six of the eight families could not have helped much.** The published
matrix turns out to be a *lossless* encoding of the 20 nt protospacer — every
position carries at least three of its four mononucleotide indicators — so any
feature that is a deterministic function of the guide adds **zero information**.
Measured, by linear regression on position-specific k-mer indicators: CRISPRoff
ΔG is 98% recoverable from the 20-mer, nearest-neighbour thermodynamics 85%.

**Zero information is not zero performance, and the distinction is the whole
point.** Re-describing the guide in better coordinates is worth **+0.011 to
+0.019** ρ to a depth-limited tree, on **two** independent guide-only
measurements (`b_energy`'s and `c_folding`'s shares of it are the same spacer
fold energy filed in two folders, Pearson 0.9993) — small, real, and not
nothing: Noshay's 316 quantum columns beat every k-mer
encoding of the same 20 bases by **+0.0114**, and of the 63 hand-named summary
columns the 46 computable from the guide carry **+0.0065** while the 17 that need
annotation carry **−0.0007**. So the families sort by **reach**, not by a
delivered/undelivered line: **nothing confined to the 20-mer clears +0.019, and
nothing worth more than +0.05 is confined to it.** That ordering is about what
the model cannot already compute, not about what the phenotype depends on — set
the increments beside each family's strongest *marginal* association and the
latter is nearly flat, 0.12 to 0.18, with the single strongest column in the
whole matrix being a guide-determined hybridisation energy at 0.184 (report,
Part 2, *Further inferences about the process*). Three more measurements close off the alternatives — the
champion has no feature left with residual correlation above 0.035, sixteen
model classes span only 0.116 Spearman, and a CNN reading raw bases *ties* the
engineered matrix on the guide (ρ 0.498 vs 0.527) while losing badly on the
flanks (+0.019 vs +0.080). Two independent screens of the same library agree at
ρ 0.810, putting the ceiling for **Noshay's** label at ~0.90, so the assay is not
the constraint either. (The curated crisprHAL label we now train on has its own,
higher ceiling of 0.946 — measured, not inherited; the two numbers belong to two
different quantities and are not interchangeable.)
`results/RESULTS_archive_2026-10-10.md` §13 has the full argument; `python -m sgrna.diagnose
--what all` reproduces it.

**Model class, measured on identical folds and identical selected features.**
Eight predictors, 3 seeds × 5 folds (`results/model_comparison.csv`):

| model | R² | ρ | top-decile precision | pick percentile |
|---|---:|---:|---:|---:|
| **LightGBM** | **0.382** | **0.609** | 0.261 | **72.1** |
| XGBoost | 0.381 | 0.608 | 0.260 | 72.1 |
| CatBoost | 0.362 | 0.601 | **0.268** | 72.0 |
| Ridge | 0.341 | 0.577 | 0.247 | 71.2 |
| random forest | 0.292 | 0.550 | 0.236 | 70.2 |

Model choice spans 0.09 R² but only **1.8 percentile points on the guide you
actually pick**, and a linear model gets within 0.03 ρ of boosting — so most of
the signal is additive. On the interpretability side
(`results/interpretability.csv`), **XGBoost's two importance methods rank its
top features independently (ρ −0.02)** while LightGBM's agree (+0.44) at the
same accuracy and 1.5× the selection stability. **The champion is therefore
LightGBM**; the frozen XGBoost numbers above stay as the historical baseline.
`results/RESULTS_archive_2026-10-10.md` §11–12 defines every metric and gives the full tables.

---

## Which document to read

| file | what it is for |
|---|---|
| [`docs/PROGRESS_REPORT.md`](PROGRESS_REPORT.md) | **start here** — the whole project for an outside reader: the seven findings and the evidence for each, how it compares to published work, what is original rather than imported, where it stops, and the controls in the appendices, with a glossary and a guide to reading the statistics |
| [`results/RESULTS.md`](results/RESULTS.md) | **claim → evidence index**: for every claim the report makes, the file that backs it and the command that produced it, including the thirteen files whose producer is no longer in the repo |
| [`results/RESULTS_archive_2026-10-10.md`](results/RESULTS_archive_2026-10-10.md) | the long-form commentary as it stood on 10 October 2026, kept verbatim — nulls, tangents and all |
| `docs/qna_archive.md` | *(local only)* questions asked, answered and closed without moving a number — kept for conference Q&A, deliberately not in the report |
| `docs/FINDINGS_LOG.md` | *(local only)* the chronological record — every result as it was first believed, every withdrawal where it happened, every bug. Git history is the authority; this is the reading copy |
| [`lit_review/beyond_QCT_literature_review.md`](lit_review/beyond_QCT_literature_review.md) | why each feature family was tried in the first place |
| `local/MISCONCEPTIONS.md` | *(local only, gitignored)* every claim we withdrew, why it was wrong and how it was caught |

---

## Folder map

```
RSI09/
├── data/
│   ├── raw/            ecoli_feature_matrix.csv  (Noshay Supp. Table 2, untouched)
│   ├── reference/      NC_000913.2 genome + gene coordinates
│   ├── interim/        guide index, QCT lookup tables, cached arrays
│   │   └── features/   one CSV per feature family
│   └── processed/      merged model-ready matrices (only if you export them)
├── external_data/      downloaded datasets — see its own README
├── lit_review/         papers + the review that motivates each family
├── src/sgrna/          the pipeline (imported by the notebooks)
├── results/            335 result files + RESULTS.md (claim → evidence index)
├── notebooks/          ← start here
│   ├── RSI09_pipeline.ipynb    the full walkthrough, step by step
│   └── RSI09_quickstart.ipynb  three cells, if you just want it running
├── docs/               PROGRESS_REPORT.md + qna_archive.md, FINDINGS_LOG.md (local)
├── colab/              scripts that need a GPU — run on free Colab, not here
├── archive/            earlier one-off scripts, kept for reference
├── sync_to_shared.sh   publishes the documents, code and cited results below
└── RSI-Crispr/         the team's shared git repo
    └── Runjia/         **a build artefact, not a second copy to maintain.**
                        Edit the sources above and run `./sync_to_shared.sh`;
                        `--check` reports drift without changing anything
```

**On there being two of some files.** The shared repo sits *inside* this folder,
so anything teammates read has to exist in both places. Those pairs are
generated, not maintained: `sync_to_shared.sh` is the only thing that should
ever write to `RSI-Crispr/Runjia/`, and it rewrites the one link that differs
between the two locations (this README's pointer to the report, which is in
`docs/` here and a sibling there). Four things are deliberately *not* published:
`docs/qna_archive.md`, `docs/FINDINGS_LOG.md`, the 19 MB Colab input, and the
255 result files no claim cites.

---

## Running it — Colab on a local runtime

The pipeline reads a 198 MB matrix and several GB of external data from your own
disk, so it runs on a **local runtime**: Colab's interface, your computer's CPU
and filesystem. Nothing is uploaded.

In a terminal:

```bash
pip install jupyter
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0 \
  --NotebookApp.allow_credentials=True
```

Copy the `http://localhost:8888/?token=…` URL it prints. In Colab, click the
**Connect** dropdown → **Connect to a local runtime** → paste it. Then open
`notebooks/RSI09_pipeline.ipynb` through that local Jupyter server, so its
working directory is inside the project and the setup cell finds everything
with no paths to edit.

The whole pipeline is four calls:

```python
import sgrna.notebook as nb
nb.setup()                    # find the project, install what is missing
nb.build_index()              # ~3 min once: locate all 13,880 guides
nb.build_features("all")      # ~25 min once: all eleven families
nb.cache_matrix()             # ~1 min once
nb.ablate(["c_folding"])      # ~2 min per family
```

`nb.status()` shows what has been built already; every step skips work it has
done before. Requirements (`pandas numpy scipy scikit-learn xgboost viennarna
biopython`) are installed by `nb.setup()`.

> **Hosted Colab instead?** Put the folder in Drive and call
> `nb.setup(root='/content/drive/MyDrive/RSI09')`. It works, but the external
> datasets have to live in Drive too, which is slow.

> **Terminal instead?** Every step also has a CLI: `python -m
> sgrna.build_guide_index`, `python -m sgrna.build_features --all`,
> `python -m sgrna.run_ablation --families a_flank --permute`, and so on.

---

## The pipeline, step by step

### Step 0 — `nb.build_index()`

The published matrix identifies each guide only as `aaeAb3241_107_Cas9`. It
never stores the sequence, and without a genomic coordinate five of the eight
feature families are impossible. This step recovers both.

The trick is that the QCT descriptors are a pure function of the nucleotide at
each position. `p{i}monomer.No.electronsraw` takes exactly four values across
all 13,880 guides — 42, 48, 50, 56 — which are the **valence-electron counts**
of cytosine, thymine, adenine and guanine. So the protospacer can be read
straight back out of the matrix, one position at a time.

Every decoded guide is then confirmed against the genome rather than trusted:

| check | result |
|---|---|
| decoded from QCT columns | 13,879 / 13,880 (one row has a missing descriptor) |
| found in NC_000913.2 | 13,879 |
| NGG PAM immediately 3′ | 13,879 |
| single-copy in the genome | 13,877 |
| GC-skew origin vs annotated *oriC* | 3,923,620 vs ~3,923,800 |

The same step inverts every QCT column into a k-mer → value lookup table. All
4 monomers, 16 dimers, 64 trimers and 256 tetramers appear in the data, with
no conflicts, so **the paper's quantum parameterisation is recovered exactly**
and can be applied to sequence the paper never encoded.

Outputs: `data/interim/guide_index.csv`, `data/interim/qct_lookup_tables.json`.

### Step 1 — `nb.build_features("all")`

| Family | Module | Adds | Needs |
|---|---|---|---|
| A | `a_flank` | sequence **and QCT tensors** outside the 20-mer | genome only |
| B | `b_energy` | CRISPRoff ΔG decomposition (ΔG_H, ΔG_O, ΔG_U, ΔG_B) | crisproff repo |
| C | `c_folding` | spacer MFE, spacer:scaffold duplex, seed accessibility | ViennaRNA |
| D1 | `d_mechanics` | nearest-neighbour duplex stability, melting profile | genome only |
| D2 | `d_supercoiling` | GapR-seq positive-supercoiling density | GSE152880 |
| E | `e_nucleoid` | Hi-C 3D crowding, NAP dependence | Lioy et al. 2018 |
| F | `f_methylation` | Dam `GATC` / Dcm `CCWGG` at PAM and seed | genome only |
| H | `h_offtarget` | target copy number, near-cognate burden | genome only |
| G | `g_transcription` | TSS distance, template strand, expression | RegulonDB + PRECISE-1K |

Families are independent. One whose external dataset is missing is skipped
with a message; it never takes the run down. Build a subset with
`nb.build_features(["a_flank", "c_folding"])`, or
`nb.build_features("self-contained")` for the ones that need no external data.

### Step 2 — `nb.cache_matrix()`

Caches the 198 MB published matrix as a float32 `.npy` so later steps load in
seconds instead of a minute. `nb.export_csv([...])` writes a merged CSV in the
original shape (title row included) if you want to point one of the team's
existing notebooks at it by changing `DATA_PATH`.

### Step 3 — `nb.ablate(...)`

Reproduces the frozen protocol — seeds 41–45, 5 folds, top-300 by XGBoost
gain, 400-tree champion — with imputation, ranking and selection all fitted
**inside the training split**. A validation fold that helped choose the
features is not a validation fold.

```python
nb.ablate(["a_flank", "b_energy", "c_folding"], permute=True)
nb.ablate(["d_supercoiling"], group="bin100k", tag="grouped")
nb.summary("notebook")
```

Four things decide whether a family earned its place:

- **`d_spearman`** — change against the baseline in the same run, not against
  the number in the paper.
- **`permute=True`** — the same columns with their rows shuffled.
  This is the noise floor. A gain inside the permuted band is not a gain.
- **`group="bin100k"`** — hold out whole 100 kb blocks of chromosome instead
  of random rows. The library puts ~20 guides in every gene and neighbouring
  genes share supercoiling domains, so a positional feature can identify the
  locus and get rewarded for memorising its mean score. **Any positional
  family (D2, E, G) must be reported grouped.** In this project's first pass
  that cost `e_nucleoid` 42% of its apparent gain and `g_transcription` 36%,
  while `a_flank` and `d_supercoiling` survived nearly intact.
- **`selected_from_family`** — how many of the family's columns survived into
  the top 300. Useful context, but on its own it proves nothing: 24.7
  *permuted* `a_flank` columns still got selected.

Runs append to `results/ablation_<tag>.csv` and skip experiments already
recorded, so a sweep can be built up one cell at a time and survives a
dropped Colab session — re-running the cell picks up where it stopped.

---

## Where each family came from

Short version; the full argument and citations are in the review.

- **A** is first because it is the largest documented gain for bacterial Cas9
  (+0.143 Spearman on this exact screen, Browne et al. 2025) and the cheapest
  to obtain. Extending the recovered QCT tables into the flanks turns the
  project's claim from *quantum descriptors of the protospacer* into *quantum
  descriptors of the target locus*.
- **B** and **C** are the two established biophysical families. B is computed
  by importing the CRISPRoff authors' own pipeline and parameters, with only
  the folding call redirected to the ViennaRNA API for speed. C folds the
  transcript E. coli actually makes — spacer **plus scaffold**, which is the
  usual reason this family gets wrongly written off.
- **D2** and **E** address the "unknown chromosomal factors" Guo et al. named
  but could not explain. Neither had been tested against bacterial sgRNA
  efficiency before, which made them the highest-risk, highest-novelty part
  of the project. D2 pays (+0.139 on the curated cohort) but **not for the stated
  reason**: the untagged no-antibody control predicts as well as the GapR ChIP, so
  what the family measures is DNA read density, not supercoiling.
- **F** is a regex with a mechanism attached, and two 2026 papers behind it.
- **G** was deliberately constrained on the reasoning that Guo et al. had already
  reported the marginal effects as null, so only the interaction terms were worth
  arguing about. **Measured, that is backwards** — the interactions add +0.0004
  over their parents and the marginals carry the family's +0.058, almost all of it
  gene geometry rather than transcription (see "Known nulls" below).
- **H** is not about Cas9 at all — it is about the half of `cut.score` that
  measures whether the cell survived the cut.

## Known nulls — don't spend another iteration here

Guo et al. tested **gene essentiality**, **chromosome position** and **target
gene expression** against this exact label and reported no meaningful
correlation. The Keio essentiality experiment in `RSI-Crispr/Jacky/` is
therefore a replication of a published null, which is worth a paragraph in the
paper rather than another round of tuning.

Measured here on the curated cohort, that null splits three ways
(`python -m sgrna.txn_interactions`,
[`results/txn_interactions.csv`](results/txn_interactions.csv)): **essentiality
is exactly nothing** (+0.00004 as a lone column, confirming Guo), **expression is
not** (+0.0213), and the largest part of `g_transcription`'s +0.058 is **gene
geometry** — distance and orientation to the nearest gene and promoter, worth
+0.0393 with every expression column removed. The family's own stated rationale,
that transcription matters through *interactions* with local DNA, is the part that
fails: the four interaction terms add **+0.0004** over their parents. So
`g_transcription` is kept, and read as a positional family rather than a
transcriptional one.

## Three caveats worth writing into the methods

1. **Coordinate systems.** RegulonDB reports positions on U00096.3; the genome
   here, the GapR-seq wiggles and the Bikard screen all use U00096.2, and the
   two differ by up to ~2 kb. Rather than lift over, `g_transcription`
   re-locates each promoter by searching for the 80 nt sequence RegulonDB
   ships with it and reading the TSS off the single upper-case base — exact,
   offline, and self-checking (3,964 promoters anchor successfully).

2. **Collinearity with QCT.** Nearest-neighbour thermodynamics and hydrogen-
   bond/stacking tensors encode overlapping physics. Test every family both
   alone and stacked on the baseline, or a reported gain may just be a
   re-parameterisation of something already in the matrix.

3. **Adaptive overfitting, and the size of it.** Many choices here were made
   against the same curated five folds, so those figures carry an inflation
   component that no amount of cross-validation removes. The analysis is frozen
   as of 11 October 2026, and the component is **measured**: sealing 20% of genes
   by whole gene, re-deriving selection and hyperparameters inside the
   development pool only, and differencing against a reference arm chosen
   elsewhere puts the automatable part at **+0.0087 ρ [+0.0020, +0.0153]** over
   **fifteen** sealing draws — 12 of 15 positive, t = 2.56, p = 0.022
   (`python -m sgrna.sealed_holdout --seal-seeds ...`). **It lands on the
   differences, not the levels**: absolute scores do not move detectably
   (0.7043 in pool CV against 0.7077 sealed), while **family A's gain reads
   +0.164 in cross-validation and +0.156 on genes the pipeline never saw**.
   Treat that as a *lower* bound, since no re-derivation reaches the human
   choices, and expect the margin over crisprHAL 2 on our own folds to carry it
   too — our configuration was chosen on those rows and theirs was not, so it
   does not cancel. An earlier three-draw version reported +0.0005 ± 0.016 and
   "no inflation detectable"; three draws could not resolve a 0.009 effect
   against a reference arm whose own gap swings ±0.015.
