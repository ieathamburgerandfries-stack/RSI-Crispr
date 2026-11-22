# Code overview: what the important code does, and why it matters

*For a reader who has not seen the project, such as a judge or reviewer. This documents the code in `Runjia/`; it is not paper text. It
covers the eleven blocks that the report's claims depend on (about a third of the modules). Every other module is a control, a null or an
appendix check, and is indexed in [`CODE_MAP.md`](CODE_MAP.md). Every number below appears in the project's own documents (a check in `./run_audit.sh` enforces that), and the numbers from after the freeze are
also re-computed from their result files by the same audit.*

## The project in one paragraph

A 2023 resource published 6,232 features for 13,880 *E. coli* CRISPR guides, but not the guide sequences, so it could not be used on a new target.
The code (1) decodes that matrix back into DNA, (2) finds that the DNA *around* the cut site, which Cas9 does not read, carries the largest
signal in the data, (3) attacks that finding with controls until it survives, (4) builds a small, fast model that matches and, once tuned, beats the
state-of-the-art predictor crisprHAL 2, (5) measures how much of the signal can be predicted at all, and (6) characterises what the signal is made of.

## Terms used

- **ρ** is the Spearman rank correlation between predicted and measured guide efficiency: 1 is a perfect ranking, 0 is none. Ranking is what a guide designer acts on.
- **The published matrix** is the 13,880-guide resource the project started from; **the curated cohort** is crisprHAL's 33,567-guide cleaned version of the same screen, which the models train on.
- **Flank** is the DNA on either side of the 20-nt guide target; **PAM** is the short motif Cas9 needs immediately after it.
- **An ablation** adds or removes one group of features and measures the change in ρ on identical folds. **Gene-grouped cross-validation** keeps all guides of a gene on the same side of the split.
- **Finding N** refers to the numbered findings in `PROGRESS_REPORT.md`.

## How the pieces fit

```
published matrix: 6,232 columns, no sequences
        |  1. decode                           sgrna/build
        v
20-mers, located on the genome (NC_000913.2)
        |  2. read the DNA around each cut     sgrna/features/a_flank.py
        v
+348 flank-composition columns  --->  3. leakage-safe cross-validation harness   sgrna/harness
                                          |
              4. controls (is it real?)   |   5. model vs crisprHAL 2
              6. speed   7. explanation   |   8. ceiling   9. what the signal is
              10. ideas that failed       |
                                          v
              11. an audit re-computes every number the report quotes
```

The pipeline is drawn as a figure in `results/post_freeze/figures/fig7_pipeline.png`.

## The eleven blocks

### 1. The decoder: an opaque matrix back into DNA *(Finding 1)*
`sgrna/build/qct.py`, `sgrna/build/decode_v_columns.py`, `sgrna/build/featurise.py`, `sgrna/build/build_guide_index.py`, `sgrna/build/genome.py`

The published matrix names each guide only by an ID. One per-position quantum-chemical column takes exactly four values across all guides, the
valence-electron counts of A, C, G and T, so the 20-mer can be read straight back out. 5,853 of the 5,887 anonymous binary `V` columns turn out to be
the one-hot (position, k-mer) indicators for k = 1 to 4. Every decoded guide is checked by finding it in the genome with an NGG PAM in the right place.
**Result:** 6,169 of 6,232 columns (99.0%) regenerate from a bare 20-mer, verified to 100.000000% over 1.17 million rebuilt values, and 13,879 of
13,880 guides are located. Any guide can now be featurised, and imported feature families that are functions of the guide could not have added information.

### 2. Family A: features from the DNA around the cut site *(Finding 2)*
`sgrna/features/a_flank.py` (the other ten families, B to J, are in the same folder; several show standalone gains, which the report explains as restatements of family A's composition, of the guide itself, or of gene dosage and strand, not as new information: Appendix B3 and Findings 3 and 8)

For each guide the code reads up to 1,000 nt either side, in the guide's orientation, and builds 348 columns: the identity of the nearest bases
on each side, the paper's own quantum-chemical tables extended past the guide, and window summaries (GC fraction, purine fraction, longest
homopolymer, mean and spread of four descriptor series) over the nearest 50, 250, 500 and 1,000 nt. The windows are nested, so each profile is computed once.
**Result: +0.164 ρ on the curated cohort**, +0.156 measured on genes no step of the pipeline had seen, and +0.080 on the published matrix. It is a smooth
compositional gradient, not a motif, and it is the largest effect in the project.

### 3. The harness: what every number is measured through *(throughout; Appendix B)*
`sgrna/harness/run_ablation.py`, `sgrna/cohorts/curated_ablation.py`

One rule governs it: **every fit happens inside the training fold**, including the imputation, the ranking of columns and the choice of the top 300 by
gain. An "ablation" is the same rows, folds and model with only the columns changed, so each difference is paired fold by fold. Folds are
grouped by gene (and in some runs by 100 kb block), so a position-based feature cannot score by recognising a locus. Five seeds by five folds; each fold is
written to disk as it finishes. Every claim in the report is a paired difference measured this way.

### 4. The controls: why +0.164 is not an artefact *(Finding 2; Appendix A)*
`sgrna/controls/shiftnull.py`, `sgrna/controls/sealed_holdout.py`, `sgrna/controls/selector_audit.py`, `sgrna/controls/floors.py`,
`sgrna/interpret/gcstrata.py`, `sgrna/post_freeze/human/fold_distance.py`

Each of these is an attack on the result, not a confirmation of it.
- **Circular-shift null.** Re-assign each guide the flanks of a position offset around the chromosome. Composition, smoothness and every column's distribution are preserved
  exactly; only the correspondence to the guide is broken. A single offset would be ambiguous, because chromosome-scale GC structure could survive a short one, so the code sweeps
  the offset and reads off how fast the gain decays: a 1 kb shift retains 8% and shifts of 10 kb or more retain none. The effect is local to about a kilobase.
- **Sealed-gene hold-out.** Re-derive every mechanisable choice on a development pool only, then score once on genes no step ever saw. The adaptive-overfitting component is **+0.0087**.
- **Permutation floor.** Shuffled flank rows (the harness's `--permute` arm, summarised by `floors.py`) score 0.5264 against a 0.5272 baseline, so the floor is zero.
- **Target-GC strata.** Conditioning on the target's own GC raises the effect (+0.044 to +0.082 on the published matrix), so it is not target GC in disguise.
- **Sequence distance.** The minimum Hamming distance between any validation and training guide is 3, with no identical guide across any split. The same statistic recovers a known
  duplicate-guide leak in the human matrix at 39.1%, which is the positive control.

### 5. The model against crisprHAL 2 *(Part 6, the head-to-head)*
`sgrna/compare/headtohead.py`, `sgrna/compare/final_eval.py`, `sgrna/post_freeze/tuning/tune_ecoli.py`, `sgrna/compare/vs_crisprhal.py`

The comparison is run under four conditions, because the answer depends on which is held equal. On our folds, with both models tuned here, SLICER leads by +0.0172. On crisprHAL 2's own splits with its shipped weights, the frozen model is
statistically equivalent (Δρ +0.0021, 95% CI [−0.0031, +0.0072]) at about 150× less CPU time to fit. On a different assay neither model was trained on (a boundary of the claim, not a like-for-like comparison), crisprHAL 2 leads by +0.026.
The tuning is designed so that it cannot fool itself: genes are split once into DEV (70%) and CONFIRM (30%), a random search runs on DEV only, and the winners are scored on genes never used to choose anything,
giving **+0.014 ρ**, which replicates on a second split. After tuning, the lead over crisprHAL 2 is **+0.023 to +0.029 ρ** on both of its published splits, measured against its shipped weights.

### 6. Speed: 1.41 ms, then 0.093 ms, then 0.036 ms per guide *(Part 6, the cost half of the head-to-head)*
`sgrna/post_freeze/speed/a_flank_fast.py`, `sgrna/post_freeze/speed/base_fast.py`, `sgrna/post_freeze/speed/flank_locus.py`, `sgrna/post_freeze/speed/e2e_bench.py`

The original featuriser walked every nucleotide in Python, which was 1.33 ms of the 1.41 ms it cost to score a guide. The array rewrite produces the same 348 columns, identical on all
33,567 guides. The per-locus version goes further. Neighbouring guides read almost the same DNA, so it scans a stretch of chromosome once into cumulative tracks; every window statistic then
becomes a subtraction, and the one column that is not a sum (the longest homopolymer run) becomes two lookups in a sparse table. It matches the rewrite to 4.6 × 10⁻¹². End to end the model scores a guide in
0.093 ms, and 0.036 ms when designing across a locus, which is **13 to 21×** and **34×** faster than crisprHAL 2.

### 7. Can the model be explained? Redundancy, importance and headroom *(Findings 4 and 5)*
`sgrna/interpret/importance.py`, `sgrna/interpret/importance_models.py`, `sgrna/interpret/faithfulness.py`, `sgrna/interpret/representation.py`

Because 99% of the columns restate the same 20-mer, which column a model credits is close to arbitrary. Across model families the library-default importance agrees between models at only **0.050** per column;
SHAP agrees at 0.496 (0.800 per block), so the project reports SHAP, at block level only. `representation.py` shows that the 412 portable columns reproduce the full 6,232 (a difference of
−0.00003 ρ), and that the model is **out of material**: no column it was given correlates with its residuals above 0.028, while family A, held back, correlated at 0.147. The test flagged family A *before
family A existed*, which is its positive control.

### 8. The ceiling: how much can be predicted at all *(Finding 6)*
`sgrna/cohorts/curated_ceiling.py`, `sgrna/post_freeze/signal/ceiling_table.py`

crisprHAL publishes its per-replicate counts. Pairing each Cas9 replicate with its matched dead-nuclease control gives two independent estimates of the same quantity, which agree at ρ 0.8097;
Spearman–Brown gives the released score's reliability (0.8948), and its square root is the highest ρ any predictor can reach: **0.946** [0.944, 0.948]. The field's "~0.7 upper limit" drops both the square
root and the averaging step; corrected, its own numbers give 0.91 to 0.93, so models at ρ ≈ 0.70 are at about 75 to 77% of what is attainable, not at the top. `ceiling_table.py` tabulates the correction
for any replicate agreement.

### 9. What the signal is made of *(Finding 7, after the freeze)*
`sgrna/post_freeze/signal/mechanism.py`, `sgrna/post_freeze/signal/novel_followups.py`

Because model importance is unreliable here (block 7), nothing in this block reads importance off the model. The code uses partial correlations computed from sequence and label directly, a resolution ablation,
and a depth-2 model whose splits are the pairs. **Results:** the effect is GC, not purine; it is side-symmetric (the two sides agree to within 0.026); it has a single length scale of 250 to 500 nt with no structure
finer than about 250 nt, so four columns (GC and purine over the whole kilobase on each side) carry about three quarters of it; and it interacts with the PAM-proximal seed. Separately, the guide's own position-by-base preferences are uncorrelated between
*E. coli* and human (ρ = −0.052), and 15 of 27 jointly significant cells invert.

### 10. Ideas that were tested and failed, and why that counts *(Appendix B4)*
`sgrna/post_freeze/shape/geom.py`, `sgrna/post_freeze/shape/shape_ablate.py`, `sgrna/post_freeze/shape/local_sequence_bound.py`, `sgrna/post_freeze/signal/lit_followups.py`

DNA shape and 3D structure were tested in the form an earlier null had left open: vectors with direction, including a rigid-body reconstruction of the helical path from the six step parameters. They add −0.0008 to +0.0006 ρ over family A,
and their standalone gains are composition (shuffling the flank keeps them). `local_sequence_bound.py` closes the question for any local shape predictor by handing the model the inner ±50 nt directly: it adds
+0.0002 ρ. Two methods taken from the published literature, a mixed-effect model and a
binarised classifier, were also null, each with a mechanism. The shape null was re-run at larger model capacity and against shuffled-flank controls before being believed.

### 11. The audit: how every quoted number is checked *(throughout)*
`sgrna/post_freeze/paper/verify_numbers.py`, `sgrna/post_freeze/paper/figures.py`, `sgrna/post_freeze/paper/import_check.py`, `sgrna/post_freeze/paper/doc_refs.py`, `sgrna/post_freeze/paper/code_map.py`

`verify_numbers.py` recomputes each number the report quotes from the result file that produced it and compares at the precision quoted; it exits non-zero on any mismatch (more than nine hundred checks). It has repeatedly caught the authors' own
transcription and rounding errors. Four smaller guards sit beside it: every module imports and importing writes nothing (`import_check.py`); every module, path and script a
document names exists (`doc_refs.py`); every module is documented (`code_map.py`); and the numbers in this overview all appear in the project's documents. `figures.py` builds the seven publication
figures to ACS specification from the result files.

## What a judge can run

These read the stored results and the genome, so they need no GPU and no re-training.

```bash
./run_audit.sh                                          # re-compute every number the report quotes from its file (about 2 seconds)
./run_ext.sh sgrna.post_freeze.speed.a_flank_fast       # the fast featuriser equals the original on 300 guides (under 1 second)
./run_ext.sh sgrna.post_freeze.signal.ceiling_table     # the ceiling correction, as arithmetic (about 1 second)
```

## What this code does not show

- The lead over crisprHAL 2 is against the model **as shipped**. It was never retuned on its own splits (about 40 GPU-hours), so the honest claim is that a tuned boosted tree beats the released model, not its architecture.
- No untouched external cutting dataset remains, so the adaptive-overfitting component is measured at +0.0087 rather than eliminated.
- Whether the flank effect needs a double-strand break is **not** established. dCas9 CRISPRi screens were tried and could not separate cutting from binding (Appendix B5); their code is not in this tour, and no claim rests on it.
- One organism and two relatives; the human arm was stopped, and its rules are uncorrelated with *E. coli*'s.
- The physical reading of the signal (a bulk property over a few hundred bases) is a hypothesis, not a result.
- A design tool built on this model is not yet built. Block 6 is the groundwork for it.
