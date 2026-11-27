# Code overview: what the important code does, and why it matters

*For a reader who has not seen the project, such as a judge or reviewer. This documents the code in `Runjia/`; it is not paper text. It
covers the ten blocks that the report's claims depend on (about a third of the modules). Every other module is a control, a null or an
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
              4. controls (is it real?)   |   5. model vs crisprHAL 2 (accuracy and cost)
              6. explanation  7. ceiling |   8. what the signal is  9. ideas that failed
                                          v
              10. an audit re-computes every number the report quotes
```

The pipeline is drawn as a figure in `results/post_freeze/figures/fig7_pipeline.png`.

## The ten blocks

### 1. The decoder: an opaque matrix back into DNA *(Finding 1)*
`sgrna/build/qct.py`, `sgrna/build/decode_v_columns.py`, `sgrna/build/featurise.py`, `sgrna/build/build_guide_index.py`, `sgrna/build/genome.py`

The published matrix names each guide only by an ID. One per-position quantum-chemical column takes exactly four values across all guides, the
valence-electron counts of A, C, G and T, so the 20-mer can be read straight back out. 5,853 of the 5,887 anonymous binary `V` columns turn out to be
the one-hot (position, k-mer) indicators for k = 1 to 4. Every decoded guide is checked by finding it in the genome with an NGG PAM in the right place.
**Result:** 6,169 of 6,232 columns (99.0%) regenerate from a bare 20-mer, verified to 100.000000% of the 3.08 million values rebuilt for 500 guides and of the 85.6 million over all 13,880 guides, and 13,879 of
13,880 guides are located. Any guide can now be featurised, and imported feature families that are functions of the guide could not have added information.

### 2. Family A: features from the DNA around the cut site *(Finding 2)*
`sgrna/features/a_flank.py` (the other ten families, B to J, are in the same folder; several show standalone gains, which the report explains as restatements of family A's composition, of the guide itself, or of gene dosage and strand, not as new information: Appendix B3 and Findings 3 and 8)

For each guide the code reads up to 1,000 nt either side, in the guide's orientation, and builds 348 columns: the identity of the nearest bases
on each side, the paper's own quantum-chemical tables extended past the guide, and window summaries (GC fraction, purine fraction, longest
homopolymer, mean and spread of four descriptor series) over the nearest 50, 250, 500 and 1,000 nt. The windows are nested, so each profile is computed once.
**Result: +0.161 ρ on the curated cohort** (whole genes held out, three seeds; +0.164 on random rows), +0.156 measured on genes no step of the pipeline had seen, and +0.080 on the published matrix. It is a smooth
compositional gradient, not a motif, and it is the largest effect in the project.

### 3. The harness: what every number is measured through *(throughout; Appendix B)*
`sgrna/harness/run_ablation.py`, `sgrna/cohorts/curated_ablation.py`

One rule governs it: **every fit happens inside the training fold**, including the imputation, the ranking of columns and the choice of the top 300 by
gain. An "ablation" is the same rows, folds and model with only the columns changed, so each difference is paired fold by fold. Folds are
grouped by gene (and in some runs by 100 kb block), so a position-based feature cannot score by recognising a locus. Five seeds by five folds; each fold is
written to disk as it finishes. Every claim in the report is a paired difference measured this way.

### 4. The controls: why +0.161 is not an artefact *(Finding 2; Appendix A)*
`sgrna/controls/shiftnull.py`, `sgrna/controls/sealed_holdout.py`, `sgrna/controls/selector_audit.py`, `sgrna/controls/floors.py`,
`sgrna/interpret/gcstrata.py`, `sgrna/controls/fold_distance.py`, `sgrna/compare/sealed_organism.py`

Each of these is an attack on the result, not a confirmation of it.
- **Circular-shift null.** Re-assign each guide the flanks of a position offset around the chromosome. Composition, smoothness and every column's distribution are preserved
  exactly; only the correspondence to the guide is broken. A single offset would be ambiguous, because chromosome-scale GC structure could survive a short one, so the code sweeps
  the offset and reads off how fast the gain decays: a 1 kb shift retains 8% and shifts of 10 kb or more retain none. The effect is local to about a kilobase.
- **Sealed-gene hold-out.** Re-derive every mechanisable choice on a development pool only, then score once on genes no step ever saw. The sealed-gene drop in family A's gain is **+0.0087**, small and not shown to be adaptive overfitting (whole-gene folds inside the pool remove only +0.0007 of it).
- **Permutation floor.** Shuffled flank rows (the harness's `--permute` arm, summarised by `floors.py`) score 0.5264 against a 0.5272 baseline, so the floor is zero.
- **Target-GC strata.** Conditioning on the target's own GC raises the effect (+0.044 to +0.082 on the published matrix), so it is not target GC in disguise.
- **Sequence distance.** The minimum Hamming distance between any validation and training guide is 3, with no identical guide across any split. The same statistic recovers a known
  duplicate-guide leak in the human matrix at 39.1%, which is the positive control.
- **A sealed organism.** The tuned settings were chosen on *E. coli* genes; trained on all of *E. coli* and scored zero-shot on the *C. rodentium* screen, which no choice ever saw, the tuned model still beats the frozen champion by +0.0172 (0.7160 against 0.6986).

### 5. The model against crisprHAL 2: accuracy, then cost *(Part 5, the head-to-head)*
`sgrna/compare/headtohead.py`, `sgrna/compare/final_eval.py`, `sgrna/tuning/tune_ecoli.py`, `sgrna/compare/vs_crisprhal.py`; for the cost half, `sgrna/build/a_flank_fast.py`, `sgrna/build/base_fast.py`, `sgrna/build/flank_locus.py`, `sgrna/cost/e2e_bench.py`

**Accuracy.** The final SLICER model, as shipped, leads crisprHAL 2, as released, by **+0.023 to +0.029 ρ** on both of crisprHAL 2's own splits. The comparison is run under four conditions, because the answer depends on which is held equal. On our folds, with both models tuned here, SLICER leads by +0.0172. On a different assay neither model was trained on (a boundary of the claim, not a like-for-like comparison), crisprHAL 2 leads by +0.026.
The tuning is designed so that it cannot fool itself: genes are split once into DEV (70%) and CONFIRM (30%), a random search runs on DEV only, and the winners are scored on genes never used to choose anything,
giving **+0.014 ρ**, which replicates on a second split. Before tuning, no difference was detected against crisprHAL 2's released weights (Δρ +0.0021, 95% CI [−0.0031, +0.0072]), and the model fits 81 times faster on the same laptop.

**Cost: 1.41 ms, then 0.093 ms, then 0.036 ms per guide.** The original featuriser walked every nucleotide in Python, which was 1.33 ms of the 1.41 ms it cost to score a guide. The array rewrite produces the same 348 columns, identical on all
33,567 guides. The per-locus version goes further. Neighbouring guides read almost the same DNA, so it scans a stretch of chromosome once into cumulative tracks; every window statistic then
becomes a subtraction, and the one column that is not a sum (the longest homopolymer run) becomes two lookups in a sparse table. It matches the rewrite to 4.6 × 10⁻¹². End to end the model scores a guide in
0.093 ms, and 0.036 ms when designing across a locus, which is 15 to 25 times faster than crisprHAL 2 on one CPU thread for the default model, and faster again when designing across a locus (on a Colab T4, crisprHAL 2 costs 3.9 times SLICER's laptop path from raw contexts, on two machines: `EXTENSION_LOG.md` section 33).

### 6. Can the model be explained? Redundancy, importance and headroom *(Findings 4 and 5)*
`sgrna/interpret/importance.py`, `sgrna/interpret/importance_models.py`, `sgrna/interpret/faithfulness.py`, `sgrna/interpret/representation.py`, `sgrna/interpret/compact_model.py`, `sgrna/interpret/vs_crisprhal_interpret.py`, `sgrna/interpret/hal_readout.py`

Because 99% of the columns restate the same 20-mer, which column a model credits is close to arbitrary. Across model families the library-default importance agrees between models at only **0.050** per column;
SHAP agrees at 0.496 (0.800 per block), so the project reports SHAP, at block level only. `representation.py` shows that the 412 portable columns reproduce the full 6,232 (a difference of
−0.00003 ρ), and that the model is **out of material**: no column it was given correlates with its residuals above 0.028, while family A, held back, correlated at 0.147. The test flagged family A *before
family A existed*, which is its positive control. `compact_model.py` then removes the redundancy on purpose: on 80 guide indicators plus 24 flank composition columns (88% of the full model's ρ) SHAP agrees between model families at 0.87 per column, against 0.50 on the full matrix, and reads out the model-free design rules, so the instability is the redundancy and not the models. `vs_crisprhal_interpret.py` (its crisprHAL 2 half, `hal_readout.py`, runs in the TensorFlow environment) then puts SLICER and crisprHAL 2's released network through the same four tests, with the predictions written first: SLICER's attributions are the more faithful (sensitivity-*n* 0.96–0.97 against 0.86–0.88), and on stability under refitting, agreement with the model-free map and compressibility the two are tied or crisprHAL 2 is ahead, so SLICER is interpretable and is not shown to be more interpretable than crisprHAL 2.

### 7. The ceiling: how much can be predicted at all *(Finding 6)*
`sgrna/cohorts/curated_ceiling.py`, `sgrna/cohorts/ceiling_table.py`

crisprHAL publishes its per-replicate counts. Pairing each Cas9 replicate with its matched dead-nuclease control gives two independent estimates of the same quantity, which agree at ρ 0.8097;
Spearman–Brown gives the released score's reliability (0.8948), and its square root is the highest ρ any predictor can reach: **0.946** [0.944, 0.947]. The field's "~0.7 upper limit" drops both the square
root and the averaging step; corrected, its own numbers give 0.91 to 0.93, so models at ρ ≈ 0.70 are at about 75 to 77% of what is attainable, not at the top. `ceiling_table.py` tabulates the correction
for any replicate agreement.

### 8. What the signal is made of *(Finding 7, after the freeze)*
`sgrna/interpret/mechanism.py`, `sgrna/interpret/novel_followups.py`, `sgrna/interpret/hypotheses.py`

Because model importance is unreliable here (block 6), nothing in this block reads importance off the model. The code uses partial correlations computed from sequence and label directly, a resolution ablation,
and a depth-2 model whose splits are the pairs. **Results:** the effect is GC, not purine; it is side-symmetric (the two sides agree to within 0.026); it has a single length scale of 250 to 500 nt with no structure
finer than about 250 nt, so four columns (GC and purine over the whole kilobase on each side) carry about three quarters of it; and it interacts with the PAM-proximal seed. Separately, the guide's own position-by-base preferences are uncorrelated between
*E. coli* and human (ρ = −0.052), and 15 of 27 jointly significant cells invert. `eukaryote_rules.py` repeats the comparison on eleven public datasets from human, mouse and zebrafish: the inversion is **bacteria against eukaryotes** (mouse and human maps agree at +0.56, *E. coli*'s agree with none of them), and a model trained on human guides keeps about 70% of its accuracy on mouse guides.

`hypotheses.py` then asks what the *response* to that composition looks like, with each prediction written into the file before it was run. Of three, two held (the flank's effect is steeper beside a favourable seed base, and PAM density does not carry it) and one failed, which was the informative one: pooled across the genome the response to flank GC is not a slope but a **step at the AT-rich end** (the lowest tenth of guides, below 44.4% GC, sit 3.10 label units lower, and nothing varies above it), with a gentler graded component inside genes. The report turns this into four hypotheses with the test that would settle each; the leading one is that nucleoid proteins shield AT-rich DNA, which needs an occupancy track that is not on disk.

### 9. Ideas that were tested and failed, and why that counts *(Appendix B4)*
`sgrna/features/geom.py`, `sgrna/cohorts/shape_ablate.py`, `sgrna/cohorts/tuned_ablation.py` (`--design local_sequence`), `sgrna/compare/lit_followups.py`

DNA shape and 3D structure were tested in the form an earlier null had left open: vectors with direction, including a rigid-body reconstruction of the helical path from the six step parameters. They add −0.0008 to +0.0006 ρ over family A,
and their standalone gains are composition (shuffling the flank keeps them). `tuned_ablation.py --design local_sequence` closes the question for any local shape predictor by handing the model the inner ±50 nt directly: it adds
+0.0002 ρ. Two methods taken from the published literature, a mixed-effect model and a
binarised classifier, were also null, each with a mechanism. The shape null was re-run at larger model capacity and against shuffled-flank controls before being believed.

### 10. The audit: how every quoted number is checked *(throughout)*
`sgrna/paper/verify_numbers.py`, `sgrna/paper/figures.py`, `sgrna/paper/import_check.py`, `sgrna/paper/doc_refs.py`, `sgrna/paper/code_map.py`

`verify_numbers.py` recomputes each number the report quotes from the result file that produced it and compares at the precision quoted; it exits non-zero on any mismatch (more than nine hundred checks). It has repeatedly caught the authors' own
transcription and rounding errors. Four smaller guards sit beside it: every module imports and importing writes nothing (`import_check.py`); every module, path and script a
document names exists (`doc_refs.py`); every module is documented (`code_map.py`); and the numbers in this overview all appear in the project's documents. `figures.py` builds the seven publication
figures to ACS specification from the result files.

## How the modules are organised

Each module answers one question. A variation of that question (another family, another fold design, another window set, another split) is a flag of the same module, not another file. The
flank-reach questions show the pattern: `flank_scale` is the one module for them, with the parts `window` (symmetric windows), `partition` (which pair of base classes), `resolution`, `displacement`
(reading the flank from a shifted site) and `asymmetric` (the upstream and downstream windows chosen separately), and it reproduces the stored result of the module it replaced byte for byte.
A reader looking for a control should look for its question (block 4 and `CODE_MAP.md`) and not for the name of a variation. An analysis of the whole package found twenty-six clusters of modules that run one
pipeline with different arm tables, folds or seeds; two modules are merged and the plan for the rest, with the result files each must keep reproducing, is in `EXTENSION_LOG.md` section 33.

## What a judge can run

These read the stored results and the genome, so they need no GPU and no re-training.

```bash
./run_audit.sh                                          # re-compute every number the report quotes from its file (about 2 seconds)
./run_ext.sh sgrna.build.a_flank_fast       # the fast featuriser equals the original on 300 guides (under 1 second)
./run_ext.sh sgrna.cohorts.ceiling_table     # the ceiling correction, as arithmetic (about 1 second)
```

## What this code does not show

- The lead over crisprHAL 2 is a statement about the two models **as shipped**, which is how both are used in practice. Whether boosted trees beat its architecture under equal tuning is a separate question that has not been tested (the Colab script `colab/crisprhal_retune_own_splits.py` is written, not run).
- No untouched external cutting dataset remains at the same loci. The sealed-gene drop is small (+0.0087) and not shown to be adaptive overfitting, and the tuned settings' benefit survives a genome no choice ever saw (+0.0172); what cannot be undone is the choice of cohort and of family A, both made on *E. coli*.
- Whether the flank effect needs a double-strand break is **not** established. dCas9 CRISPRi screens were tried and could not separate cutting from binding (Appendix B5); their code is not in this tour, and no claim rests on it.
- One organism and two relatives; the human arm was stopped, and its rules are uncorrelated with *E. coli*'s.
- The physical reading of the signal (a bulk property over a few hundred bases) is a hypothesis, not a result.
- A design tool built on this model is not yet built. The cost half of block 5 is the groundwork for it.
