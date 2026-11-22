# SLICER — what determines whether a CRISPR guide cuts, in *E. coli*

**S**imple **L**ightweight **I**nterpretable **C**RISPR **E**fficiency **R**anker. The analysis is frozen as of 11 October 2026 (Part 4).

## Summary

Seven claims, each defended in the finding or part named beside it.

1. **A published feature matrix is a re-encoding of the guide.** 99% of its 6,232 columns regenerate from the 20-mer and spell about twenty variables; 412 columns computable for any guide match all 6,232 to 0.00003 ρ. No deterministic function of the 20-mer can add information to it (Finding 1).
2. **The DNA around the cut site predicts cutting, and it is the one large effect.** Flank composition within about a kilobase adds **+0.164 ρ** on the curated cohort and **+0.156 on genes sealed from every choice**. It is local (8% survives a 1 kb shift, none survives 10 kb), is not mappability or gene ranking, and carries across nuclease and organism with 88–91% retained and no retraining (Findings 2–3).
3. **The signal is GC over a few hundred bases, coupled to the seed.** It is GC and not composition in general, side-symmetric, with one length scale of 250–500 nt. Pooled across the genome the response is a step at the AT-rich end with a gentler graded component inside genes. The flank's slope is steeper beside a favourable seed base. The guide's own position × base rules invert between *E. coli* and eukaryotes (G at position 20 is disfavoured here and favoured in human, mouse and zebrafish), while mouse and human rules agree (Finding 7).
4. **The model is out of material, and its per-column importances are not stable across fits.** No column it was given correlates with its residuals above 0.028. On a redundant matrix, per-column importance and drop-and-refit faithfulness are properties of the matrix, while SHAP at block level and flank-edit tests are faithful to the fitted model, so the model can be interpreted at the block level (Findings 4–5).
5. **The label's ceiling is 0.946 [0.944, 0.948]**; the best model reaches 0.7314, so about 0.21 ρ of repeatable variance is unexplained. The field's own ceiling is mis-derived: the same replicate agreement gives 0.91–0.93, not "about 0.7" (Finding 6; Part 5).
6. **The final SLICER model, as shipped, leads crisprHAL 2, as released, on crisprHAL 2's own splits**: by **+0.0286** [+0.0239, +0.0331] on eSpCas9 and **+0.0226** [+0.0139, +0.0314] on SpCas9. It scores a guide 13–21× faster end to end and reaches crisprHAL 2's accuracy with about a third of the labelled guides. Before tuning, the same model was statistically equivalent (Δρ +0.0021, 95% CI [−0.0031, +0.0072] on eSpCas9) at about 150× less CPU time to fit. Both models are compared as used in practice, which is as released; whether the architecture itself is better is a separate method-level question that is not tested (Part 5).
7. **Adaptive overfitting is measured and small.** Sealing genes gives +0.0087 ρ on arm-to-arm differences, and the tuned settings' benefit survives a genome no choice ever saw (+0.0172 over the frozen champion on *C. rodentium*) (Part 4).

## How to read this document

* **Part 1** is the seven findings, each a claim followed by the evidence and its limit. **Part 2** states the contributions and what is new about the biology. **Part 3** says where the work stops. **Part 4** is the validation: the freeze, the audit of what could have inflated the figures, and the sealed checks. **Part 5** places the work against the published predictors.
* **Appendices.** A: every control, with what it rules out and what would be believed without it, and the inventory of nulls. B: the methods and controls behind the findings. B2–B4: the imported families that did not work, what curation is, the family table on the curated cohort, and the null on the flank's shape and 3D structure. B5: a boundary experiment on dCas9 screens that no claim rests on. C0 and C: reproducibility and data sources. D: references.
* **Two cohorts.** Noshay et al.'s published 13,880-row matrix (label `cut.score`) and crisprHAL 2's curated 33,567-guide release of the same screen (ALDEx2 score) are different frames; every number belongs to one of them and the text says which. The working cohort is the curated one.
* **Provenance.** [`results/INDEX.md`](results/INDEX.md) lists every result file by the section it supports and [`results/RESULTS.md`](results/RESULTS.md) goes from each claim to its file and command. [`CODE_OVERVIEW.md`](CODE_OVERVIEW.md) tours the code the claims depend on and [`CODE_MAP.md`](CODE_MAP.md) maps every module. `./run_audit.sh` recomputes every post-freeze number quoted here from the file that produced it and exits non-zero on any mismatch (1027 checks as of 22 November), and checks that the package imports without writing anything. `EXTENSION_LOG.md` is the working record: what was tried, in what order, and with what expectation.

## What earns a place, and what the work is for

A statement is included only if it is insightful, verified, and not at serious risk of retraction, and only if it makes a claim in Parts 1–3 or defends one. A null result is a finding only if it shows something that had been missed; otherwise it is an appendix entry. A hypothesis is included only if the findings support it and it can be falsified.

The *E. coli* work is scaffolding for a human workflow. Its results divide into **method**, which travels (grouped cross-validation by gene and block, the texture-preserving null, replicate-derived ceilings, supplying an assay's dominant covariate before reading a subgroup, the negative results about tuning, selection and label weighting), and ***E. coli* biology**, which does not (the size and shape of the flanking effect, the GC signs, the dosage result, the mechanism constraints). Where the biology has been tested elsewhere it inverted: target GC against activity is −0.20 here and +0.02 in human.

---

## Part 1 — Findings

### 1. The published representation is a re-encoding of the guide, and holds about twenty variables

**6,169 of the 6,232 columns (99.0%) regenerate from any 20 nt protospacer, so the matrix holds
about twenty variables** (a 20-mer carries at most 40 bits). Regeneration agrees exactly,
**100.000000%** over 1.17 million rebuilt values, and 13,879 of 13,880 guides were located in
NC_000913.2 with the required PAM in the right place (13,877 unique). The matrix partitions with
nothing left over (`portability_audit.csv`): **5,853** (position, k-mer) indicators (5,853 of 5,887
anonymous `V####` columns, none ambiguous), **316** quantum-chemical tables (exact k-mer lookups,
no conflicts) and **63** hand-named summaries. *Limit:* the 63 are outside the regenerated set; 46
are computable from the 20-mer and 17 are not (below).

**Seventy-four mononucleotide columns recover 96.4% of full-matrix ρ on the published cohort (95.4%
on the curated one); higher-order k-mers add almost nothing.** No selection in any arm
(`curated_ladder.py`, 3 seeds × 5 folds; family A adds 348 flank columns):

| arm (`curated_ladder.csv`) | columns | ρ | vs full base | ρ with family A | vs full base |
|---|---:|---:|---:|---:|---:|
| mononucleotide only | 74 | 0.5198 ± 0.0021 | −0.0251 | — | — |
| + dinucleotide | 364 | 0.5331 ± 0.0010 | −0.0118 | 0.6937 ± 0.0010 | −0.0143 |
| + trimer | 1,510 | 0.5354 ± 0.0003 | −0.0096 | 0.7011 ± 0.0009 | −0.0068 |
| + tetramer (**all 5,853 k-mers**) | 5,853 | 0.5335 ± 0.0013 | −0.0114 | 0.7005 ± 0.0004 | −0.0074 |
| **+ 316 quantum columns (the base)** | **6,169** | **0.5449 ± 0.0013** | — | **0.7079 ± 0.0004** | — |

Trimers add **+0.0023** and tetramers then **−0.0019** (−0.0006 with family A): 4,343 tetramer
indicators are too sparse to carry anything, as in Finding 4 (+0.0035 and −0.0028 on the published
matrix). The one step that pays is the last: the 316 quantum columns, pure lookups that add no
information, are worth **+0.0114** over a complete k-mer encoding (+0.0074 with family A).

**Only 2 of the 63 hand-named columns need anything beyond the guide, and they carry none of the
gain.** The 46 computable from the 20-mer are 20 whole-guide k-mer counts, guide GC, Tm and RNA
structure, 4 PAM indicators and 19 of the 34 undecoded `V` columns; the 17 that are not are the
two annotation distances (nearest gene, PAM) and 15 undecoded `V` columns. Published matrix, no
selection (`handnamed_split.py`, `handnamed_split.csv`): 364 sequence indicators give ρ 0.5189 ±
0.0010; adding the 46 computable gives **0.5254 ± 0.0013 (+0.0065)**, adding the 17 that are not
gives 0.5182 ± 0.0009 (**−0.0007**), and adding all 63 (427 columns) gives 0.5256 ± 0.0018 (+0.0066).
The 427-column result is therefore not evidence of non-sequence information, and annotation helps
only at the locus scale, where `d_supercoiling`, `e_nucleoid` and `g_transcription` are worth +0.145
between them as one variable (Appendix B3).

**The matrix is 74 sequence columns, 63 hand-named columns and ~6,095 restatements, and the
restatement is worth about +0.015 to a depth-limited tree.** Independent families agree: quantum
tables **+0.0114**, `b_energy` (CRISPRoff on the 23-mer) **+0.019**, `c_folding` (ViennaRNA on the
spacer and its scaffold) **+0.014**, `d_mechanics`'s protospacer half **+0.011**, the hand-named 46
**+0.0065**. The gain belongs to the particular functions, not to re-description as a class and not
to the tree's difficulty in computing them. On the curated base, **random** position-weight tables
(4, 16, 64, 256; three draws each) give −0.0004, −0.0004, −0.0002 and −0.0015, and random
**non-linear** functions of order 2 to 8, matched on out-of-fold recoverability to the real fold
energy (R² 0.316), give +0.0001 to +0.0007. What pays is tracking the phenotype:
`eng.dg.guide_selffold`, fully determined by the 20-mer, has marginal \|ρ\| 0.102 and is worth
**+0.0116** alone, whereas a random function of the same difficulty has no association and buys
nothing (Appendix B). Selection is not the cause: the full `d_mechanics` arm gives **+0.0993** with
every column against **+0.0988** selecting 300 (`results/selector_control.csv`).

The size must be measured on columns that are functions of the guide, and `d_mechanics` is only
partly one: its eight `eng.mech.flank.{up,dn}{25,100}` columns are thermodynamics of flanking DNA
the base does not contain (out-of-fold R² ≈ 0), while nine ΔG/ΔH/ΔS sums reconstruct at
**R² = 1.0000** (Appendix A). Against the base (6,169 columns, ρ 0.5445; `no_information.py`,
`no_information.csv`, same 5 × 5 protocol) the 22 **protospacer** columns add **+0.0110**, the 8
**flank** columns **+0.0973** and all 30 +0.0991, so eight flank columns carry 98% of that arm's
gain. *Limit:* a deterministic function of the 20-mer adds no information and cannot raise the
ceiling; this is not a screen (Appendix B2), and the effect size lies elsewhere: family A's access to
DNA outside the 20-mer is worth **+0.164**.

**A model given 300 of the 6,169 regenerated columns is at parity with one given all of them, but a
hand-picked reduced set is not a substitute for the pool.** Selecting 300 columns inside each
training fold against using every column gives **0.5445 against 0.5449** on the curated base and
**0.7084 against 0.7079** with family A (±0.0005 either way); quadrupling the budget changes
nothing, so no arm above needs a selection-budget caveat. Reduction of the representation itself:

| reduced set | cohort | ρ loss vs all columns | reading |
|---|---|---:|---|
| 427 = 364 indicators + 63 hand-named | published | **+0.0023** [−0.0008, +0.0054] (0.5249 against 0.5272) | non-inferior at any margin from 0.0054 up (one-sided p = 0.035 against 0.005, just outside the two-sided interval; margin not pre-declared, Appendix B) |
| 412 = 364 + 48 portable hand-named | published | 0.00003 | non-inferior at any margin worth naming; computable for any 20-mer plus the reference genome |
| 364 indicators only | published | 0.0083 | loses, so the 48 portable columns earn their place |
| 364 and 1,510 columns (ladder above) | curated | 0.0118 and 0.0096; with family A 0.0143 and 0.0068 | falls short of the 0.005 margin 427 passed |

So "412 ≈ all 6,232" holds for the published matrix under a 300-column budget and does not transfer
to the curated cohort, whose arms (6,517 columns = 6,169 regenerated + 348 flank) contain none of
the 63 hand-named columns in any case. The same redundancy accompanies sixteen model classes
spanning only 0.116 ρ (eight span 0.1076 with family A) and no correlate of the residuals above 0.031.

**The gain is better located by how much of a column the guide determines than by which family
built it** (Appendix B). Of the 636 family columns that have an R² against the guide, the 18 the
guide determines are worth **+0.008** as a block, the 16 partly recoverable **+0.020** and the 602 it
cannot predict **+0.201**. *Limit:* the blocks differ in size as much as in recoverability, so Δρ is
a difference and not a ratio, and finer banding puts the step at about R² = 0.2, not 0.99.

### 2. The DNA around the cut site predicts cutting — the one large effect, and it is local and compositional

**Windowed composition of the flanking DNA (family A, 348 columns: GC, purine, longest run and
quantum-table averages at 50/250/500/1,000 nt each side, plus the nearest 10 bases) is the one large
effect, larger than every other idea tested combined, and Cas9 does not read this DNA.** It is worth
**+0.164 ρ** on the curated cohort, **+0.156** on genes no step of the pipeline has seen (Part 4)
and +0.080 on the published matrix, and it survives holding out whole genes, losing nothing beyond
the 6% that grouping costs a feature which cannot memorise a locus (Appendix B3). It grows with the
model: under one protocol (the base-only selector never offered a flank column) the increment is
**+0.1563** [+0.1515, +0.1616] at the champion's settings and **+0.1697** [+0.1659, +0.1738] at the
tuned ones, 15 of 15 folds either way, and tuning is worth **+0.0162** with these columns against
**+0.0028** without, so almost all of what a full hyperparameter search bought is the ability to use
the flanking DNA (Finding 4, `EXTENSION_LOG.md` §16).

**The effect is spatially local.** Giving each guide the flank track read from a position offset by
a circular chromosome shift keeps every column's distribution, smoothness and long-range
autocorrelation and breaks only the correspondence (Appendix A):

| flanks read from (`shift_null.csv`) | ρ | gain over no flanks |
|---|---:|---:|
| **the right place** | **0.6060** | **+0.0769** |
| 1 kb away | 0.5353 | +0.0061 (8% of the effect) |
| 10 kb away | 0.5160 | −0.0132 |
| 100 kb – 2 Mb away | 0.5194–0.5229 | −0.006 to −0.010 |

At 1 kb the shifted columns are still 0.41 correlated with the real ones; by 10 kb the gain is gone,
so the effect is not chromosome-scale structure.

**Target GC does not explain the effect; it suppresses it** (`gcstrata.py --corr`, 33,567 curated
guides). Target GC correlates **−0.147** with the label and **+0.245** with flank GC. Partialling it
out raises flank GC's association from **+0.176 marginal to +0.221 partial**, positive in all five
target-GC strata (**+0.425, +0.278, +0.180, +0.107, +0.068**, strongest where the target is
GC-poor): the signature of a suppressor, not a confound. The full ablation inside each stratum,
where target composition is near-constant:

| target GC (`gc_strata_ablation.csv`) | n | baseline ρ | + family A | Δ |
|---|---:|---:|---:|---:|
| 0.20–0.40 | 2,776 | 0.3353 | 0.4630 | **+0.128** |
| 0.40–0.50 | 2,776 | 0.4045 | 0.4797 | +0.075 |
| 0.50–0.55 | 2,775 | 0.4627 | 0.5221 | +0.059 |
| 0.55–0.60 | 2,776 | 0.4760 | 0.5220 | +0.046 |
| 0.60–0.85 | 2,776 | 0.4443 | 0.5023 | +0.058 |

Family A is worth +0.046 to +0.128 inside every stratum. *Limit:* the gain is **not monotone** (the
top stratum rises) and covaries with headroom, since the GC-poor stratum has the lowest baseline;
the ordering is an observation, not a dose-response curve.

**Hypothesis (untested).** Flank composition acts through a duplex-opening step limited by local GC.
Consistent with it: target GC correlates **−0.201** with the score, the effect survives a change of
enzyme and of organism, and the model run backwards prefers GC-poor targets (35% against a 51%
mean). Its one test is the stratified ablation above, where a GC confound would have vanished and
did not. *Limit:* the assay measures depletion, not cutting; separating a strand-invasion-limited
from a hybridisation-limited step would need kinetic data.

**A convolutional network given the raw DNA recovers only a fraction of the effect.** The network
has the architecture family of the best published competitor, on the same rows and folds (the
pooled response to flank GC is mostly a step at the AT-rich end; Finding 7):

| CNN given (`seqnet_matched.csv`) | protospacer + PAM | ±100 nt | ±250 nt | ±1,000 nt |
|---|---:|---:|---:|---:|
| ρ | 0.4980 | 0.5203 | **0.5247** | 0.5190 |

The curve is flat from ±100 and then declines: given the exact DNA the windows summarise, the
network adds **+0.019** where the windowed means add **+0.080**. A windowed average is representable
by its mean-pooling context branch, so the barrier is inductive bias and sample efficiency (the
prior that the right summary of 500 positions is their unweighted mean is expensive to learn from
33,567 noisy examples), not a representational limit.

**On the DNA a tuned CNN actually holds, it beats windowed composition; SLICER's lead is long-range
reach.** crisprHAL 2's released rows are 378 nt for a 23 nt target (**189 nt upstream, 166
downstream**). Restricting family A to the columns that fit inside that span (the 50 nt windows and
the near-cut columns; the 250/500/1,000 nt windows reach past their input) gives a like-for-like
contest:

| arm | flank columns | ρ | vs crisprHAL 2 |
|---|---:|---:|---:|
| near-cut only (±10 nt) | 260 | 0.6178 | −0.079 |
| **matched to their span (±189 / ±166)** | **282** | **0.6724 ± 0.0094** | **−0.025** |
| full family A | 348 | **0.7072** | **+0.010** |

Inside their own span their convolutions beat windowed composition by **0.025**, so windowed means
are not a smarter encoding of the same bases; the lead comes from the 250–1,000 nt windows their
input never reaches, worth 0.7072 − 0.6724 = **+0.035**. A tuned CNN also does considerably better
than the in-house one on the same span.

**Given the same reach, crisprHAL 2 at its published hyperparameters gets worse, not better.**
Retrained at four input widths on the same rows, label and fold (batch 256, as cuDNN cannot allocate
the GRU workspace at 2,023 nt and batch 1,024; the 378 nt arm at both batches bounds that cost at
**−0.011**, a fifth of the effect below):

| DNA both models see (`crisprhal_wide_window.csv`) | crisprHAL 2 ρ | vs its 378 nt | SLICER ρ | lead |
|---|---:|---:|---:|---|
| guide + PAM (23 nt) | 0.5494 | −0.131 | 0.5473 | **tied** (0.002) |
| **378 nt (as published)** | **0.6799** | — | 0.6568 | crisprHAL **+0.023** |
| 1,023 nt | 0.6602 | **−0.020** | — | — |
| ±1,000 nt (2,023 nt) | 0.6299 | **−0.050** | **0.6944** | **SLICER +0.065** |

crisprHAL 2 peaks at the width it was designed for and declines monotonically beyond it, by roughly
7× its across-fold sd at 2,023 nt, while SLICER improves with width. Browne et al. (2026) stopped
their per-dataset length search at 378 nt for *E. coli* SpCas9 (406 nt for eSpCas9), reporting that
gains level off (+0.00034 ρ per nucleotide upstream, +0.00074 downstream; 0.554 on the sgRNA alone
to 0.697 at 378 nt), so the decline continues a curve they had already flattened. *Limit:* one
fold, one seed; the claim is only that the published configuration is degraded by input wider than
it was tuned for. Equalising the DNA removes the information asymmetry (they receive raw sequence,
family A a 348-column summary) but not the opportunity to use it, since epochs, dropout, filters and
GRU width were chosen at 378 nt; a re-tune at each width is not yet run
(`colab/crisprhal_tune_by_width.py` pre-registers what each outcome would support; Appendix A).

### 3. The effect belongs to the DNA, and transfers

**Changing the enzyme or the organism costs about a tenth of the performance; changing the kingdom costs all of it.** In the hardest bacterial transfer, a model imported from a 236 kb *C. rodentium* screen into the *E. coli* genome, the flank features carry **+0.150 of the 0.628** that arrives. That is the sense in which the effect belongs to the DNA: it is the part of the model that survives a change of protein and of host.

Each row trains SLICER on one screen and predicts another with no retraining or recalibration. *Transferred* is the imported model's ρ on the target; *local* is the same architecture trained on the target, the ceiling for a transfer; *retained* is their ratio; the last column is the part of the transferred ρ owed to flank features (base + family A minus base alone, both transferred; `results/transfer_model.csv`, `results/transfer.csv`, 3 seeds).

| variable changed | from → to | transferred ρ | local ρ | retained | of which family A |
|---|---|---:|---:|---:|---:|
| **nuclease** | SpCas9 → eSpCas9, same genome | 0.689 | 0.784 | **88%** | +0.098 |
| **nuclease + organism** | *E. coli* SpCas9 → *C. rodentium* TevSpCas9 | 0.698 | 0.766 | **91%** | +0.072 |
| **the same, reversed** | *C. rodentium* TevSpCas9 → *E. coli* SpCas9 | 0.628 | 0.708 | **89%** | **+0.150** |
| **kingdom**, shared columns only | *E. coli* → human (6,216 shared columns, no family A on the human side) | −0.039 | 0.508 | **0%** | — |

A surviving effect under a new nuclease is not about SpCas9's idiosyncrasies; under a new organism it is not about *E. coli*'s composition. The human row failing is the control on these: a model that transferred everywhere would indicate a methodological artefact rather than biology.

*Limits.* The kingdom row uses only the shared columns, which carry no family A on the human side; built on both sides, family A transfers with a positive sign (Part 3). Both that row and its reversal are on row-wise human folds that include repeated guides and have not been re-run (Part 3). A fifth change, from a cleavage assay to a dCas9 silencing screen, alters what is measured and not only what is varied (zero-shot ρ 0.061 against 0.458 trained locally); it is a boundary experiment and is not in the table (Appendix B5).

#### Two confound controls

These are not transfer tests; each removes one way the effect could be an artefact of this dataset.

| could the effect be an artefact of... | result |
|---|---|
| read depth or mappability? | **unsupported** (see Limits) |
| gene ranking in disguise? | no: within-gene ρ 0.571 against 0.606 global; flanks worth **+0.061 inside genes** |

*Limits.* The mappability control was meant to test whether family A predicts read misassignment (a guide in a repeat, IS element or rRNA operon has counts that can be wrong in a way that tracks local DNA) by restricting to a high-uniqueness stratum; the stated result is a gain of +0.078 → +0.051 before and after restriction. The two files cited for it do not contain those numbers: `ablation_abund_split.csv` is the `d_supercoiling` abundance split, and `abund_local_vs_mappability.csv` is a track-versus-uniqueness correlation matrix whose `uniq` definition is lost. The control is treated as unsupported and the mappability objection as open. The within-gene row is traceable. Neither control is an intervention; together they only rule things out.

#### Scope of the predictor

**The predictor estimates cleavage efficiency, and its applicability follows from that.** In these pooled screens a guide's score falls because the cell died, and the cell died because Cas9 cut the chromosome; death is the readout, cutting efficiency the quantity predicted. This is a statement about the training data, not about the enzyme: the model was fitted to survival after cleavage and says nothing about binding-only assays.

| application | does this predictor apply? |
|---|---|
| gene knockout, HDR knock-in, any edit needing a double-strand break | **yes**: this is the quantity predicted |
| pooled knockout screens using Cas9 cutting as the perturbation | **yes**: the screens it was fitted on |
| CRISPRi / CRISPRa with dCas9 | **no, not as it stands**: guide strand dominates that assay and a cutting model has no use for it (Appendix B5); a guide-sequence model trained on silencing data is the right tool |
| base and prime editing (nickase-based) | **untested**: these bind and nick rather than cut, and nothing here speaks to them |

For an application that is silencing, crisprHAL 2's raw-sequence model transfers to a CRISPRi screen about half again as well as SLICER (|ρ| 0.087 against 0.061; Part 5), plausibly because what survives that change of assay is guide-sequence effects, not flanking DNA.

### 4. The model is out of *material*, and the tuning gain is evidence for it

**The learner is not the limit. The same tuning that is worth +0.016 ρ with family A present is worth +0.003 without it**, so the tuned settings buy the ability to use the flanking columns, not generic capacity, and without those columns there is almost nothing for extra capacity to do. A model one hyperparameter search from +0.016 might look short of any limit; asking where that +0.016 comes from shows it is not.

The four configurations were run on a matrix with family A and on one without it, under the standard protocol (gene-grouped 5 folds × 3 seeds, selection fitted inside each training fold, the base-only arm's selector never offered a flank column; `family_a_at_tuned.py`, `results/post_freeze/family_a_at_tuned*.csv`, `EXTENSION_LOG.md` §16):

| (`family_a_at_tuned_summary.csv`; 15 folds, paired, bootstrap over folds) | base only | base + family A | family A's increment |
|---|---:|---:|---:|
| champion (depth 4, lr 0.05, 400 trees, 300 columns) | 0.5588 | 0.7151 | **+0.1563** [+0.1515, +0.1616] |
| depth 6 | 0.5569 | 0.7223 | +0.1655 |
| **tuned d099x** (depth 5, lr 0.013, 4,000 trees, 450 columns) | 0.5617 | **0.7314** | **+0.1697** [+0.1659, +0.1738] |
| tuned d077x | 0.5631 | 0.7310 | +0.1680 |
| **the tuning gain itself (d099x − champion)** | **+0.0028** [+0.0009, +0.0050], 9/15 folds | **+0.0162** [+0.0152, +0.0173], 15/15 folds | |

On the base representation, ten times the trees and a quarter of the learning rate buy +0.003, positive in only 9 of 15 folds; on top of family A the same change buys +0.016 in all 15. Consequently **family A's increment grows with the model, from +0.156 to +0.170**, with the best-of-four bound (tuned arm over the best base-only arm, guarding against the settings having been chosen with family A present) at +0.168 [+0.164, +0.173]. The champion's base + A figure (0.7151) and d099x's (0.7314) are identical to `tune_fullcv.csv`, so this run reproduces the established protocol.

*Limits.* The tuned settings were selected on DEV genes with family A present, so applying them to a base-only matrix is not a matched tuning effort and the base-only column could be slightly pessimistic; a matched base-only search has not been run. For family A's increment at the tuned settings to fall back to the champion's +0.1563, a base-only search would have to reach ρ **0.5751**: +0.0120 beyond the best of the four tried (d077x, 0.5631) and about 2.9× the largest base-only gain any of them produced (+0.0042).

**Residual correlation agrees, and reducing the matrix does not change it.** Correlating every column against the champion's out-of-fold residuals (`representation_headroom.csv`):

| model | strongest \|ρ\| among columns it has | among columns held back |
|---|---:|---:|
| all 6,232 | 0.028 | **0.147**: family A |
| the 427 | 0.022 | **0.146**: family A |
| 427 + family A | 0.028 | 0.070, none above 0.10 |

Dropping 5,805 columns did not change what the model is missing. The test has a positive control: it flagged family A at 0.147 before family A existed, and family A was then worth +0.080.

**The usable k-mer order stops at three.** Without feature selection, trimers help and tetramers hurt, and the two cancel (paired over 3 seeds × 5 folds, `kmer_order_paired.csv`):

| contrast | mean Δρ | folds | paired *t* |
|---|---:|---:|---:|
| **+ trimers** | **+0.0035** | 12/15 | **p = 0.025** |
| **+ tetramers on top** | **−0.0028** | 4/15 | **p = 0.013** |
| net | +0.0007 | 8/15 | p = 0.62 |

4,343 tetramer columns for 256 possible 4-mers per position are too sparse to carry anything. Through the top-300 selector adding trimers hurt, so a conclusion drawn downstream of feature selection is a statement about the selector until shown otherwise.

### 5. On a redundant matrix, per-column importance is unstable across fits, but SHAP at block level and edit tests are faithful

**Feature importance disagrees between models on this matrix, and the standard "drop the top features and refit" probe returns ≈0 for every model.** Both are properties of a matrix on which 6,095 columns restate 20 variables, not of boosting or of the attributions. The defensible report is SHAP, at block level.

**Importance.** Seven model families (two boosters, two forests, two linear, CatBoost; 3 folds; `importance_model_pairs.csv`). The top three rows are the five tree families (10 pairs), the only ones for which SHAP is defined; the linear models are added beneath:

| method | agreement between models, per column | per block |
|---|---:|---:|
| the library default (gain / impurity / coef) | **0.050** | 0.531 |
| permutation | 0.258 | 0.581 |
| **SHAP** | **0.496** | **0.800** |
| *with the two linear models too (all 21 pairs): default* | *−0.057* | *0.522* |
| *with the two linear models too: permutation* | *0.187* | *0.512* |

The instability survives bagged forests and penalised linear models, which make the default and permutation agreement worse, not better. What did not survive is the apparent finding that models prefer different kinds of column: it was an artefact of split gain. XGBoost's share on binary indicators falls from 57% to 31% under SHAP and lands on LightGBM's 31% exactly; the largest between-model gap on any kind of column goes from 31 points to 0.3.

*Limits.* Report SHAP at block level and never gain. The pipeline's own column selector uses gain, which costs no accuracy but does bias what the model is shown. Block-level reporting is the only fix available: selecting for decorrelation lowers ρ and leaves a less readable column set, because on this matrix the redundant columns are the informative ones (Appendix B).

**Faithfulness by drop-and-refit is uninformative here.** Dropping the top *k* columns removes almost no information because near-duplicates remain to carry it, so the probe reads ≈0 for every model and representation. Equalising the *share of attributed importance* removed, rather than the number of columns, makes the comparison about what each representation loses: at a matched share (~35%) the full matrix loses 0.002 and the 775-column reduced set loses **0.015**, six times the damage for the same share of the explanation removed, while both concentrate importance almost identically (44 and 37 columns for half of it).

#### Three tests that need no refit

Refitting is where redundancy does its damage, so three tests use one fixed model: the production recipe (base + family A, gain-ranked top 300, champion XGBoost) fitted on a random 80% of the curated cohort, ρ 0.712 on the other 20% (`interp_extra.py`; one fit, so the figures are not averaged over seeds).

**1. SHAP is faithful to the fitted model, at column level as well as block level (sensitivity-*n*).** For random subsets of *n* columns, the summed SHAP over the subset correlates, across subsets and within each of 100 held-out guides, with the prediction lost when those columns are replaced by values from 50 training guides (`results/interp_sensitivity_n.csv`):

| subset | n | SHAP | ridge surrogate | SHAP with columns permuted |
|---|---:|---:|---:|---:|
| columns | 1 | **+0.928** | +0.079 | −0.000 |
| columns | 100 | **+0.957** | +0.210 | +0.007 |
| blocks | 1 | **+0.956** | +0.396 | −0.011 |
| blocks | 6 | **+0.969** | +0.423 | +0.038 |

The permuted column is the control: the same numbers on the wrong columns correlate at zero, so 0.93–0.97 is the attribution's content. The ridge surrogate fails because it attributes a different model. This is consistent with the claim above: SHAP accounts faithfully for this fitted model, even for one column, but a column's credit does not mean the same thing in a second fit, where a correlated neighbour carries it. Faithful to the model and unstable across models is the pattern redundancy predicts. *Limits:* the correlation tests the ordering of subsets, not the attribution's scale; the SHAP is the path-dependent variant (the interventional one is refused by this XGBoost build), so agreement with an interventional ablation also shows the two conventions coincide here.

**2. About 15% of the output's variance is interaction, mostly among correlated descriptors.** SHAP interaction values on 400 held-out guides, all 300 columns (`interp_interactions.csv`, `interp_interaction_blocks.csv`): single-column terms carry **70.2%** of the variance and pairwise terms **14.7%**, the remainder being their covariance. Off-diagonal terms hold 52.7% of the absolute interaction mass, of which 46% lies between two protospacer blocks, 15% between two flank blocks and **39% between a flank block and a protospacer block**. The strongest pairs are descriptors of the same base pairs (`p18dimer.Hbond.energyraw` with `p20basepair.Hbond.energyraw`, 0.034, and other terminal-position pairs), which is the model splitting credit between near copies. This is consistent with a ridge on a decorrelated basis reproducing 95.6% of boosting's ρ (Appendix B): the interaction part of the output is small relative to the part that orders guides correctly. *Limits:* with credit split among correlated columns, a flank × guide term is at least as likely to be redundancy as mechanism, so no claim about biological interaction between flank and guide follows.

**3. Edited flanks move the prediction through composition and little else.** Rewriting everything beyond the 20 nt next to the cut on 1,500 held-out guides and rebuilding family A from the edited DNA (the unedited rebuild reproduces the stored matrix exactly; `interp_counterfactual.csv`): GC +5 points moves the prediction by +0.46 [+0.42, +0.50] and −5 points by −0.88 [−0.93, −0.83], against a label sd of 3.29, while changing as many bases without moving GC moves it by 0.003 to 0.016 (the control). The response is asymmetric, about twice as far for a fall as for a rise, which the AT-rich step of Finding 7 predicts. Shuffling order at fixed composition moves it by +0.23, through the run-length and windowed quantum columns. *Limits:* these are properties of the fitted model, not measurements of the genome. 1 kb flank GC has an sd of 4.5 points across guides, so ±10 is a 2.2 sd extrapolation where trees are cliff-edged, and no experiment has varied flank composition at a fixed guide (Part 3).

**In what sense SLICER is interpretable.** Its attributions are faithful to the fitted model (SHAP, sensitivity-*n* +0.928 for one column and +0.956 for one block), agree between model families at block level (0.800), and its response to an edited flank is composition and little else (test 3); the signal itself is described without the model in Finding 7. It is not interpretable in the sense of a per-column ranking that survives a refit, because correlated neighbours carry a column's credit in a second fit.

### 6. The label's ceiling is 0.946, measured from crisprHAL's own replicates

**No predictor can exceed ρ 0.946 [0.944, 0.948] against the curated label, and SLICER at ρ 0.707–0.721 leaves about 0.22 of ρ unaccounted for** (0.225 against 0.946; with the tuned model at 0.7314 it is **0.215**, or 77.3% of attainable). The figure is a replicate-agreement one, from crisprHAL 2's published per-replicate counts, not an inherited figure (`curated_ceiling.py`).

crisprHAL 2's paper reports no replicate statistics, but its counts for two Cas9 replicates and two dead-nuclease controls are published. Pairing Cas9 replicate 1 with control 1 and replicate 2 with control 2 gives two independent estimates of the same quantity, agreeing at **ρ 0.8097**. Spearman–Brown makes the released two-replicate score's reliability **0.8948**, and its square root is the highest ρ any predictor can reach against it. The other pairing gives 0.9453, so the figure does not depend on how replicates are matched.

| check | result |
|---|---|
| does their **cohort** reproduce? | their R script filters on an average control count of **73**; applying it gives **33,871** guides against the 33,567 they release |
| does their **label** reproduce? | a plain CPM log-ratio of the two conditions correlates **\|ρ\| = 0.957** with their released score, so these are the replicates behind that label |
| is it conservative? | **yes**: ALDEx2's posterior shrinks noisy guides toward the prior, which can only raise reliability above a plain log-ratio's, so 0.946 is a **lower bound** |

The curated label is cleaner than the published one, **0.946 against 0.937**, as the crisprHAL 2 paper claims of curation, now measured from its own counts. The published figure came from a different route (Guo's tiling library, not replicates) and should not be reused for curated numbers (Appendix B).

**The field's version of this ceiling is mis-derived.** The standard benchmarking frame, Konstantakos et al. (2022, *NAR*), takes the replicate Spearman range of 0.71–0.77 as an upper limit of about 0.7 on any predictor, with 0.4 as the threshold for a successful tool. That treats a **reliability** as a **ceiling**. Two corrections push the same way: the largest correlation any predictor can have with a noisy target is the **square root** of that target's reliability (Spearman's 1904 correction for attenuation), and when the target is an average over replicates, as essentially every released activity score is, Spearman–Brown raises the reliability first. On the review's own numbers the attainable ceiling is **0.911 to 0.933**, not about 0.7: understated by 0.21–0.23, of which 0.14–0.18 is the missing square root alone (Part 5, `EXTENSION_LOG.md` §19).

| replicate agreement *r* | √*r* | reliability of a 2-replicate mean | attainable ceiling |
|---:|---:|---:|---:|
| 0.71 | 0.843 | 0.830 | **0.911** |
| 0.77 | 0.877 | 0.870 | **0.933** |
| **0.8097** (measured here) | 0.900 | 0.895 | **0.946** |

*Limits.* Attenuation theory is exact for Pearson under classical test theory; applying it to Spearman, as both the review and this report do, is an approximation that holds well for roughly normal data and is not exact.

---

### 7. What the signal is made of: GC over a few hundred bases, a seed region that inverts between kingdoms, and the two coupled

**Claim.** The flanking effect of Findings 2–3 is carried by **GC specifically**, is **side-symmetric**, has one length scale of **250–500 nt** with no structure finer
than ~250 nt, and **interacts with the PAM-proximal seed** rather than adding to it. Pooled across the genome its response to composition is mostly a **step at the
AT-rich end**, with a gentler graded component inside genes. Separately, the guide's own position-specific base preferences are confined to the seed and are **unrelated
between *E. coli* and eukaryotes**; the strongest one inverts. No importance is read off the model (Finding 5) (`mechanism.py`, `novel_followups.py`, `hypotheses.py`,
`eukaryote_rules.py`; `EXTENSION_LOG.md` §14, §20–21, §24–25).

**(a) The flank effect is GC, not composition in general.** Partial Spearman of each flank window against the label, holding target GC fixed (33,567 guides):

| | 50 nt | 250 nt | 500 nt | 1,000 nt |
|---|---:|---:|---:|---:|
| **GC**, upstream | +0.172 | **+0.203** | +0.196 | +0.173 |
| **GC**, downstream | +0.146 | **+0.197** | +0.194 | +0.181 |
| purine, upstream | −0.000 | +0.001 | −0.005 | −0.005 |
| purine, downstream | −0.007 | −0.018 | −0.014 | −0.000 |

Purine fraction is within ±0.018 of zero everywhere while GC runs +0.15 to +0.20, which excludes a mechanism keyed to purine/pyrimidine asymmetry. The effect tracks
**absolute** composition, not enrichment against the genome's own background, and DNA-shape descriptors are a reparameterisation of composition, not a second channel
(Appendix B4). *Limit:* this identifies the variable, not the physics (see I1).

**(b) It is side-symmetric, so it is not the orientation effect.** Upstream and downstream agree to within 0.026 at every window. A directional process (fork direction,
transcription direction, one-sided resection) cannot produce that, so the orientation effect (Part 3; Δ 0.051 against a curated TSS) and the composition effect are two
different things.

**(c) There is one length scale, 250–500 nt, and no structure below ~250 nt.** The association in (a) rises from 50 to 250 nt and falls by 1,000 nt on both sides. Binning
the flank at increasing resolution saturates, then turns over (reference: family A's four nested windows, 24 columns, ρ 0.6732):

| flank described by | columns | ρ | share of the composition signal |
|---|---:|---:|---:|
| nothing (base only) | 0 | 0.5617 | |
| GC and purine over the whole kilobase, each side | **4** | 0.6462 | **75.7%** |
| 250 nt bins | **16** | 0.6679 | **95.2%** |
| 20 nt bins | 200 | 0.6687 | 96.0%, *turns over* |

Doubling the resolution past 250 nt buys +0.0022 (32 columns, ρ 0.6701) and 200 columns score below 32: what a bulk property averaged over a few hundred bases predicts,
not a motif, a positioned protein or a periodic feature. *Limit:* that there is one scale is the measurement; its physical meaning (one to three persistence lengths of
duplex DNA, far longer than the ~20–30 nt Cas9 contacts) is a hypothesis.

**(d) The guide's own base preferences sit in the seed, and they invert between *E. coli* and eukaryotes.** In *E. coli*, per-base effects on the label (SD units) are
within ±0.04 of zero at positions 1–7 and the twelve strongest are all at 14–20: at position 20, G **−0.322** and T **+0.348**; at position 18, G −0.266 and T +0.210. The
model's gain-by-position profile (63% of split gain in the eight PAM-proximal positions) and Orro et al.'s human SHAP profile (peak at 19–20) agree. The identical
estimator on the human matrix (13,138 distinct guides; sign convention checked: target GC against label −0.147 here, +0.018 there) gives at position 20 G **+0.131** and T
**−0.170**. Across the 80-cell matrix the rank correlation is **−0.052**, with **15 of 27** jointly-significant cells opposite in sign. G immediately 5′ of the PAM is the
most disfavoured base-position combination in *E. coli* and a favoured one in human, consistent with mammalian-trained predictors failing in bacteria. Practical
consequence: **avoid G and prefer T immediately 5′ of the PAM in *E. coli***.

*Bacteria against eukaryotes, or E. coli against one human cohort?* The same estimators were run on 23 tables of the CRISPOR benchmark collection
(`external_data/crispor_effdata/`). The inclusion rule, fixed beforehand, was at least 300 guides: eight human datasets (six libraries), two mouse, one zebrafish. Each
dataset is oriented by its median Spearman correlation with six published predictors (all six agree in sign for all eleven; weakest anchor Wang 2015 at −0.10;
`eukaryote_orientation.csv`); split-half reliability of the 80-cell maps is 0.66–0.95.

| pair of datasets (different libraries) | pairs | median ρ between the 80-cell maps | range | corrected for reliability |
|---|---:|---:|---|---:|
| human – human | 26 | +0.456 | −0.047 to +0.711 | +0.59 |
| human – mouse | 16 | +0.443 | +0.101 to +0.695 | +0.56 |
| human – zebrafish | 8 | +0.221 | −0.239 to +0.381 | +0.28 |
| ***E. coli* – human** | 8 | **−0.140** | −0.276 to +0.078 | −0.16 |

Mouse and human share their guide rules to the degree two human libraries do (corrected +0.56 against +0.59); *E. coli* shares none. The position-20 contrast (G minus
mean of A and C, SD units) says the same:

| | *E. coli* | human (21,567 guides) | mouse (2,239) | zebrafish (1,233) |
|---|---:|---:|---:|---:|
| G minus mean(A, C) at position 20 | **−0.333** [−0.359, −0.308] | **+0.277** [+0.248, +0.306] | **+0.398** [+0.308, +0.482] | **+0.430** [+0.303, +0.548] |

All eleven datasets of at least 300 guides have the eukaryotic sign (the smallest, Wang 2015, +0.08 [−0.004, +0.160], is indistinguishable from zero), so the inversion is
**bacteria against eukaryotes**. Zero-shot transfer agrees (SLICER base representation, ridge in parentheses; label rank only), against the target's own five-fold ρ of
0.18–0.55:

| trained → scored (different libraries) | pairs | median ρ (ridge) | share of the target's own ρ retained (ridge) |
|---|---:|---|---|
| **human → mouse** | 16 | **+0.342** (+0.283) | **73%** (65%) |
| **mouse → human** | 16 | **+0.261** (+0.269) | **70%** (66%) |
| ***E. coli* → eukaryote** | 11 | **−0.041** (−0.056) | −13% (−14%) |

*Limits:* labels are not the same quantity across tables (indel frequency or depletion); zebrafish differs in system (injected embryos); datasets scoring one library in
two cell lines are flagged `same_library` and left out of the pair table; no table has gene identifiers. Why the rules invert is not answered: Moreb and Lynch (2021; 44
datasets, abstract only) attribute cross-species failure to a genome-dependent search time, which these data cannot test (Family H, genome-wide seed-match counts, is null
at +0.000 on a cohort of uniquely targeting guides).

**(e) The flank and the seed are coupled, and the coupling is synergy, not compensation.** Of what trees add over a linear model, 37% is pairwise; a depth-2 model's
splits are pairs. The leading block pairs are flank composition × flank quantum (0.058), seed indicators (16–20) × protospacer dimer quantum (0.048) and **flank
composition × protospacer basepair quantum (0.048)**; the strongest single pair is `eng.flank.comp.up250.gc` against a property of **position 20** (1.9%). The coupling is
diffuse (no block pair above 5.8%, no column pair above 1.9%). Direct test (H1), Spearman of label on flank GC within each base at position 20 (gene-bootstrap intervals,
30,203 genic guides):

| base at position 20 | A | T | C | G |
|---|---:|---:|---:|---:|
| that base's own effect on the label (SD) | +0.193 | +0.348 | −0.142 | −0.322 |
| ρ of label on flank GC | **0.219** [0.188, 0.242] | 0.184 [0.154, 0.217] | 0.169 [0.144, 0.192] | **0.118** [0.090, 0.147] |

T minus G is **+0.066 [+0.029, +0.102]**. The flank's slope is shallowest where the seed base is most disfavoured: a guide needs a competent seed *and* an open region. It
is not a compressed label: the label's spread is the same in all four strata (SD 3.08–3.20), and the ordering survives dropping the lowest quartile of the label (A 0.084,
T 0.083, C 0.060, G 0.030), whose share of guides differs only as the seed's own effect predicts (G 35%, C 29%, A 19%, T 15%). *Limit:* these checks make a floor effect
unlikely but do not formally exclude it; depth 2 cannot see third-order structure (bounded at 4%).

**(f) Pooled, the response to flank GC is a step at the AT-rich end; within genes there is a gentler gradient.** A linear dependence was predicted and failed. Mean label
by decile of flank GC (250 nt each side averaged, 30,203 genic guides):

| decile of flank GC | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean flank GC | 0.406 | 0.460 | 0.483 | 0.498 | 0.510 | 0.520 | 0.530 | 0.541 | 0.554 | 0.579 |
| mean label | **−3.09** | **−0.40** | +0.11 | +0.01 | +0.05 | +0.07 | +0.14 | +0.14 | +0.02 | −0.05 |

Spearman of label with flank GC is **+0.485** among the lowest 10% of guides (below 44.4% GC, 3,086 guides) and **+0.021** among the other 27,117. The lowest decile sits
**3.10 label units lower [−3.37, −2.78], 0.95 SD**, than the rest, and not because of a few genes (857 genes; the ten largest contributors hold 4.4%; without them the gap
is −2.97). **Within genes** (variables centred on the gene mean) the dependence is graded (decile means −0.58 rising to +0.50), and **+0.066** survives setting the lowest
10% aside. So there are two components: an AT-rich penalty of about one SD, mostly *between* genes, and a graded gradient *within* genes (the +0.061 of Finding 3); the
first is why the flank effect looks strongest among GC-poor targets (Finding 2, +0.425 against +0.068). The gap is −0.76 SD at 50 nt, **−0.95 at 250**, −0.92 at 500 and
−0.78 at 1,000 nt (ρ inside the tail 0.435, **0.485**, 0.416, 0.389), so the step belongs to regions a few hundred nucleotides wide.

| robustness of the step (lowest decile of flank GC against the rest) | n | deficit, label units [95% interval] | in SD |
|---|---:|---|---:|
| *E. coli*, WT SpCas9, all guides (bootstrap over 100 kb blocks) | 33,567 | −3.29 [−3.76, −2.84] | −1.00 |
| *E. coli*, eSpCas9, a different enzyme at the **same loci** (not independent of the row above) | 59,489 | −2.15 [−2.43, −1.86] | −0.80 |
| *C. rodentium*, TevSpCas9, at its own decile edge (GC 50.8%) | 25,210 | −0.57 [−0.75, +0.14] | −0.21 |

The step replicates across an enzyme on the same genome and is **not established in a second genome**; with *E. coli*'s absolute threshold in *C. rodentium* (653 guides)
the interval includes zero (−1.97 [−2.04, +1.10]). It is not expression (flank GC and expression are uncorrelated, ρ −0.003; the association is 0.153 marginal and 0.154
given expression; the deficit is −3.79 / −2.80 / −1.77 in the lowest, middle and highest expression thirds), and it is not PAM density: holding flank GC fixed, NGG plus
CCN density (both flanks, 250 nt) has a partial Spearman of **+0.022**, the **52nd percentile** of 5,000 random sums of four of the 32 strand-collapsed trimer classes
(null mean +0.005, sd 0.040).

*Does the step exist in eukaryotes?* At the 77 nt scale the eukaryote tables allow (30 nt before the guide, 47 after the PAM), *E. coli*'s lowest decile sits **−0.70 SD**
[−0.73, −0.66] below the rest, ρ **+0.33** inside the tail. In the eleven eukaryotic datasets the gap is smaller and inconsistent: five have an interval that excludes
zero (from Xu HL60 −0.16 to Doench 2016 −0.45 SD), six do not (including zebrafish), and ρ inside the tail never exceeds 0.20: a weaker, inconsistent AT-rich penalty in
about half of the mammalian datasets and not in zebrafish. *Limit:* no gene identifiers (between and within genes cannot be separated), depletion screens carry gene
effects, and mammalian GC content tracks chromatin state.

Pre-specified predictions (decision rules written into the module docstrings before the runs):

| Predicted in advance | Observed |
|---|---|
| **P1.** Mammalian maps agree (corrected correlation at least 0.4 for every pair) | Held in the median, failed in the tails |
| **P2.** *E. coli* unrelated to every eukaryote and opposite at position 20 | Held |
| **P3.** Zebrafish agrees with mammals less than they agree with each other | Held, more strongly than predicted |
| **P4.** Mammal-to-mammal transfer keeps at least 60% of the target's own ρ | Held (73% and 70%) |
| **P5.** No AT-rich step in mammals | Partly failed: weaker, inconsistent step in about half |
| **H1.** The flank's slope is steeper where the position-20 base is favourable | Held (T minus G +0.066) |
| **H2.** A roughly linear dependence on flank GC (probability about 0.6) | **Failed**: a step at the AT-rich end |
| **H3.** PAM density is not what the GC effect is | Held (partial +0.022, 52nd percentile of the null) |

**(g) What follows.** *Established:* (a)–(f). *Inferences, each with what would show it wrong:*

| inference | rests on | would show it wrong |
|---|---|---|
| **I1. Melting-limited unwinding is not the main driver:** more GC means *more* cutting, the opposite of what a melting-limited step predicts | the sign of the step and of the within-gene gradient | a locus-matched design in which AT-rich flanks cut better; a within-gene gradient that reverses with another covariate held fixed |
| **I2. Flank and seed act as a gate, not as compensation** | (e): slope falls from A 0.219 to G 0.118 across seed bases; spread equal across strata | a multiplicative flank × seed term that fits no better than an additive one on gene-grouped folds; the order reversing on another screen |
| **I3. The AT-rich penalty belongs to regions a few hundred nucleotides wide,** not single sites or kilobase domains | the scale check in (f) | the gap growing toward a kilobase on a wider window or in another strain |

*One candidate explanation: AT-rich DNA is less accessible to Cas9.* The pattern alone supports it (a step at the AT-rich end, between regions, not expression, not
outlier genes). **The data do not support any particular accessibility mechanism.** Naming a nucleoid protein such as H-NS, which binds AT-rich DNA and silences
horizontally acquired regions, is a guess from outside the data: nothing here distinguishes it from another AT-binding factor, from AT-rich DNA being more flexible or
bent, or from a cell-level effect (copy number, replication timing). The scale check (I3) fits a local property better than a silenced domain. (The genes most often in
the lowest decile, `wbbJ`, `waaL`, `glf`, `waaO`, `alpA`, `trkG`, `lit`, `mcrC` and `casA`, look like horizontally acquired loci by name; the data carry no annotation, so
this is not evidence.) Falsifiers: (1) a published occupancy track for an AT-binding protein (guides inside bound regions should carry the deficit and the step should
shrink once occupancy is conditioned on; an equal deficit inside and outside kills the protein version); (2) a Δ*hns*-type strain that keeps the deficit; (3) a eukaryotic
screen, where a step would mean generic accessibility. The third was run (above): the step is weaker and inconsistent, so a bacteria-specific protein is not required, and
generic accessibility is neither shown nor excluded.

**Limits.** (i) The cross-kingdom comparison is between different assays (survival after cutting against indel, knockout or depletion scores), so it compares design
rules, not mechanisms; 15 of 27 is sign agreement among jointly-significant cells, not a formal test. (ii) H2 and every follow-up in (f) were found in one screen and
designed after the result they follow, so they describe the data and do not test a prediction. (iii) The step is established in *E. coli* K-12 SpCas9 on one curated label
and not in a second genome.

## Part 2 — Contributions

**1. The published representation is redundant and can be decoded.** 99% of its 6,232 columns regenerate from a 20-mer
and spell out about twenty variables; on the published matrix under a 300-column budget, 412 columns computable for
any guide match all 6,232 to 0.00003 ρ. The encoding is lossless, so **no deterministic function of the 20-mer can
add information**, though it can add *performance* (+0.011 to +0.019 for a depth-limited tree); this is why the
imported biophysical families could not have helped (Finding 1; Appendices B2, B3).

**2. Genomic flanking context is the main new signal: +0.164 ρ** on the curated cohort, **+0.156 on sealed genes**
(+0.080 on the published matrix) and **+0.170 once the model is tuned** (Finding 4). It is a local compositional
gradient, not a motif (texture-preserving null, which also measures its autocorrelation length), not mappability or
gene ranking, and it survives holding out whole genes. It transfers across nuclease and organism (88–91% retained with
no retraining) and, built on both sides, across kingdoms, where the shared published columns alone do not: it is worth
**+0.023 ρ** on the human matrix on its own ground (0.3063 → 0.3289, sd 0.003 over 15 fits, grouped folds;
`EXTENSION_LOG.md` §16.2) and turns transfer from ≈−0.049 to +0.022 and +0.017 (Part 3; Jacky's result), though the
transfer figures have not been re-run on grouped human folds. Whether it needs a break is **not settled**: on dCas9
silencing screens a weaker within-gene effect remains whose cause the data cannot identify (Appendix B5; Findings 2–3).

**3. The signal is GC over a few hundred bases, and the guide's own preferences invert between kingdoms.** The effect
is **GC**, not composition in general (purine is flat at every distance and side), and is **side-symmetric**, so it is
not the directional orientation effect. It has **one characteristic length of 250–500 nt**, with no structure below
about 250 nt; four numbers per guide carry three quarters of it. It tracks absolute, not genome-relative, composition:
per-genome percentiles change *C. rodentium* transfer by −0.0037 (0.7142 → 0.7105) (`EXTENSION_LOG.md` §20.1). The
guide's own preferences sit in the PAM-proximal seed and are **uncorrelated between *E. coli* and human (ρ = −0.052)**
and between *E. coli* and mouse or zebrafish, while human and mouse agree; G immediately 5′ of the PAM is the most
disfavoured base here and a favoured one in human. The GC association itself inverts (−0.20 against +0.02) with no
fitted model, so the inversion and the transfer failure belong to the *guide* half of the representation. The two
halves are **coupled**: the largest pairwise interaction is 250 nt flank GC against position 20 (Finding 7).

### What is new about the *E. coli* biology, separately from the model

One positive localisation, one directional effect, and constraints that are worth as much because each rules out a
plausible mechanism.

**Positive.**

1. **DNA outside the protospacer, which Cas9 never contacts, predicts whether a target is cut: +0.164 ρ**, most of it
   within 500 nt each side and essentially none past a kilobase; the largest of eleven families (next +0.139). Browne
   et al. established that the effect exists; what is added is its **shape**: compositional and smooth, no motif, with
   a measured autocorrelation length (8% survives a 1 kb shift, none survives 10 kb) (Finding 2).
2. **It is a local property of the target, not of its gene:** inside genes, where gene identity is constant, the flanks
   are still worth **+0.061** (Finding 3).

**Directional.**

3. **Orientation relative to transcription matters** (Jacky's result): Δ **0.051** [0.020, 0.082] against a curated
   RegulonDB TSS, so the surrounding DNA is not acting through composition alone. Transcription, replication or the
   supercoiling domains they create could carry the direction; which one is unidentified (Finding 7).

**Constraints, each ruling out an obvious mechanism.**

4. **Not helical phasing or intrinsic curvature:** GC skew, A-tract phasing and bendability periodicity at the helical
   repeat give −0.001 alone and +0.0004 on top of the flank composition (Appendix A).
5. **Not the physical shape of the flanks, but their letter composition:** shape descriptors are worth +0.120 alone and
   **+0.0007** once composition is present, and a 3D reconstruction of the helical path adds −0.0008 to +0.0006 on top
   of family A, replicated on a second cohort and under shuffled-flank controls (Appendix B4).
6. **Not supercoiling**, though the family built to test it pays +0.139: the untagged no-antibody GapR control predicts
   as well as the ChIP, so the family measures read density, mostly **gene dosage**; four columns from distance to
   *oriC* alone recover as much as the ChIP track (+0.0112 against +0.0111). Guo et al.'s "unknown chromosomal
   factors" are, here, copy number under multifork replication (Appendix A).
7. **Not gene essentiality, as Guo et al. reported; expression is not null:** the transcription family's +0.058 is
   gene geometry (+0.0393) plus expression (+0.0213) plus essentiality (+0.00004); the published essentiality null
   reproduces to four decimals, the published expression null does not (Appendix C).
8. **Not a restatement of target GC:** conditioning on target GC *raises* flank GC's association, **+0.176 marginal to
   +0.221 partial**, and the flank effect is positive inside all five target-GC strata (Finding 2).
9. **Most of the phenotype is still unexplained:** the label's reliability ceiling is **0.946** and the best model
   reaches **0.7314** after post-freeze tuning (0.7218 before), leaving about **0.21 ρ of real, repeatable biology**
   unaccounted for by every family tried; the same tuning is worth +0.003 on the base representation alone, so
   capacity is not what the gap is made of (Findings 4, 6).

### Three inferences from results in combination

**1. "GC content affects cutting" is two effects of opposite sign, and the window decides which one is measured.**
Spearman with the label: GC of the 20-mer guide **−0.147**, of the target (protospacer and PAM) **−0.201**, of the flank
(±250 nt) **+0.161**. Inside the protospacer GC-rich predicts *less* cutting (a GC-rich duplex costs more to unwind in
the R-loop); outside it the same composition predicts *more*, which that account cannot produce at any distance. The two
are positively correlated (+0.23 for the guide, +0.176 for the target), so each suppresses the other, which is why
conditioning on target GC raises the flank association (item 8). *Limit:* marginal associations on one cohort; signs
and scales, not magnitudes.

**2. Family increments measure what is new, not what matters.** Marginal |ρ| is 0.12 to 0.18 for nine of the eleven
families while increments over a guide-holding model span 0.001 to 0.164 (`recoverability_trend_columns.csv`). The
strongest of all 6,232 columns, `eng.dg.hybridisation_weighted` (**R² 1.00 recoverable from the guide's indicators, 0.90
correlated with guide GC**), has |ρ| **0.184** for a family increment of +0.018; `a_flank`'s strongest
(`eng.flank.comp.up250.gc`) is 0.161 for **+0.164**. So "nothing confined to the 20-mer clears +0.019, nothing above
+0.05 is confined to it" ranks families by what the model cannot already compute, not by what cutting depends on.

**3. The guide's own folding is a real determinant.** The spacer's minimum free energy is worth **+0.0119** on the
curated base with marginal **|ρ| 0.102**; it is fully determined by the twenty bases, so its small increment says
nothing about importance, and the control does: the same ViennaRNA computation on each guide with **one fixed
permutation of its twenty positions** is worth **+0.0000 and −0.0007**, and random non-linear functions of the guide
matched on out-of-fold recoverability (R² 0.34–0.48 against 0.32) are worth +0.0001 to +0.0007 at every interaction
order (Appendix B). So it is the particular structure the real spacer forms, not composition or tree-difficulty.
*Limit:* one cohort.

---

## Part 3 — Where this stops

### What the work does not show

- **One organism and two relatives.** No genome-wide Cas9-cutting screen exists outside *Enterobacteriaceae*; what
  exists elsewhere is CRISPRi, a different measured quantity, so closing this needs wet-lab work. On the **shared
  published columns**, cross-kingdom transfer fails in both directions: *E. coli* → human **ρ −0.046 [−0.061,
  −0.031]**, human → *E. coli* **−0.016 [−0.032, +0.001]** (`results/kingdom_transfer_boot.csv`, 2,000 test-row
  resamples, champion settings both sides). With flanking context built on both sides the sign flips to **+0.022**
  and **+0.017**, flanks alone **+0.042** and **+0.108** (`Jacky/human-boundary/results/family_a_kingdom_transfer.csv`);
  residualising GC, GC-range matching and histogram matching leave the shared-column figure at ≈−0.049, so the fix
  is the representation, not the covariate. **The human-side figures are on row-wise folds of a matrix in which 8,499
  of 17,421 rows (49%) repeat a guide that appears elsewhere, and have not been re-run under grouped folds**; on
  grouped folds the human champion reaches **0.3377** against 0.7314 here, and the matrix needs deduplication first.
- **The label is survival, not cutting.** Measured efficiency is **P(cut) × P(the break is unsurvivable)**, every model
  here is fitted to the product, and no experiment separates the terms; the dCas9 screens did not (Appendix B5).
  RecBCD attenuation at Chi sites, the one repair route testable from sequence, is excluded: Chi sits at the 41st
  percentile of 200 GC-matched controls (Appendix B).
- **The label is crisprHAL 2's curated score, which is a fitted procedure.** It is **bit-identical** to their released
  `y` on all 33,567 guides, so the head-to-head compares two models on one quantity; but the two cohorts' labels are
  different quantities (Noshay's `cut_score` runs 0 to 48.05, crisprHAL's −9.05 to +6.21 with 47.5% negative; rank
  agreement ρ 0.945 on 8,617 overlapping guides, same sign for target GC, −0.147 against −0.201), so every number
  belongs to one label or the other. The filter is an average control read count with cutoff **73** for the SpCas9 set
  (31 for eSpCas9, 56 for *C. rodentium*), **chosen by cross-validation for model performance** and applied
  **before** the split, so it is a hyperparameter and cleans the test set too, raising ρ partly by making evaluation
  easier. At fixed rows a label swap is worth only +0.004; the family table grew because the flank-GC slope is about
  **3× steeper** outside Guo's 1,222-gene panel (Appendix B2a). Further curation does not help: dropping the noisiest
  decile of training guides costs **−0.0016** and a *random* decile **−0.0014**; inverse-variance weighting costs
  **−0.0050**, of which −0.0041 is reproduced by permuting the weights, so the cost is effective sample size
  (Appendix B). Curation buys a cleaner label (ceiling 0.946 against 0.937; +0.022 of fit against the raw log-ratio).
- **No specificity claim.** Six of `h_offtarget`'s thirteen columns (`mm1`, `mm2`, `seed_exact_mm1..4`) are zero for
  every guide because the cohort keeps only uniquely-targeting guides, so the family's +0.0009 is not evidence that
  off-target descriptors are uninformative; a cohort retaining multi-targeting guides is needed.
- **No nickase claim.** Base and prime editors bind and nick rather than cut, which puts them between cutting and
  binding-only assays; this needs an editing-outcome readout. Nuclease-information analysis was removed as having no
  substance for this project (`REMOVED.md`).

### Table of limits

**Addressable from data already on disk.**

| limit | why it binds | what would move it |
|---|---|---|
| Interpretation is conditioned on a redundancy-blind selector | the 300 columns are the top *k* of a gain ranking, and gain over-weights binary columns by 14 points of composition; "45 columns carry half the gain" and every family increment inherit this | **not the selector, tested:** a redundancy-penalised rule costs −0.0023 and leaves a *more* binary set (Appendix B); the fix is block-level attribution (Finding 5) |
| The label is P(cut) × P(the break kills) | survival is the only readout | partial: Cui's screen carries **two time points** (18 and 75 generations); their difference is a depletion-rate signal an endpoint cannot give; dCas9 cohort only (Appendix B5) |
| The gene effect is removed by design, not modelled | folds are grouped by gene; Yu et al. (2024) fit a mixed-effect model with gene as a random term (Part 5) | **tested, does not bind** (`EXTENSION_LOG.md` §18.1): shrunk per-gene intercepts move accuracy from between-gene to within-gene (−0.0080 / +0.0016) and leave pooled ρ at −0.0003; between-gene ρ is 0.729, so there is little nuisance to absorb |

**Addressable, but only with compute not spent.**

| limit | why it binds | what would move it |
|---|---|---|
| "Degraded by wider input" is a claim about their *published configuration* | their epochs, dropout, filters and GRU width were all chosen at 378 nt | 4–6 GPU-hours: `colab/crisprhal_tune_by_width.py`, anchored on the 378 nt positive control, interpretation pre-registered both ways |
| **The lead is a product-level claim (both models as shipped), not an architecture claim** | SLICER's settings were searched (200 draws plus a five-family mix; 97–157 draws per fold on our folds); crisprHAL 2 is used as released. A nested search of its own on our folds found nothing better than its published configuration in four of five folds, and a comparable small search bought SLICER +0.0103, but at 5–8 draws, not at the present scale | a matched search on its own splits: `colab/crisprhal_retune_own_splits.py` (selects on an inner split of their *training* file, scores their test file once, smoke-tested on CPU, not yet run on a GPU; 8–10 T4-hours at 40 draws for SpCas9, about three times that for eSpCas9, so a null from it is a lower bound). The as-shipped comparison does not depend on it |

**Not addressable without new experiments.**

| limit | why it binds | what it would take |
|---|---|---|
| **~0.21 ρ of repeatable variance is unexplained** (Part 2, item 9) | eleven families from genome, thermodynamics, ChIP-seq and quantum chemistry have not closed it, and capacity is not the cause (tuning: +0.016 with family A, **+0.003** on the base representation; Finding 4) | not more features of the kinds tried and not a better learner: the sequence-derived candidates with a mechanism are *excluded* (shape and 3D path, Appendix B4; periodicity; Chi; supercoiling), so this is a measurement problem |
| The long-range mechanism is unidentified | the dCas9 screens do not settle whether it needs a break (Appendix B5); orientation is a live candidate with one ruled-out alternative; GC skew and helical periodicity are a null | an intervention: a designed library varying flank composition at a **fixed guide**; in every existing pooled screen guide and locus are confounded by construction |
| No untouched external set remains **for the cutting claims at the same loci** | eSpCas9 proved confirmatory and Cui's screen is spent. Sealing 20% of genes over **fifteen draws** puts the automatable adaptive component at **+0.0087 ρ [+0.0020, +0.0153]**, a *lower* bound, landing on differences not levels (family A **+0.164** in cross-validation, **+0.156** on unseen genes); on a sealed genome (*C. rodentium*) the tuned settings' benefit is intact at **+0.0172** over the frozen champion | permanent on the cutting side: Rousset and Wang are untouched but are CRISPRi screens (Appendix B5). The human part of the adaptive component and the cohort choice are unquantified; the margin over crisprHAL 2 on our folds is read the same way, and the parity claim on their splits is untouched (Part 4) |

## Part 4 — The freeze, and what remains

**The analysis is frozen as of 11 October 2026: no model, feature or hyperparameter choice is made against the curated
five-fold split.** Settings chosen afterwards were chosen on the tuning part (70% of genes) of a gene split and judged on the
held-out 30% or on other data (Appendix B). A guide-design tool is not a deliverable: the findings are the contribution, and
`featurise.matrix()` turns any 20-mer into the published representation if one is built later. Importing the package is
side-effect free: `paper/import_check.py` asserts that every module imports and that importing changes nothing under `results/`.

### Adaptive overfitting is measured, and it is small

**The curated five-fold figures carry an adaptive-overfitting component of about +0.009 ρ [+0.0020, +0.0153] from the
automatable part of the pipeline; the human choices above it are unquantified, and one of them, the cohort, cannot be.** It
falls on differences between arms rather than on levels: family A reads +0.164 in cross-validation and +0.156 on sealed genes,
and the paper should quote the sealed figure or both.

**Most of the pipeline is not adaptive.** Every decision that could have inflated the five-fold figures, with where it was
made:

| decision | chosen where | adaptive w.r.t. the curated folds? | effect |
|---|---|---|---|
| 300 columns; champion XGBoost settings; gain selector | inherited from the baseline script (`N_FEATURES = 300`), not searched on either cohort | no | none: ±0.0005 against every column; 0.0002 against LightGBM at its own defaults; selector 0.7033 against 0.7025 for LightGBM |
| family A's existence | designed on the published cohort, from residuals computed there | no (different cohort) | it is the effect |
| family A's window set (`config.FLANK_WINDOWS`: 50/250/500/1,000) | swept on the published cohort | no, but a sweep winner carries its own curse | none: rebuilt at 75/300/600/1,200 nt, sharing no value, **+0.1634 against +0.1638** (0.0005 against the ±0.005 declared in advance) |
| per-fold hyperparameter search (627 draws) | curated cohort, nested inside each training fold | no: it cannot see a validation row | +0.0131 gain, leak-free by construction |
| the 412/427-column reduced set | published cohort | n/a: it does not transfer (Finding 1) | none |
| ρ as primary metric | crisprHAL 2's reporting convention | no | nine-metric panel in `docs/qna_archive.md` |
| **the choice of cohort** | **here, partly because curation improved the score** (ρ 0.607 → 0.628 on shared rows) | **yes** | **unquantified** |

The settings a reviewer would suspect first were inherited and are insensitive, and the search that was run is nested.
The cohort choice is adaptive and cannot be undone: every figure is computed on the rows it selected, and the alternative
cohort is a different, smaller, noisier draw rather than a held-out sample of the same thing.

**The automatable part is +0.009 ρ, not zero.** Twenty per cent of the cohort is sealed by whole gene
(`sealed_holdout.py`); selection and hyperparameters are re-derived inside the development pool, and each arm is scored once
on the sealed genes. A reference arm whose configuration was not chosen on these rows absorbs the population difference, and
the adaptive component is the difference of differences. Fifteen sealing draws, training sizes matched so the sealed fit sees
no more data than a CV fold:

| | mean | sd across draws | 95% CI |
|---|---:|---:|---|
| reference gap (nothing chosen on these rows) | −0.0121 | 0.0148 | [−0.0196, −0.0047] |
| frozen: the configuration the report quotes | −0.0035 | 0.0077 | [−0.0074, +0.0004] |
| **adaptive component, frozen** | **+0.0087** | 0.0131 | **[+0.0020, +0.0153]** |
| adaptive component, dev-tuned | +0.0082 | 0.0140 | [+0.0011, +0.0153] |

Twelve of the fifteen draws are positive (t = 2.56, p = 0.022; Wilcoxon p = 0.015; sign test p = 0.035; 20,000-draw bootstrap
[+0.0024, +0.0152]), and dropping any single draw moves the mean only between +0.0065 and +0.0106. The reference gap itself
swings with sd 0.0148 between sealings, which is why fewer draws cannot resolve an effect of this size.

**The levels are not detectably inflated; the arm differences are.** The frozen arm scores 0.7043 in pool CV and 0.7077 on
sealed genes (gap −0.0035, interval contains zero). The base-only arm, on which nothing was chosen, moves from 0.5396 to 0.5517
(+0.0121), so the family-A gain reads **0.1647 in pool CV and 0.1560 on sealed genes (−0.0087)**: optimism cancels between
arms only when both were chosen on the same rows, and these were not.

The same reasoning applies to the margin over crisprHAL 2 on our own five folds (our configuration was chosen on those rows,
theirs was not): +0.0161 to +0.0172 carries up to this much self-inflation, leaving roughly **+0.008** on unseen genes. That
is consistent with the +0.002 to +0.005 margin on their held-out splits and is a second contributor beside the data-volume
account in Part 5. The head-to-head on crisprHAL 2's own splits is the cleaner comparison: no weights of either model were fitted on a test row, nothing was selected on the eSpCas9 split, and the CONFIRM genes are the clean check for the WT-SpCas9 split (Part 5).

*Limit.* +0.0087 is a lower bound on total adaptivity: no re-derivation reaches the human choices (building family A, the
window set, the cohort), and the sealed-organism check below reaches the settings but not these.

### For the settings chosen after the freeze, adaptive overfitting is not an issue

**Their benefit holds on 1,204 genes untouched by any search, on a second gene split, and on a genome no choice ever saw
(+0.0172).** Which data could have been sealed depends on what each was used for:

| data | used for | anything chosen on it? | at loci the *E. coli* cohort does not hold? |
|---|---|---|---|
| Noshay's published matrix, 13,880 guides (same screen as the curated cohort; label ρ 0.945 on 8,617 shared guides) | Finding 1, the 412-column result, the harness anchor, family A's design and window sweep | **yes**: family A's existence and windows | no |
| crisprHAL curated WT-SpCas9, 33,567 guides | the working cohort: every model and figure | **yes**: the cohort itself; hyperparameters (on 70% of its genes) | n/a |
| eSpCas9, 59,489 guides (their split 47,591 / 11,898) | transfer; the head-to-head on their split; an external check during tuning | nothing selected, but scores were visible: confirmatory | no |
| *C. rodentium* TevSpCas9, 25,210 guides (5.35 Mb genome) | the frozen champion's transfer test (Finding 3); the sealed check below | **no** | **yes** |
| human matrix, 17,421 rows (13,138 distinct) | kingdom transfer; the design-rule inversion (Finding 7) | human-specific analyses only; its row-wise folds leaked duplicate guides (Part 2) | yes |
| three *E. coli* CRISPRi screens (Cui, Rousset, Wang) | Appendix B5 only | no model or setting | partly |

Not all of it was needed: the CRISPRi screens underpin no claim, and they cannot seal a cutting claim because they measure
another quantity. The first three rows are one screen at one set of loci, so holding one out as a dataset leaves the loci in,
and flank composition is a deterministic function of locus. The seal that counts within *E. coli* is by gene and by block
(sealed genes: +0.0087 of adaptive component; 1,204 CONFIRM genes: a +0.0324 lead over crisprHAL 2's published configuration).
That leaves one dataset independent of the loci and never chosen on, *C. rodentium*. eSpCas9 and crisprHAL 2's shipped
SpCas9 split (which has carried seven arms: SLICER's original 0.6997 and six data-ablation arms) were never selected on, so
they are confirmatory but not untouched; no untouched external set remains at the same loci.

**The sealed check.** Train on all 33,567 *E. coli* guides and score the 25,210 *C. rodentium* guides zero-shot, with the same
protocol for each arm (`sealed_organism.py`; three seeds; 16 blocks of 100 kb resampled for the interval). Predicted in
advance: tuned at least as good as the champion, by less than within *E. coli* (+0.014 to +0.026). Observed: at least as good,
and by as much.

| arm | ρ on *C. rodentium*, mean of 3 seeds (sd) | minus champion, seed-averaged predictions [95% over blocks] |
|---|---:|---|
| frozen champion | 0.6986 (0.0038) | |
| tuned XGBoost (d099x) | **0.7160** (0.0014) | **+0.0172 [+0.0162, +0.0206]** |
| XGBoost average of 5 | 0.7170 (0.0012) | +0.0165 [+0.0154, +0.0195] |

The tuning's benefit is the same size as inside *E. coli* (+0.014 on genes it never saw; +0.019 and +0.026 on crisprHAL 2's
two splits), so the settings are not specific to the data they were chosen on. The champion's 0.6986 reproduces the transfer
figure of Finding 3 (0.698), so the protocol is the same one.

*Limit.* The check does not reach the cohort choice or family A's design (both made on *E. coli*; the transfer shows the
effect crosses to another genome, not that the choice was unbiased), the question of equal tuning against crisprHAL 2 (Part 5), or the
opposite direction, which would score the genes the settings were chosen on. For a next organism: seal first, seal by gene, and
keep a whole genome back.

### What remains open

Each question is open because of something this project does not have.

| question | what would answer it | why it is open |
|---|---|---|
| Does the lead survive equal tuning of crisprHAL 2? (a method-level question about the architecture) | their search on their splits, about 40 GPU-hours at our scale (`colab/crisprhal_retune_own_splits.py`, estimated 8–10 T4-hours for 40 draws on one split) | not run. The as-shipped comparison in Part 5 stands either way; this is the arm that would let the claim be about the architecture |
| Would re-tuning crisprHAL 2's other hyperparameters at each input width change the width result? | `colab/crisprhal_tune_by_width.py`, anchored on the 378 nt positive control; 4–6 GPU-h | not run. Browne et al.'s own per-dataset length search stopped at 378 nt (*E. coli* SpCas9) and 406 nt (eSpCas9) with gains levelling off (+0.00034 ρ per upstream nucleotide, +0.00074 per downstream one; 0.554 on the sgRNA alone to 0.697 at 378 nt), so a change is unlikely |
| Does measured nucleoid-protein occupancy add anything beyond family A? (H-NS/Fis ChIP, GSE24991; IHF/HU, SRP008538; IPOD-HR, GSE164796) | one processed track mapped to the 33,567 loci, plus one ablation arm | no processed track was fetched. Bounded by argument, not measured: Hi-C-derived family E is null, the flank effect has no structure below ~250 nt (Finding 7), and coarse NAP occupancy is AT-driven, which family A's composition columns carry |
| Would a measured DNA shape or stiffness matrix differ from the tabulated pentamer means? | ABC/muABC tetranucleotide stiffness (Pasi et al. 2014) | no download page was found; the local-predictor half is closed by measurement (`EXTENSION_LOG.md` §22.3) |
| How much of the unexplained ρ is repair rather than cutting? | an assay separating cutting from survival for the same guide | no pooled survival screen can provide it; this is the honest form of the 0.215 ρ gap |
| Does the lead hold on an untouched cutting screen? | a new, never-used *E. coli* cutting screen | none remains at the same loci; a different genome carries the sealed check on the tuned settings (+0.0172). The dimension on which the project is furthest behind the field |
| Why do the 1,222 panel genes differ from the rest? | per-guide read counts, to separate dynamic range from different biology | the counts were not used; the essentiality explanation is argued against (family G's essentiality column is worth +0.00004) |
| Do the human-arm results hold on a deduplicated matrix, and does the human shape result (+0.018 on top of family A, where *E. coli* gives zero) mean anything? | the 13,138-guide deduplicated matrix searched under twin-grouped folds; human family A rebuilt on our own hg38 flanks | the cross-kingdom transfer figures (shared-column: no transfer in either direction, [−0.023, +0.011]; family-A-on-both-sides, which reverses it) are on row-wise folds and not re-run; human runs are outside the current scope |

## Part 5 — Where this sits in the field

Performance numbers from different cohorts, labels and metrics are not commensurable, so each comparison below states what is held equal. The comparator is crisprHAL 2 (Browne et al., 2026), the strongest published predictor of bacterial Cas9 cutting known to us. A literature search on 22 November 2026 found no newer bacterial cutting model; the nearest neighbours measure CRISPRi depletion, base-editing fitness, or SaCas9 PAM preference (the crisprHAL group's own, *Nucleic Acids Research* 2026, doi:10.1093/nar/gkaf1520), and a search that fails to find something is weak evidence that nothing exists.

### The head-to-head with crisprHAL 2

**The final SLICER model, as shipped, leads crisprHAL 2, as released, by +0.023 to +0.029 ρ on both of its authors' own splits, at a fraction of the cost.** Both are used as released in practice, so that is the comparison: SLICER with its settings chosen on held-out genes (XGBoost over the base representation plus family A, Finding 4) and crisprHAL 2 with its published weights. Before tuning, the same model was statistically equivalent to it. The result depends on what is held equal, so four conditions are reported; a single headline number would be a choice of condition dressed as a finding.

| condition | what is held equal | result for the final model |
|---|---|---|
| **their two splits, their published weights** | rows, label, split, and each model as its authors ship it | **SLICER +0.0286** [+0.0239, +0.0331] on eSpCas9 and **+0.0226** [+0.0139, +0.0314] on WT-SpCas9 |
| **our five folds, both tuned here** | rows, label, folds, and a search run for both | **SLICER +0.0288** [+0.0251, +0.0326] against their published configuration, **+0.0216** [+0.0180, +0.0252] against the best configuration found for them |
| **equal input DNA** | all of the above *plus* the DNA each model reads | **depends on the width**: crisprHAL +0.023 at 378 nt, SLICER +0.065 at ±1,000 nt (original model) |
| **a different assay, neither model trained on it** | rows, label, and that neither has seen the data | **crisprHAL +0.026** [+0.023, +0.029] (original model; not like-for-like, see the last subsection) |

**What the comparison claims.** A product-level claim: used as released, SLICER scores higher on crisprHAL 2's own test splits. Published predictors are compared as released in this field's benchmarks (Konstantakos et al., 2022, ran published tools through their pre-trained models and stand-alone packages and retrained only where a tool could not be reproduced), and it is what a user experiences. It is not, by itself, a claim that boosted trees beat convolutional-recurrent networks: SLICER's settings were searched and crisprHAL 2's were not, and general guidance on method comparisons asks for equal tuning budgets before an architecture is credited (Lones, 2021). Two things bound that reading: a nested search of crisprHAL 2's own on our folds found nothing better than its published configuration in four of five folds (Appendix B), and `colab/crisprhal_retune_own_splits.py`, which would supply the matched search on its own splits, has not been run.

#### The final model leads on both of crisprHAL 2's splits

**With hyperparameters chosen on 70% of the genes (DEV) and confirmed on the other 30% (CONFIRM), SLICER leads crisprHAL 2's released weights by +0.0286 ρ [+0.0239, +0.0331] on its authors' eSpCas9 split and +0.0226 [+0.0139, +0.0314] on their SpCas9 split, with τ, Pearson and R² clear in the same direction and the two top-end metrics tied.** Every number in this subsection is audited in `sgrna/paper/verify_numbers.py`; the working record is `EXTENSION_LOG.md` §13. The equivalence of the frozen model before tuning is therefore a property of its settings, not of the model class.

**Tuning design.** Settings were selected on the DEV genes (70%) of the 33,567-guide WT-SpCas9 cohort. The CONFIRM set is the remaining 1,204 genes (8,943 guides), untouched by any search; SLICER trained on the DEV genes only (21,260 guides) gains **+0.014 [+0.011, +0.017]** over the frozen champion there. No model's weights were fitted on a test row, and each SLICER arm trains on the split's training rows only, with selection inside them. The eSpCas9 split is a different cohort with nothing selected on it; the WT-SpCas9 split and our five folds hold the same guides as the DEV selection, so their labels fed the choice of settings (not any weights), which is why the CONFIRM row is the clean check for them. The lead replicates across all of them:

| paired on guides, 95% bootstrap CI | crisprHAL 2 ρ | frozen champion Δρ | tuned ρ | **tuned Δρ** |
|---|---:|---:|---:|---:|
| their eSpCas9 split | 0.7821 | +0.0021 [−0.0031, +0.0072] | 0.8107 | **+0.0286 [+0.0239, +0.0331]** |
| their WT-SpCas9 split | 0.6945 | +0.0039 [−0.0047, +0.0124] | 0.7172 | **+0.0226 [+0.0139, +0.0314]** |
| our five folds vs their published configuration | 0.6968 | +0.0115 | 0.7256 | **+0.0288 [+0.0251, +0.0326]** |
| our five folds vs the best configuration found for them | 0.7040 | +0.0043 | 0.7256 | **+0.0216 [+0.0180, +0.0252]** |
| CONFIRM genes vs their published configuration | 0.6892 | +0.0181 [+0.0098, +0.0265] | 0.7215 | **+0.0324 [+0.0248, +0.0403]** |
| CONFIRM genes vs the best configuration found for them | 0.6976 | +0.0097 [+0.0018, +0.0179] | 0.7215 | **+0.0240 [+0.0162, +0.0318]** |

On the CONFIRM rows crisprHAL 2 is at its stored out-of-fold predictions, trained on 80% of all guides by random folds (including guides from the same genes), which favours it, and the bootstrap is over genes; the lead there is larger than on the five folds. The lead is stable across five training seeds (selection and trees refitted): +0.0280 to +0.0286 (mean 0.0284) on eSpCas9, +0.0208 to +0.0228 (mean 0.0221) on WT-SpCas9.

**Metric by metric** (tuned minus crisprHAL 2, paired bootstrap over guides): ρ (above), τ, Pearson and R² are clear on both splits and the two top-end metrics are tied.

| | eSpCas9 (11,898 test) | WT-SpCas9 (6,737 test) |
|---|---:|---:|
| Kendall τ | **+0.0301** [+0.0257, +0.0346] | **+0.0221** [+0.0147, +0.0297] |
| Pearson | **+0.0234** [+0.0191, +0.0278] | **+0.0173** [+0.0094, +0.0254] |
| R² | **+0.0435** [+0.0366, +0.0503] | **+0.0285** [+0.0169, +0.0397] |
| top-decile precision | +0.0034 [−0.0151, +0.0286] | +0.0193 [−0.0119, +0.0520] |
| pick percentile of the 10 best-scoring guides | −1.92 [−6.70, +3.93] | −0.44 [−10.35, +9.73] |

The pick percentile rests on ten guides (intervals of about ±10 percentile points). Precision of the top 5% is +0.019 [−0.017, +0.049] on eSpCas9 and **+0.048 [+0.003, +0.074]** on WT-SpCas9, and the mean percentile of the 50 and 100 best-scoring guides is tied on both. Objectives aimed at the top do not help: squared error gives top-decile precision 0.2912 on the held-out genes, weighting high-activity guides 0.2852 (weight 3) and 0.2890 (weight 9), pairwise and NDCG ranking objectives reach ρ 0.5758 and 0.5119 (0.6017 at best) with lower top-end precision. Averaging the tuned model with crisprHAL 2's shipped predictions at equal weight adds nothing on eSpCas9 (0.8099, −0.0008 [−0.0030, +0.0015]) and **+0.0085 [+0.0040, +0.0127]** on WT-SpCas9 (0.7257); the two models' residuals correlate at 0.83 and 0.81, so the network contributes only at the smaller training size.

**Data efficiency** (mean ρ over the five folds, same subsets as the stored curve; crisprHAL 2 at seed 41):

| training guides | 4,000 | 8,000 | 13,880 | 20,000 | 26,853 |
|---|---:|---:|---:|---:|---:|
| crisprHAL 2 | 0.5235 | 0.6323 | 0.6733 | 0.6879 | 0.6950 |
| frozen champion | 0.6524 | 0.6814 | 0.6988 | 0.7053 | 0.7076 |
| **tuned XGBoost** | **0.6576** | **0.6915** | **0.7106** | **0.7198** | **0.7251** |

The tuned model is ahead on every fold at every size (leads +0.134, +0.059, +0.037, +0.032, +0.030). With 8,000 guides it scores 0.6915, within 0.004 of crisprHAL 2 trained on 26,853, and by interpolation reaches that figure at about 9,100 guides, **about three times fewer labels**.

#### Before tuning, the same model was equivalent to crisprHAL 2 at a fraction of the cost

**On crisprHAL 2's own two splits, against its shipped weights, the frozen champion is equivalent to it, measured as an interval rather than as a failure to find a difference.** On the 11,898-guide eSpCas9 test file (`final_eval.py`) its arm reproduces the published figures (ρ 0.782, Pearson 0.780; 0.7821 and 0.7801 here), so the comparison is with their model and not a re-implementation of it.

| their split (`final_eval_espcas9.csv`, `their_split_slicer.csv`) | crisprHAL 2, published ρ (shipped weights) | frozen SLICER | paired Δρ vs shipped weights [95% CI] |
|---|---:|---:|---:|
| eSpCas9 (47,591 train / 11,898 test) | 0.782 (0.7821) | 0.7842 | +0.0021 [−0.0031, +0.0072] |
| WT-SpCas9 (26,830 train / 6,737 test) | 0.695 (0.6945) | 0.6997 (re-derived 0.6985) | +0.0039 [−0.0047, +0.0124] |

The eSpCas9 interval supports equivalence at any margin from 0.0072 upward; no margin was pre-declared, so the interval is reported in place of a verdict (Appendix A audits the dates). On eSpCas9 SLICER leads on ρ (0.7842 against 0.7821) and τ (0.5942 against 0.5920), crisprHAL 2 on Pearson (0.7801 against 0.7782), R² (0.6009 against 0.5979), top-decile precision (0.494 against 0.488) and expected pick percentile (95.5 against 94.8); **all six differences have intervals containing zero** (Appendix A). On our five folds (33,567 out-of-fold predictions per model, paired on guides, `heldout_compare.py`; per-fold figures at seed 41 in `results/folds_ladder_base_full+a_flank.csv` and `results/crisprhal_rerun.csv`) the frozen model leads in five of five folds, by +0.0115 [+0.0074, +0.0157] against crisprHAL 2's published configuration and +0.0043 [+0.0005, +0.0080] against the best configuration found for it. The margin shrinks with training size (26,853 guides per fold on our folds, 47,591 on their eSpCas9 split), as the data-efficiency curves predict, so at 47.6k labelled guides the frozen model does not beat crisprHAL 2.

Search effort differs by construction: SLICER's settings were searched (97–157 draws per fold on our folds, 627 in total) and crisprHAL 2 is used as released. Its published configuration won the inner search in four of five folds in each of two independent nested searches (0.6987 and 0.6989 against 0.7218 for ours, paired +0.0225, 5/5 folds, p < 0.001), the best configuration found by any search lifts it only from 0.6971 to 0.7051, and a comparable small search (12 draws per fold) buys our model +0.0103 and theirs nothing. `tune.DEFAULT`, the "default" arm on our side, is a configuration inherited from the starting baseline, not LightGBM's defaults, while crisprHAL 2's arm is its published configuration (Appendix B).

**The cost and data-efficiency claims are therefore the ones that survive every condition for the frozen model:** roughly 150× less CPU time to fit and about half the labelled guides to reach a given accuracy, with 125× fewer fitted numbers (cost table below). Under equal input DNA its ranking depends on the window (next subsection).

#### Equal input DNA: their convolutions win inside their own span, ours wins past it

**Given the same DNA, the ranking depends on the window: crisprHAL 2 is better inside the 378 nt it was designed for, and SLICER is better once the window reaches the 250–1,000 nt its input cannot see.** Same fold, original model (Finding 2):

| DNA both models see | SLICER | crisprHAL 2 | lead |
|---|---:|---:|---|
| guide + PAM only | 0.5473 | 0.5494 | tied (0.002) |
| their 378 nt span | 0.6568 | 0.6799 | crisprHAL **+0.023** |
| ±1,000 nt | **0.6944** | 0.6299 | **SLICER +0.065** |

Limit: crisprHAL 2 was retrained at each width with its published configuration, whose epochs, dropout, filters and GRU width were chosen at 378 nt, on one fold and one seed. The result supports "the published configuration is degraded by input wider than it was tuned for", not that convolutional-recurrent networks cannot use long-range context (Finding 2, Appendix A).

#### Cost: fitting, data, scoring and size

**SLICER is cheaper on every axis measured: about 150× less CPU time to fit, about half to a third of the labels, a model 125× smaller, and 13× to 34× faster scoring end to end.**

| quantity (one thread unless stated) | SLICER | crisprHAL 2 | ratio |
|---|---|---|---:|
| fitted numbers | 6,348 leaf values over 400 trees; 18,244 counting each node's column and threshold (`complexity.py`, `results/complexity.json`) | 790,533 trainable parameters | 125× (43×) |
| CPU time to fit, per fold (`cost_panel.csv`) | 65.8 CPU-s (frozen); on 47.6k guides 2 s frozen and 30 s tuned, 4,000 trees (`cost_bench.py`) | 10,000 CPU-s | about 150× (45–113× wall-clock) |
| labelled guides to reach ρ ≥ 0.62 / 0.66 / 0.68 / 0.70 (`sample_complexity.py`) | 4,000 / 8,000 / 8,000 / 20,000 (frozen) | 8,000 / 13,880 / 20,000 / not reached on 26,853 | about 2×; 2.5× at 0.68 |
| guides to reach crisprHAL 2's 26,853-guide ρ (0.6950) | about 9,100 (tuned) | 26,853 | about 3× |
| forward pass alone, per guide, batch 100, laptop (`results/inference_onecore_local.csv`) | 0.002 ms frozen; 0.018 ms tuned | 1.32 ms | 329×–817× frozen over 1 to 10,000 guides (177×–344× on a Colab vCPU; `colab/inference_bench_onecore.py`, `results/inference_onecore.csv`) |
| **end to end, per guide, 1 / 100 / 10,000 guides** | tuned 0.72 / 0.089 / 0.093 ms | 15.29 / 1.365 / 1.214 ms | **21× / 15× / 13×** |
| **end to end, every NGG site in a 100 kb stretch** (`flank_locus.py`) | tuned 0.0361 ms; frozen 0.0223 ms | 1.214 ms | **34×** (54× frozen) |

Three readings of this table. (1) CPU time is a statement about the hardware, not a law, and there is no architecture-independent measure of cost; the one that is architecture-free is sample complexity, which is why the data-efficiency rows are the defensible form of the claim. (2) Forward-pass ratios omit featurisation, which the end-to-end rows include: the array featurisers `a_flank_fast.py` and `base_fast.py` give columns identical to the stored matrices on all 33,567 guides (maximum absolute difference 0.0; 0.068 ms per guide for family A and 0.019 ms for the base), and `flank_locus.py` scans a stretch once into cumulative tracks so each window statistic is a subtraction (largest relative difference 4.6 × 10⁻¹²; `EXTENSION_LOG.md` §22.1). The ratio is not an artefact of optimising one side: 95% of crisprHAL 2's 1.214 ms is its forward pass (1.157 ms), so the same trick could remove at most 4.7%. The 34× is a statement about designing guides across a locus: the precompute is paid per nucleotide, so at the curated cohort's density (one guide per 138 nt) it gives only 3–5×, and at 1 Mb the 2.95 GB feature matrix means a design run should chunk at tens of kilobases. (3) On all ten laptop cores the forward-pass factor is about 1,700 on throughput (`inference_cost.py`, `results/inference_cost.csv`), not the headline because the two models use ten cores very differently; on a T4 GPU it is 15× to 20× with input on the host and 12× to 110× with input on the card (same files), because a 400-tree depth-4 ensemble has little use for the card; cold start is 6.0 s (CPU) and 9.9 s (GPU) for crisprHAL 2 against 1.4–1.9 s for SLICER (`results/inference_colab.csv`).

**One tuned model is worth its cost; averaging or mixing models is not.** The tuned model has ten times the trees of the champion (0.018 ms against 0.002 ms per guide) and carries nearly all the gain. Averaging five XGBoost models adds +0.003 on the held-out genes and nothing on eSpCas9 (0.8104 against 0.8107); mixing nine models from three libraries adds +0.0014 [+0.0005, +0.0025] over that average and is lower than the single model on eSpCas9 (0.8095), for 24,800 trees and a separate process for the neural network.

**Size is not capacity.** The 125× is the size of the stored model at equal accuracy on the two splits where accuracies are equivalent; a parameter count is a loose proxy for capacity in both model classes, SLICER's count omits the choice of which 300 of 6,517 columns to offer, and depth 4 was inherited rather than chosen (Appendix B). It supports only that no part of the accuracy requires a model of the size the field has been fitting.

#### Other comparisons and boundary conditions

**Against Noshay et al.'s iRF on its own cohort SLICER leads by +0.128 ρ (0.6060 against 0.4785; 13,880 guides, `cut_score`, five folds)**, of which +0.049 is the model class (champion XGBoost on the identical 6,232 columns reaches 0.5278) and +0.078 is family A. The iRF is our reimplementation (`noshay_replicate.py`) and reproduces their published figures (Pearson 0.488 and R² 0.235 against 0.502 and 0.249). On crisprHAL 2's cohort the same comparison is +0.023 to +0.029 for the final model (equivalent before tuning), so a paper quoting only the Noshay margin overstates the contribution.

**Parity depends on the training rows.** With the architecture fixed and only the training data varied, scored on crisprHAL 2's WT-SpCas9 test partition (6,737 of its 6,749 guides locate in our feature build), SLICER reaches 0.6537 on Noshay's 1,222-gene panel with Noshay's `cut_score` (n = 6,860), 0.6600 on the same rows with the curated label, 0.6683 on count-matched random curated rows, and 0.6985 on all 26,830 curated rows (`data_ablation.py`, `results/data_ablation.csv`). Of the +0.0448 between the first and last, the label is worth +0.0063, which rows +0.0083 and how many rows +0.0302. Family A is worth +0.1554 on the same partition at full data, more than the whole move between cohorts. The Noshay-panel row is 0.041 below crisprHAL 2's published 0.695, but crisprHAL 2 was trained on all curated rows throughout, so the comparison is SLICER with less data against crisprHAL 2 with all of it and does not show how crisprHAL 2 would fare on Noshay's rows.

**On a different assay, crisprHAL 2 transfers better.** Zero-shot on Cui's CRISPRi screen, |ρ| is 0.0867 for crisprHAL 2 against 0.0609 for SLICER, a paired difference of −0.0258 [−0.0289, −0.0227] for SLICER (`cui_final_exam.py`); this is a boundary of the claim and not a like-for-like comparison (assay, library and laboratory all differ), both values are small against the 0.458 reached by a model trained on that screen, and a plausible reason is that transferable dCas9 signal lies in guide-sequence effects, which is their input, while SLICER's advantage is flanking DNA (a hypothesis; the CRISPRi screens are Appendix B5).

**Structure-aware inputs do not change the comparison.** Adding tensor-valued 3D descriptors of the flank (three table sources, up to 1,887 extra columns) changes the champion's ρ by −0.0008 to +0.0006 on the curated cohort and by +0.0022 on crisprHAL's eSpCas9 screen, where a flank shuffled to destroy geometry gives +0.0016; the descriptors are functions of window composition and of the first ten or so base pairs, which family A already holds (Appendix B4). Not tested: the same descriptors restricted to crisprHAL 2's 378 nt span, where its convolutions lead.

### What other predictors read, and how they are validated

**No published predictor reads flanking DNA beyond 406 nt; SLICER reads ±1,000 nt, and that reach is the one large effect.** The flank
features add **+0.164 ρ** (curated base 0.545 → 0.709) and, on crisprHAL's held-out partition, **+0.155** (base-only 0.543 → 0.698):
the distance between a model that does not match the state of the art and one that does. On the decision scale this moves the
expected pick from roughly the 70th to the 81st percentile of ten candidates (see the head-to-head above).

| predictor | organism / assay | input beyond the 20-mer | model | split | reported performance |
|---|---|---|---|---|---|
| Noshay et al. 2023 (QCT + iRF) | *E. coli* and human, cutting | none; all columns are functions of the guide (Finding 1) | iterative random forest | random rows | Pearson 0.502, R² 0.249 (13,880 guides) |
| crisprHAL (Ham et al. 2023) | *E. coli*, SpCas9 / TevSpCas9 | **37 nt**: 3 upstream, 20 guide, 14 downstream | CNN–BiGRU, transfer learning | held-out split | n/a |
| crisprHAL 2 (Browne et al. 2026) | *E. coli*, SpCas9 and eSpCas9 | **378 nt** (SpCas9), 406 nt (eSpCas9), set by their own length search | CNN–BiGRU | their own two splits | ρ 0.695 (SpCas9), 0.782 (eSpCas9) |
| **SLICER, frozen champion** | *E. coli*, cutting | **±1,000 nt** windowed composition | XGBoost, 300 columns | **gene-grouped** 5×5 | ρ 0.7218 CV; 0.6997 / 0.7842 on their splits |
| **SLICER, tuned (post-freeze)** | same | same | XGBoost, 450 columns, 4,000 trees | gene-grouped; DEV/CONFIRM genes | ρ 0.7314 CV; 0.7172 / 0.8107 on their splits |
| Yu et al. 2024 | bacteria, **CRISPRi** depletion | gene-level covariates: expression, transcription-unit structure, gene GC, gene length, distance to operon start | mixed-effect random forest (129 guide features fixed, 9 gene features random, 301 genes as clusters) | **gene-wise** CV, train/test Hamming distances reported | ρ ≈ 0.21 guide-only → 0.25 with thermodynamics → **0.66** with gene features |
| Orro et al. 2026 | human, **base-editing** fitness screen; ~11,000 sgRNAs over 47 genes | **±15 nt** context sweep (total 20–50 nt), gene strand, DepMap essentiality, mutation type, three external editing-efficiency predictors | six classifiers at library defaults | gene-held-out per the published abstract; the preprint's methods describe 5-fold CV × 10 seeds | AUC-ROC 0.838–0.948 (classification, not commensurable with ρ) |
| CRISPRon (Xiang et al. 2021) | human, Cas9 knockout | **experimental chromatin accessibility** | CNN | internal + external test sets | ρ 0.80 internal, ≈0.46–0.68 external |

Other flank readers are shorter still (Yu et al. 4 nt upstream and 3 beyond the PAM; CRISPOR 50 bp each side), and the field's other
routes outside the protospacer are experimental (chromatin accessibility) or gene-level (expression, operon structure, essentiality).
No surveyed predictor builds compositional descriptors of flanking DNA in windows of hundreds to a thousand nucleotides and measures
their increment, and this one needs only a reference genome. A plant-focused study reports no measurable flank effect.

*Independent agreement.* (i) Gene-grouped evaluation is also the practice of Yu et al. (2024) and Orro et al. (2026), and Yu et al.'s
separation of guide and gene effects is the within-gene / between-gene decomposition used here (Finding 2's +0.061, asked entirely
inside genes). (ii) Orro et al.'s SHAP importance rises toward the 3′ end of the guide, peaking at position 19 (0.237) and 20 (0.328),
with positions 1–5 minimal (0.065–0.068); here the eight PAM-proximal positions carry **63% / 62%** of the gain and positions 1–12
only 10% / 17% (`EXTENSION_LOG.md` §14), on a different organism, nuclease, label and importance measure.

*Reference figures.* crisprHAL 2's highest value, 0.754, belongs to crisprHAL Tev on *C. rodentium* (a different organism and nuclease),
so 0.695 and 0.782 are the right figures to compare against. The shipped weights reproduce ρ 0.6945 against the published 0.695; a
35-guide difference between their SpCas9 split (26,795 / 6,700) and the one loaded here (26,830 / 6,737) is unexplained.

### What is new, and what is not

**The paper this work supports is about what determines cutting and how much of it is knowable, not a better predictor.** The
modelling is incremental; the measurement is not.

*Not new.* (a) Boosted trees over engineered features match a deep model: BoostMEC (Zarate et al. 2022) is LightGBM over sgRNA sequence
features, benchmarked against 10 models on 13 external datasets, with far more external validation than is available here. What is
specific to this work is why the tree wins: the representation reaches DNA the network's input does not, a claim about features and
range, not the learner. Finding 4 is the sharpest internal evidence: the tuning gain collapses from +0.016 to +0.003 without the flank
columns. (b) Gene-grouped evaluation (Yu et al. 2024; Orro et al. 2026). (c) That flanking sequence matters at all: crisprHAL 2 is built on
±190 nt and searched the length itself.

*New, as three findings (statements about what determines cutting) and five warnings or methods (things wrong, or measured wrongly, in
the field or in earlier versions of this work).*

1. **Finding 1: a published feature matrix is a lossless re-encoding of the guide, so no deterministic function of the guide can add
   information to it.** No paper found makes the arithmetic argument; it settled several imported biophysical families without running them.
2. **Finding 2, characterisation not existence: the flank effect is compositional, not a motif.** 8% survives a 1 kb shift and none a
   10 kb shift; it does not depend on window edges (+0.1634 against +0.1638 at four widths sharing no value with the originals); its
   reach is 250–1,000 nt against a longest published input of 406 nt.
3. **Finding 7: the effect is GC specifically** (purine is flat at every window and side), **side-symmetric** to within 0.026, with
   **one length scale of 250–500 nt and no structure below ~250 nt** (four columns carry 76% of it; 200 columns score below 32), and
   **coupled to the PAM-proximal seed** (largest pairwise interaction: 250 nt flank GC against position 20). The guide's own
   position × base preferences are **uncorrelated between *E. coli* and human (ρ = −0.052)**, the strongest inverting: G immediately
   5′ of the PAM is the most disfavoured base in *E. coli* and a favoured one in human, which accounts for the failure of
   mammalian-trained predictors here and is a design rule applicable today.
4. **The corrected ceiling** (next subsection; Finding 6): the field's ceiling of ~0.7 is mis-derived and the attainable value is 0.91–0.93.
5. **The supercoiling deconstruction.** A ChIP track worth +0.139 was read density, and read density was gene dosage, recoverable from
   distance to *oriC* with no experimental data: a warning for every coverage-derived feature, including CRISPRon's chromatin tracks.
6. **The duplicate-guide result.** 49% of a widely used published human matrix's rows repeat a guide; under folds that keep them together
   the best-looking configuration is 0.08 worse than the default one (`EXTENSION_LOG.md` §16.2).
7. **The adaptive-overfitting measurement.** +0.0087 ρ [+0.0020, +0.0153] from the automatable part, over fifteen sealed-gene draws,
   and a sealed genome on which the tuned settings' benefit survives (+0.0172). No other guide-prediction paper found measures its own
   selection bias.
8. **The strand warning (Appendix B5, a boundary experiment).** A feature family worth +0.27 in a dCas9 screen is worth +0.007 to
   +0.025 once the assay's own strand rule is supplied: a family containing a lossy copy of an assay's dominant covariate looks
   powerful. Orro et al. (2026) independently find gene strand their most influential non-sequence feature (0.493, ahead of
   essentiality 0.179), so any subgroup result from a pooled screen should control strand first. Whether the effect needs a break was
   not settled.

Items 4, 5 and 8 generalise beyond this project and are the most likely to be useful to others.

### The field's ceiling is mis-derived: attainable ρ is 0.91–0.93, not ~0.7

**A replicate-to-replicate correlation is a reliability, not a ceiling, and treating it as one understates the attainable ρ by 0.21–0.23.**
Konstantakos et al. (2022) set the benchmarking frame: replicate Spearman correlations of 0.71–0.77 are read as showing that ∼0.7 is an
upper limit of any prediction, and ρ 0.4 is the threshold for a successful tool. Two corrections push the same way. The largest correlation any predictor can have with a noisy target is the
**square root** of the target's reliability; and when the target is an *average* over replicates, as essentially every released
activity score is, Spearman–Brown raises its reliability to *kr*/(1+(*k*−1)*r*).

| replicate agreement *r* | ceiling vs one replicate, √*r* | reliability of a 2-replicate mean | **ceiling vs that mean** |
|---:|---:|---:|---:|
| 0.71 | 0.843 | 0.830 | **0.911** |
| 0.77 | 0.877 | 0.870 | **0.933** |
| 0.8097 *(ours, measured)* | 0.900 | 0.895 | **0.946** |

On the review's own numbers the ceiling is **0.91–0.93, not ∼0.7**, understated by **0.21–0.23**, of which **0.14–0.18 is the missing
square root alone**. Finding 6 is the worked example, from crisprHAL's released per-replicate counts; no other ceiling in this
literature was found derived this way. The correction is arithmetic in two numbers a dataset's own paper states, so it does not need
the per-replicate counts of other datasets (`EXTENSION_LOG.md` §22.2, `ceiling_parametric.csv`): along the review's *r* = 0.71–0.77, a
model at ρ 0.70 sits at **80–83%** of attainable if the released label is a single unaveraged measurement and **75–77%** if it averages
two, so the conclusion holds for every replicate count.

**Consequence.** Under the field's ceiling a bacterial model at 0.695–0.782 looks saturated; under the corrected one, models at ρ ≈ 0.70 sit at **75–77%** of attainable, and the tuned model at 0.7314 sits at **77.3%** of
0.946, leaving **0.215 ρ** of real, repeatable signal unexplained (Part 3). The field has roughly **three times more headroom than it
believes**, and the 0.4 success threshold is far too lenient.

### Scope and calibration: what is not claimed

**The lead over crisprHAL 2 is +0.023 to +0.029 ρ on two splits of one organism's screens, with both models as shipped; it supports a
claim about reach, not about one architecture being better.** It is measured, audited and confirmed on untouched genes. In recent
human benchmarks CRISPRon, Rule Set 3 and DeepSpCas9 reorder between datasets and organisms by differences of the same size, and one
plant evaluation found older, simpler scores tracking in-planta efficiency better than either deep model.

Absolute ρ is not comparable across assays: CRISPRon reports 0.80 internally and 0.46–0.68 externally, Rule Set 3 and DeepSpCas9 sit
at *r* ≈ 0.40–0.45 in independent benchmarks, and DeepHF reports 0.867 within a cell line and 0.433 on a new one. A 0.4 and a 0.8 can
describe equally good models on differently noisy data, which is why the ceiling above is the comparison that matters.

- **Specificity.** Six of `h_offtarget`'s thirteen columns are constant zero because the cohort keeps only uniquely-targeting guides, so
  its +0.0009 is a filter, not evidence about off-target descriptors (Part 3).
- **Nickase editors (base and prime editing).** Untested: they bind and nick rather than cut. The nearest literature, Orro et al. (2026),
  fits a binarised human fitness label with ±15 nt of context, so the kilobase window is untested there. Gene essentiality predicts their
  label (0.179) and is a null here (+0.00004, reproducing Guo et al. to four decimals) because the labels measure different things:
  a fitness effect against survival of a break that is lethal whichever gene carried it.
- **Human.** The *E. coli* flank result should not be assumed to transfer: the one cross-kingdom quantity tested inverted (GC against
  activity, −0.20 here against +0.02 there), and the seed preferences are uncorrelated (ρ = −0.052). Cross-kingdom tests are in Part 3.
- **External breadth.** One cohort, two relatives and, for the cutting claims, no untouched external set (BoostMEC uses 13); this needs
  new data (Part 3).
- **Gene effect as a random term.** Yu et al. model it; a mixed-effect version trades −0.008 between-gene for +0.0016 within-gene and
  moves pooled ρ by −0.0003, because on a cutting screen the gene effect is largely signal (between-gene ρ 0.729), not nuisance
  (`EXTENSION_LOG.md` §18.1).
- **Not yet reported or shipped.** A train/test sequence-distance statistic (Browne et al. 2026 and Yu et al. 2024 report one); exact
  duplicates are removed and folds grouped by gene and by 100 kb block, which should already cover it in *E. coli*. No installable
  design tool is shipped.

# Appendices

The controls, the methods each claim rests on, and the provenance. Nothing here is a finding.

## Appendix A — Controls and nulls

Each control is given as what it rules out, the design, the result, and what would be believed without it.

### Nulls: is the feature–label pairing real?

**Permutation floor.** *Rules out:* gain that a selector and a boosted tree manufacture from 300 columns of correctly shaped noise. *Design:* every family with a gain re-run with its feature rows shuffled against the labels (marginals and capacity kept). *Result:* floor **0.5264 against a 0.5272 baseline**, i.e. zero; `j_periodicity`'s shuffled arm scores higher than its real one. *Without it:* family A's +0.080 would rest on the assumption that a 6,580-column search cannot overfit a 13,880-row training split.

**Texture-preserving null for family A** (`shiftnull.py`). *Rules out:* that family A learned the texture of bacterial DNA rather than the local DNA at the cut site; row shuffling cannot, since it also destroys genomic autocorrelation. *Design:* each guide receives the flank track read from a position offset by a circular chromosome shift, which keeps every column's marginal, smoothness and long-range autocorrelation and breaks only the alignment (correlation with the aligned track 1.00 / 0.41 / 0.05 / −0.03 at 0 / 1 kb / 100 kb / 2 Mb on `dn500.gc`). The shift is swept, since a 1 kb shift is still 0.41 correlated and a single 2 Mb shift shows only that a distant locus is uninformative; Δρ against shift distance is the effect's autocorrelation length. Grouped CV (100 kb blocks) throughout. *Result:* Finding 2. *Without it:* the result would also fit a model that has learned the texture of the genome.

**Negative control from the source experiment** (the dCas9 / no-antibody arm of the GapR-seq experiment; one family only). *Rules out:* that `d_supercoiling`'s +0.045 (published matrix) measures supercoiling. *Design:* the same assay with the biology removed; a feature that predicts equally well from the control track measures the assay. *Result:* the signal is DNA read density, mostly gene dosage. Four columns computed from the genome coordinate alone (distance to *oriC*, that distance as a fraction of a replichore, a 2<sup>−2f</sup> copy-number proxy, which replichore) are worth **+0.0112** on the published cohort under 100 kb blocking, against **+0.0111** for the GapR ChIP track alone and **+0.0053** for the supercoiling-specific log-ratio part (`dissection_supercoiling.csv`, rows `z_dosage`, `z_signal_only`, `z_sc_enrichment`). Origin-proximal loci are over-represented in any DNA-derived track of multi-forked, exponentially growing *E. coli*, which is why the untagged control is interchangeable with the ChIP; dosage is also a candidate mechanism for the phenotype (more target copies to cut). *Limit:* small quantities (+0.011 against family A's +0.164); the claim concerns the source of `d_supercoiling`'s signal, not the Cas9 effect.

### Splits: is the gain prediction or address memorisation?

**Grouped cross-validation** (`arc5`, 100 kb blocks, or whole genes held out). *Rules out:* locus recognition: guides in one gene share a locus and positional columns identify it, so plain KFold rewards learning that guides around here score about this much (sequence features are unaffected). *Design:* hold out whole groups. *Calibration:* families that are functions of the guide lose 6% in aggregate under grouping, the cost of the harder partition rather than memorisation; `a_flank`, `d_supercoiling` and `i_shape` lose nothing beyond it, `e_nucleoid` and `g_transcription` lose 6 and 15 points more (Appendix B3). *Without it:* positional gains would be indistinguishable from locus recognition, whose budget is the between-gene share of label variance, **16% on the published matrix, 29% on the curated one** (`cohort_groups.py`; Appendix B). *Re-run on the curated cohort:* every curated family number had been measured under random KFold on a cohort whose between-gene share nearly doubled; under gene grouping the families keep 79–96% of their gains (Appendix B3), so the inflation is not memorisation.

### Matching: is the comparison fair?

| control | rules out; design | result | believed without it |
|---|---|---|---|
| Matched-span head-to-head against crisprHAL 2 | the lead attributed to architecture when the models read different amounts of DNA; crisprHAL also run at our window widths, SLICER at theirs | at their span our windowed composition loses by 0.025; at ±1,000 nt we lead by 0.065 | "we out-encode a tuned deep model", which is false; the defensible claim is access to DNA they never see |
| Distance- and GC-matched orientation comparison, bootstrap over 47 genomic blocks | a cohort confound (opposite-orientation guides sit at different distances from the gene and in different GC); blocks rather than rows, since nearby guides are not independent | 0.047 against the nearest gene becomes **0.007 [−0.024, 0.038]**, P = 0.34; against a curated **TSS** the effect is real (Δ 0.051 [0.020, 0.082]) | the 0.047 read as an orientation effect |
| Conditional GC test within matched target-GC strata | flank GC acting as target GC seen through a correlated variable; bin on target GC and measure inside each bin | curated cohort: **+0.176 marginal → +0.221 partial**, positive in all five strata: a suppressor, the opposite of a confound | the GC account would rest on a marginal correlation |

**What the comparison on their splits holds equal.** Rows, labels and splits are equal (theirs, untouched by us). Each model is used as released: crisprHAL 2 with its published weights, SLICER with settings chosen on held-out genes of the WT-SpCas9 screen, which the eSpCas9 split never entered; for the WT-SpCas9 split the CONFIRM genes are the clean check (Part 5). Input DNA is **unequal, in our favour**: family A reads ±1,000 nt and their eSpCas9 input is 406 nt, so each model is compared as released, not with equal information, and equalising the DNA makes the ranking depend on width (Finding 2). Our features were designed on *E. coli* WT cutting data, so design knowledge leaks into the eSpCas9 arm in a way no split removes. For the frozen model the SpCas9 margin compares our 0.6997 with their *published* 0.695, while our re-run of their model gives 0.6940: +0.006 measured our way, +0.005 against their number. Equal DNA is not equal opportunity to use it: at 2,023 nt the published configuration's flattened layer is ~5× larger with regularisation and training budget chosen at 378 nt, so Finding 2 claims only that the published configuration is degraded by input wider than it was tuned for, not that the architecture cannot use long-range context (a re-tune at each width, `colab/crisprhal_tune_by_width.py`, is not yet run). The experiment extends the curve of Browne et al., whose own search stopped at 378 nt (Part 5).

### Instruments: does the measurement measure what its name says?

| control | rules out; design | result | believed without it |
|---|---|---|---|
| Selector control: 300 columns against all | an arm gaining only because it was offered more candidates; selection switched off | `d_mechanics` **+0.0988 selecting 300, +0.0993 using all 6,199** | gain attributed to selection |
| Faithfulness positive control | an attribution read as a measurement when the method assigns importance to real and permuted features alike; the probe is fed a known null | at a matched ~35% of attributed importance, permuted columns score 0.258 against real columns' 0.581 (Finding 5) | attribution read as measurement |
| Pairing alignment, held-out comparison | a mis-rotated join, which gives a tight, plausible, meaningless interval (the failure shape of the three execution guards in `docs/qna_archive.md`); crisprHAL's out-of-fold predictions have no guide key, so the code rebuilds `KFold(5, shuffle=True, random_state=41)` in fold order and asserts that the two label vectors match guide for guide | labels match guide for guide | a precise interval about the wrong pairs |

**Reconstructibility probe** (each of a family's columns predicted out of fold from the base). *Rules out:* the claim that a family adds no new DNA, which reading the builder checks only for inputs someone read: a column the base predicts at R² = 1.0000 adds no information, one at R² ≈ 0 reads something the base lacks. *Result:* R² from the base is **1.0000** for the 9 ΔG/ΔH/ΔS sums over the protospacer and its windows, 0.997–0.9995 for the 4 melting temperatures, 0.20–0.98 for the 7 stacking and slide summaries and **−0.006 to +0.042** for the 8 flank thermodynamics (±25, ±100 nt); those eight carry +0.097 of `d_mechanics`' +0.099 (Finding 1). The guide-only / flank-only decomposition (`no_information.py`) uses the same base, rows, folds and selection budget: +0.011 for the 22 protospacer columns, +0.097 for the 8 flank columns. *Without it:* `d_mechanics` would remain the headline evidence that representation beats information.

**Equivalence reported as a bound, not against a margin.** *Rules out:* arguing equivalence from a non-significant p-value, or from a margin whose declaration cannot be audited. *Design:* no margin is claimed as pre-declared, since both were first committed after their results existed (427 columns, 0.005 ρ: result `136755f`, margin `0f66a2e`, 7 days later; parity with crisprHAL 2, ±0.01 ρ: result `6e2f32d`, margin `872e2d4`, 54 minutes later); each claim is reported with the smallest margin the data support, the larger end of its interval. Paired on the same 11,898 eSpCas9 guides, 2,000 resamples, SLICER minus crisprHAL 2, with the smallest margin at which equivalence holds in brackets: Spearman ρ +0.0021 [−0.0031, +0.0072] (**0.0072**); Kendall τ +0.0023 [−0.0024, +0.0073] (0.0073); Pearson r −0.0019 [−0.0066, +0.0030] (0.0066); R² −0.0030 [−0.0103, +0.0041] (**0.0103**); top-decile precision −0.0059 [−0.0244, +0.0202]; expected pick percentile −0.70 [−17.91, +3.98]. The 427 columns against 6,232 give [−0.0008, +0.0054], equivalent at any margin ≥ 0.0054. The rank metrics support equivalence with room to spare, but **R² does not at ±0.01**; it needs 0.0103. No metric resolves, so "better at ranking against better at picking" is sampling noise, not model character, and a top-10 statistic should not be headlined (its interval is 22 percentile points wide).

**Attenuation ceiling (Spearman–Brown on replicate statistics).** *Rules out:* reading "out of material" as a claim about the model: ρ cannot exceed the label's reliability, so the ceiling must come from the measurement. *Design:* two Cas9 replicates and two controls give two independent estimates of one quantity; their agreement is the reliability of one, and Spearman–Brown converts it to that of the released mean of two. Two checks make it an estimate of *their* label: their cohort reproduces under their stated cutoff (33,871 against 33,567), and a plain log-ratio reproduces their score at |ρ| 0.957. *Result:* **0.946 [0.944, 0.948]** for the label trained on (Finding 6), conservative because their posterior shrinkage can only raise reliability above a plain log-ratio's. *Without it:* the curated arms would keep quoting 0.937, derived for a different label by a different route (Appendix B).

### Inventory of nulls

Eight nulls, one small real gain and one robustness control that passed. For this phenotype the lever is what DNA the model is shown and nothing about the estimator repays attention, which licenses the cost claim in Part 5.

| what was tried | result | why |
|---|---|---|
| model class (16 classes, then 8 on the curated base) | spans 0.116 ρ, and 0.108 *with* family A | every class reads the same columns; the gain lies in the representation |
| the selector: budget (300 against every column), library, gain against SHAP, redundancy penalty | ±0.0005 (600, 1,200 and every column move family A by at most 0.003); library 0.0008 at 3× the speed; SHAP +0.0007; a redundancy-penalised rule **−0.0023** (Appendix B) | selection is not binding (selector control above); the redundant columns here are the informative ones, so penalising them leaves a more binary, less readable set |
| flank descriptions beyond composition: tetramers, DNA shape, helical periodicity and GC skew | tetramers −0.0019 against dimers; shape +0.0007 and periodicity/skew +0.0004 on top of composition | 4,343 tetramer indicators are too sparse to carry anything; composition already holds the shape (Appendix B4); the signal is not helical phasing or curvature |
| explicit interaction terms (family G's rationale) | +0.0004 over their parents | Clarke et al.'s transcription×strand mechanism does not reproduce |
| per-guide label reliability as a training weight | −0.0050, four fifths of it reproduced by permuting the weights (Appendix B) | non-uniform weights spend effective sample size whatever they mean |
| *small real gain:* 627-draw nested hyperparameter search | **+0.013**, of which **+0.0087** is the measured adaptive component; a later joint search judged on unseen genes gives +0.0143 [+0.0111, +0.0174] (Appendix B) | capacity, not policy |
| *control that passed:* family A rebuilt at 75/300/600/1,200 nt against 50/250/500/1,000 | +0.1634 against +0.1638 | the window choice carries no weight |

## Appendix B — The methods and controls behind the findings

### Search effort in the head-to-head with crisprHAL 2

The final model is compared with crisprHAL 2 as released, and the search effort differs. Our side used 97–157 draws per fold (627 in total). crisprHAL 2's published configuration was given one nested search of 33 fits (~2.6 GPU-hours; patience stopped every fold after 5–8 draws) and won the inner search in four of five folds: 0.6987 against its published 0.6971 (a second search 0.6989; the best configuration found by any search 0.7051; Part 5). Pre-specified brackets were ≥ ~0.716 (margin mostly a tuning artefact), 0.704–0.710 (margin survives at +0.005 to +0.010) and above 0.7213 (we lose on our own folds); observed 0.6987, below all three; draw 0, the published configuration, was the positive control and landed inside the 0.686–0.702 band `crisprhal_rerun.csv` predicts. The published configuration fits one fold in 5.48 min on a free Colab T4 against 17.0 min locally, a measured **3.1×** (`crisprhal_wide_window.csv`, `crisprhal_rerun.csv`). On crisprHAL 2's own two splits the comparison is against released weights and our harness reproduces their numbers to three decimals (ρ 0.782 → 0.7821 on eSpCas9), so the +0.005 and +0.002 margins are against whatever search produced the released weights.

*Open (method level):* whether a search matched to ours (~100 draws per fold at ~5 GPU-minutes each, ~40 GPU-hours) would change a claim about the architecture, as opposed to the released model, was not run; the as-released comparison does not depend on it (Part 4 lists the retune on their own splits).

### Hyperparameters are a hundredth-scale knob, and the one gain the sweep found is capacity, not policy

Of the champion's 18 real `XGBRegressor` knobs, six had been varied by a search and **eleven had never moved**; the 627-draw search ran on LightGBM (`tune.py`) and the only earlier XGBoost search was the 8-draw one inside the sealed holdout. The eleven were swept one at a time (25 arms, curated base with family A, selection cached per fold; `xgb_knobs.py`, `results/xgb_knobs.csv`) against the champion's 0.7084: `grow_policy` = `lossguide` with `max_leaves` 63 **+0.0069**, `num_parallel_tree` 4 +0.0025, three knobs at most +0.0004, column subsampling −0.0009 to −0.0014, `max_delta_step` **−0.0080**. A 25-arm sweep reports a maximum, and the adaptive component is **+0.0087**, so the winner was re-run on three fresh fold partitions (fifteen paired comparisons, each ahead on 15/15 folds; `xgb_knobs.py --verify`, `results/xgb_knobs_verify.csv`): `lossguide`, 63 leaves +0.0067 [+0.0054, +0.0080]; **`depthwise`, `max_depth=6` +0.0063 [+0.0053, +0.0073]**; `num_parallel_tree=4` +0.0024 [+0.0019, +0.0029]. `lossguide` does nothing that four times the leaves does not do under the original policy: `max_depth=4` is too shallow for this cohort by about six thousandths of ρ. It is not adopted into the frozen champion (Part 4), since +0.0063 is below the +0.0087 adaptive component on these folds, which is what a sealed holdout exists to remove.

*Scale and limit.* The whole measured span of tuning is **+0.0131**, of which **+0.0087 is adaptive inflation**, against **+0.164** for one feature family (0.108 ρ across model classes, ±0.0005 for the selection budget), on a problem whose open variance is 0.22 ρ. On the human matrix the larger depth gains (+0.10 for depth and learning rate, +0.0585 for `max_depth` 4 → 6) came from row-wise folds in which 49% of rows carry a duplicate guide; under twin-grouped folds depth is worth **−0.0062** and the champion is the best of six arms (`EXTENSION_LOG.md` §16.2). Tuning buys insurance against a change of signal shape, not a better number on a signal already fitted.

### A joint search judged on unseen genes finds +0.0143 [+0.0111, +0.0174], and the optimum is slower, not deeper

Genes were split once into a tuning part (70%) and a confirmation part (30%, 1,204 genes, never used to choose anything). Two hundred random configurations were scored on gene-grouped folds of the tuning genes; the best twelve were re-scored on three fresh partitions; the survivors were trained on all tuning genes and scored on the confirmation genes and on crisprHAL's eSpCas9 screen (`EXTENSION_LOG.md` §11). The champion (depth 4, 400 trees, lr 0.05, 300 columns) scores 0.7073 on the confirmation genes, 0.7840 on eSpCas9 and 0.7151 on the whole cohort (standard protocol). Depth 6 gains +0.0053 [+0.0027, +0.0083], +0.0174 [+0.0165, +0.0183] and +0.0072 [+0.0062, +0.0082] on those three; the best tuned configuration (depth 5, lr 0.013, 4,000 trees, 450 columns, subsample 0.51) gains **+0.0143 [+0.0111, +0.0174]**, **+0.0255 [+0.0246, +0.0262]** (10 folds) and **+0.0162 [+0.0152, +0.0173]**; the average of the five best configurations gains **+0.0174 [+0.0144, +0.0204]** on confirmation genes and **+0.0183 [+0.0175, +0.0191]** on the whole cohort.

The gain holds on never-seen genes, at the size it has on the tuning folds (0.7121 → 0.7248 for the best tuned), and carries to a different nuclease on a different cohort (+0.023 to +0.026, ahead on 10 of 10 folds; 59,489 guides). The better direction is slower, not deeper: mean ρ over the 200 draws falls with depth (depth 3–5 about 0.713, depth 8–10 about 0.70) and with learning rate and rises with tree count, with a plateau at depth 4–5, lr 0.013–0.04 and 1,600–4,000 trees; the other knobs are each within ±0.001 once these are fixed, and the column budget stops mattering above about 450.

*Replication.* Genes were re-split 70/30 with a different seed and a **fresh 40-draw search** run on the new development half (`tune_split2.py`); re-scoring split 1's winner would have been contaminated, since 847 of the new confirmation set's 1,204 genes fed split 1's selection. The new winner is a different configuration in the same region (depth 6, lr 0.020, 2,000 trees; 450 columns, raw target): **+0.0121 [+0.0092, +0.0152]** against +0.0143, and +0.0159 against +0.0174 for the ensemble. The gap, **+0.0022**, estimates what selecting on the same genes adds and agrees with the "about +0.002" from an unrelated route.

*Limits.* Levels move far more than differences: the champion scores **0.7073** on split 1's held-out genes and **0.7329** on split 2's, a **0.026** swing between two random 30% gene partitions, roughly twice the whole tuning gain, while differences between arms reproduce to within 0.002; every tuned-model figure is therefore quoted as a difference, and a level belongs to its split. The benefit is about one and a half hundredths of ρ on a problem whose open variance is 0.22, and it concerns the flank columns more than the learner: without family A the +0.016 collapses to +0.003 (Finding 4). The champion elsewhere in this report keeps its original settings, since every comparison was made against them.

### Mixing model families adds about +0.0015, and the tuned settings describe an additive-plus-pairs signal

*Mixing* (`EXTENSION_LOG.md` §12, §14). XGBoost, LightGBM (40 draws) and CatBoost (30 draws) were each searched on the tuning genes and scored on the held-out genes. As single models they land together (LightGBM 0.7206, CatBoost 0.7207, tuned XGBoost 0.7215; champion 0.7073), with prediction correlations 0.985 (XGBoost–LightGBM) and 0.977 (with CatBoost) and residual correlations 0.973 and 0.959, which is why blending helps so little. Against the XGBoost average of five (0.7246), the equal-weight blend with CatBoost gives 0.7262 (**+0.0016 [+0.0006, +0.0026]**), with LightGBM and CatBoost 0.7261 (+0.0014 [+0.0005, +0.0025]), and fitted weights 0.7264 (+0.0017 [+0.0005, +0.0030]). Models that err differently help only if good enough: extra-trees (0.666), a neural network on the selected columns (0.692) and ridge on all columns (0.647) each lower the equal-weight blend (to 0.7224, 0.7234, 0.7226). *Limit:* one split's held-out genes; Part 5 gives the cost and the eSpCas9 result, where mixing is lower than the single model.

*What the settings say about the data* describes the fitted ensemble, not the molecule: trees split on correlated columns, so "interaction order" is order among columns and one biological factor can appear as several. **Additive plus pairs:** tree depth is the interaction order; on held-out genes stumps reach 0.6905 against 0.647 for ridge, depth 2 0.7177 (+0.027), depth 3 0.7206 (+0.003), depth 4 0.7210 (+0.0004), depth 8 0.7158; of the 0.074 that trees add to the linear model, 59% (0.0435) is nonlinearity within a column, 37% (0.0272) pairs and 4% (0.0033) third and fourth order together. **Read more directly from sequence:** the tuned model's share of gain moves from dense summary descriptors to position-and-k-mer indicators (35% → 51%; protospacer quantum descriptors 37% → 27%, flank descriptors 27% → 22%). **Both levels of ranking:** on held-out genes the tuned model gains +0.018 [+0.008, +0.028] ranking genes against each other and +0.018 [+0.014, +0.022] ranking guides within a gene.

### Grouped cross-validation means something different on the curated cohort, and the gap narrows

Gene identity explains **16%** of the published label and **29%** of the curated one, because the published rows target a gene panel and crisprHAL's curated release is genome-wide: within-gene share **0.843** on the published matrix (13,829 of 13,880 guides placed in 1,222 genes, 11.3 guides/gene) against **0.713** on the curated one (30,203 of 33,567 guides in 4,013 genes, 7.5 guides/gene). Pre-specified: twice the between-gene variance should widen the gap between random and gene-grouped CV. Observed: it narrows; grouping cost `e_nucleoid` 42% of its gain on the published matrix and 12% here, `g_transcription` 36% and 21% (Appendix B3). Between-gene variance sets how much a locus-memorising model could win and guides per gene how well it can learn each locus's mean; both moved, in opposite directions, and only the measurement says which won. *Check on the method:* genes are assigned by coordinate on NC_000913.2 against the reference gene table for both cohorts; on the published rows this agrees with Noshay's own assignment for **97.7%** of guides and gives the same share to three decimals (0.843 against 0.844), so the change is the cohort and not the method (`python -m sgrna.harness.cohort_groups`, `results/cohort_groups.csv`).

### The published label's ceiling is 0.937 [0.929, 0.945], from tiling-library agreement

Two screens of the same library agree at ρ 0.810, which naively gives 0.90, but the disagreement is predictable at ρ 0.546, so it is enzyme biology and not noise. Using the source paper's own agreement statistics, with the published score a two-replicate geometric mean, the tiling route (a *different* library; r 0.878, reliability 0.878) gives **0.937 [0.929, 0.945]**. The replicate-agreement route with Spearman–Brown (r >0.883, reliability 0.938) gives 0.969 and is rejected: it must assume the activity score inherits the single-arm read counts' reliability, which it cannot, since the label is the `|Z|` of a Cas9/dCas9 **ratio** and a difference is never more reliable than its components (at 50% shared signal between arms the implied ceiling is already 0.940). *Limit:* 0.937 is the ceiling for Noshay's `cut_score`, not for the curated label trained on, whose ceiling (0.946, Finding 6) comes from a replicate split that exists for that label; both are quoted because arms on the two cohorts are compared, and quoting 0.937 against a curated number would understate the headroom.

### Per-guide label reliability cannot be used as a training weight

Weighting training rows by measured per-guide reliability costs ρ, and four fifths of the damage comes from having non-uniform weights at all, not from where they point. Pairing Cas9 replicate 1 with control 1 and replicate 2 with control 2 gives two independent estimates of each curated guide's score, whose disagreement runs **0.06 / 0.71 / 2.39** log2 units at the 5th / 50th / 95th percentile; it is not a read-depth effect (ρ(disagreement, total counts) = −0.053) and is mildly tied to the label (ρ(|y|, disagreement) = −0.236). Weights apply to training rows only; validation rows are scored unweighted against the released label. Champion, selection inside each training fold, 3 seeds × 5 folds (`replicate_weight.py`, `results/replicate_weight.csv`): unweighted **0.7083**; inverse variance, moderated, 0.7032 (**−0.0050**); *the same weights permuted across guides* 0.7041 (*−0.0041*); inverse variance flattened within \|y\| bins −0.0031; weighted by read depth −0.0241; drop the noisiest decile of training guides −0.0016, *drop a random decile* *−0.0014*. Permuting keeps the weight distribution and destroys its correspondence with measured noise, yet costs −0.0041 of the −0.0050: a 98:1 weight ratio spends effective sample size, about four thousandths whatever the weights mean. Read depth is the wrong instrument by an order of magnitude because it tracks almost none of the noise. On a label with no shrinkage (the plain CPM log-ratio of the same counts; `--raw-label`, `results/replicate_weight_raw.csv`) inverse-variance weighting costs −0.0040 and permuted weights −0.0043, so the correspondence with measured noise is worth +0.0003. *Limit:* 33,567 rows against a feature set worth +0.164. The raw label is also less predictable, 0.6864 against 0.7083 on identical rows, folds and features: ALDEx2's processing is worth about +0.022 of fit, five times what swapping the label at fixed rows buys (+0.004, Part 3), and it is done in the label's construction, where no downstream reweighting reaches it.

### The reduced-representation ladder

**On the published matrix, 412 portable columns match all 6,232 (ρ 0.52713 against 0.52716; Finding 1).** One protocol, 5
seeds × 5 folds: 74 mononucleotide columns 0.5081; 427 (adding dinucleotide and non-sequence features) 0.5249; 412 (minus 15
undecoded) **0.52713 ± 0.0011**; published full 0.52716 ± 0.0022. The 427's loss is **+0.0023, 95% CI [−0.0008, +0.0054]**,
one-sided p = 0.035 against a 0.005 ρ margin set after the result was seen. *Limit:* every arm selects **300 columns from
its candidate pool** (`cross_validate` resolves `n_features=None` to `config.N_FEATURES`), so the ladder compares pools
under a fixed budget, not full representations. On the curated cohort the budget does not bind (±0.0005, Finding 1), but the
reduction fails there once each arm is re-run with `n_features` set to its own column count; the curated tetramer (−0.0019),
quantum-column (+0.0114) and non-inferiority results use that no-selection protocol.

**B.2 — What is the model class worth?**

### Sixteen model classes and the cost spread

**Boosting beats a tuned CNN–BiGRU (crisprHAL 2) by +0.003 at 1/113th of its wall-clock cost, and ridge is within 0.058 of
it at 1/220th.** Curated cohort, family A, one seed × 5 folds, crisprHAL 2 on the same folds (`curated_models.csv`,
`crisprhal_tuned_folds.csv`). Wall-clock for 5 folds: XGBoost (champion) ρ **0.7084**, 1.9 min; ridge 0.6473, 1.0 min;
crisprHAL 2 published configuration 0.6971, 85 min (45×); its best configuration 0.7051, 220 min (**113×**). The ~150×
headline in Part 5 is CPU-seconds per fold, 65.8 against 10,000 (`cost_panel.csv`, 152×); the two differ because the models
multithread differently (4.8 CPU-seconds per wall-second for SLICER, 7.9 for crisprHAL).

* **Ridge, PLS, random forest and extra trees agree within 0.009** and LightGBM (0.7082) matches XGBoost, so the problem is
  **bias-limited, not variance-limited**, at this sample size. k-NN is worst (0.6007), as a smooth global signal rather than
  a clustered one predicts (Finding 2's gradient, not a motif).
* **Boosting is worth +0.061 over ridge**, as a sum of many small, partly interacting effects requires (the margin was 0.03
  without family A); model class decides about a tenth of a correlation (Finding 1).

### Interpretability figures and the gain-ranked selector

**Of the three interpretability figures in Part 5, the ridge share of boosting is the strongest, the depth cap holds for the
frozen configuration only and the gain concentration is the weakest; gain-ranked selection changes which columns are shown,
not how well the model scores.**

* **Ridge reproduces 95.6% of boosting's ρ on a decorrelated basis.** With one coefficient per column and no term in two
  columns, the gap measures how much accuracy needs interactions, about 4%, so Finding 2's account can be read off the model
  as per-column effects.
* **`max_depth = 4` caps the interaction order at four, for the frozen configuration only.** Depth 4, `learning_rate=0.05`,
  400 trees and the 300-column budget are inherited from `FINAL_baseline_ecoli_sgRNA_model_5seeds.py`, not selected; the
  post-freeze joint search (Finding 4, `EXTENSION_LOG.md` §11) prefers depth 5, learning rate 0.013, 4,000 trees
  (**+0.016**). Third- and fourth-order terms are 4% of what trees add over a linear model.
* **45 of 300 columns carry half the gain** (the top ten 19.5%): this rules out a two-feature rule and an undifferentiated
  smear, and little else.

`run_ablation.make_selector` picks 300 of 6,517 columns by XGBoost gain, which does not survive a change of model and
credits a column for being binary (Finding 5). Audit (`selector_audit.py`, `results/selector_audit.csv`,
`results/selector_redundancy.csv`): 300 against 600, 1,200 and every column, within ±0.0005; top 300 by gain against by mean
|SHAP|, +0.0007 (both arms selected on all rows, so only their difference is quotable); top *k* against greedy
relevance-over-redundancy, **−0.0023** (0.7084 against 0.7061). Gain puts 144 binary indicators (48%) in the top 300 against
102 (34%) under SHAP, and the redundancy-penalised rule still picks more of them (174 against 145). *Limit:* no rule is both
decorrelated and informative, so attribution is reported at **block** level (Finding 5), and every statement about the
selected set is conditioned on gain-based selection.

**B.3 — What is the representation worth?**

### Recoverability, re-description and the family split: a column pays when it names something the phenotype depends on

**A column's increment falls from +0.180 to +0.008 as the guide explains more of it, with the step at about R² = 0.2, while
its marginal association with the label stays flat.** All 636 family columns with a defined R² were regressed on the guide's
one-hot encoding (384 predictors), out of fold, and banded; Δρ is the block's gain over the curated base (`redescription.py
--arm trend`, `results/recoverability_trend.csv`, 1 seed × 5 folds). R² ≤ 0.01: 425 columns, **+0.180**; R² 0.20–0.50: 13
columns, **+0.018**; R² > 0.99: 18 columns, **+0.008**, with strongest marginal |ρ| 0.168, 0.143 and 0.184. Over the coarse
cut, R² ≥ 0.99 is worth +0.008 and R² ≤ 0.50 +0.201 (`results/figures/recoverability_vs_power.png`). *Limits.* A bound, not
a prohibition: one at a time (`--arm percolumn`, `results/recoverability_percolumn.csv`), the 18 fully recoverable columns
are each worth up to +0.0065 (median +0.0038), including `eng.dg.hybridisation_weighted` (R² 1.000, |ρ| 0.184). The probe is
linear: the seven not-cheap `c_folding` columns are deterministic in the guide and none is linearly recoverable.

**Re-description of the guide is worth nothing as a class; the spacer fold energy's gain is physics, not non-linearity.**
Predicted: any precomputed sum of position-wise terms of the guide should pay a tree (Finding 1); observed: zero for random
weights per position and base (`redescription.py`, three draws per count, two model seeds). The fold gain has two readings:
(a) RNA secondary structure is part of the mechanism, or (b) any non-linear, non-local summary of the guide would pay a
depth-limited tree the same hundredth. `shuffled-position MFE` folds every guide with real ViennaRNA after one fixed
permutation of its twenty positions (`--arm folds`, `results/redescription_folds.csv`): same energy model, value
distribution and composition, structure removed (conservative, since composition-driven foldability survives). Random
non-linear functions sum 200 random weights over position tuples (3 draws per order, one model seed; `--arm nonlinear`,
`results/redescription_nonlinear.csv`). Δρ over the curated base (out-of-fold linear R² in brackets): the real
`eng.dg.guide_selffold` **+0.0117** (0.316) and the spacer MFE recomputed through the same code path **+0.0119**;
**shuffled-position MFE**, permutations 0, 1, 2, **+0.0000, −0.0007, +0.0002**; random linear tables, 4 / 64 / 256 added,
−0.0004 / −0.0002 / **−0.0015** (1.000); random non-linear functions of order 2 +0.0000 (0.402; individual R² 0.339, 0.383,
0.484) and of order 8 −0.0005 (−0.008).

Reading (a) survives and (b) is excluded; the recomputation rules out a plumbing artefact, and no random arm is further than
0.0016 from zero. The mechanism supported is a spacer that base-pairs with itself being less available to invade the target
duplex. Being expensive for the tree is neither sufficient nor probably necessary (the nine cheap `b_energy` columns are
worth +0.0131 as a block); what separates a function that pays is whether it tracks the phenotype (`guide_selffold` has
marginal |ρ| **0.102**, a random function none):

> **A deterministic function of the guide pays a depth-limited tree about a hundredth of ρ when it names something the
> phenotype depends on, and nothing at all otherwise, however expensive it is to compute.**

*Scope:* the positive side rests on one column at +0.0116 and seven in `c_folding` at +0.0113, on one cohort, under the
300-column budget; difficulty survives only as an ordering among qualifying functions (per column, +0.0116 for the one
expensive aligned column against an average +0.0015 for nine cheap ones).

**Splitting each family by the recoverability probe reproduces the `d_mechanics` split (20/10 at +0.0113/+0.0970, against
22/8 measured by hand at +0.0110/+0.0973), and in several families the family figure reports one half.** `redescription.py
--arm families` (`results/redescription_family_split.csv`; `f_methylation` and `h_offtarget` omitted), not-cheap half
against whole family: `a_flank` 320 columns **+0.1643** of +0.1635; `b_energy` 1 column **+0.0116** of +0.0181; `c_folding`
7 columns +0.0113 of +0.0147; `d_mechanics` 10 columns **+0.0970** of +0.0987. The probe measures cheapness, not provenance:
in `a_flank`, `d_mechanics` and `h_offtarget` the not-cheap half is **outside the 20-mer**; in `b_energy` and `c_folding` it
is ViennaRNA folds and a guide-only self-folding energy. In `d_mechanics` the kinds separate by name (`--arm provenance`,
`results/redescription_provenance.csv`): the eight `eng.mech.flank.*` columns at ±25 and ±100 nt are worth **+0.0968**, the
two guide-only non-linear ones +0.0019, so it is a provenance effect, not a non-linearity effect. `eng.dg.guide_selffold` is
`b_energy`'s whole not-cheap half and the same quantity as `eng.fold.spacer_mfe` (Pearson 0.9993), so it carries 76% of
`c_folding`'s +0.0153: the independent evidence for the fold effect is **two measurements, not three** (the spacer fold
energy; `d_mechanics`'s 20 cheap guide-only columns), because cross-family duplicates are invisible to a family ablation.

### The human baseline

**On the human matrix (17,421 guides × 6,233 columns) the champion reaches ρ 0.3377 under twin-grouped folds, against 0.7314
here; row-wise folds overstate it.** The rows decode to 13,138 distinct 20-mers, 8,499 (49%) have a twin at the same
position; twin labels agree at Spearman 0.80 (`EXTENSION_LOG.md` §10), so row-wise folds reward memorisation; the bacterial
cohort has none (33,567 distinct in 33,567) and `kingdom_transfer.py` is unaffected. Twin-grouped
(`human_additivity_twin.py`, `results/post_freeze/human_additivity_twin*.csv`; row-wise: `human_baseline.py`,
`results/human_baseline*.csv`, `human_additivity.csv`), `max_depth` 4 → 6 moves +0.0585 to **−0.0062** [−0.0093, −0.0034].

**B.4 — Mechanism: what the effect is, and what it is not**

### Long-range features and the Chi-site (RecBCD) null

**The long-range candidates with a mechanism attached are built and null, and Finding 7 predicts most of what is left to be
null.** GC skew, A-tract phasing and bendability periodicity (family J) score −0.001 alone and +0.0004 on top of family A,
with their own permutation control scoring higher. The unbuilt candidates (k-mer entropy, homopolymer run-length, REP
density, RegulonDB distances to TSS, terminator and operon boundary) are predicted null, because the effect is GC and not
composition in general (purine is flat at every window), so non-GC re-descriptions are reparameterisations, as DNA shape was
(Appendix B4); only the RegulonDB orientation items address the separate orientation effect.

**With flank GC held fixed, Chi (`GCTGGTGG`) carries no association, and its coefficient sits at the 41st percentile of 200
GC-matched octamers.** RecBCD degrades a double-strand break over kilobases until a Chi site, so it would predict the flank
effect's length scale, missing motif and orientation dependence (Findings 2, 7), but the direction disagrees: Chi sites are
GC-rich, yet flank GC correlates **+0.161** with the label (`eng.flank.comp.up250.gc`; the −0.201 quoted elsewhere is
*target* GC, Finding 5). `chi_sites.py` counts `GCTGGTGG` and its reverse complement on each side for all 33,567 curated
guides (`results/chi_sites.csv`; 11,677 carry one within ±1 kb). The Chi count's β at ±250, ±500 and ±1,000 nt is +0.306,
+0.287 and +0.219 alone but **−0.103 ± 0.053, −0.047 ± 0.037 and −0.053 ± 0.026 with flank GC held fixed**: the marginal
association is flank GC. The deciding control is **200 other octamers of the same GC content** (6 of 8), each entered the
same way: mean β −0.022, sd 0.201, range −1.25 to +1.20; **Chi's −0.053 is at the 41st percentile**, so a specific
recognition sequence behaves like any GC-matched eight-mer. *Limits:* only the Chi-attenuation route is excluded; repair in
general (RecBCD processivity, RecA loading, replication-fork proximity) is not, and is not testable from sequence alone.
With flank GC fixed at ±500 nt, `GCTGGTGG` downstream is +0.169 ± 0.094 and its reverse complement upstream −0.110 ± 0.060,
within the null's sd of 0.201, so this orientation hint is not claimed.

### Why ρ, and why R² beside it

**ρ and R² are reported together: a LightGBM with a robust loss scored R² 0.192 with ρ 0.568**, the same model looking
broken on one metric and respectable on the other. ρ is primary because it is crisprHAL 2's own reporting convention and
ranking is the decision the model serves; what ρ means on that scale is in Part 5.

## Appendix B2 — Why the imported families failed, and why that is not a screen

**Two of the eleven imported families are columns the 20-mer already determines, and neither helped; the table does not support a general rule beyond that.** What survives is Finding 1: a deterministic function of the 20-mer cannot add information to a model that already encodes the 20-mer.

Out-of-fold R² of predicting each new column from position-specific mono- and dinucleotide indicators of the protospacer. Δρ is the family's standalone increment over the base on the published matrix (Appendix B3).

| family | columns | median R² | **share of columns with R² > 0.9** | Δρ |
|---|---:|---:|---:|---:|
| `b_energy` (CRISPRoff ΔG) | 10 | 0.984 | **70%** | +0.007 |
| `d_mechanics` (duplex stability) | 30 | 0.845 | **50%** | +0.020 |
| `c_folding` | 9 | 0.297 | 0% | +0.007 |
| `f_methylation` | 17 | 0.141 | 0% | +0.004 |
| the other seven (`a_flank`, `d_supercoiling`, `g_transcription`, `e_nucleoid`, `i_shape`, `j_periodicity`, `h_offtarget`) | 7–322 each | −0.012 to −0.022 | 0% | −0.001 to **+0.080** (`a_flank`) |

**The structure is binary.** `b_energy` and `d_mechanics` contain columns the 20-mer determines; the other nine contain none (`c_folding` and `f_methylation` average weak partial fits). Nearest-neighbour thermodynamics is a sum over adjacent pairs, so binding energy is a second-order quantity the matrix already holds: importing CRISPRoff imported the matrix into itself. A negative median R² (10th percentiles −0.019 to −0.022) is what out-of-fold R² against a target unrelated to the basis looks like; it is set by basis size and row count.

**Why this is not a general screen.**

* **It applies to two families.** Only `b_energy` (0.984) and `d_mechanics` (0.845) exceed 0.30. A rule with n = 2 is an observation, and for `b_energy` it is arithmetic: ΔG is computed from the sequence by nearest-neighbour summation, so adding it to a model holding the sequence losslessly adds nothing. The same holds for `d_mechanics`.
* **The rest carry no signal about the rule.** Among the families with negative R², R² spans 0.010 while their gains span 0.081; across all eleven, ρ(R², Δρ) = **0.224, p = 0.508**. Their failures have unrelated causes: `h_offtarget` needs the rest of the genome, `j_periodicity` is a genuine null, `i_shape` duplicates family A (+0.036 standalone, +0.0009 on top of it), and `d_supercoiling` reads gene dosage (Part 2, biology item 6).
* **No tally is supportable.** Only four families have positive R², no threshold for "high" can be set after the fact, and `d_mechanics` (R² 0.845) returned +0.020.

**Limit.** The usable direction is qualitative and carries no score: if a candidate's columns are individually reconstructible from the 20-mer, do not build it.

## Appendix B2a — Curation: what it is, and why family A doubled between cohorts

**Family A's gain is +0.080 on the published matrix and +0.164 on crisprHAL's curated cohort because the curated cohort contains 24,950 additional guides in which the flank effect is steeper; the label, the row count and the base account for almost none of the difference.** Neither figure is an attenuated version of the other: +0.080 is the right figure for the 1,222-gene panel around which Guo's screen was built (Noshay published a subset of it), +0.164 the right figure genome-wide. Comparing them across cohorts without this reads as the model improving.

**Curation drops rows; selection drops columns, and only curation can flatter a score.** crisprHAL's curation removes guides whose label is unreliable and is applied to the whole dataset **before splitting**, so it also cleans the test set and raises ρ partly by making evaluation easier; feature selection is fitted inside each training fold and cannot. The figure for curation here, ρ 0.607 → 0.628 on restricting to the 8,617 retained rows, contains both effects and has not been separated.

**Procedure** (Browne et al., 2026). The score is the ALDEx2 `diff.btw` log2 fold-change between the Cas9 and dead-nuclease arms, shrunk toward the prior. The filter is an average control-condition read count across the two control replicates, with a threshold of **73** for the *E. coli* SpCas9 set used here, chosen by 5-fold cross-validation on the training set and applied before the 80/20 split. The cohorts are not nested: 8,617 guides are shared with Noshay's published 13,880, 24,950 are only in crisprHAL's and 5,263 only in Noshay's. Each candidate explanation was tested with everything else held fixed:

| candidate | test | effect on family A's Δρ |
|---|---|---:|
| the **label** | same 8,617 rows, published base, swap `cut_score` for the curated score | **+0.004** (+0.0892 → +0.0936) |
| **row count** | random 13,880-row subsamples of the curated cohort, 3 draws | **−0.003** (+0.1672 ± 0.0027 against +0.1638 at 33,567) |
| the **base** | add 41 dense guide-only summary columns to the curated base (Appendix B3) | **+0.013** (+0.1638 → +0.1504) |
| **which rows** | the remainder of a +0.084 gap | **≈ +0.070** |

**The relationship is steeper in the added rows, not wider-spread.** The marginal rank correlation between flank GC and activity needs no model: +0.069 on the 8,617 shared guides under Noshay's label and +0.067 under crisprHAL's (the label does nothing), and **+0.172 on the 24,950 guides only crisprHAL has**. Regressing the curated label on flank GC at ±500 nt gives a slope β of 5.35 on the shared rows, 8.63 on added rows inside Noshay's panel and **17.03** outside it (`curation_variance.py`, `results/curation_variance.csv`). The predictor's spread rises only 1.21× and the label's 1.34×, which pushes r down, so the 2.7× rise in r is β (2.96×); which guides were published inside the panel's genes accounts for about ×1.6 of it and the choice of genes for ×2.0.

**Limit.** Why panel genes differ is open: separating dynamic range from different biology needs per-guide read counts that were not used. An essentiality explanation is argued against by family G's essentiality column being worth +0.00004 on its own.

## Appendix B3 — The family ablations on the curated cohort, and what they measure

**Of eleven feature families, one is the finding (family A, +0.164 ρ on the curated cohort); the other ten add only +0.039 beyond it (all eleven jointly give +0.203), and nothing confined to the 20-mer exceeds +0.019.** Five of the ten return almost nothing (`b_energy`, `c_folding`, `j_periodicity`, `f_methylation`, `h_offtarget`); five show real standalone gains that are family A, or gene dosage, in other clothes. The table is a ranking of families, not a budget.

Design: base representation against base + one family, 5 seeds × 5 folds, selection inside each training split (`python -m sgrna.cohorts.curated_ablation --all`); baseline ρ 0.5445 ± 0.0010. The published column is each family's own run in Appendix C (`ablation_main`; `ablation_grouped` for the three positional families; `ablation_shape`; `ablation_jper`), whose baselines differ slightly between rows (0.5272 random-row, 0.5283 100 kb-grouped). The gene-grouped column (`curated_ablation --grouping gene`) holds out whole genes on the 30,203 guides inside an annotated gene (4,013 groups, 5 seeds × 5 folds, baseline ρ 0.5584; the 3,364 outside every annotated gene are dropped), so Δρ compares across columns and the ρ levels do not.

| family | reads | columns (dense continuous) | published Δρ | **curated Δρ** | gene-grouped Δρ | retained |
|---|---|---:|---:|---:|---:|---:|
| `a_flank` | flanks, ±1,000 nt | 348 (155) | +0.080 | **+0.164** | **+0.157** | 96% |
| `d_supercoiling` | external (GapR-seq track) | 26 (26) | +0.045 | **+0.139** | **+0.131** | 94% |
| `i_shape` | flanks | 152 (152) | +0.036 | **+0.120** | **+0.112** | 94% |
| `d_mechanics` | guide (22 of 30 columns) and flanks (8) | 30 (29) | +0.020 | **+0.099** | **+0.092** | 92% |
| `e_nucleoid` | external (Hi-C, nucleoid proteins) | 20 (20) | +0.016 | **+0.072** | **+0.063** | 88% |
| `g_transcription` | external (transcription, replication) | 19 (12) | +0.015 | **+0.058** | **+0.046** | 79% |
| `b_energy` | guide (R-loop free energy) | 10 (9) | +0.007 | +0.019 | +0.017 | 91% |
| `c_folding` | guide (sgRNA folding) | 9 (7) | +0.007 | +0.014 | +0.014 | 97% |
| `j_periodicity` | position (strand asymmetry, helical phasing) | 24 (24) | −0.001 | +0.011 | +0.007 | 64% |
| `f_methylation` | short flank (±20 nt at the PAM; Dam, Dcm) | 17 (2) | +0.003 | +0.008 | +0.007 | 93% |
| `h_offtarget` | genome (off-target burden) | 13 (1) | +0.000 | +0.001 | +0.000 | 23% |

**What became of each family.** `a_flank` is the finding (Finding 2). `d_supercoiling` is read density (the untagged GapR control predicts as well as the ChIP), mostly gene dosage recoverable from distance to *oriC*: a trap, not biology (Part 2, biology item 6). `i_shape` is family A re-expressed as bendability, **+0.036** standalone and **+0.0009** on top of A (`a_flank` + `i_shape` +0.1647 against `a_flank` +0.1638; Appendix B4): a near-duplicate, not a null. In `d_mechanics` the 8 flanking columns carry +0.097 of the +0.099 (Finding 1); the guide-only effect is +0.011 to +0.019 on two independent measurements, because `b_energy`'s and `c_folding`'s shares of it are the same spacer-fold column entered twice (Appendix B). `e_nucleoid` and `g_transcription` read external position-indexed tracks and carry a real locus component (below). `c_folding` is small but real (Part 2, further inferences, item 3). `j_periodicity`, `f_methylation` and `h_offtarget` are null (six of `h_offtarget`'s thirteen columns are constant zero by construction of the cohort).

**The ranking transfers; the scale does not.** The published and curated columns rank-correlate at **ρ = 0.97**, every family gains, and the median inflation is **3.3×** (1.9× for `c_folding` to 4.9× for `d_mechanics`); ratios of Δρ are not ratios of information, and every family statement elsewhere is a difference. The multiplier is the **2.96× steeper** flank-composition slope in the guides crisprHAL adds (Appendix B2a), which every family that reads genomic context inherits. The two Δρ columns are not comparable as levels: the published base contains the 63 hand-named summaries and the curated base (6,169 sparse indicator columns, 300 selected) has none.

**Neither the selector nor column shape produces the gain.** On `d_mechanics`: base with 300 selected 0.5443, base + `d_mechanics` 0.6432; base with all 6,169 columns 0.5439, base + `d_mechanics` with all 6,199 columns 0.6432, so the gain is **+0.0988 with selection and +0.0993 without**. ρ(dense continuous columns, Δρ) = **0.882, p = 0.0003, n = 11** (columns with more than ten distinct values, `--shape-audit`, `results/ablation_curated_shape.csv`), but supply is not sufficient: `j_periodicity` supplies 24 dense columns, as many as `d_supercoiling`'s 26, and returns +0.011 against +0.139; dense columns and reach beyond the 20-mer are correlated, and reach is the one that separates. Giving the curated base 41 dense guide-only summary columns (`b_energy`, `c_folding` and `d_mechanics`'s protospacer half; `base_enrichment.py`) raises it from 0.5445 to 0.5692 and takes **0.013** off family A's gain, 8% of it. The published matrix's own 63 hand-named summaries take **+0.0015** off, 1.8% of it (`base_enrichment.py --published`, `results/base_enrichment_published.csv`): sparse base of 6,169 columns ρ 0.5227, + family A 0.6050, Δ **+0.0823**; with the 63, 6,232 columns, 0.5272, 0.6080, **+0.0808**. The second base's 0.5272 reproduces Finding 1's 0.52716 through a different code path, certifying that the arms differ only by those columns. Column shape is therefore not the source of the published-to-curated tripling; the row population accounts for essentially all of it (Appendix B2a).

**Reach separates the families better than any property of their columns: nothing confined to the 20-mer exceeds +0.019, and nothing above +0.05 is confined to it.** Curated Δρ by reach: the 20-mer and its PAM only (`b_energy`, `c_folding`, `d_mechanics`'s 22 guide columns) **+0.011 to +0.019**; a short flank (`f_methylation`) +0.008; the flanking DNA (`a_flank`, `i_shape`, `d_mechanics`'s 8 flank columns) **+0.097 to +0.164**; the rest of the genome or external measurement (`d_supercoiling`, `e_nucleoid`, `g_transcription`) +0.058 to +0.139; position on the chromosome (`j_periodicity`, `h_offtarget`) +0.001 to +0.011. Access to information outside the guide is necessary for a large gain and not sufficient: `j_periodicity` and `h_offtarget` read the whole chromosome and return nothing. This is the conclusion Finding 2 reaches from the other direction.

**Gene-grouped cross-validation: the gains are not locus memorisation.** Gene identity explains **29%** of the curated label against 16% of the published one (Appendix A), so more is available to memorise, yet with whole genes held out the three largest effects survive intact. The guide-only families (`b_energy`, `c_folding`, `f_methylation`) cannot memorise a locus and still retain only **94% in aggregate**: that 6% is the **grouping tax**, the cost of a harder partition (less locus diversity in training; 4,013 groups split five ways is coarser than 30,203 rows). `a_flank` (96%), `d_supercoiling` (94%) and `i_shape` (94%) lose nothing beyond it; `e_nucleoid` (88%) and `g_transcription` (79%) lose 6 and 15 points beyond it, a real locus component in two external position-indexed tracks; `j_periodicity` (64%) and `h_offtarget` (23%) lose most of gains of +0.011 and +0.001. *Predicted in advance:* twice the between-gene variance should widen the gap between random and grouped CV. *Observed:* the gap is smaller (grouping costs `e_nucleoid` 12% and `g_transcription` 21% of their gains here, 42% and 36% on the published cohort), because the curated cohort has **7.5 guides per gene against 11.3**: between-gene variance sets how much a locus-memorising model could win, guides per gene sets how well it learns each locus's mean, and the measurement says the second dominated.

**The column must not be added up.** Joint arms (`curated_joint.py`, same base, seeds and folds), joint Δρ against the sum of singles: `a_flank` + `i_shape` +0.1647 against +0.2833 (58% realised); `a_flank` + `d_supercoiling` +0.1882 against +0.3029 (62%); `d_supercoiling` + `e_nucleoid` + `g_transcription` +0.1454 against +0.2692 (54%); `a_flank` + `d_supercoiling` + `i_shape` +0.1882 against +0.4224 (45%); the guide-only control `b_energy` + `c_folding` + `f_methylation` +0.0289 against +0.0409 (**70%**); **all eleven families +0.2030** against +0.7046 (**29%**). Summing the column would claim +0.70 ρ; the real figure for all eleven is +0.203, only +0.039 beyond family A alone. `i_shape` adds +0.0009 on top of family A and exactly nothing once `d_supercoiling` is also present (the two three-family arms agree to four decimals, +0.1882). The guide-only control licenses the reading: those families overlap only through the 20-mer and are the most additive at **70%**, while the flank and positional families, which read overlapping DNA, realise 45–62%; had the control also come in near 50%, the conclusion would have been a pipeline artefact rather than an interpretation.

## Appendix B4 — The flank's shape and 3D structure add nothing that its composition does not already hold

**Claim.** Describing the flank by its DNA shape (13 base-pair and step features from three independent sources), or by a rigid-body reconstruction of its 3D helical path in a frame attached to the cut, recovers up to 90% of family A's gain on its own and adds **−0.0008 to +0.0006 ρ** to it. These descriptors carry window composition plus the identity of the first ten or so base pairs; family A holds both. Shuffling the flank, which destroys everything else, keeps the composition part of the gain. The test addresses whether the scalar-summary null for shape (`i_shape`, +0.0007 over family A) reflected the summary rather than the information: shape is a vector per step with direction, which a window mean cannot express. Post-freeze experiment; results in `results/post_freeze/`.

**Design.** The DNAshapeR pentamer table (512 pentamers × 90 numbers) was parsed into named columns by source (`mc` Monte Carlo, `xrc` X-ray, `md` molecular dynamics) and kind, and checked for reverse-complement consistency (a duplex read from either strand gives the same step parameters and, for the 3D chain, the same end-to-end distance and radius of gyration to four decimals). Three blocks per source: `mean` (arm × four windows × 13 features, 104 columns), `vec` (the first 8 bp of each arm position by position, 208) and `geom` (201: path positions, axis tangents, gyration tensor and its eigen-structure, straightness and excess twist, in the frame of the base pair 5′ of the cut). Gene-grouped CV, three seeds × five folds, 30,203 genic guides, the same folds for every arm, so differences are paired. Over the base (ρ 0.5588), alone, for md / xrc / mc: `mean` +0.099 / +0.102 / +0.103, `vec` +0.052 / +0.056 / +0.053, `geom` **+0.031** / +0.078 / +0.080, all three blocks (513 columns) +0.138 / +0.140 / +0.141, every cell ahead in 15 of 15 folds. References: family A (348 columns) +0.156; plain A/C/G/T fractions in the same windows (40 columns) **+0.105**; A's own 24 composition columns +0.107.

**On top of family A (ρ 0.7151)**, paired over the same folds: mc `mean` +0.0001, `vec` −0.0005, `geom` −0.0008, all three −0.0007; md / xrc `geom` −0.0006 / −0.0004; md / xrc all three −0.0007 / **+0.0006**; letters in windows (40 columns) −0.0001; all three sources (1,887 columns) −0.0006. **Shuffled-flank controls** (mc, all three blocks): flank shuffled in 25-nt blocks **−0.0012**, shuffled whole-arm **−0.0009**; the real blocks sit 0.0002–0.0018 above them, which is the cost of adding five hundred columns that cannot help. The null survives the obvious alternatives:

* **Displacement and depth.** At a 600-column budget instead of 300: A + mc all −0.0008, mc `geom` −0.0018, against A at 600. At depth 6 and 800 trees instead of the champion's 4 and 400 (A itself rises 0.0073 to 0.7225): A + mc all −0.0012, A + arm-shuffled mc all −0.0023.
* **A second cohort** (crisprHAL's *E. coli* eSpCas9 screen, 59,489 guides, 100-kb-block CV): the blocks alone recover 94% of A (+0.0969 against +0.1034); A + mc all +0.0022 (14/15 folds), A + mc `geom` +0.0008, **A + arm-shuffled mc all +0.0016**.
* **Fresh seeds** (44–46, folds never used before), curated cohort: A + mc all −0.0006, A + arm-shuffled mc all −0.0010, mc all alone +0.1411 against +0.1570 for A.

**Why the standalone gains exist, each explanation tested.**

1. *The sources agree and letters do as well.* The three sources agree on `mean` and `vec` to 0.004 although their parameter values differ (mean twist 34.3/33.1/32.5° and spread across pentamers 1.5/3.8/4.1°, mc/xrc/md), so the blocks do not read source-specific physics; forty columns of base fractions give +0.105 against +0.103 for the 104 shape columns, and adding the shape means to the letters changes ρ by −0.0002.
2. *The geometry's gain is composition.* `mc geom` on a flank shuffled within 25-nt blocks gives +0.080 (real +0.080); shuffled across the whole arm, keeping only its composition, +0.077.
3. *The geometry columns are dinucleotide counts plus the first base pairs.* Of the 27 geometry columns that survive selection, the five largest are excess twist (summed step twist minus 10.5 bp per turn) over 100–499 nt. Letter fractions predict them out of fold with R² 0.43–0.67 and dinucleotide fractions with R² **0.86–0.97** by ridge (0.85–0.94 by boosting): **0.968** for the downstream 499-nt window, 0.86 at 100 nt (`geom_importance.py`). A sum of step twists is, to that accuracy, a weighted dinucleotide count. The rest are path coordinates at +12 and +25 bp, with R² ≈ 0 from letters and at most 0.15 from dinucleotides: the identity of the first 12–25 bp after the PAM.
4. *Order matters in exactly one place.* `vec` loses 86% of its gain when the flank is shuffled in blocks (+0.053 → +0.008) and 94% when shuffled whole-arm; `mean` loses 4%, `geom` 0%.
5. *MD geometry is weaker alone (+0.031)* because the sources disagree on each guide's excess twist (cross-source correlations 0.27–0.66) and the MD value correlates with the label half as strongly (ρ −0.056 against −0.112 and −0.122). A phase mismatch from MD's mean twist is rejected: recentring every source to 34.3° leaves ρ unchanged (md 0.5914 against 0.5897). The cause beyond that was not identified.

**What family A is made of.** A is four kinds of column: 104 position-specific one-hot columns near the cut (`seq`), 156 per-position quantum-chemical tensor values at the same positions (`qct`), 64 windowed means and sds of four tensor series at 50–1,000 nt (`win`), and 24 composition summaries (`comp`). Alone over the base: `seq` +0.055, `qct` +0.070, `win` +0.109, **`comp` +0.107**. Removing one kind from A moves ρ by +0.0004 (`seq`, i.e. nothing), −0.0003 (`comp`), −0.0020 (`win`) and −0.0054 (`qct`). The flank effect therefore has two parts: a **window-scale composition gradient worth about +0.105, carried by 24 columns of GC and purine fraction**, and a **local component within ~10–15 nt of the cut worth +0.05 to +0.07**; together +0.156, less than their sum. Everything in the shape and 3D blocks falls inside those two parts.

**Limits.**

* **The null is about the boosted pipeline.** A *linear* model does see information beyond A in the same columns: ridge on all columns gives A + mc all **+0.0097** (5/5 folds) and A + md all +0.0078, against +0.0002 for the arm-shuffled control, mostly from the near-junction `vec` (+0.0048, control −0.0004) and the window means (+0.0054, control +0.0018). The boosted trees do not show it at depth 4 or 6; whether they already extract it from A and the base, or the gain is too small to split on, was not separated.
* **Tabulated pentamer means, static.** The values are not measurements of this genome's DNA, and no stiffness matrix or fluctuation was used. A neural shape predictor with seven base pairs of context (Deep DNAshape) was not run. Its output at a position is a deterministic function of the few nucleotides around it, so the model was given the inner ±50 nt one-hot explicitly (the 75 positions family A does not reach, 300 columns): at the tuned configuration this is worth **+0.0002 ρ [−0.0004, +0.0008]**, indistinguishable from the same columns shuffled across guides (−0.0004), with the selector picking 8.7 of the real columns and 5.3 of the shuffled ones out of 450 (`EXTENSION_LOG.md` §22.3). Any deterministic function of that window, Deep DNAshape included, is capped by it; what remains possible is a learnability claim (a better-*shaped* feature for information worth at most +0.0008 ρ), not an information one.
* **One organism; 3D path from mean step parameters.** The human case needs hg38 flanks for the human matrix, which the project's human file does not carry. The 3D path says nothing about nucleosome-free DNA looping or the supercoiled chromosome; the nucleoid's own 3D structure (Hi-C) is family E, whose null stands.
* **Still open** is narrower than "shape": a *measured*, non-sequence-derived quantity (a stiffness matrix, or a shape measurement of this genome's own DNA) that no local sequence predictor supplies. This is row 3 of the open-questions register in Part 4.

## Appendix B5 — A boundary experiment, demoted: the dCas9 / CRISPRi screens

**No headline claim rests on this experiment, and Parts 1–3 do not use it.** It was meant to test whether the flank effect needs a double-strand break, but the dCas9 CRISPRi screen (Cui et al., 2018) and the cutting screens differ in enzyme state, library, laboratory and phenotype (fitness after silencing against survival after a break) at once, so no mechanism follows from a difference in how much family A adds, however scaled. Neither reading below (13–14% or 26% of the attainable gain) would change the candidate mechanisms of Parts 2–3, and the one repair route sequence can test (Chi sites, Appendix B) was excluded on its own evidence. Four analyses of the same screens (`cui_screen.py`, `crispri_screens.py`, `strand_*.py`, `cut_within_gene.py`) supported four readings; only the last survives, and it is weakly identified.

| round | analysis | reading it supported | what overturned or qualified it |
|---|---|---|---|
| 1 | family A, base and flank-only, whole screen | **specific to cleavage**: −0.009 against +0.157 | a screen-wide average hides the guides that carry the signal |
| 2 | the same, split by gene essentiality | inside essential genes +0.27, which would **retire the localisation** | that +0.27 is the assay's own strand rule |
| 3 | strand supplied as one column | **the null holds**: +0.007 to +0.025 on Cui and Rousset; Wang does not replicate | the pooled ranking is dominated by strand, which hides a within-gene effect |
| 4 | within-gene ranking, coding strand only, ceiling-scaled share | **the effect exists without a break**: 13–14% of the attainable gain against 26% on cutting | not overturned; three caveats below |

### Family A adds nothing pooled; inside essential genes its +0.27 is the assay's strand rule

**Family A is worth −0.0093 on a silencing screen and +0.1567 on a cutting screen, measured the same way** (same pipeline and champion, 3 seeds × 5 folds, whole genes held out because 30% of the label's variance is between genes). Cui et al.'s 76,325 uniquely-targeting guides were located in NC_000913.2, 70,298 of them in 4,265 named genes; the 20-mer base reads 0.4577 ± 0.0013 and 0.4484 ± 0.0006 with family A, against 0.5584 and 0.7151 on cutting (`cui_screen.csv`; `ablation_curated.csv`), and family A alone reads 0.1333 (not zero, which rules out selection dilution), so the null belongs to the family and not to the run. *Limit: the average is not the informative subset.* A CRISPRi label needs the gene to matter, 85% of the genic cohort is non-essential, and essential-gene guides have label sd 2.886 against 1.133 (`cui_screen.py --essentiality`, `results/cui_essentiality_split.csv`).

**Inside essential genes family A reads +0.27, but it is a lossy copy of which strand the guide targets, the dominant covariate of a CRISPRi screen: one binary column beats the 348-column family (0.6621 against 0.5227), and given strand family A is worth +0.0069 (100 kb blocks) to +0.0148 (genes), an order of magnitude below the +0.157 of cutting.** dCas9 blocks RNA polymerase better from one strand (a property of the assay, not of Cas9). On Cui's 5,514 essential-gene guides (44 blocks, 297 genes; `cui_screen.py --marginals`, `results/cui_strand_summary.csv`) coding-strand guides average −5.07 against −1.19 (1.35 sd), strand alone explains 45% of the label's variance, and a coding-strand guide is more depleted than an off-strand one 88% of the time; a binary column cannot exceed ρ 0.863 at these group sizes and strand reaches 0.662 (77%). The control is the non-essential genes, where the same statistic is flat (gap 0.15 sd, 0.6% of variance, AUC 0.54 against 0.88). The Finding 2 column carries nothing here (flank GC at ±500 nt, marginal Spearman −0.0004); family A's strongest column, `eng.flank.comp.up1000.purine` (+0.4045), counts purines on the guide's own strand and so flips with orientation.

| ρ, or increment of family A (`cui_screen.py --strand`, `cui_essential_strand.csv`, `crispri_screen_family_a.csv`) | Cui, 100 kb blocks | Cui, genes | Rousset, genes |
|---|---:|---:|---:|
| increment, all genic guides | | −0.0093 | −0.0206 |
| increment, non-essential genes | | −0.0101 | −0.0250 |
| *essential genes:* 20-mer base | 0.2805 | 0.2718 | 0.2141 |
| base + family A | 0.5484 | 0.5468 | 0.5043 |
| *family A, apparently* | *+0.2679* | *+0.2749* | *+0.2902* |
| **which strand the guide targets, alone (1 column)** | **0.6621** | | |
| base + strand | 0.6506 | 0.6462 | 0.5878 |
| **family A, given strand** | **+0.0069** | **+0.0148** | **+0.0245** |

Holding out 100 kb blocks instead of genes, the control for essential genes clustering in the genome, leaves the +0.27 unchanged. A position-only control rules out a location lookup (the bare coordinate buys +0.0394 under block grouping; `results/cui_essential_position.csv`), and telling the model the wrong strand on every held-out guide ranks the screen backwards (ρ −0.528 against 0.646; `strand_strata.py --flip`). No other family beats the strand call: `strand_families.py` (all eleven families, 5,514 guides, three seeds, both groupings; `results/strand_families.csv`) gives `a_flank` +0.015 (genes) and +0.007 (blocks) over `base + strand`, `g_transcription` (expression, gene geometry, replication context) +0.057 and +0.028, and the other nine −0.013 to +0.030; seed sd is 0.002–0.01 pooled and 0.01–0.03 within a strand. Only `g_transcription` is clearly positive in both groupings and it is not a flank family; the largest other gains (`d_supercoiling`, `e_nucleoid`) change between groupings and are not claimed. *Limits:* dropping the columns most correlated with strand does not make an arm strand-free (an out-of-fold classifier on the rest still reads strand at AUC 0.79 for A and 0.9997 for `g_transcription`), so the `-proxy` arms are not quoted (`strand_recoverable.py`, `results/strand_families_recoverable.csv`); the +0.007 to +0.025 is the gain to the pooled ranking, which strand dominates (0.28 to 0.65 from one column), and does not say A is inert within a strand (below); the informative subset is 5,514 guides in 44 blocks, the control rests on one strand indicator, and nothing here is an intervention.

**On Rousset all three steps hold** (59,246 guides, the Cui library under a weaker dCas9 promoter; `crispri_screens.py --family-a`, `results/crispri_screen_family_a.csv`); under 100 kb blocks it gives +0.2916 apparently and +0.0213 given strand, against +0.2679 and +0.0069 on Cui. The strand rule replicates (`crispri_screens.py --strand`, `results/crispri_screen_strand.csv`): ρ(coding strand, label) is −0.633 on Rousset and −0.599 on Cui inside essential genes against −0.056 and −0.048 outside. **Wang does not replicate, and the locus control does not explain it away** (all genic +0.1019; essential genes given strand +0.2669 by gene and +0.2195 by block). Its strand control is vacuous (52,104 of 52,104 guides are coding-strand), 0.766 of its label variance is between genes (0.476 Rousset, 0.327 Cui) and its 20-mer base reaches only ρ 0.0829 in essential genes (0.2141, 0.2718); the within-gene comparison has not been run on Wang, so the null reproduces on two of three screens and the third is unexplained. All three are *E. coli* K-12 MG1655 CRISPRi, so replication adds robustness for one control, not reach.

### Within genes, on the coding strand, family A keeps 13–14% of the attainable gain against 26% on cutting (hypothesis)

**The pooled null is false inside the coding strand.** Fitted and scored within one strand, family A gains +0.124 (genes) and +0.119 (blocks) over `base` on Cui's 2,535 coding-strand guides and +0.159 and +0.133 on Rousset's 2,074, and loses on the template strand (−0.026, −0.040 and −0.019, −0.022); Wang's +0.267 is this stratum (`strand_families.py`; Rousset `strand_strata.py --rousset`); gene-pooled within the coding strand `g_transcription` gains more than A (+0.198 blocks, +0.267 genes; the other nine −0.008 to +0.081 and −0.000 to +0.154). Within-gene ρ (out-of-fold predictions with each gene's mean removed; `strand_within_gene.py`, ten seeds; 206 Cui and 182 Rousset genes with at least five coding-strand guides) is largest for family A of any family, and adding A to `g_transcription`, which carries the guide's relative position in its gene, still leaves **+0.096** within genes (seed sd 0.012; between genes +0.006). The cutting screen decomposes as +0.275 between genes and +0.155 overall (30,203 genic guides, 4,013 genes, ten seeds; `ext/cut_within_gene.py`; within-gene seed sd 0.0007). Raw dCas9 gains are not smaller but start from within-gene bases of 0.17 and 0.08 against 0.62, so they are scaled by the attainable gain and by variance explained:

| within-gene ranking | genes | base | with A | Δ within [95% bootstrap over genes] | ceiling | share of attainable gain | Δρ² |
|---|---:|---:|---:|---:|---:|---:|---:|
| cutting screen | 4,013 | 0.618 | 0.703 | +0.085 [+0.078, +0.090] | 0.946 (Finding 6) | **26%** | +0.113 |
| Cui, coding strand | 206 | 0.172 | 0.274 | +0.102 [+0.081, +0.164] | ≈ 0.93 | **13%** | +0.045 |
| Rousset, coding strand | 182 | 0.083 | 0.196 | +0.114 [+0.084, +0.189] | ≥ 0.864 | **14%** | +0.031 |

The dCas9 effect is between a quarter and a half of the cutting one (half by share of the attainable gain, 27–40% by variance explained). The dCas9 ceiling is the Spearman–Brown reliability of the mean of the screen's two time points (18 and 75 generations; within-gene ρ 0.768 over 2,289 guides), square-rooted. *Three limits keep this out of Parts 1–3:* (i) the two time points are not true replicates, so a systematic difference between them would raise the true ceiling and shrink the dCas9 share; (ii) Rousset has one readout, so its ceiling is a lower bound from its agreement with the E18 screen of the same guides (within-gene ρ 0.805 over 1,759 guides) and its share is at most 14%; (iii) a within-gene CRISPRi effect could be gene-local transcription context, which the `g_transcription` control narrows without closing. Whether the gain is a local flank property or a position effect that relative position does not capture is not identified; settling it needs a design that varies flank composition at a fixed guide, which no existing screen provides.

### A second ceiling: screens agree with each other far less than replicates do

**The transferable-signal ceiling across libraries and laboratories is ≈ 0.77–0.79 inside essential genes and ≈ 0.27–0.30 outside them, against 0.946 for replicates (Finding 6), which is a within-experiment quantity.** Rousset and Cui are one 59,246-guide library under a weaker and a stronger dCas9 promoter; Wang is a different library sharing 6,139 guides with them (`crispri_screens.py --agreement`, `results/crispri_screen_agreement.csv`; n 6,139–59,246 all guides, 529–4,640 essential, 5,610–54,606 non-essential).

| pair | what differs | all guides | essential genes | non-essential |
|---|---|---:|---:|---:|
| Rousset vs Cui | dCas9 promoter strength only | +0.689 | **+0.902** | +0.659 |
| Rousset vs Wang | library, laboratory, protocol | +0.429 | **+0.790** | +0.298 |
| Cui vs Wang | the same | +0.401 | **+0.765** | +0.271 |

A model fitted on one screen cannot score above that screen's agreement with another, so the smaller ceiling binds any transfer claim. *Limit:* these are CRISPRi screens, so it bounds the dCas9 claims and not the cutting ones; the curated Cas9 cohort has only its own replicates.

**Method result: supply the assay's dominant covariate before reading a subgroup.** A family that holds a lossy copy of a screen's dominant covariate looks powerful: family A was worth +0.27 inside the essential genes and +0.007 to +0.025 once strand was supplied (Part 2, method list); the same strand term dominates the non-sequence features of a published human base-editing model (Part 5). Scope: the predictor is for cleavage, and a CRISPRi user needs a guide-sequence model.

## Appendix C0 — Reproducibility: versions, seeds and protocol

| | |
|---|---|
| software | Python 3.12.15; numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1; LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10; SHAP 0.52.0; PyTorch 2.14.1 (the CNN controls) |
| machine, for every timing in this report | Apple M5, 10 cores, 16 GB, macOS 26.6.2 |
| seeds | `config.SEEDS = (41, 42, 43, 44, 45)`; `config.N_SPLITS = 5`; every seed is passed explicitly to the model, the selector and the fold splitter, so no arm depends on global RNG state |
| replication | ablations five seeds × five folds (25 fits per arm); shift-distance null and GC-strata test three seeds × five folds; head-to-head against crisprHAL 2 seed 41 only, because each of their fits costs ~21 min |
| selector | `config.SELECTOR = "xgboost"`, a throwaway model that only ranks columns; fitted **inside the training split of each fold**, never on the full data |
| resumability | each fold is written to `results/folds_<tag>.csv` as it finishes and a recorded fold is skipped, so a crash loses one fold rather than an experiment |

**Timings.** Every millisecond figure in this report (featuriser costs, end-to-end tables, ratios against crisprHAL 2) is wall-clock on **one thread** of the machine above, median of repeated calls after a warm-up, with `OMP_NUM_THREADS` and the three sibling variables pinned to 1 so no arm uses more cores than another. The **ratios travel and the absolutes do not**: another machine will move 0.036 and 1.214 ms together, and the 34× with much less. The audit compares each quoted timing with the measurement stored in `results/post_freeze/`, so re-measuring on other hardware makes those lines fail, correctly; re-quote them rather than widening the tolerance. A module that measures at import can silently re-measure them under load (`EXTENSION_LOG.md` §22.4).

## Appendix C — Feature sets and data sources

### Eleven folders are not eleven kinds, and which nulls are worth keeping

**Six of the eleven feature folders bundle measurements of different things, and in four cases the bundling hides the result.** Δρ is the curated-cohort increment over the 6,169-column base (Appendix B3); published-matrix figures are uniformly smaller and rank almost identically (ρ = 0.97), for the reason Appendix B2a gives. Guide-derived folders are deterministic functions of the 20-mer, so they can add performance but not information (Finding 1). A null earns its place when it kills a mechanism someone would otherwise have to assume, not when it is an artefact of how the cohort was built: five rows below do (marked **Null**), one does not.

| kind | set | columns | what it carries; what a null rules out | Δρ |
|---|---|---:|---|---:|
| flank | `a_flank` | 348 | DNA outside the protospacer in three layers: **near-cut sequence** (±10 nt) +0.074 alone; **crisprHAL's own ±189/166 nt span** +0.055 more; **50/250/500/1,000 nt windowed composition** +0.035 more. Only the windows are information crisprHAL 2's input cannot reach (Finding 2) | **+0.164** |
| guide-derived | `b_energy` | 10 | R-loop free-energy budget (hybridisation, DNA opening, guide self-folding, net binding, distance from the −64.5 to −47.1 kcal/mol window where efficient guides cluster) | +0.019 |
| guide-derived | `c_folding` | 9 | secondary structure of the transcript made: spacer MFE and spacer-against-scaffold pairing | +0.014 |
| guide-derived | `f_methylation` | 17 | Dam `GATC` and Dcm `CCWGG` motifs by position relative to the PAM. **Null:** a published ~10× lower SaCas9 activity when a Dam `GATC` overlaps the PAM-proximal positions does not reproduce for SpCas9 on this screen | +0.008 |
| target mechanics | `d_mechanics` | 30 | **two sets**: 22 columns of protospacer duplex thermodynamics **+0.011**; 8 columns of the same on the flanks at ±25 and ±100 nt **+0.097**, which is family A under another name | +0.099 |
| target mechanics | `i_shape` | 152 | dinucleotide-step deformability of the flanks; **+0.120 alone**, **+0.0007** once `a_flank` is present. **Null:** shape and composition are one channel twice, a statement about DNA descriptors and not about Cas9 | +0.120 |
| locus | `d_supercoiling` | 26 | GapR-seq positive-supercoiling signal (GSE152880), several window widths. **Null, the most useful here:** survives replacing the track with the **untagged control**, so it measures sequencing read density; this answers Guo et al.'s "unknown chromosomal factors" for this family, and was caught only because the experiment shipped an untagged control (Appendix B3) | +0.139 → artefact |
| locus | `e_nucleoid` | 20 | **two sets**: Hi-C 3D position; NAP dependence (contacts per bin in wild type against H-NS, HU, Fis, MatP, MukB and ZapB mutants) | +0.072 |
| locus | `g_transcription` | 19 | **three sets, one pays**: gene geometry **+0.0393**; PRECISE-1K expression **+0.0213**; essentiality **+0.00004**. **Null:** the essentiality term replicates Guo et al.'s essentiality null to four decimal places on a different cohort, while their **expression** null does not: one confirmation and one correction | +0.058 |
| cohort-defined-away | `h_offtarget` | 13 | **7 live, 6 dead**: protospacer copy number plus 6 near-cognate-mismatch columns identically zero because the curated cohort keeps only uniquely-targeting guides. **Not a null:** the question was removed before the measurement, and quoting it beside the others would imply a specificity conclusion the data cannot support (Part 2) | +0.001 |
| periodicity | `j_periodicity` | 24 | GC skew and helical-repeat phasing. **Null:** excludes the most mechanistically motivated explanations for a long-range, strand-asymmetric effect | flat |

`b_energy` and `c_folding` are not independent witnesses: one column is the same quantity computed twice (Pearson 0.9993), so the folding evidence is two measurements, not three (Appendix B). The three locus folders are close to one variable measured three ways (jointly +0.145).

### Teammates' results on the published 13,880-row matrix, quoted in R² (not comparable with the Δρ column)

| set | who | result |
|---|---|---|
| **gene essentiality** | Jacky | **+0.0037 R²** over 0.2927, **selected in 5/5 folds**: the only non-QCT idea on that matrix with a consistent gain, and not a function of the 20-mer |
| **orientation controls** | Jacky | closes Part 4 item 3: curated RegulonDB TSS replaces nearest gene as the orientation variable (Δ 0.051 [0.020, 0.082]); the nearest-gene version is largely a distance/GC cohort confound (Δ 0.007, P = 0.34 after matching); block bootstrap over 47 genomic blocks, promoter/RBS separated, operon compared |
| QCT variants | Joshua | seed non-additive ~+0.005 R² (absolute-difference forms carry more SHAP than signed, 1.09 against 0.92); log-ratio forms ~+0.005, best seed 0.3015; regional summaries +0.0013 against a permuted control at 0.2857; PAM-weighted, profile-shape, coupling and HOMO–LUMO asymmetry −0.0010 to +0.0006, null |

The non-additive families are **independent evidence for Finding 4**: they gain from exactly the higher-order structure the k-mer ladder found worth +0.0035 ρ once the selector was removed.

### Data added after the freeze

| data | source | size | used for |
|---|---|---|---|
| 23 efficiency tables for human, mouse, zebrafish, rat, fly and worm (`*.scores.tab`, plain text; read as data, never executed) | the CRISPOR benchmark collection, github.com/maximilianh/crisporPaper, `effData/`, which redistributes the supplementary tables of the original papers (Doench 2014 and 2016, Chari 2015, Xu 2015, Wang 2015, Hart 2016, Moreno-Mateos 2015 and others) | about 7 MB, in `external_data/crispor_effdata/` (not committed) | Finding 7, the eukaryote comparison (`eukaryote_rules.py`) |

## Appendix D — Sources (APA 7th edition)

Browne, T. S., Edgell, D. R., & Gloor, G. B. (2026). Better data for better
predictions: Data curation improves deep learning for sgRNA/Cas9 prediction.
*PeerJ*, *14*, e20706. https://doi.org/10.7717/peerj.20706
— the state of the art; code and curated label re-run here as the baseline.

Chiu, T.-P., Comoglio, F., Zhou, T., Yang, L., Paro, R., & Rohs, R. (2016).
DNAshapeR: an R/Bioconductor package for DNA shape prediction and feature
encoding. *Bioinformatics*, *32*(8), 1211–1213.
https://doi.org/10.1093/bioinformatics/btv735
— source of the pentamer shape table (`src/TableCompiler.cpp`, read as text) used in
Appendix B4; the table lumps Monte Carlo, X-ray and molecular-dynamics values.

Cui, L., Vigouroux, A., Rousset, F., Varet, H., Khanna, V., & Bikard, D. (2018).
A CRISPRi screen in *E. coli* reveals sequence-specific toxicity of dCas9.
*Nature Communications*, *9*, 1912. https://doi.org/10.1038/s41467-018-04209-5
— the dCas9 CRISPRi screen used as the final untouched test (Appendix B5); its two
measurement arms also set the ceiling range quoted in Appendix B.

DeWeirdt, P. C., McGee, A. V., Zheng, F., Nwolah, I., Hegde, M., & Doench, J. G.
(2022). Accounting for small variations in the tracrRNA sequence improves sgRNA
activity predictions for CRISPR screening. *Nature Communications*, *13*(1), 5255.
https://doi.org/10.1038/s41467-022-33024-2
— Rule Set 3; cited in Part 5 for how the human predictors are ranked and how the
ranking moves between datasets.

Doench, J. G., Fusi, N., Sullender, M., Hegde, M., Vaimberg, E. W., Donovan, K. F.,
Smith, I., Tothova, Z., Wilen, C., Orchard, R., Virgin, H. W., Listgarten, J., &
Root, D. E. (2016). Optimized sgRNA design to maximize activity and minimize
off-target effects of CRISPR-Cas9. *Nature Biotechnology*, *34*(2), 184–191.
https://doi.org/10.1038/nbt.3437
— Rule Set 2, the reference point for guide-intrinsic design rules (Part 5).

Guo, J., Wang, T., Guan, C., Liu, B., Luo, C., Xie, Z., Zhang, C., & Xing, X.
(2018). Improved sgRNA design in bacteria via genome-wide activity profiling.
*Nucleic Acids Research*, *46*(14), 7052–7069. https://doi.org/10.1093/nar/gky572
— the screen behind the label; Figure 2b and 2c agreement statistics for the
published label's ceiling; the intergenic sub-library. Its per-replicate counts,
as re-released by Browne et al., give the curated label's ceiling (Finding 6).

Guo, M. S., Kawamura, R., Littlehale, M. L., Marko, J. F., & Laub, M. T. (2021).
High-resolution, genome-wide mapping of positive supercoiling in chromosomes.
*eLife*, *10*, e67236. https://doi.org/10.7554/eLife.67236
— GapR-seq; positive-supercoiling density (GEO accession GSE152880).

Ham, D. T., Browne, T. S., Banglorewala, P. N., Wilson, T. L., Michael, R. K.,
Gloor, G. B., & Edgell, D. R. (2023). A generalizable Cas9/sgRNA prediction
model using machine transfer learning with small high-quality datasets. *Nature
Communications*, *14*(1), 5514. https://doi.org/10.1038/s41467-023-41143-7
— the *C. rodentium* screen used for cross-organism transfer.

Haeussler, M., Schönig, K., Eckert, H., Eschstruth, A., Mianné, J., Renaud, J.-B.,
Schneider-Maunoury, S., Shkumatava, A., Teboul, L., Kent, J., Joly, J.-S., &
Concordet, J.-P. (2016). Evaluation of off-target and on-target scoring algorithms
and integration into the guide RNA selection tool CRISPOR. *Genome Biology*,
*17*(1), 148. https://doi.org/10.1186/s13059-016-1012-2
— the source of the replicate correlations (0.71–0.77) that the field's ∼0.7 ceiling
is built on, and an early demonstration that duplicate on-target entries make random
cross-validation train and test on the same guide (Part 5).

Konstantakos, V., Nentidis, A., Krithara, A., & Paliouras, G. (2022). CRISPR–Cas9
gRNA efficiency prediction: an overview of predictive tools and the role of deep
learning. *Nucleic Acids Research*, *50*(7), 3616–3637.
https://doi.org/10.1093/nar/gkac192
— the review that sets this field's benchmarking frame, and the source of the
"∼0.7 constitutes an upper limit of any prediction" claim that Part 5 shows to be
understated by 0.21–0.23 ρ. Quoted from the full text.

Li, J., Sagendorf, J. M., Chiu, T.-P., Pasi, M., Perez, A., & Rohs, R. (2017).
Expanding the repertoire of DNA shape features for genome-scale studies of
transcription factor binding. *Nucleic Acids Research*, *45*(22), 12877–12887.
https://doi.org/10.1093/nar/gkx1145
— the 13-feature shape set (Monte Carlo, with X-ray and MD comparison columns).

Lioy, V. S., Cournac, A., Marbouty, M., Duigou, S., Mozziconacci, J., Espéli, O.,
Boccard, F., & Koszul, R. (2018). Multiscale structuring of the *E. coli*
chromosome by nucleoid-associated and condensin proteins. *Cell*, *172*(4),
771–783.e18. https://doi.org/10.1016/j.cell.2017.12.027
— Hi-C contact structure and nucleoid-associated protein dependence.

Lones, M. A. (2021). How to avoid machine learning pitfalls: A guide for academic researchers. *arXiv*, 2108.02497.
https://arxiv.org/abs/2108.02497
— cited in Part 5 for the guidance that a comparison between a tuned method and an untuned baseline is not a fair test of the method.

Lu, X.-J., & Olson, W. K. (2003). 3DNA: a software package for the analysis,
rebuilding and visualization of three-dimensional nucleic acid structures.
*Nucleic Acids Research*, *31*(17), 5108–5121. https://doi.org/10.1093/nar/gkg680
— the step-parameter convention used to compose the 3D path (mid-step frame).

Noshay, J. M., Walker, T., Alexander, W. G., Klingeman, D. M., Romero, J.,
Walker, A. M., Prates, E., Eckert, C., Irle, S., Kainer, D., & Jacobson, D. A.
(2023). Quantum biological insights into CRISPR-Cas9 sgRNA efficiency from
explainable-AI driven feature engineering. *Nucleic Acids Research*, *51*(19),
10147–10161. https://doi.org/10.1093/nar/gkad736
— the feature matrix decoded here; their iRF re-runs in our harness at Pearson
0.4881 against their reported 0.5019, on a third of their rows.

Moreb, E. A., & Lynch, M. D. (2021). Genome dependent Cas9/gRNA search time
underlies sequence dependent gRNA activity. *Nature Communications*, *12*, 5034.
https://doi.org/10.1038/s41467-021-25339-3
— a retrospective analysis of 44 published gRNA datasets across species; read from its abstract
only. It reports that prediction algorithms do not carry between species and proposes that
sequence acts mainly through the time Cas9 takes to find its target, in a genome-dependent way
(Finding 7, the hypotheses table).

Orro, A., Consiglio, A., Curci, M. I., Scichilone, M., Hasin, F., Minervini, M.,
Mencar, C., De Bellis, G., Cocola, C., Pelucchi, P., & Selmi, T. (2026). Machine
learning-based prediction of base editor sgRNA fitness score. *Scientific Reports*.
Advance online publication, 10 November 2026.
https://doi.org/10.1038/s41598-026-79920-1
(preprint: https://doi.org/10.21203/rs.3.rs-9224117/v1, which is the version read here)
— base-editing fitness prediction over ~11,000 sgRNAs and 47 genes. Its SHAP position
profile independently reproduces this project's gain-by-position result, and gene strand
is its largest non-sequence feature, as strand was in our dCas9 screen. It finds gene
essentiality weakly predictive where we find it null, which the two labels explain
(Part 5).

SantaLucia, J., & Hicks, D. (2004). The thermodynamics of DNA structural motifs.
*Annual Review of Biophysics and Biomolecular Structure*, *33*(1), 415–440.
https://doi.org/10.1146/annurev.biophys.32.110601.141800
— nearest-neighbour thermodynamic parameters.

Sastry, A. V., Gao, Y., Szubin, R., Hefner, Y., Xu, S., Kim, D., Choudhary, K. S.,
Yang, L., King, Z. A., & Palsson, B. O. (2019). The *Escherichia coli*
transcriptome mostly consists of independently regulated modules. *Nature
Communications*, *10*(1), 5536. https://doi.org/10.1038/s41467-019-13483-w
— the iModulon decomposition behind the PRECISE expression compendium.

Xiang, X., Corsi, G. I., Anthon, C., Qu, K., Pan, X., Liang, X., Han, P., Dong, Z.,
Liu, L., Zhong, J., Ma, T., Wang, J., Zhang, X., Jiang, H., Xu, F., Liu, X., Xu, X.,
Wang, J., Yang, H., Bolund, L., Church, G. M., Lin, L., Gorodkin, J., & Luo, Y.
(2021). Enhancing CRISPR-Cas9 gRNA efficiency prediction by data integration and
deep learning. *Nature Communications*, *12*(1), 3238.
https://doi.org/10.1038/s41467-021-23576-0
— CRISPRon; the one predictor in Part 5's table that reads outside the protospacer
via an *experimental* track (chromatin accessibility) rather than sequence.

Yu, Y., Gawlitt, S., de Andrade e Sousa, L. B., Merdivan, E., Piraud, M.,
Beisel, C. L., & Barquist, L. (2024). Improved prediction of bacterial CRISPRi guide
efficiency from depletion screens through mixed-effect machine learning and data
integration. *Genome Biology*, *25*(1), 13.
https://doi.org/10.1186/s13059-023-03153-y
— the closest methodological neighbour: gene-wise cross-validation and a mixed-effect
model separating guide efficiency from the targeted gene, on bacterial CRISPRi. Their
guide-only model reaches ρ ≈ 0.21 and gene-level features take it to ≈ 0.66, which is
the same guide/gene decomposition this project makes (Part 5).

Zarate, O. A., Yang, Y., Wang, X., & Wang, J.-P. (2022). BoostMEC: predicting
CRISPR-Cas9 cleavage efficiency through boosting models. *BMC Bioinformatics*,
*23*(1), 446. https://doi.org/10.1186/s12859-022-04998-z
— the published precedent for this project's *architectural* choice: LightGBM over
engineered sgRNA features, benchmarked against 10 models on 13 external datasets and
argued for on interpretability. Cited in Part 5 to show that "boosted trees can match a
deep model" is not what is new here.

Zhou, T., Yang, L., Lu, Y., Dror, I., Dantas Machado, A. C., Ghane, T., Di Felice,
R., & Rohs, R. (2013). DNAshape: a method for the high-throughput prediction of DNA
structural features on a genomic scale. *Nucleic Acids Research*, *41*(W1), W56–W62.
https://doi.org/10.1093/nar/gkt437
— the original pentamer-based shape method.

**Two database citations still need their canonical form**, because Crossref
does not resolve them from the version strings we use and they should not be
guessed: **RegulonDB v14.5** (release citation) and **PRECISE-1K** (the 1,035-run
compendium, successor to Sastry et al. 2019). Everything above was verified
against Crossref metadata rather than transcribed from memory.
