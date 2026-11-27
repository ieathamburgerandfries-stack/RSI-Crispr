# SLICER — what determines whether a CRISPR guide cuts, in *E. coli*

**S**imple **L**ightweight **I**nterpretable **C**RISPR **E**fficiency **R**anker. The analysis is frozen as of 11 October 2026 (Part 4).

## Summary

Eight claims, in the order of the paper, each defended in the finding or part named beside it. **One protocol unless a number is labelled otherwise:** five-fold cross-validation that holds whole genes out (all 33,567 curated guides; guides outside genes grouped in 100 kb blocks), three seeds, SLICER's default model (400 trees, 300 columns chosen inside each training fold, nothing searched) and the 20-mer alone as the baseline. The team's shared harness assigns rows to folds at random (five seeds of five-fold `KFold`); the model and its selection are the same, so only the fold assignment differs, and Appendix B3 gives both side by side (the flank gain is +0.161 on the paper's folds, +0.164 on random rows, +0.157 on the 30,203 genic guides only).

1. **A published feature matrix is a re-encoding of the guide, and the guide-derived families add under 0.02.** 99.0% of its 6,232 columns regenerate from the 20-mer and spell about twenty variables; 412 columns computable for any guide lose 0.00003 ρ against all 6,232 (published matrix, random-row folds of the team's harness; paired interval [−0.0029, +0.0030]). On the paper's folds the energy, folding, methylation, periodicity and off-target families add +0.0181, +0.0143, +0.0077, +0.0081 and −0.0003 ρ (the same 300-column selection on both arms), and thermodynamics adds +0.0972, 99% of it from the eight columns that read flanking DNA (Finding 1; Appendix B3).
2. **The DNA around the cut site predicts cutting, and it is the one large effect.** Flank features raise ρ from 0.547 to 0.707: **+0.161** (sd 0.010 over 15 folds, positive in all). Sealing 20% of the guides by whole gene and re-deriving selection on the rest gives +0.156; scored with eSpCas9 labels the sealed genes give +0.103 [+0.100, +0.107]; inside genes alone +0.085 [+0.078, +0.090] survives (ten seeds), so it is not only gene ranking. Reading the flank features from the guide nearest to an offset of 100 bp, 300 bp, 1 kb or 3 kb keeps 68%, 57%, 32% and 6% of the gain and from 10 kb or more none of it. Repeats, mappability and read depth do not account for it: +0.157 among the 26,146 guides whose every 25-mer within 1 kb is unique, +0.261 among 852 guides with over 10% repeated 25-mers, +0.160 given the control read depth as a column. Across organism it keeps 91% (to *C. rodentium*) and 89% (back) of the locally trained ρ (Findings 2 and 3).
3. **The signal is GC over a few hundred bases, a step at the AT-rich end.** Two GC columns (the GC fraction of the W nt before the protospacer and after the PAM) give +0.106 at W = 200 nt, the highest of 14 windows (10 to 5,000 nt) and within 0.01 of it from 150 to 500 nt; +0.042 at 5 kb. The purine fraction and the amino fraction in the same windows give no gain (−0.0014 to 0.0000), and the four bases separately at most +0.103. The lowest decile of flank GC (mean −3.154) sits 3.16 label units below the rest and deciles 3 to 10 lie within 0.15 of zero; inside genes the dependence is graded. The dependence on flank GC differs with the base at guide position 20 (Spearman 0.219 [0.186, 0.246] with A, 0.119 [0.089, 0.146] with G), but a flank × guide-score interaction term fits only negligibly better than an additive model (R² +0.0035, ρ +0.0007). The guide's own position × base rules are unrelated between *E. coli* and eukaryotes (median map correlation −0.14 over eight pairs; G at position 20 is −0.33 SD here and positive in all eleven eukaryotic datasets) (Findings 3 and 7).
4. **The model is out of material, and it is interpretable without being shown to be more interpretable than crisprHAL 2.** No published-matrix column correlates with the champion's out-of-fold residuals above 0.028. Per-column importance on the full matrix is unstable because 6,095 columns restate about twenty variables; a 104-column non-redundant representation (80 base indicators, 24 flank columns; 88% of the full model's ρ) is readable column by column: SHAP agrees at 0.87 across five tree families, puts 83% of the attribution on the guide indicators (61% of it on positions 15 to 20; G at position 20 −0.33 SD) and 17% on the flank columns (79% of it GC), and the lowest decile of flank GC contributes −0.78 SD, 2.4 times the largest single-base effect (Finding 5; Appendix A2 of the paper).
5. **The label's ceiling is about 0.946 [0.944, 0.947]** (from crisprHAL 2's replicate counts; 0.942 with a second estimator of the released score), and the best single model reaches 0.7314 (gene-grouped CV, 30,203 genic guides; 0.7215 on the 1,204 genes the settings never touched), so 0.21 to 0.22 of ρ is unexplained. The field's ceiling of about 0.7 (Haeussler et al. 2016, repeated by Konstantakos et al. 2022) treats a reliability as a ceiling: for a score averaged over two replicates the same agreement of 0.71 to 0.77 gives 0.91 to 0.93 (Finding 6).
6. **SLICER, as shipped, leads crisprHAL 2, as released, on crisprHAL 2's own splits** by **+0.0286** [+0.0239, +0.0331] on eSpCas9 and **+0.0226** [+0.0139, +0.0314] on SpCas9 (paired 95% bootstrap over test guides; the settings search used the labels of the SpCas9 test guides in its development genes, though no weights were fitted on a test row; on the 1,204 genes the search never touched the lead is +0.0324 [+0.0248, +0.0403]). Before tuning no difference is detected (+0.0021 [−0.0031, +0.0072] on eSpCas9, +0.0039 [−0.0047, +0.0124] on SpCas9). On the whole-gene folds with crisprHAL 2 retrained at its published settings (one seed) it reaches 0.693 against 0.707 for the default model (+0.014, ahead in 5 of 5 folds) and 0.723 for the tuned model (+0.031). Reach explains the default model's small lead and not the tuned model's larger one. Both were tuned with comparable compute, not equal draws: 627 draws in 2.4 hours for SLICER against 28 draws (33 fits, about 3.0 GPU-hours) for crisprHAL 2 after an earlier 10-draw search of 7.3 CPU-hours; its published configuration won four of five folds of its own search. BoostMEC, retrained on the cohort from its own pipeline's features, reaches 0.620 (released hyperparameters) and 0.623 (its own search), −0.066 as released (Part 5).
7. **SLICER is cheaper to run and to fit than the state of the art, by hardware.** On a quiet laptop CPU, one thread, from the raw sequence, the default model scores a guide 15.4 to 24.5 times faster than crisprHAL 2 and 42.3 times faster than BoostMEC (batch of 100), and fits 81 times faster on the same laptop (12.8 s against 17.3 min); against crisprHAL 2 on a Colab T4 the fit ratio is 25 times. CPU-seconds are not compared (XGBoost's threads spin while they wait). On the T4 session (one VM, not a quiet machine) crisprHAL 2 costs 3.9 times SLICER's laptop path from raw contexts (two machines) and its forward pass 13 times the forward pass of SLICER's trees on the same VM. SLICER's default model stores 6,338 leaves (0.68 MB) against 136,142 for BoostMEC's released model (11.5 MB) and 790,533 parameters for crisprHAL 2 (Part 5).
8. **The sealed-gene drop is small, and not shown to be adaptive overfitting.** Sealing 20% of the guides by whole gene lowers family A's gain by +0.0087 ρ [+0.0020, +0.0153; mean ± 1.96 se over 15 sealings]; the frozen configuration re-derives no hyperparameter, an 8-draw search adds nothing detectable, and holding out whole genes inside the pool removes only +0.0007 of it, so the figure is the most that automatable adaptive overfitting could be; the tuned settings' benefit survives a genome nothing was chosen on (+0.0172 over the frozen champion, zero-shot on *C. rodentium*, one 236 kb region) (Part 4).

## How to read this document

* **Part 1** is the seven findings, each a claim followed by the evidence and its limit. **Part 2** states the contributions and what is new about the biology. **Part 3** says where the work stops. **Part 4** is the validation: the freeze, the audit of what could have inflated the figures, and the sealed checks. **Part 5** places the work against the published predictors.
* **Appendices.** A: every control, with what it rules out and what would be believed without it, and the inventory of nulls. B: the methods and controls behind the findings. B2–B4: the imported families that did not work, what curation is, the family table on the curated cohort, and the null on the flank's shape and 3D structure. B5: a boundary experiment on dCas9 screens that no claim rests on. C0 and C: reproducibility and data sources. D: references.
* **Two *E. coli* SpCas9 cohorts.** Noshay et al.'s published 13,880-row matrix (label `cut.score`; their iRF used 40,468 sgRNAs of Guo et al.'s screen, 32,374 for training; the released file holds 13,880 rows, the first 1,225 genes of the screen in alphabetical order (aaeA to leuC, 1,222 after our gene assignment), cut off mid-file; the paper does not say why) and crisprHAL 2's curated 33,567-guide genome-wide release of the same screen (ALDEx2 score; 8,617 guides in common) are different frames; every number on *E. coli* SpCas9 belongs to one of them and the text says which, as it does for the other data used (eSpCas9, *C. rodentium*, human, CRISPRi, the CRISPOR tables). The working cohort is the curated one.
* **Provenance.** [`results/INDEX.md`](results/INDEX.md) lists every result file by the section it supports and [`results/RESULTS.md`](results/RESULTS.md) goes from each claim to its file and command. [`CODE_OVERVIEW.md`](CODE_OVERVIEW.md) tours the code the claims depend on and [`CODE_MAP.md`](CODE_MAP.md) maps every module. `./run_audit.sh` recomputes a large set of the post-freeze numbers quoted here from the file that produced each and exits non-zero on any mismatch (its last line is the number of checks it ran; a number without a check line is not covered), and checks that the package imports without writing anything. `EXTENSION_LOG.md` is the working record: what was tried, in what order, and with what expectation.

## What earns a place, and what the work is for

A statement is included only if it is insightful, verified, and not at serious risk of retraction, and only if it makes a claim in Parts 1–3 or defends one. A null result is a finding only if it shows something that had been missed; otherwise it is an appendix entry. A hypothesis is included only if the findings support it and it can be falsified.

The *E. coli* work is scaffolding for a human workflow. Its results divide into **method**, which travels (grouped cross-validation by gene and block, the texture-preserving null, replicate-derived ceilings, supplying an assay's dominant covariate before reading a subgroup, the negative results about tuning, selection and label weighting), and ***E. coli* biology**, which does not (the size and shape of the flanking effect, the GC signs, the dosage result, the mechanism constraints). Where the guide-side biology has been tested in human it did not carry over: target GC against activity is −0.201 in the published *E. coli* matrix (−0.147 on the curated cohort) and +0.018 in the human matrix, which is about zero, and the base preference at position 20 reverses sign (Finding 7).

---

## Part 1 — Findings

### 1. The published representation is a re-encoding of the guide, and holds about twenty variables

**6,169 of the 6,232 columns (99.0%) regenerate from any 20 nt protospacer, so the matrix holds
about twenty variables** (a 20-mer carries at most 40 bits). Regeneration agrees exactly,
**100.000000%** of the 3.08 million values rebuilt for the first 500 guides (`featurise.verify_against_published`) and of the 85.6 million over all 13,880 guides, and 13,879 of 13,880 guides were located in
NC_000913.2 with the required PAM in the right place (13,877 unique). The matrix partitions with
nothing left over (`portability_audit.csv`): **5,853** (position, k-mer) indicators (5,853 of 5,887
anonymous `V####` columns, none ambiguous), **316** quantum-chemical tables (exact k-mer lookups,
no conflicts) and **63** hand-named summaries. *Limit:* the 63 are outside the regenerated set; 46
are computable from the 20-mer and its PAM and 17 are not (below).

**Seventy-four mononucleotide columns recover 96.4% of full-matrix ρ on the published cohort (0.5081
against 0.5272, the full matrix at the 300-column budget) and 95.4% on the curated one; higher-order k-mers add almost nothing.** Curated cohort, random-row folds, no selection in any arm
(`curated_ladder.py`, 3 seeds × 5 folds; family A adds 348 flank columns):

| arm (`curated_ladder.csv`) | columns | ρ | vs full base | ρ with family A | vs full base |
|---|---:|---:|---:|---:|---:|
| mononucleotide only | 74 | 0.5198 ± 0.0021 | −0.0251 | — | — |
| + dinucleotide | 364 | 0.5331 ± 0.0010 | −0.0118 | 0.6937 ± 0.0010 | −0.0143 |
| + trimer | 1,510 | 0.5354 ± 0.0003 | −0.0096 | 0.7011 ± 0.0009 | −0.0068 |
| + tetramer (**all 5,853 k-mers**) | 5,853 | 0.5335 ± 0.0013 | −0.0114 | 0.7005 ± 0.0004 | −0.0075 |
| **+ 316 quantum columns (the base)** | **6,169** | **0.5449 ± 0.0013** | — | **0.7079 ± 0.0004** | — |

Trimers add **+0.0023** and tetramers then **−0.0018** (−0.0007 with family A): 4,343 tetramer
indicators are too sparse to carry anything, as in Finding 4 (+0.0035 and −0.0028 on the published
matrix). The one step that pays is the last: the 316 quantum columns, pure lookups that add no
information, are worth **+0.0114** over a complete k-mer encoding (+0.0075 with family A).

**Only 17 of the 63 hand-named columns are not functions of the guide and its PAM, and they carry none of the
gain.** The 46 computable from the 20-mer and its PAM are 20 whole-guide k-mer counts, guide GC, Tm and RNA
structure, 4 PAM indicators and the 19 of the 34 undecoded `V` columns that the 364 indicators reconstruct exactly; the 17 that are not are the
two annotation distances (nearest gene, PAM) and 15 undecoded `V` columns (in-sample R² from the 364 indicators 0.06–0.65). Published matrix, no
selection (`handnamed_split.py`, `handnamed_split.csv`): 364 sequence indicators give ρ 0.5189 ±
0.0010; adding the 46 computable gives **0.5254 ± 0.0013 (+0.0065)**, adding the 17 that are not
gives 0.5182 ± 0.0009 (**−0.0007**), and adding all 63 (427 columns) gives 0.5256 ± 0.0018 (+0.0066).
The 427-column result is therefore not evidence of non-sequence information, and annotation helps
only at the locus scale, where `d_supercoiling`, `e_nucleoid` and `g_transcription` are worth +0.145
between them on the curated cohort (random-row folds), as one variable (Appendix B3).

**The matrix is 74 sequence columns, 63 hand-named columns and 6,095 restatements, and the
restatement is worth +0.0065 to +0.019 to a depth-limited tree.** Families that are functions of the guide agree
(curated cohort, random-row folds; the hand-named 46 exist only in the published matrix): quantum
tables **+0.0114**, `b_energy` (CRISPRoff, Alkan et al., 2018, on the 23-mer, with the binding-energy window of Corsi et al., 2022) **+0.019**, `c_folding` (ViennaRNA on the
spacer and its scaffold) **+0.014**, `d_mechanics`'s 22 protospacer columns **+0.011**, the hand-named 46
**+0.0065**; `b_energy` and `c_folding` share the spacer's fold energy (Appendix B). The gain belongs to the particular functions, not to re-description as a class and not
to the tree's difficulty in computing them. On the curated base, **random** linear position-weight tables
(4, 16, 64, 256; mean of three draws each) give −0.0004, −0.0004, −0.0002 and −0.0015, and random
**non-linear** functions of order 2 (out-of-fold R² 0.34–0.48, against 0.316 for the real fold
energy) give −0.0004 to +0.0007, with orders 3 to 8 (R² 0.2 down to 0) between −0.0010 and +0.0016 (three draws each, one model seed). What pays is tracking the phenotype:
`eng.dg.guide_selffold`, fully determined by the 20-mer, has marginal \|ρ\| 0.102 and is worth
**+0.0116** alone, whereas a random function of the same difficulty has no association and buys
nothing (Appendix B). Selection is not the cause: the full `d_mechanics` arm gives **+0.0993** with
every column against **+0.0988** selecting 300 (curated cohort, one seed; `results/selector_control.csv`).

The size must be measured on columns that are functions of the guide, and `d_mechanics` is only
partly one: its eight `eng.mech.flank.{up,dn}{25,100}` columns are thermodynamics of flanking DNA
the base does not contain (out-of-fold R² ≈ 0), while nine ΔG/ΔH/ΔS sums reconstruct at
**R² = 1.0000** (Appendix A). Against the base (6,169 columns, ρ 0.5445; `no_information.py`,
`no_information.csv`, same 5 × 5 protocol) the 22 **protospacer** columns add **+0.0110**, the 8
**flank** columns **+0.0973** and all 30 +0.0991, so eight flank columns carry 98% of that arm's
gain. *Limit:* a deterministic function of the 20-mer adds no information and cannot raise the
ceiling; this is not a screen (Appendix B2), and the effect size lies elsewhere: family A's access to
DNA outside the 20-mer is worth **+0.164** (curated cohort, random-row 5 × 5).

**A model given 300 of the 6,169 regenerated columns is at parity with one given all of them, but on the curated
cohort a hand-picked reduced set is not a substitute for the pool.** Selecting 300 columns inside each
training fold against using every column gives **0.5445 against 0.5449** on the curated base and
**0.7084 against 0.7079** with family A (5 seeds with selection against 3 seeds without; ±0.0005 either way, and 0.5443 against 0.5439 in the one-seed control `selector_control.csv`); quadrupling the budget to 1,200 columns
on the published matrix moves family A's gain from +0.080 to +0.083 (main-project `results/ablation_nfeat1200.csv`), so no arm above needs a selection-budget caveat. Reduction of the representation itself:

| reduced set | cohort | ρ loss vs all columns | reading |
|---|---|---:|---|
| 427 = 364 indicators + 63 hand-named | published | **+0.0023** [−0.0008, +0.0054] (paired t over 5 seeds; 0.5249 against 0.5272) | non-inferior at any margin from 0.0054 up (one-sided p = 0.035 against 0.005, just outside the two-sided interval; margin not pre-declared, Appendix B) |
| 412 = 364 + 48 portable hand-named | published | 0.00003 [−0.0029, +0.0030] | non-inferior at any margin from 0.0030 up (same test); computable for any 20-mer plus the reference genome and its gene annotation |
| 364 indicators only | published | 0.0083 [+0.0057, +0.0109] | loses (all 5 seeds), so the 48 portable columns earn their place |
| 364 and 1,510 columns (ladder above) | curated | 0.0118 and 0.0096; with family A 0.0143 and 0.0068 | falls short of the 0.005 margin that 427 met only on a one-sided test |

So "412 ≈ all 6,232" holds for the published matrix under a 300-column budget and does not transfer
to the curated cohort, whose arms (6,517 columns = 6,169 regenerated + 348 flank) contain none of
the 63 hand-named columns in any case. The same redundancy accompanies sixteen model classes
spanning only 0.116 ρ (published matrix, 3 seeds × 5 folds; eight span 0.1076 on the curated cohort with family A, one seed) and no correlate of the champion's residuals above 0.031 (published matrix with family A given; 0.028 without).

**The gain is better located by how much of a column the guide determines than by which family
built it** (Appendix B). Of the 636 family columns that have an R² against the guide, the 18 the
guide determines are worth **+0.008** as a block, the 16 partly recoverable **+0.020** and the 602 it
cannot predict **+0.201**. *Limit:* the blocks differ in size as much as in recoverability, so Δρ is
a difference and not a ratio, and finer banding puts the step at about R² = 0.2, not 0.99.

### 2. The DNA around the cut site predicts cutting — the one large effect, and it is local and compositional

**Windowed composition of the flanking DNA (family A, 348 columns: GC, purine, longest run and
quantum-table averages at 50/250/500/1,000 nt each side, plus a near-cut block of 10 bases 5' of the
protospacer, the PAM's N and 15 bases 3' of the PAM) is the one large effect: the ten other families
add only +0.039 beyond it (Appendix B3), and its 250–1,000 nt windows cover DNA that Cas9 does not
contact.** It is worth **+0.164 ρ** on the curated cohort (random-row folds), **+0.156** on sealed genes
(15 sealings of whole genes holding 20% of the guides, with column selection and settings re-derived on the
other 80%; family A's design and windows were fixed beforehand, and the pair 0.1647 in pool CV against 0.1560
sealed is a third protocol: random-row folds in the pool, XGBoost with 300 of 6,517 columns minus the
6,169-column base; Part 4) and +0.080 on the published matrix (random-row folds), and it survives holding
out whole genes (+0.157 on the 30,203 genic guides), losing nothing beyond the 6% that grouping costs
a feature which cannot memorise a locus (Appendix B3). It grows with the model: under one protocol
(gene-grouped folds, 3 seeds, the same genic guides; the base-only selector never offered a flank
column) the increment is
**+0.1563** [+0.1515, +0.1616] at the champion's settings and **+0.1697** [+0.1659, +0.1738] at the
tuned ones, 15 of 15 folds either way, and tuning is worth **+0.0162** with these columns against
**+0.0028** without, so almost all of what the hyperparameter search bought is the ability to use
the flanking DNA (the base-only arm was not tuned in its own right; Finding 4, `EXTENSION_LOG.md` §16).

**The effect is spatially local.** Giving each guide the flank track read from a position offset by
a circular chromosome shift keeps every column's distribution, smoothness and long-range
autocorrelation and breaks only the correspondence (Appendix A; published matrix, 100 kb-block folds,
3 seeds × 5 folds):

| flanks read from (`shift_null.csv`) | ρ | gain over no flanks |
|---|---:|---:|
| **the right place** | **0.6060** | **+0.0769** |
| 1 kb away | 0.5353 | +0.0061 (8% of the effect) |
| 10 kb away | 0.5160 | −0.0132 |
| 100 kb – 2 Mb away | 0.5194–0.5229 | −0.006 to −0.010 |

At 1 kb the shifted columns are still correlated with the real ones in proportion to their window (median
0.03 for the near-cut columns, 0.11 at 50 nt, 0.28 at 250 nt, 0.40 at 500 nt, 0.58 at 1,000 nt; the
downstream 500 nt GC column 0.41); by 10 kb no window's median exceeds 0.07 and the gain is gone, so
the effect is not chromosome-scale structure.

**Target GC does not explain the effect; it suppresses it** (`gcstrata.py --corr`, 33,567 curated
guides). Target GC correlates **−0.147** with the label and **+0.245** with flank GC (the mean GC of the 500 nt
windows either side). Partialling it
out raises flank GC's association from **+0.176 marginal to +0.221 partial**, positive in all five
target-GC strata (**+0.425, +0.278, +0.180, +0.107, +0.068**, strongest where the target is
GC-poor): the signature of a suppressor, not a confound. The full ablation inside each stratum of the
published matrix (13,879 guides, 100 kb-block folds; strata of equal count, so their edges differ from
the curated ones above), where target composition is near-constant:

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

**Hypothesis (untested).** Target GC and flank GC act through different steps: a GC-rich target slows a
strand-invasion-limited step, while GC-rich flanking DNA helps by a route that is not melting-limited
unwinding (inference I1, Finding 7: more flank GC means more cutting, the opposite of what melting
predicts). Consistent with it: target GC correlates **−0.201** with the score on the published matrix
(−0.147 on the curated cohort) while flank GC correlates +0.176 with it on the curated cohort, the effect
survives a change of enzyme and of organism, and the one sequence a beam search returns as the
published-matrix model's optimum has 35% GC against a 51% mean (a 20-mer absent from the genome;
`model_inversion.csv` in the main project's `results/`). Its one test is the stratified ablation above,
where a GC confound would have vanished and did not; a locus-matched design in which AT-rich flanks cut
better would refute it (I1). *Limit:* the assay measures depletion, not cutting; separating a
strand-invasion-limited from a hybridisation-limited step would need kinetic data.

**A convolutional network given the raw DNA recovers only a fraction of the effect.** The network
has the architecture family of the best published competitor, on the same rows and folds (published
13,880-row matrix, the tabular models' random-row folds, seed 41; the pooled response to flank GC is
mostly a step at the AT-rich end; Finding 7):

| CNN given (`seqnet.csv`, `seqnet_matched.csv`) | protospacer + PAM | ±100 nt | ±250 nt | ±1,000 nt |
|---|---:|---:|---:|---:|
| ρ | 0.4980 | 0.5203 | **0.5247** | 0.5190 |

The curve is flat from ±100 (the four widths differ by 0.006, within the fold sd of 0.007–0.012):
given the exact DNA the windows summarise (±1,000 nt), the network adds **+0.021** over the guide alone
(+0.027 at best, ±250 nt) where the windowed means add **+0.080** on the same matrix. A windowed
average is representable by its mean-pooling context branch, so the barrier is not a representational
limit; inductive bias and sample efficiency are the candidate causes (the prior that the right summary
of 500 positions is their unweighted mean may be expensive to learn from 13,880 noisy examples), and
neither is tested.

**On the DNA a tuned CNN actually holds, it beats windowed composition; SLICER's lead is long-range
reach.** crisprHAL 2's released rows are 378 nt for a 23 nt target (**189 nt upstream, 166
downstream**). Restricting family A to the columns that fit inside that span (the 50 nt windows, the
widest that fit, and the near-cut columns; the 250/500/1,000 nt windows reach past their input) gives a
like-for-like contest (curated cohort, five random-row folds, seed 41; family A in LightGBM, crisprHAL 2 at
its published configuration on the same folds, ρ 0.6971):

| arm | flank columns | ρ | vs crisprHAL 2 |
|---|---:|---:|---:|
| near-cut only (10 nt 5' of the protospacer to 15 nt 3' of the PAM; folds 1–3 only) | 260 | 0.6178 | −0.076 (same three folds) |
| **matched to their span (±189 / ±166): near-cut + 50 nt windows** | **282** | **0.6724 ± 0.0094** | **−0.025** |
| full family A | 348 | **0.7078** | **+0.011** |

Inside their own span their convolutions beat windowed composition by **0.025** (in five of five folds;
one seed), so windowed means are not a smarter encoding of the same bases; the lead comes from the
250–1,000 nt windows their input never reaches, worth 0.7078 − 0.6724 = **+0.035**. Their tuned CNN
reaches 0.697 on the curated cohort against 0.525 for the in-house network on the published matrix; the
cohorts differ, so that gap mixes architecture and data.

**Given the same reach, crisprHAL 2 at its published hyperparameters gets worse, not better.**
Retrained at four input widths on the same rows, label and folds, on all five folds (batch 256, as cuDNN cannot allocate
the GRU workspace at 2,023 nt and batch 1,024; the 378 nt arm at both batches bounds that cost at
**−0.011** on fold 1, a fifth of the effect below, and its five-fold mean is 0.6810 at batch 256 against 0.6971 at the published batch).
Of the twenty arms, the 378 nt arms of folds 2 and 3 and the 23 nt arms of folds 2 to 5 were trained on a CPU and the other fourteen on a T4.
Fold 1's four T4 arms are in the first run's file; the other ten were transcribed from the Colab console, because its csv files were not downloaded
(`crisprhal_wide_window_fcolab.csv` and `crisprhal_wide_window_fcolab2.csv`: seven to four decimals, fold 5's three to six);
a harness check on the T4 gave 0.6809 for fold 1's 378 nt arm against 0.6799 in the first run. `colab/crisprhal_wide_window.py` reproduces the whole experiment in one paste.

| DNA both models see (`crisprhal_wide_window_summary.csv`) | crisprHAL 2 ρ | vs its 378 nt (paired, five folds) | SLICER ρ (same folds) | lead |
|---|---:|---:|---:|---|
| guide + PAM (23 nt) | 0.5556 | −0.125 | 0.5440 | crisprHAL **+0.012** (ahead in 5 of 5) |
| **378 nt (as published)** | **0.6810** | — | 0.6724 | crisprHAL **+0.009** (ahead in 4 of 5; −0.013 on fold 3) |
| 1,023 nt | 0.6615 | **−0.020** (below in 5 of 5; −0.005 to −0.040) | — | — |
| ±1,000 nt (2,023 nt) | 0.6421 | **−0.039** (below in 5 of 5; −0.003 to −0.058) | **0.7078** | **SLICER +0.066** (ahead in 5 of 5; +0.055 to +0.079) |

(SLICER's arms are the 20-mer representation alone, family A restricted to 189 / 166 nt, and the full family A, on the same folds.)
crisprHAL 2 peaks at the width it was designed for and falls below its own 378 nt score at both wider inputs in all five folds,
by 0.020 on average at 1,023 nt and 0.039 at 2,023 nt, while SLICER improves with width (0.5440, 0.6724, 0.7078). The size of the fall varies
(sd across folds 0.013 and 0.024; fold 3 is almost flat at −0.003, and its 378 nt arm, a CPU run, is the lowest of the five at 0.6602),
so the direction is established on five folds and one seed and the magnitude is not tightly measured. At its own span crisprHAL 2's lead over SLICER's windowed composition is
+0.009 at batch 256 (sd 0.014 across folds, reversed on fold 3) against +0.025 at its published batch (five folds, above): small, and not stable in sign. Browne et al. (2026) settled
on 378 nt for *E. coli* SpCas9 (406 nt for eSpCas9) as the optimum of their per-dataset length search, reporting a small
mean gain per added nucleotide (+0.00034 ρ per nucleotide upstream, +0.00074 downstream, averaged over the whole added span; 0.554 on the sgRNA alone
to 0.697 at 378 nt, five-fold CV), so the decline beyond it follows a curve whose per-nucleotide gain was already small. *Limit:* one seed per fold, so the claim is only that the published configuration is degraded by input wider than
it was tuned for. Equalising the DNA removes the information asymmetry (they receive raw sequence,
family A a 348-column summary) but not the opportunity to use it, since epochs, dropout, filters and
GRU width were chosen at 378 nt; a re-tune at each width is not yet run
(`colab/crisprhal_tune_by_width.py` pre-registers what each outcome would support; Appendix A).

### 3. The effect belongs to the DNA, and transfers

**Changing the enzyme or the organism costs about a tenth of the performance; changing the kingdom costs all of it.** In the bacterial transfer with the lowest transferred ρ, a model imported from a 236 kb *C. rodentium* screen (Ham et al., 2023) into the *E. coli* genome, the flank features carry **+0.149 of the 0.628** that arrives. That is the sense in which the effect belongs to the DNA: it is the part of the model that survives a change of protein and of host.

Each row trains SLICER on one screen and predicts another with no retraining or recalibration. *Transferred* is the imported model's ρ on the target; *local* is the same architecture trained on the target (random-row 5-fold CV, seed 41), the ceiling for a transfer; *retained* is their ratio; the last column is the part of the transferred ρ owed to flank features (base + family A minus base alone, both transferred; `results/transfer_model.csv`, `results/transfer.csv`, 3 seeds).

| variable changed | from → to | transferred ρ | local ρ | retained | of which family A |
|---|---|---:|---:|---:|---:|
| **nuclease** | SpCas9 → eSpCas9, same genome and library (see Limits) | 0.689 | 0.784 | **88%** | +0.098 |
| **nuclease + organism** | *E. coli* SpCas9 → *C. rodentium* TevSpCas9 | 0.698 | 0.766 | **91%** | +0.072 |
| **the same, reversed** | *C. rodentium* TevSpCas9 → *E. coli* SpCas9 | 0.628 | 0.708 | **89%** | **+0.149** |
| **kingdom**, shared columns only | *E. coli* → human (published matrices; 6,216 shared columns, no family A on the human side; human-fitted settings) | −0.039 | 0.508 | **0%** | — |

A surviving effect under a new organism and nuclease (rows 2–3) is not about SpCas9's idiosyncrasies nor about *E. coli*'s composition; the eSpCas9 row supports the nuclease half only through the guides absent from the training screen (Limits). The human row failing is the control on these: a model that transferred everywhere would indicate a methodological artefact rather than biology.

*Limits.* The first row shares its library with the training screen: all 33,567 SpCas9 guides are among the 59,489 eSpCas9 guides, so 56% of the eSpCas9 test rows are training guides and the row shows label transfer across nuclease on one library, not sequence generalisation; on the 25,922 eSpCas9 guides absent from the SpCas9 screen the transferred ρ is 0.662 and the family A share +0.103 (seed 41; 0.689 and +0.098 on all rows), so the share does not rest on the shared guides; the two *C. rodentium* rows are sequence-disjoint from the *E. coli* screens (one shared 20-mer). The first three rows are LightGBM on base + family A; the kingdom row is the XGBoost champion on the shared columns only, which carry no family A on the human side; built on both sides, family A transfers with a positive sign (Part 3). The kingdom row's −0.039 [−0.054, −0.024] (bootstrap over test rows) is the human-fitted-settings figure; at the champion's settings it is −0.046 [−0.061, −0.031]. The transferred ρ is the prediction of the whole human matrix by a model trained on *E. coli* and does not depend on how human rows are folded, but the local ρ of 0.508 is on row-wise human folds in which 49% of rows have a twin in training (twin-grouped: 0.2515, `human_twin_check.csv`); the retained share is negative under either denominator, and the row has not been re-run on the deduplicated matrix (Part 3). A fifth change, from a cleavage assay to a dCas9 silencing screen, alters what is measured and not only what is varied (zero-shot ρ 0.061 on Cui et al.'s screen against 0.458 for a model trained on it, gene-grouped folds); it is a boundary experiment and is not in the table (Appendix B5).

#### Two confound controls

These are not transfer tests; each removes one way the effect could be an artefact of this dataset.

| could the effect be an artefact of... | result |
|---|---|
| read depth or mappability? | **excluded**: +0.157 among the 26,146 guides whose every 25-mer within 1 kb is unique, +0.261 among 852 guides with over 10% repeated 25-mers, +0.160 given the control read depth as a column (`mappability_stratum.csv`, `mappability_depth.csv`) |
| gene ranking in disguise? | no: inside genes (gene-centred) ρ rises from 0.618 to 0.703, so flanks are worth **+0.085 [+0.078, +0.090] inside genes** (`cut_within_gene_summary.csv`; curated genic guides, gene-grouped folds, 10 seeds, bootstrap over genes) |

*Limits.* The mappability control restricts to a high-uniqueness stratum to test whether family A predicts read misassignment (a guide in a repeat, IS element or rRNA operon has counts that can be wrong in a way that tracks local DNA). No result in this folder gives the gain inside such a stratum: `ablation_abund_split.csv` is the `d_supercoiling` abundance split, and `abund_local_vs_mappability.csv` is a track-versus-uniqueness correlation matrix whose `uniq` definition cannot be reproduced. Jacky's rebuild with genome-derived 25-mer uniqueness (`Jacky/agent-notes/results/flank_high_uniqueness_stratum.csv`: top quartile of mean window uniqueness, 4,949 of the 13,880 published guides, gene-grouped folds, one seed) gives 0.490 → 0.540 against 0.529 → 0.607 on all guides, a gain of +0.051 against +0.078; it was not audited, and the strata of `mappability_stratum.csv` (above) supersede it: they exclude the objection. The within-gene row is traceable. Neither control is an intervention, and only the within-gene one rules anything out.

#### Scope of the predictor

**The predictor estimates cleavage efficiency as read out by survival, and its applicability follows from that.** In these pooled screens a guide's score falls because the cell died, and the cell died because Cas9 cut the chromosome; death is the readout and cutting is what the score is taken to track, though the score is the product of P(cut) and P(the break is unsurvivable) and no experiment here separates the two (Part 3). This is a statement about the training data, not about the enzyme: the model was fitted to survival after cleavage and says nothing about binding-only assays.

| application | does this predictor apply? |
|---|---|
| gene knockout, HDR knock-in, any edit needing a double-strand break | **yes**: cutting is what the score tracks |
| pooled knockout screens using Cas9 cutting as the perturbation | **yes**: the screens it was fitted on |
| CRISPRi / CRISPRa with dCas9 | **no, not as it stands**: guide strand dominates that assay and a cutting model has no use for it (Appendix B5); a guide-sequence model trained on silencing data is the right tool |
| base and prime editing (nickase-based) | **untested**: these bind and nick rather than cut, and nothing here speaks to them |

For an application that is silencing, crisprHAL 2's raw-sequence model transfers to a CRISPRi screen about 1.4 times as well as SLICER (|ρ| 0.087 against 0.061 on Cui et al.'s screen; Part 5), plausibly because what survives that change of assay is guide-sequence effects, not flanking DNA.

### 4. The model is out of *material*, and the tuning gain is evidence for it

**The learner is not the limit. The same tuning that is worth +0.016 ρ with family A present is worth +0.003 without it**, so the tuned settings buy the ability to use the flanking columns, not generic capacity, and without those columns there is almost nothing for extra capacity to do. A model one hyperparameter search from +0.016 might look short of any limit; asking where that +0.016 comes from shows it is not.

The four configurations were run on a matrix with family A and on one without it, under the standard protocol (the 30,203 genic guides of the curated cohort, gene-grouped 5 folds × 3 seeds, selection fitted inside each training fold, the base-only arm's selector never offered a flank column; `family_a_at_tuned.py`, `results/family_a_at_tuned*.csv`, `EXTENSION_LOG.md` §16):

| (`family_a_at_tuned_summary.csv`; 15 folds, paired, bootstrap over folds) | base only | base + family A | family A's increment |
|---|---:|---:|---:|
| champion (depth 4, lr 0.05, 400 trees, 300 columns) | 0.5588 | 0.7151 | **+0.1563** [+0.1515, +0.1616] |
| depth 6 | 0.5569 | 0.7223 | +0.1655 |
| **tuned d099x** (depth 5, lr 0.013, 4,000 trees, 450 columns) | 0.5617 | **0.7314** | **+0.1697** [+0.1659, +0.1738] |
| tuned d077x | 0.5631 | 0.7310 | +0.1680 |
| **the tuning gain itself (d099x − champion)** | **+0.0028** [+0.0009, +0.0050], 9/15 folds | **+0.0162** [+0.0152, +0.0173], 15/15 folds | |

On the base representation the tuned settings (ten times the trees at a quarter of the learning rate, depth 5, 450 selected columns and other changes) buy +0.003, positive in only 9 of 15 folds; on top of family A the same change buys +0.016 in all 15. Consequently **family A's increment grows with the model, from +0.156 to +0.170**, with the best-of-four bound (tuned arm over the best base-only arm, guarding against the settings having been chosen with family A present) at +0.168 [+0.164, +0.173]. The champion's base + A figure (0.7151) and d099x's (0.7314) are identical to `tune_fullcv.csv`, so this run reproduces the established protocol.

*Limits.* The tuned settings were selected on DEV genes with family A present, so applying them to a base-only matrix is not a matched tuning effort and the base-only column could be slightly pessimistic; a matched base-only search has not been run. For family A's increment at the tuned settings to fall back to the champion's +0.1563, a base-only search would have to reach ρ **0.5751**: +0.0120 beyond the best of the four tried (d077x, 0.5631) and about 2.9× the largest base-only gain any of them produced (+0.0042).

**Residual correlation agrees, and reducing the matrix does not change it.** Correlating every column against the champion's out-of-fold residuals (`representation_headroom.csv`; published 13,880-row matrix, champion XGBoost with 300 selected columns, five random-row folds, seed 41):

| model | strongest \|ρ\| among columns it has | among columns held back |
|---|---:|---:|
| all 6,232 | 0.028 | **0.147**: family A |
| the 427 | 0.022 | **0.146**: family A |
| 427 + family A | 0.028 | 0.070, none above 0.10 |
| all 6,232 + family A | 0.031 | nothing held back |

Dropping 5,805 columns did not change what the model is missing. The test has a positive control: with family A held back from the model it flagged family A at 0.147, and giving the model family A was then worth +0.080 (published matrix, random-row folds).

**The usable k-mer order stops at three.** Without feature selection, trimers help and tetramers hurt, and the two cancel (published matrix, paired over 3 seeds × 5 folds, `kmer_order_paired.csv`):

| contrast | mean Δρ | folds | paired *t* |
|---|---:|---:|---:|
| **+ trimers** | **+0.0035** | 12/15 | **p = 0.025** |
| **+ tetramers on top** | **−0.0028** | 4/15 | **p = 0.013** |
| net | +0.0007 | 8/15 | p = 0.62 |

4,343 tetramer columns for 256 possible 4-mers per position are too sparse to carry anything. Through the top-300 selector adding trimers hurt (0.5189 → 0.5163, `representation_depth.csv`, champion depth 4), so a conclusion drawn downstream of feature selection is a statement about the selector until shown otherwise.

### 5. On a redundant matrix, per-column importance is unstable across fits, but SHAP at block level and edit tests are faithful; SLICER is interpretable, more faithful than crisprHAL 2, and ties or splits with it on stability, truth and compressibility depending on the reading used

**Feature importance disagrees between models on this matrix, and the standard "drop the top features and refit" probe returns ≈0 for each of the four models tried.** Both are properties of the published matrix with family A (13,880 guides, 6,580 columns), on which 6,095 columns restate about twenty variables, not of boosting or of the attributions. The defensible report is SHAP, at block level.

**Importance.** Seven model families (two boosters, two forests, two linear, CatBoost) on the published matrix with family A (13,880 guides, 6,580 columns), random-row three-fold CV, seed 41 (`importance_model_pairs.csv`). Agreement is the Spearman correlation between two models' importance vectors, per column over the union of the two models' 300 selected columns and per block over 14 blocks, averaged over pairs and folds. The top three rows are the five tree families (10 pairs), the only ones for which SHAP is defined; the linear models are added beneath:

| method | agreement between models, per column | per block |
|---|---:|---:|
| the library default (gain / impurity / coef) | **0.050** | 0.531 |
| permutation | 0.258 | 0.581 |
| **SHAP** | **0.496** | **0.800** |
| *with the two linear models too (all 21 pairs): default* | *−0.057* | *0.522* |
| *with the two linear models too: permutation* | *0.187* | *0.512* |

The instability survives bagged forests and penalised linear models, which make the default and permutation agreement worse, not better. What did not survive is the apparent finding that models prefer different kinds of column: it was an artefact of split gain. XGBoost's share on binary indicators (the guide's one-hot columns and the flank's bases within 10 nt) falls from 57% under gain to 31% under SHAP and lands on LightGBM's 31% (`importance_model_blocks.csv`, three folds); in a separate two-model run on the 300 columns each model selects (one fold of five), the largest XGBoost–LightGBM gap on any of six kinds of column goes from 31 points under gain (49% against 18% on guide indicators) to 0.3 under SHAP (`attribution_summary.csv`, `attribution_summary_shap.csv`).

*Limits.* Report SHAP at block level and never gain. The pipeline's own column selector uses gain, which costs no accuracy but does bias what the model is shown. Block-level reporting is the only fix that costs no accuracy: selecting for decorrelation lowers ρ and leaves a less readable column set, because on this matrix the redundant columns are the informative ones (Appendix B), and the non-redundant representation below costs 12% of ρ.

**Faithfulness by drop-and-refit is uninformative on the full matrix.** Dropping the top *k* columns removes almost no information because near-duplicates remain to carry it: refitting XGBoost and LightGBM without their 20 top SHAP columns, or without their largest one to five correlation clusters (3 to 12 columns), costs ≈0 on the full 6,580-column matrix (three random folds; `faithfulness_top20_full.csv`, `faithfulness_groups.csv`). On the 775-column reduced set (364 mono- and dinucleotide indicators, the 63 hand-named columns and family A's 348) the same probe does not read ≈0. Equalising the *share of attributed importance* removed, rather than the number of columns, makes the comparison about what each representation loses: at a matched share (the top 20 columns hold 34.6% of SHAP on the full matrix and 37.5% on the reduced set) the full matrix loses 0.002 (changes of +0.002 to −0.005 over the six model-fold fits) and the reduced set loses **0.015** (−0.011 to −0.020 over its six; +0.001 for 20 random columns), several times the damage for the same share of the explanation removed, while both concentrate importance almost identically (44 and 37 columns for half of it; `faithfulness_groups_reduced.csv`, `faithfulness_shap_share.csv`).

#### Three tests that need no refit

Refitting is where redundancy does its damage, so three tests use one fixed model: the production recipe (base + family A, gain-ranked top 300, champion XGBoost) fitted on four of the five whole-gene folds of the paper's design (26,853 guides; seed 41), ρ 0.722 on the fifth, whose genes it has not seen (6,714 guides; `interp_extra.py`; one fit, so the figures are not averaged over seeds). The first version of these tests used a random 80% of the guides (ρ 0.712) and gave the same readings; it was rerun on whole genes on 26 November, and the numbers below are the whole-gene ones.

**1. SHAP is faithful to the fitted model, at column level as well as block level (sensitivity-*n*).** For random subsets of *n* columns, the summed SHAP over the subset correlates, across subsets and within each of 100 held-out guides (60 random column subsets for each *n*, 40 block subsets), with the prediction lost when those columns are replaced by values from 50 training guides (`results/interp_sensitivity_n.csv`):

| subset | n | SHAP | ridge surrogate | SHAP with columns permuted |
|---|---:|---:|---:|---:|
| columns | 1 | **+0.922** | +0.190 | +0.022 |
| columns | 100 | **+0.961** | +0.213 | −0.006 |
| blocks | 1 | **+0.960** | +0.273 | +0.039 |
| blocks | 6 | **+0.977** | +0.316 | +0.043 |

The permuted column is the control: the same numbers on the wrong columns correlate at zero, so 0.92–0.98 is the attribution's content. The ridge surrogate fails because it attributes a different model. This is consistent with the claim above: SHAP accounts faithfully for this fitted model, even for one column, but a column's credit does not mean the same thing in a second fit, where a correlated neighbour carries it. Faithful to the model and unstable across models is the pattern redundancy predicts. *Limits:* the correlation tests the ordering of subsets, not the attribution's scale; the SHAP is the path-dependent variant (the interventional one is refused by this XGBoost build), so agreement with an interventional ablation also shows the two conventions coincide here.

**2. About 14% of the output's variance is interaction, mostly among correlated descriptors.** SHAP interaction values on 400 held-out guides, all 300 columns (`interp_interactions.csv`, `interp_interaction_blocks.csv`): single-column terms carry **69.8%** of the variance and pairwise terms **13.9%**, the remainder being their covariance. Off-diagonal terms hold 52.6% of the absolute interaction mass, 42.8% of it between different blocks and 9.8% within a block; of the between-block mass, 44% lies between two protospacer blocks, 14% between two flank blocks and **41% between a flank block and a protospacer block**. The strongest pairs are mostly descriptors of the same base pairs (`p18dimer.Hbond.energyraw` with `p19dimer.Hbond.stackingraw`, 0.036, `p20basepair.Hlgap.eVEraw` with `p19dimer.Hbond.stackingraw`, 0.033, and other terminal-position pairs) and the GC fraction of the 250 nt windows on the two sides (0.035), which is the model splitting credit between near copies. This is consistent with a ridge on a decorrelated basis reproducing 95.6% of boosting's ρ (Appendix B; on the published matrix's 427-column reduced set, ridge 0.503 against LightGBM 0.526, `additivity_reduced.csv`): the interaction part of the output is small relative to the part that orders guides correctly. *Limits:* with credit split among correlated columns, a flank × guide term is at least as likely to be redundancy as mechanism, so no claim about biological interaction between flank and guide follows.

**3. Edited flanks move the prediction through composition and little else.** Editing the flank DNA (1 kb each side) beyond the 20 nt next to the cut on 1,500 held-out guides and rebuilding family A from the edited DNA (the unedited rebuild reproduces the stored matrix exactly; `interp_counterfactual.csv`): GC +5 points moves the prediction by +0.41 [+0.37, +0.46] and −5 points by −0.81 [−0.86, −0.77], against a label sd of 3.29, while swapping the same number of bases (A↔T, G↔C) or four times as many without moving GC moves it by 0.004 and 0.021 (the control). The response is asymmetric, about twice as far for a fall as for a rise, which the AT-rich step of Finding 7 predicts. Shuffling the order of the bases beyond the 20 nt moves it by +0.19: the 1 kb composition is unchanged, but the GC of the 50 to 500 nt windows changes by 2 to 6 points on average, and the windowed quantum columns (+0.14 when edited alone) and windowed GC columns (+0.06) carry it; the model holds no run-length column. *Limits:* these are properties of the fitted model, not measurements of the genome. 1 kb flank GC has an sd of 4.5 points (upstream) and 4.6 (downstream) across guides, so ±10 is a 2.2 sd extrapolation where trees are cliff-edged, and no experiment has varied flank composition at a fixed guide (Part 3).

#### A non-redundant representation is readable column by column

Finding 5's explanation of the instability is redundancy, which predicts that it disappears when no column restates another. This was tested on a **compact representation**: 80 position × base indicators for the guide and 24 flank composition columns (GC, purine fraction and longest homopolymer run over 50, 250, 500 and 1,000 nt on each side), 104 columns, every one named (`compact_model.py`). Gene-grouped five-fold CV on the 30,203 genic guides, three seeds, champion settings, against the production recipe on the same folds:

| representation | columns | ρ | share of the full recipe | SHAP agreement between the five tree families, per column (per block) |
|---|---:|---:|---:|---|
| full recipe (base + family A, gain-ranked top 300) | 300 of 6,517 | **0.7161** | 100% | 0.496 (0.800); default importance 0.050 (published matrix, 6,580 columns, random folds: the seven-family run above) |
| compact + the 304 adjacent-pair indicators | 408 | 0.6541 | 91% | not measured |
| **compact** | **104** | **0.6335** | **88%** | **0.872 (0.923)**; default importance 0.698 (0.734) |
| guide + the four whole-kilobase numbers | 84 | 0.6115 | 85% | not measured |
| guide only | 80 | 0.5402 | 75% | not measured |

Predicted in advance: the compact model keeps at least 90% of the full ρ (**failed**: 88%, 91% with adjacent pairs); SHAP agrees at 0.8 or more per column (**held**: 0.872 ± 0.006 over three splits, against 0.496 on the full matrix); default importance improves but stays below SHAP (**held**: 0.698 against 0.872). So the instability goes with the redundancy and not with the model family: with no restated columns the per-column ranking survives a change of model family, at a cost of about an eighth of ρ. The contrast is not paired: the 0.496 is the published matrix under random folds, the 0.872 the curated genic guides under gene-grouped folds, and the block counts differ (14 and 28).

**The compact model's SHAP reads out what the model-free analysis found.** The SHAP-implied effect of each base at each guide position (summed over the four columns at the position, in SD units) correlates with the model-free position × base map of Finding 7 at **ρ 0.851 (Pearson 0.963) over 80 cells** (predicted at 0.8 or more: **held**). The seven largest columns by mean |SHAP| are G, T, C and A at position 20, G at position 18 and C at positions 15 and 14, with the 250 nt flank GC columns (upstream 0.203, downstream 0.182) the largest flank terms. The flank's total contribution, by decile of flank GC, is **−0.78 SD in the lowest decile, −0.04 in the second, and +0.07 rising to +0.17 above**: the step at the AT-rich end that the response curve of Finding 7 shows without any model (predicted in advance: **held**). *Limits:* one fold's model for the read-out; the compact model is a readable counterpart and not the shipped model; the 0.496 and 0.800 on the full matrix come from the seven-family protocol above (published matrix, random folds), not from these three splits.

**In what sense SLICER is interpretable.** The frozen production recipe's attributions are faithful to it (SHAP, sensitivity-*n* +0.928 for one column and +0.956 for one block; 0.961–0.971 at block level for the tuned model as shipped), agree between model families at block level (0.800), and its response to an edited flank is local composition and little else (test 3); the signal itself is described without the model in Finding 7. A per-column ranking that survives a refit needs a non-redundant representation, and the compact model above is one, at a cost of 12% of ρ.

#### Is SLICER more interpretable than crisprHAL 2? Only in faithfulness: on stability, truth and compressibility the two tie or split depending on the reading used

"More interpretable" was split into four questions that either model can be put through unchanged, with the predictions and the decision rule written first (`vs_crisprhal_interpret.py`). Both models are the released ones on crisprHAL 2's own WT-SpCas9 split (their 26,994 training and 6,749 test rows; SLICER sees the 26,830 and 6,737 that locate in the genome): crisprHAL 2's shipped weights, SLICER's tuned model fitted on the same training rows, and SLICER-104 where the question is about positions. The readings are TreeSHAP for the trees, expected gradients for the network (the estimator behind SHAP's GradientExplainer), and one reading that needs no attribution method, the **substitution map**: the mean change in a model's prediction when one guide base is set to each base. The shipped SLICER's columns are descriptors, so its edited 20-mer is re-featurised, which reads it at the same unit as the network. The attribution, substitution-map and stability readings use the same 1,000 test guides; sensitivity-*n* uses the first 100 of them (40 random block subsets per block count, 50 background guides); the flank edits use the 1,500 guides they were drawn from. Refitting is three seeds and two pairs of random halves of the training rows for each model, the network by crisprHAL's own code and 48 epochs (`interp_compare_*.csv`; `hal_readout.py` runs in the TensorFlow environment).

| question | measure | SLICER (shipped) | SLICER-104 | crisprHAL 2 | which is ahead |
|---|---|---:|---:|---:|---|
| faithful | sensitivity-*n*, 1–6 blocks removed (100 guides; 12, 28 and 25 blocks) | 0.961–0.971 | 0.953–0.962 | 0.860–0.881 | **SLICER**, by +0.081 to +0.111 (every 95% interval above 0: ±1.96 se over the 100 guides, unpaired) |
| stable under refit | substitution map, Spearman between refits: seeds (3 pairs) | 0.998 | 0.999 | 0.989 | SLICER, by 0.01 (ranges do not overlap) |
| | the same, between random halves (2 pairs) | 0.972 | 0.953 | 0.981 | crisprHAL 2, by 0.01–0.03 |
| | per input unit, seeds / halves | 0.578 / 0.278 (columns) | 0.989 / 0.843 (80 channels) | 0.986 / 0.957 (80 channels) | crisprHAL 2 over halves and against the shipped SLICER; a tie with SLICER-104 over seeds |
| true | substitution map against the model-free map, ρ | 0.810 | 0.853 | 0.783 | no difference for the shipped model (+0.027 [−0.010, +0.071]); SLICER-104 ahead, +0.070 [+0.033, +0.122] (bootstrap over the 80 cells) |
| compressible | a 104-column boosted surrogate's ρ with the model's out-of-fold predictions (random-row five-fold predictions, gene-grouped surrogate), 30,203 guides | 0.871 | – | 0.831 | fidelity: SLICER by +0.041 [+0.038, +0.044] (bootstrap over guides); share of accuracy kept: 0.876 for SLICER and 0.890 for crisprHAL 2, crisprHAL 2 by 0.015 |

SLICER is ahead of crisprHAL 2 on every reading of one question, faithfulness; on the other three the two tie or split depending on the reading used. On stability, SLICER's substitution map is steadier across seeds (0.998 against 0.989, ranges that do not overlap) and the network's is steadier across data halves (0.981 against 0.972 for the shipped SLICER and 0.953 for SLICER-104); on the units each model reads natively the network's is steadier (0.986 over seeds and 0.957 over halves, against 0.578 and 0.278 per column for the shipped SLICER; SLICER-104 ties over seeds at 0.989). On truth, the two match the model-free map as well on each model's own attribution read-out (0.862 against SLICER-104's SHAP-implied 0.857), but on the substitution map that all three models share SLICER-104 is ahead (0.853 against 0.783, +0.070 [+0.033, +0.122]) and the shipped SLICER is not distinguishable (0.810, +0.027 [−0.010, +0.071]). On compressibility a readable surrogate keeps slightly more of the network's accuracy (0.890 against 0.876), although it reproduces SLICER's predictions better (fidelity 0.871 against 0.831, interval above zero); the two point opposite ways because SLICER is the more accurate model, and in absolute terms the same readable columns recover 0.642 of ρ from its predictions and 0.627 from the network's. The unit decides the rest. The network's input is the letters, so its attributions are position × base natively and stable (0.986 over seeds, 0.957 over halves); the shipped SLICER's per-column attributions are not (0.578, 0.278), which is Finding 5's redundancy again, and it is readable at block level, through the substitution map, or on the 104-column model. Predicted in advance: faithfulness gap of at least 0.03 (held); crisprHAL 2 at 0.8 or more per channel (held); both read-outs at 0.8 or more and within 0.1 (held except the network's substitution map, 0.783); SLICER's own per-column agreement below crisprHAL 2's (held: 0.578 and 0.278 against 0.986 and 0.957); no difference in compressibility (did not hold: fidelity favours SLICER by 0.02–0.04, the share of accuracy kept favours crisprHAL 2 by 0.01–0.02); the flank response below (held).

**The two models say the same thing.** Their substitution maps agree at ρ 0.974 (crisprHAL 2 with the shipped SLICER) and 0.968 (with SLICER-104), and both track the model-free map of Finding 7. They also respond alike to the flank-edit tests of test 3, on the DNA each sees: GC −5 points moves crisprHAL 2 by −0.57 [−0.60, −0.54] against SLICER's −0.88, and +5 points by +0.31 against +0.50, so the fall is 1.8 times the rise in both, at about 0.6 of SLICER's size; swapping bases at fixed GC moves the network by −0.002 to −0.009 (SLICER 0.015–0.055). A network built independently, from a window of 189 nt before and 169 nt after the protospacer, learned the same asymmetric GC dependence, which is evidence that the effect is in the data and not in SLICER's features. Shuffling the order of the bases beyond the near 20 nt moves the network by +0.03 and SLICER by +0.26, through its windowed columns (windowed quantum +0.21 and windowed GC +0.07 when edited alone; the model holds no run-length column).

**What can be claimed.** SLICER is interpretable: faithful attributions, a stable block-level and substitution-map reading, a compact counterpart. It is more faithful than crisprHAL 2, and on stability, truth and compressibility the two tie or split by reading, so it is not shown to be more interpretable overall: crisprHAL 2 can be read with the same standard tools to a similar standard. What SLICER adds is scope: its flank columns are named quantities (GC, purine fraction and longest run over 50 to 1,000 nt) that let Finding 7 state a length scale, which a 189 and 169 nt window cannot. *Limits:* the estimators differ (exact path-dependent TreeSHAP against expected gradients sampled 50 times), and the network's faithfulness rises from 0.874 to 0.896 with one block removed, and to 0.901–0.906 with three and six, at 800 draws, so about a quarter of the gap is estimator noise and the rest remains; the block sets differ (12, 28 and 25), and the 28-block SLICER-104 gives the same ordering, so granularity does not explain it; the network's refits on halves use its 48 epochs on half the steps (ρ 0.662–0.671 against 0.691–0.692 on all rows); "interpretable" is measured functionally, with no person in the loop; and the substitution map of the network is an edit test of that network, so its agreement across refits is stability, not evidence of truth, which the comparison with the model-free map supplies.

### 6. The label's ceiling is about 0.946 (0.942 to 0.946 by estimator), measured from crisprHAL's own replicates

**No predictor can exceed ρ of about 0.946 [0.944, 0.947] against the curated label, and SLICER leaves 0.21 to 0.22 of ρ unaccounted for** (0.946 against the best tuned model's 0.7314 and the tuned model's 0.7215 on the 1,204 CONFIRM genes; 0.946 against the frozen champion's 0.707 would be 0.24; against the best single tuned model, 0.7314 in gene-grouped CV (an average of the five best tuned configurations reaches 0.7334), the gap is **0.215**, or 77.3% of attainable). The figure is a replicate-agreement one, from crisprHAL 2's published per-replicate counts, not an inherited figure (`curated_ceiling.py`); the interval is a bootstrap over guides for one estimator of the replicate log-ratio, and a second estimator gives 0.942 (below).

crisprHAL 2's paper reports no replicate statistics, but its counts for two Cas9 replicates and two dead-nuclease controls are published. Pairing Cas9 replicate 1 with control 1 and replicate 2 with control 2 gives two independent estimates of the same quantity, agreeing at **ρ 0.8097** (plain log-ratio of counts per million, on the 33,871 guides that pass their filter). Spearman–Brown makes the released two-replicate score's reliability **0.8948**, and its square root is the highest ρ any predictor can reach against it. The other pairing gives 0.9453, so the figure does not depend on how replicates are matched.

| check | result |
|---|---|
| does their **cohort** reproduce? | their R script filters on an average control count of **73**; applying it gives **33,871** guides against the 33,567 distinct guides in their release (33,495 in their paper's Table 2) |
| does their **label** reproduce? | a plain CPM log-ratio of the two conditions correlates **\|ρ\| = 0.957** with their released score, so these are the replicates behind that label |
| is it conservative? | **not shown**: ALDEx2's posterior shrinks noisy guides toward the prior, which was expected to raise reliability above a plain log-ratio's. A Dirichlet–CLR re-implementation of its estimator (prior 0.5, 128 draws; \|ρ\| 0.993 with the released score) gives replicate agreement 0.798 and a ceiling of **0.942**, so the ceiling is 0.942 to 0.946 by estimator and 0.946 is not a lower bound |

The curated label's ceiling is higher than the published one's, **0.946 against 0.937**, in line with the crisprHAL 2 paper's claim for curation, measured here from its own counts. The published figure came from a different route (Guo's tiling library, not replicates) and its interval [0.929, 0.945] overlaps the curated one [0.944, 0.947], so the difference is indicative only, and the published figure should not be reused for curated numbers (Appendix B).

**The field's version of this ceiling is mis-derived.** The standard benchmarking frame, Konstantakos et al. (2022, *NAR*), takes the replicate Spearman range of 0.71–0.77 as an upper limit of about 0.7 on any predictor, with 0.4 as the threshold for a successful tool. That treats a **reliability** as a **ceiling**. Two corrections push the same way: the largest correlation any predictor can have with a noisy target is the **square root** of that target's reliability (Spearman's 1904 correction for attenuation), and when the target is an average over replicates, as essentially every released activity score is, Spearman–Brown raises the reliability first. On the review's own numbers the attainable ceiling is **0.911 to 0.933**, not about 0.7: understated by 0.21–0.23, of which 0.14–0.18 is the missing square root alone (Part 5, `EXTENSION_LOG.md` §19).

| replicate agreement *r* | √*r* | reliability of a 2-replicate mean | attainable ceiling |
|---:|---:|---:|---:|
| 0.71 | 0.843 | 0.830 | **0.911** |
| 0.77 | 0.877 | 0.870 | **0.933** |
| **0.8097** (measured here) | 0.900 | 0.895 | **0.946** |

*Limits.* Attenuation theory is exact for Pearson under classical test theory; applying it to Spearman, as both the review and this report do, is an approximation that holds well for roughly normal data and is not exact.

---

### 7. What the signal is made of: GC over a few hundred bases, a seed region that inverts between kingdoms, and the two coupled

**Claim.** The flanking effect of Findings 2–3 is carried by **GC specifically**, is **side-symmetric**, is highest at **200 nt** of 14 windows, with a plateau from 150 to 500 nt and no structure finer
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
**absolute** composition, not enrichment against the genome's own background (expressing every window as a percentile of its own genome's distribution does not improve
*E. coli* → *C. rodentium* transfer: 0.7142 → 0.7105, seed 41, no interval), and DNA-shape descriptors are a reparameterisation of composition, not a second channel
(Appendix B4). *Limit:* of the three two-way splits of the four bases, strong/weak (GC) carries the effect and purine/pyrimidine does not; amino/keto was run later (`flank_scale.csv`, partition arms): purine and amino give no gain at any of 14 windows (−0.0014 to 0.0000), and the four bases separately at most +0.103. This
identifies the variable, not the physics (see I1).

**(b) It is side-symmetric, so it is not the orientation effect.** Upstream and downstream agree to within 0.026 at every window. A directional process (fork direction,
transcription direction, one-sided resection) cannot produce that, so the orientation effect (Part 2, biology item 3; Δ 0.051 [0.020, 0.082] against a curated TSS, 0.025 [−0.003, +0.053] once matched for distance and GC) and the composition effect are two
different things.

**(c) The signal is highest at 200 nt, within 0.01 from 150 to 500 nt, and has no structure below ~250 nt.** The association in (a) rises from 50 to 250 nt and falls by 1,000 nt on both sides. Binning
the flank at increasing resolution saturates (curated genic guides, gene-grouped five-fold CV, seeds 41–43, tuned settings with the top-450 selector; reference: family A's
four nested windows, 24 columns, ρ 0.6732):

| flank described by | columns | ρ | share of the composition signal |
|---|---:|---:|---:|
| nothing (base only) | 0 | 0.5617 | |
| GC and purine over the whole kilobase, each side | **4** | 0.6462 | **75.7%** |
| 250 nt bins | **16** | 0.6679 | **95.2%** |
| 20 nt bins | 200 | 0.6687 | 96.0%, no gain over 16 columns |

Doubling the resolution past 250 nt buys +0.0022 (32 columns, ρ 0.6701), and 200 columns score 0.0014 below 32 and 0.003 below 80, inside the fold-to-fold SD of
0.006–0.008 (no gain, not a demonstrated decline): what a bulk property averaged over a few hundred bases predicts, not a motif, a positioned protein or a periodic feature. *Limit:* that there is one scale is the measurement; its physical meaning (one to three persistence lengths of
duplex DNA, far longer than the ~20–30 nt Cas9 contacts) is a hypothesis.

**(d) The guide's own base preferences sit in the seed, and they invert between *E. coli* and eukaryotes.** In *E. coli* (33,567 curated guides), per-base effects on the label (SD units) are
within ±0.04 of zero at positions 1–7 and the twelve strongest are all at 14–20: at position 20, G **−0.322** and T **+0.348**; at position 18, G −0.266 and T +0.210. The
model's gain-by-position profile (63% of the champion's split gain, 62% of the tuned model's, in the eight PAM-proximal positions) and Orro et al.'s human base-editing SHAP profile (A375; concentrated at the PAM-proximal end, with its largest values at positions 16, 20 and 14 in the published model; a different organism, nuclease, label and importance measure) agree in that. The identical
estimator on the human matrix (13,138 distinct guides; sign convention: target GC against label is −0.147 in the 33,567-guide cohort and +0.018 in the human matrix, too close to zero to fix a sign alone) gives at position 20 G **+0.131** and T
**−0.170**. Across the 80-cell matrix the rank correlation is **−0.052**, with **15 of 27** jointly-significant cells opposite in sign (the chance expectation for unrelated maps:
binomial p = 0.70), so the maps are unrelated overall and the reversal is established for G and T at position 20 and for five cells at positions 18–19. G immediately 5′ of the PAM is the
most disfavoured base-position combination in *E. coli* and a favoured one in human, consistent with mammalian-trained predictors failing in bacteria. Practical
consequence: **avoid G and prefer T immediately 5′ of the PAM in *E. coli*** (marginal effects in this screen).

*Bacteria against eukaryotes, or E. coli against one human cohort?* The same estimators were run on 24 tables of the CRISPOR benchmark collection (Haeussler et al., 2016;
`external_data/crispor_effdata/`). The inclusion rule, fixed beforehand, was at least 300 guides: eight human datasets (six libraries), two mouse, one zebrafish. Each
dataset is oriented by its median Spearman correlation with six published predictors (all six agree in sign for all eleven; weakest anchor Wang 2015 at −0.10;
`eukaryote_orientation.csv`); four datasets (Wang 2015, Xu HL60, Xu KBM7, Xu mESC) were flipped by that rule and the other seven kept their native direction.
Split-half reliability of the 80-cell maps is 0.66–0.95.

| pair of datasets (different libraries) | pairs | median ρ between the 80-cell maps | range | corrected for reliability |
|---|---:|---:|---|---:|
| human – human | 26 | +0.456 | −0.047 to +0.711 | +0.59 |
| human – mouse | 16 | +0.443 | +0.101 to +0.695 | +0.56 |
| human – zebrafish | 8 | +0.221 | −0.239 to +0.381 | +0.28 |
| ***E. coli* – human** | 8 | **−0.140** | −0.276 to +0.078 | −0.16 |

Mouse and human share their guide rules to the degree two human libraries do (corrected +0.56 against +0.59); *E. coli* shares none (*E. coli* – mouse, 2 pairs, median
−0.01; *E. coli* – zebrafish, 1 pair, −0.21; the 11 *E. coli* pairs run from −0.28 to +0.08). The position-20 contrast (G minus mean of A and C, SD units; eukaryotes pooled
over every table of the species, including those under 300 guides) says the same:

| | *E. coli* | human (21,567 guides) | mouse (2,239) | zebrafish (1,233) |
|---|---:|---:|---:|---:|
| G minus mean(A, C) at position 20 | **−0.333** [−0.359, −0.308] | **+0.277** [+0.248, +0.306] | **+0.398** [+0.308, +0.482] | **+0.430** [+0.303, +0.548] |

All eleven datasets of at least 300 guides have the eukaryotic sign as point estimates and ten of the eleven intervals exclude zero (Wang 2015, +0.08 [−0.004, +0.160], is
indistinguishable from zero). The orientation step is not independent of this for the four datasets it flipped, whose direction was set by predictors trained on eukaryotic
data (Wang 2015 has the weakest anchor); the seven that kept their native direction all carry the sign with intervals that exclude zero. The sign is also the same in a
second bacterial genome (*C. rodentium*, TevSpCas9, 25,210 guides: −0.42 [−0.45, −0.39]; *E. coli* eSpCas9 at the same loci as the WT screen: −0.21), so the inversion is
**bacteria against eukaryotes**. Zero-shot transfer agrees (SLICER base representation, ridge in parentheses; label rank only), against the target's own five-fold ρ of
0.18–0.55:

| trained → scored (different libraries) | pairs | median ρ (ridge) | median share of the target's own ρ retained (ridge) |
|---|---:|---|---|
| **human → mouse** | 16 | **+0.342** (+0.283) | **73%** (65%) |
| **mouse → human** | 16 | **+0.261** (+0.269) | **70%** (66%) |
| ***E. coli* → eukaryote** | 11 | **−0.041** (−0.056) | −13% (−14%) |

*Limits:* labels are not the same quantity across tables (indel frequency or depletion); zebrafish differs in system (injected embryos); datasets scoring one library in
two cell lines are flagged `same_library` and left out of the pair table; no table has gene identifiers. Why the rules invert is not answered: Moreb and Lynch (2021; 44
datasets, abstract only) attribute cross-species failure to a genome-dependent search time, which these data cannot test (Family H, genome-wide seed-match counts, adds
+0.001 on the curated cohort, +0.000 gene-grouped, and six of its thirteen columns are constant because the cohort keeps only uniquely targeting guides).

**(e) The flank and the seed are coupled, and the coupling is synergy, not compensation.** Of what trees add over a linear model (depth 4 against ridge, on held-out CONFIRM genes), 37% is pairwise; a depth-2 model's
splits are pairs. The leading block pairs are flank composition × flank quantum (0.058), seed indicators (16–20) × protospacer dimer quantum (0.048) and **flank
composition × protospacer basepair quantum (0.048)**; the strongest single pair is `eng.flank.comp.up250.gc` against a property of **position 20** (1.9%). The coupling is
diffuse (no block pair above 5.8%, no column pair above 1.9%). Direct test (H1), Spearman of label on flank GC within each base at position 20 (gene-bootstrap intervals,
30,203 genic guides):

| base at position 20 | A | T | C | G |
|---|---:|---:|---:|---:|
| that base's own effect on the label (SD; all 33,567 guides, from (d)) | +0.193 | +0.348 | −0.142 | −0.322 |
| ρ of label on flank GC | **0.219** [0.188, 0.242] | 0.184 [0.154, 0.217] | 0.169 [0.144, 0.192] | **0.118** [0.090, 0.147] |

T minus G is **+0.066 [+0.029, +0.102]**, and the steepest minus the shallowest slope (A minus G) is +0.101 [+0.066, +0.131]. The flank's slope is shallowest where the seed base is most disfavoured: a guide needs a competent seed *and* an open region. It
is not a compressed label: the label's spread is the same in all four strata (SD 3.08–3.20), and the ordering survives dropping the lowest quartile of the label (A 0.084,
T 0.083, C 0.060, G 0.030), whose share of guides differs only as the seed's own effect predicts (G 35%, C 29%, A 19%, T 15%). *Limit:* these checks make a floor effect
unlikely but do not formally exclude it; depth 2 cannot see third-order structure (depth 4 adds a further 4% of what trees add).

**(f) Pooled, the response to flank GC is a step at the AT-rich end; within genes there is a gentler gradient.** A linear dependence was predicted and failed. Mean label
by decile of flank GC (250 nt each side averaged, 30,203 genic guides):

| decile of flank GC | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean flank GC | 0.406 | 0.460 | 0.483 | 0.498 | 0.510 | 0.520 | 0.530 | 0.541 | 0.554 | 0.579 |
| mean label | **−3.09** | **−0.40** | +0.11 | +0.01 | +0.05 | +0.07 | +0.14 | +0.14 | +0.02 | −0.05 |

Spearman of label with flank GC is **+0.485** among the lowest 10% of the genic guides (below 44.4% GC, 3,086 guides) and **+0.021** among the other 27,117. The lowest decile sits
**3.10 label units lower [−3.37, −2.78] (gene bootstrap), 0.95 SD**, than the rest, and not because of a few genes (857 genes; the ten largest contributors hold 4.4%; without them the gap
is −2.97). **Within genes** (variables centred on the gene mean) the dependence is graded (decile means −0.58 rising to +0.50), and **+0.066** survives setting the lowest
10% aside. So there are two components: an AT-rich penalty of about one SD, mostly *between* genes, and a graded gradient *within* genes (its model counterpart is the +0.085 within-gene flank gain of Finding 3); the
first accounts for most of why the flank effect looks strongest among GC-poor targets (Finding 2, +0.425 against +0.068; with the lowest 10% of flank GC, 500 nt, set aside,
the five target-GC strata read +0.155 to +0.037). The gap is −0.76 SD at 50 nt, **−0.95 at 250**, −0.92 at 500 and
−0.78 at 1,000 nt (ρ inside the tail 0.435, **0.485**, 0.416, 0.389), so the step belongs to regions a few hundred nucleotides wide (point estimates on the same guides; no interval on the
difference between windows).

| robustness of the step (lowest decile of flank GC against the rest) | n | deficit, label units [95% interval] | in SD of that dataset's label |
|---|---:|---|---:|
| *E. coli*, WT SpCas9, all guides (bootstrap over 100 kb blocks) | 33,567 | −3.29 [−3.76, −2.84] | −1.00 |
| *E. coli*, eSpCas9, a different enzyme at the **same loci** (not independent of the row above) | 59,489 | −2.15 [−2.43, −1.86] | −0.80 |
| *C. rodentium*, TevSpCas9, at its own decile edge (GC 50.8%) | 25,210 | −0.57 [−0.75, +0.14] | −0.21 |

The step replicates across an enzyme on the same genome and is **not established in a second genome**; with *E. coli*'s absolute threshold in *C. rodentium* (653 guides)
the interval includes zero (−1.97 [−2.04, +1.10]). It is not expression (among the 29,097 guides with an expression value, flank GC and expression are uncorrelated, ρ −0.003; the association is 0.153 marginal and 0.154
given expression; the deficit is present in all three expression thirds, −3.79 / −2.80 / −1.77 from lowest to highest, and about twice as large in the lowest), and it is not PAM density: holding flank GC fixed, NGG plus
CCN density (both flanks, 250 nt) has a partial Spearman of **+0.022**, the **52nd percentile** of 5,000 random sums of four of the 32 strand-collapsed trimer classes
(null mean +0.005, sd 0.040).

*Does the step exist in eukaryotes?* At the 77 nt scale the eukaryote tables allow (30 nt before the guide, 47 after the PAM), *E. coli*'s lowest decile sits **−0.70 SD**
[−0.73, −0.66] below the rest, ρ **+0.33** inside the tail. In the eleven eukaryotic datasets (intervals bootstrapped over guides, not over blocks as in the table above) the gap is smaller and inconsistent: five have an interval that excludes
zero (from Xu HL60 −0.16 to Doench 2016 −0.45 SD), six do not (including zebrafish), and ρ inside the tail is at most 0.20 (largest 0.2003, Xu mESC; P5 asked for under 0.15): a weaker, inconsistent AT-rich penalty in
about half of the mammalian datasets and not in zebrafish. *Limit:* no gene identifiers (between and within genes cannot be separated), depletion screens carry gene
effects, and mammalian GC content tracks chromatin state.

Predictions written into the module docstrings before the runs (`sgrna/compare/eukaryote_rules.py` P1–P5, `sgrna/interpret/hypotheses.py` H1–H3; the order rests on the author's log, `EXTENSION_LOG.md` §24–25, because each module was first committed together with its results):

| Predicted in advance | Observed |
|---|---|
| **P1.** Mammalian maps agree (corrected correlation at least 0.4 for every pair; mouse – human no more than 0.2 below human – human) | Held in the median (corrected +0.59 human – human, +0.56 human – mouse) and in the mouse – human clause; 9 of the 43 mammal pairs fall below 0.4, 7 of them involving Wang 2015 or Doench 2016 |
| **P2.** *E. coli* unrelated to every eukaryote (ρ within ±0.2) and opposite at position 20 in every eukaryotic dataset (contrast G minus T) | Held in the median (−0.14 uncorrected against human, −0.01 against mouse) and in sign at position 20; 2 of 11 pairs lie beyond 0.2 (Doench 2016 −0.28, zebrafish −0.21). The reported contrast is G minus mean(A, C), not the pre-specified G minus T, which is undefined for the two Hart libraries (no T at position 20) |
| **P3.** Zebrafish agrees with mammals less than they agree with each other (predicted 0.2 to 0.5) | Held: human – zebrafish +0.28 corrected, inside the predicted band; mouse – zebrafish +0.07 (2 pairs), below it |
| **P4.** Mammal-to-mammal transfer keeps at least 60% of the target's own ρ; *E. coli* to any eukaryote lies within 0.05 of zero | Held for mammals in the median (73% and 70%; 13 and 12 of 16 pairs at or above 60%); the *E. coli* clause held only in the median (−0.041; 5 of 11 pairs beyond ±0.05, down to −0.24 for zebrafish) |
| **P5.** No AT-rich step in mammals (gap interval including 0, ρ inside the tail under 0.15; probability about 0.7) | Partly failed: weaker, inconsistent step in about half (five of the ten mammalian datasets have an interval excluding 0; tail ρ reaches 0.19 and 0.20 in two) |
| **H1.** The flank's slope is steeper where the position-20 base is favourable | Held (heterogeneity present, A minus G +0.101 [+0.066, +0.131]; T steeper than G by +0.066) |
| **H2.** A roughly linear dependence on flank GC (probability about 0.6) | **Failed**: a step at the AT-rich end |
| **H3.** PAM density is not what the GC effect is (probability about 0.8) | Held (partial +0.022, 52nd percentile of the null) |

**(g) What follows.** *Measured (E. coli K-12, one curated label):* (a)–(f) (the step of (f) is not in a second genome; (c) and (f) each rest on one protocol). *Inferences, each with what would show it wrong:*

| inference | rests on | would show it wrong |
|---|---|---|
| **I1. Melting-limited unwinding is not the main driver:** more GC means *more* cutting, the opposite of what a melting-limited step predicts | the sign of the step and of the within-gene gradient | a locus-matched design in which AT-rich flanks cut better; a within-gene gradient that reverses with another covariate held fixed |
| **I2. Flank and seed act as a gate, not as compensation** | (e): slope falls from A 0.219 to G 0.118 across seed bases; spread equal across strata | a multiplicative flank × seed term that fits no better than an additive one on gene-grouped folds; the order reversing on another screen |
| **I3. The AT-rich penalty belongs to regions a few hundred nucleotides wide,** not single sites or kilobase domains | the scale check in (f) (point estimates, no interval on the contrast between windows) | the gap growing toward a kilobase on a wider window or in another strain |

*One candidate explanation: AT-rich DNA is less accessible to Cas9.* The pattern alone supports it (a step at the AT-rich end, between regions, not expression, not
outlier genes). **The data do not support any particular accessibility mechanism.** Naming a nucleoid protein such as H-NS, which binds AT-rich DNA and silences
horizontally acquired regions, is a guess from outside the data: nothing here distinguishes it from another AT-binding factor, from AT-rich DNA being more flexible or
bent, or from a cell-level effect (copy number, replication timing). The scale check (I3) fits a local property better than a silenced domain. (Nine of the fifteen genes most often in
the lowest decile, `wbbJ`, `waaL`, `glf`, `waaO`, `alpA`, `trkG`, `lit`, `mcrC` and `casA`, look like horizontally acquired loci by name; the data carry no annotation, so
this is not evidence.) Falsifiers: (1) a published occupancy track for an AT-binding protein (guides inside bound regions should carry the deficit and the step should
shrink once occupancy is conditioned on; an equal deficit inside and outside kills the protein version); (2) a Δ*hns*-type strain that keeps the deficit; (3) a eukaryotic
screen, where a step would mean generic accessibility. The third was run (above): the step is weaker and inconsistent, so a bacteria-specific protein is not required, and
generic accessibility is neither shown nor excluded.

**Limits.** (i) The cross-kingdom comparison is between different assays (survival after cutting against indel, knockout or depletion scores), so it compares design
rules, not mechanisms; 15 of 27 jointly-significant cells opposite in sign is what unrelated maps give (binomial p = 0.70) and is not a test of inversion; the reversal rests on G and T at position 20,
five cells at positions 18–19 and the position-20 contrast. (ii) The step that H2's failure revealed and every follow-up in (f) were found in one screen and
designed after the result they follow, so they describe the data and do not test a prediction. (iii) The step is established in *E. coli* K-12 SpCas9 on one curated label
and not in a second genome.

## Part 2 — Contributions

**1. The published representation is redundant and can be decoded.** 99% of its 6,232 columns regenerate from a 20-mer
and spell out about twenty variables; on the published matrix under a 300-column budget, 412 columns computable for
any guide lose 0.00003 ρ against all 6,232 (random-row 5 × 5 CV, `portability.csv`). The encoding is lossless, so **no deterministic function of the 20-mer can
add information**, though it can add *performance* (+0.011 to +0.019 for a depth-limited tree, curated base); this is why the
imported biophysical families could not have added information (Finding 1; Appendices B2, B3).

**2. Genomic flanking context is the main new signal: +0.164 ρ** on the curated cohort (random-row 5 × 5 CV; +0.157
gene-grouped), **+0.156 on sealed genes** (a 20%-of-guides whole-gene holdout whose sealing re-derives only the automatable
choices, not family A's design; +0.080 on the published matrix) and **+0.170 once the model is tuned** (gene-grouped, 3 seeds,
30,203 genic guides; the champion's figure on the same folds is +0.156; Finding 4). It is a local compositional
gradient, not a motif (texture-preserving null, which also measures its autocorrelation length), not gene ranking
(+0.085 survives inside genes; Finding 3), and it survives holding out whole genes; read depth and mappability are excluded (strict stratum +0.157, repeated +0.261, given depth +0.160; Finding 3). It transfers across nuclease and organism (88–91% of the locally trained ρ retained with
no retraining; the nuclease row scores guides that were in the training library, so the *C. rodentium* rows carry the
sequence-disjoint claim) and, built on both sides, across kingdoms, where the shared published columns alone do not: it is
worth **+0.023 ρ** on the human matrix on its own ground (0.3063 → 0.3289, twin-grouped folds, ahead in 15 of 15 folds;
`EXTENSION_LOG.md` §21.6) and the transfer point estimates change from −0.049 and −0.010 (base columns, same rows) to
+0.022 and +0.017 (Part 3; Jacky's result, no intervals). Whether it needs a break is **not settled**: on dCas9
silencing screens a weaker within-gene effect remains whose cause the data cannot identify (Appendix B5; Findings 2–3).

**3. The signal is GC over a few hundred bases, and the guide's own preferences invert between kingdoms.** The effect
is **GC**, not composition in general (purine is flat at every distance and side), and is **side-symmetric**, so it is
not the directional orientation effect. It is **highest at 200 nt of 14 windows, within 0.01 from 150 to 500 nt**, with no structure below
about 250 nt; four numbers per guide carry three quarters of it. A genome-relative description does not help it:
re-expressing the 16 raw composition columns as per-genome percentiles changes *C. rodentium* transfer by −0.0037
(0.7142 → 0.7105; one tuned fit, no interval) (`EXTENSION_LOG.md` §20.1). The
guide's own preferences sit in the PAM-proximal seed and are **unrelated between *E. coli* and human** (position × base
maps: ρ = −0.052 against the human matrix of Noshay et al.; median −0.14 over eight other human datasets) and between
*E. coli* and two mouse and one zebrafish dataset (−0.21 to +0.08), while human and mouse agree (median +0.44, 16 pairs); G
immediately 5′ of the PAM is the most disfavoured base here and a favoured one in human. The GC association loses its
size with no fitted model: target GC (the 20-nt protospacer) correlates −0.147 with the label in the curated *E. coli*
cohort (−0.201 on the published matrix) and +0.018 in human (13,138 distinct guides), which is about zero, so the sign does
not reverse; this loss and the transfer failure belong to the *guide* half of the representation. The two
halves are **coupled**: the largest column-pair interaction (1.9% of interaction gain) is 250 nt flank GC against a
position-20 column (Finding 7).

### What is new about the *E. coli* biology, separately from the model

One positive localisation, one directional effect, and constraints that are worth as much because each rules out a
plausible mechanism.

**Positive.**

1. **DNA flanking the target predicts whether it is cut: +0.164 ρ** for family A (curated cohort, random-row CV), whose 24
   window-scale composition columns (50–1,000 nt) alone are worth +0.107 and whose local block within ~10–15 nt of the cut,
   partly inside Cas9's PAM-adjacent footprint, +0.05 to +0.07 alone (gene-grouped, Appendix B4); the association with the
   label is highest at about 200 nt each side (plateau 150 to 500 nt) and is lower by 1 kb (Finding 7); the largest of eleven families (next +0.139). Browne
   et al. showed that adjacent nucleotides carry activity information (0.554 on the guide alone to 0.697 at 378 nt); what is
   added is the effect's **shape**: compositional and smooth, no motif, with a measured autocorrelation length (published
   matrix: 8% survives a 1 kb shift, none survives 10 kb) (Finding 2).
2. **It is a local property of the target, not of its gene:** inside genes, where gene identity is constant, the flanks
   are still worth **+0.085** [+0.078, +0.090] (gene-centred, curated genic guides, 10 seeds; Finding 3).

**Directional.**

3. **Orientation relative to transcription may matter** (Jacky's result): Δ **0.051** [0.020, 0.082] against a curated
   RegulonDB TSS, but **0.025** [−0.003, +0.053] (P(Δ ≤ 0) = 0.04) once opposite- and same-orientation guides are matched on
   distance to the TSS and flank GC (nearest-gene version after matching: 0.007 [−0.024, +0.038]), so that the surrounding DNA
   acts through more than composition is suggested, not shown. Transcription, replication or the supercoiling domains they
   create could carry the direction; which one is unidentified (Finding 7).

**Constraints, each ruling out an obvious mechanism.**

4. **Not helical phasing or intrinsic curvature:** GC skew, A-tract phasing and bendability periodicity at the helical
   repeat give −0.001 alone and +0.0004 on top of the flank composition (published matrix; on the curated cohort the family
   is +0.011 alone and was not run on top of family A) (Appendix A).
5. **Not the physical shape of the flanks, but their letter composition:** shape descriptors are worth +0.120 alone and
   **+0.0009** once composition is present (curated cohort), and a 3D reconstruction of the helical path adds −0.0008 to +0.0006 on top
   of family A (curated cohort; +0.0022 on a second cohort, where an arm-shuffled flank gives +0.0016; shuffled-flank controls
   on the curated cohort −0.0012 and −0.0009) (Appendix B4).
6. **Not supercoiling**, though the family built to test it pays +0.139 (curated, random-row; +0.044 on the published matrix,
   where the controls were run): the untagged no-antibody GapR control alone predicts almost as well as the ChIP (+0.0087
   against +0.0111, 78%) and raw read density alone reproduces 98% of the family's gain, so the family measures read density,
   of which gene dosage is a part: four columns from distance to *oriC* alone are worth as much as the ChIP track (+0.0112
   against +0.0111; published matrix, 100 kb blocks). The "unknown chromosomal factors" of Guo et al. (2018) are, for this
   family, read density, in part copy number under multifork replication (Appendix A).
7. **Not gene essentiality; expression is not null, unlike in Guo et al. (2018):** the transcription family's +0.058 (curated) is
   carried by gene geometry (+0.0393 alone) and expression (+0.0213 alone), essentiality being +0.00004 alone (the three sets
   overlap: they would sum to +0.061); Guo et al. reported gene expression (and
   position in the gene, operon position and replication strand) as unrelated to activity, and the expression term here is not (Appendix C).
8. **Not a restatement of target GC:** conditioning on target GC *raises* flank GC's association, **+0.176 marginal to
   +0.221 partial** (curated cohort; flank GC is the mean of the 500 nt windows on both sides), and the flank effect is
   positive inside all five target-GC strata (Finding 2).
9. **Most of the phenotype is still unexplained:** the label's reliability ceiling is **0.946** [0.944, 0.947] and the best
   single model reaches **0.7314** after post-freeze tuning (gene-grouped CV; 0.7151 for the frozen champion on the same folds),
   leaving about **0.21 ρ of repeatable variance** unaccounted for by every family tried; the same tuning (settings chosen
   with family A present) is worth +0.003 on the base representation alone, so capacity is unlikely to be what the gap is
   made of (Findings 4, 6).

### Three inferences from results in combination

**1. "GC content affects cutting" is two effects of opposite sign, and the window decides which one is measured.**
Spearman with the label (curated cohort): GC of the 20-nt protospacer, which Findings 2 and 7 call target GC, **−0.147**
(−0.201 on the published matrix), of the upstream 250 nt flank window **+0.161**; the sign is that of the label, whose higher
values are more cutting (the 118 guides with more than one genomic copy have median label 2.11 against 0.27 for single-copy
guides, `label_orientation.csv`). Inside the protospacer GC-rich predicts *less* cutting (a GC-rich duplex costs more to unwind in
the R-loop); outside it the same composition predicts *more*, which that account cannot produce at any distance. The two
are positively correlated (target GC against flank GC +0.245, curated), so each suppresses the other, which is why
conditioning on target GC raises the flank association (item 8). *Limit:* marginal associations on one cohort; signs
and scales, not magnitudes.

**2. Family increments measure what is new, not what matters.** The strongest column of each of nine of the eleven
families has marginal |ρ| 0.12 to 0.18, while family increments over a guide-holding model span 0.001 to 0.164
(`recoverability_trend_columns.csv` for the marginals, `ablation_curated.csv` for the increments). The strongest of the 636
family columns, `eng.dg.hybridisation_weighted` (**R² 1.00 recoverable from the guide's indicators, 0.90 correlated with
guide GC**), has |ρ| **0.184** for a family increment of +0.019 (some base columns are stronger, up to 0.255); `a_flank`'s strongest
(`eng.flank.comp.up250.gc`) is 0.161 for **+0.164**. So "nothing confined to the 20-mer clears +0.019, nothing above
+0.05 is confined to it" ranks families by what the model cannot already compute, not by what cutting depends on.

**3. The guide's own folding is a real determinant.** The spacer's minimum free energy (`eng.dg.guide_selffold`) is worth **+0.0117** on the
curated base (+0.0119 recomputed through the same code path) with marginal **|ρ| 0.102**; it is fully determined by the twenty bases, so its small increment says
nothing about importance, and the control does: the same ViennaRNA computation on each guide with **one fixed
permutation of its twenty positions** is worth **+0.0000, −0.0007 and +0.0002** (three permutations), and random non-linear functions of the
guide of order 2, matched on out-of-fold recoverability (R² 0.34–0.48 against 0.32), are worth −0.0004 to +0.0007 (orders 3 to
8, with R² ≤ 0.20, −0.0010 to +0.0016) (Appendix B). So it is the particular structure the real spacer forms, not composition or tree-difficulty.
*Limit:* one cohort.

---

## Part 3 — Where this stops

### What the work does not show

- **One organism and two relatives** (the same genome under eSpCas9, and *C. rodentium*). No genome-wide Cas9-cutting
  screen outside *Enterobacteriaceae* is known to us; what exists elsewhere is CRISPRi, a different measured quantity, so
  closing this needs wet-lab work. On the **shared published columns** (6,216 columns shared by the published 13,880-row
  *E. coli* and 17,421-row human matrices), cross-kingdom transfer is absent in both directions: *E. coli* → human
  **ρ −0.046 [−0.061, −0.031]**, human → *E. coli* **−0.016 [−0.032, +0.001]** (`kingdom_transfer_boot.csv` in the main project `results/`; the fork's copy holds only
  the two `human_fitted` rows; 2,000 test-row resamples that treat repeated human guides as independent, champion settings
  both sides). With flanking context
  built on both sides (6,169 base columns plus family A, 16,490 human rows) the point estimates become **+0.022** and
  **+0.017**, against −0.049 and −0.010 for the base columns on the same rows, and flanks alone give **+0.042** and
  **+0.108** (`Jacky/human-boundary/results/family_a_kingdom_transfer.csv`; no intervals were computed). Residualising GC and
  matching the GC histogram leave the base-column *E. coli* → human figure at −0.049 and restricting to a GC range gives
  −0.060 (`gc_matched_transfer.csv`), so the fix is the representation, not the covariate. **The human matrix has 8,499
  of 17,421 rows (49%) that repeat a guide appearing elsewhere.** The zero-shot figures use no human folds, but the human
  local figures do: on twin-grouped folds the local ρ behind Finding 3's 'retained' share falls from 0.508 to 0.2515, the
  family-A increment halves (+0.048 to +0.023) and the human champion reaches **0.3377** against 0.7314 here (a different
  cohort and representation); the matrix needs deduplication first, and the local figures of Finding 3 have not been re-run.
- **The label is survival, not cutting.** Measured efficiency is **P(cut) × P(the break is unsurvivable)**, every model
  here is fitted to the product, and no experiment separates the terms; the dCas9 screens did not (Appendix B5).
  RecBCD attenuation at Chi sites, the one repair route testable from sequence, is excluded: Chi sits at the 41st
  percentile of 200 GC-matched controls (Appendix B).
- **The label is crisprHAL 2's curated score, which is a fitted procedure.** It is **bit-identical** to their released
  `y` on all 33,567 guides, so the head-to-head compares two models on one quantity; but the two cohorts' labels are
  different quantities (Noshay's `cut_score` runs 0 to 48.05, crisprHAL's −9.05 to +6.21 with 47.5% negative; rank
  agreement ρ 0.945 on 8,617 overlapping guides, same sign for target GC: −0.147 on the curated cohort, −0.201 on the published matrix), so every number
  belongs to one label or the other. The filter is an average control read count with cutoff **73** for the SpCas9 set
  (31 for eSpCas9, 56 for *C. rodentium*), **chosen by cross-validation for model performance** and applied
  **before** the split, so it is a hyperparameter and cleans the test set too, raising ρ partly by making evaluation
  easier. At fixed rows a label swap is worth only +0.004; the family table grew because the flank-GC slope is about
  **3× steeper** in the guides crisprHAL added (2.96×; 3.2× outside the published matrix's 1,222 genes; the other candidates
  tested, label, row count and base, account for +0.014 of the +0.084 gap) (Appendix B2a). Further curation does not help: dropping the noisiest
  decile of training guides costs **−0.0016** and a *random* decile **−0.0014**; inverse-variance weighting costs
  **−0.0050**, of which −0.0041 is reproduced by permuting the weights, so the cost is effective sample size
  (Appendix B). Curation buys a cleaner label (ceiling 0.946 against 0.937; ρ 0.708 against 0.686, +0.022, for the same model fitted to and
  scored on the raw log-ratio).
- **No specificity claim.** Six of `h_offtarget`'s thirteen columns (`mm1`, `mm2`, `seed_exact_mm1..4`) are zero for
  every guide because the cohort keeps only uniquely-targeting guides, so the family's +0.0011 (curated cohort) is not evidence that
  off-target descriptors are uninformative; a cohort retaining multi-targeting guides is needed.
- **No nickase claim.** Base and prime editors bind and nick rather than cut, which puts them between cutting and
  binding-only assays; this needs an editing-outcome readout. Nuclease-information analysis was removed as having no
  substance for this project (`REMOVED.md`).

### Table of limits

**Addressable from data already on disk.**

| limit | why it binds | what would move it |
|---|---|---|
| Interpretation is conditioned on a redundancy-blind selector | the 300 columns are the top *k* of a gain ranking, and gain over-weights binary columns by 14 points of composition; "45 columns carry half the gain" and every family increment inherit this | **not the selector, tested:** a redundancy-penalised rule costs −0.0023 and leaves a *more* binary set (Appendix B); the fix is block-level attribution (Finding 5) |
| The label is P(cut) × P(the break kills) | survival is the only readout | not separated: Cui's screen carries two dCas9 expression arms (strains LC-E18 and LC-E75, both grown 17 generations), two readouts of one experiment and not a time course; dCas9 cohort only (Appendix B5) |
| The gene effect is removed by design, not modelled | the gene-grouped protocol and the CONFIRM genes hold out whole genes (the head-to-head folds are random-row); Yu et al. (2024) fit a mixed-effect model with gene as a random term (Part 5) | **tested, does not bind** (`EXTENSION_LOG.md` §18.1, tuned model on the CONFIRM genes): shrunk per-gene intercepts move accuracy from between-gene to within-gene (−0.0080 / +0.0016) and leave pooled ρ at −0.0003; between-gene ρ is 0.729, so there is little nuisance to absorb |

**Addressable, but only with compute not spent.**

| limit | why it binds | what would move it |
|---|---|---|
| "Degraded by wider input" is a claim about their *published configuration* | their epochs, dropout, filters and GRU width were all chosen at 378 nt | 4–6 GPU-hours: `colab/crisprhal_tune_by_width.py`, anchored on the 378 nt positive control, interpretation pre-registered both ways |
| **The lead is a product-level claim (both models as shipped), not an architecture claim** | SLICER's settings were searched (200 draws plus a five-family mix; 97–157 draws per fold on our folds); crisprHAL 2 is used as released. A nested search of its own on our folds found nothing better than its published configuration in four of five folds, and a comparable small search (12 draws per fold) bought SLICER's LightGBM +0.0103 where theirs, at 5–8 draws per fold, bought nothing; neither is the scale of SLICER's later search | a matched search on its own splits: `colab/crisprhal_retune_own_splits.py` (selects on an inner split of their *training* file, scores their test file once, smoke-tested on CPU, not yet run on a GPU; 8–10 T4-hours at 40 draws for SpCas9, about three times that for eSpCas9, so a null from it is a lower bound). The as-shipped comparison does not depend on it |

**Not addressable without new experiments.**

| limit | why it binds | what it would take |
|---|---|---|
| **~0.21 ρ of repeatable variance is unexplained** (Part 2, item 9) | eleven families from genome, thermodynamics, ChIP-seq and quantum chemistry have not closed it, and capacity is not the cause (tuning: +0.016 with family A, **+0.003** on the base representation; Finding 4) | not more features of the kinds tried and not a better learner: the sequence-derived candidates with a mechanism returned nothing (shape and 3D path under boosted trees, Appendix B4; periodicity; Chi; supercoiling), so this is a measurement problem |
| The long-range mechanism is unidentified | the dCas9 screens do not settle whether it needs a break (Appendix B5); orientation relative to a TSS is a candidate (Δ 0.025 [−0.003, +0.053] after matching; Part 2, item 3), the nearest-gene version is not; GC skew and helical periodicity add nothing on the published matrix | an intervention: a designed library varying flank composition at a **fixed guide**; in every existing pooled screen guide and locus are confounded by construction |
| No untouched external set remains **for the cutting claims at the same loci** | eSpCas9 proved confirmatory and Cui's screen is spent. Sealing whole genes holding 20% of the guides (about 22% of the genes) over **fifteen draws** lowers family A's gain by **+0.0087 ρ [+0.0020, +0.0153]** (mean ± 1.96 se), from **+0.1647** in random-row cross-validation inside the development pool to **+0.1560** on the sealed genes: the most the automatable adaptive component could be (the frozen arm re-derives nothing, and holding out whole genes inside the pool removes only +0.0007 of the drop, 8 sealings), landing on differences not levels and not reaching the human choices (Part 4); on a sealed genome (*C. rodentium*) the tuned settings' benefit is intact at **+0.0172** over the frozen champion | permanent on the cutting side: Rousset and Wang are untouched but are CRISPRi screens (Appendix B5). The human part of the adaptive component and the cohort choice are unquantified; the margin over crisprHAL 2 on our folds is read the same way, and the parity claim on their splits is untouched (Part 4) |

## Part 4 — The freeze, and what remains

**The analysis is frozen as of 11 October 2026: no model, feature or hyperparameter choice is made against the curated
five-fold split.** Settings chosen afterwards were chosen on the tuning part (70% of genes) of a gene split and judged on the
held-out 30% or on other data (Appendix B). A guide-design tool is not a deliverable: the findings are the contribution, and
`featurise.matrix()` turns any 20-mer into the published representation if one is built later. Importing the package is
side-effect free: `sgrna/paper/import_check.py` asserts that 161 of the 163 modules import (the other two need the TensorFlow
environment) and that importing changes nothing under `results/`.

### The sealed-gene drop is small, and not shown to be adaptive overfitting

**Sealing whole genes lowers family A's gain by +0.0087 ρ [+0.0020, +0.0153] between random-row cross-validation inside the
development pool and the sealed genes. That is the most this design could attribute to adaptive overfitting of the automatable
choices, and little of it can be: the frozen configuration re-derives no hyperparameter, an 8-draw search adds nothing
detectable, and holding out whole genes inside the pool as well removes only +0.0007 of it (8 sealings, below). The human
choices above it are unquantified, and one of them, the cohort, cannot be.** It falls on differences between arms rather than on levels. Under this protocol (random-row
CV inside the 80% pool, an 80%-sized sealed fit, frozen XGBoost on 300 of 6,517 columns minus base-only XGBoost on 6,169
columns with no selection) family A reads +0.1647 in pool CV and +0.1560 on sealed genes; the full-cohort random-row figure
is +0.164, the gene-grouped one +0.157 (+0.1563 for the champion over 3 seeds), so the sealed figure, or both, is the one to quote.

**Most of the pipeline is not adaptive.** The decisions identified that could have inflated the five-fold figures, with where
each was made:

| decision | chosen where | adaptive w.r.t. the curated folds? | effect |
|---|---|---|---|
| 300 columns; champion XGBoost settings; gain selector | inherited from the baseline script (`N_FEATURES = 300`), not searched on either cohort | no | none: ±0.0005 against every column; 0.0002 against LightGBM at its own defaults; a LightGBM gain selector 0.7033 against 0.7025 for the XGBoost one used |
| family A's existence | designed on the published cohort, from residuals computed there | no (a different but overlapping cohort: 8,617 of the curated 33,567 guides are in it, same screen) | it is the effect |
| family A's window set (`config.FLANK_WINDOWS`: 50/250/500/1,000) | swept on the published cohort | no, but a sweep winner carries its own curse | none: rebuilt at 75/300/600/1,200 nt, sharing no value, **+0.1634 against +0.1638** (0.0005 against the ±0.005 declared in advance) |
| per-fold hyperparameter search (627 draws) | curated cohort, nested inside each training fold | no: it cannot see a validation row | +0.0131 gain (LightGBM, seed 41), leak-free by construction |
| the 412/427-column reduced set | published cohort | n/a: it does not transfer (Finding 1) | none |
| ρ as primary metric | crisprHAL 2's reporting convention | no | nine-metric panel in `results/metric_panel.csv` |
| **the choice of cohort** | **here, partly because curation improved the score** (ρ 0.607 on the published matrix → 0.628 on restricting it to the 8,617 rows crisprHAL kept) | **yes** | **unquantified** |

The settings a reviewer would suspect first were inherited and are insensitive, and the search that was run is nested.
The cohort choice is adaptive and cannot be undone: every figure is computed on the rows it selected, and the alternative
cohort is a different, smaller, noisier draw rather than a held-out sample of the same thing.

**The sealed-gene drop is +0.009 ρ, not zero.** Whole genes holding 20.0% of the guides (21.5–23.2% of the 4,013 genes; the
3,364 intergenic guides stay in the pool) are sealed (`sealed.py --parts holdout`); column selection is re-derived inside the development
pool for both arms with family A and an 8-draw hyperparameter search for the dev-tuned arm, cross-validation inside the pool is
random-row 5-fold, and each arm is scored once on the sealed genes. A reference arm whose configuration was not chosen on these rows absorbs the population difference, and
the difference of differences is the measured drop. Fifteen sealing draws, training sizes matched so the sealed fit sees
no more data than a CV fold:

| | mean | sd across draws | 95% CI (mean ± 1.96 se over the 15 draws) |
|---|---:|---:|---|
| reference gap (nothing chosen on these rows) | −0.0121 | 0.0148 | [−0.0196, −0.0047] |
| frozen: the configuration the report quotes | −0.0035 | 0.0077 | [−0.0074, +0.0004] |
| **difference of differences, frozen (the sealed-gene drop)** | **+0.0087** | 0.0131 | **[+0.0020, +0.0153]** |
| difference of differences, dev-tuned | +0.0082 | 0.0140 | [+0.0011, +0.0153] |

Twelve of the fifteen draws are positive (t = 2.56, p = 0.022; Wilcoxon p = 0.015; sign test p = 0.035; the t-interval of the frozen component is [+0.0014, +0.0159]
and its 20,000-draw bootstrap [+0.0024, +0.0152]; the draws overlap, each sealing about a fifth of the same genes, so these
describe the spread across sealings), and dropping any single draw moves the mean only between +0.0065 and +0.0106. The reference gap itself
swings with sd 0.0148 between sealings, which is why fewer draws cannot resolve an effect of this size.

**The levels are not detectably inflated; the arm differences are.** The frozen arm scores 0.7043 in pool CV and 0.7077 on
sealed genes (gap −0.0035, interval contains zero). The base-only arm, on which nothing was chosen, moves from 0.5396 to 0.5517
(+0.0121), so the family-A gain reads **0.1647 in pool CV and 0.1560 on sealed genes (−0.0087)**, the same drop again by
algebra and not a second measurement: optimism cancels between arms only when both were chosen on the same rows, and these
were not. *Splitting the drop.* Pool CV is random-row and the sealed score is on whole genes, so the drop could also contain
what unseen genes cost a feature that reads the locus. The pool CV was therefore re-run with whole genes held out (guides
outside any gene grouped by 100 kb block) for the first eight sealings (`sealed.py --parts gene`,
`results/sealed_holdout_gene.csv`; eight of the fifteen for compute, and their random-row component, +0.0088,
matches the 15-draw +0.0087):

| 8 sealings (mean; interval mean ± 1.96 se) | frozen | dev-tuned |
|---|---:|---:|
| difference of differences, random-row pool CV (as above) | +0.0088 | +0.0079 |
| difference of differences, whole-gene pool CV | +0.0081 [−0.0037, +0.0199] | +0.0060 |
| cost of random-row folds (the first minus the second) | +0.0007 [−0.0011, +0.0024] | +0.0019 [+0.0006, +0.0032] |

Family A's gain is +0.1648 in the pool under random-row folds, +0.1641 under whole-gene folds and +0.1560 on the sealed genes, so
random-row folds account for about +0.001 of the drop and the rest remains when both the pool CV and the sealed score hold out
whole genes. Nor does a re-derived choice produce it: the frozen arm re-derives none, and the dev-tuned arm, which adds an
8-draw search inside the pool, is not larger. What remains is a difference between family A's gain in the pool genes and in the
sealed genes, which this design cannot attribute to adaptivity (the intervals of the eight-sealing components include zero, and
the draws overlap).

The same reasoning applies to the margin over crisprHAL 2 on our own five folds (our configuration was chosen on those rows,
theirs was not): the pre-freeze margin of the per-fold-tuned model over their best configuration, +0.0161 to +0.0172 (the
final model's is +0.0216), would carry up to this much self-inflation, leaving roughly **+0.008** on unseen genes. That is
consistent with the +0.002 to +0.004 margin of the frozen champion on their held-out splits and is a second contributor
beside the data-volume account in Part 5. The head-to-head on crisprHAL 2's own splits is the cleaner comparison: no weights of either model were fitted on a test row, nothing was selected on the eSpCas9 split, and the CONFIRM genes are the clean check for the WT-SpCas9 split (Part 5).

*Limit.* The sealed-gene design reaches only the automatable choices, and the frozen arm has none to re-derive: no
re-derivation reaches the human choices (building family A, the window set, the cohort), so +0.0087 says nothing about their
size, and the sealed-organism check below reaches the settings but not these. The figure is therefore not a lower bound on total
adaptivity: it is the measured sealed-gene drop, which equals the automatable adaptive component plus any difference between the
pool genes and the sealed genes.

### For the settings chosen after the freeze, adaptive overfitting is not detected

**Their benefit holds on 1,204 genes untouched by any search, on a second gene split, and on a genome nothing was chosen on
(+0.0172).** Which data could have been sealed depends on what each was used for:

| data | used for | anything chosen on it? | at loci the *E. coli* cohort does not hold? |
|---|---|---|---|
| Noshay's published matrix (Supplementary Table 2: 13,880 guides in the alphabetical block of genes aaeA to leuC, 1,222 genes under our assignment; the paper's own table counts 40,468 sgRNAs; same screen as the curated cohort, label ρ 0.945 on 8,617 shared guides) | Finding 1, the 412-column result, the harness anchor, family A's design and window sweep | **yes**: family A's existence and windows | no |
| crisprHAL curated WT-SpCas9, 33,567 guides | the working cohort: every model and figure | **yes**: the cohort itself; hyperparameters (on 70% of its genes) | n/a |
| eSpCas9, 59,489 guides (their split 47,591 / 11,898) | transfer; the head-to-head on their split; an external check during tuning | nothing selected, but scores were visible: confirmatory | no |
| *C. rodentium* TevSpCas9, 25,210 guides (5.35 Mb genome) | the frozen champion's transfer test (Finding 3); the sealed check below | **no** | **yes** |
| human matrix, 17,421 rows (13,138 distinct) | kingdom transfer; the design-rule inversion (Finding 7) | human-specific analyses only; its row-wise folds leaked duplicate guides (Part 3) | yes |
| three *E. coli* CRISPRi screens (Cui, Rousset, Wang) | Appendix B5 only | no model or setting | partly |

Not all of it was needed: the CRISPRi screens underpin no claim, and they cannot seal a cutting claim because they measure
another quantity. The first three rows are one screen at one set of loci, so holding one out as a dataset leaves the loci in,
and flank composition is a deterministic function of locus. The seals that count within *E. coli* are by whole gene, not by block, so a sealed guide's flank windows (up to 1 kb) can
overlap training loci beside its gene (sealed genes: a +0.0087 drop in family A's gain; the 1,204 CONFIRM genes, 30% of the
genes and never used by any search: a +0.0324 lead over crisprHAL 2's published configuration). That leaves one dataset
independent of the loci and never chosen on, *C. rodentium*. eSpCas9 was never selected on; crisprHAL 2's shipped SpCas9 split
is made of guides from the cohort whose DEV genes fed the settings search (no weights were fitted on its test rows), and has
carried seven arms (SLICER's original 0.6997 and six data-ablation arms). Both are confirmatory but not untouched; no
untouched external set remains at the same loci.

**The same seal, scored on a second screen.** Train on the WT-SpCas9 guides of the development genes of each of the 15 sealing draws and score the
*eSpCas9* labels of every eSpCas9 guide that lies in a sealed gene (11,609 to 12,200 guides in 861 to 932 genes per draw; `sealed.py --parts crossscreen`,
`results/sealed_crossscreen.csv`). The enzyme changes as well as the genes, and no step has fitted these labels, which answers the
suggestion of holding one screen out: dropping a screen after it has been looked at cannot make it untouched, but sealing its guides by gene
while training on the other screen's labels gives a test no earlier one was. The prediction was written into the module before the run: family A's gain
(300 columns selected inside the training genes, against the base alone) positive in at least 14 of 15 draws with a mean of at least +0.08.
Observed: **+0.1033 [+0.0997, +0.1070], positive in 15 of 15 (range +0.0917 to +0.1157)**, against +0.1560 [+0.1506, +0.1615] for the
WT labels of the same sealed genes (eSpCas9 base 0.5656 → 0.6689; two-thirds of the WT gain; mean ± 1.96 se over draws, which share genes). The status
stays *confirmatory*: eSpCas9 scores were visible during tuning and in the head-to-head, and the loci are the same genome (a sealed guide's
flank windows can overlap training loci beside its gene). What the test adds is that the gain survives a change of enzyme and of gene together and is not a property
of one screen's labels.

**The sealed check.** Train on all 33,567 *E. coli* guides and score the 25,210 *C. rodentium* guides zero-shot, with the same
protocol for each arm (`sealed_organism.py`; three seeds; 16 blocks of 100 kb resampled for the interval). Predicted in
advance: tuned at least as good as the champion, by less than within *E. coli* (+0.014 to +0.026). Observed: at least as good
(the first half held), and by +0.0172, inside that range rather than below it.

| arm | ρ on *C. rodentium*, mean of 3 seeds (sd) | minus champion, seed-averaged predictions [95% over blocks] |
|---|---:|---|
| frozen champion | 0.6986 (0.0038) | |
| tuned XGBoost (d099x) | **0.7160** (0.0014) | **+0.0172 [+0.0162, +0.0206]** |
| XGBoost average of 5 | 0.7170 (0.0012) | +0.0165 [+0.0154, +0.0195] |

The tuning's benefit lies within the range seen inside *E. coli* (+0.014 on genes it never saw; +0.019 and +0.026 on crisprHAL 2's
two splits), so the settings are not specific to the data they were chosen on. The two mean-of-seeds columns differ by
+0.0174; the +0.0172 is the difference of the seed-averaged predictions, which the interval (16 blocks, so coarse) is computed on. The champion's 0.6986 reproduces the transfer
figure of Finding 3 (0.698), so the protocol is the same one.

*Limit.* The check does not reach the cohort choice or family A's design (both made on *E. coli*; the transfer shows the
effect crosses to another genome, not that the choice was unbiased), the question of equal tuning against crisprHAL 2 (Part 5), or the
opposite direction, which would score the genes the settings were chosen on. For a next organism: seal first, seal by gene, and
keep a whole genome back.

### What remains open

Each question is open because of something this project does not have.

| question | what would answer it | why it is open |
|---|---|---|
| Does the lead survive equal tuning of crisprHAL 2? (a method-level question about the architecture) | their search on their splits, about 57 GPU-hours for 627 draws (`colab/crisprhal_retune_own_splits.py`, estimated 8–10 T4-hours for 40 draws on one split) | not run. The as-shipped comparison in Part 5 stands either way; this is the arm that would let the claim be about the architecture |
| Would re-tuning crisprHAL 2's other hyperparameters at each input width change the width result? | `colab/crisprhal_tune_by_width.py`, anchored on the 378 nt positive control; 4–6 GPU-h | not run. Browne et al.'s own per-dataset length search (sequence extended 500 nt each side) settled on 378 nt (*E. coli* SpCas9) and 406 nt (eSpCas9), with a small mean gain per added nucleotide (+0.00034 ρ per upstream nucleotide, +0.00074 per downstream one, averaged over the added span; 0.554 on the sgRNA alone to 0.697 at 378 nt), so a change is unlikely |
| Does measured nucleoid-protein occupancy add anything beyond family A? (H-NS/Fis ChIP, GSE24991; IHF/HU, SRP008538; IPOD-HR, GSE164796) | one processed track mapped to the 33,567 loci, plus one ablation arm | no processed track was fetched. Bounded by argument, not measured: family E (Hi-C and NAP dependence) is worth +0.072 alone but is one variable with the read-density families, all ten other families together add only +0.039 beyond family A (Appendix B3), the flank effect has no structure below ~250 nt (Finding 7), and coarse NAP occupancy is AT-driven, which family A's composition columns carry |
| Would a measured DNA shape or stiffness matrix differ from the tabulated pentamer means? | ABC/muABC tetranucleotide stiffness (Pasi et al. 2014) | no download page was found; the local-predictor half is closed by measurement (`EXTENSION_LOG.md` §22.3) |
| How much of the unexplained ρ is repair rather than cutting? | an assay separating cutting from survival for the same guide | no pooled survival screen can provide it; this is the honest form of the 0.215 ρ gap |
| Does the lead hold on an untouched cutting screen? | a new, never-used *E. coli* cutting screen | none remains at the same loci (the flank gain, though not the lead over crisprHAL 2, was tested on eSpCas9's labels in sealed genes: +0.1033, but that screen is confirmatory); a different genome carries the sealed check on the tuned settings (+0.0172). The dimension on which this project is weakest |
| What about the genes of the published matrix (aaeA to leuC) lowers the flank gain there (+0.102 against +0.180 in all other genes, curated cohort)? | gene-level covariates (length, expression, function, essentiality) tested as moderators of the gain | not run. The file is an alphabetical block of the screen's genes and not a designed panel, so the difference is whatever separates genes named a to l from the rest (their base is easier: ρ 0.607 against 0.542); the essentiality explanation is argued against (family G's essentiality column is worth +0.00004) |
| Do the human-arm results hold on a deduplicated matrix, and does the human shape result (+0.018 on top of family A, where *E. coli* gives zero) mean anything? | the 13,138-guide deduplicated matrix searched under twin-grouped folds; human family A rebuilt on our own hg38 flanks | the cross-kingdom transfer figures (shared columns: −0.046 [−0.061, −0.031] and −0.016 [−0.032, +0.001]; family A on both sides: +0.022 and +0.017, no intervals) use no human folds but rest on a matrix with repeated guides, and the human local figures behind them are row-wise and not re-run; human runs are outside the current scope |

## Part 5 — Where this sits in the field

Performance numbers from different cohorts, labels and metrics are not commensurable, so each comparison below states what is held equal. The comparator is crisprHAL 2 (Browne et al., 2026), the strongest published predictor of bacterial Cas9 cutting known to us. A literature search on 22 November 2026 found no newer bacterial cutting model; the nearest neighbours measure CRISPRi depletion, base-editing fitness, or SaCas9 PAM preference (the crisprHAL group's own, *Nucleic Acids Research* 2026, doi:10.1093/nar/gkaf1520), and a search that fails to find something is weak evidence that nothing exists.

### The head-to-head with crisprHAL 2

**The final SLICER model, as shipped, leads crisprHAL 2, as released, by +0.023 to +0.029 ρ on both of its authors' own splits, at a fraction of the cost on a CPU (§30 of the extension log for the GPU).** Both are used as released in practice, so that is the comparison: SLICER with its settings chosen on held-out genes (XGBoost over the base representation plus family A, Finding 4) and crisprHAL 2 with its published weights. Before tuning, no difference is detected (paired Δρ +0.0021 [−0.0031, +0.0072] on eSpCas9 and +0.0039 [−0.0047, +0.0124] on WT-SpCas9). The result depends on what is held equal, so four conditions are reported; a single headline number would be a choice of condition dressed as a finding.

| condition | what is held equal | result for the final model |
|---|---|---|
| **their two splits, their published weights** | rows, label, split, and each model as its authors ship it | **SLICER +0.0286** [+0.0239, +0.0331] on eSpCas9 and **+0.0226** [+0.0139, +0.0314] on WT-SpCas9 |
| **our five folds** | rows, label and folds; crisprHAL 2 at its published configuration and at the best configuration any search found for it | **SLICER +0.0288** [+0.0251, +0.0326] against their published configuration, **+0.0216** [+0.0180, +0.0252] against the best configuration found for them |
| **equal input DNA** | all of the above *plus* the DNA each model reads | **depends on the width**: crisprHAL +0.012 on the guide and PAM alone and +0.009 at 378 nt (batch 256), SLICER +0.066 at ±1,000 nt (original model; five folds each, one seed) |
| **a different assay, neither model trained on it** | rows, label, and that neither has seen the data | **crisprHAL +0.026** [+0.023, +0.029] (frozen model, 76,325 guides; not like-for-like, see the last subsection) |

**What the comparison claims.** A product-level claim: used as released, SLICER scores higher on crisprHAL 2's own test splits. Published predictors are compared as released in this field's benchmarks (Konstantakos et al., 2022, ran published tools through the authors' code, weights or web servers on sequence-only inputs, and retrained only two CRISPRLearner models that could not be reproduced and, on sequence features alone, CNN-SVR and C-RNNCrispr), and it is what a user experiences. It is not, by itself, a claim that boosted trees beat convolutional-recurrent networks: SLICER's settings were searched and crisprHAL 2's were not, and general guidance on method comparisons asks for equal tuning budgets before an architecture is credited (Lones, 2021). Two things bound that reading: a nested search of crisprHAL 2's own on our folds found nothing better than its published configuration in four of five folds (Appendix B), and `colab/crisprhal_retune_own_splits.py`, which would supply the matched search on its own splits, has not been run. As an extrapolation and not a measurement: the best configuration any search found for crisprHAL 2 lifted its per-fold mean ρ by +0.0080 on our folds (0.6971 to 0.7051; +0.0072 on the pooled out-of-fold predictions, 0.6968 to 0.7040); the same gain on its own splits would leave SLICER's lead at about +0.015 on SpCas9 and +0.021 on eSpCas9. Interpretability is not among the things the comparison claims for SLICER: Finding 5 puts both models through the same four tests and finds SLICER's attributions the more faithful and the two tied, or crisprHAL 2 ahead, on stability under refitting, agreement with the model-free design rules and compressibility.

#### The final model leads on both of crisprHAL 2's splits

**With hyperparameters chosen on 70% of the genes (DEV) and confirmed on the other 30% (CONFIRM), SLICER leads crisprHAL 2's released weights by +0.0286 ρ [+0.0239, +0.0331] on its authors' eSpCas9 split and +0.0226 [+0.0139, +0.0314] on their SpCas9 split, with τ, Pearson and R² clear in the same direction and the two top-end metrics of the table below (top-decile precision, pick percentile) tied.** The numbers in this subsection are recomputed from stored files by `sgrna/paper/verify_numbers.py`, `sgrna/paper/paper_numbers_C.py` and `sgrna/paper/report_checks_R6.py`, except where a sentence says otherwise; the working record is `EXTENSION_LOG.md` §13. That the frozen model only matched crisprHAL 2 before tuning is therefore a property of its settings, not of the model class.

**Tuning design.** Settings were selected on the DEV genes (70% of the 4,013 genes) of the 33,567-guide WT-SpCas9 cohort (30,203 genic guides). The CONFIRM set is the remaining 1,204 genes (8,943 guides), untouched by any search; SLICER trained on the DEV genes only (21,260 guides) gains **+0.014 [+0.011, +0.017]** over the frozen champion there. No model's weights were fitted on a test row, and each SLICER arm trains on the split's training rows only, with selection inside them. No eSpCas9 label entered the search, but 57% of the eSpCas9 test guides (6,778 of 11,898) are also guides of the WT-SpCas9 cohort, so the settings were chosen partly on the other nuclease's labels of the same guides; the WT-SpCas9 split and our five folds hold guides of the DEV genes, so their labels fed the choice of settings (not any weights), which is why the CONFIRM row is the clean check for them. The lead replicates across all of them:

| paired on guides, 95% bootstrap CI | crisprHAL 2 ρ | frozen champion Δρ | tuned ρ | **tuned Δρ** |
|---|---:|---:|---:|---:|
| their eSpCas9 split | 0.7821 | +0.0021 [−0.0031, +0.0072] | 0.8107 | **+0.0286 [+0.0239, +0.0331]** |
| their WT-SpCas9 split | 0.6945 | +0.0039 [−0.0047, +0.0124] | 0.7172 | **+0.0226 [+0.0139, +0.0314]** |
| our five folds vs their published configuration | 0.6968 | +0.0115 | 0.7256 | **+0.0288 [+0.0251, +0.0326]** |
| our five folds vs the best configuration found for them | 0.7040 | +0.0043 | 0.7256 | **+0.0216 [+0.0180, +0.0252]** |
| CONFIRM genes vs their published configuration | 0.6892 | +0.0181 [+0.0098, +0.0265] | 0.7215 | **+0.0324 [+0.0248, +0.0403]** |
| CONFIRM genes vs the best configuration found for them | 0.6976 | +0.0097 [+0.0018, +0.0179] | 0.7215 | **+0.0240 [+0.0162, +0.0318]** |

On the CONFIRM rows crisprHAL 2 is at its stored out-of-fold predictions, trained on 80% of all guides by random folds (including guides from the same genes), which favours it, and the bootstrap is over genes; the lead there is larger than on the five folds. The lead is stable across five training seeds (selection and trees refitted): +0.0280 to +0.0286 (mean 0.0284) on eSpCas9, +0.0208 to +0.0228 (mean 0.0221) on WT-SpCas9.

**Metric by metric** (tuned minus crisprHAL 2, paired bootstrap over guides): ρ (above), τ, Pearson and R² are clear on both splits and the two top-end metrics of this table are tied.

| | eSpCas9 (11,898 test) | WT-SpCas9 (6,737 test) |
|---|---:|---:|
| Kendall τ | **+0.0301** [+0.0257, +0.0346] | **+0.0221** [+0.0147, +0.0297] |
| Pearson | **+0.0234** [+0.0191, +0.0278] | **+0.0173** [+0.0094, +0.0254] |
| R² | **+0.0435** [+0.0366, +0.0503] | **+0.0285** [+0.0169, +0.0397] |
| top-decile precision | +0.0034 [−0.0151, +0.0286] | +0.0193 [−0.0119, +0.0520] |
| pick percentile of the 10 best-scoring guides | −1.92 [−6.70, +3.93] | −0.44 [−10.35, +9.73] |

The pick percentile rests on ten guides (intervals of about ±10 percentile points). Precision of the top 5% is +0.019 [−0.017, +0.049] on eSpCas9 and **+0.048 [+0.003, +0.074]** on WT-SpCas9, and the mean percentile of the 50 and 100 best-scoring guides is tied on both. Objectives aimed at the top did not raise top-decile precision (three seeds, CONFIRM genes, no intervals): squared error gives 0.2912, weighting high-activity guides 0.2852 (weight 3) and 0.2890 (weight 9), and pairwise and NDCG ranking objectives reach ρ 0.5758 and 0.5119 (0.6017 at best) with lower top-end precision. Weight 9 is nominally ahead of squared error on the mean pick-10 percentile (87.2 against 85.1) and top-1% precision (0.0487 against 0.0412), a difference three seeds do not resolve. Averaging the tuned model with crisprHAL 2's shipped predictions at equal weight adds nothing on eSpCas9 (0.8099, −0.0008 [−0.0030, +0.0015]) and **+0.0085 [+0.0040, +0.0127]** on WT-SpCas9 (0.7257); the two models' residuals correlate at 0.83 and 0.81; the network adds to the blend on the WT-SpCas9 split, which has the smaller training set, and not on eSpCas9 (the two splits also differ in nuclease).

**Data efficiency** (mean ρ over the five folds, same subsets as the stored curve; crisprHAL 2 at seed 41):

| training guides | 4,000 | 8,000 | 13,880 | 20,000 | 26,853 |
|---|---:|---:|---:|---:|---:|
| crisprHAL 2 | 0.5235 | 0.6323 | 0.6733 | 0.6879 | 0.6950 |
| frozen champion | 0.6524 | 0.6814 | 0.6988 | 0.7053 | 0.7076 |
| **tuned XGBoost** | **0.6576** | **0.6915** | **0.7106** | **0.7198** | **0.7251** |

The tuned model is ahead on every fold at every size (fold-mean leads +0.134, +0.059, +0.037, +0.032, +0.030; crisprHAL 2 at its published configuration at every size, and the tuned settings were chosen on genes shared with these folds). With 8,000 guides it scores 0.6915, within 0.004 of crisprHAL 2 trained on 26,853, and by interpolation reaches that figure at about 9,100 guides, **about three times fewer labels**.

#### Before tuning, no difference from crisprHAL 2 is detected, at a fraction of the cost on a CPU

**On crisprHAL 2's own two splits, against its shipped weights, no difference is detected for the frozen champion: the 95% interval of the paired Δρ is [−0.0031, +0.0072] on eSpCas9 and [−0.0047, +0.0124] on WT-SpCas9.** On the 11,898-guide eSpCas9 test file (`final_eval.py`) its arm reproduces the published figures (ρ 0.782, Pearson 0.780; 0.7821 and 0.7801 here), so the comparison is with their model and not a re-implementation of it.

| their split (`final_eval_espcas9.csv`, `their_split_slicer.csv`) | crisprHAL 2, published ρ (shipped weights) | frozen SLICER | paired Δρ vs shipped weights [95% CI] |
|---|---:|---:|---:|
| eSpCas9 (47,591 train / 11,898 test) | 0.782 (0.7821) | 0.7842 | +0.0021 [−0.0031, +0.0072] |
| WT-SpCas9 (26,830 train / 6,737 test) | 0.695 (0.6945) | 0.6997 (re-derived 0.6985) | +0.0039 [−0.0047, +0.0124] (from the re-derived 0.6985) |

The eSpCas9 interval excludes a lead of crisprHAL 2 above 0.0031 and a lead of SLICER above 0.0072, the WT-SpCas9 interval above 0.0047 and 0.0124 (the ±0.01 margin that `final_eval_espcas9.csv` applies to eSpCas9 is not a margin declared in advance); the interval is reported in place of a verdict (Appendix A audits the dates). On eSpCas9 SLICER leads on ρ (0.7842 against 0.7821) and τ (0.5942 against 0.5920), crisprHAL 2 on Pearson (0.7801 against 0.7782), R² (0.6009 against 0.5979), top-decile precision (0.494 against 0.488) and expected pick percentile (95.5 against 94.8); **all six differences have intervals containing zero** (Appendix A). On our five folds (33,567 out-of-fold predictions per model, paired on guides, `heldout_compare.py`; per-fold figures at seed 41 in `results/folds_ladder_base_full+a_flank.csv` and `results/crisprhal_rerun.csv`) the frozen model leads in five of five folds (per-fold ρ at seed 41: mean gap +0.0105, 0.7076 against 0.6971) and, on the pooled out-of-fold predictions, by +0.0115 [+0.0074, +0.0157] against crisprHAL 2's published configuration and +0.0043 [+0.0005, +0.0080] against the best configuration found for it. The margin is smaller on the 47,591-guide eSpCas9 split than on our 26,853-guide folds, the direction the data-efficiency curves predict, but the two also differ in nuclease and screen, so this is not a measured trend; on eSpCas9 the frozen model is level with crisprHAL 2 (+0.0021, interval spanning zero).

Search effort differs by construction: SLICER's settings were searched (97–157 draws per fold on our folds, 627 in total) and crisprHAL 2 is used as released. Its published configuration won the inner search in four of five folds in each of two independent nested searches (0.6987 and 0.6989 against 0.7212 for ours, the mean of the five per-fold tuned scores; paired +0.0225 against the first search, 5/5 folds, paired t-test p < 0.001; the second search's per-fold scores are recorded in `docs/FINDINGS_LOG.md` only), the best configuration found by any search lifts its per-fold mean only from 0.6971 to 0.7051, and a comparable small search (12 draws per fold) buys our model +0.0103 (`tuning_trials12.csv`, in the main project's `results/`) and theirs nothing. `tune.DEFAULT`, the "default" arm on our side, is a configuration inherited from the starting baseline, not LightGBM's defaults, while crisprHAL 2's arm is its published configuration (Appendix B).

**The cost claims do not depend on which accuracy comparison is used, and the data-efficiency claim was measured on our five folds only:** 81× less wall-clock to fit on the same laptop and 25× against a Colab T4 (measured on 24 Nov; the earlier 150× was an estimate and CPU-seconds are not compared, because XGBoost's threads spin while they wait; see §30), about half the labelled guides to reach a given accuracy, and 125× fewer leaf values (43× counting split structure). Under equal input DNA its ranking depends on the window (next subsection).

#### Equal input DNA: their convolutions win inside their own span, ours wins past it

**Given the same DNA, the ranking depends on the window: crisprHAL 2 is better on the guide and PAM alone and inside the 378 nt it was designed for, and SLICER is better once the window reaches the 250–1,000 nt its input cannot see.** Original model, crisprHAL 2 at batch 256, means over the folds shown (Finding 2):

| DNA both models see | folds | SLICER | crisprHAL 2 | lead |
|---|---:|---:|---:|---|
| guide + PAM only | 5 | 0.5440 | 0.5556 | crisprHAL **+0.012** (5 of 5 folds) |
| their 378 nt span | 5 | 0.6724 | 0.6810 | crisprHAL **+0.009** (4 of 5 folds) |
| ±1,000 nt | 5 | **0.7078** | 0.6421 | **SLICER +0.066** (5 of 5 folds) |

Limit: crisprHAL 2 was retrained at each width with its published configuration, whose epochs, dropout, filters and GRU width were chosen at 378 nt, on five folds and one seed. The result supports "the published configuration is degraded by input wider than it was tuned for", not that convolutional-recurrent networks cannot use long-range context (Finding 2, Appendix A).

#### Cost: fitting, data, scoring and size

**SLICER is cheaper on every axis measured on a CPU: 81× less wall-clock to fit on the same laptop (measured on 24 Nov; 25× against a Colab T4), about half to a third of the labels, 125× fewer fitted values (43× counting split structure), and 15× to 25× faster scoring on one CPU thread (default model, 1 to 10,000 guides). On a T4 crisprHAL 2 costs 3.9 times SLICER's laptop path from raw contexts (two machines) and 13 times SLICER's trees' forward pass on the same VM (§30).**

| quantity (one thread unless stated) | SLICER | crisprHAL 2 | ratio |
|---|---|---|---:|
| fitted numbers | 6,338 leaf values over 400 trees (`model_size.csv`; `results/complexity.json` from an older fit has 6,348; 18,244 counting each node's column and threshold) | 790,533 trainable parameters | 125× on leaf values (43× with structure) |
| CPU time to fit, per fold (`cost_panel.csv`) | 65.8 CPU-s (frozen); on 47.6k guides 2 s frozen and 30 s tuned, 4,000 trees (`results/vs_crisprhal_espcas9.csv`) | 10,000 CPU-s, an estimate entered by hand in `sgrna/cost/cost.py` (1,266 s wall-clock × 7.9 CPU-s per wall-second) | withdrawn: the 150× was an estimate (CPU-seconds entered by hand); measured, 81× on wall-clock on the same laptop (24 Nov) |
| labelled guides to reach ρ ≥ 0.62 / 0.66 / 0.68 / 0.70 (`sample_complexity.py`) | 4,000 / 8,000 / 8,000 / 20,000 (frozen) | 8,000 / 13,880 / 20,000 / not reached on 26,853 | 2.0× / 1.7× / 2.5× at 0.62 / 0.66 / 0.68, at least 1.3× at 0.70 (resolution is the five training sizes) |
| guides to reach crisprHAL 2's 26,853-guide ρ (0.6950) | about 9,100 (tuned) | 26,853 | about 3× |
| forward pass alone, per guide, batch 100, laptop (`results/cost_slicer.csv`, `cost_crisprhal.csv`) | 0.002 ms frozen; 0.018 ms tuned | 1.32 ms | 329×–817× frozen over 1 to 10,000 guides (`results/inference_onecore_local.csv`, a separate run on the same laptop; 175×–344× on a Colab vCPU; `colab/inference_bench_onecore.py`, `results/inference_onecore.csv`) |
| **end to end, per guide, 1 / 100 / 10,000 guides** | tuned 0.72 / 0.089 / 0.093 ms | 15.29 / 1.365 / 1.214 ms | **21× / 15× / 13×** |
| **end to end, every NGG site in a 100 kb stretch** (`flank_locus.py`) | tuned 0.0361 ms; frozen 0.0223 ms | 1.214 ms | 34× (54× frozen; locus mode, not in the paper) |

Three readings of this table. (1) CPU time is a statement about the hardware, not a law, and there is no architecture-independent measure of cost; the one that is architecture-free is sample complexity, which is why the data-efficiency rows are the defensible form of the claim. (2) Forward-pass ratios omit featurisation, which the end-to-end rows include: the array featurisers `a_flank_fast.py` and `base_fast.py` give columns identical to the stored matrices on all 33,567 guides (maximum absolute difference 0.0; 0.068 ms per guide for family A and 0.019 ms for the base), and `flank_locus.py` scans a stretch once into cumulative tracks so each window statistic is a subtraction (largest relative difference 4.6 × 10⁻¹²; `EXTENSION_LOG.md` §22.1). The ratio is not an artefact of optimising one side: 95% of crisprHAL 2's 1.214 ms is its forward pass (1.157 ms), so the same trick could remove at most 4.7%. The 34× is a statement about designing guides across a locus: the precompute is paid per nucleotide, so at the curated cohort's density (one guide per 138 nt) it gives only 3–5× (`EXTENSION_LOG.md` §22.1; no stored file), and at 1 Mb the 2.95 GB feature matrix means a design run should chunk at tens of kilobases. (3) On all ten laptop cores the forward-pass factor is about 1,700 on throughput (`inference_cost.py`, `results/inference_cost.csv`), not the headline because the two models use ten cores very differently; on a T4 GPU it is 15× to 20× with input on the host and 12× to 110× with input on the card (same files), because a 400-tree depth-4 ensemble has little use for the card; cold start is 6.0 s (CPU) and 9.9 s (GPU) for crisprHAL 2 against 1.4–1.9 s for SLICER (`results/inference_colab.csv`).

**One tuned model is worth its cost; averaging or mixing models is not.** The tuned model has ten times the trees of the champion (0.018 ms against 0.002 ms per guide) and carries nearly all the gain. Averaging five XGBoost models adds +0.003 on the held-out genes and nothing on eSpCas9 (0.8104 against 0.8107); mixing nine models from three libraries adds +0.0014 [+0.0005, +0.0025] over that average and is lower than the single model on eSpCas9 (0.8095), for 24,800 trees across three libraries (five XGBoost, two LightGBM, two CatBoost).

**Size is not capacity.** The 125× counts leaf values only (43× with split structure) for the frozen model, whose accuracy is indistinguishable from crisprHAL 2's on the two splits; a parameter count is a loose proxy for capacity in both model classes, SLICER's count omits the choice of which 300 of 6,517 columns to offer, and depth 4 was inherited rather than chosen (Appendix B). It supports only that no part of the accuracy requires a model of the size the field has been fitting.

#### Other comparisons and boundary conditions

**Against Noshay et al.'s iRF on the released 13,880-row matrix SLICER leads by +0.128 ρ (0.6068 against 0.4785; 13,880 guides, `cut_score`, random-row five-fold CV; SLICER the mean of 5 seeds × 5 folds, `ablation_main.csv`, the iRF seed 41, `noshay_replication.csv`)**, of which +0.049 is the model class (champion XGBoost on the identical 6,232 columns reaches 0.5273) and +0.080 is family A. The iRF is our reimplementation (`noshay_replicate.py`) and approaches their published figures on a third of their rows (Pearson 0.488 and R² 0.234 against 0.502 and 0.249 on 40,468 guides). On crisprHAL 2's cohort the same comparison is +0.023 to +0.029 for the final model (indistinguishable before tuning), so a paper quoting only the Noshay margin overstates the contribution.

**Parity depends on the training rows.** With the architecture fixed and only the training data varied, scored on crisprHAL 2's WT-SpCas9 test partition (6,737 of its 6,749 guides locate in our feature build), SLICER reaches 0.6537 on Noshay's 1,222-gene panel with Noshay's `cut_score` (n = 6,860), 0.6600 on the same rows with the curated label, 0.6683 on count-matched random curated rows, and 0.6985 on all 26,830 curated rows (`data_ablation.py`, `results/data_ablation.csv`). Of the +0.0448 between the first and last, the label is worth +0.0064, which rows +0.0082 and how many rows +0.0302. Family A is worth +0.1554 on the same partition at full data, more than the whole move between cohorts. The Noshay-panel row is 0.041 below crisprHAL 2's published 0.695, but crisprHAL 2 was trained on all curated rows throughout, so the comparison is SLICER with less data against crisprHAL 2 with all of it and does not show how crisprHAL 2 would fare on Noshay's rows.

**On a different assay, crisprHAL 2 transfers better.** Zero-shot on Cui's CRISPRi screen, |ρ| is 0.0867 for crisprHAL 2 against 0.0609 for SLICER, a paired difference of −0.0258 [−0.0289, −0.0227] for SLICER (`cui_final_exam.py`); this is a boundary of the claim and not a like-for-like comparison (assay, library and laboratory all differ), both values are small against the 0.458 that SLICER's base columns reach when trained on that screen (gene-grouped CV, `cui_screen.csv`; nothing was fitted or selected on the screen for either model, though family A had been evaluated against its labels earlier, with a null result), and a plausible reason is that transferable dCas9 signal lies in guide-sequence effects, which is their input, while SLICER's advantage is flanking DNA (a hypothesis; the CRISPRi screens are Appendix B5).

**Structure-aware inputs do not change the comparison.** Adding tensor-valued 3D descriptors of the flank (three table sources, up to 1,887 extra columns) changes the champion's ρ by −0.0008 to +0.0006 on the curated cohort (gene-grouped CV, 30,203 genic guides) and by +0.0022 on crisprHAL's eSpCas9 screen, where a flank shuffled to destroy geometry gives +0.0016; the descriptors are functions of window composition and of the first ten or so base pairs, which family A already holds (Appendix B4). Not tested: the same descriptors restricted to crisprHAL 2's 378 nt span, where its convolutions lead.

### What other predictors read, and how they are validated

**No surveyed predictor reads flanking DNA beyond 406 nt; SLICER reads ±1,000 nt, and that reach is the one large effect.** The flank
features add **+0.164 ρ** on the curated cohort under random-row five-by-five CV (base 0.545 → 0.708; +0.157 with whole genes held out, +0.156 on sealed genes) and, on crisprHAL 2's WT-SpCas9 test partition with training on its 26,830 training rows, **+0.155** (base-only 0.543 → 0.698):
the distance between a model that does not match the state of the art and one that does. On the decision scale this moves the
measured expected pick of ten candidates from the 69.6th to the 75.9th percentile (`headtohead.csv`, curated cohort, seed 41, five random-row folds).

| predictor | organism / assay | input beyond the 20-mer | model | split | reported performance |
|---|---|---|---|---|---|
| Noshay et al. 2023 (QCT + iRF) | *E. coli* and human, cutting | none beyond 17 of the 6,232 columns (two annotation distances and 15 undecoded columns); the rest are functions of the guide (Finding 1) | iterative random forest | random rows | Pearson 0.502, R² 0.249 (their Table 1: 40,468 guides, 32,374 train; the released matrix holds 13,880 rows) |
| crisprHAL (Ham et al. 2023) | *E. coli*, SpCas9 / TevSpCas9 | **28 nt**: 20 guide, PAM and 5 downstream, none upstream (the 37-nt input, 3 upstream, 20 guide, 14 downstream, belongs to the 2026 crisprHAL Tev of Browne et al.) | CNN–BiGRU, transfer learning | 5-fold CV on their own TevSpCas9 data (n = 279); held-out unique Guo SpCas9 sequences | ρ 0.630 (TevSpCas9, 5-fold CV); 0.682 (held-out Guo SpCas9 sequences) |
| crisprHAL 2 (Browne et al. 2026) | *E. coli*, SpCas9 and eSpCas9 | **378 nt** (SpCas9), 406 nt (eSpCas9), set by their own length search | CNN–BiGRU | their own two splits | ρ 0.695 (SpCas9), 0.782 (eSpCas9) |
| **SLICER, frozen champion** | *E. coli*, cutting | **±1,000 nt** windowed composition | XGBoost, 300 columns | **gene-grouped**, 3 seeds × 5 folds | ρ 0.7151 CV; 0.6997 / 0.7842 on their splits |
| **SLICER, tuned (post-freeze)** | same | same | XGBoost, 450 columns, 4,000 trees | gene-grouped, 3 seeds × 5 folds; DEV/CONFIRM genes | ρ 0.7314 CV; 0.7172 / 0.8107 on their splits |
| Yu et al. 2024 | bacteria, **CRISPRi** depletion | gene-level covariates: expression, transcription-unit structure, gene GC, gene length, distance to operon start | mixed-effect random forest (129 guide features fixed, 9 gene features random, 301 genes as clusters) | **gene-wise** CV, train/test Hamming distances reported | mixed-effect model, gene-wise CV on three screens: median within-gene ρ 0.396. (The ρ ≈ 0.21 sequence only → 0.25 with distance and thermodynamics → **0.66** with gene features belongs to their auto-sklearn depletion models under guide-wise CV on one screen, 1,951 guides in 293 essential genes.) |
| Orro et al. 2026 | human, **base-editing** fitness screen (the Hanna et al. 2021 library: ~11,000 sgRNAs over 47 genes, five cell lines, one model each) | a short sequence context (**±15 nt** in the preprint), GC content, melting temperature, DepMap essentiality, mutation type, external editing-efficiency predictors | gradient-boosted trees (CatBoost best) among six classifiers | leave-one-gene-out in the published version (the preprint's main protocol was random 5-fold CV × 10) | CatBoost AUC-ROC 0.914–0.943 across the five cell lines (87–94% across the classifiers compared); classification, not commensurable with ρ |
| CRISPRon (Xiang et al. 2021) | human, Cas9 knockout | a 30-nt window (protospacer, PAM and neighbouring bases) plus a computed gRNA–DNA binding energy (ΔG_B); no experimental track | CNN | internal + external test sets | ρ 0.80 internal, ≈0.46–0.68 external |

Other flank readers are shorter still (Yu et al. 4 nt upstream and 3 beyond the PAM; CRISPOR 50 bp each side, Haeussler et al. 2016), and the field's other
routes outside the protospacer are experimental (epigenetic tracks, as in DeepCRISPR, CNN-SVR and C-RNNCrispr; Konstantakos et al. 2022) or gene-level (expression, operon structure, essentiality).
No surveyed predictor builds compositional descriptors of flanking DNA in windows of hundreds to a thousand nucleotides and measures
their increment, and this one needs only a reference genome.

*Independent agreement.* (i) Gene-held-out evaluation is also reported by Yu et al. (2024) for their guide-efficiency model and by Orro et al. (2026) (leave-one-gene-out is the main protocol of the published version; the preprint's was random 5-fold CV), and Yu et al.'s
separation of guide and gene effects is the within-gene / between-gene decomposition used here (Finding 3's +0.085, asked entirely
inside genes). (ii) Orro et al.'s SHAP importance for the A375 model rises toward the 3′ end of the guide in both versions of their paper (published CatBoost model: largest at positions 16 (1.119), 20 (0.750) and 14 (0.617); the preprint's LightGBM model peaked at position 20, 0.328); here the eight PAM-proximal positions carry **63% / 62%** of the gain (frozen / tuned) and positions 1–12
only 10% / 17% (`EXTENSION_LOG.md` §14), on a different organism, nuclease, label and importance measure.

*Reference figures.* crisprHAL 2's *C. rodentium* TevSpCas9 model (a different organism and nuclease) reports 0.754 by five-fold CV and 0.746 on its held-out split,
so the held-out *E. coli* values 0.695 and 0.782 are the right figures to compare against. The shipped weights reproduce ρ 0.6945 against the published 0.695. Their paper's SpCas9 split (33,495 curated guides: 26,795 train / 6,700 test) differs from the released files (26,994 / 6,749 rows) and from the guides located here (26,830 / 6,737); the differences are unexplained.

### What is new, and what is not

**The paper this work supports is about what determines cutting and how much of it is knowable, not a better predictor.** The
modelling is incremental; the measurement is not.

*Not new.* (a) Boosted trees over engineered features match a deep model: BoostMEC (Zarate et al. 2022) is LightGBM over sgRNA sequence
features, benchmarked against 10 models on 13 external datasets, with far more external validation than is available here. What is
specific to this work is why the tree wins: the representation reaches DNA the network's input does not, a claim about features and
range, not the learner. Finding 4 is the sharpest internal evidence: the tuning gain collapses from +0.016 to +0.003 without the flank
columns (gene-grouped, 3 seeds × 5 folds). (b) Gene-held-out evaluation (Yu et al. 2024; Orro et al. 2026, per its published abstract). (c) That flanking sequence matters at all: crisprHAL 2 is built on
±190 nt and searched the length itself.

*New, as three findings (statements about what determines cutting) and five warnings or methods (things wrong, or measured wrongly, in
the field or in earlier versions of this work).*

1. **Finding 1: a published feature matrix is a lossless re-encoding of the guide, so no deterministic function of the guide can add
   information to it.** No paper found makes the arithmetic argument; it settled several imported biophysical families without running them.
2. **Finding 2, characterisation not existence: the flank effect is compositional, not a motif.** 8% survives a 1 kb shift and none a
   10 kb shift; it does not depend on window edges (+0.1634 against +0.1638 at four widths sharing no value with the originals; curated, random-row); its
   reach is 250–1,000 nt against a longest published input of 406 nt.
3. **Finding 7: the effect is GC specifically** (purine is flat at every window and side), **side-symmetric** to within 0.026, with
   **a plateau from 150 to 500 nt (highest at 200 nt) and no structure below ~250 nt** (four columns carry 76% of it; 200 columns are no better than 16 or 32, the differences of 0.0014 and 0.003 lying inside the fold SD of 0.006–0.008), and
   **coupled to the PAM-proximal seed** (largest pairwise interaction: 250 nt flank GC against position 20). The guide's own
   position × base preferences are **uncorrelated between *E. coli* and human (ρ = −0.052)**, the strongest inverting: G immediately
   5′ of the PAM is the most disfavoured base in *E. coli* and a favoured one in human, which is consistent with the failure of
   mammalian-trained predictors here and is a design rule applicable today.
4. **The corrected ceiling** (next subsection; Finding 6): the field's ceiling of ~0.7 is mis-derived and the attainable value is 0.91–0.93.
5. **The supercoiling deconstruction.** A ChIP track worth +0.139 on the curated cohort (random-row CV) was read density, and read density was gene dosage, recoverable from
   distance to *oriC* with no experimental data (published matrix, 100 kb blocks: +0.0112 for four *oriC*-distance columns against +0.0111 for the ChIP track and +0.0087 for the untagged control): a warning for every coverage-derived feature, including the chromatin tracks of human predictors such as DeepCRISPR.
6. **The duplicate-guide result.** 49% of a widely used published human matrix's rows repeat a guide; under folds that keep them together
   the best-looking configuration is 0.086 worse than the default one (0.2515 against 0.3377; `EXTENSION_LOG.md` §16.2).
7. **The sealed-gene measurement.** A +0.0087 ρ drop in family A's gain between pool cross-validation and sealed genes (mean ± 1.96 standard errors over fifteen draws: [+0.0020, +0.0153]; the most the automatable choices could contribute, not reaching the cohort choice, family A's design or the window set, and nearly all of it persists with whole-gene folds in the pool),
   and a sealed genome on which the tuned settings' benefit survives (+0.0172). No other guide-prediction paper found measures its own
   selection bias.
8. **The strand warning (Appendix B5, a boundary experiment).** A feature family worth +0.27 in a dCas9 screen is worth +0.007 to
   +0.025 once the assay's own strand rule is supplied: a family containing a lossy copy of an assay's dominant covariate looks
   powerful. the nearest literature does not settle it (Orro et al. (2026) model the gene's orientation, not the guide's strand relative to the gene, and
   the preprint's gene-orientation feature was dropped from the published version), so any subgroup result from a pooled screen should control strand first. Whether the effect needs a break was
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

**Consequence.** Under the field's ceiling a bacterial model at 0.695–0.782 looks saturated; under the corrected one, models at ρ ≈ 0.70 sit at **75–77%** of attainable, and the best single tuned model at 0.7314 (the five-model average reaches 0.7334) sits at **77.3%** of
0.946, leaving **0.215 ρ** of real, repeatable signal unexplained (Part 3). The field has roughly **three times more headroom than it
believes** (0.21–0.23 for a model at ρ 0.70, against 0.07 below the highest replicate figure, 0.77), and the 0.4 success threshold is far too lenient.

### Scope and calibration: what is not claimed

**The lead over crisprHAL 2 is +0.023 to +0.029 ρ on two splits of one organism's screens, with both models as shipped; it supports a
claim about reach, not about one architecture being better.** It is measured, audited and confirmed on untouched genes. On the four
external human test sets of Xiang et al. (2021), predictors' ρ spans 0.46–0.68 (CRISPRon), 0.44–0.52 (DeepSpCas9) and 0.42–0.46 (DeepHF): ranges far wider
than this lead, and overlapping between predictors.

Absolute ρ is not comparable across assays: CRISPRon reports 0.80 internally and 0.46–0.68 externally (Xiang et al. 2021), DeepSpCas9 ≈ 0.44–0.52 and DeepHF ≈ 0.42–0.46 on the same external sets, and DeepHF 0.867 within a cell line and 0.433 on a new one (as tabulated by Konstantakos et al. 2022). A 0.4 and a 0.8 can
describe equally good models on differently noisy data, which is why the ceiling above is the comparison that matters.

- **Specificity.** Six of `h_offtarget`'s thirteen columns are constant zero because the cohort keeps only uniquely-targeting guides, so
  its +0.0011 (curated cohort, random-row CV; +0.0003 gene-grouped) is a filter, not evidence about off-target descriptors (Part 3).
- **Nickase editors (base and prime editing).** Untested: they bind and nick rather than cut. The nearest literature, Orro et al. (2026),
  fits a binarised human fitness label with a short sequence context (±15 nt in the preprint), so the kilobase window is untested there. Gene essentiality
  carries marginal weight in their published models (DepMap, SHAP 0.388, third among the non-sequence features in A375) and is a null here (+0.00004), plausibly because the labels measure different things (untested):
  a fitness effect against survival of a break that is lethal whichever gene carried it.
- **Human.** The *E. coli* flank result should not be assumed to transfer: the one cross-kingdom quantity tested inverted (GC against
  activity, −0.147 on the curated cohort [−0.201 on the published 13,880-row matrix] against +0.018 in the decoded human matrix), and the seed preferences are uncorrelated (ρ = −0.052). Cross-kingdom tests are in Part 3.
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

**Permutation floor.** *Rules out:* gain that a selector and a boosted tree manufacture from 300 columns of correctly shaped noise. *Design:* on the published matrix, each family with a gain except `i_shape` (which has no permuted arm) was re-run with its feature rows shuffled against the labels (marginals and capacity kept; `permutation_floors.csv`, `ablation_main.csv`, `ablation_grouped.csv`, `ablation_jper.csv`). *Result:* family A's floor is **0.5264 against a 0.5273 baseline** (−0.0009, i.e. zero; 5 seeds × 5 folds); the floors of the other families run from −0.0001 to +0.0011 (`h_offtarget` is indistinguishable from its floor, `f_methylation` marginal), and `j_periodicity`'s shuffled arm scores higher than its real one (0.5273 against 0.5264). *Without it:* family A's +0.080 would rest on the assumption that a 6,580-column search cannot overfit a 13,880-row matrix.

**Texture-preserving null for family A** (`shiftnull.py`). *Rules out:* that family A learned the texture of bacterial DNA rather than the local DNA at the cut site; row shuffling cannot, since it also destroys genomic autocorrelation. *Design:* each guide receives the flank track read from a position offset by a circular chromosome shift, which keeps every column's marginal, smoothness and long-range autocorrelation and breaks only the alignment (correlation with the aligned track 1.00 / 0.41 / 0.05 / −0.03 at 0 / 1 kb / 100 kb / 2 Mb on `eng.flank.comp.dn500.gc`; published matrix). The shift is swept, since a 1 kb shift is still correlated with the aligned columns in proportion to their window (median 0.03 for the near-cut columns, 0.11 at 50 nt, 0.28 at 250 nt, 0.40 at 500 nt, 0.58 at 1,000 nt; 0.41 is the `dn500.gc` column alone) and at 10 kb no window's median exceeds 0.07, while a single 2 Mb shift shows only that a distant locus is uninformative; Δρ against shift distance is the effect's autocorrelation length. Grouped CV (100 kb blocks, 3 seeds × 5 folds) throughout, on the published matrix. *Result:* Finding 2. *Without it:* the result would also fit a model that has learned the texture of the genome.

**Negative control from the source experiment** (the untagged-GapR control arm of the GapR-seq experiment, GSE152880, which has no FLAG-tagged GapR to pull down; one family only). *Rules out:* that `d_supercoiling`'s +0.044 (published matrix, 100 kb blocks) measures supercoiling. *Design:* the same assay with the biology removed; a feature that predicts equally well from the control track measures the assay. *Result:* the signal is DNA read density, mostly gene dosage. Four columns computed from the genome coordinate alone (distance to *oriC*, that distance as a fraction of a replichore, a 2<sup>−2f</sup> copy-number proxy, which replichore) are worth **+0.0112** on the published cohort under 100 kb blocking, against **+0.0111** for the GapR ChIP track alone, **+0.0087** for the untagged control track alone and **+0.0053** for the supercoiling-specific log-ratio part (`dissection_supercoiling.csv`, rows `z_dosage`, `z_signal_only`, `z_control_only`, `z_sc_enrichment`; 3 seeds × 5 folds). Origin-proximal loci are over-represented in any DNA-derived track of multi-forked, exponentially growing *E. coli*, which is why the untagged control predicts almost as well as the ChIP (78% of its gain); dosage is also a candidate mechanism for the phenotype (more target copies to cut). *Limit:* small quantities (+0.011 against family A's +0.077 under the same 100 kb blocking on the published matrix); the claim concerns the source of `d_supercoiling`'s signal, not the Cas9 effect.

### Splits: is the gain prediction or address memorisation?

**Grouped cross-validation** (`arc5`, 100 kb blocks, or whole genes held out). *Rules out:* locus recognition: guides in one gene share a locus and positional columns identify it, so plain KFold rewards learning that guides around here score about this much (sequence features are unaffected). *Design:* hold out whole groups. *Calibration:* families that are functions of the guide lose 6% in aggregate under grouping, the cost of the harder partition rather than memorisation; `a_flank`, `d_supercoiling` and `i_shape` lose nothing beyond it, `e_nucleoid` and `g_transcription` lose 6 and 15 points more (Appendix B3). *Without it:* positional gains would be indistinguishable from locus recognition, whose budget is the between-gene share of label variance, **16% on the published matrix, 29% on the curated one** (`cohort_groups.py`; Appendix B). *Re-run on the curated cohort:* every curated family number had been measured under random KFold on a cohort whose between-gene share nearly doubled; under gene grouping the six families with gains above +0.05 keep 79–96% of them (the guide-only families 94% in aggregate, `j_periodicity` 64% and `h_offtarget` 23% of gains of +0.011 and +0.001; Appendix B3), so the inflation is not memorisation.

### Matching: is the comparison fair?

| control | rules out; design | result | believed without it |
|---|---|---|---|
| Matched-span head-to-head against crisprHAL 2 | the lead attributed to architecture when the models read different amounts of DNA; crisprHAL also run at our window widths, SLICER at theirs | at their span our windowed composition loses by 0.025 (our five folds, seed 41, crisprHAL 2 at its published batch; +0.009 over the five folds retrained at batch 256 in the Part 5 table); at ±1,000 nt we lead by 0.066 (five folds and one seed, original model) | "we out-encode a tuned deep model", which is false; the defensible claim is access to DNA they never see |
| Distance- and GC-matched orientation comparison, bootstrap over 47 genomic blocks | a cohort confound (opposite-orientation guides sit at different distances from the gene and in different GC); blocks rather than rows, since nearby guides are not independent | opposite-minus-same asymmetry against the nearest gene is 0.026 [−0.006, 0.058] unmatched (a first unweighted pass gave 0.047 against 0.005 and is withdrawn) and **0.007 [−0.024, 0.038]**, P = 0.34, after matching; against a curated **TSS** it is 0.051 [0.020, 0.082] unmatched and **0.025 [−0.003, 0.053]**, one-sided P = 0.04, after the same matching: the TSS orientation survives matching attenuated, with an interval that includes zero (Jacky's `intergenic_orientation_controls_delta_ci.csv`) | the unmatched nearest-gene difference read as an orientation effect |
| Conditional GC test within matched target-GC strata | flank GC acting as target GC seen through a correlated variable; bin on target GC and measure inside each bin | curated cohort, flank GC the mean of ±500 nt: **+0.176 marginal → +0.221 partial**, positive in all five strata: a suppressor, the opposite of a confound | the GC account would rest on a marginal correlation |

**What the comparison on their splits holds equal.** Rows, labels and splits are equal (theirs, untouched by us). Each model is used as released: crisprHAL 2 with its published weights, SLICER with settings chosen on the DEV genes of the WT-SpCas9 screen. No eSpCas9 label entered that choice, but all 33,567 WT guides are in the eSpCas9 screen and 57% of the eSpCas9 test guides (6,778 of 11,898, by protospacer) are among them, so their WT labels informed the settings (never a weight); for the WT-SpCas9 split the CONFIRM genes are the clean check (Part 5). Input DNA is **unequal, in our favour**: family A reads ±1,000 nt and their eSpCas9 input is 406 nt, so each model is compared as released, not with equal information, and equalising the DNA makes the ranking depend on width (Finding 2). Our features were designed on *E. coli* WT cutting data, so design knowledge leaks into the eSpCas9 arm in a way no split removes. For the frozen model the SpCas9 margin compares our 0.6997 with their *published* 0.695, while our re-run of their model gives 0.6940 (`crisprhal_rerun.csv`, arm `their_split`, their 26,994/6,749-row files, seed 41): +0.006 measured our way, +0.005 against their number. Equal DNA is not equal opportunity to use it: at 2,023 nt the published configuration's flattened layer is ~5× larger with regularisation and training budget chosen at 378 nt, so Finding 2 claims only that the published configuration is degraded by input wider than it was tuned for, not that the architecture cannot use long-range context (a re-tune at each width, `colab/crisprhal_tune_by_width.py`, is not yet run). The experiment extends the curve of Browne et al. (2026), whose per-dataset length search settled on 378 nt (Part 5).

### Instruments: does the measurement measure what its name says?

| control | rules out; design | result | believed without it |
|---|---|---|---|
| Selector control: 300 columns against all | an arm gaining only because it was offered more candidates; selection switched off | `d_mechanics` **+0.0988 selecting 300, +0.0993 using all 6,199** (curated cohort, `selector_control.csv`) | gain attributed to selection |
| Faithfulness null control (sensitivity-*n*) | an attribution scored as faithful because any attribution correlates with the prediction lost when columns are replaced; the same SHAP values are assigned to permuted columns | SHAP on permuted columns correlates at −0.045 to +0.038 with the prediction lost, against +0.93 to +0.97 for SHAP on the real columns (one fit, 100 held-out guides; `interp_sensitivity_n.csv`, Finding 5) | attribution read as measurement |
| Pairing alignment, held-out comparison | a mis-rotated join, which gives a tight, plausible, meaningless interval (the failure shape of the three execution guards in `docs/qna_archive.md`); crisprHAL's out-of-fold predictions have no guide key, so the code rebuilds `KFold(5, shuffle=True, random_state=41)` in fold order and asserts that the two label vectors match guide for guide | labels match guide for guide | a precise interval about the wrong pairs |

**Reconstructibility probe** (each of a family's columns predicted out of fold from the base's 364 mono- and dinucleotide indicators). *Rules out:* the claim that a family adds no new DNA, which reading the builder checks only for inputs someone read: a column the base predicts at R² = 1.0000 adds no information, one at R² ≈ 0 reads something the base lacks. *Result:* R² from the base is **1.0000** for the 9 ΔG/ΔH/ΔS sums over the protospacer and its windows, 0.997–0.9995 for the 4 melting temperatures, 0.20–0.98 for the 9 stacking and slide summaries and **−0.006 to +0.042** for the 8 flank thermodynamics (±25, ±100 nt); those eight carry +0.097 of `d_mechanics`' +0.099 (Finding 1). The guide-only / flank-only decomposition (`no_information.py`) uses the same base, rows, folds and selection budget: +0.011 for the 22 protospacer columns, +0.097 for the 8 flank columns. *Without it:* `d_mechanics` would remain the headline evidence that representation beats information.

**Equivalence reported as a bound, not against a margin.** *Rules out:* arguing equivalence from a non-significant p-value, or from a margin whose declaration cannot be audited. *Design:* no margin is claimed as pre-declared, since both were first committed after their results existed (427 columns, 0.005 ρ: result `136755f`, margin `0f66a2e`, 7 days later; parity with crisprHAL 2, ±0.01 ρ: result `6e2f32d`, margin `872e2d4`, 54 minutes later); each claim is reported with the smallest margin the data support, the larger end of its interval. Paired on the same 11,898 eSpCas9 guides, 2,000 resamples, frozen SLICER minus crisprHAL 2's shipped weights (ρ from `final_eval_espcas9.csv`, the other metrics from `final_eval_metric_bootstrap.csv`, a second bootstrap whose ρ interval is [−0.0028, +0.0071]), with the smallest margin at which equivalence holds in brackets: Spearman ρ +0.0021 [−0.0031, +0.0072] (**0.0072**); Kendall τ +0.0023 [−0.0024, +0.0073] (0.0073); Pearson r −0.0019 [−0.0066, +0.0030] (0.0066); R² −0.0030 [−0.0103, +0.0041] (**0.0103**); top-decile precision −0.0059 [−0.0244, +0.0202]; expected pick percentile −0.70 [−17.91, +3.98]. The 427 columns against 6,232 give [−0.0008, +0.0054], equivalent at any margin ≥ 0.0054. On eSpCas9 the rank metrics lie within ±0.0073, inside a ±0.01 margin, but **R² does not at ±0.01**; it needs 0.0103. On crisprHAL 2's WT-SpCas9 split (6,737 test guides, `vs_crisprhal_wt.csv`) the ρ difference is +0.0039 [−0.0047, +0.0124], wider than ±0.01 (smallest margin 0.0124), so there the data show no resolved difference and do not establish equivalence. No metric resolves on either split, so "better at ranking against better at picking" cannot be told from sampling noise, and a top-10 statistic should not be headlined (its interval is 22 percentile points wide on eSpCas9).

**Attenuation ceiling (Spearman–Brown on replicate statistics).** *Rules out:* reading "out of material" as a claim about the model: ρ cannot exceed the label's reliability, so the ceiling must come from the measurement. *Design:* two Cas9 replicates and two controls give two independent estimates of one quantity; their agreement is the reliability of one, and Spearman–Brown converts it to that of the released mean of two. Two checks make it an estimate of *their* label: their cohort reproduces under their stated cutoff (33,871 against 33,567), and a plain log-ratio reproduces their score at |ρ| 0.957. *Result:* **0.946 [0.944, 0.947]** for the label trained on (Finding 6; first pairing, 1-1 against 2-2, a plain log-ratio of the replicates; the other pairing gives 0.945). The released score is an ALDEx2 posterior mean and the direction of the resulting bias is not established: its shrinkage was expected to raise reliability, but a Dirichlet–CLR re-implementation of that estimator (|ρ| 0.993 with the released score) gives replicate agreement 0.798 and a ceiling of 0.942, so the ceiling is 0.942–0.946 by estimator and 0.946 is not shown to be a lower bound. *Without it:* the curated arms would keep quoting 0.937, derived for a different label by a different route (Appendix B).

### Inventory of nulls

Five nulls (several with sub-arms), one small real gain and one robustness control that passed. For this phenotype the lever is what DNA the model is shown and nothing about the estimator repays attention, which licenses the cost claim in Part 5.

| what was tried | result | why |
|---|---|---|
| model class (16 classes on the published matrix, then 8 on the curated base, both with family A) | spans 0.116 ρ and 0.108 ρ | every class reads the same columns; the gain lies in the representation |
| the selector: budget (300 against every column), library, gain against SHAP, redundancy penalty | ±0.0005 for 300 columns against every column (curated cohort, with and without family A); on the published matrix 600 and 1,200 columns move the ρ with family A by +0.0011 and +0.0014 and family A's gain by +0.0013 and +0.0036 (the base arm falls at 1,200; 3 seeds × 5 folds); library 0.0008 at 3× the speed; SHAP +0.0007; a redundancy-penalised rule **−0.0015** (Appendix B) | selection binds by at most a few thousandths (selector control above); the redundant columns here are the informative ones, so penalising them leaves a more binary, less readable set |
| descriptions beyond what the model already has: guide tetramers, flank DNA shape, helical periodicity and GC skew | tetramers −0.0018 on top of trimers (curated, no selection; −0.0028 on the published matrix, paired by fold); shape +0.0009 on top of family A (curated cohort) and periodicity/skew +0.0004 (published matrix) | 4,343 tetramer indicators are too sparse to carry anything; composition already holds the shape (Appendix B4); the signal is not helical phasing or curvature |
| explicit interaction terms (family G's rationale) | +0.0004 over their parents (curated cohort) | Clarke et al.'s (2018) transcription×strand mechanism does not reproduce |
| per-guide label reliability as a training weight | −0.0050 (curated, 3 seeds × 5 folds), four fifths of it reproduced by permuting the weights (Appendix B) | non-uniform weights spend effective sample size whatever they mean |
| *small real gain:* 627-draw nested hyperparameter search | **+0.013** (curated cohort, seed 41, five folds; nested inside each training fold, so leak-free; the +0.0087 sealed-gene drop of Part 4 arises with no search and is not part of it); a later joint search judged on unseen genes gives +0.0143 [+0.0111, +0.0174] (Appendix B) | capacity, not policy |
| *control that passed:* family A rebuilt at 75/300/600/1,200 nt against 50/250/500/1,000 | +0.1634 against +0.1638 (curated, 5 seeds × 5 folds) | the window set moves the gain by 0.0005, inside the ±0.005 declared in advance |

## Appendix B — The methods and controls behind the findings

### Search effort in the head-to-head with crisprHAL 2

The final model is compared with crisprHAL 2 as released, and the search effort differs. Our side used 97–157 draws per fold (627 in total). crisprHAL 2's published configuration was given one nested search of 33 fits (28 draws, about 3.0 GPU-hours at the measured 326 s per fit on a T4; patience stopped every fold after 5–8 draws; SLICER's 627 draws took 2.4 hours on the laptop, crisprHAL 2's earlier 10-draw search 7.3 CPU-hours) and won the inner search in four of five folds: 0.6987 against its published 0.6971 (a repeat on a second Colab session 0.6989, its fold scores not stored in `results/`; the best configuration found by any search 0.7051; Part 5). Pre-specified brackets were ≥ ~0.716 (margin mostly a tuning artefact), 0.704–0.710 (margin survives at +0.005 to +0.010) and above 0.7213 (we lose on our own folds); observed 0.6987, below all three; draw 0, the published configuration, was the positive control and landed inside the 0.686–0.702 band `crisprhal_rerun.csv` predicts (outer-fold scores 0.689–0.701 in the four folds where it won). The published configuration fits one fold in 5.48 min on a free Colab T4 against 17.0 min locally, a measured **3.1×** (`crisprhal_wide_window.csv`, `crisprhal_rerun.csv`). On crisprHAL 2's own two splits the comparison is against released weights and our harness reproduces their numbers to three decimals (ρ 0.782 → 0.7821 on eSpCas9), so the +0.005 and +0.002 margins are against whatever search produced the released weights.

*Open (method level):* whether a search matched to ours (627 draws at 5.4 GPU-minutes each, about 57 GPU-hours) would change a claim about the architecture, as opposed to the released model, was not run; the as-released comparison does not depend on it (Part 4 lists the retune on their own splits).

### Hyperparameters are a hundredth-scale knob, and the one gain the sweep found is capacity, not policy

Six of the champion's `XGBRegressor` knobs (`max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_lambda`, `min_child_weight`) had been varied by a search and **eight others had never moved** (`gamma`, `reg_alpha`, `colsample_bylevel`, `colsample_bynode`, `max_bin`, `grow_policy` with `max_leaves`, `max_delta_step`, `num_parallel_tree`); the 627-draw search ran on LightGBM (`tune.py`) and the only earlier XGBoost search was the 8-draw one inside the sealed holdout. The eight were swept one at a time (24 settings besides the champion, 16 of them non-default; curated base with family A, selection cached per fold, seed 41, five folds; `xgb_knobs.py`, `results/xgb_knobs.csv`) against the champion's 0.7084: `grow_policy` = `lossguide` with `max_leaves` 63 **+0.0069**, `num_parallel_tree` 4 +0.0025, three knobs at most +0.0004, column subsampling −0.0004 to −0.0014, `max_delta_step` **−0.0080**. A sweep of 16 non-default settings reports a maximum, and the sealed-gene drop is **+0.0087**, so the winner was re-run on three fold partitions (seeds 41–43; seed 41 is the sweep's own partition; fifteen paired fold comparisons per setting, each ahead on 15/15 folds; `xgb_knobs.py --verify`, `results/xgb_knobs_verify.csv`): `lossguide`, 63 leaves +0.0067 [+0.0054, +0.0080]; **`depthwise`, `max_depth=6` +0.0063 [+0.0053, +0.0073]**; `num_parallel_tree=4` +0.0024 [+0.0019, +0.0029]. `lossguide` does nothing that four times the leaves does not do under the original policy: `max_depth=4` is too shallow for this cohort by about six thousandths of ρ. It is not adopted into the frozen champion (Part 4), since +0.0063 is below the +0.0087 sealed-gene drop on these folds, which is what a sealed holdout exists to remove.

*Scale and limit.* The whole measured span of tuning is **+0.0131** (a nested search, leak-free by construction; the **+0.0087** sealed-gene drop of Part 4 arises with no search at all and is not part of it), against **+0.164** for one feature family (curated cohort; 0.108 ρ across model classes, ±0.0005 for 300 columns against all), on a problem whose open variance is 0.22 ρ. On the human matrix the larger gains (+0.10 for the boosting defaults: depth 6, learning rate 0.3, 1,000 trees, no subsampling; +0.0585 for `max_depth` 4 → 6) came from row-wise folds in which 49% of rows carry a duplicate guide; under twin-grouped folds depth is worth **−0.0062** and the champion is the best of six arms (`EXTENSION_LOG.md` §16.2). Tuning buys insurance against a change of signal shape, not a better number on a signal already fitted.

### A joint search judged on unseen genes finds +0.0143 [+0.0111, +0.0174], and the optimum is slower, not deeper

Genes were split once into a tuning part (70%) and a confirmation part (30%, 1,204 genes, never used to choose anything). Two hundred random configurations were scored on gene-grouped folds of the tuning genes; the best twelve were re-scored on three fresh partitions; the survivors were trained on all tuning genes and scored on the confirmation genes and on crisprHAL's eSpCas9 screen (`EXTENSION_LOG.md` §11). The champion (depth 4, 400 trees, lr 0.05, 300 columns) scores 0.7073 on the confirmation genes, 0.7840 on eSpCas9 and 0.7151 on the whole cohort (gene-grouped 5 folds × 3 seeds on the 30,203 genic guides, selection inside folds; the tuned settings were chosen on 70% of those genes, so this compares with earlier tables and is not an unbiased gain). Depth 6 gains +0.0053 [+0.0027, +0.0083], +0.0174 [+0.0165, +0.0183] and +0.0072 [+0.0062, +0.0082] on those three; the best tuned configuration (depth 5, lr 0.013, 4,000 trees, 450 columns, subsample 0.51) gains **+0.0143 [+0.0111, +0.0174]**, **+0.0255 [+0.0246, +0.0262]** (10 folds) and **+0.0162 [+0.0152, +0.0173]**; the average of the five best configurations gains **+0.0174 [+0.0144, +0.0204]** on confirmation genes and **+0.0183 [+0.0175, +0.0191]** on the whole cohort.

The gain holds on never-seen genes, at the size it has on the tuning folds (0.7121 → 0.7248 for the best tuned), and carries to a different nuclease on a different cohort (+0.023 to +0.026, ahead on 10 of 10 folds; 59,489 guides). The better direction is slower, not deeper: mean ρ over the 200 draws falls with depth (depth 3–5 about 0.713, depth 8–10 about 0.70) and with learning rate and rises with tree count, with a plateau at depth 4–5, lr 0.013–0.04 and 1,600–4,000 trees; the other knobs are each within ±0.001 once these are fixed, and the column budget stops mattering above about 450.

*Replication.* Genes were re-split 70/30 with a different seed and a **fresh 40-draw search** run on the new development half (`tune_ecoli.py --split 2`); re-scoring split 1's winner would have been contaminated, since 847 of the new confirmation set's 1,204 genes fed split 1's selection. The new winner is a different configuration in the same region (depth 6, lr 0.020, 2,000 trees; 450 columns, raw target): **+0.0121 [+0.0092, +0.0152]** against +0.0143, and +0.0159 against +0.0174 for the ensemble. The gap, **+0.0022**, estimates what selecting on the same genes adds and agrees with the "about +0.002" from an unrelated route.

*Limits.* Levels move far more than differences: the champion scores **0.7073** on split 1's held-out genes and **0.7329** on split 2's, a **0.026** swing between two random 30% gene partitions, roughly twice the whole tuning gain, while differences between arms reproduce to within 0.002; every tuned-model figure is therefore quoted as a difference, and a level belongs to its split. The benefit is about one and a half hundredths of ρ on a problem whose open variance is 0.22, and it concerns the flank columns more than the learner: without family A the whole-cohort +0.016 collapses to +0.003 (same gene-grouped protocol, tuned minus champion; Finding 4). The champion elsewhere in this report keeps its original settings, since every comparison was made against them.

### Mixing model families adds about +0.0015, and the tuned settings describe an additive-plus-pairs signal

*Mixing* (`EXTENSION_LOG.md` §12, §14). XGBoost, LightGBM (40 draws) and CatBoost (30 draws) were each searched on the tuning genes and scored on the held-out genes. As single models they land together (LightGBM 0.7206, CatBoost 0.7207, tuned XGBoost 0.7215; champion 0.7073), with prediction correlations 0.985 (XGBoost–LightGBM) and 0.977 (with CatBoost) and residual correlations 0.973 and 0.959, which is why blending helps so little. Against the XGBoost average of five (0.7246), the equal-weight blend with CatBoost gives 0.7262 (**+0.0016 [+0.0006, +0.0026]**), with LightGBM and CatBoost 0.7261 (+0.0014 [+0.0005, +0.0025]), and fitted weights 0.7264 (+0.0017 [+0.0005, +0.0030]). Models that err differently help only if good enough: extra-trees (0.666), a neural network on the selected columns (0.692) and ridge on all columns (0.647) each lower the equal-weight blend (to 0.7224, 0.7234, 0.7226). *Limit:* one split's held-out genes; Part 5 gives the cost and the eSpCas9 result, where mixing is lower than the single model.

*What the settings say about the data* describes the fitted ensemble, not the molecule: trees split on correlated columns, so "interaction order" is order among columns and one biological factor can appear as several. **Additive plus pairs:** tree depth is the interaction order; on held-out genes stumps reach 0.6905 against 0.647 for ridge, depth 2 0.7177 (+0.027), depth 3 0.7206 (+0.003), depth 4 0.7210 (+0.0003), depth 8 0.7158; of the 0.074 that trees add to the linear model, 59% (0.0435) is nonlinearity within a column, 37% (0.0272) pairs and 4% (0.0033) third and fourth order together. **Read more directly from sequence:** the tuned model's share of gain moves from dense summary descriptors to position-and-k-mer indicators (35% → 51%; protospacer quantum descriptors 37% → 27%, flank descriptors 27% → 22%). **Both levels of ranking:** on held-out genes the tuned model gains +0.018 [+0.008, +0.028] ranking genes against each other and +0.018 [+0.014, +0.022] ranking guides within a gene.

### Grouped cross-validation means something different on the curated cohort, and the gap narrows

Gene identity explains **16%** of the published label and **29%** of the curated one, because the published rows are an alphabetical block of the screen's genes (aaeA to leuC) and crisprHAL's curated release is genome-wide: within-gene share **0.843** on the published matrix (13,829 of 13,880 guides placed in 1,222 genes, 11.3 guides/gene) against **0.713** on the curated one (30,203 of 33,567 guides in 4,013 genes, 7.5 guides/gene). Pre-specified: twice the between-gene variance should widen the gap between random and gene-grouped CV. Observed: it narrows; grouping cost `e_nucleoid` 42% of its gain on the published matrix and 12% here, `g_transcription` 36% and 21% (Appendix B3; the published random-row arms of these two families are not stored in `results/`). Between-gene variance sets how much a locus-memorising model could win and guides per gene how well it can learn each locus's mean; both moved, in opposite directions, and only the measurement says which won. *Check on the method:* genes are assigned by coordinate on NC_000913.2 against the reference gene table for both cohorts; on the published rows this agrees with Noshay's own assignment for **97.7%** of guides and gives the same share to three decimals (0.843 against 0.844), so the change is the cohort and not the method (`python -m sgrna.harness.cohort_groups`, `results/cohort_groups.csv`).

### The published label's ceiling is 0.937 [0.929, 0.945], from tiling-library agreement

Two screens of the same library (WT-SpCas9 and eSpCas9, 33,567 shared guides) agree at ρ 0.810, which naively gives 0.90, but the disagreement is predictable from sequence (ρ 0.546; analysis not stored in `results/`), so it is enzyme biology and not noise. Using the source paper's (Guo et al., 2018) own agreement statistics, with the published score a two-replicate geometric mean, the tiling route (a *different* library, 901 shared guides, Figure 2C: R² = 0.771, r 0.878, reliability 0.878) gives **0.937 [0.929, 0.945]** (r in [0.863, 0.893]; read from the paper's text and figure, not from stored data). The replicate-agreement route with Spearman–Brown (Figure 2B: R² > 0.78, r > 0.883, reliability 0.938) gives 0.969 and is rejected: it must assume the activity score inherits the single-arm read counts' reliability, which it cannot, since the label is the `|Z|` of a Cas9/dCas9 **ratio** and a difference is never more reliable than its components (at 50% shared signal between arms the implied ceiling is already 0.940). *Limit:* 0.937 is the ceiling for Noshay's `cut_score`, not for the curated label trained on, whose ceiling (0.946 [0.944, 0.947] from a plain log-ratio of the replicates, Finding 6) comes from a replicate split that exists for that label; both are quoted because arms on the two cohorts are compared, and quoting 0.937 against a curated number would understate the headroom.

### Per-guide label reliability cannot be used as a training weight

Weighting training rows by measured per-guide reliability costs ρ, and four fifths of the damage comes from having non-uniform weights at all, not from where they point. Pairing Cas9 replicate 1 with control 1 and replicate 2 with control 2 gives two independent estimates of each curated guide's score, whose disagreement runs **0.06 / 0.71 / 2.39** log2 units at the 5th / 50th / 95th percentile; it is not a read-depth effect (ρ(disagreement, total counts) = −0.053) and is mildly tied to the label (ρ(|y|, disagreement) = −0.236). Weights apply to training rows only; validation rows are scored unweighted against the released label. Champion, selection inside each training fold, 3 seeds × 5 folds (`replicate_weight.py`, `results/replicate_weight.csv`): unweighted **0.7083**; inverse variance, moderated, 0.7032 (**−0.0050**); *the same weights permuted across guides* 0.7041 (*−0.0041*); inverse variance flattened within \|y\| bins −0.0031; weighted by read depth −0.0241; drop the noisiest decile of training guides −0.0016, *drop a random decile* *−0.0014*. Permuting keeps the weight distribution and destroys its correspondence with measured noise, yet costs −0.0041 of the −0.0050: a 98:1 weight ratio spends effective sample size, about four thousandths whatever the weights mean. Read depth is the wrong instrument by an order of magnitude because it tracks almost none of the noise. On a label with no shrinkage (the plain CPM log-ratio of the same counts; `--raw-label`, `results/replicate_weight_raw.csv`) inverse-variance weighting costs −0.0040 and permuted weights −0.0043, so the correspondence with measured noise is worth +0.0004. *Limit:* 33,567 rows against a feature set worth +0.164. The raw label is also less predictable, 0.6864 against 0.7083 on identical rows, folds and features: ALDEx2's processing is worth about +0.022 of fit, five times what swapping the label at fixed rows buys (+0.004, Part 3), and it is done in the label's construction, where no downstream reweighting reaches it.

### The reduced-representation ladder

**On the published matrix, 412 portable columns match all 6,232 (ρ 0.52713 against 0.52716; Finding 1).** One protocol, 5
seeds × 5 folds: 74 mononucleotide columns 0.5081; 427 (adding dinucleotide and non-sequence features) 0.5249; 412 (minus 15
undecoded) **0.52713 ± 0.0011**; published full 0.52716 ± 0.0022. The 427's loss is **+0.0023, 95% CI [−0.0008, +0.0054]**,
one-sided p = 0.035 against a 0.005 ρ margin set after the result was seen. *Limit:* every arm selects **300 columns from
its candidate pool** (`cross_validate` resolves `n_features=None` to `config.N_FEATURES`), so the ladder compares pools
under a fixed budget, not full representations. On the curated cohort the budget does not bind (±0.0005, Finding 1), but the
reduction fails there once each arm is re-run with `n_features` set to its own column count; the curated tetramer (−0.0018),
quantum-column (+0.0114) and non-inferiority results use that no-selection protocol.

**B.2 — What is the model class worth?**

### Sixteen model classes and the cost spread

**Boosting beats a tuned CNN–BiGRU (crisprHAL 2) by +0.003 at 1/113th of its wall-clock cost, and ridge is within 0.058 of
it at 1/220th.** Curated cohort, family A, one seed × 5 folds, crisprHAL 2 on the same folds (`curated_models.csv`,
`crisprhal_tuned_folds.csv`). Wall-clock for 5 folds: XGBoost (champion) ρ **0.7084**, 1.9 min; ridge 0.6473, 1.0 min;
crisprHAL 2 published configuration 0.6971, 85 min (44×); its best configuration 0.7051, 220 min (**113×**). The ~150×
headline in Part 5 is CPU-seconds per fold, 65.8 against 10,000 (`cost_panel.csv`, 152×; crisprHAL 2's 10,000 is an estimate entered by hand in `sgrna/cost/cost.py`, 1,266 s wall × 7.9); the two differ because the models
multithread differently (4.8 CPU-seconds per wall-second for SLICER, 7.9 for crisprHAL).

* **Ridge, PLS, random forest and extra trees agree within 0.009** and LightGBM (0.7082) matches XGBoost, so the problem is
  **bias-limited, not variance-limited**, at this sample size. k-NN is worst (0.6007), as a smooth global signal rather than
  a clustered one predicts (Finding 2's gradient, not a motif).
* **Boosting is worth +0.061 over ridge**, as a sum of many small, partly interacting effects requires (the margin was 0.03
  on the published matrix, with family A and 300 selected columns); model class decides about a tenth of a correlation (Finding 1).

### Interpretability figures and the gain-ranked selector

**Of the three interpretability figures in Part 5, the ridge share of boosting is the strongest, the depth cap holds for the
frozen configuration only and the gain concentration is the weakest; gain-ranked selection changes which columns are shown,
not how well the model scores.**

* **Ridge reproduces 95.6% of boosting's ρ on a decorrelated basis** (427 columns, published matrix, no family A, no selection; 93.7% with family A on the 775-column basis; `additivity_reduced.csv`). With one coefficient per column and no term in two
  columns, the gap measures how much accuracy needs interactions, about 4–6%, so Finding 2's account can be read off the model
  as per-column effects.
* **`max_depth = 4` caps the interaction order at four, for the frozen configuration only.** Depth 4, `learning_rate=0.05`,
  400 trees and the 300-column budget are inherited from `FINAL_baseline_ecoli_sgRNA_model_5seeds.py`, not selected; the
  post-freeze joint search (Finding 4, `EXTENSION_LOG.md` §11) prefers depth 5, learning rate 0.013, 4,000 trees
  (**+0.016** on the whole cohort, +0.0143 on held-out genes). Third- and fourth-order terms are 4% of what trees add over a linear model.
* **45 of 300 columns carry half the gain** (the top ten 19.5%; average gain per split, as `feature_importances_` reports it; by total gain about 24 columns carry half and the top ten 32%): this rules out a two-feature rule and an undifferentiated
  smear, and little else.

`run_ablation.make_selector` picks 300 of 6,517 columns by XGBoost gain, which does not survive a change of model and
credits a column for being binary (Finding 5). Audit (`selector_audit.py`, `results/selector_audit.csv`,
`results/selector_redundancy.csv`): 300 against every column, within ±0.0005 (curated; 600 and 1,200 columns, Appendix A); top 300 by gain against by mean
|SHAP|, +0.0007 (both arms selected on all rows, so only their difference is quotable); top *k* against greedy
relevance-over-redundancy, **−0.0015** (0.7083 against 0.7068; 3 seeds × 5 folds). Gain puts 144 binary indicators (48%) in the top 300 against
102 (34%) under SHAP, and the redundancy-penalised rule still picks more of them (175 against 146). *Limit:* no rule is both
decorrelated and informative, so attribution is reported at **block** level (Finding 5), and every statement about the
selected set is conditioned on gain-based selection.

**B.3 — What is the representation worth?**

### Recoverability, re-description and the family split: a column pays when it names something the phenotype depends on

**A column's increment falls from +0.180 to +0.008 as the guide explains more of it, with the step at about R² = 0.2, while
the strongest marginal association with the label in each band stays at |ρ| 0.14–0.18.** All 636 family columns with a defined R² were regressed on the guide's
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
0.0015 from zero on its mean over draws (single draws reach −0.0021). The mechanism the result is consistent with is a spacer that base-pairs with itself being less available to invade the target
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
`c_folding`'s +0.0153 in the same run: the independent evidence for the fold effect is **two measurements, not three** (the spacer fold
energy; `d_mechanics`'s 20 cheap guide-only columns), because cross-family duplicates are invisible to a family ablation.

### The human baseline

**On the human matrix (17,421 guides × 6,233 columns) the champion reaches ρ 0.3377 under twin-grouped folds, against 0.5273
on the *E. coli* published matrix (0.7314 for the best tuned model with family A on the curated cohort); row-wise folds overstate it.** The rows decode to 13,138 distinct 20-mers, 8,499 (49%) have a twin at the same
position; twin labels agree at Spearman 0.80 (`EXTENSION_LOG.md` §10), so row-wise folds reward memorisation; the bacterial
cohort has none (33,567 distinct in 33,567) and `kingdom_transfer.py` is unaffected. Twin-grouped
(`human_additivity_twin.py`, `results/human_additivity_twin*.csv`; row-wise: `human_baseline.py`,
`results/human_baseline*.csv`, `human_additivity.csv`), `max_depth` 4 → 6 moves +0.0585 to **−0.0062** [−0.0093, −0.0034].

**B.4 — Mechanism: what the effect is, and what it is not**

### Long-range features and the Chi-site (RecBCD) null

**The long-range candidates with a mechanism attached are built and null, and Finding 7 predicts most of what is left to be
null.** GC skew, A-tract phasing and bendability periodicity (family J) score −0.001 alone and +0.0004 on top of family A (published matrix),
with their own permutation control scoring higher. The unbuilt candidates (k-mer entropy, homopolymer run-length, REP
density, RegulonDB distances to TSS, terminator and operon boundary) are predicted null, because the effect is GC and not
composition in general (purine is flat at every window), so non-GC re-descriptions are reparameterisations, as DNA shape was
(Appendix B4); only the RegulonDB orientation items address the separate orientation effect.

**With flank GC held fixed, Chi (`GCTGGTGG`) carries nothing beyond what any GC-matched octamer carries: its coefficient sits at the 41st percentile of 200
of them.** RecBCD degrades a double-strand break over kilobases until a Chi site, so it would predict the flank
effect's length scale, missing motif and orientation dependence (Findings 2, 7), but the direction disagrees: Chi sites are
GC-rich, yet flank GC correlates **+0.161** with the label (`eng.flank.comp.up250.gc`, Spearman, curated cohort; target GC is −0.147 on that cohort and the −0.201 quoted elsewhere is the published-matrix figure). `chi_sites.py` counts `GCTGGTGG` and its reverse complement on each side for all 33,567 curated
guides (`results/chi_sites.csv`; 11,677 carry one within ±1 kb). The Chi count's β at ±250, ±500 and ±1,000 nt is +0.306,
+0.287 and +0.219 alone but **−0.103 ± 0.053, −0.047 ± 0.037 and −0.053 ± 0.026 with flank GC held fixed** (± standard error): the marginal
association is flank GC. The deciding control is **200 other octamers of the same GC content** (6 of 8), each entered the
same way at ±1,000 nt: mean β −0.022, sd 0.201, range −1.25 to +1.20; **Chi's −0.053 is at the 41st percentile**, so a specific
recognition sequence behaves like any GC-matched eight-mer. *Limits:* only the Chi-attenuation route is excluded; repair in
general (RecBCD processivity, RecA loading, replication-fork proximity) is not, and is not testable from sequence alone.
With flank GC fixed at ±500 nt, `GCTGGTGG` downstream is +0.169 ± 0.094 and its reverse complement upstream −0.110 ± 0.060,
within the ±1,000 nt null's sd of 0.201 (total counts), so this orientation hint is not claimed.

### Why ρ, and why R² beside it

**ρ and R² are reported together: a LightGBM with a robust (Huber) loss scored R² 0.192 with ρ 0.500**, against 0.382 and 0.609 for the same model with squared error (published matrix with family A, 3 seeds × 5 folds; `model_comparison.csv`), so R² halves where ρ falls by a fifth. ρ is primary because it is crisprHAL 2's own reporting convention and
ranking is the decision the model serves; what ρ means on that scale is in Part 5.

## Appendix B2 — Why the imported families failed, and why that is not a screen

**Two of the eleven imported families are columns the 20-mer already determines, and their standalone gains are small (+0.007 and +0.020 on the published matrix); the table does not support a general rule beyond that.** What survives is Finding 1: a deterministic function of the 20-mer cannot add information to a model that already encodes the 20-mer.

Out-of-fold R² of predicting each new column from position-specific mono- and dinucleotide indicators of the protospacer (ridge, five folds, the published matrix's 13,880 guides; `redundancy.csv`). Δρ is the family's standalone increment over the base on the published matrix (Appendix B3).

| family | columns | median R² | **share of columns with R² > 0.9** | Δρ |
|---|---:|---:|---:|---:|
| `b_energy` (CRISPRoff ΔG) | 10 | 0.984 | **70%** | +0.007 |
| `d_mechanics` (duplex stability) | 30 | 0.845 | **50%** | +0.020 |
| `c_folding` | 9 | 0.297 | 0% | +0.007 |
| `f_methylation` | 17 | 0.141 | 0% | +0.003 |
| the other seven (`a_flank`, `d_supercoiling`, `g_transcription`, `e_nucleoid`, `i_shape`, `j_periodicity`, `h_offtarget`) | 7–322 each | −0.012 to −0.022 | 0% | −0.001 to **+0.080** (`a_flank`) |

**The structure is binary.** `b_energy` and `d_mechanics` contain columns the 20-mer determines; the other nine contain none (`c_folding` and `f_methylation` average weak partial fits). Nearest-neighbour thermodynamics is a sum over adjacent pairs, so binding energy is a second-order quantity the matrix already holds: importing CRISPRoff imported the matrix into itself. A negative median R² (10th percentiles −0.019 to −0.025) is what out-of-fold R² against a target unrelated to the basis looks like; it is set by basis size and row count.

**Why this is not a general screen.**

* **It applies to two families.** Only `b_energy` (0.984) and `d_mechanics` (0.845) exceed 0.30. A rule with n = 2 is an observation, and for `b_energy` it is arithmetic: ΔG is computed from the sequence by nearest-neighbour summation, so adding it to a model holding the sequence losslessly adds nothing. The same holds for `d_mechanics`.
* **The rest carry no signal about the rule.** Among the families with negative R², R² spans 0.010 while their gains span 0.080; across all eleven, Spearman ρ(R², Δρ) = **0.227, p = 0.50** (family medians of `redundancy.csv` against the published increments, unrounded). Their failures have unrelated causes: `h_offtarget` needs the rest of the genome, `j_periodicity` is null (−0.001 alone, +0.0004 on top of family A, published matrix), `i_shape` duplicates family A (+0.036 standalone, +0.0010 on top of it, published matrix), and `d_supercoiling` reads gene dosage (Part 2, biology item 6).
* **No tally is supportable.** Only four families have positive R², no threshold for "high" can be set after the fact, and `d_mechanics` (R² 0.845) returned +0.020.

**Limit.** The usable direction is qualitative and carries no score: if a candidate's columns are individually reconstructible from the 20-mer, do not build it.

## Appendix B2a — Curation: what it is, and why family A doubled between cohorts

**Family A's gain is +0.080 on the published matrix and +0.164 on crisprHAL's curated cohort (both five seeds × five random-row folds) because the curated cohort contains 24,950 additional guides in which the flank effect is steeper: the label, the row count and the base account for +0.014 of the +0.084 difference, and the remaining ≈ +0.070 is shown by elimination and, directly, by a 2.96× steeper flank-GC slope in the added guides.** Neither figure is an attenuated version of the other: +0.080 is the right figure for the alphabetical block of genes (aaeA to leuC, 1,222 under our assignment) in which the published matrix's rows lie (a subset of Guo et al.'s genome-wide library), +0.164 the right figure genome-wide. On the curated cohort itself, gene-grouped, the gain is +0.102 [+0.089, +0.115] in those genes and +0.180 [+0.164, +0.197] in all the others (`mappability_stratum.csv`), so the block alone accounts for most of the gap. Comparing them across cohorts without this reads as the model improving.

**Curation drops rows; selection drops columns, and only curation can flatter a score.** crisprHAL's curation removes guides whose label is unreliable and is applied to the whole dataset **before splitting**, so it also cleans the test set and raises ρ partly by making evaluation easier; feature selection is fitted inside each training fold and cannot. The figure for curation here, ρ 0.607 → 0.628 for base + family A (Noshay's label) on restricting the 13,880 rows to the 8,617 that curation retains, contains both effects and has not been separated.

**Procedure** (Browne et al., 2026). The score is the ALDEx2 `diff.btw` log2 fold-change between the Cas9 and dead-nuclease arms, shrunk toward the prior. The filter is an average control-condition read count across the two control replicates, with a threshold of **73** for the *E. coli* SpCas9 set used here, chosen by 5-fold cross-validation on the training set and applied before the 80/20 split. The cohorts are not nested: 8,617 guides are shared with Noshay's published 13,880, 24,950 are only in crisprHAL's and 5,263 only in Noshay's. Each candidate explanation was tested with everything else held fixed:

| candidate | test | effect on family A's Δρ |
|---|---|---:|
| the **label** | same 8,617 rows, published base, swap `cut_score` for the curated score | **+0.004** (+0.0892 → +0.0936) |
| **row count** | random 13,880-row subsamples of the curated cohort, 3 draws | **−0.003** (+0.1672 ± 0.0027 against +0.1638 at 33,567) |
| the **base** | add 41 dense guide-only summary columns to the curated base (Appendix B3) | **+0.013** (+0.1638 → +0.1504) |
| **which rows** | the remainder of a +0.084 gap (by elimination) | **≈ +0.070** |

**The relationship is steeper in the added rows, not wider-spread.** The marginal rank correlation between the GC fraction of the 500 nt upstream of the protospacer and activity needs no model: +0.069 on the 8,617 shared guides under Noshay's label and +0.067 under crisprHAL's (the label does nothing), and **+0.172 on the 24,950 guides only crisprHAL has**. Regressing the curated label on that same upstream-500 GC gives a slope β of 5.35 on the shared rows, 8.63 on added rows inside Noshay's panel and **17.03** outside it (`curation_variance.py`, `results/curation_variance.csv`). The predictor's spread rises only 1.21× and the label's 1.34×, which pushes r down, so the 2.7× rise in r is β (2.96×); which guides were published inside the panel's genes accounts for about ×1.6 of it and the choice of genes for ×2.0.

**Limit.** Why panel genes differ is open: separating dynamic range from different biology needs per-guide read counts that were not used. An essentiality explanation is argued against by family G's essentiality column being worth +0.00004 on its own.

## Appendix B3 — The family ablations on the curated cohort, and what they measure

**Of eleven feature families, one is the finding (family A, +0.164 ρ on the curated cohort, random-row folds); the other ten add only +0.039 beyond it (all eleven jointly give +0.203), and nothing confined to the 20-mer exceeds +0.019.** Five of the ten return +0.001 to +0.019 (`b_energy`, `c_folding`, `j_periodicity`, `f_methylation`, `h_offtarget`); five show real standalone gains: three are family A or gene dosage in other clothes (`i_shape`, `d_mechanics`'s flank columns, `d_supercoiling`) and two read external position-indexed tracks (`e_nucleoid`, `g_transcription`). The table is a ranking of families, not a budget.

Design: base representation against base + one family, 5 seeds × 5 folds, selection inside each training split (`python -m sgrna.cohorts.curated_ablation --all`); baseline ρ 0.5445 ± 0.0010. The published column is each family's own run in Appendix C (`ablation_main`; `ablation_grouped` for the three positional families; `ablation_shape`, recorded as +0.0356 in `results/RESULTS_archive_2026-10-10.md`, with its joint arm in `ablation_shape_joint.csv`; `ablation_jper`), whose baselines differ slightly between rows (0.5272–0.5273 random-row, 0.5283 100 kb-grouped). The gene-grouped column (`curated_ablation --grouping gene`) holds out whole genes on the 30,203 guides inside an annotated gene (4,013 groups, 5 seeds × 5 folds, baseline ρ 0.5584; the 3,364 outside every annotated gene are dropped), so Δρ compares across columns and the ρ levels do not.

| family | reads | columns (dense continuous) | published Δρ | **curated Δρ** | gene-grouped Δρ | retained |
|---|---|---:|---:|---:|---:|---:|
| `a_flank` | flanks, ±1,000 nt | 348 (155) | +0.080 | **+0.164** | **+0.157** | 96% |
| `d_supercoiling` | external (GapR-seq track) | 26 (26) | +0.044 | **+0.139** | **+0.131** | 94% |
| `i_shape` | flanks | 152 (152) | +0.036 | **+0.120** | **+0.112** | 94% |
| `d_mechanics` | guide (22 of 30 columns) and flanks (8) | 30 (29) | +0.020 | **+0.099** | **+0.092** | 92% |
| `e_nucleoid` | external (Hi-C, nucleoid proteins) | 20 (20) | +0.016 | **+0.072** | **+0.063** | 88% |
| `g_transcription` | external (transcription, replication) | 19 (12) | +0.015 | **+0.058** | **+0.046** | 79% |
| `b_energy` | guide (R-loop free energy) | 10 (9) | +0.007 | +0.019 | +0.017 | 91% |
| `c_folding` | guide (sgRNA folding) | 9 (7) | +0.007 | +0.014 | +0.014 | 97% |
| `j_periodicity` | position (strand asymmetry, helical phasing) | 24 (24) | −0.001 | +0.011 | +0.007 | 64% |
| `f_methylation` | short flank (±20 nt at the PAM; Dam, Dcm) | 17 (2) | +0.003 | +0.008 | +0.007 | 93% |
| `h_offtarget` | genome (off-target burden) | 13 (1) | +0.000 | +0.001 | +0.000 | 23% |

**What became of each family.** `a_flank` is the finding (Finding 2). `d_supercoiling` is read density (the untagged GapR control predicts almost as well as the ChIP, +0.0087 against +0.0111 on the published matrix with 100 kb blocks), mostly gene dosage recoverable from distance to *oriC*: a trap, not biology (Part 2, biology item 6). `i_shape` is family A re-expressed as bendability, **+0.120** standalone (+0.036 on the published matrix) and **+0.0009** on top of A (`a_flank` + `i_shape` +0.1647 against `a_flank` +0.1638; +0.0010 on the published matrix; Appendix B4): a near-duplicate, not a null. In `d_mechanics` the 8 flanking columns carry +0.097 of the +0.099 (Finding 1); the guide-only effect is +0.011 (the 20 `d_mechanics` columns the guide determines linearly) to +0.019 (`b_energy`) on two independent measurements, because `b_energy`'s and `c_folding`'s shares of it are the same spacer-fold column entered twice (Appendix B). `e_nucleoid` and `g_transcription` read external position-indexed tracks and carry a real locus component (below). `c_folding` is small but real (Part 2, further inferences, item 3). `j_periodicity`, `f_methylation` and `h_offtarget` are null or nearly so (+0.011, +0.008 and +0.001 on the curated cohort; six of `h_offtarget`'s thirteen columns are constant zero by construction of the cohort).

**The ranking transfers; the scale does not.** The published and curated columns rank-correlate at **ρ = 0.97**, every family gains, and the median inflation is **3.2×** (the ten families with a positive published increment; 1.9× for `c_folding` to 4.9× for `d_mechanics`); ratios of Δρ are not ratios of information, and every family statement elsewhere is a difference. For family A the 2.1× rise is accounted for by the **2.96× steeper** flank-composition slope in the guides crisprHAL adds (Appendix B2a); that the other families that read genomic context inherit the same multiplier is an inference, not tested family by family. The two Δρ columns are not comparable as levels: the published base contains the 63 hand-named summaries and the curated base (6,169 sparse indicator columns, 300 selected) has none.

**Neither the selector nor column shape produces the gain.** On `d_mechanics` (seed 41, five folds; `selector_control.csv`): base with 300 selected 0.5443, base + `d_mechanics` 0.6432; base with all 6,169 columns 0.5439, base + `d_mechanics` with all 6,199 columns 0.6432, so the gain is **+0.0988 with selection and +0.0993 without**. ρ(dense continuous columns, Δρ) = **0.882, p = 0.0003, n = 11** (columns with more than ten distinct values, `--shape-audit`, `results/ablation_curated_shape.csv`), but supply is not sufficient: `j_periodicity` supplies 24 dense columns, as many as `d_supercoiling`'s 26, and returns +0.011 against +0.139; dense columns and reach beyond the 20-mer are correlated, and reach is the one that separates. Giving the curated base 41 dense guide-only summary columns (`b_energy`, `c_folding` and `d_mechanics`'s protospacer half; `base_enrichment.py`) raises it from 0.5445 to 0.5692 and takes **0.013** off family A's gain, 8% of it. The published matrix's own 63 hand-named summaries take **+0.0015** off, 1.8% of it (`base_enrichment.py --published`, `results/base_enrichment_published.csv`): sparse base of 6,169 columns ρ 0.5227, + family A 0.6050, Δ **+0.0823**; with the 63, 6,232 columns, 0.5272, 0.6080, **+0.0808**. The second base's 0.5272 reproduces Finding 1's 0.52716 through a different code path, certifying that the arms differ only by those columns. Column shape is therefore not the main source of the published-to-curated inflation; for family A the row population accounts for most of it (about +0.070 of the +0.084 gap, Appendix B2a).

**Reach separates the families better than any property of their columns: nothing confined to the 20-mer exceeds +0.019, and nothing above +0.05 is confined to it.** Curated Δρ by reach: the 20-mer and its PAM only (`b_energy`, `c_folding`, `d_mechanics`'s 22 guide columns) **+0.011 to +0.019**; a short flank (`f_methylation`) +0.008; the flanking DNA (`a_flank`, `i_shape`, `d_mechanics`'s 8 flank columns) **+0.097 to +0.164**; the rest of the genome or external measurement (`d_supercoiling`, `e_nucleoid`, `g_transcription`) +0.058 to +0.139; position on the chromosome (`j_periodicity`, `h_offtarget`) +0.001 to +0.011. Access to information outside the guide is necessary for a large gain and not sufficient: `j_periodicity` and `h_offtarget` read the whole chromosome and return +0.011 and +0.001. This is the conclusion Finding 2 reaches from the other direction.

**Gene-grouped cross-validation: the gains are not locus memorisation.** Gene identity explains **29%** of the curated label against 16% of the published one (Appendix A), so more is available to memorise, yet with whole genes held out the three largest effects survive intact. The guide-only families (`b_energy`, `c_folding`, `f_methylation`) cannot memorise a locus and still retain only **94% in aggregate**: that 6% is the **grouping tax**, the cost of a harder partition (less locus diversity in training; 4,013 groups split five ways is coarser than 30,203 rows). `a_flank` (96%), `d_supercoiling` (94%) and `i_shape` (94%) lose nothing beyond it; `e_nucleoid` (88%) and `g_transcription` (79%) lose 6 and 15 points beyond it, a real locus component in two external position-indexed tracks; `j_periodicity` (64%) and `h_offtarget` (23%), with gains of only +0.011 and +0.001, lose 30 and 71 points beyond it. *Predicted in advance:* twice the between-gene variance should widen the gap between random and grouped CV. *Observed:* the gap is smaller (grouping costs `e_nucleoid` 12% and `g_transcription` 21% of their gains here, 42% and 36% on the published cohort), because the curated cohort has **7.5 guides per gene against 11.3**: between-gene variance sets how much a locus-memorising model could win, guides per gene sets how well it learns each locus's mean, and the measurement says the second dominated.

**The column must not be added up.** Joint arms (`curated_joint.py`, same base, seeds and folds), joint Δρ against the sum of singles: `a_flank` + `i_shape` +0.1647 against +0.2833 (58% realised); `a_flank` + `d_supercoiling` +0.1882 against +0.3029 (62%); `d_supercoiling` + `e_nucleoid` + `g_transcription` +0.1454 against +0.2692 (54%); `a_flank` + `d_supercoiling` + `i_shape` +0.1882 against +0.4224 (45%); the guide-only control `b_energy` + `c_folding` + `f_methylation` +0.0289 against +0.0409 (**70%**); **all eleven families +0.2030** against +0.7046 (**29%**). Summing the column would claim +0.70 ρ; the real figure for all eleven is +0.203, only +0.039 beyond family A alone. `i_shape` adds +0.0009 on top of family A and +0.00005 once `d_supercoiling` is also present (the arms with and without it agree to four decimals, +0.1882). The guide-only control licenses the reading: those families overlap only through the 20-mer and are the most additive at **70%**, while the flank and positional families, which read overlapping DNA, realise 45–62%; had the control also come in near 50%, the conclusion would have been a pipeline artefact rather than an interpretation.

## Appendix B4 — The flank's shape and 3D structure add nothing that its composition does not already hold

**Claim.** Describing the flank by its DNA shape (13 base-pair and step features from three independent sources), or by a rigid-body reconstruction of its 3D helical path in a frame attached to the cut, recovers up to 90% of family A's gain on its own (one source at a time) and adds **−0.0008 to +0.0006 ρ** to it (curated cohort, gene-grouped CV, boosted trees; the ridge and eSpCas9 results are below). These descriptors carry window composition plus the identity of the first 8–25 base pairs beside the cut; family A holds both. Shuffling the flank, which destroys everything else, keeps the composition part of the gain. The test addresses whether the scalar-summary null for shape (`i_shape`, +0.0009 over family A) reflected the summary rather than the information: shape is a vector per step with direction, which a window mean cannot express. Post-freeze experiment; results in `results/`.

**Design.** The DNAshapeR pentamer table (512 pentamers × 90 numbers) was parsed into named columns by source (`mc` Monte Carlo, `xrc` X-ray, `md` molecular dynamics) and kind, and checked for reverse-complement consistency (a duplex read from either strand gives the same step parameters and, for the 3D chain, the same end-to-end distance and radius of gyration to four decimals). Three blocks per source: `mean` (arm × four windows × 13 features, 104 columns), `vec` (the first 8 bp of each arm position by position, 208) and `geom` (201: path positions, axis tangents, gyration tensor and its eigen-structure, straightness and excess twist, in the frame of the base pair 5′ of the cut). Gene-grouped CV, three seeds × five folds, 30,203 genic guides, the same folds for every arm, so differences are paired. Over the base (ρ 0.5588), alone, for md / xrc / mc: `mean` +0.099 / +0.102 / +0.103, `vec` +0.052 / +0.056 / +0.053, `geom` **+0.031** / +0.078 / +0.080, all three blocks (513 columns) +0.138 / +0.140 / +0.141, every cell ahead in 15 of 15 folds. References: family A (348 columns) +0.156; plain A/C/G/T fractions in the same windows (40 columns) **+0.105**; A's own 24 composition columns +0.107.

**On top of family A (ρ 0.7151)**, paired over the same folds: mc `mean` +0.0001, `vec` −0.0005, `geom` −0.0008, all three −0.0007; md / xrc `geom` −0.0006 / −0.0004; md / xrc all three −0.0007 / **+0.0006**; letters in windows (40 columns) −0.0001; all three sources (1,887 columns) −0.0006. **Shuffled-flank controls** (mc, all three blocks): flank shuffled in 25-nt blocks **−0.0012**, shuffled whole-arm **−0.0009**; the real all-block arms (mc, md, xrc, all sources) sit 0.0002–0.0018 above them, which is the cost of adding five hundred columns that cannot help. The null survives the obvious alternatives:

* **Displacement and depth.** At a 600-column budget instead of 300: A + mc all −0.0008, mc `geom` −0.0018, against A at 600. At depth 6 and 800 trees instead of the champion's 4 and 400 (A itself rises 0.0073 to 0.7225): A + mc all −0.0012, A + arm-shuffled mc all −0.0023.
* **A second cohort** (crisprHAL's *E. coli* eSpCas9 screen, 59,489 guides, 100-kb-block CV): the blocks alone recover 94% of A (+0.0969 against +0.1034); A + mc all +0.0022 (14/15 folds), A + mc `geom` +0.0008, **A + arm-shuffled mc all +0.0016**.
* **Fresh seeds** (44–46, folds never used before), curated cohort: A + mc all −0.0006, A + arm-shuffled mc all −0.0010, mc all alone +0.1411 against +0.1570 for A.

**Why the standalone gains exist, each explanation tested.**

1. *The sources agree and letters do as well.* The three sources agree on `mean` and `vec` to 0.004 although their parameter values differ (mean twist 34.3/33.1/32.5° and spread across pentamers 1.5/3.8/4.1°, mc/xrc/md), so the blocks do not read source-specific physics; forty columns of base fractions give +0.105 against +0.103 for the 104 shape columns, and adding the shape means to the letters changes ρ by −0.0002.
2. *The geometry's gain is composition.* `mc geom` on a flank shuffled within 25-nt blocks gives +0.080 (real +0.080); shuffled across the whole arm, keeping only its composition, +0.077.
3. *The geometry columns are dinucleotide counts plus the first base pairs.* Of the 27 geometry columns that survive selection, the five largest are excess twist (summed step twist minus 10.5 bp per turn) over 100–499 nt. Letter fractions predict them out of fold with R² 0.43–0.67 and dinucleotide fractions with R² **0.86–0.97** by ridge (0.85–0.94 by boosting): **0.968** for the downstream 499-nt window, 0.86 at 100 nt (`geom_importance.py`). A sum of step twists is, to that accuracy, a weighted dinucleotide count. The rest are path coordinates at +12 and +25 bp, with R² ≈ 0 from letters and at most 0.15 from dinucleotides: the identity of the first 12–25 bp after the PAM.
4. *Order matters in exactly one place.* `vec` loses 86% of its gain when the flank is shuffled in blocks (+0.053 → +0.008) and 94% when shuffled whole-arm; `mean` loses 4% and `geom` 0% under block shuffling (12% and 3% whole-arm).
5. *MD geometry is weaker alone (+0.031).* The sources disagree on each guide's excess twist (cross-source correlations 0.27–0.66) and the MD value correlates with the label half as strongly (ρ −0.056 against −0.112 and −0.122). A phase mismatch from MD's mean twist is rejected: recentring every source to 34.3° leaves ρ unchanged (md 0.5914 against 0.5897). The cause beyond that was not identified.

**What family A is made of.** A is four kinds of column: 104 position-specific one-hot columns near the cut (`seq`), 156 per-position quantum-chemical tensor values at the same positions (`qct`), 64 windowed means and sds of four tensor series at 50–1,000 nt (`win`), and 24 composition summaries (`comp`). Alone over the base: `seq` +0.055, `qct` +0.070, `win` +0.109, **`comp` +0.107**. Removing one kind from A moves ρ by +0.0004 (`seq`, i.e. nothing), −0.0003 (`comp`), −0.0020 (`win`) and −0.0054 (`qct`). The flank effect therefore has two parts: a **window-scale composition gradient worth about +0.105, carried by 24 columns of GC and purine fraction**, and a **local component within ~10–15 nt of the cut worth +0.05 to +0.07**; together +0.156, less than their sum. Everything in the shape and 3D blocks falls inside those two parts.

**Limits.**

* **The null is about the boosted pipeline.** A *linear* model does see information beyond A in the same columns: ridge on all columns (one seed, gene-grouped five folds) gives A + mc all **+0.0097** (5/5 folds) and A + md all +0.0078, against +0.0002 for the arm-shuffled control, mostly from the near-junction `vec` (+0.0048, control −0.0004) and the window means (+0.0054, control +0.0018). The boosted trees do not show it at depth 4 or 6; whether they already extract it from A and the base, or the gain is too small to split on, was not separated.
* **Tabulated pentamer means, static.** The values are not measurements of this genome's DNA, and no stiffness matrix or fluctuation was used. A neural shape predictor with seven base pairs of context (Deep DNAshape) was not run. Its output at a position is a deterministic function of the few nucleotides around it, so the model was given the inner ±50 nt one-hot explicitly (the 75 positions family A does not reach, 300 columns): at the tuned configuration this is worth **+0.0002 ρ [−0.0004, +0.0008]**, indistinguishable from the same columns shuffled across guides (−0.0004), with the selector picking 8.7 of the real columns and 5.3 of the shuffled ones out of 450 (`EXTENSION_LOG.md` §22.3). Any deterministic function of that window, Deep DNAshape included, is capped by it; what remains possible is a learnability claim (a better-*shaped* feature for information worth at most +0.0008 ρ, the upper end of the interval at the tuned configuration), not an information one.
* **3D path from mean step parameters; one organism in the main arm.** The same design on the human matrix (`human_shape.py`; 16,490 guides located on hg38, family A from a teammate's hg38 build, three seeds × five folds, random-row folds; `results/human_shape_ablation.csv`) gives the three mc blocks on top of family A **+0.0183** (15 of 15 folds; **+0.0177** under twin-grouped folds, `human_shape_ablation_twin.csv`) where *E. coli* gives −0.0007. That was stopped at the observation and is not explained (`EXTENSION_LOG.md` §21.6). The 3D path says nothing about nucleosome-free DNA looping or the supercoiled chromosome; the nucleoid's own 3D structure (Hi-C) is family E, whose null stands.
* **Still open** is narrower than "shape": a *measured*, non-sequence-derived quantity (a stiffness matrix, or a shape measurement of this genome's own DNA) that no local sequence predictor supplies. This is row 3 of the open-questions register in Part 4.

## Appendix B5 — A boundary experiment, demoted: the dCas9 / CRISPRi screens

**No headline claim rests on this experiment, and Parts 1–3 do not use it.** It was meant to test whether the flank effect needs a double-strand break, but the dCas9 CRISPRi screen (Cui et al., 2018; its E75 arm is the dataset released by Rousset et al., 2018) and the cutting screens differ in enzyme state, library, laboratory and phenotype (fitness after silencing against survival after a break) at once, so no mechanism follows from a difference in how much family A adds, however scaled. Neither reading below (13–15% or 26% of the attainable gain) would change the candidate mechanisms of Parts 2–3, and the one repair route sequence can test (Chi sites, Appendix B) was excluded on its own evidence. Four analyses of the same screens (`cui_screen.py`, `crispri_screens.py`, `strand.py`, `strand.py` (`--parts cutting`)) supported four readings; only the last survives, and it is weakly identified. The independent third screen, "Wang" below, is Wang et al. (2018).

| round | analysis | reading it supported | what overturned or qualified it |
|---|---|---|---|
| 1 | family A, base and flank-only, whole screen | **specific to cleavage**: −0.009 against +0.156 | a screen-wide average hides the guides that carry the signal |
| 2 | the same, split by gene essentiality | inside essential genes +0.27, which would **retire the localisation** | that +0.27 is the assay's own strand rule |
| 3 | strand supplied as one column | **the null holds**: +0.007 to +0.025 on Cui and on its E75 arm ("Rousset", the same experiment); Wang, the one independent screen, does not replicate | the pooled ranking is dominated by strand, which hides a within-gene effect |
| 4 | within-gene ranking, coding strand only, ceiling-scaled share | **the effect exists without a break**: 13–15% of the attainable gain against 26% on cutting | not overturned; three caveats below |

### Family A adds nothing pooled; inside essential genes its +0.27 is the assay's strand rule

**Family A is worth −0.0093 on a silencing screen and +0.1563 on a cutting screen, measured the same way** (same pipeline and champion, 3 seeds × 5 folds, whole genes held out because 30% of Cui's label variance is between genes; five seeds give +0.1567 on cutting). Cui et al.'s 76,325 uniquely-targeting guides were located in NC_000913.2, 70,298 of them in 4,265 named genes; the 20-mer base reads 0.4577 ± 0.0013 and 0.4484 ± 0.0006 with family A, against 0.5588 and 0.7151 on cutting (`cui_screen.csv`; `family_a_at_tuned_summary.csv`), and family A alone reads 0.1333 (not zero, which rules out selection dilution), so the null belongs to the family and not to the run. *Limit: the average is not the informative subset.* A CRISPRi label needs the gene to matter, 92% of the genic cohort is non-essential, and essential-gene guides have label sd 2.886 against 1.133 (`cui_screen.py --essentiality`, `results/cui_essentiality_split.csv`).

**Inside essential genes family A reads +0.27, but it is a lossy copy of which strand the guide targets, the dominant covariate of a CRISPRi screen: one binary column beats the 348-column family (0.6621 against 0.5265 under 100 kb blocks, 0.6630 against 0.5227 by gene), and given strand family A is worth +0.0069 (100 kb blocks) to +0.0148 (genes), an order of magnitude below the +0.156 of cutting.** dCas9 blocks RNA polymerase better from one strand (a property of the assay, not of Cas9). On Cui's 5,514 essential-gene guides (44 blocks, 297 genes; `cui_screen.py --marginals`, `results/cui_strand_summary.csv`) coding-strand guides average −5.07 against −1.19 (1.35 sd), strand alone explains 45% of the label's variance, and a coding-strand guide is more depleted than an off-strand one 88% of the time; a binary column cannot exceed ρ 0.863 at these group sizes and strand reaches 0.662 (77%). The control is the non-essential genes, where the same statistic is flat (gap 0.15 sd, 0.6% of variance, AUC 0.54 against 0.88). The Finding 2 column carries nothing here (GC of the 500 nt upstream, marginal Spearman −0.0004); family A's strongest column, `eng.flank.comp.up1000.purine` (+0.4045), counts purines on the guide's own strand and so flips with orientation.

| ρ, or increment of family A (`cui_screen.py --strand`, `cui_essential_strand.csv`; the E75 column is described above) | Cui, 100 kb blocks | Cui, genes | E75 arm ("Rousset"), genes |
|---|---:|---:|---:|
| increment, all genic guides | | −0.0093 | −0.0206 |
| increment, non-essential genes | | −0.0101 | −0.0250 |
| *essential genes:* 20-mer base | 0.2805 | 0.2718 | 0.2141 |
| base + family A | 0.5484 | 0.5468 | 0.5043 |
| *family A, apparently* | *+0.2679* | *+0.2749* | *+0.2902* |
| **which strand the guide targets, alone (1 column)** | **0.6621** | **0.6630** | |
| base + strand | 0.6506 | 0.6462 | 0.5878 |
| **family A, given strand** | **+0.0069** | **+0.0148** | **+0.0245** |

Holding out 100 kb blocks instead of genes, the control for essential genes clustering in the genome, leaves the +0.27 unchanged. A position-only control rules out a location lookup (the bare coordinate buys +0.0394 under block grouping; `results/cui_essential_position.csv`), and telling the model the wrong strand on every held-out guide ranks the screen backwards (ρ −0.528 against 0.646; `strand.py --parts flip`). No other family beats the strand call: `strand.py` (`--parts families`) (all eleven families, 5,514 guides, three seeds, both groupings; `results/strand_families.csv`) gives `a_flank` +0.015 (genes) and +0.007 (blocks) over `base + strand`, `g_transcription` (expression, gene geometry, replication context) +0.057 and +0.028, and the other nine −0.013 to +0.030; seed sd over the three seeds is 0.0001–0.02 pooled (median 0.005) and 0.0015–0.05 within a strand (median 0.013). Only `g_transcription` is clearly positive in both groupings and it is not a flank family; the largest other gains (`d_supercoiling`, `e_nucleoid`) change between groupings and are not claimed. *Limits:* dropping the columns most correlated with strand does not make an arm strand-free (an out-of-fold classifier on the rest still reads strand at AUC 0.79 for A and 0.9997 for `g_transcription`), so the `-proxy` arms are not quoted (`strand.py` (`--parts recoverable`), `results/strand_families_recoverable.csv`); the +0.007 to +0.025 is the gain to the pooled ranking, which strand dominates (0.28 to 0.65 from one column), and does not say A is inert within a strand (below); the informative subset is 5,514 guides in 44 blocks, the control rests on one strand indicator, and nothing here is an intervention.

**On the E75 arm ("Rousset") all three steps hold, but that is one experiment and not a second screen.** `E75_Rousset.tsv` is the `fit75` column of Cui's own `screen_data.csv` and `E18_Cui.tsv` its `fit18` column (Spearman 1.0000 on 54,326 shared single-target guides): Cui's strains LC-E75 (dCas9 at a lower concentration) and LC-E18, both grown 17 generations, here on the 59,246 guides that Rousset et al. (2018) retained after filtering (`crispri_screens.py --family-a`; the stored `results/crispri_screen_family_a.csv` holds Wang only, and the E75 arms are the means of `folds_cris_Rousset_*.csv` in the project's results/). On the E75 arm, under 100 kb blocks it gives +0.2916 apparently and +0.0213 given strand, against +0.2679 and +0.0069 on Cui (label: the mean of the two arms). The strand rule holds on both arms (`crispri_screens.py --strand`, `results/crispri_screen_strand.csv`): ρ(coding strand, label) is −0.633 on the E75 arm and −0.599 on the E18 arm inside essential genes against −0.056 and −0.048 outside. **Wang, the one independent screen, does not replicate, and the locus control does not explain it away** (all genic +0.1019; essential genes given strand +0.2669 by gene and +0.2195 by block). Its strand control is vacuous (52,104 of 52,104 guides are coding-strand), 0.713 of its label variance is between genes (0.375 on the E75 arm, 0.296 on Cui's mean label; all genic guides) and its 20-mer base reaches only ρ 0.0829 in essential genes (0.2141 on the E75 arm, 0.2718 on Cui's mean label); the within-gene comparison has not been run on Wang, so the null holds on Cui's experiment, fails on Wang, and the difference is unexplained. All are *E. coli* K-12 MG1655 CRISPRi and the first two are one experiment, so there is one replication attempt, on Wang, and it fails; the E75 arm checks the label (one arm, guides filtered for bad seeds and off-targets), not the screen.

### Within genes, on the coding strand, family A keeps 13–15% of the attainable gain against 26% on cutting (hypothesis)

**The pooled null is false inside the coding strand.** Fitted and scored within one strand, family A gains +0.124 (genes) and +0.119 (blocks) over `base` on Cui's 2,535 coding-strand guides and +0.159 and +0.133 on the E75 arm's 2,074, and loses on the template strand (−0.026, −0.040 and −0.019, −0.022); Wang's +0.267 is this stratum (`strand.py` (`--parts families`); Rousset `strand.py --parts rousset`); gene-pooled within the coding strand `g_transcription` gains more than A (+0.198 blocks, +0.267 genes; the other nine −0.008 to +0.081 and −0.000 to +0.154). Within-gene ρ (out-of-fold predictions with each gene's mean removed; `strand.py` (`--parts within-cui within-joint within-rousset`), ten seeds; 206 genes on Cui's mean label and 182 on the E75 arm, which is the same experiment, with at least five coding-strand guides) is largest for family A among the families run within genes on Cui (`g_transcription`, `d_supercoiling`, `d_mechanics`, `i_shape`; on Rousset only family A was run), and adding A to `g_transcription`, which carries the guide's relative position in its gene, still leaves **+0.096** within genes (seed sd 0.012; between genes +0.006). The cutting screen decomposes as +0.275 between genes and +0.155 overall (30,203 genic guides, 4,013 genes, ten seeds; `strand.py` (`--parts cutting`); within-gene seed sd 0.0007). Raw dCas9 gains are not smaller but start from within-gene bases of 0.17 and 0.08 against 0.62, so they are scaled by the attainable gain and by variance explained:

| within-gene ranking | genes | base | with A | Δ within [95% bootstrap over genes] | ceiling | share of attainable gain | Δρ² |
|---|---:|---:|---:|---:|---:|---:|---:|
| cutting screen | 4,013 | 0.618 | 0.703 | +0.085 [+0.078, +0.090] | 0.946 (Finding 6) | **26%** | +0.113 |
| Cui (mean of both arms), coding strand | 206 | 0.172 | 0.274 | +0.102 [+0.081, +0.164] | ≈ 0.93 | **13%** | +0.046 |
| E75 arm ("Rousset"), coding strand | 182 | 0.083 | 0.196 | +0.114 [+0.084, +0.189] | ≥ 0.864 | **15%** | +0.032 |

The dCas9 effect is between a quarter and a half of the cutting one (half by share of the attainable gain, 28–40% by variance explained). The dCas9 ceiling is the Spearman–Brown reliability of the mean of the screen's two arms (the strains LC-E18 and LC-E75, which differ in dCas9 expression; within-gene ρ 0.768 over 2,289 guides), square-rooted. *Three limits keep this out of Parts 1–3:* (i) the two arms are not true replicates, so a systematic difference between them would raise the true ceiling and shrink the dCas9 share; (ii) the E75 arm has one readout, so its ceiling is a lower bound from its agreement with the E18 arm on the same guides (within-gene ρ 0.805 over 1,759 guides) and its share is at most 15%, and its row and Cui's are two labels of one experiment, not two screens; (iii) a within-gene CRISPRi effect could be gene-local transcription context, which the `g_transcription` control narrows without closing. Whether the gain is a local flank property or a position effect that relative position does not capture is not identified; settling it needs a design that varies flank composition at a fixed guide, which no existing screen provides.

### A second ceiling: an independent screen agrees with Cui's far less than replicates do

**The transferable-signal ceiling across libraries and laboratories is ≈ 0.77–0.79 inside essential genes and ≈ 0.27–0.30 outside them, against 0.946 for replicates (Finding 6), which is a within-experiment quantity.** The Rousset and Cui rows are the two dCas9-expression arms (strains LC-E75, with dCas9 at a lower concentration, and LC-E18) of one experiment, on the 59,246 guides that Rousset et al. retained (`E75_Rousset.tsv` is the `fit75` and `E18_Cui.tsv` the `fit18` column of Cui's `screen_data.csv`); Wang is a different library, laboratory and protocol sharing 6,139 guides with them, so the two Wang rows are one independent screen set against two arms of one experiment (`crispri_screens.py --agreement`, `results/crispri_screen_agreement.csv`; n 6,139–59,246 all guides, 529–4,640 essential, 5,610–54,606 non-essential).

| pair | what differs | all guides | essential genes | non-essential |
|---|---|---:|---:|---:|
| E75 arm ("Rousset") vs E18 arm (Cui) | dCas9 expression level only (two arms of one experiment) | +0.689 | **+0.902** | +0.659 |
| E75 arm ("Rousset") vs Wang | library, laboratory, protocol | +0.429 | **+0.790** | +0.298 |
| E18 arm (Cui) vs Wang | the same | +0.401 | **+0.765** | +0.271 |

A model fitted on one screen cannot score above that screen's agreement with another, so the smaller ceiling binds any transfer claim. *Limit:* these are CRISPRi screens, so it bounds the dCas9 claims and not the cutting ones; only one independent screen (Wang) enters; the curated Cas9 cohort has only its own replicates.

**Method result: supply the assay's dominant covariate before reading a subgroup.** A family that holds a lossy copy of a screen's dominant covariate looks powerful: family A was worth +0.27 inside the essential genes and +0.007 to +0.025 once strand was supplied (Part 2, method list); the same strand term dominates the non-sequence features of a published human base-editing model (Part 5). Scope: the predictor is for cleavage, and a CRISPRi user needs a guide-sequence model.

## Appendix C0 — Reproducibility: versions, seeds and protocol

| | |
|---|---|
| software | Python 3.12.15; numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1; LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10; SHAP 0.52.0; PyTorch 2.14.1 (the CNN controls) |
| machine, for every timing in this report | Apple M5, 10 cores, 16 GB, macOS 26.6.2 |
| seeds | `config.SEEDS = (41, 42, 43, 44, 45)`; `config.N_SPLITS = 5`; every seed is passed explicitly to the model, the selector and the fold splitter, so no arm depends on global RNG state |
| replication | ablations five seeds × five folds (25 fits per arm); the tuned-model numbers (the 70/30 search, `tuned_ablation.py --design family_a`), the Appendix B4 shape arms, the CRISPRi screens, the shift-distance null and the GC-strata test three seeds (41–43) × five folds; the within-gene decompositions ten seeds; head-to-head against crisprHAL 2 seed 41 only, because each of their fits costs ~21 min |
| selector | `config.SELECTOR = "xgboost"`, a throwaway model that only ranks columns; fitted **inside the training split of each fold**, never on the full data |
| resumability | each fold is written to `results/folds_<tag>.csv` as it finishes and a recorded fold is skipped, so a crash loses one fold rather than an experiment |

**Timings.** Every millisecond figure in this report (featuriser costs, end-to-end tables, ratios against crisprHAL 2) is wall-clock on **one thread** of the machine above, median of repeated calls after a warm-up, with `OMP_NUM_THREADS` and the three sibling variables pinned to 1 so no arm uses more cores than another. The **ratios travel and the absolutes do not**: another machine will move 0.036 and 1.214 ms together, and the 34× with much less. The audit compares each quoted timing with the measurement stored in `results/`, so re-measuring on other hardware makes those lines fail, correctly; re-quote them rather than widening the tolerance. A module that measures at import can silently re-measure them under load (`EXTENSION_LOG.md` §22.4).

## Appendix C — Feature sets and data sources

### Eleven folders are not eleven kinds, and which nulls are worth keeping

**Six of the eleven feature folders bundle measurements of different things, and in four cases the bundling hides the result.** Δρ is the curated-cohort increment over the 6,169-column base (Appendix B3); published-matrix figures (Noshay et al.'s Supplementary Table 2: 13,880 rows by 6,232 columns, lying in 1,222 genes; their Table 1 lists 40,468 *E. coli* sgRNAs, 32,374 for training) are uniformly smaller and rank almost identically (ρ = 0.97), for the reason Appendix B2a gives. Guide-derived folders are deterministic functions of the 20-mer, so they can add performance but not information (Finding 1). A null earns its place when it kills a mechanism someone would otherwise have to assume, not when it is an artefact of how the cohort was built: five rows below do (marked **Null**), one does not.

| kind | set | columns | what it carries; what a null rules out | Δρ |
|---|---|---:|---|---:|
| flank | `a_flank` | 348 | DNA outside the protospacer in three layers: **near-cut sequence** (10 bases before the protospacer to 15 after the PAM, 260 columns) +0.074 alone (layer increments from LightGBM, seed 41, `headtohead.csv`; the near-cut arm has three of the five folds, +0.073 against the base on those folds); **crisprHAL's own ±189/166 nt span** +0.055 more; **50/250/500/1,000 nt windowed composition** +0.035 more. Only the windows are information crisprHAL 2's input cannot reach (Finding 2) | **+0.164** |
| guide-derived | `b_energy` | 10 | R-loop free-energy budget (hybridisation, DNA opening, guide self-folding, net binding, distance from the −64.5 to −47.1 kcal/mol window where efficient guides cluster) | +0.019 |
| guide-derived | `c_folding` | 9 | secondary structure of the transcript made: spacer MFE and spacer-against-scaffold pairing | +0.014 |
| guide-derived | `f_methylation` | 17 | Dam `GATC` and Dcm `CCWGG` motifs by position relative to the PAM. **Null:** a ~10× lower SaCas9 activity at sites whose PAM carries a Dam `GATC` (Ham et al., 2026, who report it for a Dam site in the PAM, 5′-NNGGAT[C]-3′, in *E. coli* and *C. rodentium*) does not reproduce for SpCas9 on this screen | +0.008 |
| target mechanics | `d_mechanics` | 30 | **two sets**: 22 columns of protospacer duplex thermodynamics (**+0.011** for the 20 the guide determines linearly); 8 columns of the same on the flanks at ±25 and ±100 nt **+0.097**, which is family A under another name | +0.099 |
| target mechanics | `i_shape` | 152 | dinucleotide-step deformability of the flanks; **+0.120 alone**, **+0.0009** once `a_flank` is present. **Null:** shape and composition are one channel twice, a statement about DNA descriptors and not about Cas9 | +0.120 |
| locus | `d_supercoiling` | 26 | GapR-seq positive-supercoiling signal (GSE152880), several window widths. **Null, the most useful here:** survives replacing the track with the **untagged control** (+0.0087 alone against +0.0111 for the ChIP track, published matrix, 100 kb blocks), so it measures sequencing read density; this answers the "unknown chromosomal factors" of Guo et al. (2018) for this family, and was caught only because the experiment shipped an untagged control (Appendix B3) | +0.139 → artefact |
| locus | `e_nucleoid` | 20 | **two sets**: Hi-C 3D position; NAP dependence (contacts per bin in wild type against H-NS, HU, Fis, MatP, MukB and ZapB mutants) | +0.072 |
| locus | `g_transcription` | 19 | **three sets, one pays**: gene geometry **+0.0393**; PRECISE-1K expression **+0.0213**; essentiality **+0.00004**. **Null:** the essentiality term adds +0.00004 on this cohort; Guo et al. (2018) reported gene expression as unrelated to activity, and the **expression** term here is not: one correction | +0.058 |
| cohort-defined-away | `h_offtarget` | 13 | **7 live, 6 dead**: protospacer copy number plus 6 near-cognate-mismatch columns identically zero because the curated cohort keeps only uniquely-targeting guides. **Not a null:** the question was removed before the measurement, and quoting it beside the others would imply a specificity conclusion the data cannot support (Part 2) | +0.001 |
| periodicity | `j_periodicity` | 24 | GC skew and helical-repeat phasing. **Null:** −0.001 on the published matrix (+0.0004 on top of family A) and +0.011 on the curated cohort (64% of it survives gene-grouping); bounds the most mechanistically motivated explanations for a long-range, strand-asymmetric effect at that size | +0.011 |

`b_energy` and `c_folding` are not independent witnesses: one column is the same quantity computed twice (Pearson 0.9993), so the folding evidence is two measurements, not three (Appendix B). The three locus folders are close to one variable measured three ways (jointly +0.145).

### Teammates' results on the published 13,880-row matrix, quoted in R² (not comparable with the Δρ column)

| set | who | result |
|---|---|---|
| **gene essentiality** | Jacky | **+0.0037 R²** over 0.2927, **selected in 5/5 folds**: the only non-QCT idea on that matrix with a consistent gain, and not a function of the 20-mer |
| **orientation controls** | Jacky | addresses Part 4 item 3: curated RegulonDB TSS replaces nearest gene as the orientation variable (Δ 0.051 [0.020, 0.082] unmatched; 0.025 [−0.003, +0.053], p = 0.04, after distance/GC matching); the nearest-gene version is largely a distance/GC cohort confound (Δ 0.007, P = 0.34 after matching); block bootstrap over 47 genomic blocks, promoter/RBS separated, operon compared |
| QCT variants | Joshua | seed non-additive ~+0.005 R² (absolute-difference forms carry more SHAP than signed, 1.09 against 0.92); log-ratio forms ~+0.005, best seed 0.3015; regional summaries +0.0013 against a permuted control at 0.2857; PAM-weighted, profile-shape, coupling and HOMO–LUMO asymmetry −0.0010 to +0.0006, null |

The non-additive families are **independent evidence for Finding 4**: they gain from exactly the higher-order structure the k-mer ladder found worth +0.0035 ρ at the trimer step once the selector was removed (12 of 15 folds, published matrix; the tetramer step on top is −0.0028, net +0.0007).

### Data added after the freeze

| data | source | size | used for |
|---|---|---|---|
| 24 efficiency tables for human, mouse, zebrafish, rat, fly and worm (`*.scores.tab`, plain text; read as data, never executed) | the CRISPOR benchmark collection, github.com/maximilianh/crisporPaper, `effData/`, which redistributes the supplementary tables of the original papers (Doench et al. 2014 and 2016, Chari et al. 2015, Xu et al. 2015, Wang et al. 2015, Hart et al. 2015 (named `hart2016` in the collection), Moreno-Mateos et al. 2015 and others; Appendix D) | about 7 MB, in `external_data/crispor_effdata/` (not committed) | Finding 7, the eukaryote comparison (`eukaryote_rules.py`) |

## Appendix D — Sources (APA 7th edition)

Alkan, F., Wenzel, A., Anthon, C., Havgaard, J. H., & Gorodkin, J. (2018).
CRISPR-Cas9 off-targeting assessment with nucleic acid duplex energy parameters.
*Genome Biology*, *19*(1), 177. https://doi.org/10.1186/s13059-018-1534-x
— CRISPRoff, the duplex-energy model behind family B (`b_energy`), run with the authors' own pipeline and parameters.

Browne, T. S., Edgell, D. R., & Gloor, G. B. (2026). Better data for better
predictions: Data curation improves deep learning for sgRNA/Cas9 prediction.
*PeerJ*, *14*, e20706. https://doi.org/10.7717/peerj.20706
— the state of the art; code and curated label re-run here as the baseline.

Chari, R., Mali, P., Moosburner, M., & Church, G. M. (2015). Unraveling CRISPR-Cas9
genome engineering parameters via a library-on-library approach. *Nature Methods*,
*12*(9), 823–826. https://doi.org/10.1038/nmeth.3473
— the HEK293T library of the CRISPOR collection used in Finding 7.

Chiu, T.-P., Comoglio, F., Zhou, T., Yang, L., Paro, R., & Rohs, R. (2016).
DNAshapeR: an R/Bioconductor package for DNA shape prediction and feature
encoding. *Bioinformatics*, *32*(8), 1211–1213.
https://doi.org/10.1093/bioinformatics/btv735
— source of the pentamer shape table (`src/TableCompiler.cpp`, read as text) used in
Appendix B4; the table lumps Monte Carlo, X-ray and molecular-dynamics values.

Clarke, R., Heler, R., MacDougall, M. S., Yeo, N. C., Chavez, A., Regan, M.,
Hanakahi, L., Church, G. M., Marraffini, L. A., & Merrill, B. J. (2018). Enhanced
bacterial immunity and mammalian genome editing via RNA-polymerase-mediated
dislodging of Cas9 from double-strand DNA breaks. *Molecular Cell*, *71*(1),
42–55.e8. https://doi.org/10.1016/j.molcel.2018.06.005
— translocating RNA polymerase dislodges Cas9 from its cut, from one direction only:
the transcription × strand mechanism behind family G's interaction terms.

Corsi, G. I., Qu, K., Alkan, F., Pan, X., Luo, Y., & Gorodkin, J. (2022). CRISPR/Cas9
gRNA activity depends on free energy changes and on the target PAM context. *Nature
Communications*, *13*(1), 3006. https://doi.org/10.1038/s41467-022-30515-0
— efficient guides sit in a window of binding free energy (roughly −64.5 to −47.1 kcal/mol) and fall
off on both sides, which is why `b_energy` also carries the distance from that window.

Cui, L., Vigouroux, A., Rousset, F., Varet, H., Khanna, V., & Bikard, D. (2018).
A CRISPRi screen in *E. coli* reveals sequence-specific toxicity of dCas9.
*Nature Communications*, *9*(1), 1912. https://doi.org/10.1038/s41467-018-04209-5
— the dCas9 CRISPRi screen used as the final untouched test (Appendix B5); its two
screens (strains LC-E18 and LC-E75, strong and reduced dCas9 expression) also set the ceiling range quoted in Appendix B.

DeWeirdt, P. C., McGee, A. V., Zheng, F., Nwolah, I., Hegde, M., & Doench, J. G.
(2022). Accounting for small variations in the tracrRNA sequence improves sgRNA
activity predictions for CRISPR screening. *Nature Communications*, *13*(1), 5255.
https://doi.org/10.1038/s41467-022-33024-2
— Rule Set 3, named in Part 5's scope paragraph; its own pairwise benchmark (Spearman on held-out
datasets) ranked CRISPRon first among the four published models it compared.

Doench, J. G., Fusi, N., Sullender, M., Hegde, M., Vaimberg, E. W., Donovan, K. F.,
Smith, I., Tothova, Z., Wilen, C., Orchard, R., Virgin, H. W., Listgarten, J., &
Root, D. E. (2016). Optimized sgRNA design to maximize activity and minimize
off-target effects of CRISPR-Cas9. *Nature Biotechnology*, *34*(2), 184–191.
https://doi.org/10.1038/nbt.3437
— Rule Set 2 (Azimuth), one of the six published predictors that orient the eukaryotic
datasets in Finding 7 (`eukaryote_rules.py`); its Doench 2016 library is one of those datasets.

Doench, J. G., Hartenian, E., Graham, D. B., Tothova, Z., Hegde, M., Smith, I.,
Sullender, M., Ebert, B. L., Xavier, R. J., & Root, D. E. (2014). Rational design of
highly active sgRNAs for CRISPR-Cas9-mediated gene inactivation. *Nature
Biotechnology*, *32*(12), 1262–1267. https://doi.org/10.1038/nbt.3026
— the human and mouse libraries of the CRISPOR collection called `doench2014` in Finding 7.

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

Hanna, R. E., Hegde, M., Fagre, C. R., DeWeirdt, P. C., Sangree, A. K., Szegletes, Z.,
Griffith, A., Feeley, M. N., Sanson, K. R., Baidi, Y., Koblan, L. W., Liu, D. R., Neal, J. T.,
& Doench, J. G. (2021). Massively parallel assessment of human variants with base editor
screens. *Cell*, *184*(4), 1064–1080.e20. https://doi.org/10.1016/j.cell.2021.01.012
— the base-editing screen whose data Orro et al. model (as they describe it).

Haeussler, M., Schönig, K., Eckert, H., Eschstruth, A., Mianné, J., Renaud, J.-B.,
Schneider-Maunoury, S., Shkumatava, A., Teboul, L., Kent, J., Joly, J.-S., &
Concordet, J.-P. (2016). Evaluation of off-target and on-target scoring algorithms
and integration into the guide RNA selection tool CRISPOR. *Genome Biology*,
*17*(1), 148. https://doi.org/10.1186/s13059-016-1012-2
— the source of the replicate correlations (0.71–0.77) that the field's ∼0.7 ceiling
is built on (Part 5), and the CRISPOR collection of benchmark datasets used in Finding 7; it also
shows that scores evaluated on their own training data are inflated ('algorithmic overfitting').

Ham, D. T., Browne, T. S., Banglorewala, P. N., Wilson, T. L., Michael, R. K.,
Gloor, G. B., & Edgell, D. R. (2023). A generalizable Cas9/sgRNA prediction
model using machine transfer learning with small high-quality datasets. *Nature
Communications*, *14*(1), 5514. https://doi.org/10.1038/s41467-023-41143-7
— the *C. rodentium* screen used for cross-organism transfer.

Ham, D. T., Browne, T. S., Zhang, C. Q., Foo, G. W., Uruthirapathy, A. S., Gloor, G. B.,
& Edgell, D. R. (2026). Machine learning reveals sequence and methylation determinants of
SaCas9–PAM interactions in bacteria. *Nucleic Acids Research*, *54*(2), gkaf1520.
https://doi.org/10.1093/nar/gkaf1520
— reports about 10-fold lower SaCas9 activity at a PAM carrying a Dam `GATC` in *E. coli* and
*C. rodentium*, which `f_methylation` tests for SpCas9 and does not reproduce.

Hart, T., Chandrashekhar, M., Aregger, M., Steinhart, Z., Brown, K. R., MacLeod, G., Mis, M.,
Zimmermann, M., Fradet-Turcotte, A., Sun, S., Mero, P., Dirks, P., Sidhu, S., Roth, F. P.,
Rissland, O. S., Durocher, D., Angers, S., & Moffat, J. (2015). High-resolution CRISPR
screens reveal fitness genes and genotype-specific cancer liabilities. *Cell*, *163*(6),
1515–1526. https://doi.org/10.1016/j.cell.2015.11.015
— the HCT116 and HeLa libraries (`hart2016-*` in the CRISPOR collection) of Finding 7.

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

Moreb, E. A., & Lynch, M. D. (2021). Genome dependent Cas9/gRNA search time
underlies sequence dependent gRNA activity. *Nature Communications*, *12*(1), 5034.
https://doi.org/10.1038/s41467-021-25339-3
— a retrospective analysis of 44 published gRNA datasets across species; read from its abstract
only. It reports that prediction algorithms do not carry between species and proposes that
sequence acts mainly through the time Cas9 takes to find its target, in a genome-dependent way
(Finding 7, the hypotheses table).

Moreno-Mateos, M. A., Vejnar, C. E., Beaudoin, J.-D., Fernandez, J. P., Mis, E. K.,
Khokha, M. K., & Giraldez, A. J. (2015). CRISPRscan: designing highly efficient sgRNAs for
CRISPR-Cas9 targeting in vivo. *Nature Methods*, *12*(10), 982–988.
https://doi.org/10.1038/nmeth.3543
— the zebrafish library of the CRISPOR collection used in Finding 7.

Noshay, J. M., Walker, T., Alexander, W. G., Klingeman, D. M., Romero, J.,
Walker, A. M., Prates, E., Eckert, C., Irle, S., Kainer, D., & Jacobson, D. A.
(2023). Quantum biological insights into CRISPR-Cas9 sgRNA efficiency from
explainable-AI driven feature engineering. *Nucleic Acids Research*, *51*(19),
10147–10161. https://doi.org/10.1093/nar/gkad736
— the feature matrix decoded here; their iRF re-runs in our harness at Pearson
0.4881 against their reported 0.5019, on a third of their rows.

Orro, A., Consiglio, A., Curci, M. I., Scichilone, M., Hasin, F., Minervini, M.,
Mencar, C., De Bellis, G., Cocola, C., Pelucchi, P., & Selmi, T. (2026). Machine
learning-based prediction of base editor sgRNA fitness score. *Scientific Reports*.
Advance online publication (accelerated preview), 10 November 2026;
volume, issue and article number not yet assigned. https://doi.org/10.1038/s41598-026-79920-1
(preprint: Research Square, 10 April 2026, https://doi.org/10.21203/rs.3.rs-9224117/v1; both versions read)
— base-editing fitness prediction (the Hanna et al. 2021 library, about 11,000 sgRNAs over 47 genes, five cell
lines), gradient-boosted trees with leave-one-gene-out validation. Its SHAP position profile concentrates at the
PAM-proximal end, as this project's gain-by-position result does; gene essentiality carries marginal weight
where we find it null, a difference the report attributes to the two labels (Part 5; untested). The preprint's
SHAP values (peak at position 20) and its gene-strand result differ from the published version and are not relied on.

Pasi, M., Maddocks, J. H., Beveridge, D., Bishop, T. C., Case, D. A., Cheatham, T.,
Dans, P. D., Jayaram, B., Lankas, F., Laughton, C., Mitchell, J., Osman, R., Orozco, M.,
Pérez, A., Petkevičiūtė, D., Spackova, N., Sponer, J., Zakrzewska, K., & Lavery, R.
(2014). μABC: A systematic microsecond molecular dynamics study of tetranucleotide
sequence effects in B-DNA. *Nucleic Acids Research*, *42*(19), 12272–12283.
https://doi.org/10.1093/nar/gku855
— tetranucleotide stiffness from microsecond molecular dynamics; listed in Part 5 among the
measurements not run (no download page was found).

Rousset, F., Cui, L., Siouve, E., Becavin, C., Depardieu, F., & Bikard, D. (2018).
Genome-wide CRISPR-dCas9 screens in *E. coli* identify essential genes and phage host
factors. *PLOS Genetics*, *14*(11), e1007749. https://doi.org/10.1371/journal.pgen.1007749
— its dCas9 screen is the E75 arm of Cui et al.'s experiment (Appendix B5), not an independent screen.

SantaLucia, J., & Hicks, D. (2004). The thermodynamics of DNA structural motifs.
*Annual Review of Biophysics and Biomolecular Structure*, *33*(1), 415–440.
https://doi.org/10.1146/annurev.biophys.32.110601.141800
— nearest-neighbour thermodynamic parameters.

Sastry, A. V., Gao, Y., Szubin, R., Hefner, Y., Xu, S., Kim, D., Choudhary, K. S.,
Yang, L., King, Z. A., & Palsson, B. O. (2019). The *Escherichia coli*
transcriptome mostly consists of independently regulated modules. *Nature
Communications*, *10*(1), 5536. https://doi.org/10.1038/s41467-019-13483-w
— the iModulon decomposition behind the PRECISE expression compendium.

Spearman, C. (1904). The proof and measurement of association between two things.
*The American Journal of Psychology*, *15*(1), 72–101. https://doi.org/10.2307/1412159
— the correction for attenuation behind the square-root ceiling (Part 5).

Wang, T., Birsoy, K., Hughes, N. W., Krupczak, K. M., Post, Y., Wei, J. J., Lander, E. S., &
Sabatini, D. M. (2015). Identification and characterization of essential genes in the human
genome. *Science*, *350*(6264), 1096–1101. https://doi.org/10.1126/science.aac7041
— the library called `wang2015` in the CRISPOR collection (Finding 7); its guides were designed with a
scoring algorithm and carry little signal there (Haeussler et al., 2016), so it is the weakest anchor.

Wang, T., Guan, C., Guo, J., Liu, B., Wu, Y., Xie, Z., Zhang, C., & Xing, X. (2018). Pooled
CRISPR interference screening enables genome-scale functional genomics study in bacteria
with superior performance. *Nature Communications*, *9*(1), 2475.
https://doi.org/10.1038/s41467-018-04899-x
— the independent *E. coli* CRISPRi library ("Wang", Appendix B5), a different laboratory from Cui and Rousset.

Xiang, X., Corsi, G. I., Anthon, C., Qu, K., Pan, X., Liang, X., Han, P., Dong, Z.,
Liu, L., Zhong, J., Ma, T., Wang, J., Zhang, X., Jiang, H., Xu, F., Liu, X., Xu, X.,
Wang, J., Yang, H., . . . Luo, Y.
(2021). Enhancing CRISPR-Cas9 gRNA efficiency prediction by data integration and
deep learning. *Nature Communications*, *12*(1), 3238.
https://doi.org/10.1038/s41467-021-23576-0
— CRISPRon; a CNN over a 30-nt window plus a computed gRNA–DNA binding energy, with no
experimental track; cited in Part 5 for its reported ρ (0.80 internal, ≈0.46–0.68 external).

Xu, H., Xiao, T., Chen, C.-H., Li, W., Meyer, C. A., Wu, Q., Wu, D., Cong, L., Zhang, F.,
Liu, J. S., Brown, M., & Liu, X. S. (2015). Sequence determinants of improved CRISPR sgRNA
design. *Genome Research*, *25*(8), 1147–1157. https://doi.org/10.1101/gr.191452.115
— the HL60, KBM7 and mESC libraries of the CRISPOR collection (reprocessed screens of Wang et al., 2014,
and Koike-Yusa et al., 2014) used in Finding 7.

Yu, Y., Gawlitt, S., de Andrade e Sousa, L. B., Merdivan, E., Piraud, M.,
Beisel, C. L., & Barquist, L. (2024). Improved prediction of bacterial CRISPRi guide
efficiency from depletion screens through mixed-effect machine learning and data
integration. *Genome Biology*, *25*(1), 13.
https://doi.org/10.1186/s13059-023-03153-y
— the closest methodological neighbour: gene-wise cross-validation (for the mixed-effect guide-efficiency model) and a mixed-effect
model separating guide efficiency from the targeted gene, on bacterial CRISPRi. Their
guide features alone reach ρ ≈ 0.21 (sequence) to ≈ 0.25 (with distance and thermodynamics) and gene-level
features take it to ≈ 0.66 (depletion, guide-wise cross-validation, one screen), the same guide/gene
decomposition this project makes (Part 5).

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

## Appendix E — The paper's Results, Discussion and appendices, bullet by bullet

The paper (the point-form write-up in the project's Google Doc) is the concise version of this report. This appendix repeats its Results, Discussion and appendix bullets as the Doc states them (regenerated from the Doc text of 25 November 2026), so that every number the paper states is also stated here; the paper's protocols (random rows, whole genes, sealed genes, development and confirmation genes, crisprHAL 2's own splits, an unseen genome) are listed in its Appendix A3 and in this report's Part 4; where a Finding above says something different, the Finding was patched to agree with this appendix on 25 November 2026 (the record is in `EXTENSION_LOG.md`, sections 29 to 33). The paper's Methods are this report's Part 4 and Appendices C0 and C; its references are in Appendix D (six more are listed at the end of this appendix). Figure numbers are the paper's (main Figures 1 to 4, appendix Figures A1 to A4; files `p_fig*.png` in `results/figures/`, drawn by `sgrna/paper/figures_paper.py`).


### Results


#### The 20-mer regenerates 99% of the published matrix

- 6,169 of the published matrix's 6,232 columns (99.0%) regenerate exactly from the protospacer, and 412 columns computable for any guide match all 6,232 to 0.00003 ρ [−0.0029, +0.0030] (published matrix, random rows)

#### Flank features add 0.161 in ρ

- Flank features raise ρ from 0.547 ± 0.013 to 0.707 ± 0.009: a gain of +0.161 (sd 0.010 over the 15 folds, positive in 15 of 15; default model, whole genes held out). As a check on optimism, sealing 20% of the guides by whole gene and re-deriving selection on the rest gives +0.156 (Appendix A3); scored with eSpCas9 labels the sealed genes give +0.103 [+0.100, +0.107] (positive in 15 of 15 sealings). Inside genes alone +0.085 [+0.078, +0.090] survives (ten seeds)
- Two GC columns, the GC fraction of the W nt before the protospacer and of the W nt after the PAM, give +0.106 at W = 200 nt, the highest of 14 windows (10 to 5,000 nt), are within 0.01 of it from 150 to 500 nt, and give +0.042 at 5 kb (Figure 1b). The purine fraction (A+G) and the amino fraction (A+C) in the same windows give no gain (−0.0014 to 0.0000) and the four bases separately at most +0.103 (Appendix A1)
- Reading the flank features from the guide nearest to an offset of 100 bp, 300 bp, 1 kb or 3 kb (seed 41, five folds) keeps 68%, 57%, 32% and 6% of the gain, and from 10 kb, 100 kb or 1 Mb away none of it (gains −0.0001, −0.0001 and −0.0008; Figure 1a)

> Figure 1. How far the flank signal reaches (default model, seed 41, five folds; error bars: sd over folds). (a) Gain in ρ over the 20-mer when each guide is given the 348 flank features of the guide whose protospacer lies the stated distance away (circular genome). (b) Gain from the two GC columns of the text as W grows; dashed: the full 348-column set.


#### The shape of the flank effect

- With the target's own GC held fixed, flank GC correlates +0.15 to +0.20 with activity at every window from 50 to 1,000 nt, equal before and after (Appendix A1). The lowest decile of flank GC (mean −3.154) sits 3.16 label units below the rest, while deciles 3 to 10 are within 0.15 of zero (Figure 2a); inside genes the dependence is graded. The step replicates for a second enzyme at the same loci and is not established in a second genome
- The dependence on flank GC differs with the base at guide position 20, next to the PAM: Spearman correlation 0.219 [0.186, 0.246] with A, 0.119 [0.089, 0.146] with G, 0.184 with T and 0.169 with C (genic guides; Figure 2b)
- In a model of 104 named columns (88% of the full model's ρ), SHAP puts 83% of the attribution on the 80 guide indicators (61% of it on positions 15 to 20) and 17% on the 24 flank columns, 79% of it GC; the lowest decile of flank GC contributes −0.78 SD (Appendix A2)

> Figure 2. The shape of the flank effect. (a) Mean cutting score by decile of flank GC (250 nt each side averaged; 30,203 genic guides; bars: 95% interval from resampling genes). (b) Mean score in quintiles of flank GC by the base at guide position 20, with the Spearman correlation within each base.


#### The replicate ceiling is 0.946 and the best model reaches 0.73

- The two replicates agree at ρ 0.8097 (33,871 guides; Figure 3); Spearman–Brown gives a two-replicate score a reliability of 0.8948, so the highest attainable ρ is its square root, 0.946 [0.944, 0.947]. The best model reaches 0.7314 (tuned, whole-gene folds, 30,203 genic guides; 0.7215 on the 1,204 genes the settings never touched), a gap of 0.21 to 0.22
- A ceiling of about 0.7 is quoted for the field (Haeussler et al., 2016, from replicate correlations of 0.71 to 0.77; repeated by Konstantakos et al., 2022); for a score that is the mean of two replicates, as crisprHAL 2's is, the same numbers give 0.911 to 0.933

> Figure 3. Replicate agreement and what it caps. (a) Cutting score of replicate pair 1 against pair 2 for the 33,871 guides with at least 73 unselected counts (counts per million; hexagons shaded by guides). (b) ρ of crisprHAL 2 (published settings), BoostMEC (released hyperparameters) and SLICER (default and tuned) on the five whole-gene folds of seed 41 (mean ± sd over folds); dotted lines: the field's ceiling (0.7) and the attainable ceiling of the released score (0.946).


#### SLICER, crisprHAL 2 and BoostMEC: accuracy and cost

- SLICER is gradient-boosted trees (XGBoost; Chen & Guestrin, 2016) on base features that describe the guide (which nucleotide sits where, quantum-chemical values for them) and flank features that describe the DNA around the cut (GC and purine content in windows of 50 to 1,000 nt each side, the bases next to the guide and its PAM; Appendix A4), keeping the most informative columns, chosen inside each training fold; the tuned model's settings were searched on 70% of genes and confirmed on the other 30%. It is compared with crisprHAL 2 (Browne et al., 2026), the state of the art, a network that reads 378 nt around the guide, and with BoostMEC (Zarate et al., 2022), boosted trees on 149 hand-built features of a 30-nt window, trained on human cells; both run as released, and BoostMEC's design is also retrained on the E. coli cohort (Appendix A6)
- Accuracy: SLICER has a slight edge over crisprHAL 2 in ρ. As released, on crisprHAL 2's own random splits, tuned SLICER leads by +0.0286 [+0.0239, +0.0331] on eSpCas9 and +0.0226 [+0.0139, +0.0314] on SpCas9 (95% bootstrap intervals over test guides); the other accuracy measures agree (Appendix A2)
- On the whole-gene folds used everywhere else, with crisprHAL 2 retrained at its published settings on the same folds (one seed), crisprHAL 2 reaches ρ 0.693 against 0.707 for SLICER's default model (+0.014, sd 0.006 over the folds, ahead in 5 of 5) and 0.723 for the tuned model (+0.031, ahead in 5 of 5).
- On random-row folds (seed 41, SLICER's original LightGBM version), inside crisprHAL 2's 378-nt span its convolutions beat SLICER's windowed composition by 0.025, and the 250 to 1,000 nt windows beyond its input are worth +0.035 (the default model's lead on those folds is +0.0115). Given wider input (batch 256) and no re-tuning, crisprHAL 2 falls below its own 378-nt score in all five folds, by 0.020 at 1,023 nt and 0.039 at 2,023 nt (Appendix Figure A2)
- BoostMEC, retrained on the E. coli cohort from the features of its own pipeline, reaches ρ 0.619 ± 0.013 on the folds of Figure 3b (0.620 ± 0.009 over three seeds with its released hyperparameters, −0.087 against SLICER's default model; 0.623 after its own search; Appendix A6) and scores ρ −0.066 as released, trained on human cells
- Cost, quiet laptop CPU (one thread, raw sequence to score; Figure 4a): per guide at a batch of 100, SLICER's default model takes 0.0726 ms and its tuned model 0.0885 ms, crisprHAL 2 1.37 ms (18.8 [18.8, 18.9] times the default model; ratio of medians [95% interval over repeats]; 24.5 [24.4, 24.5] for one guide, 15.4 [15.3, 15.4] at 10,000) and BoostMEC 3.07 ms (42.3 [42.2, 42.5] times; 5.35 ms per guide run as its authors ship it, Appendix A6)
- On one Colab T4 session (Figure 4b; a shared VM, not a quiet machine), crisprHAL 2 takes 0.311 ms per guide at 10,000 from raw contexts (3.9 times SLICER's whole path on the laptop; two machines), and its forward pass with the batch on the card 0.0837 ms, 13 times the forward pass of SLICER's trees on the same VM's CPU (0.00636 ms; Appendix A6)
- Fitting (wall-clock, ten laptop cores; Figure 4c): SLICER's default model 12.8 s, tuned 24.5 s; BoostMEC with its released hyperparameters 13.2 s; crisprHAL 2 17.3 min (81 times the default model), or 5.4 min on a T4 (25 times; SLICER's shape fits in 3.20 s on the same T4)
- Stored model: SLICER's default model 6,338 leaves (0.68 MB), tuned 113,540 (10.2 MB), BoostMEC's released model 136,142 in 1,582 trees (11.5 MB), crisprHAL 2 790,533 parameters (9.6 MB)

> Figure 4. The cost; bars: median with a 95% bootstrap interval over repeats, log scales. (a) Quiet laptop CPU, one thread: time to score one guide from the raw sequence at batches of 1, 100 and 10,000. (b) One Colab T4 session: crisprHAL 2 from raw contexts and its forward pass with the batch on the card, and the forward pass of SLICER's trees on the same VM's CPU. (c) One fit, wall-clock seconds (BoostMEC's second bar adds the ten-fold cross-validation that fixes its tree count).


### Discussion

- What the flank effect is remains open. It is a bulk property of a few hundred bases, specific to GC (the purine and amino fractions give no gain and the four bases separately do no better than GC; Appendix A1), with a gain highest at 200 nt and within 0.01 of it from 150 to 500 nt. That fits a local property of DNA such as accessibility or flexibility, or a protein that binds AT-rich DNA; nothing here tells them apart (Moreb and Lynch, 2021, tie guide-sequence preferences to target search time; untested for flank composition)
- The controls point to the guide's own flank and not to an artefact: the gain is the same among guides whose surroundings are unique as on all guides and larger where the genome is repetitive, and read depth does not account for it (Appendix A3); reading the flank features from another site removes it beyond about 3 kb (the shifted arms also lose the 260 cut columns, so part of the loss at 100 bp is theirs); and it is not only gene ranking (+0.085 inside genes) nor shown to be adaptive overfitting (the sealed-gene drop is small). Not tested: the Cas9-selected arm's sequencing depth, the label's own numerator
- The effect is a penalty for AT-rich surroundings, larger than any single guide base: a step and not a slope, costing 0.78 SD in the lowest decile of flank GC in the 104-column model, 2.4 times the largest single-base effect (G at position 20, −0.33 SD). A flank × guide-score interaction, predicted to make the two act as a gate, fits only slightly better than an additive model on whole-gene folds (R² +0.0035, better in 15 of 15 folds; ρ +0.0007, 14 of 15): negligible
- Short flanks were known to help (Browne et al., 2026); what is new is the reach beyond 378 nt and that the gain is carried by GC. The windows beyond the network's input account for the default model's small lead over crisprHAL 2 and not for the tuned model's larger one, and given wider input without re-tuning crisprHAL 2 does worse; this describes the two models as used, not the architectures
- The two models were tuned with comparable compute, not equal draws: SLICER's original version 627 draws in 2.4 hours on a laptop, crisprHAL 2 28 draws (33 fits) in about 3.0 GPU-hours after an earlier 10-draw search of 7.3 CPU-hours (the same 627 draws would have cost it about 57 GPU-hours); its published configuration won four of five folds of its own search. Unequal tuning remains a known pitfall of model comparisons (Lones, 2021)
- A ceiling of about 0.7 treats a reliability as a ceiling. The model looks out of material rather than capacity (settings tuned with the flank features add only +0.0028 on the 20-mer alone); the best model reaches 77% of the 0.946 ceiling, and the remaining 0.21 is repeatable between replicates, so explaining it may need an experiment in which guide and locus are not confounded, which no pooled screen provides
- That the base next to the PAM acts in the opposite direction in E. coli and in eukaryotic predictors (Appendix A5) fits the failure of eukaryote-trained predictors on bacterial data that Ham et al. (2023) report
- Cost, as far as it goes: of the three predictors costed here, SLICER's default model is the cheapest to store and, on a CPU, to score, fits as fast as BoostMEC and is the most accurate; but its scoring advantage is that of array featurisers written for this work, since through the original per-guide code it takes 19.6 times longer, about as long as crisprHAL 2 (Appendix A6), and on a T4 crisprHAL 2 costs 3.9 times SLICER's laptop path from raw contexts (two machines; a shared VM). This does not show that SLICER is the cheapest predictor of bacterial guide activity: the others were not costed (a linear model on the 20-mer was not run)
- Limits: one organism, the same genome under another nuclease, and C. rodentium; the label is survival, not cutting; the step and slope analyses were designed after the result they follow; the head-to-head on crisprHAL 2's own splits is on random rows and the whole-gene comparison has five folds, one seed and no per-guide interval; crisprHAL 2 was not re-tuned on its own splits or at each width; cost comes from one laptop and one Colab session

### The paper's appendices


### Appendix A1. Hypotheses tested and not supported

- Composition classes (Figure A1a): the purine and amino fractions give a gain in at most 2 of 5 folds at any of the 14 windows (seed 41); the four bases separately (8 columns, which contain the C and G fractions) peak at +0.103 (250 nt). At the four windows of the full flank set, eight GC columns give +0.113, with purine +0.113 and with all four bases +0.108. With the target's own GC held fixed, flank purine correlates 0.00 to −0.02 with activity, against +0.15 to +0.20 for GC, upstream and downstream within 0.026 of each other
- Fine structure (Figure A1b; the columns are the GC and purine fraction of equal-width bins that tile the kilobase on each side, added to the 6,169 base columns): four numbers (one 1,000-nt bin per side) carry 75.7% of the composition signal, 16 columns of 250-nt bins carry 95.2%, 32 columns of 125-nt bins 97.2%, 80 columns of 50-nt bins 98.7% and 200 columns of 20-nt bins 96.0%: finer bins add little beyond about 250 nt. The 10 to 15 nt next to the cut are worth +0.055 (sequence block) and +0.070 (quantum tables) alone; dropping the sequence block from the flank features changes ρ by +0.0004 and the quantum-table block by −0.0054
- PAM density: holding flank GC fixed, NGG plus CCN density has a partial correlation of +0.022, the 52nd percentile of a random null whose 97.5th percentile is +0.060; adds nothing detected
- Target GC: it correlates −0.147 with activity and +0.245 with flank GC, and conditioning on it raises flank GC's association from +0.176 to +0.221, so target GC suppresses the effect and is not a confound; the flank gain is positive in every quintile of the target's own GC, from +0.072 among GC-rich guides to +0.349 among GC-poor ones
- A linear response to flank GC (a prediction recorded in the module docstring and log before the run, probability about 0.6) failed: activity is flat from the third decile and the lowest decile is the exception (Figure 2a); within genes the dependence is graded (+0.066 survives setting the lowest tenth aside)

> Figure A1. (a) Gain in ρ over the 20-mer from two columns (eight for the four bases) of one composition class as the window on each side grows (seed 41, five folds; error bars: sd over folds). (b) ρ of a model given the flank as equal-width bins tiling the kilobase on each side (4 columns = one 1,000-nt bin, 200 = fifty 20-nt bins; genic guides, three seeds, tuned settings; error bars: sd over fits).


### Appendix A2. Comparisons with crisprHAL 2 that tie or are small, and interpretability


> Table A2. Tuned SLICER against crisprHAL 2 as released, on crisprHAL 2's own test splits (SpCas9: 6,737 guides; eSpCas9: 11,898). Each cell gives the two models' values (SLICER against crisprHAL 2) and the difference with its 95% bootstrap interval over test guides; the last two rows contrast other models, as labelled.

| Measure | SpCas9 | eSpCas9 |
|---|---|---|
| Spearman ρ | 0.717 against 0.695 (difference and interval in the main text) | 0.811 against 0.782 (difference and interval in the main text) |
| Pearson r | 0.738 against 0.720: +0.017 [+0.010, +0.026] | 0.804 against 0.780: +0.023 [+0.019, +0.028] |
| Kendall τ | 0.529 against 0.507: +0.022 [+0.015, +0.030] | 0.622 against 0.592: +0.030 [+0.026, +0.034] |
| R² | 0.544 against 0.515: +0.029 [+0.018, +0.041] | 0.644 against 0.601: +0.044 [+0.036, +0.050] |
| AUROC, efficient guides | 0.825 against 0.814: +0.011 [+0.005, +0.019] | 0.909 against 0.892: +0.016 [+0.013, +0.020] |
| AUROC, duds | 0.895 against 0.886: +0.008 [+0.000, +0.017] | 0.907 against 0.896: +0.011 [+0.006, +0.016] |
| Top-decile precision | 0.300 against 0.280: +0.019 [−0.012, +0.052] | 0.497 against 0.494: +0.003 [−0.017, +0.029] |
| NDCG over the top decile | 0.823 against 0.811: +0.012 [−0.003, +0.025] | 0.902 against 0.898: +0.004 [−0.003, +0.012] |
| Top-5% precision | 0.164 against 0.116: +0.048 [+0.003, +0.074] | 0.362 against 0.343: +0.019 [−0.017, +0.049] |
| Mean score percentile of the hundred best-picked guides | +1.0 [−3.1, +5.2] | −0.8 [−3.2, +1.4] |
| Calibration slope | 1.01 against 1.08 | 1.05 against 1.05 |
| Spearman ρ of the default model, minus crisprHAL 2 | +0.0039 [−0.0047, +0.0124] | +0.0021 [−0.0031, +0.0072] |
| Spearman ρ of tuned SLICER averaged with crisprHAL 2 at equal weight, minus tuned SLICER | +0.0085 [+0.0040, +0.0127] | −0.0008 [−0.0030, +0.0015] |

- Reading Table A2: the correlation measures, R² and both AUROCs favour SLICER on both splits (the lower end for duds on SpCas9 is +0.000), the top-end measures are not detected (top-5% precision is detected on SpCas9 only), and SLICER's calibration slope is closer to 1 on SpCas9 and equal on eSpCas9. They measure the same thing, accuracy, and none changes the ranking. On our own random-row folds the default model is ahead of crisprHAL 2's published configuration by +0.0115 [+0.0076, +0.0156] pooled and in five of five folds, a small lead. Objectives aimed at the top end (three seeds, held-out genes, no intervals) did not improve top-decile precision
- Interpretability, four tests on both models as released: faithfulness (block-level sensitivity-n, Ancona et al., 2018, with TreeSHAP, Lundberg et al., 2020, and expected-gradients, Erion et al., 2021, attributions), stability under refitting, agreement with the model-free design rules and compressibility (a prediction recorded before the run that did not hold as stated). SLICER's attributions are more faithful (sensitivity-n 0.96 to 0.97 against 0.86 to 0.88, with different estimators, so part of the gap is estimator noise), its substitution map is more stable under refitting on seeds (0.998 against 0.989; the two tie on attribution blocks) and its 104-column version agrees better with the model-free map (+0.070 [+0.033, +0.122]); crisprHAL 2 leads on stability across data halves and on the accuracy kept after compression. SLICER is interpretable (Murdoch et al., 2019) and not shown to be more so, the tests are proxies, and both models learned the same guide map (ρ 0.97 between their position × base maps) and the same asymmetric response to flank GC
- Column by column, per-column importance is unstable on the published 6,232-column matrix, where 6,095 columns restate about twenty variables; on the 104-column non-redundant version it is readable: SHAP agrees at 0.87 across five tree families per column (native importance 0.70, against 0.50 on the published matrix). Position 20 alone holds 20% of the guide attribution; G at position 20 and at 18 lower the score by 0.33 and 0.28 SD and T at position 20 raises it by 0.36 SD, and the 80 SHAP-implied base effects agree with the model-free map at ρ 0.85; the other nine deciles of flank GC contribute −0.04 to +0.17 SD, against −0.78 SD for the lowest
- Faithfulness of SHAP without refitting (one default model fitted on four of the five whole-gene folds of the paper's design, seed 41; ρ 0.722 on the fifth, whose genes it has not seen; 100 of its guides): the summed TreeSHAP over a random subset of columns correlates, within each guide, with the prediction lost when those columns are replaced by values from 50 training guides at +0.922 for one column and +0.961 for 100, and at +0.960 for one block of columns and +0.977 for six (+0.922 to +0.977 over all ten subset sizes). A ridge surrogate's attributions give +0.127 to +0.316 and the same SHAP values assigned to permuted columns −0.023 to +0.047. SHAP is faithful to a fitted model even column by column, which is not the same as stable across fits (the previous bullet)
- SHAP interaction values on the same model (400 held-out guides, all 300 columns): single-column terms carry 69.8% of the variance of the prediction and pairwise terms 13.9%, and 52.6% of the absolute mass lies off the diagonal. The strongest pairs are different descriptors of the same base pairs and the GC of the 250 nt windows on the two sides, the model splitting credit between near copies, so no flank × guide interaction follows, in line with the near-additive gate test of the Discussion
- Counterfactual flank edits on the same model (1,500 held-out guides, DNA beyond the 20 nt next to the cut edited, the flank features rebuilt from it; label sd 3.29): GC +5 points moves the prediction by +0.41 [+0.37, +0.46] and −5 points by −0.81 [−0.86, −0.77]; swapping 5% or 20% of the bases at fixed GC moves it by 0.004 and 0.021, so the response is to composition; the fall is about twice the rise, as the AT-rich step of the Discussion predicts; shuffling the order beyond 20 nt moves it by +0.19 because the GC of the 50 to 500 nt windows changes. These are properties of the fitted model, not measurements of the genome

> Figure A2. Input width. ρ of crisprHAL 2 (published settings with batch 256, one seed per fold) and of SLICER's original version given the 20-mer (23 nt), the flank features that fit inside 378 nt, or the full family with windows to 1 kb each side (2,023 nt; flank summaries, not raw DNA). Thick lines: means over five random-row folds; thin lines: the folds.


### Appendix A3. Controls, robustness checks and protocols

- Protocols. (1) The paper's folds: five-fold cross-validation keeping every gene whole (guides outside genes in 100 kb blocks; 33,567 guides), seeds 41 to 43 (15 folds), the default model (A4; the 20-mer baseline uses all 6,169 columns). (2) The team's shared harness: rows assigned at random, five seeds of five-fold KFold (seeds 41 to 45), the same model and selection, so only the fold assignment differs. (3) Whole genes on the 30,203 genic guides only, five seeds. (4) Sealed genes: whole genes holding 20% of guides are sealed away before any choice is made, 15 draws, selection re-derived on the rest. (5) Development and confirmation genes: genes split 70/30, the tuned settings searched on the first and scored on the second. (6) crisprHAL 2's own train and test files, each model as released, with a paired bootstrap over test guides: SLICER's settings were searched on development genes of the SpCas9 screen, so SpCas9 test guides in those genes informed the search (no weights were fitted on a test guide), and 57% of eSpCas9 test guides are cohort guides; the interval does not include variation from the training split or the seed. (7) A genome no choice used (C. rodentium). Spearman ρ is the metric because the label's scale is arbitrary and a design tool ranks guides (Zarate et al., 2022, say it is what most work uses); Pearson r is higher on the same predictions and orders the models alike, as do Kendall τ and R² (Appendix A2). Differences carry 95% bootstrap intervals because an sd says how much a figure moves from one split to the next and an interval on a paired difference says whether it can be told from zero
- Which control ran on which folds: the displacement, window, strata, gate, interpretability and BoostMEC tests use the paper's folds; the sealing tests use sealed genes. The fold design hardly moves the flank gain, +0.161 on the paper's folds against +0.164 on random rows; whole genes are the headline design because the claim is about held-out genes
- The tuned lead over crisprHAL 2 on our five random-row folds is +0.0288 [+0.0251, +0.0326] against its published configuration and +0.0216 [+0.0180, +0.0252] against the best configuration found for it (these folds share genes with the settings search, which may carry up to about 0.009; not measured for this margin); on the 1,204 genes held out of the search +0.0324 [+0.0248, +0.0403] and +0.0240 [+0.0162, +0.0318]; across five seeds of SLICER +0.0280 to +0.0286 on eSpCas9. Settings searched with the flank features present gain +0.0162 over the default settings with them and +0.0028 on the 20-mer alone (30,203 genic guides, whole genes held out; positive in 9 of 15 folds; no 20-mer-only search was run)
- Flank gain with the same 300-column selection on both arms: +0.1612 on the paper's folds, where selection leaves the 20-mer where it was (ρ 0.5461 selected, 0.5466 not); on the 30,203 genic guides alone (three seeds), where the 20-mer scores 0.559, the gain is +0.1563 [+0.1515, +0.1616] at the default and +0.1697 [+0.1659, +0.1738] at the tuned settings, in 15 of 15 folds each
- The gain depends on the genes: on the curated cohort it is +0.102 [+0.089, +0.115] inside the 10,530 guides of the alphabetical block of genes (aaeA to leuC) that Noshay et al.'s matrix holds, and +0.180 [+0.164, +0.197] in the 19,673 guides of all other genes (whole genes held out), which accounts for part of why a gain measured on that matrix is smaller than on the curated cohort
- Strata of the flank gain (whole-gene folds, default model): +0.161 [+0.149, +0.171] on all guides; +0.157 [+0.145, +0.169] among the 26,146 guides whose every 25-mer within 1 kb is unique, with no seed duplicate and one genomic copy; +0.175 [+0.150, +0.199] in the rest, with any repeat signal (7,421 guides); +0.261 [+0.190, +0.334] among the 852 guides with over 10% repeated 25-mers; +0.142 [+0.113, +0.173] (1,938) for guides with another genomic site sharing the 12 PAM-proximal bases; +0.160 [+0.148, +0.171] given the control read depth as a column and +0.159 given the uniqueness measures
- The step at the AT-rich end (exploratory: these follow-ups were designed after the linear prediction failed): ρ +0.485 inside the lowest decile (3,086 genic guides; cut on a different flank-GC summary from Figure 2a's decile, so not the same guides) against +0.021 outside; a second enzyme at the same loci gives −0.80 SD, and C. rodentium −0.21 SD with an interval that includes zero, so it is not detected there
- Data efficiency: labels to reach ρ 0.62 / 0.66 / 0.68 / 0.70 are 4,000 / 8,000 / 8,000 / 20,000 for the default model against 8,000 / 13,880 / 20,000 / not reached at 26,853 for crisprHAL 2 (five training sizes, our folds)
- Sealing (protocol 4) over 15 draws lowers the flank gain by +0.0087 [+0.0020, +0.0153] (mean ± 1.96 se over draws, which share data and so understate the uncertainty): it reads +0.1647 in cross-validation within the unsealed pool and +0.1560 on sealed genes. With whole genes also held out inside the pool (first 8 draws) the drop is +0.0081, so random-row pool folds account for only +0.0007; the default configuration re-derives no hyperparameter and an 8-draw search adds nothing detectable, so the drop is not shown to be adaptive overfitting, and the choice of cohort and of the feature design is not reached by this design. The eSpCas9 sealed-gene gain was predicted before the run (at least +0.08, positive in at least 14 of 15 sealings) and observed positive in 15 of 15; the tuned settings' advantage on a genome no choice used is +0.0172 [+0.0162, +0.0206] (C. rodentium: one 236 kb region, so the interval rests on about four 100 kb blocks)
- The flank signal is not made by which guides the read-count cutoff keeps: on the 60,811 raw guides at a single site (33,449 of them kept) the chance of being kept depends only weakly on flank GC (0.53 to 0.57 across deciles), and the flank association is present, about 10 to 15% weaker, in the 18,733 guides with 30 to 72 unselected counts that the cutoff discards (ρ 0.159, lowest-decile deficit 0.94 SD, plain log-ratio label) than in those it keeps (0.175 to 0.188; 0.97 to 1.02 SD)

### Appendix A4. Data, features, models and the label in detail

- Datasets. The cohort: 33,567 SpCas9 guides of the E. coli screen of Guo et al. (2018) that remain after curation (Browne et al., 2026), 30,203 of them in 4,013 genes. crisprHAL 2's own SpCas9 and eSpCas9 files (guides located in the genome: 26,830 train / 6,737 test and 47,591 / 11,898). The eSpCas9 screen: 59,489 guides, the same library under another nuclease. The C. rodentium TevSpCas9 screen: 25,210 guides located in the genome, covering one 236 kb region (Ham et al., 2023). Eleven eukaryotic datasets of 300 or more guides (CRISPOR collection, Haeussler et al., 2016; eight human, two mouse, one zebrafish), oriented so that higher means more active. Noshay et al.'s released matrix: 13,880 guides × 6,232 columns
- Features. Base: 6,169 columns regenerated from the 20-nt protospacer (position × k-mer indicators with k up to 4, quantum-chemical lookup tables). Flank features: 348 columns, 88 summarising windows of 50, 250, 500 and 1,000 nt on each side of the cut (GC and purine fraction, longest run, quantum-table averages) and 260 describing the cut itself (10 nt before the protospacer, the PAM's N, 15 nt after it)
- Models. SLICER's default model: 400 trees, 300 columns chosen inside each training fold; tuned: 450 columns, 4,000 trees, depth 5; the input-width arms use SLICER's original LightGBM version of the same columns. crisprHAL 2: convolutional network with a bidirectional GRU, 790,533 parameters, 378-nt input, SpCas9 weights
- The label. ALDEx2 (Fernandes et al., 2014) was written for differential abundance in sequencing counts. Here each guide's counts in two unselected (dead-Cas9) and two Cas9-selected replicates are resampled 128 times from a Dirichlet distribution (low counts get wide spread), each draw is centred on the geometric mean of all retained guides (a centred log ratio), and the median difference between the conditions is kept, unselected minus Cas9-selected: a log2 fold change relative to the typical guide that is high for a guide depleted by cutting. The centre moves when other guides are removed, so scores are compared by rank. The direction is checked on the 118 guides with more than one genomic copy, whose median label is 2.11 against 0.27 for single-copy guides
- Genes. The reference is the E. coli K-12 MG1655 chromosome from NCBI RefSeq as released with the screen of Cui et al. (2018): NC_000913.2, 4,639,675 bp (NCBI has since released NC_000913.3, 4,641,652 bp; looked up on 24 Nov 2026). A guide takes the gene whose coordinates contain the leftmost base of its protospacer, from a table of 4,499 genes with their coordinates; where guides are grouped for cross-validation, the 3,364 guides in no gene are grouped by 100 kb block so that neighbours stay in one fold (the tuned-model figures use the 30,203 genic guides only). Noshay et al.'s released matrix holds 13,880 of the 40,468 guides of their table, in a contiguous alphabetical range of 1,225 gene identifiers from aaeA to leuC (the gain on it is smaller than on the full cohort; A3)

### Appendix A5. Transferability across nuclease, organism and kingdom

- Across nuclease, on the 25,922 eSpCas9 guides absent from the SpCas9 screen, trained on the same 26,830 SpCas9 guides: SLICER's default model reaches ρ 0.659, its tuned model 0.678 and crisprHAL 2's released weights 0.633; SLICER leads by +0.025 [+0.021, +0.030] (default) and +0.044 [+0.040, +0.049] (tuned), and the flank features carry +0.110 of the default model's ρ
- Across organism, SLICER's original version trained on E. coli keeps 91% of its locally trained ρ on C. rodentium (0.698) and 89% going back (0.628), where flank features carry +0.149 of the 0.628 that arrives (the original LightGBM version, transferred arms over 3 seeds, local folds random-row with seed 41)
- Across kingdom nothing carries from E. coli: scored on the eleven eukaryotic datasets, SLICER's 20-mer features give a median ρ of −0.041 (2 of 11 positive, largest +0.108) and crisprHAL 2's architecture on its guide-and-PAM input −0.015 (5 of 11 positive, largest +0.137); the other way, every eukaryotic model scored on E. coli is negative (11 of 11, median ρ −0.058). Among the eukaryotic datasets all 106 ordered pairs from different libraries are positive: human to human keeps 70% of the target's own ρ (median ρ 0.26), human to mouse 73%, mouse to human 70%, and the zebrafish pairs about a third (Figure A3)
- The diagonal is low because the tables are small and noisy and the model sees only the 20-mer: the eleven tables hold 881 to 4,293 guides, and on the same tables the six published predictors read from the CRISPOR files (some trained on these very tables) reach a median ρ of 0.10 to 0.42 and the best of them 0.17 to 0.70, against 0.18 to 0.55 for the 20-mer pipeline in cross-validation on the table itself (above the median published predictor on 11 of the 11 tables and above the best on 3); in E. coli the same pipeline on the 20-mer alone reaches 0.544 in the same kind of random folds
- Position 20 illustrates the difference: G next to the PAM is −0.33 SD in E. coli (G minus the mean of A and C) and positive in all eleven eukaryotic datasets as a point estimate (+0.08 to +0.57; ten of eleven intervals exclude zero, and the orientation of four datasets was set by eukaryote-trained predictors, which makes this contrast and the eukaryote-to-eukaryote transfer partly circular); C. rodentium has the bacterial sign (−0.42 [−0.45, −0.39]). The E. coli and human position × base maps are unrelated overall (median rank correlation −0.14 over eight pairs; Figure A4)

> Figure A3. Transfer between datasets (SLICER, 20-mer features, default model). (a) ρ of a model trained on the row and scored on the column; the bold diagonal is five-fold cross-validation on the dataset itself (random folds, as the eukaryotic tables carry no gene information; E. coli diagonal 0.544). Every cell between eukaryotic datasets is positive; E. coli to eukaryote cells are near zero (median −0.041, 2 of 11 positive) and eukaryote to E. coli cells are all negative. (b) The E. coli row for SLICER and for crisprHAL 2's architecture retrained on the 23-nt guide-and-PAM input, both on the same 33,567 guides: neither transfers.


> Figure A4. The guide's last base. Effect of G at guide position 20 against the mean of A and C, in SD of the label with a 95% bootstrap interval, for E. coli (orange; with the same-loci eSpCas9 screen and C. rodentium) and for each eukaryotic dataset of at least 300 guides and each species pooled (blue). E. coli is negative; every eukaryotic dataset has a positive point estimate (ten of eleven intervals exclude zero).


### Appendix A6. Cost: how it was measured, and BoostMEC retrained and as shipped

- BoostMEC retrained on the cohort as its authors trained it (`boostmec_replica.py`; github.com/oazarate/BoostMEC, commit cdf190c): 149 features of the 30-mer (two RNAfold energies, four melting temperatures, the G and C count, poly-T counts, 28 position-specific nucleotides and 28 dinucleotides as categorical features, and the counts of every 1-, 2- and 3-mer), the released model's hyperparameters (`model.txt`: learning rate 0.0099, 701 leaves, depth 12, 203 rows per leaf, 48% of columns per tree, L2 1.48, 343 bins), the rule for the number of trees (ten-fold cross-validation with early stopping, limit 7,000) and the search (Bayesian optimisation, 10 initial points and 100 rounds, each scored by that cross-validation). The features are those of the authors' own pipeline, run unmodified in their image on all 33,567 guides (below). Arm 1, released hyperparameters on SLICER's default protocol over 15 folds (919 trees, 88,309 leaves; the result is in the main text). Arm 2, their search run on the development genes (70%), the model refitted on all of them and scored on the 8,943 guides of the genes held out of the search, where SLICER's default model reaches 0.707 and its tuned model 0.722: BoostMEC 0.623 (−0.084 [−0.103, −0.066] against the default model, −0.099 [−0.117, −0.080] against the tuned model; bootstrap over genes). Not published, and so chosen here: the search ranges (wide around the released optimum) and the optimiser, a Gaussian process with the upper-confidence-bound rule written with scikit-learn
- Laptop session: Apple M5, ten cores; one chained run with nothing else of ours running (load average 1.8 to 2.4 while scoring). Scoring on one thread, from the raw sequence, after one warm-up call (300, 60 and 15 repeats; 200, 40 and 8 for crisprHAL 2; coefficient of variation at most 4%); median ms per guide at 1 / 100 / 10,000 guides: SLICER default 0.629 / 0.0726 / 0.0795, tuned 0.716 / 0.0885 / 0.0935, BoostMEC (its features computed in Python, the released model) 2.94 / 3.07 / 3.11, crisprHAL 2 15.4 / 1.37 / 1.22. The input windows differ: 30 nt for BoostMEC, 378 nt for crisprHAL 2, up to 1 kb either side of the guide for SLICER (read from the cohort's table; fetching a genome is outside every model's cost). Fits: ten of SLICER's default model, five of its tuned model and of BoostMEC, three of crisprHAL 2 (published configuration: 48 epochs, batch 1,024), on the training part of one fold (26,853 guides); BoostMEC with its ten-fold choice of tree count 29.2 s and its search 17 min, training features 2.13 s for SLICER and 83.5 s for BoostMEC, stored model sizes compressed 0.15 MB (SLICER default), 2.71 MB (tuned), 4.14 MB (BoostMEC)
- The array featurisers behind SLICER's scoring time (`base_fast.py`, `a_flank_fast.py`) give the same 6,517 columns as the original per-guide code on all 33,567 guides (same names and missing values; maximum absolute difference 0.0 after the float32 cast every model reads), and the default and tuned models predict identically from either. With the two paths alternating in one laptop session (load average 2.3 to 2.6), the default model takes 6.05 ms per guide through the original code and 0.617 ms through the array code for one guide, 1.41 and 0.0718 at a batch of 100 (19.6 [19.5, 19.7] times; 20 repeats) and 1.40 and 0.0811 at 10,000 (17.3 [16.9, 17.8] times; 3 repeats). Through the original code SLICER therefore takes about as long per guide as crisprHAL 2 (1.37 ms at a batch of 100, above), and its scoring advantage in Figure 4 is the rewrite's, which is exact; crisprHAL 2's encoding was run as shipped
- The Colab session was one VM that other tenants may share, so it is noisier than the laptop: repeat variation (coefficient of variation) is 2.7% for crisprHAL 2's forward pass at 10,000 guides and up to 30% for the VM's CPU timings of the trees. No whole-path timing of SLICER exists on the VM, so the only same-machine pair is the forward pass of the two models (13 times at 10,000 guides); the 3.9 times from raw contexts compares the T4 with the laptop.
- Process CPU seconds, for the cores kept busy: SLICER default 81.5, tuned 132; BoostMEC 46.0 for the final fit and 181 with the ten-fold cross-validation (ten one-thread fits run in parallel); crisprHAL 2 9,617 (160 CPU-minutes, 9.3 cores busy on average). They are not compared, because XGBoost's threads spin while they wait and so overstate the work; wall-clock is.
- GPU session (`colab/cost_gpu.py`, one Colab Tesla T4): random 378-nt contexts and labels (neither the cost of a forward pass nor of a fixed number of epochs depends on the values). crisprHAL 2's batch size was swept (32, 256, 1,024, 2,048) and the best kept (2,048 guides at 10,000 on the card). Tree models of SLICER's shapes (default: 400 trees, depth 4, 300 columns; tuned: 4,000 trees, depth 5, 450 columns), forward pass per guide at 10,000: 0.00636 ms on one CPU thread, 0.00519 on the T4 with the matrix on the host and 0.000617 with it on the card (tuned shape: 0.0690, 0.0194, 0.00803); fits on synthetic data of those shapes: 3.20 s on the T4 against 23.8 s on Colab's CPU (default shape) and 29.0 s against 479 s (tuned shape); crisprHAL 2 326 s (3 fits, 5.4 to 5.6 min). An earlier Colab session gave 0.0762 ms per guide for crisprHAL 2 on the card at 10,000
- BoostMEC as shipped: its image (`oazarate/boostmec:latest`, 1.2 GB, linux/amd64, Ubuntu 18.04, R 4.1.0, ViennaRNA 2.4.14; its `model.txt` and `rules.rds` are byte-identical to the repository's) run unmodified on this laptop under Docker Desktop's x86 emulation, so its R and Python are emulated; only RNAfold was calibrated (natively 2.9 ms for the 101-nt guide with scaffold, 1.1 times that in the container). Whole `boostmec-pipeline.py`: 3.74 s for 10 guides, 4.17 s for 100 (41.7 ms per guide), 53.5 s for 10,000 (52.7 to 54.5; 3 runs), 1.72 times the Python version of its features. At 10,000 guides the stages took 31.6 s for RNAfold on the guide with its scaffold (60% of 52.5 s), 16.8 s for the R feature script, 2.18 s for the R prediction script and 0.523 s for RNAfold on the guide. A one-guide input fails in `predict.R` (R's CSV reader turns a position-specific nucleotide that is the same in every row into a logical column), so 10 guides is the smallest batch timed
- The Python version of BoostMEC's features (`cost_boostmec.py`, ViennaRNA's binding and Biopython) against the shipped pipeline on 100 guides: identical energies, G and C counts, poly-T features, all 84 k-mer counts and all 56 categorical columns; the four melting temperatures differ by 1.95, 0.91, 1.46 and 1.10 degrees on average (R's TmCalculator against Biopython), and predictions by 0.34 on average (3.19 at most) on the model's 0 to 100 scale, Spearman 0.9990. It reproduces the paper's two worked examples (Supplementary Tables S2 and S3) as 49.19 and 28.41 against the published 49.53 and 28.42. The released model run in Python on the shipped features gives the shipped predictions exactly, for all 33,567 guides. The retrained arms use the shipped features

References the paper cites that Appendix D does not yet list: Ancona et al. (2018), Erion et al. (2021), Lundberg et al. (2020), Murdoch et al. (2019), Chen and Guestrin (2016) and Fernandes et al. (2014); their full entries are in the paper's reference lists.

