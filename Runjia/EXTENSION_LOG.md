# Extension work: 3D structure and nuclease information (private-fork experiments)

> **Finding numbers in this log are those of 20 November 2026 and are not rewritten, because the log is a record of what was
> believed when.** On 22 November the report was reorganised: the old **Finding 7** (the comparison with crisprHAL 2) is the first
> section of **Part 6**; the old **Finding 8** (shape and 3D) is **Appendix B4**; the old **Finding 9** (nuclease) was removed
> (`REMOVED.md`); the old **Finding 10** is now **Finding 7**. Findings 1–6 are unchanged.

Everything here lives in the fork, outside the frozen pipeline. New code is `Runjia/sgrna/post_freeze/`;
results are `Runjia/results/post_freeze/`. The frozen modules are imported, never edited. The shared
repository (smal-boi/RSI-Crispr) is untouched.

This file is the running record: what was run, what it showed, and the inference drawn, with each
inference marked **verified**, **partly verified** or **open**, and the check that decided it.
The report (`PROGRESS_REPORT.md`) carries only what survived.

## 0. Provenance of the data used

| data | source | how obtained | what it is |
|---|---|---|---|
| Cas9-variant activity, 40,128 targets x 9 nucleases | Kim et al. 2020, Nat Biotechnol 38:1328, Supplementary Table 6 (`41587_2020_537_MOESM10_ESM.xlsx`) | publisher static content, 6.4 MB | the sheet lumps four kinds of column: the 30-mer input, a library covariate (tRNA-N20 vs (G/g)N19 sgRNA), the authors' train/test flag, nine outcome columns |
| PAM profiles of the nine nucleases | same paper, Supplementary Table 7 (`MOESM7`) | same | mean indel frequency for each of 256 four-nucleotide PAMs per nuclease |
| pentamer DNA-shape table | DNAshapeR (Chiu et al. 2016) `src/TableCompiler.cpp`, GitHub TsuPeiChiu/DNAshapeR | read as text, parsed with a regex; no code from the repository was run | 512 pentamers x 90 numbers; lumps MC, X-ray and MD sources and means, sds and counts (see `ext/dnashape.py`) |
| R-loop complex structure | PDB 5F9R (Jiang et al. 2016), RCSB | `files.rcsb.org/download/5F9R.pdb` | Cas9 (chain B), sgRNA (A), target strand (C), non-target strand (D) |
| curated E. coli cutting cohort | crisprHAL curated 33,567 guides, already in the project | -- | the cohort every cutting-side number in the report uses |

Deep DNAshape (JinsenLi/deepDNAshape) was cloned to `external_data/deepDNAshape_src` and **not run**:
the sandbox refused to execute third-party model code. The pentamer table is the data route, and 22.3 closes the
question a different way -- by bounding what *any* local shape predictor could add, which does not require running one.

### 0b. The search: what was found, what was used, what was not

| candidate | what it offers | outcome |
|---|---|---|
| Kim et al. 2020 (Nat Biotechnol) Supplementary Tables | 40,128 targets x 9 Cas9 variants with the authors' split; 256-PAM profiles | **used** (Finding 9); the only public sheet found that gives >= 9 nucleases on the same targets with processed activities |
| DNAshapeR pentamer table (Rohs lab) | 512 pentamers x 90 numbers: MC, X-ray and MD shape parameters, step parameters included | **used** (Finding 8); the "several kinds of column in one table" case |
| PDB 5F9R (Jiang et al. 2016) | Cas9 - sgRNA - target / non-target DNA R-loop | **used** for nuclease structure descriptors |
| Deep DNAshape (Li, Chiu, Rohs; GitHub JinsenLi/deepDNAshape) | neural predictor, 14 shape features with up to 7 bp of context | cloned, **not run**: the sandbox refused to execute third-party model code, and running a downloaded model is not prudent in any case. The DNAshapeR table gives the 5-mer-context version. **Closed by a bound instead of by running it** (22.3): its output over the inner flank is a deterministic function of the local sequence, and giving the model that sequence explicitly adds nothing |
| ABC / muABC tetranucleotide MD stiffness (Pasi et al. 2014) | 6x6 stiffness per tetranucleotide | **not obtained**: no direct download page found; this is the data that a true stiffness-tensor encoding would need, and its absence is the main limit of Finding 8 |
| E. coli Hi-C / 3D chromosome (Lioy et al. 2018, GSE107301) | contact maps, 3D coordinates | already family E (`eng.nuc`), a null; not repeated |
| E. coli NAP ChIP-seq (GSE24991 H-NS/Fis; SRP008538 IHF/HU) and IPOD-HR protein occupancy (GSE164796) | protein-bound DNA tracks | **not used**: no processed track was fetched. Registered as open with a bound (report Part 4, open-questions register, row 2): Hi-C-derived family E was already null; 21.4 shows the flank effect has no structure below ~250 nt, so a track can only enter through its coarse average; and the coarse average of H-NS/Fis/IHF occupancy is AT-driven, which family A's composition columns already carry. That is an argument, not a measurement, and what would settle it is one processed track mapped to the 33,567 loci plus a single ablation arm |
| DeepSpCas9variants (GitHub NahyeKim) | trained models only, no training data; published accuracies are in the paywalled article | weights **not run**; published per-variant Spearman (Fig. 4b of the article, supplied by the user on 9 Nov, `lit_review/`) compared with ours in `results/post_freeze/kim2020_published_vs_ours.csv`: mean 0.896 against 0.881, ahead on 5 of 9 variants, evoCas9 0.683 against 0.81 |
| Schmid-Burgk et al. 2020 (Mol Cell), others | specificity profiling with 59 guides | too small to condition a model on |
| human flank sequences for the project's human matrix | would let Finding 8 be run on human | not available locally (the matrix carries computed columns, not sequence); needs hg38 coordinates |

## 1. Prerequisite: the within-gene effect of family A on the cutting screen

**Run.** `ext/cut_within_gene.py` -- the same decomposition used on Cui and Rousset
(`strand_within_gene.py`): gene-grouped CV, ten seeds (41-50), genes with at least five guides;
30,203 genic guides in 4,013 genes.

| | base | + family A | gain | seed sd of gain | 95% bootstrap over genes |
|---|---:|---:|---:|---:|---:|
| overall rho | 0.561 | 0.717 | +0.155 | 0.0008 | |
| between genes | 0.438 | 0.713 | +0.275 | 0.0029 | |
| **within genes** | 0.618 | 0.703 | **+0.085** | 0.0007 | [+0.078, +0.090] |

**Inference (the earlier dCas9 claim, re-examined).** Finding 3 said family A adds nothing to the
dCas9 screens beyond the strand rule. The within-gene decomposition already showed +0.102 (Cui,
coding strand) and +0.114 (Rousset) there. The cutting screen's within-gene gain is +0.085, so in
raw rho the dCas9 coding-strand gains are *not smaller*. That comparison is not valid as it stands,
because the three screens start from very different bases (within-gene rho 0.62 on the cutting
screen, 0.17 and 0.08 on the dCas9 coding strands) and a gain of 0.1 from 0.08 is not a gain of 0.1
from 0.62. Two scale-free comparisons:

* *Fraction of the attainable gain closed.* Cutting: ceiling 0.946 (the label's replicate ceiling,
  Finding 6), (0.7035 - 0.6181) / (0.946 - 0.6181) = **26%**. Cui coding strand: the two time points
  of the screen (fit18, fit75) agree within genes at rho 0.768 on exactly these 2,289 guides, which
  by Spearman-Brown puts the reliability of their mean at 0.869 and the ceiling near 0.93;
  (0.274 - 0.172) / (0.93 - 0.172) = **13%**. Rousset has no second readout, but the same 1,759 coding-strand guides
  were screened at E18 and E75 and agree within genes at rho 0.805, which with Cui's reliability puts Rousset's
  ceiling at 0.864 or above (r^2 / rel_C = 0.746); (0.196 - 0.083) / (0.864 - 0.083) = **14%**.
* *Variance explained* (rho squared): +0.113 (cutting) against +0.045 (Cui) and +0.031 (Rousset).

Both say the same thing: the dCas9 within-gene effect is **real and between a third and a half the size of
the cutting one** (half by share of the attainable gain, 27-40% by variance explained), not absent and not equal. **Status: verified** for "not absent" (ten seeds, bootstrap
interval clear of zero, two screens, survives conditioning on transcription context); **open** for
"a third to a half the size" (rests on a reliability ceiling estimated from time points that are not true
replicates, and on assuming Rousset's ceiling equals Cui's).

## 2. Verification of the building blocks (done before any result below was trusted)

| check | result |
|---|---|
| Kim 2020: authors' split leaks no protospacer | 0 of 5,415 test rows share a 20-mer with a training row |
| Kim 2020: column-to-nuclease mapping | mean activity by PAM class reproduces the known PAM preferences (NG: broad; VRQR: NGA; evoCas9: lowest overall; high-fidelity variants: NGG only) |
| 5F9R numbering matches UniProt for all 32 substituted residues | every wild-type residue found at its number (e.g. R1335, T1337, K848, Q695) |
| 5F9R distances are physically sensible | PAM-expanding substitutions (NG, VRQR) sit 3.1-4.6 A from the non-target strand; REC3 substitutions (HF1, Hypa, evo) sit 3.1-3.5 A from the target strand |
| pentamer table parsed correctly | A-tract MGW 3.42 A against GCGCG 5.45 A; legacy and MC columns agree to <=0.46; mean rise 3.2-3.3 A, mean twist 32.5-34.3 deg |
| reverse-complement handling | step parameters read from the other strand equal the table's, with Shift/Tilt sign flips, to 0.0; end-to-end distance and radius of gyration of a 3D chain are identical (correlation 1.0, max relative difference 0.0) whichever strand it is read on, for all three sources. This requires the mid-step convention for the translation, so it also tests the chain composition |
| vectorised features match an independent loop | 100 values (both arms, bp and step features, window means) agree to 3e-6 (float32) |
| ideal B-DNA and phased roll | straight (radius 0.000) for ideal parameters; 10.5-bp-phased roll bends end-to-end/contour to 0.095, constant roll leaves 0.978 |

## 3. Nuclease information (Kim et al. 2020, nine nucleases, 40,128 targets)

All arms: XGBoost on the same 596-column featurisation of the 30-mer (one-hot, dinucleotide one-hot,
7 composition columns, the tRNA library covariate); scores are Spearman on the authors' test split
(5,415 targets, none sharing a 20-mer with training), per nuclease then averaged over the nine.
"rho" is over all targets (so it includes the PAM compatibility of each enzyme); "NGG" is over the
targets with an NGG PAM only, i.e. sequence-dependent efficiency under the canonical PAM.

### 3.1 Column kinds of the sheet, removed one at a time (per-nuclease models)
`results/post_freeze/nuclease_indist.csv`. Mean over nine nucleases (rho / NGG):

| columns | rho | NGG |
|---|---:|---:|
| all | 0.881 | 0.793 |
| - library covariate (tRNA flag) | 0.875 | 0.759 |
| - PAM one-hot (3 nt) | 0.876 | 0.790 |
| - flank one-hot (4 nt left, 3 nt right) | 0.880 | 0.789 |
| - dinucleotide one-hot | 0.869 | 0.774 |
| - composition | 0.879 | 0.785 |
| spacer one-hot only | 0.341 | 0.613 |
| spacer + PAM one-hot only | 0.842 | 0.700 |

Findings. (i) The **library covariate matters on the canonical-PAM subset** (-0.034 NGG; -0.095 for
eSpCas9, -0.089 for HF1): sgRNA type changes activity more than most sequence columns, and a model
that ignores it is partly fitting library design. It is a property of how the library was built, not
of the target, and it must be a column or an adjustment in any analysis of this sheet. (ii) Removing
the PAM one-hot costs almost nothing in rho (-0.005) *because the PAM is also present in the dinucleotide
columns*: the columns are redundant, and a one-column-at-a-time ablation understates their importance
(spacer alone 0.34 overall, spacer + PAM 0.84). (iii) The flank (4 + 3 nt) adds +0.004 NGG.

### 3.2 Pooling nine nucleases: first reading was a capacity artefact
First run (500 trees, depth 6, everywhere): per-nuclease 0.881 / 0.793; pooled with a one-hot
nuclease label 0.866 / 0.736; a deficit of 0.015 / 0.057. Read at face value this says pooling hurts.
**It does not.** At matched and larger capacity (`nuclease_capacity.csv`), mean rho / NGG:

| trees x depth | pooled, one-hot | per-nuclease |
|---|---:|---:|
| 500 x 6 | 0.866 / 0.736 | 0.881 / 0.793 |
| 1500 x 6 | 0.879 / 0.776 | 0.886 / 0.808 |
| 500 x 10 | 0.878 / 0.778 | 0.881 / 0.796 |
| 1500 x 10 | 0.881 / 0.796 | 0.882 / 0.797 |
| 3000 x 8 | 0.886 / 0.798 | not run |

The pooled model needs more trees to cover nine tasks. At sufficient capacity it ties with the
best per-nuclease fit (0.886 against 0.886). **Pooling neither helps nor hurts in distribution.**
The conclusion "pooled is worse" in the first run would have been reported as a finding if the
capacity had not been varied.

### 3.3 Descriptors in distribution carry no information beyond the label
Replacing the one-hot with descriptors gave 0.868-0.872 / 0.747-0.753 against 0.866 / 0.736 for the
one-hot (500 x 6): apparently +0.002-0.006 / +0.011-0.017. The control: the **same descriptors with
the nuclease rows permuted** (`nuclease_indist_null.csv`, 8 permutations each) give 0.866-0.870 /
0.731-0.750, indistinguishable. The gain over one-hot is the gain of a dense continuous code over a
sparse one at fixed capacity, not information about the enzymes. In distribution every injective code
of the nuclease is equivalent, as it must be when the nuclease was seen in training.

### 3.4 Leave one nuclease out: where descriptors do help, and where they cannot
`results/post_freeze/nuclease_lono_rep.csv`, saved predictions in `lono_preds/`; 3 replicates (different 15k-row
training subsamples and seeds); 4 shuffled-descriptor controls per replicate; the unit of analysis is the
held-out nuclease (n = 9) and intervals are a bootstrap over nuclease. Reference U = a pooled model with
no nuclease information, trained on the same rows.

| arm | rho | NGG | PAM-class ordering | within-PAM-class rho |
|---|---:|---:|---:|---:|
| U (no nuclease information) | 0.795 | 0.613 | 0.855 | 0.410 |
| descriptors, label-free (mutations + 5F9R structure) | 0.809 | 0.635 | 0.908 | 0.408 |
| descriptors, all (adds PAM profile + activity scale) | 0.813 | 0.634 | 0.910 | 0.427 |
| the same descriptors, shuffled across nucleases | 0.757-0.761 | 0.594-0.597 | | |
| mean of per-source models (no descriptors) | 0.798 | 0.614 | 0.869 | 0.423 |
| own model (reference; not held out) | 0.881 | 0.793 | 0.972 | 0.545 |

Differences from U (bootstrap over nuclease, 95%): overall rho +0.014 [-0.001, +0.039] (label-free) and
+0.018 [+0.000, +0.042] (all); NGG +0.022 [-0.015, +0.065]; **PAM-class ordering +0.053 [+0.007, +0.103]**
(label-free); **within-PAM-class rho -0.002 [-0.039, +0.040]**. Shuffled descriptors are *worse* than
no descriptors (-0.034 to -0.039 rho, intervals clear of zero, 0-1 of 9 nucleases ahead).

**Inference.** Nuclease descriptors tell a model for an enzyme it has not seen *which PAM classes the
enzyme can use*: the ordering of PAM classes improves (0.855 -> 0.908) and this is what raises overall
rho. They do not tell it the enzyme's own sequence-dependent efficiency rules: within a PAM class
the held-out score is unchanged (0.410 -> 0.408) and sits 0.135 below the enzyme's own model. The
label-free descriptors do this as well as the ones built from a PAM screen of the new enzyme (class
ordering 0.908 against 0.900 for the PAM profile alone), so the information about PAM usage can be
read from the substitutions and their position in the PI domain without measuring the enzyme. Largest
single gain: SpCas9-NG, where U fails (rho 0.703) and descriptors recover 0.80.
**What cannot be concluded:** which property of the enzyme carries the information. With nine enzymes
the mutations, the structure-derived distances and the PAM profile are collinear (each group alone
gives the same class-ordering gain, 0.045-0.060), so this is a statement about descriptors as a set.
**Status: verified** for the PAM-ordering effect (three replicates, shuffled control, bootstrap),
**verified** that within-class efficiency is not transferred by descriptors, **open** for attribution
to a particular descriptor.

## 4. Shape and 3D path descriptors of the flank (DNAshapeR pentamer table; E. coli curated cutting cohort)

**Setup.** Base representation (6,169 columns regenerable from the 20-mer) plus blocks built from the
locus (500 nt each side of the guide, read on the guide's strand). Pentamer table split by source (mc,
xrc, md; the 27 `legacy` and 3 `ep` numbers are not used) and each source turned into three blocks:
`mean` (104 columns: arm x 4 windows x 13 per-bp and per-step features), `vec` (208: the same features
for the first 8 bp of each arm, kept position by position), `geom` (201: rigid-body reconstruction of
the helical path in the frame of the base pair 5' of the cut -- path positions, axis tangents,
gyration-tensor components and eigen-structure, straightness, excess twist). Gene-grouped CV, three
seeds x five folds, 30,203 genic guides, top-300 selection inside each training fold, identical folds
for every arm so differences are paired. `analyze.py` prints fold-paired differences.

### 4.1 Alone, on top of the base representation (rho 0.5588)
| block | md | xrc | mc |
|---|---:|---:|---:|
| mean (104) | +0.0991 | +0.1018 | +0.1026 |
| vec (208) | +0.0523 | +0.0557 | +0.0534 |
| geom (201) | **+0.0308** | +0.0779 | +0.0799 |
| mean + vec + geom (513) | +0.1376 | +0.1399 | +0.1405 |
| all three sources (1,539) | | | +0.1446 |
| family A (348) for reference | | | +0.1563 |

Every cell is ahead of the base in 15 of 15 folds.

### 4.2 On top of family A (rho 0.7151): nothing, in the boosted pipeline
Paired difference from A (15 folds, 3 seeds): A + mc mean +0.0001; + mc vec -0.0005; + mc geom -0.0008;
+ mc all -0.0007; + md geom -0.0006; + md all -0.0007; + xrc geom -0.0004; + xrc all **+0.0006** (11 of 15
folds); + all three sources (1,887 columns) -0.0006. The **shuffled-flank controls** (same pipeline,
flank bases shuffled inside 25-nt blocks, or shuffled across the whole arm) on top of A: -0.0012 and
-0.0009. Real columns sit 0.0002-0.0020 above their own noise floor, which is the cost of adding 500
columns that cannot help. **Tensor-valued 3D descriptors of the flank add nothing to family A in the
champion pipeline.**

### 4.3 Why the standalone gains exist, with the explanations tested rather than assumed
1. *The three sources agree on `mean` and `vec` to within 0.004* although their parameter values differ
   (mean twist 34.3 / 33.1 / 32.5 deg; sd of twist across pentamers 1.5 / 3.8 / 4.1 deg). The `mean`
   block is not reading source-specific physics.
2. *Plain letters do as well.* Base composition in the same windows (40 columns: A/C/G/T fractions + GC,
   arm x 4 windows) gives **+0.1046**, against +0.1026 for the 104 shape-mean columns; adding the shape
   means to the letters gives **-0.0002**. Dinucleotide fractions (128 columns) give +0.0970 alone (more columns dilute the 300-slot selection); with the shape means 0.6639, i.e. +0.0025 over the shape means alone. The shape window mean is a re-expression of composition.
3. *The geometry's standalone gain is composition too, and the shuffle shows it directly.* `mc_geom` on a
   flank shuffled within 25-nt blocks: +0.0800 (real: +0.0799). Shuffled across the whole arm (only the
   arm's composition kept): +0.0772. Shuffling destroys all local order and loses 3% of the gain.
4. *Which geometry columns are used:* of 27 geometry columns surviving selection (6.1% of model
   importance), the five largest are excess twist over 100-499 nt (the summed step twist minus 10.5 bp per
   turn), predictable from letter composition with out-of-fold R2 0.43-0.67 and from **dinucleotide
   composition with R2 0.86-0.97 by ridge (0.85-0.94 by boosting)**, 0.968 for the downstream 499-nt window:
   a sum of step twists is, to that accuracy, a weighted dinucleotide count. The others are path coordinates
   at +12 and +25 bp (R2 ~0 from letters, at most 0.15 from dinucleotides: they encode the identity of the
   first 12-25 nt after the PAM, which family A holds as position-specific columns).
   *Correction, 22 Nov:* this item originally read "R2 0.963". That figure came from a session whose code was not
   kept; re-running the analysis from `geom_importance.py` (now extended to compute it) gives 0.968 by ridge and
   0.937 by boosting for the same column, so the old figure is retired. The letter-composition values reproduced
   to three decimals, which is the check that the re-run is the same analysis.
5. *Why MD geometry is weaker alone (+0.031 against +0.078 / +0.080).* First hypothesis: the MD table's
   mean twist (32.5 deg) puts roll and twist out of phase. **Rejected**: recentring each source's mean twist
   to 34.3 deg changes nothing (md 0.5914 against 0.5897; xrc 0.6377 against 0.6367; mc unchanged). What
   does differ: the three sources disagree on each guide's excess twist (cross-source correlations 0.27 to
   0.66), and the MD value correlates with the label half as strongly as the others (rho -0.056 against
   -0.112 and -0.122). The excess twist is the carrier of dinucleotide composition, and the MD table
   assigns twist to dinucleotides in a way that tracks the label less. Cause beyond that not identified.
6. *Order matters in one place.* `mc_all` loses 0.043 when the flank is block-shuffled (0.6994 -> 0.6558)
   while `mean` and `geom` lose ~0.004 and 0.000; by elimination the loss is in `vec` (the first 8 bp of each arm;
   a block-shuffled `vec` alone was not run -- see 4.5), the region where family A also holds per-position columns.

### 4.4 The same question, asked of a different model class
Ridge regression on all columns (no selector, no trees; `ridge_check.py`, one seed, five gene-grouped
folds, alpha 30,000): base 0.5440; + A 0.6581; + mc_all 0.6547 (alone nearly as good as A);
**A + mc_all 0.6678 (+0.0097 over A, 5 of 5 folds)**, A + md_all +0.0078 (5 of 5), A + arm-shuffled
mc_all **+0.0002**. By block over A: mean +0.0054 (block-shuffled control +0.0018), vec +0.0048
(control -0.0004), geom +0.0020 (control +0.0004).

**So "shape adds nothing to family A" is not a property of the data; it is a property of the pipeline.**
A linear model can use real-order shape information beyond A (about +0.01, mostly from the
near-junction positions and the window means); the boosted trees already extract it from A and the base,
and gain +0.0006 at best. The claim that survives is: *for the model this project uses, shape and 3D
path descriptors are redundant given family A*; the claim that does not survive is that the pentamer
table carries no information beyond A. Whether the 300-column budget hides a small gain is tested in 4.5.

### 3.5 What a nuclease-conditioned pool is worth for a new enzyme: data efficiency
`nuclease_model.py --stage 6`, two replicates. The held-out nuclease contributes k labelled targets
(its training split) to the pooled fit, given 10% of the total sample weight whatever k is; the
alternative is a model trained on those k targets alone (200 trees, depth 4, not tuned). Mean test
rho over the nine enzymes:

| k labelled targets of the new enzyme | own model on k only | pooled + one-hot | pooled + label-free descriptors |
|---:|---:|---:|---:|
| 0 | -- | 0.792 | 0.808 |
| 100 | 0.713 | 0.820 | 0.828 |
| 500 | 0.807 | 0.843 | 0.851 |
| 2,000 | 0.843 | 0.859 | 0.865 |
| all (about 30,000) | 0.881 | | |

On the canonical-PAM subset (NGG): own-only 0.358 / 0.561 / 0.666 at k = 100 / 500 / 2,000; pooled with
descriptors 0.622 / 0.686 / 0.722 (k = 0: 0.634).

**Inference.** Pooling with the other eight enzymes is worth a lot when the new one has few labels: 500
labelled targets plus the pool (0.851) beat 2,000 targets alone (0.843), at least a 4x saving in
labelled data at that operating point; at k = 100 the pool is +0.115 over the enzyme alone. The
descriptors' own contribution, once labelled targets exist, is small but consistent: descriptors minus one-hot
+0.017 [-0.002, +0.046] at k = 0, +0.008 [-0.003, +0.021] at 100, **+0.008 [+0.003, +0.014] at 500 (8 of 9
enzymes ahead)**, +0.005 [-0.000, +0.010] at 2,000 (8 of 9). So the descriptor's contribution is a head start
that fades as the enzyme's own labels accumulate; the pooling is what carries the benefit.
**Limits:** the 10% weighting and the untuned k-only model are choices, not findings; the k-only baseline
could be stronger with tuning (it is the weaker, simpler arm here). **Status: verified** that pooling
beats the k-only fit at every k; **verified** for the sign and size of the descriptor gain at k = 500;
the k-only comparison's size is **partly verified** (untuned baseline).

## 5. Family A taken apart by its own column kinds (E. coli curated cohort, gene-grouped, 3 seeds)

Family A is four kinds of column lumped under one name: 104 position-specific one-hot columns near the
cut (`seq`), 156 per-position quantum-chemical tensor values at the same positions (`qct`), 64 windowed
means and sds of four tensor series at 50-1,000 nt (`win`), 24 composition summaries (`comp`: GC fraction,
purine fraction, longest run, each at four windows on each side). Base alone 0.5588; A 0.7151.

| columns | alone, over base | A minus this kind, versus A |
|---|---:|---:|
| `seq` (104) | +0.0552 | +0.0004 |
| `qct` (156) | +0.0701 | **-0.0054** |
| `win` (64) | +0.1088 | -0.0020 |
| `comp` (24) | **+0.1072** | -0.0003 |
| `seq` + `qct` (local, 260) | +0.0710 | |
| `win` + `comp` (windowed, 88) | +0.1088 | |
| letters in the same windows (40) | +0.1046 | |
| all of A (348) | +0.1563 | |

**Inference.** The flank effect has two parts. A **window-scale composition gradient** is worth +0.105 to
+0.109 and is carried by as few as 24 columns (8 of them selected): GC and purine fraction in windows of
50-1,000 nt are the whole of it, and plain A/C/G/T fractions in four windows do as well (40 columns). A
**local component** within about 10-15 nt of the cut is worth +0.05 to +0.07 alone, and the two overlap:
together they give +0.156, less than the sum. Removing any one kind from A costs at most 0.005, so A is
redundant within itself; the `qct` columns are the one kind whose removal registers. Nothing in the
shape/3D blocks of section 4 is outside these two parts: the window means reproduce the first (4.3.2), the
`vec` block reproduces the second (block-shuffling it loses 85% of its gain), and the 3D geometry is
excess twist (a dinucleotide count) plus the identity of the first 12-25 bp.

### 4.5 Checks on the null "shape / 3D adds nothing to A"

| check | result | what it rules out |
|---|---|---|
| vec block, flank shuffled (`mc_blockshuf_vec`, `mc_armshuf_vec`) | real vec +0.0534; block-shuffled +0.0076; arm-shuffled +0.0034 (13/15 folds) | vec's gain is local order, not composition: it loses 86-94% when order is destroyed |
| 600-column budget instead of 300 | A@600 0.7158; A + mc_all @600 0.7150 (-0.0008 against A@600); A + mc_geom @600 0.7141 (-0.0017) | displacement inside the 300-slot budget hiding a small gain (the budget itself is worth +0.0007 to A) |
| independent cohort: crisprHAL's *E. coli* eSpCas9 screen, 59,489 guides, 100-kb-block CV | base 0.6802, A 0.7836 (+0.1034), mc_all alone +0.0969 (94% of A), letters in windows alone +0.0784; A + mc_all **+0.0022** (14/15 folds), A + mc_geom +0.0008, **A + arm-shuffled mc_all +0.0016** (13/15) | the null being specific to one cohort or one nuclease; the real columns exceed their shuffled control by 0.0006 |
| fresh seeds 44-46 (folds never used) | A +0.1570 over base; mc all alone +0.1411; A + mc all -0.0006 (5/15 folds); A + arm-shuffled mc all -0.0010 | the null being specific to the seeds/folds used while the analysis was built |
| deeper trees: depth 6, 800 trees (the champion is depth 4, 400) | A@d6 0.7225 (+0.0073 over A: the champion is under-capacity for A); A + mc all -0.0012; A + arm-shuffled mc all -0.0023 | the null being the shallow trees' inability to use the extra columns |
| ridge instead of boosted trees | A + mc_all +0.0097 over A (5/5 folds); arm-shuffled +0.0002 (see 4.4) | the null being a property of the data rather than of the pipeline: it is not |

**Inference (Finding 8 in the report).** Everything cheap and sequence-derived that describes the flank's
shape or 3D path is a function of (a) its window composition and (b) the identity of its first ~10 bp;
family A holds both. The boosted pipeline gains at most +0.002 (eSpCas9) and at worst loses 0.001 (curated)
from adding them, and a shuffled-flank control with the same number of columns lands within 0.001 of the real
columns in both cohorts. A linear model sees about +0.01 of real, order-dependent information in the same
columns, which the trees extract from A and the base. **Status: verified** (two cohorts, three sources,
shuffled controls, two model classes, two budgets) that tensor-valued 3D flank descriptors do not improve the
champion; **verified** that the standalone gain is composition; **open**: whether a *measured* (not tabulated)
shape or a stiffness matrix would differ, and the human case.

### 3.6 Correction to 3.4: the within-PAM-class result depends on model capacity

Section 3.4 concluded that descriptors "do not tell the model the enzyme's own sequence-dependent
efficiency rules" (within-class rho 0.410 -> 0.408). That was measured at 500 trees x depth 6, the same
setting that had just produced the false "pooling hurts" result of 3.2. Repeating the leave-one-nuclease-out
run at 1,500 x 8 (three replicates, `nuclease_lono_bigcap*.csv`, `lono_preds_bigcap/`):

| (differences from U, label-free descriptors) | 500 x 6 | 1,500 x 8 |
|---|---:|---:|
| overall rho | +0.014 [-0.001, +0.039] | **+0.023 [+0.004, +0.055]**, 8/9 ahead |
| NGG only | +0.022 [-0.015, +0.065] | **+0.056 [+0.004, +0.117]**, 6/9 |
| PAM-class ordering | **+0.053 [+0.007, +0.103]** | **+0.047 [+0.007, +0.090]**, 5/9 |
| within PAM class | -0.002 [-0.039, +0.040] | +0.035 [-0.021, +0.100], 5/9 |

**Inference, revised.** The effect on PAM-class ordering is robust to capacity (+0.05 both times, intervals
clear of zero). The effect on ranking inside a PAM class is not: nil at the smaller model, +0.035 at the
larger with an interval that still includes zero; NGG-only ranking, which is the same thing restricted to
the canonical PAM, crosses zero at the larger model. A pooled model with enough capacity can learn
nuclease x sequence interactions, and that is the only reading that explains both this and the capacity
dependence of 3.2. **The report therefore says "PAM usage and little else", not "PAM usage, not sequence
rules".** Status: PAM ordering **verified** (two capacities x three replicates, shuffled control);
within-class transfer **open** (sign depends on capacity, nine enzymes cannot settle it).
This is the second time in this work that a capacity change reversed a reading; the first (3.2) would
have been reported as a finding had capacity not been varied.

### 3.7 The few-shot baseline is not a tuning artefact
`nuclease_fewshot_tuned.csv`: the k-only model with hyperparameters chosen by 4-fold CV on the k targets
(24-point grid over trees, depth, learning rate, column fraction) scores 0.712 / 0.804 / 0.847 at
k = 100 / 500 / 2,000, against 0.713 / 0.807 / 0.843 for the fixed setting. Pooled with descriptors (500 x 6):
0.828 / 0.851 / 0.865. So 500 labels plus the pool (0.851) match 2,000 labels alone (0.847): about a fourfold saving.
The pooled arms used the lower-capacity setting, so the saving is if anything conservative.

## 6. Audit

`ext/verify_numbers.py` recomputes 98 of the numbers quoted in Findings 8-9 and the Finding 3 addendum from
the result files and compares them with the report's text. First run: 3 mismatches, all rounding (A + mc geom
at 600 columns -0.0018 not -0.0017; the cutting screen's within-gene rho with A 0.7035 -> 0.703 not 0.704;
tuned k-only at k = 500 0.804 not 0.805). Corrected; second run 98 / 98.

*Current state of the audit (20 Nov):* it has grown with each section and now stands at **822 checks, 0 mismatches**,
covering sections 7-22. Run it with `./run_audit.sh` from `Runjia/` -- it sets the three environment variables the
modules need and exits non-zero if any quoted number no longer matches its file. Since 22.4 it also checks three things
about itself: that every module in the package imports and that importing one writes nothing; that four claims which
later work made false have not come back; and that the check count quoted in this sentence, in the report, in
`results/README.md` and in `PAPER_FLOW.md` is the count it actually ran. The four had drifted to 682, 784, 799 and 682.

Errors made and caught on the way, because they bear on how much to trust the rest:
* a splice that duplicated 350 lines of the report (the end marker preceded the start marker after an earlier
  reordering); caught by a line count, restored from the saved copy and redone with order assertions;
* "pooling hurts" (3.2) and "descriptors do not transfer sequence rules" (3.4): both capacity artefacts;
* a hypothesis that the MD geometry deficit came from its mean twist (4.3.5): tested and rejected;
* a dinucleotide-composition result first written as "+0.0005" that was +0.0025 over the shape means
  alone (corrected before use).

### 3.8 Sensitivity of the PAM-class-ordering metric to the class-size threshold
Label-free descriptors minus U, 500 x 6, bootstrap over nuclease: classes with >= 300 test targets: ordering +0.111 [+0.015, +0.178],
within class +0.006 [-0.011, +0.025]; >= 100 (the threshold used throughout): +0.053 [+0.009, +0.106], -0.002 [-0.038, +0.037]; >= 30: +0.014
[-0.004, +0.032], -0.011 [-0.036, +0.020]. The ordering gain is real for the well-populated classes and not detectable once rare
classes with noisy means are included. The report quotes the >= 100 figure and states this dependence.


## 7. Giving the model more of the enzyme: composition and structure, learned from many variants (added 9 Nov)

*Why this section exists.* In section 3 the enzyme enters the model as a handful of numbers per nuclease and there are
only nine nucleases, so nothing richer than a few columns can be learned. A richer description of the enzyme (its residues,
their chemistry, where they sit in the structure) only makes sense where there are enough enzymes to learn from. The one
public set found with a few hundred variants is Silverstein et al. 2025 (Nature 643): 770 SpCas9 variants that differ at six
PI-domain positions (D1135 S1136 G1218 E1219 R1335 T1337), each with the HT-PAMDA log10 rate constant for 64 PAMs (N + 3 nt).
Data: `training_data/220924_6pos_selected_random_only.csv` of github.com/RachelSilverstein/PAMmla (BSD-3), 729 KB, read as
text, no code from the repository run. Code: `pam_variants.py` (loader, descriptors, metrics), `pam_learn.py`,
`pam_analyze.py`. Results: `results/post_freeze/pam_variants/`.

**Column kinds of the sheet** (separated; only the six residues are inputs): identifiers (sample, prep, name); provenance
(selected/rational, variant source, source library, PAM selected on, shuffled?, extra mutations); a derived flag
(`Active variant?`, computed from the outcome columns: using it as an input would be leakage; used only to define an
evaluation subset); the six residues; the 64 outcomes. The `variant source` column is the selection campaign (7 of them:
five selected on different PAM sets, two unselected random libraries expRAS11/expRAS25); it defines the held-out unit.

**Descriptor groups** (per position, so added and removed independently): `id` one-hot residue identity (what PAMmla uses);
`comp` composition (hydropathy, side-chain volume, charge, H-bond donors, acceptors, aromatic, Gly and Pro flags: 8 per
position); `reach` structure x composition (the longest C-alpha to side-chain-atom distance of the residue, taken from the
structure itself, set against the C-alpha distance to the nearest atom of each of the four PAM base pairs in 5F9R; varies with
both residue and site); `env` structure only (the site's distances to the PAM base pairs; a constant per site, hence constant
over variants when the sites are fixed).
Structure helper checked against the biology: with antiparallel pairing (template residue 31 - r for non-target residue r) the
C1'-C1' distances are 10.3-10.6 A for the four PAM pairs; R1333 contacts non-target residue 22 (PAM G2) and R1335 residue 23
(PAM G3), as known for NGG readout. (A first version paired by nearest C1' and mispaired two of the four; caught by the
distances, replaced.)

**Design.** Leave one selection campaign out (7 folds; the analogue of leave-one-nuclease-out) and, for reference, random
10-fold. Three model classes (ridge, extra-trees, a 128-64 MLP) because two earlier readings in this project reversed with
capacity. Metrics are computed *inside* each held-out fold and averaged: pooling predictions across folds makes the
training-mean baseline score -0.38 (the held-out campaign is missing from its own training mean) and would hand every model
a spurious +0.4. `rho_pam` = mean over PAMs of the Spearman correlation across held-out variants (does the model order the
variants correctly for a given PAM); `skill` = 1 - SSE / SSE of the training-mean model. Nulls: identity columns shuffled
across training variants (`id_perm`), and the structure-only group.

| leave-one-campaign-out, rho_pam | ridge | extra-trees | MLP |
|---|---:|---:|---:|
| residue identity (one-hot, 120 cols) | 0.306 | 0.277 | 0.277 |
| composition only (48 cols) | 0.313 | 0.298 | 0.312 |
| identity + composition | 0.318 | 0.292 | 0.292 |
| identity + composition + structure x composition | 0.316 | 0.303 | 0.290 |
| structure x composition only | 0.213 | 0.288 | 0.243 |
| structure only (site distances) | 0.000 | 0.000 | 0.000 |
| identity, shuffled across variants | 0.001 | 0.010 | 0.002 |

Paired differences to identity, with a bootstrap over variants (200 resamples) and the number of the 7 held-out campaigns on
which the arm is ahead: identity + composition + structure x composition: ridge +0.010 [+0.006, +0.014] (7/7), extra-trees
+0.026 [+0.007, +0.040] (7/7), MLP +0.013 [+0.005, +0.022] (5/7). Composition alone vs identity: ridge +0.007
[-0.008, +0.021], extra-trees +0.021 [-0.001, +0.036], MLP +0.036 [+0.023, +0.050]. Random 10-fold gives the same ordering
at slightly higher levels (identity 0.349-0.373, identity + composition + structure x composition 0.366-0.383).

**Inferences, with their status.**
1. *A description of the residue's chemistry does as well as its identity, with fewer columns* (48 against 120) and better
   for the nonlinear models. **Verified** for three model classes and both splits; the size of the gain is small (+0.01 to
   +0.04) and the ridge difference is within noise.
2. *Adding the structure x composition feature gives a small, consistent further gain* (+0.01 to +0.03, ahead on 7/7 campaigns
   for ridge and extra-trees, 5/7 for the MLP). **Verified as a small effect**; whether it is the structure or the extra
   engineered residue-length feature that does it was not separated (`reach` alone is worse than `comp` for ridge and MLP).
3. *Structure that is fixed per site cannot help when the sites are fixed.* The structure-only group scores exactly the mean
   model in every model class, by construction (constant columns). This is not a finding about structure; it is the
   statement that structure can enter a learned model only through (a) the interaction with a variable residue, as `reach`
   does, or (b) sharing parameters across sites, which this dataset cannot test because the six sites never change.
4. *Which composition property matters.* Single-property ablations (extra-trees / MLP / ridge, rho_pam; full composition
   0.298 / 0.312 / 0.313): keeping only side-chain volume 0.284 / 0.193 / 0.154, only charge 0.256 / 0.272 / 0.250, only
   hydropathy 0.253 / 0.174 / 0.148, only H-bond donors 0.213 / 0.251 / 0.235, acceptors 0.135 / 0.136 / 0.124, aromatic
   0.103 / 0.103 / 0.120, Gly flag 0.051 / 0.064 / 0.066, Pro flag 0.027 / 0.022 / 0.025; dropping any single property leaves
   0.292-0.307 / 0.300-0.314 / 0.302-0.314. Size and charge carry most of it and no single property is necessary. The
   ranking of volume and hydropathy differs between model classes (a linear model cannot use a U-shaped dependence), so only
   the coarse statement "size, charge and H-bonding, not aromaticity or backbone flags" is supported. **Partly verified.**
5. *The ceiling is unknown.* The sheet has one measurement per variant, so there is no replicate-based reliability estimate;
   the three model classes agree with each other (0.28-0.32), which says the learners are saturated, not that the data are.

**What the bootstrap does not cover.** It resamples variants, not campaigns; with 7 campaigns the campaign counts above are
the more honest unit and they are reported next to the intervals.

**Whole-protein composition is not informative for these enzymes (computed).** Applying the mutation lists to the 1,368-residue
SEQRES of 5F9R chain B (every wild-type residue matches, which also verifies the lists' numbering) gives 20-amino-acid
composition vectors that differ from SpCas9's by L1 distance 0.004-0.010 (largest single-amino-acid change 0.003). A
whole-protein composition or sequence embedding of the nine Kim enzymes is therefore SpCas9's plus a perturbation of at most
1%; only the environment of the substituted residues distinguishes them.

## 8. A sequence-only PAM descriptor in the guide-efficiency model (stage 10; 9 Nov)

The PAM profile was the descriptor that carried the nuclease effect in section 3.4, but it was measured on each enzyme. Here it
is predicted from the six PI residues alone by models trained on the 770 Silverstein variants (ridge on identity + composition
and extra-trees on identity + composition + structure x composition, equal weight; variants with the identical six-residue
string removed from training; `pam_predict.py`) and handed to the same leave-one-nuclease-out experiment in place of the
measured profile. The nine Kim enzymes reduce to four six-residue strings (SpCas9, Sniper, eSp, HF1, Hypa, evo are wild type
at all six; NG V-S-R-F-V-R; VRQR V-S-R-E-Q-R; xCas9 D-S-G-V-R-T), so this descriptor takes four distinct values; it
cannot tell the high-fidelity enzymes apart from SpCas9, which is right for PAM usage and says nothing about their activity.
NG also carries L1111R and A1322R, outside the six positions, which this descriptor cannot see.

Check of the descriptor against the measured profile (class shares over the ten NxxN classes): Spearman over the 90
(nuclease x class) entries 0.886; per nuclease 0.67 (evoCas9) to 0.98 (SpCas9); the top class matches for 8 of 9 (NG: GT
predicted, GG measured). The predicted shares are sharper (GG share 0.98 for SpCas9 against 0.45 measured: HT-PAMDA rate
constants span far more orders of magnitude than library indel frequencies); the descriptor is z-scored over the enzymes, so
only the ordering enters.

Training rows were replayed from the stage-4 random stream, so the new arms are paired with U, D_label-free, D_pam and D_all
on identical rows; the first fit reproduces the stored stage-4 U prediction exactly (max difference 0.0). Nine nuclease,
3 replicates, bootstrap over nuclease (4,000):

| arm | rho | vs U | PAM-class ordering | vs U | within PAM class | vs U |
|---|---:|---:|---:|---:|---:|---:|
| U (no information) | 0.795 | | 0.855 | | | |
| measured PAM profile | 0.806 | +0.011 [-0.016, +0.037] | 0.900 | +0.045 [+0.007, +0.085] | 0.435 | +0.024 [-0.023, +0.068] |
| mutations + structure (no measurement) | 0.809 | +0.014 [-0.001, +0.039] | 0.908 | +0.053 [+0.008, +0.104] | 0.408 | -0.003 [-0.039, +0.036] |
| **sequence-predicted PAM profile** | **0.820** | **+0.025 [+0.002, +0.057] (7/9)** | **0.899** | **+0.044 [+0.0004, +0.091]** | 0.410 | -0.0005 [-0.056, +0.057] |
| sequence-predicted PAM + mutations + structure | 0.811 | +0.016 [+0.001, +0.039] (7/9) | 0.908 | +0.053 | 0.418 | +0.008 |
| same descriptors, shuffled across enzymes (nulls) | 0.752-0.777 | -0.018 to -0.043 | | | | |

Sequence-predicted minus measured PAM profile: rho +0.014 [-0.008, +0.038], PAM-class ordering -0.001 [-0.010, +0.008],
within class -0.025 [-0.076, +0.018] (4/9). **Inference: for PAM-class ordering, a descriptor computed from the enzyme's six
PI residues is as good as the measured PAM profile (difference -0.001, interval 0.02 wide), so the PAM screen need not be
run to get that part of the benefit.** Status: verified on nine enzymes of one protein, four distinct descriptor values;
what it does not show is that this transfers to enzymes whose changes lie outside the six positions (NG's two extra
substitutions, xCas9's six others), or to any other protein. Within a PAM class, neither measured nor predicted PAM
information helps (+0.024 and -0.0005, both intervals include zero), the same answer as sections 3.4 and 3.6. The largest
single beneficiary is VRQR (0.803 against 0.693 for U), whose six-residue string is not in the training set (nearest
variant at Hamming distance 2).

## 9. The enzyme as a position-resolved object: where its substitutions sit along the target (9 Nov)

A global enzyme descriptor treats every position of the target alike. Mapping each substituted residue through 5F9R onto the
target positions it neighbours (weight exp(-d/4 A), d the nearest heavy-atom distance to template-strand residue 10 + i for
protospacer position i counted from the PAM, and to the four PAM base pairs) gives each enzyme a 24-entry contact profile
(`enzyme_contacts.py`). The profiles are structurally sensible (HF1: positions 8-12 and 18-20; HypaCas9: 17-20; evoCas9: 7-11
and 18-19; Sniper: 11-14; NG and VRQR: the PAM and the first protospacer position). Test of whether the profile carries
enzyme-specific information: each enzyme's sequence dependence at a position (spread of the one-hot ridge coefficients,
fitted on its own training targets, scaled to sum to one over the positions so a global change in activity spread cannot
mimic a local one) minus SpCas9's, correlated across positions with the enzyme's contact profile; null by all 8! (or 5!)
pairings of profiles to enzymes (exact).

| enzymes | statistic (mean Spearman) | null mean (sd) | exact p |
|---|---:|---:|---:|
| all eight, protospacer positions | 0.264 | 0.081 (0.107) | 0.056 |
| all eight, with PAM positions | 0.368 | 0.083 (0.122) | 0.013 |
| **five high-fidelity only** (protospacer / with PAM) | 0.417 / 0.565 | 0.404 (0.035) / 0.551 (0.025) | 0.34 / 0.28 |
| three PAM variants only | 0.009 / 0.039 | 0.123 / 0.100 | 1.0 / 1.0 |

Charge, volume and H-bond-capacity weighted profiles: no pairing better than shuffled (p 0.12-0.86).

**Inference: the position-resolved enzyme profile is not supported as an enzyme-specific predictor.** The all-eight p-values
come from the contrast between the high-fidelity group and the PAM variants (shuffling across the two groups breaks it);
inside the high-fidelity group the correct pairing is no better than a shuffled one (0.417 against a null mean of 0.404), i.e.
the profile does not say which positions a *given* enzyme changed, only which kind of enzyme it is. A test run on the raw
(un-normalised) change gave p = 0.027 / 0.013 and was not used because it mixes in the enzymes' overall activity scale.
Eight tests were run (four profile types x two site sets, plus the group splits), so even the group-level p values would not
survive a multiplicity correction. **Verified as a null at this sample size; power is limited to eight enzymes.**

## 10. Human cohort: a side result on the human matrix, stopped on request (9 Nov)

*Scope note.* This was started as a replication of Finding 8 on human (the "human flank sequences" open item). The user then
said the aim is to improve predictive power on the *bacterial* datasets, so the human runs were stopped; nothing below
changes any bacterial result. They are recorded because one of them touches the correctness of numbers already in the report.

*What was built.* The 17,421 human protospacers were decoded from the monomer electron-count columns and located on hg38 as
single-copy NGG-flanked 20-mers (UCSC `hg38.fa.gz`, 983,659,424 bytes, gzip test passed; numpy 2-bit scan; `human_flank.py`):
16,490 located, 906 with more than one NGG copy dropped, 25 not found. The count equals the teammate's earlier located set,
which is an independent reproduction. Flanks of 1,000 nt per side in the guide's orientation (`curated_guide_index.csv` layout).
A first run of the "on top of A" arms had no family A at all, because `a_flank.build` skips rows without a `located` flag and
returned empty columns; caught because `n_ext` was 0 and the arm equalled the base to 16 digits; those arms were discarded and
re-run, the one valid result among them (shape without A) relabelled `h_base+mc`.

**Finding that matters: the human matrix has repeated guides, and random folds leak them.** Of 17,421 rows 13,138 are distinct
20-mers; 4,216 sequences occur more than once (4,149 twice, 67 three times), so 8,499 rows (49%) have a twin. Twins sit at the
same genomic position and their labels agree at Spearman 0.80 (59% exactly equal). With row-wise random folds a test guide's
twin is usually in training.

| human, 3 seeds x 5 folds | random folds | twin-grouped folds | locus-grouped (10 kb) |
|---|---:|---:|---:|
| base (guide sequence only), 300 cols | 0.3807 | 0.3063 | 0.2987 |
| + family A | 0.4287 | | 0.3169 |
| + A + shape/3D (mc) | 0.4470 | | 0.3319 |
| project's human baseline, champion, 300 cols (17,421 rows) | 0.4036 | **0.3356** | |
| project's human baseline, "library defaults, 1,000 trees, 400 cols" | 0.5076 | **0.2515** | |

Keeping identical protospacers together removes 0.074 of the 0.082 that locus grouping removes from the base, so the leak
is the twins, not locus structure. The high-capacity configuration that reached 0.5076 (and that the report reads as "depth
and learning rate are worth +0.10 on human") falls to 0.2515, *below* the champion's 0.3356: its gain was memorisation of
repeated sequences. **Status: verified for the two baseline arms; the other human statements in the report that rest on
row-wise folds (family A +0.052, "a linear model reproduces 64% of boosting on human", the cross-kingdom transfer
comparisons) were not re-run.**

**Partial and unexplained (not for the report body).** (i) Under locus-grouped folds the shape/3D blocks are +0.015 over A
(0.3169 -> 0.3319; at a 900-column budget 0.3193 -> 0.3311, +0.012), but plain composition columns added to A are +0.007 and
+0.005 there, and under random folds they were -0.035 and -0.025 (A+comp 0.3942, A+dcomp 0.4039 against A 0.4287). (ii) Adding
861 columns of pure N(0,1) noise to the base lowers random-fold rho from 0.3807 to 0.3372, and the shuffled-flank shape controls
(0.378, 0.373) sit at that level, so on the human matrix with random folds extra columns cost rho by displacing useful ones
in the 300-column selection; the comparison "shape vs composition" there is confounded by that, and I have not found the
mechanism. The grouped runs were stopped before the shuffled and noise controls could be repeated, so (i) is a hint, not a
result. (iii) Under random folds shape without A (0.4402) beat A (0.4287) and composition (0.3888-0.3953) by far; this is
the opposite of the bacterial cohort and may be the same twin effect (shape columns fingerprint a guide's locus).

**The bacterial cohort does not have this problem (checked).** `curated_guide_index.csv`: 33,567 rows, 33,567 distinct
protospacers; 87 guides have 2-8 genomic copies; 29% of guides have another within 19 bp (overlapping tiling), which the
gene-grouped folds used throughout keep together.

## 11. Tuning the E. coli model, outside the freeze (9 Nov)

*Aim (from the user): improve the E. coli model's rho; improvements may later carry to human data.* A joint search replaces the
earlier one-knob-at-a-time sweep, judged on genes it never saw. Code: `tune_ecoli.py`, `tune_analyze.py`; results
`results/post_freeze/tune_*.csv`, `tune_configs.json`.

**Design.** Curated cutting cohort, base + family A (6,517 columns), 30,203 guides with a gene label. Genes split once into DEV
(70%: 2,809 genes, 21,260 guides) and CONFIRM (30%: 1,204 genes, 8,943 guides); the split was asserted to share no gene.
Stage 1: 200 random draws + the champion + the old depth-6 setting, 5 gene-grouped folds of DEV (seed 41). Stage 2: the best 12
re-scored on three further gene-grouped partitions of DEV (seeds 42-44). Confirmation: train on all of DEV, score CONFIRM genes,
three seeds, paired bootstrap over genes. External: the same configurations on crisprHAL's eSpCas9 screen (59,489 guides, a
different nuclease, 5 folds grouped by 100-kb block, two seeds), nothing tuned on it. Search space: max_depth 3-10, learning rate
0.01-0.2 (log), subsample 0.5-1, colsample_bytree 0.3-1, min_child_weight 1-60, lambda 0.1-50, alpha {0..5}, gamma {0..2},
max_bin {64, 256}, column budget {150, 300, 450, 600, 900} (top of the fixed selector's ranking, computed once per fold), target
raw or rank-normalised, and the number of trees as a free dimension (each model fitted with up to 2,000 trees, scored at a grid
through `iteration_range`; finalists whose best count sat at the cap were re-run with up to 4,000, kept under a new name `...x`).
Column selection stays the project's fixed selector (XGBoost, 300 trees, depth 4); the champion is depth 4, lr 0.05, 400 trees,
subsample 0.8, colsample 0.8, 300 columns.

**What 200 draws say about each knob** (marginal means; `results/post_freeze/tune_stage1_effects.txt`). Shallower and slower is better:
mean rho by depth 3/4/5/6/7/8/9/10 = 0.712/0.714/0.713/0.707/0.706/0.704/0.700/0.698; by learning-rate quartile (low to high)
0.713/0.713/0.707/0.696; by column budget 150/300/450/600/900 = 0.698/0.709/0.711/0.711/0.708. Holding the others fixed (ridge on
the standardised settings), +1 sd of tree count is +0.006, of depth -0.003, of budget +0.002, of learning rate -0.002, rank-normalised
target -0.0015; subsample, colsample, min_child_weight, lambda, alpha, gamma and max_bin are within +-0.001. The optimum is a broad
plateau (depth 4-5, learning rate 0.013-0.04, 1,600-4,000 trees, 450-900 columns, raw target), not a point: the ten best draws
differ widely in every other knob and score within 0.003 of each other.

**Result on genes the search never saw** (CONFIRM, trained on DEV only, rho; difference to the champion with a 95% bootstrap
over genes; three seeds):

| model | rho | vs champion |
|---|---:|---:|
| champion (depth 4, 400 trees, lr 0.05) | 0.7073 | |
| depth 6 (the earlier finding) | 0.7126 | +0.0053 [+0.0027, +0.0083] |
| best tuned draw d099x (depth 5, lr 0.013, 4,000 trees, 450 columns, subsample 0.51) | 0.7215 | **+0.0143 [+0.0111, +0.0174]** |
| d077x / d021x / d000x | 0.7215 / 0.7216 / 0.7210 | +0.0143 / +0.0143 / +0.0138 |
| average of the five best (z-scored predictions) | **0.7246** | **+0.0174 [+0.0144, +0.0204]** |

The gain on unseen genes (+0.014) is as large as on the tuning folds (+0.012 at stage 1, +0.011-0.013 at stage 2), so the winner's
curse that inflated the earlier sweep is not what produces it; the interval resamples genes of one split and does not include
split-to-split variation.

**Carry-over to a different nuclease** (eSpCas9 screen, 10 paired folds; champion 0.7840): d021x +0.0227 [+0.0221, +0.0233],
d077x +0.0257 [+0.0247, +0.0266], d099x +0.0255 [+0.0246, +0.0262], depth 6 +0.0174 [+0.0165, +0.0183]; ahead on 10 of 10 folds
for every configuration. The intervals are over folds of one partition. The cohort is larger (59,489 guides), which is the
likeliest reason the gain is bigger there; it is not the screen the earlier final evaluation used as untouched, and nothing was
selected on it.

**The column budget is not a lever at the tuned setting** (one fold, config d077x): the selector gives nonzero gain to only 878
of 6,517 columns, and rho at 450 / 900 / 1,500 / 2,500 columns is 0.7208 / 0.7205 / 0.7202 / 0.7199 (best tree count each).

**The project's standard protocol on the whole cohort** (gene-grouped 5 folds x 3 seeds, selection inside folds, 15 paired
folds; `tune_fullcv.csv`): champion 0.7151 (identical to the earlier family-A figure, which checks that this code reproduces
the pipeline), depth 6 0.7223 (+0.0072 [+0.0062, +0.0082]), tuned d099x 0.7314 (+0.0162 [+0.0152, +0.0173]), average of the
five best 0.7334 (+0.0183 [+0.0175, +0.0191]); every tuned arm ahead on 15 of 15 folds. The tuned settings were chosen on 70%
of these genes, so this is the number to compare with earlier tables; the unbiased gain is the held-out-gene one above
(+0.014 single, +0.017 average), and the difference (about +0.002) is the size of what selecting on the same genes adds.

## 12. Mixing model families (9 Nov; `mix_ecoli.py`, `mlp_worker.py`)

*Why.* The five tuned XGBoost models are alike, so averaging them adds only +0.003 over the best; averaging helps in proportion
to how differently members err. Same DEV / CONFIRM genes as §11; each family gets its own random search on the same DEV folds
(seed 41) and cached column rankings: LightGBM 40 draws (leaves 7-63, lr 0.01-0.1, up to 3,000 trees), CatBoost 30 draws (depth
4-8, lr 0.03-0.15, up to 2,000 trees), extra-trees 3, a neural network 8, ridge 1 (all 6,517 columns). Finalists (2 LightGBM, 2 CatBoost, 1 of each other, the 5
XGBoost from §11) get out-of-fold predictions on DEV (seed-42 partition, for blend weights) and are trained on all of DEV for CONFIRM
(three seeds). **Process note:** the first smoke test stalled silently at 0% CPU for ten minutes at the neural-network step: PyTorch next
to XGBoost and LightGBM in one process (each ships an OpenMP runtime) hangs on macOS. The user noticed before I did. The network now runs in its own process
(`mlp_worker.py`), and later runs were confirmed busy before waiting.

Stage-1 bests on DEV (seed 41; max over the tree grid): LightGBM 0.7233, CatBoost 0.7237, XGBoost 0.7248, neural network 0.6988,
extra-trees 0.6642, ridge 0.6458. LightGBM and several XGBoost finalists sat at the tree-count cap (3,000), as in §11; not extended.

| CONFIRM genes (rho; difference to the XGBoost average of five, 95% bootstrap over genes) | rho | vs XGBoost average |
|---|---:|---:|
| champion | 0.7073 | -0.0174 |
| XGBoost average of 5 | 0.7246 | |
| LightGBM best / average of 2 | 0.7206 / 0.7235 | -0.0041 / -0.0011 |
| CatBoost best / average of 2 | 0.7207 / 0.7242 | -0.0039 / -0.0004 |
| extra-trees / neural network / ridge | 0.6657 / 0.6920 / 0.6470 | -0.059 / -0.033 / -0.078 |
| equal blend XGBoost + LightGBM | 0.7250 | +0.0004 [-0.0003, +0.0010] |
| equal blend XGBoost + CatBoost | 0.7262 | **+0.0016 [+0.0006, +0.0026]** |
| equal blend XGBoost + LightGBM + CatBoost | 0.7261 | +0.0014 [+0.0005, +0.0025] |
| ... + extra-trees / + neural network / + ridge (all equal weight) | 0.7224 / 0.7234 / 0.7226 | -0.0022 / -0.0012 / -0.0020 |
| weights fitted on DEV out-of-fold predictions (XGBoost .42, CatBoost .21, network .18, LightGBM .15, extra-trees .02, ridge .02) | 0.7264 | +0.0017 [+0.0005, +0.0030] |

Prediction correlations with XGBoost (Spearman, DEV out-of-fold): LightGBM 0.985, CatBoost 0.977, neural network 0.934, extra-trees 0.903, ridge 0.860; residual
correlations 0.973 / 0.959 / 0.882 / 0.836 / 0.767. **Inference:** the gradient-boosted families agree closely, so mixing them adds +0.0015;
families that do differ are too weak to help at equal weight. **Status: verified on one gene split;** the blend weights come from one OOF partition. (The *tuning* result was replicated on a second split in section 16.3; the **mixing** result was not, and mixing is the part the cost verdict rejects anyway.)

## 13. Against crisprHAL 2 again, and the cost of the extra accuracy (9 Nov; `vs_crisprhal.py`, `vs_crisprhal_wt_hal.py`, `cost_bench.py`)

Three conditions of the original comparison with four SLICER arms (champion, tuned XGBoost d099x, XGBoost average of 5, mixed ensemble =
the mean of the XGBoost, LightGBM and CatBoost group averages). crisprHAL 2's shipped WT-SpCas9 weights were run on their own WT test file (6,737 guides,
Spearman 0.6945 against their published 0.695) to get paired predictions; the eSpCas9 and five-fold predictions were the stored ones. Results, rho
and the paired difference to crisprHAL 2 (95% bootstrap over guides, 4,000 resamples) are tabulated in the report ("After retuning") and in
`results/post_freeze/vs_crisprhal_{espcas9,wt,folds}.csv`. The champion arm reproduces the earlier figures (0.7842 and +0.0021 [-0.0031, +0.0072] on their eSpCas9
split; +0.0115 [+0.0076, +0.0156] and +0.0044 [+0.0005, +0.0082] on our folds), which checks the pipeline; on the WT split it gives 0.6985
against the 0.6997 of the earlier `their_split_slicer` run, a 0.001 difference between two fits of the same configuration. **Result:** tuned XGBoost leads by
+0.0286 [+0.0239, +0.0331] (eSpCas9) and +0.0226 [+0.0139, +0.0314] (WT-SpCas9); the average of five and the mixed ensemble are within 0.003 of it
(0.8104, 0.8095 on eSpCas9).

*Cost* (one thread, same laptop; fits on 47,591 guides): original champion 400 trees, 0.04 / 0.21 / 16.2 ms for 1 / 100 / 10,000 guides, 2 s to fit; tuned 4,000 trees, 0.13 / 1.81 / 149 ms, 30 s;
average of 5 (16,000 trees) 0.54 / 6.28 / 535 ms, 132 s; mixed (9 models, 24,800 trees) 1.56 / 16.7 / 1,399 ms, 202 s; crisprHAL 2 (TensorFlow, one thread) 15.6 / 132 / 11,705 ms
(1.17-1.32 ms per guide in batches). **The model-only ratios of the inference section overstate the end-to-end case:** computing the 6,517 columns costs 1.41 ms
per guide on one thread (base 0.08, family A 1.33: warm, 300 guides), so end to end SLICER is about 10x faster for one guide and 7-22% slower per guide in
large batches (champion and tuned model; the mix 20-32% slower). A first, cold measurement gave 1.69 ms because the quantum tables were reloaded on each call; the warm figure is the one used.
**Verdict:** one tuned model is worth it (ten times the trees, 0.02 ms per guide, 30 s); averaging and mixing are not.

## 14. What the tuned settings say about the signal (9 Nov; `interpret.py`)

Seven readings, each tested rather than assumed (numbers in the report's Appendix B subsection and `results/post_freeze/interpret_*.csv`): the interaction-order curve (stumps 0.6905, depth 2
0.7177, depth 3 0.7206, depth 4 0.7210 on CONFIRM, ridge 0.647: 59% of what trees add to a linear model is per-column nonlinearity, 37% pairs, 4% third and
fourth order); the gain by column kind (indicators 35% -> 51%, quantum 37% -> 27%, flank 27% -> 22%; 131 -> 223 indicator columns in the top
300 / 450); the gain by position (eight PAM-proximal positions 63% / 62%; position 18 alone 14%; 3 nt after the PAM 6% / 5%); within-gene against
between-gene gain (+0.018 each); data scaling (+0.0095, +0.0061, +0.0143 at 25%, 50%, 100% of the tuning genes). Limits: gain shares are model-dependent importances over correlated columns, positions
are k-mer start positions (positions 19 and 20 have few columns), the data-scaling points use 3 random gene subsets and the 50% point is out of order.

## 15. Overnight: where crisprHAL 2 could still lead, and what was done about it (9 Nov; the user asked for further tests aimed at beating crisprHAL 2 on any earlier metric)

Selection of tests (each argued before it was run): the original panel had crisprHAL 2 ahead on Pearson, R², top-decile precision and pick percentile; speed had just
turned out to be a tie once featurisation was counted; data efficiency had not been re-run with the tuned model. So: (1) the full six-metric panel for the tuned and mixed
models on both of crisprHAL 2's splits, with raw predictions kept (`panel_vs_hal.py`; R² and calibration need unscaled predictions, which the earlier z-scored blends did not
give); (2) smoother top-end metrics and five-seed stability (`topk_panel.py`); (3) top-end objectives (`topend.py`: weighted regression, pairwise, NDCG), with a capacity-fair
rerun of the ranking objective before calling it a null (`topend_rank_capacity.csv`: best over three learning rates, two depths, four tree counts 0.6017, against 0.7215 for squared error);
(4) the data-efficiency curve with the tuned model against the stored crisprHAL 2 curve (`data_efficiency.py`); (5) array-based featurisers (`a_flank_fast.py`, `base_fast.py`,
`e2e_bench.py`, `equiv_check.py`); (6) blending with crisprHAL 2 (`blend_with_hal.py`); (7) a comparison on genes untouched by any search (`clean_vs_hal.py`).

**A correction made on the way.** In §13 and the report I wrote that nothing was selected on crisprHAL's splits. That is true of the eSpCas9 split only. crisprHAL's WT-SpCas9
data are the same 33,567 guides as the curated cohort whose genes were split into tuning and confirmation parts in §11, so the tuned *settings* (not any model's weights) were selected using
labels of guides that appear in the WT-SpCas9 test split and in the five-fold comparison. The report now says so, and (7) gives the comparison that excludes them: on the 8,943 CONFIRM guides,
tuned XGBoost trained on the tuning genes only leads crisprHAL 2's out-of-fold predictions by +0.0324 [+0.0248, +0.0403] (published configuration) and +0.0240 [+0.0162, +0.0318] (best-found
configuration), a conservative design because crisprHAL 2's models were trained on guides from the same genes.

**Results** are tabulated in the report subsection "Metric by metric, seeds, data efficiency, an untouched-genes check, and a faster featuriser". In short: four of six metrics clearly ahead on both splits
(ρ, τ, Pearson, R²), top-decile precision and pick percentile tied (intervals include zero; precision of the top 5% +0.048 [+0.003, +0.074] on WT-SpCas9); no objective change improves the top end; the lead is
stable across five seeds (+0.0284 eSpCas9, +0.0221 WT-SpCas9 on average); the tuned model is ahead on every fold at every training size and matches crisprHAL 2's 26,853-guide accuracy at about 9,100 guides; the array-based
featurisers reproduce the stored columns exactly (0.0 difference on 33,567 guides; predictions identical) and make end-to-end scoring 21× / 15× / 13× faster than crisprHAL 2 for 1 / 100 / 10,000 guides
(0.72 / 0.089 / 0.093 ms per guide against 15.29 / 1.365 / 1.214); averaging with crisprHAL 2 adds +0.0085 [+0.0040, +0.0127] on WT-SpCas9 and nothing on eSpCas9.

**Not done / limits.** crisprHAL 2 was not retuned or seed-ensembled on its own splits (GPU time), so the lead is against its shipped model. The panel intervals resample guides of one split. The end-to-end timing
excludes retrieving the flanking sequence from a genome (the same for both models) and the first timing run shared the machine with a finishing job (rerun on an idle machine gave the figures quoted). GPU timings were not repeated.

## 16. Three gaps closed after a review of the whole project (19 Nov; the user asked for all items of a critique to be done)

*Why these three.* A full read of the report against its own result files found three things worth compute, each one a
claim that was either resting on the wrong model, resting on leaked folds, or resting on a single split.

### 16.1 Family A's increment at the tuned settings — the headline biology number against the headline model

*The gap.* Every version of the +0.164 flank increment was measured with the **champion**, and the fork's search then
found +0.016 from hyperparameters alone, so the report's headline accuracy figure and its headline biology figure
belonged to two different models. `tune_fullcv.csv` had four arms and **all of them included family A**; no base-only
arm existed at the tuned settings. Two outcomes were possible: if the tuned model extracts more from the 6,169 base
columns, the flank increment shrinks and part of what family A was credited with was unexploited capacity; if the extra
capacity goes into the flank columns, it grows.

*Design* (`family_a_at_tuned.py`). Standard protocol so the numbers line up with `tune_fullcv.csv`: gene-grouped 5 folds
x 3 seeds on the 30,203 gene-labelled guides, selection fitted inside each training fold, four configurations
(champion, depth6, d099x, d077x) on two matrices. The matrices differ only in what the selector is offered -- base-only
ranks the 6,169 base columns and an assertion rejects any ranking that reaches a family A column. Resumable per fit.

| 15 folds, paired, bootstrap over folds | base only | base + family A | family A's increment |
|---|---:|---:|---:|
| champion | 0.5588 | 0.7151 | **+0.1563** [+0.1515, +0.1616], 15/15 |
| depth6 | 0.5569 | 0.7223 | +0.1655 |
| **tuned d099x** | 0.5617 | **0.7314** | **+0.1697** [+0.1659, +0.1738], 15/15 |
| tuned d077x | 0.5631 | 0.7310 | +0.1680 |
| best-of-four (d099x over d077x) | 0.5631 | 0.7314 | +0.1683 [+0.1642, +0.1728] |
| **tuning gain (d099x - champion)** | **+0.0028** [+0.0009, +0.0050], **9/15** | **+0.0162** [+0.0152, +0.0173], **15/15** | |

**Result, and it is the cleanest thing the fork has produced.** Family A's increment **grows** with tuning, +0.156 ->
+0.170. The mirror image is sharper: the same hyperparameter change is worth **+0.003 on the base representation alone**
(positive in only 9 of 15 folds) and **+0.016 with family A present** (15 of 15). The tuned settings are not buying
generic capacity the champion left on the table; they are buying the ability to *use the flanking columns*. That turns
the apparent refutation of Finding 4 ("the model is at its material limit") into the sharpest evidence for it, and the
finding's heading was changed from "at its material limit" to "out of *material*" to say so.

*Harness check:* champion base+A = 0.7151 and d099x = 0.7314, both identical to `tune_fullcv.csv`, so this is the
established protocol and not a third one.

*Caveat, with the size of it computed.* The tuned configurations were chosen on DEV genes **with family A present**, so
the base-only column may be slightly pessimistic; four configurations per side and the best-of-four bound guard against
it, but a matched search over base-only columns has not been run. For the reading to be overturned -- family A's tuned
increment falling back to the champion's +0.1563 -- a base-only search would have to reach rho **0.5751**: **+0.0120
beyond the best of the four tried** (d077x, 0.5631) and about **2.9x the largest base-only gain any of them produced**
(+0.0042). Worth one night if this conclusion ever has to stand alone, but the margin is not marginal.

### 16.2 The human additivity claim under twin-grouped folds

*The gap.* Section 10 found that 8,499 of the human matrix's 17,421 rows (49%) are a repeat measurement of a guide that
appears elsewhere, and that the teammate's configuration falls from 0.5076 to 0.2515 under twin-grouped folds. The
correction was recorded in Appendix B.4 with an honest list of what had not been re-run -- but three statements resting
on those folds were still in the **main text** without a caveat, one of them a contribution-level biology claim ("the two
kingdoms differ in interaction order, and two unrelated instruments agree on it").

*The logical problem, which is worse than the missing caveat.* Every knob that "helped" on human is a memorisation
knob: depth 4->6 (+0.0585), learning rate 0.05->0.3 (+0.0416), trees 400->1,000 (+0.0361), while turning subsampling
*off* cost -0.0075. Memorising a sequence whose twin is in the training fold is exactly what deep, fast-learning trees
do, so the "second, independent instrument" is the first artefact's signature rather than separate evidence -- and the
63.7% ratio has the leaked 0.5076 in its denominator.

*Design* (`human_additivity_twin.py`). Ridge and the whole knob sweep, under **both** fold schemes in one run with the
same columns and seeds, so the only thing that differs is the folds and the row-wise arms act as a positive control for
the harness. Twin-grouped folds are asserted to share no protospacer across the split.

*Expectation recorded before the run:* the ratio should *rise* under grouped folds, because ridge has no mechanism for
memorising a twin and boosting does; if it rises far enough the claim inverts and should be withdrawn rather than
rescoped.

| 400 columns, 3 seeds x 5 folds | row-wise rho | twin-grouped rho |
|---|---:|---:|
| champion (depth 4, lr 0.05, 400 trees) | 0.4074 | **0.3377** <- now the best boosting arm |
| `max_depth` 4 -> 6 | 0.4659 | 0.3315 |
| `learning_rate` 0.05 -> 0.3 | 0.4490 | 0.2560 |
| `n_estimators` 400 -> 1,000 | 0.4435 | 0.3226 |
| no row/column subsampling | 0.3999 | 0.3333 |
| all three ("library defaults"; the 0.5076 configuration) | 0.5076 | **0.2515** |
| ridge | 0.3232 | 0.3001 |
| **ridge / best boosting** | **63.7%** | **88.8%** |
| **`max_depth` 4 -> 6** | **+0.0585** | **-0.0062** [-0.0093, -0.0034] |

**The positive control is exact.** Every row-wise figure reproduces the published one to four decimals (0.40741 against
0.4074, 0.50761 against 0.5076, 0.46588 against 0.4659, 0.44896 against 0.4490, 0.44354 against 0.4435, 0.39990 against
0.3999, 0.32315 against 0.3232; ratio 63.66% against 63.7%; depth +0.05847 against +0.0585). So the difference between
the columns is the folds and nothing else.

**Result: the claim is withdrawn, and the expectation was right about the direction and wrong about how far.**
(i) Every knob that helped now hurts, and the **champion is the best of the six arms** -- so "the champion is
mis-specified by 0.10 rho on the human matrix" is withdrawn too; that was the leak, not a shape mismatch.
(ii) The ridge/boosting gap between kingdoms falls from 31.9 points to **6.8** (95.6% against 88.8%), on column sets
that were never comparable (427 decorrelated for *E. coli*, 400 selected for human) -- not enough to carry a biology
claim. (iii) The second instrument **changes sign**: depth is +0.0063 in *E. coli* and -0.0062 in human, interval clear
of zero. The claim's whole force was that two independent instruments agreed; they now disagree, and the one that looked
decisive was measuring memorisation.

*Withdrawn, not rescoped:* Part 2 biology claim 4, the additivity half of the two cross-kingdom observations, the
champion-mis-specification reading in Part 3, and Part 4's item 4 as it stood. Nothing replaces them; establishing a
kingdom-level interaction-order difference would need a deduplicated human matrix and a search run under grouped folds.
*Still standing:* the GC inversion (a marginal association, no model to fit), the *E. coli* additivity result on its own,
and the plain observation that the human matrix is much harder (0.3377 against 0.7314).
*Not re-run and still on row-wise folds:* the +0.052 human flank gain and the cross-kingdom transfer figures, including
the family-A-on-both-sides result that reverses the sign. Those are flagged in Part 2 and Part 3.

*An uncomfortable corollary worth stating.* Appendix B argued that hyperparameters are a hundredth-scale knob on this
problem, with the human matrix as the one exception where they were worth +0.10. The exception is gone, so the
generalisation is **stronger** than it was: tuning is a hundredth-scale knob on both matrices. That, together with 16.1
(the tuning gain collapsing to +0.003 without family A), is two independent routes to the same conclusion -- the
learner was never the binding constraint.

### 16.3 A second gene split, with its own search

*The gap.* The whole tuning result rests on one partition (`tune_ecoli.SPLIT_SEED = 2026`): the +0.0143 on held-out
genes, the configurations, the blend weights, the untouched-genes comparison. The bootstrap intervals resample genes
*within* that split, so they cannot show split-to-split variation.

*What would be wrong.* Re-scoring d099x on a different CONFIRM set is contaminated -- d099x was chosen using split 1's
DEV genes, and a new split reshuffles them into both halves. The claim at risk is not "d099x is good" but "a search on
70% of the genes yields a configuration that beats the champion on the other 30%", so the replication has to re-run the
**procedure**: a fresh search on the new DEV, a fresh winner, confirmed on genes that search never saw. A different
winner is the expected outcome, not a problem.

*Design* (`tune_split2.py`). Genes re-split 70/30 with seed 7 (asserted to share no gene, and the overlap with split 1's
CONFIRM set is printed); 40 fresh configurations from a different stream of the same search space; the best 3 re-scored
on one further partition with the extended tree grid; champion, the new winner and the top-3 average confirmed on the
new CONFIRM over three seeds with a gene bootstrap. d099x is also scored there as a secondary row, labelled
`contaminated`. Resumable draw by draw, so a partial run is still an answer.

Split 2 gives DEV 21,222 / CONFIRM 8,981 guides (2,809 / 1,204 genes), and **357 of the 1,204 new CONFIRM genes were
also CONFIRM in split 1** -- so 847 of them fed split 1's selection, which is exactly why d099x's row here is labelled
contaminated and why a fresh search was necessary.

| split-2 CONFIRM genes (rho; difference to the champion, 95% bootstrap over genes, three seeds) | rho | vs champion |
|---|---:|---:|
| champion | 0.7329 | |
| depth6 | 0.7374 | +0.0045 [+0.0013, +0.0077] |
| **split-2 winner s2d017** (depth 6, lr 0.020, 2,000 trees, 450 columns) | **0.7450** | **+0.0121 [+0.0092, +0.0152]** |
| s2d002x / s2d000x | 0.7446 / 0.7435 | +0.0117 / +0.0105 |
| **average of the top 3** | **0.7488** | **+0.0159 [+0.0132, +0.0186]** |
| *d099x, split 1's winner (contaminated: 847 of these genes fed its selection)* | *0.7472* | *+0.0143 [+0.0112, +0.0176]* |

**Result: the tuning payoff replicates.** A fresh 40-draw search on a different 70% of the genes yields
**+0.0121 [+0.0092, +0.0152]** on genes it never saw, against **+0.0143 [+0.0111, +0.0174]** from the 200-draw search on
split 1. The intervals overlap substantially, and the ensemble figures agree too (+0.0159 against +0.0174). The split-2
estimate is a little smaller, which is consistent both with less winner's curse and with less search power at 40 draws
instead of 200; this design cannot separate those two.

**A different winner, the same region of the space.** s2d017 is depth 6, lr 0.020, 2,000 trees; d099x is depth 5,
lr 0.013, 4,000 trees. Both sit at 450 columns on the raw target with slow learning, and the split-2 top three span
depth 3 to 6, min_child_weight 2 to 51 and lambda 0.76 to 15.3 while scoring within 0.0015 of each other. That is
section 11's "broad plateau, not a point" reading, confirmed on an independent partition.

**A bonus that was not the point of the run: an independent estimate of the selection effect.** d099x scored on these
genes reads +0.0143, but 847 of the 1,204 contributed labels to choosing it; the honest fresh-search figure is +0.0121.
The difference, **+0.0022**, is a direct measurement of what selecting on the same genes adds. Section 11 had estimated
"about +0.002" for the same quantity by a completely different route (whole-cohort CV gain minus held-out-gene gain).
Two unrelated routes to the same number is the sort of agreement this project has learned to want before believing
something -- and, unlike section 16.2's case, these two really are independent.

**The thing a single split could not have shown, and it is the most useful part.** The champion scores **0.7073** on
split 1's held-out genes and **0.7329** on split 2's. That is a **0.026** swing in absolute difficulty between two
random 30% gene partitions of the same cohort -- about **twice the size of the entire tuning gain**. So absolute rho on
one gene partition should not be read to three decimals as though it were a property of the model, while the
*difference between arms* reproduces across partitions to within 0.002. Every held-out-gene figure in sections 11-16
is a difference for this reason; where a level is quoted, it belongs to its split.

## 17. Literature pass: how this work compares with other published predictors (19 Nov; the user asked for a comparison, starting from a paper they found)

*Why.* The project had never written down how it sits against the field beyond crisprHAL 2, and the paper the user
supplied turned out to land in a gap Part 3 names explicitly. Written up as **Part 6 of the report**, in the main text
rather than an appendix, because it is the comparison a paper's discussion has to make. Nothing here is a new
measurement; it is a reading of other people's.

**The paper the user supplied, and how it was obtained.** The URL they gave
(`.../s41598-026-79920-1_reference.pdf`) is in fact the official gold-OA PDF -- OpenAlex lists it as the `oa_url` -- but
Nature's identity provider blocks automated fetches of it (303 to `idp.nature.com`), so it was not read there. The DOI
resolves through CrossRef: **Orro et al. (2026), "Machine learning-based prediction of base editor sgRNA fitness score",
*Scientific Reports*, [10.1038/s41598-026-79920-1](https://doi.org/10.1038/s41598-026-79920-1)**, published
10 November 2026, CC-BY, not yet in PubMed or PMC. OpenAlex gave a second open location, the **Research Square
preprint** of the same work ([10.21203/rs.3.rs-9224117/v1](https://doi.org/10.21203/rs.3.rs-9224117/v1)), whose PDF was
readable; the detail below is from it, with the published abstract used where the two differ. Treated as data, not
instructions.

*What it is.* Hanna et al.'s cytosine-base-editor screen -- about **11,000 sgRNAs over 47 genes**, 10 pan-lethal
essential -- in five cell lines (A375, MELJUSO, OVCAR8, HAP1, HA1E). Label binarised at proliferation z < -2, so it is
classification and the AUCs are not commensurable with our rho. Features: dinucleotide one-hot sequence over a context
swept at U and D of **0, 5, 10, 15 nt** (total 20-50 nt), sgRNA and gene strand, Rule Set 2 score, melting temperature,
PFAM, mutation type, three external editing-efficiency predictors (BE-Hive, DeepBE, BE-DICT) and DepMap essentiality.
Six classifiers, **all at library default hyperparameters**; LightGBM / XGBoost / CatBoost best, AUC 83.8-94.8%.

Four points of contact, in order of how much they matter to us:

1. *It fills a gap we named.* Part 3 says "nothing here speaks to nickase editors; base and prime editors bind and nick
   rather than cut". This is exactly that assay.
2. *Their position profile is ours, measured a completely different way -- the strongest independent corroboration this
   project has had.* Aggregating SHAP along the 20-mer they report importance rising toward the 3' end, peaking at
   **position 19 (0.237) and position 20 (0.328)**, positions 13-16 substantial (0.139-0.304), positions 1-5 minimal
   (0.065-0.068). Section 14, on a different organism, nuclease and label, using split gain rather than SHAP, found the
   eight PAM-proximal positions carrying **63% / 62%** of the gain and positions 1-12 only 10% / 17%.
3. *Strand dominates their non-sequence features, as it dominated our dCas9 screen.* Their largest non-sequence feature
   is **gene strand (0.493)**, ahead of DepMap essentiality (0.179) and mutation type (0.101). Finding 3's hardest
   correction was that the +0.268 family-A gain inside essential genes was *the screen's own strand rule*, worth
   +0.007 to +0.025 once a strand indicator was supplied. Two assays, two kingdoms, same trap: control strand before
   reading any subgroup result off a pooled screen.
4. *An apparent contradiction that resolves into a mechanism.* They find essentiality predictive (0.179, and it "drives
   the global difference between BE3.9 and Cas9 activity"); we find it a flat null (+0.00004, reproducing Guo et al. to
   four decimals). Both are right, because the labels differ: a **fitness** effect in human cells depends on whether the
   edited gene matters, while survival after a double-strand break in *E. coli* is lethal more or less regardless of
   which gene was cut. Essentiality should predict one and not the other. This is the report's own "two screens' numbers
   are comparable only when the measured quantity is" rule, applied to someone else's paper.

*One concern worth recording, stated as a question rather than a criticism.* With 47 genes, gene-level features among
the inputs (DepMap essentiality is constant within a gene), and a preprint whose methods describe plain 5-fold
cross-validation repeated over ten seeds, sgRNAs from the same gene would sit on both sides of most splits. The
**published abstract specifies "gene-held out validation"**, so the final version appears to have addressed precisely
this; the preprint's methods do not say it. If the published scheme is gene-held-out the concern is answered -- and that
it needed answering is the lesson this project learned twice, once by gene and once, expensively, by duplicate guide
(section 16.2).

**The closest methodological neighbour, which we had not cited at all: Yu et al. (2024), *Genome Biology*,
[10.1186/s13059-023-03153-y](https://doi.org/10.1186/s13059-023-03153-y)** (Barquist lab), bacterial CRISPRi guide
efficiency. Found via PubMed/Europe PMC; full text read. It matters for three reasons:

- *It reaches our decomposition independently.* Their headline is "the surprising discovery that gene-specific features
  substantially impact prediction", and their answer is a **mixed-effect random forest** — 129 guide features as fixed
  effects, 9 gene features as random effects, 301 genes as clusters — separating guide efficiency from the targeted
  gene. That is the same guide/gene split as our within-gene versus between-gene analysis and as Finding 2's "+0.061
  asked entirely inside genes".
- *It validates the way we evaluate.* They abandon guide-wise cross-validation for a **gene-wise** scheme for our
  reason, and additionally report train/test Hamming distances (minimum 7, median 21). Orro et al. also hold out genes.
  So gene-grouped evaluation is what a reviewer will expect, which is what this report quotes.
- *It is ahead of us in one respect.* They *model* the gene effect as a random term; we group folds and decompose
  afterwards. Logged here as an open item and **run the same day** -- see 18.1: a back-fitted mixed-effect version of
  the tuned model moves accuracy from between-gene to within-gene exactly as predicted (-0.0080 and +0.0016) and leaves
  pooled rho unchanged at -0.0003, because on a *cutting* screen the gene term is signal rather than nuisance.

Their numbers, which are not comparable to ours (CRISPRi depletion, different label) but are informative about the
assay: guide sequence plus PAM plus 4 nt upstream and 3 nt after the PAM gives rho ~0.21; adding thermodynamics and
distance to the start codon gives ~0.25; adding gene-level covariates (expression, transcription-unit structure, gene
GC, gene length, distance to operon start) gives **~0.66**. Their dominant feature is maximal RNA expression. This is
consistent with our Finding 3 reading that in a dCas9 screen the gene context dominates the ranking and the flank
effect is weak and strand-confounded.

**The novelty question, asked properly.** Of everything read in this pass, the longest input any predictor takes is
crisprHAL 2's 378 nt (SpCas9) / 406 nt (eSpCas9), which its authors reached by their own length search. Everyone else who
reads flanking sequence reads very little of it: Orro et al. sweep +-15 nt, crisprHAL 37 nt in total, Yu et al. 4 nt
upstream and 3 beyond the PAM, CRISPOR 50 bp each side for scoring. The other routes outside the protospacer are an
*experimental* track (CRISPRon's chromatin accessibility; Xiang et al. 2021) or *gene-level* covariates (Yu et al.'s
expression and operon structure; Orro et al.'s essentiality). **Nothing found builds compositional descriptors of
flanking DNA in windows of hundreds to a thousand nucleotides and measures their increment.** Caveat stated rather than
buried: a literature search failing to find something is weak evidence that it does not exist, and this was a handful of
searches plus citation chasing, not a systematic review.

**The claim we should stop making, found by looking.** "A boosted tree over engineered features matches a tuned deep
model" has a published precedent on human data: **BoostMEC** (Zarate et al. 2022, *BMC Bioinformatics*,
[10.1186/s12859-022-04998-z](https://doi.org/10.1186/s12859-022-04998-z)) is LightGBM over direct and derived sgRNA
sequence features, benchmarked against **10 models on 13 external datasets**, and its authors argue for it on exactly
our grounds (competitive accuracy, better interpretability). They have far more external validation than we do. So the
architecture is not the contribution; the *reach* of the representation is, and section 16.1's result -- the tuning gain
falling from +0.016 to +0.003 without the flank columns -- is the sharpest internal evidence that the features and not
the learner are doing the work.

**A calibration against our own headline.** Recent independent benchmarks of the human predictors put CRISPRon first
with Rule Set 3 (DeWeirdt et al. 2022) and DeepSpCas9 close behind, and the ordering **changes between datasets and
organisms** - one plant evaluation found older, simpler scores tracking in-planta efficiency better than either deep
model. Differences of that size are the size of our lead over crisprHAL 2 (+0.023 to +0.029 rho on two splits of one
organism's screens). So the lead is worth stating as measured, audited and confirmed on untouched genes, against
crisprHAL 2 **as shipped** - and not as a claim that one architecture beats another. The sentence this report has always
defended, that the advantage is about *reach*, is the one that survives the comparison.

**Also noted, not pursued.** The same group's 2025 SaCas9 preprint reports that dinucleotides at +1/+2 beyond the
NNGRRN PAM and PAM adenine methylation affect activity in bacteria - independent evidence that flanking sequence
matters in bacteria, at a range (1-2 nt) three orders of magnitude shorter than ours, and through a mechanism
(methylation) we have no feature for. A possible addition to family A if the project ever returns to feature building.

## 18. Importing two published design choices, and both are nulls with a mechanism (20 Nov; `lit_followups.py`)

*Why.* Section 17 found two things other people do that we never had. Importing them is the cheap way to find out
whether our choices were right or merely ours. Both run on `tune_ecoli`'s DEV/CONFIRM gene split with the tuned
configuration, so the regression baseline reproduces the 0.7215 already on record.

### 18.1 Yu et al.'s mixed-effect model: the gain is assay-specific, and it is not ours to have

Yu et al. (2024) fit bacterial CRISPRi guide efficiency with a mixed-effect random forest -- guide features fixed, the
targeted gene random -- and report the gene term as their main methodological contribution. We only ever *grouped
folds* by gene, which removes gene leakage but not gene-level noise from the training signal. Hypothesis recorded before
running: a random intercept absorbs the between-gene level differences that cannot transfer to an unseen gene, so the
fixed part should become a cleaner estimate of *within-gene* efficiency even if pooled rho does not move.

Implementation is back-fitting: alternate a boosted fixed part with shrunk per-gene intercepts,
`b_g = n_g/(n_g+lam) * mean(residual in g)` and `lam = sigma2_eps/sigma2_b` re-estimated each round, six rounds.
Held-out genes get `b = 0` by construction, which is the point -- the fixed part has to carry everything that transfers.

| CONFIRM genes, 3 seeds | pooled rho | between-gene | within-gene |
|---|---:|---:|---:|
| tuned regression (d099x) | 0.7215 | 0.7293 | 0.7124 |
| mixed effects, gene intercept | 0.7213 | 0.7213 | **0.7140** |
| difference | -0.0003 | **-0.0080** | **+0.0016** |

**The hypothesis was right about the direction and small about the size, and the reason is the assay.** The random
intercept does what it is supposed to: it moves accuracy from between-gene to within-gene, trading -0.008 for +0.0016.
But on a *cutting* screen the gene-level component is not nuisance -- our between-gene rho is **0.729**, i.e. gene-level
prediction is good, and much of it is carried by family A, because neighbouring loci share flank composition. In Yu et
al.'s CRISPRi setting the gene term *is* mostly nuisance: guide-only features reach rho ~0.21 and it takes gene-level
covariates (expression, operon structure) to reach ~0.66, so there the gene effect dominates and removing it is worth a
lot. **The mixed-effect gain is therefore a property of depletion-screen labels, not of bacterial guide prediction**, and
our fold-grouping had already collected the part that transfers. That is a cleaner answer than "we should have done it
too", and it is only available because the two assays sit side by side in this project.

### 18.2 Orro et al.'s binarised classifier: discretising the label loses more than the top end gains

Orro et al. (2026) binarise their fitness label and fit classifiers. Section 15 had tried weighting high-activity
guides, pairwise ranking and NDCG -- none helped -- but never the plain thing: train on "is this guide in the top X%".
Top-decile precision and the pick percentile of the ten best-scoring guides are the two metrics where we only tie
crisprHAL 2, and the two a designer actually uses, so this was aimed at the known weakness. A regression+classifier
blend is included because a classifier can be better at the top while worse overall.

| CONFIRM genes, 3 seeds | rho | top-decile precision | pick percentile of top 10 |
|---|---:|---:|---:|
| tuned regression | **0.7215** | **0.2912** | 85.10 |
| classifier, top 10% | 0.6135 | 0.2509 | 79.97 |
| classifier, top 20% | 0.6679 | 0.2744 | 83.47 |
| classifier, top 30% | 0.6913 | 0.2867 | **86.18** |
| regression + classifier, top 10% | 0.7019 | 0.2651 | 81.73 |
| regression + classifier, top 20% | 0.7099 | 0.2785 | 84.43 |
| regression + classifier, top 30% | 0.7156 | **0.2938** | 83.32 |

**Null.** Every classifier arm is worse on rho, by 0.03 to 0.11, and the best top-end numbers (+0.0026 precision for the
top-30% blend, +1.1 percentile points for the top-30% classifier) are inside the seed-to-seed spread. The coarser the
threshold the less is lost, which is the signature of information thrown away by discretising rather than of a better
objective. **So four separate ways of targeting the top end -- activity weighting, pairwise ranking, NDCG and now
binarised classification -- all fail**, which makes a stronger statement than any one of them: the top end is hard here
because the label is noisiest where activity is highest, not because squared error is the wrong objective. Worth noting
that Orro et al.'s own task *is* classification (their label is a binary fitness call), so this is not a criticism of
their design -- it is a finding about ours.

## 19. The field's performance ceiling is mis-derived, and correcting it triples the apparent headroom (20 Nov)

This is the most consequential thing the literature pass produced, and it is a contribution rather than an import.

**What the field uses.** Konstantakos et al. (2022, *Nucleic Acids Research* 50:3616-3637), a widely-cited review that
sets up the benchmarking frame for this field, writes: *"The upper bound is based on a previous study (8), which showed
that the measured efficiencies are not perfectly correlated even for data from the same assay and cell line.
Specifically, they showed that for datasets where replicates are available, the Spearman correlation is in the range of
0.71-0.77. This illustrates the quality of the data and suggests that a correlation of ~0.7 constitutes an upper limit
of any prediction."* Reference (8) is Haeussler et al. (2016). They then set 0.4 as the threshold for calling a tool
successful. Verified in the full text, not from a snippet.

**Why that is wrong, by two steps, both in the same direction.** A replicate-to-replicate correlation *r* is a
reliability, not a ceiling. The largest correlation any predictor can have with a noisy observed target is the square
root of that target's reliability -- the standard attenuation relation -- and when the target is an *average* of
replicates, as essentially every released activity score is, its reliability is raised by Spearman-Brown to
`k*r/(1+(k-1)*r)`. Taking *r* itself as the ceiling drops the square root and ignores the averaging.

| replicate agreement *r* | ceiling vs a single replicate, sqrt(r) | reliability of a 2-replicate mean | **ceiling vs that mean** |
|---:|---:|---:|---:|
| 0.71 | 0.843 | 0.830 | **0.911** |
| 0.77 | 0.877 | 0.870 | **0.933** |
| 0.8097 *(ours, measured)* | 0.900 | 0.895 | **0.946** |

On the review's own numbers the attainable ceiling is **0.91-0.93, not ~0.7** -- an understatement of **0.21-0.23**, and
**+0.14-0.18 from the missing square root alone**, before any averaging correction. Our Finding 6 is the worked example
of the corrected route: replicate agreement 0.8097 on crisprHAL's own per-replicate counts, Spearman-Brown 0.8948,
square root **0.946**.

**What it changes, and this is the point.** Under the field's ceiling a bacterial model at rho 0.695-0.782 or a human one
at 0.6-0.8 looks saturated, and the rational conclusion is that the remaining error is irreducible noise. Under the
corrected ceiling:

* models at rho ~0.70 sit at **75-77%** of what is attainable;
* our own tuned model at 0.7314 sits at **77.3%** of 0.946, with **0.215 rho** of real, repeatable signal unexplained;
* so *everyone*, us included, is at about three-quarters of the ceiling while the frame says they are at the top of it.

That reframes this project's "0.21 rho unexplained" from a modest caveat into a claim about the field: there is roughly
three times more headroom than the standard benchmarking frame implies, and the reason nobody is looking for it is a
missing square root. It also means the 0.4 success threshold is far too lenient. **This is a self-contained result that
needs no new data and would stand as a short methodological comment on its own.**

*What would make it stronger, and is not done:* recomputing corrected ceilings for the specific benchmark datasets the
review tabulates, which needs their per-replicate counts rather than the published aggregate scores. We did it for one
label (crisprHAL's curated score) because their replicate counts are released; most are not.

## 20. Two of our own questions, not imported ones (20 Nov; `novel_followups.py`, `fold_distance.py`)

### 20.1 Species-relative composition: the report's "cheapest portability lever" is a null, and the null is mechanistic

Part 3's limits table named this as *"the single cheapest thing that could make the representation species-portable"*:
family A's strongest column is a **raw** GC fraction, K-12 is 50.79% GC and *C. rodentium* 54.72% (both reproduced here
from the fastas), so a flank at 52% GC is above background in one genome and below it in the other -- the feature
carries the opposite sign of local enrichment across the two. Expressing each window as a percentile of its **own**
genome's distribution of that statistic should fix the sign.

A design note that decided the experiment: **within one genome this cannot change anything**, because a percentile map
of a single column is monotone and boosted trees are invariant to monotone transforms of individual features. So the
home cohort is only a check that it is exactly a no-op, and the real test is transfer.

| 16 raw composition columns, tuned configuration | *C. rodentium* rho (25,210 guides) | home gene-grouped CV |
|---|---:|---:|
| absolute, as shipped | **0.7142** | 0.7342 |
| species-relative percentile | 0.7105 | 0.7340 |
| difference | **-0.0037** | -0.0003 |

**Null, and slightly the wrong way.** The home difference is -0.0003, which confirms the monotone-invariance prediction
and so validates the setup. On transfer the absolute fraction is **better** by 0.0037.

*The reading, which is worth more than the portability gain would have been.* If the flank effect were about being
GC-rich *for this genome* -- local enrichment against background -- the percentile should have won. It loses, so the
effect tracks **absolute base composition**, not composition relative to genomic context. That points at a physical
property of the DNA itself (duplex stability, bendability, protein affinity at a given composition) rather than at
anything the cell reads relative to its own background, and it is consistent with Finding 8's result that shape
descriptors are a reparameterisation of composition. The limits-table item is retired: absolute is the right
parameterisation, and the 9-12% lost on cross-organism transfer is not recoverable this way.

### 20.2 The train/test sequence-distance statistic our comparators report, and ours

Browne et al. check Hamming distance between training and held-out guides; Yu et al. report minimum and median (7 and
21). We checked exact duplicates and grouped folds, but never published a distance. Adopting it (`fold_distance.py`):
for every validation guide, the minimum Hamming distance to any training guide.

| cohort / fold scheme | min | 1st pct | median | fraction with an identical training guide |
|---|---:|---:|---:|---:|
| curated *E. coli* / gene-grouped | **3** | 4 | 6 | **0.000** |
| curated *E. coli* / random rows | 3 | 4 | 6 | 0.000 |
| human / twin-grouped | **3** | 4 | 6 | **0.000** |
| human / random rows | **0** | 0 | 6 | **0.391** |

**Two things, and the second is the better one.** First, the bacterial cohort is clean: no validation guide has a
training guide closer than 3 substitutions, under either scheme, so the report's claim that grouping covers
near-duplicate leakage is now shown rather than asserted. (The gene and 100 kb grouping still matter, but for the
*positional* features, which is a different axis of leakage from sequence similarity.)

Second, **the statistic recovers section 16.2's leak from an entirely different measurement, and the two agree
numerically.** Under row-wise human folds 39.1% of validation rows have an *identical* guide in training; under
twin-grouped folds, none. And 39.1% is what the twin count predicts independently: 49% of rows have a twin somewhere,
and 5-fold CV puts 80% of the matrix in training, so 0.49 x 0.8 = 39.2% of validation rows should find their twin
across the split. A performance collapse (0.5076 -> 0.2515) and a distance statistic computed from sequence alone,
arriving at the same number from opposite directions.

## 21. Biology out of the model, by routes redundancy does not disqualify (20 Nov; `mechanism.py`)

*Why this was overdue.* The user's criticism was fair: most of what this project contributes beyond the model is a
warning. Finding 5 is partly to blame -- it says that on a matrix where 99% of columns regenerate from the 20-mer,
column-level importance is a statement about the selector -- so the project had been cautious about reading mechanism
off the model at all. The way out is to use routes where there is no model to be confounded: the label against the
sequence directly, and partial associations.

### 21.1 The flank effect is GC, it is symmetric, and it peaks at 250-500 nt

Partial Spearman of each flank window's composition with the label, holding **target** GC fixed (33,567 guides, no
model involved; `mechanism_distance_profile.csv`). Sign convention checked twice against the report: flank GC is
positively associated with activity and target GC negatively, and both reproduce here (+0.14 to +0.16 marginal for
flank GC, -0.147 for target GC), so a positive number means more cutting.

| statistic | side | 50 nt | 250 nt | 500 nt | 1,000 nt |
|---|---|---:|---:|---:|---:|
| GC | upstream | +0.172 | **+0.203** | +0.196 | +0.173 |
| GC | downstream | +0.146 | **+0.197** | +0.194 | +0.181 |
| purine | upstream | -0.000 | +0.001 | -0.005 | -0.005 |
| purine | downstream | -0.007 | -0.018 | -0.014 | -0.000 |

**Three statements, each a constraint on mechanism rather than a feature importance.**

1. **It is GC, not base composition in general.** Purine fraction is flat at every distance and on both sides, within
   +-0.018 of zero, while GC runs +0.15 to +0.20. So this is not a generic composition or strand-chemistry effect. It
   rules out anything keyed to purine/pyrimidine asymmetry and points at GC-specific physics -- duplex stability,
   stacking, melting -- which is the same direction as 20.1's finding that the effect tracks *absolute* composition
   rather than enrichment against genomic background, and as Finding 8's result that shape descriptors are a
   reparameterisation of composition.
2. **It is symmetric.** Upstream and downstream profiles agree to within 0.026 at every window. A directional process
   -- replication fork direction, transcription direction, one-sided resection -- would not produce that. Worth putting
   beside the orientation result (Jacky's, delta 0.051 against a curated TSS): orientation matters, but it cannot be
   acting through flank *composition*, because composition is side-blind. Those are two separate effects.
3. **There is a length scale, and it is 250-500 nt.** The association rises from 50 nt to 250 nt and falls again by
   1,000 nt, on both sides. That is far longer than anything Cas9 contacts (~20-30 nt) and far shorter than nucleoid
   macrodomains (hundreds of kb). It is one to three persistence lengths of duplex DNA (~150 bp). *Hypothesis, not a
   result:* the effect tracks the local mechanical or topological environment over a few persistence lengths. It is
   stated as a hypothesis because the measurement is a length scale and nothing here tests the physics.

### 21.2 Position x base design rules, and one of them inverts between kingdoms

Effect of each base at each guide position on the label, in SD units, computed straight from the 33,567 protospacers
and their labels -- no model, no selector (`mechanism_position_base.csv`; 95% intervals from the normal approximation
to the SE of a mean over thousands of guides). Position 1 is PAM-distal, 20 is PAM-proximal.

| | A | C | G | T |
|---|---:|---:|---:|---:|
| positions 1-7 | all within +-0.04 | | | |
| 14 | -0.008 | **-0.147** | +0.012 | **+0.141** |
| 15 | -0.033 | -0.133 | +0.103 | +0.088 |
| 17 | -0.022 | +0.125 | -0.086 | -0.015 |
| 18 | +0.075 | -0.060 | **-0.266** | **+0.210** |
| 19 | -0.115 | -0.161 | +0.138 | +0.152 |
| 20 | +0.193 | -0.142 | **-0.322** | **+0.348** |

**The seed region carries all of it.** Positions 1-7 are flat -- every base within 0.04 SD of zero -- and the strongest
twelve effects are all at positions 14-20. That agrees with two independent measurements: our own gain-by-position
profile (eight PAM-proximal positions carry 63% / 62% of split gain, section 14) and Orro et al.'s SHAP profile on a
human base-editing screen, which peaks at positions 19 and 20. Three routes, two organisms, two assays, one answer
about where along the guide the signal lives.

**It is mostly the GC rule, but not entirely.** Positions 18 and 20 follow it -- G and C depressed, A and T raised --
which is the target-GC effect resolved by position. Position 19 breaks it: G is *favoured* there (+0.138) while C is
disfavoured (-0.161). So there is position-specific base chemistry on top of composition.

**And the largest single effect appears to invert between kingdoms.** G at position 20, immediately 5' of the PAM, is
the most disfavoured base-position combination we measure, at **-0.322 SD** [-0.343, -0.300]. The eukaryotic literature
consistently reports the opposite: G at position 20 is a *favoured* base and one of the main drivers of the Doench
score. If that holds, it joins the GC inversion already in the report (target GC against activity: -0.20 here, +0.02 in
human) to make a coherent statement -- **the guide-intrinsic design rules invert between *E. coli* and human** -- which
would also explain, mechanistically rather than by appeal to "different organism", the field's repeated observation
that mammalian-trained predictors transfer poorly to bacteria.

*Status: our measurement is firm, the comparison is not yet.* The human side comes from reviews and a third-party
calculator, not from primary weights, and positional numbering conventions differ between papers (some number from the
PAM, some from the 5' end), which is exactly the kind of detail that can flip a sign. Confirming it needs Doench et
al.'s 2016 supplementary weights read directly. **Do not put the inversion in a draft until that is done**; the
*E. coli* matrix itself can go in now.

### 21.3 The design rules do not transfer between kingdoms, and the strongest one inverts

The right way to settle 21.2's open question was not to read Doench et al.'s fitted weights -- those come from a
regularised regression over many correlated features and are not per-position marginals, so comparing them with our
statistic would be apples to oranges. Instead: **run our estimator on their data.** Same code, same units, both
organisms. The human matrix is deduplicated first (17,421 rows decode to 13,138 distinct 20-mers, and a repeat is a
repeat *measurement*, so labels are averaged within a guide), which also sidesteps section 16.2's problem.

*Sign convention checked before reading anything*, because it is the one thing that could manufacture this result.
Target GC against the label: **-0.147 in *E. coli*** (report: -0.20) and **+0.018 in human** (report: +0.02). Both
reproduce the report's independently established signs, so in both cohorts a higher label means more activity.

| position 20 (PAM-proximal) | *E. coli* | human |
|---|---:|---:|
| G | **-0.322** | **+0.131** |
| T | **+0.348** | **-0.170** |
| A | +0.193 | +0.060 |
| C | -0.142 | -0.121 |

| | value |
|---|---|
| rank correlation of the two full position x base matrices (80 cells) | **-0.052** |
| cells clear of zero in *both* organisms that have **opposite** signs | **15 of 27** |

**Result: the position-specific base preferences of the guide are essentially unrelated between the two kingdoms**
(rho = -0.05, i.e. no relationship rather than an attenuated one), **and the strongest preference in each inverts.** G
immediately 5' of the PAM is the most *disfavoured* base-position combination in *E. coli* and a *favoured* one in
human -- which is what the eukaryotic literature reports independently, where G at position 20 is described as a main
driver of the Doench score. T inverts with it.

This is the finding 21.2 flagged and would not claim, now settled by matched measurement rather than by cross-reading
someone else's weights. Three things follow:

1. **It is a positive biological result, not a warning.** Together with the GC inversion already in the report it says
   the *guide-intrinsic* design rules are kingdom-specific, while the *flanking-context* effect is the half that
   crosses (Part 3).
2. **It explains an observation the bacterial literature keeps making without an explanation** -- that
   mammalian-trained predictors predict poorly in bacteria. The reason is not "different organism" in the abstract: the
   position weights they encode point the wrong way.
3. **It is a design rule a bench scientist can use today.** Avoid G and prefer T immediately 5' of the PAM in
   *E. coli*; the opposite of what a human-derived tool will tell you.

*Limits.* Two different assays (survival after cutting against an indel/knockout score), so the comparison is of
design rules rather than of mechanisms, and nothing here says *why* they invert. The human matrix is a single cohort.
The 15-of-27 figure is about sign agreement among jointly-significant cells, not a formal test of the matrices'
independence.

### 21.4 The flank signal has no structure finer than about 250 nt

Browne et al. searched input *length* and found the gains level off. The orthogonal question is *resolution*: how
finely does the flank have to be described? If Finding 2 is right that the effect is a smooth gradient with no motif
and no dependence on window edges, there should be a resolution floor. Equal-width bins of the 1,000 nt each side, GC
and purine per bin, gene-grouped 5 folds x 3 seeds, tuned configuration (`novel_resolution.csv`):

| description | columns added | rho | share of the composition signal |
|---|---:|---:|---:|
| base only | 0 | 0.5617 | |
| one bin per side (1,000 nt) | 4 | 0.6462 | **75.7%** |
| two bins (500 nt) | 8 | 0.6616 | 89.6% |
| four bins (250 nt) | 16 | 0.6679 | **95.2%** |
| eight bins (125 nt) | 32 | 0.6701 | 97.2% |
| twenty bins (50 nt) | 80 | 0.6718 | 98.7% |
| fifty bins (20 nt) | 200 | 0.6687 | 96.0% -- *it turns over* |
| family A's four nested windows | 24 | **0.6732** | 100% (reference) |
| all of family A | 348 | 0.7314 | |

**There is a floor and it is coarse.** Four numbers -- GC and purine over the whole kilobase on each side -- carry
three quarters of the composition signal. Sixteen carry 95%. Doubling the resolution past 250 nt bins buys +0.0022 and
redoubling buys +0.0017. And at 20 nt bins it **turns over** -- 200 columns score below 80 and below 32 -- because fine bins are noisier
individually and crowd the base columns out of a fixed selector budget. So **the flank effect has essentially no
spatial structure below ~250 nt, and describing it more finely actively costs**, which is the same
length scale 21.1's distance profile peaks at, reached by a completely different construction. (Family A's 24 nested
columns beat 80 equal bins, which is what one would expect: nested windows at four scales are a better basis for a
signal with one characteristic scale than fine bins at a single scale.)

Two consequences. *Mechanistically*, a process that produced fine structure -- a motif, a positioned protein, a
periodic feature -- would not give this curve; a bulk property averaged over a few hundred bases would. *Practically*,
the flank featuriser could be about four columns per side rather than 174, which is most of the remaining end-to-end
cost (section 13) and would make the representation trivially portable.

### 21.5 Which pairs carry the interaction: the flank meets the seed

Section 14 found 37% of what trees add over a linear model is pairwise but never said *which* pairs. A depth-2 model
can express exactly pairs, so its (root, child) splits **are** the interactions, with none of SHAP's additivity
assumptions. Gain share by block pair and by specific column pair (`mechanism_pairs_*.csv`):

| block pair | share of interaction gain |
|---|---:|
| flank window composition x flank window quantum | 0.058 |
| **protospacer indicators, positions 16-20 (seed) x protospacer dimer quantum** | 0.048 |
| **flank window composition x protospacer basepair quantum** | 0.048 |
| flank window quantum x protospacer basepair quantum | 0.047 |
| flank window composition x protospacer dimer quantum | 0.043 |
| seed indicators x seed indicators | 0.039 |

| strongest specific pairs | share |
|---|---:|
| `eng.flank.comp.up250.gc` x `p20basepair.Hlgap.eVEraw` | 0.0186 |
| `p18dimer.Hbond.energyraw` x `p19dimer.Hbond.stackingraw` | 0.0163 |
| `eng.flank.win.dn500.trimer.Hbondenergy.mean` x `p20basepair.Hlgap.eVEraw` | 0.0159 |
| `eng.flank.comp.up250.gc` x `eng.flank.win.dn500.trimer.Hbondenergy.mean` | 0.0137 |

**The dominant interaction is between the flanking kilobase and the seed region**, and the single strongest pair is
250 nt upstream GC against a property of **position 20** -- the same window that 21.1 finds the distance profile peaks
at and the same position that 21.2 finds carries the strongest base preference and the cross-kingdom inversion. Three
analyses built on different principles (partial correlation, marginal base effects, depth-2 split gain) converge on
the same two objects. So the flank effect and the seed effect are **not additive**: how much the surrounding
composition matters depends on what the PAM-proximal bases are.

*And the interaction is diffuse.* No block pair exceeds 5.8% of interaction gain and no specific pair exceeds 1.9%, so
this is not a handful of special pairs but a broad coupling between two regions -- which is what Finding 2's "smooth
gradient, no motif" predicts, and the reason the pairwise term is 37% of a small total rather than a discrete rule.

### 21.6 The human flank gain, re-run honestly: it survives at half the size

Section 16.2 left three statements resting on row-wise human folds. The additivity ratio was withdrawn there. This is
the second: the **+0.052** (measured here as +0.048) family-A increment on the human matrix, which Part 2 uses for the
claim that flanking context is the half of the representation that crosses kingdoms. Re-run with folds that keep
identical protospacers together (`human_shape --stage 7 --twin`, 3 seeds x 5 folds, 16,490 located guides):

| arm | row-wise folds | twin-grouped folds |
|---|---:|---:|
| `h_base` | 0.3807 | 0.3063 |
| `h_base+A` | 0.4287 | **0.3289** |
| `h_base+A+mc` (shape on top) | 0.4470 | 0.3466 |
| **family A increment** | **+0.0480** | **+0.0226** (sd 0.003 over 15 fits) |

**The claim survives at half the size.** Family A is still worth +0.0226 on the human matrix with honest folds, clear
of zero, so "flanking context is the part of the representation that crosses kingdoms" stands -- it is simply a +0.023
effect rather than a +0.05 one. Both levels drop under grouped folds (0.38 -> 0.31 for the base), which is the leak
being removed from the baseline as well as from the arm, and the *increment* is what the claim rests on.

*An observation not developed here.* Shape on top of family A adds **+0.0177** in human under twin-grouped folds and
+0.0183 under row-wise ones -- stable across fold schemes, so not a leak artefact -- against essentially zero in
*E. coli* (Finding 8). That is a kingdom difference in whether DNA-shape descriptors are a second channel or a
reparameterisation, and it is the kind of thing Finding 8's limits section anticipated. The human family A is a
teammate's hg38 build rather than ours, so the two are not strictly comparable, and the human arm is otherwise stopped;
flagged for whoever picks it up.

## 22. The per-locus precompute, and a sweep for loose ends (20 Nov; the user asked for the precompute, then for the whole project to be checked for open questions and each closed rigorously)

### 22.1 Family A from a locus instead of from a guide (`flank_locus.py`)

Section 13 made the featuriser fast; it did not make it *cheap*. `a_flank_fast` still gives every guide its own two
1,000-nt flanks and re-indexes four k-mer series over each of them, so at 0.068 ms per guide it was **three quarters of
the 0.093 ms end-to-end bill**. That is work paid per guide for information that is shared: *E. coli* carries an
NGG-flanked site about every 8.6 nt, so under the old path each base of the chromosome is scanned roughly 150 times.

**What replaces it.** Scan a stretch of chromosome once, store cumulative tracks along it, and answer every guide inside
that stretch by differencing them:

| column group | how a window is answered |
|---|---|
| `comp.*.gc`, `comp.*.purine` | cumulative base counts; one subtraction (purine is complemented for a minus-strand guide, GC is not) |
| `win.*.mean`, `win.*.sd` | cumulative sum and sum of squares of the tensor series, centred on the lookup table's own mean so the variance does not cancel; two subtractions |
| `comp.*.maxrun` | the one column that is **not** a sum of anything -- the longest homopolymer run inside a window. Still O(1): list the maximal runs of the stretch once, and the answer is the better of the two truncated boundary runs and the longest run strictly between them, which a sparse table over run lengths gives in two lookups. A window of *w* nt spans at most *w* runs, so the table needs `ceil(log2(w)) + 1` levels |
| one-hot, per-position tensors | direct gathers at the 26 labelled offsets around the cut, complemented and reversed for a minus-strand guide |

Nothing reads a stored flanking string: a guide is a plus-strand coordinate and a strand, which is what a design tool
actually has. `base_fast.matrix_fast` gained a `codes=` argument for the same reason -- a design run holds its 20-mers as
genome array slices, and turning them into Python strings only to turn them back into the same codes is pure overhead.

**Verified against the path it replaces**, on all 33,567 curated guides: same column names in the same order, same
missing-value pattern, **largest relative difference 4.6e-12** (floating-point reassociation, nothing else), zero columns
above 1e-9. Two checks come free with it and both pass: the stored flanking strings in the guide index agree with the
genome exactly, and the protospacers regathered from the chromosome by coordinate match the screen's own to the base.

**Cost, at the density a design tool meets** (every NGG site in a stretch, one per ~8.6 nt):

| stretch | sites | `a_flank_fast` | `flank_locus` | of which precompute | speed-up |
|---|---:|---:|---:|---:|---:|
| 1 kb (about one gene) | 123 | 0.0614 | 0.0113 | 0.0051 | 5.4x |
| 10 kb | 1,153 | 0.0604 | 0.0044 | 0.0009 | 13.6x |
| 100 kb | 11,671 | 0.0632 | **0.0039** | 0.0005 | **16.1x** |
| 1 Mb | 113,201 | 0.1383 | 0.0042 | 0.0005 | 33.1x |

(ms per guide, one thread.) End to end -- featurise, select, impute, predict -- on the same stretches:

| ms per guide, one thread | 1 kb | 100 kb | 1 Mb |
|---|---:|---:|---:|
| original champion | 0.0234 | **0.0223** | 0.0300 |
| **tuned XGBoost (d099x)** | 0.0392 | **0.0361** | 0.0477 |
| average of 5 XGBoost | 0.0890 | 0.0835 | 0.1003 |

**The tuned model goes from 0.093 to 0.036 ms per guide, and from 13x to 34x faster than crisprHAL 2** (54x for the
champion). Three honest qualifications, because each is a real limit of the measurement:

1. *The precompute is paid per nucleotide, not per guide*, so the gain depends on guide density. At the **curated
   cohort's** density -- one guide every 138 nt, because that screen samples genes rather than tiling -- the same code
   gives only 3-5x. The 16x is a statement about designing guides, not about re-scoring a screen.
2. *At 1 Mb the bottleneck moves.* 113,201 guides x 6,517 float32 columns is a 2.95 GB feature matrix, and per-guide cost
   rises again (0.0039 -> 0.0042 for the flank block, 0.036 -> 0.048 end to end). A design run should chunk at tens of
   kilobases; nothing is gained by going wider.
3. *No number in the report moves.* The columns are identical to 4.6e-12, so accuracy, every ablation and every finding
   are untouched. This is an engineering result and is reported as one.

**It also retires an open item.** Section 21.4 ended with "the flank featuriser could be about four columns per side
rather than 174, which is most of the remaining end-to-end cost". The cost half of that argument is now gone: the flank
block is 0.0039 of a 0.0361 ms bill, so the four-column version would save about 0.003 ms per guide -- and 21.4's own
ladder prices it at **0.6462 against 0.7314 rho**, a loss of 0.085. It was only ever attractive because featurisation
was expensive. It is not, so it should not be done.

### 22.2 The ceiling correction, parametrically, so it no longer needs anyone's replicate counts (`ceiling_table.py`)

Section 19 corrected the field's ceiling and had to end *"Not done: corrected ceilings for the specific datasets the
review tabulates, which needs their per-replicate counts. Most are not released."* That is the wrong shape of gap to
leave open, because the correction needs only two numbers a dataset's own paper states -- the replicate agreement *r*
and how many replicates *k* the released label averages -- and both steps are arithmetic. So it is tabulated over the
whole plausible range of both (`ceiling_parametric.csv`), which lets anyone apply it to any row of that review, or to a
dataset published tomorrow, without re-deriving it. The three rows the report already quotes are reproduced exactly
(0.9113, 0.9328, 0.9460).

The useful consequence is that **the conclusion does not depend on pinning down anyone's *r* and *k* at all**:

| | *r* = 0.50 | 0.60 | 0.70 | 0.71 | 0.77 | 0.80 | 0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ceiling, *k* = 1 (no averaging at all) | 0.707 | 0.775 | 0.837 | 0.843 | 0.877 | 0.894 | 0.949 |
| ceiling, *k* = 2 | 0.817 | 0.866 | 0.908 | 0.911 | 0.933 | 0.943 | 0.973 |
| where rho 0.70 sits, *k* = 2 | 86% | 81% | 77% | 77% | 75% | 74% | 72% |

Read along the review's **own** replicate range, *r* = 0.71-0.77: a model at rho 0.70 is at **80-83%** of attainable
even under the most conservative possible assumption, that the released label is a *single* measurement with no
averaging at all, and at **75-77%** if it averages two. It is never at the ceiling, for any *k*. So the correction does
not depend on recovering anyone's replicate count -- the only way "~0.7 is an upper limit" comes out right is if
replicate agreement were near 0.5 *and* labels were unaveraged, and the review itself states that agreement is 0.71-0.77.

*What the table is not.* Exact for Pearson correlations with independent additive replicate noise; Spearman's rho is a
rank correlation, so these are close approximations. And both steps assume replicate noise is independent -- shared batch
effects inflate *r* and so inflate the ceiling, a real systematic difference between replicates deflates it. That is
an assumption here as in the field's own derivation, and nothing in this section tests it. (An earlier attempt
compared the WT and eSpCas9 screens and found 30% of their disagreement predictable from sequence; since those are two
different enzymes that is expected biology, not evidence about replicate independence -- see REMOVED.md. Finding 6 avoids
the question by using crisprHAL's released per-replicate counts.)

### 22.3 An upper bound on every local shape predictor, including the one we could not run (`local_sequence_bound.py`)

Finding 8 tested tabulated DNA shape -- DNAshapeR's pentamer table, 5 bp of context -- and found nothing on top of family
A. **Deep DNAshape**, which reads up to **7 bp** of context, was cloned and never run: the sandbox refuses to execute
third-party model code, and running a downloaded model is not prudent in any case. So the finding has had to carry the
caveat *"the wider context is what was not tested"*, which leaves it open to the obvious objection that the next shape
predictor along might work.

That objection can be closed without running anyone's model, because of what a shape predictor **is**. Its output at a
position is a deterministic function of the handful of nucleotides around it. Over the inner flank, therefore, its entire
output is a function of the local sequence -- so hand the model that sequence explicitly and see what it buys. Family A
already one-hot encodes u10..u1, the PAM's N and d1..d15; this adds the 75 positions it does not reach, out to +-50 nt
(u50..u11 and d16..d50; 300 columns), which makes every 7 bp-context feature from u47 to d47 a function of columns the
model now holds. Protocol is the project's standard and identical to 16.1's, so the numbers line up with it: gene-grouped
5 folds x 3 seeds on the 30,203 guides with a gene label, top-900 ranking fitted inside each training fold, and a
**shuffled control** carrying the same 300 columns permuted across guides, because 300 new candidates could dilute a
fixed selector budget whatever they contain.

| configuration | arm | rho | difference from base+A | 95% CI | folds up | local columns picked, of the budget |
|---|---|---:|---:|---|---:|---:|
| **tuned (d099x)** | base+A | **0.7314** | -- | | | 0 of 450 |
| | + the inner +-50 nt | 0.7316 | **+0.0002** | [-0.0004, +0.0008] | 8/15 | 8.7 |
| | + the same, shuffled | 0.7309 | -0.0004 | [-0.0012, +0.0003] | 6/15 | 5.3 |
| champion | base+A | 0.7151 | -- | | | 0 of 300 |
| | + the inner +-50 nt | 0.7144 | -0.0008 | [-0.0015, -0.0001] | 6/15 | 2.1 |
| | + the same, shuffled | 0.7152 | +0.0001 | [-0.0006, +0.0008] | 8/15 | 0.9 |

**Nothing is there.** At the capacity that *can* use flank columns -- Finding 4's point, and the reason the bound is
measured at the tuned settings rather than only the champion -- the explicit local sequence is worth **+0.0002 rho, with
an upper confidence bound of +0.0008**. It is indistinguishable from the same columns shuffled; the selector does not
reject it, it picks 8.7 of them out of 450 and picks 5.3 of the noise ones, which is the clearest form of the result: the
ranking cannot tell the real local sequence from noise. At the champion's smaller budget the real block is very slightly
negative and the shuffle is not, which is dilution noise at a 300-column budget rather than anything about the columns.

**What this bounds, stated exactly, because it would be easy to overclaim.** Two measurements sit side by side: the
pipeline gains at most +0.0008 rho from holding the inner +-50 nt explicitly, and Finding 8 separately measured the
5 bp-context *reparameterisation* of exactly that sequence and got nothing. What remains logically possible is that a
7 bp-context predictor is a better-*shaped* feature for the same information -- a learnability claim, not an information
one. That is the smallest residue this question reduces to without executing the model, and the prize is capped at
+0.0008 rho either way. The report's Finding 8 limits section now says that instead of "not tested".

*Two checks that came free.* `base+A` at the tuned configuration gives **0.7314**, reproducing the report's headline rho
to four decimals through a separately written cohort builder; and the register's row 3 is narrowed to what it should have
been all along -- not "shape", but specifically a *measured*, non-sequence-derived quantity such as a stiffness matrix,
which no local predictor supplies.

### 22.4 Two modules did their work at import, and one of them overwrote its own result file (`import_check.py`)

Found by accident, while writing a script that only meant to check that every module in the package loads. Importing
`sgrna.post_freeze.speed.equiv_check` refitted two models and **rewrote `featuriser_equivalence.csv`** with timings measured on
a machine that happened to be running an experiment, which then made the audit fail on two numbers that had nothing wrong
with them. Importing `sgrna.post_freeze.paper.verify_numbers` ran the whole audit and raised `SystemExit` **in the importing
process**, which is why the sweep never finished and the cause took a moment to find. `strand/strand_recoverable.py` has
the same shape and writes a results file at import too.

All three were written as scripts with their body at the top level, which is convenient to run and quietly unsafe to
import. Their bodies now live in `main()` behind a `__main__` guard; the audit still reports the identical 806 checks and
0 mismatches afterwards, which is the evidence that the move changed nothing.

**The guard against it coming back** is `import_check.py`, which asserts two things and is run by the audit:

* every module under `sgrna/` imports without raising -- 150 modules when this was written (146 after the 22 Nov
  reorganisations, which removed modules, added package markers, a reference checker and a map generator, and merged `screens` into `compare`), 2 excluded by name with the reason recorded
  (`vs_crisprhal_wt_hal` reads `sys.argv[1]` at the top level and needs the TensorFlow venv; the second was `controls/numcheck`, a
  document-coverage script pinned to the shared repo's absolute paths, removed on 22 Nov -- see REMOVED.md). This is also the only check in the project that
  the code *loads*, as distinct from the audit's check that the numbers match their files;
* the sweep leaves `results/` byte-identical -- all 675 files hashed before and after, 0 changed.

The sweep runs in a subprocess, because the entire point is that an unsafe module may exit or write and neither should be
able to reach the caller.

*Worth saying plainly:* this is a mistake of mine that cost two wrong audit lines and a confusing half hour, and the only
reason it did not quietly corrupt a reported number is that the two rows it overwrote were timings the audit checks.

### 22.5 The sweep itself: what was found, and where the answer now lives

The user asked for the whole project to be checked for loose ends and open questions, and for each found to be closed
rigorously. What the sweep turned up, in the order it was found:

| found | kind | what was done |
|---|---|---|
| Three places said the mixed-effect formulation was **untried**; 18.1 had run it the day before | stale claim | all three rewritten with the result; the audit now fails if any of the four stale phrases returns |
| The audit's check count was quoted as **682, 784, 799 and 682** in four documents | drift | all four set to the real count, and the audit now checks its own quoted count in all four, so they cannot separate again |
| `PAPER_FLOW.md` planned four figures and listed seven built ones in a second table | internal contradiction | one table, with the section each figure belongs to |
| `fig7_pipeline` had "799 checks" drawn into it | a volatile number inside a vector figure | replaced with "every number recomputed". A figure should not need regenerating when a check is added; `figures.py --fig 7` also no longer truncates the manifest to one row |
| `PROGRESS_REPORT.md` cited a file called final_eval, which does not exist | dangling reference | the file is `final_eval_espcas9.csv`; the numbers it supports (0.7842 against 0.7821) check out |
| Importing the package ran the audit, exited the process, and overwrote a result file | hazard, and it had already caused two wrong audit lines | 22.4 |
| "Not done: corrected ceilings for the review's datasets" | an open item that did not need to be | 22.2 |
| "Deep DNAshape: not run" | an open item that could be bounded instead | 22.3 |
| "the flank featuriser could be four columns per side" | an open item whose premise had changed | 22.1 |
| Open questions scattered across two documents in four phrasings | no single list, so staleness was invisible | **the open-questions register**, report Part 4. Nine genuinely open, six closed with where |

**Ten questions remain open and the register says why each one is** (nine when this section was first written; the tenth, on 22 Nov, is in section 23). None of them is open through oversight: four need
data this project does not have (a retuned crisprHAL 2, a processed occupancy track, a measured stiffness matrix, an
untouched cutting screen), two need an experiment no pooled screen can provide, one was stopped by the user, one needs
per-guide read counts, and one needs a document I cannot read. That is the honest end state, and it is now in one place
where the next person can see it without reading 5,700 lines.

## 23. Consolidation: what the code is, what was removed, and what the sweep found in the evidence (22 Nov)

The user asked for the code to be explained, overlapping or outdated modules combined, results nobody agreed were worth keeping
removed (with a file listing them), and for the repository to be made readable. Where each part is recorded:
`CODE_MAP.md` (what everything is, generated from the code), `REMOVED.md` (what went and what was brought in), the report's Part 4
register (what is still open).

### 23.1 The finding about the size: it is not padded

The test applied to every module and result file: kept if the report names it, the audit reads it, a kept module uses it (including
through file names built at run time), or a teammate's document cites it. Only three modules and one directory of tracked debris
failed every test. Across the package only one 8-line function is duplicated. The code is roughly a hundred one-question scripts,
each the provenance of a sentence in the report; the lever for a smaller repository is the report's scope, not tidying. The
by-section table in `CODE_MAP.md` is the price list.

### 23.2 What was done

* **Removed** `catboost_info/` (tracked training-log debris), `interpret/decision_scale.py` (never run), `cohorts/ceiling2.py`
  (superseded by Finding 6; its result -- 30% of the WT-vs-eSp disagreement is predictable -- is recorded in `REMOVED.md`),
  `controls/numcheck.py` (superseded by the audit), and two superseded Colab drafts folded into `crisprhal_wide_window.py`.
* **Reorganised** `post_freeze/`'s 52 flat modules into eight themed subpackages (`tuning`, `vs_crisprhal`, `speed`, `shape`,
  `nuclease`, `human`, `signal`, `paper`) by `git mv`, so history follows. The themes were not imposed: they follow the clusters
  of the import graph (the shape modules import each other, the nuclease modules import each other, and so on). Verified by import of all 157 modules, by the audit, and by re-running `clean_vs_hal` (three model
  fits through the `tuning -> vs_crisprhal` chain) into a scratch copy of the results: every cell identical.
* **Did not merge modules line by line.** There is nothing to deduplicate, every module has its own `main`/`run`, and merging them would
  make the one-module-one-claim provenance harder to follow. The one real merge was the Colab script.

### 23.3 What the sweep found, in the order it found it

| found | what it was | what was done |
|---|---|---|
| the report's **R2 0.963** (excess twist from dinucleotides) had no producer | a number from a session whose code was not kept | `geom_importance` now computes it; it does not reproduce (0.968 ridge, 0.937 boosting; 0.86-0.97 across the five); report and log corrected, audited |
| **221 stale references in 77 files**, all from my own earlier reorganisation | every frozen module's "how to run" line, `RESULTS.md`, the README | repaired mechanically where a name maps to exactly one module; `paper/doc_refs.py` now fails the audit if one goes stale |
| `vs_crisprhal`, `data_efficiency` and four modules built on them **could not be re-run as documented** | `ROOT.parent / "results"` assumes a checkout layout that `run_ext.sh` does not produce | one resolver in `common.py` that tries both layouts, does not raise at import, and fails clearly on use |
| Finding 5's table **cited seven families but is five** | SHAP is undefined for the two linear models | caption corrected; the linear models make the default agreement worse (-0.057), which strengthens the claim |
| Finding 4's headroom table **cited the wrong file** | `headroom.csv` is a later 6,769-column run (strongest held-back 0.155), not the table's source | cited `representation_headroom.csv`, which reproduces all three rows; row 1 also re-derived from code |
| four report tables **had their data only outside the repository** | importance (3 files), k-mer order, crisprHAL re-run, representation headroom | each checked against the report, copied in, audited |
| `RESULTS.md` cited **`human_ladder.csv`, which never existed** | the aggregate was never saved | its figures are the means of two fold files that were already in the repo; now cited and audited |
| ~30 further small summary files **still exist only outside the repository** | written by frozen-phase modules whose conclusions the report states | **not copied** -- unchecked copies would look like evidence. Listed in `CODE_MAP.md`; open-questions register row 10 |

Two mistakes of mine, corrected on the way: a sentence in 22.2 and in `ceiling_table.py` said `ceiling2` "tests directly" the
replicate-independence assumption, when comparing two *different enzymes* cannot; and the first draft of `REMOVED.md` said the human fold
files' means matched the report beyond chance, which held for only about half of them.
