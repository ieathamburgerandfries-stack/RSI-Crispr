# Extension work: 3D structure and nuclease information (private-fork experiments)

Everything here lives in the fork, outside the frozen pipeline. New code is `Runjia/sgrna/ext/`;
results are `Runjia/results/ext/`. The frozen modules are imported, never edited. The shared
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
the sandbox refused to execute third-party model code. The pentamer table is the data route.

### 0b. The search: what was found, what was used, what was not

| candidate | what it offers | outcome |
|---|---|---|
| Kim et al. 2020 (Nat Biotechnol) Supplementary Tables | 40,128 targets x 9 Cas9 variants with the authors' split; 256-PAM profiles | **used** (Finding 9); the only public sheet found that gives >= 9 nucleases on the same targets with processed activities |
| DNAshapeR pentamer table (Rohs lab) | 512 pentamers x 90 numbers: MC, X-ray and MD shape parameters, step parameters included | **used** (Finding 8); the "several kinds of column in one table" case |
| PDB 5F9R (Jiang et al. 2016) | Cas9 - sgRNA - target / non-target DNA R-loop | **used** for nuclease structure descriptors |
| Deep DNAshape (Li, Chiu, Rohs; GitHub JinsenLi/deepDNAshape) | neural predictor, 14 shape features with up to 7 bp of context | cloned, **not run**: the sandbox refused to execute third-party model code. The DNAshapeR table gives the 5-mer-context version of the same features; the wider context is what was not tested |
| ABC / muABC tetranucleotide MD stiffness (Pasi et al. 2014) | 6x6 stiffness per tetranucleotide | **not obtained**: no direct download page found; this is the data that a true stiffness-tensor encoding would need, and its absence is the main limit of Finding 8 |
| E. coli Hi-C / 3D chromosome (Lioy et al. 2018, GSE107301) | contact maps, 3D coordinates | already family E (`eng.nuc`), a null; not repeated |
| E. coli NAP ChIP-seq (GSE24991 H-NS/Fis; SRP008538 IHF/HU) and IPOD-HR protein occupancy (GSE164796) | protein-bound DNA tracks | **not used**: no processed track was fetched; these describe the nucleoid, and Hi-C-derived family E was already null. An open item |
| DeepSpCas9variants (GitHub NahyeKim) | trained models only, no training data; published accuracies are in the paywalled article | **not used**; no state-of-the-art comparison for human nuclease variants is made |
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
`results/ext/nuclease_indist.csv`. Mean over nine nucleases (rho / NGG):

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
`results/ext/nuclease_lono_rep.csv`, saved predictions in `lono_preds/`; 3 replicates (different 15k-row
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
   composition with R2 0.963**: a sum of step twists is, to that accuracy, a weighted dinucleotide count.
   The others are path coordinates at +12 and +25 bp (R2 from composition ~0: they encode the identity of the
   first 12-25 nt after the PAM, which family A holds as position-specific columns).
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
