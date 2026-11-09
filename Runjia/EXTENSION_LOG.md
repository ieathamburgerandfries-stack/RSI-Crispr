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
| DeepSpCas9variants (GitHub NahyeKim) | trained models only, no training data; published accuracies are in the paywalled article | weights **not run**; published per-variant Spearman (Fig. 4b of the article, supplied by the user on 9 Nov, `lit_review/`) compared with ours in `results/ext/kim2020_published_vs_ours.csv`: mean 0.896 against 0.881, ahead on 5 of 9 variants, evoCas9 0.683 against 0.81 |
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


## 7. Giving the model more of the enzyme: composition and structure, learned from many variants (added 9 Nov)

*Why this section exists.* In section 3 the enzyme enters the model as a handful of numbers per nuclease and there are
only nine nucleases, so nothing richer than a few columns can be learned. A richer description of the enzyme (its residues,
their chemistry, where they sit in the structure) only makes sense where there are enough enzymes to learn from. The one
public set found with a few hundred variants is Silverstein et al. 2025 (Nature 643): 770 SpCas9 variants that differ at six
PI-domain positions (D1135 S1136 G1218 E1219 R1335 T1337), each with the HT-PAMDA log10 rate constant for 64 PAMs (N + 3 nt).
Data: `training_data/220924_6pos_selected_random_only.csv` of github.com/RachelSilverstein/PAMmla (BSD-3), 729 KB, read as
text, no code from the repository run. Code: `pam_variants.py` (loader, descriptors, metrics), `pam_learn.py`,
`pam_analyze.py`. Results: `results/ext/pam_variants/`.

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
`results/ext/tune_*.csv`, `tune_configs.json`.

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

**What 200 draws say about each knob** (marginal means; `results/ext/tune_stage1_effects.txt`). Shallower and slower is better:
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
families that do differ are too weak to help at equal weight. **Status: verified on one gene split;** the blend weights come from one OOF partition.

## 13. Against crisprHAL 2 again, and the cost of the extra accuracy (9 Nov; `vs_crisprhal.py`, `vs_crisprhal_wt_hal.py`, `cost_bench.py`)

Three conditions of the original comparison with four SLICER arms (champion, tuned XGBoost d099x, XGBoost average of 5, mixed ensemble =
the mean of the XGBoost, LightGBM and CatBoost group averages). crisprHAL 2's shipped WT-SpCas9 weights were run on their own WT test file (6,737 guides,
Spearman 0.6945 against their published 0.695) to get paired predictions; the eSpCas9 and five-fold predictions were the stored ones. Results, rho
and the paired difference to crisprHAL 2 (95% bootstrap over guides, 4,000 resamples) are tabulated in the report ("After retuning") and in
`results/ext/vs_crisprhal_{espcas9,wt,folds}.csv`. The champion arm reproduces the earlier figures (0.7842 and +0.0021 [-0.0031, +0.0072] on their eSpCas9
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

Seven readings, each tested rather than assumed (numbers in the report's Appendix B subsection and `results/ext/interpret_*.csv`): the interaction-order curve (stumps 0.6905, depth 2
0.7177, depth 3 0.7206, depth 4 0.7210 on CONFIRM, ridge 0.647: 59% of what trees add to a linear model is per-column nonlinearity, 37% pairs, 4% third and
fourth order); the gain by column kind (indicators 35% -> 51%, quantum 37% -> 27%, flank 27% -> 22%; 131 -> 223 indicator columns in the top
300 / 450); the gain by position (eight PAM-proximal positions 63% / 62%; position 18 alone 14%; 3 nt after the PAM 6% / 5%); within-gene against
between-gene gain (+0.018 each); data scaling (+0.0095, +0.0061, +0.0143 at 25%, 50%, 100% of the tuning genes). Limits: gain shares are model-dependent importances over correlated columns, positions
are k-mer start positions (positions 19 and 20 have few columns), the data-scaling points use 3 random gene subsets and the 50% point is out of order.
