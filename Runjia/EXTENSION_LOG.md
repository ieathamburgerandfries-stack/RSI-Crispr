# Extension work: 3D structure and nuclease information (private-fork experiments)

> **Finding numbers in this log are those of 20 November 2026 and are not rewritten, because the log is a record of what was
> believed when.** On 22 November the report was reorganised: the old **Finding 7** (the comparison with crisprHAL 2) is the first
> section of **Part 5**; the old **Finding 8** (shape and 3D) is **Appendix B4**; the old **Finding 9** (nuclease) was removed
> (`REMOVED.md`); the old **Finding 10** is now **Finding 7**. Findings 1–6 are unchanged.

Everything here lives in the fork, outside the frozen pipeline. The code is organised by function across `Runjia/sgrna/` (it lived in `sgrna/post_freeze/` until 22 Nov);
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

**Run.** `ext/cut_within_gene.py` -- the same decomposition used on Cui and on its E75 arm
(`strand_within_gene.py`): gene-grouped CV, ten seeds (41-50), genes with at least five guides;
30,203 genic guides in 4,013 genes.

| | base | + family A | gain | seed sd of gain | 95% bootstrap over genes |
|---|---:|---:|---:|---:|---:|
| overall rho | 0.561 | 0.717 | +0.155 | 0.0008 | |
| between genes | 0.438 | 0.713 | +0.275 | 0.0029 | |
| **within genes** | 0.618 | 0.703 | **+0.085** | 0.0007 | [+0.078, +0.090] |

**Inference (the earlier dCas9 claim, re-examined).** Finding 3 said family A adds nothing to the
dCas9 screens beyond the strand rule. The within-gene decomposition already showed +0.102 (Cui,
coding strand) and +0.114 (E75 arm, one experiment with Cui) there. The cutting screen's within-gene gain is +0.085, so in
raw rho the dCas9 coding-strand gains are *not smaller*. That comparison is not valid as it stands,
because the three screens start from very different bases (within-gene rho 0.62 on the cutting
screen, 0.17 and 0.08 on the dCas9 coding strands) and a gain of 0.1 from 0.08 is not a gain of 0.1
from 0.62. Two scale-free comparisons:

* *Fraction of the attainable gain closed.* Cutting: ceiling 0.946 (the label's replicate ceiling,
  Finding 6), (0.7035 - 0.6181) / (0.946 - 0.6181) = **26%**. Cui coding strand: the screen's two
  dCas9 arms (strains LC-E18 and LC-E75, `fit18` and `fit75`; two expression levels of one experiment,
  not a time course and not true replicates) agree within genes at rho 0.768 on exactly these 2,289
  guides, which by Spearman-Brown puts the reliability of their mean at 0.869 and the ceiling near 0.93;
  (0.274 - 0.172) / (0.93 - 0.172) = **13%**. The E75 arm ("Rousset": the `fit75` column of Cui's own
  `screen_data.csv`, the same experiment) has one readout, but the same 1,759 coding-strand guides carry
  the E18 arm and agree within genes at rho 0.805, which with Cui's reliability puts its ceiling at
  0.864 or above (r^2 / rel_C = 0.746); (0.196 - 0.083) / (0.864 - 0.083) = **at most 15%**.
* *Variance explained* (rho squared): +0.113 (cutting) against +0.046 (Cui) and +0.032 (E75 arm).

Both say the same thing: the dCas9 within-gene effect is **real and between a quarter and a half the size of
the cutting one** (half by share of the attainable gain, 28-40% by variance explained), not absent and not equal. **Status: verified** for "not absent" (ten seeds, bootstrap
interval clear of zero, two arms of one screen, survives conditioning on transcription context); **open** for
"a quarter to a half the size" (rests on a reliability ceiling estimated from two dCas9 expression arms that are not
true replicates, and on an E75-arm ceiling that is only a lower bound; the two dCas9 rows are one experiment, not two screens).

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

*Current state of the audit (24 Nov):* it has grown with each section and now stands at **3359 checks, 0 mismatches**,
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
supplied turned out to land in a gap Part 3 names explicitly. Written up as **Part 5 of the report**, in the main text
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
  gene. That is the same guide/gene split as our within-gene versus between-gene analysis and as Finding 3's "+0.085
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
`sgrna.controls.equiv_check` refitted two models and **rewrote `featuriser_equivalence.csv`** with timings measured on
a machine that happened to be running an experiment, which then made the audit fail on two numbers that had nothing wrong
with them. Importing `sgrna.paper.verify_numbers` ran the whole audit and raised `SystemExit` **in the importing
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

### 23.4 Later on 22 November (recorded here because they were done after 23.2 and 23.3 were written)

* **The dCas9 / CRISPRi experiment was demoted** from the second half of Finding 3 to **Appendix B5**. Its reading changed over four
  rounds and ended weakly identified, and the screens differ in assay, library, laboratory and phenotype at once, so no mechanism
  follows from it. The strand lesson stays as a method note; Figure 2 lost its cut-versus-bind panel.
* **`sgrna/` was reorganised by function**: `post_freeze/` and `strand/` were dissolved into `tuning`, `build`, `features`, `cohorts`,
  `controls`, `interpret`, `compare`, `cost` and `paper` (46 modules, `git mv`, imports rewritten from the AST).
* **`NOTE_human_twin_leak.md` was removed** at the user's word that the hand-off was done; `results/INDEX.md` was added (generated and
  audited), the top-level `README.md` was brought up to date, and the speed block of `CODE_OVERVIEW.md` was merged into the head-to-head.
* **`colab/crisprhal_retune_own_splits.py`** was written for the missing arm of the head-to-head (not yet run on a GPU).
* Two errors of mine found on the way: Figure 2's baseline averaged three repeated fold rows from the frozen file (so it read "1 kb
  retains 6%" where the report says 8%), and two shift-table cells had been rounded from rounded values.

## 24. What the response curve says: three predictions, one failure, and the hypotheses that follow (22 Nov; the user asked to expand the interpretation, "to extract more insight or at least make some hypotheses that they support")

**Why these three.** Finding 7 says what the flank signal is made of and leaves "why" open. Three of the open readings could be turned
into statistics the cohort can answer without a model, so I wrote each prediction and its decision rule into the docstring of
`sgrna/interpret/hypotheses.py` *before running anything*, with a stated probability where I had one.

| | prediction (written first) | outcome |
|---|---|---|
| H1 the form of the flank x position-20 coupling | synergy: steeper flank slope where the seed base is favourable | **held**: T minus G +0.066 [+0.029, +0.102] at position 20, +0.087 at 18 |
| H2 the shape of the dependence on flank GC | roughly linear (probability ~0.6) | **failed**: a step at the AT-rich end pooled, graded within genes |
| H3 PAM density rather than GC | composition only (probability ~0.8) | **held**: PAM density at the 52nd percentile of a GC-matched null |

**What H2 led to.** The failure was the useful result. Pooled, Spearman of the label with flank GC is +0.485 inside the lowest 10% of
guides and +0.021 in the rest; the lowest decile sits 3.10 label units lower (0.95 SD) and spans 857 genes. I then ran follow-ups that
were designed after seeing it, so they are exploratory and the report calls them descriptions: robustness to dropping the ten biggest
contributing genes (a gap of −2.97 remains), replication with another enzyme at the same loci (−2.15, replicates) and in another
genome (not established), the effect by expression tertile (present in all three, larger in the lowest), and a within-gene check that
shows a second, graded component (+0.066 once the tail is set aside).

**What I got wrong along the way.** The first version of the hinge check clipped flank GC at a threshold and took the Spearman of the
clipped variable. Clipping ties most of the sample, so it overstated the tail's share (it gave 0.268 against a full-range 0.158);
the table in the report is Spearman *inside* each part of the range, which is the right statistic.

**What it changes in the report.** "A smooth compositional gradient" is true in space (the shift null) and not in composition, so Finding 2's
subsection, Part 2 and the Finding 7 claim now say so. The "duplex stability, melting" reading of the GC specificity is weakened,
because the sign and the shape are wrong for it, and a hypothesis about access (nucleoid proteins on AT-rich, horizontally acquired
DNA) is stated with the test that would settle it, which needs a published H-NS occupancy track that is not on disk (register item 2).

**A correction to the first draft of this section (same day).** The first draft listed four "hypotheses" and presented H-NS as the leading one.
Challenged on whether they followed from the findings or were unfalsifiable, I graded them and cut two. The H-NS reading came from outside the data:
the findings support a pattern (an AT-rich penalty, between regions, not expression, not outlier genes) and say nothing about which factor causes it,
and my by-name reading of the lowest-decile genes was recollection and not annotation. The tree-versus-CNN idea had no support at all. The gate
reading had a plausible alternative (a compressed label) that I had listed and not checked; I checked it (equal label spread, ordering survives
dropping the lowest quartile). I also ran the one check that bears on the protein reading, the scale of the step, which peaks at 250 nt and
softens toward a kilobase, so it argues against the domain-scale version. What is left is sorted into established results, inferences with what
would falsify each, and one candidate explanation, with its three falsifiers.

## 25. Eukaryotes: the guide's rules and the flank step across human, mouse and zebrafish (22 Nov; the user asked to compare different eukaryotes, "since we essentially have the whole pipeline down")

**What the pipeline did and did not make easy.** The guide-intrinsic part is organism-agnostic by construction (any 20-mer goes through the base
representation), so the maps, the contrast and the transfer matrix were cheap. The flank part is not: it needs a genome and coordinates, and the public
tables carry only 77 nt of context (30 before the guide, 47 after the PAM), so the flank check is at that scale and not at 250 nt. The hard part was
the data, not the code: 23 plain-text tables from the CRISPOR benchmark collection (github.com/maximilianh/crisporPaper, `effData/`), about 7 MB, into
`external_data/crispor_effdata/`, read as data and never executed.

**Predictions, written into `sgrna/compare/eukaryote_rules.py` before the run, and what happened.**

| | prediction | outcome |
|---|---|---|
| P1 | mammalian maps agree, corrected correlation at least 0.4 for every pair; mouse-human no lower than human-human by more than 0.2 | **median held** (human-human +0.59, human-mouse +0.56) **tails failed**: human-human ranges from −0.05 to +0.71 raw, the weak pairs involve Wang 2015, Doench 2016 and zebrafish |
| P2 | E. coli unrelated to every eukaryote (|rho| at most 0.2) and opposite in the position-20 contrast | **held**: median −0.14, two pairs beyond 0.2 (Doench 2016 −0.28, zebrafish −0.21); contrast opposite in all eleven |
| P3 | zebrafish agrees with mammals less than they agree with each other (0.2 to 0.5) | **held, more strongly**: human-zebrafish +0.28 corrected, mouse-zebrafish +0.07 |
| P4 | mammal-to-mammal transfer keeps 60% of the target's own cross-validated rho; E. coli to eukaryote within 0.05 of zero | **held for mammals** (70-73%); **E. coli median −0.04 but down to −0.24** (zebrafish), so not within 0.05 everywhere |
| P5 | no AT-rich step in mammals (probability 0.7) | **partly failed**: a weaker step in five of eleven datasets (−0.16 to −0.45 SD against −0.70 in E. coli at the same scale), none in the other six |

**What I had to be careful about.** Scores run in opposite directions, so each dataset was oriented by a consensus of six published predictors used
for the sign only; the weakest anchor (Wang 2015, −0.10) is printed. The two Hart libraries contain no T at position 20, so G minus T is undefined
there and the contrast was changed to G minus the mean of A and C, which is defined everywhere. A first summary of the transfer matrix pooled both models
and printed identical rows for each; I caught it because they were identical to three decimals, and the numbers in the report are per model.

## 26. A representation on which the model can be read column by column (22 Nov; the user asked whether SLICER is interpretable at all and to run the compact-model test)

**Why.** Finding 5 attributes the instability of per-column importance to redundancy, which predicts that it disappears on a non-redundant representation. `sgrna/interpret/compact_model.py` tests that on 80 position x base indicators plus 24 flank composition
columns, with predictions written into the docstring first.

| | prediction | outcome |
|---|---|---|
| P1 | the 104-column model keeps at least 90% of the full model's rho (probability about 0.6) | **failed**: 88% (91% with the 304 adjacent-pair indicators) |
| P2 | SHAP agrees between the five tree families at rho >= 0.8 per column (probability about 0.7) | **held**: 0.872 against 0.496 on the full matrix |
| P3 | default importance improves but stays below SHAP | **held**: 0.698 |
| P4 | the SHAP-implied position x base effects reproduce the model-free map at rho >= 0.8 | **held**: 0.851 (Pearson 0.963) |
| P5 | the flank contribution is a step at the AT-rich end | **held**: -0.78 SD in the lowest decile, -0.04 in the second, +0.07 to +0.17 above |

**What it settles.** The redundancy account of Finding 5 holds: with no restated columns the per-column ranking survives a change of model family. The cost is about an eighth of rho, so the readable model is a counterpart and not a replacement.

## 27. Is SLICER more interpretable than crisprHAL 2? (22 Nov; the user: interpretability is a valid claim, the issue is whether SLICER is *more* interpretable)

**Why.** "Interpretable" is not one quantity, so it was split into four questions that either model can be put through unchanged: faithful (sensitivity-n at block level), stable (agreement between independently refitted models), true (agreement with the model-free position x base map of Finding 7) and compressible (how much of the model's output a 104-column readable surrogate reproduces). Both models as released, on crisprHAL 2's own WT-SpCas9 split: its shipped weights, and SLICER (tuned, d099x) fitted on the same training rows, plus SLICER-104 where the question is about positions. Predictions and the decision rule are in the docstring of `sgrna/interpret/vs_crisprhal_interpret.py`, written before any crisprHAL 2 number existed; the decision rule for stability was tightened from "bootstrap over refit pairs" to "the ranges of the five pairs do not overlap" before the run, since five dependent pairs are too few for a bootstrap. One reading needs no attribution method (the substitution map: the model's mean response to setting a guide base to each base) and is the unit common to all three models; the shipped SLICER's edited 20-mer is re-featurised with `base_fast.matrix_fast`, and an assertion checks that re-featurising an unedited 20-mer reproduces the stored matrix exactly.

**Cost.** crisprHAL 2: six refits by their own code and 48 epochs (about 17 minutes on all 26,994 rows, 9 to 11 on halves, on this machine's CPU) plus four to seven minutes of expected-gradient attribution each; SLICER: about 35 seconds per fit.

| | prediction | outcome |
|---|---|---|
| P1 | SLICER's block-level sensitivity-n at least 0.9; crisprHAL 2's lower by at least 0.03 (probability about 0.5) | **held**: 0.961-0.971 against 0.860-0.881; 0.896-0.906 for the network at 800 estimator draws, so about a quarter of the gap is Monte Carlo noise |
| P2 | crisprHAL 2 at 0.8 or more per channel across refits (about 0.7); lower on SLICER's own selected columns (about 0.8) | **held**: 0.986 over seeds and 0.957 over halves; SLICER's columns 0.578 and 0.278. So there is no stability claim at the position x base level, and across data halves the network is steadier |
| P3 | both read-outs at 0.8 or more against the model-free map, within 0.1 of each other (about 0.7) | **held except one cell**: the network's substitution map 0.783 (its attribution-implied map 0.862); SLICER shipped 0.810, SLICER-104 0.853 |
| P4 | compressibility within 0.03; crisprHAL 2 the better described (about 0.3) | **did not hold as stated**: fidelity favours SLICER by 0.020 (additive) and 0.041 (boosted); the share of accuracy kept favours crisprHAL 2 by 0.015-0.023. The two disagree because SLICER is the more accurate model |
| P5 | crisprHAL 2 moves the same way under flank-GC edits, by less than SLICER's -0.88 at GC -5 (about 0.8, 0.7) | **held**: -0.57, and +0.31 against +0.50; the fall is 1.8 times the rise in both |

**What it settles.** SLICER is interpretable and is not more interpretable than crisprHAL 2 by these measures; its attributions are the more faithful and nothing else separates them. Two by-products: the two models' substitution maps agree at rho 0.97, and an independently built network with a 189 and 169 nt window reproduces the asymmetric flank-GC response, which supports Finding 7 and says the effect is not an artefact of SLICER's features.

**What I had to be careful about.** The attribution estimators differ (exact TreeSHAP; expected gradients sampled 50 times), so a control raised the draws to 200 and 800 on the same 100 guides; the block sets differ (12, 28 and 25 blocks), so SLICER-104 (28) is reported beside the shipped model (12). The network's refits on halves use its 48 epochs on half the steps (rho 0.662-0.671 against 0.691-0.692 on all rows). The shipped SLICER's per-column stability is measured over the union of the columns the two fits selected, with zeros for the columns one did not use. Nothing was tuned on these guides in this test, but the tuned settings were chosen on curated folds that contain many of them, which does not affect a comparison of readings. Intermediate readings are tracked in `results/post_freeze/interp_hal_cache/` (the refitted weights are not kept) so the comparison tables rebuild without the 90 minutes of refitting: `./run_ext.sh sgrna.interpret.vs_crisprhal_interpret --step assemble`.

## 28. A line-by-line review of the report, and what it corrected (23 Nov; the user: "make sure it is ALL correct")

**Why.** The report was about to be condensed into the paper (the Google Doc) and then committed in two places, so every statement of `PROGRESS_REPORT.md` was checked against its stored file, its protocol and its source. Nine reviewers each took a slice of the report and every correction was applied as a patch whose `old` text had to occur exactly once. Each number a reviewer could recompute became a check in `sgrna/paper/report_checks_*.py` (1,298 checks, run by `sgrna/paper/report_numbers.py` from `./run_audit.sh`); R3's refits (about two minutes) run every time, R2's SpCas9 to eSpCas9 refit is stored in `results/post_freeze/transfer_esp_unseen_guides.csv` and redone only with `RSI_AUDIT_HEAVY=1`.

**What was wrong, by kind** (about sixty real errors; none moved a headline, several moved how a headline is worded):

| kind | examples |
|---|---|
| a number quoted without its protocol, or with the wrong one | the +0.164 flank gain is random-row five-by-five CV on the curated cohort, +0.157 gene-grouped, +0.156 on sealed genes; 0.7215 is the 1,204 CONFIRM genes and 0.7314 gene-grouped CV; SLICER's tuned mean over its own five folds is 0.7212 and the old 0.7218 is in no file |
| a number with no source | unsourced Part 5 claims were removed or replaced by one that a file backs; the hand-entered 10,000 CPU-s for crisprHAL 2 is labelled an estimate |
| a source misdescribed | Cui's two readouts are two dCas9 strains (LC-E18, LC-E75), not time points, and "Rousset" is Cui's E75 arm, one experiment and not a second screen (Wang is the one independent screen, and the within-gene null does not replicate there); Alkan 2022 was Corsi et al. 2022 (CRISPRoff itself is Alkan et al. 2018); Noshay's table lists 40,468 sgRNAs, the released matrix 13,880 rows in a 1,222-gene panel; eleven sources cited in text had no entry in Appendix D and now have one, Crossref-checked |
| a bound that is not one | the label's ceiling 0.946 is not a lower bound: a Dirichlet-CLR re-implementation of ALDEx2 gives replicate agreement 0.798 and ceiling 0.942, so it is 0.942 to 0.946 by estimator |
| a transfer row that shares its library | SpCas9 to eSpCas9 is one guide library; on the 25,922 eSpCas9 guides absent from the SpCas9 screen the transferred ρ is 0.662 and family A's share +0.103 (all rows 0.689 and +0.098) |
| an assumption the label never checked | higher label values are more cutting: the 118 multi-copy guides have median 2.11 against 0.27 (`label_orientation.csv`) |
| a scale-free comparison that was not | the dCas9 within-gene gain is 13% (Cui) and at most 15% (E75 arm) of the attainable gain against 26% on cutting, a quarter to a half by variance explained (§1) |
| the eukaryote claim stated too strongly | all eleven datasets carry the eukaryotic sign as point estimates, ten of eleven intervals exclude zero (Wang 2015 does not), the four whose direction the orientation rule set are not independent of it, and a second bacterial genome has the bacterial sign (*C. rodentium* −0.42 [−0.45, −0.39]) |
| the sealed-gene drop called adaptive overfitting | see below |

**The sealed-gene drop (`sgrna/controls/sealed_holdout_gene.py`).** Sealing whole genes lowers family A's gain by +0.0087 [+0.0020, +0.0153] (15 sealings), measured as a difference of differences between random-row CV inside the development pool and the sealed genes. A random-row pool fold can reward recognising a guide's neighbourhood, so the control repeats the pool CV with whole genes held out (guides outside genes grouped by 100 kb block). On the first eight sealings (the compute allowed; their random-row figure is +0.0088 against +0.0087 over all fifteen) the drop is +0.0081 [-0.0037, +0.0199] with whole-gene pool folds, so random-row folds account for only +0.0007. The hypothesis that most of the figure was the cost of unseen genes to a locus-reading feature was therefore wrong. What the figure is not: the frozen arm re-derives no hyperparameter and an eight-draw search adds nothing, so the drop is not shown to be adaptive overfitting, and the design cannot reach the human choices (the cohort, family A's design and its window set). The wording changed from "adaptive overfitting is measured, and it is small" to "the sealed-gene drop is small, and not shown to be adaptive overfitting".

**The width experiment on all five folds (`colab/crisprhal_wide_window.py`, `sgrna/compare/crisprhal_wide_window_folds.py assemble`).** Finding 2's width result had been one fold. crisprHAL 2 at its published configuration (batch 256, no re-tuning) now has every arm on all five folds. Fourteen of the twenty arms ran on a T4 (the Colab csv files were not downloaded, so ten are transcribed from the console: `results/crisprhal_wide_window_fcolab.csv` and `_fcolab2.csv`, seven to four decimals and fold 5's three to six) and six on this machine's CPU (the 23 nt arms of folds 2 to 5 and the 378 nt arms of folds 2 and 3: 76 to 82 minutes for a 378 nt arm against about 6 on a T4, which is why the wider arms went to Colab only). What changed from one fold: crisprHAL 2 leads SLICER by +0.012 on the guide and PAM alone (5 of 5 folds; fold 1 had read as a tie), by only +0.009 at 378 nt (4 of 5 folds, sd 0.014, reversed on fold 3; +0.025 at its published batch), falls below its own 378 nt score at both wider inputs in every fold (-0.020 at 1,023 nt, -0.039 at 2,023 nt, from -0.003 to -0.058), and SLICER's full flank leads it at 2,023 nt by +0.066 (5 of 5 folds, +0.055 to +0.079). The direction is consistent and the size is not tightly measured: fold 3 is nearly flat, and its 378 nt arm, a CPU run, is the lowest (0.6602; the other CPU arm, fold 2, is the highest, so the hardware is not the explanation). A T4 re-run of the two CPU 378 nt arms (about 12 minutes) would show whether fold 3's is noise in the fit.

**What stays open.** crisprHAL 2 was not re-tuned on its own splits with the search SLICER got (`colab/crisprhal_retune_own_splits.py` is the script); Orro et al.'s SHAP details were read from a preprint; the hand-entered crisprHAL 2 fitting time; whether the flank effect is a physical property of the DNA or a position effect that relative position does not capture.

## 29. The second review round: one protocol, the reach of the flank signal, controls, transfer to every dataset, and the cost with its spread (23 Nov; the user's list of fourteen questions and requests on the Doc)

**Why.** After §28 the user asked, in one message, that the Doc be re-checked claim by claim and that thirteen further things be done: justify or drop each dataset; a full transfer table and a comparison with crisprHAL 2's transfer; an explanation of curation and ALDEx2; whether the counts of A, C, G and T beat GC and purine; why Spearman and not Pearson; how genes are assigned and what NC_000913.2 is; the other metrics a model is judged on and how SLICER compares; drop the vague "Analyses" section; why confidence intervals and not ± sd; report every figure under the one protocol with the least leakage; what "displacement" means and a plot of ρ against window size; redraw Figures 2 and 4 as data; make the cost section concrete with repetitions and compare it with BoostMEC.

**One protocol.** Every figure in the Doc's Results is now the default model under five-fold cross-validation with whole genes held out, three seeds, selection inside each training fold, the 20-mer alone (all 6,169 columns, no selection) as the baseline; ± is the sd over the 15 folds and differences carry bootstrap intervals. The selected-300 baseline (0.559) survives only in Appendix A3 with its label. The headline flank gain under it is +0.161 (ρ 0.547 to 0.707; sd 0.010 over 15 folds, positive in all; `flank_scale.csv`, arm `family A shifted 0` against `base`).

**New analyses, each a module with its own result file in `results/post_freeze/` (all recomputed by `paper_numbers_D.py`).**

| question | module | result |
|---|---|---|
| how far does the flank reach, and is it GC or composition | `sgrna/interpret/flank_scale.py` → `flank_scale.csv`, `flank_scale_summary.csv` | two GC columns peak at W = 250 nt (+0.105, sd 0.011) and fall to +0.042 at 5 kb; the four base fractions are never better (+0.103 at 250 nt); at the four family windows (3 seeds) GC +0.113, GC and purine +0.113, A/C/G/T +0.108; reading the family from the guide 100 bp, 300 bp, 1 kb, 3 kb away keeps 68%, 57%, 32%, 6% of the gain and from 10 kb or more none (−0.0001 to −0.0008) |
| what the signal looks like | `sgrna/interpret/signal_profile.py` | lowest decile of flank GC −3.154 [−3.359, −2.944]; deciles 3 to 10 within 0.15 of zero; slope by base at guide position 20 (A 0.219, C 0.169, G 0.119, T 0.184); the gain is positive in all five target-GC quintiles (+0.349 GC-poor to +0.072 GC-rich) |
| is it a gate | `sgrna/interpret/gate_test.py` | interaction over additive: R² +0.0035 (15 of 15 folds), ρ +0.0007 (14 of 15) |
| repeats, mappability, read depth | `sgrna/controls/mappability_stratum.py` | all +0.1606 [+0.1493, +0.1715]; strict 26,146 guides +0.1569; larger in repeated sequence (+0.2607 over 852 guides); given control depth +0.1600 |
| second screen as the sealed set | `sgrna/controls/sealed_crossscreen.py` | eSpCas9 labels in sealed genes +0.1033 [+0.0997, +0.1070], 15 of 15 (prediction written before the run: at least +0.08, 14 of 15) |
| curation | `sgrna/cohorts/curation_check.py` | inclusion by flank-GC decile 0.531 to 0.574; the 18,733 single-site guides with 30 to 72 unselected counts show the same flank association (ρ 0.159, deficit 0.94 SD) as those kept (0.175 to 0.188; 0.97 to 1.02 SD) |
| SLICER against crisprHAL 2 on more metrics | `sgrna/compare/metrics_panel.py` | AUROC for efficient guides +0.011 / +0.016 (SpCas9 / eSpCas9 split), for duds +0.008 / +0.011; top-decile precision and NDCG not detected; calibration slopes 1.01 against 1.08 and 1.05 against 1.05 |
| transfer, nuclease and kingdom, against crisprHAL 2 | `sgrna/compare/transfer_vs_crisprhal.py` | on the 25,922 eSpCas9 guides absent from the SpCas9 screen: SLICER default 0.659, tuned 0.678, crisprHAL 2 0.633; E. coli to the eleven eukaryotic tables: median −0.041 (SLICER) against −0.015 (crisprHAL 2's architecture on 23 nt) |
| the full transfer table | `sgrna/compare/transfer_to_ecoli.py` (the E. coli column; the other 12 x 11 cells were `eukaryote_transfer.csv`) | every eukaryotic model scored on E. coli is negative (median −0.058, −0.120 to −0.012) |
| cost with its spread | `sgrna/cost/cost_repeats.py`, `sgrna/cost/model_size.py` | see §29 results below |
| BoostMEC rebuilt as its authors built it | `sgrna/compare/boostmec_replica.py` → `boostmec_released.csv`, `boostmec_searched.csv`, `boostmec_search_trace.csv`, `boostmec_vs_slicer_confirm.csv` | released hyperparameters on SLICER's default protocol: ρ 0.620 ± 0.009 (−0.087 against SLICER default, 15 of 15 folds); its own search on the development genes, scored on the 8,943 guides of the held-out genes: 0.624 against 0.707 (default) and 0.722 (tuned) |
| why the diagonal of the transfer table is low | `sgrna/compare/transfer_diagonal.py` → `transfer_diagonal.csv` | the 20-mer pipeline's own cross-validation (0.18 to 0.55 on tables of 881 to 4,293 guides) beats the median of the six published predictors on all 11 tables (their medians 0.10 to 0.42) and their best on 3 |

**BoostMEC (rebuilt as its authors built it; the user: "do what they did or else it wouldn't be fair").** A first attempt (a feature set reconstructed from the paper's description, melting temperatures and energies approximated, a fixed LightGBM setting) was replaced after the user objected. The paper's Methods (Zarate et al., 2022, BMC Bioinformatics 23:446) and, more usefully, its repository (github.com/oazarate/BoostMEC, read in the browser pane, nothing downloaded) specify everything but the search ranges: the 149 features line by line (`model/featurize-dataset.R`, `boostmec-pipeline.py`: counts and not frequencies of every 1-, 2- and 3-mer; the 81-nt scaffold; TmCalculator 1.0.1's Tm_NN with table DNA_NN4 and 50 mM Na on the 21-mer and its positions 1-4, 5-12, 16-20; RNAfold energies of the guide alone and with the scaffold; poly-T runs; 28 + 28 categorical position features with the PAM's GG removed), the hyperparameters of the released model (`model/model.txt`, whose 'parameters' block gives learning rate 0.0099, 701 leaves, depth 12, 203 rows per leaf, 48% of columns, L2 1.48, 343 bins, row subsampling off) and its size (1,582 trees, 136,142 leaves, 11,455,146 bytes, counted from that file; these figures are in the repository and not in the article, and had briefly been removed from the Doc as unverified before this check), and the paper gives the tree-count rule (ten-fold cross-validation, early stopping after 10 rounds, limit 7,000, mean best iteration) and the search (Bayesian optimisation, 10 initial points and 100 rounds scored by that cross-validation). As a spot check (not a validation: the released model was not run), the values of a test 30-mer for the two energies and the four melting temperatures fall inside the `feature_infos` ranges stored in `model.txt`. Two arms: the released hyperparameters with the tree count by their rule on SLICER's default protocol (ρ 0.620 ± 0.009 over 15 folds, +0.073 over the 20-mer alone, −0.087 against SLICER's default model in 15 of 15 folds; 904 trees and 86,762 leaves on average), and their search run on the development genes (110 evaluations of ten-fold cross-validation, 16 minutes on ten cores; the ranges are mine because the authors' are not published, Supplementary Table S4 giving only the optimum), the model refitted on all development genes and scored on the 8,943 guides of the held-out genes: ρ 0.624 against 0.707 for SLICER's default model (−0.084 [−0.102, −0.065]) and 0.722 for its tuned model (−0.098 [−0.117, −0.080]). Searching moves BoostMEC by about +0.004, so the gap is the features: they stop 3 nt past the PAM. Not done: the released model was not downloaded and scored (it needs a 11.5 MB download, which was left to the user), and Supplementary Table S4 was not read. The first attempt's file (`boostmec_style.py`) and numbers (0.616) were removed.

**Transferability moved to its own appendix (A5).** The user judged the transfer results "pretty meh across the board" and asked for them out of the main Results; the main text now has two figures. The user also asked why the diagonal of the table is low: the 20-mer pipeline's own cross-validation on each eukaryotic table is 0.18 to 0.55, but the six published predictors read from the same CRISPOR files reach a median ρ of only 0.10 to 0.42 on those tables (their best 0.17 to 0.70; some were trained on them), so the pipeline is above the median predictor on 11 of 11 tables and above the best on 3: the tables are small (881 to 4,293 guides) and noisy and the model sees only the 20-mer.

**Two independent checks of the final text and what they corrected.** Two fact-checkers, one for the main text and one for the appendix, each recomputed the claims from the stored files (read-only). Corrected: the label's sign (it is unselected minus Cas9-selected, so it is high for a guide depleted by cutting, and the replicate-agreement caption now says minus the log ratio); curation described as keeping guides found in the genome with an unselected count of at least 73 (33,871 pass, 33,567 located; the earlier 'single-site' wording described the repo's own subset used for the check); 'does not depend on flank GC' became 'depends only weakly' and 'as strong' became 'about 10 to 15% weaker'; the 20-to-80 range of the cutoff (not in the source) was dropped; 'only ranks are used' became 'compared by rank'; 'which is why' the Noshay block gives a smaller gain became 'part of why'; the 0.559 against 0.547 baseline; the near-cut ablation (+0.055 and +0.070 alone, +0.0004 and −0.0054 on dropping a block from family A); the fine-structure percentages (95.2, 97.6, 98.6 and 96.0%); the width figure's y-axis (random-row folds, not gene-blind) and caption (SLICER has no 1,023-nt arm); the stability readout (SLICER's substitution map 0.998 against 0.989; attribution blocks tie at 0.995 and 0.997); the transfer caption (E. coli to eukaryote cells are near zero, 2 of 11 positive, not all negative); the displacement protocol (seed 41, five folds, the guide nearest to the offset); the sealing wording (20% of guides sealed by whole gene, intervals mean ± 1.96 se over overlapping sealings); the tuning sentence (settings searched with the flank present add +0.0028 on the 20-mer alone; no 20-mer-only search was run); the reach decomposition (explains the default model's lead, not the tuned model's); the C. rodentium interval (about four blocks); the 0.7215 held-out-gene figure beside 0.7314; and several protocol labels.

**Datasets pruned from the Doc.** Dropped: the human Noshay matrix (as a transfer table) and the CRISPRi screens (Appendix B5 of the report keeps them, the Doc does not). Kept, each with a stated purpose in the Methods: the curated cohort, crisprHAL 2's two splits, the eSpCas9 screen, C. rodentium, the eleven eukaryotic tables, the published matrix (only for the re-description finding and the importance instability), and the per-replicate counts (the ceiling and the read-depth and mappability controls).

**Corrections made on the way.** The width figure's axis said "gene-blind folds" for what are random-row folds (fixed); the replicate-agreement figure first gave ρ 0.804 because counts per million were computed after filtering (now over the whole library: 0.8097); a depth control used the numerator of the label (ρ 0.998) and was redone with the control counts only; flank windows were off by one and were validated exactly against the index flanks.

**Still open.** The head-to-head on crisprHAL 2's own splits is on random rows; the gene-grouped version needs `colab/crisprhal_gene_grouped.py` on a GPU (five folds of about six minutes), then `python -m sgrna.compare.gene_grouped_vs_crisprhal assemble`.


## 30. The cost split by hardware, BoostMEC run as its authors ship it, and the gene-grouped head-to-head (24 Nov; the user: "download boostMEC and check its cost ... split into quiet CPU and colab GPU times ... compared against boostmec ... does that mean our computational cost is lowest in this niche? ... check the rounding again"; then "surely the github repo is usable as-shipped")

**What was done, in order.**
(1) BoostMEC was cloned (`external_data/BoostMEC/repo`, commit cdf190c; read only). Its released `model/model.txt` (1,582 trees, 136,142 leaves, 11,455,146 bytes) is run through LightGBM with the category codes of `model/rules.rds`, which a small read-only parser (`sgrna/cost/rds_read.py`) extracts: the dinucleotide at positions 25-26 saw four categories in training and so is coded 1 to 4, not 1 to 16, and a first version that coded every dinucleotide 1 to 16 was off by 0.8 on the paper's worked examples until this was fixed.
(2) A Python version of its 149 features (`boostmec_replica.features`, from `featurize-dataset.R`) reproduces the paper's two worked predictions (Supplementary Tables S2 and S3: 49.53 and 28.42) as 49.19 and 28.41.
(3) The user asked why it was reimplemented when the repository is usable as shipped. The repository has no training code and needs R and RNAfold, which are not installed here; the authors' Docker image (`oazarate/boostmec:latest`, 1.2 GB, linux/amd64 only) was pulled with the user's approval and the pipeline (`model/boostmec-pipeline.py`: Python, RNAfold twice, two R scripts) run unmodified (`sgrna/cost/cost_boostmec_shipped.py`). The image's `model.txt` and `rules.rds` are byte-identical to the repository's. On an Apple M5 it runs under x86 emulation; RNAfold, the one compiled program, takes 1.1 times its native time on the 101-nt folds that dominate, and the R and Python parts could not be calibrated.
(4) Shipped against the Python version on 100 guides: identical energies, G and C counts, poly-T features, all 84 k-mer counts and all 56 categorical columns; only the four melting temperatures differ (R's TmCalculator against Biopython, 1.95, 0.91, 1.46, 1.10 degrees on average; no setting of Biopython's `Tm_NN` reproduces it), and predictions differ by 0.34 on average (3.19 at most; Spearman 0.9990). The released model in Python on the shipped features reproduces the shipped predictions exactly on all 33,567 guides (Spearman 1, largest difference 7e-15).
(5) The authors' pipeline was run on every guide of the cohort (171 s) and its features now feed the retraining arms of `boostmec_replica.py` (`BOOSTMEC_FEATURES=shipped`, the default; `--features python` keeps the reimplementation). Released hyperparameters: ρ 0.620 ± 0.009 over 15 folds, −0.087 against SLICER's default model, behind in 15 of 15 folds; the Python features had given 0.620 as well (paired difference +0.0002 ± 0.0009), so the melting-temperature discrepancy did not matter. Its own search, scored on the 8,943 guides of the genes held out of the search: ρ 0.623 against 0.707 (SLICER default) and 0.722 (tuned): −0.084 [−0.103, −0.066] and −0.099 [−0.117, −0.080]. The released model, trained on human cells, scores ρ −0.066 on the E. coli cohort, as the other eukaryote-trained models do (Appendix A5 of the paper).

**A finding about the shipped pipeline.** It cannot score a single guide: when a position-specific nucleotide column is the same T in every sequence, R's CSV reader makes it logical and `lgb.convert_with_rules` stops with a recycling error (eight such columns for one guide, one for two; a batch of 10 is safe). Nothing was changed to get round it; 10 guides is the smallest batch timed.

**Quiet CPU, one thread, from the raw sequence** (`cost_chain.sh` ran every timing one after another; the load average, 1.8 to 2.4 on ten cores while scoring, is stored with every configuration; `cost_repeats_machine.json` has the processor and library versions). Median ms per guide at 1 / 100 / 10,000: SLICER default 0.629 / 0.0726 / 0.0795, tuned 0.716 / 0.0885 / 0.0935, BoostMEC (Python features, released model) 2.94 / 3.07 / 3.11, crisprHAL 2 15.4 / 1.37 / 1.22; coefficient of variation at most 4%. crisprHAL 2 over SLICER default: 24.5 [24.4, 24.5], 18.8 [18.8, 18.9], 15.4 [15.3, 15.4]; BoostMEC over SLICER default at 100: 42.3 [42.2, 42.5]. The shipped pipeline: 3.74 s for 10 guides, 4.17 s for 100, 53.5 s for 10,000 (5.35 ms per guide, 67.3 times SLICER default; 1.72 times the Python version); at 10,000 guides RNAfold on the guide with its scaffold takes 31.6 s (60% of 52.5 s), the R feature script 16.8 s, the R prediction script 2.18 s.

**Colab T4** (`colab/cost_gpu.py`, which replaces four older Colab benchmarks, see `REMOVED.md`; synthetic inputs; every repeat stored in `results/post_freeze/cost_gpu.csv`). crisprHAL 2 from raw contexts: 74.5 ms for one guide, 1.07 per guide at 100, 0.311 at 10,000; with the encoded batch already on the card 40.7, 0.529 and 0.0837 (batch size 2,048); that last figure is 1.05 [1.04, 1.07] times SLICER default's whole one-thread CPU path. **So the 18-fold CPU scoring advantage does not carry to a GPU at large batches**; the paper now says so. SLICER-shaped trees gain little from the card (forward pass at 10,000 on one CPU thread 0.00636 ms per guide, 0.000617 on the card), because the features, built on the host, are the cost. Fits on the T4: crisprHAL 2 326 s (3 fits), SLICER-shaped XGBoost 3.20 s against 23.8 s on Colab's CPU (default) and 29.0 s against 479 s (tuned).

**Fits on the laptop** (wall-clock, all ten cores): SLICER default 12.8 s, tuned 24.5 s, BoostMEC 13.2 s (29.2 s with its ten-fold tree count; its search took 17 min), crisprHAL 2 17.3 min (three fits, 17.0 to 17.3). That is 81 times SLICER default on the same machine and 25 times on the T4. CPU seconds are given in the paper's appendix with a caveat: XGBoost's threads spin while waiting, so they overstate the work. **The earlier "about 150x less CPU time to fit" was an estimate (crisprHAL 2's CPU time had been entered by hand); measured, it is 118x in CPU-seconds (9,617 against 81.5), 81x in wall-clock and 25x against a T4. The Abstract tab still says 150.**

**Sizes** (`model_size.csv`, now with gzip): SLICER default 6,338 leaves, 0.68 MB (0.15 MB compressed); tuned 113,540, 10.2 MB (2.71); BoostMEC 136,142 leaves in 1,582 trees, 11.5 MB as LightGBM text (4.14 compressed); crisprHAL 2 790,533 parameters, 9.6 MB. The tuned SLICER model is not smaller than BoostMEC's: only the default is.

**Is SLICER's cost lowest "in this niche"?** Among the three models costed, the default model is cheapest to fit and store and, on a CPU, to score, and it is the most accurate; BoostMEC costs more on every measure and scores lower. That does not extend to the niche: no other bacterial predictor was costed (most publish nothing), a linear model on the 20-mer costs less and was not run, and crisprHAL 2's scoring cost equals SLICER's for large batches on a GPU. The paper says this in its Discussion.

**The gene-grouped head-to-head** (`colab/crisprhal_gene_grouped.py`, whole genes held out, the folds of every other SLICER number; crisprHAL 2's published configuration, 378 nt, batch 1,024, one seed, a Colab T4). crisprHAL 2 scores 0.6928 (sd 0.0137 over the five folds) against SLICER default 0.7070 (lead +0.0143, sd 0.0059, ahead in 5 of 5 folds) and tuned 0.7234 (+0.0306, sd 0.0077, 5 of 5). On random-row folds crisprHAL 2 scores 0.6971, so removing locus leakage cost it only 0.004: leakage did not favour it. A first run at batch 256 (the batch of the width experiment) scored 0.6796, a handicap of 0.013, and made the leads look twice as large (+0.0274, +0.0438); it is kept as `crisprhal_gene_grouped_fcolab_b256.csv` and described in Appendix A2 as not the comparison. The published-batch result is in the paper (Results, head-to-head), as fold means with no per-guide interval (crisprHAL 2's predictions were not stored).

**Is there an optimal window?** No definitive one. The two GC columns were tested at nine windows (10, 25, 50, 100, 250, 500, 1,000, 2,000, 5,000 nt, seed 41, five folds): +0.023, +0.051, +0.076, +0.095, +0.105, +0.097, +0.081, +0.061, +0.042. 250 nt is the best of the grid and beats both neighbours in 5 of 5 folds (by +0.0095 over 100 nt and +0.0080 over 500 nt), but the curve is flat to within 0.01 from 100 to 500 nt, the grid is coarse (a factor of two or more between neighbours), and a quadratic through the points peaks near 240 nt. The paper now says 'highest at 250 nt of the windows tried'. A finer grid (150, 200, 300, 400 nt) was then run (section 31): the highest of 14 symmetric windows is 200 nt (+0.106), the plateau runs 150 to 500 nt, and tuning the two sides separately finds nothing that replicates; the paper now says 'highest at 200 nt of the 14 windows tried'.

**What the rounding check found.** Every number the audit recomputes was re-tested against its full-precision value (`AUDIT_RAW`): of 3,244 claims, none is an exact tie, and at the precision it is printed only seven are not the nearest rounding: three internal reproduction checks, two intentional 'about' figures, and two real errors in the Doc's Appendix A1, which had said that 32 columns of 125-nt bins carry 97.6% and 80 columns of 50-nt bins 98.6% of the composition signal; the stored table gives 97.2% and 98.7% (`novel_resolution.csv`). They had never been audited (the section 29 fact-check listed them as verified). Corrected in the Doc, and checked in `paper_numbers_A.py`. A scan of every number in the final text against the audit's claims (911 numbers) left 50 unmatched, almost all settings, seeds, published constants and version numbers; the computed ones among them (the 25,198 guides, 0.559, the dataset totals, BoostMEC's hyperparameters from `model.txt`, the pick-100 interval ends, the image size) were given checks or corrected: **25,198 'guides in one 236 kb region' had no source in any stored file and was replaced by the traceable 25,210 located guides of 25,244 in the files.**

**Housekeeping.** The page estimator did not let paragraphs split across pages and demanded that a whole figure caption fit beside its figure; both fixed (the earlier "6.8 pages" was 6.25). The main tab is 7.5 pages without references after this round. The four superseded Colab benchmarks are removed (`REMOVED.md`). The shared repository's checkout has no `data`/`external_data` links, so the audit is run from the fork.

**Still open.** The Abstract's "150 times" (the user's tab); the 88-91% of the Abstract against 89 and 91 in Appendix A5 (the audit's [20] checks give 91 and 89); a linear 20-mer baseline to put a floor under the cost-accuracy plot.

**The Methods were rewritten (24 Nov, the user: "talk more about what it DOES instead of what it IS").** Each Methods bullet now says what a dataset or step does for the study; the counts, column numbers, split sizes and model settings moved to new bullets of Appendix A4 ("Data, features, models, label and metrics in detail"), and curation is one clause with its mechanics in A4. The reference genome is no longer named in the main text. It is NCBI's NC_000913.2 (4,639,675 bp), the build released with the screen of Cui et al. (2018) and used by the GapR-seq tracks; NCBI lists it as replaced by NC_000913.3 (4,641,652 bp), recorded with its lookup date in `results/post_freeze/reference_genome_versions.csv` and checked against the fasta's length in `paper_numbers_D.py` [D20]. The project's notes give no download year, so none is stated. The Results bullets keep their numbers: they are the findings.


## 31. The flank windows tuned upstream and downstream separately, one protocol for every Δρ, and the models merged into the comparison (24 Nov; the user: "could you tune the window size, both up- and down-stream, some more? ... for completeness"; "i thought we agreed on one protocol for Δρ?"; "the section on features and models can be merged with ... slicer vs crisprhal2 ... 1. accuracy ... 2. cost"; "we don't really need all those different metrics")

**Window tuning** (`python -m sgrna.interpret.flank_scale --parts asymmetric`, which replaced the module `flank_window_tune` on 25 Nov; `flank_window_tune.csv`, `flank_window_tune_summary.csv`). Same protocol as the first sweep (`flank_scale.py`): two GC columns, the GC fraction of the U nt before the protospacer and of the D nt after the PAM, added to the 20-mer; five-fold cross-validation with whole genes held out; the default model with nothing searched; seed 41, so every arm is paired by fold with the earlier arms (the symmetric arms and the base of seed 41 are read from `flank_scale.csv`, not rerun). (A) Five symmetric windows were added to the nine of the first sweep (75, 150, 200, 300, 400 nt: 14 in all). (B) Each side was varied alone over 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1,000 nt with the other held at 250 nt. (C) The best cell and its neighbours were refined (3 more cells). That is 33 combinations. (D) The selected cell, 250/250 and the base were rerun on seeds 42 and 43 (new fold assignments of the same genes), because a cell picked as the best of 33 on the folds it is scored on overstates its lead.
- **Symmetric windows:** the highest is now **200 nt (+0.106)**, not 250 nt (+0.105); every window from 150 to 500 nt is within 0.01 of it (100 nt is 0.011 below). So the earlier "250 nt is the best of the nine windows" was true of that grid only; the Doc and Appendix now say "the highest of the 14 windows tried", and no window length is singled out.
- **Separate sides:** the best of 33 is **200 nt upstream, 250 nt downstream**: +0.0021 over 250/250 on the folds it was chosen on (5 of 5 folds) and **+0.0001 on seeds 42 and 43 (5 of 10 folds)**, the gain being 0.106 for both. The 13 combinations with both windows from 150 to 300 nt lie within 0.004 of the best. Changing one side alone to 100, 50, 750 or 1,000 nt costs −0.0031, −0.0107, −0.0097, −0.0114 (upstream) and −0.0028, −0.0121, −0.0056, −0.0085 (downstream); the two sides never differ by more than 0.004 at any window. **Tuning the windows buys nothing that replicates.**
- Figure 1b now plots the 14 symmetric windows for GC (the four-base curve keeps its nine). Audit: 24 checks, `[D24]`; the audit now has 3,287 checks.
- The Abstract's "length scale of 250–500 nucleotides" is not what the finer grid shows (the peak is at 200 and the plateau runs 150 to 500); it was changed to "a few hundred nucleotides".

**One protocol.** The Abstract's Δρ was the sealed-gene +0.156 while the Results quoted the five-fold +0.161. The Abstract now says +0.161 and, at first mention, what the protocol is (five-fold cross-validation that holds out whole genes, so no tested locus was seen in training); the Results label the sealed-gene +0.156 as the optimism check of that figure. Not yet on the single protocol: the Abstract's "Δρ < 0.02" for the guide-derived families (random-row folds of the published matrix and of the curated cohort); a whole-gene rerun of the family ablation (about 25 minutes) would put it there.

**Models merged into the comparison, metrics dropped.** The Methods "Features and models" subsection is gone. The Results section on the crisprHAL 2 comparison now opens with what SLICER and BoostMEC are, then "Accuracy" (a slight edge in ρ: +0.0286 and +0.0226 as released; +0.014 default and +0.031 tuned on whole-gene folds at the published batch) and "Cost" (scoring 18.8 times faster than crisprHAL 2 and 42.3 times faster than BoostMEC at 100 guides on one CPU thread; smaller than both). The other accuracy measures (Pearson r, R², Kendall τ, the metrics of choosing from the top of a list) are in Appendix A2 only. BoostMEC's paper does not claim to be light, so the Doc calls it a boosted-tree model like SLICER. SLICER's default model fits in the same time as BoostMEC (12.8 and 13.2 s), so "lower cost" holds for scoring and storage, not fitting.

**Compute.** The window sweep ran locally (about 40 minutes, ten cores). It was not sent to cloud compute: its input matrix (`data/interim/expanded_X.npy`, 828 MB) is outside the repository, the remaining arms were about nine minutes of work, and the arms must be paired fold by fold with those already run on the same library build.


## 32. The family ablation on the paper's own folds, the protocols compared, and which controls ran on which folds (24 Nov; the user: "rerun the family ablation on whole-gene folds. actually, what are the different protocols we've used ... i don't want to be using a different protocol from the rest of the team"; then "do the appendix bullet, and rerun the ablation on the exact folds. do the various protocols in the table mirror the control experiments performed?")

**What existed.** `curated_ablation.py` had two fold designs: random rows (the team's shared harness: `KFold(5, shuffle)`, seeds 41 to 45, 300 columns chosen inside each training fold, 400 trees; `ablation_curated.csv`) and `--grouping gene` (whole genes, but only the 30,203 genic guides, five seeds; `ablation_curated_gene.csv`). Neither is the fold design of the paper's other numbers (`controls/sealed_holdout_gene._fold_ids` over all 33,567 guides, intergenic guides in 100 kb blocks, seeds 41 to 43). The Abstract's "Δρ < 0.02" and the first Results bullet (thermodynamics +0.0991, 98%) rested on random rows.

**New: `python -m sgrna.cohorts.curated_ablation --paper-folds`** (written as the module `ablation_paper_folds`, merged into `curated_ablation` as the design `paper_folds` on 25 Nov; the merged module resumes from the stored rows and reproduces the summary byte for byte) (`ablation_paper_folds.csv`, `ablation_paper_folds_summary.csv`; about 55 s per arm and seed, 36 arms). Each family block added to the 6,169 base columns, 300 columns selected inside each training split, on exactly the folds of `flank_scale.csv`; the aligned family A arm and the 20-mer with no selection are read from `flank_scale.csv`. Gains are reported against the like-for-like baseline (the 20-mer with the same 300-column selection) and against the paper's (no selection).
- **Guide-derived families (paper folds, same selection):** energy +0.0181, folding +0.0143, methylation +0.0077, periodicity +0.0081 (each positive in 15 of 15 folds), off-target −0.0003 (7 of 15). None reaches 0.02, so the Abstract's claim now stands on the single protocol. On random rows: +0.0187, +0.0144, +0.0078, +0.0108, +0.0011.
- **Thermodynamics (`d_mechanics`)** +0.0972; split by `.flank.` in the column name: the 22 guide columns +0.0103, the 8 flank columns +0.0963, **99% of the whole** (98% on random rows). The first Results bullet now quotes these.
- **Fold design barely matters for the headline and for most families** (same model, same cohort): the paper's folds keep 98% of the random-row flank gain (+0.1612 against +0.1638), energy 97%, folding 99%, methylation 99%, thermodynamics, shape and supercoiling 98%. Only the families that track the locus lose more: nucleoid 90%, transcription 78%, periodicity 75%. (The earlier "retained %" table in `grouped_retention.csv` compares random rows on 33,567 guides with whole genes on 30,203, so its levels are not on one cohort.)
- **A correction found on the way.** Appendix A3 said the 20-mer "scores 0.559 and not 0.547" under the same selection, which reads as an effect of the selection. On the paper's folds the selected 20-mer scores 0.5461 and the unselected 0.5466: selection changes nothing. The 0.559 is the 30,203-guide genic subset (an easier set), where `family_a_at_tuned*.csv` was run. The bullet and the audit label of check [51] now say so. The same-selection flank gain on the paper's folds is +0.1612 (+0.1608 against no selection).

**The protocols and the controls (the user's question).** Protocols: (1) team harness, random rows; (2a) whole genes on the 30,203 genic guides, five seeds; (2b) the paper's folds; (3) sealed genes, 15 draws; (4) DEV/CONFIRM genes for the tuned settings; (5) crisprHAL 2's own train/test files, models as released; (6) external genomes and transfer. Checked in code (fold construction in each module): on the paper's folds, `flank_scale` (displacement, windows, resolution), `flank_window_tune`, `mappability_stratum` (read depth, mappability), `gate_test`, `boostmec_replica`; sealed genes, `sealed_holdout_gene` and `sealed_crossscreen`; random rows, `no_information` (the thermodynamics split, now also rerun on the paper's folds), `window_robustness`, `curated_joint`, `redescription` (the random position-weight and random-function nulls: `KFold`, seeds 41 to 43); 100 kb blocks, `shiftnull` (the spatial null, a design missing from the first table); whole genes on genic guides only, `curated_ablation --grouping gene`, `shape_ablate`, `family_a_at_tuned`. So the controls that defend the flank claim mirror the headline protocol; the claim that guide-derived families add little, and its random nulls, sat on random rows. The family gains and the thermodynamics split are now on the paper's folds; the random nulls stay on random rows and are labelled so in Appendix A1.

**Doc changes.** First Results bullet (the guide-derived numbers on the paper's folds, thermodynamics +0.0972 and 99%); Appendix A1 (same numbers, nulls labelled random-row); Appendix A3 (the same-selection bullet corrected; a new last bullet, Fold design, with the random-row, genic and paper-fold numbers side by side); Abstract (the fold-design clause moved to the first Δρ, so the justification comes at first mention). Applied in place with tab-scoped `replaceAllText` and one `insertText`; both tabs re-read and equal to the local text. Audit: 35 checks `[D25]`; the audit now has 3,322 checks.


## 33. Other base-pair classes, the paper's Results rewritten as numbers, the cost comparison made fair, the report brought into line, and code consolidation started (25 Nov; the user's long list of 24-25 Nov)

**Other pairs of bases (Figure 1b).** The four-base curve added nothing to GC, so the user asked about the other ways to split four bases into two classes. `flank_scale.py --parts partition` runs the two columns of the purine fraction (A+G against C+T) and of the amino fraction (A+C against G+T) at the same 14 windows as GC (seed 41, five folds): **neither gains anything at any window** (−0.0014 to 0.0000; positive in at most 2 of 5 folds), and the four bases separately (eight columns, which contain C and G) peak at +0.103 at 250 nt against GC's +0.106 at 200 nt. GC is the only class that works, so the main text keeps GC alone (Figure 1b is the GC curve and the dashed full family); the alternatives, the purine partial correlations and the resolution of the composition are Appendix Figure A1 (a: gain by class; b: partial ρ for GC and purine, upstream and downstream, in one panel; c: ρ against the number of equal-width columns, which are the GC and purine fraction of bins that tile the kilobase on each side) and Appendix A1. Figure 2 lost its panel c (the resolution) to A1 for the same reason. Old Figure A1d (the largest pairwise interaction shares) was deleted: no search over all interactions was ever claimed, so showing the top of one was misleading. Window tuning (upstream and downstream separately) is no longer mentioned in the paper, since Figure 1b shows the windows; the stored results and the code stay (`flank_scale.py --parts asymmetric`).

**Highest ρ achieved, and under what conditions** (stored files). A single model on held-out data: 0.8107, tuned SLICER on crisprHAL 2's eSpCas9 test split (11,898 test guides, 47,591 training guides, random rows as released; averaging with crisprHAL 2 at weight 0.25 gives 0.8140, a weight chosen on the test set). On the E. coli SpCas9 cohort: 0.7314 (tuned, whole-gene folds, 30,203 genic guides, five folds); 0.7215 on the 1,204 genes the search never touched; 0.745 to 0.749 on a second 70/30 gene split's CONFIRM genes (easier genes; three tuned models averaged 0.7488); 0.7475 with all eleven families jointly on random rows (the team harness; includes locus-tracking families). The 0.827 in `blend_with_hal.csv` is the correlation of the two models' residuals, not a ρ.

**Tuning effort: compute, not draws** (`tuning.csv`, `crisprhal_matched_folds.csv`, `crisprhal_tuning.csv`, `cost_gpu.csv`; audit `[D26]`). SLICER's original version: 627 draws in 2.4 hours on the laptop. crisprHAL 2: 28 draws (33 fits with the five final refits) at 326 s per fit on a T4 = about 3.0 GPU-hours, after an earlier 10-draw search of 7.3 CPU-hours; its published configuration won four of five folds of its own search. The same 627 draws would cost it about 57 GPU-hours. So the fair statement is comparable compute, not equal draws; the paper's Discussion says so.

**The GPU comparison was not like for like, and is rewritten.** The earlier "1.05 times SLICER's whole path" compared crisprHAL 2's forward pass with the batch already on the card (no input encoding) on a Colab T4 with SLICER's whole path on the laptop: two machines and two different paths. The T4 session is one VM that is not quiet (repeat variation 2.7% for the forward pass on the card but 30% for the VM's CPU timings of the trees). The paper now gives the two comparisons that exist: crisprHAL 2 from raw contexts on the T4 is 3.9 times SLICER's laptop path (two machines, said so), and its forward pass is 13 times the forward pass of SLICER's trees on the same VM. No whole-path SLICER timing exists on the VM; measuring it needs the featuriser inside `colab/cost_gpu.py`. Figure 4 is now bars (median, 95% bootstrap interval over repeats, log scales); panel b shows the three same-session quantities. The Abstract's "25 times faster on a GPU" compares SLICER on the laptop with crisprHAL 2 on the T4; on the same T4 SLICER's shape fits in 3.2 s against 326 s (about 100 times).

**Figure 3b** is a bar graph of ρ (mean ± sd over the five whole-gene folds of seed 41, the folds crisprHAL 2 was run on) for crisprHAL 2, BoostMEC, SLICER default and tuned, with two dotted lines: the field's ceiling (0.7) and the attainable ceiling (0.946). **Results are numbers; interpretation is in the Discussion**: every interpretive clause was moved out of the Results bullets (controls, the step, reach, the ceiling, prior work), the Discussion has nine bullets, and the main tab estimates at 7.6 pages. **BoostMEC** is a cost reference only in the main text (one clause in the cost bullet, its accuracy in one bullet and in Figure 3b; the retraining and shipped-pipeline detail are in Appendix A6). The batch-256 handicap run and the "published batch" parentheticals are gone; the input-width experiment (Figure A2) is stated to have used batch 256, because it did.

**Figure A2 into Figure 1b: not merged.** crisprHAL 2's ρ against input width exists only on random rows, batch 256, one seed, with SLICER's original LightGBM version; Figure 1b is the default model on whole genes. One axis would mix three protocols. A merge needs crisprHAL 2 at 23, 378, 1,023 and 2,023 nt on the paper's folds (a Colab T4 job; the batch would then be whatever fits).

**The SHAP reading of the 104-column model** (`compact_model_*.csv`, audit `[D26]`): 83% of the attribution is on the 80 guide indicators and 17% on the 24 flank columns (79% of it GC); positions 15 to 20 hold 61% of the guide attribution and position 20 alone 20%; G at 20 and 18 lower the score by 0.33 and 0.28 SD and T at 20 raises it by 0.36 SD (the 80 SHAP-implied base effects agree with the model-free map at ρ 0.85); the lowest decile of flank GC contributes −0.78 SD, 2.4 times the largest single-base effect, against −0.04 to +0.17 SD for the other nine deciles. It is in the paper's Results (one bullet), Discussion and Appendix A2.

**The report against the paper.** A read-only audit (14 differences of number or conclusion, 24 stale statements, 18 results the report never stated, 8 protocol-label differences, 9 imprecisions in the paper) found the report was not in step. Done on 25 Nov: the Summary rewritten in the paper's order with one protocol key; the mappability objection, the one-length-scale reading, "statistically indistinguishable", the 150 times CPU claim, the 13 to 34 times scoring claim, the GPU 1.05 times claim, the leaf count, the ceiling gap and the search-effort numbers corrected; README and PAPER_FLOW rewritten; and Appendix E added, which repeats the paper's bullets in the paper's own wording so that every number the paper states is in the report with its protocol label. The nine imprecisions in the paper were fixed (cross-references, "its authors' folds" that were the team harness, the located-guide counts, the tuned settings of the fine-structure experiment, the released BoostMEC model, the dead-Cas9 control name, the two meanings of "lowest decile", the batch of the width runs). Not done: rewriting Findings 2, 3 and 7 and Part 2 in place around the paper's order (they are patched where they contradicted Appendix E, not restructured), and the Finding-level re-basing of the Appendix B3 retention table (the paper's folds are in `ablation_paper_folds_summary.csv` and Appendix A3).

**Code consolidation** (analysis: `scratchpad/code_analysis.md` of this session, summarised here). 174 modules in 11 folders; 26 clusters of 72 modules that run the same pipeline with a different arm table, fold scheme, split seed or model, and 102 modules that each ask a different question. What makes them duplicates is the call pattern (23 modules call `run_ablation.cross_validate`, 56 import `curated_ablation`, 7 call `_fold_ids`) more than copied text. The plan is one module per engine with one `--design` per question, and every result file keeps its name and its fold-checkpoint tags (a merged module is correct when the same command reproduces the stored csv; `cross_validate` resumes from `folds_{tag}.csv`). Clusters: (1) the family ablations `curated_ablation`, `curated_joint`, `txn_interactions`, `no_information`, `window_robustness`, `why_inflation`, `ablation_paper_folds` into `curated_ablation --design` (**`ablation_paper_folds` done**, as `--paper-folds`; the other five remain); (2) `family_a_at_tuned`, `local_sequence_bound`, `novel_followups.resolution`; (3) `xgb_knobs`, `curated_models`, `replicate_weight`, `data_ablation`; **(4) `flank_scale` and `flank_window_tune` — done**; (5) `shape_ablate`, `shape_analyze`, `ridge_check`; (6) `representation`, `curated_ladder`; (7) `portability`, `handnamed_split`; (8) `sealed_holdout`, `sealed_holdout_gene`, `sealed_crossscreen`; (9) `tune_ecoli`, `tune_split2`; (10) `topend`, `lit_followups`; (11) `importance`, `importance_models`; (12) `build_features`, `build_curated_features`; (13) seven comparisons with crisprHAL 2 (`vs_crisprhal`, `panel_vs_hal`, `topk_panel`, `metrics_panel`, `blend_with_hal`, `clean_vs_hal`, `gene_grouped_vs_crisprhal`); (14) `panel`, `heldout_compare`; (15) `sample_complexity`, `data_efficiency`; (16) `crisprhal_rerun`, `crisprhal_tune`; (17) `cui_screen`, `crispri_screens`; (18) five strand modules; (19) `human_additivity_twin`, `human_twin_check`; (20) `cost`, `bench`; (21) `cost_bench`, `e2e_bench`, `inference_cost`; (22) `geom_importance`, `redundancy`; (23) `curated_ceiling`, `ceiling_table`; (24) `eukaryote_rules`, `transfer_diagonal`, `transfer_to_ecoli`; (25) `model_size`, `complexity`; (26) `cost_boostmec`, `cost_boostmec_shipped`. Thirteen helper families are copied between modules (the eSpCas9 split rebuild five times, ranking metrics seven, z-score six, bootstraps eight). Nothing is safe to delete outright: nine modules write files nothing reads, but six are cited in documents. Layout proposal: ten packages as now (the audit asserts ten), `cohorts` renamed `ablations`, and `compare` (43 modules) split into five nested sub-packages. Done on 25 Nov: cluster 4 and the paper-folds part of cluster 1 (verified: each merged module reproduces its stored summary byte for byte), the generator defect that listed `cost_boostmec_shipped` as the writer of nearly every result file (a Docker `--format '{{.Id}}'` literal gave an empty prefix), and a safety tag `pre-merge-2026-11-25`. Remaining: the other 25 clusters, in the order the analysis gives (merge with the stored results as oracle, then the package moves), run one cluster at a time with `import_check`, `doc_refs` and the audit after each.

## 34. The paper cut to seven pages, the appendix de-duplicated, and every protocol folded into the appendix on controls (25 Nov; the user: "the appendix seems quite long, check through it again to make sure you aren't repeating content in both places ... report itself should be 7 pages max, not 9 ... the various protocols [can] be folded into the appendix on controls")

**Page limit.** The paper is 7 pages at most (it was 9), references not counted. The introduction is another tab and is taken as one page, so the "Paper (point form)" tab is cut to about 6 pages (page estimate 7.59 to 5.93 without references; `pages2.py` of the scratchpad lays the Doc's own fonts and spacing out; the true count is the Doc's).

**Protocols.** Methods keep two data bullets and one protocol sentence (whole-gene five-fold cross-validation, three seeds, default model, 20-mer baseline, mean ± sd over 15 folds, 95% bootstrap intervals, Spearman ρ); the other protocols and the controls run on each are in Appendix A3, now "Controls, robustness checks and protocols": (1) the paper's folds, (2) the team's random-row harness (seeds 41 to 45), (3) whole genes of the genic guides only, (4) sealed genes (20%, 15 draws), (5) development and confirmation genes (70/30), (6) crisprHAL 2's own splits (search overlap, 57% of eSpCas9 test guides are cohort guides, interval over test guides only), (7) an unseen genome (C. rodentium); the metric and interval rationale that sat in A4 "Metrics" moved there. A4 lost its Sealing and Metrics bullets, A3 lost its Fold design bullet (the percentages it held are in "which control ran on which folds").

**Repetition removed.** Main text and appendix no longer state the same number twice: the SHAP bullet of A2 keeps only what Results do not (positions 15 to 20, the 88% and 83/17/79 are in Results); the composition-class bullet of A1 keeps only what the Results bullet does not; the strata of the flank gain moved from Results into the one "Strata" bullet of A3 (Discussion states the conclusion); the "settings searched with the flank features" bullet moved to A3 (Discussion keeps +0.0028 with a pointer); the sealed-drop bullet (0.0087) and C. rodentium +0.0172 are in A3 only; the melting-limited bullet of A1 was deleted; BoostMEC arm 1 in A6 no longer repeats the Results numbers; cross-references repaired (metrics now point to A3). Results bullets for the GC correlation and the step were merged; Figure 1 to 4 captions shortened.

**Appendix A3 sealing bullet** now carries the pre-registered eSpCas9 prediction (at least +0.08, positive in at least 14 of 15 sealings; observed +0.103, positive in 15 of 15).

`PROGRESS_REPORT.md` Appendix E is regenerated from the Doc text (`mk_appendix_e.py` of the scratchpad), `PAPER_FLOW.md` updated. Nothing in the audit changed: 3359 checks, the new text's numbers are all matched to an audit claim or are settings, seeds or published constants.
