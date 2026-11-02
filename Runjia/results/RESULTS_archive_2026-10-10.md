# Results and interpretation

Everything measured, and what it means. Updated 2026-10-02; supersedes the
separate `INTERPRETATION.md` and the former `superseded/` directory of early
ablation runs, both since removed (the superseded runs were notebook-era
smoke tests and an `arc5` fold file, all re-measured under the settings
recorded here). Two companion documents:
[`docs/PROGRESS_REPORT.md`](../docs/PROGRESS_REPORT.md) explains the whole
project in plain language, with a glossary, for a reader without a biology or
machine-learning background — it also places these numbers against the
published state of the art and separates what is original from what is
imported, having absorbed the former `NOVELTY.md` on 2026-10-02.
`docs/FINDINGS_LOG.md` is the chronological record, including which earlier
claims were later overturned; it is kept in the working folder and not
published here, since git history is the authority for what was believed when.

Protocol unless stated otherwise: seeds 41–45, 5 folds, top-300 features by
XGBoost gain, 400-tree champion, with imputation, ranking and selection all
fitted **inside the training split**.

**Contents**

1. [What the numbers buy you](#1-what-the-numbers-buy-you)
2. [Feature family ablations](#2-feature-family-ablations)
3. [The supercoiling result is not supercoiling](#3-the-supercoiling-result-is-not-supercoiling)
4. [Three positional families, one variable](#4-three-positional-families-one-variable)
5. [The thermodynamic families are GC content](#5-the-thermodynamic-families-are-gc-content)
6. [Family A: two findings](#6-family-a-two-findings)
7. [The two small families](#7-the-two-small-families)
8. [Cross-validation strictness](#8-cross-validation-strictness)
9. [Improvement levers, measured](#9-improvement-levers-measured)
10. [Lifting the 13,880-row cap](#10-lifting-the-13880-row-cap)
11. [**How to quantify predictive power and interpretability**](#11-how-to-quantify-predictive-power-and-interpretability)
12. [**Model class: what changes**](#12-model-class-what-changes)
13. [**Why the new libraries didn't move the needle**](#13-why-the-new-libraries-didnt-move-the-needle)
14. [**Head-to-head with crisprHAL 2, on identical rows**](#14-head-to-head-with-crisprhal-2-on-identical-rows)
15. [**Does the flank effect transfer?**](#15-does-the-flank-effect-transfer-one-clean-answer-one-dataset-that-cannot-answer)
16. [What to do next](#16-what-to-do-next)

---

## 1. What the numbers buy you

R² is the wrong unit for this project, because nobody predicts a cut score —
they pick a guide. Given ten candidates and taking the model's top pick, where
does that guide land in the true efficiency distribution? Measured empirically
on held-out folds (§11), not simulated:

| model | Spearman | guide you end up picking |
|---|---:|---|
| pick at random | — | 50th percentile |
| published baseline | 0.527 | ~70th |
| + family A | 0.609 | 72nd |
| **all nine families, LightGBM** | **0.641** | **73rd** |
| **base + family A on curated rows (§14)** | **0.707** | **76th** |
| crisprHAL 2, published, same curated rows | 0.697 | — |
| the reproducibility ceiling (§13) | ≈0.94 (0.94–0.97) | ~89th |
| an oracle | 1.000 | 91st |

**The gains are real but modest in practice.** Do not imply otherwise: +0.114
Spearman over the published baseline moves the guide you pick from the 70th
percentile to the 73rd. The interesting output of this project is mechanistic,
not predictive — which is fortunate, because that is where the results are
strong.

Two things to read off this. The middle rows are **parity with the published
state of the art on identical rows** (§14) — reached with a model that fits in
about 3 minutes of CPU. And the ceiling is still far away: §13 shows the remaining gap
is not in the label, the model or the guide's representation, but in features
that do not exist yet.

The permutation floor is **per family**, not global (§17). Quoting one floor
measured on `a_flank` for all of them was wrong: the floor depends on how many
columns a family contributes and how easily they survive selection. Measured
properly, `h_offtarget` sits *below* its own floor and `f_methylation` only 3×
above it.

## 2. Feature family ablations

Three tags, three cross-validation schemes. **Rows are only comparable within
a tag.** `main` holds out random rows, `grouped` whole 100 kb chromosome
blocks, `arc` one of five contiguous chromosome arcs.

**`main`** — baseline R² 0.2937 ± 0.0017, ρ 0.5273 ± 0.0013. (The frozen
reference is 0.2937 / 0.5278, reproduced to four decimals — the check that the
harness is faithful.)

| family | new cols | ΔR² | Δρ | in top 300 |
|---|---:|---:|---:|---:|
| **A** `a_flank` | 348 | **+0.0867** | **+0.0795** | 73.5 |
| A permuted control | 348 | −0.0018 | −0.0009 | 19.8 |
| **D1** `d_mechanics` | 30 | +0.0203 | +0.0201 | 10.7 |
| **C** `c_folding` | 9 | +0.0073 | +0.0074 | 3.5 |
| **B** `b_energy` | 10 | +0.0071 | +0.0074 | 7.2 |
| **F** `f_methylation` | 17 | +0.0035 | +0.0034 | 6.2 |
| **H** `h_offtarget` | 13 | +0.0002 | +0.0003 | 0.3 |
| **I** `i_shape` | 152 | +0.0377 | +0.0356 | 36.9 |
| I permuted control | 152 | −0.0021 | −0.0022 | 11.0 |

**19.8 of the *permuted* family A's columns still reached the top 300** —
selection counts say nothing about whether a family works.

**Family I's +0.0377 is family A's effect by another route — read the two
together.** I describes the same flanking DNA as A, but by its physical
deformability (dinucleotide bending and stiffness scales) rather than its
letter composition. Added *on top of* A it is worth **+0.0007 R²** against a
±0.0032 seed spread (`ablation_shape_joint.csv`: A alone 0.3805, A + I 0.3811),
and its best column correlates with the champion's residuals at 0.028, below
the 0.035 dust level (§13). A gets more than twice as much from the same
information, because A is not a letter count — it is position-resolved
composition over four window sizes, which already spans what a 2 nt-step
average can express.

This is the counter-example to `d_mechanics`, where a physical
reparameterisation of information already present *did* win (+0.0203 over raw
GC). The "better-shaped variable" argument has now been tested twice and gone
both ways; quoting only the success would overstate it.

**`grouped`** — baseline R² 0.2920 ± 0.0044, ρ 0.5283.

| family | ΔR² | Δρ | permuted ΔR² |
|---|---:|---:|---:|
| **D2** `d_supercoiling` | **+0.0475** | +0.0445 | +0.0014 |
| **E** `e_nucleoid` | +0.0174 | +0.0155 | +0.0006 |
| **G** `g_transcription` | +0.0159 | +0.0146 | −0.0000 |

**`arc`** — baseline R² 0.2730 ± 0.0011, ρ 0.5348. The published feature set is
itself ~7% worse on an unseen arc than on unseen random guides, which belongs
in the methods.

| model | ΔR² | Δρ |
|---|---:|---:|
| `a_flank` | **+0.0867** | +0.0756 |
| `d_supercoiling` | +0.0455 | +0.0293 |
| **A + D2 + D1 together** | **+0.1050** | +0.0799 |

Two things. **Family A gives exactly +0.0867 under both random-row and
contiguous-arc CV** — a sequence family should be indifferent to how the
chromosome is split, and it is. And **the combined model is 70% additive in R²
but barely additive in rank**: adding the positional and thermodynamic blocks
to family A moves Spearman by +0.004. Chromosomal position improves the
*calibration* of the prediction, not its *ranking* — and ranking is what guide
selection uses.

## 3. The supercoiling result is not supercoiling

`d_supercoiling` gives +0.0475 R² with a permutation control of +0.0014. Then
look inside:

| feature | ρ with cut score | ρ with distance from *oriC* |
|---|---:|---:|
| GapR signal, 10 kb window | +0.116 | **−0.834** |
| untagged control, 10 kb window | +0.114 | **−0.898** |
| + rifampicin, 10 kb window | +0.114 | **−0.968** |
| log2(signal / control) — the actual supercoiling measure | +0.071 | −0.280 |

The **untagged no-antibody control predicts cutting as well as the GapR ChIP
does**. Six follow-up ablations (`dissection_supercoiling.csv`; 3 seeds,
100 kb-grouped, baseline 0.2924):

| given to the model | cols | ΔR² |
|---|---:|---:|
| the whole family | 26 | +0.0465 |
| **raw read density only, no ratios** | 16 | **+0.0453** |
| signal track alone | 4 | +0.0110 |
| untagged control alone | 4 | +0.0091 |
| **ratios only — dosage divided out** | 10 | **+0.0040** |
| analytic *oriC*-distance proxy | 4 | +0.0127 |

**97% of the effect is raw sequencing read density.** Do not write "DNA
supercoiling predicts sgRNA efficiency in *E. coli*". Write that **a measured
profile of local chromosomal DNA abundance does** — a different, still
unreported claim. Columns are now named `eng.sc.abund.*` / `eng.sc.torsion.*`.

**What is it, then?** Answered on 2026-10-02, and the answer retires most of
this section's own headline. Three steps, all `arc5`, 3 seeds × 5 folds
(`ablation_abund_split.csv`, `ablation_abund_cause.csv`):

| block | cols | ΔR² | Δρ |
|---|---:|---:|---:|
| `z_abund_smooth` — mean density of the guide's 100 kb bin | 16 | +0.0134 | **−0.0017** |
| `z_abund_local` — the within-bin residual | 16 | **+0.0356** | +0.0326 |
| **`z_mapgc` — GC + 25-mer uniqueness, reference genome only** | **8** | **+0.0377** | +0.0299 |

1. **Not replication timing.** Split non-parametrically, the smooth positional
   component gives *no rank improvement* (−0.0017 ρ) and its ΔR² reproduces the
   analytic `z_dosage` proxy to three decimals. The signal is local.
2. **It is GC bias and mappability.** Local density tracks window-matched GC at
   r = −0.70 (rifampicin, 500 bp) and −0.44 (ChIP signal, 2 kb), and the
   untagged control's 10 kb density tracks 25-mer uniqueness at −0.69
   (`abund_local_vs_mappability.csv`; 97.35% of the genome has unique 25-mers).
   **Eight genome-computed columns reproduce the whole 16-column measured
   gain** — so this effect needs no GapR-seq download, and transfers to any
   organism with a reference genome.
3. **Against family A it nearly vanishes.** Paired across the same 15 folds:
   `a_flank` + `z_mapgc` is **−0.0002 ρ** (8/15 folds, p = 0.84), while
   `a_flank` + `z_abund_local` is **+0.0061 ρ** (15/15 folds, p = 2.5e-05).

**The elimination argument above was wrong, and the way it was wrong is worth
recording.** GC was dismissed because *GC correlates only +0.03 to +0.06 with
the cut score* — true, but the wrong test. GC's correlation with the **read
density track** is an order of magnitude larger, and a feature can route a
signal into a model without correlating with the label itself. Test a candidate
cause against the *mediator*, not only against the outcome.

So: replication timing (no rank gain) + GC bias and mappability
(genome-computable, and already inside family A) + **≈+0.007 R² that is
genuinely new and not reproducible from sequence**. Write that the measured
abundance profile is *mostly a sequencing artifact that happens to encode flank
composition*, with a small real residual — not that chromosome state predicts
cutting.

## 4. Three positional families, one variable

|  | supercoiling signal | Hi-C contacts | HU dependence | *oriC* distance |
|---|---:|---:|---:|---:|
| supercoiling signal | 1.00 | 0.80 | −0.77 | −0.83 |
| Hi-C total contacts | | 1.00 | −0.93 | −0.96 |
| HU dependence | | | 1.00 | 0.94 |
| *oriC* distance | | | | 1.00 |

Hi-C contacts and ChIP read density are both **read counts**, so both inherit
the copy-number gradient. Stacked, the three give +0.051 where their individual
gains sum to +0.081 — two thirds is double-counting. **Report them as one
result.** A referee who computes that matrix will notice.

## 5. The thermodynamic families are GC content

| feature | ρ with cut score | ρ with guide GC |
|---|---:|---:|
| protospacer ΔG37 | +0.156 | **−0.961** |
| protospacer Tm | −0.157 | **+0.959** |
| CRISPRoff ΔG hybridisation | −0.195 | **+0.909** |
| … position-weighted | −0.213 | **+0.898** |
| spacer self-folding MFE | +0.098 | −0.463 |
| guide GC content itself | **−0.201** | 1.000 |

`d_mechanics` and `b_energy` are 0.90–0.96 correlated with a column the matrix
always had. Their gains are a better-*shaped* GC variable: `d_mechanics` wins
(+0.020 vs +0.007) because it says *where* the duplex is weak.

> **GC-rich protospacers cut worse** (ρ −0.20), and the decomposition says why:
> a target duplex that is harder to melt is harder to invade. A *more* stable
> RNA:DNA hybrid goes with *lower* activity (ρ −0.21) — **strand-invasion
> limited**, not hybridisation limited.

`c_folding` is the cleanest of the group (only −0.46 correlated with GC) and
independently replicates both of Moreb & Lynch's (2022) thresholds: 4.0% of
guides below −5 kcal/mol spacer MFE and 8.2% below −15 kcal/mol spacer:scaffold
duplex stability, both less active, in a dataset they never used.

**A published human design rule that does not transfer.** After the ΔG window
fix, `hyb_in_window` correlates **−0.133** with cut score: guides inside
Alkan et al.'s −64.5 to −47.1 kcal/mol optimum are, if anything, *less* active
in *E. coli*.

## 6. Family A: two findings

+0.0867 R², identical under random-row and arc CV, permutation floor −0.0018.
Where the gain sits:

| block | share |
|---|---:|
| single positions downstream of the PAM (d1–d15) | 29% |
| the PAM's variable N base | 27% |
| window summaries at 500 nt | 12% |
| single positions upstream (u1–u10) | 11% |
| window summaries at 250 nt | 10% |
| window summaries at 1000 nt | 9% |

**(a) A nucleotide penalty right after the PAM.** `seq.d1.G` — a guanine at the
first position 3′ of the NGG — is the strongest single column (ρ −0.108).
Alkan et al. (2022) reported exactly this in human data at ~12.6% efficiency
cost. Gain decays d1 > d2 ≈ d3 and fades by d8, matching crisprHAL's 5-nt
downstream window.

**(b) The neighbourhood wants the opposite, and it saturates.**

| window | ρ (upstream GC) | ρ (downstream GC) |
|---|---:|---:|
| 50 nt | +0.047 | +0.012 |
| **250 nt** | **+0.051** | +0.039 |
| 500 nt | +0.046 | +0.034 |
| 1000 nt | +0.032 | +0.022 |

Against guide GC at **−0.20**. Opposite signs; the flank effect peaks at
250–500 nt and decays by 1 kb. **This question is closed — do not widen
further.**

Family A is near-orthogonal to everything positional (|ρ| ≤ 0.045), which is
why the combined model gains in R². **For choosing guides, sequence is nearly
the whole story.**

### Where in the flank the gain lives

Splitting family A by side and by distance (`flank_sides.csv`, LightGBM, seed
41, 5 folds — arms comparable within this run only):

| what is added to the baseline | Δρ |
|---|---:|
| upstream side only | +0.019 |
| downstream side only | +0.065 |
| ±10 nt of immediate context (`seq.*`, `qct.*`) | +0.049 |
| windowed composition at 50–1000 nt (`comp.*`, `win.*`) | +0.026 |
| **everything** | **+0.075** |

Two clean decompositions, both close to additive. **Downstream beats upstream
3.4 to 1**, which agrees with crisprHAL's report that downstream context helps
out to +8 nt past the PAM — except that here upstream is *independently
positive*, where crisprHAL 1 found adding upstream sequence "immediately
decreased predictive ability" in a CNN.

And **two thirds of the gain is in the ±10 nt that published bacterial models
already read; one third is in the 50–1000 nt windows.** Those long windows are
the part that is not in the current tools, and §13 shows why a CNN cannot pick
them up: the signal there is a smooth compositional gradient, not a motif.

### Is the downstream advantage a transcription effect? No — and half the test is impossible

`src/sgrna/asymmetry.py`. Two tests follow from the transcription account; one of
them the screen cannot support.

**The orientation test cannot be run** (`asymmetry_strand_audit.csv`). Checked
two independent ways — the guide index joined to `genes_NC_000913.2.tsv`, and
`g_transcription`'s separately-derived `template_strand` column — the library
targets the gene's template strand for **13,825 of 13,879 located guides, 4
exceptions**. There is no variation to condition on. A side effect worth
recording: `eng.txn.gene.template_strand` is therefore a near-constant column,
one of several reasons that family managed only +0.015.

**The expression test can be, and the asymmetry is flat**
(`asymmetry_expression.csv`; guides split into terciles by their gene's
`median_log_tpm`, seed 41, 5 folds, ~4,600 guides per band):

| band | baseline | Δρ upstream | Δρ downstream | Δρ downstream, count-matched | asymmetry | matched |
|---|---:|---:|---:|---:|---:|---:|
| low | 0.4733 | +0.0296 | +0.0782 | +0.0790 | 0.0486 | 0.0494 |
| mid | 0.4802 | +0.0177 | +0.0635 | +0.0604 | 0.0458 | 0.0427 |
| high | 0.4836 | +0.0177 | +0.0656 | +0.0539 | 0.0479 | 0.0362 |

The transcription account predicts the asymmetry grows with expression. It does
not move (0.0486 / 0.0458 / 0.0479), and the count-matched version declines
slightly. One seed per band, so this rules out a gradient rather than proving
independence — but a predicted gradient that is absent is evidence against.

**The count-matched arm is a control the original sides split lacked.** Family A
is not symmetric: 194 downstream columns against 144 upstream, because the
windows and the PAM-side k-mers are not mirror images, so up to a third of the
downstream advantage could have been column budget. Restricting downstream to a
random 144 of its 194 leaves the gain essentially unchanged (+0.0790 against
+0.0782 in the low band). **It is the side, not the budget.**

## 7. The two small families

**`f_methylation` (+0.0035)** sits at the noise floor but is **not a GC proxy**
(ρ with guide GC +0.11), and the direction is right: a Dcm `CCWGG` overlapping
the PAM associates with lower activity (ρ −0.124), any methylation motif in the
seed −0.127. Test it with a **GC-matched subgroup comparison**, not a bigger
feature set.

**`h_offtarget` (+0.0002)** is a true null and informative: the median guide is
5 mismatches from the nearest other NGG site and essentially nothing is
multi-copy, so the "multiple cuts inflate lethality" confound **does not apply
to this dataset**.

## 8. Cross-validation strictness

The full ladder for the read-density block (3 seeds):

| held out per fold | baseline R² | + read density | Δ |
|---|---:|---:|---:|
| random rows | 0.2947 | 0.3501¹ | +0.055 |
| whole genes | 0.2916 | 0.3454¹ | +0.054 |
| 100 kb blocks | 0.2920 | 0.3395¹ | +0.048 |
| 500 kb blocks | 0.2866 | 0.3358 | +0.049 |
| **five contiguous arcs** | 0.2730 | 0.3185¹ | **+0.046** |

¹ full `d_supercoiling` family; others are the 16-column read-density block.

Flat gain while the baseline falls — the signature of a covariate that
transfers. Use `group="bin100k"` routinely and `group="arc5"` for the
manuscript.

## 9. Improvement levers, measured

| lever | expected | measured |
|---|---|---:|
| curate the label | "+0.03 Spearman" | **+0.003 to the baseline** — but see below |
| more training rows | "large" | **+0.010**, saturating at ~20k (confirmed §14) |
| raise the top-300 cap | "the selector is under pressure" | **+0.001** |
| widen family A's flanks | "no sign of saturating" | **saturates at 250–500 nt** |
| *sequence context outside the 20-mer* | — | **+0.080**, and **+0.164** on a curated label (§14) |

**Label curation — no effect on the baseline, and a large effect on family A.**
crisprHAL's read-count-filtered scores cover 8,617 of our guides and correlate
+0.945 with `cut.score`. On the same rows:

| label | baseline ρ | + family A ρ |
|---|---:|---:|
| published `cut.score` | 0.5387 | 0.6279 |
| crisprHAL curated | 0.5370 | 0.6305 |

Read alone, that table says curation buys +0.003 and is noise. **That reading
was too quick, and §14 corrects it.** Run at full scale — all 33,567 curated
rows rather than the 8,617 overlap — family A is worth **+0.164** against
+0.080 on the published scores. Curation does not make the model better
directly; it removes noise that was **masking a real biological signal**, and
only a feature family large enough to be masked shows the difference.

So the honest version of this lever: curating the label is worth nothing if you
change nothing else, and worth twice the flank effect if you are trying to see
the flank effect. The *level* still matters too — ρ 0.628 on the curated 8,617
against 0.607 on all 13,880 — so dropping bad rows beats relabelling good ones.

**Feature cap — not binding.** ρ 0.6068 / 0.6079 / 0.6082 at 300 / 600 / 1200.

**More rows — saturates.** Fixed test set, growing training subsets, 2 seeds
(`learning_curve.csv`): 4,000 → ρ 0.488; 8,000 → 0.520; 13,880 → 0.535;
20,000 → 0.544; 26,000 → **0.545**.

## 10. Lifting the 13,880-row cap

5,853 of the 5,887 opaque `V####` columns were identified as (position, k-mer)
one-hot indicators — zero ambiguous, **100% verification on 1.17 M rebuilt
values**. With the QCT tables that is **6,169 of 6,232 columns (99.0%)
regenerable from a 20 nt protospacer**, confirmed at 100% agreement over
6,169 columns × 500 guides.

### What decoding reveals: 6,232 columns, ~20 variables

The decoding is usually framed as a reuse result — the matrix now works for any
guide. It is also a statement about **what the matrix contains**, and that is
the more important half.

A 20 nt protospacer carries at most 40 bits. The matrix spends 6,232 columns
describing it, in three overlapping spellings of the same thing:

- **5,853 one-hot indicators** — "is there k-mer X at position Y". Each
  position's four base indicators sum to 1, so any one is recoverable from the
  other three (§12's lossless-encoding argument).
- **316 quantum-chemical columns** — fixed per-base lookups (§5). `p7monomer.
  HLgap` and `p7monomer.No.electrons` are two numeric codings of "what letter
  is at position 7".
- **a handful of summary columns** — Tm, GC, folding — which are weighted sums
  of the same positions.

Measured directly (`selector_comparison.csv`, and the footprint analysis in
§15a): two selectors that share **only 114 of 300 chosen columns** nevertheless
cover **all 20 protospacer positions, 100% overlap**, at a median of 9 columns
per position. They are not selecting different information. They are selecting
different spellings of the same 20 letters.

**Three results elsewhere in this document are corollaries of that fact**, and
should be presented together rather than as separate surprises:

| observation | section | why it follows |
|---|---|---|
| sixteen model classes span only 0.116 ρ | §12 | few effective dimensions to exploit differently |
| quadrupling the selected columns (300 → 1200) changes nothing (+0.0014; the baseline gets *worse*) | §9 | the extra columns are re-descriptions |
| nothing correlates with the residuals above 0.035 | §13 | the representation, not the model, is the limit |

**So "6,232 features" is not a rich representation.** It is an elaborate
spelling of a 20-letter word. That is why family A mattered: windowed flank
composition was the first thing added that described DNA *outside* the 20-mer,
and therefore the first thing that could add information rather than
re-describe it.

### Tested: how few columns does the model actually need?

The paragraphs above are an argument about *information*. They do not by
themselves license giving the model less, because a tree can only use what it
can split on, and a redundant encoding might be the shape that makes the signal
reachable. That is a separate, testable claim, and it had never been tested.

The published matrix partitions exactly — 5,853 indicators + 316 quantum
lookups + 63 remainder = 6,232 — so every part can be run on its own through
the standard harness: same rows, same CV, same champion, 5 seeds × 5 folds,
seed SD 0.001–0.003 (`representation.csv`, `python -m sgrna.representation
--run`).

| arm | columns | ρ | vs full |
|---|---:|---:|---:|
| `published_full` | 6,232 | 0.5272 | — |
| `mono_di_364` | 364 | 0.5189 | −0.0083 |
| `qct_only` | 316 | 0.5154 | −0.0118 |
| `v_onehot_only` | 5,853 | 0.5119 | −0.0153 |
| **`mono_74`** | **74** | **0.5081** | **−0.0191** |
| `hand_named_only` | 63 | 0.3044 | −0.2227 |
| **`mono_74` + `a_flank`** | **422** | **0.5462** | **+0.0190** |
| `published_full` + `a_flank` | 6,580 | 0.6080 | +0.0808 |

`published_full` reproducing 0.5272 / 0.2934 against the frozen anchor's
0.5273 / 0.2937 is a harness check in passing.

**74 columns recover 96.4% of the full matrix's ρ.** `mono_74` is the
mononucleotide indicators alone, and it is a *lossless* encoding: every position
carries at least three of its four base indicators, so the fourth is one minus
the others and the 20-mer is recoverable exactly. An 84× reduction in columns
costs 0.019 Spearman.

**`v_onehot_only` (5,853 columns) scores below `mono_di_364` (364).** More
columns, worse model. The 5,489 extra trimer and tetramer indicators are
re-descriptions that dilute selection, which is the clearest single
demonstration that "6,232 features" was never a rich representation.

**And 422 columns beat the whole published matrix.** `mono_74` + family A is
+0.019 ahead of all 6,232 published columns with 15× fewer of them: one feature
family describing DNA *outside* the 20-mer outweighs the entire published
feature engineering. (Selection pressure differs between arms — the small arms
select 300 of 422, the full arm 300 of 6,580 — so read the *sign* of that
comparison, not the size of the flank gain across arms.)

#### What the remaining 0.019 is, after two wrong guesses

It cannot be information: there is nothing in 6,232 columns that is not in the
74. Two explanations were proposed and tested, and **both were refuted**, which
is worth recording in full because the surviving account is the third.

**Rejected — redundancy as pre-computed interaction terms.** The champion is
`max_depth=4`. A tetramer indicator states a four-base interaction in one
split; from mononucleotide indicators the same statement costs four stacked
splits, the entire depth budget. So more depth should close the gap
(`representation_depth.csv`). It does the opposite, monotonically:

| | depth 4 | 6 | 8 | 12 |
|---|---:|---:|---:|---:|
| `mono_74` | **0.5081** | 0.4953 | 0.4739 | 0.4557 |
| `mono_di_364` | **0.5189** | 0.5120 | 0.4988 | 0.4858 |
| `mono_di_tri_1510` | **0.5163** | 0.5081 | 0.4972 | 0.4781 |

Depth only overfits, and handing the depth-4 model explicit trimers *hurts*
relative to dimers alone (0.5163 vs 0.5189). Whatever the gap is, it is not
interaction order.

**The accounting that does work.** The 63 leftover columns are the one part of
the matrix §13 says is *not* a function of the protospacer — they include
`gene.distance0`, which is positional, and base counts that disagree with a
direct count of the guide by up to 13. On their own they reach ρ 0.304. Added
back (`representation_residual.csv`):

| arm | columns | ρ |
|---|---:|---:|
| `mono_di_364` + `hand_63` | **427** | **0.5249** |
| `qct_316` + `hand_63` | 379 | 0.5194 |
| `mono_74` + `hand_63` | 137 | 0.5008 |

**427 of 6,232 columns — 6.8% — are statistically indistinguishable from the
whole matrix** (0.5249 vs 0.5272; paired over 5 seeds, mean difference 0.0023,
p = 0.11). So the matrix decomposes into **74 columns of sequence, 63 columns
of something that is not sequence, and ~6,095 columns of restatement.**

One oddity left unexplained: `mono_74 + hand_63` (0.5008) is *worse* than
`mono_74` alone (0.5081), a 7σ drop, and the sign reverses once dinucleotide
indicators are present. The leftover block only helps on top of a representation
that already carries dinucleotide structure. We do not have an account of that
and are not claiming one.

**Practical consequence, measured.** A reduced drop-in replacement exists, and
on the §14 pipeline it is worth the following (`memory_comparison.csv`,
`net_cost.csv`; peak RSS is a median of three repetitions in fresh processes):

| | all columns | reduced | change |
|---|---:|---:|---|
| peak RSS, reduced matrix **stored** reduced | 3.33 GB | **0.65 GB** | 5× less |
| peak RSS, subset taken *after* loading | 3.33 GB | 2.83 GB | 14% less |
| feature selection | 12.0 s | 1.4 s | 8× faster |
| dense net on the feature table, training | 6.1 s | 1.4 s | 4.4× faster |
| that net's parameters | 1,685,121 | 199,041 | 8.5× fewer |
| ρ on the §14 arm | 0.6939 | 0.6823 | −0.0116 |

Two qualifications that matter more than the headline. **The saving requires the
reduced matrix to be what gets stored** — subsetting columns after the full
matrix is in memory recovers almost none of it, because peak RSS is set by the
load. And **the sequence CNN gains nothing**: `seqnet.py` and crisprHAL 2 read
raw flanking DNA, never the feature table, so their cost is epochs ×
convolutions over sequence length. The reduced set speeds up nets that take the
table as input, which is not the architecture this project competes against.

It is not wired in as the default, because the 6,232-column form is what the
frozen baseline anchor reproduces.

### Inverting the model: what does it think a perfect guide looks like?

Because the base features are a deterministic function of the protospacer, the
model can be run backwards: search sequence space for the 20-mer it scores
highest. Beam search, width 60, LightGBM champion trained on the regenerated
base features (`model_inversion.csv`):

| | sequence | predicted | GC |
|---|---|---:|---:|
| the model's ideal guide | `ACTGCACAAAAGATGTCTTT` | 38.76 | **35%** |
| the model's worst guide | `AAAAAAATTGCCCCCCGGGG` | −14.14 | 55% |
| *observed 1st–99th percentile* | — | *0.98 – 41.73* | *mean 51%* |

Two checks pass. The optimum lands **inside** the observed range rather than
extrapolating absurdly, and it is **GC-poor (35% against a 51% average)** —
which is §6's strand-invasion mechanism recovered by a completely different
route. The worst case pairs a poly-A run with poly-C/poly-G blocks.

**And it is fictional.** That exact 20-mer appears **nowhere in NC_000913.2**.
This is the honest limit of inversion for this problem: a guide is not chosen
from sequence space, it is chosen from the handful of NGG-flanked sites inside
the gene you intend to cut. The global optimum is unreachable by construction.

What inversion is good for is **diagnosis, not design**: it shows what the
model has learned in a form a biologist can read, and here it says "prefer
AT-rich targets, avoid homopolymer runs" — consistent with the mechanism and
with the published literature. The design question a practitioner actually has
is the constrained one: *of the guides available in my target, which is best?*
That is ranking, which is what §1's pick-percentile measures.


```python
from sgrna import featurise
X, names = featurise.matrix(["ACGT..."])     # any guide, published representation
featurise.verify_against_published(500)      # 100.000000%
```

The ~31 hand-named columns are **not** pure functions of the protospacer —
base counts disagree with a direct count by up to 13, `sgRNA.temp` takes
exactly 14 values spaced 2.05 apart, PAM columns take fractions like 0.25.
Discretised or aggregated somewhere undocumented, so they are omitted rather
than guessed. **Featurise every row through the same code rather than mixing
sources.**

---

## 11. How to quantify predictive power and interpretability

Both get asserted more often than measured. These are the measurements, all
implemented in `src/sgrna/evaluate.py`.

### Predictive power — four numbers, not one

| metric | what it answers | chance |
|---|---|---|
| **R²** | how much variance is explained | 0 |
| **Spearman ρ** | does it get the *order* right | 0 |
| **top-decile precision** | of the guides it ranks in the top 10%, how many really are | 0.10 |
| **expected pick percentile** | draw 10 candidates, take its favourite — how good is that guide | 50th |
| **calibration slope** | is the *scale* right, or only the order | 1.0 |

The last two are the ones to lead with in a paper about guide design, because
they are the experiment a bench scientist actually runs. They also disagree
with R² often enough to matter: CatBoost has the **best top-decile precision
(0.268)** in this project while sitting third on R² (0.362). If you rank models
by R² you pick the wrong one for the job.

Report all four. Rank by the one that matches the use case, and say which.

### Interpretability — five numbers

"Interpretable" usually means "I read the feature list and it seemed
reasonable". That is not a measurement. These are.

| metric | what it answers | how |
|---|---|---|
| **selection stability** | re-run with a different split — same features? | Kuncheva index over the selected sets; 1 = identical, 0 = chance overlap |
| **concentration** | how few features carry the story | features needed for 50% / 90% of importance; Gini; effective feature count |
| **method agreement** | do two importance methods rank features the same? | Spearman between split gain and permutation importance |
| **faithfulness** | does removing the "important" features actually hurt? | ΔSpearman from dropping the top-20, minus ΔSpearman from dropping 20 random |
| **direction consistency** | does the model use each feature the way the data does? | sign agreement between univariate ρ and the model's response |

Why the Kuncheva index rather than Jaccard: two random selections of 300 out of
6,580 overlap by ~14 features through luck alone. Jaccard rewards that;
Kuncheva subtracts it, so 0 really means "no better than chance".

Why faithfulness is the one people skip: with 6,580 correlated columns a model
can route around any feature you remove. If dropping the top-20 costs no more
than dropping 20 at random, the importance ranking is describing correlations,
not the model's reasoning — and any mechanism you read off it is decoration.

## 12. Model class: what changes

**Prediction** — identical folds, identical selected features (XGBoost gain,
top 300), only the predictor swapped. 3 seeds × 5 folds, base + family A
(`model_comparison.csv`):

| model | R² | ρ | top-decile precision | pick percentile | fit (s) |
|---|---:|---:|---:|---:|---:|
| Stacked blend | **0.3870** | **0.6108** | 0.2633 | 72.07 | 14.02 |
| **LightGBM** | 0.3824 | 0.6086 | 0.2606 | **72.09** | 0.29 |
| XGBoost | 0.3812 | 0.6076 | 0.2602 | 72.06 | 0.43 |
| CatBoost | 0.3615 | 0.6010 | **0.2681** | 71.99 | 0.63 |
| Hist gradient boosting | 0.3700 | 0.5997 | 0.2597 | 71.79 | 1.29 |
| SVR (RBF) | 0.3529 | 0.5864 | 0.2445 | 71.21 | 8.11 |
| Ridge | 0.3406 | 0.5770 | 0.2467 | 71.16 | **0.02** |
| Elastic net | 0.3401 | 0.5761 | 0.2436 | 71.06 | 0.19 |
| Nyström + ridge | 0.3276 | 0.5749 | 0.2460 | 71.23 | 0.48 |
| PLS (25 components) | 0.3381 | 0.5747 | 0.2460 | 71.07 | 0.21 |
| MLP | 0.3387 | 0.5721 | 0.2342 | 70.36 | 0.76 |
| LightGBM DART | 0.2671 | 0.5684 | 0.2443 | 70.78 | 0.91 |
| Extra trees | 0.3021 | 0.5562 | 0.2383 | 70.33 | 1.09 |
| Random forest | 0.2915 | 0.5497 | 0.2361 | 70.24 | 2.33 |
| LightGBM (Huber loss) | 0.1921 | 0.5000 | 0.2294 | 68.86 | 0.37 |
| k-NN (k=50) | 0.2167 | 0.4951 | 0.2159 | 68.80 | 0.19 |

Sixteen model classes, spanning boosting, bagging, kernels, a neural net, a
latent-factor linear model, a purely local learner and a blend of the best
three. **The whole space is 0.116 wide in Spearman and 3.3 percentile points
wide on the guide you actually pick.**

Four things worth saying out loud.

**Boosting wins, bagging loses.** Random forest and extra trees are 0.05–0.06
Spearman behind despite being tree ensembles. Whatever is in these features is
not something bagging with `max_features='sqrt'` finds.

**A linear model gets within 0.03.** Ridge on the same 300 features reaches
ρ 0.577 against boosting's 0.608 — so **most of the signal is additive**. That
is a real result about the biology, not just about modelling: the determinants
of cutting efficiency behave largely independently, and the interactions
boosting captures are worth about 0.03 Spearman.

> **Checked on a basis with the redundancy removed** (`additivity_reduced.csv`,
> 5 folds, no selection step), because a gap measured on gain-selected columns
> could be an artefact of which columns the selector happened to pick:
>
> | columns given | ridge | LightGBM | XGBoost | boosting − ridge |
> |---|---:|---:|---:|---:|
> | **427** (mono+di+hand) | 0.5029 | 0.5258 | 0.5255 | **+0.0230** |
> | **775** (+ family A) | 0.5618 | 0.5993 | 0.5962 | **+0.0375** |
> | 6,232 (base only) | 0.4233 | 0.5298 | 0.5267 | +0.1064 |
> | 6,580 (+ family A) | 0.4941 | 0.6100 | 0.6070 | +0.1160 |
>
> **Additivity holds, and more cleanly than the headline suggests**: on 427
> columns ridge reaches 95.6% of boosting's ρ. Two further things fall out.
> **Ridge on the raw 6,232 collapses to 0.423** — collinearity across thousands
> of near-duplicate columns, not biology — so §10's redundancy was *concealing*
> the additivity from the linear model rather than manufacturing it; removing it
> is worth +0.08 to ridge. And the **gap widens when family A is added** (0.023
> → 0.038), so the flanking-DNA columns carry proportionally more non-additive
> structure than the protospacer columns do.

**Model choice barely moves the thing you care about.** Across all sixteen,
the guide you pick spans 3.3 percentile points, and across the eleven that are
not actively broken, 1.8.

**Blending is not the answer either.** The stacked ridge-over-
(LightGBM + random forest + Ridge) is the best single row in the table, by
+0.002 ρ over LightGBM alone, at 48× the fit cost. Its own fitted weights say
why: LightGBM 0.775, Ridge 0.324, **random forest −0.003**. The forest's errors
are not complementary — it is simply a worse version of the same model. Two
learners is the whole ensemble, and one of them is a linear model.

Two failures are informative rather than embarrassing. **k-NN at ρ 0.495** says
the signal is not local: guides with similar feature vectors do not have
similar scores, which is what you expect when the response is a sum of many
small additive terms rather than a landscape with neighbourhoods.
**LightGBM with a Huber loss collapses to R² 0.192** while keeping ρ 0.568 — the
robust loss shrinks toward the median and destroys calibration while leaving
the ranking intact, a clean demonstration of why R² and ρ must both be
reported.

**Interpretability** — each model selecting with *its own* importances, seed
41, 5 folds, run twice: once from each library's default importance and once
from SHAP (`interpretability.csv`, `--importance native|shap`). The SHAP rows
are what this project now reports; §12a is why.

| model | importance | stability | rank agreement | method agreement | faithfulness | features for 50% |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | gain | 0.49 | 0.55 | **−0.02** | +0.003 | 330 |
| XGBoost | **SHAP** | **0.75** | **0.64** | **+0.39** | +0.006 | **44** |
| LightGBM | gain | 0.72 | 0.67 | +0.44 | +0.007 | 151 |
| LightGBM | **SHAP** | **0.74** | **0.71** | **+0.40** | +0.010 | **43** |
| Random forest | impurity | **0.91** | 0.78 | +0.30 | **+0.022** | 175 |
| Random forest | **SHAP** | **0.90** | **0.79** | +0.20 | +0.003 | 87 |
| Ridge | \|coef\| | 0.45 | 0.42 | +0.28 | **−0.012** | 1,371 |
| Ridge | **SHAP** | 0.61 | 0.54 | +0.37 | −0.004 | 1,224 |

Four things to take from it.

**Changing the importance method helps more than changing the model.** XGBoost
read through SHAP goes from the worst explainer in the table to the joint best:
stability 0.49 → 0.75, method agreement −0.02 → +0.39, and the features needed
for half the importance from **330 to 44**. Nothing about the fitted model
changed.

**The reason to prefer LightGBM for interpretability disappears.** Under gain
LightGBM looked 1.5× more stable with half as many load-bearing features; under
SHAP the two are 0.75 vs 0.74 and 44 vs 43 columns. The recommendation below is
revised accordingly — and conveniently so, since the frozen baseline anchor
wants XGBoost anyway (§15a).

**Ridge is still the outlier, and SHAP only half-fixes it.** Faithfulness is
−0.004 against −0.012: dropping its "top" features still hurts no more than
dropping random ones, and it still spreads importance over >1,200 columns.
Coefficient magnitude on 6,580 correlated columns is not an explanation, and
neither is its Shapley equivalent.

**Random forest keeps the most reproducible ranking either way** (0.90–0.91)
while predicting worst (ρ 0.550) — the two axes are separate, which is the
reason it remains the confirmation model below.

⚠️ **The faithfulness column is measured on the full matrix and is therefore
not a property of the rankings.** All four values are ≈0 because the matrix can
substitute for anything removed; on a representation without restatement the
same probe gives −0.015 for the top 20 against −0.001 at random. See §12b
before quoting this column.

### 12a. Seven model families, three importance methods: gain fails, SHAP does not

Everything above rests on two models, both boosted trees, and on split gain.
Two objections follow, and both are now tested (`importance_models.py`,
`importance_model_*.csv`, 3 folds, base + family A).

**"You only tried boosted trees."** Bagged forests fit independently rather
than on residuals; penalised linear models have no selection step at all and
score every column. Seven families were run — XGBoost, LightGBM, CatBoost,
random forest, extra trees, ridge, elastic net — with three methods where each
applies: native (gain / impurity decrease / |coefficient| on standardised
columns), TreeSHAP mean |φ| on validation rows, and permutation. Pairs are
matched to SHAP's coverage so the methods are compared on identical model
pairs:

| method | ρ between models, per column | ρ between models, per block |
|---|---:|---:|
| native | **0.050** | 0.531 |
| permutation | 0.258 | 0.581 |
| **SHAP** | **0.496** | **0.800** |

**The column-level/block-level split survives the change of model class** —
native importance agrees at 0.050 across seven families, against 0.531 at block
level. So that result is a property of the matrix, not of boosting.

**"Use SHAP" was the right objection, and it was expected to fail.** The
prediction recorded before running it was that SHAP would show the same
instability: Shapley values divide credit for one prediction among its inputs,
so two columns carrying identical information should split it, and the split
should depend on which the trees happened to use. **That prediction is wrong.**
XGBoost and LightGBM rank each other's columns at **−0.070 by gain and +0.867
by SHAP** (block level 0.590 → 0.984).

The reason is visible directly, and it retires a result reported above:

| | binary indicators' share of importance, native | under SHAP |
|---|---:|---:|
| XGBoost | **57.4%** | **31.0%** |
| LightGBM | 29.7% | 31.0% |
| CatBoost | 27.3% | 25.1% |
| random forest | 16.0% | 24.0% |
| extra trees | 46.3% | 47.4% |

**The "data-type affinity" gap between XGBoost and LightGBM is an artefact of
split gain.** XGBoost's preference for binary indicators halves under SHAP and
lands on LightGBM's figure exactly. Gain counts fit improvement at split time,
which over-credits a binary column used in many shallow nodes; SHAP measures
effect on the output. These models fit nearly the same function — they span
0.08 ρ — so once the function is measured instead of the fitting procedure,
they agree.

Under SHAP all five tree models agree on the same two leading blocks: flank
quantum within 10 nt (≈14%) and protospacer dinucleotide quantum (≈13%), while
the 3,383 middle-position indicators carry ≈10% between them.

**What this changes.** "Never read mechanism off an importance plot" is too
strong and should be narrowed to **never read it off gain**. SHAP at 0.496 per
column is substantial agreement but not unanimity, so a claim still belongs at
block level (0.800), and the three kinds of evidence listed below are still what
the biology rests on. But a SHAP ranking is reproducible across model classes in
a way a gain ranking is not, and that is a usable interpretability result rather
than a caveat.

### 12b. Faithfulness cannot be measured on a redundant representation

The standard faithfulness probe — drop what the ranking points at, refit, see
whether it hurts — returns "the ranking is not load-bearing" on this matrix:
dropping each model's top 20 columns costs **−0.0024** against **+0.0015** for
20 at random. §12's interpretability table reported that as the decisive
evidence that neither model had found the real features.

**It is not decisive, it is confounded by §10.** With 6,232 columns of
restatement in play, removing the 20 columns a model used leaves many other ways
to state the same thing, so the probe cannot distinguish an unfaithful ranking
from a substitutable encoding. Re-run on the 775-column reduced representation
(`faithfulness.py`, `faithfulness_groups*.csv`; 3 folds, XGBoost and LightGBM
averaged, SHAP-ranked, each arm refitted from scratch):

| dropped and refitted | full, 6,580 columns | reduced, 775 columns |
|---|---:|---:|
| top 20 | −0.0024 | **−0.0150** |
| 20 at random | +0.0015 | −0.0006 |
| top 50 | −0.0102 | **−0.0453** |
| 50 at random | −0.0008 | −0.0015 |
| top 100 | −0.0226 | **−0.0816** |
| 100 at random | +0.0016 | −0.0054 |

The random control is ≈0 in both, so the top-minus-random gap is the statistic:
**0.015 / 0.045 / 0.077 on the reduced set against 0.003 / 0.009 / 0.024 on the
full one.**

**The normalisation matters, and column fraction is the wrong one.** Twenty
columns is 0.3% of 6,580 and 2.6% of 775. Matching on fraction instead makes the
comparison worse: the top 170 of 6,580 (also 2.6%) costs **−0.027** against +0.001 at random,
but it removes **80% of the model's attributed importance** rather than 35%. Matching on share
of importance removed is the comparison that means something
(`faithfulness_shap_share.csv`):

| | top 20 | top 50 | top 100 | top 170 | n for 50% of SHAP |
|---|---:|---:|---:|---:|---:|
| full, 6,580 columns | 34.6% | 53.1% | 68.6% | 79.6% | 44 |
| reduced, 775 columns | 37.5% | 57.5% | 73.8% | 85.2% | 37 |

At an almost identical share of attributed importance removed, the full matrix
loses 0.002 and the reduced one 0.015 — **six times the damage from taking away
the same share of the explanation.** Note also that the two representations
concentrate importance almost identically (44 and 37 columns for half of it), so
this is not a difference in how peaked the attribution is; it is a difference in
whether the information is reachable elsewhere.

**Two consequences for how this project reports interpretability.** A
faithfulness result measured on a redundant matrix is a statement about the
matrix. And the "method agreement" metric in §12 measures how concentrated and
individually irreplaceable an attribution is, which here is largely a property of
the encoding rather than of the model — it was never a measure of whether an
attribution is biologically correct, and the group-level ablations, single named
quantities and transfer tests remain what the biology rests on.

A related negative result from the same run, worth keeping. Clustering columns by
|r| ≥ 0.7 around each model's top SHAP columns gives clusters of **1–3 columns**:
the leading columns have almost no close twins. Removing the top five clusters —
10–11 columns carrying **19% of total SHAP** — still costs nothing on the full
matrix. So the substitution that protects the model is **not pairwise**; a tree
rebuilds the same function from many weakly-correlated columns, which is why
pairwise correlation only ever put a floor under replaceability (§12a).

### The recommendation

1. **Leave each configuration on the library its numbers were produced with;
   report SHAP rather than gain.** The former reason to prefer LightGBM was
   explanation quality, and under SHAP that gap is gone (0.75 vs 0.74 stability,
   44 vs 43 load-bearing columns) while accuracy was always a tie — so there is
   no longer an argument for changing either. Concretely: the **baseline** on the
   13,880 published rows stays **XGBoost**, because that is what reproduces the
   published 0.2937 / 0.5278 (§15a); the **curated head-to-head** stays
   **LightGBM** as predictor with XGBoost selecting columns, because that is what
   produced 0.7078 and what the tuning search in §16 operates on. The two differ
   by ≤0.001 wherever both were run.
2. **Confirm every mechanistic claim under random forest** before it goes in
   the paper. It is the worst predictor and the most trustworthy explainer;
   if a feature matters under both RF and LightGBM, it matters.
3. **Report Ridge as a reference row.** "A linear model gets within 0.03" is a
   finding about the biology and it costs 0.02 seconds to produce.
4. **Quote SHAP, not gain, and quote it at block level.** Gain agrees across
   model families at ρ 0.050 and carries a splitting-algorithm artefact worth
   26 percentage points of XGBoost's attribution; SHAP agrees at 0.496 per
   column and 0.800 per block (§12a). Report the method, and the
   between-model correlation for it.

```bash
python -m sgrna.evaluate --what prediction --models xgboost lightgbm ridge
python -m sgrna.evaluate --what interpretability --models lightgbm random_forest
```

## 13. Why the new libraries didn't move the needle

Eight feature families were added, each with a mechanism and a published
dataset behind it, and six of them moved Spearman by less than 0.01. The
natural worry is that something is broken. It is not, and the reason is
specific enough to be measured. Everything here is reproducible with
`python -m sgrna.diagnose --what all`.

### The published matrix is a lossless encoding of the protospacer

5,853 of the 5,887 `V####` columns are position-specific k-mer indicators
(§10). Crucially, **every one of the 20 positions carries at least three of its
four mononucleotide indicators** — 14 positions carry all four — so the fourth
is one minus the others. The 20-mer is therefore recoverable from the matrix
exactly.

That has a hard consequence: **any feature that is a deterministic function of
the 20 nt protospacer adds literally zero information.** It can only re-express
what is already there in a shape the model finds easier. Whatever gain such a
family shows is the convenience of a pre-computed nonlinear summary, not new
evidence.

### Which families were in that position — measured

For every new column, the out-of-fold R² of predicting it by **linear**
regression on position-specific mono- and dinucleotide indicators — i.e. on
information the matrix already contains (`redundancy.csv`):

| family | median R² from the 20-mer | share above 0.9 | its Δρ |
|---|---:|---:|---:|
| `b_energy` (CRISPRoff ΔG) | **0.984** | 70% | +0.007 |
| `d_mechanics` (nearest-neighbour duplex) | **0.845** | 50% | +0.020 |
| `c_folding` (RNA secondary structure) | 0.297 | 0% | +0.007 |
| `f_methylation` (Dam/Dcm motifs) | 0.141 | 0% | +0.004 |
| `a_flank` | −0.012 | 0% | **+0.080** |
| `d_supercoiling` | −0.013 | 0% | +0.045 |
| `e_nucleoid` | −0.016 | 0% | +0.016 |
| `g_transcription` | −0.016 | 0% | +0.015 |
| `h_offtarget` | −0.022 | 0% | +0.000 |

`b_energy` and `d_mechanics` are not merely correlated with the sequence — they
are nearly *linear* in its dinucleotide content, which they should be: a
nearest-neighbour thermodynamic sum is by construction a weighted count of
dinucleotides, and CRISPRoff's ΔG is dominated by the same term. Importing
them was importing the matrix back into itself.

**The gain tracks one thing, and it is not the quality of the biology: it is
whether the feature describes anything outside the 20-mer.** Every family whose
R² from the protospacer is ~0 delivered; every family whose R² is high did not.

This is also the diagnostic to run *before* adding the next dataset. If a
candidate feature regresses onto the k-mer basis at R² > 0.9, it cannot help,
however good the mechanism sounds.

### The model has already wrung its features dry

Take the champion's out-of-fold residuals and correlate every column against
them (`headroom.csv`, out-of-fold ρ 0.606):

| | max abs ρ with residual |
|---|---:|
| the 6,480 columns the model was given | **0.035** (zero above 0.10) |
| `d_supercoiling`, held back | 0.154 |
| `e_nucleoid`, held back | 0.151 |
| `g_transcription`, held back | 0.146 |
| `c_folding`, held back | 0.067 |
| `b_energy`, held back | 0.045 |
| `f_methylation` / `d_mechanics` / `h_offtarget`, held back | 0.015–0.035 |
| `i_shape`, held back (`headroom_with_shape.csv`) | 0.028 |

Not one of 6,480 available columns retains a residual correlation above 0.10.
**The model is not underfitting; it is out of material.** The only features
with anything left are the positional ones — and §4 showed those are one
variable measured three ways.

**Why this test can be trusted: it has a positive control.** A diagnostic that
only ever answers "nothing here" is not evidence. When family A was held back,
the same test flagged it at **0.147**, four times anything the model had — and
adding it was then worth +0.080. The test found the one thing that was there.

#### And it still holds after the matrix is cut to 427 columns

The reduced representation of §10 drops 5,805 columns. They are all functions of
the 20-mer, but the 427 that remain carry only its first and second order, so a
trimer or tetramer indicator could in principle track something they cannot
express. Re-run on the subset's own out-of-fold residuals
(`representation_headroom.csv`):

| model | out-of-fold ρ | max \|ρ\| among columns it has | among columns dropped |
|---|---:|---:|---:|
| all 6,232 | 0.531 | 0.028 | 0.147 *(family A, held back)* |
| the 427 | 0.525 | 0.022 | 0.146 *(family A, held back)* |
| 427 + family A | 0.597 | 0.028 | **0.070**, zero above 0.10 |

**Rows one and two are the result: dropping 5,805 columns did not change what
the model is missing.** The strongest residual correlate is the same family-A
column at the same magnitude, so the reduction removed restatement rather than
creating a blind spot.

Row three is the honest limit. With family A included, the dropped columns top
out at 0.070 — under the 0.10 threshold, but **2.5× the strongest correlate
among the columns the model holds** — and the leading ones are `V####`
higher-order k-mer indicators. So a sliver of higher-order sequence signal does
exist, which is consistent with the reduced set's −0.01 ρ cost, and §10's depth
ladder shows why taking it is unprofitable: handing the model 1,146 trimer
indicators to reach it loses more to selection dilution than the sliver is
worth.

### And a neural net on raw sequence does not rescue it

If the representation were the problem, a convolutional network reading the
bases directly should beat gradient boosting on the engineered columns. Same
rows, same folds, a conv–conv–BiGRU encoder close to the published bacterial
architecture (`seqnet.csv`):

| model | input | R² | ρ |
|---|---|---:|---:|
| CNN + BiGRU | 20 nt protospacer + PAM | 0.257 | 0.498 |
| XGBoost | the 6,232 published columns | 0.294 | **0.527** |
| CNN + BiGRU | protospacer + ±100 nt of genome | 0.272 | 0.516 |
| LightGBM | published columns + `a_flank` | **0.382** | **0.609** |

Two readings, both useful.

**On the guide itself the two representations tie** (0.498 vs 0.527, the CNN
slightly behind). The published encoding is not costing anything; a network
given exactly the same information arrives at the same answer.

**On the flanks the engineered features win outright.** Raw context buys the
CNN +0.019; the hand-computed windowed composition buys boosting **+0.080**.
That is a statement about the biology: the flank effect is a *long-range
compositional average* over hundreds of bases (§6), which a windowed GC or
tensor mean expresses exactly and a 5 nt convolution with pooling expresses
badly. Motif-finders are the wrong tool for a signal that has no motif.

### The label is not the ceiling either

Two independent screens of the same library — WT-SpCas9 and eSpCas9, different
enzymes, separate experiments — share **33,567 protospacers**, and their scores
agree at **ρ 0.810** (`ceiling.json`). Under the usual attenuation argument
(each observed score is a shared signal plus independent noise) that rank
correlation *is* the reliability, so the highest correlation any model could
reach against a single observed cut score is √0.810 ≈ **0.90**.

> **Recomputed 2026-10-04, and the assumption behind it is false.** The
> attenuation argument requires the disagreement between the two screens to be
> *independent noise*. It is not: a model fitted to the rank difference between
> the two screens' scores for the same guide predicts it at **ρ 0.546**, against
> **0.708** for predicting the cut score itself (`ceiling_recomputed.json`,
> `src/sgrna/ceiling2.py`). A systematic, sequence-dependent difference is
> exactly what two enzymes engineered to differ in specificity should show.
>
> | assumption about the 0.19 disagreement | reliability | ceiling |
> |---|---:|---:|
> | all of it is noise (as originally assumed) | 0.810 | 0.90 |
> | the predictable ~30% is biology | 0.866 | **0.93** |
>
> Two further biases push in opposite directions: the two screens share their
> **guide library** (100% identical sequences, eSpCas9 being WT plus three point
> mutations), which flatters agreement; and the enzyme difference, which
> depresses it. **Quote 0.90–0.93 as a bracket, never 0.90 as a figure.** A
> trustworthy ceiling needs same-library, same-enzyme independent repeats. The
> conclusion that neither model is near the limit survives either way.

> **Superseded 2026-10-05 — use the source paper's own agreement statistics
> instead.** The claim that no suitable repeats are published was wrong. Guo et
> al. 2018 ran **two biological replicates per condition by independent
> transformation** and published the agreement, and the score we train on is
> their **geometric mean** of the two (verified against *NAR* 46:7052, Fig 2).
> That changes the arithmetic twice: the attenuation bound applies to predicting
> *one* observation, and under Spearman–Brown a 2-item mean has reliability
> `2r/(1+r)`, above `r`.
>
> | route | published statistic | r | reliability | ceiling |
> |---|---|---:|---:|---:|
> | **tiling library, Fig 2c** — 901 guides shared with an independent library; a test–retest of the published quantity, so no correction needed | R² = 0.771 | 0.878 | 0.878 | **0.937** |
> | replicate agreement, Fig 2b + Spearman–Brown | R² > 0.78 | >0.883 | 0.938 | **0.969** |
> | colony counting, Fig 2d (15 sgRNAs, orthogonal assay) | R² = 0.840 | 0.917 | 0.917 | 0.957 |
>
> **Quote ≈0.94–0.97, and lead with the tiling figure**: it uses a *different*
> library, which is exactly the objection the two-enzyme estimate cannot answer,
> and it is conservative because library design differences are not label noise.
>
> **The two routes are not equally safe.** Guo's label is the `|Z|` of a
> **Cas9/dCas9 ratio** (`Jacky/agent-notes/CUT_SCORE_PROVENANCE.md`), so the
> replicate route has to assume the ratio score inherits the single-arm read
> counts' reliability. A difference is never more reliable than its components,
> and the shortfall scales with how much true signal the two arms share — which
> the dCas9 control arm exists to maximise. At each log-count's Fig 2b floor of
> r = 0.883:
>
> | shared signal between arms | ratio-score reliability | ceiling |
> |---:|---:|---:|
> | 0% (what the replicate route assumes) | 0.883 | **0.968** |
> | 30% | 0.841 | 0.956 |
> | 50% | 0.791 | **0.940** |
> | 70% | 0.694 | 0.905 |
>
> So 0.968 is an **upper bound** that holds only if the control shares nothing.
> The tiling route needs none of this: both sides of Fig 2c are already the
> published score, so r = 0.878 *is* its reliability, and on 901 guides
> r ∈ [0.863, 0.893] gives **0.937 [0.929, 0.945]** — biased conservatively,
> because library design differences are not label noise.
>
> **Quote ≈0.94 as the working figure and 0.94–0.97 as the range.** Two caveats
> apply to both: the published values are Pearson while the ceiling is quoted in
> Spearman, and Fig 2b is an inequality across ten libraries so 0.883 is a floor.
>
> Consequence: SLICER at 0.707–0.721 is **~77%** of the way to the working figure
> of 0.94 (74% against the loose upper bound), not ~80%. Narrowing the range needs per-guide replicate values, which are in SRA
> (PRJNA450978) as raw reads only. Correction due to Jacky's work plan.

Two supporting numbers: 85% of cut-score variance is *within* gene rather than
between genes, so the label really is measuring guide quality and not gene
essentiality; and the published `cut.score` agrees with the crisprHAL
re-derivation at ρ 0.945, confirming the two are the same measurement.

A caveat worth writing into the methods: part of that shared 0.810 is shared
*systematic* structure — same library, same growth conditions, same locus
effects — which a model could only capture with the right positional features,
not with sequence. So 0.90 bounds what is predictable in principle, not what is
predictable from sequence alone.

### So: no, nothing is broken — and the combination does work

The impression of "no improvement" comes partly from reading one-family-at-a-
time ablations. Put all nine families in at once, with LightGBM, and the model
reaches **ρ 0.641 and the 73.4th pick percentile** (seed 41, 5 folds, random-row
CV, `best_current_model.csv`) — against the published baseline's 0.527 and ~70th.
That is **+0.114 Spearman over the starting point**, which is not a small
result; it is just distributed across families rather than sitting in one.

(Random-row CV flatters the positional families, which can partly recognise a
locus. The honest contiguous-arc figure for `a_flank + d_supercoiling +
d_mechanics` is ρ 0.615, §2.)

Putting the four measurements together, the remaining headroom — roughly 0.61
to 0.90 in Spearman — is:

- **not in the current features** — residual correlations max out at 0.035;
- **not in the model** — sixteen classes span 0.116 and the best blend adds
  0.002;
- **not in the representation of the guide** — a CNN on raw bases ties;
- **not in the label** — two screens agree at 0.810.

Which leaves exactly one place: **features that are outside the 20-mer and that
nobody has built yet.** That is a much better position to be in than a noise
ceiling, and it is what §14 is about.

## 14. Head-to-head with crisprHAL 2, on identical rows

Every number above sits on a different row set from the model it is being
compared to — 13,880 guides here against the 33,495 curated guides crisprHAL 2
trains on — which makes the gap uninterpretable. This section removes the
excuse, and it is only possible because §10 made the published representation
computable for any guide.

Same curated guides, their own activity label, 5-fold CV, LightGBM, top-300
selection fitted inside the training split (`headtohead.csv`,
`python -m sgrna.headtohead --run`):

| what the model sees | rows | cols | ρ | R² | pick %ile |
|---|---:|---:|---:|---:|---:|
| published rows only, base features | 8,617 | 6,169 | 0.536 | 0.299 | 69.5 |
| all curated rows, base features | 33,567 | 6,169 | 0.544 | 0.305 | 69.6 |
| + immediate ±10 nt (`seq.*`, `qct.*`) | 33,567 | 6,429 | 0.618 | 0.383 | 72.9 |
| + windowed composition, 50–1000 nt | 33,567 | 6,257 | 0.660 | 0.468 | 74.0 |
| **+ all of family A** | **33,567** | **6,517** | **0.707 ± 0.008** | **0.522** | **75.9** |
| *crisprHAL 2, as published* | *33,495* | — | *0.697* | — | — |
| *crisprHAL 2, re-run here on these folds* | *33,567* | — | *0.6971 ± 0.0067* | *0.517* | — |

Eight folds across two seeds for the headline arm (0.7078 at seed 41, 0.7062 at
seed 42), three folds for the two decomposition arms.

### The re-run, 2026-10-02 — the comparison is no longer against a published figure

`src/sgrna/crisprhal_rerun.py` trains crisprHAL 2's WT-SpCas9 model — their
architecture, their hyper-parameters, their 48 epochs at batch 1024, imported
unchanged from `external_data/crisprHAL` — on our rows and our folds, with
their own 378 nt input looked up by protospacer (`crisprhal_rerun.csv`).

Two independent checks that the harness is faithful: on their shipped
train/test split it scores **0.6940** against their published hold-out of
0.695, and its mean over our five folds is **0.6971** against their published
5-fold 0.697.

Paired on identical rows, identical folds and the same label (seed 41):

| fold | ours (LightGBM, base + family A) | crisprHAL 2, re-run | difference |
|---:|---:|---:|---:|
| 1 | 0.6944 | 0.6855 | +0.0089 |
| 2 | 0.7079 | 0.6988 | +0.0091 |
| 3 | 0.7121 | 0.6978 | +0.0142 |
| 4 | 0.7112 | 0.7022 | +0.0090 |
| 5 | 0.7136 | 0.7012 | +0.0125 |
| **mean** | **0.7078 ± 0.0078** | **0.6971 ± 0.0067** | **+0.0107** |

**Ahead in 5 of 5 folds**, paired t-test p = 0.0006 (sign test p = 0.0625 —
quote both; n = 5 is too small for the t-test alone to carry the claim).

> This arm compares **untuned against untuned**. §16a redoes it with both models
> searched, where the margin becomes **+0.0161**, also 5/5 folds. Quote §16a for
> any claim about which model is better; quote this arm only as the like-for-like
> default-settings comparison.

**How to state it: a small but consistent advantage, not a victory.** +0.011 ρ
is a fraction of a pick percentile and would not change anyone's guide choice.
Three caveats belong with it: their arm is one seed (ours is two), their
hyper-parameters were tuned by them on their own split rather than re-tuned
here, and the crisprHAL line has cross-species evidence this project does not
(but see §15 for what that evidence actually consists of, and whose paper it
belongs to). What makes the result worth reporting is the cost —
the whole five-fold run takes **about 3 minutes end to end** against
**105 minutes** for crisprHAL's five folds on the same machine — roughly
**36×**, measured rather than asserted (see below) — and every column has a
name.

The September caveat "compared against a published figure, not a re-run" is
retired.

### What the speed claim actually is, measured

Earlier versions of this document said the model "fits in 0.3 s on a CPU".
That figure **was** measured — `model_comparison.csv` records LightGBM's fit at
0.291 s, tightly reproducible across 15 folds (0.28–0.31). The error was one of
**scope**, not fabrication: it is the cost of the final fit on 300 already-
selected columns, quoted as though it were the cost of training the model. The
XGBoost pass that chooses those 300 from 6,517 is excluded, and it dominates.

Measured end to end on 2026-10-03 (`results/timing.csv`), §14 configuration —
33,567 rows × 6,517 columns, one fold:

| stage | time |
|---|---:|
| impute | ~1.8 s |
| **XGBoost feature selection, 6,517 → 300** | **~31 s** |
| **LightGBM fit on the 300** | **~2.4 s** |
| predict | 0.02 s |
| **one fold, end to end** | **~35 s** |
| **five folds** | **~3 min** |

The feature selection is roughly 13× the fit it feeds. (The fit itself also
measures 1.5–2.4 s on the larger §14 matrix against 0.291 s on the 13,880-row
one, so part of the gap is simply the bigger problem.)

Against crisprHAL 2 on the same machine (1,265.6 s per fold, logged per fold in
`crisprhal_rerun.csv`, consistent across all five):

Those first measurements were unusable: identical work took 13.0, 34.7, 116.2
and 35.4 s. Two causes, both ours — `n_jobs=-1` let the selector take whatever
cores were free, and the matrix was held resident and copied twice, pushing a
16 GB host into swap. Fixing the thread count at 8 and reading the matrix by
`mmap` makes it reproducible: **IQR 0.71 s across 5 repeats**
(`timing_controlled.csv`).

| | per fold (IQR) | five folds | peak memory, one fold |
|---|---:|---:|---:|
| **current pipeline** (XGBoost selector) | **17.07 s** (±0.71) | **1.42 min** | 3.33 GB (3.23–3.68) |
| with a LightGBM selector (§15a) | **6.63 s** (±0.37) | **0.55 min** | 3.05 GB (3.05–3.05) |
| crisprHAL 2 | 1,266 s | 105 min | 4.55 GB |
| **ratio, as the pipeline stands** | **74×** | **74×** | **1.37×** |
| **ratio with the cheaper selector** | **191×** | **191×** | — |

Conservative on two counts: we used 8 of 10 threads where crisprHAL used all
ten, and the memory figures are within 1.4× of each other.

> **How the memory column must be measured, because it was wrong twice.**
> `timing_controlled.csv` reports **4034.3 MB for both selectors**, identically,
> because both ran in one process and `ru_maxrss` is a high-water mark that never
> falls — the figure belonged to neither arm. The column above comes from
> `bench.py`, which runs every repetition in a fresh process, three repetitions
> per arm. That matters: two single-shot runs of the *same* configuration came
> back 2.95 and 3.29 GB, so run-to-run spread is ~0.3 GB and **the 0.3 GB gap
> between the two selectors is not a real difference.** Peak memory needs
> separate processes and repetitions exactly as timing does.

**The win is time, not memory.** Peak resident memory is the same order — 3.33 GB
against 4.55 GB — because our cost is a 33,567 × 6,517 float32 matrix carried
through imputation and selection, while theirs is a 310 MB one-hot input
expanded by a network over 48 epochs. Anyone expecting the tree pipeline to be
light on RAM as well as fast should not: it is not. The one lever that does work
is storing fewer columns (§10): 0.65 GB on a reduced store, a 5× reduction.

Both figures are **CPU-to-CPU on the same machine**. crisprHAL 2 as published
is GPU-trained, so "faster than a GPU model" is not what was measured here —
what was measured is that their architecture costs ~74× more CPU time than ours
for the same predictive performance.

**A health warning on that ratio.** Four measurements of the identical work gave
13.0, 34.7, 116.2 and 35.4 s per fold; the selection stage alone ranged 9.6 to
112.7 s. This host has 16 GB against a 198 MB matrix, a 250 MB matrix and
several GB of caches, and logged 970k pageouts — the timings are dominated by
memory pressure, not by the algorithms. The ~35 s figure is the median of the
two agreeing isolated runs. The reliable qualitative claim is **minutes against
hours, on a CPU, for equal predictive performance**; the precise multiple needs
a quiet machine with enough RAM to avoid swapping, which this is not.



Three results fall out, and they matter more than the headline.

**Four times the data buys +0.008.** This is the learning curve of §9 confirmed
on real rows rather than extrapolated: the assay saturates near 20,000 guides.
Anyone planning a larger screen to get a better predictor should read that line
first.

**The label was hiding the effect, not damping it.** Family A is worth +0.080 ρ
on the published scores and **+0.164 on the curated scores, on the same rows**.
§9 measured label curation as worth +0.003 *to the baseline* and concluded it
was empty — that conclusion stands for the baseline and is wrong for family A.
Curation does not make the model better directly; it makes a real biological
signal visible. Those are different claims and only the second one is
interesting.

**The long-range half is the bigger half — once the label is clean.** Windowed
composition out to 1 kb alone reaches 0.660, against 0.618 for the immediate
±10 nt. On the noisy label the ranking was the other way round (§6: +0.049 local
vs +0.026 long-range). The signal furthest from the cut site is the one that
survives curation best, which is what you would expect if it is a smooth
property of the locus and the local signal is partly noise-coupled.

### 15a. The selection stage is the cost, and it is replaceable

The expensive stage is not the model. It is an XGBoost fit over all 6,517
columns whose *only* output is a feature ranking — the fitted model is
discarded — refitted inside every fold, because selecting on all the data would
leak validation rows into the choice of columns. Five selectors, same folds,
same downstream LightGBM champion (`selector_comparison.csv`):

| selector | ρ | vs current | select | total/fold | speed-up |
|---|---:|---:|---:|---:|---:|
| **LightGBM gain, 300 trees** | **0.7033** | **+0.0008** (p = 0.029) | **3.25 s** | 4.87 s | **2.3×** |
| XGBoost gain, 100 trees | 0.7007 | −0.0018 (p = 0.12) | 5.95 s | 7.78 s | 1.5× |
| XGBoost gain, 300 trees (current) | 0.7025 | — | 10.29 s | 11.43 s | 1.0× |
| variance filter (label-free) | 0.6968 | −0.0057 (p = 0.006) | 0.07 s | 1.94 s | 5.9× |
| \|correlation\| filter | 0.6751 | **−0.0274** (p < 0.001) | 0.55 s | 2.37 s | 4.8× |

**Swapping XGBoost for LightGBM in the selector is free**: 3× faster at
selection, accuracy indistinguishable (+0.0008). Not adopted as the default
yet — `selector_params` is marked "identical to the frozen baseline script",
and changing it would mean every number in this document was produced by a
selector the code no longer describes. It should be switched deliberately, with
the headline arms re-run to confirm nothing moves.

**The univariate filter is the worst option**, by a wide margin. Ranking
columns by their own correlation with the label loses 0.027 — so the gain
ranking is capturing conditional structure that single-column association
cannot see. The feature that predicts best alone is not the feature a tree
wants.

#### What the variance filter means, and what it does not

A filter that never looks at the label loses only 0.006. That is not evidence
that selection can ignore the label — it is an artifact of scale, and a
restatement of the matrix's redundancy.

| | QCT | family A | V indicators | median variance of its picks |
|---|---:|---:|---:|---:|
| variance-selected | 148 | 127 | 25 | **14.02** |
| gain-selected | 98 | 77 | **125** | **0.066** |

The two sets **share only 114 of 300 columns**. Ranking by raw variance ranks
by *units*: the continuous QCT and windowed-composition columns have large
numeric ranges, while one-hot indicators are capped at 0.25 by construction. So
the variance filter sweeps up the continuous columns and discards the
indicators that gain-based selection prefers — and still lands within 0.006.

**Two nearly disjoint 300-column subsets performing equally is a statement
about the matrix, not about selection.** The signal is spread across many
correlated columns, which is the same conclusion as §4 (three positional
families, one variable), §5 (thermodynamics is GC) and §13 (nothing correlates
with the residuals above 0.035). *Which* 300 columns you take matters far less
than taking 300 of a reasonable kind.

It also belongs beside §11's interpretability result. That section showed 19.8
of the *permuted* family A columns were still selected as "important"; this
shows a selector that never sees the label scores within 0.006 of one that
does. Both say selection is weak evidence about what matters.

### Why a cheap CPU tree matches a GPU-trained network

crisprHAL 2 already feeds on 189 nt upstream and 166 downstream, so it is not
starved of context. The difference is how the context is encoded. §13 measured
both on the same rows:

| encoding of the surrounding DNA | Δρ |
|---|---:|
| raw bases into a conv + BiGRU (±100 nt) | +0.019 |
| windowed composition (GC, purine, runs, tensors) | **+0.164** |

**The flank effect is a smooth compositional gradient, not a motif.** A 5 nt
convolution is built to find local patterns and has to approximate "the GC
content of the next 500 bases"; a windowed mean computes it exactly. That is a
statement about the biology, and it is the mechanistic reason the cheap model
keeps up.

## 15. Does the flank effect transfer? One clean answer, one dataset that cannot answer

The effect in §14 was established on one screen: *E. coli*, WT-SpCas9. Is it a
property of the DNA, of that enzyme, or of that genome? crisprHAL ships three
further screens (`src/sgrna/transfer.py`, `results/transfer.csv`). Only one of
them turns out to be able to answer, and establishing *which* is itself a
result.

| screen | organism | nuclease | context shipped | usable? |
|---|---|---|---:|---|
| WT-SpCas9 | *E. coli* | SpCas9 | 378 nt | §14 |
| eSpCas9 | *E. coli* | **eSpCas9** | 406 nt | **yes** |
| TevSpCas9 | ***C. rodentium*** | TevSpCas9 | 37 nt | partly — see below |
| TevSaCas9 | *C. rodentium* | TevSaCas9 | 29 nt | no: SaCas9 reads NNGRRT, so only 45% of its sites are NGG and this pipeline's position labels would be misaligned |

### Check the screen before trusting the screen

Flank features are positional, so §8's rule applies: they must be validated
under grouped cross-validation. Applying `arc5` to *C. rodentium* failed —
**98% of its 25,210 guides fall in a single 500 kb arc**, which produced a
one-group fold and a `nan`. That failure is informative:

| screen | guides located | genomic spread | density |
|---|---:|---|---:|
| eSpCas9 | **59,489 / 59,489** in NC_000913.2 | **all five arcs** | genome-wide |
| TevSpCas9 | 25,210 / 25,244 in NC_013716.1 | **229 kb — 4.3% of the chromosome** | **110 guides/kb** |

**The *C. rodentium* screen is not genome-wide, by design.** Ham et al. state
it: *"A 236-kb fragment of the Citrobacter rodentium DBS100 genome was screened
for 5'-NGG-3' PAM sequences and 31,596 sites were identified."* Our independent
measurement of the shipped data — a 230 kb span — matches that to within the
fragment's own length, which is a useful mutual check but not a discovery.

The consequence is specific: within 236 kb, flank composition barely varies, so
there is almost no spread for a *correlation* to work with (flank GC against
the label: **−0.022**, against +0.159 and +0.138 in the two *E. coli* screens).
**A correlation test on this screen cannot reveal a long-range gradient.**
Whether a *model* can still use one is a different question with a different
answer — see the transfer results below, which is the point of running both.

### The cross-enzyme result: solid, and the shape transfers

eSpCas9 is an engineered high-fidelity SpCas9 — different enzyme, same
organism, genome-wide. Same protocol as §14; `arc5` holds out a whole fifth of
the chromosome per fold.

| arm | random-row ρ | **arc5 ρ** | arc5 Δ |
|---|---:|---:|---:|
| base | 0.6804 | 0.6849 | — |
| + immediate ±10 nt (260 cols) | 0.7222 | 0.7278 | +0.0429 |
| + windowed composition 50–1000 nt (88 cols) | 0.7562 | **0.7610** | **+0.0761** |
| + all of family A (348 cols) | 0.7837 | **0.7892** | **+0.1043** |

**Strict and loose CV agree to three decimals**, exactly as family A did in
*E. coli* WT (+0.0867 under both). The gain is +0.1043, every fold improving
(p = 7.4e-06), and **the long-range half again beats the local half**
(+0.0331, 5/5 folds, p = 0.0012) — the same ordering as §14's 0.660 vs 0.618.

So the compositional-gradient finding is **not an artifact of SpCas9**. Within
*E. coli* it survives changing the enzyme, in both size and shape.

### The *C. rodentium* result: real, but not the claim it looks like

Grouped by 10 kb blocks (38 occupied), which is the finest honest scheme this
screen supports:

| arm | random-row ρ | **bin10k ρ** | bin10k Δ |
|---|---:|---:|---:|
| base | 0.7066 | 0.7039 | — |
| + immediate ±10 nt | 0.7402 | 0.7372 | +0.0333 |
| + windowed composition | 0.7399 | 0.7364 | +0.0325 |
| + all of family A | 0.7662 | 0.7641 | +0.0602 |

The gain is real and survives blocking (+0.0602, 5/5 folds, p = 0.0051) — a
leakage explanation was tested and **rejected**. But the long-range half no
longer beats the local half; they are identical (−0.0008, 3/5 folds, p = 0.94),
and no single long-range column correlates with the label above **0.035**, the
dust level of §13.

Read on its own, this says family A's features help within one locus of another
organism and nothing more. The model-transfer arm below shows that reading is
incomplete: the long-range signal *is* present in this data, it simply cannot
be found by looking inside a 236 kb window.

### Why, from the mechanism side

Spearman of GC content against the cut score, same measurement on all three:

| screen | **target** GC | flank GC, 250 nt downstream |
|---|---:|---:|
| WT-SpCas9 | −0.147 | **+0.159** |
| eSpCas9 | −0.107 | **+0.138** |
| TevSpCas9 | **−0.379** | **−0.022** |

Two mechanisms, behaving differently:

- **The target-GC penalty replicates everywhere and is strongest in
  *C. rodentium***. This is §6's strand-invasion claim: cutting is limited by
  prying the duplex open, which is physics and should not care about the
  organism or the enzyme variant. It does not.
- **The flank-GC benefit is present in both *E. coli* screens and absent in
  *C. rodentium*** — consistent with the span problem rather than with biology:
  across 229 kb there is no gradient to detect.

### What can and cannot be claimed

- **Can:** the flank effect, in both magnitude and shape, is a property of the
  DNA rather than of SpCas9 — demonstrated on a genome-wide screen with a
  different nuclease, under the strictest CV this project uses.
- **Can:** the target-GC/strand-invasion mechanism replicates across two
  organisms and three nucleases.
- **Can, via transfer rather than correlation:** that the long-range gradient
  holds in *C. rodentium* — the long-range columns carry the cross-organism
  transfer in both directions (below). What the 236 kb screen cannot do is
  *reveal* that gradient from within itself.
- **Cannot:** that a model trained on one organism ranks another's guides. Every
  arm here is trained and tested within its own screen.

### Model transfer: train on one organism, rank another's guides

Every arm above trains and tests inside one screen, so none of them answers the
question a reader actually assumes: does a model built here work there?
`transfer_model` fits on all of one screen and predicts all of another
(`results/transfer_model.csv`).

| train → test | base | + family A | family A adds | target's own within-screen ρ | retained |
|---|---:|---:|---:|---:|---:|
| *E. coli* WT → ***C. rodentium*** | 0.626 | **0.700** | **+0.074** | 0.764 | **92%** |
| ***C. rodentium*** → *E. coli* WT | 0.478 | **0.628** | **+0.150** | 0.707 | **89%** |
| *E. coli* eSp → ***C. rodentium*** | 0.587 | 0.656 | +0.069 | 0.764 | 86% |
| *E. coli* WT → *E. coli* eSp | 0.591 | 0.689 | +0.098 | 0.789 | 87% |

**Cross-organism transfer works.** A model trained only on *E. coli* ranks
*C. rodentium* guides at **ρ 0.700**, 92% of what a model trained on
*C. rodentium* itself reaches. The two guide sets are sequence-disjoint and the
organisms differ, so this is generalisation rather than recall.

#### Which half of family A survives the crossing

| train → test | base | + local | + long-range | + full A |
|---|---:|---:|---:|---:|
| *E. coli* → *C. rodentium* | 0.6261 | 0.6614 (+0.035) | **0.6647 (+0.039)** | 0.7002 (+0.074) |
| *C. rodentium* → *E. coli* | 0.4785 | 0.5365 (+0.058) | **0.5855 (+0.107)** | 0.6282 (+0.150) |

**The long-range half carries the transfer in both directions**, and dominates
in the harder one: crossing out of *C. rodentium*, 88 long-range columns are
worth +0.107 against 260 local columns' +0.058 — 71% of the total family A
gain. In the forward direction the two halves are near-additive
(0.035 + 0.039 ≈ 0.074), so they carry largely independent information.

The second row is the striking one. That model saw only 25,210 guides from a
**single 236 kb window**, where flank GC correlates −0.022 with the label — and
its long-range features still rank guides across an entire other genome at
+0.107 over the published representation.

**So a relationship can be learnable from data too narrow to reveal it.** A
correlation test needs spread; a fitted function does not, to the same degree.
Within 236 kb there is too little variation in flank composition for a
correlation to register, yet enough to constrain a function that extrapolates.
Discovery and verification have different data requirements, and conflating
them is what made the within-screen reading above look conclusive when it was
not.

#### Does the *E. coli* model therefore beat *C. rodentium*'s own?

No — but the part that matters does. Both numbers below are out-of-sample on
the same 25,210 *C. rodentium* guides:

| predicting *C. rodentium* | its own model | the *E. coli* model |
|---|---:|---:|
| base features only | **0.7039** | 0.6261 |
| + full family A | **0.7641** | 0.7002 |
| *gain contributed by family A* | *+0.0602* | ***+0.0741*** |
| *of which long-range* | *+0.0325* | ***+0.0386*** |

The local model wins overall by 0.064, and **all of its advantage sits in the
base features** (+0.078): it knows that screen's own label scale, its nuclease
and its 20-mer-to-score mapping. But the *E. coli* model extracts **more** from
*C. rodentium*'s flanks than *C. rodentium*'s own model does — +0.074 against
+0.060 overall, +0.039 against +0.033 long-range.

That reading suggested the flank relationship might be *better* learned from a
genome-wide screen in a different organism than from a narrow one in the right
organism, and therefore that importing it should improve the local model. **It
does not** — and the comparison above is not sound: the two gains sit on
different baselines (0.626 against 0.704), so the model with more room to
improve shows the larger gain for free. The hybrid below is the clean test.

#### The hybrid: import the flank relationship, keep the local labels

Stage 1 trains on *E. coli* with **family A alone** — no base columns, so the
prior carries the flank relationship and nothing about that screen's label
scale or nuclease — and scores every *C. rodentium* guide. Stage 2 is ordinary
`bin10k` CV on *C. rodentium* with that one extra column. No target label
enters stage 1 (`hybrid.csv`). The prior alone correlates with the
*C. rodentium* label at ρ 0.227.

| arm | ρ | against the arm above |
|---|---:|---|
| base | 0.7039 | — |
| **base + prior** (one imported column) | **0.7513** | **+0.0474**, 5/5 folds, p = 0.0039 |
| base + family A (348 local columns) | 0.7641 | +0.0128, 5/5 folds, p = 0.029 |
| base + family A + prior | 0.7644 | **+0.0002**, 3/5 folds, p = 0.80 |

**The hypothesis is rejected.** Importing the relationship on top of the local
flank features is worth +0.0002 — nothing. Whatever the *E. coli* model learned
about flanks, *C. rodentium*'s own 236 kb already supplies.

**But one imported number carries 78% of what 348 local columns do** (+0.047 of
+0.060), losing to the full local block by only 0.013. That is a compression
result, not an information result: the two are redundant, and the
*E. coli*-trained function packages the same signal into a single feature.

Which sharpens the conserved-relationship claim rather than weakening it. If
the flank-to-efficiency function were organism-specific, a prior fitted on
*E. coli* could not substitute for 348 locally-computed columns at 78%
strength. It can. The relationship is shared between the two organisms; the
narrow screen is simply not short of it.

**Family A is what makes the model portable**, in every direction and most
dramatically in the hardest one: *C. rodentium* → *E. coli* starts at 0.478 on
the published representation alone and reaches 0.628 once the flank features
are added (+0.150). The flank encoding is not merely locally informative; it is
the part that survives a change of organism.

**One arm is not a generalisation test and should not be quoted as one.** The
WT and eSpCas9 screens share **100% of their guides** — the same library scored
under two nucleases — so WT → eSp is label transfer with the sequences already
seen in training. It is reported because it isolates the enzyme change
perfectly, holding sequence constant, but it says nothing about unseen DNA.

### This resolves the tension in the two sections above

The long-range gradient **cannot be learned from** *C. rodentium*'s own screen:
229 kb gives nothing to learn it from, and flank GC correlates −0.022 with the
label there. Yet a model that **learned that gradient in *E. coli* transfers it
successfully** to *C. rodentium* guides.

Both are true, and together they are a stronger statement than either alone:
**the compositional relationship generalises across organisms even though one
organism's dataset is too narrow to discover it in.** The limitation measured
earlier is a property of the available *data*, not of the biology — which is
precisely what the model-transfer arm can distinguish and the within-screen
arms cannot.

So the project's standing on the species question, stated exactly:

- **Shown:** a flank-aware model trained in *E. coli* ranks another organism's
  guides at ~90% of a locally-trained model, in both directions.
- **Shown:** family A is the component responsible for that portability.
- **Still not shown:** that the long-range gradient is itself detectable in
  another organism — that needs a genome-wide screen outside *E. coli*, which
  neither this project nor the published alternatives have.

### The GC mechanism is not absent in human cells — it is a different shape

Measured on both matrices, fitting GC and GC² against the cut score
(`gc_shape.csv`, `gc_shape_vertex.csv`):

| | linear ρ | linear R² | quadratic R² | curved term | vertex GC | vertex sits at |
|---|---:|---:|---:|---:|---:|---|
| human | +0.017 | 0.00019 | **0.00423** | F = 70.8, p < 10⁻¹⁵ | **0.571** | the **60th** percentile |
| *E. coli* | −0.201 | 0.0407 | 0.0452 | F = 64.7, p < 10⁻¹⁵ | 0.294 | the **1st** percentile |

Both have a statistically significant curved term; what differs is **where the
turning point falls relative to the data**. In human guides it is interior — a
genuine optimum at GC ≈ 0.57, with decile means rising 0.235 → **0.262** and
then falling to **0.192** at GC 0.78. In *E. coli* the vertex sits below the
1st percentile, so across the bulk of the data the relationship is monotonic
decline.

**The quadratic fit explains 22× more of the human variance than the linear one**
(0.0042 against 0.0002), which is exactly why §15's linear ρ of +0.017 read as
"no relationship at all". It upgrades the cross-kingdom finding from *the
mechanism does not transfer* to **the mechanism is replaced by one of a
different shape** — and it is consistent with the published human literature
(an optimum near 40–60% GC, with no significant *linear* association).

### Where it stops: the human boundary

Noshay et al. published a *H. sapiens* matrix alongside the *E. coli* one, in
the same format — **6,216 of the columns are shared**, so a model can be moved
between them with no change of representation at all. That makes the
cross-kingdom test unusually clean: anything that fails cannot be blamed on
mismatched features. 17,421 human guides against 13,880 *E. coli*
(`results/human_transfer.csv`).

| arm | ρ |
|---|---:|
| *E. coli* within-screen CV (harness check against the frozen 0.5278) | **0.5305** |
| human within-screen CV | 0.4043 |
| ***E. coli* model → human** | **−0.048** |
| **human model → *E. coli*** | **−0.017** |

**Both directions are at or just below zero** — not weak but useless, and
fractionally worse than random. This reproduces crisprHAL's reported −0.2 to
0.1 for human models applied to bacteria, now measured both ways on an
identical feature set.

#### The mechanism inverts, which is why transfer is negative rather than weak

| | *E. coli* | human |
|---|---:|---:|
| GC content → cut score | **−0.201** | **+0.017** |
| melting temperature → cut score | **−0.201** | **+0.017** |
| guide secondary structure → cut score | +0.071 | +0.028 |

§6's strand-invasion result — GC-rich targets cut worse because prying the
duplex open is rate-limiting — **has no counterpart in human cells**. A model
trained on *E. coli* therefore carries a rule ("avoid GC-rich targets") that
is inert in human data, which produces a slightly negative transfer rather
than a merely small one.

That the project's single strongest mechanism is the one that fails to cross is
informative: strand invasion is limiting *in bacteria*, and whatever limits
editing in human cells — chromatin, repair pathway choice, delivery — is not
visible in these features at all.

#### The scope statement this supports

| distance | ρ retained against a locally-trained model |
|---|---:|
| same organism, different nuclease (shared guides) | 87% |
| **different organism, same kingdom** (*E. coli* ↔ *C. rodentium*) | **89–92%** |
| **different kingdom** (*E. coli* ↔ human) | **0%** |

Everything in this project is a **bacterial** claim, and the boundary is now
measured rather than assumed. Two caveats belong with it: the labels measure
different quantities (human `cut.score` is a 0–1 fraction, the *E. coli* score
a 0–48 depletion value — Spearman handles the scale but not the difference in
what is being measured), and the human arm has no family A, since that would
require locating 17,421 guides in hg38.

### Seed robustness: every arm in this section, three seeds

Item 3 of the next-steps list. The transfer, hybrid and human arms were
originally seed 41 only. Repeating all of them at seeds 42 and 43 (grouped CV
is deterministic, so the seed varies the model's own randomness — subsampling,
column sampling, tie-breaking — not the fold assignment):

| conclusion | across 3 seeds | stands? |
|---|---:|---|
| eSpCas9 flank gain (`arc5`) | **+0.1040 ± 0.0003** | ✓ |
| eSpCas9: long-range beats local | **+0.0330 ± 0.0003** | ✓ |
| *C. rodentium* flank gain (`bin10k`) | **+0.0587 ± 0.0015** | ✓ |
| *C. rodentium*: the two halves are equal | **−0.0010 ± 0.0002** | ✓ |
| *E. coli* → *C. rodentium* model transfer | **0.6980 ± 0.0033** (92% retention) | ✓ |
| *C. rodentium* → *E. coli* | **0.6276 ± 0.0009** (89%) | ✓ |
| *E. coli* WT → eSpCas9 | **0.6891 ± 0.0003** (87%) | ✓ |
| eSpCas9 → *C. rodentium* | **0.6556 ± 0.0007** (86%) | ✓ |
| *E. coli* model on human | **−0.0474 ± 0.0014** | ✓ |
| human model on *E. coli* | **−0.0184 ± 0.0087** | ✓ |
| hybrid prior on top of family A | **+0.0011 ± 0.0008** | ✓, with a caveat |

**Nothing moves.** The largest seed-to-seed spread anywhere is 0.009 (the human
→ *E. coli* arm, whose value is ≈0 regardless), and every headline is within
0.003 of the seed-41 figure originally reported.

**The one nuance worth stating rather than rounding away.** The hybrid prior
measured +0.0002 at seed 41; over three seeds it is **+0.0011 ± 0.0008, and
positive in all three**. The conclusion is unchanged — 0.001 is an order of
magnitude below the flank effect and below the ±0.003 ablation noise floor, so
the two flank sources remain redundant rather than complementary — but the sign
is consistent rather than genuinely zero, and a single seed happened to show
the smallest of the three values.

### What the field's own cross-species evidence consists of

Worth checking before treating cross-species validation as the bar this
project fails to clear. Two corrections to how this document previously
described it.

**It is crisprHAL 1's claim, not crisprHAL 2's.** The cross-species results are
in Ham, Browne et al., *Nat Commun* 14:5514 (2023). crisprHAL 2 (PeerJ 2026),
the model benchmarked in §14, is a data-curation paper whose own repository
describes it as rebuilding "two prior *E. coli* Cas9 datasets" and producing an
improved *E. coli* eSpCas9 model.

**And the evidence is thinner than the phrase "three species" implies:**

| organism | what the data is | sgRNAs | their ρ |
|---|---|---:|---:|
| *E. coli* | training and held-out test | 45,010 / 7,821 | 0.657–0.682 |
| *C. rodentium* | a screen in that organism | 31,796 designed | 0.635, 0.612 |
| *S. enterica* | **a 2 kb fragment of its *katG* gene cloned into *E. coli*** | **~270–300** | 0.678, 0.648 |

The *S. enterica* arm is *Salmonella* **sequence inside an *E. coli* cell**: no
*Salmonella* chromosome, no *Salmonella* cellular context, ~300 guides across
2 kb. It tests tolerance of foreign sequence composition, which is a real
result, but it is not a screen in another organism. And the *C. rodentium* arm
is the same data measured above — 229 kb, 4.3% of a chromosome.

So the honest competitive statement is not "they generalise across three
species and we do not". It is: **the field's cross-species evidence is one
confined screen plus a 2 kb insert, and this project has now measured why that
is not enough to support a long-range claim.** Both sides are limited by the
same missing dataset — a genome-wide Cas9 screen in a non-*E. coli* bacterium.

*(Verified against the* Nat Commun *full text on 2026-10-02: the* C. rodentium
*fragment is 236 kb with 31,596 PAM sites, and the* S. enterica *pool is 296
sgRNAs on a pTox plasmid. The crisprHAL 2 attribution rests on that paper's
scope and its repository description; the PeerJ full text returned 403 here, so
confirm that one point from the PDF before a manuscript relies on it.)*

### Four bugs, three of which returned confident numbers

Building this module took four attempts, and the failure modes generalise:

- **`a_flank` returned zero features silently** — it skips rows whose `located`
  flag is unset, which only the genome-lookup path sets. No error. (All 33,567
  curated guides were located, so §14 is unaffected — checked, not assumed.)
- **An off-by-one in PAM detection scored 1.000 while being wrong.** NGG's two
  Gs sit at +21 and +22 from the protospacer start; testing +20 and +21 also
  scores perfectly, one base off. Caught only by requiring the detector to
  return 189 for WT-SpCas9, which `crisprhal_rerun.py` independently hard-codes.
  **A validation a wrong answer can pass is not a validation** — the same
  failure as §3's GC elimination, which tested the candidate against the
  outcome instead of the mediator. Second instance.
- **A de-duplicating `groupby` silently dropped the genomic positions**, so the
  grouped CV had nothing to group by.
- A tuple passed where an array was expected, which at least failed loudly.

## 16. Hyper-parameter tuning, and what it does to the §14 margin

§14 is untuned against untuned: our LightGBM on hand-chosen settings, crisprHAL
2 on the settings its authors shipped. Defensible as like-for-like, but it
leaves the obvious objection that the +0.0107 margin might be an accident of
which defaults suited which model.

Tuning both would settle it. Tuning theirs is not affordable here — one
configuration is ~105 min of CPU for five folds, so a twelve-point search is a
day of compute. So the affordable half was measured instead, and is reported as
an **uncertainty band rather than an improvement**: how far can tuning move
*our* number?

Protocol (`src/sgrna/tune.py`, `results/tuning.csv`). Twelve random draws over
capacity (`num_leaves`, `max_depth`, `min_child_samples`), learning rate and
`n_estimators`, subsampling and `lambda_l2`. The search runs **inside each
training fold** on an inner split, and the chosen configuration touches the
validation fold exactly once — picking by validation score would report the best
of twelve draws on the test set, which is not a held-out number. Feature
selection is done once per outer fold and shared across draws, since it is 31 s
of the 35 s fold and does not depend on the LightGBM settings.

| | ρ |
|---|---:|
| default settings (every number in this project so far) | 0.7078 |
| tuned inside the fold, searched to convergence | **0.7212** |
| **gain** | **+0.0131 ± 0.0016, 5/5 folds, paired p = 0.00005** |

### 16a. crisprHAL 2 tuned too, and what it does to the margin

`src/sgrna/crisprhal_tune.py`. Thirteen of their constructor's arguments
(learning rate, CNN filters and window, three dropout rates, four dense widths,
GRU width, epochs, batch size), their architecture untouched, draw 0 their
published configuration exactly. Searched on one inner 80/20 split of outer fold
1's training rows, then the winner refitted on all five outer folds.

Their guides, their 378-nt context, and a label verified **bit-identical** to
theirs; only the split differs (our 5-fold CV rather than their shipped
hold-out, which `crisprhal_rerun --arm their_split` separately reproduces at
0.6940 against a published 0.695).

| | default | tuned | gain | folds |
|---|---:|---:|---:|---|
| crisprHAL 2 | 0.6971 ± 0.0067 | **0.7051 ± 0.0092** | +0.0080 | 5/5, p = 0.018 |
| SLICER | 0.7082 | **0.7212 ± 0.0016** | +0.0131 | 5/5, p = 0.0001 |

Per fold (crisprHAL default → tuned): 0.6855→0.6892, 0.6988→0.7081,
0.6978→0.7124, 0.7022→0.7056, 0.7012→0.7102.

| paired comparison | mean | folds | paired *t* |
|---|---:|---:|---:|
| untuned, SLICER − crisprHAL | +0.0111 | 5/5 | p = 0.0002 |
| **tuned, SLICER − crisprHAL** | **+0.0161** | **5/5** | **p = 0.0003** |

**Tuning both widens the margin**, so it is not an artefact of default settings.

The winning configuration against their published one: learning rate halved
(5e-4 → 2.5e-4), wider convolutions (128 → 192 filters) and recurrence
(128 → 192 units, dense 128/64 → 256/128), more dropout on the CNN and fusion
branches (0.3 → 0.4), more epochs (48 → 64), half the batch (1024 → 512). A
coherent "bigger model, trained more slowly, regularised harder" pattern rather
than a lucky corner — and 3–4× the fit cost, 21 → 69 min per fold.

**The search asymmetry, and a measured bound on it.** SLICER ran 627
configurations across five folds; crisprHAL ran 10 on one inner split, because
one of its fits costs ~1,000× one of ours. SLICER's trace
(`tuning_trace.csv`) says what a 10-draw budget would have found for us:

| draws | share of total tuning gain captured |
|---:|---:|
| 5 | 47% |
| **10** | **66%** |
| 20 | 74% |
| 40 | 100% |

At 66%, crisprHAL's converged gain would be ≈ +0.012 and its tuned score ≈
0.709, leaving a lead of ≈ 0.012 rather than 0.016. Erasing the margin needs
+0.024 from their tuning — three times what it found.

**Three draws of ten scored below 0.66 and one below 0.40**, so the space
contains genuinely bad regions and a single draw's score carries real variance.
The outer-fold result is what validates the choice: the selected configuration
generalises, +0.0080 on held-out folds in 5 of 5, so the gain is not a selection
artefact even though the inner-split figure (+0.0088) may be slightly
optimistic.

Per fold: +0.0087, +0.0125, +0.0143, +0.0077, +0.0081.

**The consequence for §14 was immediate: tuning our own model is worth more than
our entire untuned lead over theirs.** So +0.0107 could never support "SLICER is
better" on its own — it is smaller than an effect available to whichever model
gets searched. That is what forced §16a, where both models are tuned and the
comparison is redone on equal terms.

### 16a. crisprHAL 2 tuned too, and what it does to the margin

`src/sgrna/crisprhal_tune.py`. Thirteen of their constructor's arguments
(learning rate, CNN filters and window, three dropout rates, four dense widths,
GRU width, epochs, batch size), their architecture untouched, draw 0 their
published configuration exactly. Searched on one inner 80/20 split of outer fold
1's training rows, then the winner refitted on all five outer folds.

Their guides, their 378-nt context, and a label verified **bit-identical** to
theirs; only the split differs (our 5-fold CV rather than their shipped
hold-out, which `crisprhal_rerun --arm their_split` separately reproduces at
0.6940 against a published 0.695).

| | default | tuned | gain | folds |
|---|---:|---:|---:|---|
| crisprHAL 2 | 0.6971 ± 0.0067 | **0.7051 ± 0.0092** | +0.0080 | 5/5, p = 0.018 |
| SLICER | 0.7082 | **0.7212 ± 0.0016** | +0.0131 | 5/5, p = 0.0001 |

Per fold (crisprHAL default → tuned): 0.6855→0.6892, 0.6988→0.7081,
0.6978→0.7124, 0.7022→0.7056, 0.7012→0.7102.

| paired comparison | mean | folds | paired *t* |
|---|---:|---:|---:|
| untuned, SLICER − crisprHAL | +0.0111 | 5/5 | p = 0.0002 |
| **tuned, SLICER − crisprHAL** | **+0.0161** | **5/5** | **p = 0.0003** |

**Tuning both widens the margin**, so it is not an artefact of default settings.

The winning configuration against their published one: learning rate halved
(5e-4 → 2.5e-4), wider convolutions (128 → 192 filters) and recurrence
(128 → 192 units, dense 128/64 → 256/128), more dropout on the CNN and fusion
branches (0.3 → 0.4), more epochs (48 → 64), half the batch (1024 → 512). A
coherent "bigger model, trained more slowly, regularised harder" pattern rather
than a lucky corner — and 3–4× the fit cost, 21 → 69 min per fold.

**The search asymmetry, and a measured bound on it.** SLICER ran 627
configurations across five folds; crisprHAL ran 10 on one inner split, because
one of its fits costs ~1,000× one of ours. SLICER's trace
(`tuning_trace.csv`) says what a 10-draw budget would have found for us:

| draws | share of total tuning gain captured |
|---:|---:|
| 5 | 47% |
| **10** | **66%** |
| 20 | 74% |
| 40 | 100% |

At 66%, crisprHAL's converged gain would be ≈ +0.012 and its tuned score ≈
0.709, leaving a lead of ≈ 0.012 rather than 0.016. Erasing the margin needs
+0.024 from their tuning — three times what it found.

**Three draws of ten scored below 0.66 and one below 0.40**, so the space
contains genuinely bad regions and a single draw's score carries real variance.
The outer-fold result is what validates the choice: the selected configuration
generalises, +0.0080 on held-out folds in 5 of 5, so the gain is not a selection
artefact even though the inner-split figure (+0.0088) may be slightly
optimistic.

Per fold: +0.0087, +0.0125, +0.0143, +0.0077, +0.0081.

**The consequence for §14 is immediate: tuning our own model is worth as much
as our entire lead over theirs.** So +0.0107 cannot support "SLICER is better".
It is the same size as an effect available to whichever model gets tuned.

**The defensible claim is parity** — two models built on entirely different
principles landing within a hundredth of each other on identical rows, with the
difference between them no larger than the difference between tuned and untuned
versions of either. §14's "small but consistent advantage" wording is withdrawn.

What remains unaffected is the cost comparison (§15a) and the fact that our
*tuned* 0.7212 is the highest number anyone has reported on these rows — which
is worth stating as exactly that, a tuned-vs-untuned comparison, and not more.

## 17. Permutation floors, per family and paired

Two problems with how the permutation floor was being used.

**It was measured on four families and quoted for all nine.** The floor is not a
property of the harness; it is a property of the family. A 348-column block and
a 13-column block differ in how many of their shuffled columns survive top-300
selection by luck, so they have different floors.

**And the delta was computed across mismatched folds.** `ablation_*.csv` stores
each experiment's mean over whatever folds it happens to have, so a family run
on three seeds was being differenced against a baseline run on five — mixing the
effect with seed-to-seed spread. Every number below is **paired on (seed,
fold)**, so the two arms always see identical data
(`src/sgrna/floors.py`, `results/permutation_floors.csv`).

| family | real Δρ | its own floor | ratio | verdict |
|---|---:|---:|---:|---|
| `a_flank` | +0.0795 | −0.0009 | 88× | clear of the floor |
| `d_mechanics` | +0.0201 | +0.0001 | 153× | clear of the floor |
| `b_energy` | +0.0074 | −0.0001 | 92× | clear of the floor |
| `c_folding` | +0.0074 | +0.0010 | 7.7× | clear of the floor |
| `f_methylation` | +0.0034 | +0.0011 | 3.0× | **marginal** |
| `h_offtarget` | +0.0003 | +0.0004 | 0.8× | **indistinguishable from noise** |

Two changes to how results are reported: **`h_offtarget` is a null result**, not
a small positive one, and **`f_methylation` is marginal**. Both were previously
listed with their nominal gains as though they were real but small.

Finding the missing control also exposed a bug worth recording: `b_energy`'s
permuted arm had run on 2026-09-12 and its 15 folds were sitting in
`folds_main.csv`, but the summary row was never written, because the resume gate
sees the fold checkpoint as complete and skips the experiment before the
summary-writing step. Same class as the gate bug fixed on 2026-09-13 — a
finished result that silently does not appear.

### What a permutation control cannot do

It asks "is this better than noise?". It cannot ask "is my biological
interpretation of this correct?" — that needs a control built into the *source
experiment*, like the untagged no-antibody arm that overturned the supercoiling
reading (§3). Stating the coverage plainly, because it is uneven:

| data source | source-experiment control available? |
|---|---|
| GapR-seq | **yes** — untagged, no-antibody and rifampicin arms; used, and it changed the conclusion |
| Hi-C (`e_nucleoid`) | no mock arm published. Checked only indirectly, by showing its signal correlates 0.77–0.96 with tracks the untagged control had already discredited — weaker evidence, and it should be described as such |
| RegulonDB / PRECISE-1K (`g_transcription`) | not that kind of data; annotation rather than a measurement with a control |
| reference genome (`a_flank`, `f_methylation`, `h_offtarget`, `i_shape`, `z_mapgc`) | not applicable — no experiment to control |

## 18. What to do next

§13 closed off three of the four candidate explanations. Model class is
exhausted (0.116 across sixteen), the guide's representation is not the problem
(a CNN ties), and the assay is not the ceiling (two screens agree at 0.810, so
~0.90 is reachable in principle). Four earlier levers were already measured and
empty (§9).

So the work is in **features outside the 20-mer that do not yet exist**, and the
diagnostics say where to look.

1. **Describe the flanks structurally, not compositionally.** The flank effect
   is currently a GC average over 250–500 nt with ρ +0.05 (§6), and a CNN
   reading the same bases does *worse*. That means the signal is a smooth
   property of the region, and the obvious candidates are physical: intrinsic
   DNA curvature and bendability (DNase-derived or dinucleotide-model), local
   duplex breathing/melting propensity, and nucleosome-free-equivalent
   accessibility. These are computable from the genome with no new dataset.

2. **Explain the read-density signal instead of using it.** §3 showed the
   `d_supercoiling` gain is DNA read density, reproduced by an untagged
   no-antibody control — so it is copy number, mappability or accessibility,
   not torsion. It is also the single largest residual correlation left
   (0.154). Whatever it really is, it is the biggest unexplained thing in the
   project, and identifying it is worth more than another library.

3. **Prune the redundant families out of the paper's model.** `b_energy` and
   `d_mechanics` are re-encodings of the matrix (R² 0.984 and 0.845 from the
   20-mer). Keep them only if the point being made is that a hand-derived
   thermodynamic summary helps a tree that cannot compute one — which is a
   real, small, honest result worth one sentence, not a feature family.

4. **Screen every future dataset before building it.** `sgrna.diagnose`'s
   redundancy test costs seconds. If a candidate feature regresses onto the
   position-specific k-mer basis above R² 0.9, it cannot add information, and
   the effort belongs elsewhere.

**That experiment has now been run — §14.** It answered both halves: more rows
are worth +0.008 and are not the constraint, while family A on a curated label
is worth +0.164 and takes the model to parity with the state of the art. The
next experiments are therefore:

**Re-run crisprHAL 2 on these folds.** Parity against a published number is
weaker than parity against a model you ran yourself. Their code and weights are
public; this settles the comparison and costs an afternoon.

**Test whether the compositional encoding transfers.** Done, in §15: it
transfers across nucleases within *E. coli*, and the species question turns out
not to be answerable with the available data — by either side. See §15's
reading of what the field's cross-species evidence actually consists of.

One framing point for the write-up. The results above make this a paper about
**what does and does not determine cutting efficiency**, with a quantitative
account of why six well-motivated feature families could not have helped. That
is a stronger contribution than another 0.02 of Spearman, and it is supported
by measurements rather than by argument.
