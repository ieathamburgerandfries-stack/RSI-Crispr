# SLICER — what determines whether a CRISPR guide cuts, in *E. coli*

**S**imple **L**ightweight **I**nterpretable **C**RISPR **E**fficiency **R**anker.
RSI09, as of 11 October 2026. **The analysis is frozen** (Part 4).

**In one paragraph.** A 2023 resource published 6,232 quantum-chemical and
positional features for 13,880 *E. coli* guides but **not the guide sequences**,
which made it unusable for new targets. We decoded it: 99% of those columns
regenerate from a 20-mer, and they turn out to be an elaborate spelling of about
twenty variables. That settles by arithmetic — not by measurement — why several
imported biophysical feature families could not have helped, since they are
deterministic functions of the guide the matrix already encodes losslessly
(Finding 1). **The signal the matrix lacks is outside the guide: DNA
composition in windows of 50–1,000 nt around the cut site, worth +0.164 ρ on the
curated cohort** — the largest effect in the project, a smooth compositional
gradient rather than a motif, local to within a kilobase, and specific to
cleavage rather than to Cas9 binding (Findings 2–3). The part that decides the
comparison with the state of the art is its **range**: inside the ±189/166 nt
span crisprHAL 2 already reads, its convolutions beat our windowed composition,
and our advantage comes entirely from the 250–1,000 nt its input never reaches —
handed that reach, it gets worse rather than better (Finding 2). The resulting
model is **statistically equivalent to crisprHAL 2 on both of its authors' own
published splits** — Δρ +0.0021, 95% CI [−0.0031, +0.0072] — at **~150× less CPU
time**, reaching a given accuracy on about **half the labelled guides**. On our
own folds we lead by +0.017; on a different assay their model transfers better.
**The claim that survives every condition is parity at a fraction of the cost,
plus better data efficiency** — not a better model (Finding 7).

**How this document is organised.** **Part 1** is seven findings, each stated as
a claim, followed by the evidence it rests on and the limit of what it supports.
They are ordered as an argument rather than chronologically: **1** decodes the
published representation and shows what it cannot contain, **2–3** establish the
one large effect, what shape it has and where it lives, **4–6** are the
supporting results — the model's material limit, what its internals will and
will not tell you, and the ceiling on the label — and **7** is the comparison
with the state of the art. **Part 2** states the three contributions and the
biology separately from the modelling, **Part 3** where the work stops,
**Part 4** the freeze and the remaining items, **Part 5** what the adversarial
phase changed.

Everything that supports a finding without being one lives in the appendices:
**A** every control, each with what it rules out and what would be believed
without it; **B** the methods each claim rests on, grouped by what they defend;
**B2/B2a/B3** the imported families that did not work, what curation actually
is, and the family table on the curated cohort; **C0/C** reproducibility and
data sources; **D** references. Null results that are null for unrelated reasons
are in B2 and B3 rather than in Part 1, because a list of things that did not
work is not a finding. `../results/RESULTS.md` indexes every claim here to the file
that backs it and the command that produced it; per-fold numbers and the
long-form commentary to 10 October are in `../results/` beside it. The
chronological working log has been retired: the shared repository's git history
is the record of what was believed when, and it does the job without a second
copy drifting out of date.

**What earns a place in this document.** A statement goes in only if it is
insightful, fully verified, and not at serious risk of retraction — and only if
it either makes a claim in Parts 1–3 or defends one. Questions that were asked,
answered and closed without moving a number are not findings and are not kept
here; they are in `docs/qna_archive.md`, which is local, for conference Q&A.

---

## Part 1 — Findings

### 1. The published representation is a re-encoding of the guide, and holds about twenty variables

5,853 of 5,887 anonymous `V####` columns are position-specific k-mer indicators,
none ambiguous; the 316 quantum columns are exact k-mer lookup tables with no
conflicts. Together **6,169 of 6,232 columns (99.0%) regenerate from any 20 nt
protospacer**, verified at **100.000000%** over 1.17 million rebuilt values.
13,879 of 13,880 guides were located in NC_000913.2 with the required PAM in the
right place; 13,877 are unique.

*Rests on:* exact agreement, not a fit. *Limit:* the ~63 hand-named columns are
not pure functions of the protospacer and are not reproduced.


**And those 6,232 columns hold about twenty variables.**

A 20-mer carries at most **40 bits**. The matrix spends 6,232 columns on it.

**Start with an inventory, because the matrix partitions exactly.** Every column
is in one of three groups and the counts add up with nothing left over:

| group | columns | what each column is |
|---|---:|---|
| **(position, k-mer) indicators** | **5,853** | is the base at position *p* an A? is the pair at *p* a CG? — mono, di, tri and tetramer |
| **quantum-chemical tables** | **316** | a looked-up physical property of the base, pair or triple at a position |
| **hand-named summaries** | **63** | guide GC, Tm, k-mer counts, PAM identity, two annotation distances |
| | **6,232** | |

Read that table and the conclusion is already visible: the first two groups are
*definitions* — given the 20 letters, every one of those 6,169 numbers follows by
arithmetic or by lookup. The question is how much of the matrix is restatement,
and the rest of this finding answers it one layer at a time.

**Layer one: 74 columns are a lossless encoding of the guide.** There are only
four bases, so the mononucleotide indicators are *position-specific*: one per
(position, base) pair over 20 positions. A complete one-hot would be 80, and
three of four determine the fourth, so 60 would do — the matrix carries **all
four indicators at 14 positions and three at the other six** (positions 3, 4, 5,
6, 7 and 20, each missing a different base). Those six omissions are not absent
bases: every dropped base occurs in 20–27% of guides. **The matrix is not even
internally consistent in its own encoding**, which is the cleanest illustration
of what "elaborate spelling" means here. Those 74 columns recover **96.4%** of
full-matrix ρ on the published cohort and 95.4% on the curated one.

**Layer two: the higher-order spellings add almost nothing, and then stop.**
Climbing the ladder with **no selection in any arm**, so these are
representations compared in full rather than candidate pools under a budget
(`curated_ladder.py`, 3 seeds × 5 folds):

| arm | columns | ρ | vs the full base |
|---|---:|---:|---:|
| mononucleotide only | 74 | 0.5198 ± 0.0021 | −0.0251 |
| + dinucleotide | 364 | 0.5331 ± 0.0010 | −0.0118 |
| + trimer | 1,510 | 0.5354 ± 0.0003 | −0.0096 |
| + tetramer (**all 5,853 k-mers**) | 5,853 | 0.5335 ± 0.0013 | −0.0114 |
| **+ the 316 quantum columns** | **6,169** | **0.5449 ± 0.0013** | — |
| 364 + family A | 712 | 0.6937 ± 0.0010 | −0.0143 |
| 1,510 + family A | 1,858 | 0.7011 ± 0.0009 | −0.0068 |
| all k-mers + family A | 6,201 | 0.7005 ± 0.0004 | −0.0074 |
| **base + family A** | **6,517** | **0.7079 ± 0.0004** | — |

Trimers buy **+0.0023** and tetramers then **cost −0.0019** (−0.0006 with family
A present) — 4,343 tetramer indicators are too sparse to carry anything. That is Finding 4 reproducing on a
second cohort (+0.0035 and −0.0028 on the published matrix).

**The one step that does buy something is the last, and it should not.** The 316
quantum columns are worth **+0.0114** over a *complete* k-mer encoding — every
mono-, di-, tri- and tetramer indicator the matrix contains — and +0.0074 with
family A also present. Every one of them is a lookup of a physical property for
the base, pair or triple at a position, so they add **no information** whatsoever.
They buy a hundredth of a correlation all the same.

**Layer three: the 63 hand-named columns are mostly the guide re-described**, so
"non-sequence" is a misnomer worth retiring (`results/portability_audit.csv`):

| | count | computable from the 20-mer? |
|---|---:|---|
| whole-guide k-mer counts (4 mono + 16 di) | 20 | yes |
| guide GC, guide Tm, guide RNA secondary structure | 3 | yes |
| PAM base indicators | 4 | yes |
| undecoded `V` columns | 34 | 19 yes, 15 no |
| **distance to the nearest gene and to the PAM** | **2** | **no — needs annotation** |

**Only two of the sixty-three need anything beyond the guide.** Split along that
line, all of their value is in the re-description (`handnamed_split.py`,
published matrix, no selection in any arm):

| arm | columns | ρ | over the base |
|---|---:|---:|---:|
| 364 sequence indicators | 364 | 0.5189 ± 0.0010 | — |
| **+ the 46 computable from the 20-mer** | 410 | **0.5254 ± 0.0013** | **+0.0065** |
| + the 17 that are not (2 annotation distances, 15 undecoded `V`) | 381 | 0.5182 ± 0.0009 | **−0.0007** |
| + all 63 | 427 | 0.5256 ± 0.0018 | +0.0066 |

**The computable 46 carry the whole gain; the 17 carry none of it.** So the
427-column result this project quoted from September is not evidence that the
matrix holds non-sequence information. And it closes the annotation question at
the guide level: the two columns that genuinely need annotation contribute
nothing. Annotation helps this task only at the *locus* scale, where
`d_supercoiling`, `e_nucleoid` and `g_transcription` are worth +0.145 between
them — and even there they overlap so heavily that they are one variable
measured three ways (Appendix B3).

**So the matrix decomposes as 74 sequence + 63 non-sequence + ~6,095
restatement**, and the restatement is not worthless: re-describing the guide in
better coordinates is worth **one to two hundredths** to a depth-limited tree,
measured four independent ways — quantum tables **+0.0114**, `b_energy`
(CRISPRoff on the 23-mer) **+0.019**, `c_folding` (ViennaRNA on the spacer and
its scaffold) **+0.014**, `d_mechanics`'s protospacer half **+0.011**, the
hand-named 46 **+0.0065**. The consistency across four families is the result,
and it puts the effect at about **+0.015**.

**"Better" is carrying that sentence, and it was tested.** If the mechanism were
simply that a depth-limited tree cannot cheaply compute a sum of position-wise
terms, then *any* such sum would pay and the content would not matter. It does
not: adding 4, 16, 64 or 256 **random** position-weight tables — structurally
identical to a nearest-neighbour energy sum, with the physics replaced by noise
— is worth **−0.0004, −0.0004, −0.0002 and −0.0015** on the curated base, three
independent draws each. Nothing at every count, and enough of them begin to cost.
So the hundredth belongs to the *particular* functions and not to re-description
as a class. **Nor is it that the tree cannot compute them cheaply**, which was
this project's reading for a time and does not survive its own control: random
*non-linear* functions of the guide — expensive by construction, and matched on
out-of-fold recoverability to the real fold energy's R² 0.316 — are worth +0.0001
to +0.0007 at every interaction order from 2 to 8. What the functions that pay
have in common is that they **track the phenotype**: `eng.dg.guide_selffold`, a
single guide-RNA self-folding energy fully determined by the 20-mer, carries a
marginal \|ρ\| of 0.102 against the label and is worth **+0.0116 by itself**,
while a random function of the same difficulty carries no association and buys
nothing (Appendix B, *Every family split the way `d_mechanics` was*). The physics
is load-bearing, and difficulty only orders the functions that already qualify.
None of it is a feature-selection artefact: on the full `d_mechanics` arm, using
every column rather than selecting 300 gives the same gain, **+0.0993 against
+0.0988** (`results/selector_control.csv`).

*That last point is often stated as a theorem and is false as one.* If the matrix
encodes the 20-mer losslessly, it is tempting to conclude that any deterministic
function of it adds nothing. **That holds for an unconstrained learner and not
for ours.** A depth-limited gradient-boosted tree over one-hot indicators cannot
cheaply compute "sum of position-wise energies"; handed that sum precomputed, it
gains. What survives of the original claim is narrower: a deterministic function
of the 20-mer adds no **information**, so it cannot raise the ceiling, and it is
a poor bet when the model can already compute it. It is not a screen (Appendix
B2), and it is not where this project's effect size lives — family A's access to
DNA outside the 20-mer is worth **+0.164**, an order of magnitude more.

**The size had to be measured on columns that really are functions of the guide,
and `d_mechanics` is only partly one.** Eight of its thirty columns —
`eng.mech.flank.{up,dn}{25,100}` — are duplex thermodynamics of the *flanking*
DNA, which the base does not contain: a reconstruction probe puts them at
out-of-fold R² ≈ 0 against the base's indicators, while nine of its ΔG/ΔH/ΔS
sums come back at **R² = 1.0000** (Appendix A). Split in two
(`no_information.py`, same base, same 5 × 5 protocol):

| arm | columns | ρ | Δρ |
|---|---:|---:|---:|
| base | 6,169 | 0.5445 | — |
| + the 22 **protospacer** columns | 6,191 | 0.5555 | **+0.0110** |
| + the 8 **flank** columns | 6,177 | 0.6418 | **+0.0973** |
| + all 30 | 6,199 | 0.6437 | +0.0991 |

**Eight columns of flanking-DNA thermodynamics carry 98% of that gain**, so the
arm once quoted as the cleanest case for representation-over-information was
mostly family A's effect wearing another family's name.

---

**Which brings the finding to its point: how few columns does the model actually
need?** Two different questions hide behind that, they have different answers,
and conflating them is how a reduction claim becomes unstable.

**Question one — selection: can the model be handed fewer *candidates*? Yes, and
it is free.** Selecting 300 columns inside each training fold against using every
column: **0.5445 against 0.5449** on the curated base, and **0.7084 against
0.7079** with family A. **±0.0005 in both directions.** The budget was never the binding constraint, so
no arm above needs a protocol caveat for it. Quadrupling the budget changes
nothing either.
**A model given 300 of 6,169 columns is at parity with one given all of them** —
that is the reduction claim that holds, and it holds on both cohorts.

**Question two — reduction: can the *representation* be replaced by a smaller
one?** On the published matrix, yes. Adding the 63 hand-named columns to the 364
indicators gives **427 columns at ρ 0.5249 against 0.5272** for all 6,232: loss
**+0.0023, 95% CI [−0.0008, +0.0054]**, non-inferior at any margin from 0.0054
upward (one-sided p = 0.035 against 0.005, which the two-sided interval sits just
outside; the margin was not pre-declared, and Appendix B audits that).
And the set is **portable**, which is what a move to another cohort needs. Of the
63 hand-named columns 26
are computable from the 20-mer and its PAM, 2 need gene annotation, and 34 are
undecoded `V` columns, 19 of them reconstructible. Dropping the 15 that are not
costs nothing, leaving **412 columns computable for any 20-mer plus the reference
genome**, landing **0.00003 ρ** from all 6,232 — non-inferior at any margin worth
naming. Going down to the 364 indicators alone *does* cost 0.0083 ρ, so the 48
portable hand-named columns earn their place rather than surviving by inertia.

**On the curated cohort, no.** With the selection caveat removed, every reduced
representation falls short of the regenerated base: 364 columns lose **0.0118**,
1,510 lose **0.0096**, and with family A present the shortfalls are **0.0143** and
**0.0068** — all past the 0.005 ρ margin that 427 passed on the published matrix.
So **"412 ≈ all 6,232" is a statement about the published matrix under a
300-column budget and does not transfer.** What does transfer is the lossless
part: 74 columns recover 95.4% of full ρ here against 96.4% there.

*Already in force, which is why this matters less than it reads.* The curated
arms carry 6,517 columns = 6,169 reconstructable + 348 flank, so the 63
hand-named columns are **already absent** there. Any claim about "the 427" on
crisprHAL's rows was never about the columns actually used.

**The defensible pair of sentences, then:** the matrix spends 6,232 columns on
about twenty variables, and a model handed 300 of them does as well as one handed
all 6,232 — but the 6,169 regenerated columns are the right *pool* to select
from, and a hand-picked 364 is not a substitute for it. Three results elsewhere
are corollaries of the same redundancy: sixteen model classes span only 0.116 ρ,
quadrupling the selected columns changes nothing, and nothing correlates with the
model's residuals above 0.031. The first of those holds on a representation that
*does* contain a large effect, which is the form that matters — on the curated
base **with** family A, eight model classes span **0.1076 ρ** against 0.116 for
sixteen without it, so adding the one family that matters does not widen the gap
between model classes at all.

**And the family is the wrong unit for any of this.** Sorting all 668 family
columns by how much of each the guide itself explains, rather than by which
folder they came in, gives the cleanest statement this project has of where its
performance lives: columns the guide determines are worth **+0.008**, columns it
does not are worth **+0.201**, and the ratio is twenty-four to one
(Appendix B).

*Limit:* "non-inferior by 0.005" was the claim, not "identical" — higher-order
k-mers do carry a little, which is Finding 4.


### 2. The DNA around the cut site predicts cutting — the one large effect, and it is a gradient

Windowed composition of the flanking DNA (GC, purine, longest run, quantum-table
averages at 50/250/500/1,000 nt each side, plus the nearest 10 bases) is worth
**+0.164 ρ** on the curated cohort we train on — **+0.156 measured on genes no
step of the pipeline has seen** (Part 4), and +0.080 on the published matrix —
larger than every other idea tested here combined, and it survives holding out
whole genes, losing nothing beyond the 6% that grouping costs a feature which
*cannot* be memorising a locus (Appendix B3). **Cas9 does not read this DNA**,
which is what makes the effect interesting rather than merely useful.

**The effect is spatially local, shown with a control that preserves genomic
texture.** Re-assigning each guide the flank track read from a position offset by
a circular chromosome shift keeps every column's distribution, smoothness and
long-range autocorrelation and breaks only the correspondence (Appendix A):

| flanks read from | ρ | gain over no flanks |
|---|---:|---:|
| **the right place** | **0.6060** | **+0.0769** |
| 1 kb away | 0.5353 | +0.0062 (8% of the effect) |
| 10 kb away | 0.5160 | −0.0131 |
| 100 kb – 2 Mb away | 0.5194–0.5229 | −0.006 to −0.010 |

At 1 kb the shifted columns are still 0.41 correlated with the real ones and only
8% of the gain survives; by 10 kb the gain is gone. So this is not an artefact of
chromosome-scale structure, and the effect is sharply localised.

**And the mechanism it points at, as a hypothesis.** The size and the
locality above say what the effect is; they do not say why flanking DNA
should matter at all. The account this project works with, and the one test
that could have embarrassed it:

*Mechanism, as a hypothesis.* Target GC correlates **−0.201** with the score;
the effect survives a change of enzyme and of organism; and the model run
backwards prefers GC-poor targets (35% against a 51% mean). That is consistent
with a duplex-opening step limited by local GC. **It is not an intervention** —
the assay measures depletion, not cutting — so it is stated as a hypothesis, and
distinguishing a strand-invasion-limited from a hybridisation-limited step would
need kinetic data.

**The GC account survives conditioning, which is the test that matters**
(`gcstrata.py --corr`, 33,567 curated guides). Target GC is a **suppressor**: it
correlates **−0.147** with the label and **+0.245** with flank GC, so it masks
the flank term. Partialling it out *raises* flank GC's association from
**+0.176 marginal to +0.221 partial**, and the sign is positive in all five
target-GC strata — **+0.425, +0.278, +0.180, +0.107, +0.068**, strongest where
the target itself is GC-poor. That is the signature of a suppressor, and the
opposite of what a confound looks like.

The same measurement on the 13,880-row published cohort gives +0.044 marginal
and +0.082 partial. Both are right for their cohort, and the 4× gap is a third
independent measurement of the curation mechanism in Appendix B2a: the flank-GC
slope is **2.96×** steeper in the rows crisprHAL added, which is what produces a
marginal correlation four times larger.

The full ablation inside each stratum, where target composition is near-constant
by construction:

| target GC | n | baseline ρ | + family A | Δ |
|---|---:|---:|---:|---:|
| 0.20–0.40 | 2,776 | 0.3353 | 0.4630 | **+0.128** |
| 0.40–0.50 | 2,776 | 0.4045 | 0.4797 | +0.075 |
| 0.50–0.55 | 2,775 | 0.4627 | 0.5221 | +0.059 |
| 0.55–0.60 | 2,776 | 0.4760 | 0.5220 | +0.046 |
| 0.60–0.85 | 2,776 | 0.4443 | 0.5023 | +0.058 |

**Family A is not target GC in disguise** — it is worth between +0.046 and +0.128
inside every stratum. *Limit:* the gain is largest where the target is GC-poor,
but it is **not monotone** (the top stratum rises again), and it covaries with
headroom — the GC-poor stratum has the lowest baseline and so the most room to
gain. So the ordering is an observation, not a dose-response curve, and it should
not be read as quantitative support for the duplex-opening account.


**It is a gradient, not a motif**

A convolutional network of the same architecture family as the best published
competitor, given raw DNA on the same rows and folds:

| sequence given | ρ |
|---|---:|
| protospacer + PAM | 0.4980 |
| ±100 nt | 0.5203 |
| ±250 nt | **0.5247** |
| ±1,000 nt | 0.5190 |

**Flat from ±100 and then declining.** Given the exact DNA the windows summarise,
the network adds **+0.019** where the windowed means add **+0.080**.

*What this does and does not show.* Not a receptive-field limit — the sequence was
there and went unused; the context branch mean-pools, so a windowed average is
representable. The barrier is **inductive bias and sample efficiency**: the prior
"the right summary of 500 positions is their unweighted mean" is free to assert
and expensive to learn from 33,567 noisy examples. Nothing here shows a
representational limit for CNNs.

**But a *tuned* CNN is not so easily beaten on the DNA it actually holds, and
this is the result that bounds the claim.** crisprHAL 2's released rows are 378
nt for a 23 nt target — **189 nt upstream, 166 downstream** — so its network
already sees that span. Restricting family A to the columns that fit inside it
(the 50 nt windows and the near-cut sequence and quantum columns; the
250/500/1,000 nt windows reach past their input and are dropped) gives a
like-for-like contest in which neither model holds sequence the other lacks:

| arm | flank columns | ρ | vs crisprHAL 2 |
|---|---:|---:|---:|
| no flank | 0 | 0.5440 | — |
| near-cut only (±10 nt) | 260 | 0.6178 | −0.079 |
| **matched to their span (±189 / ±166)** | **282** | **0.6724 ± 0.0094** | **−0.025** |
| full family A | 348 | **0.7072** | **+0.010** |

**Inside their own receptive span, their convolutions beat our windowed
composition by 0.025.** Our lead is not a better encoding of the same bases: it
comes from the 250–1,000 nt windows their input never reaches, worth
0.7072 − 0.6724 = **+0.035** on top of a matched-span representation.

*What this costs and what it buys.* It costs the reading that windowed means are
simply a smarter encoding than convolution — on equal DNA they are not. It buys
a cleaner statement of the contribution: **the signal is long-range, and the
advantage is access to it**, not a representational trick. It also makes the
in-house CNN of the table above the weaker comparison it always was — a tuned
CNN does considerably better than ours on the same span, so "CNNs cannot learn
this" was never the right conclusion, and only the sample-efficiency reading
survives.

**And then we gave their architecture the same reach, which is the test that
decides it.** Their input width is a parameter, so crisprHAL 2 was retrained at
four widths on the same rows, label and fold. cuDNN cannot allocate the GRU
workspace at 2,023 nt and their published batch of 1,024, so every width was
re-run at batch 256; the 378 nt arm at both batches brackets that cost at
**−0.011**, a fifth of the effect below.

| crisprHAL 2 input | ρ (batch 256, fold 1) | vs their 378 nt design point |
|---:|---:|---:|
| 23 nt (guide + PAM) | 0.5494 | −0.131 |
| **378 nt (as published)** | **0.6799** | — |
| 1,023 nt | 0.6602 | **−0.020** |
| 2,023 nt | 0.6299 | **−0.050** |

**Their model peaks at the width it was designed for and declines monotonically
beyond it.** Handed the DNA family A reads, it gets worse, not better — roughly
7× its across-fold sd. The head-to-head on the same fold:

| DNA both models see | SLICER | crisprHAL 2 | lead |
|---|---:|---:|---|
| guide + PAM only | 0.5473 | 0.5494 | **tied** (0.002) |
| their 378 nt span | 0.6568 | 0.6799 | crisprHAL **+0.023** |
| ±1,000 nt | **0.6944** | 0.6299 | **SLICER +0.065** |

So the three input widths tell three different stories, and only the last one is
ours: on the guide alone the two architectures are **indistinguishable**; on
their design width **theirs is better**; and at the reach family A uses, **ours
is better by more than either of those gaps** — because width helps us and hurts
them.

**What this supports is narrow and worth stating narrowly: crisprHAL 2, at its
published hyperparameters, is degraded by input wider than it was designed
for.** Not that convolutional networks cannot learn long-range composition — its
48 epochs and every other setting were chosen at 378 nt, and a re-tune at each
width has not been run (`colab/crisprhal_tune_by_width.py` is written and
pre-registers which reading each outcome supports). One fold, one seed.

*Is "as published, with more of the genome" a fair test?* Of one claim only.
Equalising the input DNA removes the information asymmetry — in fact it reverses
it, since they receive **raw** sequence over the window while family A receives a
348-column lossy summary of the same span. What it does not equalise is the
*opportunity to use* that DNA: their epochs, dropout, filters and GRU width were
all chosen at 378 nt. So this supports **"the published configuration, used as
published, is degraded by input wider than it was tuned for"** and not "a
convolutional-recurrent architecture cannot exploit long-range context".
Appendix A sets out why, and why the experiment is still worth reporting.

*Their own paper supplies part of that control, and it points the same way.*
Browne et al. (2026) searched input length **per dataset** and stopped at 378 nt
for *E. coli* SpCas9 and 406 nt for eSpCas9, reporting that the gains level off:
+0.00034 ρ per nucleotide upstream and +0.00074 downstream, carrying the model
from 0.554 on the sgRNA alone to 0.697 at 378 nt. So 378 nt is not an arbitrary
inherited setting — it is where **their** search stopped improving, and the
decline we measure at 1,023 and 2,023 nt is the continuation of a curve they had
already flattened. What remains unrun is re-tuning the other hyperparameters at
each width, which is a weaker objection than it was before this was read, and is
the one item on Part 4's list still waiting on a GPU rather than on a decision.


### 3. The effect belongs to the DNA, and transfers

#### Transfer, one variable at a time

**Transfer, stated as one variable at a time.** Each row trains SLICER on one
screen and predicts another without retraining or recalibration. *Transferred* is
what the imported model scores on the target; *local* is the same architecture
trained on the target itself, which is the ceiling a transfer could reach;
*retained* is the ratio. The last column is how much of the transferred ρ the
flank features are responsible for — the difference between the base alone and
base + family A, both transferred (`results/transfer_model.csv`,
`results/transfer.csv`, 3 seeds).

| variable changed | from → to | transferred ρ | local ρ | retained | of which family A |
|---|---|---:|---:|---:|---:|
| **nuclease** | SpCas9 → eSpCas9, same genome | 0.689 | 0.784 | **88%** | +0.098 |
| **nuclease + organism** | *E. coli* SpCas9 → *C. rodentium* TevSpCas9 | 0.698 | 0.766 | **91%** | +0.072 |
| **the same, reversed** | *C. rodentium* TevSpCas9 → *E. coli* SpCas9 | 0.628 | 0.708 | **89%** | **+0.150** |
| **kingdom** | *E. coli* → human (6,216 shared columns) | −0.039 | 0.508 | **0%** | — |
| **the assay** | Cas9 cleavage → dCas9 silencing, zero-shot | 0.061 | 0.458 | **13%** | — |

**Changing the enzyme or the organism costs about a tenth of the performance;
changing the kingdom or the assay costs all of it.** The three bacterial rows
retain 88–91% with no retraining at all, and in the hardest of them — importing a
model from a 236 kb *C. rodentium* screen into the *E. coli* genome — the flank
features carry **+0.150 of the 0.628**, nearly a quarter of everything that
arrives. That is the sense in which the effect "belongs to the DNA": it is the
part of the model that survives changing the protein and the host.

#### Three confound controls

**Three confound controls, which are not transfer tests and answer a different
question:**

| could the effect be an artefact of... | result |
|---|---|
| read depth or mappability? | no — survives a high-uniqueness stratum, +0.078 → +0.051 |
| gene ranking in disguise? | no — within-gene ρ 0.571 against 0.606 global; flanks worth **+0.061 inside genes** |
| **Cas9 binding rather than cutting?** | no — but only after a second control. Family A adds −0.009 across a dCas9 CRISPRi screen and reads **+0.268** inside its essential genes; that +0.268 is the screen's own strand rule, and given a strand indicator the family adds **+0.007** there |

*Why these tests and not others.* Each removes one way the effect could be an
artefact of this dataset rather than a property of the DNA. A new **nuclease**
changes the protein but keeps the genome, so a surviving effect is not about
SpCas9's idiosyncrasies. A new **organism** changes the genome too, so a
surviving effect is not about *E. coli*'s composition; the **human** arm
failing is the control on that control, since a model transferring everywhere
would be evidence of a methodological artefact rather than of biology. The
**uniqueness stratum** removes the mappability explanation by restricting to
guides whose reads cannot be misassigned. The **within-gene** comparison removes
the gene-ranking explanation by asking the question entirely inside genes, where
"which gene" is constant. Nothing here is an intervention, so together they still
only rule things out.

#### The cross-assay test, and what it took to trust it

**The one test that could have falsified the mechanism, and what it took to
trust it.** Finding 2's account is a duplex-opening step limited by local DNA
composition at the cut site. Every other external check here is another Cas9
*cutting* screen, which cannot distinguish that from a general property of Cas9
binding or of local chromatin accessibility — both would predict a flank effect
too. Cui et al.'s (2018) dCas9 **CRISPRi** screen can: dCas9 binds and does not
cut, and the measured quantity is fitness under transcriptional silencing rather
than survival after cleavage. The screen-wide null below is the headline; it then
survived a challenge that looked fatal, and the two tables after it are that
challenge and its resolution.

It had never been used for anything here except a single ceiling-range number, so
it was the last genuinely untouched dataset available. 76,325 uniquely-targeting
guides located in NC_000913.2, 70,298 of them inside 4,265 named genes, same
pipeline, same champion, 3 seeds × 5 folds, **whole genes held out** —
mandatory here, because 30% of this label's variance is between genes (within-gene
share 0.704) and essential genes sit 2.05 fitness units below non-essential ones,
so a model that recognised the locus would be richly rewarded:

| arm | CRISPRi (dCas9) | cutting (curated Cas9) |
|---|---:|---:|
| 20-mer base, 6,169 columns | 0.4577 ± 0.0013 | 0.5584 ± 0.0011 |
| base + family A | 0.4484 ± 0.0006 | 0.7151 ± 0.0005 |
| **family A's increment** | **−0.0093** | **+0.1567** |
| family A alone, 348 columns | 0.1333 ± 0.0013 | 0.3408 ± 0.0005 |

**Family A adds nothing to a silencing screen and +0.157 to a cutting screen,
measured the same way.** The flank-only arm is what makes that a clean null
rather than a selection artefact: adding 348 columns to a 6,169-column pool under
a 300-column budget could dilute selection, but family A on its own cannot be
diluted, and on its own it reaches 0.133 against 0.341 on the cutting screen.

**The challenge: a screen-wide average can hide the only guides that carry a
signal.** The reading above assumes the two screens differ only in whether a cut
is made. They also differ in what the *phenotype* requires: a Cas9 depletion
screen scores a guide by whether its cells die, which works at any locus, while a
CRISPRi screen scores the growth change from knocking a gene **down**, which
needs that gene to be expressed and to matter. Where it does not, the guide
scores ~0 however well dCas9 bound — and the label says that is most of this
cohort. Essential-gene guides have a label sd of **2.886** against **1.133** for
non-essential ones, and **85% of the genic cohort is non-essential**.

Split on exactly that (`cui_screen.py --essentiality`,
`results/cui_essentiality_split.csv`):

| CRISPRi subset | guides | label sd | base | + family A | family A's increment |
|---|---:|---:|---:|---:|---:|
| all genic | 70,298 | 1.463 | 0.4577 | 0.4484 | **−0.0093** |
| non-essential genes | 64,784 | 1.133 | 0.5141 | 0.5039 | **−0.0101** |
| essential genes, gene-grouped | 5,514 | 2.886 | 0.2718 | 0.5468 | **+0.2749** |
| essential genes, 100 kb blocks | 5,514 | 2.886 | 0.2805 | 0.5484 | **+0.2679** |

**Restricted to loci where knockdown has a phenotype, family A reads +0.27 in a
dCas9 screen.** The non-essential half reproduces the published null exactly
(−0.0101), so the −0.009 headline figure is an average dominated by guides whose
knockdown barely moves growth — which, taken at face value, retires the
localisation entirely.

The obvious objection is locus memorisation: essential genes cluster in the
genome — ribosomal operons above all — flank composition is autocorrelated over
kilobases, and in CRISPRi the locus largely sets the phenotype, so a model could
learn "this GC signature is the ribosomal region" and carry it to a held-out gene
next door. Holding out 100 kb blocks instead of genes is the control for that,
and the effect is **unchanged**: +0.2679 against +0.2749 across 44 blocks against
297 genes.

**The resolution: that +0.27 is not the flank effect, it is the dCas9 strand
rule.** Block-grouping answers the locus objection and raises a sharper one —
*what is family A predicting here?* Not what it predicts on a cutting screen:

* **The Finding 2 column carries nothing.** Flank GC at ±500 nt, the column the
  whole compositional account runs through, has a marginal Spearman of
  **−0.0004** with the CRISPRi label inside these essential genes. Zero, to four
  decimal places.
* **The column that does carry it is strand-asymmetric.** Family A's strongest
  column here is `eng.flank.comp.up1000.purine` at **+0.4045**. Purine content
  counted on the guide's own strand flips with the guide's orientation, because
  coding strands are purine-rich; it is a proxy for *which strand the guide
  targets*, not for composition.
* **And that is the dominant covariate of this assay.** Coding-strand guides
  average a fitness score of **−5.07** against **−1.19** off it, and their mean
  flank purine differs accordingly (0.4845 against 0.5137). dCas9 blocks
  transcription far more effectively from one strand than the other; this is a
  known property of CRISPRi and has nothing to do with Finding 2.

Hand the model the strand indicator directly and family A has nothing left to
add (`cui_screen.py --strand`, `results/cui_essential_strand.csv`, same
seeds, 100 kb blocks):

| inside the 5,514 essential-gene guides | columns | ρ |
|---|---:|---:|
| 20-mer base | 6,169 | 0.2805 |
| base + family A | 6,517 | 0.5484 |
| family A alone | 348 | 0.5227 |
| **which strand the guide targets** | **1** | **0.6621** |
| base + strand | 6,170 | 0.6506 |
| **base + strand + family A** | **6,518** | **0.6575** |

**One column beats the entire flank family, and once it is present family A is
worth +0.0069** under block grouping and **+0.0148** under gene grouping — both
an order of magnitude below the +0.157 the same family is worth on a cutting
screen. A position-only control rules out the other obvious explanation: the bare
genomic coordinate buys +0.0394 under block grouping, so the 348 columns were not
acting as a location lookup (`results/cui_essential_position.csv`).

**So the null holds where it matters, and it now has a positive control it
previously lacked.** In the half of this cohort where the phenotype is
measurable, with the assay's own dominant covariate supplied, flanking
composition adds **+0.0069** (blocks) and **+0.0148** (genes) against **+0.157**
on a cutting screen measured the same way. And the screen demonstrably *can*
detect a guide-positioning effect — it detects one at ρ 0.66 from a single column
— so an accessibility- or binding-limited flank effect of the size family A
shows on a cutting screen would not have gone unseen here.

*What this still is not.* The informative subset is 5,514 guides across 44
blocks, the control rests on one strand indicator, and nothing here is an
intervention — so this localises the effect to a step that requires a break
rather than proving a mechanism. It is the strongest form of that claim the data
support, and it was nearly lost: on the screen-wide average alone the
localisation was right by accident, for a reason the average could not show.

Two qualifications on the original null still stand. The 0.133 flank-only figure
is **not zero** — flanking composition does carry information about CRISPRi
fitness, plausibly because a 2 kb window around a gene says something about the
gene. And this is a different enzyme, library, assay and laboratory at once, so a
null has many possible authors; what makes it interpretable is that the *base*
predicts 0.458
on the same rows, so the pipeline plainly works on this screen and the null
belongs to family A rather than to the run.

#### What the predictor is therefore for

**Does cleavage-specificity make the predictor useless for real editing work?**
It is the first thing to ask after the dCas9 result, and the answer turns on
separating the assay's *readout* from the quantity being predicted. In these
pooled screens a guide's score falls because the cell died, and the cell died
because Cas9 cut the chromosome — so death is the **readout for cutting
efficiency**, not the thing anyone wants. The predicted quantity is "how
efficiently is this target cleaved", which is exactly what a genome-editing
experiment needs to maximise.

So the scope is narrower than "CRISPR" and wider than "killing bacteria":

| application | does this predictor apply? |
|---|---|
| gene knockout, HDR knock-in, any edit that needs a double-strand break | **yes** — this is the quantity being predicted |
| pooled knockout screens using Cas9 cutting as the perturbation | **yes**, and these are the screens it was fitted on |
| CRISPRi / CRISPRa with dCas9 | **no** — measured, −0.009 across the screen and +0.007 where its phenotype is measurable, and a guide-sequence model is the right tool there |
| base and prime editing (nickase-based) | **untested**, and genuinely uncertain — these bind and nick rather than cut, which puts them between the two cases |

In each of those the quantity wanted is cleavage efficiency — whether a knockout
takes, how many guides per gene a library has to carry, whether a gene scored
non-essential because its guides cut badly — which is what the screens measure
through survival and what this model predicts. **The dCas9 null is therefore a
specification, not a defect**: it says which library a user should and should not
apply this to, and it is the reason the flank columns are worth building at all
— a feature set that helped equally in both assays would be describing
accessibility, which is already well covered. The one honest gap is the nickase
row: nothing here speaks to base or prime editors, and the Cui result makes it a
real question rather than a safe extrapolation.

For anyone whose application *is* silencing, the comparison in Finding 7 is the
practical guidance: crisprHAL 2's raw-sequence model transfers to a CRISPRi
screen about half again as well as ours (\|ρ\| 0.087 against 0.061), because what
survives that change of assay is guide-sequence effects and not flanking DNA.


### 4. The model is at its material limit

Measured through the top-300 selector, adding trimers *hurt*. Without selection it
helps, paired over 3 seeds × 5 folds:

| contrast | mean Δρ | folds | paired *t* |
|---|---:|---:|---:|
| **+ trimers** | **+0.0035** | 12/15 | **p = 0.025** |
| **+ tetramers on top** | **−0.0028** | 4/15 | **p = 0.013** |
| net | +0.0007 | 8/15 | p = 0.62 |

Both are significant and they cancel. The usable order stops at three: 4,343
tetramer columns for 256 possible 4-mers per position are too sparse to carry
anything. **A conclusion drawn downstream of feature selection is a statement
about the selector until proven otherwise** — this ladder produced three confident
wrong answers before that was understood.


**The model is out of material, and the reduction does not change that**

Correlating every column against the champion's out-of-fold residuals:

| model | strongest \|ρ\| among columns it has | among columns held back |
|---|---:|---:|
| all 6,232 | 0.028 | **0.147** — family A |
| the 427 | 0.022 | **0.146** — family A |
| 427 + family A | 0.028 | 0.070, none above 0.10 |

Rows one and two are the point: **dropping 5,805 columns did not change what the
model is missing.** The test is trustworthy because it has a positive control — it
flagged family A at 0.147 *before* family A existed, and family A was then worth
+0.080.


### 5. On a redundant matrix, importance and faithfulness are both uninterpretable

Across seven model families (two boosters, two forests, two linear, CatBoost):

| method | agreement between models, per column | per block |
|---|---:|---:|
| the library default (gain / impurity / coef) | **0.050** | 0.531 |
| permutation | 0.258 | 0.581 |
| **SHAP** | **0.496** | **0.800** |

The instability is a property of the matrix, not of boosting — it survives bagged
forests and penalised linear models. But **the "models prefer different kinds of
column" finding was an artefact of split gain**: XGBoost's share on binary
indicators falls from 57% to 31% under SHAP and lands on LightGBM's 31% exactly.
The largest between-model gap on any kind of column goes from 31 points to 0.3.

*Consequence:* report SHAP, at block level, and never gain.


**The standard faithfulness probe is uninterpretable on a redundant matrix**

"Drop the top features and refit" returns ≈0 on the full matrix for every model,
and that is a property of the matrix rather than of the attributions. The fix is
to match what is removed rather than how many columns are removed. **Why the
naive version fails:** on a matrix where
6,095 columns restate 20 variables, dropping the top *k* columns removes almost
no information, because near-duplicates remain to carry it. The probe then
reports ≈0 for every model and every representation, which is a property of the
redundancy, not of the attributions. Equalising the *share of attributed
importance* removed makes the comparison about what each representation actually
loses. At a matched share (~35%),
the full matrix loses 0.002 and the 775-column reduced set loses **0.015** — six
times the damage for the same share of the explanation removed, while both
concentrate importance almost identically (44 and 37 columns for half of it).


### 6. The label's ceiling is 0.946, measured from crisprHAL's own replicates

**Measured for the curated label, 15 October: 0.946 [0.944, 0.948]**, and the
route is a replicate-agreement one rather than the inherited figure
(`curated_ceiling.py`). crisprHAL 2's paper reports no replicate statistics, but
their **per-replicate counts are published** — two Cas9 replicates and two
dead-nuclease controls. Pairing Cas9 replicate 1 with control 1 and replicate 2
with control 2 gives two independent estimates of the same quantity, agreeing at
**ρ 0.8097**; Spearman–Brown makes the released two-replicate score's reliability
**0.8948**, and its square root is the highest ρ any predictor can reach against
it. The other pairing gives 0.9453, so the figure does not depend on how the
replicates are matched.

Three things make it trustworthy, and they are why this is worth more than the
inherited number:

| check | result |
|---|---|
| does their **cohort** reproduce? | their R script filters on an average control count of **73**; applying it gives **33,871** guides against the 33,567 they release |
| does their **label** reproduce? | a plain CPM log-ratio of the two conditions correlates **\|ρ\| = 0.957** with their released score, so these are the replicates behind that label |
| is it conservative? | **yes** — ALDEx2's posterior shrinks noisy guides toward the prior, which can only raise reliability above a plain log-ratio's, so 0.946 is a **lower bound** |

So the curated label is *cleaner* than the published one — **0.946 against
0.937** — which is exactly what their paper claims curation does, now measured
from their own counts rather than taken on trust. The published figure came from
a different route (Guo's tiling library, not replicates) and should not be reused
for curated numbers; that derivation is in Appendix B.

So SLICER at ρ 0.707–0.721 leaves **about 0.22 of ρ unaccounted for** — 0.225
against the curated label's measured 0.946, and the gap is not a rounding
artefact of an inherited ceiling.

---


### 7. Against the state of the art: parity, and the margin depends on the question

crisprHAL 2 (Browne et al., 2026) is the state of the art for bacterial Cas9,
and the comparison with it was measured under four conditions over three days.
**They do not agree, and the disagreement is the result** — a single headline
number here would be a choice of condition dressed as a finding.

| condition | what is held equal | result |
|---|---|---|
| **our five folds, both tuned here** | rows, label, folds, and a tuning protocol we ran for both | **SLICER +0.0172** [+0.0136, +0.0209] |
| **their two splits, their published weights** | rows, label, split, and each model as its authors ship it | **parity** — +0.002 and +0.005 |
| **equal input DNA** | all of the above *plus* the DNA each model reads | **depends on the width** — crisprHAL +0.023 at 378 nt, SLICER +0.065 at ±1,000 nt |
| **a different assay, neither model trained on it** | rows, label, and that neither has seen the data | **crisprHAL +0.026** [+0.023, +0.029] |

#### Condition one: our five folds, both models tuned here

**On held-out folds of our own cohort, we are ahead by a margin that resolves.**
Each guide is predicted exactly once by a model that never saw it, so the five
validation folds give 33,567 out-of-fold predictions per model over the same
rows, and the difference can be bootstrapped paired on guides rather than across
five folds (`heldout_compare.py`): **+0.0115** [+0.0074, +0.0157] at each side's starting
configuration, **+0.0172** [+0.0136, +0.0209] with both tuned, and **+0.0043**
[+0.0005, +0.0080] pitting our untuned model against their tuned one. Five of
five folds favour SLICER. At seed 41, the seed for which both models' per-fold
figures were recorded, SLICER's starting configuration gives 0.6950 / 0.7059 /
0.7128 / 0.7121 / 0.7124 against crisprHAL's published one at 0.6855 / 0.6988 /
0.6978 / 0.7022 / 0.7012 (`results/folds_ladder_base_full+a_flank.csv`,
`results/crisprhal_rerun.csv`).

*"Both at defaults" is a courtesy, not a fact.* `tune.DEFAULT` is a
configuration inherited from the baseline script this project started from, not
LightGBM's defaults (100, 0.1, 31, −1, no subsampling), while crisprHAL's arm
really is its published configuration. **No row in this table is symmetric in
search effort**, which is why the matched search below matters.

#### Was the search unequal? A search of their own, twice

**A hyperparameter search of their own, run 16 October, does not close that gap —
it widens it, and the reason is more interesting than the number.** Given a
nested search on our folds, crisprHAL 2 lands at **0.6987** (per fold 0.6890,
0.7011, 0.7044, 0.6993, 0.6997) against our 0.7218, a paired **+0.0225, 5/5
folds, p < 0.001**. Their *published configuration won the inner search in four
of the five folds*; the one fold that chose otherwise picked a draw 0.0016 ahead
on the inner split, which is noise. So:

| crisprHAL 2 arm | mean ρ over folds | our margin |
|---|---:|---:|
| published configuration | 0.6971 | +0.0241 |
| **best configuration found by any search** | **0.7051** | **+0.0161** |
| nested search, 5–8 draws per fold | 0.6987 | +0.0225 |
| *the same search repeated, 26 October* | *0.6989* | *+0.0224* |

**The search replicates on a second GPU session.** Run again from scratch on
26 October it returns **0.6989** (folds 0.6895, 0.7012, 0.7063, 0.6983, 0.6994)
against 0.6987, with their published configuration again winning four of five
inner searches — and in both runs the one fold that chose otherwise did so on an
inner margin of 0.0016, which is noise. A deep model's search landing within
0.0002 across independent sessions is worth stating, because the usual objection
to a single tuning run is that it is a single tuning run.

**The conservative number to quote is +0.0161**, against the best configuration
anyone has found for their model on these folds. And the asymmetry that matters
is not in effort but in **tunability**: an equivalent small search on our side
buys **+0.0103** (12 draws per fold), while theirs buys nothing. Their published
settings are at or very near the optimum of the space we searched; ours are not.
That dissolves the tuning-effort objection in a way equalising effort could not —
equal effort yields them nothing.

*The searches are still not equal, and the residual is ~20× in our favour.* Our
tuned arm used **97–157 draws per fold** (627 in total); theirs explored 5–8 per
fold before patience stopped it. Matching that would be ~100 draws per fold,
about **40 GPU-hours**, and has not been run — so the margin remains formally an
upper bound, now one with a measurement behind it rather than only a caveat
(Appendix B).

#### Condition two: their splits, their published weights — parity

**On crisprHAL 2's own two splits, evaluated against their own published
weights, it is parity — and parity as a measured equivalence rather than as a
failure to find a difference.** Run once on their 11,898-guide eSpCas9 test file
(`final_eval.py`):

| | ρ | τ | Pearson | R² | top-decile | pick %ile |
|---|---:|---:|---:|---:|---:|---:|
| **SLICER** — champion XGBoost, 6,517 columns, selection fitted inside their training rows | **0.7842** | **0.5942** | 0.7782 | 0.5979 | 0.488 | 94.8 |
| **crisprHAL 2** — their shipped `ESPCAS9.keras`, as distributed | 0.7821 | 0.5920 | **0.7801** | **0.6009** | **0.494** | **95.5** |

Their arm **reproduces their published figure** — Browne et al. report ρ 0.782
and Pearson 0.780, and their weights give 0.7821 and 0.7801 here — which is what
makes the comparison quotable rather than a re-implementation of it. Paired on
the same 11,898 guides, Δρ = **+0.0021**, 95% CI **[−0.0031, +0.0072]** — so the
data support equivalence at **any margin from 0.0072 upward**, and no margin had
to be chosen to say so: no margin here was pre-declared, so the interval is
reported in place of a verdict (Appendix B audits the dates).
And on both of their splits the margin is small and ours:

| their split | their published ρ | SLICER | margin |
|---|---:|---:|---:|
| *E. coli* SpCas9, 26,830 train | 0.695 | **0.6997** | +0.005 |
| *E. coli* eSpCas9, 47,591 train | 0.782 | **0.7842** | +0.002 |

**All six metrics are individually indistinguishable from zero** — we lead on
the two rank metrics, they lead on
Pearson, R², top-decile precision and expected pick percentile, and every
interval contains zero (Appendix A). So "they win some, we win others" is the
right reading, and the reason is not the split itself but that no metric resolves
a difference at this sample size.

**The two conditions disagree because of data volume, which is a prediction the
sample-complexity curves had already made.** Our folds train on 26,853 guides;
their eSpCas9 split on 47,591. The curves say our advantage is largest where
labels are scarce and closes as they stop being (Appendix B), and the paired gap
goes +0.0133 at 26,853 to +0.0021 at 47,591. **So the data-efficiency claim
survives and the ranking claim does not.** "SLICER beats crisprHAL 2" is a claim
about the small-data regime; at 47.6k labelled guides it is false.

#### Each published model on the cohort it was built for

**The two like-for-like comparisons, each on its own home ground.** Before any
ablation: the two published models this project is measured against were fitted
on different cohorts, so each is compared on the data it was built for, with
SLICER trained on that same data.

| | the other model | SLICER, same data | margin |
|---|---:|---:|---:|
| **Noshay's cohort** — 13,880 guides, `cut_score`, 5 folds | iRF **0.4785** | **0.6060** | **+0.128** |
| **crisprHAL's cohort** — their shipped SpCas9 split | **0.695** | **0.6985** | +0.004 |
| **crisprHAL's cohort** — our five curated folds | 0.7051 | 0.7218 | +0.017 |

*The iRF row is our reimplementation of Noshay et al.'s model* (`noshay_replicate.py`,
1,000 trees, 10 iterations, same folds), and it reproduces their published figures
— Pearson 0.488 and R² 0.235 against their reported 0.502 and 0.249 — so it is
their model and not a strawman. Against it we are ahead by **+0.128 ρ on their own
data**, and that splits cleanly: **+0.049 is the model class** (champion XGBoost
on the identical 6,232 columns reaches 0.5278) and **+0.078 is family A**. Against
crisprHAL 2 on *its* data it is parity.

**So the two comparisons say different things, and both are real.** Noshay's model
is beaten decisively on its own cohort, by a margin that is mostly the flank
features; crisprHAL 2 is matched, not beaten. A paper that quotes only the first
overstates the contribution and a paper that quotes only the second understates
how far the field moved between the two.

#### What parity is bought with, holding the architecture fixed

**Hold the architecture fixed and change only the training data, and parity turns
out to be bought with rows.** The comparison above varies the condition; this one
varies the *data* and nothing else. Same champion XGBoost, same 6,517 columns,
same 300-column budget, selection fitted inside the training rows — scored every
time on crisprHAL 2's **own held-out WT-SpCas9 test partition** (6,737 of their
6,749 guides locate in our feature build; Browne et al. report **0.695** on it,
and their shipped weights re-run here give 0.6940). `data_ablation.py`,
`results/data_ablation.csv`:

| training data | n | ρ | against crisprHAL 2 |
|---|---:|---:|---:|
| all curated rows, **base only** (no family A) | 26,830 | 0.5431 | −0.152 |
| Noshay's 1,222-gene panel, **Noshay's own `cut_score`** | 6,860 | 0.6537 | −0.041 |
| the same rows, curated label | 6,860 | 0.6600 | −0.035 |
| random curated rows, count matched (3 draws) | 6,860 | 0.6683 | −0.027 |
| **all curated rows** | **26,830** | **0.6985** | **+0.0035** |

*Two numbers for one split, and they are not interchangeable:* **0.695** is what
Browne et al. report on this partition and is the reference above; **0.6971** is
their published *configuration* re-trained on our five curated folds, which is a
different split and appears only in the matched-search table. The SLICER figure
here, 0.6985, is the independently re-derived one; the recorded run of the same
setup gives 0.6997 (`their_split_slicer.csv`), which is the 0.005 quoted in the
two-split table above.

Read down the middle three rows and the move from "the data Noshay published" to
"the data crisprHAL published" decomposes: the **label** is worth **+0.0063**,
**which rows** another **+0.0083**, and **how many rows** the remaining
**+0.0302** — two thirds of the total is simply row count. **On Noshay's data
alone this architecture would sit 0.043 behind the state of the art.** Parity is
not a property of the model by itself.

Two things follow that are worth keeping apart. First, the **feature set still
dominates every data effect**: family A is worth **+0.1554** on this same held-out
partition at full data, against +0.0448 for the entire move between cohorts. The
ordering of the contributions is features, then row count, then which rows, then
the label — and the label is the part the curation paper is about.

Second, and this is the one to read carefully: **only the last row of the table
is a comparison between two models.** The rest are SLICER against itself, with
crisprHAL 2's figure printed beside them for scale — their model was trained on
the full curated training set in every case, so "−0.043" is *SLICER with a
sixth of the data against crisprHAL 2 with all of it*, and says nothing about
how crisprHAL 2 would fare on Noshay's rows. It is not evidence that their model
is better on Noshay's cohort; nothing here tests that, and the sample-complexity
curves predict the opposite, since their model degrades faster as labels get
scarce (Appendix B).

#### Size and interpretability, where there is no common unit

**On model size there is no common unit, so both are reported.** A CNN is quoted
by its trainable parameter count; crisprHAL 2's WT-SpCas9 model holds **790,533**
of them across 37 layers (823,301 for eSpCas9). A boosted ensemble has no weight
matrices, and the closest analogue is what the fit stores: **6,348 leaf values**
over 400 trees, or **18,244 numbers** counting each internal node's feature index
and threshold (`complexity.py`). That is **125×** fewer fitted values, or 43× on
the inclusive count — the same order as the compute ratio, which is not a
coincidence.

Three things say more about interpretability than any of those totals, and have
no CNN equivalent: the fit **caps interaction order at four** by construction,
**45 of the 300 offered columns carry half its gain** (280 are ever split on, the
top ten hold 19.5%), and a **ridge reproduces 95.6% of its ρ** on a decorrelated
basis — so about 4% of the accuracy needs interactions at all, and the rest reads
as a sum of per-column effects in named units. What those numbers do and do not
license, including the fact that depth 4 was inherited rather than chosen, is in
Appendix B.

**And on a screen neither model trained on, their raw-sequence model transfers
better than our engineered one** — \|ρ\| 0.0867 against 0.0609 zero-shot on Cui's
CRISPRi screen, a paired difference of **−0.0258, 95% CI [−0.0289, −0.0227]**
(`cui_final_exam.py`).
Both numbers are near-nothing against the 0.458 a model *trained* on that screen
reaches, so this compares two small quantities. The reason is specific and worth
stating rather than burying: the transferable signal in a dCas9 screen is
**guide-sequence** effects, which is what their input is, while our advantage is
flanking DNA, which contributes nothing to silencing (Finding 3). **Our
representation is the more specialised of the two**, and that is a real cost of
the design.

#### The sentence that survives every condition

*The claim that survives all four conditions*, and the sentence the paper should
use:

> Given each model its own best input and its own tuning, SLICER is statistically
> equivalent to crisprHAL 2 on both of crisprHAL 2's own published splits — Δρ
> +0.0021, 95% CI [−0.0031, +0.0072] on eSpCas9, +0.005 on SpCas9 — at roughly
> 150× less compute, and reaches a given accuracy on about half the labelled
> guides. Under equal input DNA the ranking depends on the window: crisprHAL 2 is
> better inside its own 378 nt span, and SLICER is better once the window reaches
> the 250–1,000 nt its input cannot see.

#### How to read these numbers: cost, and what ρ buys

**The nuance that "150× less compute" needs, stated here rather than buried.**
There is **no architecture-independent measure of cost**, and FLOPs is not one:
it is hardware-independent but computed *from* the architecture, so it cannot
adjudicate between architectures without already assuming what work costs. CPU
time is the honest version of the claim and it is a statement about our hardware,
not a law. The one measure that is genuinely architecture-free is **sample
complexity**, because it says nothing about how the computation is done — and it
is the stronger claim anyway, measured over all five folds:

| guides needed to reach | SLICER | crisprHAL 2 |
|---|---:|---:|
| ρ ≥ 0.62 | **4,000** | 8,000 |
| ρ ≥ 0.66 | **8,000** | 13,880 |
| ρ ≥ 0.68 | **8,000** | 20,000 |
| ρ ≥ 0.70 | **20,000** | not reached on all 26,853 |

About half the labelled guides at every threshold, and **2.5×** at ρ ≥ 0.68
(`sample_complexity.py`). So the defensible form of the cost claim is the
data-efficiency one: it does not depend on which machine either model ran on.

**What these ρ values mean on the decision scale**, since every number in Part 1
is one. ρ measures whether the ordering is right, which is the right headline
because nobody needs a guide's exact efficiency — but **ratios of Δρ are not
ratios of information**, since on the ρ² scale the same Δρ is worth twice as much
higher up, which is why this report quotes paired differences and raw ρ rather
than percentages. Translated into the choice a user actually makes: given ten
candidate guides, a model at ρ 0.71 picks one landing at the **81st percentile**
of the ten, against the 50th for blind choice and the 88th at the label's
ceiling. The whole range from useless to perfect spans the 50th to the 91st,
because ten random candidates need not contain a good one — so the distance
between this model and a perfect one is smaller, in decisions, than the distance
between ρ 0.71 and ρ 0.946 looks.

*Why the advantage exists at all is Finding 2*, which is a statement about
long-range context rather than about architecture. *The conditions that make
"fair" ambiguous* — and why the phrase should not be used unqualified — are in
Appendix A.

---


## Part 2 — Contributions

**1. A published representation decoded, and shown to be redundant.** 99% of its
6,232 columns regenerate from a 20-mer, and they spell out about twenty
variables; on the published matrix under a 300-column budget, 412 columns
computable for any guide match all 6,232 to 0.00003 ρ. The lossless encoding also
settles by arithmetic that **no deterministic function of the 20-mer can add
information** — though it can still add *performance*, +0.011 to +0.019 for a
depth-limited tree, which is a much smaller effect than this project first
claimed and the reason several imported biophysical families could not have
helped (Finding 1; Appendices B2, B3).

**2. Genomic flanking context identified and validated as the main new signal.**
**+0.164 ρ** on the curated cohort, **+0.156 on sealed genes** (+0.080 on the
published matrix) — the largest
effect here; a spatially local compositional gradient rather than a motif, shown
with a texture-preserving null that also measures its autocorrelation length;
not mappability, not gene ranking, and it survives holding out whole genes. It
transfers across nuclease and organism — 88–91% retained with no retraining —
and not across kingdoms. It does **not** appear in a dCas9
silencing screen — −0.009 across the screen, and **+0.007** inside the essential
genes where the phenotype is measurable, once that assay's own strand rule is
controlled for — which localises it to a step requiring a break rather than to
binding or accessibility (Findings 2–3).

**3. Parity with the state of the art, from a representation of long-range
context that a tuned convolutional network does not extract at this sample
size.** Statistically equivalent to crisprHAL 2 on both of its authors' own
published splits — Δρ +0.0021, 95% CI [−0.0031, +0.0072] on eSpCas9 and +0.005 on
SpCas9 — at **~150× less CPU time** and reaching a given accuracy on about **half
the labelled guides**. On our own five folds we are ahead by +0.0172
[+0.0136, +0.0209], under a search that was not matched; on a different assay
their model transfers better. **The contribution is not a faster or better
model** — it is that a boosted tree over 300 engineered columns matches a tuned
deep network at a fraction of the cost, because the signal it encodes is
long-range and the network's input does not reach it (Findings 2 and 7).

The qualifier is load-bearing, and the evidence runs both ways: given the *same*
input the two architectures are indistinguishable on the guide alone (0.5473
against 0.5494), crisprHAL 2 is **better** inside its own 378 nt span (0.6568
against 0.6799), and SLICER leads by **0.065** only once the window reaches
±1,000 nt, where their model degrades. That is an argument about *reach*, not
about architecture, and Finding 7 tabulates every condition under which the
comparison has been asked rather than choosing one.

### What is new about the biology, separately from the model

The three contributions above are partly about modelling. Stripped of everything
computational, this is what the project claims about *E. coli* Cas9 biology that
was not known before it — one positive localisation, one directional effect, and
a set of constraints that are worth as much because each kills a plausible
mechanism.

**Positive.**

1. **The DNA outside the protospacer — DNA Cas9 never contacts — predicts whether
   a target is cut**, and it is the largest known effect for this phenotype:
   +0.164 ρ, out to roughly 500 nt each side. Browne et al. had shown that a
   network reading ±190 nt beats one reading the guide; the claim here is
   stronger and different in kind — the information is **compositional and
   smoothly distributed**, with a measured autocorrelation length (8% of the
   effect survives a 1 kb shift, none survives 10 kb) and no dependence on where
   the window edges are put (+0.1634 against +0.1638 at four different widths).
   There is no motif to find.
2. **The effect belongs to a step that requires a break, not to binding or
   accessibility.** In a dCas9 CRISPRi screen on the same genome the same
   features add **−0.009** across the screen and **+0.007** inside the essential
   genes where knockdown has a measurable phenotype, once that screen's own
   strand rule is controlled for — against **+0.157** on a cutting screen. Both
   binding-limited and chromatin-limited accounts predict a flank effect in a
   dCas9 screen and neither produces one. This is the one result here that
   localises a mechanism rather than bounding one, and it is the claim that came
   closest to being lost: taken on the screen-wide average alone it would have
   been right for a reason the average cannot show (Finding 3).
3. **It is a local property of the target, not a property of its gene.** Asked
   entirely inside genes, where gene identity is constant, the flanks are still
   worth **+0.061**, and the whole effect survives holding out whole genes.

**Directional.**

4. **Orientation relative to transcription matters — Jacky's result, not this
   project's**, and listed here because the mechanism argument above leans on
   it: Δ **0.051** [0.020, 0.082] against a curated RegulonDB TSS, which
   replaced our own nearest-gene proxy after his matched comparison showed that
   version to be a cohort confound (0.007 [−0.024, 0.038]). So the surrounding
   DNA is not acting through composition alone. Something with a direction is
   involved — transcription, replication, or the supercoiling domains they
   create — and which one is unidentified. It is the most mechanistically
   suggestive loose end around this work, and it belongs in his write-up.

**Constraints — each rules out a mechanism that would otherwise be the obvious
candidate.**

5. **Not helical phasing or intrinsic curvature.** GC skew, A-tract phasing and
   bendability periodicity at the helical repeat — the most mechanistically
   motivated features buildable from the genome — are a null: −0.001 alone,
   +0.0004 on top of the flank composition.
6. **Not the physical shape of the flanks, but their letter composition.**
   Flank DNA-shape descriptors are worth +0.120 alone and **+0.0007** once
   composition is present: a reparameterisation, not a second channel.
7. **Not supercoiling**, despite the family built to test it paying +0.139: the
   untagged no-antibody GapR control predicts as well as the ChIP, so what that
   family measures is sequencing read density. Guo et al.'s "unknown chromosomal
   factors" are, in this family, coverage.
8. **Not gene essentiality — exactly as Guo et al. reported — but expression is
   not null either.** The transcription family's +0.058 is gene geometry
   (+0.0393) plus expression (+0.0213) plus essentiality (+0.00004). The
   published essentiality null reproduces to four decimal places; the published
   expression null does not.
9. **Not a restatement of target GC.** Conditioning on target GC *raises* flank
   GC's association, **+0.176 marginal to +0.221 partial**, and the flank effect
   is positive inside all five target-GC strata.
10. **And the phenotype is mostly still unexplained.** The label's reliability
    ceiling is **0.946** and the best model here reaches 0.7218, so about **0.22 ρ
    of real, repeatable biology is unaccounted for** by every feature family
    tried. That number is the most useful thing in this list for anyone deciding
    what to work on next.

**One cross-kingdom observation**, which is biology rather than bookkeeping: the
association between GC content and cutting efficiency **inverts** between
*E. coli* and human (−0.20 against +0.02), and the human phenotype is far less
additive — a linear model reproduces 96% of a boosted tree's ρ in *E. coli* and
only 64% in human. Bacterial cutting efficiency is close to a sum of small
independent effects; human cutting efficiency is not.

### Further inferences about the process, from results not collected to make them

The list above reports what each experiment was designed to find. Several things
follow from the results *in combination* that no single experiment was aimed at,
and they are the most biology-facing statements the project can currently
support. Each is given with the measurement that licenses it and the reason it
could still be wrong.

**1. "GC content affects cutting" is two effects with opposite signs, and the
window decides which one you measure.**

| | Spearman with the label |
|---|---:|
| GC of the 20-mer guide | **−0.147** |
| GC of the target (protospacer and PAM) | **−0.201** |
| GC of the flank, ±250 nt | **+0.161** |

Inside the protospacer, GC-rich predicts *less* cutting — the textbook account,
and the sign is right for it: a GC-rich duplex costs more to unwind, and Cas9
must unwind it to complete the R-loop. Outside the protospacer the same
composition predicts *more* cutting, which that account cannot produce at any
distance. The two are positively correlated with each other (+0.23 for the guide,
+0.176 for the target), so **each suppresses the other**, which is exactly why
conditioning on target GC *raises* the flank association rather than removing it
(item 9 above: +0.176 marginal becomes +0.221 partial). A single GC coefficient fitted over an unspecified window is an
average of two mechanisms pointing opposite ways, and its sign is determined by
how much flank the window happened to include.

*Why it could be wrong:* both are marginal associations on one cohort, and the
flank figure is the one that survives every control in Findings 2–3 while the
target figure is inherited. The claim is about signs and scales, not magnitudes.

**2. The family table measures how much is *new*, not how much *matters* — and
this is the single biggest gap between the informatics and the biology here.**

Every family number in this report is an **increment**: what the family adds on
top of a model that already holds the guide. Set the increments beside the
strongest single-column association within each family and they do not agree at
all:

| family | strongest single column, \|ρ\| | family increment Δρ |
|---|---:|---:|
| `b_energy` | **0.184** `eng.dg.hybridisation_weighted` | +0.018 |
| `i_shape` | 0.165 | +0.0007 |
| `a_flank` | 0.161 `eng.flank.comp.up250.gc` | **+0.164** |
| `d_mechanics` | 0.154 | +0.099 |
| `f_methylation` | 0.143 | +0.008 |
| `c_folding` | 0.119 | +0.015 |
| `h_offtarget` | 0.024 | +0.001 |

**Marginal association is nearly flat across six of the seven families — 0.12 to
0.18 — while the increments span 0.001 to 0.164.** The strongest single column in
the entire 6,232-column feature set is not a flank column at all: it is
guide–DNA hybridisation energy, which is **R² 1.00 recoverable from the guide's
own indicators and 0.90 correlated with plain guide GC**, and its family is worth
+0.018.

So the ordering this report leans on — *nothing confined to the 20-mer clears
+0.019, nothing above +0.05 is confined to it* — is **true as stated and not a
statement about biology**. It ranks families by what the model cannot already
compute. Flanking DNA dominates the increments because it is the only family
carrying information a one-hot guide does not contain; the guide-derived families
score low not because the phenotype depends on them weakly but because the model
reaches them another way. The two questions —

> *what should I add to this model?* and *what does cutting depend on?*

— have different answers, and this project has mostly been answering the first.
`h_offtarget` is the only family weak on both measures, and that is by
construction (see 6).

**3. The guide's own folding is a real determinant, and the control is what makes
that a biological claim rather than a modelling one.**

The spacer's minimum free energy is worth **+0.0119** on top of the curated base
and carries a marginal **\|ρ\| 0.102** on its own (0.119 for the thresholded
version). It is fully determined by the twenty bases the model already holds, so
on the reasoning of 2 its small increment says nothing about its importance. What
settles it is the control: the same ViennaRNA computation, the same energy model,
the same base composition and the same non-linearity, run on each guide with
**one fixed permutation of its twenty positions**, is worth **+0.0000 and
−0.0007** on two permutations. Scrambling which base sits where destroys the
entire effect while preserving everything else about the quantity.

**So it is not composition, and it is not the fact that folding is hard for a
tree to compute. It is the particular structure the real spacer forms.** A second
control says the same thing from the other direction: random *non-linear*
functions of the guide, built to be as hard for the tree as the real fold energy
and matched on out-of-fold recoverability (R² 0.34–0.48 against the real column's
0.32), are worth +0.0001 to +0.0007 at every interaction order tried
(Appendix B). Difficulty without meaning buys nothing. The mechanism this
supports is the standard one — a spacer that base-pairs with itself is less
available to invade the target duplex — and the measurement says it operates in
this cohort at a marginal strength comparable to flank GC.

*Why it could be wrong:* one cohort, and the shuffled control is conservative in
the other direction — it preserves composition, so if composition carried the
effect the control would have shown it. It did not, which is the result.

**4. The two kingdoms differ in interaction order, and two unrelated instruments
agree on it.**

A ridge model recovers **95.6%** of a boosted tree's ρ in *E. coli* and **63.7%**
in human. Independently, raising `max_depth` from 4 to 6 — which raises the
maximum interaction order the trees can express — is worth **+0.0063** here and
**+0.0585** on the human matrix. Model class and model capacity are different
instruments and give the same answer: **bacterial cutting efficiency is close to
a sum of small independent contributions, and human cutting efficiency is not.**

That fits the rest of the picture. The *E. coli* flank effect is a smooth
gradient with no motif and no window-edge dependence — the signature of a bulk,
graded property. A eukaryotic nucleosome either occludes a site or does not,
which is a threshold, and thresholds are interactions. It also carries a
practical consequence: a model developed on bacterial screens will under-fit
human CRISPR for a **structural** reason, not for want of data.

*Why it could be wrong:* the human matrix is a different feature set on a smaller
cohort, so some of the additivity gap could be parameterisation rather than
biology.

**5. The phenotype these screens measure is death, not cutting — and that bounds
what any of this can mean.**

In a pooled Cas9 depletion screen a guide scores as efficient when the cells
carrying it fail to survive. Measured efficiency is therefore
**P(cut) × P(the break is unsurvivable)**, and every model in this literature,
including this one, is fitted to the product, and no experiment here separates
the two terms. The dCas9 contrast puts the effect in the term that
requires a break (Finding 3), which is as far as sequence data reach: it does not
say whether what varies is P(cut) or P(the break is unsurvivable), so **both of
those terms remain open**. One repair route testable from sequence alone —
RecBCD attenuation at Chi sites — is **excluded** on its own merits: Chi behaves
like any other octamer of its GC content, sitting at the 41st percentile of 200
GC-matched controls (Appendix B).

This is also the sharpest answer available to the question of whether a
cleavage-specific effect is useful when the goal is editing rather than killing.
**If the effect sits in P(cut)** it transfers directly to editing applications.
**If it sits in P(break kills)** it does not predict whether Cas9 cut — but the
fate of a break is precisely what sets indel spectra and HDR rates, so it would
bear on editing *outcome* instead. On either reading it is not useless. What it
is not, on any reading, is a guide-design rule: it is a property of the target.

**6. Nothing in this project speaks to specificity, by construction.**

Six of `h_offtarget`'s thirteen columns — `mm1`, `mm2`, `seed_exact_mm1..4` — are
**zero for every guide**, because the curated cohort retains only
uniquely-targeting guides. The family's +0.0009 is therefore **not** evidence that
off-target descriptors are uninformative; the cohort defined them away. Any
reader taking a specificity conclusion from this work is reading a filter.

**7. And the single most useful number for a biologist is still the gap.** The
label's reliability ceiling is **0.946**, the best model reaches **0.7218**, and
eleven feature families built from the genome, from thermodynamics, from
ChIP-seq and from quantum chemistry have not closed **0.22 ρ of real, repeatable
biology**. Items 1 to 6 are refinements inside the quarter of the variance that
is already explained.

### What else distinguishes this model, and what does not

A fair way to ask it: in drug-property prediction, handing a model 3D structure
instead of a 2D graph is the change that moves the field. What is the equivalent
here, and is there more than one?

**There is one, and it is the flanking DNA.** It should not be filed as a small
improvement: +0.164 ρ on a base of 0.545 is a **30% relative gain**, and on
crisprHAL's held-out partition the same features are worth +0.155 against a
base-only model at 0.543 — the difference between a model that is useless for
ranking and one that matches the state of the art. The 3D-structure analogy is
apt in size as well as in kind: in both cases the representation is given
information the old one could not express, rather than a better way of fitting
the same information.

**There is a second contribution that is not an accuracy lever at all**, and it
is worth naming separately because it is what made everything else possible:
**decoding the published matrix**. 5,853 of the 5,887 opaque `V####` columns were
identified as (position, k-mer) indicators, so 99% of the feature matrix
regenerates from a bare 20-mer. That buys no ρ on Noshay's cohort — it is the
same matrix — but without it the project could not have been run on crisprHAL's
33,567 guides, on eSpCas9, on *C. rodentium* or on Cui's screen, because those
guides have no published feature row. It converts a dataset into a method.

**Everything else that was tried is a null, and the list is the useful part**:

| candidate change | measured |
|---|---|
| model class (16 classes, then 8 on the curated base) | spans 0.116 ρ, and 0.108 *with* family A — not where the gain is |
| hyperparameters | a 627-draw nested search buys +0.013 |
| feature budget (300 against every column) | ±0.0005 |
| selector (XGBoost against LightGBM gain) | 0.0008, at 3× the speed |
| higher k-mer order in the flanks | tetramers −0.0019 against dimers |
| DNA shape of the flanks | +0.0007 on top of composition |
| helical periodicity and GC skew | +0.0004 on top of composition |
| explicit interaction terms (family G's own rationale) | +0.0004 over their parents |
| wider windows (75/300/600/1,200 against 50/250/500/1,000) | 0.0005 |

**So the answer is: one representational change, one piece of infrastructure, and
nine nulls.** That is a less exciting inventory than a list of wins, but it is
the more useful one — it says that for this phenotype the lever is *what DNA the
model is shown*, and that nothing about the estimator repays attention. The same
finding is what licenses the cost claim in Finding 7: if the estimator does not
matter, use the cheap one.

Findings 4–6 support these three. They are not co-equal headlines, with one
exception worth separating for an ML audience: **Finding 5 is a methodological
result about interpretability that generalises beyond this dataset**, and would
be better served by a short companion note than by being compressed into a
biology paper.

---

## Part 3 — Where this stops

- **Everything is bacterial.** Cross-kingdom transfer is useless in both
  directions, and the two directions are not the same kind of useless. With the
  human model fitted to the human matrix rather than to *E. coli*'s
  hyperparameters, *E. coli* → human is **ρ −0.039 [−0.054, −0.024]**, clear of
  zero, while human → *E. coli* is **−0.006 [−0.023, +0.011]**, which is zero
  (`results/kingdom_transfer_boot.csv`, 2,000 test-row resamples). Both
  directions excluded zero only while the human model was undertuned; fitting it
  properly moved one of them to zero. The boundary is real either way — nothing
  transfers — but it is a wall, not a reversal.
- **Our human baseline was undertuned, and by more than was thought.** At the
  teammate's settings the human matrix reaches **ρ 0.5076 ± 0.0149** against the
  0.404 this harness had been quoting — so the ρ 0.472 attributed to an early
  personal script not only reproduces, it is exceeded. **The gap is
  hyperparameters, not features**: 400 selected columns at our champion settings
  score 0.4074, within noise of 300 (Appendix B). This widens the cross-kingdom
  gap rather than narrowing it, but it carries a caution that reaches further
  than the human arm — **the champion hyperparameters are inherited from the
  *E. coli* baseline and lose 0.10 ρ on a matrix of a different shape.** Every
  external arm in this report is fitted with them. The reason is identified, and
  it is the more interesting half: **the human signal is much less additive.**
  Ridge reaches 95.6% of boosting's ρ in *E. coli* and only **63.7%** in human,
  and the largest single hyperparameter term is tree depth (+0.0585 of the
  +0.1002). A champion tuned where the signal is nearly linear is the wrong model
  where it is not — which also explains why it travels across the bacterial sets
  without incident (Appendix B).
- **One organism and two relatives.** No genome-wide Cas9-cutting screen exists
  outside *Enterobacteriaceae*; what exists elsewhere is CRISPRi, a different
  measured quantity. Closing this needs wet-lab work.
- **The label is survival, not cutting**, and we train on **crisprHAL 2's curated
  score**, not the published one. Verified: our target is **bit-identical** to
  their released `y` on all 33,567 guides, so the head-to-head compares two models
  on one quantity. But the two cohorts' labels are **different quantities** —
  Noshay's `cut_score` runs 0 to 48.05 with no negative values, crisprHAL's runs
  −9.05 to +6.21 with 47.5% negative. They agree in rank at ρ 0.945 on the 8,617
  overlapping guides and give target GC the same sign (−0.147 against −0.201), so
  nothing downstream reverses — but **every number in this report belongs to one
  label or the other and they are not interchangeable.**
- **And "curated" is a procedure, not a judgement — but it is a *fitted*
  procedure.** The guides were not cherry-picked: the score is ALDEx2's
  `diff.btw`, the posterior-mean log2 fold change between the Cas9 and
  dead-nuclease arms over two replicates each, and the filter is an average
  control-condition read count with a cutoff of **73** for the SpCas9 set we
  train on (31 for eSpCas9, 56 for *C. rodentium*). Two properties of it are not
  flattering and we state them because we train on its output: the **threshold is
  itself chosen by cross-validation for model performance**, so it is a
  hyperparameter rather than an independent cleanliness rule; and it is applied
  **before** the train/test split, so it cleans the test set too and raises ρ
  partly by making the evaluation easier. **None of this is what inflated our
  family table**: at fixed rows a label swap is worth +0.004. The 3.3× is that
  the flank effect is genuinely ~3× steeper outside Guo's 1,222-gene panel than
  inside it (Appendix B2a).
- **The long-range mechanism is unidentified.** Orientation is a live candidate
  with one ruled-out alternative; the most mechanistically motivated feature built
  to explain it (GC skew and helical periodicity) is a null.
- **Adaptive overfitting, now measured rather than bounded.** Many choices were
  made against the same outer folds, and sealing 20% of genes across **fifteen
  draws** puts the automatable component at **+0.0087 ρ [+0.0020, +0.0153]** —
  small, real, and a *lower* bound, since no re-derivation reaches the human
  choices. It lands on the **differences**, not the levels: family A's gain is
  **+0.164 in cross-validation and +0.156 on genes the pipeline never saw**,
  while the absolute figures do not move detectably. The margin over crisprHAL 2
  on our own folds should be read the same way; the parity claim on their splits
  is untouched (Part 4).

---

## Part 4 — The freeze, and what remains

**The analysis is frozen as of 11 October 2026.** No further model, feature or
hyperparameter choice is made against the curated five-fold split.

**Where the model stands against the state of the art is Finding 7.** The freeze
matters to it in one way worth stating here: **eSpCas9 was designated "the untouched final evaluation" and was not
untouched** — four feature arms had already been run on it across three seeds and
two CV schemes (`results/transfer.csv`), with a flank gain of **+0.104 under
`arc5`** quoted in `RESULTS.md` (now `RESULTS_archive_2026-10-10.md`). Nothing was ever *selected* on it, so it is
confirmatory rather than contaminated, but "evaluated once" was never
available, and the last genuinely unused dataset (Cui's CRISPRi screen) has now
been spent too. **No untouched external set remains.**


Two existing external checks, neither a substitute for that: the baseline anchor
reproduces Noshay et al.'s published figure through our harness, and SLICER
scores 0.6997 on crisprHAL 2's own shipped SpCas9 split, a partition untouched
during development.

*That partition's status, stated once so it is not overclaimed later.* It has now
carried **seven** arms: the original SLICER run, and six more in the data
ablation (Finding 7). None of them selected anything — every feature set,
hyperparameter and split was fixed before the partition was read, and the arms
differ only in which training rows they see — so it remains confirmatory rather
than contaminated. But "evaluated once" is no longer true of it either, and a
seventh reading is the point at which that should be written down rather than
assumed. **The curated five-fold figures carry an adaptive-overfitting component of
about +0.009 ρ** [+0.0020, +0.0153] from the automatable part, measured below,
with the human part unquantified above it. It falls on the arm-to-arm
differences rather than on the levels — family A reads +0.164 in CV and +0.156
on sealed genes — and the paper should quote the sealed figure or both.

Remaining before submission:

| | item | cost |
|---|---|---|
| 1 | ~~eSpCas9 final evaluation~~ | **done, and it is a tie** — ρ 0.7842 against 0.7821 on crisprHAL's own eSpCas9 split with their published weights, at 47,591 training guides. Also corrected: eSpCas9 was never an untouched holdout (above) |
| 2 | ~~crisprHAL 2 given a search of its own~~ | **run 16 October, and it found nothing**: 0.6987 against the 0.6971 their published *configuration* reaches on these same folds, with that configuration winning the inner search in 4 of 5 folds. **Independently repeated on a second Colab session, 26 October: mean 0.6989** (0.6895, 0.7012, 0.7063, 0.6983, 0.6994), again with their published configuration winning four of the five inner searches — a GPU result reproducing to 0.0002. The margin against their *best known* configuration is **+0.0161**. Still not a fully matched search — ours used 97–157 draws per fold, theirs 5–8, so ~40 GPU-h would be needed to equalise (Appendix B) |
| 3 | ~~orientation controls~~ | **done** — Jacky's, and reported in his own write-up rather than here |
| 4 | ~~human baseline re-tuned at the teammate's settings~~ | **done, and it reproduces: ρ 0.5076 against the 0.472 quoted.** The settings arrived 16 October; the gain is hyperparameters, not columns (+0.1002 against +0.0038), and the reason is that the human signal is far less additive than *E. coli*'s (ridge/boosting 63.7% against 95.6%), so the champion is the wrong shape for it by 0.10 ρ. **This cost a claim**: refitting the human side moved one cross-kingdom transfer direction from −0.016 to −0.006 [−0.023, +0.011], i.e. to zero, so "negative in both directions with intervals excluding zero" is withdrawn in favour of "no transfer in either direction" (Part 5, Appendix B) |
| 5 | ~~conditional test of the GC hypothesis within matched GC strata~~ | **done** — Finding 3 |
| 6 | ~~sample-complexity curve repeated across folds and seeds~~ | **done** — Appendix B |
| 7 | re-tune crisprHAL 2 at **each input width** — **partly supplied by their own paper.** Browne et al. searched input length *per dataset* and stopped at 378 nt (*E. coli* SpCas9) and 406 nt (eSpCas9), reporting that gains level off: +0.00034 ρ per upstream nucleotide and +0.00074 per downstream one, from 0.554 on the sgRNA alone to 0.697 at 378 nt. So the width is their optimum, found by their own search, and our 1,023/2,023 nt degradation is the continuation of a curve they had already flattened. What is still unrun is re-tuning the *other* hyperparameters at each width. The script is written and anchored on the 378 nt positive control (`colab/crisprhal_tune_by_width.py`); only the GPU time is outstanding | 4–6 GPU-h, now optional |
| 8 | ~~re-derive the ceiling for the curated label~~ | **done: 0.946 [0.944, 0.948]**, from their own published per-replicate counts, cohort and label both verified to reproduce, and conservative by construction (Finding 6) |
| 9 | ~~re-run the gene-grouped ablation on the curated cohort~~ | **done** — all eleven families, Appendix B3. The three largest effects lose nothing beyond the 6% grouping tax; `e_nucleoid` and `g_transcription` carry a real locus component of 6 and 15 points |

**Paper writing is no longer blocked by this list — the list is closed.** Every
item is done except the optional remainder of item 7, re-tuning crisprHAL 2's
*other* hyperparameters at each input width, which their own published length
search makes unlikely to change the answer. What the draft must carry, and what
this list exists to make unavoidable:

**The last untouched dataset has now been spent, deliberately, on the one
question that could have falsified the mechanism — and it took two rounds to
read.** Cui et al.'s dCas9 CRISPRi screen was the only external set never used
for a model, feature, split or hyperparameter. It was spent on "is the flank
effect about cutting or about binding?" Family A adds **−0.009** across that
screen against **+0.157** on the cutting screen; restricted to the essential
genes where knockdown has a phenotype at all it reads **+0.268**, and that +0.268
turns out to be the screen's own strand rule, leaving **+0.007** once a strand
indicator is supplied (Finding 3). The answer is cutting, and the second round is
what makes it trustworthy. **No untouched dataset remains**, so the
adaptive-overfitting caveat below is permanent rather than pending, and further
external validation needs new data.


* the headline comparison is **parity, not a win** — +0.005 and +0.002 on
  crisprHAL's own two splits — and the data-efficiency result is the claim that
  survives a larger screen;
* the curated five-fold figures carry an **unknown adaptive-overfitting
  component**, because eSpCas9 turned out to be confirmatory rather than
  untouched and nothing fully clean is left;
* the 412-column reduction is a **published-matrix result under a 300-column
  budget** and does not transfer to the curated cohort;
* the ceiling for the label we train on is **0.946 [0.944, 0.948]**, measured;
  0.937 belongs to the published label and should not be quoted for curated
  figures.

The shared Google Doc is **frozen as of 13 October 2026** and is no longer kept
in sync; this report is the single maintained document.

**A guide-design tool is not a deliverable.** The findings are the contribution,
the field already has several predictors, and everything needed to build one
later exists (`featurise.matrix()` turns any 20-mer into the published
representation).

### The decision register: which choices could have inflated the five-fold figures

"The curated five-fold figures carry an unknown adaptive-overfitting component"
is the honest caveat, and it is also lazy — the component is not uniformly
unknown, because most of the choices have a traceable provenance and several are
measurably insensitive. Every decision that could inflate those figures, audited
16 October:

| decision | chosen where | adaptive w.r.t. the curated folds? | does it matter? |
|---|---|---|---|
| 300 selected columns | **inherited** from the baseline script (`N_FEATURES = 300`) | no | **no** — ±0.0005 against using every column |
| champion XGBoost settings | **inherited** from the same script, not searched by us on either cohort | no | **no** — 0.0002 against LightGBM at its own unrelated defaults |
| selector = XGBoost gain | inherited; the LightGBM alternative was measured | no | **no** — 0.7033 against 0.7025 |
| 5 folds, `random_state=41` | fixed once, reused by every arm in the report | no | reported over 5 seeds × 5 folds |
| family A's existence | designed in **September on the published cohort**, from residuals computed there | no — a different cohort | it is the effect, so it must matter |
| family A's window set (50/250/500/1,000) | swept on the **published** cohort, winner carried over | no, but a sweep winner carries its own curse | **no** — rebuilt at 75/300/600/1,200, which shares no value with it: **+0.1634 against +0.1638**, a difference of 0.0005 against the ±0.005 declared in advance |
| per-fold hyperparameter search (627 draws) | curated cohort, **nested inside each training fold** | no — cannot see a validation row by construction | +0.0131 |
| the 412/427 reduced column set | published cohort | n/a — it does not transfer, and the report says so | — |
| which families to report | all eleven are reported, including the nulls | no | — |
| ρ as the primary metric | crisprHAL 2's own reporting convention | no | nine-metric panel, `docs/qna_archive.md` |
| **the choice of cohort** | **here, and partly because curation improved our score** (ρ 0.607 → 0.628 on shared rows) | **yes** | unquantified |

**Two conclusions, and the second is the uncomfortable one.**

*Most of the pipeline is less adaptive than the caveat implies.* The settings a
reviewer would suspect first — 300 columns, the booster's hyperparameters, the
selector — were **inherited from the baseline script this project started from**,
and are measurably insensitive besides, at ±0.0005 and 0.0002. The 627-draw
per-fold search is a separate thing and cannot leak: it runs inside each training
fold. The per-fold search is leak-free by
construction. So the mechanisable part of the pipeline has little room to have
inflated anything.

*The cohort choice is genuinely adaptive and cannot be undone.* We moved to
crisprHAL's curated label partly on the evidence that it improved scores, and
every figure since is computed on the rows that choice selected. No re-derivation
reaches it, because the alternative cohort is not a held-out sample of the same
thing — it is a different, smaller, noisier draw.

**The automatable part is measured, and it is not zero: +0.009 ρ**
(`sealed_holdout.py`). Twenty per cent of the cohort is sealed by whole gene;
selection and hyperparameters are re-derived inside the development pool; each
arm is then scored once on the sealed genes. A reference arm whose configuration
was *not* chosen on these rows absorbs the population difference, and the
adaptive component is the difference of differences. Fifteen sealing draws, with
training sizes matched so the sealed fit sees no more data than a CV fold:

| | mean | sd across draws | 95% CI |
|---|---:|---:|---|
| reference gap (nothing chosen on these rows) | −0.0121 | 0.0148 | [−0.0196, −0.0047] |
| frozen — the configuration the report quotes | −0.0035 | 0.0077 | [−0.0074, +0.0004] |
| dev-tuned — re-searched inside the pool | −0.0039 | 0.0090 | [−0.0085, +0.0006] |
| **adaptive component, frozen** | **+0.0087** | 0.0131 | **[+0.0020, +0.0153]** |
| adaptive component, dev-tuned | +0.0082 | 0.0140 | [+0.0011, +0.0153] |

**Twelve of the fifteen draws are positive**, t = 2.56 (p = 0.022), Wilcoxon
p = 0.015, sign test p = 0.035, and a 20,000-draw bootstrap gives
[+0.0024, +0.0152]. Dropping any single draw moves the mean only between +0.0065
and +0.0106, so it does not rest on one sealing. **The curated five-fold figures
are optimistic by about 0.009 ρ from the automatable part alone**, and that is a
*lower* bound on total adaptivity, because no re-derivation reaches the human
choices — building family A at all, the window set, the cohort.

*This supersedes the three-draw result, and the way it failed is the lesson.*
The first three draws gave **+0.0005 ± 0.016**, which was reported as "no
inflation is detectable, and the component is not larger than about 0.02". That
was the right reading of three draws and the wrong conclusion: the reference
arm's own gap swings ±0.015 between sealings, so three draws could not resolve a
0.009 effect, and a point estimate of zero from an underpowered design is not
evidence of zero. Twelve more draws cost one night of a plugged-in laptop.

*Which number it moves, and it is not the one intuition reaches for.* The
**absolute** figures are not detectably inflated: the frozen arm scores 0.7043
in pool CV and **0.7077** on sealed genes, a gap of −0.0035 whose interval
contains zero — the sealed genes are, if anything, marginally easier. What
moves is the **difference between arms**, which is where the headline lives:

| | pool CV | sealed genes | change |
|---|---:|---:|---:|
| base only (nothing chosen here) | 0.5396 | 0.5517 | +0.0121 |
| frozen (the quoted configuration) | 0.7043 | 0.7077 | +0.0035 |
| **the gain from the chosen configuration** | **0.1647** | **0.1560** | **−0.0087** |

**So the family-A gain reads +0.164 in cross-validation and +0.156 on genes no
part of the pipeline has seen.** That is the honest form of the number, and it
is the same ±0.0087 as above, seen from the side that matters.

*The intuition that a difference cancels shared optimism is wrong here, and this
is the measurement that shows it.* Cancellation needs both arms to have been
chosen on the same rows. The base-only arm was not, so the optimism does not
subtract out — it appears in full in the difference. The same logic applies to
**the margin over crisprHAL 2 on our own five folds**: our configuration was
chosen on those rows and theirs was not, so +0.0161 to +0.0172 should be read as
carrying up to this much self-inflation, leaving roughly **+0.008** on unseen
genes. That is independently consistent with the margin on *their* held-out
splits being +0.002 to +0.005, and it is a second contributor alongside the
data-volume account in Finding 7 rather than a replacement for it.

**The parity claim is untouched**, because it is measured on crisprHAL 2's own
held-out partitions, which no step of our pipeline was derived on.

What the experiment *cannot* reach is the cohort choice itself, and nothing in
this design ever could.

*One of the two was testable, and it passed.* The window set is the kind of
choice that carries a sweep's winner's curse even when the sweep happened on
another cohort. Rebuilding family A at **75/300/600/1,200 nt** — four windows
sharing no value with the published four — gives **+0.1634 against +0.1638**.
A difference of **0.0005**, a tenth of the declared margin, so the four numbers
in `config.FLANK_WINDOWS` are not where any of family A's gain lives. That is
what Finding 2 predicts for a smooth gradient and it is one fewer thing for a
reviewer to ask about.

**What would have prevented the one that remains:** sealing 20% of the curated
cohort on 13 October, before the pivot, rather than on 16 October after it. That is the one procedural
lesson worth carrying to the next organism — seal first, and seal by gene.

## Part 5 — What the control phase changed

Work resumed on **2 October** after a pause following family A's discovery
(12–13 September) and the first head-to-head (15 September). From **10–13
October** the work was almost entirely adversarial: nulls, matched comparisons,
conditional tests and robustness checks aimed at the claims already made rather
than at new ones, and it has continued in that mode through **26 October**. It is worth stating separately what that bought and what it
cost, because the cost was not small.

### What survived, and got stronger

| claim | what the control added |
|---|---|
| family A is real and local | a **texture-preserving** null, not just row shuffling: aligned +0.0769, 1 kb retains 8%, ≥10 kb is null. Converts a control into a measurement of the effect's autocorrelation length |
| the GC account | conditioning *raises* flank GC's association — on the curated cohort, **+0.176 marginal → +0.221 partial** — and it is positive in all five target-GC strata (+0.425 down to +0.068), strongest where the target itself is GC-poor |
| the reduced representation | **412 columns** computable from any 20-mer land 0.00003 ρ from all 6,232 — portability, not just sufficiency |
| data efficiency | survives all five folds; the ρ ≥ 0.68 ratio goes **1.4× → 2.5×** |
| the mechanism | the dCas9 null survived the one re-examination that looked fatal to it. Restricted to the essential genes where knockdown has a phenotype, family A reads **+0.268** — but that is the screen's own strand rule, and given a strand indicator the family is worth **+0.007** there against +0.157 on a cutting screen. The control the claim previously lacked, and the claim is stronger for it (Finding 3) |
| the flanking-context thesis | on the curated cohort nothing confined to the guide clears **+0.019** and nothing above **+0.05** is confined to it. Arrived at by withdrawing a claim that pointed the other way, and since re-scoped: the ordering is by what the model cannot already compute, not by what the phenotype depends on (Part 2) |

### What that means

**The three contributions survive; several of the sentences supporting them did
not.** The decoding and the flanking-context effect came through every control
intact or stronger. What changed is the *framing* of the third: we do not
out-encode a tuned deep model on DNA both sides hold, we have access to DNA it
never sees. That is a narrower claim and a much harder one to attack.

Every withdrawal but one was caught by **our own tests**; the exception, the
nearest-gene orientation confound, was caught by a teammate's. The newest one
came from chasing down a teammate's *unexplained number* rather than a claim:
reproducing ρ 0.472 on the human matrix showed our human model was the wrong
shape for its data, and the transfer interval was resting on that. None were caught by
re-reading the text, and in every case the data needed was already on disk: the
bottleneck was that nobody had asked the number to defend itself.
`docs/FINDINGS_LOG.md`, which is local, records each one where it happened,
with what replaced it.

---

# Appendices

Everything below supports a finding without being one: the controls, the
methods each claim rests on, and the provenance. Nothing here should appear in
the main text of a paper.

## Appendix A — Controls and nulls

Every control here is written the same way: **what it rules out**, **why the
design rules that out** rather than merely gesturing at it, and **what would be
believed without it**. The third line is the one worth reading — several of these
controls exist because the thing they rule out was, for a while, believed.

### Nulls — is the pairing between feature and label real?

**Permutation floors.** Every family with a gain was re-run with its feature rows
shuffled against the labels.
*Why it works:* shuffling destroys the guide↔feature correspondence while
preserving each column's marginal distribution and the model's capacity exactly,
so the arm measures how much ρ a selector and a boosted tree can manufacture
from 300 columns of correctly-shaped noise. That floor is not zero and has to be
measured rather than assumed.
*Without it:* family A's +0.080 would rest on the assumption that a 6,580-column
search cannot overfit a 13,880-row training split. The measured floor is
**0.5264 against a 0.5272 baseline** — i.e. zero — and `j_periodicity`'s
shuffled arm scores *higher* than its real one, which is what a null family
looks like from the inside.

**The texture-preserving null for family A** (`shiftnull.py`). Each guide is
re-assigned the flank track read from a position offset by a **circular
chromosome shift**: the same track, at the wrong place.
*Why it works:* row shuffling tests a weaker null than family A's claim.
Family A asserts that *this* guide's local DNA predicts *this* guide's activity;
row shuffling breaks that but also destroys genomic autocorrelation, so it
cannot distinguish "local DNA matters" from "DNA-like texture at the right
smoothness matters". A circular shift preserves every column's marginal, its
smoothness and its long-range autocorrelation exactly, and breaks only the
alignment. Verified on `dn500.gc`: the mean moves 0.5103 → 0.497–0.508 while the
correlation with the aligned track falls 1.00 / 0.41 / 0.05 / −0.03 at 0 / 1 kb /
100 kb / 2 Mb.
*Without it:* the family A result would be compatible with a model that has
learned the *texture* of bacterial DNA rather than anything about the cut site.

**One shift would not have been enough**, which is why the shift is swept. At
1 kb the shifted columns are still 0.41 correlated with the real ones, so a
single 1 kb control would have reported a spurious residual effect and a single
2 Mb control would have proved only that a distant genome position is
uninformative. Sweeping converts the control into a measurement: Δρ against
shift distance **is** the autocorrelation length of the effect, read off
independently of the window sizes family A happens to use. Results in Finding 2.
Grouped CV (100 kb blocks) throughout, because a positional feature under random
KFold can score by locus recognition and the shifted arms would inherit that too.

**Negative controls from the source experiment.** Available for exactly one
family: the dCas9 / no-antibody arm of the GapR-seq experiment.
*Why it works:* it is the same assay with the biology removed, so a feature that
predicts equally well from the control track is measuring the assay, not the
phenomenon the track is named after.
*Without it:* `d_supercoiling`'s +0.045 would still be reported as supercoiling.
It is **DNA read density**, and the family was overturned by its own control
rather than by an argument — which is the only reason it was overturned at all.

### Splits — is the gain prediction, or address memorisation?

**Grouped cross-validation.** Anything positional is reported under `arc5`,
100 kb blocks, or whole genes held out.
*Why it works:* the screen puts several guides in every gene, and guides in one
gene share a locus. Plain KFold therefore puts guides from the same gene on both
sides of the split. For a sequence feature that is harmless, because the
sequences differ; for anything positional it is not, because those columns
*identify* the locus, and a model can be rewarded at validation time for
learning "guides around here score about this much". Holding out whole groups
removes that route entirely, since no held-out gene contributes a training row.
*Without it:* the positional families' gains would be indistinguishable from
locus recognition, and the share of label variance that lies *between* genes is
exactly the budget available for that trick — **16% on the published matrix, 29%
on the curated one** (`cohort_groups.py`).
*And the control needs its own calibration.* Families that are functions of the
guide cannot be invalidated by a change of split, so whatever *they* lose under
grouping is the cost of the harder partition rather than memorisation. They lose
6% in aggregate, which sets the benchmark the positional families are read
against — `a_flank`, `d_supercoiling` and `i_shape` lose nothing beyond it,
`e_nucleoid` and `g_transcription` lose 6 and 15 points more (Appendix B3).
Without that calibration every family would appear to lose something.

**Why gene-grouping became necessary on the curated cohort, specifically.** The
grouped arms on the published matrix were a precaution against a small risk:
with 84.4% of label variance inside genes, only ~15% was available to memorise.
The curated cohort is **genome-wide** — 4,013 genes at 7.5 guides each, against a
1,222-gene panel at 11.3 — so gene identity explains **29%** of its label, nearly
twice as much. The risk the control guards against roughly doubled when the
cohort moved, while every curated family number was measured under random KFold.
That is why the re-run was not optional, and the result is in Appendix B3: the
families keep 79–96% of their gains, so the inflation is not memorisation. **A
control that had not been run would have left the single largest table in the
report resting on the assumption that the published cohort's grouping behaviour
transferred — and the variance share says it does not.**

### Matching — is the comparison fair?

**Matched-span head-to-head against crisprHAL 2.** crisprHAL is also run at our
window widths, and SLICER at theirs.
*Why it works:* the two models differ in architecture *and* in how much DNA they
read, and a head-to-head at each side's native width cannot separate them.
Equalising the input DNA leaves architecture as the only difference, in both
directions: at their span our windowed composition loses by 0.025; at ±1,000 nt
we lead by 0.065.
*Without it:* "we out-encode a tuned deep model" — which was claimed, and is
false. The honest claim is access to DNA they never see, and it is narrower and
much harder to attack.

**Distance- and GC-matched orientation comparison, with a block bootstrap**
(Jacky).
*Why it works:* opposite-orientation guides are not a random sample of
same-orientation guides — they sit at different distances from the gene and in
different GC. Matching on distance and flank GC compares like with like, and
bootstrapping over 47 genomic blocks rather than over rows accounts for the fact
that nearby guides are not independent observations, which is what sets the
interval width honestly.
*Without it:* "0.047 against the nearest gene" stands. After matching it is
**0.007 [−0.024, 0.038]**, P = 0.34 — a cohort confound. The effect is real
against a curated **TSS** (Δ 0.051 [0.020, 0.082]), which is a different and
better-specified claim.

**The conditional GC test, within matched target-GC strata.**
*Why it works:* flank GC and target GC are correlated, so a marginal association
between flank GC and activity could be target GC seen through it. Binning on
target GC and measuring inside each bin removes that path by construction.
*Without it:* the GC account would rest on a marginal correlation. Conditioning
*raises* the association — on the curated cohort **+0.176 marginal → +0.221
partial**, positive in all five strata — the signature of a suppressor
relationship, and the opposite of what a confound looks like.

**Why "a fair comparison" has to be qualified, and what each condition holds
equal.** The phrase *"parity under the fairest conditions for both models"*
should not be used, because three things can be equalised and no condition
equalises all three:

| axis | on their splits | effect |
|---|---|---|
| **rows, labels, splits** | equal — theirs, untouched by us | fair |
| **tuning effort** | equal, arguably in their favour: their arm is their *published* model, ours is a config frozen on a different dataset and never tuned for eSpCas9 | fair, or against us |
| **input DNA** | **unequal, in our favour** — family A reads ±1,000 nt, their eSpCas9 input is 406 nt | against them |

So the parity result is "each model at its own best configuration" — a real and
reportable condition, and how both models would actually be used — but **not** the
equal-information condition. Equalising the DNA instead makes the ranking depend
on the width (Finding 2). Two further asymmetries belong beside the numbers: our
features were designed on *E. coli* WT cutting data, the same organism, so design
knowledge leaks into the eSpCas9 arm in a way no split can remove; and the SpCas9
margin compares our 0.6997 against their *published* 0.695 rather than a re-run
under identical preprocessing — our own re-run of their model gives 0.6940, so
that margin is +0.006 measured our way and +0.005 against their number.

**Equal DNA is not equal opportunity to use it, and Finding 2 claims only the
weaker thing.** Giving crisprHAL 2 more of the genome at its published
configuration multiplies its flattened layer ~5× at 2,023 nt while leaving the
regularisation and training budget that were chosen at 378 nt untouched. The
measurement therefore supports *"the published configuration is degraded by
input wider than it was tuned for"*, not *"this architecture cannot exploit
long-range context"* — the second needs a re-tune at each width.
*Why it is still worth reporting:* **378 nt is their own searched optimum**, not
an inherited default. Browne et al. searched input length per dataset and
stopped there, reporting gains that level off at +0.00034 ρ per upstream
nucleotide and +0.00074 downstream. So the experiment extends a curve past the
point their own search stopped improving, rather than handicapping them at a
setting nobody chose.

**The metric panel, bootstrapped: is the metric split a profile difference or
noise?** It would be easy to read "we win the rank metrics, they win the
top-of-list metrics" as a genuine difference in character. Paired on the same
11,898 guides, 2,000 resamples:

| metric | Δ (SLICER − crisprHAL) | 95% CI |
|---|---:|---|
| Spearman ρ | +0.0021 | [−0.0028, +0.0071] |
| Kendall τ | +0.0023 | [−0.0024, +0.0073] |
| Pearson r | −0.0019 | [−0.0066, +0.0030] |
| R² | −0.0030 | [−0.0103, +0.0041] |
| top-decile precision | −0.0059 | [−0.0244, +0.0202] |
| expected pick percentile | −0.70 | [−17.91, +3.98] |

*Why it works:* six metrics on one test set invite reading a pattern into six
differences. Bootstrapping each one paired on guides asks whether any of them
resolves, and none does. *Without it:* the split would read as "better at ranking
against better at picking", which is a claim about model character rather than
about sampling noise. The last row is also the argument against ever headlining a
top-10 statistic: its interval is 22 percentile points wide.

**The held-out pairing had to reconstruct an alignment, and it is asserted rather
than assumed.** crisprHAL's out-of-fold predictions were saved one file per fold
with no guide key, so the concatenation is in fold order, not cohort order.
Pairing them with SLICER's out-of-fold vector requires rebuilding
`KFold(5, shuffle=True, random_state=41)` and concatenating its validation
indices in the same order; the code then asserts that the two label vectors match
guide for guide. *Without it:* a mis-rotated join would have produced a tight,
plausible and meaningless interval — the same failure shape as the three
execution guards recorded in `docs/qna_archive.md`.

### Instruments — is the measurement measuring what its name says?

**The selector control: 300 columns against all of them.**
*Why it works:* every arm selects 300 columns from its candidate pool, so an arm
with more candidates could gain simply by being given more to choose from.
Re-running with selection switched off removes the budget as a variable.
*Without it:* `d_mechanics`'s +0.099 on a base starved of dense columns would be
open to the obvious objection. It is not selection: **+0.0988 selecting 300,
+0.0993 using all 6,199**.

**The reconstructibility probe: each of a family's columns predicted out of fold
from the base.**
*Why it works:* "this feature is a deterministic function of the 20-mer" is a
claim about the feature's *inputs*, and reading the builder is the fast way to
check it — but a probe makes it quantitative and catches columns whose inputs
nobody read. A column the base predicts at R² = 1.0000 adds no information; one
it predicts at R² ≈ 0 is reading something the base does not contain.
*Without it:* the claim that `d_mechanics` adds no new DNA stands, and its
+0.099 remains the report's headline evidence that representation beats
information. Eight of its thirty columns came back at **R² ≈ 0** — flanking-DNA
thermodynamics — and they carry +0.097 of the +0.099 (Finding 1). **This control
cost the report its most quotable claim, and it is the single most useful thing
run in two days.**

Per-column, the probe separates the family cleanly, which is what made the
decomposition obvious once it was run:

| `d_mechanics` columns | out-of-fold R² from the base |
|---|---:|
| 9 ΔG/ΔH/ΔS sums over the protospacer and its windows | **1.0000** |
| 4 melting temperatures | 0.997–0.9995 |
| 7 stacking and slide summaries (means, sd, extrema, argmax) | 0.20–0.98 |
| **8 flank thermodynamics (±25, ±100 nt)** | **−0.006 to +0.042** |

**The guide-only / flank-only decomposition** (`no_information.py`) is the
follow-up that same probe demanded: split the family by what its columns read and
run both halves against the same base. +0.011 for the 22 protospacer columns,
+0.097 for the 8 flank columns. *Why it works:* the two halves share the base,
the rows, the folds and the selection budget, so the only difference is reach.

**The faithfulness positive control.**
*Why it works:* an attribution method that assigns importance to real features
and to permuted ones alike is uninformative, so the probe is calibrated by
feeding it a known null and checking what it reports. At a **matched share of
attributed importance** (~35%), permuted columns score 0.258 against real
columns' 0.581.
*Without it:* the standard faithfulness probe's output on a redundant matrix
would read as a measurement rather than as the uninterpretable number it is
(Finding 5).

**Equivalence testing, reported as a bound rather than against a margin.**
*Why it works:* "the reduced set is as good" cannot be established by failing to
reject a difference — absence of significance is not evidence of equivalence. An
equivalence test fixes that by asking whether the whole interval falls inside
some margin. *Without it:* "427 ≈ 6,232" would be an argument from a
non-significant p-value, which is the most common error in this literature.

**But a margin is only meaningful if its declaration can be checked, and ours
could not.** Earlier versions of this report called both margins "pre-declared".
The repository says otherwise, and it is worth stating because the word was doing
real work:

| claim | result first exists | margin first appears | gap |
|---|---|---|---|
| 427 columns non-inferior, 0.005 ρ | 4 Oct (`136755f`) | 11 Oct (`0f66a2e`) | **7 days after** |
| parity with crisprHAL 2, ±0.01 ρ | 15 Oct 19:13 (`6e2f32d`) | 15 Oct 20:07 (`872e2d4`) | **54 minutes after** |

Nothing was fabricated — both margins are defensible and neither was tuned to
produce a verdict — but "pre-declared" asserted a timeline that the commits
contradict, and a reader has no way to audit such a claim except by doing what is
done above. **So the word is withdrawn, and the claims are restated in the one
form that needs no declaration: the smallest margin the data themselves
support**, which is just the larger end of the confidence interval.

| claim | interval | equivalence holds at any margin ≥ |
|---|---|---:|
| SLICER vs crisprHAL 2, ρ, eSpCas9 split | [−0.0031, +0.0072] | **0.0072** |
| the same, Kendall τ | [−0.0024, +0.0073] | 0.0073 |
| the same, Pearson | [−0.0066, +0.0030] | 0.0066 |
| the same, R² | [−0.0103, +0.0041] | **0.0103** |
| 427 columns against 6,232 | [−0.0008, +0.0054] | 0.0054 |

That table is a fact about the data and cannot be gamed by when anyone wrote
anything down. It also exposes something a fixed ±0.01 hid: on **R² the data do
not support equivalence at 0.01** — they need 0.0103. The rank metrics do, with
room to spare.

**What a credible pre-declaration would look like**, for the next organism:
commit the margin, in code, with the hash recorded, *before* the result file
exists — then `git log -S` settles it in one command, as it did here. Until that
is done, report the bound.

**The attenuation ceiling** (Spearman–Brown on replicate statistics).
*Why it works:* a model's ρ cannot exceed the label's own reliability, so the
ceiling is a property of the measurement and has to be estimated from replicates
rather than from model performance.
*Without it:* "the model is out of material" would be a claim about the model.
With it, it is a claim about how much signal the label contains — **0.946
[0.944, 0.948]** for the label we train on, from crisprHAL's own per-replicate
counts (Finding 6), and 0.937 for the published one by a different route.

**The cross-assay control: a screen where the mechanism should *not* apply.**
*Why it works:* every other external check is another Cas9 cutting screen, so a
flank effect there is consistent with cutting, with binding and with local
accessibility alike. dCas9 binds without cutting, so a flank effect about the
cleavage step must disappear while one about binding or accessibility must
survive. It is the only control here that could have falsified a mechanism rather
than a number.
*Without it:* Finding 2's mechanism sentence would rest on plausibility. Family A
adds **−0.009** to a CRISPRi screen and **+0.157** to a cutting screen under one
protocol, with a flank-only arm (0.133 against 0.341) to rule out selection
dilution (Finding 3).
*The control it needed in turn, because a cross-assay comparison is a mechanism
test only if the two assays differ in exactly one respect.* They differ in two:
CRISPRi also needs the silenced gene to matter, and 85% of that cohort is genes
whose knockdown barely moves growth. Restricted to the essential genes the same
features read **+0.268** — which is the screen's own strand rule, not the flank
effect, since flank GC's marginal association there is −0.0004 and a single
strand indicator scores ρ 0.662 on its own. Given that indicator, family A adds
**+0.007**. The null holds in the informative half, and the strand result is the
positive control the original comparison lacked: this assay plainly *can* see a
guide-positioning effect.

**The replicate-split ceiling.**
*Why it works:* a ceiling has to come from the measurement, not the model. Two
Cas9 replicates and two controls allow two independent estimates of the same
quantity; their agreement is the reliability of one, and Spearman–Brown converts
that to the reliability of the released mean of two. Two checks make it an
estimate of *their* label rather than of our arithmetic: their cohort reproduces
under their stated cutoff (33,871 against 33,567) and a plain log-ratio
reproduces their score at |ρ| 0.957.
*Without it:* the curated arms would keep quoting 0.937, a figure derived for a
different label by a different route. It is **0.946 [0.944, 0.948]**, and
conservative, since their posterior shrinkage can only raise reliability above a
plain log-ratio's.

## Appendix B — The methods and controls behind the findings

This appendix was a chronological list of questions and answers. It is now
grouped by what each section defends, and the sections that defended nothing
have been moved to `docs/qna_archive.md`. Every section below is here because a
claim in Parts 1–3 rests on it.

**B.1 — Can the head-to-head be trusted?**

### The search asymmetry: what closing it cost, and why it was not closed further

Finding 7 carries the result — 0.6987 against their published configuration's
0.6971, that configuration winning four of five inner searches, and a residual
~20× asymmetry. Four things about the run belong here rather than there.

*The hardware is what made any version of it affordable.* Their published
configuration fits one fold in **5.48 min on a free Colab T4** against **17.0 min
locally**, a measured **3.1×** (`crisprhal_wide_window.csv` against
`crisprhal_rerun.csv`). A TPU would have been the wrong instrument: TPUs earn
their keep on large dense matmul under XLA, and this model's recurrent branch is
a bidirectional GRU, the shape that maps worst onto them.

*It was costed three times and only the third was a measurement.* First ~460
CPU-hours, for 627 draws locally. Then "2–4 GPU-hours", which priced their
*default* configuration for a search that mostly does not use it — the local
search's median fit was 44 min against 17 for the default, because the draws
average larger networks. Then the run: **33 fits, ~2.6 GPU-hours**, at 2.3–8.6
min each, because patience stopped every fold after 5–8 draws. The cheap version
is cheap *because* it gives up early, and why it gives up is the result.

*The prediction was wrong, and the reasoning was the mistake.* I predicted ~0.716
— the margin narrowing to ~+0.005 — by anchoring on their published WT model
scoring 0.695 on their own WT split, where we score 0.6997. That anchor is a
single 6,737-guide partition on which the standard error of ρ is ≈0.006, so
+0.005 and +0.017 sit about two standard errors apart and the split cannot
distinguish them. A noisy one-split estimate was treated as a precise prediction
for a five-fold paired one.

*And the pre-registration helped less than it should have.* The interpretation
was fixed in three brackets beforehand: **≥ ~0.716** would mean the margin was
mostly a tuning artefact, **0.704–0.710** that it survives at +0.005 to +0.010,
**above 0.7213** that we lose on our own folds. The outcome, 0.6987, fell **below
all three**. Draw 0 doubled as the positive control and behaved — inner 0.6909,
0.6851, 0.6875, 0.6909, 0.6905, with finals in the 0.686–0.702 band
`crisprhal_rerun.csv` predicts. Brackets that do not span the outcome are a limit
of pre-registration, not a success of it.

**Decision, 16 October: the ~40 GPU-hour fully-matched search will not be run.**
A genuinely matched search is ~100 draws per fold at ~5 GPU-minutes each, and it
would most likely buy a tighter bound on the same conclusion, since the
2.6-hour version already shows the search reverting to their published
configuration in four folds of five. The honest headline is the margin against
their **best known** configuration, **+0.0161**, with the asymmetry stated.
Reversing this decision needs a reason beyond wanting a larger number.

**On their own two splits none of this applies**, because there the comparison is
against their published weights — their tuning, their search, their published
numbers, which our harness reproduces to three decimals (ρ 0.782 → 0.7821 on
eSpCas9). Whatever budget they used is already inside the +0.005 and +0.002.

### What are all of XGBoost's hyperparameters, and does this project need them?

The champion is an `XGBRegressor`, so "have we tuned it?" has a countable
answer. The sklearn API exposes **40 constructor parameters**, twenty-two of them
plumbing — objective, threading, seeds, verbosity, device, the categorical group
this project never enables, the classification-only ones — which leaves **18
real knobs**. Six have been varied by a search here; **eleven had never moved.**

One piece of bookkeeping goes with that, because the report leaned on it for
weeks: **the 627-draw search was run on LightGBM, not XGBoost** (`tune.py`'s
`sample_config` draws `num_leaves` and `min_child_samples`, which are LightGBM
names). The only XGBoost search in the project is the 8-draw one inside the
sealed holdout.

**So the eleven were swept** — one at a time around the champion, 25 arms, on the
curated base with family A, selection cached per fold so the whole sweep costs
about one search (`xgb_knobs.py`, `results/xgb_knobs.csv`). Three had a reason to
be there beyond completeness: `colsample_bylevel` and `colsample_bynode` exist
for exactly the wide, redundant matrix this project has, and `max_bin` interacts
with a sparse binary one.

| knob | best value found | Δρ against the champion's 0.7084 |
|---|---|---:|
| `grow_policy` = `lossguide`, `max_leaves` 63 | — | **+0.0069** |
| `num_parallel_tree` | 4 | +0.0025 |
| `reg_alpha` | 1.0 | +0.0004 |
| `max_bin` | 64 | +0.0001 |
| `gamma` | any | −0.0002 to +0.0000 |
| `colsample_bynode` | 1.0 (the default) | −0.0009 at 0.8 |
| `colsample_bylevel` | 1.0 (the default) | −0.0014 at 0.5 |
| `max_delta_step` | 0 (the default) | **−0.0080** at 1 |

**Ten of the eleven do nothing, and two do harm.** Per-level and per-node column
subsampling — the two with the best prior reason to help — both want to be
switched off, which says the 300-column budget has already done the
decorrelating they exist for. `max_delta_step=1` costs eight thousandths, as it
should on an unbounded regression target.

**The eleventh looked like a find, and the control says it is capacity.** A
25-arm sweep reports the maximum of 25 draws on one fold partition, against an
adaptive component measured elsewhere here of **+0.0087** — larger than the
+0.0069 at issue. So the winner was re-run on fold partitions it had never seen,
beside a control that isolates what actually changed: `lossguide` with 63 leaves
is *two* changes, a different grow policy **and** four times the capacity, so
`depthwise` at `max_depth=6` was run next to it (`xgb_knobs.py --verify`,
`results/xgb_knobs_verify.csv`). Three fold partitions, fifteen paired
comparisons, bootstrapped:

| arm | Δρ against the champion | 95% CI | folds improved |
|---|---:|---|---:|
| `lossguide`, 63 leaves | +0.0067 | [+0.0054, +0.0080] | 15/15 |
| **`depthwise`, `max_depth=6`** | **+0.0063** | [+0.0053, +0.0073] | 15/15 |
| `num_parallel_tree=4` | +0.0024 | [+0.0019, +0.0029] | 15/15 |

Overlapping intervals 0.0004 apart, so `lossguide` is doing nothing that four
times the leaves does not do under the original policy. What the sweep found is
that **`max_depth=4` is too shallow for this cohort by about six thousandths of
ρ**, reproducibly, on every one of fifteen folds — and the champion's depth had
never been set by a measurement on this cohort, since it is inherited from the
baseline script and the LightGBM search varied LightGBM's own depth alongside
`num_leaves`. **It is not adopted:** +0.0063 is smaller than the +0.0087 adaptive
component on these same folds, the freeze is the whole point of Part 4, and a
six-thousandth gain on the cohort the configuration was selected against is
exactly what a sealed holdout removes. It is recorded as the first place to look
if the freeze is ever lifted.

**So, does the project need the other eleven?** No, and the reason is worth
stating in proportion. The whole measured span of hyperparameter tuning here is
**+0.0131**, of which **+0.0087 is adaptive inflation**; eight model classes span
0.108 ρ; the selection budget is worth ±0.0005; and one feature family is worth
**+0.164**. Hyperparameters are a hundredth-scale knob on a problem whose open
variance is 0.22 ρ. The one place they are not is a *different dataset*: on the
human matrix depth and learning rate are worth **+0.10**, and `max_depth` 4 → 6
alone is **+0.0585** against +0.0063 here. Tuning buys insurance against a change
of signal shape, not a better number on a signal already fitted.

### Does grouped cross-validation still mean the same thing after the cohort moved?

No, and the direction is the uncomfortable one. Every grouped-CV argument rests
on one number: the share of label variance lying **within** genes rather than
between them. If nearly all of it is within, a model cannot score by memorising
which gene a guide sits in, and a random split and a gene-grouped split measure
nearly the same thing.

| cohort | guides placed in a gene | genes | guides/gene | within-gene share |
|---|---:|---:|---:|---:|
| published (13,880 rows) | 13,829 | 1,222 | 11.3 | **0.843** |
| curated (33,567 rows) | 30,203 | 4,013 | 7.5 | **0.713** |
| published, 100 kb blocks | 13,879 | 47 | 295 | 0.980 |
| curated, 100 kb blocks | 33,567 | 47 | 714 | 0.973 |

**Gene identity explains 16% of the published label and 29% of the curated one.**
The reason is what each cohort is: Noshay's rows target a gene panel, while
crisprHAL's curated release is genome-wide — 4,013 genes at 7.5 guides each. More
genes, fewer guides in each, and more of the variance sits between them.

**The prediction that follows from this is wrong, which is why it was tested.**
Twice the between-gene variance should mean a larger gap between random and
gene-grouped CV. The gap is *smaller*: grouping cost `e_nucleoid` 42% of its gain
on the published matrix and 12% here, `g_transcription` 36% and 21%
(Appendix B3). The missing variable is **guides per gene** — 7.5 here against
11.3 — because between-gene variance sets how much a locus-memorising model could
win while guides per gene sets how well it can learn each locus's mean. Both
moved, in opposite directions, and only the measurement says which won.

Gene assignment is by coordinate on NC_000913.2 against the reference gene
table, not from either dataset's own annotation, so both cohorts are measured the
same way. On the published rows that assignment agrees with Noshay's own for
**97.7%** of guides and gives the same share to three decimals (0.843 against
0.844) — which is the check that the change above is the cohort and not the
method. `python -m sgrna.cohort_groups`, `results/cohort_groups.csv`.

### Where the published label's 0.937 ceiling came from

Two screens of the same library agree at ρ 0.810, which naively gives 0.90 — but
the disagreement is **predictable at ρ 0.546**, so it is enzyme biology, not
noise. Using the source paper's own agreement statistics instead, and accounting
for the published score being a **two-replicate geometric mean**:

| route | r | reliability | ceiling |
|---|---:|---:|---:|
| **tiling library, a *different* library** | 0.878 | 0.878 | **0.937 [0.929, 0.945]** |
| replicate agreement + Spearman–Brown | >0.883 | 0.938 | 0.969 |

**Lead with 0.937.** The replicate route must assume the activity score inherits
the single-arm read counts' reliability, and it cannot: the label is the `|Z|` of
a Cas9/dCas9 **ratio**, and a difference is never more reliable than its
components — at 50% shared signal between arms the implied ceiling is already
0.940. The tiling route needs none of that machinery.


*Why this is kept.* It is the ceiling for Noshay's `cut_score`, not for the
curated label we train on, and the curated figure (0.946, Finding 6) comes from
a replicate split that *is* available for that label. Both appear in the report
because arms on the two cohorts are compared, and quoting 0.937 against a curated
number would understate the headroom.

### The reduced-representation ladder, and the budget that was never tested

Finding 1's reduction was established on the published matrix. All arms from one
protocol, 5 seeds × 5 folds, same rows:

| arm | columns | ρ | vs full matrix |
|---|---:|---:|---:|
| mononucleotide only | 74 | 0.5081 ± 0.0010 | −0.0191 |
| + dinucleotide | 364 | 0.5189 ± 0.0009 | −0.0083 |
| + non-sequence features | 427 | 0.5249 ± 0.0014 | −0.0023 |
| − 15 undecoded (**portable**) | **412** | **0.52713 ± 0.0011** | **−0.00003** |
| published full | 6,232 | 0.52716 ± 0.0022 | — |

Non-inferiority of the 427 is formal: loss **+0.0023, 95% CI [−0.0008,
+0.0054]**, one-sided p = 0.035 against a 0.005 ρ margin that was written down a
week after the result rather than before it (Appendix B). That is
the right form of claim — "as good as" cannot be established by failing to
reject a difference.

*Protocol caveat, found 13 October.* Every arm above selects **300 columns from
its candidate pool**, because `cross_validate` resolves `n_features=None` to
`config.N_FEATURES`. So "all columns" was never tested here: the 74-column arm
uses all of its candidates while the 6,232-column arm uses 300 of 6,232. The
ladder measured *candidate pools under a fixed selection budget*, not
representations in full. The budget turns out not to bind — on the curated
cohort, selecting 300 against using every column differs by ±0.0005 in both
directions (Finding 1) — but the caveat had to be measured rather than waved
away, and when it was, **the reduction itself stopped holding on the curated
cohort**. Every arm in the table above was then re-run with **no selection at
all**, `n_features` set to each arm's own column count, which is what makes the
tetramer result (−0.0019), the quantum-column result (+0.0114) and the
non-inferiority failures statements about representations rather than about the
selector.

**B.2 — What is the model class worth?**

### What sixteen model classes and 220× of compute tell us about the task

Accuracy against cost, all on the curated cohort with family A, one seed × five
folds, against crisprHAL 2 on the same folds (`curated_models.csv`,
`crisprhal_tuned_folds.csv`):

| | ρ | CPU-minutes, 5 folds | relative cost |
|---|---:|---:|---:|
| **XGBoost** (champion) | **0.7084** | 1.9 | 1× |
| LightGBM | 0.7082 | 1.4 | 0.7× |
| extra trees | 0.6537 | 1.2 | 0.6× |
| random forest | 0.6482 | 1.3 | 0.7× |
| ridge | 0.6473 | 1.0 | 0.5× |
| PLS | 0.6450 | 1.0 | 0.5× |
| elastic net | 0.6369 | 1.0 | 0.5× |
| k-NN | 0.6007 | 1.0 | 0.5× |
| crisprHAL 2, published configuration | 0.6971 | 85 | 45× |
| crisprHAL 2, its best configuration | 0.7051 | 220 | **113×** |

**Boosting beats a tuned CNN–BiGRU by +0.003 at 1/113th of the compute, and
*ridge* is within 0.058 of it at 1/220th.** A linear model on 300 engineered
columns, fitted in one CPU-minute, gets within six hundredths of a deep network
that takes three and a half hours. That is the cost argument in its strongest
form, and it does not depend on our model winning anything.

**What the spread says about the task.** Reading the classes by what each is good
at is more informative than ranking them:

* **ridge ≈ PLS ≈ random forest ≈ extra trees, all within 0.009.** Three very
  different inductive biases — penalised linear, low-rank linear, bagged trees —
  land in the same place. When bagging buys nothing over a straight line, the
  problem is **bias-limited, not variance-limited**, at this sample size.
* **boosting is worth +0.055 over ridge.** Sequential bias reduction is the one
  thing that helps, which is what a sum of many small, partly-interacting effects
  calls for. On the published matrix *without* family A that margin was 0.03;
  adding family A nearly doubles it, so **family A is also where the exploitable
  interaction structure is**, not merely where the signal is.
* **k-NN is worst, at 0.601.** Nearest neighbours in 300 dimensions find almost
  nothing, so the signal is **smooth and global rather than clustered** — there
  is no neighbourhood of similar guides to borrow an answer from. That is
  consistent with Finding 2's gradient and inconsistent with a motif.

**And the comparison is not architecture against architecture.** crisprHAL 2 at
0.7051 sits between our boosters and our forests. The thing that separates the
top two rows from the bottom six is not depth, recurrence or ensembling — it is
**which columns the model is given**, since every row above reads the same 300.
Representation decides the task; model class decides a tenth of a correlation
(Finding 1).

### Why does boosting beat random forest, and ridge beat both forests?

| | ρ |
|---|---:|
| LightGBM / XGBoost | 0.609 / 0.608 |
| **Ridge** | **0.577** |
| extra trees / random forest | 0.556 / 0.550 |
| k-NN | 0.495 |

**The signal is overwhelmingly additive** — a straight line gets within 0.03, so
the interactions boosting captures are worth ~0.03 ρ. Boosting reduces bias
sequentially, which is the right move for a sum of many small independent
effects; bagging reduces variance, which is not the binding constraint. Random
forest's `max_features='sqrt'` offers only ~81 of 6,517 columns per node, most of
them restatement. And **k-NN failing confirms the shape**: guides with similar
features do not have similar scores, which is what many small additive terms
look like.

Checked on a basis with the redundancy removed, because the gap was measured on
selected columns: on 427 columns ridge reaches **95.6%** of boosting's ρ
(0.5029 against 0.5258), re-derived independently at 95.8%. **So the additivity
is not an artefact of feature selection.**

*What does not survive re-derivation is the companion sentence.* This appendix
said that ridge on the raw 6,232 columns "collapses to 0.423 from collinearity",
and the collapse is real only at a **fixed** penalty: `Ridge(alpha=1)` gives
0.4231, reproducing the stored figure, while `RidgeCV` over 10⁻²–10⁴ gives
**0.4945**. Choosing the penalty is what ridge is *for* under collinearity, so
the original number measured an unsearched hyperparameter rather than a property
of the matrix. The reduced basis still helps a linear model, by **+0.009**
(0.5036 against 0.4945) rather than by +0.071 — a real effect and a small one
(`python -m sgrna.rederive --only additivity_reduced`).

### What the interpretability numbers license, and what they do not

Finding 7 quotes three of them. Each is worth stating precisely, because two are
weaker than they look and one is stronger.

**`max_depth = 4` caps the interaction order at four** — no term in the fitted
model combines five features, which is a structural guarantee rather than an
empirical observation. But it is a guarantee about *this configuration*, and the
configuration was **inherited, not chosen**: depth 4 comes from
`FINAL_baseline_ecoli_sgRNA_model_5seeds.py` along with `learning_rate=0.05`,
400 trees and the 300-column budget. It was tested after the fact rather than
selected, and it survives the test — a 627-draw nested search buys only +0.0131 ρ,
and XGBoost's champion and LightGBM at unrelated defaults land within 0.0002 of
each other on the curated cohort with family A. What the search does *not* do is
endorse the number: the five per-fold winners chose depths of **4, 8, −1, 3 and
−1**, while three of them chose `num_leaves = 8`.

| fold | `max_depth` | `num_leaves` | `n_estimators` | Δρ over the inherited default |
|---:|---:|---:|---:|---:|
| 1 | 4 | 63 | 1,600 | +0.0139 |
| 2 | 8 | 8 | 1,600 | +0.0111 |
| 3 | −1 | 8 | 1,600 | +0.0149 |
| 4 | 3 | 127 | 1,600 | +0.0117 |
| 5 | −1 | 8 | 1,600 | +0.0136 |

What the folds agree on is **low capacity** and 1,600 trees at a small learning
rate; depth is only one of the two ways the space offers to express it, and
`num_leaves = 8` is the other. So the defensible statement is that the fitted
model is shallow, not that 4 is optimal — and the setting is data-specific, since
on the human matrix, where the signal is not additive, depth 6 is worth +0.0585.

**280 of 300 columns are ever split on, 45 carry half the gain, top ten 19.5%.**
This is the weakest of the three as an interpretability claim: it says the model
is neither a two-feature rule nor an undifferentiated smear, which rules out the
two extremes and little else.

**Ridge reproduces 95.6% of boosting's ρ on a decorrelated basis**, and this one
is the strongest. *What ridge is here, since the comparison means nothing if that
is unclear:* not a line through two numbers, but a **weighted sum with one
coefficient per column** — 428 fitted numbers on the 427-column basis, a
hyperplane in 427 dimensions. "Linear" means each column contributes a fixed
amount per unit with no term depending on two columns at once. That is exactly
what makes the comparison informative: boosting may use interactions and ridge
cannot, so the gap measures **how much of the accuracy needs interactions at
all**. About 4% — which is why the account in Finding 2 can be read off
the model as a set of per-column effects rather than inferred from attributions.

**B.3 — What is the representation worth?**

### Is re-description worth anything, or only the *right* re-description?

Finding 1 reports that a deterministic function of the 20-mer adds no
information but buys a depth-limited tree one to two hundredths of ρ, agreed
across four families. The proposed mechanism — the tree cannot cheaply compute a
sum of position-wise terms, so handing it one precomputed pays — has a sharp
consequence nobody had tested: **the function would not have to mean anything.**
Any deterministic summary packaging a higher-order effect should do. And since
performance cannot rise without bound by bolting on arbitrary functions, the
effect has a size and a saturation point.

**It has neither, because it is zero.** `redescription.py` builds functions
structurally identical to the physical ones and semantically empty: a random
weight per (position, base), summed over the guide — exactly the shape of a
nearest-neighbour energy sum with the physics replaced by noise. Three
independent draws of tables at each count, two model seeds each, added to the
curated base:

| random tables added | mean Δρ | range across three draws |
|---:|---:|---|
| 4 | −0.0004 | −0.0006 to −0.0002 |
| 16 | −0.0004 | −0.0007 to +0.0002 |
| 64 | −0.0002 | −0.0010 to +0.0006 |
| 256 | **−0.0015** | −0.0021 to −0.0010 |
| *for comparison, the physical families* | *+0.011 to +0.019* | |

**Nothing, at every count, and at 256 it starts to cost** — enough useless
columns begin to crowd the 300-column budget. So the gain is **not** a property
of re-description as a class. It belongs to the particular functions, which
means the physics is doing the work and the repackaging is only how the work
gets delivered. "Representation beats information" was too strong: the honest
version is that a *well-chosen* deterministic summary pays a constrained learner
about a hundredth, and choosing it well is domain knowledge, not formatting.

**That last sentence is the one that survived**, and it survived a second
control. The random tables are linear sums over position-specific weights, which
is exactly the class `b_energy`, `d_mechanics` and the quantum tables live in —
so for a while the reading was that linearity, not meaninglessness, was why they
paid nothing, and that any function a shallow tree cannot reconstruct would do.
Random **non-linear** functions of the guide, matched on out-of-fold
recoverability to the real fold energy, say otherwise: +0.0001 to +0.0007 at
every interaction order from 2 to 8, against +0.0117 for the real column (see
*Every family split the way `d_mechanics` was*). Choosing the summary well is
domain knowledge, and the domain is doing the work.

### Columns sorted by what the guide determines, instead of by family

A family is a folder, not a kind. `d_mechanics` bundles 22 columns that restate
the guide with 8 that read flanking DNA, and its family-level Δρ hides the split
(Finding 1). So the taxonomy was rebuilt at the column level: regress **all 668
family columns** on the guide's own one-hot encoding (20×4 mono + 19×16 di = 384
predictors), out of fold, and sort by how much of each column the guide explains
(`redescription.py --arm taxonomy`, `results/redescription_columns.csv`).

| | columns | Δρ over the curated base |
|---|---:|---:|
| **linearly recoverable from the guide** (R² ≥ 0.99) | 18 | **+0.0083** |
| partly recoverable (0.50 < R² < 0.99) | 16 | +0.0199 |
| **not linearly recoverable** (R² ≤ 0.50) | 602 | **+0.2012** |
| constant on this cohort, so no R² exists | 32 | −0.0003 |
| all 668 together | 668 | +0.2027 |

**Everything the guide determines is worth +0.008; everything it does not is
worth +0.20** — a factor of twenty-four, measured without reference to anyone's
choice of folder. The sum of the three parts (+0.229) exceeds the joint arm
(+0.203), which is the same overlap the family table shows.

The classification checks itself against two known cases: 13 of `d_mechanics`'s
30 columns come back as guide-determined and 10 as outside, against the 22/8
split measured a different way in Finding 1; and 320 of `a_flank`'s 348 land
outside, as do all 152 of `i_shape`, as they must.

**And it turned up 32 columns that can never contribute anything, for two
different reasons.** Both are properties of the feature set rather than of the
model, and neither had been noticed:

* **26 of family A** are `eng.flank.qct.dimer.Noelectrons.*` — the electron
  count of the dinucleotide at each position — and every one of them is
  **196 for every guide**. That is chemistry, not a bug: an A·T base pair and a
  G·C base pair contain the same number of electrons, so a per-base-pair electron
  count cannot vary with sequence. The descriptor is degenerate by construction
  and has been carried in every family A run since September.
* **6 of `h_offtarget`** — `mm1`, `mm2` and `seed_exact_mm1..4` — are **zero for
  every guide**, because the curated cohort keeps only uniquely-targeting guides,
  so no close off-target survives the filter. Those columns measure something
  the cohort has defined away.

Added as a block they are worth **−0.0003**, which is the right answer for 32
constants competing for a 300-column budget.

**A third defect of the same kind turned up later: duplication across families.**
`eng.dg.guide_selffold` (family B) and `eng.fold.spacer_mfe` (family C) are the
same spacer fold energy at Pearson 0.9993, and `eng.fold.scaffold_delta` is
`eng.fold.full_mfe` shifted by the constant scaffold energy. Constants, degenerate
descriptors and cross-family duplicates are all invisible to a family-level
ablation, which is the recurring lesson of this appendix: **the folder is not the
unit of analysis, and nobody had read the column lists.**

*What the probe measures, and what it does not.* It measures **linear**
recoverability from mono- and dinucleotide indicators, which is a lower bound on
determinism and a reasonable proxy for *cheapness* to a depth-limited tree — the
quantity the mechanism in Finding 1 is actually about. It is not a test of
whether a column is a function of the guide: `c_folding` is deterministic in the
guide and all seven of its columns land in the "not linearly recoverable" group,
because RNA folding is not a linear sum. Read the groups as *cheap*, *partly
cheap* and *not cheap*, not as *determined* and *free*.

### Is the folding gain the physics, or only the non-linearity?

Two results above are consistent with two incompatible readings. Random
*linear* functions of the guide buy nothing, so re-description as a class is not
worth anything. But the guide-determined columns that *do* pay are the ones a
linear probe cannot see — the spacer fold energy at **+0.0119**. Either

* **(a)** RNA secondary structure is part of the mechanism, and an explicitly
  computed fold delivers a real effect the model cannot reach from indicators; or
* **(b)** nothing biological is involved — any non-linear, non-local summary of
  the guide would pay a depth-limited tree the same hundredth, and folding is
  merely the one that happened to be computed.

The two are separated by **keeping the algorithm and removing the biology**
(`redescription.py --arm folds`, `results/redescription_folds.csv`).

**The control that decides it.** `shuffled-position MFE` applies **one fixed
permutation of the twenty positions** to every guide and folds the result with
real ViennaRNA. Same algorithm, same energy model, same value distribution, same
non-linearity, still a deterministic function of the guide — and still the same
base composition, which makes the control conservative: whatever part of the
gain is composition-driven foldability survives in it, and only the
position-specific structure is removed.

| arm | Δρ over the curated base |
|---|---:|
| real `eng.dg.guide_selffold` (the published column) | **+0.0117** |
| spacer MFE, recomputed here through the same code path | **+0.0119** |
| real `c_folding` block, 9 columns | +0.0153 |
| **shuffled-position MFE**, permutation 0 | **+0.0000** |
| **shuffled-position MFE**, permutation 1 | **−0.0007** |
| **shuffled-position MFE**, permutation 2 | **+0.0002** |

**Scrambling which base sits where destroys the entire effect.** Three
permutations average **−0.0002** against the real column's +0.0119. So it is
neither the composition nor the fact that folding is expensive for a tree to
compute: it is the particular structure the real spacer forms. Reading (a)
survives and (b) is excluded.

That is worth stating plainly because it is the first result in this project
where a **zero-information** feature is shown to carry **real biology** rather
than a representation convenience. The mechanism it supports is the standard one
— a spacer that base-pairs with itself is less available to invade the target
duplex — and the recomputation through an independent code path (+0.0119 against
the published column's +0.0117) rules out a plumbing artefact.

**Where the earlier random-table null fits.** The random tables were *linear*
sums over position weights, so they lose twice over: no information, and cheap
for the tree to approximate from indicators it already holds. A real fold loses
once and wins once. A secondary arm closes the remaining cell — random symmetric
4×4 pairing energies run through a Nussinov minimisation, which is non-linear
and non-local but carries no chemistry — and it is a null as well
(**+0.0008** on the first table).

*A design limit worth recording, because it nearly went unnoticed.* The
random-energy arm is **not** a matched control, and the matching was checked
rather than assumed: the out-of-fold linear recoverability of a random-energy
fold runs **0.391 to 0.855** across three tables against the real column's
**0.316**, because a random table usually has one dominant pair type and the
minimisation collapses to counting it. So most random-energy folds are not
actually expensive, and their null would have been uninformative on its own.
The position-shuffled arm is the one that carries the argument, which is why it
was built after the random one and not instead of it.

### Every family split the way `d_mechanics` was

The 22/8 split inside `d_mechanics` was found by hand, on one family, and used
to argue that family-level ablations can hide structure. The column probe makes
that test cheap to repeat everywhere, so all six families with enough columns
were run as two arms rather than one
(`redescription.py --arm families`,
`results/redescription_family_split.csv`).

| family | cheap half | not-cheap half | whole family |
|---|---|---|---|
| `a_flank` | 2 cols, +0.0037 | 320 cols, **+0.1643** | +0.1635 |
| `b_energy` | 9 cols, +0.0131 | **1 col, +0.0116** | +0.0181 |
| `c_folding` | 2 cols, +0.0046 | 7 cols, +0.0113 | +0.0147 |
| `d_mechanics` | 20 cols, +0.0113 | 10 cols, **+0.0970** | +0.0987 |
| `f_methylation` | 1 col, −0.0002 | 16 cols, +0.0080 | +0.0080 |
| `h_offtarget` | — | 7 cols, +0.0003 | +0.0009 |

**`d_mechanics` reproduces.** The probe splits it 20/10 at +0.0113/+0.0970,
against the 22/8 split measured by hand at +0.0110/+0.0973 — two unrelated
methods, agreeing to a thousandth. The lesson generalises in the weak sense: in
every family the two halves differ, and in four of them the whole family's
number is within 0.001 of its larger half, so the family-level figure is
reporting one half and carrying the other for free.

**But the halves are not the same kind of thing in every family, and this is
where the earlier wording was wrong.** The probe measures *cheapness*, not
provenance (see the limit above), and the two coincide only sometimes:

* In `a_flank`, `d_mechanics` and `h_offtarget` the not-cheap half really is
  **outside the 20-mer** — flanking DNA at ±25 and ±100 nt, genome-wide
  off-target counts. Provenance and cheapness point the same way.
* In `b_energy` and `c_folding` they do **not**. Every one of those eight
  not-cheap columns is computed from the guide and nothing else; they are
  ViennaRNA folds and a self-folding energy, which are not linear sums, so a
  linear probe cannot see them. The information content is zero and the gain is
  not.
* `f_methylation` is mixed by construction: `_motif_features` searches a window
  of `upstream + protospacer + PAM + downstream`, so `in_pam`, `in_pam_flank`,
  `n_in_window` and `nearest_to_cut` read beyond the guide by definition while
  `in_protospacer`, `in_seed` and `in_distal` are all but guide-only.

**In `d_mechanics` the two kinds are separable by name, so the distinction was
measured rather than argued** (`redescription.py --arm provenance`,
`results/redescription_provenance.csv`). Its ten not-cheap columns are eight
`eng.mech.flank.*` that read ±25 and ±100 nt, plus `slide.strongest_from_pam`
and `stack.min`, which are an argmax and a min over the guide's own dinucleotide
stack — guide-only, and non-linear, which is why the linear probe files them
outside. Run as separate arms:

| | columns | Δρ |
|---|---:|---:|
| reads outside the 20-mer | 8 | **+0.0968** |
| guide-only but non-linear | 2 | +0.0019 |
| both halves together | 10 | +0.0970 |

**So `d_mechanics` is a provenance story and not a non-linearity story**: the
eight flank columns carry essentially all of it, and the two expensive guide-only
columns are worth two thousandths. The non-linearity effect is real but it lives
in the folding columns, not here — which is why both kinds have to be named
separately instead of being read off one R² threshold.

**The sharpest single column in the project sits in that second group.**
`eng.dg.guide_selffold` — the folding energy of the guide RNA with itself — is
`b_energy`'s entire not-cheap half: **one column, R² 0.316 from the guide's own
mono- and dinucleotide indicators, worth +0.0117 on its own.** It adds no
information whatsoever. It is a deterministic function of twenty bases the model
already has. And it buys more than the other nine columns of its family put
together.

**And `b_energy` and `c_folding` are not two independent witnesses to that.**
`eng.dg.guide_selffold` and
`eng.fold.spacer_mfe` are **the same quantity computed twice in two different
folders** — Pearson 0.9993, identical on all but a handful of guides where the
two modules prepare the sequence slightly differently. So the single column
carries +0.0117 of `c_folding`'s whole-block +0.0153 as well, which is 76% of
it, and the apparent agreement between two families is one measurement filed in
two places. (`eng.fold.scaffold_delta` is likewise `eng.fold.full_mfe` minus a
constant — the fixed scaffold's own energy, −15.3 — so those two are collinear
by construction.) The independent evidence for the effect is therefore **two
measurements, not three**: the spacer fold energy, and `d_mechanics`'s 20 cheap
guide-only columns at +0.0113.

**That reconciles the random-tables null instead of contradicting it — and the
cell those two results leave empty has now been run.** The random tables were
linear sums over position-specific weights, which loses twice: they carry no
information *and* a tree can approximate them cheaply from the one-hot columns it
already holds. `guide_selffold` loses only once, because no shallow tree is going
to reconstruct a base-pairing partition function from indicators. That left one
reading standing — that difficulty alone is what pays — and one cell untested: a
random function that is **non-linear** rather than linear.

It has now been run, with recoverability measured rather than assumed
(`redescription.py --arm nonlinear`, `results/redescription_nonlinear.csv`). A
sum of 200 random weights over random position *tuples* is uninformative by
construction and becomes arbitrarily expensive as the tuple size rises, with
nothing else about it changing. Three independent draws at each order, one model
seed, five folds:

| random function of the guide | out-of-fold linear R² | Δρ |
|---|---:|---:|
| order 1 — the earlier random tables | 1.000 | +0.0001 |
| order 2 | 0.402 | +0.0000 |
| order 3 | 0.178 | +0.0002 |
| order 4 | 0.075 | −0.0007 |
| order 6 | −0.001 | +0.0007 |
| order 8 | −0.008 | −0.0005 |
| **real `eng.dg.guide_selffold`, for scale** | **0.316** | **+0.0117** |

**Difficulty is dialled across the whole available range and the gain never
moves.** The order-2 arms are the matched ones — individually R² 0.339, 0.383 and
0.484 against the real column's 0.316 — and they buy −0.0004, −0.0002 and
+0.0007. No arm in the table is further than 0.0016 from zero, while a real fold
energy of the same recoverability buys +0.0117. Order 1 reproduces the earlier
random-tables null at +0.0001, which is the check that this instrument agrees
with the one it extends.

**So "cheap against expensive" was the wrong axis, and this retires it.** Being
expensive for the tree is not sufficient, and the family table above says it is
probably not necessary either: `b_energy`'s nine *cheap* columns are worth
+0.0131 as a block, about what its one expensive column is worth alone. What
separates a function that pays from one that does not is whether it tracks the
phenotype. `guide_selffold` carries a marginal \|ρ\| of **0.102** against the
label; a random function of the guide carries none, by construction. So the
statement this line of work can support is the one it started from, now with a
matched control under it rather than an assumption:

> **a deterministic function of the guide pays a depth-limited tree about a
> hundredth of ρ when it names something the phenotype depends on, and nothing
> at all otherwise, however expensive it is to compute.**

Difficulty survives as a second-order ordering *among* the functions that
qualify, and that is where the original observation was right: per column, the
one expensive aligned column is worth +0.0116 where nine cheap aligned ones
average +0.0015 each.

*And it agrees with the shuffle, which is the stronger of the two controls.* The
position-shuffled fold holds the algorithm, the energy model, the base
composition **and** the difficulty fixed, and destroys only the correspondence
between structure and guide; it is worth +0.0000 and −0.0007 (Part 2). Two
different ways of removing the meaning while keeping the difficulty, and both are
nulls.

*Scope.* The positive side still rests on `n = 1` column at +0.0116 and seven at
+0.0113 in `c_folding`, on one cohort, with the usual 300-column budget competing
underneath. It is a mechanism with a measurement attached, not a law.

### Does the human baseline reproduce ρ 0.472, and what produces it?

Part 3's cross-kingdom sentence had carried, for weeks, a figure with no
provenance: a teammate's grid search reaching ρ 0.472 on the human iRF matrix
against the 0.404 this harness measures. The settings arrived on 16 October
(`Jacky/human-boundary/HUMAN_BASELINE_SETTINGS.md`): **400 selected columns,
1,000 estimators, every other XGBoost knob at library defaults**, from a personal
script predating the merged pipeline. Three arms on the same folds of
`data/raw/human_feature_matrix.csv` (17,421 guides × 6,233 columns), 3 seeds × 5
folds (`src/sgrna/human_baseline.py`, `results/human_baseline.csv`):

| arm | columns | XGBoost settings | ρ |
|---|---:|---|---:|
| `harness_default` | 300 | champion | 0.4036 ± 0.0128 |
| `jacky_cols_only` | 400 | champion | 0.4074 ± 0.0148 |
| **`jacky_400_1000`** | **400** | **library defaults, 1,000 trees** | **0.5076 ± 0.0149** |

**It reproduces, and overshoots: 0.5076 against the 0.472 quoted.** The third
arm exists because the first two differ in one thing and the quoted
configuration differs in two, and only one of them can be doing the work. The
answer is unambiguous — **the column count contributes +0.0038, inside a single
fold's spread; the hyperparameters contribute +0.1002.**

*Why the design settles it.* Holding the folds, the selector and the seeds fixed
and moving one factor at a time is the only way to attribute a difference
between two historical runs, because the two differences are confounded in the
original comparison. Without `jacky_cols_only` the natural reading would have
been "human needs more features", which is the opposite of true.

*Why the library defaults are not a detail.* XGBoost defaults to
`max_depth=6, learning_rate=0.3`; the champion uses `4` and `0.05` with
`subsample=0.8, colsample_bytree=0.8`. On a 17,421-row matrix that is a
substantially higher-capacity, faster-learning model, not a tweak — and the
teammate's note records that on human `n_estimators` kept improving with no
plateau, unlike bacteria.

**What this costs the report.** Not the cross-kingdom conclusion — nothing
transfers either way, and that survives (below) — but one sentence supporting it
does, and the assumption underneath both: **the champion is tuned on one
*E. coli* screen and is mis-specified by 0.10 ρ on a matrix of different
shape.** Every external arm here — TevSpCas9, *C. rodentium*, eSpCas9, Cui — is
fitted with it, and each therefore reports a lower bound on what a locally-tuned
model would reach. For the comparisons that are *differences between arms on one
dataset* this mostly cancels; for absolute transfer figures it does not, which is
exactly where it bit.

**Which knob, and what it says.** The quoted configuration changes three things
at once, so each was moved alone from the 400-column reference
(`results/human_baseline_knobs.csv`):

| change from the champion | ρ | Δ |
|---|---:|---:|
| `max_depth` 4 → 6 | 0.4659 | **+0.0585** |
| `learning_rate` 0.05 → 0.3 | 0.4490 | +0.0416 |
| `n_estimators` 400 → 1,000 | 0.4435 | +0.0361 |
| no row/column subsampling | 0.3999 | −0.0075 |
| all three (= `jacky_400_1000`) | 0.5076 | +0.1002 |

Sub-additive, as three capacity knobs should be. **The largest single term is
tree depth** — not shrinkage, not tree count — and depth is the interaction-order
knob, which makes this a candidate statement about the signal rather than about
our settings. Depth also buys plain capacity, though, so the hyperparameter
cannot establish it alone.

**The direct test says the human signal really is less additive.** Appendix B's
model-class comparison found *E. coli*'s signal overwhelmingly additive: ridge
reaches **95.6%** of boosting's ρ. Repeating exactly that test on the human
matrix, on the same 400 selected columns, standardised, with `RidgeCV`
(`results/human_additivity.csv`):

| | *E. coli* | human |
|---|---:|---:|
| boosting | 0.608 | **0.5076** |
| ridge | 0.577 | **0.3232** |
| **ridge / boosting** | **95.6%** | **63.7%** |

A straight line gets within 0.03 ρ of boosting in *E. coli* and falls 0.18 short
in human. **Human guide efficiency carries interaction structure that *E. coli*'s
does not**, and that is why a champion tuned on a nearly-linear signal — shallow
trees, heavy shrinkage — is the wrong model there. Against our champion alone
ridge still reaches 79.3%, which is the same fact from the other side: the
champion's effective capacity is close to a linear model's.

This also **bounds the caution above**. The champion fails to travel across
kingdoms because the shape of the signal changes, not because hyperparameters are
fragile in general — which is consistent with it transferring without incident
across the bacterial sets, where the near-additivity measured above holds. The two
percentages are not measured on an identical basis (*E. coli*'s on 427
decorrelated columns, human's on 400 selected ones), so the comparison is a
factor-of-magnitude one, not a precise ratio.

**And it did bite, once.** Part 3's boundary claim had both transfer directions
negative with intervals excluding zero — measured with the human model fitted at
champion settings. Refitted at the human-appropriate ones and bootstrapped over
test rows (`src/sgrna/kingdom_transfer.py`, 6,216 columns shared by name, 2,000
resamples):

| direction | champion | human-fitted | human-fitted, seeds pooled | 95% CI |
|---|---:|---:|---:|---|
| *E. coli* → human | −0.0462 | −0.0367 | −0.0393 | [−0.0542, −0.0242] |
| human → *E. coli* | −0.0155 | −0.0055 | **−0.0060** | **[−0.0227, +0.0111]** |

The first two columns are means of three single-seed fits; the third averages the
three seeds' predictions before scoring, which is what the interval is
bootstrapped on. The two conventions agree to 0.003 and are kept separate rather
than mixed.

Reproduces the previously quoted −0.048 and −0.017 at champion settings, and
then moves one direction to zero. **The claim that both intervals exclude zero
was an artefact of the undertuned human model** and is withdrawn (Part 5); the
boundary itself is not, since neither direction transfers. Note which direction
moved: the one *trained* on human, which is the only one whose hyperparameters
were wrong for its training data. The seed-to-seed spread of three fits is model
noise and says nothing about this — at ρ this close to zero only resampling the
test rows can, which is why the interval is bootstrapped rather than taken from
the seeds.

The one place it could bite is the Cui null in Finding 3, where family A adds
−0.009 across that CRISPRi screen. An underfit model that could not use the
flank columns at all would produce the same null. It did not happen here: given
nothing else, those same 348 columns reach ρ **0.1333** on Cui under gene-grouped
CV, so the model demonstrably reads them and still gains nothing from them
alongside the base — which is a displacement result inside a 300-column budget,
not an inability to fit.

**B.4 — Mechanism: what the effect is, and what it is not**

### What other long-range features could be built without new data?

Two of the three most mechanistically motivated candidates have now been built
and are nulls: **GC skew, A-tract phasing and bendability periodicity**, as
family J (−0.001 alone, +0.0004 on top of family A, with its own permutation
control scoring higher), and **Chi sites**, in the section below. That makes the
long-range asymmetry harder to explain rather than easier — the candidates that
came with a mechanism attached are now excluded rather than unexamined. What
remains unbuilt is k-mer entropy, homopolymer run-length and REP density from the
genome, and from RegulonDB the distance and orientation to the nearest annotated
TSS, terminator and operon boundary. None of the first three carries a mechanism
of its own; the RegulonDB items are the properly-annotated version of the
nearest-gene proxy behind Jacky's orientation result, and so the highest-value
thing left on this list.

### Is the flank effect a *repair* effect? The Chi-site test, and it is a null

**What motivates this section.** The dCas9 comparison in Finding 3 puts the
effect at or after the cut, which is a hard constraint: almost nothing at a cut
site has any reason to read five hundred nucleotides. The route below is the one
mechanism in *E. coli* that does. It is excluded by measurement against a
GC-matched null, so the constraint stands and this candidate does not.

The flank effect is **large** (+0.164), **smooth and compositional with no
motif** out to ~500 nt (Finding 2), and **orientation-dependent** — the last
being Jacky's result, not this project's.

That is an awkward constraint, because almost nothing at a cut site has any
reason to read five hundred nucleotides. The Cas9 HNH conformational change does
not. One thing in *E. coli* does: **RecBCD** enters a double-strand break and
degrades processively over kilobases until it meets a Chi site (`GCTGGTGG`),
where it is attenuated and hands off to RecA for homologous repair. And in a
pooled depletion screen the phenotype is not cutting but **dying** — a guide
scores as efficient when its cells fail to survive the break — so anything that
improves repair must lower measured efficiency.

The prediction is specific. It would explain the length scale (processive
degradation over kilobases), the absence of a motif at any fixed position (what
matters is the density over a long tract), the cleavage-specificity (dCas9 makes
no break to process), and the orientation dependence (Chi is functional only
when RecBCD approaches it from one side).

**The signs, however, go the wrong way, and that was worth noticing before
running anything.** Better repair means less death means *lower* measured
efficiency, and Chi sites are GC-rich — so the mechanism predicts GC-rich flanks
scoring low. They score **high**: flank GC correlates **+0.161** with the label
(`eng.flank.comp.up250.gc`). The −0.201 association this report quotes
elsewhere is **target** GC, inside the protospacer, which carries the opposite
sign to the flanks and is the reason conditioning on it *raises* the flank term
(Finding 5, biology item 9). So the hypothesis was already in difficulty on
direction, and the test below settles it for a different and more decisive
reason.

**It is a null.** `chi_sites.py` counts `GCTGGTGG` and its reverse complement on
each side at ±250, ±500 and ±1,000 nt across all 33,567 curated guides
(`results/chi_sites.csv`). Chi is common enough to test — 11,677 guides carry at
least one within ±1 kb.

| | ±250 nt | ±500 nt | ±1,000 nt |
|---|---:|---:|---:|
| guides carrying ≥1 Chi | 3,305 | 6,469 | 11,677 |
| Spearman with the label | +0.028 | +0.039 | +0.048 |
| β, Chi count alone | +0.306 ± 0.055 | +0.287 ± 0.038 | +0.219 ± 0.026 |
| **β, flank GC held fixed** | **−0.103 ± 0.053** | **−0.047 ± 0.037** | **−0.053 ± 0.026** |

The marginal association is real and **it is entirely GC content**: holding
flank GC fixed removes it and flips its sign to something indistinguishable from
zero. And the control that decides the question is not the standard error but
the comparison against **200 other octamers of the same GC content** (6 of 8),
each entered the same way with flank GC held fixed:

> control octamers: mean β **−0.022**, sd **0.201**, range −1.25 to +1.20.
> **Chi's −0.053 sits at the 41st percentile of that null** — the middle of it.

**Chi sites behave exactly like any other GC-rich eight-mer.** A specific
recognition sequence with a specific enzyme and a specific mechanism is
indistinguishable from an arbitrary string of the same composition, which is
what "the effect is composition, not sequence" means when it is tested properly.

*What this does and does not exclude.* It excludes the Chi-attenuation route
specifically, and it is worth having because that route predicted all four
observed properties of the effect and is the most parsimonious mechanism anyone
in this project has proposed. It does **not** exclude repair in general: RecBCD
processivity itself, RecA loading, proximity to a replication fork and the
availability of a sister chromosome are all untouched, and none of them is
testable from sequence alone. The surviving statement is the one the dCas9
contrast already licensed — the effect is downstream of binding — with one
candidate mechanism now removed rather than merely unexamined.

*An orientation hint, reported and not promoted.* With flank GC held fixed, the
four orientation-resolved counts do not behave alike: at ±500 nt, `GCTGGTGG`
downstream is +0.169 ± 0.094 while its reverse complement upstream is
−0.110 ± 0.060, and the pattern repeats in sign at all three widths. That is the
shape a directional mechanism would make. But the octamer null puts sd 0.201 on
a coefficient of this kind, which is larger than the contrast, so it is a hint
at the edge of the noise the control establishes — recorded here so it is not
rediscovered as news, not claimed.

### Why ρ, and why R² beside it

**Report ρ and R² together.** A LightGBM with a robust loss scored R² 0.192 with
ρ 0.568 — the same model looking broken on one metric and respectable on the
other. ρ is the primary metric here because it is crisprHAL 2's own reporting
convention and because ranking is the decision the model is used for; what ρ
means on that decision scale is in Finding 7, since it governs how every headline
in Part 1 should be read rather than only these tables.

## Appendix B2 — Why the imported families failed, and why that is not a screen

This began as a numbered finding: a rule that a feature predictable from the
20-mer cannot help, with a table of eleven families as evidence. **The table does
not support a rule**, and it is demoted here rather than deleted because the
measurements are real and the failure mode is instructive.

Out-of-fold R² of predicting each new column from position-specific mono- and
dinucleotide indicators of the protospacer, over all eleven families:

| family | columns | median R² | **share of columns with R² > 0.9** | Δρ |
|---|---:|---:|---:|---:|
| `b_energy` (CRISPRoff ΔG) | 10 | 0.984 | **70%** | +0.007 |
| `d_mechanics` (duplex stability) | 30 | 0.845 | **50%** | +0.020 |
| `c_folding` | 9 | 0.297 | 0% | +0.007 |
| `f_methylation` | 17 | 0.141 | 0% | +0.004 |
| `a_flank` | 322 | −0.012 | 0% | **+0.080** |
| `d_supercoiling` | 26 | −0.013 | 0% | +0.045 → artefact |
| `g_transcription` | 18 | −0.016 | 0% | +0.015 |
| `e_nucleoid` | 20 | −0.016 | 0% | +0.016 |
| `i_shape` | 152 | −0.017 | 0% | +0.036† |
| `j_periodicity` | 24 | −0.021 | 0% | −0.001 |
| `h_offtarget` | 7 | −0.022 | 0% | +0.000 |

† `i_shape` is the standalone increment over the base, like every other row.
Earlier versions of this table printed +0.001, which is its increment *on top of
family A* — a different quantity, and the one place in the report where a
marginal-on-A figure sat in a column of base-relative gains (Appendix B3).

**Read the last-but-one column, not the median.** The structure is binary: two
families contain columns the 20-mer *determines*, and nine contain none.
`c_folding` and `f_methylation` have middling medians but **no** determined
column — their medians average weak partial fits and detect nothing.
Nearest-neighbour thermodynamics is a sum over adjacent pairs, so binding energy
is a second-order quantity the matrix already holds: importing CRISPRoff was
importing the matrix back into itself.

**Why the nine null families agree so closely.** Out-of-fold R² against a target
with no relationship to the basis is not 0 but a small negative set by the basis
size and the row count, not by the target. All nine land between −0.012 and
−0.022 with 10th percentiles of −0.019 to −0.022: they are sitting on a shared
null floor, which is what "no relationship" looks like rather than a defect.

*Status of this rule.* **A qualitative screen, and it is not quoted with a
score.** Of the two families whose columns are determined, neither produced a
material gain, and that is the direction worth using: if a candidate's columns
are individually reconstructible from the 20-mer, do not build it. The converse
has no force — `h_offtarget` and `j_periodicity` pass the screen and deliver
nothing, and `i_shape` passes it and delivers +0.036 standalone, falling to
+0.001 only once family A is present.

*Why this is stated as a mechanism and never as a tally.* **No count of
families is supportable here.** Only four families have positive R² at all, no
threshold for "high" can be declared after the fact, and `d_mechanics` —
unambiguously high at 0.845 — returned +0.020, the second-largest gain in the
table. A score like "8 of 8" would be doing rhetorical work this design cannot
carry.


### Why this is not a general screen

Three things, and the first is decisive.

**The rule applies to two families.** Only `b_energy` (R² 0.984) and
`d_mechanics` (0.845) are substantially determined by the 20-mer. Everything else
sits below 0.30. A rule with n = 2 is an observation.

**For the strongest case it is a tautology.** CRISPRoff ΔG is *computed from the
sequence* by nearest-neighbour summation. That it adds no information to a model
holding the sequence losslessly is arithmetic, not an empirical discovery; the R²
of 0.984 confirms the arithmetic. The same applies to `d_mechanics`. So the two
supporting cases are the two cases where no measurement was needed.

**The other nine carry no signal about the rule.** Their R² spans **0.010**
(−0.022 to −0.012) — null-floor variation set by family size, not by
predictability — while their gains span **0.081** (−0.001 to +0.080). Across all
eleven families ρ(R², Δρ) = **0.224, p = 0.508** — it was quoted as 0.489,
p = 0.127 while `i_shape` sat at +0.001, and correcting that entry weakens the
association further. Arranging nulls in a table
sorted by a statistic with no resolving power produces the *appearance* of a
trend from a pile of things that did not work, for unrelated reasons:
`h_offtarget` needs the rest of the genome, `j_periodicity` is a genuine null,
`i_shape` duplicates family A, `d_supercoiling` was an artefact its own control
overturned.

**What survives** is the statement in Finding 1: a deterministic function of the
20-mer cannot add information to a model that already encodes the 20-mer. That is
worth saying because people keep importing such features. It is not a predictive
screen, and this report no longer quotes it as one.

## Appendix B2a — Curation and feature selection are not the same kind of dropping

Both discard data, and it is easy to treat them as interchangeable. They are not.

**Curation drops rows; selection drops columns.** crisprHAL's curation removes
guides whose *label* is unreliable — it is a read-count filter — which reduces
noise in the target. Feature selection removes *inputs* that do not predict,
which changes what the model sees. Neither touches the other's object.

**Only one of them can flatter a score without improving a prediction.**
Selection is fitted **inside each training fold**, so it cannot see the test rows
and cannot leak. Curation is applied to the whole dataset **before splitting**,
so it also removes noisy rows from the *test* set — raising ρ partly by making
the evaluation easier rather than the model better. Our own figure for the value
of curation, ρ 0.607 → 0.628 on restricting to the 8,617 retained rows, contains
both effects and **we have never separated them**.

**What curation is worth here, measured.** On the 8,617 guides both cohorts
share, swapping Noshay's label for crisprHAL's changes almost nothing — baseline
0.5387 → 0.5370, with family A 0.6279 → 0.6305, a difference of +0.003 inside the
noise. The value is in *which rows survive*, not in how the survivors are scored.

**The two cohorts are not nested**: 8,617 shared, 24,950 only in crisprHAL,
**5,263 only in Noshay**. So curation is not a filter applied to Noshay's
selection — it is a different, larger draw from the same Guo screen with a
quality threshold. The guides crisprHAL drops look unremarkable on copy number
and uniqueness; their only visible signature is a compressed label range.

**What the procedure actually is** (Browne et al., 2026, read 15 October — worth
setting out in full, because we train on their label):

* **The score is not a read-count ratio.** It is the ALDEx2 `diff.btw` log2
  fold-change between the experimental (Cas9) and control (dead-nuclease) arms —
  *"the expected value of a Bayesian posterior derived from the data"*, over two
  control and two experimental replicates. That matters for the ceiling argument
  below: a posterior mean is shrunk toward the prior, so its reliability is a
  property of the prior as well as of the read counts, and the Spearman–Brown
  machinery we applied to Noshay's two-replicate geometric mean does not carry
  over unchanged.
* **The filter is an average control-condition read count across the two control
  replicates**, and the threshold is **73** for the *E. coli* SpCas9 set we train
  on (31 for eSpCas9, 56 for *C. rodentium* TevSpCas9). So "a read-count filter"
  was the right inference.
* **The threshold is itself a fitted quantity** — chosen by 5-fold
  cross-validation on the training set. Curation is therefore not a fixed
  data-cleaning rule but a hyperparameter optimised for model performance, which
  is worth knowing before describing their cohort as independently cleaner.
* **And it is applied before the split.** Their Table 2 reports the cutoffs
  against the unfiltered pools, before the 80/20 partition — their own text flags
  the tension with the hold-out being untouched. This confirms the concern above
  from their methods rather than from our inference: curation cleans the test set
  too, and our 0.607 → 0.628 figure still contains both effects.

### Why the columns behave *qualitatively* differently after curation

The objection is the right one: if curation only removed noise, every family's
gain should have been scaled by a single attenuation factor and nothing should
have changed in kind. `i_shape` should not have looked like a null and then like
a third-largest effect; `d_mechanics` should not have gone from +0.020 to +0.099.
Noise removal does not reorganise a table.

**It did not only remove noise. Four things changed at once, and three of them
are not about noise at all.**

| what changed | is it noise removal? |
|---|---|
| the **label** — crisprHAL's read-count-filtered score replaces `cut_score` | yes |
| **which rows** — not nested: 8,617 shared, 24,950 only crisprHAL, 5,263 only Noshay; genome-wide (4,013 genes at 7.5 guides) against a panel (1,222 at 11.3) | no — a different draw |
| **how many rows** — 13,880 → 33,567, 2.4× | no |
| the **base representation** — the curated base is the 6,169 regenerated columns and carries **none** of the published base's 63 hand-named summaries | no — a different comparison |

**The part that is noise removal is nearly inert, and this is measured.** At
fixed rows — the 8,617 both cohorts share, published base, same folds — swapping
the label moves family A from **+0.0892** against `cut_score` to **+0.0936**
against the curated score. A **0.004** move, against a cohort move of +0.080 →
+0.164. Whatever tripled the table, it was not the label, and the attenuation
argument is bounded at a few per cent of the effect rather than 230% of it.

**Still, noise removal *can* change behaviour in kind, and it is worth being
precise about why**, because the intuition that it cannot is the thing that makes
this surprising. Two reasons:

*Feature selection and tree splitting are not continuous in signal-to-noise.* A
family is only worth anything if its columns are selected into the top 300 and
then split on. Both are threshold operations. A weak-but-real column sitting just
below the selector's cut contributes **exactly zero**, not a small amount; drop
the label noise slightly and the same column crosses the threshold and
contributes its full value. So a small change in reliability can produce a step
change in measured Δρ. This is the mechanism by which "just less noise" shows up
as a qualitative difference, and it is specific to a pipeline with a hard
selection budget — which ours has, at 300.

*Label noise is not uniform across rows, so removing noisy rows is not a
null operation on the covariates.* A read-count filter removes
low-abundance guides preferentially, and abundance is **not** spatially random —
we measured that directly when `d_supercoiling`'s gain turned out to be DNA read
density rather than supercoiling. A filter on read count is therefore a filter
that touches the joint distribution of every positional feature and the label,
not just the label's variance. That is a change of population, and it is the one
the table is most likely responding to.

### The objection answered with a mechanism, not an exclusion

Everything above rules things out. It does not yet say *why* the increments rose
while the baseline rose too, and that is the part of the objection that has
force: attenuation predicts the two move in opposite directions.

The answer is that **the added guides carry a genuinely stronger relationship
between flanking composition and the label**, and it can be measured directly
rather than inferred from model gains. For a relationship y = βx + noise, the
correlation is β · sd(x) / sd(y), so exactly three things can raise it: a steeper
β, a wider spread of the predictor, or a narrower spread of the label. All three
are observable. Taking x = flank GC at ±500 nt and y = **the curated label on
every row**, so the units are identical across groups
(`curation_variance.py`, `results/curation_variance.csv`):

| rows | n | sd(x) | sd(y) | **β** | Pearson r |
|---|---:|---:|---:|---:|---:|
| the 8,617 Noshay also published | 8,617 | 0.0431 | 2.58 | **+5.35** | +0.089 |
| crisprHAL only, **inside Noshay's gene panel** | 2,045 | 0.0475 | 3.92 | **+8.63** | +0.105 |
| crisprHAL only, **outside it** | 22,905 | 0.0524 | 3.38 | **+17.03** | +0.264 |
| the whole curated cohort | 33,567 | 0.0503 | 3.29 | +14.76 | +0.225 |

**It is not spread.** The predictor's spread rises only 1.21× between the first
and fourth rows, and the *label's* spread rises 1.34× — which pushes the
correlation **down**. Both together cannot produce a 2.7× rise in r. What does is
β, which rises **2.96×**: a unit of flank GC is worth three times as much score
in the guides crisprHAL added. The arithmetic closes: 2.96 × 1.21 / 1.34 = 2.67,
against an observed r ratio of 2.67.

**And it is the genes, not the guides — mostly.** Splitting the added rows by
whether their host gene is one of the 1,222 in Noshay's panel separates two
effects that were confounded. Inside the panel's genes the relationship stays
weak (β 8.63, r 0.105) even for guides Noshay never published; outside it the
relationship is twice as steep again (β 17.03, r 0.264). So the 2.96× factors
into **×1.6 from which guides were published inside those genes** and **×2.0 from
the choice of genes itself** (1.6 × 2.0 = 3.2, against 2.96 observed).

**What that means for every number in the family table.** The published matrix
did not measure a noisier version of the same effect — it measured a real effect
in a **part of the genome where that effect is weak**, because Guo's screen was
built around a 1,222-gene panel and Noshay published a subset of it. The +0.080
is the correct figure for that panel and the +0.164 is the correct figure
genome-wide; neither is an attenuated version of the other, and the right one to
quote depends on where a user intends to design guides. Why panel genes behave
differently is **open**: the obvious candidate is essentiality — if cutting kills
the cell whatever the efficiency, variation in efficiency has less room to show
in the score — but that is a hypothesis, not a result, and family G's
essentiality column is worth +0.00004 on its own, which argues against the
simplest version of it.

This also explains the uniformity across families without appealing to the
measurement scale: every family that reads genomic position or context inherits
the same β, so a change in β multiplies all of them together and preserves the
ordering, which is what the rank correlation of 0.97 shows.

**A second, smaller contributor remains the base representation.** The
inflation is nearly uniform — rank correlation 0.97 between the published and
curated columns, median 3.3× — and a uniform multiplier is also the signature of
something that changed the *scale of the measurement*. The base
losing its 63 hand-named dense columns does exactly that: every family that
supplies dense continuous summaries is now measured against a weaker comparison,
and the weaker the base, the more any such family is worth. It predicts
multiplication with the ordering preserved, which is what is observed.

**All four were tested, and the answer is the rows.** Each row below holds
everything else fixed and changes one thing:

| candidate | test | effect on family A's Δρ |
|---|---|---:|
| the **label** | same 8,617 rows, published base, swap `cut_score` for the curated score | **+0.004** (+0.0892 → +0.0936) |
| **row count** | random 13,880-row subsamples of the curated cohort, 3 draws | **−0.003** (+0.1672 ± 0.0027 against +0.1638 at 33,567) |
| the **base** | add 41 dense guide-only summary columns to the curated base, then measure family A on top | **+0.013** (+0.1638 → +0.1504) |
| **which rows** | the remainder of a +0.084 gap | **≈ +0.070** |

Row count is not merely small, it is **absent**: at the published cohort's own
row count, the curated cohort still gives family A **+0.167**. The base matters a
little — giving the base `b_energy`, `c_folding` and `d_mechanics`'s 22 dense
guide-only columns raises it from 0.5445 to 0.5692 and takes 0.013 off family
A — which is 8% of the gap, not the explanation. The label is 5%.

**The 24,950 guides crisprHAL has and Noshay never published are a different
population, and the flank effect is 2.5× stronger in them.** The cleanest form of
this needs no model at all — the marginal rank correlation between a single
flank-GC column and activity:

| rows | label | `up500.gc` | `up50.gc` | `dn500.gc` |
|---|---|---:|---:|---:|
| the 8,617 **shared** | Noshay's `cut_score` | +0.069 | +0.056 | +0.052 |
| the 8,617 **shared** | crisprHAL's score | +0.067 | +0.060 | +0.054 |
| the 24,950 **crisprHAL-only** | crisprHAL's score | **+0.172** | **+0.150** | **+0.180** |

**Same rows, two labels: identical. Different rows, same label: 2.5× to 3.5×.**
That is the whole answer to the original objection, in a statistic with no
selector, no tree and no folds in it.

*What is different about those rows, as far as we can see.* They are not an
obviously different class of gene — 6.9% essential against 8.0%, 6.8 guides per
gene against 7.4. What differs is **dynamic range**: within the same curated
label, the added rows have **1.34×** the label spread (sd 3.46 against 2.58) and
**1.21×** the flank-GC spread (0.052 against 0.043). A correlation is
scale-invariant, so spread alone changes nothing — but spread in the *signal*
with the noise floor unchanged raises reliability, and attenuation is
multiplicative in exactly the way the family table is. Noshay published the
subset of the screen with the least room to measure anything in.

*What would separate "more dynamic range" from "different biology" properly:*
the per-guide read counts, which give a noise variance per row and hence a
reliability per subset. We do not have them, and they are in the same PeerJ
supplement nobody has read. Until then, "the added rows have more dynamic range,
and that is sufficient to explain a 2–2.5× attenuation difference" is as far as
this goes.

**So the premise of the objection was right.** If curation had only removed noise
from the label, the columns should not have changed behaviour — and the label
change is worth 0.004. What changed behaviour is that crisprHAL's release is a
**different and wider draw from the Guo screen**, four fifths of which Noshay
never published, and in which the effect this project is about is simply easier
to see.

*Consequence for the findings, and it is a favourable one.* The published matrix
**understated** the flanking-context effect, because it is the narrower,
lower-range subset. The curated figures are the ones that describe genome-wide
guide design; the published figures describe a 1,222-gene panel. Nothing in
Part 1 needs weakening, and Finding 2's effect size needs *raising* when it is
restated on this cohort — but the restatement has to carry this appendix with it,
because a reader comparing +0.080 to +0.164 across cohorts without it will think
the model got better.

## Appendix B3 — The family ablations on the curated cohort, and what they measure

Every Δρ in Appendix C was measured on the published 13,880-row matrix. The model
layer moved to crisprHAL's 33,567 curated guides and the family measurements did
not follow, so **all eleven families were re-run there**: same design, base
representation against base + one family, 5 seeds × 5 folds, selection inside
each training split, `python -m sgrna.curated_ablation --all`.

| family | columns added | published Δρ | **curated Δρ** | ratio |
|---|---:|---:|---:|---:|
| `a_flank` | 348 | +0.080 | **+0.164** | 2.1× |
| `d_supercoiling` | 26 | +0.045 | **+0.139** | 3.1× |
| `i_shape` | 152 | +0.036 | **+0.120** | 3.4× |
| `d_mechanics` | 30 | +0.020 | **+0.099** | 4.9× |
| `e_nucleoid` | 20 | +0.016 | **+0.072** | 4.6× |
| `g_transcription` | 19 | +0.015 | **+0.058** | 4.0× |
| `b_energy` | 10 | +0.007 | +0.019 | 2.5× |
| `c_folding` | 9 | +0.007 | +0.014 | 1.9× |
| `j_periodicity` | 24 | −0.001 | +0.011 | — |
| `f_methylation` | 17 | +0.003 | +0.008 | 2.3× |
| `h_offtarget` | 13 | +0.000 | +0.001 | 3.8× |

Baseline ρ 0.5445 ± 0.0010 on the curated cohort. The published column takes each
family from its own run (`ablation_main`, `ablation_grouped` for the three
positional families, `ablation_shape`, `ablation_jper`), so its baselines differ
slightly between rows — 0.5272 random-row, 0.5283 100 kb-grouped.

**The ranking transfers; the scale does not.** Rank correlation between the two
columns is **ρ = 0.97**, and every family gains, with a median inflation of
**3.3×**. That is the opposite of what the migration was expected to show. The
cohort change does not reorder which families matter — it multiplies all of them
by roughly the same factor.

**Why a multiplier, and why upward**, is answered in Appendix B2a and is not
about noise: the slope of the flank-composition effect is **2.96× steeper** in
the guides crisprHAL adds, and the weak slope belongs to Guo's 1,222-gene panel
rather than to the guides Noshay selected from it. Every family that reads
genomic context inherits that same slope, which is why one factor moves the whole
column and leaves the ordering intact.

*Two different `i_shape` numbers exist and only one belongs in this column.*
**+0.001** is `i_shape`'s increment *on top of family A* (`a_flank+i_shape`
0.6078 against `a_flank` 0.6068); every row here is an increment over the base,
where `i_shape` is **+0.036** on the published matrix, third-largest then as it
is now. The family is a near-duplicate of family A, which is why its marginal
value alongside it is ~0 — a real and reported result, but not a null, and not a
base-relative gain. Quoting the +0.001 in this column would read as a reversal
from a flat null, and there is no reversal.

**It does not measure the selector.** The curated base is 6,169 sparse indicator
columns with none of the 63 hand-named summaries the published base carried, and
300 are selected. The obvious hypothesis was starvation. Tested directly on
`d_mechanics`:

| arm | ρ |
|---|---:|
| base, select 300 | 0.5443 |
| base + `d_mechanics`, select 300 | 0.6432 |
| base, **all 6,169 columns** | 0.5439 |
| base + `d_mechanics`, **all 6,199 columns** | 0.6432 |

**+0.0988 with selection, +0.0993 without.** The gain is not an artefact of
selecting 300 columns.

### What it does measure

The inflation tracks **what kind of column a family supplies**. Counting columns
that take more than ten distinct values on the cohort as dense and continuous
(`--shape-audit`, `results/ablation_curated_shape.csv`):

| family | columns | dense continuous | curated Δρ |
|---|---:|---:|---:|
| `a_flank` | 348 | 155 | +0.164 |
| `d_supercoiling` | 26 | 26 | +0.139 |
| `i_shape` | 152 | 152 | +0.120 |
| `d_mechanics` | 30 | 29 | +0.099 |
| `e_nucleoid` | 20 | 20 | +0.072 |
| `g_transcription` | 19 | 12 | +0.058 |
| `b_energy` | 10 | 9 | +0.019 |
| `c_folding` | 9 | 7 | +0.014 |
| `j_periodicity` | 24 | 24 | +0.011 |
| `f_methylation` | 17 | 2 | +0.008 |
| `h_offtarget` | 13 | 1 | +0.001 |

**ρ(dense continuous columns, Δρ) = 0.882, p = 0.0003, n = 11.** The published
base already contained dense summaries among its 63 hand-named columns, including
guide Tm; the curated base contains none, so families that restore them are paid
for it.

**Supply is not sufficient, which is the useful part.** `j_periodicity` supplies
24 dense continuous columns — as many as `d_supercoiling`'s 26 — and returns
+0.011 against +0.139. Column shape buys a family the *chance* to help; what it
measures decides whether it does.

**And supply is not the whole account.** Dense continuous columns and reach
beyond the 20-mer are themselves correlated — a family that measures something
about 2 kb of DNA naturally reports it as real numbers — so ρ = 0.882 is not
evidence that shape is the operative variable. The section below separates
them, and reach wins.

### What it measures, second answer: how far outside the 20-mer a family can see

The shape account above is a correlation, and it is confounded. Sorting the same
eleven families by **what DNA or data they are allowed to read** separates them
better than any property of their columns:

| reach | families | curated Δρ |
|---|---|---|
| the 20-mer and its PAM only | `b_energy`, `c_folding`, `d_mechanics` (22 of 30 cols) | **+0.011 to +0.019** |
| a short flank as well | `f_methylation` (±20 nt at the PAM) | +0.008 |
| the flanking DNA | `a_flank` (±1,000 nt), `i_shape`, `d_mechanics` (8 of 30) | **+0.097 to +0.164** |
| the rest of the genome, or external measurement | `d_supercoiling`, `e_nucleoid`, `g_transcription` | +0.058 to +0.139 |
| position on the chromosome | `j_periodicity`, `h_offtarget` | +0.001 to +0.011 |

**Nothing confined to the 20-mer exceeds +0.019, and nothing above +0.05 is
confined to it.** Access to information outside the guide is necessary for a
large gain here and not sufficient — `j_periodicity` and `h_offtarget` read the
whole chromosome and return nothing. That is a sharper statement than the column
shape correlation, it is the same conclusion Finding 2 reaches from the other
direction, and it survives the correction that produced it.

*What this appendix said until 13 October.* That `d_mechanics` adds **no new DNA
at all** and is therefore "the cleanest available demonstration that
representation matters independently of information" at +0.099. Eight of its
thirty columns are flanking-DNA thermodynamics, and those eight carry +0.097 of
the +0.099 (Finding 1). The representation effect is real at **+0.011 to +0.019**,
on **two** independent guide-only measurements rather than the three this
appendix claimed: `b_energy`'s and `c_folding`'s shares of it are the same
spacer-fold column entered twice (Appendix B, *Every family split the way
`d_mechanics` was*). The original claim was inflated nine-fold by a family whose
column list nobody had read.

### Under gene-grouped cross-validation: the gains are not locus memorisation

Every number above is from random KFold, and on this cohort that needed
checking rather than assuming — gene identity explains **29%** of the curated
label against 16% of the published one, so there is nearly twice as much
available to memorise (Appendix A). All eleven families were therefore re-run
with **whole genes held out**: `curated_ablation --grouping gene`, the 30,203
guides that fall inside an annotated gene, 4,013 gene groups, 5 seeds × 5 folds,
baseline ρ 0.5584.

| family | reach | random-row Δρ | **gene-grouped Δρ** | retained |
|---|---|---:|---:|---:|
| `a_flank` | flanks | +0.164 | **+0.157** | 96% |
| `d_supercoiling` | external | +0.139 | **+0.131** | 94% |
| `i_shape` | flanks | +0.120 | **+0.112** | 94% |
| `d_mechanics` | guide + flanks | +0.099 | **+0.092** | 92% |
| `e_nucleoid` | external | +0.072 | **+0.063** | 88% |
| `g_transcription` | external | +0.058 | **+0.046** | 79% |
| `b_energy` | guide | +0.019 | +0.017 | 91% |
| `c_folding` | guide | +0.014 | +0.014 | 97% |
| `j_periodicity` | position | +0.011 | +0.007 | 64% |
| `f_methylation` | guide | +0.008 | +0.007 | 93% |
| `h_offtarget` | genome | +0.001 | +0.000 | 23% |

**The guide-only families calibrate the control, which is what makes the rest of
the column readable.** `b_energy`, `c_folding` and `f_methylation` are functions
of the guide, so grouping *cannot* invalidate them — a sequence feature does not
become less true when the split changes. They nevertheless retain only **94% in
aggregate**, and that 6% is the **grouping tax**: holding out whole genes is a
harder task for reasons that have nothing to do with memorisation (less locus
diversity in training, and 4,013 groups split five ways is a coarser partition
than 30,203 rows split five ways). Without that calibration a family retaining
94% would read as having lost 6% to memorisation, when the right reading is that
it lost nothing.

Against the 94% benchmark:

* **`a_flank` (96%), `d_supercoiling` (94%) and `i_shape` (94%) lose nothing at
  all.** The three largest effects in the table survive gene-grouping intact.
* **`e_nucleoid` (88%) and `g_transcription` (79%)** carry a real locus
  component, 6 and 15 percentage points beyond the tax. Both read external
  tracks indexed by position, so this is the expected direction and the expected
  families.
* **`j_periodicity` (64%) and `h_offtarget` (23%)** lose most of their gain, but
  their gains are +0.011 and +0.001. Large relative losses of nothing.

**This is a smaller memorisation component than the published cohort showed**,
where grouping cost `e_nucleoid` 42% of its gain and `g_transcription` 36%. The
apparent paradox — twice the between-gene variance, half the memorisation — is
resolved by what else changed: the curated cohort has **7.5 guides per gene
against 11.3**. Between-gene variance sets how much a locus-memorising model
*could* win; guides per gene sets how well it can *learn* each locus's mean. The
two moved in opposite directions, and the measurement says the second dominated.
That is not something that could have been reasoned out from the variance share
alone, which is the argument for running the control rather than arguing about
it.

*Caveat on levels.* The grouped arms run on 30,203 guides, not 33,567, because
3,364 fall outside every annotated gene and were dropped rather than given
singleton groups. Its baseline is accordingly different (0.5584 against 0.5445),
so the Δρ columns compare and the ρ columns do not.

### How to read these numbers, and how not to

**They are not comparable to the published Δρ column as levels, and the reason is
now known.** The two cohorts have different baselines (0.544 against 0.527),
different labels and different base representations — but none of those is what
tripled the table. **The rows are** (Appendix B2a): the label is worth 0.004, the
row count nothing, the weaker base 0.013, and the remaining ~0.070 of family A's
+0.084 gap belongs to the 24,950 guides crisprHAL publishes and Noshay did not,
in which a single flank-GC column's marginal correlation with activity is 2.5×
higher. The *ordering* transfers (ρ = 0.97); the magnitudes describe a different
and wider population.

**They are not clean estimates of each family's biological contribution.** A
family that supplies dense summaries is rewarded for its shape as well as its
content on this base. Separated two ways, and both say the shape is worth little.
On the curated cohort (`base_enrichment.py`), giving the base 41 dense guide-only
summary columns — `b_energy`, `c_folding` and `d_mechanics`'s protospacer half,
none of which carries information beyond the 20-mer — raises it from 0.5445 to
0.5692 and takes **0.013** off family A's gain, 8% of it.

That was a stand-in, because the published matrix's own 63 hand-named summaries
— guide GC, Tm, k-mer counts, PAM indicators — exist on the 13,880-row cohort and
nowhere else. Run there, where the two bases differ by exactly those 63 columns
(`base_enrichment.py --published`, `results/base_enrichment_published.csv`):

| base | columns | ρ | + family A | Δ |
|---|---:|---:|---:|---:|
| sparse: indicators + quantum tables | 6,169 | 0.5227 | 0.6050 | **+0.0823** |
| the same, plus the 63 hand-named | 6,232 | 0.5272 | 0.6080 | **+0.0808** |

**The real summaries take +0.0015 off family A's gain, 1.8% of it** — less than
the stand-in suggested, not more. The second base's 0.5272 reproduces Finding 1's
0.52716 for the full published matrix through a different code path, which is
what certifies the two arms differ only by those columns. So column shape is not
where the published-to-curated tripling comes from, and the row population
accounts for essentially all of it (Appendix B2a).

### The joint arms: the column must not be added up, and here is by how much

Every Δρ above is a single-family increment over a common base, and the warning
against summing them was inherited from the published cohort rather than measured
here. Measured (`curated_joint.py`, same base, seeds and folds):

| arm | joint Δρ | sum of singles | realised |
|---|---:|---:|---:|
| `a_flank` + `i_shape` | +0.1647 | +0.2833 | 58% |
| `a_flank` + `d_supercoiling` | +0.1882 | +0.3029 | 62% |
| `d_supercoiling` + `e_nucleoid` + `g_transcription` | +0.1454 | +0.2692 | 54% |
| `a_flank` + `d_supercoiling` + `i_shape` | +0.1882 | +0.4224 | 45% |
| **`b_energy` + `c_folding` + `f_methylation`** (guide-only control) | +0.0289 | +0.0409 | **70%** |
| **all eleven families** | **+0.2030** | +0.7046 | **29%** |

**Summing the column would claim +0.70 ρ. The real figure for all eleven
together is +0.203** — 29% of it, and only +0.039 beyond family A alone. So the
table is a ranking of families, not a budget.

**`i_shape` adds +0.0009 on top of family A, and exactly nothing once
`d_supercoiling` is also present** — the two three-family arms give identical
joint gains to four decimals (+0.1882). Its +0.120 standalone is family A's
effect re-expressed as bendability, which reproduces the published cohort's
+0.001-on-top-of-A result on data where that number was independently measured.

**The control is what licenses reading the rest.** The three guide-only families
overlap only through the 20-mer and are the most additive at **70%**; the flank
and positional families, which read overlapping DNA, land at 45–62%. Had the
control also come in near 50%, the right conclusion would have been a pipeline
artefact rather than an interpretation.

## Appendix C0 — Reproducibility: versions, seeds and protocol

Recorded so a rerun is a rerun, not an approximation.

| | |
|---|---|
| Python | 3.12.15 |
| numpy / pandas / scipy | 2.5.3 / 3.0.6 / 1.18.1 |
| scikit-learn | 1.9.1 |
| LightGBM / XGBoost / CatBoost | 4.7.0 / 3.4.1 / 1.2.10 |
| SHAP | 0.52.0 |
| PyTorch (the CNN controls) | 2.14.1 |

**Seeds.** `config.SEEDS = (41, 42, 43, 44, 45)`; `config.N_SPLITS = 5`. Ablations
use all five seeds × five folds (25 fits per arm). The head-to-head against
crisprHAL 2 uses seed 41 only, because each of their fits costs ~21 min. The
shift-distance null and the GC-strata test use three seeds × five folds. Every
seed is passed explicitly to the model, the selector and the fold splitter, so
no arm depends on global RNG state.

**Selector.** `config.SELECTOR = "xgboost"` — the throwaway model whose only job
is to rank columns; its predictions are never used. Feature selection is fitted
**inside the training split of each fold**, never on the full data.

**Resumability.** Every fold is written to `results/folds_<tag>.csv` as it
finishes and a fold already recorded there is skipped, so a long run can be
rebuilt over several sittings without recomputing work — and so that a crash
loses one fold rather than an experiment.

## Appendix C — Feature sets and data sources

### Ours — eleven families, four distinct ideas

Δρ below is on the **published** 13,880-row matrix, which is what these were
built against. All eleven have since been re-measured on the curated cohort,
where the ordering is nearly identical and every figure is about three times
larger — Appendix B3 has that table and the reason.

| set | what it measures | source | Δρ |
|---|---|---|---:|
| `a_flank` | composition and quantum averages outside the 20-mer, 50–1,000 nt | reference genome | **+0.080** |
| `b_energy` | guide–DNA binding energy by component | CRISPRoff, unchanged | +0.007 |
| `c_folding` | whether the guide RNA folds on itself | ViennaRNA | +0.007 |
| `d_mechanics` | duplex opening, position by position | SantaLucia & Hicks (2004) | +0.020 |
| `d_supercoiling` | positive-supercoiling density | GapR-seq, GSE152880 | +0.045 → artefact |
| `e_nucleoid` | 3D contact structure, NAP dependence | Lioy et al. 2018 Hi-C | +0.016 |
| `f_methylation` | Dam/Dcm motifs at PAM and seed | reference genome | +0.004 |
| `g_transcription` | TSS distance, strand, expression | RegulonDB + PRECISE-1K | +0.015 |
| `h_offtarget` | off-target burden, target copies | reference genome | +0.000 |
| `i_shape` | flank bendability / deformability | published dinucleotide scales | +0.036 (+0.001 on top of `a_flank`) |
| `j_periodicity` | GC skew, helical-repeat periodicity | reference genome | −0.001 |

Groups 1 (`b`,`c`,`d_mechanics`,`f`) are redundant by construction (Appendix B2); group 2 (`a`) is
the one that worked; group 3 (`d_supercoiling`, `e`, `g`) collapse into one
positional variable; group 4 (`h`) needs the rest of the genome.

### Teammates' — on the published 13,880-row matrix, quoted in R²

Not directly comparable with the Δρ column above; recorded so the paper does not
under-credit them.

| set | who | result |
|---|---|---|
| **gene essentiality** | Jacky | **+0.0037 R²** over 0.2927, **selected in 5/5 folds** — the only non-QCT idea on that matrix with a consistent gain, and it is not a function of the 20-mer |
| **orientation controls** | Jacky | closes Part 4 item 3: curated RegulonDB TSS replaces nearest gene as the orientation variable (Δ 0.051 [0.020, 0.082]), and the nearest-gene version we had been quoting is shown to be largely a distance/GC cohort confound (Δ 0.007, P = 0.34 after matching). Block bootstrap over 47 genomic blocks, promoter/RBS separated, operon compared |
| seed non-additive QCT | Joshua | ~+0.005 R²; absolute-difference forms carry more SHAP than signed (1.09 against 0.92) |
| log-ratio QCT forms | Joshua | ~+0.005 R², best seed 0.3015 |
| regional QCT summaries | Joshua | +0.0013 R², against a permuted control at 0.2857 |
| PAM-weighted, profile-shape, coupling, HOMO–LUMO asymmetry | Joshua | −0.0010 to +0.0006 — null |

The non-additive families are **independent evidence for Finding 4**: they gain
from exactly the higher-order structure the k-mer ladder found worth +0.0035 ρ
once the selector was removed.

## Appendix D — Sources (APA 7th edition)

Browne, T. S., Edgell, D. R., & Gloor, G. B. (2026). Better data for better
predictions: Data curation improves deep learning for sgRNA/Cas9 prediction.
*PeerJ*, *14*, e20706. https://doi.org/10.7717/peerj.20706
— the state of the art; code and curated label re-run here as the baseline.

Cui, L., Vigouroux, A., Rousset, F., Varet, H., Khanna, V., & Bikard, D. (2018).
A CRISPRi screen in *E. coli* reveals sequence-specific toxicity of dCas9.
*Nature Communications*, *9*, 1912. https://doi.org/10.1038/s41467-018-04209-5
— the dCas9 CRISPRi screen used as the final untouched test (Finding 3); its two
measurement arms also set the ceiling range quoted in Appendix B.

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

Lioy, V. S., Cournac, A., Marbouty, M., Duigou, S., Mozziconacci, J., Espéli, O.,
Boccard, F., & Koszul, R. (2018). Multiscale structuring of the *E. coli*
chromosome by nucleoid-associated and condensin proteins. *Cell*, *172*(4),
771–783.e18. https://doi.org/10.1016/j.cell.2017.12.027
— Hi-C contact structure and nucleoid-associated protein dependence.

Noshay, J. M., Walker, T., Alexander, W. G., Klingeman, D. M., Romero, J.,
Walker, A. M., Prates, E., Eckert, C., Irle, S., Kainer, D., & Jacobson, D. A.
(2023). Quantum biological insights into CRISPR-Cas9 sgRNA efficiency from
explainable-AI driven feature engineering. *Nucleic Acids Research*, *51*(19),
10147–10161. https://doi.org/10.1093/nar/gkad736
— the feature matrix decoded here; their iRF re-runs in our harness at Pearson
0.4881 against their reported 0.5019, on a third of their rows.

SantaLucia, J., & Hicks, D. (2004). The thermodynamics of DNA structural motifs.
*Annual Review of Biophysics and Biomolecular Structure*, *33*(1), 415–440.
https://doi.org/10.1146/annurev.biophys.32.110601.141800
— nearest-neighbour thermodynamic parameters.

Sastry, A. V., Gao, Y., Szubin, R., Hefner, Y., Xu, S., Kim, D., Choudhary, K. S.,
Yang, L., King, Z. A., & Palsson, B. O. (2019). The *Escherichia coli*
transcriptome mostly consists of independently regulated modules. *Nature
Communications*, *10*(1), 5536. https://doi.org/10.1038/s41467-019-13483-w
— the iModulon decomposition behind the PRECISE expression compendium.

**Two database citations still need their canonical form**, because Crossref
does not resolve them from the version strings we use and they should not be
guessed: **RegulonDB v14.5** (release citation) and **PRECISE-1K** (the 1,035-run
compendium, successor to Sastry et al. 2019). Everything above was verified
against Crossref metadata rather than transcribed from memory.
