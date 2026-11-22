# Paper flow — skeleton only

**This is a flow, not a draft.** Headings, what each section has to establish, which result goes
where, and what the figures are. No prose: the text is yours to write.

Two things to settle before writing, because they change the shape:

1. **Which paper is this?** See "The choice of frame" below. The flow assumes Frame A.
2. **The Google Doc.** I don't have access to the one you shared (the report records it as frozen
   on 13 October and no longer synced). If it already has an outline, reconcile against this rather
   than replacing it — and tell me where they differ.

---

## The field's conventions, from the exemplars

| | crisprHAL 2 (Browne 2026, *PeerJ*) | Yu et al. (2024, *Genome Biology*) | Orro et al. (2026, *Sci Rep*) |
|---|---|---|---|
| order | Intro → Methods → Results → Discussion → Conclusion | Background → Results → Discussion → Conclusions → Methods | Intro → Methods → Results → Discussion |
| Results headings | **claim-shaped sentences** | claim-shaped | topic labels |
| length | ~6,000 words, 5 figures, 3 tables | similar | shorter |
| what leads | the model and its performance | the *surprise* (gene features dominate) | the task |

**Convention to adopt:** claim-shaped Results headings. Our report already states findings as
claims, so each finding maps to one heading. **Convention to reject:** leading with the model.
Browne et al. can, because the model *is* their contribution. Ours is not (Part 6).

---

## The choice of frame

| | frame | leads with | risk |
|---|---|---|---|
| **A** | **Determinants** — what decides whether a guide cuts, and how much is knowable | the flanking-DNA effect and what it is made of | reviewers ask for the predictor anyway |
| B | Method/measurement — the field's headroom is mis-measured | the ceiling correction | thin as a standalone; better as a section |
| C | Model — a better bacterial predictor | the +0.023–0.029 lead | weakest. The lead is over a *shipped* model, the architecture claim is not novel (BoostMEC), and the search asymmetry is on our side |

**Recommend A, with B as a full Results section.** A carries the two largest novel results (the
flank effect's characterisation; the cross-kingdom inversion) and B reframes every number in the
field. C becomes one short section, honestly scoped.

---

## Flow

### Title + abstract
- One sentence each: the effect, its shape, the kingdom inversion, the corrected ceiling.
- Do **not** put the crisprHAL 2 margin in the abstract. It invites the frame we are rejecting.

### Introduction (~700 words)
- Establish: bacterial Cas9 as antimicrobial/engineering tool; activity varies by target.
- Establish: predictors exist, mostly mammalian-trained, and **transfer badly to bacteria** — cite
  that the bacterial literature says this repeatedly *without explaining it*. This is the gap we close.
- Name what is not known: what the determinants physically are, and how much of the variance is
  reachable at all.
- Close on the three things the paper establishes. No results numbers here.

### Results 1 — "The published representation is a re-encoding of the guide" (Finding 1)
- Why first: it sets up why new *information* has to come from outside the 20-mer. The rest of the
  paper is the consequence.
- Short. 99% regenerate; the 412-column equivalent; the arithmetic consequence.
- **Fig 1**: decoding schematic + the 412-vs-6,232 equivalence.

### Results 2 — "DNA outside the protospacer predicts cutting" (Findings 2–3)
- The increment (+0.164 CV, +0.156 sealed genes), gene-grouped.
- Locality: the texture-preserving null and the autocorrelation length.
- Not mappability, not gene ranking, survives holding out whole genes, within-gene +0.061.
- Cut vs bind: 2–3× larger with a break (two dCas9 screens, strand controlled).
- **Fig 2**: Δρ vs shift distance (the autocorrelation measurement) + the cut/bind comparison.

### Results 3 — "The effect is GC over a few hundred bases, and it is coupled to the seed" (Finding 10)
- **This is the paper's centre of gravity under Frame A.**
- GC not purine (the partial-correlation table).
- Side-symmetric → not directional → separate from the orientation effect.
- One length scale, 250–500 nt; the resolution curve saturating and turning over.
- The flank × seed interaction (depth-2 pairs).
- **Fig 3**: four panels — distance profile by side; resolution curve; purine-vs-GC null; interaction heatmap.

### Results 4 — "Guide design rules do not transfer between kingdoms" (Finding 10, second half)
- Position × base matrix, *E. coli*.
- Same estimator on the human matrix; ρ = −0.052; 15 of 27 cells invert; position-20 G.
- Tie back to the Introduction's gap: this is *why* mammalian predictors fail here.
- **Fig 4**: the two position × base heatmaps side by side.
- State the sign-convention verification in the legend or methods. It is the obvious attack.

### Results 5 — "The attainable ceiling is 0.946, and the field's ~0.7 is mis-derived" (Finding 6)
- Our replicate-derived ceiling, from crisprHAL's released per-replicate counts.
- The correction: attenuation √, then Spearman–Brown; the review's own numbers give 0.91–0.93.
- Consequence: models at ρ ≈ 0.70 are at 75–77% of attainable, not at the top.
- **Table 1**: *r* → √*r* → *k*-replicate reliability → ceiling, with our row, and the parametric grid
  (`ceiling_parametric.csv`) so the claim does not rest on recovering anyone's replicate count.
- **Fig 5** carries the same thing as a picture; use one or the other, not both.
- Caveat in text: exact for Pearson, approximate for Spearman.

### Results 6 — "A compact model reaches this with far fewer labels" (Finding 7, scoped)
- Short and honest. Parity for the frozen model; +0.023–0.029 once tuned; ~3× fewer labels; 13–21× faster,
  34× on the design path — and say why that ratio is fair (95% of crisprHAL 2's cost is its forward pass,
  so the same precompute could take at most 4.7% off it).
- **Say in the text**: measured against crisprHAL 2 *as shipped*, not retuned on its own splits; the
  architecture claim is not novel (cite BoostMEC).
- Top-end metrics tied, and four objectives aimed at the top end all failed.
- **Table 2**: metric panel on both of their splits + the data-efficiency curve.

### Discussion (~900 words)
- What the determinants imply mechanically: bulk GC property over a few hundred bases, no fine
  structure, no direction, coupled to the seed. Name the persistence-length reading **as a hypothesis**.
- What is excluded: shape/3D, periodicity, Chi, supercoiling (→ read density → gene dosage).
- Why the kingdom inversion matters for anyone using a mammalian tool in bacteria.
- The ~0.21 ρ that is real, repeatable and unexplained — and that closing it needs an experiment in
  which guide and locus are not confounded, which no pooled screen can provide.
- Limitations: one organism plus two relatives; survival not cutting; the curated label is a fitted
  procedure; no untouched external set remains for the cutting claims; adaptive component +0.0087.

### Methods
- Follow crisprHAL 2's order: Datasets → Data processing → Features → Model → Metrics → Splits.
- **Splits section must carry**: gene-grouped and 100 kb-block CV; the train/test Hamming statistic
  (min 3, no identical guides) — both comparators report this and reviewers will look for it.
- Reproducibility: `run_audit.sh`, 940 checks; `import_check.py` for the code itself.

---

## Figures and tables

**The figures exist** (`results/post_freeze/figures/`, built by `sgrna/post_freeze/paper/figures.py`,
vector PDF + 600 dpi PNG each, geometry and completeness checked by the audit). One table, so the
plan and the built set cannot disagree:

| # | file | size (in) | content | source | where it goes |
|---|---|---|---|---|---|
| Fig 1 | `fig1_representation` | 7.00 × 1.85 | decoding + the 412-column equivalence; a schematic, deliberately not a plot — see below | Finding 1 | Results 1 |
| Fig 2 | `fig2_locality_cut_vs_bind` | 7.00 × 2.35 | (a) Δρ vs shift distance, (b) cut vs bind as a dumbbell by screen | Findings 2–3 | Results 2 |
| Fig 3 | `fig3_what_the_signal_is` | 7.00 × 4.60 | (a) GC distance profile by side, (b) the purine null on the same scale, (c) the resolution curve, (d) interaction pairs | Finding 10 | Results 3 |
| Fig 4 | `fig4_design_rules_by_kingdom` | 4.73 × 3.50 | position × base heatmaps, *E. coli* vs human | Finding 10 | Results 4 |
| Fig 5 | `fig5_attainable_ceiling` | 3.33 × 2.60 | the corrected ceiling, single column | Finding 6 | Results 5 |
| Fig 6 | `fig6_flank_windows_schematic` | 7.00 × 2.15 | what family A actually reads | Findings 2, 10 | Methods or Results 2 |
| Fig 7 | `fig7_pipeline` | 7.00 × 2.50 | how a ρ in this paper is produced, with the two leakage controls | Part 4 | Methods |
| Table 1 | — | — | the ceiling correction, now parametric in *r* and *k* (`ceiling_parametric.csv`) | Finding 6 | Results 5 |
| Table 2 | — | — | metric panel on both splits + data efficiency + the design-path cost | Finding 7 | Results 6 |
| Supp | — | — | family ablations, adaptive-overfitting draws, nuclease and shape nulls (Findings 8–9), the local-sequence bound | Appendices | Supplementary |

ACS applied: ≤ 3.33 in single column and ≤ 7 in double, ≤ 9.167 in depth, Helvetica at ≥ 7 pt
(the floor is 4.5 pt but ACS pages disagree, so everything clears every variant), no rule thinner
than 0.5 pt, vector PDF so the raster minima are satisfied by construction. Colour is secondary
throughout — marker shape and line style carry identity too, so each figure survives greyscale.

**Two judgement calls worth knowing about before you write captions:**

* **Fig 1 is a schematic, not a chart.** The 412-vs-6,232 equivalence spans two protocols and no
  single result file holds both arms. A first attempt assembled a two-point chart from
  incommensurable sources and produced a spurious −0.019 ρ. The numbers are stated as annotation
  instead.
* **Fig 2b is a dumbbell, not three bars of Δρ.** Plotting Δρ side by side reads "binding beats
  cutting" (+0.102 and +0.114 against +0.085), which is the opposite of the claim. The screens
  start from very different within-gene baselines — 0.62 for cutting, 0.17 and 0.08 for binding —
  so the figure shows the baselines and lets the reader see why the raw numbers are not comparable.
  The share-of-attainable-gain form (26% against 13–14%) stays in the text.

---

## Things to decide, not assume

- **Venue** sets order (Methods before or after Results) and length. Not assumed here.
- **Findings 8–9** (shape/3D, nuclease descriptors) are nulls. Supplementary under Frame A, unless a
  reviewer asks.
- **Jacky's orientation result and the human arm** are his; agree authorship and which figures are
  shared before drafting.
- **The three withdrawn claims** must not reappear. The report lists them once, at the top.
