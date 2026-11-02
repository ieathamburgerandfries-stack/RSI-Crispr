# Beyond the quantum descriptors: what else moves `cut.score`?

**RSI09 — Enhancing sgRNA efficiency in CRISPR-Cas9 genome editing**
Literature review, September 2026. Baseline for comparison: XGBoost, top-300 features, 5-seed CV **R² 0.294 / Spearman 0.528 / Pearson 0.542** on 13,880 *E. coli* sgRNAs.

---

## 0. Read the target variable before adding features

Everything below is constrained by what `cut.score` actually measures. In Guo *et al.* (2018) — the source of the *E. coli* labels — the score is the log2 change in sgRNA abundance between a Cas9 condition and a dCas9 control in a pooled library of ~70,000 guides. It is a **survival assay**, not a direct measurement of cleavage.

That distinction matters. A guide gets a low score either because Cas9 never cut, or because the cell survived the cut. *E. coli* has no NHEJ, so a chromosomal DSB is usually lethal — but cells escape by deleting the target region, by losing the guide, or when the break sits in a dispensable, duplicated, or already-deleted locus. Anything that changes escape probability appears in the label as if it were a cutting-efficiency effect. Families D–H below act partly on that second half of the pipeline, and are worth separating in the write-up.

**Structural gap in the current matrix.** Positions `p1`–`p20` plus `PAM` encode the protospacer and nothing else. Every family that depends on *where the guide sits in the genome* is currently invisible to the model. Recovering genomic coordinates for each `sgRNAID` is the one step that unlocks families D through H at once.

---

## 1. The eight factor families

Status legend: **[in matrix]** already represented · **[computable]** derivable from sequence or the reference genome, no new dataset · **[needs dataset]** requires an external join · **[reported null]** tested in the literature and found weak.

### A. Sequence context outside the 20-mer — [computable]

The single largest documented gain for bacterial Cas9 prediction, and the cheapest to obtain. Curated re-analysis of the Guo *E. coli* SpCas9 screen found that extending the input window beyond the protospacer improved Spearman by **+0.143**, using up to 189 nt upstream and 166 nt downstream. The effect concentrates in the ~11 nt immediately 3′ of the PAM, but AT-richness of the wider flank carries real signal too (especially for eSpCas9, which disfavours AT-rich flanks over >100 nt in both directions). crisprHAL, the current best bacterial model, uses a 28-nt input (20 spacer + 3 PAM + 5 downstream) for exactly this reason.

- **Add:** one-hot *and* QCT encodings at positions +1…+15 downstream of PAM and −1…−10 upstream; AT fraction and GC skew in ±200 nt windows; melting temperature of the flank.
- **Why it fits this project:** the QCT pipeline already generates monomer/basepair/dimer/trimer/tetramer descriptors per position. Running it over flanking positions is a mechanical extension — and it turns "quantum descriptors of the protospacer" into "quantum descriptors of the target locus", which is a stronger paper.
- **Source:** MG1655 reference `NC_000913.3`. No external dataset needed.

### B. R-loop free-energy decomposition — [computable]

Cas9 pays to unwind the DNA duplex and to unfold its own guide, and is repaid by RNA–DNA hybridisation. The CRISPRspec/CRISPRoff energy model splits this into ΔG_H (hybridisation), ΔG_U (guide unfolding penalty), ΔG_O (DNA opening penalty), and net ΔG_B. Highly efficient guides cluster in a narrow ΔG_H window of roughly **−64.5 to −47.1 kcal/mol**; efficiency falls off on both sides — a non-monotonic relationship plain GC content cannot express. The same work found a G immediately 3′ of the PAM costs ~12.6% efficiency while a G immediately 5′ gains ~7.2%. CRISPRon showed that feeding ΔG_B into a model alongside sequence lowered MSE by ~2.7%.

- **Add:** ΔG_H, ΔG_U, ΔG_O, ΔG_B, plus `|ΔG_H − optimum|` as an explicit distance-from-sweet-spot term so the tree model doesn't have to discover the window itself.
- **Tool:** CRISPRoff 1.1.1 energy function over ViennaRNA (RNAfold 2.x). Minutes for 13,880 sequences.
- **Caveat:** nearest-neighbour thermodynamics overlaps conceptually with your QCT H-bond and stacking descriptors. That overlap is itself a result — report the correlation between the two families.

### C. sgRNA folding and spacer–scaffold interference — [partly in matrix] [computable]

The matrix has a single `sgRNA.structure` column, and the team has tested structure *proxies* but not the real quantity. A meta-analysis of 39 guide-library screens — including a 1.2-million-guide self-targeting *E. coli* library — found two distinct sigmoidal failure modes: spacer self-folding (activity collapses below **spacer MFE −5 kcal/mol**) and spacer–scaffold base-pairing that disrupts the repeat:anti-repeat duplex (**duplex stability below −15 kcal/mol**). Both are threshold effects on a minority of guides — exactly the shape gradient-boosted trees capture well and linear terms do not.

- **Add:** spacer-only MFE; full sgRNA (spacer + SpCas9 scaffold) MFE; spacer–scaffold duplex stability; per-base unpaired probability across the seed (positions 1–10 from PAM) via RNAplfold.
- **Tool:** ViennaRNA `RNAfold` / `RNAduplex` / `RNAplfold`.
- **Note:** fold the transcript actually expressed in Guo's library, scaffold included. Folding the bare 20-mer is the most common way this feature gets wrongly reported as useless.

### D. DNA supercoiling and local mechanics — [needs dataset] [computable]

Guo *et al.* explicitly attributed part of the unexplained variance to "unknown chromosomal factors" — loci resistant to cutting for reasons the sequence does not carry. Supercoiling is the leading candidate: R-loop formation requires local unwinding, so negatively supercoiled DNA should be easier for Cas9 to invade. A biophysical model of Cas9 activity (Farasat & Salis) treats supercoiling as a first-class term, and recent biochemistry shows supercoiling changes the *sequence-dependence* of cleavage, not just its rate. *E. coli* now has genome-wide supercoiling maps at kb resolution.

- **Add:** supercoiling density at the target bin (10 kb and 1 kb); distance to nearest highly transcribed operon (transcription dominates the supercoiling landscape); DNA-shape descriptors — minor groove width, roll, propeller twist, helix twist — over protospacer and PAM.
- **Datasets:** GapR-seq positive-supercoiling map, GEO `GSE152880`; psoralen-seq landscape (Visser 2022); DNA shape via `DNAshapeR` (no dataset needed).
- **Evidence:** DNA-shape features already improve Cas9 activity prediction in eukaryotic data; the supercoiling join appears untested for sgRNA efficiency in *E. coli*. **Most novel item on this list.**

### E. Nucleoid organisation — the bacterial analogue of chromatin — [needs dataset]

In eukaryotes chromatin accessibility is one of the strongest non-sequence predictors: nucleosomes measurably impede Cas9 in vitro and in vivo, and a 2024 study found that adding 3D-genome spatial-density features from Hi-C raised model R² by **11–12%** on human datasets. Bacteria have no nucleosomes, but they do have nucleoid-associated proteins — H-NS in particular coats AT-rich, horizontally acquired regions in dense filaments — and a well-mapped 3D nucleoid with macrodomains and MukBEF-organised loops. Whether NAP occupancy blocks Cas9 in *E. coli* is an open question with a clean answer available from existing public data.

- **Add:** H-NS / Fis / IHF / HU ChIP-seq occupancy at the target; macrodomain identity (Ori, Ter, Left, Right, NS); Hi-C spatial-density features (mean distance to *n* nearest bins), transposing the human method to the *E. coli* contact map.
- **Datasets:** Lioy *et al.* 2018 (Cell) Hi-C + NAP ChIP-seq for MG1655, code at `github.com/koszullab/E_coli_analysis`; Kahramanoglou *et al.* H-NS/Fis ChIP-seq.
- **Why it's a good student project:** a direct, defensible port of a published eukaryotic result into a bacterial system — clear hypothesis, public data, and a negative result that is still publishable.

### F. DNA methylation at the target site — [computable]

Concrete very recently. Adenine methylation at Dam `GATC` motifs inhibits SaCas9 cleavage roughly **10-fold** when the motif overlaps PAM-proximal positions, confirmed by comparing wild-type and Dam-deficient *E. coli*; structural work on ThermoCas9 shows methylation of a PAM cytosine blocks *binding* outright rather than catalysis (0% editing at methylated vs 16–33% at unmethylated sites). SpCas9 is generally described as methylation-tolerant in the protospacer body, but the PAM and its immediate flank are the sensitive zone — and MG1655 methylates every `GATC` (Dam, 6mA) and `CCWGG` (Dcm, 5mC) in the genome.

- **Add:** binary and positional indicators for `GATC` and `CCWGG` overlapping the protospacer, the PAM, and positions +1…+5; count of methylated bases in the seed region.
- **Cost:** a regex over the target sequences. Cheapest genuinely novel feature on the list.
- **Expectation:** likely small-but-real, confined to a subset of guides — a clean SHAP story even if global R² barely moves.

### G. Transcription and replication context — [needs dataset] [computable] [partly null]

Elongating RNA polymerase dislodges Cas9 from its target after cleavage, asymmetrically: the effect depends on which strand the guide matches and how heavily the gene is transcribed. But note the discipline required here — Guo *et al.* tested the coarse versions (target gene expression, chromosome position, leading vs lagging strand) and reported no meaningful correlation. Only the mechanistically specific forms are worth building.

- **Add:** template vs non-template strand *interacted with* local transcription level; RNAP occupancy at the target; distance to the operon TSS; co-directional vs head-on orientation relative to the replication fork; distance to *oriC* as a gene-dosage proxy.
- **Datasets:** PRECISE / iModulon *E. coli* RNA-seq compendium (Sastry 2019) for expression; RegulonDB / EcoCyc for operon and TSS structure; strand and *oriC* distance computable from the reference genome.
- **Rule:** only claim this family if the interaction terms beat *both* parents in the ablation. Otherwise report it as a replication of the null — still a result.

### H. Post-cleavage survival and assay artefacts — [computable]

The part of the label that isn't about Cas9 at all. A guide with several perfect or near-perfect genomic matches — rRNA operons, IS elements, duplicated genes — causes multiple simultaneous breaks and is far more lethal, inflating apparent efficiency; conversely, Cas9 toxicity from off-target load has been used as a genome-scale *readout* in human screens. Guides in deletable regions let cells escape. And on the measurement side, curated re-analysis of the same Guo data found that filtering guides by minimum control-condition read count improved Spearman by up to **+0.036** even after discarding 45% of rows — **data quality outweighed data quantity**.

- **Add:** count of exact and ≤3-mismatch genomic matches (Bowtie or Cas-OFFinder); repeat/IS-element overlap; distance to nearest prophage boundary; duplicated-gene-family flag.
- **Also do:** apply a control-read-count floor to the label before training and report performance both ways. This is a change to *y*, not *X*, and may raise the ceiling more than any single feature family.

---

## 2. Ranked shortlist

Ordered by expected gain per unit of work, assuming the existing ablation protocol (5 seeds x 5 folds, leakage-safe, one family at a time).

| # | Addition | Family | Effort | Evidence for the gain | Tier |
|---|---|---|---|---|---|
| 1 | Genomic coordinate lookup for every `sgRNAID` | enabler | low | Prerequisite for D, E, F, G, H | do first |
| 2 | Flanking sequence + QCT beyond the protospacer | A | low | +0.143 Spearman on this exact dataset | do first |
| 3 | Real ViennaRNA folding features (spacer, full, duplex) | C | low | Sigmoidal thresholds at −5 / −15 kcal/mol | do first |
| 4 | CRISPRoff ΔG decomposition + sweet-spot distance | B | low | −2.7% MSE in CRISPRon; non-monotonic | do first |
| 5 | Control-read-count filter on the label | H | low | +0.029–0.036 Spearman after curation | do first |
| 6 | Dam/Dcm methylation motifs at PAM and seed | F | low | ~10x effect for SaCas9; novel for SpCas9 | high novelty |
| 7 | Expand rows: full Guo SpCas9 + eSpCas9 sets | data | med | 13,880 → 33k–59k curated guides | high novelty |
| 8 | DNA-shape descriptors (DNAshapeR) | D | med | Published gain on Cas9 activity prediction | high novelty |
| 9 | Supercoiling density join (GapR-seq) | D | med | Untested here; addresses "unknown loci" | exploratory |
| 10 | NAP occupancy + Hi-C spatial density | E | high | +11–12% R² for the human analogue | exploratory |
| 11 | Off-target / multi-copy burden counts | H | med | Mechanistic; confounds the label | exploratory |
| 12 | Transcription x strand interactions | G | high | Marginals reported null; interactions untested | exploratory |

---

## 3. Datasets you can actually join

| Dataset | What it gives you | Organism / scale | Join key | Access |
|---|---|---|---|---|
| Guo *et al.* 2018 full screens | More labelled guides; eSpCas9 as a second nuclease | *E. coli*, 40,308 SpCas9 + 45,010 eSpCas9 | sgRNA sequence | NAR 2018 supplementary; curated copies in crisprHAL repo |
| crisprHAL curated sets | Read-count-filtered labels; external validation sets | *E. coli* 33k / 59k; *C. rodentium* 25k | sgRNA sequence | `github.com/tbrowne5/crisprHAL` |
| Rousset / Bikard CRISPRBact screen | Independent *E. coli* Cas9 cutting screen | *E. coli* MG1655, ~92,000 guides | genomic coordinate | bioRxiv 308148 + CRISPRBact tool |
| Lioy *et al.* 2018 | Hi-C contact maps; H-NS/HU/Fis/MatP/MukBEF profiles | *E. coli* MG1655, 5 kb bins | genomic coordinate | `github.com/koszullab/E_coli_analysis` |
| GapR-seq supercoiling map | Positive-supercoiling density genome-wide | *E. coli*, high resolution | genomic coordinate | GEO `GSE152880` |
| PRECISE / iModulon | Expression level and regulatory module per gene | *E. coli* MG1655, 278+ conditions | b-number | iModulonDB; Sastry 2019 |
| RegulonDB / EcoCyc | Operon structure, TSS positions, strand | *E. coli* MG1655 | b-number / coordinate | regulondb.ccg.unam.mx |
| Keio collection (Baba 2006) | Gene essentiality | *E. coli*, 4,288 genes | b-number | already built into your matrix |
| Kim *et al.* 2019 (DeepSpCas9) | Human on-target efficiency with 30-mer context | *H. sapiens*, 13,359 targets | target sequence | published supplementary |
| Xiang *et al.* 2021 (CRISPRon) | Human efficiency + precomputed ΔG terms | *H. sapiens*, 10,592 (23,902 merged) | target sequence | Nat Commun 2021 supplementary |

The two human sets matter for the cross-species half of the proposal: the Noshay *et al.* human matrix was built on Doench 2014 (1,278 guides) and Chuai 2018 (16,749). Adding CRISPRon's merged 23,902-guide set roughly doubles the human side and brings matched thermodynamic features with it.

---

## 4. What the literature says *not* to spend time on

Three variables have already been tested against this exact label and reported as uncorrelated by Guo *et al.*: **gene essentiality**, **chromosome position**, and **target gene expression level**. The Keio essentiality experiment is therefore a replication of a published null rather than a failed idea — worth one paragraph in the paper, not another iteration. Note the specific confound: in a depletion screen every successful cut is lethal regardless of which gene it hits, so essentiality has little room to act.

Two further cautions:

1. **Eukaryotic rules do not transfer.** The poly-T Pol III termination signal, CpG methylation effects, and nucleosome positioning have no bacterial counterpart. The 5′-end-dominant position weighting reported for *E. coli* is the *opposite* of the PAM-proximal dominance seen in human data.
2. **Collinearity with QCT.** Hydrogen-bond and stacking tensors encode much of what nearest-neighbour thermodynamics encodes. Test each family both alone and stacked on the baseline, or you will report a gain that is really a re-parameterisation.

---

## 5. A concrete next sprint

1. **Recover coordinates and sequences.** Parse `sgRNAID` (`aaeAb3241_107_Cas9` → gene `aaeA`, b-number `b3241`, offset `107`), map through the MG1655 annotation to a genome coordinate and strand, pull the ±200 nt window from `NC_000913.3`. Sanity-check by reconstructing the protospacer and confirming it matches the positional encodings already in the matrix.
2. **Build the three cheap feature blocks.** Flanking sequence + QCT (A), ViennaRNA folding (C), CRISPRoff ΔG terms (B). Name them `eng.flank.*`, `eng.fold.*`, `eng.dg.*` so they follow the `ENGINEERED_PREFIXES` convention already in the baseline script.
3. **Run the standard ablation, one family at a time.** Five seeds, five folds, feature selection fit inside the training split only. Report ΔSpearman and ΔR² against the 0.528 / 0.294 baseline, with the random-control-seed test as the significance floor.
4. **Re-run once with the curated label.** Apply a control-read-count floor to `cut.score` and repeat the winning configurations. Report both — a change in the label changes what the numbers mean.
5. **Then attempt one genome-context family.** Supercoiling or NAP occupancy, whichever dataset aligns first. Highest risk, highest novelty, and the one that speaks directly to the "unknown chromosomal factors" the original authors could not explain.

---

## 6. Sources

1. Noshay *et al.* (2023). Quantum biological insights into CRISPR-Cas9 sgRNA efficiency from explainable-AI driven feature engineering. *Nucleic Acids Research* 51:10147. https://academic.oup.com/nar/article/51/19/10147/7279034
2. Guo *et al.* (2018). Improved sgRNA design in bacteria via genome-wide activity profiling. *NAR* 46:7052. https://academic.oup.com/nar/article/46/14/7052/5047272 — preprint: https://www.biorxiv.org/content/10.1101/272377v1.full
3. Ham *et al.* (2023). A generalizable Cas9/sgRNA prediction model using machine transfer learning with small high-quality datasets. *Nature Communications* 14:5514. https://www.nature.com/articles/s41467-023-41143-7
4. Browne *et al.* (2025). Better data for better predictions: data curation improves deep learning for sgRNA/Cas9 prediction. *PeerJ*. https://peerj.com/articles/20706/
5. Rousset *et al.* (2018). Genome-wide CRISPR-Cas9 screen in E. coli identifies design rules for efficient targeting. https://www.biorxiv.org/content/10.1101/308148v1.full
6. Xiang *et al.* (2021). Enhancing CRISPR-Cas9 gRNA efficiency prediction by data integration and deep learning. *Nature Communications* 12:3238. https://www.nature.com/articles/s41467-021-23576-0
7. Alkan *et al.* (2022). CRISPR/Cas9 gRNA activity depends on free energy changes and on the target PAM context. *Nature Communications* 13:2601. https://www.nature.com/articles/s41467-022-30515-0
8. Moreb & Lynch (2022). A meta-analysis of gRNA library screens: gRNA folding and structural stability. *The CRISPR Journal*. https://www.biorxiv.org/content/10.1101/2021.05.29.446220v1.full.pdf
9. Farasat & Salis (2016). A biophysical model of CRISPR/Cas9 activity for rational design of genome editing and gene regulation. *PLOS Comp Biol*. https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1004724
10. Lioy *et al.* (2018). Multiscale structuring of the E. coli chromosome by nucleoid-associated and condensin proteins. *Cell* 172:771. https://www.cell.com/cell/fulltext/S0092-8674(17)31507-6 — data: https://github.com/koszullab/E_coli_analysis
11. Guo *et al.* (2021). High-resolution, genome-wide mapping of positive supercoiling in chromosomes. *eLife* 10:e67236. https://elifesciences.org/articles/67236 — GEO GSE152880
12. Differential roles of positive and negative supercoiling in organizing the E. coli genome. *NAR* 52:724. https://academic.oup.com/nar/article/52/2/724/7458899
13. Machine learning reveals sequence and methylation determinants of SaCas9–PAM interactions in bacteria (2026). *NAR* 54:gkaf1520. https://academic.oup.com/nar/article/54/2/gkaf1520/8426367
14. Molecular basis for methylation-sensitive editing by Cas9 (2026). *Nature*. https://www.nature.com/articles/s41586-026-10384-z
15. Strong association between genomic 3D structure and CRISPR cleavage efficiency (2024). *PLOS Comp Biol* 20:e1012214. https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1012214
16. DNA shape features improve prediction of CRISPR/Cas9 activity (2024). *Methods*. https://www.sciencedirect.com/science/article/abs/pii/S1046202324001026
17. Horlbeck *et al.* (2016). Nucleosomes impede Cas9 access to DNA in vivo and in vitro. *eLife* 5:e12677. https://elifesciences.org/articles/12677
18. Clarke *et al.* (2018). Enhanced bacterial immunity and mammalian genome editing via RNA-polymerase-mediated dislodging of Cas9 from double-strand DNA breaks. *Molecular Cell*. https://www.cell.com/molecular-cell/fulltext/S1097-2765(18)30447-7
19. Morgens *et al.* (2017). Genome-scale measurement of off-target activity using Cas9 toxicity in high-throughput screens. *Nature Communications* 8:15178. https://www.nature.com/articles/ncomms15178
20. Sastry *et al.* (2019). The E. coli transcriptome mostly consists of independently regulated modules. *Nature Communications* 10:5536. https://www.nature.com/articles/s41467-019-13483-w
21. Dorman (2019). CRISPR-Cas, DNA supercoiling, and nucleoid-associated proteins. *Trends in Microbiology*. https://pubmed.ncbi.nlm.nih.gov/31519332/
22. Evaluation and prediction of guide RNA activities in genome-editing tools (2025). *Nature Reviews Bioengineering*. https://www.nature.com/articles/s44222-025-00352-z
