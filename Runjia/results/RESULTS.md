# Results — claim to evidence

This file is an **index**, not a narrative. For every claim the progress report
makes, it names the file that backs it and the command that produced it, so any
number in the report can be checked without reading the report.

The narrative lives in [`../docs/PROGRESS_REPORT.md`](../docs/PROGRESS_REPORT.md);
the chronology, including claims later withdrawn, lives in
the shared repository's git history.

**What is indexed:** results a claim rests on. That includes nulls when the null
*is* the claim — the periodicity family, the shift control —
because a ruled-out mechanism is a result. The dCas9 / CRISPRi rows below are **not** claims: that experiment was demoted to
Appendix B5 of the report (22 November) because it did not identify a mechanism, and its rows are kept as the record of it.

**What is not indexed:** exploratory sweeps, superseded runs, tangential
questions, per-fold dumps (`folds_*.csv`), per-run settings (`*_settings.json`)
and prediction dumps (`preds_*.csv`). Those files are still here and still
readable; they are just not load-bearing. The long-form commentary written up to
10 October 2026 is kept verbatim in
[`RESULTS_archive_2026-10-10.md`](RESULTS_archive_2026-10-10.md).

**Which cohort a number belongs to.** The project pivoted on 13 October 2026 from
Noshay's published 13,880-row matrix and `cut.score` to crisprHAL 2's curated
33,567-guide release and its ALDEx2 score. Both frames appear below and are
marked **[P]** published or **[C]** curated. They are not interchangeable.

---

## The one large effect

| claim | number | frame | evidence | command |
|---|---|---|---|---|
| Flanking DNA composition is the largest effect measured here | +0.164 ρ | [C] | `ablation_curated.csv` | `python -m sgrna.cohorts.curated_ablation --all` |
| …and +0.080 on the published matrix, so the effect is not an artefact of the pivot | +0.080 ρ | [P] | `ablation_main.csv` | `python -m sgrna.harness.run_ablation` |
| It survives holding out whole genes, losing only the grouping tax | 6% tax | [C] | `ablation_curated_gene.csv`, `grouped_retention.csv` | `python -m sgrna.cohorts.curated_ablation --all --grouping gene`; `python -m sgrna.controls.rederive --only grouped_retention` |
| The fold design moves only the families that track the locus (same model, same cohort, random rows against the paper's folds) | flank 98% kept, energy 97%, folding 99%, methylation 99%, thermodynamics 98%, shape 98%, supercoiling 98%; nucleoid 90%, transcription 78%, periodicity 75% | [C] | `ablation_paper_folds_summary.csv`, `ablation_curated.csv` | `python -m sgrna.cohorts.curated_ablation --paper-folds` |
| It is spatially local: a texture-preserving circular shift destroys it | 8% at 1 kb, 0 at 10 kb | [P] | `shift_null.csv` | `python -m sgrna.controls.family_a_controls --parts shift` |
| It is a smooth gradient, not a motif: k-mer order buys nothing | tetramers −0.0018 | [C] | `curated_ladder.csv`, `kmer_order_noselect.csv` | `python -m sgrna.cohorts.curated_ablation --ladder`; `python -m sgrna.controls.rederive --only kmer_order_noselect` |
| The window widths are not where the gain lives | +0.1634 vs +0.1638 | [C] | `window_robustness.csv` | `python -m sgrna.controls.family_a_controls --parts windows` |
| It is not target GC in disguise: conditioning raises it | +0.044 → +0.082 | [P] | `gc_strata_corr.csv`, `gc_strata_ablation.csv` | `python -m sgrna.interpret.flank_signal --parts gc-corr gc-ablation` |
| It is not gene ranking: it holds inside genes | +0.085 within-gene [+0.078, +0.090] | [P] | `cut_within_gene_summary.csv` | `ext/cut_within_gene.py` (EXTENSION_LOG §1) |
| It is not read depth or mappability: the gain is +0.1569 [+0.1448, +0.1694] among the 26,146 guides with no repeated 25-mer within 1 kb, no seed duplicate and one copy (all guides +0.1606), larger in repeated sequence (+0.2607, 852 guides), and +0.1600 given the control read depth as a column | +0.157 ρ | [C] | `mappability_stratum.csv`, `mappability_depth.csv` | `python -m sgrna.controls.mappability_stratum` (second review round, 23 Nov; supersedes the unsupported row of the published matrix) |

## The second review round (23 Nov): one protocol, the reach of the flank, transfer to every dataset, cost with its spread

All under five-fold cross-validation with whole genes held out, three seeds, the default model, the 20-mer alone as the baseline (frame **[C]**); every number is recomputed by `sgrna/paper/paper_numbers_D.py`.

| claim | number | evidence | command |
|---|---|---|---|
| The flank gain under the one protocol | +0.161 ρ (0.547 → 0.707; sd 0.010; 15 of 15 folds) | `flank_scale.csv` | `python -m sgrna.interpret.flank_scale` (seed 41: window, resolution, displacement), then `--parts resolution --seeds 42 43` and `--parts full --seeds 42 43` |
| Two GC columns carry most of it; highest at 200 nt of 14 symmetric windows, within 0.01 of it from 150 to 500 nt; the four bases add nothing | +0.106 at W = 200 nt, +0.105 at 250 nt, +0.042 at 5 kb; GC +0.113, A/C/G/T +0.108 | `flank_scale.csv`, `flank_scale_summary.csv`, `flank_window_tune.csv` | `python -m sgrna.interpret.flank_scale --parts window partition asymmetric` |
| Tuning the upstream and downstream windows separately finds nothing that replicates | best of 33 combinations (200 nt up, 250 nt down): +0.0021 over 250/250 on its own folds (5 of 5), +0.0001 on seeds 42 and 43 (5 of 10 folds); the two sides never differ by more than 0.004 | `flank_window_tune.csv`, `flank_window_tune_summary.csv` | `python -m sgrna.interpret.flank_scale --parts asymmetric` |
| It needs the guide's own flank and fades over about 3 kb | 68%, 57%, 32%, 6% at 100 bp, 300 bp, 1 kb, 3 kb; 0 from 10 kb | `flank_scale.csv` | same |
| A step at the AT-rich end, graded by the base at position 20 | lowest decile −3.154; ρ with flank GC 0.219 (A), 0.119 (G) | `signal_decile.csv`, `signal_slope.csv`, `signal_target_gc.csv`, `signal_corr.csv` | `python -m sgrna.interpret.flank_signal --parts profile` |
| Flank and guide score are close to additive | R² +0.0035 (15 of 15), ρ +0.0007 | `gate_test.csv` | `python -m sgrna.interpret.flank_signal --parts gate` |
| It survives a second screen's labels at the sealed genes | +0.1033 [+0.0997, +0.1070], 15 of 15 | `sealed_crossscreen.csv` | `python -m sgrna.controls.sealed --parts crossscreen` |
| Curation does not select on the flank | kept by flank-GC decile 0.53 to 0.57; discarded guides carry the same association | `curation_check.csv` | `python -m sgrna.cohorts.curation --parts check` |
| SLICER against crisprHAL 2 on more metrics | AUROC +0.011 / +0.016; top-decile precision and NDCG not detected | `metrics_panel.csv` | `python -m sgrna.compare.panel --parts splits` |
| Transfer across nuclease, against crisprHAL 2 | 0.659 / 0.678 against 0.633 | `transfer_vs_crisprhal_nuclease.csv` | `python -m sgrna.compare.vs_crisprhal --condition transfer-prepare / transfer-tf / transfer-report` |
| Transfer across kingdom, against crisprHAL 2 | median −0.041 against −0.015 | `transfer_vs_crisprhal_kingdom.csv` | same |
| The full 12 x 12 transfer table | every eukaryote scored on E. coli negative (−0.120 to −0.012) | `eukaryote_transfer.csv`, `transfer_to_ecoli.csv` | `python -m sgrna.compare.eukaryote_rules --parts maps transfer`; `python -m sgrna.compare.eukaryote_rules --parts to-ecoli` |
| Cost with its spread | crisprHAL 2 takes 24.8, 18.1 and 15.1 times as long as SLICER's default model per guide at batches of 1, 100 and 10,000; fits 14 s (default), 25 s (tuned), 17 min (crisprHAL 2, CPU) | `cost_repeats_inference.csv`, `cost_repeats_fit.csv`, `model_size.csv` | `python -m sgrna.cost.cost_repeats inference / hal / fit`; `python -m sgrna.cost.model_size` |
| BoostMEC retrained on this cohort as its authors trained it, on the features of their own pipeline (run in their Docker image) | released hyperparameters ρ 0.620 ± 0.009 (−0.087 against SLICER default, behind in 15 of 15 folds); its own search 0.623 on the held-out genes against 0.707 and 0.722; the released model (human cells) scores ρ −0.066 on E. coli | `boostmec_released.csv`, `boostmec_searched.csv`, `boostmec_search_trace.csv`, `boostmec_vs_slicer_confirm.csv`, `boostmec_shipped_cohort.csv` | `python -m sgrna.compare.boostmec cohort`, then `python -m sgrna.compare.boostmec released / search / compare` |
| Cost on a quiet CPU (Apple M5, one thread for scoring, ten cores for fits), 24 Nov | per guide at 100: SLICER default 0.0726 ms, tuned 0.0885, crisprHAL 2 1.37 (18.8×), BoostMEC 3.07 (42.3×); fits 12.8 s, 24.5 s, BoostMEC 13.2 s, crisprHAL 2 17.3 min (81×) | `cost_repeats_inference.csv`, `cost_repeats_fit.csv`, `cost_repeats_machine.json`, `model_size.csv` | `sgrna/cost/cost_chain.sh` |
| Cost on a Colab T4, 24 Nov | crisprHAL 2 from raw contexts 0.311 ms per guide at 10,000, 0.0837 with the encoded batch on the card (1.05× SLICER default's whole CPU path); fit 326 s | `cost_gpu.csv` | `colab/cost_gpu.py` |
| BoostMEC as shipped (authors' Docker image, unmodified, under emulation) | 10,000 guides in 53.5 s (5.35 ms per guide, 67×); a one-guide input fails; features identical to the Python version except four melting temperatures | `boostmec_shipped_timings.csv`, `boostmec_shipped_stages.csv`, `boostmec_shipped_check.csv`, `boostmec_shipped_rnafold.csv` | `python -m sgrna.compare.boostmec time / stages / check / rnafold / cohort` |
| crisprHAL 2 on whole-gene folds at its published batch (1,024; Colab T4; batch 256 was a handicap, 0.680, and is kept apart) | crisprHAL 2 0.693 against SLICER default 0.707 (+0.014, 5 of 5 folds) and tuned 0.723 (+0.031, 5 of 5); on random-row folds crisprHAL 2 scores 0.697 | `crisprhal_gene_grouped_fcolab_b1024.csv`, `crisprhal_gene_grouped_fcolab_b256.csv`, `gene_grouped_vs_crisprhal.csv` | `colab/crisprhal_gene_grouped.py`, `python -m sgrna.compare.vs_crisprhal --condition gene-assemble` |
| Why the diagonal of the transfer table is low | 20-mer cross-validation 0.18 to 0.55 against a median published predictor of 0.10 to 0.42 on the same tables | `transfer_diagonal.csv` | `python -m sgrna.compare.eukaryote_rules --parts diagonal` |


## The representation

| claim | number | frame | evidence | command |
|---|---|---|---|---|
| 99% of the 6,232 published columns regenerate from a 20-mer | 5,853 of 5,887 `V` columns decoded | [P] | `portability_audit.csv` | `python -m sgrna.build.curated_matrix --build` |
| 412 columns computable from any guide match all 6,232 | 0.00003 ρ | [P] | `portability.csv`, `portability_364.csv` | `python -m sgrna.interpret.representation --portability --audit --run`; `python -m sgrna.controls.rederive --only portability_364` |
| …but no reduced set is non-inferior on the curated cohort | −0.0118 / −0.0143 | [C] | `curated_ladder.csv` | `python -m sgrna.cohorts.curated_ablation --ladder` |
| A deterministic function of the guide adds no information, but does add performance | +0.011 to +0.019 | [C] | `no_information.csv`, `base_enrichment.csv` | `python -m sgrna.controls.diagnose --what no-information`, `python -m sgrna.interpret.representation --base-enrichment` |
| On the paper's own folds (whole genes, 33,567 guides, seeds 41 to 43) no guide-derived family gains 0.02; the same ablation on random rows (team harness) agrees to within 0.002 for all but periodicity | energy +0.0181, folding +0.0143, methylation +0.0077, periodicity +0.0081, off-target −0.0003; thermodynamics +0.0972, 99% of it from its 8 flank columns | [C] | `ablation_paper_folds.csv`, `ablation_paper_folds_summary.csv`, `ablation_curated.csv` | `python -m sgrna.cohorts.curated_ablation --paper-folds` |
| …but **random** functions of the same shape add nothing | −0.0004 to −0.0015 at 4–256 tables | [C] | `redescription_random.csv` | `python -m sgrna.interpret.redescription --arm random` |
| **Sorted by column rather than by family**: what the guide determines is worth +0.008, what it does not +0.201 | 24× | [C] | `redescription_taxonomy.csv`, `redescription_columns.csv` | `python -m sgrna.interpret.redescription --arm taxonomy` |
| Every family split the same way: the two halves differ in all six | see table | [C] | `redescription_family_split.csv` | `python -m sgrna.interpret.redescription --arm families` |
| One guide-determined but non-linear column, `eng.dg.guide_selffold`, carries `b_energy`'s whole gain | +0.0116 of +0.0181 | [C] | `redescription_family_split.csv`, `redescription_columns.csv` | `python -m sgrna.interpret.redescription --arm families` |
| The eleven never-tuned XGBoost knobs are worth nothing; two do harm | +0.0069 best, −0.0080 worst | [C] | `xgb_knobs.csv` | `python -m sgrna.tuning.xgb_knobs` |
| …and the one apparent winner is capacity, not grow policy: `max_depth` 6 beats 4 | +0.0063 [+0.0053, +0.0073], 15/15 folds | [C] | `xgb_knobs_verify.csv` | `python -m sgrna.tuning.xgb_knobs --verify` |
| The folding gain is the **real structure**: position-shuffled folds are a null | +0.0119 vs −0.0002 over 3 permutations | [C] | `redescription_folds.csv` | `python -m sgrna.interpret.redescription --arm folds` |
| **Recoverability bounds what a column can add, and is unrelated to its own association**: the strongest \|ρ\| per band is flat while the increment collapses. A ρ of +0.5574 on *mean* \|ρ\| is a composition artefact and is withdrawn | strongest \|ρ\| 0.168/0.156/0.165/0.143/0.183/0.184 against Δρ +0.180→+0.008 | [C] | `recoverability_trend.csv`, `recoverability_trend_columns.csv`, `figures/recoverability_vs_power.png` | `python -m sgrna.interpret.redescription --arm trend` |
| **Helping does not require low recoverability**: the most recoverable, most strongly associated column in the matrix still adds a hundredth on its own. Recoverability bounds the increment, it does not forbid one | `eng.dg.hybridisation_weighted` R² 1.000, \|ρ\| 0.184, **+0.0063** alone; the whole 18-column block is +0.0083, so they overlap | [C] | `recoverability_percolumn.csv` | `python -m sgrna.interpret.redescription --arm percolumn` |
| …and recoverability is a **threshold, not a gradient**: once the guide explains ~20% of a column it contributes nothing | Δρ +0.180/+0.148/+0.129 below R² 0.2 against +0.018/+0.021/+0.008 above it | [C] | `recoverability_trend.csv` | `python -m sgrna.interpret.redescription --arm trend` |
| Grouping the columns into the 66 variables they measure does not recover a trend either — every group falls below recoverability 0.5, and the ordering tracks how much DNA a group summarises | single-position indicators ~0 on both axes; multi-position summaries carry the power | [C] | `recoverability_groups.csv` | `scratchpad grouptrend` |
| **Difficulty without meaning buys nothing**: random non-linear functions of the guide, matched on recoverability to the real fold energy, are null at every interaction order | +0.0001 to +0.0007 (R² 1.00 → −0.008) vs +0.0117 for the real column at R² 0.316 | [C] | `redescription_nonlinear.csv` | `python -m sgrna.interpret.redescription --arm nonlinear` |
| **The column selector uses split gain, the one metric Finding 5 says never to trust** — it over-weights binary indicators by 14 points and costs no accuracy, so the bias is interpretive, not predictive | gain 144/300 binary vs SHAP 102/300; ρ 0.7096 vs 0.7103 (both selection-leaked, not quotable as levels) | [C] | `selector_audit.csv`, `selector_audit_ranks.csv` | `python -m sgrna.controls.selector_audit` |
| **And a redundancy-penalised selection rule is worse, not neutral** — decorrelation and relevance point opposite ways here, because the redundant columns are the informative dense ones | top-*k* ρ 0.7083 at redundancy 0.647; MRMR ρ 0.7068 at 0.544, with *more* binary columns (175 vs 146) | [C] | `selector_redundancy.csv` | `python -c "from sgrna.controls.selector_audit import redundancy; redundancy()"` |
| **Per-guide label reliability is not usable**: inverse-variance weighting on the training loss costs accuracy, and four fifths of the cost is reproduced by permuting the weights — ALDEx2's posterior has already spent the information | unweighted 0.7083; inv-var −0.0050; permuted −0.0041; flat-in-|y| −0.0031; depth −0.0241; worst decile −0.0016 vs random decile −0.0014 | [C] | `replicate_weight.csv` | `python -m sgrna.cohorts.replicate_weight` |
| …and the shrinkage explanation for that is **wrong**: on the raw log-ratio, where nothing is shrunk, weighting still costs −0.004 and the permuted control reproduces all of it. The cost is effective sample size | raw label: unweighted 0.6864, weighted 0.6824, permuted 0.6820. The raw label is also 0.022 less predictable than the released one | [C] | `replicate_weight_raw.csv` | `python -m sgrna.cohorts.replicate_weight --raw-label` |
| The published matrix's **real 63 hand-named columns** take almost nothing off family A's gain, so base shape is not the source of the cohort tripling | +0.0823 → +0.0808 (−1.8%); the 41-column stand-in's 0.013 is in `base_enrichment.csv` | [P] | `base_enrichment_published.csv`, `base_enrichment.csv` | `python -m sgrna.interpret.representation --base-enrichment --published` |
| `d_mechanics`'s outside half is provenance, not non-linearity | 8 flank cols +0.0968; 2 guide-only +0.0019 | [C] | `redescription_provenance.csv` | `python -m sgrna.interpret.redescription --arm provenance` |
| Chi sites are indistinguishable from any GC-matched octamer — the RecBCD repair route is excluded | 41st percentile of 200 controls | [C] | `chi_sites.csv` | `python -m sgrna.controls.family_a_controls --parts chi` |
| The dCas9 screen-wide null is an average: inside its essential genes family A reads +0.27 | +0.2679 (100 kb blocks), +0.2749 (gene-grouped) vs −0.0101 non-essential | [C] | `cui_essentiality_split.csv` | `python -m sgrna.compare.cui_screen --essentiality` |
| **…but that +0.27 is the dCas9 strand rule, not the flank effect** — given a strand indicator family A adds almost nothing to the pooled ranking (Appendix B5, a boundary experiment) | strand alone ρ 0.6621; family A then +0.0069 (blocks), +0.0148 (genes), against +0.2679/+0.2749 without it | [C] | `cui_essential_strand.csv`, `cui_essentiality_split.csv` | `python -m sgrna.compare.cui_screen --strand` |
| …and not a location lookup either: the bare coordinate buys nothing | ρ 0.0394 alone; 0.2596 with the base, against the base's own 0.2805 | [C] | `cui_essential_position.csv`, `cui_essentiality_split.csv` | `python -m sgrna.compare.cui_screen --position` |
| Flank GC — the Finding 2 column — carries nothing in that screen; flank **purine** carries it | −0.0004 vs +0.4045 marginal ρ; coding-strand guides average −5.07 against −1.19 off it | [C] | `cui_essential_marginals.csv`, `cui_strand_summary.csv` | `python -m sgrna.compare.cui_screen --marginals` |
| Flank GC conditioning, re-measured on the curated cohort | +0.176 marginal → +0.221 partial; positive in all 5 strata | [C] | `gc_strata_corr_curated.csv` | `python -m sgrna.interpret.flank_signal --parts gc-corr` |
| Flank GC's marginal association is 3.6× stronger on the curated cohort than the published one | +0.161 vs +0.044 | [C] | — (`gcstrata.py` is pre-pivot) | `python -m sgrna.interpret.flank_signal --parts gc-corr --cohort published` |
| Noshay's quantum columns beat every k-mer encoding of the same bases | +0.0114 | [C] | `curated_ladder.csv` | `python -m sgrna.cohorts.curated_ablation --ladder` |
| Of the 63 hand-named columns, the computable 46 carry the gain | +0.0065 vs −0.0007 | [C] | `handnamed_split.csv` | `python -m sgrna.interpret.representation --handnamed` |
| `d_mechanics`'s gain is flanking DNA, not re-description | +0.097 of +0.099 | [C] | `no_information.csv`, `d_mechanics_reconstructibility.csv` | `python -m sgrna.controls.diagnose --what no-information`; `python -m sgrna.controls.rederive --only d_mechanics_reconstructibility` |
| Model class barely matters, with or without the one real effect | 0.1076 ρ over 8 classes | [C] | `curated_models.csv` | `python -m sgrna.cohorts.curated_ablation --model-classes` |
| Summing the family table would overstate it 3.5× | +0.2030 against +0.7046 | [C] | `curated_joint.csv` | `python -m sgrna.cohorts.curated_ablation --joint` |
| The 300-column budget and the selector are both free | ±0.0005 | [C] | `selector_control.csv`, `selector_comparison.csv` | `python -m sgrna.controls.rederive --only selector_control` |

## Mechanism — what it is, and what it is not

| claim | number | frame | evidence | command |
|---|---|---|---|---|
| **But Wang does not replicate**: family A is +0.102 all-genic and +0.2195 in essential genes under 100 kb blocking, where Cui and Rousset are negative. Unexplained; Wang has 77% between-gene label variance and a base ρ of 0.08 | +0.1019 / +0.2669 / +0.2195 | Wang | `crispri_screen_family_a.csv` | `python -m sgrna.compare.crispri_screens --family-a Wang` |
| **And the whole structure reproduces on the E75 arm ("Rousset", the same experiment as Cui's `fit75`, so not a second screen)**: family A looks worth +0.29 inside essential genes, a single strand indicator beats the 348-column family, and given strand family A adds +0.025 — against +0.157 on a cutting screen | base 0.2141 → +a_flank 0.5043; +strand 0.5878 → +strand+a_flank 0.6124; screen-wide −0.0206 | Rousset | `crispri_screen_family_a.csv` | `python -m sgrna.compare.crispri_screens --family-a Rousset` |
| **The strand result reproduces in Barquist et al.'s processing of Cui's E75 arm** ("Rousset"; the same experiment, not an independent screen) | ρ(coding strand, label) **−0.633** essential / −0.056 non-essential; Cui −0.599 / −0.048; mean label −4.74 on strand vs −0.92 off | Rousset | `crispri_screen_strand.csv` | `python -m sgrna.compare.crispri_screens --strand` |
| **A second, lower ceiling: cross-screen agreement bounds transfer** — and non-essential guides barely agree between screens at all | Cui's two dCas9-expression arms (E18 and E75), same experiment: +0.902 essential / +0.659 non-ess; across libraries and labs: +0.765 to +0.790 essential / +0.271 to +0.298 non-ess | Rousset, Cui, Wang | `crispri_screen_agreement.csv` | `python -m sgrna.compare.crispri_screens --agreement` |
| **Withdrawn as a claim (Appendix B5): the dCas9 screens do not identify whether the flank effect needs a break** — the earlier "specific to cleavage" and its replacement "larger with cleavage" both rest on screens that differ in assay, library, laboratory and phenotype at once | −0.009 vs +0.157; +0.007 inside essential genes with strand controlled (pooled ranking); **qualified: +0.12 among coding-strand guides alone** (rows below) | Cui | `cui_screen.csv`, `cui_essential_strand.csv` | `python -m sgrna.compare.cui_screen` |
| All eleven families with strand supplied: no family beats the strand call | increment over base+strand, genes / blocks: a_flank +0.0148 / +0.0069; g_transcription +0.0568 / +0.0284; every other family within −0.013 to +0.030 | Cui, 5,514 essential-gene guides | `strand_families.csv` | `python -m sgrna.compare.strand --parts families` |
| Dropping |ρ|>0.30 proxy columns does not remove the strand: it is still readable from what remains | out-of-fold AUC after removal: a_flank 0.794, j_periodicity 0.729, d_supercoiling 0.637, g_transcription 0.9997; families without proxy columns 0.49–0.58 | Cui | `strand_families_recoverable.csv` | `python -m sgrna.compare.strand --parts recoverable` |
| Strand held constant (fit within one strand): the gains are all on the coding strand | coding-strand increment over base, blocks / genes: a_flank +0.119 / +0.124; g_transcription +0.198 / +0.267; d_supercoiling +0.081 / +0.154; template strand −0.040 to +0.043 | Cui, 2,535 coding / 2,979 template guides | `strand_families_within.csv` | `python -m sgrna.compare.strand --parts families-within` |
| Coding-strand gain, between genes versus within them (ten seeds) | within-gene Δ over base: a_flank +0.102 (seed sd 0.016, bootstrap [0.081, 0.164]); g_transcription +0.070 [0.044, 0.111]; d_mechanics +0.043 [0.027, 0.074]; d_supercoiling +0.020 [−0.006, 0.054]; i_shape +0.005 [−0.025, 0.051]. Given g_transcription, a_flank adds +0.096 within (sd 0.012) and +0.006 between | Cui, 206 genes with ≥5 coding-strand guides | `strand_within_gene_cui_summary.csv`, `strand_within_gene_cui_joint_summary.csv` | `python -m sgrna.compare.strand --parts within-cui --joint` |
| The E75 arm ("Rousset") reproduces the within-strand result (the same experiment as Cui's, not a replication) | family A over base, coding / template: genes +0.159 / −0.019, 100 kb blocks +0.133 / −0.022; coding, ten seeds: between genes −0.059 → 0.183, within genes 0.083 → 0.196 (Δ +0.114, seed sd 0.013, bootstrap [0.084, 0.189]) | Rousset, 2,074 coding / 2,566 template essential-gene guides; 182 genes | `strand_strata_rousset.csv`, `strand_within_gene_rousset_summary.csv` | `python -m sgrna.compare.strand --parts rousset`; `python -m sgrna.compare.strand --parts within-rousset` |
| Telling the model the wrong strand | base+strand ρ: true 0.646, flipped −0.528, shuffled 0.065, set to mean 0.140; with family A 0.661 / −0.503 / 0.180 / 0.348 | Cui, essential genes, gene-grouped, 3 seeds | `strand_flip.csv` | `python -m sgrna.compare.strand --parts flip` |
| Orientation to the nearest TSS matters | Δ 0.051 [0.020, 0.082] | [P] | `../../RSI-Crispr/Jacky/orientation-controls/results/intergenic_orientation_controls_delta_ci.csv` (row `tss`); our strand audit is `asymmetry_strand_audit.csv` | Jacky's orientation controls |
| **Flanking context crosses kingdoms; the guide-derived columns do not** — built on both sides, transfer flips from ≈0.049 against to positive in both directions | E→H **+0.022**, H→E **+0.017**; flanks alone +0.042 / +0.108; family A on human +0.052 (0.382 → 0.434) | [P] | `../../RSI-Crispr/Jacky/human-boundary/results/family_a_kingdom_transfer.csv`, `folds_human_base_cv.csv`, `folds_human_base_a_cv.csv` (the ladder's two rungs; an aggregate called `human_ladder` was never saved) | Jacky's human-boundary Tier 3 |
| GC residualisation does not rescue shared-column transfer | ≈0.049 under residualise, GC-range and histogram matching | [P] | `../../RSI-Crispr/Jacky/human-boundary/results/gc_matched_transfer.csv` | Jacky's transfer autopsy |
| Human chromatin accessibility at the guide locus is near-null | K562 DNase/ATAC ±500 nt: Δρ +0.001 to +0.005; ρ vs y 0.042 / 0.027. Caveat: K562 may not match the screen's cell type | [P] | `../../RSI-Crispr/Jacky/human-boundary/results/human_chromatin.csv` | Jacky's Tier 2 |
| Not helical phasing: GC skew and periodicity are null | −0.001 alone, +0.0004 on A | [P] | `ablation_jper.csv`, `ablation_jper_joint.csv` | `python -m sgrna.harness.run_ablation --families j_periodicity --tag jper` |
| Not flank shape: shape is a reparameterisation of composition | +0.0009 on top of A (curated; +0.0010 on the published matrix) | [P] | `curated_joint.csv`, `ablation_shape_joint.csv`, `cross_redundancy_*.csv` | `python -m sgrna.harness.run_ablation --families i_shape --tag shape_joint` |
| **`d_supercoiling`'s read-density signal is mostly gene dosage**: four columns from distance-to-*oriC* alone, no sequencing data, match the ChIP track — which is why the untagged control is interchangeable with it | `z_dosage` +0.0112 vs `z_signal_only` +0.0111 vs `z_sc_enrichment` +0.0053, published cohort, 100 kb blocks | [P] | `dissection_supercoiling.csv` | `python -m sgrna.build.make_diagnostic_blocks` |
| Not supercoiling: the no-antibody control predicts as well | control ≈ ChIP | [P] | `dissection_supercoiling.csv` | `python -m sgrna.controls.rederive --only dissection_supercoiling` |
| Not essentiality (Guo confirmed); expression is not null | +0.00004 vs +0.0213 | [C] | `txn_interactions.csv` | `python -m sgrna.cohorts.curated_ablation --interactions` |
| The family's own interaction rationale fails | +0.0004 over parents | [C] | `txn_interactions.csv` | `python -m sgrna.cohorts.curated_ablation --interactions` |
| The label's ceiling, from crisprHAL's own replicate counts | 0.946 [0.944, 0.947] | [C] | `curated_ceiling.json` | `python -m sgrna.cohorts.ceiling --parts measured` |

## Against the state of the art

| claim | number | frame | evidence | command |
|---|---|---|---|---|
| Parity on crisprHAL 2's own eSpCas9 split, their weights | Δρ +0.0021 [−0.0031, +0.0072] | their split | `final_eval_espcas9.csv`, `final_eval_metric_bootstrap.csv` | `python -m sgrna.compare.vs_crisprhal --condition final-compare`; `python -m sgrna.controls.rederive --only final_eval_metric_bootstrap` |
| Parity on their SpCas9 split | +0.005 | their split | `their_split_slicer.csv`, `data_ablation.csv` | `python -m sgrna.controls.rederive --only their_split_slicer` |
| Ahead on our own held-out folds, paired on guides | +0.0115 / +0.0172 / +0.0043 | [C] | `heldout_compare.csv` | `python -m sgrna.compare.vs_crisprhal --condition paired` |
| A search of their own finds nothing better than their published config | 0.6987, 4/5 inner folds | [C] | `matched_search_folds.csv`, `crisprhal_matched_folds.csv` | `colab/crisprhal_tune_matched.py` (Colab; the CSVs were transcribed from its log †) |
| The conservative margin, against their best known configuration | +0.0161 | [C] | `crisprhal_tuned_folds.csv`, `tuning.csv` | `python -m sgrna.tuning.tune` |
| **Holding the architecture fixed, parity is bought with rows** | label +0.0063, which rows +0.0083, how many +0.0302 | their split | `data_ablation.csv` | `python -m sgrna.cohorts.data_ablation` |
| …and the feature set still dominates every data effect | +0.1554 | their split | `data_ablation.csv` | `python -m sgrna.cohorts.data_ablation` |
| Under equal input DNA the ranking depends on the window | theirs +0.012 on 23 nt and +0.009 at 378 nt, ours +0.066 at ±1 kb (five folds each) | [C] | `crisprhal_wide_window.csv`, `crisprhal_wide_window_fcolab.csv`, `crisprhal_wide_window_fcolab2.csv`, `crisprhal_wide_window_folds.csv`, `crisprhal_wide_window_summary.csv` (`seqnet_matched.csv` †) | `colab/crisprhal_wide_window.py`, `python -m sgrna.harness.seqnet` |
| On an assay neither model trained on, theirs transfers better | −0.0258 [−0.0289, −0.0227] | Cui | `cui_final_exam.csv` | `python -m sgrna.compare.cui_screen --final-exam slicer / crisprhal / compare` |
| Data efficiency: the lead is a small-data lead | +0.0133 at 26.9k, +0.0021 at 47.6k | both | `sample_complexity.csv` | `python -m sgrna.compare.sample_complexity` |
| **At inference on one Colab instance, one host thread, crisprHAL 2 (best batch size) over SLICER: CPU 177× / 208× / 344×, GPU 18× / 15× / 20× (input on host) and 12× / 12× / 110× (input on card)** for 1 / 100 / 10,000 guides. The GPU speeds up crisprHAL 2 (20.2 s → 1.08 s at 10,000) and does nothing for SLICER at that size; crisprHAL's cold start 6.0 s (CPU) / 9.9 s (T4), SLICER's 1.4–1.9 s (earlier run) | CPU: 64.9 / 222.2 / 20,202 vs 0.37 / 1.07 / 58.8 ms; T4 host input: 73.4 / 72.0 / 1,077 vs 3.99 / 4.76 / 53.3 ms | — | `inference_onecore.csv`, `inference_colab.csv`, `inference_t4.csv` | `colab/inference_bench_onecore.py`; `python -m sgrna.cost.latency --parts inference` |
| **Per core, on a laptop pinned to one host thread, crisprHAL 2 (best batch size) over SLICER: 329× / 632× / 817×** for 1 / 100 / 10,000 guides. Ten threads help crisprHAL 2 1.4× and SLICER 2.1× at 10,000 guides and hurt both at one guide | crisprHAL 2 15.6 / 130.4 / 11,579 ms; SLICER 0.047 / 0.206 / 14.2 ms; CPU-seconds / wall-seconds 1.0 | — | `inference_onecore_local.csv` | `colab/inference_bench_onecore.py` |
| Cost: ~150× in CPU-seconds, 92× wall-clock, comparable memory (superseded: measured in a quiet session on 24 Nov, 118× in CPU-seconds and 81× wall-clock; `cost_repeats_fit.csv`) | 65.8 s vs ~10,000 s per fold | — | `cost_cpu_seconds.csv`, `cost_flops.csv`, `cost_panel.csv`, `memory_comparison.csv` | `python -m sgrna.cost` |
| Model size: 6,348 fitted leaf values against 790,533 parameters | 125× | — | `complexity.json` | `python -m sgrna.cost.model_size complexity` |
| Attributions are faithful to the fixed model: sensitivity-*n* | +0.928 single columns, +0.969 six-block subsets (permuted attributions ≈ 0) | curated, 100 held-out guides | `interp_sensitivity_n.csv` | `python -m sgrna.interpret.interp_extra --sens` |
| Pairwise terms are 14.7% of the output's variance; single-column terms 70.2% | 0.147 / 0.702 | curated, 400 held-out guides | `interp_interactions.csv`, `interp_interaction_blocks.csv`, `interp_interaction_pairs.csv` | `python -m sgrna.interpret.interp_extra --inter` |
| Flank edits move the prediction through composition: GC +5 points +0.458, −5 points −0.882; GC-preserving swaps +0.003 to +0.016 | bootstrap 95% CIs in file | curated, 1,500 held-out guides | `interp_counterfactual.csv` | `python -m sgrna.interpret.interp_extra --cf` |
| **SLICER against Noshay's own model, on Noshay's data** | iRF 0.4785 against SLICER 0.6060 | [P] | `noshay_replication.csv`, `ablation_main.csv`, `shift_null.csv` | `python -m sgrna.compare.noshay_replicate` |

## Where it stops

| claim | number | frame | evidence | command |
|---|---|---|---|---|
| Transfer across nuclease, and across nuclease + organism | 88% / 91% retained | external | `transfer_model.csv`, `transfer.csv` | `python -m sgrna.compare.transfer` |
| No transfer across kingdoms, in either direction | −0.039 [−0.054, −0.024]; −0.006 [−0.023, +0.011] | both | `kingdom_transfer.csv`, `kingdom_transfer_boot.csv` | `python -m sgrna.compare.human_baseline --parts kingdom kingdom-bootstrap` |
| The human baseline reproduces at the teammate's settings | 0.5076 against 0.472 quoted | human | `human_baseline.csv`, `human_baseline_knobs.csv` | `python -m sgrna.compare.human_baseline` |
| ~~The human signal is far less additive than *E. coli*'s~~ **withdrawn 19 November**: measured on row-wise folds that split duplicate guides; with twins kept together it is 88.8% against 95.6% | ridge/boosting 63.7% vs 95.6% (row-wise folds) | human | `human_additivity.csv` | `python -m sgrna.compare.human_baseline --parts additivity` |
| **Sealed-gene drop in family A's gain, over 15 sealing draws** (not shown to be adaptive overfitting) | **+0.0087 [+0.0020, +0.0153]**, falling on the arm-to-arm difference: family A is +0.1647 in pool CV and **+0.1560** on sealed genes; with whole genes held out inside the pool as well (first 8 sealings) +0.0081, so random-row pool folds account for only +0.0007 | [C] | `sealed_holdout.csv`, `sealed_holdout_gene.csv` | `python -m sgrna.controls.sealed --parts holdout --seal-seeds 20261016 ... 20261030`; `python -m sgrna.controls.sealed --parts gene --max-draws 8` |
| The 3.2× inflation is the row population, not the label | label +0.004; population ≈ +0.070 | both | `why_inflation.csv`, `ablation_label_published.csv`, `ablation_label_curated.csv` | `python -m sgrna.controls.family_a_controls --parts rows` |
| **and the mechanism is a steeper slope outside the 1,222-gene alphabetical block (aaeA to leuC) in which the published matrix's rows lie** | β +5.4 inside, +17.0 outside | [C] | `curation_variance.csv` | `python -m sgrna.cohorts.curation --parts variance` |
| Gene identity explains more of the curated label than the published one | 0.844 → 0.713 within-gene share | both | `cohort_groups.csv` | `python -m sgrna.harness.cohort_groups` |

---

## † Files whose producer was lost, and what re-deriving them found

Thirteen files cited above were written by code no longer in `src/` or `colab/`.
`python -m sgrna.controls.rederive` now reproduces twelve of them from committed code and
**compares rather than overwrites** — re-derivation is a check, and the stored
file stays authoritative. Verdicts in `results/rederivation_check.csv`, outputs
in `results/rederived/`:

| file | verdict |
|---|---|
| `grouped_retention.csv` | **exact** |
| `selector_control.csv` | **exact** |
| `final_eval_metric_bootstrap.csv` | **exact** point estimates; CIs differ by bootstrap noise only |
| `portability_364.csv` | agrees, 0.2% |
| `kmer_order_noselect.csv` | agrees, 0.2% |
| `dissection_supercoiling.csv` | agrees on the claim (control ≥ signal), 1.2%; the stored arm names and `bin100k` grouping could not be reconstructed |
| `their_split_slicer.csv` | agrees, ρ 0.6997 against 0.6985; all metrics within 0.004 except a top-10 statistic |
| `d_mechanics_reconstructibility.csv` | agrees in distribution, median 0.832 against 0.845; the file is keyless so rows cannot be aligned |
| `additivity_reduced.csv` | **two of four arms exact, two wrong** — see below |
| `abund_local_vs_mappability.csv` | **not reproducible**: the definition of its `uniq` columns is lost. Five candidates were tried and none comes near the stored values (its `control.w10000` / `uniq.w10000` cell is −0.694; windowed exact-20-mer uniqueness gives −0.08, mean `unique_in_genome` −0.06, mean `n_genome_copies` +0.06, guide density −0.05, and the same at 2 kb). The stored quantity grows in magnitude with window width, which suggests it was derived from read data rather than from the genome — do not spend more time guessing it |
| `seqnet_matched.csv` | **exact on all four arms, all 20 folds** (f100 0.5203, f250 0.5247, f500 0.5214, f1000 0.5190 — zero difference to four decimals). `python -m sgrna.harness.seqnet --flank {100,250,500,1000} --out results/rederived/seqnet_matched.csv` |
| `matched_search_folds.csv` | **agrees to 0.0001** — it is a join of four local files, not a GPU artefact, and is now re-derived as one |
| `crisprhal_matched_folds.csv` | the only genuine GPU artefact. `colab/crisprhal_tune_matched.py` now writes it under exactly that name with the `is_published_config` flag, so a rerun reproduces it — but the rerun needs a T4 |

**The exercise paid for itself.** `additivity_reduced.csv`'s two reduced arms
re-derive to within 0.0007 (and its XGBoost column bit-identically), but its two
full-matrix arms do not: the stored "ridge on the raw 6,232 collapses to 0.423"
holds only at `alpha=1`, and a penalty search gives **0.4945**. That claim is
withdrawn in the report. A number nobody can regenerate is a number nobody can
argue with — which is exactly what happened to `human_transfer.csv`.

## Reproducing any row

Every command above assumes `pip install -e .` and the data layout in
[`../README.md`](../README.md). Seeds are fixed (41–45) and every table here was
produced by the committed code in [`../src/sgrna`](../src/sgrna); the two
GPU-dependent rows run on free Colab from [`../colab`](../colab). The harness
check is `ablation_main.csv`'s baseline row, which must reproduce R² 0.2937 /
ρ 0.5278 to four decimal places before any other number is trusted.
