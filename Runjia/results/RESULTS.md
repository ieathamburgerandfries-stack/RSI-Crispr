# Results — claim to evidence

This file is an **index**, not a narrative. For every claim the progress report
makes, it names the file that backs it and the command that produced it, so any
number in the report can be checked without reading the report.

The narrative lives in [`../docs/PROGRESS_REPORT.md`](../docs/PROGRESS_REPORT.md);
the chronology, including claims later withdrawn, lives in
the shared repository's git history.

**What is indexed:** results a claim rests on. That includes nulls when the null
*is* the claim — the dCas9 screen, the periodicity family, the shift control —
because a ruled-out mechanism is a result.

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
| Flanking DNA composition is the largest effect measured here | +0.164 ρ | [C] | `ablation_curated.csv` | `python -m sgrna.curated_ablation --all` |
| …and +0.080 on the published matrix, so the effect is not an artefact of the pivot | +0.080 ρ | [P] | `ablation_main.csv` | `python -m sgrna.run_ablation` |
| It survives holding out whole genes, losing only the grouping tax | 6% tax | [C] | `ablation_curated_gene.csv`, `grouped_retention.csv` | `python -m sgrna.curated_ablation --all --grouping gene`; `python -m sgrna.rederive --only grouped_retention` |
| It is spatially local: a texture-preserving circular shift destroys it | 8% at 1 kb, 0 at 10 kb | [P] | `shift_null.csv` | `python -m sgrna.shiftnull` |
| It is a smooth gradient, not a motif: k-mer order buys nothing | tetramers −0.0019 | [C] | `curated_ladder.csv`, `kmer_order_noselect.csv` | `python -m sgrna.curated_ladder`; `python -m sgrna.rederive --only kmer_order_noselect` |
| The window widths are not where the gain lives | +0.1634 vs +0.1638 | [C] | `window_robustness.csv` | `python -m sgrna.window_robustness` |
| It is not target GC in disguise: conditioning raises it | +0.044 → +0.082 | [P] | `gc_strata_corr.csv`, `gc_strata_ablation.csv` | `python -m sgrna.gcstrata` |
| It is not gene ranking: it holds inside genes | +0.061 within-gene | [P] | `ablation_grouped.csv` | `python -m sgrna.run_ablation --grouping gene` |
| It is not read depth or mappability | +0.078 → +0.051 | [P] | `ablation_abund_split.csv` (`abund_local_vs_mappability.csv` †) | `python -m sgrna.run_ablation --tag abund_split` |

## The representation

| claim | number | frame | evidence | command |
|---|---|---|---|---|
| 99% of the 6,232 published columns regenerate from a 20-mer | 5,853 of 5,887 `V` columns decoded | [P] | `portability_audit.csv` | `python -m sgrna.expand --build` |
| 412 columns computable from any guide match all 6,232 | 0.00003 ρ | [P] | `portability.csv`, `portability_364.csv` | `python -m sgrna.portability`; `python -m sgrna.rederive --only portability_364` |
| …but no reduced set is non-inferior on the curated cohort | −0.0118 / −0.0143 | [C] | `curated_ladder.csv` | `python -m sgrna.curated_ladder --no-select` |
| A deterministic function of the guide adds no information, but does add performance | +0.011 to +0.019 | [C] | `no_information.csv`, `base_enrichment.csv` | `python -m sgrna.no_information`, `python -m sgrna.base_enrichment` |
| …but **random** functions of the same shape add nothing | −0.0004 to −0.0015 at 4–256 tables | [C] | `redescription_random.csv` | `python -m sgrna.redescription --arm random` |
| **Sorted by column rather than by family**: what the guide determines is worth +0.008, what it does not +0.201 | 24× | [C] | `redescription_taxonomy.csv`, `redescription_columns.csv` | `python -m sgrna.redescription --arm taxonomy` |
| Every family split the same way: the two halves differ in all six | see table | [C] | `redescription_family_split.csv` | `python -m sgrna.redescription --arm families` |
| One guide-determined but non-linear column, `eng.dg.guide_selffold`, carries `b_energy`'s whole gain | +0.0116 of +0.0181 | [C] | `redescription_family_split.csv`, `redescription_columns.csv` | `python -m sgrna.redescription --arm families` |
| The eleven never-tuned XGBoost knobs are worth nothing; two do harm | +0.0069 best, −0.0080 worst | [C] | `xgb_knobs.csv` | `python -m sgrna.xgb_knobs` |
| …and the one apparent winner is capacity, not grow policy: `max_depth` 6 beats 4 | +0.0063 [+0.0053, +0.0073], 15/15 folds | [C] | `xgb_knobs_verify.csv` | `python -m sgrna.xgb_knobs --verify` |
| The folding gain is the **real structure**: position-shuffled folds are a null | +0.0119 vs −0.0002 over 3 permutations | [C] | `redescription_folds.csv` | `python -m sgrna.redescription --arm folds` |
| **Difficulty without meaning buys nothing**: random non-linear functions of the guide, matched on recoverability to the real fold energy, are null at every interaction order | +0.0001 to +0.0007 (R² 1.00 → −0.008) vs +0.0117 for the real column at R² 0.316 | [C] | `redescription_nonlinear.csv` | `python -m sgrna.redescription --arm nonlinear` |
| The published matrix's **real 63 hand-named columns** take almost nothing off family A's gain, so base shape is not the source of the cohort tripling | +0.0823 → +0.0808 (−1.8%), against 0.013 for the 41-column stand-in | [P] | `base_enrichment_published.csv` | `python -m sgrna.base_enrichment --published` |
| `d_mechanics`'s outside half is provenance, not non-linearity | 8 flank cols +0.0968; 2 guide-only +0.0019 | [C] | `redescription_provenance.csv` | `python -m sgrna.redescription --arm provenance` |
| Chi sites are indistinguishable from any GC-matched octamer — the RecBCD repair route is excluded | 41st percentile of 200 controls | [C] | `chi_sites.csv` | `python -m sgrna.chi_sites` |
| The dCas9 screen-wide null is an average: inside its essential genes family A reads +0.27 | +0.2679 (100 kb blocks), +0.2749 (gene-grouped) vs −0.0101 non-essential | [C] | `cui_essentiality_split.csv` | `python -m sgrna.cui_screen --essentiality` |
| **…but that +0.27 is the dCas9 strand rule, not the flank effect** — given a strand indicator family A adds nothing, so the cleavage localisation holds | strand alone ρ 0.6621; family A then +0.0069 (blocks), +0.0148 (genes) | [C] | `cui_essential_strand.csv` | `python -m sgrna.cui_screen --strand` |
| …and not a location lookup either: the bare coordinate buys nothing | ρ 0.0394 alone, −0.021 on the base | [C] | `cui_essential_position.csv` | `python -m sgrna.cui_screen --position` |
| Flank GC — the Finding 2 column — carries nothing in that screen; flank **purine** carries it | −0.0004 vs +0.4045 marginal ρ | [C] | `cui_essential_strand.csv` | `python -m sgrna.cui_screen --strand` |
| Flank GC conditioning, re-measured on the curated cohort | +0.176 marginal → +0.221 partial; positive in all 5 strata | [C] | `gc_strata_corr_curated.csv` | `python -m sgrna.gcstrata --corr --cohort curated` |
| Flank GC's marginal association is 3.6× stronger on the curated cohort than the published one | +0.161 vs +0.044 | [C] | — (`gcstrata.py` is pre-pivot) | `python -m sgrna.gcstrata --corr` |
| Noshay's quantum columns beat every k-mer encoding of the same bases | +0.0114 | [C] | `curated_ladder.csv` | `python -m sgrna.curated_ladder` |
| Of the 63 hand-named columns, the computable 46 carry the gain | +0.0065 vs −0.0007 | [C] | `handnamed_split.csv` | `python -m sgrna.handnamed_split` |
| `d_mechanics`'s gain is flanking DNA, not re-description | +0.097 of +0.099 | [C] | `no_information.csv`, `d_mechanics_reconstructibility.csv` | `python -m sgrna.no_information`; `python -m sgrna.rederive --only d_mechanics_reconstructibility` |
| Model class barely matters, with or without the one real effect | 0.1076 ρ over 8 classes | [C] | `curated_models.csv` | `python -m sgrna.curated_models` |
| Summing the family table would overstate it 3.5× | +0.2030 against +0.7046 | [C] | `curated_joint.csv` | `python -m sgrna.curated_joint` |
| The 300-column budget and the selector are both free | ±0.0005 | [C] | `selector_control.csv`, `selector_comparison.csv` | `python -m sgrna.rederive --only selector_control` |

## Mechanism — what it is, and what it is not

| claim | number | frame | evidence | command |
|---|---|---|---|---|
| **The effect is specific to cleavage, not Cas9 binding** | −0.009 vs +0.157; +0.007 inside essential genes with strand controlled | Cui | `cui_screen.csv`, `cui_essential_strand.csv` | `python -m sgrna.cui_screen` |
| Orientation to the nearest TSS matters | Δ 0.051 [0.020, 0.082] | [P] | `../../RSI-Crispr/Jacky/orientation-controls/results/intergenic_orientation_controls_delta_ci.csv` (row `tss`); our strand audit is `asymmetry_strand_audit.csv` | Jacky's orientation controls |
| Not helical phasing: GC skew and periodicity are null | −0.001 alone, +0.0004 on A | [P] | `ablation_jper.csv`, `ablation_jper_joint.csv` | `python -m sgrna.run_ablation --families j_periodicity --tag jper` |
| Not flank shape: shape is a reparameterisation of composition | +0.0007 on top of A | [P] | `ablation_shape_joint.csv`, `cross_redundancy_*.csv` | `python -m sgrna.run_ablation --families i_shape --tag shape_joint` |
| Not supercoiling: the no-antibody control predicts as well | control ≈ ChIP | [P] | `dissection_supercoiling.csv` | `python -m sgrna.rederive --only dissection_supercoiling` |
| Not essentiality (Guo confirmed); expression is not null | +0.00004 vs +0.0213 | [C] | `txn_interactions.csv` | `python -m sgrna.txn_interactions` |
| The family's own interaction rationale fails | +0.0004 over parents | [C] | `txn_interactions.csv` | `python -m sgrna.txn_interactions` |
| The label's ceiling, from crisprHAL's own replicate counts | 0.946 [0.944, 0.948] | [C] | `curated_ceiling.json` | `python -m sgrna.curated_ceiling` |

## Against the state of the art

| claim | number | frame | evidence | command |
|---|---|---|---|---|
| Parity on crisprHAL 2's own eSpCas9 split, their weights | Δρ +0.0021 [−0.0031, +0.0072] | their split | `final_eval_espcas9.csv`, `final_eval_metric_bootstrap.csv` | `python -m sgrna.final_eval --arm compare`; `python -m sgrna.rederive --only final_eval_metric_bootstrap` |
| Parity on their SpCas9 split | +0.005 | their split | `their_split_slicer.csv`, `data_ablation.csv` | `python -m sgrna.rederive --only their_split_slicer` |
| Ahead on our own held-out folds, paired on guides | +0.0115 / +0.0172 / +0.0043 | [C] | `heldout_compare.csv` | `python -m sgrna.heldout_compare` |
| A search of their own finds nothing better than their published config | 0.6987, 4/5 inner folds | [C] | `matched_search_folds.csv`, `crisprhal_matched_folds.csv` | `colab/crisprhal_tune_matched.py` (Colab; the CSVs were transcribed from its log †) |
| The conservative margin, against their best known configuration | +0.0161 | [C] | `crisprhal_tuned_folds.csv`, `tuning.csv` | `python -m sgrna.tune` |
| **Holding the architecture fixed, parity is bought with rows** | label +0.0063, which rows +0.0083, how many +0.0302 | their split | `data_ablation.csv` | `python -m sgrna.data_ablation` |
| …and the feature set still dominates every data effect | +0.1554 | their split | `data_ablation.csv` | `python -m sgrna.data_ablation` |
| Under equal input DNA the ranking depends on the window | theirs +0.023 at 378 nt, ours +0.065 at ±1 kb | [C] | `crisprhal_wide_window.csv` (`seqnet_matched.csv` †) | `colab/crisprhal_wide_window.py`, `python -m sgrna.seqnet` |
| On an assay neither model trained on, theirs transfers better | −0.0258 [−0.0289, −0.0227] | Cui | `cui_final_exam.csv` | `python -m sgrna.cui_final_exam` |
| Data efficiency: the lead is a small-data lead | +0.0133 at 26.9k, +0.0021 at 47.6k | both | `sample_complexity.csv`, `learning_curve.csv` | `python -m sgrna.sample_complexity` |
| Cost: ~150× in CPU-seconds, 92× wall-clock, comparable memory | 65.8 s vs ~10,000 s per fold | — | `cost_cpu_seconds.csv`, `cost_flops.csv`, `cost_panel.csv`, `memory_comparison.csv` | `python -m sgrna.cost` |
| Model size: 6,348 fitted leaf values against 790,533 parameters | 125× | — | `complexity.json` | `python -m sgrna.complexity` |
| **SLICER against Noshay's own model, on Noshay's data** | iRF 0.4785 against SLICER 0.6060 | [P] | `noshay_replication.csv`, `ablation_main.csv`, `shift_null.csv` | `python -m sgrna.noshay_replicate` |

## Where it stops

| claim | number | frame | evidence | command |
|---|---|---|---|---|
| Transfer across nuclease, and across nuclease + organism | 88% / 91% retained | external | `transfer_model.csv`, `transfer.csv` | `python -m sgrna.transfer` |
| No transfer across kingdoms, in either direction | −0.039 [−0.054, −0.024]; −0.006 [−0.023, +0.011] | both | `kingdom_transfer.csv`, `kingdom_transfer_boot.csv` | `python -m sgrna.kingdom_transfer --bootstrap` |
| The human baseline reproduces at the teammate's settings | 0.5076 against 0.472 quoted | human | `human_baseline.csv`, `human_baseline_knobs.csv` | `python -m sgrna.human_baseline` |
| The human signal is far less additive than *E. coli*'s | ridge/boosting 63.7% vs 95.6% | human | `human_additivity.csv` | `python -m sgrna.human_baseline --additivity` |
| **Adaptive overfitting, measured over 15 sealing draws** | **+0.0087 [+0.0020, +0.0153]**, falling on the arm-to-arm difference: family A is +0.1647 in pool CV and **+0.1560** on sealed genes | [C] | `sealed_holdout.csv` | `python -m sgrna.sealed_holdout --seal-seeds 20261016 ... 20261030` |
| The 3.3× inflation is the row population, not the label | label +0.004; population ≈ +0.070 | both | `why_inflation.csv`, `ablation_label_published.csv`, `ablation_label_curated.csv` | `python -m sgrna.why_inflation` |
| **and the mechanism is a steeper slope outside Guo's 1,222-gene panel** | β +5.4 inside, +17.0 outside | [C] | `curation_variance.csv` | `python -m sgrna.curation_variance` |
| Gene identity explains more of the curated label than the published one | 0.844 → 0.713 within-gene share | both | `cohort_groups.csv` | `python -m sgrna.cohort_groups` |

---

## † Files whose producer was lost, and what re-deriving them found

Thirteen files cited above were written by code no longer in `src/` or `colab/`.
`python -m sgrna.rederive` now reproduces twelve of them from committed code and
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
| `seqnet_matched.csv` | **exact on all four arms, all 20 folds** (f100 0.5203, f250 0.5247, f500 0.5214, f1000 0.5190 — zero difference to four decimals). `python -m sgrna.seqnet --flank {100,250,500,1000} --out results/rederived/seqnet_matched.csv` |
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
