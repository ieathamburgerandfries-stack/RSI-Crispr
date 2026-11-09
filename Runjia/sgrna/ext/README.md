# `sgrna.ext` — extension experiments (fork only)

Code for Findings 8 and 9 of `PROGRESS_REPORT.md` and the record in `EXTENSION_LOG.md`. Nothing in the
frozen pipeline is edited; these modules import it. Results go to `results/ext/`.

Run with the project's data visible (the fork root needs `data/` and `external_data/`, e.g. as symlinks to the
project folder) and `PYTHONPATH=<fork>/Runjia`, `EXT_OUT=<fork>/Runjia/results/ext`.

| module | what it does |
|---|---|
| `dnashape.py` | parses the DNAshapeR pentamer table (`external_data/dnashaper_tables/TableCompiler.cpp`) into named columns by source (mc / xrc / md) and kind; verification checks |
| `geom.py` | per-guide shape tracks, rigid-body 3D path (Lu & Olson mid-step convention), the `mean` / `vec` / `geom` blocks; shuffled-flank and twist-recentred controls; plain and dinucleotide composition references |
| `shape_build.py` | builds and caches the blocks for the curated cohort |
| `shape_ablate.py` | gene-grouped ablations (`--stage 1` … `9`; see the docstring and the log for what each stage is) |
| `analyze.py` | fold-paired differences between arms |
| `redundancy.py`, `geom_importance.py` | how much of each block family A predicts; which geometry columns the model uses |
| `ridge_check.py` | the shape-on-A comparison with ridge regression instead of boosted trees |
| `esp_replicate.py` | the same comparison on crisprHAL's eSpCas9 screen (independent cohort) |
| `cut_within_gene.py` | the cutting-screen within-gene decomposition (uses `sgrna.strand_within_gene.run`) |
| `nuclease_desc.py` | nuclease descriptor groups: substitutions, 5F9R structure distances, PAM profile, activity scale |
| `nuclease_model.py` | the Kim et al. 2020 experiments (`--stage 1` … `9`) |
| `lono_summary.py` | leave-one-nuclease-out summaries with bootstrap over nuclease and PAM-class decomposition |
| `pam_variants.py`, `pam_learn.py`, `pam_analyze.py` | 770 SpCas9 PI-domain variants (Silverstein 2025): descriptor groups (identity, composition, structure x composition, structure alone), leave-one-campaign-out and random folds, three model classes, within-fold metrics, bootstrap summaries |
| `pam_predict.py` | the PAM profile of each Kim nuclease predicted from its six PI residues, in the layout of `nuclease_desc.pam_group` (used by `nuclease_model --stage 10`) |
| `enzyme_contacts.py` | position-resolved contact profiles of each enzyme (substituted residues mapped through 5F9R onto target positions), the exact-permutation pairing test, whole-protein composition distances |
| `tune_ecoli.py`, `tune_analyze.py` | joint hyperparameter search for the E. coli model judged on held-out genes: 200 random draws on 70% of genes, re-scoring on fresh partitions, confirmation on the other 30%, the eSpCas9 screen as external check, and the whole-cohort standard protocol (`EXTENSION_LOG.md` section 11) |
| `mix_ecoli.py`, `mlp_worker.py` | LightGBM / CatBoost / extra-trees / neural network / ridge searches on the same DEV genes, blends on the held-out genes (the network runs in its own process: PyTorch next to XGBoost stalled silently) |
| `vs_crisprhal.py`, `vs_crisprhal_wt_hal.py`, `cost_bench.py` | the tuned and mixed models against crisprHAL 2 on its own splits and on our folds, its WT-SpCas9 weights on its test file, and one-thread cost of every model including featurisation |
| `interpret.py` | interaction-order curve, gain by column kind and by position, within- versus between-gene gain, data scaling |
| `human_flank.py`, `human_shape.py`, `human_twin_check.py` | human side result (stopped; see `EXTENSION_LOG.md` section 10): hg38 location of the human guides, the shape ablation on human, and the twin-grouped re-run of the human baseline |

Data downloaded for this work (publisher / RCSB / GitHub text, nothing executed): `external_data/kim2020_cas9_variants/`
(Supplementary Tables of Kim et al. 2020), `external_data/pdb_cas9/5F9R.pdb`, `external_data/dnashaper_tables/`,
`external_data/pammla_silverstein2025/` (one CSV and the licence, 729 KB), `external_data/hg38/hg38.fa.gz` (UCSC, 983,659,424 bytes; used only for the human side result).
`external_data/deepDNAshape_src/` was cloned and **not run**.
