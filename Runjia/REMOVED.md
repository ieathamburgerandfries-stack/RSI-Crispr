# What was removed, and why

Everything here is recoverable. The state before the 22 November 2026 consolidation is the git tag
`pre-cleanup-2026-11-22`; restore any item with

```bash
git checkout pre-cleanup-2026-11-22 -- Runjia/<path>
```

This file exists so that nobody has to read history to find out what is missing: each entry says what the
item was, what it found, and why it is gone.

## The test that was applied

A result or a module stays if **any one** of these holds, and goes only if **none** does:

1. `PROGRESS_REPORT.md`, `results/RESULTS.md` or a README names it, or names a result file it writes;
2. `sgrna/post_freeze/paper/verify_numbers.py` (the audit) or `figures.py` reads it;
3. a module that stays imports it, launches it, or reads a file it writes — including files whose names are
   built at run time (`mix_configs_{kind}.json`, `folds_<tag>.csv`), which a plain text search cannot see;
4. a teammate's document cites it as evidence (`Jacky/EXPERIMENT_CATALOG.md` indexes much of `results/`).

**The result was much smaller than expected, and that is itself a finding.** Of 151 modules and 253 tracked
result files, only the entries below failed every test. The code base is not padded with dead experiments:
it is roughly a hundred one-question scripts, each of which is the provenance of a claim the report makes,
and the problem was navigability rather than bulk. That is why the rest of the consolidation is a
reorganisation (see `CODE_MAP.md`) rather than a deletion. One further check confirmed it: across all 150
modules only one 8-line function is duplicated, so there is no copy-paste to merge away.

If you want the repository smaller than this, the lever is not more deletion of orphans: it is deciding
which *claims* leave the report, because each claim carries its code and results with it. See "What
trimming the report would free up" at the end of `CODE_MAP.md`.

## Removed

| item | what it was | why it went | recover |
|---|---|---|---|
| `catboost_info/` (4 files, 155 KB) | CatBoost's own training log, written into the working directory whenever a CatBoost model is fitted there | tracked by mistake; contains no result. Now in `Runjia/.gitignore` | `Runjia/catboost_info` |
| `sgrna/interpret/decision_scale.py` | A Gaussian-copula translation of ρ into "how good is the guide you pick out of *k*" (draw a true quality and a prediction with Pearson `r = 2 sin(πρ/6)`, let the model choose among *k*, read off the percentile of the choice; chance is the 50th, an oracle that always picks the best of ten averages the 91st). **Never run**: it has no result file, and no document, module or audit check refers to it | an unfinished idea, not a result. The measured version of the same readout is already in the report (`panel.py`'s expected pick percentile), so the idealised one added nothing. If a ρ-to-decision translation is wanted for the paper, the formula above is the whole method | `Runjia/sgrna/interpret/decision_scale.py` |
| `colab/cell3_revised.py`, `colab/cell3_v3.py` | Two successive drafts of cell 3 of `colab/crisprhal_wide_window.py`, each a "paste over the previous cell 3" patch | **folded into `crisprhal_wide_window.py`**, whose cell 3 is now v3's. v1 trained through crisprHAL's own `train()`/`predict()`, which swallow their exceptions, so a failed 2,023 nt fit scored an untrained network (ρ 0.0434 in 0.36 min); v2 added failure checks and a batch fallback but still trained through those wrappers, so the swallowed exception hid the failure from its checks; v3 calls Keras directly. The report's wide-window table (batch 256) is v3's, and `results/RESULTS.md` named v1 as its producer, which was wrong and is now right. Keeping the two superseded drafts next to the working one invited running the one that scores garbage | `Runjia/colab/cell3_revised.py`, `Runjia/colab/cell3_v3.py` |

## Kept although they look removable

Recorded so the next sweep does not repeat the reasoning.

| item | why it looked removable | why it stays |
|---|---|---|
| `sgrna/post_freeze/shape/geom_importance.py` and `geom_importance.csv` | no document names either | the report's Finding 8 states its conclusion (item 4) without naming it — and in checking this I found that the figure the report quoted, R² 0.963, had **no producer and does not reproduce** (the same analysis gives 0.968 by ridge, 0.937 by boosting for that column, and 0.86–0.97 across the five). The module now computes it and the audit checks it; the report was corrected |
| `baseline_ecoli_sgRNA_model_swappable.py`, and the fork's top-level `USE THIS!! FINAL_baseline_…py` | referenced by no code | they are the teammates' original baseline, and the report names it as where `N_FEATURES = 300` and the champion settings were *inherited from* (Part 4's decision register). Provenance, not clutter |
| `results/rederived/` and `results/rederivation_check.csv` | duplicates of seven `results/*.csv` files | the originals' producers were lost; `controls/rederive.py` rebuilt them and this is the evidence that they agree |
| `results/RESULTS_archive_2026-10-10.md` | superseded by `RESULTS.md` | the report cites it by name as the record of what was believed on 10 October, including the contaminated "+0.104 under arc5" figure |
| `results/folds_human_*.csv`, `folds_humangc_*.csv`, `folds_base_plus_{K562_ATAC,K562_DNase,dnase_atac}.csv` (12 files, each under 1 KB) | no document of mine names them, no module reads them, and for half of them even the per-arm means do not appear in the report | they are the harness's per-fold records for the human experiments that `Jacky/EXPERIMENT_CATALOG.md` lists as load-bearing (the human ladder, the chromatin tier, the GC+GC² ablation). Not part of this report's claims, but a teammate's evidence and not mine to remove; each is tiny |
| `mix_*_stage1.csv`, `mix_configs_*.json`, `tune_stage1.csv` | 100–270 KB search logs nobody cites | `mix_ecoli`/`tune_ecoli` read them by names built at run time, and `e2e_bench` times the "mixed ensemble" row of the report's cost table from them |
| `sgrna/post_freeze/tuning/mlp_worker.py`, `vs_crisprhal/vs_crisprhal_wt_hal.py` | imported by nothing | launched as subprocesses — the first to keep PyTorch out of the XGBoost process (they hang together on macOS), the second inside the TensorFlow environment |
| `sgrna/compare/crisprhal_tune.py` | never produced a result | it is the starting point for open question 1 in the report's Part 4 register (re-tune crisprHAL 2 on its own splits, ~40 GPU-hours) |
