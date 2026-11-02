# Family A, and why *E. coli* → human fails

Jacky’s entry point for the human chapter. E. coli science is largely frozen;
this folder tracks what explains the kingdom boundary.

**Lead findings (Tier 1, located n = 16,490):**

| claim | number |
|---|---|
| Human regenerated base CV ρ | **0.382** |
| Human base + Family A CV ρ | **0.434** (**+0.052**) |
| Family A alone | **0.341** |
| Shared-column *E. coli*→human transfer | **−0.047** |
| Transfer after residualising GC / GC² | **−0.048** (null — GC rule alone does not fix it) |
| Error ↔ spacer GC (Spearman) | **−0.399** |
| GC+GC² as human features | **+0.00** vs base (ablation null) |

Artefacts under `results/`. Runner: `scripts/run_human_boundary.py`.

---

## 1. What is Family A?

**Family A (`a_flank`) = features from DNA *around* the cut site, not inside the
20 nt guide.**

The published Noshay matrix only describes the protospacer (the 20 letters the
guide matches). Family A adds:

| block | what |
|---|---|
| Near-cut sequence | one-hot bases just upstream of the spacer, the PAM’s N, and ~15 nt downstream of the PAM |
| Near-cut QCT | same quantum k-mer tables as the paper, applied to those positions |
| Wide windows | composition (GC, purine, longest run) and windowed quantum summaries at **50 / 250 / 500 / 1000 nt** on both sides |

**Why it mattered in bacteria:** it was the only large new signal
(~**+0.08 ρ** on the published *E. coli* label, ~**+0.16 ρ** on the curated
33k label). Most of the wide-window signal is flank GC / composition — a
smooth gradient, not a motif. The head-to-head with crisprHAL is decided by
the **250–1000 nt** reach their network never sees.

**Why human never had it:** Family A needs each guide **located on a reference
genome** so flanks can be cut. The human arm of the transfer used the shared
6,216 matrix columns only — **no hg38 locate, no flanks**.

Plain English: Family A asks *“what’s the DNA neighborhood like?”* The old
matrix only asked *“what’s the 20-letter word?”*

---

## 2. Checklist — explain *E. coli* → human failure

Status key: **DONE** / **PARTIAL** / **BLOCKED** / **TODO**

### A. Close E. coli loose ends

| # | item | status | notes |
|---|---|---|---|
| A1 | Tell Runjia human ρ 0.472 settings | **DONE (this folder)** | `n_features=400`, `n_estimators=1000`; old personal XGBoost, not SLICER harness — see `HUMAN_BASELINE_SETTINGS.md` |
| A2 | E. coli orientation / pivot / parity | **DONE** (Runjia + Jacky) | Not Jacky’s queue unless asked |

### B. Already measured (reuse; don’t re-run unless refreshing)

| # | item | status | artefact |
|---|---|---|---|
| B1 | Shared-column transfer both ways | **DONE** | *E. coli*→human **−0.048** [−0.062, −0.033]; human→*E. coli* **−0.056** [−0.073, −0.038] — `results/cross_kingdom_bootstrap.json` |
| B2 | Within-kingdom anchors | **DONE** | *E. coli*→*E. coli* ~0.51; human→human ~0.38 (same shared cols) |
| B3 | Spacer GC shape differs | **DONE** | *E. coli* linear ρ **−0.201**, vertex below data; human linear ~0, quadratic peak at GC≈**0.57** — `results/gc_quadratic_proper.json` + `results/human_gc_shape.json` |
| B4 | Hypothesis written | **DONE** | Not “no mechanism” — **mechanism replaced by a different shape**; chromatin/repair likely outside these features |

### C. Tier 1 — next science

| # | item | status | result |
|---|---|---|---|
| C1 | Obtain human feature matrix | **DONE** | `Runjia/data/raw/human_feature_matrix.csv` — 17,421 rows |
| C2 | Decode / share columns with *E. coli* recipe | **DONE** | QCT electron decode → 17,421 spacers; 13,138 unique; V/QCT maps rebuilt from *E. coli* matrix |
| C3 | Locate human guides on **hg38** | **DONE** | Unique locate 12,222 / 13,138 wanted; multi 893; missing 23 → **16,490** usable rows — `results/human_locate_audit.json` |
| C4 | Build **Family A on human** | **DONE** | `a_flank` on located guides (348 cols); cache under `Runjia/data/interim/human_boundary/` |
| C5 | Human-only ladder | **DONE** | regenerated base **0.382** → +Family A **0.434** (+0.052); A-only 0.341; shared published cols 0.402 — `results/human_ladder.csv` |
| C6 | Transfer autopsy (“wrong GC rule”) | **DONE** | Transfer **−0.047**; after GC² residualise **−0.048** (null); error↔GC ρ **−0.399** — `results/transfer_autopsy.json` |

### D. Tier 2 — mechanism sharpeners

| # | item | status | notes |
|---|---|---|---|
| D1 | Chromatin / accessibility at guide loci | **TODO** | Public tracks (DNase/ATAC) for located hg38 coords — next after Tier 1 |
| D2 | GC + GC² as model features on human (ablation) | **DONE** | base 0.382 → base+GC/GC² **0.381** (null); GC/GC² alone 0.058 — `results/human_gc_feature_ablation.csv` |
| D3 | Label / assay sanity (score dist, uniqueness) | **DONE** | y∈[0,1], mean 0.248; 13,138 unique spacers; chrom spectrum looks sane — `results/human_label_sanity.json` |

### E. Tier 3 — later

| # | item | status |
|---|---|---|
| E1 | Cross-kingdom with Family A on both sides | after C4 — **unblocked**; not run yet |
| E2 | Human→*E. coli* with GC² / matched GC | after D2 — **unblocked**; not run yet |

---

## 3. How to re-run

```bash
# Needs: human + ecoli matrices, qct_lookup_tables.json, v_column_mapping.json, hg38.fa.gz
cd /workspace
python Jacky/human-boundary/scripts/run_human_boundary.py
# decode + autopsy only (no hg38):
python Jacky/human-boundary/scripts/run_human_boundary.py --skip-locate
```

Caches land in `Runjia/data/interim/human_boundary/` (decoded, located, `X_*.npy`, `a_flank_human.csv`).
Small tables land in `Jacky/human-boundary/results/`.

---

## 4. Plain-English takeaway

1. **Family A is real on human** (+0.052 ρ), smaller than the curated *E. coli* jump
   but not zero — flank context matters in both kingdoms.
2. **Transfer still dies** at ≈−0.047 on shared columns. Residualising out GC/GC²
   does **not** rescue it, so “bacteria learned the wrong GC rule” is a symptom,
   not the whole failure mode. Predictions are still GC-tilted (error↔GC ≈ −0.40).
3. **Putting GC/GC² in the human model** adds nothing — the quadratic shape is
   weak on human (peak ≈0.57, tiny R²). The kingdom break is not “forgot to
   include GC².”
4. **Next** (Tier 2/3): alignment of chromatin/accessibility at the 16.5k loci,
   then Family-A-aware cross-kingdom (E1) if we want a sharper autopsy.

---

## 5. What to upload (historical — now unblocked)

Drive matrices + hg38 are on disk for this run. If rebuilding elsewhere:

| priority | path under `Runjia/` | what |
|---|---|---|
| 1 | `data/raw/human_feature_matrix.csv` | Noshay human matrix (~17k guides) |
| 2–3 | `data/interim/qct_lookup_tables.json` + `v_column_mapping.json` | or rebuild from `ecoli_feature_matrix.csv` |
| 4 | `data/reference/hg38.fa.gz` | for locate + Family A |
