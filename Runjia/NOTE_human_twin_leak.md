# Note for Jacky — the human matrix has repeated guides, and it affects the 0.5076 baseline

**Runjia, 19 November 2026.** Short version: the human iRF matrix contains the same guide more than once, and
because the standard folds split on *rows* rather than on guides, a test guide's duplicate is usually sitting in
the training set. The configuration we had been treating as the human baseline is the one this helps most, so its
number does not hold up once duplicates are kept together. **Nothing on the bacterial side is affected** — that
cohort has 33,567 distinct guides in 33,567 rows and is already grouped by gene.

I am not reporting this because the configuration was a bad choice. It reproduced exactly as you described it, and
the reason it looked good is a property of the matrix that neither of us had checked.

## What is in the matrix

`data/raw/human_feature_matrix.csv` has **17,421 rows** but only **13,138 distinct 20-mers**. 4,216 sequences
appear more than once (4,149 twice, 67 three times), so **8,499 rows — 49% of the matrix — have a twin**, listed at
the same genomic position. Between twins the label agrees at **Spearman 0.80** (59% are exactly equal), so a twin is
close to a free copy of the answer.

Decoded from the `pNmonomer.No.electronsraw` columns, which invert to the protospacer exactly; the check that the
decoded table lines up row-for-row with the matrix is an assertion in the script, not an assumption.

## What it does to the baseline

Re-running with folds that keep identical protospacers together (`sgrna/compare/human_twin_check.py`, 3 seeds × 5 folds):

| arm | row-wise folds | twin-grouped folds |
|---|---:|---:|
| harness default (300 columns, champion settings) | 0.4036 | **0.3356 ± 0.0215** |
| 400 columns, 1,000 trees, library defaults | **0.5076** | **0.2515 ± 0.0191** |

So the configuration that looked **+0.10 better** is in fact **−0.08 worse**. The ordering reverses.

**The mechanism is specific, and your own observation is the evidence for it.** You noted that `n_estimators` kept
improving with no clear plateau, unlike on the bacterial data. That is what memorisation looks like: XGBoost's library
defaults are `max_depth=6, learning_rate=0.3` against the champion's `4` and `0.05`, and depth, learning rate and tree
count are exactly the knobs that let a model memorise a sequence whose twin it has already seen. On the bacterial
cohort there is nothing to memorise, which is why the plateau is there and not here.

## What I would suggest

1. **Group the folds on the decoded protospacer** for anything measured on this matrix. One line, and it makes the
   number mean what it says.
2. **Re-pick the settings under grouped folds.** The comparison that chose 400/1,000/defaults was run under the
   leak, so it was choosing partly for memorisation. The champion may not be right either — the point is that the
   earlier comparison cannot decide it.
3. **Nothing needs redoing on the flank work itself** as far as I can tell, but the numbers that quantify it on the
   human side do, because they inherit the folds. See below.

## What this cost us, and what is being re-run

Three statements in `PROGRESS_REPORT.md` rested on row-wise human folds and are now flagged in the main text rather
than only in Appendix B.4:

- the **+0.052** human flank gain,
- the **ridge/boosting ratio** (95.6% in *E. coli* against 63.7% in human) behind "the human signal is far less
  additive",
- the **cross-kingdom transfer** figures.

The additivity claim had a second supposedly independent instrument — `max_depth` 4→6 being worth +0.0585 on human
against +0.0063 in *E. coli* — but depth is a memorisation knob, so that corroboration is not independent of the
leak; it is the leak's signature.

**I have now re-run it** (`sgrna/compare/human_additivity_twin.py`), both fold schemes in one job with the same columns and
seeds, keeping the row-wise arms as a positive control. **The control is exact — every row-wise number reproduces the
published figure to four decimals** — so the only thing that differs between the columns is the folds:

| 400 columns, 3 seeds × 5 folds | row-wise ρ | twin-grouped ρ |
|---|---:|---:|
| champion (depth 4, lr 0.05, 400 trees) | 0.4074 | **0.3377** ← now the best arm |
| `max_depth` 4 → 6 | 0.4659 | 0.3315 |
| `learning_rate` 0.05 → 0.3 | 0.4490 | 0.2560 |
| `n_estimators` 400 → 1,000 | 0.4435 | 0.3226 |
| no subsampling | 0.3999 | 0.3333 |
| all three (your configuration) | 0.5076 | **0.2515** |
| ridge | 0.3232 | 0.3001 |
| **ridge / best boosting** | 63.7% | **88.8%** |
| **`max_depth` 4 → 6** | **+0.0585** | **−0.0062** [−0.0093, −0.0034] |

**Three conclusions, and the first one is good news for you.**

1. **The champion is the best of the six arms once twins are grouped**, so there was never a shape mismatch to fix. Our
   report claimed the champion was "mis-specified by 0.10 ρ on the human matrix" — that is withdrawn. Your
   configuration was not a mistake; it was the configuration that best exploited a property of the data nobody had
   noticed, which is a different thing.
2. **Every capacity knob reverses sign.** Depth, learning rate and tree count all help under row-wise folds and all
   hurt under grouped ones. That is as clean a memorisation signature as this kind of data produces, and it is why your
   "no clear plateau" observation was the most informative thing either of us had.
3. **Our cross-kingdom additivity claim is withdrawn, not rescoped.** It said the human phenotype is far less additive
   than *E. coli*'s, on a 63.7%-against-95.6% ratio, corroborated by the depth result. The ratio is 88.8% and the depth
   result changes sign, so the two instruments now disagree. A 6.8-point gap on column sets that were never comparable
   (427 decorrelated against 400 selected) is not a finding.

**What this does not touch:** the GC inversion between kingdoms, which is a marginal association and needs no model.
**What is still outstanding:** the +0.052 human flank gain and the cross-kingdom transfer figures — including the
family-A-on-both-sides result that turns transfer positive — are all on row-wise folds and have **not** been re-run.
Those are yours, and they are the ones I would do next.

Full write-up: `EXTENSION_LOG.md` §16.2; results in `results/ext/human_additivity_twin*.csv`.

## To reproduce

```bash
cd RSI-Crispr-fork/Runjia
./run_ext.sh sgrna.compare.human_twin_check          # the table above
./run_ext.sh sgrna.compare.human_additivity_twin     # ridge + knob sweep, both fold schemes
```

Background and the duplicate-count derivation: `EXTENSION_LOG.md` §10, and `PROGRESS_REPORT.md` Appendix B.4
("Correction to the human baseline"). Both are in my fork,
`github.com/ieathamburgerandfries-stack/RSI-Crispr`, under `Runjia/`.
