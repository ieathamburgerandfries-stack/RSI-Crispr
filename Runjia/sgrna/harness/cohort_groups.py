"""Grouped cross-validation means something different on the curated cohort: gene identity explains 29% of its label variance against 16% on the published cohort.

The report's grouped-CV choices rest on one number: the share of label variance
that lies *within* genes rather than between them. If almost all of it is
within, a model cannot score well by memorising which gene a guide sits in, and
a random split and a gene-grouped split measure nearly the same thing. That
number was measured on the published 13,880 rows and never re-measured after
the model layer moved to crisprHAL's 33,567, which is one of the open items in
Part 4.

It does not carry over. The published cohort targets a gene panel -- 1,225
genes, ~11 guides each -- while the curated cohort is genome-wide, so gene
identity explains more than twice as much of the label there.

Gene assignment is by coordinate on NC_000913.2 against the reference gene
table, not from either dataset's own annotation, so the two cohorts are
compared the same way. On the published rows that assignment agrees with
Noshay's annotation for 97.7% of guides and gives the same variance share to
three decimals (0.843 against 0.844), which is the check that the change below
is the cohort and not the method.

    python -m sgrna.harness.cohort_groups
"""

from __future__ import annotations

import pandas as pd

from .. import config
from ..build.build_features import _b_numbers, load_curated_index

OUT = config.RESULTS / "cohort_groups.csv"
BLOCK = 100_000


def _share(df: pd.DataFrame, group: str, y: str):
    """Within-group share of total variance, with the group count."""
    d = df[[group, y]].dropna()
    means = d.groupby(group)[y].transform("mean")
    total = ((d[y] - d[y].mean()) ** 2).sum()
    within = ((d[y] - means) ** 2).sum()
    return dict(n=len(d), n_groups=int(d[group].nunique()),
                guides_per_group=round(len(d) / d[group].nunique(), 1),
                within_share=float(within / total))


def run(verbose: bool = True) -> pd.DataFrame:
    rows = []
    pub = pd.read_csv(config.GUIDE_INDEX, low_memory=False)
    cur = load_curated_index(verbose=False)
    pub["gene"] = _b_numbers(pub)["gene_name"]
    cur["gene"] = cur["gene_name"]

    for cohort, df, y in (("published", pub, "cut_score"), ("curated", cur, "score")):
        d = df.dropna(subset=["left", y]).copy()
        d["block"] = (d["left"] // BLOCK).astype(int)
        for grouping, col in (("gene", "gene"), ("100kb_block", "block")):
            rows.append(dict(cohort=cohort, grouping=grouping,
                             **_share(d, col, y)))
    # The agreement check that makes the two cohorts comparable.
    rows.append(dict(cohort="published", grouping="gene (their annotation)",
                     **_share(pub, "gene_name", "cut_score")))

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    if verbose:
        print(out.to_string(index=False))
        print(f"\nwritten -> {OUT}")
    return out


if __name__ == "__main__":
    run()
