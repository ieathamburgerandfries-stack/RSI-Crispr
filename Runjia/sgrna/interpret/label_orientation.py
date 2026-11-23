"""Is a higher curated label more cutting? An anchor for the label's sign that does not come from a sequence rule.

Every claim about the direction of an effect (G at position 20 lowers activity in E. coli, flank GC raises it) depends on which way the
curated label runs. The label is an ALDEx2 log-ratio of Cas9 against dead-nuclease read counts, and the sign convention is the
released data's, so it is checked rather than assumed. A guide whose 20-mer plus PAM occurs at several places in the genome is cut at
every one of them, so it should deplete more than a guide with one target: if a higher label means more cutting, multi-copy guides
should score higher.

Result (`results/post_freeze/label_orientation.csv`, curated 33,567 guides): the 118 guides with more than one genomic copy have a
median label of 2.11 against 0.27 for the 33,449 single-copy guides (Mann-Whitney p about 1e-5), so a higher label is more cutting.
The test is weak in size (118 guides) but it is independent of every sequence rule the findings rely on, and it is in the direction
the labels were assumed to run.

    ./run_ext.sh sgrna.interpret.label_orientation
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import mannwhitneyu

from sgrna import config

from ..common import OUT


def main() -> pd.DataFrame:
    g = pd.read_csv(config.INTERIM / "curated_guide_index.csv")
    multi = g.n_genome_copies > 1
    u = mannwhitneyu(g.score[multi], g.score[~multi], alternative="two-sided")
    out = pd.DataFrame([dict(n_guides=len(g), n_multi_copy=int(multi.sum()), n_single_copy=int((~multi).sum()),
                             median_label_multi_copy=float(g.score[multi].median()), median_label_single_copy=float(g.score[~multi].median()),
                             auc_multi_higher=float(u.statistic / (multi.sum() * (~multi).sum())), mannwhitney_p=float(u.pvalue))])
    out.to_csv(OUT / "label_orientation.csv", index=False)
    print(out.T.to_string(header=False))
    return out


if __name__ == "__main__":
    main()
