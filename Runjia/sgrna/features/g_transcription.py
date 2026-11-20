"""Family G -- transcription and replication context. Handle with discipline.

Elongating RNA polymerase dislodges Cas9 from a cleaved target, and does so
asymmetrically: the effect depends on which strand the guide matches and on
how heavily the gene is transcribed (Clarke et al. 2018, Mol Cell). So there
is a real mechanism here.

But the coarse versions of these variables have already been tested against
this exact label and reported as uncorrelated: Guo et al. (2018) found no
meaningful relationship for target gene expression, chromosome position, or
leading-versus-lagging strand. That is the reason this module exists in the
form it does. Marginal expression, marginal strand and marginal position are
included only so the null can be *replicated*; the columns worth arguing about
are the interactions, because a mechanism that needs both transcription and
strand to line up cannot show up in either marginal alone.

The rule for using this family: claim nothing unless an interaction term beats
both of its parents in the ablation. If it does not, report the replicated
null -- which is a result, and a citable one.

Coordinates
-----------
RegulonDB reports positions on U00096.3, while the genome used here (and by
the GapR-seq wiggles and the Bikard screen) is U00096.2, and the two differ by
up to ~2 kb. Rather than lift over, each promoter is re-located by searching
the genome for the 80 nt sequence RegulonDB ships with it; the TSS is the one
upper-case base in that string. That is exact, offline, and self-checking --
promoters that fail to anchor are simply dropped, with the count reported.

Columns
-------
eng.txn.tss.*        distance and orientation to the nearest anchored TSS
eng.txn.gene.*       position within the host gene, template vs non-template
eng.txn.expr.*       PRECISE-1K expression level and variability for the gene
eng.txn.repl.*       replichore, head-on vs co-directional transcription
eng.txn.ix.*         the interaction terms this family actually rests on
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config
from ..build import genome
from ..io_utils import save_block

PREFIX = "eng.txn"

MAX_TSS_DISTANCE = 100_000  # cap so the column stays well-scaled


@lru_cache(maxsize=1)
def anchored_promoters() -> pd.DataFrame:
    """RegulonDB promoters re-located in NC_000913.2 by sequence search."""
    path = config.REGULONDB / "PromoterSet.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    pm = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    pm.columns = [c.split(")")[-1].strip() for c in pm.columns]
    pm = pm.dropna(subset=["sequence"])

    g = genome.load_genome()
    rows = []
    for _, r in pm.iterrows():
        raw = str(r["sequence"])
        upper = [i for i, ch in enumerate(raw) if ch.isupper()]
        if not upper:
            continue
        offset = upper[0]
        seq = raw.upper()
        i = g.find(seq)
        if i >= 0:
            tss = i + offset + 1
            strand = "+"
        else:
            j = g.find(genome.revcomp(seq))
            if j < 0:
                continue
            tss = j + len(seq) - offset
            strand = "-"
        rows.append(
            dict(
                promoter=r.get("name"),
                tss=tss,
                strand=strand,
                sigma=r.get("sigmaFactor"),
                first_gene=r.get("firstGeneName"),
            )
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def gene_table() -> pd.DataFrame:
    """Gene coordinates on NC_000913.2 (from the Bikard screen repository)."""
    gt = pd.read_csv(config.GENE_TABLE, sep="\t")
    gt = gt.rename(columns={"ori": "strand"})
    return gt


@lru_cache(maxsize=1)
def expression_table() -> pd.DataFrame:
    """PRECISE-1K per-gene expression summary, keyed by b-number."""
    path = config.PRECISE1K / "annotation" / "gene_info.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    gi = pd.read_csv(path, low_memory=False)
    keep = {
        "locus_tag": "b_number",
        "gene_name": "p1k_gene_name",
        "p1k_median_log_tpm": "expr_median",
        "p1k_mad_log_tpm": "expr_mad",
        "p1k_ctrl_log_tpm": "expr_control",
        "essential": "essential",
        "pseudogene": "pseudogene",
    }
    gi = gi[[c for c in keep if c in gi.columns]].rename(columns=keep)
    gi["b_number"] = gi["b_number"].str.lower()
    for col in ("essential", "pseudogene"):
        if col in gi:
            gi[col] = gi[col].map({True: 1, False: 0, "True": 1, "False": 0})
    return gi


def _nearest_tss(positions: np.ndarray, tss_sorted: np.ndarray,
                 n: int) -> np.ndarray:
    """Circular distance from each position to the nearest TSS."""
    if tss_sorted.size == 0:
        return np.full(positions.shape, np.nan)
    idx = np.searchsorted(tss_sorted, positions)
    left = tss_sorted[(idx - 1) % tss_sorted.size]
    right = tss_sorted[idx % tss_sorted.size]
    d_left = np.abs(positions - left)
    d_right = np.abs(positions - right)
    d_left = np.minimum(d_left, n - d_left)
    d_right = np.minimum(d_right, n - d_right)
    return np.minimum(d_left, d_right)


def build(index: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    n = genome.genome_length()
    out = pd.DataFrame({config.ID_COL: index[config.ID_COL].to_numpy()})

    left = index["left"].to_numpy(dtype="float64")
    valid = np.isfinite(left)
    pos = np.where(valid, left, 1).astype(np.int64)
    guide_strand = index["strand"].to_numpy()

    # ---- TSS proximity --------------------------------------------------
    try:
        prom = anchored_promoters()
        if verbose:
            print(f"  anchored {len(prom):,} RegulonDB promoters in NC_000913.2")
        for label, subset in (
            ("any", prom),
            ("plus", prom[prom["strand"] == "+"]),
            ("minus", prom[prom["strand"] == "-"]),
        ):
            tss = np.sort(subset["tss"].to_numpy(dtype=np.int64))
            d = _nearest_tss(pos, tss, n)
            out[f"{PREFIX}.tss.dist_{label}"] = np.where(
                valid, np.minimum(d, MAX_TSS_DISTANCE), np.nan
            )
        # Is the guide's own strand the same as the nearest promoter's?
        tss_all = subset  # noqa: F841  (kept for readability of the loop above)
    except FileNotFoundError as exc:
        if verbose:
            print(f"  ! skipping TSS features: {exc}")

    # ---- host gene ------------------------------------------------------
    gt = gene_table()
    starts = gt["left"].to_numpy()
    ends = gt["right"].to_numpy()
    strands = gt["strand"].to_numpy()
    order = np.argsort(starts)
    starts, ends, strands = starts[order], ends[order], strands[order]

    rel_pos = np.full(len(index), np.nan)
    template = np.full(len(index), np.nan)
    gene_len = np.full(len(index), np.nan)
    codirectional = np.full(len(index), np.nan)

    slot = np.searchsorted(starts, pos, side="right") - 1
    for row_i, (p, s, gs) in enumerate(zip(pos, slot, guide_strand)):
        if not valid[row_i] or s < 0:
            continue
        # A guide can sit inside a gene that started a little earlier; scan a
        # few candidates back rather than assuming non-overlapping genes.
        hit = None
        for k in range(s, max(-1, s - 5), -1):
            if starts[k] <= p <= ends[k]:
                hit = k
                break
        if hit is None:
            continue
        g_left, g_right, g_strand = starts[hit], ends[hit], strands[hit]
        length = g_right - g_left + 1
        gene_len[row_i] = length
        frac = (p - g_left) / length
        rel_pos[row_i] = frac if g_strand == "+" else 1.0 - frac
        # The protospacer matching the coding strand means the sgRNA has the
        # same sequence as the mRNA, i.e. it targets the non-template strand.
        template[row_i] = 0.0 if gs == g_strand else 1.0
        fork = genome.replication_fork_direction(int(p))
        codirectional[row_i] = float((g_strand == "+") == (fork == 1))

    out[f"{PREFIX}.gene.rel_position"] = rel_pos
    out[f"{PREFIX}.gene.length"] = gene_len
    out[f"{PREFIX}.gene.template_strand"] = template
    out[f"{PREFIX}.repl.codirectional"] = codirectional
    out[f"{PREFIX}.repl.ori_distance"] = index["ori_distance"].to_numpy()
    out[f"{PREFIX}.repl.leading_strand"] = index["leading_strand"].to_numpy()
    out[f"{PREFIX}.repl.right_replichore"] = (
        index["replichore"].map({"right": 1, "left": 0}).to_numpy()
    )

    # ---- expression -----------------------------------------------------
    try:
        expr = expression_table()
        merged = index[[config.ID_COL, "b_number"]].merge(
            expr, on="b_number", how="left"
        )
        for col, name in (
            ("expr_median", f"{PREFIX}.expr.median_log_tpm"),
            ("expr_mad", f"{PREFIX}.expr.mad_log_tpm"),
            ("expr_control", f"{PREFIX}.expr.control_log_tpm"),
            ("essential", f"{PREFIX}.expr.essential"),
            ("pseudogene", f"{PREFIX}.expr.pseudogene"),
        ):
            if col in merged:
                out[name] = pd.to_numeric(merged[col], errors="coerce").to_numpy()
        if verbose:
            got = out[f"{PREFIX}.expr.median_log_tpm"].notna().sum()
            print(f"  matched expression for {got:,} / {len(out):,} guides")
    except FileNotFoundError as exc:
        if verbose:
            print(f"  ! skipping expression features: {exc}")

    # ---- the interactions this family actually rests on ------------------
    expr_col = f"{PREFIX}.expr.median_log_tpm"
    if expr_col in out:
        e = out[expr_col]
        # Centre so the interaction is not just a rescaled main effect.
        e_c = e - e.mean()
        out[f"{PREFIX}.ix.expr_x_template"] = e_c * (
            out[f"{PREFIX}.gene.template_strand"] - 0.5
        )
        out[f"{PREFIX}.ix.expr_x_codirectional"] = e_c * (
            out[f"{PREFIX}.repl.codirectional"] - 0.5
        )
        if f"{PREFIX}.tss.dist_any" in out:
            out[f"{PREFIX}.ix.expr_over_tss_dist"] = e_c / (
                1.0 + out[f"{PREFIX}.tss.dist_any"]
            )
    out[f"{PREFIX}.ix.template_x_leading"] = (
        out[f"{PREFIX}.gene.template_strand"] - 0.5
    ) * (out[f"{PREFIX}.repl.leading_strand"] - 0.5)
    return out


def main() -> None:
    from ..io_utils import load_guide_index

    idx = load_guide_index()
    block = build(idx)
    path = save_block(block, "g_transcription")
    print(f"g_transcription: {block.shape[0]:,} rows x {block.shape[1] - 1:,} features -> {path}")


if __name__ == "__main__":
    main()
