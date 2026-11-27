"""Sequence distance between training and validation guides, the leakage statistic our comparators report.

Browne et al. (2026) check the Hamming distance between every training and held-out guide and confirm no
identical inputs cross the split; Yu et al. (2024) report the minimum and median (7 and 21). We have
always checked exact duplicates and grouped folds by gene and by 100 kb block, but never published a
distance statistic -- and near-duplicate leakage is exactly what invalidated the human additivity claim
(section 16.2), so asserting that grouping covers it is weaker than showing it.

For every validation guide, the minimum Hamming distance to any training guide, under each fold scheme:

  curated / gene-grouped    what every positional result in the report uses
  curated / random rows     the same cohort with KFold, as the contrast
  human   / twin-grouped    the human matrix with identical protospacers kept together
  human   / random rows     the human matrix as the published numbers were measured -- the positive
                            control, which must show minimum distance 0 if the statistic works at all

    python -m sgrna.controls.fold_distance

Report section: Part 5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from sgrna import config
from sgrna.harness.run_ablation import grouped_folds

from ..common import OUT

LUT = np.full(256, 4, np.uint8)
for _i, _c in enumerate("ACGT"):
    LUT[ord(_c)] = _i


def codes(seqs) -> np.ndarray:
    L = min(len(s) for s in seqs)
    raw = np.frombuffer("".join(s[:L] for s in seqs).encode("ascii"), np.uint8)
    return LUT[raw].reshape(len(seqs), L)


def min_hamming(va: np.ndarray, tr: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Minimum Hamming distance from each row of `va` to any row of `tr`."""
    out = np.empty(len(va), np.int16)
    for i in range(0, len(va), chunk):
        blk = va[i : i + chunk]
        d = (blk[:, None, :] != tr[None, :, :]).sum(2)
        out[i : i + chunk] = d.min(1)
    return out


def one(seqs, groups, scheme: str, n_splits=5, seed=41):
    C = codes(seqs)
    folds = (list(KFold(n_splits, shuffle=True, random_state=seed).split(C)) if scheme.endswith("random rows")
             else list(grouped_folds(groups, n_splits, seed)))
    mins = []
    for tr, va in folds:
        mins.append(min_hamming(C[va], C[tr]))
    m = np.concatenate(mins)
    return dict(scheme=scheme, n_guides=len(C), guide_len=C.shape[1], min=int(m.min()),
                p1=float(np.percentile(m, 1)), median=float(np.median(m)),
                frac_identical=float((m == 0).mean()), frac_le_2=float((m <= 2).mean()))


def main():
    rows = []
    idx = pd.read_csv(config.INTERIM / "curated_guide_index.csv")
    protos = idx["protospacer"].astype(str).tolist()
    gene = idx["gene"].astype(str).to_numpy() if "gene" in idx else np.arange(len(idx)).astype(str)
    print(f"curated: {len(protos):,} guides, {len(set(protos)):,} distinct", flush=True)
    rows.append(one(protos, gene, "curated / gene-grouped"))
    print("  " + str(rows[-1]), flush=True)
    rows.append(one(protos, gene, "curated / random rows"))
    print("  " + str(rows[-1]), flush=True)

    from ..compare.human_flank import decode
    dec = decode()
    ok = np.isfinite(pd.to_numeric(dec.y, errors="coerce").to_numpy(float))
    hp = dec.protospacer[ok].astype(str).tolist()
    hg = pd.factorize(dec.protospacer[ok])[0].astype(str)
    print(f"human: {len(hp):,} guides, {len(set(hp)):,} distinct", flush=True)
    rows.append(one(hp, hg, "human / twin-grouped"))
    print("  " + str(rows[-1]), flush=True)
    rows.append(one(hp, hg, "human / random rows"))
    print("  " + str(rows[-1]), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "fold_distance.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
