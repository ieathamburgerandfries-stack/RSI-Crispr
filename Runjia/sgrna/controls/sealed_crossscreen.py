"""A second screen as the sealed set: train on SpCas9 guides in development genes, score eSpCas9 labels in the sealed genes.

Every seal so far held out genes of the screen that supplied the training labels. This one also changes the screen. The model is fitted on the
WT-SpCas9 cohort (33,567 guides) in the development genes of one of the 15 sealing draws of `sealed_holdout.py` (same seeds, same 20% of
guides sealed by whole gene) and scored on the eSpCas9 screen's labels of every eSpCas9 guide that lies in a sealed gene: a different enzyme,
a different sequencing run, labels that no step has fitted and, for the guides absent from the SpCas9 screen, guides that no step has seen.

What this is and is not. eSpCas9's scores were visible during tuning and in the head-to-head, so the screen is confirmatory and not
untouched (Part 4 of the report). The design is new, though, and the prediction below was written before the first run. The loci are still
the same genome, so a guide in a sealed gene can have training neighbours whose flank windows (up to 1 kb) overlap its own beyond the gene's edge
(the limit that applies to every within-genome seal); the claim is that the flank gain survives a change of enzyme and of gene, not independence of
loci (that is the C. rodentium check).

PREDICTION, fixed before the run (23 Nov 2026): family A's gain (frozen configuration, 300 columns selected inside the training genes, against the
base alone) on the eSpCas9 labels of sealed genes is positive in at least 14 of the 15 draws and its mean is at least +0.08, about half of the
+0.156 on the WT screen's own sealed genes. Arms: base alone (6,169 columns, no selection) and base + family A (300 columns by gain).

    OMP_NUM_THREADS=6 python -m sgrna.controls.sealed_crossscreen [--max-draws N] [--threads T]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import config
from ..cohorts.curated_ablation import _base, _block
from . import sealed_holdout_gene as sg
from .sealed_holdout import OUT as SEALED_CSV, _seal


def _out_path():
    from ..common import OUT as POST
    return POST / "sealed_crossscreen.csv"


def _esp_matrix():
    """eSpCas9 guides: features (base + family A, 6,517 columns, as cached by `transfer.py`), labels and host genes."""
    from ..build.build_curated_features import _b_numbers
    from ..compare.transfer import build

    X, y, names = build("eSpCas9", verbose=False)
    m = json.loads((config.INTERIM / "transfer_eSpCas9" / "meta.json").read_text())
    left = np.asarray(m["left"], dtype=int)
    gdf = _b_numbers(pd.DataFrame({"left": left, "right": left + 19}))
    gene = np.array([g if isinstance(g, str) and g else "" for g in gdf["gene_name"]], dtype=object)
    return np.asarray(X, dtype=np.float32), np.asarray(y, float), gene, list(names), list(m["protospacers"])


def run(max_draws: int | None = None, threads: int = 6, verbose: bool = True) -> pd.DataFrame:
    sg.N_JOBS = threads
    out_path = _out_path()
    done = pd.read_csv(out_path) if out_path.exists() and out_path.stat().st_size else pd.DataFrame()
    ref = pd.read_csv(SEALED_CSV)
    X, y, names, protos = _base()
    A, anames = _block("a_flank", protos)
    XA = np.hstack([X, A])
    Xe, ye, gene_e, enames, _ = _esp_matrix()
    assert list(enames) == list(names) + list(anames), "eSpCas9 and curated columns differ"
    nb = X.shape[1]
    seeds = sorted(ref.seal_seed.unique())[:max_draws]
    for sd in seeds:
        if len(done) and (done.seal_seed == sd).any():
            continue
        sealed = _seal(protos, seed=int(sd), verbose=False)
        from ..build.build_curated_features import load_index
        idx = load_index(verbose=False).set_index(config.ID_COL)
        gw = idx["gene_name"].reindex(protos)
        sealed_genes = {g for g, s in zip(gw, sealed) if s and isinstance(g, str) and g}
        te = np.array([g in sealed_genes for g in gene_e])
        dev = np.where(~sealed)[0]
        t0 = time.time()
        p_base = sg._fit_predict(X[dev], y[dev], Xe[te][:, :nb], None, None, 41)
        p_A = sg._fit_predict(XA[dev], y[dev], Xe[te], None, config.N_FEATURES, 41)
        rb, ra = (float(spearmanr(ye[te], p).statistic) for p in (p_base, p_A))
        # the same arms scored on the WT screen's own sealed genes, from the sealing run
        wt = {a: float(ref[(ref.arm == a) & (ref.seal_seed == sd)].sealed.iloc[0]) for a in ("base_only", "frozen")}
        row = dict(seal_seed=int(sd), n_sealed_genes=len(sealed_genes), n_esp_guides=int(te.sum()),
                   rho_base=rb, rho_A=ra, gain_esp=ra - rb, wt_base=wt["base_only"], wt_A=wt["frozen"], gain_wt=wt["frozen"] - wt["base_only"])
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(out_path, index=False)
        if verbose:
            print(f"  seed {sd}: {te.sum():,} eSpCas9 guides in {len(sealed_genes):,} sealed genes  base {rb:.4f}  +A {ra:.4f}  gain {ra - rb:+.4f} "
                  f"(WT labels, same genes: {row['gain_wt']:+.4f})  {time.time() - t0:.0f}s", flush=True)
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-draws", type=int, default=None)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args(argv)
    d = run(a.max_draws, a.threads)
    print(f"\n{len(d)} draws: mean gain on eSpCas9 labels {d.gain_esp.mean():+.4f} (positive in {int((d.gain_esp > 0).sum())}), "
          f"on the WT screen's labels of the same sealed genes {d.gain_wt.mean():+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
