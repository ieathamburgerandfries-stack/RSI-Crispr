"""Family B -- the R-loop free-energy decomposition (CRISPRoff / CRISPRspec).

Cas9 has to pay to unwind the target duplex and to unfold its own guide, and
is repaid by RNA:DNA hybridisation. Alkan et al. (2018) and Corsi et al. (2022) split that
budget into terms that a sequence-composition feature cannot express:

    dG_H   RNA:DNA hybridisation, position-weighted
    dG_O   DNA:DNA opening penalty
    dG_U   guide self-folding penalty
    dG_B   the net binding energy, with the PAM correction applied

The relationship to activity is *not* monotonic. Corsi et al. (2022) found
efficient guides cluster in a window of roughly -64.5 to -47.1 kcal/mol and
fall off on both sides, so this module also emits the signed and absolute
distance from the middle of that window. A gradient-boosted tree can discover
a threshold on its own but not a two-sided optimum from one split, so handing
it the distance explicitly is worth a column.

Implementation
--------------
Rather than reimplement the energy model, this imports the authors' own
pipeline (`external_data/crisproff/CRISPRspec_CRISPRoff_pipeline.py`) and
calls their functions with their published parameter pickle. The only change
is that `get_rnafold_eng` is redirected from a per-sequence `RNAfold`
subprocess to the ViennaRNA Python API, which is the same calculation several
thousand times faster.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import config
from ..io_utils import save_block

PREFIX = "eng.dg"

# Midpoint of the efficient-guide hybridisation window reported by
# Corsi et al. 2022 (Nat Commun 13:2601, doi 10.1038/s41467-022-30515-0): -64.53 to -47.09 kcal/mol.
DG_H_WINDOW = (-64.53, -47.09)
DG_H_CENTRE = sum(DG_H_WINDOW) / 2


@lru_cache(maxsize=1)
def _pipeline():
    """Import the CRISPRoff pipeline and patch in fast folding."""
    script = config.CRISPROFF / "CRISPRspec_CRISPRoff_pipeline.py"
    if not script.exists():
        raise FileNotFoundError(
            f"{script} not found. Clone RTH-tools/crisproff into external_data/."
        )
    spec = importlib.util.spec_from_file_location("crisproff_pipeline", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crisproff_pipeline"] = mod
    spec.loader.exec_module(mod)
    mod.read_energy_parameters(str(config.CRISPROFF / "energy_dics.pkl"))

    try:
        import RNA
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ViennaRNA's Python bindings are needed: pip install viennarna"
        ) from exc

    cache: dict[str, float] = {}

    def fold_energy(seq, rid="guide"):
        if seq not in cache:
            cache[seq] = float(RNA.fold(seq)[1])
        return cache[seq]

    mod.get_rnafold_eng = fold_energy
    return mod


def energies_for_target(target_23: str) -> dict[str, float]:
    """All five published energy terms for one 23 nt protospacer+PAM."""
    m = _pipeline()
    dg_h_raw = -sum(m.calcRNADNAenergy(target_23, target_23))
    dg_h_weighted = m.get_eng(
        target_23, target_23, m.calcRNADNAenergy, pos_weight=True
    )
    dg_o = sum(m.calcDNAopeningScore(target_23))
    dg_u = m.get_rnafold_eng(target_23[:20])
    dg_b = m.get_eng(
        target_23,
        target_23,
        m.calcRNADNAenergy,
        pos_weight=True,
        pam_corr=True,
        grna_folding=True,
        dna_opening=True,
    )
    return {
        f"{PREFIX}.hybridisation": float(dg_h_raw),
        f"{PREFIX}.hybridisation_weighted": float(dg_h_weighted),
        f"{PREFIX}.dna_opening": float(dg_o),
        f"{PREFIX}.guide_selffold": float(dg_u),
        f"{PREFIX}.binding": float(dg_b),
    }


def build(index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in index.iterrows():
        rec: dict[str, float] = {config.ID_COL: row[config.ID_COL]}
        proto, pam = row.get("protospacer"), row.get("pam")
        if isinstance(proto, str) and isinstance(pam, str) and "N" not in proto:
            try:
                rec.update(energies_for_target(proto + pam))
            except Exception as exc:  # a single bad guide must not kill the run
                rec[f"{PREFIX}.error"] = str(exc)[:60]
        rows.append(rec)

    out = pd.DataFrame(rows)
    out = out.drop(columns=[c for c in out.columns if c.endswith(".error")],
                   errors="ignore")

    # Distance from the middle of the efficient window.
    #
    # This must use the POSITION-WEIGHTED hybridisation term, not the raw one.
    # Corsi et al. report the window on the weighted quantity; measured against
    # the raw term, zero of the 13,880 guides fall inside it (so the indicator
    # was a constant column and the offsets were meaningless), while against
    # the weighted term 64% do. Checking that the indicator is not constant is
    # the cheap way to catch this if the convention ever changes again.
    h = out.get(f"{PREFIX}.hybridisation_weighted")
    if h is not None:
        # Sign convention of the paper: favourable binding is negative.
        signed = -h
        out[f"{PREFIX}.hyb_dG"] = signed
        out[f"{PREFIX}.hyb_offset"] = signed - DG_H_CENTRE
        out[f"{PREFIX}.hyb_abs_offset"] = (signed - DG_H_CENTRE).abs()
        out[f"{PREFIX}.hyb_in_window"] = (
            (signed >= DG_H_WINDOW[0]) & (signed <= DG_H_WINDOW[1])
        ).astype("int8")
        if out[f"{PREFIX}.hyb_in_window"].nunique() == 1:
            print("  ! b_energy: hyb_in_window is constant -- check the dG "
                  "convention against Corsi et al. before trusting it")

    # The net budget, term by term, is more interpretable as ratios too.
    if all(f"{PREFIX}.{k}" in out for k in ("hybridisation", "dna_opening")):
        denom = out[f"{PREFIX}.hybridisation"].replace(0, np.nan)
        out[f"{PREFIX}.opening_over_hyb"] = out[f"{PREFIX}.dna_opening"] / denom
    return out


def main() -> None:
    from ..io_utils import load_guide_index

    idx = load_guide_index()
    block = build(idx)
    path = save_block(block, "b_energy")
    print(f"b_energy: {block.shape[0]:,} rows x {block.shape[1] - 1:,} features -> {path}")


if __name__ == "__main__":
    main()
