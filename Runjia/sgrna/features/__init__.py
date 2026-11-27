"""Feature families.

Each module exposes `build(index) -> DataFrame` keyed by sgRNAID, with every
other column prefixed `eng.`. Families are independent: build one, rebuild
one, drop one, without touching the rest.

Family   Module            What it adds                        Needs
-------  ----------------  ----------------------------------  ---------------
A        a_flank           sequence + QCT outside the 20-mer    genome only
B        b_energy          CRISPRoff dG decomposition           crisproff repo
C        c_folding         sgRNA folding, spacer:scaffold       ViennaRNA
D1       d_mechanics       DNA duplex thermodynamics            genome only
D2       d_supercoiling    GapR-seq supercoiling density        GSE152880 wigs
E        e_nucleoid        Hi-C 3D structure, NAP dependence    Lioy 2018
F        f_methylation     Dam/Dcm motifs at PAM and seed       genome only
G        g_transcription   TSS distance, strand, expression     RegulonDB + P1K
H        h_offtarget       off-target burden, target copies     genome only
I        i_shape           flank DNA bendability / deformability genome only
J        j_periodicity     GC skew + helical-repeat periodicity  genome only
"""

from __future__ import annotations

ALL = ("a_flank", "b_energy", "c_folding", "d_mechanics", "d_supercoiling", "e_nucleoid", "f_methylation", "g_transcription", "h_offtarget", "i_shape", "j_periodicity")


def __getattr__(name: str):
    """`REGISTRY` (family name -> module) is built on first use, so that `from sgrna.features import a_flank` does not import, and need the data of, all eleven families.

    The guide-design tool only needs family A and has no project data folder; importing every family at package import made it fail on a fresh clone.
    """
    if name == "REGISTRY":
        import importlib

        registry = {n: importlib.import_module(f".{n}", __name__) for n in ALL}
        globals()["REGISTRY"] = registry
        return registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Families that need nothing but the reference genome, so they always work.
SELF_CONTAINED = ("a_flank", "c_folding", "d_mechanics", "f_methylation", "h_offtarget",
                  "i_shape", "j_periodicity")
