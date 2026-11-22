"""Project paths and shared constants.

Works in three places without edits:

  1. In a notebook, after `sgrna.harness.notebook.setup()` -- which is how this
     project is normally driven (see notebooks/RSI09_pipeline.ipynb).
  2. Locally, when you run anything from inside the project folder.
  3. From anywhere else, if you set the environment variable RSI09_ROOT.

Nothing else in the package hardcodes a path -- if a file moves, change it here.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Project root
# --------------------------------------------------------------------------

_MARKERS = ("data/raw", "external_data")


def _looks_like_root(p: Path) -> bool:
    return all((p / m).exists() for m in _MARKERS)


def find_project_root() -> Path:
    env = os.environ.get("RSI09_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"RSI09_ROOT points at {p}, which does not exist.")
        return p

    # Walk up from this file: src/sgrna/config.py -> src/sgrna -> src -> ROOT
    here = Path(__file__).resolve()
    for parent in here.parents:
        if _looks_like_root(parent):
            return parent

    # Walk up from the current working directory (covers notebooks).
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if _looks_like_root(parent):
            return parent

    raise FileNotFoundError(
        "Could not find the project root. Set RSI09_ROOT to the folder that "
        "contains data/ and external_data/."
    )


ROOT = find_project_root()

# --------------------------------------------------------------------------
# Directories
# --------------------------------------------------------------------------

DATA = ROOT / "data"
RAW = DATA / "raw"
REFERENCE = DATA / "reference"
INTERIM = DATA / "interim"
FEATURE_DIR = INTERIM / "features"
PROCESSED = DATA / "processed"

EXTERNAL = ROOT / "external_data"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

for _d in (INTERIM, FEATURE_DIR, PROCESSED, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Key files
# --------------------------------------------------------------------------

# Noshay et al. 2023 supplementary table 2, unchanged. Row 1 is a title row,
# so every reader in this project passes header=1.
BASE_MATRIX = RAW / "ecoli_feature_matrix.csv"
BASE_MATRIX_HEADER_ROW = 1

ID_COL = "sgRNAID"
TARGET_COL = "cut.score"

# Reference genome. NC_000913.2 (U00096.2), 4,639,675 bp -- the build the
# GapR-seq wiggles and the Bikard screen use. Coordinates in this project are
# always NC_000913.2 unless a column name says otherwise.
GENOME_FASTA = REFERENCE / "NC_000913.2.fasta"
GENE_TABLE = REFERENCE / "genes_NC_000913.2.tsv"

# Outputs of step 0
GUIDE_INDEX = INTERIM / "guide_index.csv"
QCT_TABLES = INTERIM / "qct_lookup_tables.json"

# --------------------------------------------------------------------------
# External datasets
# --------------------------------------------------------------------------

GAPR_WIG_DIR = EXTERNAL / "GapR-seq" / "wig"
# GSE152880 sample roles, from the GEO sample titles.
GAPR_SAMPLES = {
    "signal": [  # GapR-3xFLAG, exponential phase, +aTc
        "GSM4628313_D18-11475-3531L_norm.wig.gz",
        "GSM4628314_D19-11570-4278G_MG1655_norm.wig.gz",
    ],
    "control": [  # untagged GapR -- the no-ChIP background
        "GSM4628311_D19-5504-3883G_norm.wig.gz",
        "GSM4628312_D19-11573-4278G_MG1655_norm.wig.gz",
    ],
    "rifampicin": [  # GapR-3xFLAG + rifampicin: transcription switched off
        "GSM4628315_D18-11479-3531L_norm.wig.gz",
    ],
    "truncated": [  # GapR(1-76)-3xFLAG control
        "GSM4989042_D20-5423-4700M_norm.wig.gz",
        "GSM4989043_D20-5424-4700M_norm.wig.gz",
    ],
}

LIOY_DIR = EXTERNAL / "E_coli_analysis" / "data"
LIOY_XYZ = LIOY_DIR / "3Dcoor_ecoli_5kb_3_3_copie.xyz"
LIOY_MATRIX_DIR = LIOY_DIR / "matrices_5kb"
LIOY_BIN_SIZE = 5000

REGULONDB = EXTERNAL / "RegulonDB"
PRECISE1K = EXTERNAL / "precise1k" / "data"
CRISPRHAL = EXTERNAL / "crisprHAL" / "data"
CRISPROFF = EXTERNAL / "crisproff"
BADSEED = EXTERNAL / "CRISPRBact" / "badSeed_public"

# --------------------------------------------------------------------------
# Biology constants
# --------------------------------------------------------------------------

# SpCas9 sgRNA scaffold (the constant region 3' of the 20 nt spacer), as used
# in the Guo et al. E. coli library. Needed to fold the transcript that is
# actually expressed rather than the bare spacer.
SPCAS9_SCAFFOLD = (
    "GUUUUAGAGCUAGAAAUAGCAAGUUAAAAUAAGGCUAGUCCGUUAUCAACUUGAAAAAGUGGCACCGAGUCGGUGC"
)

# How far either side of the protospacer to keep in the guide index. The
# curated re-analysis of this screen found signal out to ~190 nt upstream and
# ~166 nt downstream; the first ablation here found the wide-flank summaries
# carrying 62% of family A's gain and showing no sign of saturating at 250 nt,
# so the window was widened to 1 kb to find where it does.
FLANK = 1000

# Positions, relative to the protospacer, that family A encodes one-hot.
UPSTREAM_POSITIONS = 10    # -1 .. -10, immediately 5' of protospacer position 1
DOWNSTREAM_POSITIONS = 15  # +1 .. +15, immediately 3' of the NGG PAM

# Windows (nt) for compositional and tensor summaries of the flanks.
FLANK_WINDOWS = (50, 250, 500, 1000)

# --------------------------------------------------------------------------
# Modelling defaults -- match the frozen baseline so ablations are comparable
# --------------------------------------------------------------------------

# Which model ranks the columns before the champion is fitted. The ranking
# model is trained and thrown away, so only its speed and its ordering matter.
# Measured on 2026-10-03 (`results/selector_comparison.csv`): LightGBM gain
# matches XGBoost gain (rho 0.7033 vs 0.7025) at 3x the speed, taking a fold
# from 17.1 s to 6.6 s. Everything recorded before that date used "xgboost";
# set it back to reproduce those runs exactly.
# Default is "xgboost": it is what the frozen baseline used, and switching
# costs the exact reproduction of the published figure (R2 0.2937 / rho 0.5278
# becomes 0.2897 / 0.5251), which is the project's harness check. "lightgbm"
# ranks just as well (rho 0.7033 vs 0.7025 on the curated set) at 3x the speed
# -- use it for exploratory sweeps where the anchor does not matter.
SELECTOR = "xgboost"

N_FEATURES = 300
N_ESTIMATORS = 400
N_SPLITS = 5
SEEDS = (41, 42, 43, 44, 45)

# Reference numbers from five_seed_champion_ecoli_xgboost_baseline.
BASELINE = {"r2": 0.29368, "spearman": 0.52782, "pearson": 0.54244}


def describe() -> str:
    lines = [f"project root : {ROOT}"]
    for name, p in [
        ("base matrix", BASE_MATRIX),
        ("genome", GENOME_FASTA),
        ("gene table", GENE_TABLE),
        ("guide index", GUIDE_INDEX),
        ("feature dir", FEATURE_DIR),
        ("external", EXTERNAL),
    ]:
        mark = "ok " if p.exists() else "MISSING"
        lines.append(f"{name:12s} : [{mark}] {p}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
