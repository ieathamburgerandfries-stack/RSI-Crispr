"""Notebook front end -- everything the pipeline does, callable from a cell.

The rest of the package is written as modules with `python -m` entry points,
which is fine from a terminal but awkward in a notebook: a `!python -m ...`
cell gives you a wall of text and no object to inspect afterwards. The
functions here wrap the same code so each step is one call that prints a short
summary and hands back a DataFrame you can look at.

    import sgrna.notebook as nb
    nb.setup()
    nb.build_index()
    nb.build_features("all")
    nb.ablate(["c_folding"], group="bin100k")

Nothing here duplicates logic -- every function delegates to the module that
owns the step, so a change there shows up here automatically.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Populated by setup(); imported lazily so this module can be imported before
# the project root is known.
config = None

REQUIREMENTS = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("RNA", "viennarna"),
    ("Bio", "biopython"),
]

_MARKERS = ("data/raw", "external_data")


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------


def _find_root(start: Path | None = None) -> Path | None:
    """Walk up from the notebook's directory looking for the project root."""
    candidates = [start] if start else []
    candidates += [Path.cwd(), *Path.cwd().parents]
    for p in candidates:
        if p and all((p / m).exists() for m in _MARKERS):
            return p
    return None


def _is_hosted_colab() -> bool:
    """True on Google's VM, false on a local runtime (which is the point)."""
    if "google.colab" not in sys.modules:
        return False
    # A local runtime still imports google.colab, but the Google VM always has
    # this marker directory and a local machine essentially never does.
    return Path("/usr/local/lib/python3.10/dist-packages/google/colab").exists() \
        or Path("/content").exists() and not Path("/content").is_symlink()


def install_requirements(quiet: bool = True) -> list[str]:
    missing = []
    for module, package in REQUIREMENTS:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        print(f"Installing {', '.join(missing)} ...")
        cmd = [sys.executable, "-m", "pip", "install", *(["-q"] if quiet else []),
               *missing]
        subprocess.check_call(cmd)
    return missing


def setup(root: str | Path | None = None, install: bool = True,
          drive_path: str = "/content/drive/MyDrive/RSI09") -> Path:
    """Point the session at the project, install what is missing, report back.

    On a **local runtime** -- the setup this project is written for -- the
    notebook already sits inside the project folder, so this needs no
    arguments: the root is found by walking up from the working directory.

    On a **hosted Colab VM** the project has to come from Drive, so Drive is
    mounted and `drive_path` is used instead. That path is the only thing that
    needs editing if you keep the folder somewhere else.
    """
    global config

    found = Path(root).expanduser().resolve() if root else _find_root()

    if found is None and _is_hosted_colab():
        from google.colab import drive  # type: ignore

        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        found = Path(drive_path)

    if found is None or not found.exists():
        raise FileNotFoundError(
            "Could not find the project root.\n"
            "On a local runtime, open this notebook from inside the RSI09 "
            "folder (notebooks/ lives there).\n"
            "Otherwise call setup(root='/path/to/RSI09')."
        )

    os.environ["RSI09_ROOT"] = str(found)
    src = str(found / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    if install:
        install_requirements()

    config = importlib.import_module("sgrna.config")
    importlib.reload(config)

    where = "hosted Colab VM" if _is_hosted_colab() else "local runtime"
    print(f"Running on a {where}.")
    print(config.describe())
    return found


def status() -> pd.DataFrame:
    """What exists so far, so you can see which step to run next."""
    _require_setup()
    from sgrna.io_utils import available_blocks
    from sgrna.features import ALL

    rows = [
        dict(step="0 guide index", artefact=config.GUIDE_INDEX.name,
             ready=config.GUIDE_INDEX.exists()),
        dict(step="0 QCT tables", artefact=config.QCT_TABLES.name,
             ready=config.QCT_TABLES.exists()),
        dict(step="2 matrix cache", artefact="base_X.npy",
             ready=(config.INTERIM / "base_X.npy").exists()),
    ]
    built = set(available_blocks())
    for fam in ALL:
        rows.append(
            dict(step="1 features", artefact=fam, ready=fam in built)
        )
    df = pd.DataFrame(rows)
    df["ready"] = df["ready"].map({True: "yes", False: "-"})
    return df


def _require_setup() -> None:
    if config is None:
        raise RuntimeError("Call sgrna.notebook.setup() first.")


# --------------------------------------------------------------------------
# Step 0 -- guide index
# --------------------------------------------------------------------------


def build_index(flank: int | None = None, rebuild: bool = False,
                qct_tables: bool = True) -> pd.DataFrame:
    """Locate every guide on the chromosome and recover the QCT lookup tables.

    Takes a few minutes the first time, mostly parsing the 198 MB matrix.
    Re-running returns the saved table unless `rebuild=True`.
    """
    _require_setup()
    from sgrna.build import build_guide_index, qct

    if config.GUIDE_INDEX.exists() and not rebuild:
        idx = pd.read_csv(config.GUIDE_INDEX, low_memory=False)
        print(f"Loaded existing guide index ({len(idx):,} guides). "
              "Pass rebuild=True to redo it.")
        return idx

    t0 = time.time()
    idx = build_guide_index.build(flank=flank or config.FLANK)
    idx.to_csv(config.GUIDE_INDEX, index=False)
    print(f"\nWrote {config.GUIDE_INDEX.name} in {time.time() - t0:.0f}s")

    if qct_tables:
        tables = qct.learn_lookup_tables(idx["protospacer"], qct.load_qct_frame())
        qct.save_tables(tables)
        print("\nQCT lookup tables recovered from the published matrix:")
        display_or_print(qct.table_coverage(tables))
    return idx


def guide_index() -> pd.DataFrame:
    _require_setup()
    from sgrna.io_utils import load_guide_index

    return load_guide_index()


# --------------------------------------------------------------------------
# Step 1 -- feature families
# --------------------------------------------------------------------------


def families() -> tuple[str, ...]:
    _require_setup()
    from sgrna.features import ALL

    return ALL


def build_features(which="all", rebuild: bool = False) -> pd.DataFrame:
    """Build feature families.

    `which` takes "all", "self-contained" (the families that need nothing but
    the reference genome), a single name, or a list of names. Families already
    built are skipped unless `rebuild=True`.
    """
    _require_setup()
    from sgrna.build import build_features as bf
    from sgrna.features import ALL, SELF_CONTAINED
    from sgrna.io_utils import available_blocks, load_block

    if which == "all":
        chosen = list(ALL)
    elif which in ("self-contained", "self_contained"):
        chosen = list(SELF_CONTAINED)
    elif isinstance(which, str):
        chosen = [which]
    else:
        chosen = list(which)

    unknown = [f for f in chosen if f not in ALL]
    if unknown:
        raise ValueError(f"Unknown families {unknown}. Choose from {list(ALL)}.")

    if not rebuild:
        existing = set(available_blocks())
        skipped = [f for f in chosen if f in existing]
        chosen = [f for f in chosen if f not in existing]
        if skipped:
            print(f"Already built, skipping: {', '.join(skipped)}")
    if not chosen:
        print("Nothing to build.")
    else:
        bf.run(chosen)

    rows = []
    for fam in ALL:
        try:
            block = load_block(fam)
            rows.append(dict(family=fam, features=block.shape[1] - 1,
                             guides=len(block), built="yes"))
        except FileNotFoundError:
            rows.append(dict(family=fam, features=0, guides=0, built="-"))
    return pd.DataFrame(rows)


def feature_block(family: str) -> pd.DataFrame:
    _require_setup()
    from sgrna.io_utils import load_block

    return load_block(family)


def make_diagnostic_blocks() -> dict[str, int]:
    """Decompose d_supercoiling into what it is actually measuring.

    Builds z_signal_only / z_control_only / z_abundance_all / z_sc_enrichment /
    z_dosage, which between them separate DNA read density from
    supercoiling-specific signal. See results/INTERPRETATION.md section 1.
    Pass any of those names to `ablate()` like a normal family.
    """
    _require_setup()
    from sgrna.build import make_diagnostic_blocks as mdb

    return mdb.build_all()


# --------------------------------------------------------------------------
# Step 2 -- matrix cache
# --------------------------------------------------------------------------


def cache_matrix(force: bool = False):
    """Cache the published matrix as float32, so ablations load in seconds."""
    _require_setup()
    from sgrna.build.build_matrix import build_cache

    X, meta = build_cache(force=force)
    print(f"Base matrix ready: {X.shape[0]:,} guides x {X.shape[1]:,} features")
    return X, meta


def export_csv(families_, path=None):
    """Write a merged CSV shaped like the original, for the older scripts."""
    _require_setup()
    from sgrna.build.build_matrix import export_csv as _export

    return _export(list(families_), path)


# --------------------------------------------------------------------------
# Step 3 -- ablation
# --------------------------------------------------------------------------

# Rough wall-clock per experiment on a laptop, 3 seeds x 5 folds. a_flank is
# the outlier because its 408 dense columns slow the feature selector down.
RUNTIME_HINT = {
    "a_flank": "8-12 min", "b_energy": "~2 min", "c_folding": "~2 min",
    "d_mechanics": "~2 min", "d_supercoiling": "~2 min", "e_nucleoid": "~2 min",
    "f_methylation": "~2 min", "g_transcription": "~2 min",
    "h_offtarget": "~2 min", "baseline": "~2 min",
}


def ablate(families_=None, group: str | None = None, seeds=(41, 42, 43),
           folds: int = 5, permute: bool = False, together: bool = False,
           tag: str = "notebook", n_features: int | None = None,
           force: bool = False) -> pd.DataFrame:
    """Run the leakage-safe ablation and return the results table.

    Results accumulate in `results/ablation_<tag>.csv`; an experiment already
    in that file is skipped, so a long sweep can be built up over several
    cells or several sittings.

    `group` is the one argument to think about. Leave it None for
    sequence-only families; set it to "bin100k" for anything positional
    (d_supercoiling, e_nucleoid, g_transcription), which holds out whole
    100 kb blocks of chromosome so the model cannot score well by recognising
    the locus. "gene" is the weaker version. "arc5" is the strictest -- five
    contiguous arcs of chromosome, one held out per fold, so a smooth
    ori-to-ter trend has to be extrapolated rather than interpolated; use it
    for anything going into the manuscript.
    """
    _require_setup()
    from sgrna.harness.run_ablation import run

    chosen = [] if families_ is None else (
        [families_] if isinstance(families_, str) else list(families_)
    )
    est = " + ".join(RUNTIME_HINT.get(f, "?") for f in (chosen or ["baseline"]))
    print(f"Rough estimate: {est} (plus ~2 min for the baseline if not cached)\n")

    return run(
        chosen, together=together, permute=permute, n_features=n_features,
        seeds=tuple(seeds), n_splits=folds, tag=tag, force=force,
        group_by=group,
    )


def compare_models(families_=("a_flank",), models=None, seeds=(41, 42, 43),
                   folds: int = 5, n_features: int | None = None,
                   verbose: bool = True) -> pd.DataFrame:
    """Swap the predictor, hold everything else fixed.

    Identical folds and identical selected features (XGBoost gain, top 300)
    for every model, so the comparison isolates the predictor rather than the
    selection. Returns R2, Spearman, top-decile precision, expected pick
    percentile and calibration slope; results accumulate in
    `results/model_comparison.csv`.

    Rank by the metric that matches the use case and say which -- the orders
    disagree. Rough cost: ~1 min per model per seed on the base + a_flank set.
    """
    _require_setup()
    from sgrna.harness import evaluate

    chosen = [] if families_ is None else (
        [families_] if isinstance(families_, str) else list(families_)
    )
    return evaluate.compare(
        families=tuple(chosen), models=tuple(models or evaluate.MODELS),
        seeds=tuple(seeds), n_splits=folds, n_features=n_features,
        verbose=verbose,
    )


def interpretability(families_=("a_flank",), models=("lightgbm", "random_forest"),
                     seed: int = 41, folds: int = 3,
                     n_features: int | None = None,
                     verbose: bool = True) -> pd.DataFrame:
    """How trustworthy is each model's explanation of itself?

    Unlike `compare_models`, every model here selects with *its own*
    importances, because the question is about the explanation rather than the
    prediction. Five measurements, all in `results/interpretability.csv`:

    - `stability`     Kuncheva index across folds -- 1 identical, 0 chance.
    - `method_agree`  gain vs permutation importance. Near zero means any
                      mechanism read off an importance plot is arbitrary.
    - `faithfulness`  drop the top-20 vs drop 20 at random. Zero or negative
                      means the ranking describes correlations, not reasoning.
    - `n_for_50pct`   how few features carry half the story.
    - `direction`     sign agreement with the univariate correlation.

    Slower than it looks -- permutation importance refits nothing but scores
    the model once per feature. Start with two models and 3 folds.
    """
    _require_setup()
    from sgrna.harness import evaluate

    chosen = [] if families_ is None else (
        [families_] if isinstance(families_, str) else list(families_)
    )
    return evaluate.interpretability(
        families=tuple(chosen), models=tuple(models), seed=seed,
        n_splits=folds, n_features=n_features, verbose=verbose,
    )


def headtohead(step: str = "run", arms=None, seed: int = 41, folds: int = 5,
               model: str = "lightgbm", minutes: float | None = None,
               verbose: bool = True):
    """The like-for-like comparison against the published state of the art.

    Trains on the same ~33.5k curated guides crisprHAL 2 uses, with their
    label, so the comparison is against a published number on matching rows
    rather than across different row sets.

    Three steps, each cached and resumable -- run them in order the first time:

        nb.headtohead("index")   # locate every curated guide, cut +/-1 kb flanks
        nb.headtohead("flank")   # build family A for them
        nb.headtohead("run")     # 5-fold CV, then nb.headtohead("summary")

    `minutes` stops cleanly after that long and the next call resumes, which is
    what makes this survivable in a Colab session that may drop. Expect roughly
    2.5 minutes per fold on 33,567 rows.
    """
    _require_setup()
    import time as _time

    from sgrna.compare import headtohead as h2h

    deadline = _time.time() + minutes * 60 if minutes else None
    if step == "index":
        return h2h.build_index(deadline=deadline, verbose=verbose)
    if step == "flank":
        return h2h.build_flank(deadline=deadline, verbose=verbose)
    if step == "summary":
        return h2h.summary()
    if step == "run":
        h2h.run(arms=tuple(arms) if arms else
                ("published_rows", "curated_all", "curated_flank"),
                seed=seed, n_splits=folds, model=model, deadline=deadline,
                verbose=verbose)
        return h2h.summary()
    raise ValueError("step must be one of: index, flank, run, summary")


def diagnose(what: str = "all", families_=("a_flank",), k_max: int = 2,
             verbose: bool = True):
    """Why isn't the model improving? Three measurements instead of a guess.

    - `redundancy` asks, for every new feature column, how much of it is
      already implied by the 20 nt protospacer. The published matrix encodes
      the protospacer losslessly, so a column with R2 near 1 here carries no
      new information at all -- only a different shape.
    - `headroom` correlates every column against the champion's out-of-fold
      residuals. All near zero means the feature set is exhausted and a better
      model will not help.
    - `ceiling` estimates how reproducible the label even is, from two
      independent screens of the same library.

    Returns a dict of whatever was run. See `results/RESULTS.md` §13.
    """
    _require_setup()
    from sgrna.controls import diagnose as dg

    out = {}
    if what in ("all", "ceiling"):
        out["ceiling"] = dg.ceiling(verbose=verbose)
    if what in ("all", "redundancy"):
        df = dg.redundancy(k_max=k_max, verbose=verbose)
        out["redundancy"] = df
        out["redundancy_summary"] = dg.redundancy_summary(df)
    if what in ("all", "headroom"):
        chosen = [families_] if isinstance(families_, str) else list(families_)
        out["headroom"] = dg.headroom(families=tuple(chosen), verbose=verbose)
    return out


def results(tag: str | None = None) -> pd.DataFrame:
    """Read back one results table, or all of them stacked with a `tag` column."""
    _require_setup()

    if tag:
        path = config.RESULTS / f"ablation_{tag}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        return pd.read_csv(path)

    frames = []
    for path in sorted(config.RESULTS.glob("ablation_*.csv")):
        if path.name.endswith("_settings.json"):
            continue
        df = pd.read_csv(path)
        df.insert(0, "tag", path.stem.replace("ablation_", ""))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No ablation results in {config.RESULTS}")
    return pd.concat(frames, ignore_index=True)


def summary(tag: str = "notebook") -> pd.DataFrame:
    """The columns worth looking at, sorted by effect size."""
    df = results(tag)
    cols = [c for c in ["experiment", "n_columns", "r2", "d_r2", "spearman",
                        "d_spearman", "selected_from_family", "seconds"]
            if c in df.columns]
    return df[cols].sort_values("d_spearman", ascending=False)


# --------------------------------------------------------------------------
# Display helper -- pretty in a notebook, readable in a plain console
# --------------------------------------------------------------------------


def display_or_print(obj) -> None:
    try:
        from IPython.display import display  # type: ignore

        display(obj)
    except Exception:
        print(obj.to_string(index=False) if hasattr(obj, "to_string") else obj)
