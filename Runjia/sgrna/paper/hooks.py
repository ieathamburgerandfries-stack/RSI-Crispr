"""The audit's whole coupling to experiment code, in one table: every module, function or constant of `sgrna/` that a check imports to recompute a number.

The number checks read result files by name, and a file name survives a merge or a move. What can break is an audit file that imports an experiment's private helper (`_base`, `_seal`,
`_shifted_block`) from where it used to live. So no audit file imports experiment code directly: each does `from .hooks import _seal`, and this table says where `_seal` lives. When a merge
or a move changes that, this table is the one place to edit, and the audit's last check (`hooks.check()`) names every entry that no longer resolves instead of failing in the middle of a run.

    ./run_ext.sh sgrna.paper.hooks            # list the entries that do not resolve, exit 1 if any

Report section: Part 4.
"""
from __future__ import annotations

import importlib

# audit-facing name -> (module under sgrna, attribute of it; None when the name is the module itself)
HOOKS = {
    "_b_numbers": ("build.build_features", '_b_numbers'),
    "load_curated_index": ("build.build_features", 'load_curated_index'),
    "CACHE_META": ("build.build_matrix", 'CACHE_META'),
    "CACHE_X": ("build.build_matrix", 'CACHE_X'),
    "load_dataset": ("build.build_matrix", 'load_dataset'),
    "featurise": ("build.featurise", None),
    "genome": ("build.genome", None),
    "load_base": ("harness.curated", 'load_base'),
    "load_block": ("harness.curated", 'load_block'),
    "load_groups": ("harness.curated", 'load_groups'),
    "MIN_BIG": ("compare.eukaryote_rules", 'MIN_BIG'),
    "eukaryote_rules": ("compare.eukaryote_rules", None),
    "build": ("compare.transfer", 'build'),
    "transfer": ("compare.transfer", None),
    "SEAL_FRACTION": ("controls.sealed", 'SEAL_FRACTION'),
    "_seal": ("controls.sealed", '_seal'),
    "_shifted_block": ("controls.shiftnull", '_shifted_block'),
    "a_flank": ("features.a_flank", None),
    "dnashape": ("features.dnashape", None),
    "BLOCK": ("harness.cohort_groups", 'BLOCK'),
    "evaluate": ("harness.evaluate", None),
    "_impute": ("harness.run_ablation", '_impute'),
    "make_selector": ("harness.run_ablation", 'make_selector'),
    "reduced_columns": ("interpret.faithfulness", 'reduced_columns'),
    "_frame_curated": ("interpret.gcstrata", '_frame_curated'),
    "_strata": ("interpret.gcstrata", '_strata'),
    "_edit": ("interpret.interp_extra", '_edit'),
    "interp_extra": ("interpret.interp_extra", None),
    "vs_crisprhal_interpret": ("interpret.vs_crisprhal_interpret", None),
    "SPLIT_SEED": ("tuning.tune_ecoli", 'SPLIT_SEED'),
    "stage1_table": ("tuning.tune_ecoli", 'stage1_table'),
    "tune_ecoli": ("tuning.tune_ecoli", None),
}


def _resolve(name: str):
    mod, attr = HOOKS[name]
    m = importlib.import_module(f"sgrna.{mod}")
    return m if attr is None else getattr(m, attr)


def __getattr__(name: str):
    """`from .hooks import x` imports the target the first time it is asked for, so the heavy modules stay unloaded until a check needs them."""
    if name in HOOKS:
        return _resolve(name)
    raise AttributeError(f"{__name__} has no hook {name!r}; add it to HOOKS")


def check() -> list[str]:
    """Entries of HOOKS that do not resolve, as 'name -> module.attr: error'."""
    bad = []
    for name, (mod, attr) in HOOKS.items():
        try:
            _resolve(name)
        except Exception as exc:                       # an import error, a missing attribute, anything that stops a check from running
            bad.append(f"{name} -> {mod}{'.' + attr if attr else ''}: {type(exc).__name__}: {exc}")
    return bad


def main() -> int:
    bad = check()
    for b in bad:
        print("unresolved:", b)
    print(f"{len(HOOKS)} hooks, {len(bad)} unresolved")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
