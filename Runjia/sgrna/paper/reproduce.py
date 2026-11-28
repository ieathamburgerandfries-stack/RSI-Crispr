"""Do the modules still write the files the audit reads? Re-run each resumable module from its stored rows and fail if any tracked result changes.

The audit (`verify_numbers`) checks that the numbers in the documents equal the numbers in the result files. This is its counterpart for the *code*: that the module which wrote a result file still writes
that file, byte for byte, when run again. It is the check that proves a merge, a rename or a refactor changed no result, and before this module it was done by hand.

Each case in `CASES` is one command (a module and its arguments). It is run against a scratch copy of `results/`, never against the tracked files, so a failing case cannot damage them. A case passes when
(1) it exits 0, (2) every file it is listed to write was actually rewritten (a command that quietly does nothing must not pass), and (3) no file of the scratch tree differs from the tracked one. A case whose output
is printed rather than written names the file the output must equal (`stdout`). Only cases that finish in minutes from stored rows and caches belong here; a module that times something, or refits a model from
scratch, cannot reproduce bytes and does not.

Two options for modules that resume from fold checkpoints (`results/folds_<tag>.csv`, which live in the main project's `results/`, outside the repository): `project=True` runs the case against a scratch copy of the
whole project root (data and external_data linked, `results/` copied), so a missing checkpoint could only ever refit into the copy; `rtol=1e-9` compares the listed files as numbers instead of bytes, for results that
are re-summarised from stored folds and so agree to floating-point rounding (about 1e-14), not to the byte. Both are named in the case, never assumed.

    ./run_audit.sh --reproduce                  # every case
    ./run_audit.sh --reproduce --only tune      # the cases whose name contains `tune`
    ./run_audit.sh --reproduce --list

To add a case: append a dict to `CASES` with the command, the result files it writes (relative to `results/`, written out in full), and run it; if it differs, it does not belong.

Report section: Part 4.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import common

RUNJIA = Path(common.OUT).parent            # the folder that holds sgrna/ and results/
RESULTS = RUNJIA / "results"

CASES = [
    dict(name="tune-analysis", cmd=["sgrna.tuning.tune_ecoli", "--analyze"], stdout="tune_stage1_effects.txt"),
    dict(name="tune-confirm-split1", cmd=["sgrna.tuning.tune_ecoli", "--confirm"], writes=["tune_confirm.csv"]),
    dict(name="tune-confirm-split2", cmd=["sgrna.tuning.tune_ecoli", "--split", "2", "--confirm"], writes=["tune_split2_confirm.csv"]),
    dict(name="ablation-family-a-at-tuned", cmd=["sgrna.cohorts.tuned_ablation", "--design", "family_a", "--summarise-only"], writes=["family_a_at_tuned_summary.csv"]),
    dict(name="ablation-local-sequence", cmd=["sgrna.cohorts.tuned_ablation", "--design", "local_sequence", "--summarise-only"], writes=["local_sequence_bound_summary.csv"]),
    dict(name="family-a-by-organism", cmd=["sgrna.compare.family_a_by_organism"], writes=["family_a_by_organism.csv"]),
    dict(name="ablation-paper-folds", cmd=["sgrna.cohorts.curated_ablation", "--paper-folds", "--summary-only"], writes=["ablation_paper_folds_summary.csv"]),
    dict(name="flank-scale-asymmetric", cmd=["sgrna.interpret.flank_scale", "--parts", "asymmetric", "--summary-only"], writes=["flank_window_tune_summary.csv"]),
    dict(name="flank-scale-windows", cmd=["sgrna.interpret.flank_scale", "--summary-only"], writes=["flank_scale_summary.csv"]),
    dict(name="human-additivity-twin", cmd=["sgrna.compare.human_baseline", "--parts", "twin-additivity", "--summarise-only"], writes=["human_additivity_twin_summary.csv", "human_additivity_twin_claim.csv"]),
    dict(name="design-check", cmd=["sgrna.design.check_design"], writes=["design_check.csv"]),
    dict(name="design-benchmark", cmd=["sgrna.design.design_benchmark"], writes=["design_benchmark.csv"]),
    dict(name="shape-paired", cmd=["sgrna.cohorts.shape_ablate", "--paired"], writes=["shape_ablation_paired.csv"]),
    dict(name="strand-families", cmd=["sgrna.compare.strand", "--parts", "families"], writes=["strand_families.csv", "strand_families_proxies.csv"], project=True, rtol=1e-9),
    dict(name="strand-families-within", cmd=["sgrna.compare.strand", "--parts", "families-within"], writes=["strand_families_within.csv"], project=True, rtol=1e-9),
    dict(name="strand-recoverable", cmd=["sgrna.compare.strand", "--parts", "recoverable"], writes=["strand_families_recoverable.csv"], project=True, rtol=1e-9),
    dict(name="eukaryote-diagonal", cmd=["sgrna.compare.eukaryote_rules", "--parts", "diagonal"], writes=["transfer_diagonal.csv"]),
    dict(name="ceiling", cmd=["sgrna.cohorts.ceiling", "--parts", "measured", "table"], writes=["curated_ceiling.json", "ceiling_parametric.csv"]),
    dict(name="curation", cmd=["sgrna.cohorts.curation", "--parts", "check", "variance", "label"], writes=["curation_check.csv", "curation_variance.csv", "label_orientation.csv"]),
    dict(name="curated-joint", cmd=["sgrna.cohorts.curated_ablation", "--joint"], writes=["curated_joint.csv"], project=True),
    dict(name="curated-ladder", cmd=["sgrna.cohorts.curated_ablation", "--ladder"], writes=["curated_ladder.csv"], project=True),
    dict(name="panel-saved-predictions", cmd=["sgrna.compare.panel", "--parts", "topk", "blend", "raw", "--no-train"], writes=["topk_panel.csv", "blend_with_hal.csv", "panel_espcas9.csv", "panel_wt.csv"]),
    dict(name="family-a-controls", cmd=["sgrna.controls.family_a_controls", "--parts", "rows", "windows", "shift", "chi"], writes=["why_inflation.csv", "window_robustness.csv", "shift_null.csv", "chi_sites.csv"], project=True, rtol=1e-9),
    dict(name="vs-crisprhal-paired", cmd=["sgrna.compare.vs_crisprhal", "--condition", "paired"], writes=["heldout_compare.csv"]),
    dict(name="vs-crisprhal-confirm", cmd=["sgrna.compare.vs_crisprhal", "--condition", "confirm"], writes=["clean_vs_hal.csv"]),
    dict(name="vs-crisprhal-gene-assemble", cmd=["sgrna.compare.vs_crisprhal", "--condition", "gene-assemble"], writes=["gene_grouped_vs_crisprhal.csv"]),
    dict(name="flank-signal", cmd=["sgrna.interpret.flank_signal", "--parts", "profile", "gate", "gc-corr", "asym-audit"],
         writes=["signal_corr.csv", "signal_decile.csv", "signal_slope.csv", "signal_target_gc.csv", "gate_test.csv", "gc_strata_corr_curated.csv", "asymmetry_strand_audit.csv"]),
    dict(name="flank-signal-ablations", cmd=["sgrna.interpret.flank_signal", "--parts", "gc-ablation", "asym-run"], writes=["gc_strata_ablation.csv", "asymmetry_expression.csv"], project=True, rtol=1e-9),
    dict(name="model-size-complexity", cmd=["sgrna.cost.model_size", "complexity"], writes=["complexity.json"]),
    dict(name="mechanism-local", cmd=["sgrna.interpret.mechanism", "--test", "distance", "bases", "human", "pairs"],
         writes=["mechanism_distance_profile.csv", "mechanism_position_base.csv", "mechanism_position_base_compare.csv", "mechanism_position_base_human.csv", "mechanism_pairs_blocks.csv", "mechanism_pairs_columns.csv"]),
    dict(name="mechanism-hypotheses", cmd=["sgrna.interpret.mechanism", "--test", "h1", "h2", "h3", "followups", "audit"],
         writes=["hypotheses_h1_contrasts.csv", "hypotheses_h1_strata.csv", "hypotheses_h2_curve.csv", "hypotheses_h2_thirds.csv", "hypotheses_h3_pam_density.csv", "hypotheses_h3_trimer_classes.csv",
                 "hypotheses_followup_absolute.csv", "hypotheses_followup_concentration.csv", "hypotheses_followup_curve20.csv", "hypotheses_followup_expression.csv", "hypotheses_followup_expression_partial.csv",
                 "hypotheses_followup_hinge.csv", "hypotheses_followup_replication.csv", "hypotheses_followup_top_genes.csv", "hypotheses_followup_within.csv", "hypotheses_audit_gate.csv", "hypotheses_audit_scale.csv"]),
    dict(name="base-enrichment", cmd=["sgrna.interpret.representation", "--base-enrichment"], writes=["base_enrichment.csv"]),
    dict(name="importance-attribution", cmd=["sgrna.interpret.importance", "--parts", "attribution"], writes=["attribution_shap.csv", "attribution_summary_shap.csv"]),
    dict(name="representation-portability", cmd=["sgrna.interpret.representation", "--portability", "--audit", "--run"], writes=["portability.csv", "portability_audit.csv"], project=True, rtol=1e-9),
    dict(name="representation-handnamed", cmd=["sgrna.interpret.representation", "--handnamed"], writes=["handnamed_split.csv"], project=True),
    dict(name="strand-flip", cmd=["sgrna.compare.strand", "--parts", "flip"], writes=["strand_flip.csv"], project=True),
]


def _digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _close_csv(a: Path, b: Path, rtol: float) -> bool:
    """Two CSVs with the same shape, columns and text, whose numbers agree within `rtol` (relative) and whose missing cells coincide."""
    import numpy as np
    import pandas as pd

    x, y = pd.read_csv(a), pd.read_csv(b)
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        return False
    for c in x.columns:
        if x[c].dtype.kind in "fi" and y[c].dtype.kind in "fi":
            u, v = x[c].to_numpy(float), y[c].to_numpy(float)
            if (np.isnan(u) != np.isnan(v)).any():
                return False
            ok = ~np.isnan(u)
            if (np.abs(u - v)[ok] > rtol * np.maximum(np.abs(v[ok]), 1e-300)).any():
                return False
        elif not (x[c].astype(str) == y[c].astype(str)).all():
            return False
    return True


def _scratch_project(tmp: Path) -> Path:
    """A project root the case may write into: data and external_data linked, results copied."""
    from .. import config

    root = tmp / "project"
    root.mkdir()
    for d in ("data", "external_data"):
        (root / d).symlink_to(config.ROOT / d)
    shutil.copytree(config.ROOT / "results", root / "results", copy_function=shutil.copy2)
    return root


def _tree(root: Path) -> dict[str, Path]:
    return {str(p.relative_to(root)): p for p in sorted(root.rglob("*")) if p.is_file()}


def run_case(case: dict) -> tuple[bool, str]:
    """(passed, one-line detail) for one case, run against a scratch copy of `results/`."""
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "results"
        shutil.copytree(RESULTS, scratch, copy_function=shutil.copy2)          # copy2 keeps mtimes, so "was it rewritten" is a comparison
        before = {k: p.stat().st_mtime_ns for k, p in _tree(scratch).items()}
        env = dict(os.environ, EXT_OUT=str(scratch))
        if case.get("project"):
            env["RSI09_ROOT"] = str(_scratch_project(Path(tmp)))
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, "-m", *case["cmd"]], cwd=RUNJIA, env=env, capture_output=True, text=True)
        secs = time.perf_counter() - t0
        if proc.returncode != 0:
            return False, f"exit {proc.returncode} after {secs:.0f} s: {(proc.stderr.strip().splitlines() or ['no stderr'])[-1][:160]}"
        problems = []
        if "stdout" in case and proc.stdout.encode() != (RESULTS / case["stdout"]).read_bytes():
            problems.append(f"its output differs from {case['stdout']}")
        for rel in case.get("writes", []):
            f = scratch / rel
            if not f.exists() or f.stat().st_mtime_ns == before.get(rel):
                problems.append(f"{rel} was not rewritten")
        changed = []
        rtol = case.get("rtol")
        for rel, p in _tree(scratch).items():
            tracked = RESULTS / rel
            if not tracked.exists() or _digest(p) == _digest(tracked):
                continue
            if rtol and rel in case.get("writes", []) and _close_csv(p, tracked, rtol):
                continue
            changed.append(rel)
        if changed:
            problems.append("differs from the tracked file: " + ", ".join(changed[:5]) + (" ..." if len(changed) > 5 else ""))
        return (not problems), (f"{secs:.0f} s" if not problems else "; ".join(problems))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", help="run the cases whose name contains one of these words")
    ap.add_argument("--list", action="store_true", help="list the cases and exit")
    a = ap.parse_args(argv)
    cases = [c for c in CASES if not a.only or any(w in c["name"] for w in a.only)]
    if a.list:
        for c in cases:
            print(f"{c['name']:30s} python -m {' '.join(c['cmd'])}")
        return 0
    failed = 0
    for c in cases:
        ok, detail = run_case(c)
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {c['name']:30s} {detail}", flush=True)
    print(f"\n{len(cases)} cases, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
