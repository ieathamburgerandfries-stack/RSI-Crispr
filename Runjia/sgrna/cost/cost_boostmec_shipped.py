"""BoostMEC as shipped: the authors' own pipeline, run unmodified in their Docker image, timed and compared with the Python version of `cost_boostmec.py`.

The pipeline (`model/boostmec-pipeline.py` of github.com/oazarate/BoostMEC, commit cdf190c) is a Python script that writes FASTA files, calls the `RNAfold` program twice (the 20-nt
guide, and the guide with the 81-nt scaffold), starts `Rscript featurize-dataset.R` (TmCalculator melting temperatures, k-mer counts, with the tidyverse) and `Rscript predict.R`
(LightGBM's R package and `rules.rds`). The authors recommend their image (`oazarate/boostmec:latest`, 1.2 GB, Ubuntu 18.04, R 4.1.0, ViennaRNA 2.4.14, Python 3.7.1); it is
linux/amd64 only, so on this Apple M5 it runs under emulation, and the times below are those of the shipped software on this laptop and not of a native x86 machine.
The emulation is gauged separately: RNAfold, the one compiled program in the pipeline, is timed natively (ViennaRNA 2.7.2 Python binding) and inside the container on the same
sequences.

What is written, all under `results/`:

  boostmec_shipped_timings.csv   one row per run of the whole pipeline: batch, rep, seconds (inside the container, from the moment the script starts to its exit)
  boostmec_shipped_stages.csv    the same pipeline with an observer that logs how long each of its four child programs ran (the script itself is not changed)
  boostmec_shipped_check.csv     predictions and features of the shipped pipeline against `cost_boostmec.py`, on the same sequences
  boostmec_shipped_rnafold.csv   RNAfold natively against inside the container
  boostmec_shipped_cohort.csv    the pipeline on all 33,567 cohort guides: its time, the released model's rho on E. coli, and the Python loader against its predictions

    ./run_ext.sh sgrna.cost.cost_boostmec_shipped inspect
    ./run_ext.sh sgrna.cost.cost_boostmec_shipped check            # 100 guides, features and predictions against the Python version
    ./run_ext.sh sgrna.cost.cost_boostmec_shipped time [--reps 5 5 3]
    ./run_ext.sh sgrna.cost.cost_boostmec_shipped stages
    ./run_ext.sh sgrna.cost.cost_boostmec_shipped rnafold
    ./run_ext.sh sgrna.cost.cost_boostmec_shipped cohort
"""
import argparse
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
RUNJIA = HERE.parents[2]
ROOT = HERE.parents[3]
RESULTS = RUNJIA / "results"
WORK = ROOT / "external_data" / "BoostMEC" / "work"          # inputs and outputs of the container runs: its own directory, apart from the clone
IMAGE = "oazarate/boostmec:latest"
BATCHES = (10, 100, 10_000)         # one guide cannot be scored: if a position-specific nucleotide column is T in every sequence, R's CSV reader makes it logical and `predict.R` stops in
                                    # lgb.convert_with_rules (8 such columns for one guide, 1 for two; the attempt is recorded in the timings file), so 10 is the smallest batch timed

OBSERVER = textwrap.dedent('''
    # sitecustomize.py: logs the duration of every child program of the pipeline; changes nothing it does
    import atexit, json, subprocess, time
    _log = []
    _Popen = subprocess.Popen
    class Popen(_Popen):
        def __init__(self, *a, **k):
            self._t0 = time.time(); self._name = None
            cmd = a[0] if a else k.get("args")
            if isinstance(cmd, (list, tuple)):
                self._name = " ".join(str(c).split("/")[-1] for c in cmd[:2])
            super().__init__(*a, **k)
        def wait(self, *a, **k):
            r = super().wait(*a, **k)
            if self._name and not getattr(self, "_logged", False):
                self._logged = True; _log.append([self._name, time.time() - self._t0])
            return r
    subprocess.Popen = Popen
    atexit.register(lambda: open("/work/stages.json", "w").write(json.dumps(_log)))
''')


def docker(cmd: str, env: dict | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    WORK.mkdir(parents=True, exist_ok=True)
    args = ["docker", "run", "--rm", "--platform", "linux/amd64", "-v", f"{WORK}:/work"]
    for k, v in (env or {}).items():
        args += ["-e", f"{k}={v}"]
    return subprocess.run(args + [IMAGE, "bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)


def repo_dir() -> str:
    p = WORK / "repo_dir.txt"
    if p.exists():
        return p.read_text().strip()
    out = docker("find / -name boostmec-pipeline.py -not -path '/proc/*' 2>/dev/null | head -1").stdout.strip()
    assert out, "boostmec-pipeline.py not found in the image"
    d = str(Path(out).parent.parent)
    p.write_text(d)
    return d


def inspect() -> None:
    info = subprocess.run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}} {{.Architecture}} {{.Size}} {{json .RepoDigests}}"], capture_output=True, text=True).stdout.strip()
    host = subprocess.run(["docker", "info", "--format", "{{.Architecture}} cpus={{.NCPU}} mem={{.MemTotal}} {{.OperatingSystem}}"], capture_output=True, text=True).stdout.strip()
    inside = docker("uname -m; cat /etc/os-release | head -2; R --version | head -1; RNAfold --version; python --version 2>&1; ls /proc/sys/fs/binfmt_misc 2>/dev/null | tr '\\n' ' '").stdout
    d = repo_dir()
    print(f"image: {info}\nhost VM: {host}\nrepo in the image: {d}\n{inside}")
    (RESULTS / "boostmec_shipped_image.txt").write_text(f"image {info}\nhost VM {host}\nrepo {d}\n{inside}")


def write_input(b: int, name: str) -> Path:
    from .cost_boostmec import thirty_mers
    s, y = thirty_mers(b)
    p = WORK / f"{name}.csv"
    pd.DataFrame({"dataset": "ecoli", "x30mer": s[:b], "efficiency": y[:b]}).to_csv(p, index=False)
    return p


def run_pipeline(name: str, observe: bool = False) -> float:
    d = repo_dir()
    env = {"PYTHONPATH": "/work/obs"} if observe else {}
    if observe:
        (WORK / "obs").mkdir(exist_ok=True)
        (WORK / "obs" / "sitecustomize.py").write_text(OBSERVER)
    cmd = (f"cd {d} && TIMEFORMAT=%R && {{ time python model/boostmec-pipeline.py --file /work/{name}.csv > /work/{name}.log 2>&1 ; }} 2> /work/{name}.elapsed; "
           f"echo exit=$? >> /work/{name}.elapsed")
    r = docker(cmd, env=env)
    t = (WORK / f"{name}.elapsed").read_text().split()
    if r.returncode != 0 or not (WORK / f"{name}-boostmec-predictions.csv").exists():
        raise RuntimeError(f"pipeline failed on {name}: {r.stderr[-800:]} {(WORK / (name + '.log')).read_text()[-1500:]}")
    return float(t[0])


def check() -> None:
    """The shipped pipeline and the Python version on the same 100 E. coli 30-mers: features column by column, and predictions."""
    from scipy.stats import spearmanr

    from ..compare import boostmec_replica as BR
    from .cost_boostmec import Released
    n = 100
    p = write_input(n, "check_100")
    sec = run_pipeline("check_100")
    feats = pd.read_csv(WORK / "check_100-with-features.csv")
    pred = pd.read_csv(WORK / "check_100-boostmec-predictions.csv")
    R = Released()
    s30 = feats["x30mer"].tolist()
    X = R.matrix(s30)
    mine = R.bst.predict(X)
    theirs = pred["boostmec_predictions"].to_numpy()
    rows = [dict(item="pipeline seconds, 100 guides", value=sec)]
    rows += [dict(item="prediction: Spearman between the shipped pipeline and the Python version", value=float(spearmanr(theirs, mine).statistic)),
             dict(item="prediction: mean absolute difference", value=float(np.mean(np.abs(theirs - mine)))), dict(item="prediction: largest absolute difference", value=float(np.max(np.abs(theirs - mine))))]
    cols = R.names
    ours = pd.DataFrame(X, columns=cols)
    for c in cols[:9]:
        rows.append(dict(item=f"feature {c}: mean absolute difference", value=float(np.mean(np.abs(feats[c].to_numpy(float) - ours[c].to_numpy())))))
    counts = cols[65:]
    rows.append(dict(item="k-mer counts (84 columns): largest absolute difference", value=float(np.max(np.abs(feats[counts].to_numpy(float) - ours[counts].to_numpy())))))
    wrong = 0
    for c in cols[9:65]:                                  # the shipped file holds the letters; the column's position gives what they must be
        k = int(c[1:].replace("mono", "").replace("di", ""))
        want = [q[k - 1] if c.endswith("mono") else q[k - 1:k + 1] for q in s30]
        wrong += int(sum(a != b for a, b in zip(feats[c].astype(str), want)))
    rows.append(dict(item="categorical columns (56): cells that differ from the sequence", value=wrong))
    rows.append(dict(item="rows compared", value=len(s30)))
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "boostmec_shipped_check.csv", index=False)
    print(d.round(5).to_string(index=False))


def time_runs(reps=(5, 5, 3)) -> None:
    rows = []
    write_input(1, "in_1")
    try:
        run_pipeline("in_1")
        rows.append(dict(batch=1, rep=0, seconds=np.nan, ms_per_guide=np.nan, note="ran"))
    except RuntimeError as e:
        why = (WORK / "in_1.log").read_text().split("Backtrace")[0].strip().replace("\n", " ")
        rows.append(dict(batch=1, rep=0, seconds=np.nan, ms_per_guide=np.nan, note="fails: " + why))
        print("  batch 1 fails as shipped:", why[:160], flush=True)
    for b, r in zip(BATCHES, reps):
        write_input(b, f"in_{b}")
        for i in range(r):
            sec = run_pipeline(f"in_{b}")
            rows.append(dict(batch=b, rep=i, seconds=sec, ms_per_guide=1000 * sec / b))
            print(f"  batch {b:6d} rep {i + 1}/{r}: {sec:7.1f} s  ({1000 * sec / b:.2f} ms per guide)", flush=True)
            pd.DataFrame(rows).to_csv(RESULTS / "boostmec_shipped_timings.csv", index=False)


def stages(b_list=BATCHES) -> None:
    rows = []
    for b in b_list:
        write_input(b, f"obs_{b}")
        sec = run_pipeline(f"obs_{b}", observe=True)
        st = json.loads((WORK / "stages.json").read_text())
        for name, t in st:
            rows.append(dict(batch=b, stage=name, seconds=t))
        rows.append(dict(batch=b, stage="whole pipeline", seconds=sec))
        print(f"  batch {b}: whole {sec:.1f} s; " + "; ".join(f"{n} {t:.1f}" for n, t in st), flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "boostmec_shipped_stages.csv", index=False)


def cohort() -> None:
    """The shipped pipeline on every guide of the cohort (its features feed `boostmec_replica.py`), its time, the released model's rho on E. coli, and the Python loader against its predictions."""
    from scipy.stats import spearmanr

    from ..compare import boostmec_replica as BR
    from .cost_boostmec import Released
    if not (WORK / "cohort-with-features.csv").exists():
        write_input(33_567, "cohort")
        run_pipeline("cohort")
    sec = float((WORK / "cohort.elapsed").read_text().split()[0])
    F, y, protos, ms = BR.build("shipped")
    theirs = pd.read_csv(WORK / "cohort-boostmec-predictions.csv").boostmec_predictions.to_numpy()
    R = Released()
    mine = R.bst.predict(R.encode(F[:, :93], F[:, 93:].astype(int)))
    row = dict(n_guides=len(y), pipeline_seconds=sec, ms_per_guide=1000 * sec / len(y), rho_released_model_on_ecoli=float(spearmanr(y, theirs).statistic),
               spearman_python_loader_vs_shipped=float(spearmanr(mine, theirs).statistic), max_abs_diff_python_loader_vs_shipped=float(np.max(np.abs(mine - theirs))))
    pd.DataFrame([row]).to_csv(RESULTS / "boostmec_shipped_cohort.csv", index=False)
    print(pd.DataFrame([row]).T.to_string(header=False))


def rnafold(n=2000) -> None:
    """RNAfold on the same n guides (and on the same n guides with the scaffold): ViennaRNA 2.7.2 natively against 2.4.14 inside the container."""
    import RNA

    from ..compare.boostmec_replica import SCAFFOLD
    from .cost_boostmec import thirty_mers
    s, _ = thirty_mers(n)
    guides = [x[4:24] for x in s]
    rows = []
    for name, seqs in (("guide (20 nt)", guides), ("guide plus scaffold (101 nt)", [g + SCAFFOLD for g in guides])):
        t0 = time.perf_counter()
        for q in seqs:
            RNA.fold(q.replace("T", "U"))
        native = time.perf_counter() - t0
        fa = WORK / "rnafold.fa"
        fa.write_text("".join(f">g{i}\n{q}\n" for i, q in enumerate(seqs)))
        out = docker("TIMEFORMAT=%R; { time RNAfold --noPS -i /work/rnafold.fa --outfile=/work/rnafold.out ; } 2> /work/rnafold.elapsed; cat /work/rnafold.elapsed").stdout.strip().split()[-1]
        rows.append(dict(workload=name, sequences=len(seqs), native_seconds=native, container_seconds=float(out), ratio=float(out) / native))
        print(f"  {name}: native {native:.2f} s, in the container {float(out):.2f} s (x{float(out) / native:.2f})", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "boostmec_shipped_rnafold.csv", index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["inspect", "check", "time", "stages", "rnafold", "cohort"])
    ap.add_argument("--reps", type=int, nargs=3, default=[5, 5, 3])
    a = ap.parse_args(argv)
    {"inspect": inspect, "check": check, "time": lambda: time_runs(tuple(a.reps)), "stages": stages, "rnafold": rnafold, "cohort": cohort}[a.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
