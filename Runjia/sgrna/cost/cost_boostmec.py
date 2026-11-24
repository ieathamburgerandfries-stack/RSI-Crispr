"""What BoostMEC costs, measured with its released model, its own feature definitions and the same machine and protocol as SLICER and crisprHAL 2.

BoostMEC (Zarate et al., 2022, BMC Bioinformatics 23:446) reports no run time, memory or model size. Its repository (github.com/oazarate/BoostMEC, commit cdf190c,
cloned to `external_data/BoostMEC/repo`; nothing in it is executed, only read) holds the released LightGBM model `model/model.txt` and `model/rules.rds`, the
integer code of every category of every categorical feature in training. The shipped pipeline is a Python script that starts RNAfold and two R scripts; no R and no
RNAfold are installed here, so its 149 features are computed by `boostmec_replica.features` (ViennaRNA's Python binding for the two minimum free energies, Biopython for
the four melting temperatures) and the model is run with LightGBM's Python interface on the released file. `oracle` checks that this reproduces the paper: the two
worked examples of Supplementary Tables S2 and S3 (published predictions 49.53 and 28.42).

What is timed follows `cost_repeats.py`: raw sequence to score on one thread (here the 30-mer: 4 nt, the guide, the PAM, 3 nt), batches of 1, 100 and 10,000 guides with
300, 60 and 15 repeats; fits on the 26,853 training guides of fold 1 of SLICER's cost folds, wall and CPU seconds; load average beside every configuration.
Rows go into the same two files as the other models (`cost_repeats_inference.csv`, `cost_repeats_fit.csv`). Two fit rows are kept, because BoostMEC's procedure fixes
the number of trees by a ten-fold cross-validation with early stopping and then refits (as the paper describes): 'BoostMEC, final fit' is the refit alone (the
counterpart of SLICER's fit) and 'BoostMEC, with its tree count' adds that cross-validation (its ten fits are run in parallel; the CPU seconds are the sum of the ten).
The hyperparameter search is not repeated here (it took 985 s on ten cores in `boostmec_replica.py search`).

    ./run_ext.sh sgrna.cost.cost_boostmec oracle
    ./run_ext.sh sgrna.cost.cost_boostmec size
    ./run_ext.sh sgrna.cost.cost_boostmec inference         # one thread
    ./run_ext.sh sgrna.cost.cost_boostmec fit
"""
import os
import sys

if sys.argv[1:2] == ["inference"]:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"

import gzip
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
RUNJIA = HERE.parents[2]
ROOT = HERE.parents[3]
RESULTS = RUNJIA / "results" / "post_freeze"
REPO = ROOT / "external_data" / "BoostMEC" / "repo"
MODEL_DIR = REPO / "model"
COMMIT = "cdf190c532ad244c9d451ce22a1cf83a6203c8ef"
REPS = {1: 300, 100: 60, 10_000: 15}
PAPER_EXAMPLES = (("GTCTGCCATCTCTGATGGATGTGATGGGCA", 49.53, "S2, doench2014-Hs"), ("GGGGGGACTGTATCGACGCTGAATTGGGGG", 28.42, "S3, morenoMateos2015"))
NUC = "ACGT"


class Released:
    """The released model with the encoding it was trained with."""

    def __init__(self):
        import lightgbm as lgb

        from . import rds_read
        self.bst = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
        self.names = self.bst.feature_name()
        obj = rds_read.load(str(MODEL_DIR / "rules.rds"))
        rl = obj["values"][obj["names"].index("rules")]
        rules = {n: dict(zip(v["names"], v["values"])) for n, v in zip(rl["names"], rl["values"])}
        self.mono = [f"X{k}mono" for k in range(1, 31) if k not in (26, 27)]
        self.di = [f"X{k}di" for k in range(1, 30) if k != 26]
        assert self.names[9:37] == self.mono and self.names[37:65] == self.di, "column order of model.txt"
        # lookup tables from this repository's 0-based codes (A, C, G, T = 0..3; a dinucleotide = 4 * first + second) to the codes of the released model
        self.t_mono = np.full((28, 4), np.nan)
        self.t_di = np.full((28, 16), np.nan)
        for r, c in enumerate(self.mono):
            for j, b in enumerate(NUC):
                self.t_mono[r, j] = rules[c].get(b, np.nan)
        for r, c in enumerate(self.di):
            for j in range(16):
                self.t_di[r, j] = rules[c].get(NUC[j // 4] + NUC[j % 4], np.nan)

    def matrix(self, s30s) -> np.ndarray:
        """Raw 30-mers to the 149 columns in the order of the released model."""
        from ..compare import boostmec_replica as BR
        feats = [BR.features(s) for s in s30s]
        cont = np.asarray([f[0] for f in feats], float)
        cat = np.asarray([f[1] for f in feats], int)
        mono = self.t_mono[np.arange(28), cat[:, :28]]
        di = self.t_di[np.arange(28), cat[:, 28:]]
        return np.hstack([cont[:, :9], mono, di, cont[:, 9:]])

    def encode(self, cont: np.ndarray, cat: np.ndarray) -> np.ndarray:
        """Already computed features (continuous block and 0-based categorical block, as `boostmec_replica.build` returns them) to the released layout."""
        return np.hstack([cont[:, :9], self.t_mono[np.arange(28), cat[:, :28]], self.t_di[np.arange(28), cat[:, 28:]], cont[:, 9:]])


def thirty_mers(n=None):
    """The 30-mers of the curated cohort (4 nt before the guide, guide, PAM, 3 nt after), in the cohort's order."""
    from .. import config
    from ..build.build_curated_features import load_index
    from ..cohorts.curated_ablation import _base

    _, y, _, protos = _base()
    idx = load_index(verbose=False).set_index(config.ID_COL).reindex(protos)
    s = [(u[-4:] + p + m + d[:3]).upper() for u, p, m, d in zip(idx.upstream, idx.protospacer, idx.pam, idx.downstream)]
    return (s if n is None else s[:n]), np.asarray(y, float)


# ------------------------------------------------------------------------------------------------------------------------------------ the checks and the size
def oracle() -> None:
    R = Released()
    rows = []
    for s, paper, where in PAPER_EXAMPLES:
        pred = float(R.bst.predict(R.matrix([s]))[0])
        rows.append(dict(sequence=s, table=where, paper_prediction=paper, reproduced_prediction=round(pred, 3), difference=round(pred - paper, 3)))
        print(f"  {where}: reproduced {pred:.3f}, paper {paper}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "boostmec_oracle.csv", index=False)


def size() -> None:
    R = Released()
    d = R.bst.dump_model()
    leaves = int(sum(t["num_leaves"] for t in d["tree_info"]))
    raw = (MODEL_DIR / "model.txt").read_bytes()
    row = dict(model="BoostMEC (released)", trees=len(d["tree_info"]), nodes=np.nan, leaves=leaves, inner_nodes=leaves - len(d["tree_info"]), columns_used=np.nan, columns_selected=149,
               max_depth=12, bytes_file=len(raw), bytes_json=np.nan, parameters=np.nan, n_train=23902,
               bytes_gzip=len(gzip.compress(raw, 9)), sha256=hashlib.sha256(raw).hexdigest(), commit=COMMIT)
    print(f"  {row['trees']} trees, {leaves:,} leaves, {len(raw):,} bytes ({row['bytes_gzip']:,} gzipped)", flush=True)
    path = RESULTS / "model_size.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old[old.model != row["model"]] if len(old) else old, pd.DataFrame([row])]).to_csv(path, index=False)


# ------------------------------------------------------------------------------------------------------------------------------------ timing
def inference() -> None:
    from .cost_repeats import INF, SESSION, _timings, load1, machine

    R = Released()
    s_all, _ = thirty_mers(max(REPS))
    machine(dict(model="BoostMEC (released)", lightgbm_threads=1))
    rows = []
    for b, reps in REPS.items():
        sub = s_all[:b]

        def run():
            return R.bst.predict(R.matrix(sub), num_threads=1)
        l0 = load1()
        ts = _timings(run, reps)
        l1 = load1()
        rows += [dict(model="BoostMEC", batch=b, rep=i, ms_per_guide=1000 * t / b, session=SESSION, load1_start=l0, load1_end=l1) for i, t in enumerate(ts)]
        print(f"  BoostMEC batch {b:6d}: median {np.median([r['ms_per_guide'] for r in rows if r['batch'] == b]):.3f} ms/guide", flush=True)
    old = pd.read_csv(INF) if INF.exists() else pd.DataFrame()
    pd.concat([old[old.model != "BoostMEC"] if len(old) else old, pd.DataFrame(rows)]).to_csv(INF, index=False)


def fit(reps: int = 5) -> None:
    import lightgbm as lgb
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold

    from ..compare import boostmec_replica as BR
    from .cost_repeats import SESSION, load1, machine, save_fit

    F, y, protos, ms = BR.build()
    tr, _ = next(iter(KFold(5, shuffle=True, random_state=41).split(np.arange(len(y)))))
    Ftr, ytr = F[tr], y[tr]
    machine(dict(model="BoostMEC", n_train=len(tr)))

    def one(a, b, seed):
        c0 = time.process_time()
        m = BR._fit(Ftr[a], ytr[a], BR.HYPERPARAMETERS, BR.TREE_LIMIT, seed, valid=(Ftr[b], ytr[b]))
        return int(m.best_iteration_), time.process_time() - c0

    rows = []
    for i in range(reps):
        folds = list(KFold(BR.INNER_FOLDS, shuffle=True, random_state=2026).split(Ftr))
        l0 = load1()
        w0 = time.perf_counter()
        out = Parallel(n_jobs=BR.INNER_FOLDS)(delayed(one)(a, b, 41 + i) for a, b in folds)
        cv_wall = time.perf_counter() - w0
        n_trees = int(round(np.mean([o[0] for o in out])))
        cv_cpu = float(sum(o[1] for o in out))
        c0, w0 = time.process_time(), time.perf_counter()
        BR._fit(Ftr, ytr, BR.HYPERPARAMETERS, n_trees, 41 + i, threads=os.cpu_count())
        f_cpu, f_wall = time.process_time() - c0, time.perf_counter() - w0
        l1 = load1()
        common = dict(rep=i, n_train=len(tr), session=SESSION, load1_start=l0, load1_end=l1, trees=n_trees)
        rows.append(dict(model="BoostMEC, final fit", cpu_seconds=f_cpu, wall_seconds=f_wall, **common))
        rows.append(dict(model="BoostMEC, with its tree count", cpu_seconds=cv_cpu + f_cpu, wall_seconds=cv_wall + f_wall, **common))
        print(f"  BoostMEC fit {i + 1}/{reps}: {n_trees} trees; final fit {f_wall:.1f} s wall, {f_cpu:.0f} CPU-s; with the ten-fold tree count {cv_wall + f_wall:.1f} s wall, {cv_cpu + f_cpu:.0f} CPU-s", flush=True)
        save_fit(rows)


if __name__ == "__main__":
    {"oracle": oracle, "size": size, "inference": inference, "fit": fit}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
