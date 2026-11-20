"""Does a convolutional net on raw sequence beat gradient boosting on 6,580
engineered columns? Same rows, same folds, so the only thing that changes is
how the sequence is represented.

Three windows, deliberately:

  proto    the 20 nt protospacer plus PAM -- the information the published
           matrix already encodes losslessly. Any gain here is architecture
           alone, because the inputs carry nothing the matrix lacks.
  f100     +/- 100 nt of genomic flank
  f250     +/- 250 nt

Comparing the first against the boosted baseline separates "we need a better
representation of the guide" from "we need more context".

Needs `torch` (CPU is fine, ~2 min per fold at +/- 100 nt) and two files that
`sgrna.diagnose` / the pipeline write:

    data/interim/sequence_context.csv   sgRNAID, protospacer, pam, up250, dn250
    data/interim/cv_folds.csv           the exact folds the tabular models used

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python -m sgrna.seqnet --flank 0                      # protospacer + PAM only
    python -m sgrna.seqnet --flank 100                    # plus 100 nt either side
    python -m sgrna.seqnet --flank 1000 --source index    # the matched test
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

try:                                   # in-project import
    from .. import config
    INTERIM = config.INTERIM
except ImportError:                     # standalone copy
    import pathlib
    INTERIM = pathlib.Path(__file__).resolve().parents[2] / "data" / "interim"

IDX = {b: i for i, b in enumerate("ACGT")}


def encode(seqs, width) -> np.ndarray:
    out = np.zeros((len(seqs), 4, width), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = str(s).upper()[:width].ljust(width, "N")
        for j, ch in enumerate(s):
            k = IDX.get(ch)
            if k is not None:
                out[i, k, j] = 1.0
    return out


def build(flank: int, source: str = "cache"):
    """Encode protospacer + PAM with `flank` nt of genomic context each side.

    `source="cache"` reads `sequence_context.csv`, which stores **250 nt** each
    side and is what the original flank-0 and flank-100 arms used. Anything wider
    needs `source="index"`, which reads the 1,000 nt flanks already present in
    `guide_index.csv`. The two agree on 13,879 of 13,880 rows (the one
    unlocated guide differs), so the arms are comparable across sources -- but
    the default is left on the cache so the published numbers stay byte-for-byte
    reproducible.
    """
    folds = pd.read_csv(INTERIM / "cv_folds.csv")
    if source == "index":
        idx = pd.read_csv(INTERIM / "guide_index.csv")
        seq = idx.rename(columns={"upstream": "up250", "downstream": "dn250"})
        seq = seq[["sgRNAID", "protospacer", "pam", "up250", "dn250"]]
        widest = int(seq["up250"].astype(str).str.len().max())
        if flank > widest:
            raise SystemExit(f"flank {flank} exceeds the {widest} nt stored in "
                             f"guide_index.csv; rebuild it with a wider flank")
    elif source == "cache":
        seq = pd.read_csv(INTERIM / "sequence_context.csv")
        if flank > 250:
            raise SystemExit(f"flank {flank} exceeds the 250 nt in "
                             f"sequence_context.csv; pass --source index")
    else:
        raise ValueError(f"unknown source {source!r}")
    seq = seq.dropna(subset=["protospacer"])
    # `s[-0:]` is the whole string, not the empty one -- so flank=0 has to be
    # handled explicitly or the "protospacer only" arm silently becomes
    # "250 nt of upstream sequence".
    if flank:
        up = seq["up250"].fillna("").astype(str).str[-flank:].str.rjust(flank, "N")
        dn = seq["dn250"].fillna("").astype(str).str[:flank].str.ljust(flank, "N")
    else:
        up = dn = pd.Series([""] * len(seq), index=seq.index)
    full = up + seq["protospacer"].astype(str) + seq["pam"].fillna("NNN").astype(str) + dn
    width = 2 * flank + 23
    X = encode(full.tolist(), width)
    ids = seq["sgRNAID"].to_numpy()
    return X, ids, folds


class SeqNet(nn.Module):
    """Two convolutions, a pooled recurrent layer, a small head.

    Kept deliberately close to the published bacterial-Cas9 architecture
    (crisprHAL) so the comparison is about representation rather than about
    who tuned harder.

    With flanks the input is mostly context: at +/- 100 nt the 23 nt that
    actually get cut are 10% of the positions, and a single mean-pooled
    encoder drowns them -- the first attempt at this scored rho 0.11, against
    0.50 for the same encoder on the guide alone. So the guide and its context
    get **separate branches**, concatenated before the head. The context
    branch can still contribute, but it can no longer swamp the guide.
    """

    def __init__(self, width: int, flank: int = 0, channels: int = 64,
                 rnn: int = 32, dropout: float = 0.2):
        super().__init__()
        self.flank = flank

        def encoder():
            return nn.Sequential(
                nn.Conv1d(4, channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(channels),
                nn.Conv1d(channels, channels, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(channels),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            )

        self.core_conv = encoder()
        self.core_rnn = nn.GRU(channels, rnn, batch_first=True,
                               bidirectional=True)
        feat = 2 * rnn

        if flank:
            self.ctx_conv = nn.Sequential(
                encoder(), nn.MaxPool1d(4), nn.Dropout(dropout))
            self.ctx_rnn = nn.GRU(channels, rnn, batch_first=True,
                                  bidirectional=True)
            feat += 2 * (2 * rnn)      # upstream and downstream

        self.head = nn.Sequential(
            nn.Linear(feat, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _encode(self, x, conv, rnn):
        h = conv(x).transpose(1, 2)
        h, _ = rnn(h)
        return h.mean(dim=1)

    def forward(self, x):
        f = self.flank
        core = x[:, :, f:x.shape[2] - f] if f else x
        parts = [self._encode(core, self.core_conv, self.core_rnn)]
        if f:
            parts.append(self._encode(x[:, :, :f], self.ctx_conv, self.ctx_rnn))
            parts.append(self._encode(x[:, :, x.shape[2] - f:],
                                      self.ctx_conv, self.ctx_rnn))
        return self.head(torch.cat(parts, dim=1)).squeeze(-1)


def run_fold(Xtr, ytr, Xva, yva, width, seed, flank=0, epochs=30, batch=128,
             lr=2e-3, patience=5, verbose=False):
    torch.manual_seed(seed)
    dev = "cpu"
    model = SeqNet(width, flank=flank).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss()

    mu, sd = ytr.mean(), ytr.std()
    ytr_s = (ytr - mu) / sd

    # A held-out slice of the training fold for early stopping -- the
    # validation fold is never touched during training.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr))
    cut = int(0.9 * len(perm))
    tr_i, es_i = perm[:cut], perm[cut:]

    Xt = torch.from_numpy(Xtr[tr_i])
    yt = torch.from_numpy(ytr_s[tr_i].astype(np.float32))
    Xe = torch.from_numpy(Xtr[es_i])
    ye = torch.from_numpy(ytr_s[es_i].astype(np.float32))
    Xv = torch.from_numpy(Xva)

    best, best_state, bad = np.inf, None, 0
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        order = torch.randperm(n)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            ev = float(lossf(model(Xe), ye))
        if ev < best - 1e-4:
            best, bad = ev, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose:
            print(f"      epoch {ep + 1:3d}  es_loss {ev:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xv).numpy() * sd + mu
    return pred


def evaluate(flank: int, seeds=(41,), n_splits: int = 5, epochs: int = 30,
             tag: str | None = None, source: str = "cache",
             verbose: bool = True) -> pd.DataFrame:
    X, ids, folds = build(flank, source=source)
    width = X.shape[2]
    tag = tag or (f"f{flank}" if flank else "proto")
    rows = []
    for seed in seeds:
        f = folds[folds.seed == seed].set_index("sgRNAID")
        f = f.reindex(ids)
        y = f["y"].to_numpy(dtype=np.float32)
        assign = f["fold"].to_numpy()
        ok = np.isfinite(y) & np.isfinite(assign)
        for fold in range(1, n_splits + 1):
            va = ok & (assign == fold)
            tr = ok & (assign != fold)
            t0 = time.time()
            pred = run_fold(X[tr], y[tr], X[va], y[va], width, seed + fold,
                            flank=flank, epochs=epochs)
            row = dict(model=f"seqnet_{tag}", flank=flank, seed=seed, fold=fold,
                       r2=r2_score(y[va], pred),
                       spearman=float(spearmanr(y[va], pred).statistic),
                       seconds=round(time.time() - t0, 1))
            rows.append(row)
            if verbose:
                print(f"  {row['model']:14s} seed {seed} fold {fold}  "
                      f"rho {row['spearman']:.4f}  r2 {row['r2']:.4f}  "
                      f"{row['seconds']:.0f}s", flush=True)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flank", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=[41])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--source", choices=["cache", "index"], default="cache",
                    help="'cache' is the 250 nt sequence_context.csv the "
                         "published arms used; 'index' reads the 1,000 nt "
                         "flanks in guide_index.csv")
    ap.add_argument("--out", default=str(INTERIM.parent.parent / "results" / "seqnet.csv"))
    args = ap.parse_args(argv)
    df = evaluate(args.flank, seeds=tuple(args.seeds), n_splits=args.folds,
                  epochs=args.epochs, source=args.source)
    import os
    hdr = not os.path.exists(args.out)
    df.to_csv(args.out, mode="a", header=hdr, index=False)
    print(df.groupby("model")[["r2", "spearman"]].mean().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
