"""Neural-network member of the E. coli mix, run in its own process.

PyTorch, XGBoost and LightGBM each ship an OpenMP runtime; loading them in one process on macOS stalled silently at 0% CPU.
This worker imports only numpy and torch: it reads arrays from an .npz, writes predictions to an .npy.

    python -m sgrna.ext.mlp_worker in.npz cfg.json out.npy seed
"""
import json
import sys

import numpy as np


def run(cfg, X_tr, y_tr, X_te, seed, threads=4):
    import torch
    from torch import nn
    torch.manual_seed(seed); torch.set_num_threads(threads)
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    A = torch.tensor((X_tr - mu) / sd, dtype=torch.float32); B = torch.tensor((X_te - mu) / sd, dtype=torch.float32)
    ym, ys = y_tr.mean(), y_tr.std()
    y = torch.tensor((y_tr - ym) / ys, dtype=torch.float32)
    rng = np.random.default_rng(seed); idx = rng.permutation(len(A)); nv = len(A) // 10
    va, tr = idx[:nv], idx[nv:]
    layers, d = [], A.shape[1]
    for h in cfg["hidden"]:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(cfg["dropout"])]; d = h
    net = nn.Sequential(*layers, nn.Linear(d, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best, best_state, bad = 1e9, None, 0
    for ep in range(int(cfg["epochs"])):
        net.train(); perm = rng.permutation(tr)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad(); loss = ((net(A[b]).squeeze(1) - y[b]) ** 2).mean(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(((net(A[va]).squeeze(1) - y[va]) ** 2).mean())
        if v < best - 1e-5:
            best, bad = v, 0; best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        return net(B).squeeze(1).numpy()


if __name__ == "__main__":
    z = np.load(sys.argv[1])
    cfg = json.load(open(sys.argv[2]))
    np.save(sys.argv[3], run(cfg, z["X_tr"], z["y_tr"], z["X_te"], int(sys.argv[4])))
