"""DNA shape and 3D path descriptors of the locus around a guide, by data source.

Takes the pentamer table of `dnashape` (one source at a time: mc, md, xrc) and turns each
guide's locus -- W nt upstream, the 20-mer, the 3-nt PAM, W nt downstream, all read on the
guide's strand -- into three descriptor blocks:

  mean   window means of the per-bp and per-step shape features on each arm. This is the
         scalar summary every previous shape family used; here it is the reference.
  vec    the same features position by position for the first 8 bp of each arm. The
         vector at a position is kept as a vector.
  geom   a rigid-body reconstruction of the 3D helical path. The six step parameters
         (shift, slide, rise, tilt, roll, twist) of every step are composed into a chain of
         frames (Lu & Olson 2003 convention), and the path is described in the frame of
         the base pair just 5' of the cut: positions of the path at fixed offsets,
         bend of the axis at those offsets, and the gyration tensor of each arm and of
         the whole window (components, eigenvalues, asphericity, principal-axis
         orientation), end-to-end straightness, and excess twist.

`geom` is what the "tensor-valued flank" idea in Part 3 asked for: quantities with a
direction, defined in a frame attached to the cut, that a window mean cannot express.

Controls (`control=`): `block` shuffles each arm's bases inside consecutive 25-nt blocks
(keeps composition at the 25-nt scale, destroys local structure); `arm` shuffles each arm
whole (keeps only the arm's composition).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import dnashape
from .common import INTERIM

W = 500
STEP_ORDER = ["Shift", "Slide", "Rise", "Tilt", "Roll", "HelT"]
INTRA_ORDER = ["ProT", "Buckle", "Shear", "Opening", "Stretch", "Stagger"]
CODE = np.full(256, 4, dtype=np.int8)
for i, ch in enumerate("ACGT"):
    CODE[ord(ch)] = i


def encode(seqs) -> np.ndarray:
    return CODE[np.frombuffer("".join(seqs).encode(), dtype=np.uint8)].reshape(len(seqs), -1)


def locus_codes(idx: pd.DataFrame, control: str | None = None, seed: int = 0) -> np.ndarray:
    up = encode([s[-W:] for s in idx.upstream])
    dn = encode([s[:W] for s in idx.downstream])
    mid = encode([a + b for a, b in zip(idx.protospacer, idx.pam)])
    rng = np.random.default_rng(seed)
    if control == "block":
        for arr in (up, dn):
            n, L = arr.shape
            nb = L // 25
            key = rng.random((n, nb, 25))
            part = arr[:, :nb * 25].reshape(n, nb, 25)
            arr[:, :nb * 25] = np.take_along_axis(part, np.argsort(key, axis=2), axis=2).reshape(n, -1)
    elif control == "arm":
        for arr in (up, dn):
            key = rng.random(arr.shape)
            arr[:] = np.take_along_axis(arr, np.argsort(key, axis=1), axis=1)
    elif control not in (None, "tw"):
        raise ValueError(control)
    return np.hstack([up, mid, dn])


def _table_arrays(source: str):
    t = dnashape.load()
    pent = ["".join(p) for p in __import__("itertools").product("ACGT", repeat=5)]
    t = t.loc[pent]                                         # order = base-4 index
    intra = np.stack([t[f"{source}.{f}"].to_numpy() for f in INTRA_ORDER], 1)
    mgw = t[f"{source}.MGW"].to_numpy()[:, None]
    bp = np.hstack([intra, mgw]).astype(np.float32)          # (1024, 7)
    s1 = np.stack([t[f"{source}.{f}1"].to_numpy() for f in STEP_ORDER], 1)
    s2 = np.stack([t[f"{source}.{f}2"].to_numpy() for f in STEP_ORDER], 1)
    return bp, s1.astype(np.float32), s2.astype(np.float32)


def tracks(codes: np.ndarray, source: str):
    """bp (n, L-4, 7) and step (n, L-5, 6) from the pentamer table."""
    bp_t, s1_t, s2_t = _table_arrays(source)
    n, L = codes.shape
    c = codes.astype(np.int32)
    p = ((c[:, :-4] * 256 + c[:, 1:-3] * 64 + c[:, 2:-2] * 16 + c[:, 3:-1] * 4 + c[:, 4:]))
    bad = (codes > 3)
    badp = bad[:, :-4] | bad[:, 1:-3] | bad[:, 2:-2] | bad[:, 3:-1] | bad[:, 4:]
    p = np.where(badp, 0, p)
    bp = bp_t[p]
    st1, st2 = s1_t[p], s2_t[p]
    step = 0.5 * (st2[:, :-1] + st1[:, 1:])
    return bp, step


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def step_matrices(step: np.ndarray):
    """(n, S, 6) step parameters -> rotation (n,S,3,3) and translation (n,S,3), each in the
    coordinates of the frame the step starts from."""
    shift, slide, rise, tilt, roll, tw = [step[..., i].astype(np.float64) for i in range(6)]
    tilt, roll, tw = np.deg2rad(tilt), np.deg2rad(roll), np.deg2rad(tw)
    phi = np.sqrt(tilt ** 2 + roll ** 2)
    gamma = np.arctan2(tilt, roll)
    gamma = np.where(phi < 1e-12, 0.0, gamma)
    R = _rz(tw / 2 - gamma) @ _ry(phi) @ _rz(tw / 2 + gamma)
    Rm = _rz(tw / 2 - gamma) @ _ry(phi / 2) @ _rz(gamma)
    d = np.einsum("nsij,nsj->nsi", Rm, np.stack([shift, slide, rise], -1))
    return R, d


def chain(step: np.ndarray):
    """Compose the step transforms. Returns frame rotations (n,S+1,3,3) and origins (n,S+1,3)."""
    R, d = step_matrices(step)
    n, S = R.shape[:2]
    Rf = np.empty((n, S + 1, 3, 3))
    Of = np.zeros((n, S + 1, 3))
    Rf[:, 0] = np.eye(3)
    for s in range(S):
        Of[:, s + 1] = Of[:, s] + np.einsum("nij,nj->ni", Rf[:, s], d[:, s])
        Rf[:, s + 1] = Rf[:, s] @ R[:, s]
    return Rf, Of


def _gyration(P: np.ndarray):
    """Descriptors of the gyration tensor of points P (n, m, 3)."""
    Pc = P - P.mean(1, keepdims=True)
    S = np.einsum("nmi,nmj->nij", Pc, Pc) / P.shape[1]
    w, v = np.linalg.eigh(S)                     # ascending
    l1, l2, l3 = w[:, 2], w[:, 1], w[:, 0]
    tr = np.maximum(l1 + l2 + l3, 1e-12)
    b = l1 - 0.5 * (l2 + l3)
    c = l2 - l3
    k2 = 1.5 * (l1 ** 2 + l2 ** 2 + l3 ** 2) / tr ** 2 - 0.5
    ax = np.abs(v[:, :, 2])                      # principal axis
    return np.column_stack([S[:, 0, 0], S[:, 1, 1], S[:, 2, 2], S[:, 0, 1], S[:, 0, 2], S[:, 1, 2],
                            l1, l2, l3, b, c, k2, ax[:, 0], ax[:, 1], ax[:, 2]])


GYR_NAMES = ["Sxx", "Syy", "Szz", "Sxy", "Sxz", "Syz", "lam1", "lam2", "lam3", "asph", "acyl",
             "kappa2", "ax_x", "ax_y", "ax_z"]


def describe(bp: np.ndarray, step: np.ndarray, source: str):
    """The three blocks for one chunk. bp (n,L-4,7), step (n,L-5,6); returns dict of (arrays, names)."""
    n = bp.shape[0]
    # locus coordinates -> track coordinates: bp track t <-> locus t+2 ; step s <-> between locus s+2, s+3
    up_end_bp = W - 1 - 2                    # last upstream bp (locus W-1)
    dn_start_bp = W + 23 - 2                 # first downstream bp (locus W+23)
    out = {}
    # ---- mean and vec: per-arm features read walking away from the junction
    # upstream arm walking away = reverse-complement reading: flip order, x-axis features change sign
    up_bp = bp[:, max(0, up_end_bp - (W - 4)) + 1: up_end_bp + 1][:, ::-1]
    up_step = step[:, max(0, up_end_bp - (W - 4)): up_end_bp][:, ::-1].copy()
    flipcols = [STEP_ORDER.index("Shift"), STEP_ORDER.index("Tilt")]
    up_step[..., flipcols] *= -1
    # reverse reading also swaps buckle/shear sign at the bp level
    up_bp = up_bp.copy()
    up_bp[..., [INTRA_ORDER.index("Buckle"), INTRA_ORDER.index("Shear")]] *= -1
    m = min(up_bp.shape[1], up_step.shape[1], W - 6)
    dn_bp = bp[:, dn_start_bp:dn_start_bp + m]
    dn_step = step[:, dn_start_bp:dn_start_bp + m]
    up_bp, up_step = up_bp[:, :m], up_step[:, :m]
    arms = {"up": np.concatenate([up_bp, up_step], 2), "dn": np.concatenate([dn_bp, dn_step], 2)}
    fnames = [f"{f}" for f in INTRA_ORDER] + ["MGW"] + STEP_ORDER
    # mean windows
    wins = [(0, 25), (25, 100), (100, 250), (250, m)]
    cols, names = [], []
    for arm, arr in arms.items():
        for a, b in wins:
            cols.append(np.nanmean(arr[:, a:b], 1))
            names += [f"{arm}.w{a}_{b}.{f}" for f in fnames]
    out["mean"] = (np.hstack(cols), names)
    cols, names = [], []
    for arm, arr in arms.items():
        for k in range(8):
            cols.append(arr[:, k])
            names += [f"{arm}.p{k + 1}.{f}" for f in fnames]
    out["vec"] = (np.hstack(cols), names)
    # ---- geom: one chain across the whole locus
    Rf, Of = chain(step.copy())              # frames for track bps 0..S  (S = L-5 steps => L-4 bps)
    tc = W - 2 + 16 - 0                      # track index of the bp 5' of the cut (locus W+16)
    tc = (W + 16) - 2
    Rc, Oc = Rf[:, tc], Of[:, tc]
    RcT = np.transpose(Rc, (0, 2, 1))
    P = np.einsum("nij,ntj->nti", RcT, Of - Oc[:, None])            # path in the cut frame
    Z = np.einsum("nij,ntj->nti", RcT, Rf[:, :, :, 2])               # axis direction in the cut frame
    cols, names = [], []
    for k in (-400, -200, -100, -50, -25, -12, 12, 25, 50, 100, 200, 400):
        t = tc + k
        cols.append(P[:, t]); names += [f"path.k{k:+d}.{a}" for a in "xyz"]
    for k in (-400, -100, -25, 25, 100, 400):
        t = tc + k
        cols.append(Z[:, t]); names += [f"axis.k{k:+d}.{a}" for a in "xyz"]
    for win in (100, 250, 499 if W >= 500 else W - 1):
        for arm, sl in (("dn", slice(tc, tc + win + 1)), ("up", slice(tc - win, tc + 1)),
                        ("both", slice(tc - win, tc + win + 1))):
            g = _gyration(P[:, sl])
            cols.append(g); names += [f"gyr.{arm}{win}.{nm}" for nm in GYR_NAMES]
        e_dn = np.linalg.norm(P[:, tc + win], axis=1)
        e_up = np.linalg.norm(P[:, tc - win], axis=1)
        cols.append(np.column_stack([e_dn / (3.4 * win), e_up / (3.4 * win)]))
        names += [f"straight.dn{win}", f"straight.up{win}"]
        tw_dn = step[:, tc:tc + win, STEP_ORDER.index("HelT")].sum(1) / 360 - win / 10.5
        tw_up = step[:, tc - win:tc, STEP_ORDER.index("HelT")].sum(1) / 360 - win / 10.5
        cols.append(np.column_stack([tw_dn, tw_up]))
        names += [f"xtwist.dn{win}", f"xtwist.up{win}"]
    out["geom"] = (np.column_stack(cols), names)
    return out


def twist_offset(source: str, target: float = 34.3) -> float:
    """Shift that makes a source's mean step twist equal the B-DNA consensus (10.5 bp/turn = 34.3 deg)."""
    t = dnashape.load()
    return float(target - t[[f"{source}.HelT1", f"{source}.HelT2"]].to_numpy().mean())


def compute(idx: pd.DataFrame, source: str, control: str | None = None, chunk: int = 1500,
            seed: int = 0, verbose: bool = True):
    """control: None, 'block'/'arm' (shuffled flanks), or 'tw' (real flanks, mean twist recentred)."""
    codes = locus_codes(idx, control, seed)
    off = twist_offset(source) if control == "tw" else 0.0
    blocks = {"mean": [], "vec": [], "geom": []}
    names = {}
    for a in range(0, len(idx), chunk):
        bp, step = tracks(codes[a:a + chunk], source)
        if off:
            step = step.copy(); step[..., STEP_ORDER.index("HelT")] += off
        d = describe(bp, step, source)
        for k, (arr, nm) in d.items():
            blocks[k].append(np.asarray(arr, dtype=np.float32)); names[k] = nm
        if verbose and (a // chunk) % 5 == 0:
            print(f"   {source} {control or 'real'}: {a + chunk:,}/{len(idx):,}", flush=True)
    return {k: (np.vstack(v), names[k]) for k, v in blocks.items()}


def composition(idx: pd.DataFrame, control: str | None = None, seed: int = 0):
    """Plain base composition in the same arm windows as the shape `mean` block (the reference
    that asks whether the shape window means carry anything beyond letters).
    Returns (array, names): A/C/G/T fractions plus GC for each arm x window."""
    codes = locus_codes(idx, control, seed)
    n = codes.shape[0]
    up = codes[:, :W][:, ::-1]                    # walking away from the junction (complement not needed for GC; used for A/C/G/T of the rc strand)
    up = np.where(up < 4, 3 - up, up)             # reverse-complement reading
    dn = codes[:, W + 23:]
    cols, names = [], []
    for arm, arr in (("up", up), ("dn", dn)):
        for a, b in [(0, 25), (25, 100), (100, 250), (250, W - 6)]:
            seg = arr[:, a:b]
            fr = [(seg == k).mean(1) for k in range(4)]
            cols += fr + [fr[1] + fr[2]]
            names += [f"{arm}.w{a}_{b}.frac_{x}" for x in "ACGT"] + [f"{arm}.w{a}_{b}.GC"]
    return np.column_stack(cols).astype(np.float32), names


def dinuc_composition(idx: pd.DataFrame, control: str | None = None, seed: int = 0):
    """Dinucleotide fractions (16) per arm x window: the composition statistic that a sum of step
    parameters is a function of."""
    codes = locus_codes(idx, control, seed)
    up = codes[:, :W][:, ::-1]
    up = np.where(up < 4, 3 - up, up)
    dn = codes[:, W + 23:]
    cols, names = [], []
    for arm, arr in (("up", up), ("dn", dn)):
        d = arr[:, :-1].astype(np.int16) * 4 + arr[:, 1:]
        for a, b in [(0, 25), (25, 100), (100, 250), (250, W - 7)]:
            seg = d[:, a:b]
            for k in range(16):
                cols.append((seg == k).mean(1))
                names.append(f"{arm}.w{a}_{b}.di_{'ACGT'[k // 4]}{'ACGT'[k % 4]}")
    return np.column_stack(cols).astype(np.float32), names
