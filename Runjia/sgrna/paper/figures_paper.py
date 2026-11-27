"""Figures for the point-form paper, every one drawn from stored result files (no schematics).

Each figure shows the data its caption's claim rests on: the reach of the flank signal (displacement and window size), what the signal is (the step, the slope
by the guide's last base, the resolution of the composition), transfer between datasets (the full matrix, SLICER against crisprHAL 2, the position-20 rule),
the measured replicate agreement and what it caps, the input-width experiment, and the cost distributions.

    python -m sgrna.paper.figures_paper --fig all
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from sgrna import config

from ..common import OUT
from .figures import BLUE, DIVERGING, DOUBLE, GRID, INK, INK2, ORANGE, SINGLE, _panel, _save, _style

R = OUT
LABEL = {"chari2015Train293T": "Chari 293T", "doench2014-Hs": "Doench 2014 Hs", "doench2014-Mm": "Doench 2014 Mm", "doench2016_hg19": "Doench 2016",
         "hart2016-Hct1161lib1Avg": "Hart HCT116", "hart2016-HelaLib1Avg": "Hart HeLa", "morenoMateos2015": "Moreno-Mateos zf",
         "wang2015_hg19": "Wang 2015", "xu2015TrainHl60": "Xu HL60", "xu2015TrainKbm7": "Xu KBM7", "xu2015TrainMEsc": "Xu mESC", "E. coli": "E. coli"}
GREEN = "#3a9a5b"


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def _sd_bars(ax, x, y, e, color, marker, label, **kw):
    ax.errorbar(x, y, yerr=e, color=color, marker=marker, ms=3.4, lw=0.9, capsize=1.6, elinewidth=0.6, label=label, **kw)


# ------------------------------------------------------------------------------------------------ Figure 1: reach
def _seed41_gains(prefix_part):
    """Gain over the 20-mer per arm of one part of `flank_scale.csv`, seed 41 only (the one seed every arm of Figure 1 has): mean and sd over the five folds."""
    fr = pd.read_csv(R / "flank_scale.csv")
    fr = fr[fr.seed == 41]
    base = fr[fr.arm == "base"].set_index("fold").rho
    rows = []
    for arm, g in fr[fr.part == prefix_part].groupby("arm", sort=False):
        d = g.set_index("fold").rho - base.reindex(g.set_index("fold").index)
        rows.append(dict(arm=arm, gain=float(d.mean()), gain_sd_over_folds=float(d.std(ddof=1))))
    return pd.DataFrame(rows)


def _tune_symmetric_gains():
    """Gain over the 20-mer of the symmetric arms (same window on both sides) that the `asymmetric` part of `flank_scale.py` added to its first sweep; seed 41, five folds."""
    f = R / "flank_window_tune.csv"
    if not f.exists():
        return pd.DataFrame(columns=["arm", "gain", "gain_sd_over_folds", "W"])
    t = pd.read_csv(f)
    t = t[t.seed == 41]
    base = t[t.arm == "base"].set_index("fold").rho
    rows = []
    for arm, g in t[(t.arm != "base") & (t.up == t.down)].groupby("arm", sort=False):
        d = g.drop_duplicates("fold").set_index("fold").rho - base
        rows.append(dict(arm=f"gc W={int(g.up.iloc[0])}", gain=float(d.mean()), gain_sd_over_folds=float(d.std(ddof=1)), W=float(g.up.iloc[0])))
    return pd.DataFrame(rows)


def fig_reach():
    _style()
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE * 0.92, 2.25))
    d = _seed41_gains("displacement")
    d["shift"] = d.arm.str.extract(r"(\d+)$").astype(float)
    d = d.sort_values("shift")
    x = np.where(d["shift"] == 0, 30, d["shift"])
    _sd_bars(ax[0], x, d.gain, d.gain_sd_over_folds, BLUE, "o", "family A read from DNA the given distance away")
    full = d[d["shift"] == 0].gain.iloc[0]
    ax[0].axhline(0, color=INK2, lw=0.5)
    ax[0].set_xscale("log")
    ax[0].set_xticks([30, 100, 300, 1000, 3000, 1e4, 1e5, 1e6])
    ax[0].set_xticklabels(["0\n(aligned)", "100", "300", "1 kb", "3 kb", "10 kb", "100 kb", "1 Mb"], fontsize=6.3)
    ax[0].set_xlabel("distance from the guide to the DNA its flank features are read from", fontsize=6.6)
    ax[0].set_ylabel("gain in ρ over the 20-mer alone")
    ax[0].annotate(f"aligned: +{full:.3f}", (x[0], full), xytext=(60, full - 0.01), fontsize=6.5, color=INK)
    _clean(ax[0])
    w = _seed41_gains("window")
    w["W"] = w.arm.str.extract(r"W=(\d+)").astype(float)
    wt = _tune_symmetric_gains()                                         # the finer windows of flank_scale.py, part asymmetric (75 to 400 nt), same folds and model
    wt = wt[~wt.W.isin(w[w.arm.str.startswith("gc ")].W)]
    w = pd.concat([w, wt], ignore_index=True)
    q = w[w.arm.str.startswith("gc ")].sort_values("W")
    _sd_bars(ax[1], q.W, q.gain, q.gain_sd_over_folds, BLUE, "o", "GC fraction of each side (2 columns)")
    ax[1].axhline(full, color=INK2, lw=0.6, ls=(0, (4, 2)))
    ax[1].text(11, full + 0.004, f"full family A, 348 columns: +{full:.3f}", fontsize=6.3, color=INK2)
    ax[1].set_xscale("log")
    ax[1].set_xticks([10, 25, 50, 100, 250, 500, 1000, 2000, 5000])
    ax[1].set_xticklabels(["10", "25", "50", "100", "250", "500", "1k", "2k", "5k"])
    ax[1].set_xlabel("window length on each side of the protospacer (nt)")
    ax[1].set_ylabel("gain in ρ over the 20-mer alone")
    ax[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=6.3)
    _clean(ax[1])
    _panel(ax[0], "a")
    _panel(ax[1], "b")
    fig.tight_layout()
    return _save(fig, "p_fig1_reach")


# ------------------------------------------------------------------------------------------------ Figure 2: the signal
def fig_signal():
    _style()
    dec = pd.read_csv(R / "signal_decile.csv")
    sl = pd.read_csv(R / "signal_slope.csv")
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE * 0.72, 2.25))
    ax[0].errorbar(dec.decile, dec.mean_label, yerr=[dec.mean_label - dec.lo, dec.hi - dec.mean_label], color=BLUE, marker="o", ms=3.4, lw=0.9, capsize=1.6, elinewidth=0.6)
    ax[0].axhline(0, color=INK2, lw=0.5)
    ax[0].set_xticks(range(1, 11))
    ax[0].set_xlabel("decile of flank GC (1 = most AT-rich)")
    ax[0].set_ylabel("mean cutting score")
    ax[0].annotate(f"{dec.mean_label.iloc[0]:.2f}".replace("-", "−"), (1, dec.mean_label.iloc[0]), xytext=(1.5, dec.mean_label.iloc[0] + 0.15), fontsize=6.5)
    _clean(ax[0])
    cols = {"A": BLUE, "C": GREEN, "G": ORANGE, "T": INK2}
    mk = {"A": "o", "C": "^", "G": "s", "T": "D"}
    for b in "ACGT":
        q = sl[sl.base == b]
        r = q.iloc[0]
        ax[1].plot(q.flank_gc, q.mean_label, color=cols[b], marker=mk[b], ms=3.2, lw=0.9)
        ax[1].text(q.flank_gc.iloc[-1] + 0.004, q.mean_label.iloc[-1], f"{b}  (ρ {r.spearman:.2f})", color=cols[b] if b != "T" else INK, fontsize=6.3, va="center")
    ax[1].set_xlim(0.41, 0.66)
    ax[1].set_xlabel("flank GC (quintile means)")
    ax[1].set_ylabel("mean cutting score")
    ax[1].set_title("by the base at guide position 20", fontsize=6.4)
    _clean(ax[1])
    for a, l, dx in zip(ax, "ab", (-0.17, -0.17)):
        _panel(a, l, dx=dx)
    fig.tight_layout()
    return _save(fig, "p_fig2_signal")


# ------------------------------------------------------------------------------------------------ Figure 3: transfer
def fig_signal_appendix():
    """Appendix Figure A1: which composition works (a), how finely the flank must be described (b)."""
    _style()
    fr = pd.read_csv(R / "flank_scale.csv")
    fr = fr[fr.seed == 41]
    base = fr[fr.arm == "base"].set_index("fold").rho

    def curve(prefix, parts):
        rows = []
        for arm, g in fr[fr.part.isin(parts) & fr.arm.str.startswith(prefix + " W=")].groupby("arm", sort=False):
            dd = g.set_index("fold").rho - base.reindex(g.set_index("fold").index)
            rows.append((float(arm.split("W=")[1]), float(dd.mean()), float(dd.std(ddof=1))))
        return pd.DataFrame(sorted(rows), columns=["W", "gain", "sd"])

    gc = pd.concat([curve("gc", ("window",)), _tune_symmetric_gains().rename(columns={"gain_sd_over_folds": "sd"})[["W", "gain", "sd"]]]).drop_duplicates("W").sort_values("W")
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE * 0.72, 2.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for name, dfc, col, mk in (("GC (C+G against A+T): 2 columns", gc, BLUE, "o"), ("purine (A+G against C+T): 2 columns", curve("purine", ("partition",)), ORANGE, "s"),
                               ("amino (A+C against G+T): 2 columns", curve("amino", ("partition",)), GREEN, "^"), ("A, C, G, T separately: 8 columns", curve("base4", ("window",)), INK2, "D")):
        _sd_bars(ax[0], dfc.W, dfc.gain, dfc.sd, col, mk, name)
    ax[0].axhline(0, color=INK2, lw=0.5)
    ax[0].set_xscale("log")
    ax[0].set_xticks([10, 50, 250, 1000, 5000])
    ax[0].set_xticklabels(["10", "50", "250", "1k", "5k"])
    ax[0].set_xlabel("window length on each side (nt)")
    ax[0].set_ylabel("gain in ρ over the 20-mer alone")
    ax[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.06), fontsize=5.6, handlelength=1.6)
    _clean(ax[0])
    res = pd.read_csv(R / "novel_resolution.csv").set_index("arm")
    pts = sorted((int(a.split()[2]), float(res.loc[a, "rho"]), float(res.loc[a, "sd"])) for a in res.index if "binned" in a)
    n, r, e = (np.array(z) for z in zip(*pts))
    ax[1].errorbar(n, r, yerr=e, color=BLUE, marker="o", ms=3, lw=0.9, capsize=1.6, elinewidth=0.6)
    ref = float(res.loc["base + family A comp block (24 cols)", "rho"])
    ax[1].axhline(ref, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax[1].text(n[-1], 0.6415, "dashed: the 24 columns of the model (0.673)", fontsize=5.6, color=INK2, va="bottom", ha="right")
    ax[1].set_xscale("log")
    ax[1].set_xticks(n)
    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].set_xlabel("columns (GC, purine; equal bins of the kb)")
    ax[1].set_ylabel("Spearman ρ")
    _clean(ax[1])
    for a, l, dx in zip(ax, "ab", (-0.2, -0.2)):
        _panel(a, l, dx=dx)
    fig.tight_layout()
    return _save(fig, "p_figA1_signal")


def fig_transfer():
    _style()
    t = pd.read_csv(R / "eukaryote_transfer.csv")
    t = t[t.model.str.startswith("pipeline")]
    t = pd.concat([t[["trained_on", "scored_on", "rho"]], pd.read_csv(R / "transfer_to_ecoli.csv")[["trained_on", "scored_on", "rho"]]], ignore_index=True)
    cols = list(LABEL)
    rows = ["E. coli"] + [k for k in LABEL if k != "E. coli"]
    M = np.full((len(rows), len(cols)), np.nan)
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            q = t[(t.trained_on == a) & (t.scored_on == b)]
            if len(q):
                M[i, j] = q.rho.iloc[0]
    k = pd.read_csv(R / "transfer_vs_crisprhal_kingdom.csv")
    fig = plt.figure(figsize=(DOUBLE, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=0, vmin=-0.3, vmax=0.8), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=4.8, color=INK if abs(M[i, j]) < 0.55 else "white",
                        fontweight="bold" if rows[i] == cols[j] else "normal")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([LABEL[c] for c in cols], rotation=60, ha="right", fontsize=5.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL[r] for r in rows], fontsize=5.6)
    ax.set_xlabel("scored on")
    ax.set_ylabel("trained on")
    ax.set_title("ρ of a model trained on the row, scored on the column\n(bold diagonal: five-fold CV on the dataset itself)", fontsize=6.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=5.5)
    ax.axhline(0.5, color="white", lw=1.2)
    ax.axvline(len(cols) - 1.5, color="white", lw=1.2)
    bx = fig.add_subplot(gs[1])
    k = k.sort_values("slicer_base")
    y = np.arange(len(k))
    bx.hlines(y, k.crisprhal_core23, k.slicer_base, color=GRID, lw=1.2)
    bx.plot(k.slicer_base, y, "o", color=BLUE, ms=3.6, label="SLICER, 20-mer")
    bx.plot(k.crisprhal_core23, y, "s", color=ORANGE, ms=3.4, label="crisprHAL 2, 23 nt")
    bx.axvline(0, color=INK2, lw=0.5)
    bx.set_yticks(y)
    bx.set_yticklabels([LABEL[d] for d in k.dataset], fontsize=5.6)
    bx.set_xlabel("ρ on the eukaryotic dataset,\ntrained on E. coli only", fontsize=6.6)
    bx.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=5.8, handletextpad=0.2, columnspacing=1.0)
    _clean(bx)
    bx.grid(axis="x", color=GRID, lw=0.5)
    bx.grid(axis="y", visible=False)
    _panel(ax, "a", dx=-0.32)
    _panel(bx, "b", dx=-0.62)
    return _save(fig, "p_fig3_transfer")


def fig_rule():
    _style()
    c = pd.read_csv(R / "eukaryote_contrast_g_minus_t.csv")
    c = c[~c.underpowered.astype(bool)].copy()
    from .report_checks_R4 import _contrast_pos20
    rng = np.random.default_rng(41)
    extra = []
    for grp, name in (("TevSpCas9", "C. rodentium"), ("eSpCas9", "E. coli, eSpCas9")):
        e, lo, hi, n = _contrast_pos20(grp, rng)
        extra.append(dict(group=name, g_minus_ac_pos20=e, g_minus_ac_lo=lo, g_minus_ac_hi=hi, underpowered=False))
    c = pd.concat([c, pd.DataFrame(extra)], ignore_index=True)
    c["lab"] = c.group.map(LABEL).fillna(c.group)
    c = c.sort_values("g_minus_ac_pos20")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    y = np.arange(len(c))
    col = [ORANGE if g in ("E. coli", "C. rodentium", "E. coli, eSpCas9") else BLUE for g in c.group]
    ax.hlines(y, c.g_minus_ac_lo, c.g_minus_ac_hi, color=col, lw=1.0)
    ax.scatter(c.g_minus_ac_pos20, y, c=col, s=14, zorder=3)
    ax.axvline(0, color=INK2, lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(c.lab, fontsize=5.8)
    ax.set_xlabel("effect of G at guide position 20, against the mean of A and C\n(SD of the label, 95% bootstrap interval)")
    _clean(ax)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, "p_fig4_rule")


# ------------------------------------------------------------------------------------------------ Figure 5: ceiling
def _fold_scores():
    """rho per whole-gene fold (seed 41, the folds crisprHAL 2 was run on): crisprHAL 2 at its published batch, BoostMEC with its released hyperparameters, SLICER default and tuned."""
    hal = pd.read_csv(R / "crisprhal_gene_grouped_fcolab_b1024.csv")
    hal = hal[hal.model == "crisprHAL 2"].sort_values("fold").spearman.to_numpy()
    bm = pd.read_csv(R / "boostmec_released.csv")
    bm = bm[bm.seed == 41].sort_values("fold").rho.to_numpy()
    sl = pd.read_csv(R / "gene_grouped_slicer.csv")
    return {"crisprHAL 2": hal, "BoostMEC": bm, "SLICER default": sl[sl.model == "SLICER default"].sort_values("fold").spearman.to_numpy(),
            "SLICER tuned": sl[sl.model == "SLICER tuned"].sort_values("fold").spearman.to_numpy()}


def fig_ceiling():
    _style()
    d = pd.read_csv(config.EXTERNAL / "crisprHAL" / "rawData" / "E_coli_WTSpCas9_counts.tsv", sep="\t")
    tot = d[["Cas1_S", "Cas2_S", "dCas1_NS", "dCas2_NS"]].sum()        # counts per million over the whole library, then the control-count filter, as in cohorts/ceiling.py
    d = d[(d.dCas1_NS + d.dCas2_NS) / 2 >= 73]
    cpm = lambda c: c / tot[c.name] * 1e6
    u = np.log2((cpm(d.Cas1_S) + 0.5) / (cpm(d.dCas1_NS) + 0.5))
    v = np.log2((cpm(d.Cas2_S) + 0.5) / (cpm(d.dCas2_NS) + 0.5))
    from scipy.stats import spearmanr
    rho = float(spearmanr(u, v).statistic)
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE * 0.9, 2.45), gridspec_kw=dict(width_ratios=[1, 1.15]))
    hb = ax[0].hexbin(-u, -v, gridsize=55, bins="log", cmap="Blues", mincnt=1, linewidths=0.1)
    ax[0].set_xlabel("cutting score, replicate pair 1")
    ax[0].set_ylabel("cutting score, replicate pair 2")
    ax[0].text(0.04, 0.93, f"Spearman ρ = {rho:.3f}\n{len(d):,} guides", transform=ax[0].transAxes, fontsize=6.5, va="top")
    cb = fig.colorbar(hb, ax=ax[0], fraction=0.045, pad=0.02)
    cb.set_label("guides (log)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    rel = 2 * rho / (1 + rho)
    ceil = float(np.sqrt(rel))
    sc = _fold_scores()
    order = [("crisprHAL 2", ORANGE), ("BoostMEC", PURPLE), ("SLICER default", BLUE), ("SLICER tuned", GREEN)]
    x = np.arange(len(order))
    mean = np.array([sc[k].mean() for k, _ in order])
    sd = np.array([sc[k].std(ddof=1) for k, _ in order])
    ax[1].bar(x, mean, yerr=sd, color=[c for _, c in order], width=0.6, capsize=2.5, error_kw=dict(lw=0.7))
    for xi, m_, s_ in zip(x, mean, sd):
        ax[1].text(xi, m_ + s_ + 0.012, f"{m_:.3f}", ha="center", fontsize=6.3)
    ax[1].axhline(0.7, color=INK2, lw=0.8, ls=(0, (1.5, 2)))
    ax[1].axhline(ceil, color=GREEN, lw=0.8, ls=(0, (1.5, 2)))
    from matplotlib.lines import Line2D
    ax[1].legend(handles=[Line2D([], [], color=GREEN, lw=0.9, ls=(0, (1.5, 2)), label=f"attainable ceiling of the released score ({ceil:.3f})"),
                          Line2D([], [], color=INK2, lw=0.9, ls=(0, (1.5, 2)), label="the field's ceiling (0.7)")],
                 frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=6.2, handlelength=2.4)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([k.replace(" ", "\n", 1) for k, _ in order], fontsize=6.3)
    ax[1].set_ylim(0, 1.22)
    ax[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1].set_ylabel("Spearman ρ, whole genes held out")
    _clean(ax[1])
    _panel(ax[0], "a", dx=-0.3)
    _panel(ax[1], "b", dx=-0.2)
    fig.tight_layout()
    return _save(fig, "p_fig5_ceiling")


# ------------------------------------------------------------------------------------------------ Figure 6: input width
def fig_width():
    _style()
    a = pd.read_csv(R / "crisprhal_wide_window_folds.csv")
    h = pd.read_csv(R / "headtohead.csv")
    h = h[h.seed == 41].pivot_table(index="fold", columns="arm", values="spearman")
    order = [("core_23", 23, "curated_all"), ("native_378", 378, "curated_matched"), ("mid_1023", 1023, None), ("wide_2023", 2023, "curated_flank")]
    fig, ax = plt.subplots(figsize=(SINGLE * 1.15, 2.4))
    xs = np.array([o[1] for o in order], float)
    hal = a.pivot_table(index="fold", columns="arm", values="spearman")
    for f in hal.index:
        ax.plot(xs, [hal.loc[f, o[0]] for o in order], color=ORANGE, lw=0.4, alpha=0.35)
    ax.plot(xs, [hal[o[0]].mean() for o in order], color=ORANGE, marker="s", ms=3.6, lw=1.2, label="crisprHAL 2, published configuration")
    sx = np.array([o[1] for o in order if o[2]], float)
    for f in h.index:
        ax.plot(sx, [h.loc[f, o[2]] for o in order if o[2]], color=BLUE, lw=0.4, alpha=0.35)
    ax.plot(sx, [h[o[2]].mean() for o in order if o[2]], color=BLUE, marker="o", ms=3.6, lw=1.2, label="SLICER")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["23\n(guide+PAM)", "378\n(crisprHAL 2's\ninput)", "1,023", "2,023"])
    ax.set_xlabel("DNA both models are given (nt)")
    ax.set_ylabel("ρ, five random-row folds (thin lines: folds)")
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    _clean(ax)
    fig.tight_layout()
    return _save(fig, "p_fig6_width")


# ------------------------------------------------------------------------------------------------ Figure 7: cost
PURPLE = "#8e5fb0"


def _gpu_data():
    """crisprHAL 2 on the Colab T4 from `cost_gpu.csv` (one row per repeat, written by `colab/cost_gpu.py`): ms per guide by (placement, batch), and the fit seconds."""
    d = pd.read_csv(R / "cost_gpu.csv")
    h = d[(d.model == "crisprHAL 2") & (d.unit == "ms/guide")]
    score = {("host" if pl.startswith("raw") else "card", int(b)): g.value.to_numpy() for (pl, b), g in h.groupby(["placement", "batch"])}
    return score, d[(d.model == "crisprHAL 2") & (d.what == "fit")].value.to_numpy(), "cost_gpu.csv"


def _median_ci(v, n_boot=2000, seed=0):
    """Median and a 95% bootstrap interval over the repeats."""
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    med = float(np.median(v))
    if len(v) < 3:
        return med, med, med
    b = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    return med, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _bar(ax, x, v, color, w):
    med, lo, hi = _median_ci(v)
    ax.bar(x, med, width=w, color=color, linewidth=0)
    ax.errorbar([x], [med], yerr=[[med - lo], [hi - med]], color=INK, lw=0.7, capsize=1.6)


def fig_cost():
    _style()
    inf = pd.read_csv(R / "cost_repeats_inference.csv")
    fit = pd.read_csv(R / "cost_repeats_fit.csv")
    gpu, gpu_fit, _ = _gpu_data()
    cg = pd.read_csv(R / "cost_gpu.csv")
    colab_trees = {int(b): g.value.to_numpy() for b, g in cg[(cg.model == "SLICER default shape") & (cg.device == "Colab CPU") & (cg.what == "forward")].groupby("batch")}
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE * 0.98, 2.8), gridspec_kw=dict(width_ratios=[1.5, 1.0, 1.15]))
    models = ["SLICER default", "SLICER tuned", "BoostMEC", "crisprHAL 2"]
    cols = {"SLICER default": BLUE, "SLICER tuned": GREEN, "BoostMEC": PURPLE, "crisprHAL 2": ORANGE}
    batches = [1, 100, 10_000]
    lo, hi = 10 ** -2.4, 10 ** 2.6
    for bi, b in enumerate(batches):
        for mi, m in enumerate(models):
            _bar(ax[0], bi + (mi - 1.5) * 0.2, inf[(inf.model == m) & (inf.batch == b)].ms_per_guide.to_numpy(), cols[m], 0.18)
        for mi, (v, c) in enumerate(((gpu.get(("host", b)), ORANGE), (gpu.get(("card", b)), "#f5b08f"), (colab_trees.get(b), BLUE))):
            if v is not None:
                _bar(ax[1], bi + (mi - 1) * 0.26, v, c, 0.23)
    for a in ax[:2]:
        a.set_yscale("log")
        a.set_ylim(lo, hi)
        a.set_xticks(range(3))
        a.set_xticklabels([f"{b:,}" for b in batches])
        a.set_xlabel("guides per call", fontsize=6.6)
        _clean(a)
    ax[0].set_ylabel("ms per guide (log scale)")
    ax[0].set_title("Quiet laptop CPU, one thread, raw sequence in", fontsize=6.6, pad=3)
    ax[1].set_title("Colab T4 session (one VM, not quiet)", fontsize=6.6, pad=3)
    ax[1].set_yticklabels([])
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=cols[m], label=m) for m in models], frameon=False, loc="upper right", fontsize=5.8, ncol=2, columnspacing=0.8, handlelength=1.2)
    ax[1].legend(handles=[Patch(color=ORANGE, label="crisprHAL 2, raw contexts, host to card"), Patch(color="#f5b08f", label="crisprHAL 2, forward pass, batch on the card"),
                          Patch(color=BLUE, label="SLICER's trees (default shape), forward pass, Colab CPU")],
                 frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=5.4, handlelength=1.2)
    groups = [("SLICER default", fit[fit.model == "SLICER default"].wall_seconds.to_numpy(), BLUE),
              ("SLICER tuned", fit[fit.model == "SLICER tuned"].wall_seconds.to_numpy(), GREEN),
              ("BoostMEC, final fit", fit[fit.model == "BoostMEC, final fit"].wall_seconds.to_numpy(), PURPLE),
              ("BoostMEC, + tree count", fit[fit.model == "BoostMEC, with its tree count"].wall_seconds.to_numpy(), PURPLE),
              ("crisprHAL 2, laptop CPU", fit[fit.model == "crisprHAL 2"].wall_seconds.to_numpy(), ORANGE), ("crisprHAL 2, T4", gpu_fit, ORANGE)]
    for i, (lab, v, c) in enumerate(groups):
        if len(v):
            _bar(ax[2], i, v, c, 0.62)
    ax[2].set_yscale("log")
    ax[2].set_ylim(1, 3000)
    ax[2].set_xticks(range(len(groups)))
    ax[2].set_xticklabels([g[0] for g in groups], fontsize=5.6, rotation=40, ha="right", rotation_mode="anchor")
    ax[2].set_ylabel("seconds for one fit (log scale)")
    ax[2].set_title("Fitting (laptop: all ten cores)", fontsize=6.6, pad=3)
    _clean(ax[2])
    _panel(ax[0], "a", dx=-0.16)
    _panel(ax[1], "b", dx=-0.1)
    _panel(ax[2], "c", dx=-0.16)
    fig.tight_layout()
    return _save(fig, "p_fig7_cost")


FIGS = {"reach": fig_reach, "signal": fig_signal, "signal_appendix": fig_signal_appendix, "transfer": fig_transfer, "rule": fig_rule, "ceiling": fig_ceiling, "width": fig_width, "cost": fig_cost}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fig", default="all")
    a = ap.parse_args(argv)
    names = list(FIGS) if a.fig == "all" else a.fig.split(",")
    rows = [FIGS[n]() for n in names]
    man = OUT / "figure_manifest_paper.csv"
    old = pd.read_csv(man) if man.exists() and man.stat().st_size else pd.DataFrame(columns=["figure", "width_in", "height_in"])
    new = pd.DataFrame(rows)
    pd.concat([old[~old.figure.isin(new.figure)], new]).sort_values("figure").to_csv(man, index=False)      # a run of some figures keeps the rows of the others
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
