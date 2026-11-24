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

R = OUT.parent
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
    fr = pd.read_csv(R / "post_freeze" / "flank_scale.csv")
    fr = fr[fr.seed == 41]
    base = fr[fr.arm == "base"].set_index("fold").rho
    rows = []
    for arm, g in fr[fr.part == prefix_part].groupby("arm", sort=False):
        d = g.set_index("fold").rho - base.reindex(g.set_index("fold").index)
        rows.append(dict(arm=arm, gain=float(d.mean()), gain_sd_over_folds=float(d.std(ddof=1))))
    return pd.DataFrame(rows)


def _tune_symmetric_gains():
    """Gain over the 20-mer of the symmetric arms (same window on both sides) that `flank_window_tune.py` added to those of `flank_scale.py`; seed 41, five folds."""
    f = R / "post_freeze" / "flank_window_tune.csv"
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
    wt = _tune_symmetric_gains()                                         # the finer windows of flank_window_tune.py (75 to 400 nt), same folds and model
    wt = wt[~wt.W.isin(w[w.arm.str.startswith("gc ")].W)]
    w = pd.concat([w, wt], ignore_index=True)
    for kind, col, mk, lab in (("gc", BLUE, "o", "GC fraction of each side (2 columns)"), ("base4", ORANGE, "s", "A, C, G, T fractions of each side (8 columns)")):
        q = w[w.arm.str.startswith(kind + " ")].sort_values("W")
        _sd_bars(ax[1], q.W, q.gain, q.gain_sd_over_folds, col, mk, lab)
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
    dec = pd.read_csv(R / "post_freeze" / "signal_decile.csv")
    sl = pd.read_csv(R / "post_freeze" / "signal_slope.csv")
    s = pd.read_csv(R / "post_freeze" / "flank_scale_summary.csv")
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE, 2.2))
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
    res = s[s.part == "resolution"].copy()
    lab = {"gc": "GC", "gcpur": "GC and purine", "base4": "A, C, G, T"}
    res["kind"] = res.arm.str.split(" ").str[0]
    res = res.set_index("kind").loc[["gc", "gcpur", "base4"]]
    ax[2].bar(range(3), res.gain, yerr=res.gain_sd_over_folds, color=[BLUE, BLUE, BLUE], width=0.6, capsize=2, error_kw=dict(lw=0.6))
    ax[2].set_xticks(range(3))
    ax[2].set_xticklabels([f"{lab[k]}\n({int(n)} columns)" for k, n in zip(res.index, [8, 16, 32])])
    ax[2].set_ylabel("gain in ρ over the 20-mer alone")
    ax[2].set_ylim(0, res.gain.max() * 1.25)
    ax[2].set_xlabel("flank composition given\n(windows of 50, 250, 500, 1,000 nt)", fontsize=6.6)
    _clean(ax[2])
    for a, l, dx in zip(ax, "abc", (-0.17, -0.17, -0.2)):
        _panel(a, l, dx=dx)
    fig.tight_layout()
    return _save(fig, "p_fig2_signal")


# ------------------------------------------------------------------------------------------------ Figure 3: transfer
def fig_transfer():
    _style()
    t = pd.read_csv(R / "post_freeze" / "eukaryote_transfer.csv")
    t = t[t.model.str.startswith("pipeline")]
    t = pd.concat([t[["trained_on", "scored_on", "rho"]], pd.read_csv(R / "post_freeze" / "transfer_to_ecoli.csv")[["trained_on", "scored_on", "rho"]]], ignore_index=True)
    cols = list(LABEL)
    rows = ["E. coli"] + [k for k in LABEL if k != "E. coli"]
    M = np.full((len(rows), len(cols)), np.nan)
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            q = t[(t.trained_on == a) & (t.scored_on == b)]
            if len(q):
                M[i, j] = q.rho.iloc[0]
    k = pd.read_csv(R / "post_freeze" / "transfer_vs_crisprhal_kingdom.csv")
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
    c = pd.read_csv(R / "post_freeze" / "eukaryote_contrast_g_minus_t.csv")
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
def fig_ceiling():
    _style()
    d = pd.read_csv(config.EXTERNAL / "crisprHAL" / "rawData" / "E_coli_WTSpCas9_counts.tsv", sep="\t")
    tot = d[["Cas1_S", "Cas2_S", "dCas1_NS", "dCas2_NS"]].sum()        # counts per million over the whole library, then the control-count filter, as in curated_ceiling.py
    d = d[(d.dCas1_NS + d.dCas2_NS) / 2 >= 73]
    cpm = lambda c: c / tot[c.name] * 1e6
    u = np.log2((cpm(d.Cas1_S) + 0.5) / (cpm(d.dCas1_NS) + 0.5))
    v = np.log2((cpm(d.Cas2_S) + 0.5) / (cpm(d.dCas2_NS) + 0.5))
    from scipy.stats import spearmanr
    rho = float(spearmanr(u, v).statistic)
    fig, ax = plt.subplots(1, 2, figsize=(DOUBLE * 0.9, 2.45), gridspec_kw=dict(width_ratios=[1, 1.25]))
    hb = ax[0].hexbin(-u, -v, gridsize=55, bins="log", cmap="Blues", mincnt=1, linewidths=0.1)
    ax[0].set_xlabel("cutting score, replicate pair 1")
    ax[0].set_ylabel("cutting score, replicate pair 2")
    ax[0].text(0.04, 0.93, f"Spearman ρ = {rho:.3f}\n{len(d):,} guides", transform=ax[0].transAxes, fontsize=6.5, va="top")
    cb = fig.colorbar(hb, ax=ax[0], fraction=0.045, pad=0.02)
    cb.set_label("guides (log)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    rel = 2 * rho / (1 + rho)
    ceil = float(np.sqrt(rel))
    vals = [("agreement of two replicates", rho, INK2), ("field's ceiling: replicate\nagreement of 0.71 to 0.77", 0.77, INK2), ("crisprHAL 2, as published", 0.6971, ORANGE),
            ("SLICER, default settings", 0.7151, BLUE), ("SLICER, tuned", 0.7314, BLUE), ("attainable ceiling of the released score", ceil, GREEN)]
    y = np.arange(len(vals))[::-1]
    ax[1].barh(y, [v[1] for v in vals], color=[v[2] for v in vals], height=0.6)
    ax[1].barh(y[1], 0.06, left=0.71, color="white", height=0.6, alpha=0.55)        # the range 0.71 to 0.77 is the lighter end of that bar
    for yy, v in zip(y, vals):
        ax[1].text(v[1] + 0.01, yy, ("0.71 to 0.77" if yy == y[1] else f"{v[1]:.3f}"), va="center", fontsize=6.3)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([v[0] for v in vals], fontsize=6)
    ax[1].set_xlim(0, 1.08)
    ax[1].set_xlabel("Spearman ρ")
    ax[1].axvline(ceil, color=GREEN, lw=0.6, ls=(0, (3, 2)))
    _clean(ax[1])
    ax[1].grid(axis="x", color=GRID, lw=0.5)
    ax[1].grid(axis="y", visible=False)
    _panel(ax[0], "a", dx=-0.3)
    _panel(ax[1], "b", dx=-0.9)
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
    d = pd.read_csv(R / "post_freeze" / "cost_gpu.csv")
    h = d[(d.model == "crisprHAL 2") & (d.unit == "ms/guide")]
    score = {("host" if pl.startswith("raw") else "card", int(b)): g.value.to_numpy() for (pl, b), g in h.groupby(["placement", "batch"])}
    return score, d[(d.model == "crisprHAL 2") & (d.what == "fit")].value.to_numpy(), "cost_gpu.csv"


def fig_cost():
    _style()
    inf = pd.read_csv(R / "post_freeze" / "cost_repeats_inference.csv")
    fit = pd.read_csv(R / "post_freeze" / "cost_repeats_fit.csv")
    gpu, gpu_fit, _ = _gpu_data()
    fig, ax = plt.subplots(1, 3, figsize=(DOUBLE * 0.98, 2.8), gridspec_kw=dict(width_ratios=[1.5, 0.95, 1.15]))
    models = ["SLICER default", "SLICER tuned", "BoostMEC", "crisprHAL 2"]
    cols = {"SLICER default": BLUE, "SLICER tuned": GREEN, "BoostMEC": PURPLE, "crisprHAL 2": ORANGE}
    batches = [1, 100, 10_000]
    lo, hi = -2.2, 2.6

    def violin(a, v, pos, color, w):
        v = np.log10(np.asarray(v, float))
        if len(v) > 2:
            vp = a.violinplot(v, positions=[pos], widths=w, showextrema=False, showmedians=False)
            for body in vp["bodies"]:
                body.set_facecolor(color); body.set_alpha(0.45); body.set_edgecolor("none")
        rng = np.random.default_rng(0)
        a.scatter(pos + rng.uniform(-0.35, 0.35, len(v)) * w, v, s=2.2, color=color, alpha=0.7, linewidths=0, zorder=3)      # every repeat
        a.plot([pos - w * 0.45, pos + w * 0.45], [np.median(v)] * 2, color=INK, lw=0.8, zorder=4)
    for bi, b in enumerate(batches):
        for mi, m in enumerate(models):
            violin(ax[0], inf[(inf.model == m) & (inf.batch == b)].ms_per_guide.to_numpy(), bi + (mi - 1.5) * 0.2, cols[m], 0.18)
        for mi, (kind, lab) in enumerate((("host", "raw contexts"), ("card", "on the card"))):
            if (kind, b) in gpu:
                violin(ax[1], gpu[(kind, b)], bi + (mi - 0.5) * 0.3, ORANGE if kind == "host" else "#f5b08f", 0.26)
        sl = np.log10(inf[(inf.model == "SLICER default") & (inf.batch == b)].ms_per_guide.median())
        ax[1].plot([bi - 0.3, bi + 0.3], [sl, sl], color=BLUE, lw=1.1, ls=(0, (3, 1.5)))
    for a in ax[:2]:
        a.set_xticks(range(3))
        a.set_xticklabels([f"{b:,}" for b in batches])
        a.set_ylim(lo, hi)
        yt = [-2, -1, 0, 1, 2]
        a.set_yticks(yt)
        a.set_yticklabels([f"{10.0 ** t:g}" for t in yt])
        a.set_xlabel("guides per call", fontsize=6.6)
        _clean(a)
    ax[0].set_ylabel("ms to score one guide (log scale)")
    ax[0].set_title("CPU, one thread, raw sequence in", fontsize=6.8, pad=3)
    ax[1].set_title("Colab T4 GPU", fontsize=6.8, pad=3)
    ax[1].set_yticklabels([])
    for m in models:
        ax[0].plot([], [], color=cols[m], lw=4, label=m)
    ax[0].legend(frameon=False, loc="upper right", fontsize=5.8, ncol=2, columnspacing=0.8, handlelength=1.2)
    ax[1].plot([], [], color=ORANGE, lw=4, label="crisprHAL 2, raw contexts")
    ax[1].plot([], [], color="#f5b08f", lw=4, label="crisprHAL 2, batch on the card")
    ax[1].plot([], [], color=BLUE, lw=1.1, ls=(0, (3, 1.5)), label="SLICER default, one CPU thread (a)")
    ax[1].legend(frameon=False, loc="lower left", fontsize=5.4, handlelength=1.4, bbox_to_anchor=(-0.02, -0.02))
    # fit time, wall-clock seconds, every fit a dot
    groups = [("SLICER default", fit[fit.model == "SLICER default"].wall_seconds.to_numpy(), BLUE),
              ("SLICER tuned", fit[fit.model == "SLICER tuned"].wall_seconds.to_numpy(), GREEN),
              ("BoostMEC, final fit", fit[fit.model == "BoostMEC, final fit"].wall_seconds.to_numpy(), PURPLE),
              ("BoostMEC, + tree count", fit[fit.model == "BoostMEC, with its tree count"].wall_seconds.to_numpy(), PURPLE),
              ("crisprHAL 2, CPU", fit[fit.model == "crisprHAL 2"].wall_seconds.to_numpy(), ORANGE), ("crisprHAL 2, T4", gpu_fit, ORANGE)]
    for i, (lab, v, c) in enumerate(groups):
        if len(v):
            ax[2].scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=c, s=9, zorder=3)
            ax[2].plot([i - 0.22, i + 0.22], [np.median(v)] * 2, color=INK, lw=0.9)
    ax[2].set_yscale("log")
    ax[2].set_xticks(range(len(groups)))
    ax[2].set_xticklabels([g[0] for g in groups], fontsize=5.6, rotation=40, ha="right", rotation_mode="anchor")
    ax[2].set_ylabel("one fit, wall-clock seconds (each dot a fit)")
    ax[2].set_title("Fitting (CPU: all ten cores)", fontsize=6.8, pad=3)
    _clean(ax[2])
    _panel(ax[0], "a", dx=-0.22)
    _panel(ax[1], "b", dx=-0.1)
    _panel(ax[2], "c", dx=-0.2)
    fig.tight_layout()
    return _save(fig, "p_fig7_cost")


FIGS = {"reach": fig_reach, "signal": fig_signal, "transfer": fig_transfer, "rule": fig_rule, "ceiling": fig_ceiling, "width": fig_width, "cost": fig_cost}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fig", default="all")
    a = ap.parse_args(argv)
    names = list(FIGS) if a.fig == "all" else a.fig.split(",")
    rows = [FIGS[n]() for n in names]
    pd.DataFrame(rows).to_csv(OUT / "figure_manifest_paper.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
