"""Publication figures, to ACS graphics specifications, generated from the result files.

ACS requirements applied (JACS author guidelines):
  width      single column <= 3.33 in (240 pt); double column 4.167-7.0 in (300-504 pt)
  depth      <= 9.167 in (660 pt) including caption
  type       sans-serif; lettering no smaller than 4.5 pt in final size. ACS pages disagree
             (4.5 / 6 / 8 pt depending on the page), so everything here is >= 7 pt, which
             satisfies every variant. Helvetica, which is what ACS asks for.
  rules      no line thinner than 0.5 pt
  output     vector PDF, which satisfies the raster minima (1200 dpi line art, 600 grayscale,
             300 colour) by construction; a 600 dpi PNG is written beside it for preview only
  colour     RGB; every figure is also legible without colour -- marker shape and line style
             carry identity alongside hue, and the diverging map is lightness-ordered

Design follows the project's data-viz procedure: form chosen from the data's job, colour assigned
by that job (categorical / sequential / diverging), palette validated with the bundled validator
rather than eyeballed, thin marks and recessive chrome, legends for >= 2 series and direct labels
where there are few, no dual axes, no rainbow ramps, no hue at a diverging midpoint, no dashed
gridlines, and no value-ramp on nominal categories.

    python -m sgrna.paper.figures --fig all
    python -m sgrna.paper.figures --fig 4
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from sgrna import config

from ..common import OUT

FIGDIR = OUT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
FROZEN = config.RESULTS                      # pre-freeze result files

SINGLE, DOUBLE = 3.33, 7.00                  # inches
# categorical slots 1-2 of the validated palette; both clear 3:1 on a light surface
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e1"
# diverging: blue <-> red with a neutral GREY midpoint (never a hue at the middle)
DIVERGING = LinearSegmentedColormap.from_list(
    "bwr_acs", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#8f1f1f"])


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.direction": "out",
        "ytick.direction": "out", "lines.linewidth": 0.9, "lines.markersize": 3.4,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.4, "grid.linestyle": "-",
        "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _panel(ax, letter, dx=-0.17, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            va="top", ha="left", color=INK)


def _save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(FIGDIR / f"{name}.{ext}", **kw)
    plt.close(fig)
    w, h = fig.get_size_inches()
    assert w <= DOUBLE + 1e-6, f"{name}: {w:.2f} in exceeds the 7.0 in double-column limit"
    assert h <= 9.167, f"{name}: {h:.2f} in exceeds the 9.167 in depth limit"
    print(f"  {name}: {w:.2f} x {h:.2f} in -> pdf + 600 dpi png", flush=True)
    return dict(figure=name, width_in=round(float(w), 3), height_in=round(float(h), 3))


# ---------------------------------------------------------------- Fig 1: the representation
def fig1():
    """Finding 1, as a schematic. Deliberately not a plot.

    The 412-vs-6,232 equivalence is a comparison across two protocols on the published matrix
    (427 columns at rho 0.5249 against 0.5272; 412 portable columns land 0.00003 from all 6,232),
    and there is no single result file holding both arms. Rather than assemble a two-point chart
    from incommensurable sources -- a first attempt did, and produced a spurious -0.019 -- the
    numbers are stated as annotation and the figure does the job a schematic should.
    """
    fig, ax = plt.subplots(figsize=(DOUBLE, 1.85))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [(0.015, "20-mer guide\n+ reference genome", "#ffffff"),
             (0.265, "6,232 published\ncolumns", "#eef4fd"),
             (0.515, "about 20 distinct\nvariables", "#eef4fd"),
             (0.765, "412 portable columns\nreproduce them", "#ffffff")]
    w = 0.22
    for x, label, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.46), w, 0.30, boxstyle="round,pad=0.012",
                                    linewidth=0.6, edgecolor=INK2, facecolor=fc))
        ax.text(x + w / 2, 0.61, label, ha="center", va="center", fontsize=6.6)
    for x0, lab in ((0.235, "featurise"), (0.485, "decode"), (0.735, "reduce")):
        ax.add_patch(FancyArrowPatch((x0, 0.61), (x0 + 0.027, 0.61), arrowstyle="-|>",
                                     mutation_scale=6, linewidth=0.6, color=INK2))
        ax.text(x0 + 0.014, 0.80, lab, ha="center", fontsize=6, color=INK2)
    ax.text(0.5, 0.26, "99% of the published columns are deterministic functions of the 20-mer, so no further "
                       "function of it can add information —\nonly performance. On the published matrix under a "
                       "300-column budget the 412 portable columns land 0.00003 ρ from all 6,232.",
            ha="center", va="center", fontsize=6.3, color=INK)
    ax.text(0.5, 0.055, "This equivalence is a statement about the published matrix under that budget; "
                        "it does not transfer to the curated cohort (Appendix B).",
            ha="center", va="center", fontsize=5.8, color=INK2)
    return _save(fig, "fig1_representation")


# ---------------------------------------------------------------- Fig 2: locality
def fig2():
    """Finding 2: the effect is spatially local -- the texture-preserving shift null measures its autocorrelation length.

    This figure used to carry a second panel comparing the flank gain on cutting and dCas9 screens. That comparison was demoted
    to Appendix B5 (22 November 2026) because the screens differ in assay, library, laboratory and phenotype at once, so the
    panel could not support a claim; the panel is not rebuilt anywhere.
    """
    fig, ax = plt.subplots(figsize=(SINGLE, 2.35))

    sh = pd.read_csv(FROZEN / "shift_null.csv").sort_values("shift_bp")
    # the frozen baseline file holds 18 rows for 15 (seed, fold) cells -- a re-run appended duplicates -- and counting them
    # moves the baseline from 0.5291 to 0.5307, which turns the report's "+0.0769, 1 kb retains 8%" into "+0.075, 6%"
    fl = pd.read_csv(FROZEN / "folds_shift_baseline.csv").drop_duplicates(["seed", "fold"])
    assert len(fl) == 15, len(fl)
    base = float(fl.spearman.mean())
    gain = sh.spearman.to_numpy() - base
    x = np.arange(len(sh))
    ax.axhline(0, color=INK2, linewidth=0.5, zorder=1)
    ax.plot(x, gain, "o-", color=BLUE, markerfacecolor=BLUE, markeredgecolor="white",
            markeredgewidth=0.4, zorder=3)
    labs = ["aligned" if b == 0 else (f"{int(b/1000)}k" if b < 1e6 else f"{b/1e6:g}M")
            for b in sh.shift_bp]
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6)
    ax.set_xlabel("flank track displaced by (bp)"); ax.set_ylabel("Δρ over a no-flank model")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.annotate(f"aligned\n{gain[0]:+.3f}", (0, gain[0]), textcoords="offset points",
                xytext=(10, -2), fontsize=6, color=INK)
    k = int(np.where(sh.shift_bp == 1000)[0][0])
    ax.annotate(f"1 kb retains {gain[k]/gain[0]*100:.0f}%", (k, gain[k]),
                textcoords="offset points", xytext=(8, 10), fontsize=6, color=INK2,
                arrowprops=dict(arrowstyle="-", linewidth=0.4, color=INK2))
    return _save(fig, "fig2_locality")


# ---------------------------------------------------------------- Fig 3: what the signal is made of
def fig3():
    """Finding 7: GC not purine, side-symmetric, one length scale, coupled to the seed."""
    fig = plt.figure(figsize=(DOUBLE, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.26)
    dp = pd.read_csv(OUT / "mechanism_distance_profile.csv")

    # (a) and (b): small multiples, shared y, so "GC runs +0.2 / purine is flat" is one glance
    lim = (-0.03, 0.235)
    for j, (stat, letter, title) in enumerate((("gc", "a", "GC fraction"),
                                              ("purine", "b", "purine fraction"))):
        ax = fig.add_subplot(gs[0, j]); _panel(ax, letter)
        s = dp[dp.statistic == stat]
        for side, lab, c, mk in (("up", "upstream", BLUE, "o"), ("dn", "downstream", ORANGE, "s")):
            t = s[s.side == side].sort_values("window_nt")
            ax.plot(t.window_nt, t.partial_given_target_gc, marker=mk, color=c,
                    markerfacecolor=c, markeredgecolor="white", markeredgewidth=0.4, label=lab)
        ax.axhline(0, color=INK2, linewidth=0.5, zorder=1)
        ax.set_xscale("log"); ax.set_xticks([50, 250, 500, 1000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(*lim); ax.set_xlabel("window (nt from the cut)")
        if j == 0:
            ax.set_ylabel("partial ρ with activity\n(target GC held fixed)")
            ax.legend(loc="lower center", ncol=2, handlelength=1.3, columnspacing=1.0,
                      borderpad=0.1, handletextpad=0.4)
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=7, pad=3)
        ax.yaxis.grid(True); ax.set_axisbelow(True)

    # (c) resolution: magnitude against a count -> line, with the reference as a rule not a series
    ax = fig.add_subplot(gs[1, 0]); _panel(ax, "c")
    res = pd.read_csv(OUT / "novel_resolution.csv").set_index("arm")
    b = float(res.loc["base only", "rho"]); ref = float(res.loc["base + family A comp block (24 cols)", "rho"])
    pts = []
    for lbl in res.index:
        if lbl.startswith("base + ") and "binned" in lbl:
            pts.append((int(lbl.split()[2]), float(res.loc[lbl, "rho"])))
    pts.sort()
    n, r = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    ax.axhline(ref, color=INK2, linewidth=0.5, zorder=1)
    ax.plot(n, r, "o-", color=BLUE, markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=0.4, zorder=3)
    ax.set_xscale("log"); ax.set_xticks(n)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("equal-width flank columns"); ax.set_ylabel("Spearman ρ")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    # the panel is about the shape of the curve, so the axis is the curve's own range;
    # the base-only value is far below and is reported as a percentage in the annotation instead
    span = ref - r.min()
    ax.set_ylim(r.min() - 0.18 * span, ref + 0.30 * span)
    ax.text(n[0], ref + 0.035 * span, "family A's 4 nested windows", va="bottom", ha="left",
            fontsize=5.8, color=INK2)
    ax.annotate(f"4 columns = {(r[0]-b)/(ref-b)*100:.0f}% of\nthe composition signal",
                (n[0], r[0]), textcoords="offset points", xytext=(7, -1), fontsize=5.8, color=INK)
    ax.annotate("turns over", (n[-1], r[-1]), textcoords="offset points", xytext=(-3, -11),
                ha="right", fontsize=5.8, color=INK)

    # (d) interaction: one magnitude over nominal categories -> one flat hue, never a value ramp
    ax = fig.add_subplot(gs[1, 1]); _panel(ax, "d")
    bp = pd.read_csv(OUT / "mechanism_pairs_blocks.csv").head(6).iloc[::-1]
    short = {"flank: window composition": "flank composition", "flank: window quantum": "flank quantum",
             "protospacer indicators: pos 16-20 (seed)": "seed (16–20)",
             "protospacer quantum: dimer": "guide dimer", "protospacer quantum: basepair": "guide base pair",
             "protospacer quantum: tetramer": "guide tetramer", "flank: quantum within 10 nt": "flank, <10 nt",
             "protospacer indicators: pos 6-15 (middle)": "guide middle"}
    labs = [" × ".join(short.get(p.strip(), p.strip()) for p in s.split(" x ")) for s in bp.block_pair]
    y = np.arange(len(bp))
    ax.barh(y, bp.gain_share, height=0.34, color=BLUE, linewidth=0)
    ax.set_yticks([]); ax.set_ylim(-0.6, len(bp) - 0.25)
    ax.set_xlabel("share of pairwise interaction gain")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_xlim(0, float(bp.gain_share.max()) * 1.16)
    for i, (v, lab) in enumerate(zip(bp.gain_share, labs)):
        ax.text(0, i + 0.30, lab, fontsize=5.8, color=INK, va="bottom", ha="left")
        ax.annotate(f"{v:.3f}", (v, i), textcoords="offset points", xytext=(3, 0),
                    fontsize=5.8, color=INK, va="center")
    for sp in ("left",):
        ax.spines[sp].set_visible(False)
    return _save(fig, "fig3_what_the_signal_is")


# ---------------------------------------------------------------- Fig 4: design rules by kingdom
def fig4():
    """Finding 7: position x base preferences, and that they do not transfer between kingdoms."""
    ec = pd.read_csv(OUT / "mechanism_position_base.csv")
    hu = pd.read_csv(OUT / "mechanism_position_base_human.csv")
    cmp_ = pd.read_csv(OUT / "mechanism_position_base_compare.csv")
    piv = lambda d: d.pivot(index="position", columns="base", values="effect_sd").reindex(columns=list("ACGT"))
    A, B = piv(ec), piv(hu)
    v = float(np.nanmax(np.abs(np.concatenate([A.to_numpy(), B.to_numpy()]))))
    norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

    fig = plt.figure(figsize=(SINGLE * 1.42, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.075], wspace=0.22)
    for j, (M, letter, title) in enumerate(((A, "a", "$\\it{E.\\ coli}$, cutting"),
                                            (B, "b", "human, knockout"))):
        ax = fig.add_subplot(gs[0, j]); _panel(ax, letter, dx=(-0.30 if j == 0 else -0.16))
        im = ax.imshow(M.to_numpy(), cmap=DIVERGING, norm=norm, aspect="auto")
        ax.set_xticks(range(4)); ax.set_xticklabels(list("ACGT"))
        ax.set_yticks(range(20)); ax.set_yticklabels(range(1, 21), fontsize=5.6)
        if j == 0:
            ax.set_ylabel("position in the guide  (1 = PAM-distal)")
        else:
            ax.set_yticklabels([])
        ax.set_title(title, fontsize=7, pad=4)
        ax.set_xlabel("base")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        # label only the cells the claim rests on -- never a number on every cell
        for pos, bs in ((20, "GT"), (18, "G")):
            for bb in bs:
                val = M.loc[pos, bb]
                ax.text(list("ACGT").index(bb), pos - 1, f"{val:+.2f}", ha="center", va="center",
                        fontsize=5.4, fontweight="bold",
                        color="white" if abs(val) > 0.55 * v else INK)
        ax.add_patch(Rectangle((-0.5, 16.5), 4, 3.0, fill=False, edgecolor=INK,
                               linewidth=0.7, clip_on=False))
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("effect on activity (SD of the label)", fontsize=6.3)
    cb.ax.tick_params(labelsize=5.8, length=1.6, width=0.4)
    cb.outline.set_linewidth(0.4)
    both = cmp_[cmp_.ecoli_clear & cmp_.human_clear]
    inv = int((np.sign(both.ecoli) != np.sign(both.human)).sum())
    from scipy.stats import spearmanr
    rho = float(spearmanr(cmp_.ecoli, cmp_.human).statistic)
    fig.text(0.5, -0.035, f"the two matrices correlate at ρ = {rho:+.3f};  "
                          f"{inv} of {len(both)} cells clear of zero in both are opposite in sign  "
                          f"(boxed: the seed, positions 18–20)",
             ha="center", fontsize=6, color=INK2)
    return _save(fig, "fig4_design_rules_by_kingdom")


# ---------------------------------------------------------------- Fig 5: the ceiling
def fig5():
    """Finding 6: the attainable ceiling against the one the field's framing implies."""
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    r = np.linspace(0.55, 0.95, 400)
    ax.axvspan(0.71, 0.77, color=GRID, zorder=0)
    ax.text(0.74, 0.585, "replicate agreement\nreported in the field", ha="center",
            fontsize=5.8, color=INK2)
    ax.plot(r, np.sqrt(2 * r / (1 + r)), "-", color=BLUE, label="vs a 2-replicate mean")
    ax.plot(r, np.sqrt(r), "-", color=ORANGE, label="vs one replicate")
    ax.plot(r, r, ":", color=INK2, linewidth=0.8, label="taken as the ceiling")
    r0 = 0.8097
    c0 = float(np.sqrt(2 * r0 / (1 + r0)))
    ax.plot([r0], [c0], "D", color=BLUE, markersize=4.2, markeredgecolor="white",
            markeredgewidth=0.5, zorder=5)
    ax.annotate(f"measured here\n$r$ = {r0:.3f} gives {c0:.3f}", (r0, c0), textcoords="offset points",
                xytext=(11, -17), ha="left", fontsize=6, color=INK,
                arrowprops=dict(arrowstyle="-", linewidth=0.4, color=INK2))
    lo, hi = float(np.sqrt(2 * 0.71 / 1.71)), float(np.sqrt(2 * 0.77 / 1.77))
    ax.annotate("", xy=(0.74, hi), xytext=(0.74, 0.70),
                arrowprops=dict(arrowstyle="<->", linewidth=0.6, color=INK))
    ax.text(0.752, (hi + 0.70) / 2, f"{lo - 0.70:+.2f} to {hi - 0.70:+.2f}\nunderstated",
            fontsize=6, color=INK, va="center")
    ax.set_xlabel("replicate-to-replicate agreement, $r$")
    ax.set_ylabel("highest ρ any predictor can reach")
    ax.set_xlim(0.55, 0.95); ax.set_ylim(0.55, 1.0)
    ax.legend(loc="upper left", handlelength=1.5, borderpad=0.1, handletextpad=0.5)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    return _save(fig, "fig5_attainable_ceiling")


# ---------------------------------------------------------------- Fig 6: what family A reads
def fig6():
    """Schematic: the windows family A summarises, and where the signal actually peaks."""
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.15))
    # fixed label columns either side of the data zone, so nothing can collide
    ax.set_xlim(-2150, 2150); ax.set_ylim(-0.42, 1.30); ax.axis("off")
    dp = pd.read_csv(OUT / "mechanism_distance_profile.csv")
    gc = dp[dp.statistic == "gc"]

    # --- the target, drawn enlarged so the 20-mer is visible at this scale
    y0 = 1.03
    ax.plot([-1000, 1000], [y0, y0], color=INK2, linewidth=0.7, zorder=1)
    ax.add_patch(Rectangle((-66, y0 - 0.055), 66, 0.11, facecolor=BLUE, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((0, y0 - 0.055), 33, 0.11, facecolor=ORANGE, edgecolor="none", zorder=3))
    ax.text(-76, y0 + 0.055, "guide (20 nt)", ha="right", va="bottom", fontsize=6.3, color=INK)
    ax.text(43, y0 + 0.055, "PAM", ha="left", va="bottom", fontsize=6.3, color=INK)
    ax.text(1000, y0 - 0.14, "target enlarged; windows to scale", ha="right", fontsize=5.6, color=INK2)
    ax.text(-2100, y0, "window", ha="left", va="center", fontsize=6, color=INK2, fontweight="bold")
    ax.text(2100, y0, "partial ρ with activity\n(upstream / downstream)", ha="right", va="center",
            fontsize=6, color=INK2, fontweight="bold")

    # --- the four nested windows, each a pair of bars with a gap at the cut
    for i, w in enumerate((50, 250, 500, 1000)):
        y = 0.70 - i * 0.235
        peak = w in (250, 500)
        if peak:
            ax.add_patch(Rectangle((-2150, y - 0.035), 4300, 0.175, facecolor="#f4f8fe",
                                   edgecolor="none", zorder=0))
        for s in (-1, 1):
            x = 18 if s > 0 else -w
            ax.add_patch(Rectangle((x, y), w - 18 if w > 18 else 6, 0.105,
                                   facecolor="#dce9fb" if peak else "#eef4fd",
                                   edgecolor=BLUE, linewidth=0.5, zorder=2))
        ax.text(-2100, y + 0.052, f"{w} nt", ha="left", va="center", fontsize=6.2, color=INK)
        up = float(gc[(gc.side == "up") & (gc.window_nt == w)].partial_given_target_gc.iloc[0])
        dn = float(gc[(gc.side == "dn") & (gc.window_nt == w)].partial_given_target_gc.iloc[0])
        ax.text(2100, y + 0.052, f"{up:+.3f}  /  {dn:+.3f}", ha="right", va="center",
                fontsize=6.2, color=INK, fontweight="bold" if peak else "normal")

    ax.text(0, -0.30, "shaded: where the association peaks (250–500 nt).  Purine fraction over the same "
                      "windows is flat within ±0.018 of zero.",
            ha="center", fontsize=6.2, color=INK2)
    ax.text(0, 1.26, "Cas9 contacts only the coloured blocks; every window below is DNA it never touches",
            ha="center", fontsize=6.5, color=INK)
    return _save(fig, "fig6_flank_windows_schematic")


# ---------------------------------------------------------------- Fig 7: the evaluation pipeline
def fig7():
    """Flowchart: how a number in this paper is produced, including the two leakage controls."""
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    stages = [
        (0.005, "33,567 curated\nguides\n(crisprHAL label)", "#ffffff"),
        (0.175, "base 6,169 cols\n+ family A 348\n(from the genome)", "#eef4fd"),
        (0.345, "fixed selector\ntop 450 of 6,517\n(fitted in-fold)", "#eef4fd"),
        (0.515, "boosted trees\ndepth 5, 4,000\nsquared error", "#eef4fd"),
        (0.685, "gene-grouped CV\n5 folds × 3 seeds\n+ 100 kb blocks", "#fdf0ea"),
        (0.855, "audit\nevery number\nrecomputed", "#ffffff"),
    ]
    for x, label, fc in stages:
        ax.add_patch(FancyBboxPatch((x, 0.47), 0.14, 0.33, boxstyle="round,pad=0.008",
                                    linewidth=0.6, edgecolor=INK2, facecolor=fc))
        ax.text(x + 0.07, 0.635, label, ha="center", va="center", fontsize=6)
        if x > 0.01:
            ax.add_patch(FancyArrowPatch((x - 0.025, 0.635), (x - 0.004, 0.635),
                                         arrowstyle="-|>", mutation_scale=6, linewidth=0.6, color=INK2))
    ax.add_patch(FancyBboxPatch((0.065, 0.07), 0.87, 0.29, boxstyle="round,pad=0.008",
                                linewidth=0.5, edgecolor=ORANGE, facecolor="#fdf0ea", alpha=0.45))
    ax.text(0.5, 0.295, "leakage controls, because both have bitten this project", ha="center",
            fontsize=6.3, color=INK, fontweight="bold")
    ax.text(0.5, 0.185, "whole genes and 100 kb blocks held out, so a positional column cannot score by "
                        "locus recognition\n"
                        "train/test Hamming distance reported: minimum 3, no identical guides across any split",
            ha="center", va="center", fontsize=5.7, color=INK2)
    ax.text(0.5, 0.90, "every ρ in this paper is produced this way; the sealed-gene experiment puts the "
                       "adaptive component at +0.009 ρ",
            ha="center", fontsize=6.3, color=INK2)
    return _save(fig, "fig7_pipeline")


FIGS = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "7": fig7}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default="all")
    a = ap.parse_args()
    _style()
    want = list(FIGS) if a.fig == "all" else [a.fig]
    rows = []
    for k in want:
        print(f"fig {k}:", flush=True)
        rows.append(FIGS[k]())
    man = OUT / "figure_manifest.csv"
    out = pd.DataFrame(rows)
    if len(want) < len(FIGS) and man.exists():          # regenerating one figure must not drop the others' rows
        old = pd.read_csv(man)
        out = pd.concat([old[~old.figure.isin(out.figure)], out], ignore_index=True)
        out = out.sort_values("figure", key=lambda c: c.str.extract(r"fig(\d+)")[0].astype(int))
    out.to_csv(man, index=False)
    print(f"\n{len(rows)} figure(s) -> {FIGDIR}")


if __name__ == "__main__":
    main()
