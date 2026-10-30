"""Publication figures for the PACE paper.

  fig1   local F1 across the stream (CICIDS2017, HAT)
  fig3   anticipation lead time per seed + memory vs final F1
  fig6   frequency control: local F1 by segment and overall
  fig9   monitor swapped inside the full pipeline
  fig10  budget-matched ensembles vs continuous supervision
  fig11  trigger x base learner (CICIDS2017)
  fig12  trigger x base learner x dataset
  fig13  reactive / blind clock / PACE at matched fire count
  fig15  closed-loop stability, per-seed cumulative accuracy
  fig16  timing (PACE - label-matched clock) under label delay
  fig17  security arms without / with window rebalancing, per-seed cumulative accuracy
(fig4 resources and fig5 sensitivity live in scripts/reviewer_analysis.py,
fig7 in scripts/monitor_ablation.py; all three reuse the style defined here.)

Every figure is drawn at its printed size (FULL = \\textwidth, COL = \\columnwidth)
with no titles and no commentary: the caption in the paper carries the message.
`save` writes <name>.pdf/.png to OUT and, in OUT/variants, a grayscale print
proof and one file per panel; each figure is also rendered once more with value
labels (<name>_labels). Alternative forms of the key results (forest plots,
collapse counts, the F1 gap) go to OUT/variants as well.

Palette: one academic-blue hue in five lightness steps (validated as an ordinal
ramp), plus two blue-greys for reference arms; hatching and markers carry
identity in grayscale print.
Run:  uv run python scripts/paper_figures.py
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox
from PIL import Image

RUNS = Path("results/runs")
REC = Path("results/recovery")
SEEDS = [0, 1, 2, 3, 4]  # the monitor-ablation frontier (monitor_ablation, fig7)
CORE = list(range(10))  # the ten seeds of the core, frequency, monitor and lead-time analyses

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
FULL, COL = 489.1 / 72.27, 235.5 / 72.27  # IEEEtran \textwidth, \columnwidth (pt -> in)

BLUE = ["#8DB6DE", "#5B91C8", "#2E6AA6", "#1F4E79", "#0E2A47"]  # light -> dark
ACC = BLUE[3]  # PACE and everything that is PACE's own (lead time, AE monitor)
SLATE, SLATE_LT, EXTRA = "#44546A", "#8497B0", "#A6B2C3"  # blue-greys, reference arms
PALE = "#EEF3FA"  # background bands only
INK, MUTED = "#1A1A1A", "#7F7F7F"
BOLD = {"fontweight": "bold"}

OUT = Path("results/report/paper")
VAR = OUT / "variants"
VAR.mkdir(parents=True, exist_ok=True)

# Colour follows the role of an arm, never its rank: reactive is mid blue, the
# blind clock is light and hatched, PACE is the deep academic blue, continuous
# supervision is a hatched blue-grey.
COLOR = {
    "B": BLUE[1], "B_adwin": BLUE[1], "B_kswin": BLUE[1],
    "C_freq": BLUE[0], "C_early": ACC,
    "ARF_sched": BLUE[1], "SRP_sched": BLUE[1],
    "ARF_early": ACC, "SRP_early": ACC,
    "ARF": SLATE, "SRP": SLATE,
}
HATCH = {"B_adwin": "....", "B_kswin": "\\\\\\\\", "C_freq": "////", "ARF": "xxx", "SRP": "xxx"}
LINESTYLE = {"B": "--", "C_freq": ":", "C_early": "-"}
MARKER = {"B": "o", "B_adwin": "s", "B_kswin": "^", "C_freq": "D", "C_early": "o"}
LABEL = {
    "B": "Reactive (PHT)", "B_adwin": "Reactive (ADWIN)", "B_kswin": "Reactive (KSWIN)",
    "C_freq": "Blind clock", "C_early": "PACE",
    "ARF_sched": "ARF reactive", "SRP_sched": "SRP reactive",
    "ARF_early": "ARF + PACE", "SRP_early": "SRP + PACE",
    "ARF": "ARF continuous", "SRP": "SRP continuous",
}

plt.rcParams.update(
    {
        # 8 pt Times, the paper's own face under IEEEtran and what the template
        # asks for in figure labels. STIX (bundled with matplotlib) is a Times
        # design and sets the math, so symbols and exponents match the text;
        # both are Unicode fonts, so the PDF text layer keeps the real glyphs.
        "font.family": ["Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#E3E8EF",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "hatch.linewidth": 0.6,
        "patch.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,  # embed TrueType, as publishers require
        "ps.fonttype": 42,
    }
)


def bar_kw(phase: str) -> dict:
    """Fill, hatch and white separating edge for one arm's bars."""
    kw = {"color": COLOR[phase], "edgecolor": "white", "linewidth": 0.6}
    if phase in HATCH:
        kw["hatch"] = HATCH[phase]
        kw["hatchcolor"] = BLUE[3] if COLOR[phase] == BLUE[0] else "white"
    return kw


def tag(ax, letter: str) -> None:
    """Panel letter, the only text a figure carries besides axes and legend."""
    ax.text(0.0, 1.02, f"({letter})", transform=ax.transAxes, ha="left", va="bottom",
            gid="tag", **BOLD)


def values(ax, xs, vals, fmt: str = "{:.2f}", dy: float = 1.5) -> None:
    """Value labels above bars (labelled variants only)."""
    for x, v in zip(xs, vals, strict=True):
        ax.annotate(fmt.format(v), (x, v), xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5)


def fit_width(fig) -> Bbox:
    """Crop box exactly as wide as the figure's nominal (print) width.

    A tight crop follows the drawn content, so labels spilling past the canvas (or
    an empty margin) change the saved width and LaTeX would rescale the figure,
    and its 8 pt text with it. The canvas is stretched until the content fits the
    nominal width (text stays fixed in points; only the axes absorb it), then the
    crop is pinned to that width so small renderer differences cannot leak in.
    """
    target = fig.get_figwidth()
    pad = plt.rcParams["savefig.pad_inches"]
    for _ in range(4):
        box = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad)
        fig.set_figwidth(fig.get_figwidth() + target - box.width)
    box = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad)
    return Bbox.from_bounds(box.x0, box.y0, target, box.height)


def save(fig, name: str, labels: bool = False, panels: tuple = ()) -> None:
    """Write the figure, its grayscale proof and one file per panel."""
    crop = fit_width(fig)
    if labels:
        for ext in ("pdf", "png"):
            fig.savefig(VAR / f"{name}_labels.{ext}", bbox_inches=crop)
        plt.close(fig)
        return
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches=crop)
    Image.open(OUT / f"{name}.png").convert("L").save(VAR / f"{name}_gray.png")
    renderer = fig.canvas.get_renderer()
    for ax, letter in zip(panels, "abcd", strict=False):
        tags = [t for t in ax.texts if t.get_gid() == "tag"]
        for t in tags:
            t.set_visible(False)
        box = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        for ext in ("pdf", "png"):
            fig.savefig(VAR / f"{name}_{letter}.{ext}", bbox_inches=box.padded(0.04))
        for t in tags:
            t.set_visible(True)
    plt.close(fig)
    print(f"  wrote {name}")


def kilo(ax) -> None:
    """Stream positions on the x axis, one tick every 0.5M samples."""
    ax.set_xticks(np.arange(0, 2.01e6, 0.5e6))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e6:.1f}M" if v else "0")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_summaries() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RUNS / "*/seed_*/summary.json")):
        d = json.load(open(f))
        rows.append(d)
    return pd.DataFrame(rows)


def load_trajectory(phase: str, bin_size: int = 50_000
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local windowed F1 in fixed stream bins, aggregated across seeds.

    The raw per-window F1 is jagged with gaps (benign stretches have no positive
    labels -> F1 undefined). Binning the *valid* windows into 50K-sample buckets
    gives a smooth local-quality curve that keeps the recovery dynamics — unlike a
    cumulative mean, which would wash out the late, drift-dense regions where the
    strategies actually diverge.
    """
    frames = []
    for s in CORE:
        df = pd.read_parquet(RUNS / phase / f"seed_{s}" / "snapshots.parquet")
        d = df[df["valid"]]
        b = (d["index"] // bin_size).astype(int)
        frames.append(d.groupby(b)["f1"].mean().rename(s))
    wide = pd.concat(frames, axis=1).sort_index()
    centers = (wide.index.values + 0.5) * bin_size
    mean = wide.mean(axis=1).values
    std = wide.std(axis=1).values
    return centers, mean, std


def load_leadtimes() -> dict[int, list[int]]:
    out = {}
    for s in CORE:
        d = json.load(open(RUNS / "C_early" / f"seed_{s}" / "lead_time.json"))
        out[s] = d["lead_times"]
    return out


def parse_ci(s: str) -> tuple[float, float]:
    lo, hi = re.findall(r"[-\d.]+", s)[:2]
    return float(lo), float(hi)


def _local_f1_by_segment(phase: str, edges: list[float]) -> tuple[list, float]:
    """Mean valid-window F1 in each [edge, edge) band, pooled across the core seeds."""
    rows = []
    for s in CORE:
        df = pd.read_parquet(RUNS / phase / f"seed_{s}" / "snapshots.parquet")
        rows.append(df[df["valid"]][["index", "f1"]])
    d = pd.concat(rows)
    means, overall = [], d["f1"].mean()
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        means.append(d[(d["index"] >= lo) & (d["index"] < hi)]["f1"].mean())
    return means, overall


def _run_dir(dataset: str, phase: str, seed: int) -> Path:
    """Result directory for (dataset, phase, seed).

    Mirrors the runner's dataset namespacing (``Config._ns_segments``): flat
    ``RUNS/phase/seed_n`` for the ``cicids2017`` anchor, ``RUNS/dataset/phase/seed_n``
    for every other dataset.
    """
    base = RUNS if dataset == "cicids2017" else RUNS / dataset
    return base / phase / f"seed_{seed}"


def _cell_seeds(dataset: str, phase: str) -> list[int]:
    """Completed seeds for (dataset, phase).

    The cross-dataset figures must agree with `tier0_analysis` /
    `tier1_analysis`, which resolve seeds per cell rather than assuming one
    global count — cells were extended past the original five at different
    times. The anchor-narrative figures use ``CORE``, the ten seeds every anchor
    arm they show has completed.
    """
    d = _run_dir(dataset, phase, 0).parent
    return sorted(
        int(p.name.split("_")[1])
        for p in d.glob("seed_*")
        if (p / "snapshots.parquet").exists() and (p / "summary.json").exists()
    )


def _shared_seeds(dataset: str, *phases: str) -> list[int]:
    """Seeds completed by every one of ``phases`` (paired comparisons)."""
    # Ten-seed protocol; anchor seeds 10-19 belong to the stability controls only.
    return sorted(set.intersection(*(set(_cell_seeds(dataset, p)) for p in phases)) & set(CORE))


def _overall_local_f1(dataset: str, phase: str,
                      seeds: list[int] | None = None) -> float | None:
    """Mean valid-window F1 pooled over ``seeds`` (default: the cell's own).

    Same metric as ``_local_f1_by_segment``'s ``overall``. Returns None if there
    is nothing to summarise (run not yet present -> cell skipped).
    """
    rows = []
    for s in _cell_seeds(dataset, phase) if seeds is None else seeds:
        df = pd.read_parquet(_run_dir(dataset, phase, s) / "snapshots.parquet")
        rows.append(df[df["valid"]][["f1"]])
    return float(pd.concat(rows)["f1"].mean()) if rows else None


def _per_seed_local_f1(dataset: str, phase: str,
                       seeds: list[int] | None = None) -> np.ndarray | None:
    """Overall local F1 per seed (not pooled) for (dataset, phase), or None."""
    out = []
    for s in _cell_seeds(dataset, phase) if seeds is None else seeds:
        df = pd.read_parquet(_run_dir(dataset, phase, s) / "snapshots.parquet")
        out.append(float(df.loc[df["valid"], "f1"].mean()))
    return np.array(out) if out else None


def _mean_n_drifts(dataset: str, phase: str,
                   seeds: list[int] | None = None) -> float | None:
    """Mean adaptation count (n_drifts) over ``seeds`` (default: the cell's own)."""
    vals = [
        json.loads((_run_dir(dataset, phase, s) / "summary.json").read_text())["n_drifts"]
        for s in (_cell_seeds(dataset, phase) if seeds is None else seeds)
    ]
    return float(np.mean(vals)) if vals else None


# --------------------------------------------------------------------------- #
# Figure 1 — local F1 across the stream
# --------------------------------------------------------------------------- #
def fig1_trajectory(labels: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(FULL, 2.3))
    ax.axvspan(1.0e6, 2.05e6, color=PALE, lw=0, zorder=0)
    for phase in ["B", "C_freq", "C_early"]:
        idx, mean, std = load_trajectory(phase)
        ax.fill_between(idx, mean - std, mean + std, color=COLOR[phase], alpha=0.12,
                        lw=0, zorder=2)
        ax.plot(idx, mean, color=COLOR[phase], ls=LINESTYLE[phase],
                lw=1.6 if phase == "C_freq" else 1.3, zorder=3, label=LABEL[phase])
        if labels:
            good = np.isfinite(mean)
            ax.annotate(f"{mean[good][-1]:.2f}", (idx[good][-1], mean[good][-1]),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.5)
    ax.set_xlabel("Stream position (samples)")
    ax.set_ylabel("Local F1")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 2.05e6)
    kilo(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    save(fig, "fig1_f1_trajectory", labels)


def fig1_delta() -> None:
    """Variant: the F1 gap to the reactive baseline, bin by bin."""
    idx, mean, _ = load_trajectory("B")
    ref = pd.Series(mean, index=idx)
    fig, ax = plt.subplots(figsize=(FULL, 1.9))
    ax.axvspan(1.0e6, 2.05e6, color=PALE, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
    for phase in ["C_freq", "C_early"]:
        idx, mean, _ = load_trajectory(phase)
        gap = pd.Series(mean, index=idx) - ref
        ax.plot(gap.index, gap.values, color=COLOR[phase], ls=LINESTYLE[phase],
                lw=1.6 if phase == "C_freq" else 1.3, label=f"{LABEL[phase]} $-$ reactive")
    ax.set_xlabel("Stream position (samples)")
    ax.set_ylabel(r"$\Delta$ local F1")
    ax.set_xlim(0, 2.05e6)
    kilo(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(VAR / f"fig1_delta.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 — anticipation lead time and cost/benefit
# --------------------------------------------------------------------------- #
def fig3_early(summ: pd.DataFrame, labels: bool = False) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FULL, 2.4),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(wspace=0.3)

    lts = load_leadtimes()
    data = [np.array(lts[s]) / 1e3 for s in CORE]  # kilo-samples
    parts = axL.violinplot(data, positions=CORE, showextrema=False, widths=0.8)
    for b in parts["bodies"]:
        b.set_facecolor(ACC)
        b.set_alpha(0.25)
        b.set_edgecolor("none")
    axL.boxplot(data, positions=CORE, widths=0.22, patch_artist=True, showfliers=False,
                medianprops={"color": "white", "lw": 1.2},
                boxprops={"facecolor": ACC, "edgecolor": "none"},
                whiskerprops={"color": ACC, "lw": 0.8},
                capprops={"color": ACC, "lw": 0.8})
    axL.axhline(0, color=MUTED, lw=0.7, ls="--")
    pooled = np.concatenate(data)
    axL.axhline(pooled.mean(), color=BLUE[4], lw=0.8, ls=":")
    if labels:
        axL.annotate(f"mean {pooled.mean():.1f}K", (CORE[-1] + 0.45, pooled.mean()),
                     xytext=(0, 2), textcoords="offset points", ha="right", va="bottom",
                     fontsize=6.5)
    axL.set_ylim(-1.2, 22)
    axL.set_xticks(CORE)
    axL.set_xlabel("Seed")
    axL.set_ylabel(r"Lead time ($\times 10^3$ samples)")
    tag(axL, "a")

    # the core seeds (C_early has 20 for the comparator study)
    for ph in ["B", "B_adwin", "B_kswin", "C_freq", "C_early"]:
        sub = summ[(summ.phase == ph) & summ.seed.isin(CORE)]
        if sub.empty:
            continue
        x, y = sub["peak_rss_mb"].mean(), sub["final_f1"].mean()
        axR.scatter(x, y, s=34, marker=MARKER[ph], color=COLOR[ph], edgecolor=BLUE[4]
                    if ph == "C_freq" else "white", linewidth=0.6, zorder=3, label=LABEL[ph])
        if labels:
            axR.annotate(f"{y:.2f}", (x, y), xytext=(5, 0), textcoords="offset points",
                         va="center", fontsize=6.5)
    axR.set_xlabel("Peak memory (MB)")
    axR.set_ylabel("Final F1")
    axR.set_ylim(0.3, 1.05)
    axR.grid(axis="x")
    axR.legend(loc="lower right", handletextpad=0.3)
    tag(axR, "b")
    save(fig, "fig3_early_detection", labels, (axL, axR))


# --------------------------------------------------------------------------- #
# Figure 6 — frequency control
# --------------------------------------------------------------------------- #
def fig6_frequency_ablation(labels: bool = False) -> None:
    """C_freq control: same adaptation count as C_early, blind periodic timing."""
    edges = [0, 0.5e6, 1.0e6, 1.5e6, 2.05e6]  # the first starts after the 50K pretraining
    seg_labels = ["0–0.5M", "0.5–1.0M", "1.0–1.5M", "1.5–2.0M"]
    phases = ["B", "C_freq", "C_early"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(FULL, 2.3),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(wspace=0.25)

    seg = {p: _local_f1_by_segment(p, edges) for p in phases}
    x = np.arange(len(seg_labels))
    w = 0.27
    for i, p in enumerate(phases):
        axL.bar(x + (i - 1) * w, seg[p][0], w, label=LABEL[p], **bar_kw(p))
        if labels:
            values(axL, x + (i - 1) * w, seg[p][0])
    axL.set_xticks(x)
    axL.set_xticklabels(seg_labels)
    axL.set_xlabel("Stream segment (samples)")
    axL.set_ylabel("Local F1")
    axL.set_ylim(0, 1.0)
    axL.legend(loc="upper right")
    tag(axL, "a")

    overall = [seg[p][1] for p in phases]
    for i, p in enumerate(phases):
        axR.bar(i, overall[i], 0.6, **bar_kw(p))
    if labels:
        values(axR, range(3), overall)
    axR.set_xticks(np.arange(3))
    axR.set_xticklabels([LABEL[p].replace(" (PHT)", "") for p in phases])
    axR.set_ylabel("Overall local F1")
    axR.set_ylim(0, 1.0)
    tag(axR, "b")
    save(fig, "fig6_frequency_ablation", labels, (axL, axR))


# --------------------------------------------------------------------------- #
# Figure 9 — monitor swapped inside the pipeline
# --------------------------------------------------------------------------- #
MONITOR = {  # phase -> (label, colour)
    "C_early": ("AE", ACC),
    "C_early_pca": ("PCA", BLUE[2]),
    "C_early_hdddm": ("HDDDM", BLUE[1]),
    "C_early_d3": ("D3", BLUE[0]),
    "C_early_mean": ("Mean\nshift", SLATE_LT),
    "C_early_d3_sized": ("D3 nat.\nsized", SLATE),
    "C_early_d3_native": ("D3 nat.\ndefault", EXTRA),
}


def fig9_monitor_pipeline(summ: pd.DataFrame, labels: bool = False) -> None:
    """Final and sustained F1 with the AE replaced by PCA / mean-shift (core seeds)."""
    order = ["C_early", "C_early_pca", "C_early_mean"]
    edges = [0, 0.5e6, 1.0e6, 1.5e6, 2.05e6]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(COL, 2.0))
    fig.subplots_adjust(wspace=0.55)
    x = np.arange(len(order))
    rng = np.random.default_rng(0)
    final = []
    for i, ph in enumerate(order):
        vals = summ.loc[(summ.phase == ph) & summ.seed.isin(CORE), "final_f1"].values
        final.append(vals.mean())
        axL.bar(i, vals.mean(), 0.62, color=MONITOR[ph][1], edgecolor="white")
        axL.scatter(i + (rng.random(len(vals)) - 0.5) * 0.25, vals, color=INK, s=4,
                    lw=0, zorder=3)
    overall = [_local_f1_by_segment(ph, edges)[1] for ph in order]
    axR.bar(x, overall, 0.62, color=[MONITOR[p][1] for p in order], edgecolor="white")
    if labels:
        values(axL, x, final, "{:.3f}")
        values(axR, x, overall, "{:.3f}")
    for ax, ylab, letter in ((axL, "Final F1", "a"), (axR, "Overall local F1", "b")):
        ax.set_xticks(x)
        ax.set_xticklabels([MONITOR[p][0] for p in order])
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.05)
        tag(ax, letter)
    save(fig, "fig9_monitor_pipeline", labels, (axL, axR))


# --------------------------------------------------------------------------- #
# Figure 10 — budget-matched ensembles vs continuous supervision
# --------------------------------------------------------------------------- #
def fig10_budget_matched(labels: bool = False) -> None:
    """HAT arms, ARF/SRP on the reactive schedule (233 adaptations), ARF/SRP continuous."""
    order = ["B", "C_early", "ARF_sched", "SRP_sched", "ARF", "SRP"]
    ticks = ["HAT\nreactive", "HAT\nPACE", "ARF\nmatched", "SRP\nmatched",
             "ARF\ncontin.", "SRP\ncontin."]
    hat = _shared_seeds("cicids2017", "B", "C_early")
    f1 = [_overall_local_f1("cicids2017", p, hat if p in ("B", "C_early") else None)
          for p in order]
    print("  fig10 local F1:", dict(zip(order, np.round(f1, 3), strict=True)))
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    x = np.arange(len(order))
    for i, p in enumerate(order):
        ax.bar(i, f1[i], 0.62, **bar_kw(p))
    if labels:
        values(ax, x, f1)
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.set_ylabel("Overall local F1")
    ax.set_ylim(0, 1.0)
    save(fig, "fig10_budget_matched", labels)


# --------------------------------------------------------------------------- #
# Figures 11/12 — trigger x base learner (x dataset)
# --------------------------------------------------------------------------- #
LEARNERS = [("HAT", "B", "C_early"), ("ARF", "ARF_sched", "ARF_early"),
            ("SRP", "SRP_sched", "SRP_early")]
DSET = {"cicids2017": "CIC-IDS2017", "cicids2018": "CIC-IDS2018", "unsw": "UNSW-NB15"}


def _pairs(dataset: str) -> list[tuple[float | None, float | None]]:
    """(reactive, PACE) overall local F1 per learner, on the seeds both arms share."""
    out = []
    for _base, r, e in LEARNERS:
        shared = _shared_seeds(dataset, r, e)
        out.append((_overall_local_f1(dataset, r, shared) if shared else None,
                    _overall_local_f1(dataset, e, shared) if shared else None))
    return out


def _pair_bars(ax, pairs, labels: bool, legend: bool) -> None:
    w = 0.38
    for i, (fr, fe) in enumerate(pairs):
        for xi, v, ph, lbl in ((i - w / 2, fr, "B", "Reactive"), (i + w / 2, fe, "C_early", "PACE")):
            if v is None:
                continue
            ax.bar(xi, v, w, label=lbl if (legend and i == 0) else None, **bar_kw(ph))
            if labels:
                values(ax, [xi], [v])
        if labels and fr is not None and fe is not None:
            ax.annotate(f"{fe - fr:+.3f}", (i, max(fr, fe)), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=6.5, **BOLD)
    ax.set_xticks(np.arange(len(LEARNERS)))
    ax.set_xticklabels([b for b, _, _ in LEARNERS])
    ax.set_ylim(0, 1.08 if labels else 1.0)


def fig11_early_ensemble(labels: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    _pair_bars(ax, _pairs("cicids2017"), labels, legend=True)
    ax.set_xlabel("Base learner")
    ax.set_ylabel("Overall local F1")
    ax.legend(loc="upper left", ncol=2)
    save(fig, "fig11_early_ensemble", labels)


def fig12_external_validity(labels: bool = False) -> None:
    datasets = list(DSET)
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.2), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    for k, (ax, ds) in enumerate(zip(axes, datasets, strict=True)):
        pairs = _pairs(ds)
        print(f"  fig12 {ds}:", [(round(a, 3), round(b, 3)) for a, b in pairs if a and b])
        _pair_bars(ax, pairs, labels, legend=k == 0)
        ax.set_xlabel(DSET[ds])
    axes[0].set_ylabel("Overall local F1")
    axes[0].legend(loc="upper left", ncol=2)
    save(fig, "fig12_external_validity", labels)


def fig12_forest() -> None:
    """Variant: per-cell paired difference with its 95% CI, plus the aggregate."""
    from tier0_analysis import paired_ci, section_a

    rows = section_a()
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    marker = {"HAT": "o", "ARF": "s", "SRP": "D"}
    y, ticks, pos = 0.0, [], []
    for i, r in enumerate(rows):
        if i and r.dataset != rows[i - 1].dataset:
            y += 0.6
        ax.plot([r.t_lo, r.t_hi], [y, y], color=ACC, lw=1.0)
        ax.scatter(r.mean_delta, y, marker=marker[r.learner], s=18, color=ACC,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ticks.append(f"{DSET[r.dataset]}  {r.learner}")
        pos.append(y)
        y += 1
    cell = np.array([r.mean_delta for r in rows])
    lo, hi = paired_ci(cell)
    y += 0.6
    ax.plot([lo, hi], [y, y], color=BLUE[4], lw=1.6)
    ax.scatter(cell.mean(), y, marker="D", s=30, color=BLUE[4], zorder=3)
    ticks.append("Aggregate (9 cells)")
    pos.append(y)
    ax.axvline(0, color=MUTED, lw=0.7, ls="--")
    ax.set_yticks(pos)
    ax.set_yticklabels(ticks)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")
    ax.set_xlabel(r"$\Delta$ local F1, PACE $-$ reactive (95% CI)")
    for ext in ("pdf", "png"):
        fig.savefig(VAR / f"fig12_forest.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 13 — timing vs frequency at matched count
# --------------------------------------------------------------------------- #
CELLS13 = [(ds, learner) for ds in DSET for learner in ("HAT", "ARF", "SRP")]
ARMS13 = [("sched", "B", "Reactive"), ("freq", "C_freq", "Blind clock"), ("early", "C_early", "PACE")]


def _phases13(learner: str) -> list[str]:
    """Reactive, blind clock and PACE phase names for one learner."""
    if learner == "HAT":
        return ["B", "C_freq", "C_early"]
    return [f"{learner}_{a}" for a, _, _ in ARMS13]


def two_level_ticks(ax, cells) -> None:
    """Learner under each group, dataset centred under its learners."""
    ax.set_xticks(np.arange(len(cells)), [learner for _, learner in cells])
    names = list(dict.fromkeys(ds for ds, _ in cells))
    centres = [np.mean([i for i, (ds, _) in enumerate(cells) if ds == n]) for n in names]
    ax.set_xticks(centres, [DSET[n] for n in names], minor=True)
    ax.xaxis.remove_overlapping_locs = False  # an odd group's centre sits on a major tick
    ax.tick_params(axis="x", which="minor", length=0, pad=15)


def fig13_timing_vs_frequency(labels: bool = False) -> None:
    """All nine cells: reactive, blind clock at PACE's count, PACE."""
    vals = {}
    for ds, learner in CELLS13:
        phases = _phases13(learner)
        seeds = _shared_seeds(ds, *phases)
        vals[(ds, learner)] = [_per_seed_local_f1(ds, p, seeds) for p in phases]
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    w = 0.26
    for j, (_arm, style, lbl) in enumerate(ARMS13):
        xs = np.arange(len(CELLS13)) + (j - 1) * w
        means = [vals[c][j].mean() for c in CELLS13]
        ax.bar(xs, means, w, label=lbl, **bar_kw(style))
    if labels:  # the timing component only: per-bar values collide at this width
        for i, c in enumerate(CELLS13):
            d = float((vals[c][2] - vals[c][1]).mean())
            ax.annotate(f"{d:+.3f}", (i, max(v.mean() for v in vals[c])), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=6.5, **BOLD)
    two_level_ticks(ax, CELLS13)
    ax.set_ylabel("Overall local F1")
    ax.set_ylim(0, 1.08 if labels else 1.0)
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0), borderaxespad=0.2,
              handlelength=1.2, columnspacing=1.0)
    save(fig, "fig13_timing_vs_frequency", labels)


def fig13_forest() -> None:
    """Variant: the frequency and timing components with paired 95% CIs."""
    from tier1_analysis import paired_ci

    fig, ax = plt.subplots(figsize=(COL, 2.8))
    comp = [("Frequency (clock $-$ reactive)", 1, 0, "s", SLATE_LT, -0.14),
            ("Timing (PACE $-$ clock)", 2, 1, "o", ACC, 0.14)]
    for name, a, b, mk, col, dy in comp:
        for i, (ds, learner) in enumerate(CELLS13):
            phases = _phases13(learner)
            f = [_per_seed_local_f1(ds, p, _shared_seeds(ds, *phases)) for p in phases]
            d = f[a] - f[b]
            lo, hi = paired_ci(d)
            ax.plot([lo, hi], [i + dy] * 2, color=col, lw=1.0)
            ax.scatter(d.mean(), i + dy, marker=mk, s=18, color=col, edgecolor="white",
                       linewidth=0.5, zorder=3, label=name if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=0.7, ls="--")
    ax.set_yticks(range(len(CELLS13)))
    ax.set_yticklabels([f"{DSET[ds]}  {learner}" for ds, learner in CELLS13])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")
    ax.set_xlabel(r"$\Delta$ local F1 (95% CI, paired by seed)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, borderaxespad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(VAR / f"fig13_forest.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 15 — closed-loop stability
# --------------------------------------------------------------------------- #
def _stability_arms() -> list[tuple[str, np.ndarray]]:
    """(phase, per-seed cumulative accuracy): matched arms, then native D3.

    Within each group arms are ordered by collapse count, so the figure reads
    left to right from most to least stable.
    """
    matched = ["C_early", "C_early_hdddm", "C_early_pca", "C_early_mean", "C_early_d3"]
    native = ["C_early_d3_sized", "C_early_d3_native"]

    def collect(group):
        out = []
        for phase in group:
            accs = np.array([json.load(open(f))["final_acc"]
                             for f in glob.glob(str(RUNS / phase / "seed_*" / "summary.json"))])
            if accs.size:
                out.append((phase, accs))
        return sorted(out, key=lambda r: (r[1] < 0.90).sum())

    return collect(matched) + collect(native)


def fig15_closed_loop_stability(labels: bool = False) -> None:
    """Per-seed cumulative accuracy by trigger; every seed drawn as its own point."""
    arms = _stability_arms()
    if not arms:
        print("  (no comparator runs; skipping fig15)")
        return
    n_matched = sum(p in ("C_early", "C_early_hdddm", "C_early_pca", "C_early_mean",
                          "C_early_d3") for p, _ in arms)
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.axhspan(0.0, 0.90, color=PALE, lw=0, zorder=0)
    ax.axhline(0.90, color=BLUE[3], lw=0.7, ls="--", zorder=1)
    ax.axhline(0.80, color=MUTED, lw=0.7, ls=":", zorder=1)
    ax.axvline(n_matched - 0.5, color=MUTED, lw=0.5, zorder=1)
    rng = np.random.default_rng(0)
    for i, (phase, accs) in enumerate(arms):
        color = MONITOR[phase][1]
        ax.scatter(i + rng.uniform(-0.18, 0.18, size=len(accs)), accs, s=9, color=color,
                   edgecolor=BLUE[4] if color == BLUE[0] else "white", linewidth=0.4, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [np.median(accs)] * 2, color=INK, lw=1.2, zorder=4)
        if labels:
            ax.annotate(f"{int((accs < 0.90).sum())}/{len(accs)}", (i, 1.02), ha="center",
                        va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([MONITOR[p][0] for p, _ in arms])
    ax.set_ylabel("Cumulative accuracy (per seed)")
    ax.set_ylim(0.35, 1.07 if labels else 1.02)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    save(fig, "fig15_closed_loop_stability", labels)


def fig15_counts() -> None:
    """Variant: healthy vs collapsed seeds per trigger as stacked bars."""
    arms = _stability_arms()
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    y = np.arange(len(arms))
    bad = np.array([(a < 0.90).sum() for _, a in arms])
    n = np.array([len(a) for _, a in arms])
    ax.barh(y, n - bad, 0.62, color=BLUE[3], edgecolor="white", label="Healthy")
    ax.barh(y, bad, 0.62, left=n - bad, color=BLUE[0], edgecolor="white", hatch="////",
            hatchcolor=BLUE[3], label="Collapsed (cumulative accuracy < 0.90)")
    ax.set_yticks(y)
    ax.set_yticklabels([MONITOR[p][0].replace("\n", " ") for p, _ in arms])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")
    ax.set_xlabel("Seeds")
    ax.set_xlim(0, n.max())
    ax.set_xticks(np.arange(0, n.max() + 1, 5))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(VAR / f"fig15_counts.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 16 — timing under label delay
# --------------------------------------------------------------------------- #
DELAYS16 = [("", "0"), ("_d1k", "1"), ("_d5k", "5"), ("_d50k", "45")]  # suffix, minutes
CELLS16 = [("cicids2017", "HAT", "C_early", "C_freq_lab"), ("cicids2017", "ARF", "ARF_early", "ARF_freq_lab"),
           ("cicids2018", "HAT", "C_early", "C_freq_lab"), ("cicids2018", "ARF", "ARF_early", "ARF_freq_lab")]


def fig16_label_delay(labels: bool = False) -> None:
    """PACE minus the label-matched blind clock, per delay, with paired 95% CIs."""
    from tier1_analysis import cell_seeds, paired_ci, per_seed_local_f1

    fig, ax = plt.subplots(figsize=(COL, 2.0))
    style = {"HAT": ("o", "-"), "ARF": ("s", "--")}
    shade = {"cicids2017": BLUE[1], "cicids2018": BLUE[4]}
    for k, (ds, learner, early, clock) in enumerate(CELLS16):
        xs, ms, los, his = [], [], [], []
        for i, (suffix, _) in enumerate(DELAYS16):
            seeds = cell_seeds(ds, early + suffix, clock + suffix)
            a, b = per_seed_local_f1(ds, early + suffix, seeds), per_seed_local_f1(ds, clock + suffix, seeds)
            if a is None or b is None:
                continue
            d = a - b
            lo, hi = paired_ci(d)
            xs.append(i + (k - 1.5) * 0.09)
            ms.append(d.mean())
            los.append(lo)
            his.append(hi)
        mk, ls = style[learner]
        col = shade[ds]
        ax.vlines(xs, los, his, color=col, lw=0.9)
        ax.plot(xs, ms, ls=ls, marker=mk, color=col, lw=0.9, ms=3.5, mec="white", mew=0.4,
                label=f"{DSET[ds]} {learner}")
        if labels:
            for x, m in zip(xs, ms, strict=True):
                ax.annotate(f"{m:+.3f}", (x, m), xytext=(3, 0), textcoords="offset points",
                            fontsize=5.5, va="center")
    ax.axhline(0, color=MUTED, lw=0.7, ls=":")
    ax.set_xticks(range(len(DELAYS16)), [m for _, m in DELAYS16])
    ax.set_xlabel("Label delay (minutes of capture)")
    ax.set_ylabel(r"$\Delta$ local F1, PACE $-$ clock")
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0), borderaxespad=0.2,
              handlelength=2.0, columnspacing=1.0)
    save(fig, "fig16_label_delay", labels)


# --------------------------------------------------------------------------- #
# Figure 17 — adversaries and window rebalancing
# --------------------------------------------------------------------------- #
SECURITY17 = [("C_early", "PACE"), ("C_early_adv", "Oracle,\nattack"), ("C_early_advb", "Oracle,\nbenign"),
              ("C_early_chaff", "PACE,\nchaff"), ("B_pht", "Error\ntest"), ("B_pht_chaff", "Error t.,\nchaff"),
              ("C_freq", "Blind\nclock")]


def fig17_security(labels: bool = False) -> None:
    """Per-seed cumulative accuracy of each security arm, without and with rebalancing."""
    def accs(phase: str) -> np.ndarray:
        return np.array([json.load(open(RUNS / phase / f"seed_{s}" / "summary.json"))["final_acc"]
                         for s in range(20) if (RUNS / phase / f"seed_{s}" / "summary.json").exists()])

    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.axhspan(0.0, 0.90, color=PALE, lw=0, zorder=0)
    ax.axhline(0.90, color=BLUE[3], lw=0.7, ls="--", zorder=1)
    rng = np.random.default_rng(0)
    for i, (phase, _) in enumerate(SECURITY17):
        for dx, ph, kw, lbl in ((-0.19, phase, {"color": BLUE[0], "edgecolor": BLUE[4]}, "Undefended"),
                                (0.19, phase + "_rb", {"color": ACC, "edgecolor": BLUE[4]}, "Window rebalancing")):
            a = accs(ph)
            ax.scatter(i + dx + rng.uniform(-0.09, 0.09, size=len(a)), a, s=8, linewidth=0.4, zorder=3,
                       label=lbl if i == 0 else None, **kw)
            if labels:
                ax.annotate(f"{int((a < 0.90).sum())}", (i + dx, 1.005), ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(range(len(SECURITY17)), [lbl for _, lbl in SECURITY17])
    ax.set_ylabel("Cumulative accuracy (per seed)")
    ax.set_ylim(0.35, 1.04 if labels else 1.01)
    ax.set_xlim(-0.6, len(SECURITY17) - 0.4)
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    save(fig, "fig17_security", labels)


def main() -> None:
    summ = load_summaries()
    for labels in (False, True):
        fig1_trajectory(labels)
        fig3_early(summ, labels)
        fig6_frequency_ablation(labels)
        fig9_monitor_pipeline(summ, labels)
        fig10_budget_matched(labels)
        fig11_early_ensemble(labels)
        fig12_external_validity(labels)
        fig13_timing_vs_frequency(labels)
        fig15_closed_loop_stability(labels)
        fig16_label_delay(labels)
        fig17_security(labels)
    fig1_delta()
    fig12_forest()
    fig13_forest()
    fig15_counts()
    print("wrote to", OUT, "and", VAR)


if __name__ == "__main__":
    main()
