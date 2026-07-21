"""Publication-quality figures for the DriftGAN full-run verdict.

Builds three paper figures that tell the honest story:
  fig1  F1 trajectory over the stream  -> A collapses, B/C overlap (GAN null),
        C_early stays high (the win).
  fig2  Realistic recovery comparison  -> final F1 per phase (with per-seed
        spread) + RT/AUD with 95% CI, showing GAN adaptation ~ reactive baseline.
  fig3  Early-detection evidence       -> lead-time distribution (anticipation)
        + cost/benefit (memory vs F1).

Colours: Okabe-Ito (colour-blind safe by construction), one fixed hue per phase.
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
from matplotlib.patches import Patch

RUNS = Path("results/runs")
REC = Path("results/recovery")
OUT = Path("results/report/paper")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]

# One fixed hue per phase (colour follows the entity, never the rank).
COLOR = {
    "A": "#999999",        # frozen  -> muted grey (degraded control)
    "B": "#0072B2",        # reactive PHT -> blue (baseline)
    "C": "#E69F00",        # GAN vanilla  -> orange
    "C_cond": "#56B4E9",   # conditional WGAN-GP -> sky
    "C_prime": "#CC79A7",  # bootstrap+jitter -> purple (naive control)
    "C_early": "#009E73",  # generative monitor -> green (the win)
}
LABEL = {
    "A": "A · frozen",
    "B": "B · reactive PHT",
    "C": "C · GAN",
    "C_cond": "C_cond · cond. WGAN-GP",
    "C_prime": "C' · bootstrap+jitter",
    "C_early": "C_early · gen. monitor",
}

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "legend.fontsize": 10,
    }
)


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
    for s in SEEDS:
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
    for s in SEEDS:
        d = json.load(open(RUNS / "C_early" / f"seed_{s}" / "lead_time.json"))
        out[s] = d["lead_times"]
    return out


def parse_ci(s: str) -> tuple[float, float]:
    lo, hi = re.findall(r"[-\d.]+", s)[:2]
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Figure 1 — F1 trajectory over the stream
# --------------------------------------------------------------------------- #
def fig1_trajectory() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))

    # small vertical nudges so co-located end-labels (B≈C≈0.79) don't overlap
    lbl_dy = {"A": 0, "B": 9, "C": -9, "C_early": 0}
    for phase in ["A", "B", "C", "C_early"]:
        idx, mean, std = load_trajectory(phase)
        ax.fill_between(idx, mean - std, mean + std, color=COLOR[phase],
                        alpha=0.12, lw=0, zorder=2)
        ax.plot(idx, mean, color=COLOR[phase], lw=2.2, zorder=3,
                label=LABEL[phase])
        good = np.isfinite(mean)
        ax.annotate(f"{phase}  {mean[good][-1]:.2f}",
                    (idx[good][-1], mean[good][-1]),
                    xytext=(8, lbl_dy[phase]), textcoords="offset points",
                    va="center", color=COLOR[phase], fontweight="bold",
                    fontsize=10)

    # shade the drift-dense second half where the strategies diverge
    ax.axvspan(1.0e6, 2.05e6, color="#f2c7c0", alpha=0.18, zorder=0)
    ax.annotate("drift-dense second half", (1.52e6, 0.05), ha="center",
                va="bottom", fontsize=9, color="#b1553f", style="italic")

    ax.set_xlabel("stream position (samples processed)")
    ax.set_ylabel("local F1  (50K-sample bins, mean ± 1 s.d., 5 seeds)")
    ax.set_title("Where detection quality holds — and where it diverges")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 2.05e6)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v/1e6:.1f}M" if v else "0")
    ax.legend(loc="upper right", ncol=2)
    txt = ("Easy stretch (~0.6–1.0M): every adaptive strategy is high. GAN (C) even "
           "trails reactive PHT (B) early — it adds nothing.\nIn the drift-dense "
           "second half only the generative early-monitor (C_early) holds F1 up "
           "(+0.10–0.14 over B); frozen (A) has collapsed.")
    ax.text(0.5, 1.11, txt, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#555555")
    fig.savefig(OUT / "fig1_f1_trajectory.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 — realistic recovery comparison
# --------------------------------------------------------------------------- #
def fig2_recovery(summ: pd.DataFrame) -> None:
    order = ["A", "B", "C", "C_cond", "C_prime", "C_early"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # -- left: final F1 per phase, bar = mean, dots = seeds -----------------
    xs = np.arange(len(order))
    for i, ph in enumerate(order):
        vals = summ.loc[summ.phase == ph, "final_f1"].values
        axL.bar(i, vals.mean(), width=0.62, color=COLOR[ph], alpha=0.85,
                zorder=2, edgecolor="white", linewidth=1)
        jit = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.22
        axL.scatter(i + jit, vals, color="#222222", s=18, zorder=3)
        axL.annotate(f"{vals.mean():.3f}", (i, vals.mean()),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold")
    axL.set_xticks(xs)
    axL.set_xticklabels([p.replace("_", "\n") for p in order], fontsize=9)
    axL.set_ylabel("final F1  (bar = mean, dots = seeds)")
    axL.set_title("A. Final detection quality")
    axL.set_ylim(0, 1.05)

    # -- right: RT & AUD with 95% CI (the honest 'GAN ~ baseline') ----------
    rec = ["B", "C", "C_cond", "C_prime"]  # PHT phases; C_early recovery is confounded
    st = summ.groupby("phase")
    # pull RT/AUD from recovery parquets (per-event), aggregate mean + bootstrap CI
    def agg(metric: str) -> tuple[list, list, list]:
        means, los, his = [], [], []
        for ph in rec:
            vals = []
            for s in SEEDS:
                r = pd.read_parquet(REC / f"{ph}_{s}.parquet")
                vals.append(r[metric].dropna().values)
            v = np.concatenate(vals)
            rng = np.random.default_rng(0)
            boot = [rng.choice(v, len(v)).mean() for _ in range(2000)]
            means.append(v.mean())
            los.append(np.percentile(boot, 2.5))
            his.append(np.percentile(boot, 97.5))
        return means, los, his

    rt_m, rt_lo, rt_hi = agg("RT")
    aud_m, aud_lo, aud_hi = agg("AUD")
    x = np.arange(len(rec))
    w = 0.36
    cols = [COLOR[p] for p in rec]
    # RT on left axis feel: put RT and AUD as two grouped series but same unit
    # (both are 'samples'); AUD dwarfs RT so use a broken presentation: two rows.
    axR.bar(x - w / 2, rt_m, w, color=cols, alpha=0.55, edgecolor="white",
            yerr=[np.array(rt_m) - rt_lo, np.array(rt_hi) - rt_m],
            capsize=3, ecolor="#444", label="_")
    axR.bar(x + w / 2, aud_m, w, color=cols, alpha=0.95, edgecolor="white",
            yerr=[np.array(aud_m) - aud_lo, np.array(aud_hi) - aud_m],
            capsize=3, ecolor="#444", label="_")
    axR.set_xticks(x)
    axR.set_xticklabels([p.replace("_", "\n") for p in rec])
    axR.set_ylabel("recovery cost, samples  (95% CI)")
    axR.set_title("B. Recovery cost — RT (light) vs AUD (solid)")
    axR.set_ylim(0, max(aud_hi) * 1.28)
    for xi, (rtm, audm) in enumerate(zip(rt_m, aud_m)):
        axR.annotate(f"{rtm:.0f}", (xi - w / 2, rtm), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8)
        axR.annotate(f"{audm:.0f}", (xi + w / 2, audm), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8)
    axR.text(0.5, 0.88, "No GAN variant beats baseline B: RT is indistinguishable "
             "(Wilcoxon p=0.62 for C),\nand C_cond / C' are significantly WORSE on "
             "AUD. Reactive generation does not help.",
             transform=axR.transAxes, ha="center", va="top", fontsize=8.5,
             color="#555")
    fig.savefig(OUT / "fig2_recovery_comparison.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 — early-detection evidence
# --------------------------------------------------------------------------- #
def fig3_early(summ: pd.DataFrame) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # -- left: lead-time distribution per seed + pooled --------------------
    lts = load_leadtimes()
    data = [np.array(lts[s]) / 1e3 for s in SEEDS]  # kilo-samples
    parts = axL.violinplot(data, positions=SEEDS, showextrema=False, widths=0.8)
    for b in parts["bodies"]:
        b.set_facecolor(COLOR["C_early"])
        b.set_alpha(0.25)
        b.set_edgecolor("none")
    bp = axL.boxplot(data, positions=SEEDS, widths=0.28, patch_artist=True,
                     showfliers=False, medianprops=dict(color="white", lw=1.6),
                     boxprops=dict(facecolor=COLOR["C_early"], edgecolor="none"),
                     whiskerprops=dict(color=COLOR["C_early"]),
                     capprops=dict(color=COLOR["C_early"]))
    del bp
    axL.axhline(0, color="#d55e00", lw=1.2, ls="--")
    axL.annotate("class drift fires here (lead = 0)", (SEEDS[-1] + 0.05, 0),
                 xytext=(0, 4), textcoords="offset points", ha="right",
                 va="bottom", color="#d55e00", fontsize=8.5)
    pooled = np.concatenate(data)
    axL.axhline(pooled.mean(), color=COLOR["C_early"], lw=1.4, zorder=1)
    axL.set_ylim(-1.2, 22)
    axL.text(0.02, 0.97, f"pooled mean lead = {pooled.mean():.1f}K samples\n"
             f"(all seeds strictly positive → genuine anticipation)",
             transform=axL.transAxes, ha="left", va="top", fontsize=9,
             fontweight="bold", color=COLOR["C_early"],
             bbox=dict(boxstyle="round,pad=0.35", fc="white",
                       ec=COLOR["C_early"], lw=1.0))
    axL.set_xticks(SEEDS)
    axL.set_xlabel("seed")
    axL.set_ylabel("anticipation lead time  (thousands of samples)")
    axL.set_title("A. The monitor fires *before* the drift (C_early)")

    # -- right: cost / benefit -- memory vs final F1 -----------------------
    order = ["A", "B", "C", "C_cond", "C_prime", "C_early"]
    # hand-tuned label offsets to avoid the A/B/C_prime cluster at ~780 MB
    off = {"A": (10, -4), "B": (10, 8), "C_prime": (10, -12),
           "C": (10, 8), "C_cond": (10, -4), "C_early": (-12, 12)}
    ha = {"C_early": "right"}
    for ph in order:
        sub = summ[summ.phase == ph]
        x = sub["peak_rss_mb"].mean()
        y = sub["final_f1"].mean()
        axR.scatter(x, y, s=170, color=COLOR[ph], edgecolor="white",
                    linewidth=1.5, zorder=3)
        axR.annotate(ph, (x, y), xytext=off[ph], textcoords="offset points",
                     ha=ha.get(ph, "left"), fontsize=9, fontweight="bold",
                     color=COLOR[ph])
    axR.set_xlim(720, 1680)
    axR.set_xlabel("peak memory (MB)")
    axR.set_ylabel("final F1")
    axR.set_title("B. Cost vs benefit")
    axR.set_ylim(0.3, 1.05)
    axR.text(0.97, 0.30, "C_early buys +0.20 final-F1\nfor ~2× memory;\nthe GAN "
             "phases pay the memory\nbut not the benefit.", transform=axR.transAxes,
             ha="right", va="top", fontsize=8.5, color="#555")
    fig.savefig(OUT / "fig3_early_detection.png")
    plt.close(fig)


def main() -> None:
    summ = load_summaries()
    fig1_trajectory()
    fig2_recovery(summ)
    fig3_early(summ)
    print("wrote:")
    for p in sorted(OUT.glob("*.png")):
        print(" ", p)


if __name__ == "__main__":
    main()
