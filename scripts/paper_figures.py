"""Publication-quality figures for the DriftGAN full-run verdict.

Builds the paper figures that tell the honest story:
  fig1  local F1 across the stream    -> A collapses, B/C overlap (GAN null),
        C_early holds up in the drift-dense second half (the win).
  fig2  Realistic recovery comparison  -> final F1 per phase (with per-seed
        spread) + RT/AUD with 95% CI, showing GAN adaptation ~ reactive baseline.
  fig3  Early-detection evidence       -> lead-time distribution (anticipation)
        + cost/benefit (memory vs F1).
  fig6  Frequency-control ablation     -> C_freq (same adaptation count as
        C_early, blind periodic timing) falls below baseline B, proving the
        gain is *when* it adapts, not *how often*.
(fig4 resources and fig5 sensitivity live in scripts/reviewer_analysis.py.)

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
    "C_freq": "#D55E00",   # periodic-clock control -> vermillion
    "C_early": "#009E73",  # generative monitor -> green (the win)
}
LABEL = {
    "A": "A · frozen",
    "B": "B · reactive PHT",
    "C": "C · GAN",
    "C_cond": "C_cond · cond. WGAN-GP",
    "C_prime": "C' · bootstrap+jitter",
    "C_freq": "C_freq · periodic ctrl",
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
    summ.groupby("phase")
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
    for xi, (rtm, audm) in enumerate(zip(rt_m, aud_m, strict=False)):
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


def _local_f1_by_segment(phase: str, edges: list[float]) -> tuple[list, float]:
    """Mean valid-window F1 in each [edge, edge) band, pooled across seeds."""
    rows = []
    for s in SEEDS:
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


def _overall_local_f1(dataset: str, phase: str) -> float | None:
    """Mean valid-window F1 pooled across SEEDS for (dataset, phase).

    Same metric as ``_local_f1_by_segment``'s ``overall``. Returns None if any
    seed's snapshots.parquet is missing (run not yet present -> cell skipped).
    """
    rows = []
    for s in SEEDS:
        f = _run_dir(dataset, phase, s) / "snapshots.parquet"
        if not f.exists():
            return None
        df = pd.read_parquet(f)
        rows.append(df[df["valid"]][["f1"]])
    return float(pd.concat(rows)["f1"].mean())


def _per_seed_local_f1(dataset: str, phase: str) -> np.ndarray | None:
    """Overall local F1 per seed (not pooled) for (dataset, phase), or None."""
    out = []
    for s in SEEDS:
        f = _run_dir(dataset, phase, s) / "snapshots.parquet"
        if not f.exists():
            return None
        df = pd.read_parquet(f)
        out.append(float(df.loc[df["valid"], "f1"].mean()))
    return np.array(out)


def _mean_n_drifts(dataset: str, phase: str) -> float | None:
    """Mean adaptation count (n_drifts) across SEEDS for (dataset, phase).

    Returns None if any seed's summary.json is missing.
    """
    vals = []
    for s in SEEDS:
        f = _run_dir(dataset, phase, s) / "summary.json"
        if not f.exists():
            return None
        vals.append(json.loads(f.read_text())["n_drifts"])
    return float(np.mean(vals))


def fig6_frequency_ablation() -> None:
    """C_freq control: same adaptation count as C_early, blind periodic timing.

    Answers 'C_early only wins because it retrains more often': at equal
    frequency, blind timing is *worse than the reactive baseline*, so the gain
    is about *when* it adapts, not *how often*.
    """
    edges = [0.5e6, 1.0e6, 1.5e6, 2.05e6]
    seg_labels = ["0.5–1.0M\n(easy)", "1.0–1.5M\n(drift-dense)",
                  "1.5–2.0M\n(drift-dense)"]
    phases = ["B", "C_freq", "C_early"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    seg = {p: _local_f1_by_segment(p, edges) for p in phases}
    x = np.arange(len(seg_labels))
    w = 0.26
    for i, p in enumerate(phases):
        axL.bar(x + (i - 1) * w, seg[p][0], w, color=COLOR[p],
                edgecolor="white", label=LABEL[p])
    axL.set_xticks(x)
    axL.set_xticklabels(seg_labels, fontsize=9)
    axL.set_ylabel("local F1 (mean valid window, 5 seeds)")
    axL.set_title("A. Blind periodic adaptation collapses in the hard regions")
    axL.set_ylim(0, 1.0)
    axL.legend(loc="upper right")

    overall = [seg[p][1] for p in phases]
    axR.bar(np.arange(3), overall, width=0.6, color=[COLOR[p] for p in phases],
            edgecolor="white")
    for i, v in enumerate(overall):
        axR.annotate(f"{v:.2f}", (i, v), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontweight="bold")
    axR.set_xticks(np.arange(3))
    axR.set_xticklabels(["B\nreactive", "C_freq\nsame freq,\nblind timing",
                         "C_early\ninformed\ntiming"], fontsize=9)
    axR.set_ylabel("overall local F1")
    axR.set_title("B. Frequency is not the driver — timing is")
    axR.set_ylim(0, 0.85)
    axR.text(0.5, 0.86, "C_freq matches C_early's adaptation count (~162) but on a\n"
             "blind clock → below baseline B. The gain is *when*, not *how often*.",
             transform=axR.transAxes, ha="center", va="top", fontsize=8.5,
             color="#555")
    fig.savefig(OUT / "fig6_frequency_ablation.png")
    plt.close(fig)


def fig8_sota_baselines(summ: pd.DataFrame) -> None:
    """Compare C_early against SOTA streaming baselines (Experiment #3).

    Alternative drift detectors on the reactive HAT (B_adwin, B_kswin) and self-
    adapting ensembles (ARF, SRP), on final F1 and overall local F1. Skipped
    until the SOTA runs exist.
    """
    order = ["B", "B_adwin", "B_kswin", "ARF", "SRP", "C_early"]
    if not all((RUNS / p / "seed_0" / "summary.json").exists() for p in order):
        print("  (fig8 skipped — SOTA runs not present yet)")
        return
    pal = {"B": "#0072B2", "B_adwin": "#56B4E9", "B_kswin": "#CC79A7",
           "ARF": "#E69F00", "SRP": "#D55E00", "C_early": "#009E73"}
    edges = [0, 0.5e6, 1.0e6, 1.5e6, 2.05e6]
    # adaptation budget per phase (ARF/SRP self-adapt on *every* sample)
    def budget(ph: str) -> str:
        if ph in ("ARF", "SRP"):
            return "~2M\n(cont.)"
        nd = np.mean([json.loads((RUNS / ph / f"seed_{s}" / "summary.json").read_text())["n_drifts"]
                      for s in SEEDS])
        return f"{nd:.0f}\nadapts"
    ticks = [f"{p}\n{budget(p)}" for p in order]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.1))
    x = np.arange(len(order))
    hatch = {"ARF": "//", "SRP": "//"}  # mark the continuous-supervision baselines
    for i, ph in enumerate(order):
        vals = summ.loc[summ.phase == ph, "final_f1"].values
        axL.bar(i, vals.mean(), 0.62, color=pal[ph], alpha=0.9, edgecolor="white",
                hatch=hatch.get(ph))
        jit = (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.2
        axL.scatter(i + jit, vals, color="#222", s=16, zorder=3)
        axL.annotate(f"{vals.mean():.2f}", (i, vals.mean()), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8.5,
                     fontweight="bold")
    axL.set_xticks(x)
    axL.set_xticklabels(ticks, fontsize=8)
    axL.set_ylabel("final F1 (bar = mean, dots = seeds)")
    axL.set_title("A. Final detection quality vs SOTA")
    axL.set_ylim(0, 1.08)

    overall = [_local_f1_by_segment(ph, edges)[1] for ph in order]
    axR.bar(x, overall, 0.62, color=[pal[p] for p in order], edgecolor="white",
            hatch=[hatch.get(p, "") for p in order])
    for i, v in enumerate(overall):
        axR.annotate(f"{v:.2f}", (i, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=8.5, fontweight="bold")
    axR.set_xticks(x)
    axR.set_xticklabels(ticks, fontsize=8)
    axR.set_ylabel("overall local F1 (mean valid window)")
    axR.set_title("B. Sustained quality vs SOTA")
    axR.set_ylim(0, max(overall) * 1.25)
    fig.text(0.5, 0.005, "ARF / SRP (hatched) learn on EVERY sample — ~2M supervised "
             "updates — vs 160–340 triggered adaptations for B / C_early. The SOTA "
             "lead is real but NOT budget-matched; alternative detectors (ADWIN/KSWIN) "
             "do not beat plain PHT.", ha="center", va="bottom", fontsize=8.5,
             color="#555")
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(OUT / "fig8_sota_baselines.png")
    plt.close(fig)
    print("  wrote fig8_sota_baselines.png")


def fig9_monitor_pipeline(summ: pd.DataFrame) -> None:
    """#2 confirmation in the full pipeline: does a cheap monitor match the AE?

    Runs the C_early pipeline with the autoencoder replaced by the mean-shift and
    PCA monitors (calibrated to the AE's fire count). If final/local F1 match, the
    generative monitor is not needed. Skipped until the variant runs exist.
    """
    order = ["C_early", "C_early_mean", "C_early_pca"]
    if not all((RUNS / p / "seed_0" / "summary.json").exists() for p in order):
        print("  (fig9 skipped — C_early monitor variants not present yet)")
        return
    labels = {"C_early": "AE\n(generative)", "C_early_mean": "mean-shift\n(trivial)",
              "C_early_pca": "PCA recon\n(linear)"}
    pal = {"C_early": "#009E73", "C_early_mean": "#999999", "C_early_pca": "#0072B2"}
    edges = [0, 0.5e6, 1.0e6, 1.5e6, 2.05e6]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    x = np.arange(len(order))
    for i, ph in enumerate(order):
        vals = summ.loc[summ.phase == ph, "final_f1"].values
        axL.bar(i, vals.mean(), 0.6, color=pal[ph], edgecolor="white")
        axL.scatter(i + (np.random.default_rng(0).random(len(vals)) - 0.5) * 0.2,
                    vals, color="#222", s=16, zorder=3)
        axL.annotate(f"{vals.mean():.3f}", (i, vals.mean()), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=9, fontweight="bold")
    axL.set_xticks(x)
    axL.set_xticklabels([labels[p] for p in order], fontsize=9)
    axL.set_ylabel("final F1 (bar = mean, dots = seeds)")
    axL.set_title("A. Final F1 (monitor swapped in pipeline)", fontsize=11)
    axL.set_ylim(0, 1.08)

    overall = [_local_f1_by_segment(ph, edges)[1] for ph in order]
    axR.bar(x, overall, 0.6, color=[pal[p] for p in order], edgecolor="white")
    for i, v in enumerate(overall):
        axR.annotate(f"{v:.3f}", (i, v), xytext=(0, 3), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold")
    axR.set_xticks(x)
    axR.set_xticklabels([labels[p] for p in order], fontsize=9)
    axR.set_ylabel("overall local F1")
    axR.set_title("B. Sustained F1 (AE ≈ PCA ≫ mean-shift)", fontsize=11)
    axR.set_ylim(0, max(overall) * 1.25)
    fig.text(0.5, 0.005, "In the full pipeline a linear PCA monitor nearly matches the "
             "autoencoder (0.97 vs 0.99 final); a trivial mean-shift is clearly worse.\n"
             "A reconstruction signal is needed — but the generative/non-linear model is "
             "over-engineered: cheap PCA suffices.", ha="center", va="bottom",
             fontsize=8.5, color="#555")
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(OUT / "fig9_monitor_pipeline.png")
    plt.close(fig)
    print("  wrote fig9_monitor_pipeline.png")


def fig10_budget_matched(summ: pd.DataFrame) -> None:
    """#3b: is the SOTA lead model capacity or continuous supervision?

    ARF/SRP restricted to the same triggered schedule as B (budget-matched)
    vs their continuous-learning selves. Skipped until the _sched runs exist.
    """
    order = ["B", "C_early", "ARF_sched", "SRP_sched", "ARF", "SRP"]
    if not all((RUNS / p / "seed_0" / "summary.json").exists() for p in order):
        print("  (fig10 skipped — budget-matched SOTA runs not present yet)")
        return
    pal = {"B": "#0072B2", "C_early": "#009E73", "ARF_sched": "#E69F00",
           "SRP_sched": "#D55E00", "ARF": "#E69F00", "SRP": "#D55E00"}
    hatch = {"ARF": "//", "SRP": "//"}
    edges = [0, 0.5e6, 1.0e6, 1.5e6, 2.05e6]

    def bud(ph: str) -> str:
        if ph in ("ARF", "SRP"):
            return "~2M cont."
        nd = np.mean([json.loads((RUNS / ph / f"seed_{s}" / "summary.json").read_text())["n_drifts"]
                      for s in SEEDS])
        return f"{nd:.0f} adapts"
    ticks = [f"{p}\n{bud(p)}" for p in order]

    fig, ax = plt.subplots(figsize=(11, 5.0))
    x = np.arange(len(order))
    overall = [_local_f1_by_segment(ph, edges)[1] for ph in order]
    ax.bar(x, overall, 0.62, color=[pal[p] for p in order], edgecolor="white",
           hatch=[hatch.get(p, "") for p in order])
    for i, v in enumerate(overall):
        ax.annotate(f"{v:.2f}", (i, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, fontsize=9)
    ax.set_ylabel("overall local F1 (mean valid window, 5 seeds)")
    ax.set_title("Budget-matched SOTA: is the lead model capacity or continuous supervision?")
    ax.set_ylim(0, max(overall) * 1.2)
    ax.text(0.5, 1.13, "Same schedule as B (~233 triggered adaptations): does a stronger "
            "model alone win? | Hatched = continuous (~2M).\nThe drop from continuous to "
            "budget-matched isolates the value of full supervision.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#555")
    fig.subplots_adjust(top=0.80, bottom=0.16)
    fig.savefig(OUT / "fig10_budget_matched.png")
    plt.close(fig)
    print("  wrote fig10_budget_matched.png")


def fig11_early_ensemble() -> None:
    """#3c: does the proactive early trigger help the SOTA base learner, or only HAT?

    The base-learner x trigger 2x2 (reactive vs proactive early detection) on
    overall local F1. Skipped until ARF_early / SRP_early exist.
    """
    # (base-learner label, reactive phase, proactive-early phase)
    rows = [("HAT", "B", "C_early"), ("ARF", "ARF_sched", "ARF_early"),
            ("SRP", "SRP_sched", "SRP_early")]
    needed = {p for _, r, e in rows for p in (r, e)}
    if not all((RUNS / p / "seed_0" / "summary.json").exists() for p in needed):
        print("  (fig11 skipped — ARF_early/SRP_early not present yet)")
        return
    edges = [0, 0.5e6, 1.0e6, 1.5e6, 2.05e6]
    ov = {p: _local_f1_by_segment(p, edges)[1] for p in needed}

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(rows))
    w = 0.38
    for i, (_base, r, e) in enumerate(rows):
        ax.bar(i - w / 2, ov[r], w, color="#b0b0b0", edgecolor="white",
               label="reactive trigger" if i == 0 else None)
        ax.bar(i + w / 2, ov[e], w, color="#009E73", edgecolor="white",
               label="proactive early trigger" if i == 0 else None)
        for xi, ph in ((i - w / 2, r), (i + w / 2, e)):
            ax.annotate(f"{ov[ph]:.2f}", (xi, ov[ph]), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9,
                        fontweight="bold")
        d = ov[e] - ov[r]
        ax.annotate(f"{'+' if d >= 0 else ''}{d:.2f}", (i, max(ov[r], ov[e])),
                    xytext=(0, 20), textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold",
                    color="#009E73" if d >= 0 else "#d55e00")
    ax.set_xticks(x)
    ax.set_xticklabels([b for b, _, _ in rows], fontsize=11)
    ax.set_xlabel("base learner")
    ax.set_ylabel("overall local F1 (mean valid window, 5 seeds)")
    ax.set_title("Does proactive early detection help the base learner?")
    ax.set_ylim(0, max(ov.values()) * 1.25)
    ax.legend(loc="upper left")
    ax.text(0.5, 1.10, "Δ above each pair = gain from swapping reactive → proactive "
            "early trigger, holding the model fixed.\nIf the ensemble Δ is positive, "
            "the anticipation idea improves the SOTA model, not just the weak HAT.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#555")
    fig.subplots_adjust(top=0.82)
    fig.savefig(OUT / "fig11_early_ensemble.png")
    plt.close(fig)
    print("  wrote fig11_early_ensemble.png")


def fig12_external_validity() -> None:
    """#4: does the proactive early trigger generalize across datasets?

    Replicates fig11's base-learner x trigger 2x2 (reactive `*_sched`/B vs
    proactive `*_early`) per dataset (CICIDS2017 anchor, CICIDS2018, UNSW-NB15).
    Cells whose runs are not (yet) complete across all 5 seeds are skipped —
    the bar / delta annotation is simply not drawn.
    """
    datasets = ["cicids2017", "cicids2018", "unsw"]
    dset_label = {"cicids2017": "CIC-IDS2017", "cicids2018": "CIC-IDS2018",
                  "unsw": "UNSW-NB15"}
    learners = [("HAT", "B", "C_early"), ("ARF", "ARF_sched", "ARF_early"),
                ("SRP", "SRP_sched", "SRP_early")]
    phases = sorted({p for _, r, e in learners for p in (r, e)})

    f1_ov = {(ds, p): _overall_local_f1(ds, p) for ds in datasets for p in phases}
    n_drifts = {(ds, p): _mean_n_drifts(ds, p) for ds in datasets for p in phases}
    if not any(v is not None for v in f1_ov.values()):
        print("  (fig12 skipped — no cross-dataset runs present yet)")
        return

    fig, axes = plt.subplots(1, len(datasets), figsize=(13.5, 5.0), sharey=True)
    w = 0.36
    # label each series exactly once, in whichever axis first has data for it
    reactive_labeled = proactive_labeled = False
    legend_ax = None
    for ax, ds in zip(axes, datasets, strict=False):
        x = np.arange(len(learners))
        drew_any = False
        for i, (_base, r, e) in enumerate(learners):
            fr, fe = f1_ov[(ds, r)], f1_ov[(ds, e)]
            if fr is not None:
                lbl = None
                if not reactive_labeled:
                    lbl, reactive_labeled, legend_ax = "reactive trigger", True, ax
                ax.bar(i - w / 2, fr, w, color="#b0b0b0", edgecolor="white", label=lbl)
                ax.annotate(f"{fr:.2f}", (i - w / 2, fr), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8.5,
                            fontweight="bold")
                drew_any = True
            if fe is not None:
                lbl = None
                if not proactive_labeled:
                    lbl, proactive_labeled, legend_ax = "proactive early trigger", True, ax
                ax.bar(i + w / 2, fe, w, color="#009E73", edgecolor="white", label=lbl)
                ax.annotate(f"{fe:.2f}", (i + w / 2, fe), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8.5,
                            fontweight="bold")
                drew_any = True
            if fr is not None and fe is not None:
                d = fe - fr
                ax.annotate(f"{'+' if d >= 0 else ''}{d:.2f}", (i, max(fr, fe)),
                            xytext=(0, 16), textcoords="offset points", ha="center",
                            fontsize=9, fontweight="bold",
                            color="#009E73" if d >= 0 else "#d55e00")
        ax.set_xticks(x)
        ax.set_xticklabels([b for b, _, _ in learners], fontsize=10)
        ax.set_title(dset_label[ds], fontsize=11)
        ax.set_xlabel("base learner")
        if not drew_any:
            ax.text(0.5, 0.5, "runs pending", transform=ax.transAxes, ha="center",
                    va="center", fontsize=10, color="#999999", style="italic")
    if legend_ax is not None:
        legend_ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("overall local F1 (mean valid window, 5 seeds)")
    present = [v for v in f1_ov.values() if v is not None]
    ymax = max(present) * 1.25 if present else 1.0
    for ax in axes:
        ax.set_ylim(0, ymax)
    fig.suptitle("Does the proactive early trigger generalize across datasets?",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.01, "Δ above each pair = gain from swapping reactive -> proactive "
             "early trigger, holding the model fixed.\nPositive in all 9 cells "
             "(3 datasets x 3 base learners): the anticipation signal generalizes.",
             ha="center", va="bottom", fontsize=8.5, color="#555")
    fig.subplots_adjust(top=0.80, bottom=0.20)
    fig.savefig(OUT / "fig12_external_validity.png")
    plt.close(fig)
    print("  wrote fig12_external_validity.png")

    # -- printed cross-dataset summary table (feeds the doc write-up) -------
    print("\ncross-dataset summary (mean overall local F1, mean n_drifts):")
    header = f"{'dataset':<12} {'phase':<10} {'overall_f1':>10} {'n_drifts':>10}"
    print(header)
    print("-" * len(header))
    for ds in datasets:
        for phase in phases:
            f1, nd = f1_ov[(ds, phase)], n_drifts[(ds, phase)]
            if f1 is None and nd is None:
                continue
            f1s = f"{f1:.3f}" if f1 is not None else "n/a"
            nds = f"{nd:.1f}" if nd is not None else "n/a"
            print(f"{ds:<12} {phase:<10} {f1s:>10} {nds:>10}")


def fig13_timing_vs_frequency() -> None:
    """Tier 1: on the external datasets, is the ensemble gain *when* or *how often*?

    Three arms at comparable adaptation frequency per decisive cell (ARF/SRP on
    CIC-IDS2018 and UNSW-NB15): reactive schedule, blind periodic (`*_freq`, fire
    count matched to `*_early`), and informed proactive (`*_early`). The timing
    component `early - freq` (same count, informed vs blind placement) is the
    clean test. Skipped until the `*_freq` runs exist.
    """
    # (dataset, learner label, reactive, frequency-control, proactive)
    cells = [("cicids2018", "ARF"), ("cicids2018", "SRP"),
             ("unsw", "ARF"), ("unsw", "SRP")]
    phases = {"sched": "{L}_sched", "freq": "{L}_freq", "early": "{L}_early"}
    vals: dict[tuple, dict] = {}
    for ds, learner in cells:
        cell = {}
        for arm, tmpl in phases.items():
            cell[arm] = _per_seed_local_f1(ds, tmpl.format(L=learner))
        vals[(ds, learner)] = cell
    if not any(c["freq"] is not None for c in vals.values()):
        print("  (fig13 skipped — *_freq runs not present yet)")
        return

    dset_label = {"cicids2018": "CIC-IDS2018", "unsw": "UNSW-NB15"}
    pal = {"sched": "#b0b0b0", "freq": "#E69F00", "early": "#009E73"}
    arm_label = {"sched": "reactive", "freq": "blind, matched freq", "early": "informed timing"}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(cells))
    w = 0.26
    labelled = set()
    for j, arm in enumerate(("sched", "freq", "early")):
        for i, (ds, learner) in enumerate(cells):
            v = vals[(ds, learner)][arm]
            if v is None:
                continue
            lbl = arm_label[arm] if arm not in labelled else None
            labelled.add(arm)
            ax.bar(i + (j - 1) * w, v.mean(), w, color=pal[arm], edgecolor="white",
                   label=lbl, hatch="//" if arm == "freq" else None)
            ax.annotate(f"{v.mean():.2f}", (i + (j - 1) * w, v.mean()),
                        xytext=(0, 2), textcoords="offset points", ha="center",
                        fontsize=8)
    # timing delta (early - freq), paired by seed, annotated above each group
    for i, (ds, learner) in enumerate(cells):
        e, f = vals[(ds, learner)]["early"], vals[(ds, learner)]["freq"]
        if e is None or f is None:
            continue
        d = float((e - f).mean())
        top = max(e.mean(), f.mean(), vals[(ds, learner)]["sched"].mean())
        ax.annotate(f"timing {'+' if d >= 0 else ''}{d:.3f}", (i, top),
                    xytext=(0, 16), textcoords="offset points", ha="center",
                    fontsize=9, fontweight="bold",
                    color="#009E73" if d > 0.01 else "#777777")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{dset_label[ds]}\n{learner}" for ds, learner in cells],
                       fontsize=9)
    ax.set_ylabel("overall local F1 (mean valid window, 5 seeds)")
    ax.set_title("At matched adaptation frequency: is the gain timing or frequency?")
    present = [v[a].mean() for v in vals.values() for a in ("sched", "freq", "early")
               if v[a] is not None]
    ax.set_ylim(0, max(present) * 1.2)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(0.5, 1.12, "Timing = early − freq (same fire count, informed vs blind "
            "placement). CIC-IDS2018: informed timing wins and blind-more even "
            "*hurts*.\nUNSW: timing ≈ 0 — the gain there is frequency. The split "
            "tracks drift structure, consistent across ARF and SRP.",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5,
            color="#555")
    fig.subplots_adjust(top=0.80)
    fig.savefig(OUT / "fig13_timing_vs_frequency.png")
    plt.close(fig)
    print("  wrote fig13_timing_vs_frequency.png")


def main() -> None:
    summ = load_summaries()
    fig1_trajectory()
    fig2_recovery(summ)
    fig3_early(summ)
    fig6_frequency_ablation()
    fig8_sota_baselines(summ)
    fig9_monitor_pipeline(summ)
    fig10_budget_matched(summ)
    fig11_early_ensemble()
    fig12_external_validity()
    fig13_timing_vs_frequency()
    print("wrote:")
    for p in sorted(OUT.glob("*.png")):
        print(" ", p)


if __name__ == "__main__":
    main()
