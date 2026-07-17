"""Reusable figures: stream curves, recovery distributions, cost Pareto."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

PLOT_STYLE: dict[str, Any] = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

PHASE_COLORS = {
    "A": "#e74c3c",
    "B": "#2ecc71",
    "C": "#8e44ad",
    "C_prime": "#e67e22",
    "C_cond": "#16a085",
    "C_early": "#2c3e50",
}


def _apply_style() -> None:
    plt.rcParams.update(PLOT_STYLE)  # type: ignore[arg-type]  # stubs require literal rc keys


def _finalize(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    return fig


def stream_comparison(
    snapshots: dict[str, pd.DataFrame],
    drift_indices: list[int] | None = None,
    n_pretrain: int | None = None,
    title: str = "Concept drift — phase comparison",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Three-panel figure (rolling F1 / cumulative accuracy / interval time)
    with one line per phase. ``snapshots`` maps phase name to its
    snapshots dataframe (columns: index, f1, acc, interval_ms)."""
    _apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    ax1, ax2, ax3 = axes

    for phase, df in snapshots.items():
        color = PHASE_COLORS.get(phase, None)
        ax1.plot(df["index"], df["f1"], lw=1.5, alpha=0.85, color=color, label=phase)
        ax2.plot(df["index"], df["acc"], lw=1.5, alpha=0.85, color=color, label=phase)
        ax3.plot(df["index"], df["interval_ms"], lw=1.2, alpha=0.85, color=color, label=phase)

    if drift_indices:
        for i, dp in enumerate(drift_indices[:30]):
            ax1.axvline(dp, color="#f39c12", alpha=0.35, lw=1.0, ls="--",
                        label="drift" if i == 0 else None)
    if n_pretrain:
        for ax in axes:
            ax.axvline(n_pretrain, color="#3498db", alpha=0.6, lw=1.5, ls=":",
                       label="end of pretraining" if ax is ax1 else None)

    ax1.set_ylabel("rolling F1")
    ax1.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("cumulative accuracy")
    ax3.set_ylabel("ms per snapshot interval")
    ax3.set_xlabel("stream samples")
    for ax in axes:
        ax.legend(loc="lower right", framealpha=0.9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _finalize(fig, save_path)


def recovery_boxplots(
    recovery: dict[str, pd.DataFrame],
    metric: str = "RT",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Per-phase boxplot of a per-event recovery metric."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    phases = list(recovery)
    data = [recovery[p][metric].dropna().to_numpy(dtype=float) for p in phases]
    box = ax.boxplot(data, tick_labels=phases, patch_artist=True)
    for patch, phase in zip(box["boxes"], phases, strict=True):
        patch.set_facecolor(PHASE_COLORS.get(phase, "#95a5a6"))
        patch.set_alpha(0.6)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} per drift event, by phase")
    return _finalize(fig, save_path)


def rt_cdf(
    recovery: dict[str, pd.DataFrame],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Empirical CDF of recovery time per phase (recovered events only)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for phase, df in recovery.items():
        rt = df.loc[~df["no_recovery"], "RT"].dropna().sort_values().to_numpy(dtype=float)
        if len(rt) == 0:
            continue
        ax.step(rt, np.arange(1, len(rt) + 1) / len(rt), where="post",
                color=PHASE_COLORS.get(phase), label=f"{phase} (n={len(rt)})")
    ax.set_xlabel("recovery time (samples)")
    ax.set_ylabel("P(RT ≤ x)")
    ax.set_title("Recovery-time distribution")
    ax.legend()
    return _finalize(fig, save_path)


def pareto_cost_vs_recovery(
    points: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Pareto scatter: mean RT vs p95 latency, one point per (phase, seed).

    ``points`` columns: phase, seed, mean_RT, p95_latency_ms.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for phase, group in points.groupby("phase"):
        ax.scatter(group["p95_latency_ms"], group["mean_RT"], s=60, alpha=0.8,
                   color=PHASE_COLORS.get(str(phase), "#95a5a6"), label=str(phase))
    ax.set_xlabel("p95 latency per sample (ms)")
    ax.set_ylabel("mean recovery time (samples)")
    ax.set_title("Cost vs recovery trade-off (lower-left is better)")
    ax.legend()
    return _finalize(fig, save_path)
