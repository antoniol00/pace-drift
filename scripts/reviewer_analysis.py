"""Reviewer-facing analyses that need no re-run — built from existing outputs.

  fig4  resource / real-time profile: throughput, latency tail (p50..max),
        and where compute time goes (stacked). Surfaces the ~1 s training-burst
        spikes a reviewer will ask about for a streaming IDS.
  fig5  C_early operating-point sensitivity: an offline PageHinkley sweep over
        the recon threshold, replayed on the *persisted* recon-error series (no
        AE re-run). Shows the anticipation recall / false-positive trade-off and
        that the chosen threshold (0.1) is not an overfit knife-edge.

Run:  uv run python scripts/reviewer_analysis.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

# reuse the palette, sizes and save/variant logic of the paper figures
from paper_figures import ACC, BLUE, COLOR, CORE, FULL, MUTED, OUT, RUNS, SLATE_LT, save, tag  # noqa: E402
from river import drift as river_drift

from driftgan.early.dual_pht import match_lead_times

TIME_KEYS = [
    ("time_hat_inference_s", "Inference", BLUE[0]),
    ("time_hat_update_s", "Update", BLUE[2]),
    ("time_monitor_s", "Shift monitor", ACC),
    ("time_other_s", "Other", SLATE_LT),
]
PHASES = {  # top to bottom
    "B": "HAT reactive (PHT)", "B_adwin": "HAT reactive (ADWIN)",
    "B_kswin": "HAT reactive (KSWIN)", "C_freq": "HAT blind clock", "C_early": "HAT PACE",
    "ARF_sched": "ARF reactive", "ARF_early": "ARF PACE", "ARF": "ARF continuous",
    "SRP_sched": "SRP reactive", "SRP_early": "SRP PACE", "SRP": "SRP continuous",
}


def available_phases() -> list[str]:
    return [p for p in PHASES if load_costs(p)]


def load_costs(phase: str) -> list[dict]:
    """Every core seed that was run; ADWIN, KSWIN and continuous ARF/SRP only have five."""
    out = []
    for s in CORE:
        f = RUNS / phase / f"seed_{s}" / "cost.json"
        if f.exists():
            out.append(json.load(open(f)))
    return out


# --------------------------------------------------------------------------- #
# Figure 4 — resource / real-time profile
# --------------------------------------------------------------------------- #
def fig4_resources(labels: bool = False) -> None:
    """Throughput, latency percentiles and the compute share of each component.

    Arms are rows, shared by the three panels. Throughput and latency span three
    orders of magnitude, so both use a log axis and dots rather than bars; the
    breakdown is a share, since absolute seconds are dominated by continuous SRP.
    """
    phases = available_phases()
    print("  fig4 seeds:", {p: len(load_costs(p)) for p in phases})
    y = np.arange(len(phases))
    fig, (axT, axL, axB) = plt.subplots(1, 3, figsize=(FULL, 2.6), sharey=True,
                                        gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    fig.subplots_adjust(wspace=0.12)

    # -- (a) throughput (samples / s) --------------------------------------
    thru = []
    for ph in phases:
        vals = []
        for c in load_costs(ph):
            total = sum(c.get(k, 0.0) for k, _, _ in TIME_KEYS)
            vals.append(c["n_samples"] / total if total else np.nan)
        thru.append(float(np.nanmean(vals)))
    print("  fig4 throughput:", {p: round(t) for p, t in zip(phases, thru, strict=True)})
    for yi, ph, v in zip(y, phases, thru, strict=True):
        axT.plot([1e1, v], [yi, yi], color=COLOR[ph], lw=0.8, zorder=2)
        axT.scatter(v, yi, s=22, color=COLOR[ph], edgecolor="white", linewidth=0.5, zorder=3)
        if labels:
            axT.annotate(f"{v / 1e3:.1f}K", (v, yi), xytext=(4, 0), textcoords="offset points",
                         va="center", fontsize=6.5)
    axT.set_xscale("log")
    axT.set_xlim(1e1, 4e4)
    axT.set_xlabel("Throughput (samples/s)")
    axT.set_yticks(y)
    axT.set_yticklabels([PHASES[p] for p in phases])
    axT.invert_yaxis()
    tag(axT, "a")

    # -- (b) latency p50 / p95 / p99 / max (log) ---------------------------
    pcts = [("latency_p50_ms", "p50", "o"), ("latency_p95_ms", "p95", "s"),
            ("latency_p99_ms", "p99", "^"), ("latency_max_ms", "max", "D")]
    lat = np.array([[np.mean([c[k] for c in load_costs(ph)]) for k, _, _ in pcts]
                    for ph in phases])
    axL.hlines(y, lat[:, 0], lat[:, -1], color=MUTED, lw=0.7, zorder=2)
    for j, (_key, name, mk) in enumerate(pcts):
        axL.scatter(lat[:, j], y, s=16, marker=mk, color=BLUE[j + 1 if j else 0],
                    edgecolor="white", linewidth=0.4, zorder=3, label=name)
    axL.axvline(1.0, color=MUTED, lw=0.6, ls="--")
    axL.set_xscale("log")
    axL.set_xlabel("Per-sample latency (ms)")
    axL.legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=4, handletextpad=0.1,
               columnspacing=0.7, borderaxespad=0)
    tag(axL, "b")

    # -- (c) compute-time share by component -------------------------------
    secs = np.array([[np.mean([c.get(k, 0.0) for c in load_costs(ph)]) for k, _, _ in TIME_KEYS]
                     for ph in phases])
    share = 100 * secs / secs.sum(axis=1, keepdims=True)
    left = np.zeros(len(phases))
    for j, (_key, name, col) in enumerate(TIME_KEYS):
        axB.barh(y, share[:, j], 0.66, left=left, color=col, edgecolor="white",
                 linewidth=0.5, label=name)
        left += share[:, j]
    if labels:
        for yi, s in zip(y, share[:, 2], strict=True):
            if s >= 0.5:
                axB.annotate(f"{s:.0f}%", (100, yi), xytext=(3, 0), textcoords="offset points",
                             va="center", fontsize=6.5)
    axB.set_xlim(0, 100)
    axB.set_xlabel("Share of compute time (%)")
    axB.legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=2, handlelength=1.0,
               columnspacing=0.8, borderaxespad=0)
    tag(axB, "c")
    for ax in (axT, axL, axB):
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x")
    save(fig, "fig4_resources", labels)


# --------------------------------------------------------------------------- #
# Figure 5 — C_early operating-point sensitivity (offline sweep)
# --------------------------------------------------------------------------- #
def _sweep_seed(series: list, class_fires: list[int], thresholds: np.ndarray,
                delta: float, min_instances: int, horizon: int) -> dict:
    recall, fp, lead, nfires = [], [], [], []
    for t in thresholds:
        pht = river_drift.PageHinkley(delta=delta, threshold=float(t),
                                      min_instances=min_instances)
        fires = []
        for index, err in series:
            pht.update(err)
            if pht.drift_detected:
                fires.append(index)
        r = match_lead_times(fires, class_fires, horizon=horizon)
        recall.append(1.0 - r.missed_anticipation_rate)
        fp.append(r.anticipated_fp_rate)
        lead.append(r.mean_lead_time)
        nfires.append(len(fires))
    return {"recall": recall, "fp": fp, "lead": lead, "nfires": nfires}


OP_THRESHOLD = 0.1  # operating point (phase_c_early.yaml)


def fig5_sweep() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replay the recon Page-Hinkley over each seed's persisted score series."""
    delta, min_instances, horizon = 0.0005, 10, 20_000  # match phase_c_early.yaml
    thresholds = np.geomspace(0.01, 2.0, 24)
    per_seed = []
    for s in CORE:
        lt = json.load(open(RUNS / "C_early" / f"seed_{s}" / "lead_time.json"))
        series = lt.get("recon_error_series", [])
        if not series:
            continue
        per_seed.append(_sweep_seed(series, lt["class_fires"], thresholds,
                                    delta, min_instances, horizon))
    recall = np.array([p["recall"] for p in per_seed])
    fp = np.array([p["fp"] for p in per_seed])
    lead = np.array([p["lead"] for p in per_seed])
    return thresholds, recall, fp, lead


def fig5_sensitivity(sweep: tuple, labels: bool = False) -> None:
    thresholds, recall, fp, lead = sweep
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(FULL, 2.3))
    fig.subplots_adjust(wspace=0.28)

    # -- (a) recall & false-positive rate vs threshold (both rates -> one axis)
    r_m, f_m = recall.mean(0), fp.mean(0)
    axA.plot(thresholds, r_m, "-o", color=ACC, ms=3, label="Anticipation recall")
    axA.plot(thresholds, f_m, "--s", color=BLUE[1], ms=3, label="Anticipated false-alarm rate")
    axA.set_xlabel("Page-Hinkley threshold on the monitor score")
    axA.set_ylabel("Rate")
    axA.set_ylim(0, 1)
    axA.legend(loc="lower left", bbox_to_anchor=(0, 1.06), ncol=2, borderaxespad=0)
    tag(axA, "a")

    # -- (b) mean lead time vs threshold -----------------------------------
    l_m, l_sd = lead.mean(0), lead.std(0)
    axB.fill_between(thresholds, (l_m - l_sd) / 1e3, (l_m + l_sd) / 1e3,
                     color=ACC, alpha=0.18, lw=0)
    axB.plot(thresholds, l_m / 1e3, "-o", color=ACC, ms=3)
    axB.set_xlabel("Page-Hinkley threshold on the monitor score")
    axB.set_ylabel(r"Mean lead time ($\times 10^3$ samples)")
    tag(axB, "b")

    oi = int(np.argmin(np.abs(thresholds - OP_THRESHOLD)))
    for ax in (axA, axB):
        ax.set_xscale("log")
        ax.axvline(OP_THRESHOLD, color=MUTED, ls="--", lw=0.7)
        ax.grid(axis="x")
    if labels:
        axA.annotate(f"operating point {OP_THRESHOLD}\nrecall {r_m[oi]:.2f}",
                     (OP_THRESHOLD, 0.97), xytext=(3, 0), textcoords="offset points",
                     ha="left", va="top", fontsize=6.5)
        axB.annotate(f"{l_m[oi] / 1e3:.1f}K", (OP_THRESHOLD, l_m[oi] / 1e3), xytext=(3, 3),
                     textcoords="offset points", fontsize=6.5)
    print(f"  fig5 operating point: recall {r_m[oi]:.2f}, lead {l_m[oi]:.0f}, "
          f"max recall {r_m.max():.2f}")
    save(fig, "fig5_early_sensitivity", labels, (axA, axB))


def main() -> None:
    sweep = fig5_sweep()
    for labels in (False, True):
        fig4_resources(labels)
        fig5_sensitivity(sweep, labels)
    print("wrote fig4 and fig5 to", OUT)


if __name__ == "__main__":
    main()
