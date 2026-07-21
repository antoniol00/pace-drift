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

# reuse the exact palette/style/labels from the verdict figures
from paper_figures import COLOR, OUT, RUNS, SEEDS  # noqa: E402
from river import drift as river_drift

from driftgan.early.dual_pht import match_lead_times

TIME_KEYS = [
    ("time_hat_inference_s", "HAT inference", "#8c8c8c"),
    ("time_hat_update_s", "HAT update", "#5b5b5b"),
    ("time_monitor_s", "AE monitor", "#009E73"),
    ("time_gan_training_s", "GAN train", "#E69F00"),
    ("time_gan_generation_s", "GAN generate", "#f0c96b"),
    ("time_other_s", "other", "#d9d9d9"),
]


def available_phases() -> list[str]:
    order = ["A", "B", "C", "C_cond", "C_prime", "C_freq", "C_early"]
    return [p for p in order
            if all((RUNS / p / f"seed_{s}" / "cost.json").exists() for s in SEEDS)]


def load_costs(phase: str) -> list[dict]:
    out = []
    for s in SEEDS:
        f = RUNS / phase / f"seed_{s}" / "cost.json"
        if f.exists():
            out.append(json.load(open(f)))
    return out


# --------------------------------------------------------------------------- #
# Figure 4 — resource / real-time profile
# --------------------------------------------------------------------------- #
def fig4_resources() -> None:
    phases = available_phases()
    fig, (axT, axL, axB) = plt.subplots(1, 3, figsize=(15.5, 4.8),
                                        gridspec_kw={"width_ratios": [1, 1.2, 1.3]})
    x = np.arange(len(phases))

    # -- A: throughput (samples / s) ---------------------------------------
    thru = []
    for ph in phases:
        cs = load_costs(ph)
        vals = []
        for c in cs:
            total = sum(c.get(k, 0.0) for k, _, _ in TIME_KEYS)
            vals.append(c["n_samples"] / total if total else np.nan)
        thru.append(np.nanmean(vals))
    axT.bar(x, thru, color=[COLOR[p] for p in phases], edgecolor="white")
    for xi, v in zip(x, thru, strict=False):
        axT.annotate(f"{v/1e3:.1f}K", (xi, v), xytext=(0, 3),
                     textcoords="offset points", ha="center", fontsize=8,
                     fontweight="bold")
    axT.set_xticks(x)
    axT.set_xticklabels(phases, rotation=30, ha="right", fontsize=8)
    axT.set_ylabel("throughput (samples / s)")
    axT.set_title("A. Throughput")

    # -- B: latency tail p50/p95/p99/max (log) -----------------------------
    pcts = [("latency_p50_ms", "p50"), ("latency_p95_ms", "p95"),
            ("latency_p99_ms", "p99"), ("latency_max_ms", "max")]
    w = 0.2
    shades = [1.0, 0.75, 0.5, 0.32]
    for j, (key, name) in enumerate(pcts):
        vals = [np.mean([c[key] for c in load_costs(ph)]) for ph in phases]
        bars = axL.bar(x + (j - 1.5) * w, vals, w,
                       color=[COLOR[p] for p in phases], alpha=shades[j],
                       edgecolor="white", label=name)
        del bars
    axL.set_yscale("log")
    axL.set_xticks(x)
    axL.set_xticklabels(phases, rotation=30, ha="right", fontsize=8)
    axL.set_ylabel("per-sample latency (ms, log)")
    axL.set_title("B. Latency: median → worst-case (p50/p95/p99/max)")
    axL.axhline(1.0, color="#888", lw=0.8, ls="--")
    axL.annotate("1 ms", (x[-1] + 0.3, 1.0), fontsize=8, color="#888", va="bottom")
    axL.text(0.5, 0.97, "sub-ms typical, but GAN/AE training bursts spike the "
             "max to ~1 s", transform=axL.transAxes, ha="center", va="top",
             fontsize=8.5, color="#555")
    # simple percentile legend via grey ramp
    axL.legend(title="percentile", fontsize=8, title_fontsize=8, ncol=4,
               loc="lower center")

    # -- C: compute-time breakdown (stacked, mean seconds) -----------------
    bottoms = np.zeros(len(phases))
    for key, name, col in TIME_KEYS:
        vals = np.array([np.mean([c.get(key, 0.0) for c in load_costs(ph)])
                         for ph in phases])
        axB.bar(x, vals, bottom=bottoms, color=col, edgecolor="white",
                linewidth=0.6, label=name)
        bottoms += vals
    axB.set_xticks(x)
    axB.set_xticklabels(phases, rotation=30, ha="right", fontsize=8)
    axB.set_ylabel("compute time over the run (s)")
    axB.set_title("C. Where the time goes")
    axB.legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.savefig(OUT / "fig4_resources.png")
    plt.close(fig)


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


def fig5_sensitivity() -> None:
    # operating point + sweep params (match phase_c_early.yaml)
    delta, min_instances, horizon, op_threshold = 0.0005, 10, 20_000, 0.1
    thresholds = np.geomspace(0.01, 2.0, 24)

    per_seed = []
    for s in SEEDS:
        lt = json.load(open(RUNS / "C_early" / f"seed_{s}" / "lead_time.json"))
        series = lt.get("recon_error_series", [])
        if not series:
            continue
        per_seed.append(_sweep_seed(series, lt["class_fires"], thresholds,
                                    delta, min_instances, horizon))
    recall = np.array([p["recall"] for p in per_seed])
    fp = np.array([p["fp"] for p in per_seed])
    lead = np.array([p["lead"] for p in per_seed])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.8))
    green = COLOR["C_early"]

    # -- A: recall & false-positive vs threshold (both are rates -> one axis)
    r_m, f_m = recall.mean(0), fp.mean(0)
    oi = int(np.argmin(np.abs(thresholds - op_threshold)))
    axA.plot(thresholds, r_m, "-o", color=green, ms=4, lw=1.8,
             label="anticipation recall (drifts foreseen)")
    axA.plot(thresholds, f_m, "-s", color="#E69F00", ms=4, lw=1.8,
             label="anticipated false-positive rate")
    axA.axvline(op_threshold, color="#d55e00", ls="--", lw=1.3)
    axA.annotate(f"operating\npoint = {op_threshold}\n(recall {r_m[oi]:.2f})",
                 (op_threshold, 0.9), xytext=(6, 0), textcoords="offset points",
                 ha="left", va="top", fontsize=8.5, color="#d55e00")
    axA.set_xscale("log")
    axA.set_xlabel("recon PageHinkley threshold (log)")
    axA.set_ylabel("rate")
    axA.set_title("A. Recall / false-positive vs threshold")
    axA.set_ylim(0, 1)
    axA.legend(loc="upper right", fontsize=8.5)

    # -- B: mean lead time vs threshold ------------------------------------
    l_m, l_sd = lead.mean(0), lead.std(0)
    axB.fill_between(thresholds, (l_m - l_sd) / 1e3, (l_m + l_sd) / 1e3,
                     color=green, alpha=0.15, lw=0)
    axB.plot(thresholds, l_m / 1e3, "-o", color=green, ms=4, lw=1.8)
    axB.axvline(op_threshold, color="#d55e00", ls="--", lw=1.3)
    axB.annotate(f"operating point = {op_threshold}", (op_threshold, axB.get_ylim()[1]),
                 xytext=(6, -14), textcoords="offset points", fontsize=8.5,
                 color="#d55e00")
    axB.set_xscale("log")
    axB.set_xlabel("recon PageHinkley threshold (log)")
    axB.set_ylabel("mean anticipation lead (thousands of samples)")
    axB.set_title("B. Lead time is stable across a wide threshold band")
    fig.savefig(OUT / "fig5_early_sensitivity.png")
    plt.close(fig)


def main() -> None:
    fig4_resources()
    fig5_sensitivity()
    print("wrote:", OUT / "fig4_resources.png", "and", OUT / "fig5_early_sensitivity.png")


if __name__ == "__main__":
    main()
