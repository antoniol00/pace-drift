"""#2 monitor ablation — does the autoencoder earn its keep vs cheap statistics?

Stage 1 (slow, cached): stream the CSV once per seed and feed the cheap shift
monitors (PCA / KS / MMD / mean-shift) the *same* post-pretrain samples the AE
saw, recording each one's score series to results/monitor_ablation/.

Stage 2 (fast): for every monitor — the cheap ones plus the AE (whose series and
class-fire targets come from the existing C_early runs) — build a signal-
detectability frontier. Each series is z-scored (so monitors on different scales
compare fairly), then a threshold-with-refractory detector is swept to trace
anticipation recall vs anticipated false-positive rate, paired against the same
class-drift fires. fig7 overlays the frontiers: if a cheap monitor matches the
AE, the generative model is not justified.

Run:  uv run python scripts/monitor_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from paper_figures import ACC, BLUE, FULL, RUNS, SEEDS, SLATE, SLATE_LT, save, tag, values  # noqa: E402

from driftgan.config import load_config
from driftgan.data.stream import stream_csv
from driftgan.early.dual_pht import match_lead_times
from driftgan.early.shift_monitors import MONITOR_TYPES

CACHE = Path("results/monitor_ablation")
CACHE.mkdir(parents=True, exist_ok=True)
HORIZON = 20_000
REFRACTORY = 10_000

# (colour, line style, marker) per monitor; the AE keeps PACE's deep blue
MON_STYLE = {
    "ae": (ACC, "-", "o"), "pca": (BLUE[2], "-", "s"), "ks": (BLUE[1], "--", "^"),
    "mmd": (SLATE, "-.", "v"), "mean": (SLATE_LT, ":", "D"),
}
MON_LABEL = {
    "ae": "Autoencoder", "pca": "PCA reconstruction", "ks": "KS distance",
    "mmd": "MMD", "mean": "Mean shift",
}


# --------------------------------------------------------------------------- #
# Stage 1 — collect cheap-monitor score series
# --------------------------------------------------------------------------- #
def collect_series(seed: int, cfg) -> dict[str, list]:
    out = CACHE / f"seed_{seed}.json"
    if out.exists():
        return json.loads(out.read_text())
    monitors = {
        "pca": MONITOR_TYPES["pca"](latent=cfg.monitor_latent, check_interval=cfg.monitor_interval,
                                    clip=cfg.monitor_clip, seed=seed),
        "ks": MONITOR_TYPES["ks"](check_interval=cfg.monitor_interval, clip=cfg.monitor_clip, seed=seed),
        "mmd": MONITOR_TYPES["mmd"](check_interval=cfg.monitor_interval, clip=cfg.monitor_clip, seed=seed),
        "mean": MONITOR_TYPES["mean"](check_interval=cfg.monitor_interval, clip=cfg.monitor_clip, seed=seed),
    }
    stream = stream_csv(cfg.csv_path, drop_cols=set(cfg.drop_cols), target_col=cfg.target_col,
                        start_row=cfg.start_row, max_rows=cfg.max_samples)
    n = 0
    for x, _y, _raw in stream:
        n += 1
        if n <= cfg.n_pretrain:  # skip the pretraining prefix, exactly like the runner
            continue
        for m in monitors.values():
            m.observe(x, n)
    series = {name: m.error_history for name, m in monitors.items()}
    out.write_text(json.dumps(series))
    print(f"  seed {seed}: collected {[(k, len(v)) for k, v in series.items()]}")
    return series


# --------------------------------------------------------------------------- #
# Stage 2 — detectability frontier
# --------------------------------------------------------------------------- #
def fires_at(series: list, tau: float) -> list[int]:
    """Fire when the z-scored score first exceeds tau, with a refractory gap."""
    if not series:
        return []
    idx = np.array([p[0] for p in series])
    val = np.array([p[1] for p in series], dtype=float)
    z = (val - val.mean()) / (val.std() + 1e-12)
    fires, last = [], -np.inf
    for i, zi in zip(idx, z, strict=True):
        if zi > tau and i - last >= REFRACTORY:
            fires.append(int(i))
            last = i
    return fires


def frontier(series_by_seed: dict[int, list], class_by_seed: dict[int, list],
             taus: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recall, fp, lead = [], [], []
    for tau in taus:
        rs, fs, ls = [], [], []
        for s, series in series_by_seed.items():
            r = match_lead_times(fires_at(series, tau), class_by_seed[s], horizon=HORIZON)
            rs.append(1.0 - r.missed_anticipation_rate)
            fs.append(r.anticipated_fp_rate)
            ls.append(r.mean_lead_time)
        recall.append(np.mean(rs))
        fp.append(np.mean(fs))
        lead.append(np.nanmean(ls))
    return np.array(recall), np.array(fp), np.array(lead)


def fig7_monitor_ablation(cfg) -> None:
    # gather per-seed series + class fires
    class_by_seed, ae_by_seed = {}, {}
    cheap: dict[str, dict[int, list]] = {k: {} for k in ("pca", "ks", "mmd", "mean")}
    for s in SEEDS:
        lt = json.loads((RUNS / "C_early" / f"seed_{s}" / "lead_time.json").read_text())
        class_by_seed[s] = lt["class_fires"]
        ae_by_seed[s] = lt.get("recon_error_series", [])
        series = collect_series(s, cfg)
        for k in cheap:
            cheap[k][s] = series[k]
    monitors = {"ae": ae_by_seed, **cheap}

    taus = np.linspace(0.3, 5.0, 24)
    curves, at_budget = {}, {}
    for name, sb in monitors.items():
        recall, fp, lead = frontier(sb, class_by_seed, taus)
        curves[name] = (fp, recall)
        # recall at the tightest FP budget <= 0.5 (fall back to min FP point)
        ok = fp <= 0.5
        j = int(np.argmax(recall * ok)) if ok.any() else int(np.argmin(fp))
        at_budget[name] = (recall[j], lead[j])
    names = list(monitors)
    print("recall@FP<=0.5:", {n: round(at_budget[n][0], 3) for n in names})

    for labels in (False, True):
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(FULL, 2.4),
                                       gridspec_kw={"width_ratios": [1.3, 1]})
        fig.subplots_adjust(wspace=0.28)
        for name in names:
            col, ls, mk = MON_STYLE[name]
            axA.plot(*curves[name], ls=ls, marker=mk, color=col, ms=3, lw=1.1,
                     label=MON_LABEL[name])
        axA.set_xlabel("Anticipated false-alarm rate")
        axA.set_ylabel("Anticipation recall")
        axA.set_xlim(0, 1)
        axA.set_ylim(0, 1)
        axA.grid(axis="x")
        axA.legend(loc="upper left")
        tag(axA, "a")

        x = np.arange(len(names))
        rec = [at_budget[n][0] for n in names]
        axB.bar(x, rec, 0.62, color=[MON_STYLE[n][0] for n in names], edgecolor="white")
        if labels:
            values(axB, x, rec)
        axB.set_xticks(x)
        axB.set_xticklabels(["AE", "PCA", "KS", "MMD", "Mean\nshift"])
        axB.set_ylabel(r"Recall at false-alarm rate $\leq$ 0.5")
        axB.set_ylim(0, 1)
        tag(axB, "b")
        save(fig, "fig7_monitor_ablation", labels, (axA, axB))


def main() -> None:
    cfg = load_config(Path("configs/phase_c_early.yaml"))
    fig7_monitor_ablation(cfg)


if __name__ == "__main__":
    main()
