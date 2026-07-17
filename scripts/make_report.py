"""Aggregate all runs into the final tables and figures (``make report``).

Reads ``results/runs/*/seed_*/`` and ``results/recovery/*.parquet``, then
writes to ``results/report/``:

- ``summary_table.csv``: Phase | F1_final | mean_RT ± CI | mean_AUD ± CI |
  p95_latency_ms | peak_RSS_MB
- ``wilcoxon.csv``: paired comparisons between phases on RT and AUD
- plots: recovery boxplots, RT CDF, Pareto cost-vs-recovery
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from driftgan.config import RESULTS_DIR
from driftgan.experiments.stats import bootstrap_ci_mean, paired_wilcoxon
from driftgan.viz.plots import pareto_cost_vs_recovery, recovery_boxplots, rt_cdf


def load_recovery_frames(results_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, list[pd.DataFrame]] = {}
    for path in sorted((results_dir / "recovery").glob("*.parquet")):
        phase, seed = path.stem.rsplit("_", 1)
        df = pd.read_parquet(path)
        df["seed"] = int(seed)
        frames.setdefault(phase, []).append(df)
    return {phase: pd.concat(dfs, ignore_index=True) for phase, dfs in frames.items()}


def load_run_summaries(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((results_dir / "runs").glob("*/seed_*/summary.json")):
        rows.append(json.loads(path.read_text()))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    results_dir = args.results_dir
    report_dir = results_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    recovery = load_recovery_frames(results_dir)
    summaries = load_run_summaries(results_dir)
    if summaries.empty:
        print("no runs found — run experiments first")
        return

    # ── summary table ────────────────────────────────────────────────
    rows = []
    for phase, group in summaries.groupby("phase"):
        rec = recovery.get(str(phase), pd.DataFrame(columns=["RT", "AUD", "no_recovery"]))
        rt = rec.loc[~rec["no_recovery"], "RT"].to_numpy(dtype=float) if len(rec) else []
        aud = rec["AUD"].to_numpy(dtype=float) if len(rec) else []
        rt_mean, rt_lo, rt_hi = bootstrap_ci_mean(rt)
        aud_mean, aud_lo, aud_hi = bootstrap_ci_mean(aud)
        rows.append(
            {
                "phase": phase,
                "F1_final": group["final_f1"].mean(),
                "mean_RT": rt_mean,
                "RT_ci": f"[{rt_lo:.0f}, {rt_hi:.0f}]",
                "mean_AUD": aud_mean,
                "AUD_ci": f"[{aud_lo:.0f}, {aud_hi:.0f}]",
                "no_recovery_rate": rec["no_recovery"].mean() if len(rec) else float("nan"),
                "p95_latency_ms": group["latency_p95_ms"].mean(),
                "peak_RSS_MB": group["peak_rss_mb"].max(),
                "n_seeds": group["seed"].nunique(),
            }
        )
    table = pd.DataFrame(rows).sort_values("phase")
    table.to_csv(report_dir / "summary_table.csv", index=False)
    print(table.to_string(index=False))

    # ── paired comparisons (requires shared schedule) ────────────────
    comparisons = []
    phases = sorted(recovery)
    for i, pa in enumerate(phases):
        for pb in phases[i + 1 :]:
            for metric in ("RT", "AUD"):
                cmp_ = paired_wilcoxon(recovery[pa], recovery[pb], metric, pa, pb)
                comparisons.append(cmp_.__dict__)
    if comparisons:
        pd.DataFrame(comparisons).to_csv(report_dir / "wilcoxon.csv", index=False)

    # ── figures ──────────────────────────────────────────────────────
    if recovery:
        recovery_boxplots(recovery, "RT", save_path=report_dir / "rt_boxplot.png")
        recovery_boxplots(recovery, "AUD", save_path=report_dir / "aud_boxplot.png")
        rt_cdf(recovery, save_path=report_dir / "rt_cdf.png")

    points = []
    for _, row in summaries.iterrows():
        phase = row["phase"]
        rec = recovery.get(phase)
        if rec is None:
            continue
        seed_rec = rec[rec["seed"] == row["seed"]]
        rt = seed_rec.loc[~seed_rec["no_recovery"], "RT"]
        if len(rt) == 0:
            continue
        points.append(
            {"phase": phase, "seed": row["seed"], "mean_RT": rt.mean(),
             "p95_latency_ms": row["latency_p95_ms"]}
        )
    if points:
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        pareto_cost_vs_recovery(pd.DataFrame(points), save_path=plots_dir / "pareto_cost_vs_recovery.pdf")

    print(f"\nreport written to {report_dir}")


if __name__ == "__main__":
    main()
