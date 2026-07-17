"""Windowed attack-subtype mix over the stream (justification for C_cond).

Produces ``results/subtype_distribution.parquet`` (one row per window with
subtype proportions) and a stacked-area plot, showing that different attack
subtypes dominate different stream segments — the reason a subtype-blind
GAN averages incompatible distributions.

Usage::

    uv run python scripts/analyze_subtype_distribution.py \
        --config configs/base.yaml --window 10000
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from driftgan.config import CONFIGS_DIR, load_config
from driftgan.data.stream import stream_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIGS_DIR / "phase_b.yaml")
    parser.add_argument("--window", type=int, default=10_000)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    max_samples = args.max_samples or config.max_samples

    rows = []
    window_counts: Counter[str] = Counter()
    n = 0
    for _, _, y_raw in stream_csv(
        config.csv_path, set(config.drop_cols), config.target_col,
        start_row=config.start_row, max_rows=max_samples,
    ):
        window_counts[y_raw] += 1
        n += 1
        if n % args.window == 0:
            total = sum(window_counts.values())
            rows.append({"window_end": n, **{k: v / total for k, v in window_counts.items()}})
            window_counts.clear()

    df = pd.DataFrame(rows).fillna(0.0)
    out_dir = Path(config.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "subtype_distribution.parquet", index=False)

    subtypes = [c for c in df.columns if c != "window_end"]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.stackplot(df["window_end"], *[df[s] for s in subtypes], labels=subtypes, alpha=0.85)
    ax.set_xlabel("stream samples")
    ax.set_ylabel("subtype proportion per window")
    ax.set_title(f"Attack-subtype mix over the stream (window = {args.window:,})")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    fig.savefig(plots_dir / "subtype_distribution.png", bbox_inches="tight", dpi=200)
    print(f"wrote {out_dir / 'subtype_distribution.parquet'} and plots/subtype_distribution.png")


if __name__ == "__main__":
    main()
