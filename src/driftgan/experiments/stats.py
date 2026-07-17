"""Statistical reporting utilities (Improvement 4).

- Bootstrap percentile confidence intervals (no normality assumption).
- Paired Wilcoxon signed-rank between phases, paired by ``drift_index``
  (valid only when both phases used the same shared drift schedule).
- Effect sizes: Cliff's delta and r = Z / sqrt(N).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def bootstrap_ci_mean(
    values: np.ndarray | list[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap CI: returns (mean, low, high)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    low, high = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(low), float(high)


def cliffs_delta(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Cliff's delta in [-1, 1]: P(a > b) - P(a < b) over all pairs."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if len(a_arr) == 0 or len(b_arr) == 0:
        return float("nan")
    diff = a_arr[:, None] - b_arr[None, :]
    return float((diff > 0).mean() - (diff < 0).mean())


@dataclass
class PairedComparison:
    """Result of a paired Wilcoxon comparison of one metric between phases."""

    metric: str
    phase_a: str
    phase_b: str
    n_pairs: int
    median_diff: float
    p_value: float
    effect_r: float
    cliffs_delta: float


def paired_wilcoxon(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str,
    phase_a: str = "A",
    phase_b: str = "B",
    on: str = "drift_index",
) -> PairedComparison:
    """Wilcoxon signed-rank on ``metric``, pairing events by ``on``.

    Events present in only one phase (or with NaN metric) are dropped.
    Requires the shared drift schedule: without it, pairing by drift index
    is meaningless.
    """
    merged = df_a[[on, metric]].merge(df_b[[on, metric]], on=on, suffixes=("_a", "_b")).dropna()
    x = merged[f"{metric}_a"].to_numpy(dtype=float)
    y = merged[f"{metric}_b"].to_numpy(dtype=float)
    n = len(merged)
    if n < 2 or np.allclose(x, y):
        return PairedComparison(metric, phase_a, phase_b, n, float(np.median(x - y)) if n else float("nan"),
                                float("nan"), float("nan"), cliffs_delta(x, y))
    res = scipy_stats.wilcoxon(x, y, method="approx")
    z = float(getattr(res, "zstatistic", np.nan))
    return PairedComparison(
        metric=metric,
        phase_a=phase_a,
        phase_b=phase_b,
        n_pairs=n,
        median_diff=float(np.median(x - y)),
        p_value=float(res.pvalue),
        effect_r=abs(z) / np.sqrt(n) if np.isfinite(z) else float("nan"),
        cliffs_delta=cliffs_delta(x, y),
    )


def summarize_metric_by_phase(
    frames: dict[str, pd.DataFrame],
    metric: str,
    n_boot: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """``mean ± CI95`` table of a per-event metric across phases.

    ``frames`` maps phase name to its concatenated per-event dataframe
    (multiple seeds stacked).
    """
    rows = []
    for phase, df in frames.items():
        mean, low, high = bootstrap_ci_mean(df[metric].to_numpy(dtype=float), n_boot=n_boot, seed=seed)
        rows.append({"phase": phase, f"{metric}_mean": mean, f"{metric}_ci_low": low, f"{metric}_ci_high": high, "n": len(df)})
    return pd.DataFrame(rows)
