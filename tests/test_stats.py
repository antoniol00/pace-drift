"""Tests for bootstrap CIs, Cliff's delta and paired Wilcoxon."""

import numpy as np
import pandas as pd

from driftgan.experiments.stats import (
    bootstrap_ci_mean,
    cliffs_delta,
    paired_wilcoxon,
    summarize_metric_by_phase,
)


def test_bootstrap_ci_contains_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, low, high = bootstrap_ci_mean(values, n_boot=2000, seed=1)
    assert mean == 3.0
    assert low <= mean <= high
    assert low > 0 and high < 6


def test_bootstrap_ci_nan_and_empty():
    mean, low, high = bootstrap_ci_mean([np.nan, np.nan])
    assert np.isnan(mean) and np.isnan(low) and np.isnan(high)


def test_cliffs_delta_known_values():
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == 0.0
    assert cliffs_delta([10, 11], [1, 2]) == 1.0
    assert cliffs_delta([1, 2], [10, 11]) == -1.0


def test_paired_wilcoxon_detects_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(100, 10, size=30)
    df_a = pd.DataFrame({"drift_index": range(30), "RT": base + 20})  # consistently slower
    df_b = pd.DataFrame({"drift_index": range(30), "RT": base})
    cmp_ = paired_wilcoxon(df_a, df_b, "RT", "B", "C")
    assert cmp_.n_pairs == 30
    assert cmp_.p_value < 0.001
    assert cmp_.median_diff > 0
    assert cmp_.effect_r > 0.5


def test_paired_wilcoxon_identical_series_no_crash():
    df = pd.DataFrame({"drift_index": range(5), "RT": [1.0] * 5})
    cmp_ = paired_wilcoxon(df, df.copy(), "RT")
    assert np.isnan(cmp_.p_value)
    assert cmp_.median_diff == 0


def test_paired_wilcoxon_pairs_by_drift_index():
    df_a = pd.DataFrame({"drift_index": [1, 2, 3], "RT": [10.0, 20.0, 30.0]})
    df_b = pd.DataFrame({"drift_index": [2, 3, 4], "RT": [5.0, 6.0, 7.0]})
    cmp_ = paired_wilcoxon(df_a, df_b, "RT")
    assert cmp_.n_pairs == 2  # only drift_index 2 and 3 are shared


def test_summarize_metric_by_phase():
    frames = {
        "B": pd.DataFrame({"AUD": [10.0, 12.0, 11.0]}),
        "C": pd.DataFrame({"AUD": [5.0, 6.0, 7.0]}),
    }
    table = summarize_metric_by_phase(frames, "AUD", n_boot=500)
    assert set(table["phase"]) == {"B", "C"}
    row_c = table[table["phase"] == "C"].iloc[0]
    assert row_c["AUD_mean"] == 6.0
