"""Analytic test cases for RT / AUD / no-recovery semantics."""

import numpy as np
import pandas as pd
import pytest

from driftgan.metrics.recovery import (
    aggregate_recovery,
    compute_recovery_events,
    events_to_dataframe,
)

SNAP = 500  # snapshot spacing used in all synthetic cases


def make_series(f1_values, start=SNAP, valid=None):
    """Snapshots at start, start+SNAP, ... with given F1 values."""
    idx = np.arange(start, start + SNAP * len(f1_values), SNAP)
    f1 = np.array(f1_values, dtype=float)
    valid = np.ones(len(f1), dtype=bool) if valid is None else np.array(valid, dtype=bool)
    return idx, f1, valid


def test_immediate_recovery_rt_and_aud():
    # Drift at 0. F1: 0.5, 0.5, then >= theta for k=3 consecutive snapshots.
    idx, f1, valid = make_series([0.5, 0.5, 0.95, 0.95, 0.95, 0.95])
    events = compute_recovery_events(idx, f1, valid, [0], theta=0.9, k=3, horizon=50_000)
    assert len(events) == 1
    e = events[0]
    # Recovery run starts at the 3rd snapshot (index 1500).
    assert e.RT == 1500
    assert not e.no_recovery
    # AUD = sum over valid snapshots up to recovery snapshot (inclusive):
    # (0.9-0.5)*500 + (0.9-0.5)*500 + 0*500 = 400.
    assert e.AUD == pytest.approx(400.0)
    assert e.min_F1 == pytest.approx(0.5)


def test_no_recovery_within_horizon():
    idx, f1, valid = make_series([0.5] * 10)
    events = compute_recovery_events(idx, f1, valid, [0], theta=0.9, k=3, horizon=50_000)
    e = events[0]
    assert e.no_recovery
    assert e.RT is None
    # Deficit 0.4 held over 10 snapshots x 500 samples each = 2000.
    assert e.AUD == pytest.approx(0.4 * 500 * 10)


def test_sustained_requirement_not_met_by_short_run():
    # Two snapshots above theta, then a dip: run of k=3 never completes.
    idx, f1, valid = make_series([0.95, 0.95, 0.5, 0.95, 0.95, 0.5])
    events = compute_recovery_events(idx, f1, valid, [0], theta=0.9, k=3, horizon=50_000)
    assert events[0].no_recovery


def test_overlapping_drift_closes_event_as_no_recovery():
    # Drifts at 0 and 1500; the first event window is truncated at 1500
    # before any k=3 run can complete.
    idx, f1, valid = make_series([0.5, 0.95, 0.95, 0.95, 0.95, 0.95])
    events = compute_recovery_events(idx, f1, valid, [0, 1500], theta=0.9, k=3, horizon=50_000)
    assert len(events) == 2
    first, second = events
    assert first.no_recovery  # only snapshots at 500,1000,1500 in window -> max run 2 above theta?
    # window (0, 1500]: f1 = 0.5, 0.95, 0.95 -> run of 2 < k -> no recovery
    assert first.RT is None
    # Second event window (1500, 51500]: snapshots 2000,2500,3000 all >= theta.
    assert not second.no_recovery
    assert second.RT == 2000 - 1500


def test_invalid_windows_are_excluded_not_failed():
    # Middle snapshots have no positive labels (valid=False) with F1
    # recorded as 0; they must not break the sustained run nor add to AUD.
    idx, f1, valid = make_series(
        [0.95, 0.0, 0.0, 0.95, 0.95],
        valid=[True, False, False, True, True],
    )
    events = compute_recovery_events(idx, f1, valid, [0], theta=0.9, k=3, horizon=50_000)
    e = events[0]
    assert not e.no_recovery
    # Valid snapshots are at 500, 2000, 2500 — the k=3 run starts at 500.
    assert e.RT == 500
    assert e.AUD == pytest.approx(0.9 - 0.95 if False else 0.0)  # deficit 0 at recovery snapshot
    assert e.excluded_windows_pct == pytest.approx(40.0)


def test_event_with_only_invalid_snapshots():
    idx, f1, valid = make_series([0.0, 0.0], valid=[False, False])
    events = compute_recovery_events(idx, f1, valid, [0], theta=0.9, k=3, horizon=50_000)
    e = events[0]
    assert e.no_recovery
    assert e.RT is None
    assert e.excluded_windows_pct == pytest.approx(100.0)
    assert np.isnan(e.min_F1)


def test_horizon_truncates_event():
    # Recovery would happen at snapshot 6 but horizon cuts at 2000.
    idx, f1, valid = make_series([0.5, 0.5, 0.5, 0.5, 0.95, 0.95, 0.95])
    events = compute_recovery_events(idx, f1, valid, [0], theta=0.9, k=3, horizon=2000)
    e = events[0]
    assert e.no_recovery
    # AUD only integrates snapshots at 500..2000.
    assert e.AUD == pytest.approx(0.4 * 500 * 4)


def test_aggregate_and_dataframe_roundtrip():
    idx, f1, valid = make_series([0.5, 0.95, 0.95, 0.95])
    events = compute_recovery_events(idx, f1, valid, [0], theta=0.9, k=3, horizon=50_000)
    df = events_to_dataframe(events)
    assert set(df.columns) >= {"event_id", "drift_index", "RT", "AUD", "min_F1", "no_recovery"}
    agg = aggregate_recovery(df)
    assert agg["n_events"].iloc[0] == 1
    assert agg["RT_mean"].iloc[0] == 1000


def test_aggregate_empty():
    agg = aggregate_recovery(events_to_dataframe([]))
    assert agg["n_events"].iloc[0] == 0
    assert pd.isna(agg["RT_mean"].iloc[0]) or np.isnan(agg["RT_mean"].iloc[0])
