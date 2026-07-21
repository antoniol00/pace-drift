"""Tests for shared drift schedules and triggers."""

import pytest

from driftgan.drift.schedule import (
    DriftSchedule,
    DynamicPHTTrigger,
    PeriodicTrigger,
    ScheduledDriftTrigger,
    build_schedule_from_labels,
    load_schedule,
)


def test_periodic_trigger_fires_on_clock_after_warmup():
    trig = PeriodicTrigger(interval=10, start_after=5)
    fires = [i for i in range(46) if trig.update(i, error=0)]
    # first fire at start_after + interval = 15, then every 10
    assert fires == [15, 25, 35, 45]
    assert trig.fired_indices == fires


def test_periodic_trigger_ignores_error_signal():
    a = PeriodicTrigger(interval=7)
    b = PeriodicTrigger(interval=7)
    fa = [i for i in range(30) if a.update(i, error=0)]
    fb = [i for i in range(30) if b.update(i, error=1)]
    assert fa == fb  # firing depends only on the clock, never the error


def test_periodic_trigger_rejects_nonpositive_interval():
    with pytest.raises(ValueError):
        PeriodicTrigger(interval=0)


def test_dynamic_detector_trigger_fires_on_sustained_error():
    from river import drift as river_drift

    from driftgan.drift.schedule import DynamicDetectorTrigger

    trig = DynamicDetectorTrigger(river_drift.ADWIN(), name="adwin")
    # long clean run then a burst of errors -> ADWIN flags a change
    fired = False
    for i in range(2000):
        err = 0 if i < 1500 else 1
        if trig.update(i, err):
            fired = True
    assert fired and trig.name == "adwin"
    assert trig.fired_indices and all(idx >= 1500 for idx in trig.fired_indices)


def test_scheduled_trigger_fires_exactly_at_index():
    schedule = DriftSchedule(indices=[10, 25], source="labels")
    trig = ScheduledDriftTrigger(schedule)
    fires = [i for i in range(40) if trig.update(i, error=0)]
    assert fires == [10, 25]
    assert trig.fired_indices == [10, 25]


def test_scheduled_trigger_skipped_index_does_not_fire_late():
    # If the stream skips the exact index (e.g. filtered row), the event is
    # dropped rather than fired at index+eps: adaptation timing must never
    # depend on the phase's own dynamics.
    schedule = DriftSchedule(indices=[10], source="labels")
    trig = ScheduledDriftTrigger(schedule)
    fires = [i for i in [8, 9, 11, 12] if trig.update(i, error=0)]
    assert fires == []


def test_report_only_pht_never_triggers():
    schedule = DriftSchedule(indices=[10_000], source="pht")
    trig = ScheduledDriftTrigger(schedule, report_pht_delta=0.002)
    # Constant high error would make a dynamic PHT fire, but adaptation
    # must only trigger at the scheduled index.
    fires = [i for i in range(3000) if trig.update(i, error=1)]
    assert fires == []


def test_dynamic_trigger_records_fires():
    trig = DynamicPHTTrigger(delta=0.002)
    fired = False
    for i in range(5000):
        error = 0 if i < 1000 else 1  # abrupt error increase
        if trig.update(i, error):
            fired = True
            break
    assert fired
    assert len(trig.fired_indices) == 1


def test_schedule_save_load_roundtrip(tmp_path):
    schedule = DriftSchedule(indices=[5, 10], source="labels", metadata={"min_gap": 3})
    path = tmp_path / "drift_schedule.json"
    schedule.save(path)
    loaded = load_schedule(path)
    assert loaded.indices == [5, 10]
    assert loaded.source == "labels"
    assert loaded.metadata["min_gap"] == 3


def test_build_schedule_from_labels_blocks_and_min_gap():
    labels = (
        ["BENIGN"] * 10
        + ["DoS_Hulk"] * 5
        + ["BENIGN"] * 2
        + ["DoS_Hulk"] * 5  # same subtype resumes: no new event
        + ["PortScan"] * 5  # new subtype -> event
        + ["BENIGN"] * 3
        + ["Bot"] * 2  # new subtype but within min_gap of previous event
    )
    schedule = build_schedule_from_labels(labels, min_gap=100, n_pretrain=0)
    assert schedule.indices == [10]  # PortScan/Bot suppressed by min_gap

    schedule2 = build_schedule_from_labels(labels, min_gap=5, n_pretrain=0)
    assert schedule2.indices == [10, 22, 30]


def test_build_schedule_from_labels_skips_pretrain():
    labels = ["BENIGN"] * 5 + ["DoS_Hulk"] * 5 + ["BENIGN"] * 5 + ["PortScan"] * 5
    schedule = build_schedule_from_labels(labels, min_gap=1, n_pretrain=8)
    assert schedule.indices == [15]
