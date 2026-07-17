"""Tests for the cost tracker."""

import time

from driftgan.metrics.cost import CostTracker


def test_latency_and_sections_accounting():
    tracker = CostTracker(rss_sample_interval=2)
    for _ in range(4):
        with tracker.sample():
            with tracker.section("hat_inference"):
                time.sleep(0.001)
            with tracker.section("gan_training"):
                time.sleep(0.002)
    s = tracker.summary()
    assert s["n_samples"] == 4
    assert s["latency_p50_ms"] >= 3.0
    assert s["latency_max_ms"] >= s["latency_p50_ms"]
    assert s["time_gan_training_s"] > s["time_hat_inference_s"] > 0
    assert s["time_other_s"] >= 0
    assert s["peak_rss_mb"] > 0


def test_save_roundtrip(tmp_path):
    import json

    tracker = CostTracker()
    with tracker.sample():
        pass
    path = tmp_path / "cost.json"
    tracker.save(path)
    loaded = json.loads(path.read_text())
    assert loaded["n_samples"] == 1
