"""End-to-end smoke test on a small synthetic CICIDS-like stream.

Builds a CSV with a benign regime followed by an attack burst whose feature
distribution differs, runs every phase for a few thousand samples, and
checks that artifacts and metrics are produced.
"""

import csv
import json

import numpy as np
import pytest

from driftgan.config import ExperimentConfig
from driftgan.drift.schedule import DriftSchedule
from driftgan.experiments.runner import run_phase

N_ROWS = 6_000
PRETRAIN = 1_000
ATTACK_START = 3_000  # index in the evaluated stream (== csv row here)


@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "stream.csv"
    rng = np.random.default_rng(42)
    cols = ["Flow ID", "Source IP", "Flow Duration", "Total Fwd Packets", "Packet Length Mean", "Attack Type"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in range(N_ROWS):
            if i < ATTACK_START:
                feats = rng.normal([100, 10, 50], [10, 2, 5])
                label = "BENIGN"
            else:
                feats = rng.normal([500, 80, 200], [30, 8, 20])
                label = "DoS_Hulk" if i % 3 else "PortScan"
            writer.writerow([f"flow-{i}", "10.0.0.1", *[f"{v:.3f}" for v in feats], label])
    return path


@pytest.fixture(scope="module")
def base_config(synthetic_csv, tmp_path_factory):
    results = tmp_path_factory.mktemp("results")

    def make(phase: str, **overrides) -> ExperimentConfig:
        defaults = dict(
            phase=phase,
            csv_path=str(synthetic_csv),
            drop_cols=("Flow ID", "Source IP"),
            start_row=0,
            n_pretrain=PRETRAIN,
            max_samples=N_ROWS,
            rolling_f1_window=500,
            snap_interval=100,
            log_interval=100_000,
            train_window=500,
            buffer_size=200,
            device="cpu",
            recovery_theta=0.5,
            recovery_k=2,
            recovery_horizon=3_000,
            monitor_interval=500,
            monitor_epochs=2,
            trace_enabled=True,
            results_dir=str(results),
        )
        defaults.update(overrides)
        return ExperimentConfig(**defaults)  # type: ignore[arg-type]

    return make


def test_phase_b_detects_the_shift(base_config):
    result = run_phase(base_config("B"), seed=0)
    assert len(result.drift_indices) >= 1
    # PHT should fire after the attack regime starts.
    assert result.drift_indices[0] >= ATTACK_START
    # Reactive model should end with usable F1 on this easy stream.
    assert result.final_f1 > 0.5
    run_dir = base_config("B").run_dir(0)
    for artifact in ("manifest.json", "snapshots.parquet", "trace.csv.gz", "cost.json"):
        assert (run_dir / artifact).exists(), artifact


def test_scheduled_phases_share_drift_indices(base_config, tmp_path):
    """Scheduled arms replay identical adaptation indices, so events pair up."""
    schedule = DriftSchedule(indices=[3_200], source="labels")
    schedule_path = tmp_path / "sched.json"
    schedule.save(schedule_path)

    results = {}
    for phase in ("B", "ARF_sched", "SRP_sched"):
        config = base_config(phase, drift_schedule=str(schedule_path))
        results[phase] = run_phase(config, seed=0)

    for phase, result in results.items():
        assert result.drift_indices == [3_200], phase

    # Recovery parquet written for each phase.
    results_dir = base_config("B").results_dir
    for phase in results:
        assert (
            __import__("pathlib").Path(results_dir) / "recovery" / f"{phase}_0.parquet"
        ).exists()


def test_live_reactive_ensemble_ignores_the_schedule(base_config, tmp_path):
    """`*_pht` fires from its own error test even when a schedule is configured."""
    schedule = DriftSchedule(indices=[3_200], source="labels")
    schedule_path = tmp_path / "sched.json"
    schedule.save(schedule_path)
    result = run_phase(base_config("ARF_pht", drift_schedule=str(schedule_path)), seed=0)
    assert result.drift_indices and result.drift_indices != [3_200]
    assert {e["source"] for e in result.drift_events} == {"pht"}


def test_frequency_control_fires_on_a_blind_clock(base_config):
    """`*_freq` adapts on the periodic clock alone — no drift signal involved."""
    config = base_config("C_freq", periodic_adapt_interval=1_000)
    result = run_phase(config, seed=0)
    assert len(result.drift_indices) >= 2
    assert {e["source"] for e in result.drift_events} == {"periodic"}


def test_phase_c_early_produces_lead_report(base_config):
    config = base_config("C_early")
    result = run_phase(config, seed=0)
    assert result.lead_time_report is not None
    report = result.lead_time_report
    assert "anticipated_fp_rate" in report and "missed_anticipation_rate" in report
    lead_path = config.run_dir(0) / "lead_time.json"
    assert lead_path.exists()
    loaded = json.loads(lead_path.read_text())
    assert loaded["n_class_fires"] == len(loaded["class_fires"])
