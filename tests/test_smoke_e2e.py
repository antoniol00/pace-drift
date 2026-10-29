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


@pytest.mark.parametrize("phase", ["B_pht", "ARF_pht"])
def test_live_reactive_control_ignores_the_schedule(base_config, tmp_path, phase):
    """`*_pht` fires from its own error test even when a schedule is configured."""
    schedule = DriftSchedule(indices=[3_200], source="labels")
    schedule_path = tmp_path / "sched.json"
    schedule.save(schedule_path)
    result = run_phase(base_config(phase, drift_schedule=str(schedule_path)), seed=0)
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


def test_oracle_adversary_fires_the_monitor_branch(base_config, tmp_path):
    """The oracle replaces the monitor: its indices appear as recon fires."""
    path = tmp_path / "adv.json"
    DriftSchedule(indices=[3_100, 4_100], source="adversary_attack").save(path)
    result = run_phase(base_config("C_early_adv", adversary_schedule=str(path)), seed=0)
    assert result.lead_time_report is not None
    assert result.lead_time_report["recon_fires"] == [3_100, 4_100]
    assert {3_100, 4_100} <= set(result.drift_indices)


def test_chaff_stays_out_of_the_metrics(base_config, tmp_path):
    path = tmp_path / "chaff.json"
    DriftSchedule(indices=[2_000, 3_000], source="adversary_attack").save(path)
    config = base_config("C_early_chaff", chaff_schedule=str(path), chaff_size=300, chaff_pool=500)
    result = run_phase(config, seed=0)
    assert result.n_processed == N_ROWS
    assert result.label_dist["CHAFF"] == 600
    assert len(result.snap_indices) == (N_ROWS - PRETRAIN) // 100


def test_rebalanced_window_feeds_the_prior(base_config):
    """With the defence, what the learner is fed keeps the pinned attack share."""
    result = run_phase(base_config("B_pht_rb", rebalance_prior=0.2), seed=0)
    fed_attack = result.learned.get("ATTACK", 0) + result.learned.get("replay_ATTACK", 0)
    fed = fed_attack + result.learned.get("BENIGN", 0) + result.learned.get("replay_BENIGN", 0)
    assert result.learned.get("replay_BENIGN", 0) > 0
    assert fed_attack / fed <= 0.2 + 1e-9


def test_label_delay_shifts_the_error_test(base_config):
    """Until the first firing nothing is learned, so the error sequence is the
    same and a delayed error test fires exactly ``label_delay`` flows later."""
    immediate = run_phase(base_config("B_pht"), seed=0, write_outputs=False)
    delayed = run_phase(base_config("B_pht_d5k", label_delay=200), seed=0, write_outputs=False)
    assert delayed.drift_indices[0] == immediate.drift_indices[0] + 200


def test_labels_that_never_arrive_are_never_learned(base_config):
    """The monitor needs no label: with every label still in flight PACE can fire
    only from it, opens windows, and the learner is fed nothing."""
    result = run_phase(base_config("C_early_d5k", label_delay=N_ROWS), seed=0, write_outputs=False)
    assert result.learned == {}
    assert result.drift_events and {e["source"] for e in result.drift_events} == {"recon"}
