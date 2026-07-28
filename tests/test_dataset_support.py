"""Multi-dataset support: benign-label threading, config layering, namespacing."""
from driftgan import config as cfg_mod
from driftgan.config import ExperimentConfig, load_config
from driftgan.data.stream import ATTACK, BENIGN, preprocess_row
from driftgan.drift.schedule import build_schedule_from_labels
from driftgan.metrics.recovery import save_recovery


def test_preprocess_row_custom_benign_label():
    row = {"f1": "1.0", "f2": "2.0", "Label": "Benign"}
    x, y, y_raw = preprocess_row(row, drop_cols=set(), target_col="Label", benign_label="Benign")
    assert y == BENIGN
    assert y_raw == "Benign"
    assert x == {"f1": 1.0, "f2": 2.0}


def test_preprocess_row_attack_under_custom_benign():
    row = {"f1": "1.0", "Label": "DoS"}
    _, y, _ = preprocess_row(row, drop_cols=set(), target_col="Label", benign_label="Benign")
    assert y == ATTACK


def test_preprocess_row_2017_string_is_attack_for_other_dataset():
    # When the dataset's benign class is "Benign", the literal "BENIGN" is a
    # normal attack subtype string, not the benign class.
    row = {"f1": "1.0", "Label": "BENIGN"}
    _, y, _ = preprocess_row(row, drop_cols=set(), target_col="Label", benign_label="Benign")
    assert y == ATTACK


def test_schedule_from_labels_custom_benign():
    labels = ["Normal"] * 10 + ["DoS"] * 3 + ["Normal"] * 10 + ["Worms"] * 3
    sched = build_schedule_from_labels(labels, min_gap=1, n_pretrain=0, benign_label="Normal")
    assert sched.indices == [10, 23]


def test_schedule_from_labels_default_benign_unchanged():
    labels = ["BENIGN"] * 5 + ["DoS"] * 2 + ["BENIGN"] * 5 + ["DDoS"] * 2
    sched = build_schedule_from_labels(labels, min_gap=1, n_pretrain=0)
    assert sched.indices == [5, 12]


def test_config_defaults_dataset_and_benign():
    c = ExperimentConfig(phase="B")
    assert c.dataset == "cicids2017"
    assert c.benign_label == "BENIGN"


def test_run_dir_flat_for_default_dataset():
    c = ExperimentConfig(phase="B", results_dir="/tmp/r")
    assert c.run_dir(0).as_posix().endswith("/runs/B/seed_0")


def test_run_dir_namespaced_for_named_dataset():
    c = ExperimentConfig(phase="B", dataset="cicids2018", results_dir="/tmp/r")
    assert c.run_dir(0).as_posix().endswith("/runs/cicids2018/B/seed_0")


def test_load_config_dataset_layer_overrides_phase(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    (configs / "datasets").mkdir(parents=True)
    (configs / "base.yaml").write_text("recon_pht_threshold: 2.0\nstart_row: 0\n")
    (configs / "datasets" / "ds.yaml").write_text(
        "dataset: ds\ncsv_path: /x.csv\ntarget_col: Label\n"
        "benign_label: Benign\nrecon_pht_threshold: 0.05\n"
    )
    phase_yaml = configs / "phase_c_early.yaml"
    phase_yaml.write_text("phase: C_early\nrecon_pht_threshold: 0.1\n")
    monkeypatch.setattr(cfg_mod, "CONFIGS_DIR", configs)
    c = load_config(phase_yaml, dataset="ds")
    assert c.dataset == "ds"
    assert c.benign_label == "Benign"
    assert c.csv_path == "/x.csv"
    assert c.recon_pht_threshold == 0.05  # dataset layer overrides the phase default


def test_dataset_by_phase_block_overrides_phase_config(tmp_path, monkeypatch):
    """A (dataset, phase)-specific calibrated value must beat the phase default.

    The blind-clock interval of the `*_freq` controls is calibrated per learner
    *and* per dataset so the control stays fire-count matched. It used to live
    only in a CLI flag, and omitting it silently de-matched the control and
    invalidated a batch of runs; the value now lives in the dataset profile and
    this pins that it wins over the phase config.
    """
    configs = tmp_path / "configs"
    (configs / "datasets").mkdir(parents=True)
    (configs / "base.yaml").write_text("periodic_adapt_interval: 12000\nstart_row: 0\n")
    (configs / "datasets" / "ds.yaml").write_text(
        "dataset: ds\ncsv_path: /x.csv\ntarget_col: Label\nbenign_label: Benign\n"
        "by_phase:\n"
        "  ARF_freq:\n    periodic_adapt_interval: 9549\n"
        "  SRP_freq:\n    periodic_adapt_interval: 9857\n"
    )
    arf = configs / "phase_arf_freq.yaml"
    arf.write_text("phase: ARF_freq\nperiodic_adapt_interval: 11587\n")
    srp = configs / "phase_srp_freq.yaml"
    srp.write_text("phase: SRP_freq\nperiodic_adapt_interval: 11505\n")
    monkeypatch.setattr(cfg_mod, "CONFIGS_DIR", configs)

    assert load_config(arf, dataset="ds").periodic_adapt_interval == 9549
    assert load_config(srp, dataset="ds").periodic_adapt_interval == 9857
    # No dataset layer -> the phase default stands, and `by_phase` never leaks
    # into the config object.
    plain = load_config(arf)
    assert plain.periodic_adapt_interval == 11587
    assert "by_phase" not in plain.extra
    # An explicit CLI override still wins over the pinned value.
    assert load_config(
        arf, overrides={"periodic_adapt_interval": 7}, dataset="ds"
    ).periodic_adapt_interval == 7


def test_save_recovery_flat_for_default(tmp_path):
    p = save_recovery([], "B", 0, tmp_path)
    assert p.name == "B_0.parquet"


def test_save_recovery_namespaced(tmp_path):
    p = save_recovery([], "B", 0, tmp_path, dataset="cicids2018")
    assert p.name == "cicids2018_B_0.parquet"
