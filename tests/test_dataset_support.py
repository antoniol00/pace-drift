"""Multi-dataset support: benign-label threading, config layering, namespacing."""
from driftgan.data.stream import ATTACK, BENIGN, preprocess_row
from driftgan.drift.schedule import build_schedule_from_labels


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


from driftgan import config as cfg_mod
from driftgan.config import ExperimentConfig, load_config


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
