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
