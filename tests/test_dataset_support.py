"""Multi-dataset support: benign-label threading, config layering, namespacing."""
from driftgan.data.stream import ATTACK, BENIGN, preprocess_row


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
