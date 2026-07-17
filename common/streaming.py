"""Streaming reading of the CSV and row-by-row preprocessing."""

import csv
import math

from common.config import DROP_COLS, TARGET_COL


def preprocess_row(row: dict, drop_cols: set = DROP_COLS, target_col: str = TARGET_COL):
    """
    Transforms a CSV row into (x, y) for river.

    - Discards columns without algorithmic value.
    - Converts numeric values to float.
    - Replaces infinite/NaN values with 0.0.
    - Binarizes the target: 'BENIGN' vs 'ATTACK'.
    """
    y_raw = row.pop(target_col, "BENIGN").strip()
    y = "BENIGN" if y_raw == "BENIGN" else "ATTACK"

    x = {}
    for col, val in row.items():
        col_clean = col.strip()
        if col_clean in drop_cols:
            continue
        try:
            v = float(val)
            if math.isinf(v) or math.isnan(v):
                v = 0.0
            x[col_clean] = v
        except (ValueError, TypeError):
            continue

    return x, y, y_raw


def stream_csv(path: str, start_row: int = 0, max_rows: int = None):
    """Generator that reads the CSV row by row starting from 'start_row'."""
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        if start_row > 0:
            print(f"⏩ Skipping the first {start_row:,} rows...")
            for _ in range(start_row):
                next(reader, None)

        print("🚀 Stream generator started.")
        for row in reader:
            x, y, y_raw = preprocess_row(dict(row))
            if not x:
                continue
            yield x, y, y_raw
            count += 1
            if max_rows and count >= max_rows:
                break
