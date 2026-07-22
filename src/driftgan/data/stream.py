"""Streaming CSV reader and row-by-row preprocessing.

Benign label is configurable to support multiple datasets.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterator

Sample = tuple[dict[str, float], str, str]  # (features, binary label, raw subtype label)

BENIGN = "BENIGN"
ATTACK = "ATTACK"


def preprocess_row(
    row: dict[str, str],
    drop_cols: frozenset[str] | set[str],
    target_col: str,
    benign_label: str = BENIGN,
) -> Sample:
    """Transform a raw CSV row into ``(x, y, y_raw)`` for river.

    - Discards columns without algorithmic value (identifiers, timestamps).
    - Converts numeric values to float; non-numeric columns are dropped.
    - Replaces infinite/NaN values with 0.0.
    - Binarizes the target: ``BENIGN`` vs ``ATTACK``; the original subtype
      label is preserved in ``y_raw`` (used by the conditional GAN and the
      label-based drift schedule).
    """
    y_raw = row.pop(target_col, benign_label).strip()
    y = BENIGN if y_raw == benign_label else ATTACK

    x: dict[str, float] = {}
    for col, val in row.items():
        col_clean = col.strip()
        if col_clean in drop_cols:
            continue
        try:
            v = float(val)
        except (ValueError, TypeError):
            continue
        if math.isinf(v) or math.isnan(v):
            v = 0.0
        x[col_clean] = v
    return x, y, y_raw


def stream_csv(
    path: str,
    drop_cols: frozenset[str] | set[str],
    target_col: str,
    start_row: int = 0,
    max_rows: int | None = None,
    benign_label: str = BENIGN,
) -> Iterator[Sample]:
    """Yield ``(x, y, y_raw)`` row by row starting at ``start_row``.

    The file is never loaded fully into memory. Rows whose feature dict ends
    up empty after preprocessing are skipped and do not count toward
    ``max_rows``.
    """
    count = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for _ in range(start_row):
            next(reader, None)
        for row in reader:
            x, y, y_raw = preprocess_row(dict(row), drop_cols, target_col, benign_label)
            if not x:
                continue
            yield x, y, y_raw
            count += 1
            if max_rows is not None and count >= max_rows:
                break
