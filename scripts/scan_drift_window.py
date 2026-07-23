"""Scan attack density + subtype diversity along a stream to choose a
drift-rich ``(start_row, max_samples)`` window for a dataset profile.

The label column is read by index (via ``csv.reader``) rather than
``csv.DictReader`` so a multi-GB file streams in a couple of minutes.

Usage::

    uv run python scripts/scan_drift_window.py <csv> <label_col> <benign_label> [bucket]
"""

from __future__ import annotations

import csv
import sys
from collections import Counter

csv.field_size_limit(10_000_000)


def main() -> None:
    path, label_col, benign = sys.argv[1], sys.argv[2], sys.argv[3]
    bucket = int(sys.argv[4]) if len(sys.argv) > 4 else 250_000

    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            idx = header.index(label_col)
        except ValueError:
            print(f"label column {label_col!r} not in header; columns: {header}")
            return

        n = 0
        b_idx = 0
        b_attack = 0
        b_subtypes: Counter[str] = Counter()
        print(f"scanning {path} (label col {label_col!r} @ index {idx}, bucket={bucket})\n")
        print(f"{'bucket':>7} {'row_start':>11} {'attack%':>8} {'#subtypes':>10}  top-subtypes")
        for row in reader:
            if len(row) <= idx:
                continue
            lab = row[idx].strip()
            if lab != benign:
                b_attack += 1
                b_subtypes[lab] += 1
            n += 1
            if n % bucket == 0:
                top = ", ".join(f"{k}({v})" for k, v in b_subtypes.most_common(4))
                print(
                    f"{b_idx:>7} {b_idx * bucket:>11} {100 * b_attack / bucket:>7.1f}% "
                    f"{len(b_subtypes):>10}  {top}"
                )
                b_idx += 1
                b_attack = 0
                b_subtypes = Counter()
        if n % bucket:  # trailing partial bucket
            rows = n % bucket
            top = ", ".join(f"{k}({v})" for k, v in b_subtypes.most_common(4))
            print(
                f"{b_idx:>7} {b_idx * bucket:>11} {100 * b_attack / rows:>7.1f}% "
                f"{len(b_subtypes):>10}  {top}  (partial: {rows} rows)"
            )
        print(f"\ntotal rows: {n}")


if __name__ == "__main__":
    main()
