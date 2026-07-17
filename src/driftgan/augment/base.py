"""Common interface for drift-time data augmenters."""

from __future__ import annotations

from typing import Protocol

import numpy as np

Buffer = list[tuple[dict[str, float], str]]  # [(features, raw subtype label), ...]


class Augmenter(Protocol):
    """Generates synthetic attack samples from a small recent buffer.

    Implementations must be deterministic given the same buffer, ``n`` and
    seeded RNG/device state, so multi-seed runs are reproducible.
    """

    def fit_generate(self, buffer: Buffer, n: int) -> list[dict[str, float]]:
        """Train (or update) on ``buffer`` and return ``n`` synthetic samples."""
        ...


def buffer_to_matrix(buffer: Buffer) -> tuple[np.ndarray, list[str], list[str]]:
    """Stack a buffer into (X, feature_names, subtype_labels)."""
    keys = list(buffer[0][0].keys())
    X = np.array([[s[k] for k in keys] for s, _ in buffer], dtype=np.float32)
    labels = [lab for _, lab in buffer]
    return X, keys, labels


def matrix_to_samples(X: np.ndarray, keys: list[str]) -> list[dict[str, float]]:
    """Inverse of :func:`buffer_to_matrix` for generated rows."""
    return [{k: float(row[j]) for j, k in enumerate(keys)} for row in X]
