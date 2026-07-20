"""Common interface for drift-time data augmenters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

Buffer = list[tuple[dict[str, float], str]]  # [(features, raw subtype label), ...]


@dataclass
class FeatureScaler:
    """Per-feature standardizer used to train generators in a well-scaled space.

    CICIDS2017 features span many orders of magnitude (single features range
    into the millions), which a small MLP generator cannot fit directly: it
    collapses to near-origin output regardless of the target distribution.
    Fitting the generator on standardized features and inverse-transforming
    the samples back restores the real scale, so different generators produce
    genuinely different — and realistic-scale — synthetic attacks.

    Constant features (``sd == 0``) get ``sd = 1`` so the transform is a plain
    mean-shift and the inverse is exact.
    """

    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> FeatureScaler:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
        return cls(mu.astype(np.float32), sd)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def inverse(self, X: np.ndarray) -> np.ndarray:
        return X * self.sd + self.mu


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
