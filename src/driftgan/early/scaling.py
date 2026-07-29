"""Feature standardization and device resolution for the shift monitors.

Flow features span many orders of magnitude — single columns range into the
millions — so every monitor standardizes before it measures. Doing so is what
makes a reconstruction error, a PCA basis or a logistic fit numerically
meaningful across features; without it the widest columns dominate the statistic
and drift in the rest becomes invisible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FeatureScaler:
    """Per-feature standardizer, frozen after it is first fitted.

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


def resolve_device(requested: str) -> torch.device:
    """Return the requested device, silently falling back to CPU."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)
