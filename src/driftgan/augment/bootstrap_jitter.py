"""Naive oversampling baseline (Phase C'): bootstrap resampling + Gaussian jitter.

Purpose (Improvement 2): a cheap augmentation control. If Phase C (GAN) does
not beat this baseline, any improvement over Phase B is attributable to
"more ATTACK samples", not to the GAN learning useful structure.
"""

from __future__ import annotations

import numpy as np

from driftgan.augment.base import Buffer, buffer_to_matrix, matrix_to_samples


class BootstrapJitterAugmenter:
    """Resample the buffer with replacement and add low-variance Gaussian noise.

    Noise std per feature is ``sigma_scale * std(feature)`` computed on the
    buffer; features with zero variance receive no noise (this also protects
    integer-coded / constant features from meaningless perturbation).
    """

    def __init__(self, sigma_scale: float = 0.01, seed: int = 0):
        self.sigma_scale = sigma_scale
        self._rng = np.random.default_rng(seed)

    def fit_generate(self, buffer: Buffer, n: int) -> list[dict[str, float]]:
        import time

        self.t_train = 0.0  # no training phase in this baseline
        t0 = time.perf_counter()
        if len(buffer) == 0:
            self.t_generate = 0.0
            return []
        X, keys, _ = buffer_to_matrix(buffer)
        idx = self._rng.integers(0, len(X), size=n)
        resampled = X[idx].astype(np.float64)
        sigma = self.sigma_scale * X.std(axis=0)
        noise = self._rng.normal(0.0, 1.0, size=resampled.shape) * sigma
        samples = matrix_to_samples(resampled + noise, keys)
        self.t_generate = time.perf_counter() - t0
        return samples
