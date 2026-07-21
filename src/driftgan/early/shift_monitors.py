"""Cheap covariate-shift monitors — ablations of the autoencoder monitor.

The autoencoder in :mod:`driftgan.early.reconstruction_monitor` is the novelty
of the early-detection phase, so a reviewer will ask whether a *generative*
model is doing anything a cheap statistic cannot. These monitors answer that:
they all share the AE's exact protocol — a frozen reference frame set on the
first check, then, on every later check, **score before updating the reference**
so a distribution change shows up as a leading spike — and differ only in the
shift statistic:

- ``PCAReconstructionMonitor``  linear reconstruction error (PCA via SVD)
- ``KSMonitor``                 mean per-feature Kolmogorov–Smirnov distance
- ``MMDMonitor``               RBF maximum mean discrepancy (kernel two-sample)
- ``MeanShiftMonitor``         L2 shift of the standardized feature mean (floor)

Each exposes the same interface as the AE monitor — ``observe(x, index) ->
float | None`` and an ``error_history`` list of ``(index, score)`` — so they
drop into the runner and the offline lead-time analysis unchanged.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.stats import ks_2samp

from driftgan.augment.base import FeatureScaler


class ShiftMonitor:
    """Base monitor: buffering, cadence and the frozen-reference protocol.

    Subclasses implement three hooks operating on the standardized+clipped
    buffer ``Xn``: :meth:`_init_reference` (first check), :meth:`_score`
    (distance of the current buffer to the reference) and
    :meth:`_update_reference` (track the new normal, called *after* scoring).
    """

    def __init__(
        self,
        check_interval: int = 1_000,
        buffer_size: int = 2_000,
        clip: float = 10.0,
        min_buffer: int = 32,
        seed: int = 0,
    ):
        self.check_interval = check_interval
        self.clip = clip
        self.min_buffer = min_buffer
        self.rng = np.random.default_rng(seed)
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._keys: list[str] | None = None
        self._scaler: FeatureScaler | None = None
        self._ref_ready = False
        self._n_seen = 0
        self.error_history: list[tuple[int, float]] = []

    # -- standardization (frozen after the first check) --------------------
    def _std(self, X: np.ndarray) -> np.ndarray:
        assert self._scaler is not None
        return np.clip(self._scaler.transform(X), -self.clip, self.clip)

    # -- public API --------------------------------------------------------
    def observe(self, x: dict[str, float], index: int) -> float | None:
        if self._keys is None:
            self._keys = list(x.keys())
        self._buffer.append(np.array([x.get(k, 0.0) for k in self._keys], dtype=np.float32))
        self._n_seen += 1
        if self._n_seen % self.check_interval != 0 or len(self._buffer) < self.min_buffer:
            return None
        return self._check(index)

    def _check(self, index: int) -> float:
        X = np.stack(self._buffer)
        if not self._ref_ready:
            self._scaler = FeatureScaler.fit(X)
            Xn = self._std(X)
            self._init_reference(Xn)
            self._ref_ready = True
            score = self._score(Xn)  # self-distance -> baseline ~0
        else:
            Xn = self._std(X)
            score = self._score(Xn)  # measure before adapting (leading signal)
            self._update_reference(Xn)  # then track the new normal
        self.error_history.append((index, score))
        return score

    # -- hooks -------------------------------------------------------------
    def _init_reference(self, Xn: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def _score(self, Xn: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError

    def _update_reference(self, Xn: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError


class PCAReconstructionMonitor(ShiftMonitor):
    """Linear analogue of the AE: reconstruction error in a frozen PCA basis."""

    def __init__(self, latent: int = 16, **kw):
        super().__init__(**kw)
        self.latent = latent
        self._components: np.ndarray | None = None  # (k, d)

    def _fit(self, Xn: np.ndarray) -> None:
        # top-k right singular vectors of the centred buffer
        k = min(self.latent, min(Xn.shape) - 1)
        _, _, vt = np.linalg.svd(Xn - Xn.mean(0, keepdims=True), full_matrices=False)
        self._components = vt[:k]

    def _init_reference(self, Xn: np.ndarray) -> None:
        self._fit(Xn)

    def _score(self, Xn: np.ndarray) -> float:
        assert self._components is not None
        proj = Xn @ self._components.T @ self._components
        return float(np.mean((Xn - proj) ** 2))

    def _update_reference(self, Xn: np.ndarray) -> None:
        self._fit(Xn)


class KSMonitor(ShiftMonitor):
    """Mean per-feature two-sample Kolmogorov–Smirnov distance to the reference."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._ref: np.ndarray | None = None

    def _init_reference(self, Xn: np.ndarray) -> None:
        self._ref = Xn

    def _score(self, Xn: np.ndarray) -> float:
        assert self._ref is not None
        stats = [ks_2samp(self._ref[:, j], Xn[:, j]).statistic for j in range(Xn.shape[1])]
        return float(np.mean(stats))

    def _update_reference(self, Xn: np.ndarray) -> None:
        self._ref = Xn


class MMDMonitor(ShiftMonitor):
    """RBF maximum mean discrepancy (unbiased MMD^2) between reference and buffer.

    Both sides are subsampled to ``sample`` rows to keep the O(n^2) kernel cheap;
    the RBF bandwidth is fixed by the median heuristic on the first reference.
    """

    def __init__(self, sample: int = 256, **kw):
        super().__init__(**kw)
        self.sample = sample
        self._ref: np.ndarray | None = None
        self._gamma: float = 1.0

    def _sub(self, Xn: np.ndarray) -> np.ndarray:
        if len(Xn) <= self.sample:
            return Xn
        idx = self.rng.choice(len(Xn), self.sample, replace=False)
        return Xn[idx]

    @staticmethod
    def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aa = (a * a).sum(1)[:, None]
        bb = (b * b).sum(1)[None, :]
        return np.maximum(aa + bb - 2 * a @ b.T, 0.0)

    def _init_reference(self, Xn: np.ndarray) -> None:
        self._ref = self._sub(Xn)
        d = self._sq_dists(self._ref, self._ref)
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        self._gamma = 1.0 / (med + 1e-12)

    def _mmd2(self, a: np.ndarray, b: np.ndarray) -> float:
        kaa = np.exp(-self._gamma * self._sq_dists(a, a))
        kbb = np.exp(-self._gamma * self._sq_dists(b, b))
        kab = np.exp(-self._gamma * self._sq_dists(a, b))
        na, nb = len(a), len(b)
        # unbiased: drop the diagonal on the within-sample terms
        t1 = (kaa.sum() - np.trace(kaa)) / (na * (na - 1))
        t2 = (kbb.sum() - np.trace(kbb)) / (nb * (nb - 1))
        t3 = 2 * kab.mean()
        return float(max(t1 + t2 - t3, 0.0))

    def _score(self, Xn: np.ndarray) -> float:
        assert self._ref is not None
        return self._mmd2(self._ref, self._sub(Xn))

    def _update_reference(self, Xn: np.ndarray) -> None:
        self._ref = self._sub(Xn)


class MeanShiftMonitor(ShiftMonitor):
    """Cheapest floor: L2 distance between standardized feature means."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._ref_mean: np.ndarray | None = None

    def _init_reference(self, Xn: np.ndarray) -> None:
        self._ref_mean = Xn.mean(0)

    def _score(self, Xn: np.ndarray) -> float:
        assert self._ref_mean is not None
        return float(np.linalg.norm(Xn.mean(0) - self._ref_mean))

    def _update_reference(self, Xn: np.ndarray) -> None:
        self._ref_mean = Xn.mean(0)


MONITOR_TYPES = {
    "pca": PCAReconstructionMonitor,
    "ks": KSMonitor,
    "mmd": MMDMonitor,
    "mean": MeanShiftMonitor,
}
