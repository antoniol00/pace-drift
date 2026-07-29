"""Generative background monitor for early (pre-error) drift signals.

Hypothesis (Improvement 6, experimental): a sustained increase in the
reconstruction error of a small autoencoder trained on recent inputs is a
*leading* signal of covariate shift — the input distribution changes before
the classifier's error rises enough to fire PHT. If true, adapting on this
signal buys lead time over error-based detection.

This component is deliberately reported honestly: it may not work. The
runner records both detectors' fire indices so the lead-time analysis in
``driftgan.early.dual_pht`` can quantify anticipation, anticipated false
positives, and missed anticipations.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from driftgan.early.scaling import FeatureScaler, resolve_device


class ReconstructionMonitor:
    """Autoencoder over the recent feature buffer, as a leading drift signal.

    The coordinate system is fixed on the **first** check (a frozen reference
    :class:`FeatureScaler`): standardizing every later buffer against its own
    stats — as an earlier version did — cancels exactly the covariate shift
    the monitor is meant to detect. With a frozen scaler, a distribution
    change moves inputs away from the reference and the reconstruction error
    rises.

    Crucially, the error at each check is measured **before** the AE is
    fine-tuned on that buffer, so a shift shows up as a spike before the model
    absorbs it (a *leading* signal). After measuring, the AE is fine-tuned so
    it tracks the new normal for subsequent drifts. Standardized inputs are
    clipped to ``clip`` to stop a single heavy-tailed feature from dominating
    the error. The error series is the input to ``PHT_recon``.
    """

    def __init__(
        self,
        check_interval: int = 1_000,
        buffer_size: int = 2_000,
        hidden: int = 32,
        latent: int = 16,
        epochs: int = 5,
        lr: float = 1e-3,
        clip: float = 10.0,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.check_interval = check_interval
        self.epochs = epochs
        self.lr = lr
        self.hidden = hidden
        self.latent = latent
        self.clip = clip
        self.device = resolve_device(device)
        self.seed = seed
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._keys: list[str] | None = None
        self._scaler: FeatureScaler | None = None
        self._model: nn.Module | None = None
        self._opt: optim.Optimizer | None = None
        self._n_seen = 0
        self.error_history: list[tuple[int, float]] = []  # (stream index, mean recon error)

    def _build(self, input_dim: int) -> None:
        torch.manual_seed(self.seed)
        self._model = nn.Sequential(
            nn.Linear(input_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.latent),
            nn.ReLU(),
            nn.Linear(self.latent, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, input_dim),
        ).to(self.device)
        self._opt = optim.Adam(self._model.parameters(), lr=self.lr)

    def _tensor(self, X: np.ndarray) -> torch.Tensor:
        assert self._scaler is not None
        Xn = np.clip(self._scaler.transform(X), -self.clip, self.clip)
        return torch.tensor(Xn, dtype=torch.float32, device=self.device)

    def _train(self, t: torch.Tensor) -> None:
        assert self._model is not None and self._opt is not None
        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            self._opt.zero_grad()
            loss = loss_fn(self._model(t), t)
            loss.backward()
            self._opt.step()

    def _recon_error(self, t: torch.Tensor) -> float:
        assert self._model is not None
        with torch.no_grad():
            return float(nn.MSELoss()(self._model(t), t).item())

    def observe(self, x: dict[str, float], index: int) -> float | None:
        """Feed one stream sample; returns the mean reconstruction error at
        check points, None otherwise."""
        if self._keys is None:
            self._keys = list(x.keys())
        self._buffer.append(np.array([x.get(k, 0.0) for k in self._keys], dtype=np.float32))
        self._n_seen += 1
        if self._n_seen % self.check_interval != 0 or len(self._buffer) < 32:
            return None
        return self._check(index)

    def _check(self, index: int) -> float:
        X = np.stack(self._buffer)
        if self._scaler is None:
            # First check defines the frozen reference frame and baseline.
            self._scaler = FeatureScaler.fit(X)
            t = self._tensor(X)
            self._build(t.shape[1])
            self._train(t)
            err = self._recon_error(t)
        else:
            t = self._tensor(X)  # frozen scaler -> shift is visible
            err = self._recon_error(t)  # measure before adapting (leading signal)
            self._train(t)  # then track the new normal
        self.error_history.append((index, err))
        return err
