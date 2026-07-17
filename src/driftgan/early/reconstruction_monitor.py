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

from driftgan.augment.vanilla_gan import resolve_device


class ReconstructionMonitor:
    """Incrementally trained autoencoder over the recent feature buffer.

    Every ``check_interval`` observed samples, the AE is fine-tuned for
    ``epochs`` epochs on the buffer and the mean reconstruction error over
    the buffer is returned (otherwise ``observe`` returns None). The error
    series is the input to ``PHT_recon``.
    """

    def __init__(
        self,
        check_interval: int = 5_000,
        buffer_size: int = 2_000,
        hidden: int = 32,
        latent: int = 16,
        epochs: int = 5,
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.check_interval = check_interval
        self.epochs = epochs
        self.lr = lr
        self.hidden = hidden
        self.latent = latent
        self.device = resolve_device(device)
        self.seed = seed
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._keys: list[str] | None = None
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
        # Per-feature standardization with buffer stats keeps the error
        # scale comparable across checks despite heavy-tailed features.
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd == 0] = 1.0
        Xn = (X - mu) / sd
        t = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        if self._model is None:
            self._build(t.shape[1])
        assert self._model is not None and self._opt is not None

        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            self._opt.zero_grad()
            loss = loss_fn(self._model(t), t)
            loss.backward()
            self._opt.step()
        with torch.no_grad():
            err = float(loss_fn(self._model(t), t).item())
        self.error_history.append((index, err))
        return err
