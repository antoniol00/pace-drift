"""Vanilla binary GAN augmenter (Phase C) — the original DriftGAN, made
seedable and device-aware."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from driftgan.augment.base import Buffer, buffer_to_matrix, matrix_to_samples


def resolve_device(requested: str) -> torch.device:
    """Return the requested device, silently falling back to CPU."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


class VanillaGanAugmenter:
    """Lightweight GAN trained on the recent attack buffer at each drift.

    The generator/discriminator are intentionally tiny so training at drift
    time does not stall the stream. The network is created once (first call)
    and fine-tuned on subsequent drifts, mirroring the original PoC.
    """

    def __init__(
        self,
        noise_dim: int = 16,
        epochs: int = 30,
        lr: float = 0.005,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.lr = lr
        self.device = resolve_device(device)
        self.seed = seed
        self.generator: nn.Module | None = None
        self.discriminator: nn.Module | None = None
        self._keys: list[str] | None = None

    def _build(self, input_dim: int) -> None:
        torch.manual_seed(self.seed)
        self.generator = nn.Sequential(
            nn.Linear(self.noise_dim, 32), nn.ReLU(), nn.Linear(32, input_dim)
        ).to(self.device)
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        ).to(self.device)
        self._opt_g = optim.Adam(self.generator.parameters(), lr=self.lr)
        self._opt_d = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self._criterion = nn.BCELoss()

    def fit_generate(self, buffer: Buffer, n: int) -> list[dict[str, float]]:
        import time

        self.t_train = 0.0
        self.t_generate = 0.0
        if len(buffer) < 2:
            return []
        t0 = time.perf_counter()
        X, keys, _ = buffer_to_matrix(buffer)
        if self.generator is None or self._keys != keys:
            self._build(input_dim=X.shape[1])
            self._keys = keys
        assert self.generator is not None and self.discriminator is not None

        real = torch.tensor(X, dtype=torch.float32, device=self.device)
        batch = real.size(0)
        ones = torch.ones(batch, 1, device=self.device)
        zeros = torch.zeros(batch, 1, device=self.device)

        for _ in range(self.epochs):
            self._opt_d.zero_grad()
            d_loss_real = self._criterion(self.discriminator(real), ones)
            z = torch.randn(batch, self.noise_dim, device=self.device)
            fake = self.generator(z)
            d_loss_fake = self._criterion(self.discriminator(fake.detach()), zeros)
            (d_loss_real + d_loss_fake).backward()
            self._opt_d.step()

            self._opt_g.zero_grad()
            z = torch.randn(batch, self.noise_dim, device=self.device)
            fake = self.generator(z)
            g_loss = self._criterion(self.discriminator(fake), ones)
            g_loss.backward()
            self._opt_g.step()

        self.t_train = time.perf_counter() - t0

        t1 = time.perf_counter()
        with torch.no_grad():
            z = torch.randn(n, self.noise_dim, device=self.device)
            synthetic = self.generator(z).cpu().numpy()
        samples = matrix_to_samples(synthetic, keys)
        self.t_generate = time.perf_counter() - t1
        return samples
