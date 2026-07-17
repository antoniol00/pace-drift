"""Conditional GAN augmenter (Phase C_cond, Improvement 5).

Conditions generation on the original CICIDS2017 attack subtype so the
generator does not average structurally different attacks (DoS vs PortScan
vs Brute Force...). At drift time, synthetic samples are generated
*proportionally to the subtype mix observed in the recent buffer*, so the
augmentation reflects the attacks actually occurring.

Loss variants (ablations, see docs/gan_design.md):
- ``wgan_gp`` (default): Wasserstein + gradient penalty, spectral norm.
- ``bce``: BCE loss with optional one-sided label smoothing.
Feature matching can be enabled on top of either.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from driftgan.augment.base import Buffer, buffer_to_matrix, matrix_to_samples
from driftgan.augment.vanilla_gan import resolve_device
from driftgan.augment.wgan_gp import ConditionalCritic, ConditionalGenerator, gradient_penalty


class ConditionalGanAugmenter:
    """Subtype-conditional GAN trained at drift time on the recent buffer.

    The class vocabulary grows online: new subtypes seen in later buffers
    are added and the networks are rebuilt (buffers are tiny, rebuild cost
    is negligible compared to stream progress).
    """

    def __init__(
        self,
        noise_dim: int = 16,
        epochs: int = 30,
        lr: float = 1e-3,
        n_critic: int = 5,
        gp_lambda: float = 10.0,
        use_spectral_norm: bool = True,
        loss: str = "wgan_gp",
        label_smoothing: float = 0.0,
        feature_matching: bool = False,
        device: str = "cpu",
        seed: int = 0,
    ):
        if loss not in ("wgan_gp", "bce"):
            raise ValueError(f"unknown loss {loss!r}")
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.lr = lr
        self.n_critic = n_critic
        self.gp_lambda = gp_lambda
        self.use_spectral_norm = use_spectral_norm
        self.loss = loss
        self.label_smoothing = label_smoothing
        self.feature_matching = feature_matching
        self.device = resolve_device(device)
        self.seed = seed
        self.class_to_id: dict[str, int] = {}
        self._G: ConditionalGenerator | None = None
        self._D: ConditionalCritic | None = None
        self._keys: list[str] | None = None

    # ── vocabulary / network management ──────────────────────────────
    def _ensure_networks(self, keys: list[str], subtypes: list[str], input_dim: int) -> bool:
        new_classes = list(dict.fromkeys(s for s in subtypes if s not in self.class_to_id))
        rebuild = self._G is None or self._keys != keys or bool(new_classes)
        for s in new_classes:
            self.class_to_id[s] = len(self.class_to_id)
        if rebuild:
            torch.manual_seed(self.seed)
            self._G = ConditionalGenerator(
                self.noise_dim, max(len(self.class_to_id), 1), input_dim
            ).to(self.device)
            self._D = ConditionalCritic(
                input_dim, max(len(self.class_to_id), 1), use_spectral_norm=self.use_spectral_norm
            ).to(self.device)
            self._opt_g = optim.Adam(self._G.parameters(), lr=self.lr, betas=(0.5, 0.9))
            self._opt_d = optim.Adam(self._D.parameters(), lr=self.lr, betas=(0.5, 0.9))
            self._keys = keys
        return rebuild

    # ── training ─────────────────────────────────────────────────────
    def fit_generate(self, buffer: Buffer, n: int) -> list[dict[str, float]]:
        import time

        self.t_train = 0.0
        self.t_generate = 0.0
        if len(buffer) < 2:
            return []
        t0 = time.perf_counter()
        X, keys, subtypes = buffer_to_matrix(buffer)
        self._ensure_networks(keys, subtypes, X.shape[1])
        assert self._G is not None and self._D is not None

        real = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor([self.class_to_id[s] for s in subtypes], dtype=torch.long, device=self.device)
        batch = real.size(0)

        for _ in range(self.epochs):
            if self.loss == "wgan_gp":
                self._train_epoch_wgan(real, y, batch)
            else:
                self._train_epoch_bce(real, y, batch)

        self.t_train = time.perf_counter() - t0

        # ── generation proportional to observed subtype mix ──────────
        t1 = time.perf_counter()
        mix = Counter(subtypes)
        counts = _proportional_counts(mix, n)
        out: list[np.ndarray] = []
        with torch.no_grad():
            for subtype, count in counts.items():
                if count == 0:
                    continue
                z = torch.randn(count, self.noise_dim, device=self.device)
                yy = torch.full((count,), self.class_to_id[subtype], dtype=torch.long, device=self.device)
                out.append(self._G(z, yy).cpu().numpy())
        self.t_generate = time.perf_counter() - t1
        if not out:
            return []
        return matrix_to_samples(np.concatenate(out, axis=0), keys)

    def _train_epoch_wgan(self, real: torch.Tensor, y: torch.Tensor, batch: int) -> None:
        assert self._G is not None and self._D is not None
        for _ in range(self.n_critic):
            self._opt_d.zero_grad()
            z = torch.randn(batch, self.noise_dim, device=self.device)
            fake = self._G(z, y).detach()
            d_loss = self._D(fake, y).mean() - self._D(real, y).mean()
            d_loss = d_loss + self.gp_lambda * gradient_penalty(self._D, real, fake, y, self.device)
            d_loss.backward()
            self._opt_d.step()

        self._opt_g.zero_grad()
        z = torch.randn(batch, self.noise_dim, device=self.device)
        fake = self._G(z, y)
        g_loss = -self._D(fake, y).mean()
        if self.feature_matching:
            g_loss = g_loss + nn.functional.mse_loss(
                self._D.features(fake, y).mean(dim=0), self._D.features(real, y).mean(dim=0)
            )
        g_loss.backward()
        self._opt_g.step()

    def _train_epoch_bce(self, real: torch.Tensor, y: torch.Tensor, batch: int) -> None:
        assert self._G is not None and self._D is not None
        bce = nn.BCEWithLogitsLoss()
        real_target = torch.full((batch, 1), 1.0 - self.label_smoothing, device=self.device)
        fake_target = torch.zeros(batch, 1, device=self.device)

        self._opt_d.zero_grad()
        z = torch.randn(batch, self.noise_dim, device=self.device)
        fake = self._G(z, y).detach()
        d_loss = bce(self._D(real, y), real_target) + bce(self._D(fake, y), fake_target)
        d_loss.backward()
        self._opt_d.step()

        self._opt_g.zero_grad()
        z = torch.randn(batch, self.noise_dim, device=self.device)
        fake = self._G(z, y)
        g_loss = bce(self._D(fake, y), torch.ones(batch, 1, device=self.device))
        if self.feature_matching:
            g_loss = g_loss + nn.functional.mse_loss(
                self._D.features(fake, y).mean(dim=0), self._D.features(real, y).mean(dim=0)
            )
        g_loss.backward()
        self._opt_g.step()


def _proportional_counts(mix: Counter, n: int) -> dict[str, int]:
    """Largest-remainder allocation of ``n`` samples across the subtype mix."""
    total = sum(mix.values())
    raw = {k: n * v / total for k, v in mix.items()}
    counts = {k: int(v) for k, v in raw.items()}
    remainder = n - sum(counts.values())
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[:remainder]:
        counts[k] += 1
    return counts
