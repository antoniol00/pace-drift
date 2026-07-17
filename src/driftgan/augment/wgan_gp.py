"""WGAN-GP building blocks: conditional generator/critic and gradient penalty.

Design notes (see docs/gan_design.md): drift-time buffers are tiny (5-500
samples), where BCE GANs are prone to mode collapse and discriminator
overconfidence. Wasserstein loss with gradient penalty gives non-saturating
gradients, and spectral normalization further stabilizes the critic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as sn


def _maybe_sn(layer: nn.Module, use_sn: bool) -> nn.Module:
    return sn(layer) if use_sn else layer  # type: ignore[arg-type]


class ConditionalGenerator(nn.Module):
    """G(z, y) -> x with an embedded subtype condition."""

    def __init__(self, noise_dim: int, n_classes: int, out_dim: int, hidden: int = 64, embed_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, self.embed(y)], dim=1))


class ConditionalCritic(nn.Module):
    """D(x, y) -> score (no sigmoid: Wasserstein critic).

    ``features()`` exposes the penultimate activations for feature matching.
    """

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 64, embed_dim: int = 8, use_spectral_norm: bool = True):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.body = nn.Sequential(
            _maybe_sn(nn.Linear(in_dim + embed_dim, hidden), use_spectral_norm),
            nn.LeakyReLU(0.2),
            _maybe_sn(nn.Linear(hidden, hidden), use_spectral_norm),
            nn.LeakyReLU(0.2),
        )
        self.head = _maybe_sn(nn.Linear(hidden, 1), use_spectral_norm)

    def features(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.body(torch.cat([x, self.embed(y)], dim=1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x, y))


def gradient_penalty(
    critic: ConditionalCritic,
    real: torch.Tensor,
    fake: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Standard WGAN-GP penalty on interpolates between real and fake."""
    alpha = torch.rand(real.size(0), 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interp, y)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interp,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()
