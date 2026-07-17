"""Deterministic seeding of every randomness source (Improvement 4)."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and all CUDA devices).

    river components take their seed explicitly at construction (see
    ``driftgan.models.hat``); numpy legacy global RNG is seeded for any
    library code that still uses it, while our own code uses
    ``np.random.default_rng(seed)`` instances.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 — intentional: legacy global RNG for third-party code
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
