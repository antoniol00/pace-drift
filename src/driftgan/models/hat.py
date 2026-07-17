"""Hoeffding Adaptive Tree factory shared by all phases."""

from __future__ import annotations

from river import tree

from driftgan.config import ExperimentConfig


def make_hat(config: ExperimentConfig, seed: int) -> tree.HoeffdingAdaptiveTreeClassifier:
    """Build the HAT classifier used by every phase.

    A single factory keeps hyperparameters identical across phases so that
    differences in results are attributable to the adaptation strategy only.
    """
    return tree.HoeffdingAdaptiveTreeClassifier(grace_period=config.grace_period, seed=seed)
