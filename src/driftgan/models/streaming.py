"""Model factory: the shared HAT plus SOTA self-adapting streaming ensembles.

Experiment #3 compares the drift-adaptation strategies against established
streaming learners that manage drift internally (no external trigger): the
Adaptive Random Forest (ARF) and Streaming Random Patches (SRP), both standard
strong baselines for concept-drift classification. They are run in continuous-
learning mode by the runner (``learn_one`` on every sample).
"""

from __future__ import annotations

from typing import Any

from driftgan.config import ExperimentConfig
from driftgan.models.hat import make_hat

CONTINUOUS_PHASES = ("ARF", "SRP")


def make_model(config: ExperimentConfig, phase: str, seed: int) -> Any:
    """Return the classifier for ``phase`` (HAT unless a SOTA ensemble is named)."""
    if phase == "ARF":
        from river import forest

        return forest.ARFClassifier(seed=seed)
    if phase == "SRP":
        from river import ensemble

        return ensemble.SRPClassifier(seed=seed)
    return make_hat(config, seed)
