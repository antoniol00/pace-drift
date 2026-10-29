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

# Phases that self-adapt on every sample (continuous supervision). The "_sched"
# variants use the SAME ensemble but adapt only in triggered train windows
# (budget-matched control, Experiment #3b), so they are NOT continuous.
CONTINUOUS_PHASES = ("ARF", "SRP")


def make_model(config: ExperimentConfig, phase: str, seed: int) -> Any:
    """Return the classifier for ``phase`` (HAT unless an ensemble or ``LR`` is named).

    Dispatch is by prefix so budget-matched variants (``ARF_sched``, ``SRP_sched``)
    reuse the same ensemble as ``ARF`` / ``SRP``.
    """
    if phase.startswith("ARF"):
        from river import forest

        return forest.ARFClassifier(seed=seed)
    if phase.startswith("SRP"):
        from river import ensemble

        return ensemble.SRPClassifier(seed=seed)
    if phase.startswith("LR"):
        # A single linear model, to test whether collapse is specific to the tree.
        # Softmax regression over two classes is logistic regression, and it takes
        # the string labels; flow features span orders of magnitude, so standardise.
        from river import linear_model, preprocessing

        return preprocessing.StandardScaler() | linear_model.SoftmaxRegression()
    return make_hat(config, seed)
