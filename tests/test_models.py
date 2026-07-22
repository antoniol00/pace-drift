"""Tests for the model factory (HAT vs SOTA ensembles, budget-matched variants)."""

from pathlib import Path

from driftgan.config import load_config
from driftgan.models.streaming import CONTINUOUS_PHASES, make_model


def _cfg():
    return load_config(Path("configs/phase_arf_sched.yaml"))


def test_make_model_dispatch_by_prefix():
    cfg = _cfg()
    assert type(make_model(cfg, "ARF", 0)).__name__ == "ARFClassifier"
    assert type(make_model(cfg, "ARF_sched", 0)).__name__ == "ARFClassifier"
    assert type(make_model(cfg, "SRP", 0)).__name__ == "SRPClassifier"
    assert type(make_model(cfg, "SRP_sched", 0)).__name__ == "SRPClassifier"
    assert type(make_model(cfg, "B", 0)).__name__ == "HoeffdingAdaptiveTreeClassifier"


def test_budget_matched_variants_are_not_continuous():
    # the _sched variants must adapt only in triggered windows, not every sample
    assert "ARF" in CONTINUOUS_PHASES and "SRP" in CONTINUOUS_PHASES
    assert "ARF_sched" not in CONTINUOUS_PHASES
    assert "SRP_sched" not in CONTINUOUS_PHASES
