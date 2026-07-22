"""Tests for the model factory (HAT vs SOTA ensembles, budget-matched variants)."""

from pathlib import Path

from driftgan.config import load_config
from driftgan.models.streaming import CONTINUOUS_PHASES, make_model


def _cfg():
    return load_config(Path("configs/phase_arf_sched.yaml"))


def test_make_model_dispatch_by_prefix():
    cfg = _cfg()
    for ph in ("ARF", "ARF_sched", "ARF_early"):
        assert type(make_model(cfg, ph, 0)).__name__ == "ARFClassifier"
    for ph in ("SRP", "SRP_sched", "SRP_early"):
        assert type(make_model(cfg, ph, 0)).__name__ == "SRPClassifier"
    assert type(make_model(cfg, "B", 0)).__name__ == "HoeffdingAdaptiveTreeClassifier"


def test_triggered_ensemble_variants_are_not_continuous():
    # only the bare ARF/SRP self-adapt every sample; _sched (reactive) and _early
    # (proactive) variants adapt only in triggered windows
    assert "ARF" in CONTINUOUS_PHASES and "SRP" in CONTINUOUS_PHASES
    for ph in ("ARF_sched", "SRP_sched", "ARF_early", "SRP_early"):
        assert ph not in CONTINUOUS_PHASES
