"""Shared drift schedules so that all phases adapt at identical stream indices.

Rationale (Improvement 1): with per-phase dynamic detection, the moment a
phase modifies its model its error signal diverges and PHT fires at
different indices in each phase — the phases then respond to *different*
events and are not comparable. A shared schedule fixes the adaptation
indices so every phase reacts to the same events.

Two schedule sources are provided:

- ``build_schedule_from_pht``: a "pass 0" runs Phase B dynamics (HAT + PHT +
  retrain windows) once and records where PHT fired. Later runs of any phase
  replay those indices. This is the primary schedule used in reported
  experiments.
- ``build_schedule_from_labels``: a ground-truth schedule derived from the
  attack-subtype transitions in the CSV labels: an event is recorded at the
  first sample of each new non-benign subtype block (subject to a minimum
  gap between events). This is independent of any model and serves as a
  complementary reference.

When a fixed schedule is in use, a PHT detector still runs *in parallel* in
report-only mode: its fire indices are recorded (to measure detection delay
vs the schedule) but it never triggers retraining.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from river import drift as river_drift

from driftgan.data.stream import BENIGN


@dataclass
class DriftSchedule:
    """An ordered list of stream indices (relative to the evaluated stream,
    i.e. counting from the first sample *after* start_row, pre-training
    included) at which adaptation must trigger."""

    indices: list[int]
    source: str  # "pht" | "labels"
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        payload = {"indices": self.indices, "source": self.source, "metadata": self.metadata}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def load_schedule(path: str | Path) -> DriftSchedule:
    """Load a schedule previously saved with :meth:`DriftSchedule.save`."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return DriftSchedule(
        indices=list(payload["indices"]),
        source=str(payload["source"]),
        metadata=dict(payload.get("metadata", {})),
    )


def build_schedule_from_pht(config: Any, seed: int = 0) -> DriftSchedule:
    """Run a Phase B pass ("pass 0") and record the indices where PHT fired.

    Parameters
    ----------
    config : ExperimentConfig
        The experiment configuration. ``config.drift_schedule`` is ignored
        here (this pass is always dynamic).
    seed : int
        Seed for the HAT model of the pass.
    """
    # Imported lazily to avoid a circular import (runner uses this module).
    from driftgan.experiments.runner import run_phase

    result = run_phase(config, phase="B", seed=seed, force_dynamic=True, write_outputs=False)
    return DriftSchedule(
        indices=list(result.drift_indices),
        source="pht",
        metadata={
            "config_hash": config.config_hash(),
            "seed": seed,
            "pht_delta": config.pht_delta,
            "n_pretrain": config.n_pretrain,
        },
    )


def build_schedule_from_labels(
    labels: list[str] | Any,
    min_gap: int = 5_000,
    n_pretrain: int = 0,
    benign_label: str = BENIGN,
) -> DriftSchedule:
    """Derive a ground-truth schedule from raw subtype labels of the stream.

    An event is recorded at the index of the first sample of each new
    non-benign subtype block: i.e. whenever a non-benign label appears whose
    subtype differs from the last non-benign subtype seen. Consecutive
    events closer than ``min_gap`` samples are merged (only the first is
    kept), so short interleavings of benign traffic inside one attack burst
    do not produce spurious events.

    Parameters
    ----------
    labels : sequence of str
        Raw subtype labels (``y_raw``) in stream order, aligned with the
        evaluated stream (index 0 = first sample after start_row).
    min_gap : int
        Minimum distance in samples between two recorded events.
    n_pretrain : int
        Events inside the pre-training prefix are skipped (the model is
        openly trained there; there is nothing to adapt to).
    benign_label : str
        The label value representing benign/normal traffic (default: "BENIGN").
    """
    indices: list[int] = []
    last_subtype: str | None = None
    last_event = -min_gap
    for i, y_raw in enumerate(labels):
        if y_raw == benign_label:
            continue
        if y_raw != last_subtype:
            last_subtype = y_raw
            if i >= n_pretrain and i - last_event >= min_gap:
                indices.append(i)
                last_event = i
    return DriftSchedule(
        indices=indices,
        source="labels",
        metadata={"min_gap": min_gap, "n_pretrain": n_pretrain, "benign_label": benign_label},
    )


class DynamicPHTTrigger:
    """Adaptation trigger driven by a live Page-Hinkley detector."""

    def __init__(self, delta: float):
        self._pht = river_drift.PageHinkley(delta=delta)
        self.fired_indices: list[int] = []

    def update(self, index: int, error: int) -> bool:
        """Feed the 0/1 error at ``index``; return True if adaptation must trigger."""
        self._pht.update(error)
        if self._pht.drift_detected:
            self.fired_indices.append(index)
            return True
        return False


class DynamicDetectorTrigger:
    """Adaptation trigger driven by any river drift detector (ADWIN, KSWIN, …).

    Generalises :class:`DynamicPHTTrigger` to the alternative detectors used as
    SOTA baselines (Experiment #3): the detector is fed the 0/1 classifier error
    and fires adaptation whenever it flags drift.
    """

    def __init__(self, detector: Any, name: str = "detector"):
        self._detector = detector
        self.name = name
        self.fired_indices: list[int] = []

    def update(self, index: int, error: int) -> bool:
        self._detector.update(error)
        if self._detector.drift_detected:
            self.fired_indices.append(index)
            return True
        return False


class PeriodicTrigger:
    """Adaptation trigger firing on a blind fixed-period clock.

    Used by the frequency-matched control (Phase C_freq): it fires every
    ``interval`` samples after ``start_after`` (the pre-training prefix),
    regardless of any error or drift signal. With ``interval`` chosen so the
    total fire count matches C_early's mean, comparing C_freq against C_early
    isolates adaptation *frequency* from adaptation *timing* — the standard
    control for the "it only helps because it retrains more often" objection.
    """

    def __init__(self, interval: int, start_after: int = 0):
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.interval = interval
        self.start_after = start_after
        self.fired_indices: list[int] = []

    def update(self, index: int, error: int) -> bool:
        """Fire when ``index`` lands on the periodic clock (``error`` ignored)."""
        if index > self.start_after and (index - self.start_after) % self.interval == 0:
            self.fired_indices.append(index)
            return True
        return False


class ScheduledDriftTrigger:
    """Adaptation trigger replaying a fixed :class:`DriftSchedule`.

    A parallel PHT runs in report-only mode: its fires are recorded in
    ``report_pht_indices`` for delay analysis but never trigger adaptation.
    """

    def __init__(self, schedule: DriftSchedule, report_pht_delta: float | None = None):
        self._pending = sorted(schedule.indices)
        self._pos = 0
        self.fired_indices: list[int] = []
        self.report_pht_indices: list[int] = []
        self._report_pht = (
            river_drift.PageHinkley(delta=report_pht_delta) if report_pht_delta is not None else None
        )

    def update(self, index: int, error: int) -> bool:
        """Return True exactly when ``index`` is in the schedule."""
        if self._report_pht is not None:
            self._report_pht.update(error)
            if self._report_pht.drift_detected:
                self.report_pht_indices.append(index)

        fired = False
        while self._pos < len(self._pending) and self._pending[self._pos] <= index:
            if self._pending[self._pos] == index:
                fired = True
            self._pos += 1
        if fired:
            self.fired_indices.append(index)
        return fired
