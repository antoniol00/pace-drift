"""Computational-cost instrumentation: per-sample latency, per-component
timing breakdown, and RSS memory sampling (Improvement 7)."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import psutil


class CostTracker:
    """Tracks where the pipeline spends its time and memory.

    Usage
    -----
    >>> tracker = CostTracker()
    >>> with tracker.sample():            # one stream sample
    ...     with tracker.section("hat_inference"):
    ...         ...
    ...     with tracker.section("hat_update"):
    ...         ...
    >>> tracker.summary()["latency_p95_ms"]

    Sections nest inside a sample; time inside a sample not covered by any
    section is reported as ``other``. RSS is sampled every
    ``rss_sample_interval`` samples via psutil.
    """

    SECTIONS = ("hat_inference", "hat_update", "gan_training", "gan_generation", "monitor")

    def __init__(self, rss_sample_interval: int = 5_000):
        self._latencies_ms: list[float] = []
        self._section_totals: dict[str, float] = defaultdict(float)
        self._sample_t0 = 0.0
        self._sample_sections = 0.0
        self._rss_interval = rss_sample_interval
        self._rss_samples_mb: list[float] = []
        self._n_samples = 0
        self._process = psutil.Process()
        self.last_latency_ms = 0.0

    @contextmanager
    def sample(self) -> Iterator[None]:
        """Context spanning the processing of one stream sample."""
        self._sample_t0 = time.perf_counter()
        self._sample_sections = 0.0
        try:
            yield
        finally:
            elapsed = time.perf_counter() - self._sample_t0
            self.last_latency_ms = elapsed * 1000.0
            self._latencies_ms.append(self.last_latency_ms)
            self._section_totals["other"] += max(0.0, elapsed - self._sample_sections)
            self._n_samples += 1
            if self._n_samples % self._rss_interval == 0:
                self._rss_samples_mb.append(self._process.memory_info().rss / 1e6)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Attribute the enclosed time to component ``name``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._section_totals[name] += dt
            self._sample_sections += dt

    def add(self, name: str, seconds: float) -> None:
        """Credit externally measured time to component ``name``.

        Used when a component measures its own internal phases (e.g. an
        augmenter separating training from generation); the time still
        counts as covered within the current sample, keeping ``other``
        consistent.
        """
        self._section_totals[name] += seconds
        self._sample_sections += seconds

    @property
    def latencies_ms(self) -> np.ndarray:
        return np.asarray(self._latencies_ms)

    def summary(self) -> dict[str, Any]:
        """Percentiles, per-component totals and memory statistics."""
        lat = self.latencies_ms
        # Fall back to a point RSS reading when no periodic sample was taken
        # (short runs), avoiding NaN summaries and numpy warnings.
        rss_list = self._rss_samples_mb or [self._process.memory_info().rss / 1e6]
        rss = np.asarray(rss_list)
        out: dict[str, Any] = {
            "n_samples": int(self._n_samples),
            "latency_p50_ms": float(np.percentile(lat, 50)) if len(lat) else float("nan"),
            "latency_p95_ms": float(np.percentile(lat, 95)) if len(lat) else float("nan"),
            "latency_p99_ms": float(np.percentile(lat, 99)) if len(lat) else float("nan"),
            "latency_max_ms": float(lat.max()) if len(lat) else float("nan"),
            "peak_rss_mb": float(np.nanmax(rss)),
            "mean_rss_mb": float(np.nanmean(rss)),
        }
        for name, total in self._section_totals.items():
            out[f"time_{name}_s"] = float(total)
        return out

    def save(self, path: str | Path) -> None:
        import json

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)
