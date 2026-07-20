"""Dual Page-Hinkley detection: classification error vs reconstruction error,
plus lead-time pairing analysis (Improvement 6)."""

from __future__ import annotations

from dataclasses import dataclass

from river import drift as river_drift


class DualPHT:
    """Two independent PHT detectors over different signals.

    - ``update_class(index, error)``: 0/1 classification error, every sample.
    - ``update_recon(index, error)``: mean reconstruction error, at monitor
      check points only.

    Fire indices of both are recorded for post-hoc lead-time analysis.
    """

    def __init__(
        self,
        class_delta: float,
        recon_delta: float,
        recon_threshold: float = 2.0,
        recon_min_instances: int = 10,
    ):
        self._pht_class = river_drift.PageHinkley(delta=class_delta)
        # The recon signal is sampled sparsely (once per monitor check) and is
        # O(1)-scaled, not 0/1, so it needs its own threshold and a shorter
        # warmup than the per-sample class-error detector.
        self._pht_recon = river_drift.PageHinkley(
            delta=recon_delta, threshold=recon_threshold, min_instances=recon_min_instances
        )
        self.class_fires: list[int] = []
        self.recon_fires: list[int] = []

    def update_class(self, index: int, error: int) -> bool:
        self._pht_class.update(error)
        if self._pht_class.drift_detected:
            self.class_fires.append(index)
            return True
        return False

    def update_recon(self, index: int, error: float) -> bool:
        self._pht_recon.update(error)
        if self._pht_recon.drift_detected:
            self.recon_fires.append(index)
            return True
        return False


@dataclass
class LeadTimeReport:
    """Pairing of recon fires with subsequent class fires within a horizon."""

    pairs: list[tuple[int, int, int]]  # (recon_index, class_index, lead)
    lead_times: list[int]
    anticipated_fp_rate: float  # recon fires not followed by a class fire within horizon
    missed_anticipation_rate: float  # class fires not preceded by a recon fire within horizon
    n_recon_fires: int
    n_class_fires: int

    @property
    def mean_lead_time(self) -> float:
        return sum(self.lead_times) / len(self.lead_times) if self.lead_times else float("nan")


def match_lead_times(
    recon_fires: list[int],
    class_fires: list[int],
    horizon: int,
) -> LeadTimeReport:
    """Greedily pair each class fire with the nearest preceding unpaired
    recon fire within ``horizon`` samples.

    Lead time is ``class_index - recon_index`` (>= 0; a recon fire at the
    same index as the class fire counts as zero anticipation). Each recon
    fire anticipates at most one class fire.
    """
    recon_sorted = sorted(recon_fires)
    used = [False] * len(recon_sorted)
    pairs: list[tuple[int, int, int]] = []
    for c in sorted(class_fires):
        best: int | None = None
        for i, r in enumerate(recon_sorted):
            if used[i] or r > c:
                continue
            if c - r <= horizon:
                best = i  # keep scanning: later (closer) fires overwrite
        if best is not None:
            used[best] = True
            pairs.append((recon_sorted[best], c, c - recon_sorted[best]))

    n_recon = len(recon_sorted)
    n_class = len(class_fires)
    n_paired = len(pairs)
    return LeadTimeReport(
        pairs=pairs,
        lead_times=[p[2] for p in pairs],
        anticipated_fp_rate=(n_recon - n_paired) / n_recon if n_recon else float("nan"),
        missed_anticipation_rate=(n_class - n_paired) / n_class if n_class else float("nan"),
        n_recon_fires=n_recon,
        n_class_fires=n_class,
    )
