"""Incremental rolling binary-classification statistics over a sliding window."""

from __future__ import annotations

from collections import deque


class RollingBinaryStats:
    """O(1) rolling precision/recall/F1 over the last ``window_size`` predictions.

    Counters are maintained incrementally on append/evict instead of
    recomputing over the window on every read.

    Also tracks whether the current window contains any positive *true*
    label: when it does not, F1 is undefined for that window and downstream
    recovery metrics must exclude it (see ``driftgan.metrics.recovery``).
    """

    def __init__(self, window_size: int = 5000, pos_label: str = "ATTACK", min_samples: int = 100):
        self.window: deque[tuple[bool, bool]] = deque()
        self.window_size = window_size
        self.pos_label = pos_label
        self.min_samples = min_samples
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._n_pos_true = 0

    def update(self, y_true: str, y_pred: str) -> None:
        t = y_true == self.pos_label
        p = y_pred == self.pos_label
        if len(self.window) >= self.window_size:
            ot, op = self.window.popleft()
            if op and ot:
                self._tp -= 1
            elif op and not ot:
                self._fp -= 1
            elif not op and ot:
                self._fn -= 1
            if ot:
                self._n_pos_true -= 1
        self.window.append((t, p))
        if p and t:
            self._tp += 1
        elif p and not t:
            self._fp += 1
        elif not p and t:
            self._fn += 1
        if t:
            self._n_pos_true += 1

    @property
    def has_positive(self) -> bool:
        """True if the current window contains at least one positive true label."""
        return self._n_pos_true > 0

    def f1(self) -> float:
        """Binary F1 over the window; 0.0 while the window is warming up."""
        if len(self.window) < self.min_samples:
            return 0.0
        denom_p = self._tp + self._fp
        denom_r = self._tp + self._fn
        precision = self._tp / denom_p if denom_p > 0 else 0.0
        recall = self._tp / denom_r if denom_r > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
