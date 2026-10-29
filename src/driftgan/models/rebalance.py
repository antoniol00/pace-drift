"""Window rebalancing: decouple what the learner is fed from when the trigger fires.

A triggered window inherits the class ratio of the traffic it covers, so a
trigger that fires on an attack burst, by accident or because an adversary
steers it, skews the learner's prior toward attack; that is the closed-loop
collapse mechanism. The defence pins the attack share of what the learner is fed
inside each window to a fixed prior by replaying stored labelled samples of the
under-represented class.
"""

from __future__ import annotations

import random

from driftgan.data.stream import ATTACK, BENIGN

Labelled = tuple[dict[str, float], str]


class WindowRebalancer:
    """Keep the attack share of each window's training samples at ``prior``.

    Parameters
    ----------
    prior : float
        Target attack share, e.g. that of the pretraining labels, which the
        defender knows without looking ahead.
    size : int
        Capacity of each per-class reservoir of labelled samples.
    seed : int
        Seeds reservoir sampling and replay draws.
    """

    def __init__(self, prior: float, size: int = 5_000, seed: int = 0):
        if not 0.0 < prior < 1.0:
            raise ValueError(f"prior must lie in (0, 1), got {prior}")
        self.prior = prior
        self.size = size
        self._rng = random.Random(seed)
        self._reservoir: dict[str, list[dict[str, float]]] = {ATTACK: [], BENIGN: []}
        self._seen = {ATTACK: 0, BENIGN: 0}
        self.n_attack = 0
        self.n_benign = 0

    def remember(self, x: dict[str, float], y: str) -> None:
        """Offer a labelled sample to its class reservoir (Algorithm R)."""
        self._seen[y] += 1
        res = self._reservoir[y]
        if len(res) < self.size:
            res.append(x)
        else:
            j = self._rng.randrange(self._seen[y])
            if j < self.size:
                res[j] = x

    def start_window(self) -> None:
        """Reset the window's class counts; called on every firing."""
        self.n_attack = self.n_benign = 0

    def top_up(self, y: str) -> list[Labelled]:
        """Count one learned sample of class ``y`` and return the replays to learn.

        After the replays the window's attack share lies in
        ``(prior - 1/n, prior]``, unless the reservoir that would correct it is
        still empty.
        """
        if y == ATTACK:
            self.n_attack += 1
        else:
            self.n_benign += 1
        out: list[Labelled] = []
        benign, attack = self._reservoir[BENIGN], self._reservoir[ATTACK]
        while benign and self.n_attack > self.prior * (self.n_attack + self.n_benign):
            out.append((benign[self._rng.randrange(len(benign))], BENIGN))
            self.n_benign += 1
        while attack and self.n_attack + 1 <= self.prior * (self.n_attack + self.n_benign + 1):
            out.append((attack[self._rng.randrange(len(attack))], ATTACK))
            self.n_attack += 1
        return out
