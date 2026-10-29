"""Trigger-aware adversary (oracle and chaff) and the window-rebalancing defence."""

import pytest

from driftgan.data.stream import ATTACK, BENIGN, CHAFF, inject_chaff
from driftgan.models.rebalance import WindowRebalancer


def _filled(prior: float) -> WindowRebalancer:
    rb = WindowRebalancer(prior, size=50, seed=0)
    for i in range(200):
        rb.remember({"f": float(i)}, BENIGN if i % 5 else ATTACK)
    return rb


def test_rebalancer_pins_an_all_attack_window():
    rb = _filled(0.2)
    rb.start_window()
    for _ in range(100):
        replays = rb.top_up(ATTACK)
        assert all(y == BENIGN for _, y in replays)
        total = rb.n_attack + rb.n_benign
        assert rb.prior - 1 / total < rb.n_attack / total <= rb.prior
    assert rb.n_attack == 100  # only the real attacks; every replay was benign


def test_rebalancer_tops_up_a_benign_window_with_attacks():
    rb = _filled(0.25)
    rb.start_window()
    for _ in range(40):
        rb.top_up(BENIGN)
    assert rb.n_attack / (rb.n_attack + rb.n_benign) == pytest.approx(0.25, abs=1 / 40)


def test_rebalancer_cannot_replay_from_an_empty_reservoir():
    rb = WindowRebalancer(0.2, size=10)
    rb.start_window()
    assert rb.top_up(ATTACK) == []


def test_reservoir_is_bounded():
    rb = _filled(0.2)
    assert all(len(r) <= 50 for r in rb._reservoir.values())


def test_chaff_is_inserted_before_each_target_with_pool_values():
    stream = [({"a": float(i), "b": float(-i)}, BENIGN, "BENIGN") for i in range(1, 21)]
    out = list(inject_chaff(stream, before=[10, 15], size=3, pool_size=5, seed=0))
    assert len(out) == 26
    real = [x["a"] for x, _, raw in out if raw != CHAFF]
    assert real == [float(i) for i in range(1, 21)]
    chaff_pos = [i for i, (_, _, raw) in enumerate(out) if raw == CHAFF]
    assert chaff_pos == [9, 10, 11, 17, 18, 19]
    for x, y, _ in (out[i] for i in chaff_pos):
        assert y == BENIGN
        assert x["a"] in {1.0, 2.0, 3.0, 4.0, 5.0} and -x["b"] in {1.0, 2.0, 3.0, 4.0, 5.0}
