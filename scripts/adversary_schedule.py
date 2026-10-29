"""Firing schedules for the oracle trigger-aware adversary on the anchor.

The adversary of the threat model knows the policy and controls when its own
attack traffic arrives, so it can fire the monitor branch at moments of its
choosing. The oracle gets the monitor's own budget and constraints:

- as many firings as PACE's monitor makes per run (mean over the stored
  C_early seeds);
- only at monitor check points (``n_pretrain + k * monitor_interval``), no
  earlier than the Page-Hinkley warm-up allows, and at least
  ``recon_pht_min_instances`` checks apart, as the real test.

Two goals, each a greedy pick of the best-scoring admissible windows (ties
broken by a fixed random permutation):

- ``attack``: open the windows with the highest attack share, which the
  stability analysis found behind collapse (alert flood). A window below
  ``MIN_ATTACK_SHARE`` would counteract the skew, so the adversary leaves part
  of its budget unused rather than fire there;
- ``benign``: open attack-free windows followed by the most attack traffic
  within the lead horizon, priming the detector toward benign just before a
  burst (evasion).

The chaff arms inject their chaff before the ``attack`` targets.

Run: uv run python scripts/adversary_schedule.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from driftgan.config import CONFIGS_DIR, load_config
from driftgan.drift.schedule import DriftSchedule

RUNS = Path("results/runs")
OUT = Path("results/adversary")
LABEL_TRACE = RUNS / "B" / "seed_0" / "trace.csv.gz"  # labels do not depend on the arm
MIN_ATTACK_SHARE = 0.5


def attack_by_index() -> tuple[np.ndarray, int]:
    """Attack indicator per stream index (array position = index - offset)."""
    with gzip.open(LABEL_TRACE, "rt") as fh:
        tr = pd.read_csv(fh, usecols=["index", "y_true"])
    idx = tr["index"].to_numpy()
    assert (np.diff(idx) == 1).all(), "trace must be contiguous"
    return (tr["y_true"] == "ATTACK").to_numpy().astype(float), int(idx[0])


def monitor_budget() -> float:
    """Mean monitor firings per run of PACE on the anchor."""
    fires = [
        json.loads(p.read_text())["n_recon_fires"]
        for p in sorted((RUNS / "C_early").glob("seed_*/lead_time.json"))
    ]
    return float(np.mean(fires))


def greedy(candidates: np.ndarray, score: np.ndarray, k: int, gap: int, seed: int = 0) -> list[int]:
    """Pick up to ``k`` candidates by descending score, at least ``gap`` apart."""
    order = np.random.default_rng(seed).permutation(len(candidates))
    order = order[np.argsort(-score[order], kind="stable")]
    picked: list[int] = []
    for j in order:
        c = int(candidates[j])
        if all(abs(c - p) >= gap for p in picked):
            picked.append(c)
            if len(picked) == k:
                break
    return sorted(picked)


def main() -> None:
    cfg = load_config(CONFIGS_DIR / "phase_c_early.yaml")
    atk, offset = attack_by_index()
    w = cfg.train_window
    gap = cfg.recon_pht_min_instances * cfg.monitor_interval
    k = round(monitor_budget())
    first = cfg.n_pretrain + gap
    horizon = cfg.lead_horizon
    last = offset + len(atk) - w - horizon
    cand = np.arange(first, last, cfg.monitor_interval)
    csum = np.concatenate([[0.0], np.cumsum(atk)])

    def share(start: np.ndarray, length: int = w) -> np.ndarray:
        a = start - offset
        return (csum[a + length] - csum[a]) / length

    dense = cand[share(cand) >= MIN_ATTACK_SHARE]
    clean = cand[share(cand) == 0]
    goals = {
        "attack": (dense, share(dense)),
        "benign": (clean, share(clean + w, horizon)),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for goal, (c, score) in goals.items():
        picks = greedy(c, score, k, gap)
        p = np.array(picks)
        meta = {
            "goal": goal, "budget": k, "gap": gap, "window": w,
            "window_attack_share": float(share(p).mean()),
            "next_horizon_attack_share": float(share(p + w, horizon).mean()),
            "candidates": int(len(c)),
        }
        DriftSchedule(indices=picks, source=f"adversary_{goal}", metadata=meta).save(
            OUT / f"cicids2017_{goal}.json"
        )
        print(f"{goal}: {len(picks)} firings, window attack share "
              f"{meta['window_attack_share']:.3f}, next {horizon} {meta['next_horizon_attack_share']:.3f}, "
              f"first {picks[0]}, last {picks[-1]}")


if __name__ == "__main__":
    main()
