"""Calibrate a monitor's recon Page-Hinkley to a target fire count.

Arms in the monitor comparison must differ in the *signal*, not in how often
they fire — otherwise a difference in closed-loop stability is a fire-count
artifact rather than a property of the statistic. Every monitor therefore gets
its ``recon_pht_threshold`` tuned so its fire count matches the autoencoder's,
and the chosen value is pinned in the phase config with the evidence beside it.

The sweep is **offline**. A shift monitor watches the unlabelled input stream
only, so its score series does not depend on the threshold, on when adaptation
fires, or on the learner — one run caches the series in ``lead_time.json`` and
every candidate threshold can then be replayed against it without touching the
stream again.

Run::

    uv run python scripts/calibrate_monitor.py C_early_hdddm --target 54
    uv run python scripts/calibrate_monitor.py C_early_hdddm --reference C_early
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from river import drift as river_drift

RUNS = Path("results/runs")


def load_series(phase: str, seed: int) -> list[tuple[int, float]]:
    """Cached ``(index, score)`` series written by a completed run."""
    f = RUNS / phase / f"seed_{seed}" / "lead_time.json"
    if not f.exists():
        raise SystemExit(
            f"no cached series at {f}\n"
            f"run the phase once first, e.g.:\n"
            f"  uv run python run_experiment.py --phase {phase} --seeds {seed}"
        )
    return [(int(i), float(v)) for i, v in json.loads(f.read_text())["recon_error_series"]]


def count_fires(series: list[tuple[int, float]], threshold: float, delta: float,
                min_instances: int) -> int:
    """Fires a Page-Hinkley with these settings would produce on this series."""
    pht = river_drift.PageHinkley(
        delta=delta, threshold=threshold, min_instances=min_instances
    )
    n = 0
    for _, score in series:
        pht.update(score)
        if pht.drift_detected:
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", help="Phase whose cached score series to calibrate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=int, help="Target fire count")
    ap.add_argument("--reference", default="C_early",
                    help="Take the target from this phase's fire count (default: C_early)")
    ap.add_argument("--delta", type=float, default=0.0005)
    ap.add_argument("--min-instances", type=int, default=10)
    ap.add_argument("--n-steps", type=int, default=60, help="Thresholds to try")
    args = ap.parse_args()

    series = load_series(args.phase, args.seed)
    scores = np.array([s for _, s in series])
    target = args.target
    if target is None:
        ref = RUNS / args.reference / f"seed_{args.seed}" / "lead_time.json"
        if not ref.exists():
            raise SystemExit(f"no reference run at {ref}; pass --target instead")
        target = int(json.loads(ref.read_text())["n_recon_fires"])

    print(f"{args.phase} seed {args.seed}: {len(scores)} checks")
    print(f"  score distribution: min {scores.min():.4f} median {np.median(scores):.4f} "
          f"p90 {np.percentile(scores, 90):.4f} max {scores.max():.4f}")
    print(f"  target fire count: {target} (from {args.reference})\n")

    # The PHT cumulative-sum threshold scales with the score magnitude, so sweep
    # a band anchored to the observed spread rather than a fixed grid.
    spread = float(scores.max() - scores.min()) or 1.0
    grid = np.linspace(spread * 0.02, spread * 3.0, args.n_steps)

    rows = [(float(t), count_fires(series, float(t), args.delta, args.min_instances))
            for t in grid]
    best = min(rows, key=lambda r: (abs(r[1] - target), r[0]))

    print("  threshold -> fires")
    shown = 0
    for t, n in rows:
        if shown and n == rows[max(0, shown - 1)][1]:
            shown += 1
            continue
        print(f"   {t:9.4f} {n:5d}{'   <-- best' if (t, n) == best else ''}")
        shown += 1
    print(f"\nPin in configs/phase_{args.phase.lower()}.yaml:")
    print(f"  recon_pht_threshold: {best[0]:.4f}   # {best[1]} fires "
          f"(target {target}, delta {args.delta})")


if __name__ == "__main__":
    main()
