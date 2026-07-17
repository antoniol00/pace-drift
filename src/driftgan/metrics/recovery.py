"""Per-drift-event recovery metrics: RT, AUD, min-F1 and no-recovery flag.

Semantics (Improvement 3, see also docs/methodology.md):

- Snapshots are periodic samples of the rolling F1, taken every
  ``snap_interval`` stream samples. Each snapshot carries a ``valid`` flag:
  a snapshot is *invalid* when its rolling window contained no positive
  (ATTACK) true label, making F1 undefined. Invalid snapshots are excluded
  from RT and AUD (they are neither failure nor success); their fraction is
  reported per event as ``excluded_windows_pct``.

- An event spans from its drift index ``d`` to ``min(d + horizon,
  next_drift_index)``. If another drift occurs before recovery, the current
  event is closed with ``no_recovery=True`` and a new event starts
  (overlapping recoveries are never double-counted).

- Recovery Time (RT): number of stream samples from ``d`` until the first
  valid snapshot that starts a run of ``k`` consecutive valid snapshots with
  ``F1 >= theta``. RT is measured to the *first* snapshot of that run.
  ``None`` when the event does not recover.

- Area Under Deficit (AUD): integral of ``max(0, theta - F1(t))`` over
  valid snapshots from ``d`` until the recovery snapshot (or the event end
  when there is no recovery), rectangle rule with the actual sample spacing
  between consecutive valid snapshots. Units: deficit x samples.

- min_F1: minimum F1 over the valid snapshots of the event window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RecoveryEvent:
    """Recovery metrics for a single drift event."""

    event_id: int
    drift_index: int
    RT: int | None
    AUD: float
    min_F1: float
    no_recovery: bool
    excluded_windows_pct: float


def compute_recovery_events(
    snap_indices: np.ndarray,
    snap_f1: np.ndarray,
    snap_valid: np.ndarray,
    drift_indices: list[int],
    theta: float = 0.9,
    k: int = 3,
    horizon: int = 50_000,
) -> list[RecoveryEvent]:
    """Compute per-event recovery metrics from snapshot series.

    Parameters
    ----------
    snap_indices : array of int
        Stream index of each snapshot (monotonically increasing).
    snap_f1 : array of float
        Rolling F1 at each snapshot.
    snap_valid : array of bool
        Whether F1 was defined at each snapshot (window had positive labels).
    drift_indices : list of int
        Stream indices of drift events (the shared schedule or dynamic fires).
    theta, k, horizon :
        Recovery threshold, sustained-run length, and max samples per event.
    """
    snap_indices = np.asarray(snap_indices)
    snap_f1 = np.asarray(snap_f1, dtype=float)
    snap_valid = np.asarray(snap_valid, dtype=bool)
    if not (len(snap_indices) == len(snap_f1) == len(snap_valid)):
        raise ValueError("snapshot arrays must have equal length")

    events: list[RecoveryEvent] = []
    ordered = sorted(drift_indices)
    for event_id, d in enumerate(ordered):
        next_d = ordered[event_id + 1] if event_id + 1 < len(ordered) else None
        end = d + horizon if next_d is None else min(d + horizon, next_d)

        # Snapshots inside (d, end]
        in_window = (snap_indices > d) & (snap_indices <= end)
        idx = snap_indices[in_window]
        f1 = snap_f1[in_window]
        valid = snap_valid[in_window]

        n_total = int(len(idx))
        excluded_pct = float(100.0 * (~valid).sum() / n_total) if n_total else 100.0

        v_idx = idx[valid]
        v_f1 = f1[valid]

        if len(v_f1) == 0:
            events.append(
                RecoveryEvent(
                    event_id=event_id,
                    drift_index=int(d),
                    RT=None,
                    AUD=0.0,
                    min_F1=float("nan"),
                    no_recovery=True,
                    excluded_windows_pct=excluded_pct,
                )
            )
            continue

        # Recovery: first valid snapshot starting a run of k consecutive
        # valid snapshots with F1 >= theta. Runs are over the *valid*
        # subsequence: invalid snapshots neither break nor extend a run.
        above = v_f1 >= theta
        recovery_pos: int | None = None
        run = 0
        for i, ok in enumerate(above):
            run = run + 1 if ok else 0
            if run >= k:
                recovery_pos = i - k + 1
                break
        # A run at the tail shorter than k because the event window ended is
        # not a recovery (sustained requirement not met within the window);
        # events truncated by the next drift before recovering close as
        # no_recovery=True.

        recovered = recovery_pos is not None

        if recovery_pos is not None:
            rt = int(v_idx[recovery_pos] - d)
            aud_end_pos = recovery_pos
        else:
            rt = None
            aud_end_pos = len(v_idx) - 1

        # AUD: rectangle rule over valid snapshots from d to the recovery
        # snapshot (inclusive of segments up to it). Each valid snapshot i
        # contributes deficit(i) * (idx[i] - prev_idx), prev starting at d.
        deficit = np.maximum(0.0, theta - v_f1)
        prev = float(d)
        aud = 0.0
        for i in range(aud_end_pos + 1):
            dx = float(v_idx[i]) - prev
            aud += float(deficit[i]) * dx
            prev = float(v_idx[i])

        events.append(
            RecoveryEvent(
                event_id=event_id,
                drift_index=int(d),
                RT=rt,
                AUD=float(aud),
                min_F1=float(v_f1[: aud_end_pos + 1].min()),
                no_recovery=not recovered,
                excluded_windows_pct=excluded_pct,
            )
        )
    return events


def events_to_dataframe(events: list[RecoveryEvent]) -> pd.DataFrame:
    """Convert events to the canonical parquet schema."""
    df = pd.DataFrame([asdict(e) for e in events])
    if df.empty:
        df = pd.DataFrame(
            columns=["event_id", "drift_index", "RT", "AUD", "min_F1", "no_recovery", "excluded_windows_pct"]
        )
    return df


def save_recovery(events: list[RecoveryEvent], phase: str, seed: int, results_dir: str | Path) -> Path:
    """Write ``results/recovery/{phase}_{seed}.parquet`` and return its path."""
    out_dir = Path(results_dir) / "recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{phase}_{seed}.parquet"
    events_to_dataframe(events).to_parquet(path, index=False)
    return path


def aggregate_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-event metrics: mean, median and IQR of RT and AUD.

    ``RT`` aggregates are computed over recovered events only;
    ``no_recovery_rate`` reports the excluded fraction.
    """
    rt = df.loc[~df["no_recovery"], "RT"].astype(float)
    aud = df["AUD"].astype(float)
    def iqr(s: pd.Series) -> float:
        return float(s.quantile(0.75) - s.quantile(0.25)) if len(s) else float("nan")

    return pd.DataFrame(
        {
            "n_events": [len(df)],
            "no_recovery_rate": [float(df["no_recovery"].mean()) if len(df) else float("nan")],
            "RT_mean": [rt.mean() if len(rt) else float("nan")],
            "RT_median": [rt.median() if len(rt) else float("nan")],
            "RT_iqr": [iqr(rt)],
            "AUD_mean": [aud.mean() if len(aud) else float("nan")],
            "AUD_median": [aud.median() if len(aud) else float("nan")],
            "AUD_iqr": [iqr(aud)],
            "min_F1_mean": [float(df["min_F1"].mean()) if len(df) else float("nan")],
            "excluded_windows_pct_mean": [float(df["excluded_windows_pct"].mean()) if len(df) else float("nan")],
        }
    )
