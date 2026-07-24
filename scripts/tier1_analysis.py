"""Tier 1 analysis — timing vs frequency, decisively.

For the external-dataset cells where the proactive arm fires MORE than the
reactive schedule (ARF/SRP on CIC-IDS2018 and UNSW-NB15), Tier 0 could not tell
whether the F1 gain came from *when* the model adapts or merely *how often*.
This adds a frequency-matched control ``*_freq`` (blind periodic clock, fire
count calibrated to ``*_early``) so three arms sit at comparable frequency:

  sched  reactive (foreign schedule, fewer fires) -- the original 2x2 baseline
  freq   blind periodic, SAME fire count as early -- pure frequency
  early  proactive informed timing, same count    -- frequency + timing

Decomposition of the originally reported gain, paired by seed:

  early - sched  (reported)  =  (freq - sched)      +  (early - freq)
                                 frequency component    timing component

If the timing component (early - freq) is positive with a CI excluding 0, the
covariate-shift-informed placement helps beyond raw adaptation count, and the
"when to adapt" claim generalizes past the 2017 anchor. If it is ~0, the gain on
these datasets was frequency, and the framing must say so.

Run:  uv run python scripts/tier1_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUNS = Path("results/runs")
OUT = Path("results/report/tier1")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]

# decisive subset: (dataset, learner, reactive, frequency-control, proactive)
CELLS = [
    ("cicids2018", "ARF", "ARF_sched", "ARF_freq", "ARF_early"),
    ("cicids2018", "SRP", "SRP_sched", "SRP_freq", "SRP_early"),
    ("unsw", "ARF", "ARF_sched", "ARF_freq", "ARF_early"),
    ("unsw", "SRP", "SRP_sched", "SRP_freq", "SRP_early"),
]
DSET = {"cicids2018": "CIC-IDS2018", "unsw": "UNSW-NB15"}


def run_dir(dataset: str, phase: str, seed: int) -> Path:
    base = RUNS if dataset == "cicids2017" else RUNS / dataset
    return base / phase / f"seed_{seed}"


def per_seed_local_f1(dataset: str, phase: str) -> np.ndarray | None:
    out = []
    for s in SEEDS:
        f = run_dir(dataset, phase, s) / "snapshots.parquet"
        if not f.exists():
            return None
        df = pd.read_parquet(f)
        out.append(float(df.loc[df["valid"], "f1"].mean()))
    return np.array(out)


def per_seed_n_drifts(dataset: str, phase: str) -> np.ndarray | None:
    out = []
    for s in SEEDS:
        f = run_dir(dataset, phase, s) / "summary.json"
        if not f.exists():
            return None
        out.append(int(json.loads(f.read_text())["n_drifts"]))
    return np.array(out)


def paired_ci(d: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return m - tcrit * se, m + tcrit * se


def ci_str(d: np.ndarray) -> str:
    lo, hi = paired_ci(d)
    return f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"


def verdict(d: np.ndarray) -> str:
    lo, hi = paired_ci(d)
    same = (d > 0).all() or (d < 0).all()
    if same and lo > 0:
        return "TIMING"
    if same and hi < 0:
        return "TIMING-NEG"
    return "~FREQ"


def main() -> None:
    lines = [
        "# Tier 1 — timing vs frequency on the external datasets",
        "",
        "Frequency-matched control (`*_freq`, blind periodic clock, fire count "
        "calibrated to `*_early`). Overall local F1 paired by seed (n=5). The "
        "reported gain `early−sched` is split into a **frequency** part "
        "(`freq−sched`) and a **timing** part (`early−freq`). A positive timing "
        "part with CI excluding 0 means informed placement helps beyond count.",
        "",
        "| dataset | learner | fires sched/freq/early | F1 sched/freq/early | "
        "Δ total (early−sched) | Δ freq (freq−sched) | Δ timing (early−freq) | "
        "verdict |",
        "|---|---|---|---|---|---|---|:---:|",
    ]
    pending = []
    timing_deltas = []
    for ds, learner, r, fq, e in CELLS:
        f1r, f1f, f1e = (per_seed_local_f1(ds, p) for p in (r, fq, e))
        ndr, ndf, nde = (per_seed_n_drifts(ds, p) for p in (r, fq, e))
        if f1r is None or f1f is None or f1e is None or \
                ndr is None or ndf is None or nde is None:
            pending.append(f"{DSET[ds]}/{learner}")
            continue
        d_total = f1e - f1r
        d_freq = f1f - f1r
        d_timing = f1e - f1f
        timing_deltas.append((ds, learner, d_timing))
        lines.append(
            f"| {DSET[ds]} | {learner} | "
            f"{ndr.mean():.0f}/{ndf.mean():.0f}/{nde.mean():.0f} | "
            f"{f1r.mean():.3f}/{f1f.mean():.3f}/{f1e.mean():.3f} | "
            f"{ci_str(d_total)} | {ci_str(d_freq)} | {ci_str(d_timing)} | "
            f"{verdict(d_timing)} |")

    if pending:
        lines += ["", f"> Pending (runs not complete): {', '.join(pending)}."]

    if timing_deltas:
        n_timing = sum(1 for _, _, d in timing_deltas
                       if paired_ci(d)[0] > 0 and (d > 0).all())
        lines += [
            "",
            f"**Bottom line:** timing component positive with CI>0 in "
            f"{n_timing}/{len(timing_deltas)} decisive cells. "
            + ("The 'when to adapt' effect generalizes beyond 2017 at matched "
               "frequency." if n_timing == len(timing_deltas) and timing_deltas
               else "Where it is ~0, the external-dataset gain was adaptation "
               "frequency, not timing — reframe accordingly."),
        ]

    doc = "\n".join(lines) + "\n"
    (OUT / "analysis.md").write_text(doc)
    print(doc)
    print(f"wrote {OUT / 'analysis.md'}")


if __name__ == "__main__":
    main()
