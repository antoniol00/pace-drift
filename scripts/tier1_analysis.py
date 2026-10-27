"""Tier 1 analysis — timing vs frequency, decisively.

Tier 0 could not tell whether the F1 gain came from *when* the model adapts or
merely *how often* (the proactive arm fires more than the reactive schedule on
CIC-IDS2018 and UNSW-NB15). Every one of the nine dataset x learner cells
therefore has a frequency-matched control ``*_freq`` (blind periodic clock, fire
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
these datasets was frequency, and the framing must say so. Both components are
also aggregated over the nine cells, cell means as the unit of replication.

Two further views answer the confounds a fire count hides:

  labels  the share of the evaluated stream each arm learns from. A firing opens
          a train_window and a firing inside an open window restarts it, so
          clustered firings cost fewer labels than their count suggests.
  live    *_sched replays a schedule recorded from a live HAT pass, while the
          error test inside *_early watches the ensemble's own error. ``*_pht``
          runs that error test live with no monitor, so early - pht isolates
          the monitor. Fires of every *_early arm are split by source.

Run:  uv run python scripts/tier1_analysis.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from driftgan.metrics.cost import labelled_samples

RUNS = Path("results/runs")
OUT = Path("results/report/tier1")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]  # fallback; cells resolve their own set via cell_seeds()

# (dataset, learner, reactive, frequency control, proactive, live reactive)
CELLS = [
    (ds, learner, *arms)
    for ds in ("cicids2017", "cicids2018", "unsw")
    for learner, arms in (
        ("HAT", ("B", "C_freq", "C_early", None)),  # B's schedule is HAT's own live PHT
        ("ARF", ("ARF_sched", "ARF_freq", "ARF_early", "ARF_pht")),
        ("SRP", ("SRP_sched", "SRP_freq", "SRP_early", "SRP_pht")),
    )
]
DSET = {"cicids2017": "CIC-IDS2017", "cicids2018": "CIC-IDS2018", "unsw": "UNSW-NB15"}


def run_dir(dataset: str, phase: str, seed: int) -> Path:
    base = RUNS if dataset == "cicids2017" else RUNS / dataset
    return base / phase / f"seed_{seed}"


def cell_seeds(dataset: str, *phases: str) -> list[int]:
    """Seeds completed for every phase of a cell.

    The three-arm decomposition is paired across sched/freq/early, so a seed
    counts only when all three ran; cells therefore resolve their own set
    instead of assuming one global seed count.
    """
    sets = [
        {
            int(p.name.split("_")[1])
            for p in run_dir(dataset, ph, 0).parent.glob("seed_*")
            if (p / "summary.json").exists()
        }
        for ph in phases
    ]
    return sorted(set.intersection(*sets)) if sets else []


def per_seed_local_f1(dataset: str, phase: str, seeds: list[int]) -> np.ndarray | None:
    out = []
    for s in seeds:
        f = run_dir(dataset, phase, s) / "snapshots.parquet"
        if not f.exists():
            return None
        df = pd.read_parquet(f)
        out.append(float(df.loc[df["valid"], "f1"].mean()))
    return np.array(out) if out else None


def per_seed_n_drifts(dataset: str, phase: str, seeds: list[int]) -> np.ndarray | None:
    out = []
    for s in seeds:
        f = run_dir(dataset, phase, s) / "summary.json"
        if not f.exists():
            return None
        out.append(int(json.loads(f.read_text())["n_drifts"]))
    return np.array(out) if out else None


def per_seed_labels(dataset: str, phase: str, seeds: list[int]) -> np.ndarray:
    """Share of the evaluated stream (after pretraining) each seed learned from."""
    out = []
    for s in seeds:
        rd = run_dir(dataset, phase, s)
        last = json.loads((rd / "summary.json").read_text())["n_processed"]
        cfg = json.loads((rd / "manifest.json").read_text())["config"]
        fires = [e["index"] for e in json.loads((rd / "drift_events.json").read_text())["events"]]
        out.append(labelled_samples(fires, cfg["train_window"], last) / (last - cfg["n_pretrain"]))
    return np.array(out)


def fire_sources(dataset: str, phase: str, seeds: list[int]) -> dict[str, float]:
    """Mean firings per seed by recorded source (class, recon, recon+class)."""
    total: Counter[str] = Counter()
    for s in seeds:
        events = json.loads((run_dir(dataset, phase, s) / "drift_events.json").read_text())
        total.update(e["source"] for e in events["events"])
    return {k: v / len(seeds) for k, v in total.items()}


def paired_ci(d: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return m - tcrit * se, m + tcrit * se


def ci_str(d: np.ndarray) -> str:
    lo, hi = paired_ci(d)
    return f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"


def verdict(d_timing: np.ndarray, d_freq: np.ndarray) -> str:
    """The larger positive component whose CI excludes zero, if any."""
    clear = {name: float(d.mean()) for name, d in (("TIMING", d_timing), ("FREQ", d_freq))
             if paired_ci(d)[0] > 0}
    return max(clear, key=lambda k: clear[k]) if clear else "UNRESOLVED"


def aggregate(name: str, cell_means: np.ndarray) -> str:
    """Nine-cell aggregate with cell means as the unit of replication."""
    m = float(cell_means.mean())
    se = float(cell_means.std(ddof=1) / np.sqrt(len(cell_means)))
    tcrit = float(stats.t.ppf(0.975, df=len(cell_means) - 1))
    n_pos = int((cell_means > 0).sum())
    sign_p = float(stats.binomtest(n_pos, len(cell_means), 0.5).pvalue)
    return (f"- **{name}**: mean {m:+.3f}, 95% CI [{m - tcrit * se:+.3f}, "
            f"{m + tcrit * se:+.3f}], {n_pos}/{len(cell_means)} cells positive "
            f"(two-sided sign test p={sign_p:.4f})")


def live_control_section() -> list[str]:
    """early vs the live error-only test on the ensemble's own predictions."""
    lines = [
        "",
        "## Live reactive control for the ensembles",
        "",
        "`*_pht` runs the error Page-Hinkley live on the ensemble's own predictions "
        "with no monitor; `*_sched` replays the HAT-derived schedule. "
        "`Δ monitor = early − pht` isolates what the monitor adds to the same live "
        "error test; `labels` is the share of the evaluated stream learned from.",
        "",
        "| dataset | learner | n | fires sched/pht/early | labels sched/pht/early | "
        "F1 sched/pht/early | Δ live (pht−sched) | Δ monitor (early−pht) |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    pending = []
    for ds, learner, r, _fq, e, live in CELLS:
        if live is None:
            continue
        seeds = cell_seeds(ds, r, live, e)
        f1 = [per_seed_local_f1(ds, p, seeds) for p in (r, live, e)]
        nd = [per_seed_n_drifts(ds, p, seeds) for p in (r, live, e)]
        if not seeds or any(x is None for x in (*f1, *nd)):
            pending.append(f"{DSET[ds]}/{learner}")
            continue
        f1r, f1l, f1e = (x for x in f1 if x is not None)
        lab = [per_seed_labels(ds, p, seeds).mean() for p in (r, live, e)]
        lines.append(
            f"| {DSET[ds]} | {learner} | {len(seeds)} | "
            + "/".join(f"{x.mean():.0f}" for x in nd if x is not None) + " | "
            + "/".join(f"{100 * x:.1f}%" for x in lab) + " | "
            f"{f1r.mean():.3f}/{f1l.mean():.3f}/{f1e.mean():.3f} | "
            f"{ci_str(f1l - f1r)} | {ci_str(f1e - f1l)} |")
    if pending:
        lines += ["", f"> Pending (runs not complete): {', '.join(pending)}."]
    return lines


def sources_section() -> list[str]:
    lines = [
        "",
        "## Firings of the proactive arms by source",
        "",
        "Mean firings per seed. `recon` comes from the monitor's Page-Hinkley, "
        "`class` from the error Page-Hinkley, `recon+class` from both at the same "
        "sample. The monitor is independent of the learner, so its firings repeat "
        "across HAT, ARF and SRP at a given seed and dataset.",
        "",
        "| dataset | learner | total | class | recon | recon+class |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for ds, learner, r, _fq, e, _live in CELLS:
        seeds = cell_seeds(ds, r, e)  # the core seeds, not C_early's twenty
        src = fire_sources(ds, e, seeds)
        lines.append(
            f"| {DSET[ds]} | {learner} | {sum(src.values()):.1f} | "
            f"{src.get('class', 0):.1f} | {src.get('recon', 0):.1f} | "
            f"{src.get('recon+class', 0):.1f} |")
    return lines


def main() -> None:
    lines = [
        "# Tier 1 — timing vs frequency in all nine cells",
        "",
        "Frequency-matched control (`*_freq`, blind periodic clock, fire count "
        "calibrated to `*_early`). Overall local F1 paired by seed; `n` is the "
        "seed count available for all three arms of that cell. The "
        "reported gain `early−sched` is split into a **frequency** part "
        "(`freq−sched`) and a **timing** part (`early−freq`). A positive timing "
        "part with CI excluding 0 means informed placement helps beyond count. "
        "`labels` is the share of the evaluated stream each arm learned from: the "
        "blind clock's windows never overlap, so at a matched count it learns from "
        "more labels than the proactive arm.",
        "",
        "| dataset | learner | n | fires sched/freq/early | labels sched/freq/early | "
        "F1 sched/freq/early | Δ total (early−sched) | Δ freq (freq−sched) | "
        "Δ timing (early−freq) | verdict |",
        "|---|---|---:|---|---|---|---|---|---|:---:|",
    ]
    pending = []
    timing_deltas = []
    cell_means: dict[str, list[float]] = {"total": [], "frequency": [], "timing": []}
    for ds, learner, r, fq, e, _live in CELLS:
        seeds = cell_seeds(ds, r, fq, e)
        f1r, f1f, f1e = (per_seed_local_f1(ds, p, seeds) for p in (r, fq, e))
        ndr, ndf, nde = (per_seed_n_drifts(ds, p, seeds) for p in (r, fq, e))
        if f1r is None or f1f is None or f1e is None or \
                ndr is None or ndf is None or nde is None:
            pending.append(f"{DSET[ds]}/{learner}")
            continue
        d_total = f1e - f1r
        d_freq = f1f - f1r
        d_timing = f1e - f1f
        timing_deltas.append((ds, learner, d_timing))
        for k, d in (("total", d_total), ("frequency", d_freq), ("timing", d_timing)):
            cell_means[k].append(float(d.mean()))
        lab = [per_seed_labels(ds, p, seeds).mean() for p in (r, fq, e)]
        lines.append(
            f"| {DSET[ds]} | {learner} | {len(seeds)} | "
            f"{ndr.mean():.0f}/{ndf.mean():.0f}/{nde.mean():.0f} | "
            + "/".join(f"{100 * x:.1f}%" for x in lab) + " | "
            f"{f1r.mean():.3f}/{f1f.mean():.3f}/{f1e.mean():.3f} | "
            f"{ci_str(d_total)} | {ci_str(d_freq)} | {ci_str(d_timing)} | "
            f"{verdict(d_timing, d_freq)} |")

    if pending:
        lines += ["", f"> Pending (runs not complete): {', '.join(pending)}."]
    elif len(cell_means["timing"]) == len(CELLS):
        lines += ["", f"**Aggregate over the {len(CELLS)} cells** (cell means as the unit "
                  "of replication; the cells share a monitor design and three captures, "
                  "so the sign-test p is optimistic):"]
        lines += [aggregate(f"Δ {k}", np.array(v)) for k, v in cell_means.items()]

    if timing_deltas:
        n_timing = sum(1 for _, _, d in timing_deltas if paired_ci(d)[0] > 0)
        lines += [
            "",
            f"**Bottom line:** timing component positive with CI>0 in "
            f"{n_timing}/{len(timing_deltas)} cells. "
            + ("The 'when to adapt' effect holds in every cell at matched "
               "frequency." if n_timing == len(timing_deltas) and timing_deltas
               else "Where it is ~0, the gain was adaptation frequency, not "
               "timing — reframe accordingly."),
        ]

    lines += live_control_section() + sources_section()
    doc = "\n".join(lines) + "\n"
    (OUT / "analysis.md").write_text(doc)
    print(doc)
    print(f"wrote {OUT / 'analysis.md'}")


if __name__ == "__main__":
    main()
