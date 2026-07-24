"""Synthetic drift-*clustering* experiment (redesign B, pre-specified).

The first synthetic sweep (``synthetic_structure.py``) varied drift *block size*
and did NOT reproduce the timing effect — because (i) large blocks let a blind
periodic clock re-adapt mid-block, and (ii) with uniformly-spaced drifts a
count-matched uniform clock aligns with them by construction, forcing a tie. That
was an honest negative, kept on record.

This redesign isolates the mechanism that actually makes informed timing beat a
blind clock at matched frequency: **temporal concentration of drift onsets**. A
uniform clock spends its fixed budget uniformly; if the drifts are packed into
part of the stream, the uniform clock wastes fires on the quiet region while the
informed monitor concentrates on the drifts.

PRE-SPECIFIED DESIGN (fixed before running; do not tune to taste):
  * fixed small blocks B=1000 (<= train_window, so one fire adapts one block —
    removes the mid-block re-adaptation artifact);
  * short train_window=1000 (so *where* you adapt matters);
  * fixed E=40 drift blocks, fixed total attack volume, concepts cycling;
  * the ONLY variable is phi in {1.0, 0.7, 0.5, 0.35, 0.25}: the 40 onsets are
    spread evenly within the first phi of the eval region (phi=1 uniform,
    phi=0.25 concentrated), the rest benign.

PRE-SPECIFIED PREDICTION:
  timing(phi) = overall_local_F1(early) - overall_local_F1(freq) (paired by seed,
  matched fire count) is ~0 at phi=1.0 and INCREASES as phi decreases.

PRE-SPECIFIED DECISION RULE:
  mechanism demonstrated iff timing(0.25) > 0 with 95% CI excluding 0, AND
  timing(1.0) CI includes 0, AND timing(0.25) > timing(1.0) clearly. Otherwise
  the claim is retreated (option A) with no further attempts.

Usage: uv run python scripts/synthetic_clustering.py --all
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scripts.synthetic_structure import (
    PRETRAIN,
    D,
    L,
    _concept_centers,
)

from driftgan.config import ExperimentConfig

BLOCK = 1_000          # fixed small block (<= train_window)
N_DRIFTS = 40          # fixed number of drift onsets across all levels
TRAIN_WINDOW = 1_000   # short, so placement matters
ATTACK_SIGMA = 0.5
DATA_SEED = 3131
PHIS = [1.0, 0.7, 0.5, 0.35, 0.25]  # temporal concentration of the onsets

DATA_DIR = Path("results/synthetic/data")
OUT = Path("results/report/synthetic_clustering")
SEEDS = [0, 1, 2, 3, 4]


def _tag(phi: float) -> str:
    return f"c{int(round(phi * 100)):03d}"


def _layout(phi: float) -> list[tuple[int, int, int]]:
    """(start, end, concept) for the 40 onsets packed into the first phi."""
    eval_len = L - PRETRAIN
    active = int(phi * eval_len)
    spacing = active / N_DRIFTS
    episodes = []
    for e in range(N_DRIFTS):
        start = PRETRAIN + int(e * spacing)
        end = min(start + BLOCK, L)
        episodes.append((start, end, 1 + (e % 8)))
    return episodes


def generate(phi: float) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"synth_{_tag(phi)}.csv"
    if path.exists():
        return path
    centers = _concept_centers()
    rng = np.random.default_rng(DATA_SEED)

    concept_of = np.full(L, -1, dtype=int)
    concept_of[8_000:12_000] = 0  # known attack inside pretrain
    for start, end, concept in _layout(phi):
        concept_of[start:end] = concept

    X = rng.standard_normal((L, D))
    labels = np.full(L, "benign", dtype=object)
    for r in np.where(concept_of >= 0)[0]:
        X[r] = centers[concept_of[r]] + ATTACK_SIGMA * rng.standard_normal(D)
        labels[r] = "attack"

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(D)])
    df["label"] = labels
    df.to_csv(path, index=False)
    n_att = int((labels == "attack").sum())
    print(f"  wrote {path.name}: phi={phi} active_span={int(phi * (L - PRETRAIN))} "
          f"{N_DRIFTS} onsets, {n_att} attack ({n_att / L:.1%})")
    return path


def _cfg(phi: float, phase: str, csv: Path) -> ExperimentConfig:
    return ExperimentConfig(
        phase=phase, dataset=f"synth_{_tag(phi)}", benign_label="benign",
        csv_path=str(csv), drop_cols=(), target_col="label", start_row=0,
        n_pretrain=PRETRAIN, max_samples=L, device="cpu",
        train_window=TRAIN_WINDOW,
        monitor_type="pca", monitor_interval=500, monitor_latent=8,
        recon_pht_threshold=0.1, recon_pht_delta=0.0005,
        recon_pht_min_instances=10, lead_horizon=20_000,
    )


def run_level(phi: float) -> None:
    from driftgan.experiments.runner import run_phase

    csv = generate(phi)
    print(f"phi={phi}: reactive (B)…")
    for s in SEEDS:
        run_phase(_cfg(phi, "B", csv), seed=s)
    print(f"phi={phi}: proactive (C_early)…")
    counts = []
    for s in SEEDS:
        counts.append(len(run_phase(_cfg(phi, "C_early", csv), seed=s).drift_indices))
    mean_early = float(np.mean(counts))
    interval = max(1, round((L - PRETRAIN) / max(1.0, mean_early)))
    print(f"phi={phi}: freq interval={interval} (matching early≈{mean_early:.0f})…")
    for s in SEEDS:
        run_phase(replace(_cfg(phi, "C_freq", csv), periodic_adapt_interval=interval),
                  seed=s)


def _f1(phi: float, phase: str) -> np.ndarray | None:
    runs = Path("results/runs") / f"synth_{_tag(phi)}"
    out = []
    for s in SEEDS:
        f = runs / phase / f"seed_{s}" / "snapshots.parquet"
        if not f.exists():
            return None
        df = pd.read_parquet(f)
        out.append(float(df.loc[df["valid"], "f1"].mean()))
    return np.array(out)


def _count(phi: float, phase: str) -> float | None:
    runs = Path("results/runs") / f"synth_{_tag(phi)}"
    vals = []
    for s in SEEDS:
        f = runs / phase / f"seed_{s}" / "summary.json"
        if not f.exists():
            return None
        vals.append(json.loads(f.read_text())["n_drifts"])
    return float(np.mean(vals))


def analyze() -> None:
    from scipy import stats

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for phi in PHIS:
        f1r, f1f, f1e = _f1(phi, "B"), _f1(phi, "C_freq"), _f1(phi, "C_early")
        if any(v is None for v in (f1r, f1f, f1e)):
            continue
        assert f1r is not None and f1f is not None and f1e is not None
        d = f1e - f1f
        m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        tc = float(stats.t.ppf(0.975, df=len(d) - 1))
        rows.append({"phi": phi, "f1_r": f1r.mean(), "f1_f": f1f.mean(),
                     "f1_e": f1e.mean(), "timing": m, "lo": m - tc * se,
                     "hi": m + tc * se, "ct": _count(phi, "C_early")})
    if not rows:
        print("  (no clustering runs present yet)")
        return

    lines = ["# Synthetic drift-clustering experiment (redesign B)",
             "",
             "HAT + PCA monitor, fixed small blocks (B=1000, train_window=1000), "
             "40 onsets. Only phi (temporal concentration) varies: phi=1 uniform, "
             "phi=0.25 concentrated. Timing = overall local F1 (early − freq), "
             "paired by seed at matched fire count.",
             "",
             "| phi (concentration) | early fires | F1 react/freq/early | timing (early−freq) 95% CI |",
             "|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['phi']:.2f} | {r['ct']:.0f} | "
                     f"{r['f1_r']:.3f}/{r['f1_f']:.3f}/{r['f1_e']:.3f} | "
                     f"{r['timing']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] |")

    uniform = next(r for r in rows if r["phi"] == 1.0)
    conc = min(rows, key=lambda r: r["phi"])
    demonstrated = (conc["lo"] > 0 and uniform["lo"] <= 0 <= uniform["hi"]
                    and conc["timing"] > uniform["timing"])
    lines += ["",
              f"**Pre-specified decision:** concentrated (phi={conc['phi']}) timing "
              f"{conc['timing']:+.3f} [{conc['lo']:+.3f},{conc['hi']:+.3f}]; "
              f"uniform (phi=1.0) timing {uniform['timing']:+.3f} "
              f"[{uniform['lo']:+.3f},{uniform['hi']:+.3f}]. "
              + ("MECHANISM DEMONSTRATED — informed timing beats a blind clock "
                 "exactly when drift is temporally concentrated."
                 if demonstrated else
                 "FLAT / not demonstrated — retreat to option A (report the real-"
                 "dataset split as an empirical observation, no mechanism claim).")]
    (OUT / "analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    _plot(rows)


def _plot(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    phi = [r["phi"] for r in rows]
    tim = [r["timing"] for r in rows]
    lo = [r["timing"] - r["lo"] for r in rows]
    hi = [r["hi"] - r["timing"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.axhline(0, color="#999", lw=1, ls="--")
    ax.errorbar(phi, tim, yerr=[lo, hi], marker="o", ms=8, lw=2, capsize=4,
                color="#009E73")
    ax.invert_xaxis()  # concentrated (small phi) on the right
    ax.set_xlabel("drift onsets spread over fraction φ of the stream "
                  "(1.0 = uniform → 0.25 = concentrated)")
    ax.set_ylabel("timing component  (overall local F1, early − freq)")
    ax.set_title("Informed timing wins when drift is temporally concentrated")
    ax.grid(True, color="#eee")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig14b_synthetic_clustering.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'fig14b_synthetic_clustering.png'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gen", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--phi", type=float)
    a = p.parse_args()
    phis = [a.phi] if a.phi else PHIS
    if a.gen or a.all:
        for ph in phis:
            generate(ph)
    if a.run or a.all:
        for ph in phis:
            run_level(ph)
    if a.analyze or a.all:
        analyze()


if __name__ == "__main__":
    main()
