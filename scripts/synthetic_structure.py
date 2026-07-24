"""Synthetic drift-structure experiment — does the timing benefit track structure?

Turns the n=3 real-dataset observation ("informed timing helps on block-structured
drift, not on dense bursts") into a controlled, predicted relationship.

We generate a synthetic stream in which the ONLY thing that varies is the
*granularity* of drift, from a few large discrete attack blocks (structured) to
many small bursts (bursty). Everything else is held fixed: feature model, attack
concepts, total attack volume, and the eval-region attack duty cycle (~15%). For
each structure level we run the three arms at matched adaptation frequency —
reactive (dynamic class PHT), blind periodic ``*_freq`` (count-matched to early),
and proactive ``*_early`` (recon+class dual) — and measure the timing component

    timing(level) = overall_local_F1(early) - overall_local_F1(freq)

Prediction: timing(level) is large when drift is structured and shrinks toward 0
as drift becomes bursty.

The base learner is HAT (the canonical monitor+trigger setup, Experiment 1). The
monitor is the cheap PCA shift monitor (Experiment 2 showed it suffices).

Usage:
    uv run python scripts/synthetic_structure.py --gen          # write CSVs
    uv run python scripts/synthetic_structure.py --run          # run all arms
    uv run python scripts/synthetic_structure.py --analyze      # table + figure
    uv run python scripts/synthetic_structure.py --all          # everything
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from driftgan.config import ExperimentConfig

# ── stream design (fixed across structure levels) ───────────────────────────
D = 20                 # feature dimensionality
L = 300_000            # total stream length
PRETRAIN = 20_000      # pretraining prefix (must contain the known attack)
ATTACK_FRAC = 0.15     # eval-region attack duty cycle (held constant)
N_CONCEPTS = 8         # number of *drifting* attack concepts (excludes concept 0)
SHIFT_MAG = 6.0        # distance of an attack blob from the benign centre
ATTACK_SIGMA = 0.5     # blob radius
DATA_SEED = 2024       # fixes the CSV; model seeds (0-4) vary independently
CENTER_SEED = 777      # fixes concept centres across ALL levels

# structure sweep: block size B (samples per attack episode). Small B -> many
# small bursts; large B -> few large blocks. E (episode count) is derived so the
# total eval attack volume is identical across levels.
BLOCK_SIZES = [14_000, 6_000, 2_000, 700, 200]

DATA_DIR = Path("results/synthetic/data")
OUT = Path("results/report/synthetic")
SEEDS = [0, 1, 2, 3, 4]


def _concept_centers() -> np.ndarray:
    """Fixed attack-blob centres: concept 0 (known) + N_CONCEPTS drifting ones."""
    rng = np.random.default_rng(CENTER_SEED)
    dirs = rng.standard_normal((N_CONCEPTS + 1, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs * SHIFT_MAG


def _episode_layout(block: int) -> list[tuple[int, int, int]]:
    """(start, end, concept) for each eval attack episode at this granularity.

    Episodes are evenly spaced across the eval region; block size is 15% of the
    slot so the duty cycle is constant and blocks never overlap. Concepts cycle
    through the drifting set 1..N_CONCEPTS (recurring for small blocks).
    """
    eval_len = L - PRETRAIN
    total_attack = int(ATTACK_FRAC * eval_len)
    n_episodes = max(1, round(total_attack / block))
    spacing = eval_len / n_episodes
    episodes = []
    for e in range(n_episodes):
        start = PRETRAIN + int(e * spacing)
        end = min(start + block, L)
        concept = 1 + (e % N_CONCEPTS)
        episodes.append((start, end, concept))
    return episodes


def generate(block: int) -> Path:
    """Write (or reuse) the CSV for one structure level; return its path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"synth_b{block}.csv"
    if path.exists():
        return path
    centers = _concept_centers()
    rng = np.random.default_rng(DATA_SEED)

    # concept id per row: -1 = benign, 0 = known attack (pretrain), k = drift
    concept_of = np.full(L, -1, dtype=int)
    # known attack block inside the pretrain prefix so HAT learns the class
    concept_of[8_000:12_000] = 0
    for start, end, concept in _episode_layout(block):
        concept_of[start:end] = concept

    X = rng.standard_normal((L, D))  # benign background everywhere
    labels = np.full(L, "benign", dtype=object)
    attack_rows = np.where(concept_of >= 0)[0]
    for r in attack_rows:
        k = concept_of[r]
        X[r] = centers[k] + ATTACK_SIGMA * rng.standard_normal(D)
        labels[r] = "attack"

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(D)])
    df["label"] = labels
    df.to_csv(path, index=False)
    n_att = int((labels == "attack").sum())
    n_ep = len(_episode_layout(block))
    print(f"  wrote {path.name}: {L} rows, {n_att} attack ({n_att / L:.1%}), "
          f"{n_ep} eval episodes of {block}")
    return path


def _base_config(block: int, phase: str, csv: Path) -> ExperimentConfig:
    """Programmatic config for a synthetic run (no YAML lookup)."""
    return ExperimentConfig(
        phase=phase,
        dataset=f"synth_b{block}",
        benign_label="benign",
        csv_path=str(csv),
        drop_cols=(),
        target_col="label",
        start_row=0,
        n_pretrain=PRETRAIN,
        max_samples=L,
        device="cpu",
        monitor_type="pca",
        monitor_interval=500,
        monitor_latent=8,
        recon_pht_threshold=0.1,
        recon_pht_delta=0.0005,
        recon_pht_min_instances=10,
        lead_horizon=20_000,
    )


def run_level(block: int) -> None:
    """Run reactive / freq / early for one structure level over all seeds."""
    from driftgan.experiments.runner import run_phase

    csv = generate(block)
    print(f"level block={block}: reactive (B)…")
    for s in SEEDS:
        run_phase(_base_config(block, "B", csv), seed=s)

    print(f"level block={block}: proactive (C_early)…")
    early_counts = []
    for s in SEEDS:
        r = run_phase(_base_config(block, "C_early", csv), seed=s)
        early_counts.append(len(r.drift_indices))
    mean_early = float(np.mean(early_counts))

    interval = max(1, round((L - PRETRAIN) / max(1.0, mean_early)))
    print(f"level block={block}: freq (C_freq) interval={interval} "
          f"(matching early≈{mean_early:.0f})…")
    for s in SEEDS:
        cfg = replace(_base_config(block, "C_freq", csv),
                      periodic_adapt_interval=interval)
        run_phase(cfg, seed=s)


def _per_seed_local_f1(block: int, phase: str) -> np.ndarray | None:
    runs = Path("results/runs") / f"synth_b{block}"
    out = []
    for s in SEEDS:
        f = runs / phase / f"seed_{s}" / "snapshots.parquet"
        if not f.exists():
            return None
        df = pd.read_parquet(f)
        out.append(float(df.loc[df["valid"], "f1"].mean()))
    return np.array(out)


def _mean_count(block: int, phase: str) -> float | None:
    import json
    runs = Path("results/runs") / f"synth_b{block}"
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
    for block in BLOCK_SIZES:
        f1_r = _per_seed_local_f1(block, "B")
        f1_f = _per_seed_local_f1(block, "C_freq")
        f1_e = _per_seed_local_f1(block, "C_early")
        if any(v is None for v in (f1_r, f1_f, f1_e)):
            continue
        assert f1_r is not None and f1_f is not None and f1_e is not None
        d_timing = f1_e - f1_f
        n_ep = len(_episode_layout(block))
        m = float(d_timing.mean())
        se = float(d_timing.std(ddof=1) / np.sqrt(len(d_timing)))
        tcrit = float(stats.t.ppf(0.975, df=len(d_timing) - 1))
        rows.append({
            "block": block, "episodes": n_ep,
            "f1_r": f1_r.mean(), "f1_f": f1_f.mean(), "f1_e": f1_e.mean(),
            "timing": m, "lo": m - tcrit * se, "hi": m + tcrit * se,
            "early_ct": _mean_count(block, "C_early"),
        })
    if not rows:
        print("  (no synthetic runs present yet)")
        return

    lines = ["# Synthetic drift-structure experiment — does timing track structure?",
             "",
             "HAT + PCA monitor. Structured (few large blocks) → bursty (many small "
             "bursts), all else fixed. Timing = overall local F1 (early − freq), "
             "paired by seed at matched adaptation frequency.",
             "",
             "| block size | eval episodes | F1 react/freq/early | timing (early−freq) 95% CI |",
             "|---:|---:|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['block']} | {r['episodes']} | "
            f"{r['f1_r']:.3f}/{r['f1_f']:.3f}/{r['f1_e']:.3f} | "
            f"{r['timing']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] |")
    trend = ("decreasing" if rows[0]["timing"] > rows[-1]["timing"] else "flat/rising")
    lines += ["", f"**Trend:** timing component is {trend} from structured "
              f"({rows[0]['timing']:+.3f}) to bursty ({rows[-1]['timing']:+.3f})."]
    (OUT / "analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    _plot(rows)


def _plot(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    ep = [r["episodes"] for r in rows]
    tim = [r["timing"] for r in rows]
    lo = [r["timing"] - r["lo"] for r in rows]
    hi = [r["hi"] - r["timing"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.axhline(0, color="#999", lw=1, ls="--")
    ax.errorbar(ep, tim, yerr=[lo, hi], marker="o", ms=8, lw=2, capsize=4,
                color="#009E73")
    ax.set_xscale("log")
    ax.set_xlabel("drift granularity — eval episodes (few large blocks → many bursts)")
    ax.set_ylabel("timing component  (overall local F1, early − freq)")
    ax.set_title("The timing benefit tracks drift structure")
    ax.grid(True, which="both", color="#eee")
    ax.text(0.02, 0.04, "Left = structured (discrete blocks): informed timing wins.\n"
            "Right = bursty (dense): timing → 0, gain is frequency.",
            transform=ax.transAxes, fontsize=9, color="#555", va="bottom")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig14_synthetic_structure.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'fig14_synthetic_structure.png'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gen", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--block", type=int, help="restrict --gen/--run to one block size")
    a = p.parse_args()
    blocks = [a.block] if a.block else BLOCK_SIZES
    if a.gen or a.all:
        for b in blocks:
            generate(b)
    if a.run or a.all:
        for b in blocks:
            run_level(b)
    if a.analyze or a.all:
        analyze()


if __name__ == "__main__":
    main()
