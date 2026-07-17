# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A concept-drift research project for network intrusion detection on the CIC-IDS2017 dataset. The core research goal is a **novel approach to improve drift detection/adaptation using GANs applied early**, generating synthetic samples of a new attack type from only a handful of real examples right when drift is detected — rather than waiting for enough real samples to accumulate before retraining. Modules A and B are baselines that the proposal (Module C) is measured against.

- **Module A** (`module_a_baseline/`) — Static HAT model: pre-trained once, then frozen (predict-only). Lower-bound baseline.
- **Module B** (`module_b_pht/`) — Reactive HAT model: `river.drift.PageHinkley` (PHT) detects drift and triggers a rolling retraining window. Standard reactive baseline, no synthetic data.
- **Module C** (`module_c_gan/`) — **The proposal.** Same PHT-based detection as Module B, but at each detected drift, a lightweight `DriftGAN` (PyTorch GAN) trains on the small buffer of recent real attack samples and generates synthetic variations, which are fed into the tree via `learn_one` immediately, on top of the normal retraining window. This is the module future work should focus on strengthening.

All three run prequential (test-then-train) evaluation over the same streamed CSV using `river` for online ML and PyTorch for the GAN, and are directly comparable via `compare/plot_comparison.py`.

## Running

There is no venv/requirements-installed environment inside this repo. The Python environment lives outside the project at `/mnt/AI-DATA-SSD/alara/main_env` (pinned versions match `requirements.txt`). Use it to run anything here:

```bash
/mnt/AI-DATA-SSD/alara/main_env/bin/python3 -m module_a_baseline.run
/mnt/AI-DATA-SSD/alara/main_env/bin/python3 -m module_b_pht.run
/mnt/AI-DATA-SSD/alara/main_env/bin/python3 -m module_c_gan.run
/mnt/AI-DATA-SSD/alara/main_env/bin/python3 -m module_c_gan.visualize      # standalone plot for module C
/mnt/AI-DATA-SSD/alara/main_env/bin/python3 -m compare.plot_comparison    # A vs B vs C, run after all three above
```

The system `python3` does **not** have `river` installed. Modules must be run with `-m` (or with the repo root on `PYTHONPATH`) since they import each other via absolute package paths (`common.*`, `module_c_gan.drift_gan`).

Each `run.py` prints progress to stdout and writes its snapshot metrics to `results/module_{a,b,c}.npz` (rolling F1, cumulative accuracy, per-interval CPU time, and — for B/C — drift sample indices). These `.npz`/`.png` outputs in `results/` are regenerable build artifacts and are gitignored (only `results/.gitkeep` is tracked) — run the modules to reproduce them rather than expecting them to be present after a fresh clone.

## Repo structure

```
common/             shared config, data streaming, metrics, and plot styling used by all three modules
  config.py          all tunable constants (dataset path, stream offsets, HAT/PHT hyperparams, snapshot intervals)
  streaming.py        preprocess_row() + stream_csv(): reads the CSV row-by-row as a generator (file is ~1.5GB, never loaded fully)
  metrics.py           RollingF1: binary F1 over a sliding window of predictions
  plotting.py           shared 3-panel (F1 / Accuracy / CPU time) matplotlib figure helpers
module_a_baseline/  Module A: static model
module_b_pht/       Module B: reactive PHT baseline
module_c_gan/       Module C: the proposal
  drift_gan.py         the DriftGAN class itself (generator/discriminator, train_and_generate())
  run.py                the PHT + DriftGAN evaluation loop
  visualize.py           standalone plot for this module's results
compare/            plot_comparison.py loads all three results/module_*.npz and plots them together
results/            output dir for .npz snapshots and .png plots (gitignored except .gitkeep)
```

**Editing workflow**: `common/` holds anything shared verbatim across modules (data loading, config constants, the rolling-F1 metric, plot styling) — if you're about to copy-paste logic between `module_a_baseline`, `module_b_pht`, and `module_c_gan`, put it in `common/` instead. Module-specific logic (the retraining-window state machine, the DriftGAN integration) stays local to each module's `run.py`.

## Core experiment mechanics (shared across all three modules)

- **Data source**: `/mnt/AI-DATA-SSD/alara/Data/CIC-IDS2017.csv` (external to this repo), streamed row-by-row via `common.streaming.stream_csv`.
- **Label binarization**: multiclass `Attack Type` is collapsed to binary `BENIGN` vs `ATTACK` in `preprocess_row`.
- **Stream offset**: reading starts at `START_ROW = 540_000` (`common/config.py`) to skip the long initial all-benign block, so the first ~40K streamed rows are benign (for pre-training) before attack traffic — and thus drift — appears.
- **Pre-training**: first `N_PRETRAIN = 50_000` streamed rows are used for `learn_one` calls with no evaluation; prequential evaluation (predict-then-optionally-learn) starts after that point.
- **Drift detection** (Modules B & C): `river.drift.PageHinkley(delta=0.002)` monitors the 0/1 prediction-error signal; on `drift_detected`, the model re-enters a training-enabled window of `TRAIN_WINDOW = 5000` samples.
- **DriftGAN** (Module C only): on each drift, if there are ≥5 buffered `ATTACK` samples in a `deque(maxlen=500)`, a small GAN (single-hidden-layer generator/discriminator, `noise_dim=16`) trains for `epochs=30` on those samples and generates 200 synthetic attack samples, immediately fed into the tree via `learn_one` before the normal retraining window kicks in. This is the mechanism to iterate on when strengthening the proposal (e.g. buffer size, GAN capacity/epochs, number of synthetic samples, how synthetic samples are weighted vs. real ones).
- **Metrics**: rolling F1 (`RollingF1`, binary, window=5000, positive label `ATTACK`) and cumulative `river.metrics.Accuracy`, snapshotted every `SNAP_INTERVAL = 500` samples along with per-interval wall-clock processing time (`time.perf_counter`).

## Conventions

- Narrative text (print statements, plot labels, docstrings) is in Spanish; keep new content consistent with this.
- All tunable experiment parameters live in `common/config.py` — avoid hardcoding constants inside module `run.py` files.
