# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**PACE Drift** (repo `pace-drift`) — a concept-drift research project for streaming network intrusion detection. **PACE** = *Proactive Adaptation via Covariate-shift Estimation*: an unsupervised monitor over incoming traffic triggers adaptation *before* classification error rises, with learning gated to short triggered windows (`C_early`, `ARF_early`, `SRP_early`). The research contribution is the **when-to-adapt policy**, validated on CICIDS2017 + CIC-IDS2018 + UNSW-NB15 × HAT/ARF/SRP against budget-matched controls that separate adaptation *timing* from *frequency* (`*_freq`, blind periodic clock) and from *model capacity* (`*_sched`).

The project began as *DriftGAN* (generating synthetic attack samples at drift time, phases C/C_cond); under budget-matched controls that augmentation gave no measurable benefit, and the code path is kept only so the controlled refutation stays reproducible. **Do not reframe the project around the generative angle.** Note the Python package is still `driftgan` — that is deliberate (import stability), not a leftover: `src/driftgan/` paths in docs are correct.

Start with docs/paper/idea-central.md (the hypothesis→findings trail), then README.md, docs/methodology.md (protocol) and docs/experiments/ (per-experiment results).

## Environment and commands

The project uses **uv** with its own `.venv` (do not use the shared `/mnt/AI-DATA-SSD/alara/main_env` for this repo). `uv` is at `~/.local/bin/uv` — export `PATH="$HOME/.local/bin:$PATH"` if it's not found.

```bash
make setup        # uv sync --dev
make test         # pytest (fast: unit + synthetic-stream e2e, no real CSV needed)
make lint         # ruff check
make typecheck    # mypy (must stay clean)
make smoke        # 10K-sample run against the real CSV
uv run pytest tests/test_recovery.py            # single test file
uv run python run_experiment.py --phase C --seeds 0,1 --n-jobs 2 --max-samples 100000
make schedule && make run-all SEEDS=0,1,2,3,4   # full experiment (hours)
make report       # aggregate tables + figures into results/report/
```

`--max-samples` below `n_pretrain` auto-shrinks pretraining (see `_fit_pretrain` in run_experiment.py).

## Architecture (src/driftgan/)

One prequential engine runs every phase: `experiments/runner.py:run_phase` differs per phase only in **trigger** (none / PHT / shared schedule / dual recon+class PHT) and **augmenter** (none / vanilla GAN / jitter / conditional WGAN-GP). When adding a phase or strategy, extend `_make_augmenter` and the trigger selection there — do not fork the loop.

Load-bearing design decisions (details in docs/methodology.md):

- **Shared drift schedule** (`drift/schedule.py`): scheduled phases replay identical adaptation indices from a pass-0 JSON, so per-event metrics can be paired across phases. A report-only PHT still runs for delay measurement. C_early is deliberately exempt (its claim is anticipation). A scheduled index the stream skips is dropped, never fired late.
- **Recovery metrics** (`metrics/recovery.py`): snapshots carry a `valid` flag — windows with no positive true label have undefined F1 and are excluded from RT/AUD, not counted as failure. Overlapping drifts close the previous event as `no_recovery`. The unit tests encode the analytic semantics; change behavior only with matching test changes.
- **Cost accounting** (`metrics/cost.py`): per-sample latency + named sections; augmenters self-report `t_train`/`t_generate` attributes which the runner credits via `tracker.add` — new augmenters must set both.
- **Config** (`config.py`): every tunable lives in `ExperimentConfig`; YAMLs in `configs/` merge base.yaml ← phase yaml ← CLI overrides. Never hardcode experiment constants in modules.
- Runs write `results/runs/<phase>/seed_<n>/` (manifest with git/CSV/config hashes, snapshots parquet, cost, trace) and `results/recovery/<phase>_<seed>.parquet`. `results/` is gitignored.

## Conventions

- Everything in English. Structured JSON logging via `driftgan.logging` (no bare prints in library code; CLI prints are fine).
- ruff + mypy must pass (`make lint typecheck`); type hints everywhere; NumPy-style docstrings on public functions.
- Analysis lives in `scripts/` (load the persisted parquets/JSON and emit tables + figures); pipelines live in `src/`. The exploratory notebooks were removed in the 2026-07-25 cleanup — do not reintroduce notebooks for reported results.
- The legacy PoC modules (`module_*`, `compare/`, `common/`, `requirements.txt`) have been removed; all work lives in `src/driftgan/` with dependencies pinned in `pyproject.toml` + `uv.lock`.
- The CIC-IDS2017 CSV lives outside the repo at `/mnt/AI-DATA-SSD/alara/Data/CIC-IDS2017.csv` (`configs/base.yaml:csv_path`).
