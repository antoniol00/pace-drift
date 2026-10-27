# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**PACE Drift** (repo `pace-drift`) — a concept-drift research project for streaming network intrusion detection. **PACE** = *Proactive Adaptation via Covariate-shift Estimation*: an unsupervised monitor over incoming traffic triggers adaptation *before* classification error rises, with learning gated to short triggered windows (`C_early`, `ARF_early`, `SRP_early`). The research contribution is the **when-to-adapt policy**, validated on CICIDS2017 + CIC-IDS2018 + UNSW-NB15 × HAT/ARF/SRP against budget-matched controls that separate adaptation *timing* from *frequency* (`*_freq`, blind periodic clock), from *model capacity* (`*_sched`) and from the error test PACE runs alongside its monitor (`*_pht`, that error test run live on the learner's own predictions). The last control bounds the claim: informed placement beats blind placement (timing +0.042 [+0.008, +0.077] over the nine cells), but against `*_pht` the monitor's own contribution is **not resolved** (+0.021 [−0.002, +0.045]) — on real traffic with immediate labels the informing comes mostly from the live error test. Never claim the monitor beats a live error test. A second contribution is **closed-loop stability as the criterion for trigger quality**: at matched fire count over 20 seeds *every* trigger tested — ours, the published D3 and HDDDM, cheap PCA and mean-shift — collapses the learner into a high-false-positive regime on 20–60% of runs, through a verified training-distribution mechanism that detection metrics and final-window F1 both hide. Note what this does **not** say: our monitor is *not* measurably more stable than the published detectors. Never quote a collapse-count asymmetry from fewer than 20 seeds; the outcome is bimodal.

Note the Python package is `driftgan` — a historical name kept deliberately for import stability, not a leftover: `src/driftgan/` paths in docs are correct.

Start with docs/paper/idea-central.md (what is claimed and on what evidence), then README.md, docs/methodology.md (protocol), docs/paper/related-work.md (positioning + citation verification status) and docs/experiments/ (per-experiment results, numbered 00–08).

## Environment and commands

The project uses **uv** with its own `.venv` (do not use the shared `/mnt/AI-DATA-SSD/alara/main_env` for this repo). `uv` is at `~/.local/bin/uv` — export `PATH="$HOME/.local/bin:$PATH"` if it's not found.

```bash
make setup        # uv sync --dev
make test         # pytest (fast: unit + synthetic-stream e2e, no real CSV needed)
make lint         # ruff check src tests scripts run_experiment.py
make typecheck    # mypy (must stay clean; scoped to the driftgan package only)
make smoke        # 10K-sample run against the real CSV — see the hazard below
make report       # aggregate tables + figures into results/report/ (2017 anchor only)

uv run pytest tests/test_recovery.py             # single file
uv run pytest tests/test_early.py -k dual_pht    # single test
uv run python run_experiment.py --phase C --seeds 0,1 --n-jobs 2 --max-samples 100000
make run-anchor                                  # full 2017 experiment (hours)
```

CI (`.github/workflows/ci.yml`) runs lint + typecheck + tests on every push and PR, so all three must pass before committing.

`--max-samples` below `n_pretrain` auto-shrinks pretraining (see `_fit_pretrain` in run_experiment.py). `--n-jobs > 1` uses the `spawn` start method (fork is unsafe with torch/CUDA); `device: cuda` falls back to CPU automatically.

**Multiple datasets.** `--dataset <name>` loads `configs/datasets/<name>.yaml` (`cicids2018`, `unsw`; `cicids2017` is the implicit anchor with no profile file). Each external dataset needs its own schedule (`make schedule-2018` / `schedule-unsw`) before the scheduled phases, and pins its own calibrated monitor thresholds — see the calibration discipline note below.

## Results layout, and the overwrite hazard

Runs write `results/runs/[<dataset>/]<phase>/seed_<n>/` — the dataset segment is **omitted for the 2017 anchor** (`ExperimentConfig.run_dir` / `_ns_segments`), so 2017 results sit at flat paths like `results/runs/B/seed_0/`. Recovery frames use a different scheme: a flat `results/recovery/` with dataset-*prefixed* filenames (`unsw_C_early_0.parquet`, anchor unprefixed). Each run dir holds `manifest.json` (git/CSV/config hashes, package versions), `snapshots.parquet`, `drift_events.json`, `cost.json`, `trace.csv.gz`, `summary.json`.

**`results/` is gitignored and holds multi-hour runs that cannot be regenerated cheaply.** Because anchor paths are flat, any quick 2017 run silently overwrites the real result at the same (phase, seed) — `make smoke` targets `B/seed_0`, which is a real 2M-sample run (this has already had to be restored once). For throwaway runs on the anchor, always pass a seed outside the experiment range:

```bash
uv run python run_experiment.py --phase B --seeds 99 --max-samples 10000
```

## Architecture (src/driftgan/)

One prequential engine runs every phase: `experiments/runner.py:run_phase` differs per phase only in **trigger** (PHT / shared schedule / ADWIN / KSWIN / blind periodic / dual recon+class PHT / live PHT on the learner's own error for `B_pht`/`ARF_pht`/`SRP_pht` / continuous) and **monitor** (none / AE / PCA / KS / MMD / mean-shift / HDDDM / D3). When adding a strategy, extend `_make_monitor` and the trigger selection there — do not fork the loop. A new phase name must be registered in **three** places or it fails late: `config.py:VALID_PHASES`, `run_experiment.py:PHASE_CONFIG`, and the trigger branch in `runner.py:run_phase` (plus a `configs/phase_<name>.yaml`).

Load-bearing design decisions (details in docs/methodology.md):

- **Shared drift schedule** (`drift/schedule.py`): scheduled phases replay identical adaptation indices from a pass-0 JSON, so per-event metrics can be paired across phases. A report-only PHT still runs for delay measurement. The proactive arms are deliberately exempt (their claim is anticipation). A scheduled index the stream skips is dropped, never fired late. The schedule comes from one live **HAT** pass at seed 0 per dataset, so `ARF_sched`/`SRP_sched` replay HAT's error, not their own, and `B` at seeds 1–9 replays another run's; `B_pht`/`ARF_pht`/`SRP_pht` are the live counterparts, and `*_early − *_pht` is what the monitor adds.
- **A firing count is not a label count.** Each firing opens a `train_window` (5,000) and a firing inside an open window restarts it, so clustered firings share labels and a blind clock (never overlapping) consumes the most per firing. Report the labelled share alongside any firing count (`metrics/cost.py:labelled_samples`, used by `tier1_analysis.py`).
- **Recovery metrics** (`metrics/recovery.py`): snapshots carry a `valid` flag — windows with no positive true label have undefined F1 and are excluded from RT/AUD, not counted as failure. Overlapping drifts close the previous event as `no_recovery`. The unit tests encode the analytic semantics; change behavior only with matching test changes.
- **Cost accounting** (`metrics/cost.py`): per-sample latency + named sections (`hat_inference`, `hat_update`, `monitor`, `other`).
- **Config** (`config.py`): every tunable lives in `ExperimentConfig`; `load_config` merges base.yaml ← phase yaml ← **dataset yaml** ← CLI overrides — the dataset layer overrides the phase layer, because per-dataset calibration must win. Never hardcode experiment constants in modules.
- **Calibration discipline** (`early/`): each monitor's PHT threshold is calibrated per dataset so that arms differ in the *signal*, not in how often they fire; the calibrated values are pinned in the dataset profile with the evidence in a comment. A new monitor or dataset needs the same treatment, or comparisons become fire-count artifacts. `scripts/calibrate_monitor.py` does the sweep offline against a cached score series (one run, then any threshold replays for free). Values that depend on the *(dataset, phase)* pair rather than the dataset alone — the `*_freq` blind-clock interval, calibrated per learner — go in the profile's `by_phase:` block, which `load_config` applies after the dataset layer. **Never pass a calibrated value as a bare CLI flag:** doing so kept UNSW's `*_freq` interval out of every config, and a later batch of seeds silently ran de-matched (142 fires instead of 172) and had to be discarded.

## Analysis and figures

Analyses load the persisted parquets/JSON — they never re-run the stream. `scripts/tier0_analysis.py` (paired CIs, Kappa, event stratification) · `tier1_analysis.py` (timing vs frequency) · `comparator_analysis.py` (monitor comparison + closed-loop stability) · `monitor_ablation.py` · `reviewer_analysis.py` (fig4/fig5) · `paper_figures.py` (fig1, 3, 6, 8-13, 15) · `make_report.py` (2017 anchor tables) · `scan_drift_window.py` (dataset window selection) · `calibrate_monitor.py` (offline PHT fire-count calibration) · `synthetic_timing.py` (the controlled-stream check of the core claim).

Figures are versioned in `docs/figures/` even though `results/` is not — regenerate and commit them there when the underlying numbers change, and keep the write-up in `docs/experiments/` in sync.

## Conventions

- Everything in English. Structured JSON logging via `driftgan.logging` (no bare prints in library code; CLI prints are fine).
- ruff + mypy must pass (`make lint typecheck`); type hints everywhere; NumPy-style docstrings on public functions.
- Analysis lives in `scripts/` (load the persisted parquets/JSON and emit tables + figures); pipelines live in `src/`. Do not introduce notebooks for reported results.
- Report only what the data supports, and state refusals to claim as clearly as claims. When numbers change, update the claim rather than dropping it. Collapse-count comparisons are held to n >= 20 — that outcome is bimodal and a handful of replicates estimates its rate badly.
- The CIC-IDS2017 CSV lives outside the repo at `/mnt/AI-DATA-SSD/alara/Data/CIC-IDS2017.csv` (`configs/base.yaml:csv_path`).
