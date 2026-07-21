SEEDS ?= 0,1,2,3,4
N_JOBS ?= 5
SCHEDULE ?= results/drift_schedule.json

.PHONY: setup test lint typecheck smoke schedule run-A run-B run-C run-C-prime run-C-cond run-C-early run-all report clean-results

setup:
	uv sync --dev

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts run_experiment.py

typecheck:
	uv run mypy

# Quick end-to-end sanity check against the real CSV (10K samples, 1 seed).
smoke:
	uv run python run_experiment.py --phase B --seeds 0 --max-samples 10000

# Pass 0: build the shared drift schedule from Phase B dynamics.
schedule:
	uv run python run_experiment.py --build-schedule --out $(SCHEDULE)

run-A:
	uv run python run_experiment.py --phase A --seeds $(SEEDS) --n-jobs $(N_JOBS)

run-B:
	uv run python run_experiment.py --phase B --seeds $(SEEDS) --n-jobs $(N_JOBS) --drift-schedule $(SCHEDULE)

run-C:
	uv run python run_experiment.py --phase C --seeds $(SEEDS) --n-jobs $(N_JOBS) --drift-schedule $(SCHEDULE)

run-C-prime:
	uv run python run_experiment.py --phase C_prime --seeds $(SEEDS) --n-jobs $(N_JOBS) --drift-schedule $(SCHEDULE)

run-C-cond:
	uv run python run_experiment.py --phase C_cond --seeds $(SEEDS) --n-jobs $(N_JOBS) --drift-schedule $(SCHEDULE)

# C_early always runs dynamically: its contribution is anticipation.
run-C-early:
	uv run python run_experiment.py --phase C_early --seeds $(SEEDS) --n-jobs $(N_JOBS)

# C_freq: frequency-matched control for C_early (blind periodic clock, no schedule).
run-C-freq:
	uv run python run_experiment.py --phase C_freq --seeds $(SEEDS) --n-jobs $(N_JOBS)

# SOTA baselines (Experiment #3): alternative detectors + self-adapting ensembles.
run-sota:
	uv run python run_experiment.py --phase B_adwin --seeds $(SEEDS) --n-jobs $(N_JOBS)
	uv run python run_experiment.py --phase B_kswin --seeds $(SEEDS) --n-jobs $(N_JOBS)
	uv run python run_experiment.py --phase ARF --seeds $(SEEDS) --n-jobs $(N_JOBS)
	uv run python run_experiment.py --phase SRP --seeds $(SEEDS) --n-jobs $(N_JOBS)

run-all: schedule run-A run-B run-C run-C-prime run-C-cond run-C-freq run-C-early

report:
	uv run python scripts/make_report.py

clean-results:
	rm -rf results/runs results/recovery results/report results/plots
