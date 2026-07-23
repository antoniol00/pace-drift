SEEDS ?= 0,1,2,3,4
N_JOBS ?= 5
NJOBS ?= $(N_JOBS)
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

# Budget-matched SOTA (#3b): ARF/SRP adapting only in triggered windows on the schedule.
run-sota-matched:
	uv run python run_experiment.py --phase ARF_sched --seeds $(SEEDS) --n-jobs $(N_JOBS) --drift-schedule $(SCHEDULE)
	uv run python run_experiment.py --phase SRP_sched --seeds $(SEEDS) --n-jobs $(N_JOBS) --drift-schedule $(SCHEDULE)

# Ensemble + proactive early-detection trigger (#3c): does anticipation help the SOTA model?
run-early-ensemble:
	uv run python run_experiment.py --phase ARF_early --seeds $(SEEDS) --n-jobs $(N_JOBS)
	uv run python run_experiment.py --phase SRP_early --seeds $(SEEDS) --n-jobs $(N_JOBS)

run-all: schedule run-A run-B run-C run-C-prime run-C-cond run-C-freq run-C-early

report:
	uv run python scripts/make_report.py

clean-results:
	rm -rf results/runs results/recovery results/report results/plots

# ── External-validity datasets (#4): 2018 + UNSW ──────────────────────
DATA2018 ?= cicids2018
DATAUNSW ?= unsw
MINPHASES = C_early ARF_early SRP_early
SCHEDPHASES = B ARF_sched SRP_sched

schedule-2018:
	uv run python run_experiment.py --build-schedule --dataset $(DATA2018) \
		--config configs/phase_b.yaml --out results/drift_schedule_$(DATA2018).json

schedule-unsw:
	uv run python run_experiment.py --build-schedule --dataset $(DATAUNSW) \
		--config configs/phase_b.yaml --out results/drift_schedule_$(DATAUNSW).json

run-min-2018:
	@for p in $(SCHEDPHASES); do \
		uv run python run_experiment.py --phase $$p --dataset $(DATA2018) \
			--drift-schedule results/drift_schedule_$(DATA2018).json \
			--seeds $(SEEDS) --n-jobs $(NJOBS); done
	@for p in $(MINPHASES); do \
		uv run python run_experiment.py --phase $$p --dataset $(DATA2018) \
			--seeds $(SEEDS) --n-jobs $(NJOBS); done

run-min-unsw:
	@for p in $(SCHEDPHASES); do \
		uv run python run_experiment.py --phase $$p --dataset $(DATAUNSW) \
			--drift-schedule results/drift_schedule_$(DATAUNSW).json \
			--seeds $(SEEDS) --n-jobs $(NJOBS); done
	@for p in $(MINPHASES); do \
		uv run python run_experiment.py --phase $$p --dataset $(DATAUNSW) \
			--seeds $(SEEDS) --n-jobs $(NJOBS); done
