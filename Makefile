SEEDS ?= 0,1,2,3,4,5,6,7,8,9
N_JOBS ?= 5
NJOBS ?= $(N_JOBS)
SCHEDULE ?= results/drift_schedule.json

.PHONY: setup test lint typecheck smoke schedule report clean-results \
        run-reactive run-pace run-freq run-sota run-sota-matched run-early-ensemble \
        run-ensemble-controls \
        run-comparators run-anchor schedule-2018 schedule-unsw run-2018 run-unsw \
        analyse figures

setup:
	uv sync --dev

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts run_experiment.py

typecheck:
	uv run mypy

# Quick end-to-end sanity check against the real CSV. Seed 99 is deliberate: the
# 2017 anchor writes to flat paths, so a throwaway run on an experiment seed
# would silently overwrite a real multi-hour result.
smoke:
	uv run python run_experiment.py --phase B --seeds 99 --max-samples 10000

# Pass 0: build the shared drift schedule from the reactive baseline's dynamics.
schedule:
	uv run python run_experiment.py --build-schedule --out $(SCHEDULE)

# ── CIC-IDS2017 anchor ────────────────────────────────────────────────
# Reactive baselines: adaptation driven by the classification error.
run-reactive:
	uv run python run_experiment.py --phase B --seeds $(SEEDS) --n-jobs $(N_JOBS) --drift-schedule $(SCHEDULE)
	uv run python run_experiment.py --phase B_adwin --seeds $(SEEDS) --n-jobs $(N_JOBS)
	uv run python run_experiment.py --phase B_kswin --seeds $(SEEDS) --n-jobs $(N_JOBS)

# PACE. Always dynamic: its contribution is anticipation, which a fixed
# schedule would erase.
run-pace:
	uv run python run_experiment.py --phase C_early --seeds $(SEEDS) --n-jobs $(N_JOBS)

# Frequency-matched control: blind periodic clock at PACE's fire count.
run-freq:
	uv run python run_experiment.py --phase C_freq --seeds $(SEEDS) --n-jobs $(N_JOBS)

# Streaming ensembles learning from every sample (upper reference).
run-sota:
	uv run python run_experiment.py --phase ARF --seeds $(SEEDS) --n-jobs $(N_JOBS)
	uv run python run_experiment.py --phase SRP --seeds $(SEEDS) --n-jobs $(N_JOBS)

# The same ensembles at the reactive arm's label budget.
run-sota-matched:
	uv run python run_experiment.py --phase ARF_sched --seeds $(SEEDS) --n-jobs $(N_JOBS) --drift-schedule $(SCHEDULE)
	uv run python run_experiment.py --phase SRP_sched --seeds $(SEEDS) --n-jobs $(N_JOBS) --drift-schedule $(SCHEDULE)

# PACE on the ensembles: does the trigger help the strongest models too?
run-early-ensemble:
	uv run python run_experiment.py --phase ARF_early --seeds $(SEEDS) --n-jobs $(N_JOBS)
	uv run python run_experiment.py --phase SRP_early --seeds $(SEEDS) --n-jobs $(N_JOBS)

# Controls for the ensembles: a blind clock at ARF/SRP_early's fire count, and the
# error Page-Hinkley run live on each ensemble's own predictions (no schedule).
run-ensemble-controls:
	@for p in ARF_freq SRP_freq ARF_pht SRP_pht; do \
		uv run python run_experiment.py --phase $$p --seeds $(SEEDS) --n-jobs $(N_JOBS); done

# Monitor ablation and published comparators, all fire-count matched to the
# autoencoder. Run to 20 seeds: the collapse outcome is bimodal.
COMPARATORS = C_early_pca C_early_mean C_early_hdddm C_early_d3 C_early_d3_native C_early_d3_sized
run-comparators:
	@for p in $(COMPARATORS); do \
		uv run python run_experiment.py --phase $$p \
			--seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
			--n-jobs 10; done

run-anchor: schedule run-reactive run-pace run-freq run-sota run-sota-matched run-early-ensemble \
            run-ensemble-controls

# ── External-validity datasets: CIC-IDS2018 + UNSW-NB15 ───────────────
# Each dataset pins its own calibrated monitor threshold and blind-clock
# interval in configs/datasets/<name>.yaml — never pass those as CLI flags.
DATA2018 ?= cicids2018
DATAUNSW ?= unsw
DYNPHASES = C_early ARF_early SRP_early C_freq ARF_freq SRP_freq ARF_pht SRP_pht
SCHEDPHASES = B ARF_sched SRP_sched

schedule-2018:
	uv run python run_experiment.py --build-schedule --dataset $(DATA2018) \
		--config configs/phase_b.yaml --out results/drift_schedule_$(DATA2018).json

schedule-unsw:
	uv run python run_experiment.py --build-schedule --dataset $(DATAUNSW) \
		--config configs/phase_b.yaml --out results/drift_schedule_$(DATAUNSW).json

run-2018:
	@for p in $(SCHEDPHASES); do \
		uv run python run_experiment.py --phase $$p --dataset $(DATA2018) \
			--drift-schedule results/drift_schedule_$(DATA2018).json \
			--seeds $(SEEDS) --n-jobs $(NJOBS); done
	@for p in $(DYNPHASES); do \
		uv run python run_experiment.py --phase $$p --dataset $(DATA2018) \
			--seeds $(SEEDS) --n-jobs $(NJOBS); done

run-unsw:
	@for p in $(SCHEDPHASES); do \
		uv run python run_experiment.py --phase $$p --dataset $(DATAUNSW) \
			--drift-schedule results/drift_schedule_$(DATAUNSW).json \
			--seeds $(SEEDS) --n-jobs $(NJOBS); done
	@for p in $(DYNPHASES); do \
		uv run python run_experiment.py --phase $$p --dataset $(DATAUNSW) \
			--seeds $(SEEDS) --n-jobs $(NJOBS); done

# ── Analysis (loads persisted artifacts; never re-runs the stream) ─────
analyse:
	uv run python scripts/tier0_analysis.py
	uv run python scripts/tier1_analysis.py
	uv run python scripts/comparator_analysis.py

figures:
	uv run python scripts/paper_figures.py
	uv run python scripts/reviewer_analysis.py

report:
	uv run python scripts/make_report.py

clean-results:
	rm -rf results/runs results/recovery results/report results/plots
