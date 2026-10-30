"""Multi-seed experiment CLI (Improvement 4).

Examples
--------
Build the shared drift schedule (pass 0, Phase B dynamics)::

    uv run python run_experiment.py --build-schedule --config configs/phase_b.yaml \
        --out results/drift_schedule.json

Run a phase over 5 seeds with 4 parallel workers::

    uv run python run_experiment.py --phase C --seeds 0,1,2,3,4 --n-jobs 4 \
        --drift-schedule results/drift_schedule.json
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Any

from driftgan.config import CONFIGS_DIR, ExperimentConfig, load_config
from driftgan.logging import get_logger

logger = get_logger("driftgan.cli")

PHASE_CONFIG = {
    "B": "phase_b.yaml",
    "C_early": "phase_c_early.yaml",
    "C_freq": "phase_c_freq.yaml",
    "B_adwin": "phase_b_adwin.yaml",
    "B_kswin": "phase_b_kswin.yaml",
    "ARF": "phase_arf.yaml",
    "SRP": "phase_srp.yaml",
    "C_early_mean": "phase_c_early_mean.yaml",
    "C_early_pca": "phase_c_early_pca.yaml",
    "C_early_d3": "phase_c_early_d3.yaml",
    "C_early_d3_native": "phase_c_early_d3_native.yaml",
    "C_early_d3_sized": "phase_c_early_d3_sized.yaml",
    "C_early_hdddm": "phase_c_early_hdddm.yaml",
    "ARF_sched": "phase_arf_sched.yaml",
    "SRP_sched": "phase_srp_sched.yaml",
    "B_pht": "phase_b_pht.yaml",
    "ARF_pht": "phase_arf_pht.yaml",
    "SRP_pht": "phase_srp_pht.yaml",
    "ARF_early": "phase_arf_early.yaml",
    "SRP_early": "phase_srp_early.yaml",
    "ARF_freq": "phase_arf_freq.yaml",
    "SRP_freq": "phase_srp_freq.yaml",
    "C_early_adv": "phase_c_early_adv.yaml",
    "C_early_advb": "phase_c_early_advb.yaml",
    "C_early_chaff": "phase_c_early_chaff.yaml",
    "B_pht_chaff": "phase_b_pht_chaff.yaml",
    "C_early_rb": "phase_c_early_rb.yaml",
    "B_pht_rb": "phase_b_pht_rb.yaml",
    "C_early_adv_rb": "phase_c_early_adv_rb.yaml",
    "C_early_chaff_rb": "phase_c_early_chaff_rb.yaml",
    "C_early_advb_rb": "phase_c_early_advb_rb.yaml",
    "C_freq_rb": "phase_c_freq_rb.yaml",
    "ARF_early_rb": "phase_arf_early_rb.yaml",
    "SRP_early_rb": "phase_srp_early_rb.yaml",
    "B_pht_chaff_rb": "phase_b_pht_chaff_rb.yaml",
    "C_early_rb50": "phase_c_early_rb50.yaml",
    "C_early_chaff_rb50": "phase_c_early_chaff_rb50.yaml",
    "C_early_d1k": "phase_c_early_d1k.yaml",
    "B_pht_d1k": "phase_b_pht_d1k.yaml",
    "C_freq_d1k": "phase_c_freq_d1k.yaml",
    "ARF_early_d1k": "phase_arf_early_d1k.yaml",
    "ARF_pht_d1k": "phase_arf_pht_d1k.yaml",
    "ARF_freq_d1k": "phase_arf_freq_d1k.yaml",
    "C_freq_lab_d1k": "phase_c_freq_lab_d1k.yaml",
    "ARF_freq_lab_d1k": "phase_arf_freq_lab_d1k.yaml",
    "C_early_d5k": "phase_c_early_d5k.yaml",
    "B_pht_d5k": "phase_b_pht_d5k.yaml",
    "C_freq_d5k": "phase_c_freq_d5k.yaml",
    "ARF_early_d5k": "phase_arf_early_d5k.yaml",
    "ARF_pht_d5k": "phase_arf_pht_d5k.yaml",
    "ARF_freq_d5k": "phase_arf_freq_d5k.yaml",
    "C_early_d50k": "phase_c_early_d50k.yaml",
    "B_pht_d50k": "phase_b_pht_d50k.yaml",
    "C_freq_d50k": "phase_c_freq_d50k.yaml",
    "ARF_early_d50k": "phase_arf_early_d50k.yaml",
    "ARF_pht_d50k": "phase_arf_pht_d50k.yaml",
    "ARF_freq_d50k": "phase_arf_freq_d50k.yaml",
    "C_freq_lab": "phase_c_freq_lab.yaml",
    "C_freq_lab_d5k": "phase_c_freq_lab_d5k.yaml",
    "C_freq_lab_d50k": "phase_c_freq_lab_d50k.yaml",
    "ARF_freq_lab": "phase_arf_freq_lab.yaml",
    "ARF_freq_lab_d5k": "phase_arf_freq_lab_d5k.yaml",
    "ARF_freq_lab_d50k": "phase_arf_freq_lab_d50k.yaml",
    "LR_early": "phase_lr_early.yaml",
    "LR_pht": "phase_lr_pht.yaml",
    "LR_freq": "phase_lr_freq.yaml",
    "LR_early_rb": "phase_lr_early_rb.yaml",
    "LR_early_chaff": "phase_lr_early_chaff.yaml",
    "LR_early_chaff_rb": "phase_lr_early_chaff_rb.yaml",
}


def _fit_pretrain(config: ExperimentConfig) -> ExperimentConfig:
    """Short runs must still leave room for evaluation after pretraining."""
    if config.max_samples <= config.n_pretrain:
        config.n_pretrain = max(config.max_samples // 5, 100)
        print(f"note: n_pretrain reduced to {config.n_pretrain} to fit max_samples")
    return config


def _worker(args: tuple[ExperimentConfig, int]) -> dict[str, Any]:
    config, seed = args
    from driftgan.experiments.runner import run_phase

    result = run_phase(config, seed=seed)
    return {
        "phase": result.phase,
        "seed": seed,
        "final_acc": result.final_acc,
        "final_f1": result.final_f1,
        "n_drifts": len(result.drift_indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=sorted(PHASE_CONFIG), help="Phase to run")
    parser.add_argument("--config", type=Path, help="Config YAML (default: configs/phase_<phase>.yaml)")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds (default: 0-4)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel seed workers")
    parser.add_argument("--drift-schedule", type=Path, help="Shared drift schedule JSON (overrides dynamic PHT)")
    parser.add_argument("--max-samples", type=int, help="Override max_samples (quick runs)")
    parser.add_argument("--periodic-adapt-interval", type=int,
                        help="Override periodic_adapt_interval (frequency-matched *_freq controls)")
    parser.add_argument("--build-schedule", action="store_true",
                        help="Run pass 0 (Phase B dynamics) and write the schedule to --out")
    parser.add_argument("--out", type=Path, help="Output path for --build-schedule")
    parser.add_argument("--dataset", default="cicids2017",
                        help="Dataset profile in configs/datasets/<name>.yaml (default: cicids2017)")
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.drift_schedule:
        overrides["drift_schedule"] = str(args.drift_schedule)
    if args.max_samples:
        overrides["max_samples"] = args.max_samples
    if args.periodic_adapt_interval:
        overrides["periodic_adapt_interval"] = args.periodic_adapt_interval

    if args.build_schedule:
        config_path = args.config or CONFIGS_DIR / "phase_b.yaml"
        config = _fit_pretrain(load_config(config_path, overrides, dataset=args.dataset))
        out = args.out or Path(config.results_dir) / "drift_schedule.json"
        from driftgan.drift.schedule import build_schedule_from_pht

        schedule = build_schedule_from_pht(config, seed=int(args.seeds.split(",")[0]))
        schedule.save(out)
        print(f"schedule with {len(schedule.indices)} events written to {out}")
        return

    if not args.phase:
        parser.error("--phase is required unless --build-schedule is given")
    config_path = args.config or CONFIGS_DIR / PHASE_CONFIG[args.phase]
    config = _fit_pretrain(load_config(config_path, overrides, dataset=args.dataset))
    if config.phase != args.phase:
        parser.error(f"config {config_path} declares phase {config.phase!r}, but --phase is {args.phase!r}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    jobs = [(config, seed) for seed in seeds]

    if args.n_jobs > 1:
        ctx = mp.get_context("spawn")  # fork is unsafe with torch/CUDA
        with ctx.Pool(processes=args.n_jobs) as pool:
            summaries = pool.map(_worker, jobs)
    else:
        summaries = [_worker(job) for job in jobs]

    print(f"\n{'phase':<10}{'seed':<6}{'final_acc':<12}{'final_f1':<12}{'n_drifts':<9}")
    for s in summaries:
        print(f"{s['phase']:<10}{s['seed']:<6}{s['final_acc']:<12.4f}{s['final_f1']:<12.4f}{s['n_drifts']:<9}")


if __name__ == "__main__":
    main()
