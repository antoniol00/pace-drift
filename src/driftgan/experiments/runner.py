"""Unified prequential evaluation engine for all phases.

One loop implements every phase; the differences are confined to (a) which
trigger fires adaptation and (b) which augmenter (if any) generates
synthetic samples at drift time:

=========  ==========================  ==============================
Phase      Trigger                     Augmenter
=========  ==========================  ==============================
A          none (frozen after pretrain) none
B          PHT / shared schedule        none
C          PHT / shared schedule        VanillaGanAugmenter
C_prime    PHT / shared schedule        BootstrapJitterAugmenter
C_cond     PHT / shared schedule        ConditionalGanAugmenter
C_freq     periodic clock (blind)       none
C_early    DualPHT (recon + class)      optional (config.early_augment)
=========  ==========================  ==============================

C_early always runs dynamically: its contribution is *anticipation*, which
a fixed schedule would erase (see docs/methodology.md). C_freq is the
frequency-matched control: it fires the same number of times as C_early but
on a blind periodic clock, isolating adaptation frequency from timing.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from river import drift as river_drift
from river import metrics as river_metrics

from driftgan.augment.base import Buffer
from driftgan.config import ExperimentConfig
from driftgan.data.stream import ATTACK, BENIGN, stream_csv
from driftgan.drift.schedule import (
    DynamicDetectorTrigger,
    DynamicPHTTrigger,
    PeriodicTrigger,
    ScheduledDriftTrigger,
    load_schedule,
)
from driftgan.early.dual_pht import DualPHT, match_lead_times
from driftgan.early.reconstruction_monitor import ReconstructionMonitor
from driftgan.early.shift_monitors import MONITOR_TYPES, ShiftMonitor
from driftgan.experiments.manifest import build_manifest, save_manifest
from driftgan.experiments.seeding import seed_everything
from driftgan.logging import get_logger, log_ctx
from driftgan.metrics.cost import CostTracker
from driftgan.metrics.recovery import compute_recovery_events, save_recovery
from driftgan.metrics.rolling import RollingBinaryStats
from driftgan.models.streaming import CONTINUOUS_PHASES, make_model

logger = get_logger("driftgan.runner")


@dataclass
class RunResult:
    """In-memory summary of one (phase, seed) run."""

    phase: str
    seed: int
    n_processed: int
    final_acc: float
    final_f1: float
    snap_indices: np.ndarray
    snap_f1: np.ndarray
    snap_acc: np.ndarray
    snap_valid: np.ndarray
    snap_interval_ms: np.ndarray
    drift_indices: list[int]
    drift_events: list[dict[str, Any]]
    report_pht_indices: list[int] = field(default_factory=list)
    lead_time_report: dict[str, Any] | None = None
    cost_summary: dict[str, Any] = field(default_factory=dict)
    label_dist: dict[str, int] = field(default_factory=dict)


def _make_augmenter(phase: str, config: ExperimentConfig, seed: int) -> Any:
    if phase == "C":
        from driftgan.augment.vanilla_gan import VanillaGanAugmenter

        return VanillaGanAugmenter(
            noise_dim=config.gan_noise_dim,
            epochs=config.gan_epochs,
            lr=config.gan_lr,
            device=config.device,
            seed=seed,
        )
    if phase == "C_prime":
        from driftgan.augment.bootstrap_jitter import BootstrapJitterAugmenter

        return BootstrapJitterAugmenter(sigma_scale=config.jitter_sigma_scale, seed=seed)
    if phase == "C_cond":
        from driftgan.augment.cgan import ConditionalGanAugmenter

        return ConditionalGanAugmenter(
            noise_dim=config.gan_noise_dim,
            epochs=config.gan_epochs,
            lr=config.gan_lr,
            n_critic=config.cgan_n_critic,
            gp_lambda=config.cgan_gp_lambda,
            use_spectral_norm=config.cgan_spectral_norm,
            loss=config.cgan_loss,
            label_smoothing=config.cgan_label_smoothing,
            feature_matching=config.cgan_feature_matching,
            device=config.device,
            seed=seed,
        )
    if phase == "C_early" and config.early_augment:
        from driftgan.augment.vanilla_gan import VanillaGanAugmenter

        return VanillaGanAugmenter(
            noise_dim=config.gan_noise_dim,
            epochs=config.gan_epochs,
            lr=config.gan_lr,
            device=config.device,
            seed=seed,
        )
    return None


def _make_monitor(config: ExperimentConfig, seed: int) -> ReconstructionMonitor | ShiftMonitor:
    """Build the early-detection monitor for the phase (AE or a cheap ablation)."""
    kind = config.monitor_type
    if kind == "ae":
        return ReconstructionMonitor(
            check_interval=config.monitor_interval,
            hidden=config.monitor_hidden,
            latent=config.monitor_latent,
            epochs=config.monitor_epochs,
            clip=config.monitor_clip,
            device=config.device,
            seed=seed,
        )
    common = dict(check_interval=config.monitor_interval, clip=config.monitor_clip, seed=seed)
    if kind == "pca":
        return MONITOR_TYPES["pca"](latent=config.monitor_latent, **common)
    return MONITOR_TYPES[kind](**common)


def run_phase(
    config: ExperimentConfig,
    phase: str | None = None,
    seed: int = 0,
    force_dynamic: bool = False,
    write_outputs: bool = True,
) -> RunResult:
    """Run one phase with one seed over the stream.

    Parameters
    ----------
    config : ExperimentConfig
    phase : str, optional
        Overrides ``config.phase`` (used by the schedule "pass 0").
    seed : int
        Seeds python/numpy/torch and the HAT/augmenter.
    force_dynamic : bool
        Ignore ``config.drift_schedule`` and run dynamic PHT (pass 0).
    write_outputs : bool
        When False, nothing is written to disk (used by pass 0 and tests).
    """
    phase = phase or config.phase
    seed_everything(seed)

    model = make_model(config, phase, seed)
    continuous = phase in CONTINUOUS_PHASES  # ARF/SRP self-adapt: learn every sample
    rolling = RollingBinaryStats(window_size=config.rolling_f1_window, pos_label=ATTACK)
    acc = river_metrics.Accuracy()
    tracker = CostTracker(rss_sample_interval=config.rss_sample_interval)

    # ── trigger selection ────────────────────────────────────────────
    trigger: (
        DynamicPHTTrigger | ScheduledDriftTrigger | PeriodicTrigger
        | DynamicDetectorTrigger | None
    ) = None
    dual: DualPHT | None = None
    monitor: ReconstructionMonitor | ShiftMonitor | None = None
    if phase in ("B", "C", "C_prime", "C_cond", "ARF_sched", "SRP_sched"):
        if config.drift_schedule and not force_dynamic:
            trigger = ScheduledDriftTrigger(
                load_schedule(config.drift_schedule), report_pht_delta=config.pht_delta
            )
        else:
            trigger = DynamicPHTTrigger(delta=config.pht_delta)
    elif phase == "B_adwin":
        trigger = DynamicDetectorTrigger(river_drift.ADWIN(), name="adwin")
    elif phase == "B_kswin":
        trigger = DynamicDetectorTrigger(river_drift.KSWIN(seed=seed), name="kswin")
    elif phase == "C_freq":
        trigger = PeriodicTrigger(
            interval=config.periodic_adapt_interval, start_after=config.n_pretrain
        )
    elif phase.startswith("C_early") or phase in ("ARF_early", "SRP_early"):
        # proactive dual (recon+class) trigger; base learner varies by phase
        dual = DualPHT(
            class_delta=config.pht_delta,
            recon_delta=config.recon_pht_delta,
            recon_threshold=config.recon_pht_threshold,
            recon_min_instances=config.recon_pht_min_instances,
        )
        monitor = _make_monitor(config, seed)

    augmenter = _make_augmenter(phase, config, seed)

    # ── state ────────────────────────────────────────────────────────
    snap_indices: list[int] = []
    snap_f1: list[float] = []
    snap_acc: list[float] = []
    snap_valid: list[bool] = []
    snap_interval_ms: list[float] = []
    interval_time = 0.0
    drift_events: list[dict[str, Any]] = []
    label_dist: Counter[str] = Counter()
    recent_buffer: deque[tuple[dict[str, float], str, str]] = deque(maxlen=config.buffer_size)
    training_enabled = False
    samples_to_train = 0
    n_processed = 0

    trace_writer = None
    trace_file = None
    if write_outputs and config.trace_enabled:
        run_dir = config.run_dir(seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_file = gzip.open(run_dir / "trace.csv.gz", "wt", newline="")
        trace_writer = csv.writer(trace_file)
        trace_writer.writerow(["index", "y_true", "y_raw", "y_pred", "correct"])

    def do_augment() -> int:
        """Generate synthetic attacks from the buffer and feed them to HAT."""
        if augmenter is None:
            return 0
        attack_buffer: Buffer = [(bx, by_raw) for bx, by, by_raw in recent_buffer if by == ATTACK]
        if len(attack_buffer) < config.min_attack_samples:
            return 0
        synthetic = augmenter.fit_generate(attack_buffer, config.n_synthetic)
        tracker.add("gan_training", getattr(augmenter, "t_train", 0.0))
        tracker.add("gan_generation", getattr(augmenter, "t_generate", 0.0))
        with tracker.section("hat_update"):
            for syn_x in synthetic:
                model.learn_one(syn_x, ATTACK)
        return len(synthetic)

    log_ctx(logger, logging.INFO, "run started", phase=phase, seed=seed, dynamic=force_dynamic or not config.drift_schedule)

    stream = stream_csv(
        config.csv_path,
        drop_cols=set(config.drop_cols),
        target_col=config.target_col,
        start_row=config.start_row,
        max_rows=config.max_samples,
    )

    # ── pre-training ─────────────────────────────────────────────────
    for x, y, y_raw in stream:
        model.learn_one(x, y)
        label_dist[y_raw] += 1
        n_processed += 1
        if n_processed >= config.n_pretrain:
            break
    log_ctx(logger, logging.INFO, "pretraining done", n=n_processed)

    # ── prequential evaluation ───────────────────────────────────────
    for x, y, y_raw in stream:
        n_processed += 1
        label_dist[y_raw] += 1

        with tracker.sample():
            with tracker.section("hat_inference"):
                y_pred = str(model.predict_one(cast("dict[Any, Any]", x)) or BENIGN)

            rolling.update(y, y_pred)
            acc.update(y, y_pred)
            error = 0 if y_pred == y else 1
            recent_buffer.append((x, y, y_raw))

            fired = False
            fire_source = ""
            if continuous:
                pass  # ARF/SRP self-adapt; learning happens every sample below
            elif trigger is not None:
                fired = trigger.update(n_processed, error)
                if isinstance(trigger, ScheduledDriftTrigger):
                    fire_source = "schedule"
                elif isinstance(trigger, PeriodicTrigger):
                    fire_source = "periodic"
                elif isinstance(trigger, DynamicDetectorTrigger):
                    fire_source = trigger.name
                else:
                    fire_source = "pht"
            elif dual is not None:
                if dual.update_class(n_processed, error):
                    fired = True
                    fire_source = "class"
                if monitor is not None:
                    with tracker.section("monitor"):
                        recon_err = monitor.observe(x, n_processed)
                    if recon_err is not None and dual.update_recon(n_processed, recon_err):
                        fired = True
                        fire_source = "recon" if fire_source == "" else "recon+class"

            if fired:
                training_enabled = True
                samples_to_train = config.train_window
                n_syn = do_augment()
                drift_events.append(
                    {"index": n_processed, "source": fire_source, "n_synthetic": n_syn}
                )
                if len(drift_events) <= 30:
                    log_ctx(
                        logger, logging.INFO, "drift fired",
                        index=n_processed, source=fire_source, n_synthetic=n_syn,
                    )

            if continuous:
                with tracker.section("hat_update"):
                    model.learn_one(x, y)
            elif training_enabled:
                with tracker.section("hat_update"):
                    model.learn_one(x, y)
                samples_to_train -= 1
                if samples_to_train <= 0:
                    training_enabled = False

        interval_time += tracker.last_latency_ms

        if trace_writer is not None:
            trace_writer.writerow([n_processed, y, y_raw, y_pred, 1 - error])

        if n_processed % config.snap_interval == 0:
            snap_indices.append(n_processed)
            snap_f1.append(rolling.f1())
            snap_acc.append(acc.get())
            snap_valid.append(rolling.has_positive)
            snap_interval_ms.append(interval_time)
            interval_time = 0.0

        if n_processed % config.log_interval == 0:
            log_ctx(
                logger, logging.INFO, "progress",
                index=n_processed, acc=round(acc.get(), 4), f1=round(rolling.f1(), 4),
                drifts=len(drift_events), training=training_enabled,
            )

    if trace_file is not None:
        trace_file.close()

    # ── lead-time analysis (C_early) ─────────────────────────────────
    lead_report: dict[str, Any] | None = None
    if dual is not None:
        report = match_lead_times(dual.recon_fires, dual.class_fires, horizon=config.lead_horizon)
        lead_report = {
            "mean_lead_time": report.mean_lead_time,
            "lead_times": report.lead_times,
            "pairs": report.pairs,
            "anticipated_fp_rate": report.anticipated_fp_rate,
            "missed_anticipation_rate": report.missed_anticipation_rate,
            "n_recon_fires": report.n_recon_fires,
            "n_class_fires": report.n_class_fires,
            "recon_fires": dual.recon_fires,
            "class_fires": dual.class_fires,
            # Full recon-error series (index, error) enables offline PHT_recon
            # calibration without re-running the AE.
            "recon_error_series": monitor.error_history if monitor is not None else [],
        }

    drift_indices = [e["index"] for e in drift_events]
    report_pht = trigger.report_pht_indices if isinstance(trigger, ScheduledDriftTrigger) else []

    result = RunResult(
        phase=phase,
        seed=seed,
        n_processed=n_processed,
        final_acc=float(acc.get()),
        final_f1=float(rolling.f1()),
        snap_indices=np.array(snap_indices),
        snap_f1=np.array(snap_f1),
        snap_acc=np.array(snap_acc),
        snap_valid=np.array(snap_valid, dtype=bool),
        snap_interval_ms=np.array(snap_interval_ms),
        drift_indices=drift_indices,
        drift_events=drift_events,
        report_pht_indices=list(report_pht),
        lead_time_report=lead_report,
        cost_summary=tracker.summary(),
        label_dist=dict(label_dist),
    )

    if write_outputs:
        _write_outputs(config, result, tracker)

    log_ctx(
        logger, logging.INFO, "run finished",
        phase=phase, seed=seed, n=n_processed,
        final_acc=round(result.final_acc, 4), final_f1=round(result.final_f1, 4),
        drifts=len(drift_indices),
    )
    return result


def _write_outputs(config: ExperimentConfig, result: RunResult, tracker: CostTracker) -> None:
    run_dir = config.run_dir(result.seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    save_manifest(build_manifest(config, result.phase, result.seed), run_dir)

    pd.DataFrame(
        {
            "index": result.snap_indices,
            "f1": result.snap_f1,
            "acc": result.snap_acc,
            "valid": result.snap_valid,
            "interval_ms": result.snap_interval_ms,
        }
    ).to_parquet(run_dir / "snapshots.parquet", index=False)

    with open(run_dir / "drift_events.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "events": result.drift_events,
                "report_pht_indices": result.report_pht_indices,
                "label_dist": result.label_dist,
            },
            f,
            indent=2,
        )

    tracker.save(run_dir / "cost.json")

    if result.lead_time_report is not None:
        with open(run_dir / "lead_time.json", "w", encoding="utf-8") as f:
            json.dump(result.lead_time_report, f, indent=2)

    events = compute_recovery_events(
        result.snap_indices,
        result.snap_f1,
        result.snap_valid,
        result.drift_indices,
        theta=config.recovery_theta,
        k=config.recovery_k,
        horizon=config.recovery_horizon,
    )
    save_recovery(events, result.phase, result.seed, config.results_dir)

    summary = {
        "phase": result.phase,
        "seed": result.seed,
        "n_processed": result.n_processed,
        "final_acc": result.final_acc,
        "final_f1": result.final_f1,
        "n_drifts": len(result.drift_indices),
        **{k: v for k, v in result.cost_summary.items() if k.startswith(("latency", "peak", "time"))},
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
