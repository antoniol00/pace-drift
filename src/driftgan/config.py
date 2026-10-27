"""YAML-based experiment configuration with base + per-phase override resolution."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"

VALID_PHASES = (
    # Reactive baselines: adaptation triggered by the classification error.
    "B", "B_adwin", "B_kswin",
    # Streaming ensembles learning from every sample (upper reference).
    "ARF", "SRP",
    # Budget-matched controls: the same learners adapting only inside triggered
    # windows on the shared schedule, i.e. at the reactive arm's label budget.
    "ARF_sched", "SRP_sched",
    # Live reactive controls for the ensembles: the error Page-Hinkley runs on the
    # ensemble's own predictions instead of replaying the HAT-derived schedule, so
    # *_early vs *_pht isolates what the monitor adds to the same live error test.
    "ARF_pht", "SRP_pht",
    # PACE: the proactive covariate-shift trigger, on each base learner.
    "C_early", "ARF_early", "SRP_early",
    # Frequency-matched controls: a blind periodic clock whose fire count
    # matches the matching *_early arm, isolating timing from frequency.
    "C_freq", "ARF_freq", "SRP_freq",
    # Monitor ablation: cheaper shift statistics behind the same policy.
    "C_early_mean", "C_early_pca", "C_early_ks", "C_early_mmd",
    # Published unsupervised detectors as external comparators, at the identical
    # protocol: D3 (Gozuacik et al., CIKM 2019) and HDDDM (Ditzler & Polikar,
    # CIDUE 2011). The "d3_native"/"d3_sized" arms additionally restore D3's own
    # sliding window and AUC threshold, so the comparison does not depend on our
    # window bookkeeping.
    "C_early_d3", "C_early_d3_native", "C_early_d3_sized", "C_early_hdddm",
)


@dataclass
class ExperimentConfig:
    """Resolved configuration for a single phase run.

    Parameters mirror the YAML schema in ``configs/``. Every tunable of the
    experiment lives here so that a run is fully described by (config, seed).
    """

    phase: str

    # Dataset identity
    dataset: str = "cicids2017"  # names configs/datasets/<dataset>.yaml; also namespaces results
    benign_label: str = "BENIGN"  # raw benign class string for this dataset

    # Dataset / stream
    csv_path: str = "/mnt/AI-DATA-SSD/alara/Data/CIC-IDS2017.csv"
    drop_cols: tuple[str, ...] = ("Flow ID", "Source IP", "Destination IP", "Timestamp")
    target_col: str = "Attack Type"
    start_row: int = 540_000
    n_pretrain: int = 50_000
    max_samples: int = 2_000_000

    # Model (HAT)
    grace_period: int = 200

    # Drift detection / adaptation
    pht_delta: float = 0.002
    train_window: int = 5_000
    drift_schedule: str | None = None  # path to drift_schedule.json; overrides dynamic PHT
    # Frequency-matched controls (`*_freq`): adapt every N samples on a blind
    # periodic clock instead of any drift signal. N is set so the total number
    # of adaptations matches the matching `*_early` arm's fire count, isolating
    # adaptation *frequency* from adaptation *timing* (anticipation).
    periodic_adapt_interval: int = 12_000

    buffer_size: int = 500  # recent-sample window kept for diagnostics
    device: str = "cuda"  # falls back to cpu automatically if unavailable

    # The unsupervised covariate-shift monitor (PACE)
    # "ae" | "pca" | "ks" | "mmd" | "mean" (ablation)
    # | "hdddm" | "d3" | "d3_native"      (published comparators)
    monitor_type: str = "ae"
    monitor_interval: int = 1_000  # recon-error sample cadence (finer -> more PHT updates)
    monitor_epochs: int = 5
    monitor_hidden: int = 32
    monitor_latent: int = 16
    monitor_clip: float = 10.0  # clip standardized inputs to tame heavy-tailed features
    recon_pht_delta: float = 0.005
    recon_pht_threshold: float = 2.0  # cumulative-sum threshold for PHT_recon
    recon_pht_min_instances: int = 10  # warmup checks before PHT_recon can fire
    lead_horizon: int = 20_000  # max samples between PHT_recon and PHT_class fires to pair them

    # Native D3 comparator (monitor_type "d3_native"): the published algorithm's
    # own hyperparameters, defaults as released by Gozuacik et al. (CIKM 2019).
    # This monitor thresholds the AUC itself, so recon_pht_* does not apply.
    d3_w: int = 100  # "old" window size
    d3_rho: float = 0.1  # size of the "new" slice as a fraction of d3_w
    d3_tau: float = 0.7  # AUC above which drift is declared
    hdddm_bins: int | None = None  # histogram bins per feature (None -> sqrt(n))

    # Metrics / instrumentation
    rolling_f1_window: int = 5_000
    snap_interval: int = 500
    log_interval: int = 10_000
    recovery_theta: float = 0.9
    recovery_k: int = 3
    recovery_horizon: int = 50_000
    rss_sample_interval: int = 5_000
    trace_enabled: bool = True

    # Output
    results_dir: str = str(RESULTS_DIR)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.phase not in VALID_PHASES:
            raise ValueError(f"Unknown phase {self.phase!r}; expected one of {VALID_PHASES}")

    def resolved_dict(self) -> dict[str, Any]:
        """Return the full configuration as a plain JSON-serializable dict."""
        d = dict(self.__dict__)
        d["drop_cols"] = list(self.drop_cols)
        return d

    def config_hash(self) -> str:
        """Stable short hash of the resolved configuration (for manifests)."""
        blob = json.dumps(self.resolved_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def _ns_segments(self) -> tuple[str, ...]:
        """Result-path segment for the dataset (empty for the 2017 anchor)."""
        return () if self.dataset == "cicids2017" else (self.dataset,)

    def run_dir(self, seed: int) -> Path:
        """Directory where artifacts of (phase, seed) are written."""
        return Path(self.results_dir).joinpath("runs", *self._ns_segments(), self.phase, f"seed_{seed}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
    dataset: str | None = None,
) -> ExperimentConfig:
    """Load a phase YAML, merging it on top of ``configs/base.yaml`` if present.

    Parameters
    ----------
    path : str or Path
        Phase configuration YAML. Must contain at least ``phase:``.
    overrides : dict, optional
        Final overrides applied on top (e.g. from CLI flags).
    dataset : str, optional
        Dataset name; if provided, loads ``configs/datasets/<dataset>.yaml``
        and merges it after the phase layer (dataset overrides phase).
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        phase_cfg: dict[str, Any] = yaml.safe_load(f) or {}

    base_path = CONFIGS_DIR / "base.yaml"
    base_cfg: dict[str, Any] = {}
    if base_path.exists() and base_path.resolve() != path.resolve():
        with open(base_path, encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}

    dataset_cfg: dict[str, Any] = {}
    if dataset and dataset != "cicids2017":
        ds_path = CONFIGS_DIR / "datasets" / f"{dataset}.yaml"
        if not ds_path.exists():
            raise FileNotFoundError(f"dataset profile not found: {ds_path}")
        with open(ds_path, encoding="utf-8") as f:
            dataset_cfg = yaml.safe_load(f) or {}

    merged = _deep_merge(base_cfg, phase_cfg)
    merged = _deep_merge(merged, dataset_cfg)  # dataset overrides phase (per-dataset calibration)

    # Some calibrated values are specific to a (dataset, phase) pair rather than
    # to the dataset alone — the blind-clock interval of the *_freq controls is
    # calibrated per learner so its fire count matches that learner's *_early
    # arm. A flat dataset key cannot express that, so profiles may carry a
    # `by_phase` block. Leaving these out silently de-matches the control and
    # invalidates the comparison, which has happened once; keep them pinned in
    # the profile rather than passed on the command line.
    by_phase = merged.pop("by_phase", None) or {}
    phase_name = phase_cfg.get("phase")
    if phase_name and phase_name in by_phase:
        merged = _deep_merge(merged, by_phase[phase_name])

    if overrides:
        merged = _deep_merge(merged, overrides)
    if dataset:
        merged["dataset"] = dataset  # the --dataset flag is authoritative

    if "drop_cols" in merged:
        merged["drop_cols"] = tuple(merged["drop_cols"])
    known = {f_.name for f_ in ExperimentConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    extra = {k: v for k, v in merged.items() if k not in known}
    kwargs = {k: v for k, v in merged.items() if k in known and k != "extra"}
    return ExperimentConfig(**kwargs, extra=extra)
