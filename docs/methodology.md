# Methodology

Technical detail of the critical design decisions. The README gives the overview; this document is the reference for anyone auditing the protocol.

## 1. Shared drift schedule

**Problem.** With per-phase dynamic detection, the moment a phase modifies its model, its error signal diverges from the other phases' and PHT fires at different indices. B and C then respond to *different events*, and per-event comparisons (RT, AUD, Wilcoxon pairing) are meaningless.

**Design.**

- *Primary schedule* (`build_schedule_from_pht`): a "pass 0" runs the reactive baseline's dynamics (HAT + dynamic PHT + training windows) once and records the fire indices. Every scheduled arm (`B`, `ARF_sched`, `SRP_sched`) replays these indices via `ScheduledDriftTrigger`. The schedule is biased toward the baseline's dynamics by construction; this is acceptable because that baseline is the reference the proposal must beat, and the alternative (per-arm detection) breaks pairing entirely.
- *Complementary schedule* (`build_schedule_from_labels`): ground truth derived from label transitions. An event is recorded at the first sample of each new non-BENIGN subtype block; consecutive events closer than `min_gap` are merged (first kept). Events inside the pre-training prefix are dropped. This schedule is model-free but "oracle" (uses labels the deployed system would not have); we use it for sensitivity analysis, not headline results.
- When a schedule is active, a PHT still runs in **report-only** mode (`report_pht_indices`): its fires measure how much later error-based detection would have triggered relative to the scheduled events, but never cause adaptation.
- Index convention: indices count samples of the evaluated stream starting at 1 for the first sample after `start_row`, pre-training included. A scheduled index that the stream never reaches (filtered row) is dropped, never fired late — adaptation timing must not depend on phase dynamics.
- **The proactive arms are exempt**: their claimed contribution is anticipation, which a fixed schedule would erase. They always run dynamic dual detection, and their recovery metrics are compared with the caveat that their events differ (stated explicitly in any report).

## 2. Recovery metrics

Definitions over rolling-F1 snapshots taken every `snap_interval` samples (window = 5,000; snapshots start after 100 window samples).

- **Validity.** A snapshot is *invalid* when its rolling window contains no positive (ATTACK) true label: F1 is undefined there (the old notebooks plotted it as 0, silently corrupting recovery measurements). Invalid snapshots are **excluded** from RT and AUD — they neither break nor extend a sustained run — and their fraction is reported per event (`excluded_windows_pct`).
- **Event window.** From drift index `d` to `min(d + H, next_drift)`, with `H = 50,000` by default.
- **RT**: samples from `d` to the *first* snapshot of the first run of `k = 3` consecutive valid snapshots with `F1 >= theta` (`theta = 0.9`). A tail run cut short by the event window end does not count (the "sustained" requirement is not met inside the window).
- **AUD**: `sum over valid snapshots i of max(0, theta - F1_i) * (idx_i - idx_{i-1})`, rectangle rule from `d` up to the recovery snapshot, or to the event window end when there is no recovery. Units: deficit × samples.
- **Overlap.** If the next drift arrives before recovery, the current event closes with `no_recovery = True` and a new event opens. Events are therefore disjoint and each sample of deficit is attributed to exactly one event.
- **Aggregation.** RT statistics are computed over recovered events only, and are always reported next to `no_recovery_rate` — comparing RT between phases with different no-recovery rates is survivorship bias, and the reporting code keeps both visible.

## 3. Multi-seed protocol and statistics

- Seeds: `random`, `numpy` (global + local `default_rng`), `torch` CPU/CUDA, the base learner (`seed=` at construction). Real-data results use 10 seeds; the comparator experiment uses 20, because its collapse outcome is bimodal and a handful of replicates estimates a bimodal rate badly.
- Every run writes `manifest.json`: git commit, SHA-256 of the input CSV (cached by size+mtime), fully resolved config + hash, seed, package versions, timestamp.
- Point estimates: mean ± 95% CI via percentile bootstrap (10,000 resamples) — no normality assumption.
- Phase comparisons: Wilcoxon signed-rank **paired by `drift_index`** (valid only under the shared schedule; events missing in one phase are dropped from the pair set). Effect sizes: `r = |Z|/sqrt(n_pairs)` and Cliff's delta. With ~tens of events per run, p-values alone are not persuasive; effect sizes are primary.

## 4. The unsupervised covariate-shift trigger

- Monitor: small autoencoder (in → 32 → 16 → 32 → in) fine-tuned every `monitor_interval = 1,000` samples for 5 epochs on a 2,000-sample buffer. The reconstruction error is measured against a **frozen reference scaler** (fixed on the first check) and **before** each fine-tune, so a covariate shift shows up as a *leading* spike; standardized inputs are clipped (`monitor_clip = 10`) to tame heavy-tailed features. This redesign replaced an earlier per-buffer standardization that cancelled the very signal being detected. Cheap ablations of this monitor (PCA / KS / MMD / mean-shift) share the identical protocol — see [experiments/02-monitor-ablation.md](experiments/02-monitor-ablation.md).
- `PHT_recon` (delta = 0.0005, threshold = 0.1, calibrated offline on the persisted recon-error series) over that signal; `PHT_class` (delta = 0.002) over the 0/1 error as usual. Either can trigger adaptation in a proactive arm; the fire source is recorded per event.
- **Lead time**: greedy pairing of each class fire with the nearest preceding unpaired recon fire within `lead_horizon = 20,000` samples. Reported: mean/distribution of lead times, anticipated-FP rate (recon fires never followed by a class fire within the horizon), missed-anticipation rate (class fires with no preceding recon fire).
- Falsifiability: the hypothesis fails if lead times are ~0/negative-dominated or the anticipated-FP rate makes early adaptation more costly than the lead time is worth. The analysis reports whichever outcome the data supports.
- **Per-dataset recalibration (external validity, #4).** The recon Page-Hinkley threshold is calibrated per dataset because the reconstruction-error scale differs across feature spaces. The offline rule is the same: run the monitor over a smoke window, inspect the recon-error series and fire behaviour, and pick the threshold so the monitor fires on real bursts rather than every window. For CIC-IDS2018 and UNSW-NB15 the 2017 value (`0.1`, delta `0.0005`) was confirmed to generalize (2018: 185 fires / mean lead ~3.2K over a 1M smoke; UNSW: 25 fires / mean lead ~6.3K over a 200K smoke); the values are pinned in `configs/datasets/<name>.yaml` so every proactive arm on a dataset shares the same calibrated monitor. Values that depend on the *(dataset, phase)* pair rather than the dataset alone — the blind clock's interval, calibrated per learner — go in that profile's `by_phase:` block. `scripts/calibrate_monitor.py` performs the sweep offline against a cached score series.

## 5. Cost accounting

- Per-sample latency measured with `time.perf_counter()` around the full processing of each evaluated sample; reported p50/p95/p99/max.
- Component breakdown: `hat_inference`, `hat_update`, `monitor`, `other` (sample time not covered by a section).
- RSS sampled every 5,000 samples via psutil; peak and mean reported.
- Trade-off view: Pareto scatter of mean RT vs p95 latency, one point per (phase, seed).
