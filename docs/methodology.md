# Methodology

Technical detail of the critical design decisions. The README gives the overview; this document is the reference for anyone auditing the protocol.

## 1. Shared drift schedule (Improvement 1)

**Problem.** With per-phase dynamic detection, the moment a phase modifies its model, its error signal diverges from the other phases' and PHT fires at different indices. B and C then respond to *different events*, and per-event comparisons (RT, AUD, Wilcoxon pairing) are meaningless.

**Design.**

- *Primary schedule* (`build_schedule_from_pht`): a "pass 0" runs Phase B dynamics (HAT + dynamic PHT + retraining windows) once and records the fire indices. All phases (B, C, C', C_cond) replay these indices via `ScheduledDriftTrigger`. The schedule is biased toward B's dynamics by construction; this is acceptable because B is the reference baseline the proposals must beat, and the alternative (per-phase detection) breaks pairing entirely.
- *Complementary schedule* (`build_schedule_from_labels`): ground truth derived from label transitions. An event is recorded at the first sample of each new non-BENIGN subtype block; consecutive events closer than `min_gap` are merged (first kept). Events inside the pre-training prefix are dropped. This schedule is model-free but "oracle" (uses labels the deployed system would not have); we use it for sensitivity analysis, not headline results.
- When a schedule is active, a PHT still runs in **report-only** mode (`report_pht_indices`): its fires measure how much later error-based detection would have triggered relative to the scheduled events, but never cause adaptation.
- Index convention: indices count samples of the evaluated stream starting at 1 for the first sample after `start_row`, pre-training included. A scheduled index that the stream never reaches (filtered row) is dropped, never fired late — adaptation timing must not depend on phase dynamics.
- **Phase C_early is exempt**: its claimed contribution is anticipation, which a fixed schedule would erase. It always runs dynamic dual detection, and its recovery metrics are compared with the caveat that its events differ (stated explicitly in any report).

## 2. Recovery metrics (Improvement 3)

Definitions over rolling-F1 snapshots taken every `snap_interval` samples (window = 5,000; snapshots start after 100 window samples).

- **Validity.** A snapshot is *invalid* when its rolling window contains no positive (ATTACK) true label: F1 is undefined there (the old notebooks plotted it as 0, silently corrupting recovery measurements). Invalid snapshots are **excluded** from RT and AUD — they neither break nor extend a sustained run — and their fraction is reported per event (`excluded_windows_pct`).
- **Event window.** From drift index `d` to `min(d + H, next_drift)`, with `H = 50,000` by default.
- **RT**: samples from `d` to the *first* snapshot of the first run of `k = 3` consecutive valid snapshots with `F1 >= theta` (`theta = 0.9`). A tail run cut short by the event window end does not count (the "sustained" requirement is not met inside the window).
- **AUD**: `sum over valid snapshots i of max(0, theta - F1_i) * (idx_i - idx_{i-1})`, rectangle rule from `d` up to the recovery snapshot, or to the event window end when there is no recovery. Units: deficit × samples.
- **Overlap.** If the next drift arrives before recovery, the current event closes with `no_recovery = True` and a new event opens. Events are therefore disjoint and each sample of deficit is attributed to exactly one event.
- **Aggregation.** RT statistics are computed over recovered events only, and are always reported next to `no_recovery_rate` — comparing RT between phases with different no-recovery rates is survivorship bias, and the reporting code keeps both visible.

## 3. Multi-seed protocol and statistics (Improvement 4)

- Seeds: `random`, `numpy` (global + local `default_rng`), `torch` CPU/CUDA, HAT (`seed=` at construction), augmenters. Default 5 seeds (0–4).
- Every run writes `manifest.json`: git commit, SHA-256 of the input CSV (cached by size+mtime), fully resolved config + hash, seed, package versions, timestamp.
- Point estimates: mean ± 95% CI via percentile bootstrap (10,000 resamples) — no normality assumption.
- Phase comparisons: Wilcoxon signed-rank **paired by `drift_index`** (valid only under the shared schedule; events missing in one phase are dropped from the pair set). Effect sizes: `r = |Z|/sqrt(n_pairs)` and Cliff's delta. With ~tens of events per run, p-values alone are not persuasive; effect sizes are primary.

## 4. Augmentation control (Improvement 2)

`C_prime` (bootstrap + jitter) consumes exactly the same budget as C (`n_synthetic = 200` per event, same buffer, same `min_attack_samples`) and runs in the same engine with the same cost instrumentation. Jitter sigma is `0.01 × std(feature)` on the buffer; zero-variance features receive no noise. Interpretation rule: if C ≤ C_prime, the GAN adds no structure beyond oversampling.

## 5. Conditional GAN (Improvement 5)

See `docs/gan_design.md` for architecture and ablations. Key protocol points:

- Conditioning label = original CICIDS2017 subtype (`y_raw`), embedded (dim 8).
- Generation is proportional to the subtype mix of the current buffer (largest-remainder allocation), so the augmentation reflects the attacks actually occurring at that drift.
- Sample quality is measured per subtype and per event with multivariate energy distance and RBF-MMD (median heuristic), standardized by real-sample statistics.

## 6. Early detection (Improvement 6 — experimental)

- Monitor: small autoencoder (in → 32 → 16 → 32 → in) fine-tuned every `monitor_interval = 5,000` samples for 5 epochs on a 2,000-sample buffer; the mean reconstruction error (buffer-standardized) forms a low-frequency signal.
- `PHT_recon` (delta = 0.005) over that signal; `PHT_class` (delta = 0.002) over the 0/1 error as usual. Both trigger adaptation in C_early; fire sources are recorded.
- **Lead time**: greedy pairing of each class fire with the nearest preceding unpaired recon fire within `lead_horizon = 20,000` samples. Reported: mean/distribution of lead times, anticipated-FP rate (recon fires never followed by a class fire within the horizon), missed-anticipation rate (class fires with no preceding recon fire).
- Falsifiability: the hypothesis fails if lead times are ~0/negative-dominated or the anticipated-FP rate makes early adaptation more costly than the lead time is worth. The analysis notebook reports whichever outcome the data supports.

## 7. Cost accounting (Improvement 7)

- Per-sample latency measured with `time.perf_counter()` around the full processing of each evaluated sample; reported p50/p95/p99/max.
- Component breakdown: `hat_inference`, `hat_update`, `gan_training`, `gan_generation`, `monitor`, `other` (sample time not covered by a section). Augmenters self-report their train/generate split, credited via `CostTracker.add`.
- RSS sampled every 5,000 samples via psutil; peak and mean reported.
- Trade-off view: Pareto scatter of mean RT vs p95 latency, one point per (phase, seed).
