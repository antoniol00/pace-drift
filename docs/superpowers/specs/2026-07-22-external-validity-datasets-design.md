# External validity across datasets (#4) — design

**Date:** 2026-07-22
**Status:** approved (design), pending implementation plan
**Author:** research-engineering session

## 1. Goal

Establish external validity for the study's constructive result — *a proactive,
covariate-shift-monitor-triggered "when-to-adapt" signal improves streaming
classifiers across base learners, at a fraction of the adaptation budget*
(experiment 3c) — by reproducing the base-learner × trigger 2×2 on **two
additional naturally-drifting datasets** beyond CICIDS2017.

**Scope decided (this session):**

- **Track:** natural-drift networks. Datasets: **CICIDS2017** (existing anchor)
  **+ CIC-IDS2018 + UNSW-NB15**. Venue framing: general network intrusion
  detection.
- **Phases per new dataset (×5 seeds):** the minimal 2×2 headline set —
  **B, C_early, ARF_sched, SRP_sched, ARF_early, SRP_early** (6 phases). The
  null GAN phases (C, C_cond, C_prime) and monitor ablations are **not** rerun on
  the new datasets; those results are already established on 2017 and are
  off-message for the current framing.

### Why these datasets (evidence)

A one-pass label-ordering diagnostic (`transitions / rows`, run-length encoding)
was run on all candidate CSVs. Result:

| Dataset | transitions/rows | Structure | Natural drift usable? |
|---|---|---|---|
| CICIDS2017 (anchor) | bursty | day-concatenated, attack bursts | yes (already used) |
| **CIC-IDS2018** | 0.0005 | large benign/attack blocks, bursts | **yes — like 2017** |
| **UNSW-NB15** | 0.05 | benign with interleaved attack bursts | yes, realistic |
| CIC-DDoS2019 | (not measured) | CICFlowMeter family | probable (not selected) |
| CICIoT2023 | 0.91 | fully shuffled (runs of ~1) | **no — no drift** |
| Edge-IIoTset | 0.0000 | class-sorted hard blocks | designed drift only |

The IoT candidates were rejected for #4: CICIoT2023 is a shuffled ML-ready dump
(no temporal drift), and Edge-IIoTset is class-sorted with hard block boundaries
where covariate shift and label change coincide (lead time ≈ 0), which is exactly
where the anticipation contribution cannot be demonstrated. The CICFlowMeter
family (2018/UNSW) exhibits gradual/bursty drift matching the existing
methodology, at near-zero engineering and methodological risk. UNSW-NB15 here is
a CICFlowMeter **re-extraction** (same ~80-feature family, header begins
`Flow ID,Src IP,...,Flow Duration,Total Fwd Packet,...`), not the classic 49-feature
Argus/Bro set — so it runs on the same pipeline, while contributing a genuinely
different attack taxonomy (Exploits, Fuzzers, Reconnaissance, Generic, DoS,
Shellcode, Backdoor, Analysis, Worms).

## 2. Current-state facts the design relies on

- The engine is already **feature-agnostic**: feature dimensionality and order are
  inferred from the first sample's keys (`shift_monitors.py:_keys`,
  `reconstruction_monitor.py:_keys`, `augment/base.py` stacks `buffer[0][0].keys()`).
  river HAT/ARF/SRP consume feature **dicts** natively. No fixed feature list or
  count is hardcoded anywhere.
- `preprocess_row` (`data/stream.py`) already parameterizes `drop_cols` and
  `target_col`, converts numeric columns to float, drops non-numeric columns, and
  replaces inf/NaN with 0.0.
- The **binary** label space the model learns/predicts is always the module
  constants `BENIGN`/`ATTACK` (`data/stream.py`). Only the *raw* subtype `y_raw`
  and the benign-string comparison are dataset-specific. Therefore the only
  hardcoded 2017 assumption is the benign label string `"BENIGN"`, used in exactly
  two places: the binarization in `preprocess_row` and the benign skip in
  `build_schedule_from_labels`.
- Config merges `base.yaml ← phase yaml ← CLI` (`config.py`). Results write to
  `results/runs/<phase>/seed_<n>/` and `results/recovery/<phase>_<seed>.parquet`;
  `results/` is gitignored.
- The shared schedule (`drift/schedule.py`) is produced by a pass-0 Phase-B run
  and replayed by scheduled phases; C_early and the `*_early` phases are exempt
  (dynamic dual PHT). Monitor thresholds for C_early were calibrated offline on
  2017's persisted recon-error series (methodology §6).

## 3. Chosen approach

**Per-dataset YAML profiles + a dataset merge layer** (approach A). Rejected: a
Python dataset registry (violates "no experiment constants in modules") and
CLI-only overrides (no result namespacing, leaves `benign_label` hardcoded, does
not scale to report/figures).

## 4. Design

### 4.1 Config

Add to `ExperimentConfig`:

- `benign_label: str = "BENIGN"` — the raw benign class string for this dataset.
- `dataset: str = "cicids2017"` — a short name used to namespace results.

Extend the merge to `base.yaml ← configs/datasets/<dataset>.yaml ← phase yaml ←
CLI`, selected by a new `--dataset <name>` CLI flag on `run_experiment.py`
(default `cicids2017`, which resolves to the existing base.yaml values → fully
backward compatible).

### 4.2 Dataset profiles

`configs/datasets/cicids2018.yaml` and `configs/datasets/unsw.yaml`, each setting:

- `dataset`, `csv_path`, `target_col: Label`, `benign_label: Benign`.
- `drop_cols`: identifiers + timestamp, keeping the 2017 philosophy (drop
  identifiers, keep port/protocol). UNSW: `Flow ID, Src IP, Dst IP, Src Port,
  Timestamp`. 2018: `Timestamp` (+ any identifier columns present).
- `start_row`, `max_samples`: chosen by a **drift-density scan** (see 4.4) to land
  a drift-rich window, mirroring 2017's `start_row: 540000`.
- Monitor thresholds (`recon_pht_threshold`, `recon_pht_delta`, …): **recalibrated
  per dataset** (see 4.5).

### 4.3 Code changes (minimal)

- `data/stream.py`: `preprocess_row(row, drop_cols, target_col, benign_label)` and
  `stream_csv(..., benign_label)`; binarization compares `y_raw == benign_label`.
  `ATTACK`/`BENIGN` constants stay as the binary label space.
- `drift/schedule.py`: `build_schedule_from_labels(..., benign_label)`.
- `experiments/runner.py`: thread `config.benign_label`; **namespace outputs by
  dataset** — `results/runs/<dataset>/<phase>/seed_<n>/` and
  `results/recovery/<dataset>_<phase>_<seed>.parquet`. `cicids2017` keeps the
  current flat paths (a `results_subdir` that is empty for the default dataset and
  `"<dataset>/"` otherwise) so existing 2017 results are neither moved nor rerun.
  Schedule path becomes `results/drift_schedule_<dataset>.json`.
- `run_experiment.py`: `--dataset` flag threaded into config building and the
  schedule path.
- `Makefile`: `schedule-2018`, `schedule-unsw`, `run-min-2018`, `run-min-unsw`
  (pass-0 B then the 6 phases × seeds).

### 4.4 Drift-density calibration (per dataset, before long runs)

A short scan of attack density along the stream picks `start_row`/`max_samples`
for a window that (a) contains multiple attack subtypes/bursts (drift events) and
(b) is not degenerate (has positive labels in most rolling windows). Recorded in
the dataset YAML with a one-line justification. This is analogous to how 2017's
`start_row: 540000` was chosen.

### 4.5 Monitor recalibration (per dataset)

C_early's recon Page-Hinkley threshold depends on the reconstruction-error scale,
which differs per dataset. Reuse the offline calibration already documented in
methodology §6: run the AE monitor over a pretrain/early window, persist the
recon-error series, set the PHT threshold to a percentile of the PHT statistic so
the fire rate is sensible (target: fires on real bursts, not every window). If
C_early does not fire on a dataset with the ported threshold, recalibrate before
committing to the 5-seed run. Record the chosen values in the dataset YAML.

### 4.6 Run protocol per new dataset

1. Pass-0 Phase B → `results/drift_schedule_<dataset>.json`.
2. Scheduled phases (replay the schedule): **B, ARF_sched, SRP_sched**.
3. Dynamic early phases (exempt from schedule): **C_early, ARF_early, SRP_early**.
4. All × seeds 0–4.

### 4.7 Reporting

- `scripts/paper_figures.py`: **fig12** — the 2×2 (Δ from reactive `*_sched` to
  proactive `*_early` per base learner) shown side by side for
  2017 / 2018 / UNSW. The generalization claim: the proactive early trigger
  improves every base learner on every dataset, using fewer adaptations.
- A cross-dataset summary table: per (dataset × phase) overall local F1 and
  adaptation count.
- `docs/experiments/05-external-validity.md`: question / setup / per-dataset
  results tables / fig12 / interpretation / caveats, matching the existing
  per-experiment doc format. Copy fig12 to `docs/figures/`.
- Update README §8 (external-validity subsection + fig12) and
  `docs/architecture.md` (dataset-profile slot).

### 4.8 Tests (keep ruff / mypy / existing suite green)

- Dataset-profile merge: `base ← dataset ← phase ← CLI` resolves expected fields.
- `benign_label` threading: `preprocess_row` binarizes correctly when the benign
  string is not `"BENIGN"`; a non-`"BENIGN"` benign row is treated as BENIGN.
- `build_schedule_from_labels` with a non-`"BENIGN"` benign label detects subtype
  blocks correctly and skips benign.
- Result-path namespacing: default dataset → flat path; named dataset → subdir.

## 5. Risks and mitigations

- **Feature leakage.** Verify no retained column encodes the label/identifier that
  trivially reveals attack vs benign (audit each dataset's `drop_cols`).
- **Monitor does not fire** on a new dataset → recalibrate PHT threshold offline
  (4.5) before the long run.
- **Drift-poor window** → the density scan (4.4) fixes `start_row`/`max_samples`
  before spending hours of compute.
- **Compute** — 6 phases × 5 seeds × 2 datasets plus 2 pass-0 schedule runs. Runs
  are long; sequence per dataset and monitor progress.

## 6. Out of scope

- IoT datasets (CICIoT2023, Edge-IIoTset) and designed-drift scenarios.
- GAN augmentation phases and monitor ablations on the new datasets.
- Re-running or moving existing CICIDS2017 results.
- Cross-dataset feature alignment (each dataset runs on its own feature set; no
  shared feature vector is required or built).

## 7. Success criteria

- 2018 and UNSW each produce the 6-phase × 5-seed results with a valid per-dataset
  schedule and a firing C_early monitor.
- fig12 + doc 05 show the direction of the 3c result (proactive `*_early` ≥
  reactive `*_sched` per base learner, with fewer adaptations) on the new datasets,
  or honestly report where it does not hold.
- ruff, mypy, and the existing test suite stay green; new tests cover the profile
  merge and `benign_label` threading.
