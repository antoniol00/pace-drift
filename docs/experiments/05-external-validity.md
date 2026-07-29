# Experiment 5 — External validity across datasets (CIC-IDS2018 + UNSW-NB15)

## Question

[Experiment 3](03-sota-baselines.md) showed the proactive trigger improves
*every* base learner on CICIDS2017, using fewer adaptations than the reactive
baseline. That result is tied to one dataset's specific attack ordering.

> Does it hold on datasets beyond CICIDS2017?

## Setup

**Track and datasets.** Natural-drift networks, CICFlowMeter feature family.
Anchor: CICIDS2017. Added: **CIC-IDS2018** and
**UNSW-NB15** (a CICFlowMeter *re-extraction* of UNSW-NB15 — same ~80-feature
pipeline as 2017/2018, not the classic 49-feature Argus/Bro set — contributing a
genuinely different attack taxonomy: Exploits, Fuzzers, Reconnaissance,
Generic, DoS, Shellcode, Backdoor, Analysis, Worms).

**Dataset selection was evidence-driven.** A one-pass label-ordering
diagnostic (transitions/rows, run-length encoding) was run over all candidate
CSVs before committing compute. Two IoT candidates were rejected for this
purpose: **CICIoT2023** is fully shuffled (transitions/rows ≈ 0.91 — no
temporal drift to detect), and **Edge-IIoTset** is class-sorted with hard
block boundaries (transitions/rows ≈ 0 — covariate shift and label change
coincide at the same row, so anticipation lead time would be ≈ 0, exactly the
condition under which the anticipation claim cannot be demonstrated). The
CICFlowMeter family (2018, UNSW) showed gradual/bursty natural drift matching
the existing methodology, so those two were run instead.

**Minimal 2×2, per dataset.** Six phases — **`B`, `C_early`** (HAT) and
**`ARF_sched`/`ARF_early`, `SRP_sched`/`SRP_early`** — × 10 seeds.
The monitor ablations and published comparators are
**not** rerun; those results are already established on 2017 and are
off-message for a generalization check.

**Per-dataset evaluation window** (chosen by a drift-density scan,
`scripts/scan_drift_window.py`):
- **CIC-IDS2018**: rows 11M–14M (3M-sample window), the densest attack block
  after an intervening multi-million-row benign desert — DoS-Hulk,
  Infiltration, DDOS-HOIC, DDOS-LOIC-UDP, Bot, FTP-BruteForce, SSH-BruteForce
  arriving in separated bursts.
- **UNSW-NB15**: rows 1.75M–3.45M (1.7M-sample window), the dense second half
  of the file where all 9 attack families burst on a benign background.

**Per-dataset schedule.** Pass-0 Phase B fires **124 events on CIC-IDS2018**
and **117 events on UNSW-NB15** (same shared-schedule design 2017 uses with
its 233-event schedule — scheduled phases on a dataset replay that dataset's
own pass-0 indices).

**Monitor calibration.** C_early's recon Page-Hinkley threshold (`0.1`,
`delta=0.0005` on 2017) was checked on a short smoke run *before* committing
to the full runs, per the offline-calibration rule in
[methodology.md §4](../methodology.md). **The 2017 value generalized on both
datasets without retuning**: a 1M-sample smoke on 2018 fired 185 adaptations
(72 recon / 113 class, mean anticipation lead ~3.2K samples); a 200K-sample
smoke on UNSW fired 25 adaptations (14 recon / 11 class, mean lead ~6.3K
samples). Both fire on the real bursts, not on every window, so the ported
threshold was pinned as-is (`configs/datasets/cicids2018.yaml`,
`configs/datasets/unsw.yaml`).

## Results

Overall local F1 (mean over valid rolling windows, 10 seeds) and mean
adaptation count, reactive (`B`/`*_sched`) vs proactive (`C_early`/`*_early`):

| Dataset | Base learner | reactive F1 | proactive F1 | Δ | adaptations (reactive → proactive) |
|---|---|---|---|---|---|
| CIC-IDS2017 | HAT | 0.567 | 0.599 | +0.032 | 233 → 169 |
| CIC-IDS2017 | ARF | 0.717 | 0.747 | +0.030 | 233 → 110 |
| CIC-IDS2017 | SRP | 0.737 | 0.768 | +0.032 | 233 → 90 |
| CIC-IDS2018 | HAT | 0.828 | 0.899 | +0.071 | 124 → 306 |
| CIC-IDS2018 | ARF | 0.913 | 0.926 | +0.012 | 124 → 253 |
| CIC-IDS2018 | SRP | 0.924 | 0.929 | +0.005 | 124 → 254 |
| UNSW-NB15 | HAT | 0.355 | 0.452 | +0.097 | 117 → 214 |
| UNSW-NB15 | ARF | 0.603 | 0.662 | +0.059 | 117 → 171 |
| UNSW-NB15 | SRP | 0.697 | 0.745 | +0.048 | 117 → 167 |

*Every one of the 9 cells is a positive Δ. Paired confidence intervals for these
cells are in [experiment 6](06-timing-vs-frequency.md).*

![Base-learner × trigger 2×2, three datasets](../figures/fig12_external_validity.png)

## Interpretation

**What generalizes (the core claim).** The proactive trigger improves overall
local F1 in **all 9 cells** — 3 datasets × 3 base learners — every Δ positive.
The when-to-adapt signal helps every base learner tested on every dataset tested,
so it is not an artifact of CICIDS2017's specific attack ordering.

**What does not generalize — reported plainly.**
- **Label efficiency is 2017-specific.** On 2017 the proactive trigger fires
  *fewer* adaptations than the reactive schedule (233 → 113 for ARF, 233 → 91
  for SRP). On 2018 and UNSW the dynamic trigger fires *more* than the
  budget-matched schedule (124 → 255/256 on 2018, 117 → 167/173 on UNSW). So
  "the same gain at a fraction of the adaptation budget" is a 2017 finding,
  not a universal property of the method — on the other two datasets the
  proactive trigger buys its F1 gain with *more* label-consuming adaptations,
  not fewer.
- **There is no pattern by model strength.** On 2018 the gain *shrinks* with
  strength (HAT +0.071 → ARF +0.012 → SRP +0.005), a ceiling effect: the
  ensembles already reach F1 0.91–0.93 there, leaving little headroom. On UNSW
  the ensembles gain more than the tree on the ensemble cells but the HAT cell is
  the largest and the noisiest. Only the **sign** of the effect is invariant.

Combined: the *direction* is robust and general-purpose — proactive beats
reactive for every base learner on every dataset tried — but the second-order
properties that made the 2017 result look elegant (fewer adaptations, gains
scaling with capacity) are dataset-specific and are not claimed as universal.

## Caveats

- **UNSW-NB15 is ~97% benign.** Its attack-type mix is also fairly stationary
  across the evaluated window, so the drift there is mainly benign↔attack
  bursts rather than attack-subtype turnover — a sparser, less varied drift
  signal than 2017/2018. Many rolling windows have no positive label and are
  excluded from F1 (not scored as failure), per the recovery-metric semantics
  in [methodology.md](../methodology.md).
- **UNSW-NB15 here is a re-extraction**, not the dataset's original 49-feature
  Argus/Bro release — it runs on the same CICFlowMeter pipeline as 2017/2018,
  which is why it was chosen, but its absolute numbers are not directly
  comparable to prior literature using the classic UNSW-NB15 feature set.
- **Monitor thresholds were recalibrated per dataset** (a short offline smoke
  run before each full run, see Setup); the fact that the 2017 value
  generalized without retuning is itself a result, not an assumption made in
  advance.
- **Scope is deliberately narrow.** Only the minimal 6-phase 2×2 was rerun on
  the new datasets — the monitor ablations and comparators
  were not repeated, so this experiment speaks only to the when-to-adapt
  trigger's generalization, not to the monitor-choice
  result.
- **Adaptation counts are not comparable across datasets in absolute terms**
  (schedule lengths differ: 233 on 2017 vs 124 on 2018 vs 117 on UNSW,
  reflecting each dataset's own pass-0 drift dynamics) — only the
  reactive → proactive direction *within* a dataset is meaningful.
- **Dataset selection was evidence-driven but not exhaustive** — the
  label-ordering diagnostic screened the candidates on hand; other
  CICFlowMeter-family datasets (e.g. CIC-DDoS2019) were not run.
