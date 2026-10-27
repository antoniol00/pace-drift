# Experiment 3 — Streaming SOTA: capacity, supervision, and the trigger

## Question

Three questions that only make sense together, so they are answered together:

1. Do **other drift detectors** do better than plain Page-Hinkley as the reactive
   arm? (If so, the baseline is a straw man.)
2. Modern self-adapting ensembles beat everything on raw quality — but they learn
   from every sample. Is that lead **model capacity** or **continuous
   supervision**?
3. Does the proactive trigger help a **state-of-the-art** base learner, or was it
   only compensating for a weak one?

## Setup

All on CIC-IDS2017, 10 seeds.

- **Alternative reactive detectors** on the same HAT: `B_adwin` (ADWIN),
  `B_kswin` (KSWIN), in place of Page-Hinkley.
- **Continuous ensembles**: `ARF` (Adaptive Random Forest) and `SRP` (Streaming
  Random Patches) learning on *every* sample, managing drift internally, with no
  external trigger.
- **Budget-matched ensembles** (`ARF_sched`, `SRP_sched`): the same ensembles
  restricted to the reactive arm's regime — adaptation only inside triggered
  windows on the shared schedule.
- **PACE on the ensembles** (`ARF_early`, `SRP_early`): the same ensembles driven
  by the proactive dual Page-Hinkley over the covariate-shift monitor.

`make run-sota run-sota-matched run-early-ensemble`.

## Results

### 1. Detector choice is not the lever

| arm | overall local F1 | adaptations |
|---|---:|---:|
| reactive, Page-Hinkley | 0.567 | 233 |
| reactive, ADWIN | 0.527 | 340 |
| reactive, KSWIN | 0.454 | 138 |

ADWIN and KSWIN are erratic — they over- and under-fire respectively — and both
land *below* plain Page-Hinkley on sustained F1. The reactive baseline is not a
straw man.

### 2. The ensemble lead is capacity, not supervision

| arm | model | overall local F1 | adaptations |
|---|---|---:|---:|
| reactive | HAT | 0.567 | 233 |
| PACE | HAT | 0.599 | 169 |
| budget-matched | ARF | 0.717 | 233 |
| budget-matched | SRP | 0.737 | 233 |
| continuous | ARF | 0.772 | ~2M |
| continuous | SRP | 0.811 | ~2M |

![Budget-matched SOTA](../figures/fig10_budget_matched.png)

Two clean contrasts fall out:

- **Model capacity, at matched budget (233 adaptations):** ensembles 0.72–0.74 vs
  HAT-based 0.57–0.60 → **+0.13–0.16**.
- **Continuous supervision (233 → ~2M updates):** ARF 0.717→0.772, SRP
  0.737→0.811 → only **+0.05–0.08**.

So sparse triggered adaptation is label-efficient: the ensembles keep ~90–95% of
their continuous-learning quality on ~233 adaptations instead of ~2M.

### 3. The trigger transfers to the strongest learners

| base learner | reactive | PACE | Δ | adaptations | labels % |
|---|---:|---:|---:|---|---|
| HAT | 0.567 | 0.599 | **+0.032** | 233 → 169 | 25.5 → 29.1 |
| ARF | 0.717 | 0.747 | **+0.030** | 233 → 110 | 25.5 → 25.1 |
| SRP | 0.737 | 0.768 | **+0.032** | 233 → **90** | 25.5 → **21.3** |

`labels` is the share of the evaluated stream learned from: a firing opens a
5,000-sample window and a firing inside an open window restarts it, so the
reactive schedule's clustered firings cost fewer labels than their count
suggests.

![Base-learner × trigger 2×2](../figures/fig11_early_ensemble.png)

Against the shared schedule the proactive trigger improves **every** base
learner with **fewer** adaptations, at the same label budget on ARF and a
smaller one on SRP: **SRP + PACE = 0.768 sustained F1 on 90 adaptations and 21%
of the labels**, approaching continuous SRP's 0.811 on all of them.

**That is not yet a test of the monitor.** The shared schedule replays HAT's
error, not the ensemble's own. Run live on each ensemble with no monitor
(`ARF_pht` / `SRP_pht`), the same error test reaches **0.749 (ARF) and 0.767
(SRP) on 13.5% and 8.9% of the labels** — the whole lift, at fewer labels. On
this dataset the ensembles' gain over `*_sched` is the gain of a live over a
replayed trigger; see [experiment 6](06-timing-vs-frequency.md#part-c--live-reactive-control-new-runs).

## Interpretation

Choosing a stronger base learner buys more than any change to the trigger does.
The practical recipe on this dataset is **ensemble + a live, gated error
trigger**: the covariate-shift monitor adds nothing measurable on top of it for
the ensembles here (experiment 6, Part C).

## Caveats

- The per-cell Δ is modest (~+0.03), and for the ensembles it is the live
  trigger, not the monitor; [experiment 6](06-timing-vs-frequency.md) subjects
  it to paired confidence intervals across three datasets and to the live
  control.
- The ensembles report no drift events by construction (they self-adapt
  internally); that is not a failure to detect.
- ARF and especially SRP are **expensive**: SRP's p95 latency is ~11 ms/sample
  against sub-ms for the HAT arms, at ~130 samples/s — see
  [experiment 4](04-resources-sensitivity.md).
- The monitor's anticipation recall is modest (~0.23); a higher-recall monitor
  could widen the gap, and remains an open direction.
