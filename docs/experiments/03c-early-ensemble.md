# Experiment 3c — Ensemble + proactive early detection (the constructive result)

## Question

C_early's proactive early-detection trigger helps the weak HAT (+0.02 over the
reactive baseline). [Experiment 3b](03b-budget-matched.md) showed a stronger base
learner helps much more. These two ideas are **orthogonal** — one is *when to
adapt*, the other is *what model to adapt*. The missing cell of the 2×2:

> Does the proactive early trigger improve a **state-of-the-art** base learner, or
> was it only compensating for a weak one?

If the early trigger helps the ensembles too, the contribution is a general,
model-agnostic *when-to-adapt* signal — an **offensive** result, not a defensive
one.

## Setup

Phases **ARF_early** and **SRP_early**: a SOTA ensemble driven by C_early's
proactive **dual (recon + class) PHT** and the same calibrated autoencoder
monitor, learning only in triggered windows. This completes the base-learner ×
trigger 2×2. `make run-early-ensemble SEEDS=0,1,2,3,4`.

## Results

Overall local F1 — reactive vs proactive early trigger, per base learner:

| Base learner | reactive | proactive (early) | Δ | adaptations |
|---|---|---|---|---|
| HAT | 0.573 (B) | 0.595 (C_early) | **+0.022** | 233 → 161 |
| ARF | 0.718 (ARF_sched) | 0.748 (ARF_early) | **+0.030** | 233 → 113 |
| SRP | 0.733 (SRP_sched) | **0.769** (SRP_early) | **+0.037** | 233 → **91** |

![Base-learner × trigger 2×2](../figures/fig11_early_ensemble.png)

## Interpretation

**The proactive early trigger improves *every* base learner — and more for the
stronger ones (Δ grows 0.022 → 0.030 → 0.037) — while using *fewer* adaptations.**
So the anticipation idea is **not a crutch for a weak HAT**: it lifts the SOTA
ensemble too, and more label-efficiently.

This flips the study from defensive to **constructive**. The transferable
contribution is a cheap, model-agnostic **when-to-adapt** signal (a covariate-shift
monitor feeding a Page-Hinkley trigger), not a generative augmenter. The best
single configuration:

> **SRP + proactive early detection = 0.769 sustained F1 with only 91
> adaptations** — approaching continuous SRP's 0.811 (which needs ~2M supervised
> updates).

**Recommended paper framing:** *proactive generative-monitor-triggered adaptation
improves streaming classifiers across base learners, at a fraction of the
adaptation budget.*

## Caveats

- The absolute Δ is modest (+0.02 to +0.04 sustained F1); the strength of the
  result is its **consistency and direction** (helps all models, scales with model
  strength, costs fewer adaptations), not a large single number.
- The monitor still has modest anticipation recall (~0.23); a higher-recall
  monitor could widen the gap — an open direction.
- Cost: ARF_early/SRP_early add the AE monitor's memory (~1.3 GB peak) and
  training bursts on top of the ensemble.
