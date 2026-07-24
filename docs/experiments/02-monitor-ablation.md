# Experiment 2 — Monitor ablation (is the autoencoder justified?)

> **⚠️ Conclusion revised by [experiment 7](07-unsupervised-comparator.md).** This
> page judges the monitors on *final F1*, which masks mid-stream collapse: the PCA
> arm's seed 1 shows `final_f1 = 0.948` while its cumulative accuracy is `0.444`
> and its false-positive rate `0.657`. Judged on **closed-loop stability**, only
> the autoencoder is stable in 5/5 seeds (PCA and mean-shift collapse in 2/5).
> The autoencoder *is* justified — by stability, not by detection quality. Read
> this page together with experiment 7.

## Question

The generative autoencoder is the novelty of C_early. A reviewer will ask whether
a *generative* model is doing anything a **cheap covariate-shift statistic** could
not. If a trivial detector anticipates just as well, the autoencoder is not
justified.

## Setup

Four cheap monitors sharing the autoencoder's **exact protocol** (frozen
reference frame; score before updating the reference so a shift is a *leading*
signal): linear **PCA reconstruction**, per-feature **KS** distance, RBF **MMD**,
and a trivial **mean-shift** (L2 distance of standardized feature means).
`src/driftgan/early/shift_monitors.py`.

Two levels of comparison:
1. **Offline detectability frontier** (`scripts/monitor_ablation.py`, fig7): one
   pass per seed collects each monitor's score series; each is z-scored and swept
   with a threshold detector to trace anticipation recall vs false-positive rate,
   paired against the same class-drift fires.
2. **Full pipeline** (fig9): run the whole C_early loop with the monitor swapped
   in — phases **C_early_mean** and **C_early_pca**, PHT_recon recalibrated to the
   autoencoder's fire count. **This is the authoritative test** — it measures what
   actually matters, the downstream detection quality.

## Results

### Offline frontier — misleading on its own

![Monitor detectability frontier](../figures/fig7_monitor_ablation.png)

Under a common threshold detector the autoencoder does **not** dominate; a trivial
mean-shift matches or exceeds it, and no monitor clears recall ≈ 0.5. This
*suggested* the AE was unnecessary — but the offline frontier measures raw signal
separability, not the deployed system.

### Full pipeline — the authoritative test

![Monitor swapped in the full pipeline](../figures/fig9_monitor_pipeline.png)

| Monitor in the C_early pipeline | final F1 | overall local F1 |
|---|---|---|
| Autoencoder (generative) | 0.986 | 0.595 |
| **PCA reconstruction (linear)** | **0.969** | **0.584** |
| mean-shift (trivial) | 0.778 | 0.485 |

## Interpretation

**A reconstruction signal is needed, but the generative model is over-engineered.**
In the deployed loop, where the monitor's score dynamics interact with the
adaptation trigger, a **linear PCA reconstruction monitor nearly matches the
autoencoder** (0.969 vs 0.986 final), while a **trivial mean-shift is clearly
worse** (0.778). So you need a monitor that captures multivariate structure
(reconstruction error), but it need not be generative or non-linear — cheap PCA
suffices and is far lighter.

**Methodological lesson:** the offline signal-separability proxy was misleading
(it favoured mean-shift); monitor changes must be confirmed in the full pipeline,
where *adaptation timing* — not raw separability — is what pays off.

## Caveats

- The offline frontier z-scores each series over the whole run (uses future
  information) — fine for characterising detectability, but it is explicitly *not*
  the deployed detector, which is why the pipeline runs are the headline.
- `C_early_ks` / `C_early_mmd` phases exist but were not run in the full pipeline;
  KS/MMD looked no better than PCA offline and are more expensive.
