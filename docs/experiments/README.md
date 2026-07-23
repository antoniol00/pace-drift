# Experiments

Each document states the **question**, the **setup**, the **results** (numbers +
figure), an honest **interpretation**, and the **caveats**. All numbers are the
mean over 5 seeds on the full ~2M-sample CICIDS2017 stream. The primary quality
metric is **overall local F1** — the mean F1 over valid rolling windows (windows
with no positive label are excluded, not counted as zero) — because the single
final-window F1 is noisy and the cumulative mean washes out the drift-dense
regions where strategies actually diverge.

| # | Document | Question | Headline |
|---|---|---|---|
| — | [Core run](00-core-run.md) | Do GANs improve drift *adaptation*? Does a generative monitor *anticipate*? | GAN adaptation is **null**; the early-detection monitor is the **positive** result |
| 1 | [Frequency control](01-frequency-control.md) | Is C_early's gain *when* it adapts or *how often*? | **Timing, not frequency** — blind periodic adaptation falls *below* the baseline |
| 2 | [Monitor ablation](02-monitor-ablation.md) | Is the generative autoencoder justified as the monitor? | **Over-engineered** — linear PCA nearly matches it; a trivial mean-shift does not |
| 3 | [SOTA baselines](03-sota-baselines.md) | How does C_early compare to off-the-shelf streaming SOTA? | ARF/SRP win on raw quality — but with ~2M continuous supervised updates |
| 3b | [Budget-matched SOTA](03b-budget-matched.md) | Is the SOTA lead model capacity or continuous supervision? | Mostly **model capacity** — ensembles lead even at the baseline's sparse budget |
| 3c | [Ensemble + anticipation](03c-early-ensemble.md) | Does the early trigger help a SOTA base learner? | **Yes — it improves every base learner, more for the stronger ones, with fewer adaptations** |
| 5/6 | [Resources & sensitivity](04-resources-sensitivity.md) | Real-time cost? Is the operating point robust? | Sub-ms latency with ~1 s training bursts; lead time stable across a wide threshold band |
| 4 | [External validity](05-external-validity.md) | Does the 3c result hold beyond CICIDS2017 (CIC-IDS2018, UNSW-NB15)? | **Direction holds in all 9 dataset × base-learner cells — but label efficiency and the gains-scale-with-strength pattern are 2017-specific** |

## The story in one paragraph

Generative **augmentation** for drift adaptation does not work (C/C_cond/C' are
null-to-negative against the reactive baseline). What does work is the
**proactive early-detection trigger**: adapting *when a covariate-shift monitor
anticipates drift* rather than after error rises. Its value is the *timing*
(experiment 1), the monitor can be cheap (experiment 2), and — most importantly —
the idea is **model-agnostic**: it improves not just the weak single tree but the
state-of-the-art ensembles too (experiment 3c), at a fraction of the adaptation
budget of continuous learning (experiments 3/3b). The transferable contribution
is a cheap *when-to-adapt* signal, not a generative augmenter.

## Reproducing

See the [top-level README](../../README.md#7-reproducing-the-results) for the
exact commands (`make run-all`, `make run-sota`, `make run-sota-matched`,
`make run-early-ensemble`, and the three figure scripts under `scripts/`).
