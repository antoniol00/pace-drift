# Experiments

The trail behind **PACE** (*Proactive Adaptation via Covariate-shift
Estimation*), in the order the questions were actually asked. Phase names are the
implementation-level identifiers: the PACE policy is `C_early` on the HAT and
`ARF_early` / `SRP_early` on the ensembles, measured against the reactive
baseline (`B`, `*_sched`) and the frequency-matched blind clock (`*_freq`).

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
| 2 | [Monitor ablation](02-monitor-ablation.md) | Is the generative autoencoder justified as the monitor? | On *detection quality* linear PCA nearly matches it — but see experiment 7: **on closed-loop stability the AE is the only arm that never collapses**, so it is justified |
| 3 | [SOTA baselines](03-sota-baselines.md) | How does C_early compare to off-the-shelf streaming SOTA? | ARF/SRP win on raw quality — but with ~2M continuous supervised updates |
| 3b | [Budget-matched SOTA](03b-budget-matched.md) | Is the SOTA lead model capacity or continuous supervision? | Mostly **model capacity** — ensembles lead even at the baseline's sparse budget |
| 3c | [Ensemble + anticipation](03c-early-ensemble.md) | Does the early trigger help a SOTA base learner? | **Yes — it improves every base learner, more for the stronger ones, with fewer adaptations** |
| 4 | [External validity](05-external-validity.md) | Does the 3c result hold beyond CICIDS2017 (CIC-IDS2018, UNSW-NB15)? | **Direction holds in all 9 dataset × base-learner cells — but label efficiency and the gains-scale-with-strength pattern are 2017-specific** |
| 6 | [Timing vs frequency](06-timing-vs-frequency.md) | Does the external-dataset gain survive inferential scrutiny, and is it *when* or *how often*? | **Robust for all ensemble cells (paired CIs + Kappa); timing on CIC-IDS2018 (blind-more-hurts), frequency on UNSW — an empirical split (consistent across ARF/SRP); two pre-specified synthetics could not isolate a structure mechanism** |
| 7 | [Unsupervised comparator](07-unsupervised-comparator.md) | Does a *published* unsupervised detector (D3, CIKM'19) work as well as our monitor under the identical policy? | **No — it destabilizes the loop: 4/5 seeds collapse (FPR 0.49–0.66) vs 0/5 for the AE. Badly placed triggers bias the training set to 43–47% attack (base rate 20%). Revises experiment 2: the AE is justified by *stability*, not detection quality** |
| 5/6 | [Resources & sensitivity](04-resources-sensitivity.md) | Real-time cost? Is the operating point robust? | Sub-ms latency with ~1 s training bursts; lead time stable across a wide threshold band |

## The story in one paragraph

Generative **augmentation** for drift adaptation does not work (C/C_cond/C' are
null-to-negative against the reactive baseline). What does work is the
**proactive early-detection trigger**: adapting *when a covariate-shift monitor
anticipates drift* rather than after error rises. Its value is the *timing*
(experiment 1), the monitor must be chosen for closed-loop *stability* rather than raw
detection quality (experiments 2 and 7), and — most importantly —
the idea is **model-agnostic**: it improves not just the weak single tree but the
state-of-the-art ensembles too (experiment 3c), at a fraction of the adaptation
budget of continuous learning (experiments 3/3b). The transferable contribution
is the *when-to-adapt* policy — and the criterion for judging a trigger is what
it does to the adaptation loop, not how well it detects (experiment 7).

## Reproducing

See the [top-level README](../../README.md#7-reproducing-the-results) for the
exact commands (`make run-all`, `make run-sota`, `make run-sota-matched`,
`make run-early-ensemble`, and the three figure scripts under `scripts/`).
