# Experiments

The evidence behind **PACE** (*Proactive Adaptation via Covariate-shift
Estimation*), in the order the questions build on each other. Phase names are the
implementation-level identifiers: PACE is `C_early` on the HAT and
`ARF_early` / `SRP_early` on the ensembles, measured against the reactive
baseline (`B`, `*_sched`), the frequency-matched blind clock (`*_freq`) and the
live reactive control (`*_pht`).

Each document states the **question**, the **setup**, the **results** (numbers +
figure), an interpretation, and the **caveats**. Real-data numbers are over 10
seeds; the comparator experiment uses 20, because its outcome is bimodal. The
primary quality metric is **overall local F1** — the mean F1 over valid rolling
windows, excluding windows with no positive label rather than scoring them zero —
because a single final-window F1 is noisy and a cumulative mean washes out the
drift-dense regions where strategies actually diverge.

| # | Document | Question | Headline |
|---|---|---|---|
| 0 | [Core result](00-core-result.md) | Does an unsupervised trigger beat retraining on error? | **Yes, against the replayed schedule — higher sustained F1 on fewer adaptations at a similar label budget, with real anticipation lead times** |
| 1 | [Frequency control](01-frequency-control.md) | Is the gain *when* it adapts or *how often*? | **Timing** — a blind clock at the same count falls *below* the reactive baseline |
| 2 | [Monitor ablation](02-monitor-ablation.md) | How cheap can the shift signal be? | Cheap **linear PCA nearly matches** the autoencoder; a trivial mean-shift does not |
| 3 | [Streaming SOTA](03-sota-baselines.md) | Capacity, supervision, or the trigger? | The ensemble lead is **capacity**, not continuous supervision; the trigger beats the replayed schedule on every learner, but for the ensembles a live error test alone does as well |
| 4 | [Cost & sensitivity](04-resources-sensitivity.md) | Is it deployable? Is the operating point fragile? | **Sub-ms** typical latency; lead time stable across a wide threshold band |
| 5 | [External validity](05-external-validity.md) | Does it hold beyond CIC-IDS2017? | **Positive in all 9 (dataset × learner) cells** against the replayed schedule; label efficiency is 2017-specific |
| 6 | [Timing vs frequency](06-timing-vs-frequency.md) | Does it survive inferential scrutiny, is it timing or frequency, and is it the monitor? | Aggregate **+0.043 [+0.021, +0.065]** = timing **+0.042** + frequency +0.001; timing on 2017/2018, frequency on UNSW. Against a **live** error test the monitor adds **+0.021 [−0.002, +0.045] — not resolved** |
| 7 | [Unsupervised comparators](07-unsupervised-comparator.md) | Do published unsupervised detectors do the job as well? | **Within noise — and that is the point.** Every stream-responsive trigger on HAT, ours and the live error test included, collapses the loop on 20–60% of runs; ensembles do not (see [09](09-review-round-3.md)) |
| 8 | [Controlled stream](08-synthetic-timing.md) | Does the core claim hold where drift is known by construction? | Timing **+0.254 / +0.288** (PCA / AE, 10 seeds) at an exactly matched budget; the monitor adds only ~+0.02 over the live error test |
| 9 | [Third review round](09-review-round-3.md) | Do the claims survive clustered intervals, 20 seeds, a metric that sees benign windows, a converged D3 and stability controls? | Timing aggregate **[−0.003, +0.084]** under a hierarchical bootstrap (per-cell evidence carries it); HAT anchor vs schedule **+0.000** at 20 seeds; collapse hits **every stream-responsive trigger on HAT, incl. the live error test (8/20)**, needs feedback, and spares the ensembles |

## The argument in one paragraph

Adapting a streaming intrusion detector *when an unsupervised monitor anticipates
drift* beats replaying a reactive schedule (experiment 0), on every learner and
dataset (experiments 3, 5). The benefit is **timing**, not frequency: a blind
clock firing equally often does worse (experiments 1, 6, 8). But the last control
narrows the claim: run **live** on each learner's own predictions, the error test
PACE runs alongside its monitor matches it on CIC-IDS2017 and CIC-IDS2018 with
fewer labels, and the monitor's own contribution is not resolved (experiment 6,
Part C) — informed placement matters, but on real traffic with immediate labels
it comes mostly from the live error test. The signal itself can be cheap —
linear PCA suffices (experiment 2). The final
experiment adds the criterion that should govern this whole family of methods:
judge a trigger by **what it does to the adaptation loop**, not by how well it
detects — and, measured that way, this kind of loop is **fragile** for every
trigger we tested, ours included (experiment 7).

## Reproducing

See the [top-level README](../../README.md#7-reproducing-the-results) for the
exact commands, or `make run-anchor`, `make run-2018`, `make run-unsw`,
`make run-comparators`, then `make analyse figures`.
