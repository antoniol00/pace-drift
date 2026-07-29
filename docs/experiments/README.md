# Experiments

The evidence behind **PACE** (*Proactive Adaptation via Covariate-shift
Estimation*), in the order the questions build on each other. Phase names are the
implementation-level identifiers: PACE is `C_early` on the HAT and
`ARF_early` / `SRP_early` on the ensembles, measured against the reactive
baseline (`B`, `*_sched`) and the frequency-matched blind clock (`*_freq`).

Each document states the **question**, the **setup**, the **results** (numbers +
figure), an interpretation, and the **caveats**. Real-data numbers are over 10
seeds; the comparator experiment uses 20, because its outcome is bimodal. The
primary quality metric is **overall local F1** — the mean F1 over valid rolling
windows, excluding windows with no positive label rather than scoring them zero —
because a single final-window F1 is noisy and a cumulative mean washes out the
drift-dense regions where strategies actually diverge.

| # | Document | Question | Headline |
|---|---|---|---|
| 0 | [Core result](00-core-result.md) | Does an unsupervised trigger beat retraining on error? | **Yes — higher sustained F1 on fewer adaptations, with real anticipation lead times** |
| 1 | [Frequency control](01-frequency-control.md) | Is the gain *when* it adapts or *how often*? | **Timing** — a blind clock at the same count falls *below* the reactive baseline |
| 2 | [Monitor ablation](02-monitor-ablation.md) | How cheap can the shift signal be? | Cheap **linear PCA nearly matches** the autoencoder; a trivial mean-shift does not |
| 3 | [Streaming SOTA](03-sota-baselines.md) | Capacity, supervision, or the trigger? | The ensemble lead is **capacity**, not continuous supervision — and the trigger improves every base learner at a *lower* budget |
| 4 | [Cost & sensitivity](04-resources-sensitivity.md) | Is it deployable? Is the operating point fragile? | **Sub-ms** typical latency; lead time stable across a wide threshold band |
| 5 | [External validity](05-external-validity.md) | Does it hold beyond CIC-IDS2017? | **Positive in all 9 (dataset × learner) cells** on CIC-IDS2018 and UNSW-NB15 |
| 6 | [Timing vs frequency](06-timing-vs-frequency.md) | Does it survive inferential scrutiny, and is it timing or frequency? | **7/9 cells' CIs exclude 0**, aggregate **+0.043 [+0.021, +0.065]**; timing on CIC-IDS2018, frequency on UNSW-NB15 |
| 7 | [Unsupervised comparators](07-unsupervised-comparator.md) | Do published unsupervised detectors do the job as well? | **Within noise — and that is the point.** Every trigger, ours included, collapses the loop on 20–60% of runs, through a verified mechanism no detection metric reveals |
| 8 | [Controlled stream](08-synthetic-timing.md) | Does the core claim hold where drift is known by construction? | **+0.213 [+0.188, +0.238]** at an exactly matched adaptation budget |

## The argument in one paragraph

Adapting a streaming intrusion detector *when an unsupervised monitor anticipates
drift* beats adapting when its error rises, at a lower adaptation budget
(experiment 0). The benefit is the **timing**, not the frequency: a blind clock
firing equally often does worse than not trying at all (experiments 1, 6, 8). The
signal itself can be cheap — linear PCA suffices (experiment 2) — and the policy
is **model-agnostic**, improving state-of-the-art ensembles as well as a single
tree (experiment 3), across three datasets (experiments 5, 6). The final
experiment adds the criterion that should govern this whole family of methods:
judge a trigger by **what it does to the adaptation loop**, not by how well it
detects — and, measured that way, this kind of loop is **fragile** for every
trigger we tested, ours included (experiment 7).

## Reproducing

See the [top-level README](../../README.md#7-reproducing-the-results) for the
exact commands, or `make run-anchor`, `make run-2018`, `make run-unsw`,
`make run-comparators`, then `make analyse figures`.
