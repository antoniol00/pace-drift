# Experiment 0 — The core result: reactive vs proactive adaptation

## Question

An intrusion detector that retrains when its error rises inherits the delay of
labelled feedback. Can an **unsupervised** monitor over the incoming traffic
trigger adaptation *earlier*, and does adapting earlier actually pay?

## Setup

CIC-IDS2017, ~2M samples, HAT base learner, 10 seeds.

- **Reactive baseline (`B`)** — Page-Hinkley over the classification error opens a
  short training window when the error degrades. It replays a shared drift
  schedule (233 fire indices from a pass-0 run) so per-event metrics can be
  paired across arms.
- **PACE (`C_early`)** — schedule-exempt by construction: its claim is
  *anticipation*, which a fixed schedule would erase. A dual Page-Hinkley watches
  both the classification error and the reconstruction error of an unsupervised
  monitor over the unlabelled input; either can open the window.

Both confine learning to short triggered windows, so neither sees continuous
supervision. `make run-reactive run-pace`.

## Results

| arm | adaptations | final F1 | overall local F1 |
|---|---:|---:|---:|
| reactive (`B`) | 233 | 0.788 | 0.567 |
| **PACE (`C_early`)** | **169** | **0.986** | **0.599** |

Overall local F1 is the primary metric — the mean F1 over valid rolling windows,
excluding windows with no positive label rather than scoring them zero. The
single final-window F1 is noisy and a cumulative mean washes out precisely the
drift-dense regions where the arms diverge.

PACE reaches higher sustained quality with **fewer** adaptations, not more.

### Where quality holds and where it diverges

![Local F1 across the stream](../figures/fig1_f1_trajectory.png)

In the easy stretch (~0.6–1.0M) the arms are close. The separation is
concentrated in the drift-dense second half, which is what the aggregate hides
and why the segment view is reported alongside it.

### The monitor genuinely anticipates

![Lead time and cost/benefit](../figures/fig3_early_detection.png)

Anticipation lead times are strictly positive across every seed (pooled mean ~7K
samples): the monitor fires *before* the classification error responds. That is
the mechanism the rest of the experiments interrogate.

## Caveats

- The advantage is **region-specific** — concentrated in the drift-dense second
  half, not uniform across the stream.
- Anticipation **recall is modest** (~0.23 at the operating point): the monitor
  foresees about a quarter of the drifts it could. Its value comes from the ones
  it does catch, not from catching all of them
  ([experiment 4](04-resources-sensitivity.md)).
- `no_recovery` is high (~0.66–0.71) for *both* arms — a property of this
  stream's dense, overlapping drifts rather than a failure of either strategy.

**What this does not yet establish:** whether the gain comes from *when* PACE
adapts or merely from *how often* ([experiment 1](01-frequency-control.md)),
whether the monitor needs to be anything more than a cheap statistic
([experiment 2](02-monitor-ablation.md)), and how any of it compares to a modern
streaming ensemble ([experiment 3](03-sota-baselines.md)).
