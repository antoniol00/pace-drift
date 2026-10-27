# Experiment 1 — Frequency-matched control

## Question

PACE beats the reactive baseline. But it adapts on a schedule of its own
choosing, so the first objection is obvious: *is the gain because of **when** it
adapts, or simply because it adapts a **different number of times**?*

## Setup

Phase **`C_freq`**: adapt on a **blind periodic clock** — fire every 12,000
samples regardless of any error or drift signal — at PACE's count
(162 adaptations against PACE's 169 over the ten seeds, ~4% fewer). Same base learner, same
training-window mechanics, no monitor. This isolates adaptation **frequency**
from adaptation **timing**. `make run-freq`.

One fact weakens the objection before the control is even run: the reactive
baseline already fires **more** than PACE (233 vs 169), so "it retrains more"
was never the explanation. `C_freq` settles it cleanly.

## Results

Overall local F1 by stream segment, 10 seeds:

| Segment | reactive | blind clock (same count) | PACE |
|---|---|---|---|
| 0–0.5M | 0.581 | 0.679 | 0.617 |
| 0.5–1.0M (easy) | 0.916 | 0.803 | 0.890 |
| 1.0–1.5M (drift-dense) | 0.404 | 0.253 | 0.454 |
| 1.5–2.0M (drift-dense) | 0.354 | 0.162 | 0.425 |
| **overall** | **0.567** | **0.473** | **0.599** |

![Frequency-control ablation](../figures/fig6_frequency_ablation.png)

## Interpretation

**Timing, not frequency.** Matching PACE's adaptation *count* with blind periodic
timing does not merely fail to reproduce its advantage — it lands **below the
reactive baseline** (0.473 vs 0.567), collapsing in exactly the drift-dense
regions where PACE gains. In the first segment (from the end of the 50K
pretraining to 0.5M) the blind clock actually scores highest, so its deficit is
concentrated where drift is dense; the overall row pools every valid window,
that segment included. The advantage is therefore about adapting at the right
moments, not about adapting more.

The mechanism is intuitive, and [experiment 7](07-unsupervised-comparator.md)
measures it directly: a blind clock often opens its training window over a benign
or wrong-attack stretch, biasing the learner toward a distribution that is not
the one it will be tested on. An informed window lands on the samples that
matter.

The same comparison holds on a **controlled synthetic stream** whose drift
structure is known by construction, which rules out any explanation resting on a
peculiarity of this capture — see [experiment 8](08-synthetic-timing.md).

## Caveats

- The blind clock's *final-window* F1 is unstable (0.39 mean over ten seeds, ranging from 0.00 to 1.00).
  This is exactly why the analysis uses overall and per-segment local F1 rather
  than the last window.
- The count match is to PACE's mean; per-seed counts vary slightly.
