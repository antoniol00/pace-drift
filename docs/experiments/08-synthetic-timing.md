# Experiment 8 — The core claim on a controlled stream

## Question

Every other experiment here is measured on real traffic, where the drift process
is whatever the capture happened to contain. That leaves one objection open: is
the timing effect a property of the *method*, or of these particular captures?

> On a stream whose drift structure is known by construction, does adapting where
> an unsupervised monitor points beat spending the same adaptation budget
> uniformly?

## Setup

`scripts/synthetic_timing.py` generates the stream, so the drift process is
specified rather than inferred:

- 300K samples, 20 features. Benign traffic is standard normal; an attack concept
  is a tight blob at distance 6 from the origin.
- 8 attack concepts cycle across **40 episodes of 1000 samples**, so the stream
  drifts repeatedly rather than once. One concept appears inside the pretraining
  prefix, so the model starts knowing a single attack and must learn the rest
  online.
- `train_window = 1000` **equals** the episode length. This is the design
  decision that makes the question meaningful at all: with a window much larger
  than an episode, any fire covers any drift and placement cannot matter.
- The blind clock's interval is derived from the proactive arm's own mean fire
  count, so the two arms adapt equally often and differ **only** in placement.

HAT base learner with the cheap PCA monitor (the configuration
[experiment 2](02-monitor-ablation.md) recommends), 5 seeds.

## Results

| arm | adaptations | overall local F1 |
|---|---:|---:|
| reactive (error-triggered) | 86 | 0.821 |
| blind periodic clock | 142 | 0.635 |
| **monitor-driven (PACE)** | **142** | **0.848** |

**Timing effect (PACE − blind clock, paired by seed at matched budget):
+0.213, 95% CI [+0.188, +0.238].**

The fire counts match exactly (142 vs 142), so the gap is attributable to
placement alone.

## Interpretation

The core claim reproduces on a stream where nothing about the drift is
accidental: informed placement beats a blind clock at an identical adaptation
budget, by a wide margin. Note also that the blind clock lands **well below the
reactive baseline** (0.635 vs 0.821) despite adapting more often — the same
pattern seen on CIC-IDS2017 in [experiment 1](01-frequency-control.md) and on
CIC-IDS2018 in [experiment 6](06-timing-vs-frequency.md). Adapting at the wrong
moments is not merely wasted effort; it is actively harmful, because the training
window lands on a distribution the model will not be tested on.

## Caveats

- The margin here (+0.213) is far larger than on real data (+0.038 to +0.048 on
  CIC-IDS2018). A generated stream with clean, well-separated concepts is an
  easier setting than real flow traffic; this experiment establishes that the
  mechanism is real, **not** the effect size to expect in deployment.
- The design fixes `train_window` to the episode length deliberately. Where a
  deployment's window is much wider than its drift episodes, placement will
  matter less by construction.

## Reproducing

```bash
uv run python -m scripts.synthetic_timing --all
```
