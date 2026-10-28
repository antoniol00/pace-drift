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

HAT base learner, **10 seeds**, with the PCA monitor (the original run, which used 5
seeds) and with the autoencoder (added in the third review round, so the controlled
check uses the paper's primary monitor and the real-data seed count). Layout: 20K
pretraining with one known concept at 8K-12K; 40 episodes of 1,000 samples, one
every 7,000 samples (6,000 benign between), 14.7% attack overall; monitor checks
every 500 samples. Run with `--monitor pca|ae`; the AE runs live under
`results/runs/synthetic_timing_ae/`.

## Results

| arm | adaptations | local F1, PCA | local F1, AE |
|---|---:|---:|---:|
| reactive (error test, live) | 87 | 0.822 | 0.822 |
| blind periodic clock | 142 / 143 | 0.588 | 0.549 |
| **PACE** | **143** (87 error + 56 monitor) | **0.841** | **0.837** |

**Timing effect (PACE − blind clock, paired by seed at matched budget):
PCA +0.254 [+0.106, +0.401], AE +0.288 [+0.157, +0.419].** One clock seed per
monitor collapses entirely, which widens the intervals.

**What the monitor adds over the live error test: PCA +0.019 [+0.002, +0.037],
AE +0.015 [−0.002, +0.032].** PACE's error test fires as often as the reactive
baseline (87); the monitor fires ~1.4 times per episode (onset, and often the
return to benign — the recon PHT is two-sided). As on real traffic, almost all of
the advantage over the blind clock is already in the live error test.

The 5-seed PCA run reported +0.213 [+0.188, +0.238]; it is superseded.

## Interpretation

The core claim reproduces on a stream where nothing about the drift is
accidental: informed placement beats a blind clock at an identical adaptation
budget, by a wide margin. Note also that the blind clock lands **well below the
reactive baseline** (0.588 vs 0.822) despite adapting more often — the same
pattern seen on CIC-IDS2017 in [experiment 1](01-frequency-control.md) and on
CIC-IDS2018 in [experiment 6](06-timing-vs-frequency.md). Adapting at the wrong
moments is not merely wasted effort; it is actively harmful, because the training
window lands on a distribution the model will not be tested on.

## Caveats

- The margin here (+0.25 to +0.29) is far larger than on real data (+0.038 to +0.048 on
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
