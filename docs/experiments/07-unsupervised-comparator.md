# Experiment 7 — Unsupervised comparator (D3) and closed-loop stability

## Question

Our monitor belongs to the family of **unsupervised / feature-space drift
detectors** (see `docs/paper/related-work.md`). The decisive reviewer question is
therefore:

> Given the *identical* when-to-adapt policy, does a **published** unsupervised
> drift detector do the job as well as our reconstruction monitor?

## Setup

**Comparator.** D3 — the Discriminative Drift Detector (Gözüaçık, Büyükçakır,
Bonab, Can, *Unsupervised Concept Drift Detection with a Discriminative
Classifier*, CIKM 2019): the most-cited named unsupervised detector. It trains a
discriminative classifier to separate an old window from a new one; separability
(AUC) above a threshold means the input distribution changed. Its principle is
*orthogonal* to reconstruction error, so the comparison is informative.

**Implementation.** `D3Monitor` in `src/driftgan/early/shift_monitors.py`, phase
`C_early_d3`. river 0.23 ships no D3 (its detectors are univariate over the error
stream, i.e. supervised), so it is implemented directly. Documented deviations:
logistic regression by L2-regularised batch gradient descent (no scikit-learn
dependency), and our shared monitor protocol for window bookkeeping instead of
D3's own `(w, ρ)` roll — so every monitor sees identical data at identical check
points. The discriminative principle and the 2-fold stratified AUC are preserved.

**Calibration.** The D3 score series was cached over the full 2017 stream (AUC:
min 0.402, median 0.554, p90 0.663, max 0.848) and `PHT_recon` swept to
`threshold=0.32, delta=0.0005` → 59 fires, matching the autoencoder's 58. **The
arms are fire-count matched; only the shift statistic differs.**

## Results

Full table: `results/report/comparator/analysis.md`
(`scripts/comparator_analysis.py`).

| monitor | local F1 median | collapsed seeds | FPR med / worst | train atk% |
|---|---:|---:|---:|---:|
| autoencoder (ours) | 0.605 | **0/5** | 0.006 / 0.007 | 0.26 |
| PCA linear | 0.620 | 2/5 | 0.006 / 0.657 | 0.38 |
| mean-shift | 0.504 | 2/5 | 0.007 / 0.653 | 0.37 |
| D3 (CIKM'19) | 0.395 | **4/5** | 0.540 / 0.595 | 0.41 |

*(`train atk%` = attack fraction of the samples actually fed to the learner inside
triggered windows; the stream's own rate is 0.20.)*

### The finding is about **stability**, not detection quality

D3-triggered adaptation was **unstable**: 4 of 5 seeds entered a degenerate
feedback loop, while the surviving seed matched our monitor exactly (F1 0.984).
Report per-seed values and the collapse count — a mean ± paired-t CI is
meaningless under this bimodality and would misleadingly read
"indistinguishable" (−0.174 [−0.355, +0.007]).

### Mechanism (verified, not hypothesised)

The collapse is a **closed-loop** failure, and the trigger causes it:

1. Badly placed triggers open training windows over attack-dense stretches. The
   learner is fed **43–47% attack** in collapsed seeds vs **23–27%** in healthy
   ones, against a stream base rate of **20%**.
2. The classifier's prior skews toward ATTACK; the false-positive rate on benign
   traffic explodes to **0.49–0.66** (i.e. over half of benign traffic flagged —
   operationally unusable).
3. High error ⇒ the class-PHT fires constantly (246 vs 113 fires) ⇒ more badly
   placed windows ⇒ runaway.

Across monitors, `train atk%` (0.26 → 0.37/0.38 → 0.41) tracks the collapse count
(0 → 2 → 4) monotonically.

**Why D3 specifically.** A unit test (`tests/test_early.py`) pins the cause: D3's
*linear* discriminator detects mean shifts (AUC 0.93–1.00) but is **blind to a
variance-only change (AUC 0.50)**, which reconstruction error captures. On flow
features, attack onsets often change dispersion and correlation structure rather
than means.

## Consequence: experiment 2's conclusion is revised

Experiment 2 concluded "a cheap linear PCA monitor suffices; the autoencoder is
over-engineered." That conclusion was drawn from **final F1**, which *masks* the
failure: PCA seed 1 has `final_f1 = 0.948` while its cumulative accuracy is
`0.444` and its FPR is `0.657`. Judged on closed-loop stability instead:

> **The autoencoder is justified — not by detection quality (PCA matches it when
> it works) but by stability.** Only the AE keeps the adaptation loop stable in
> 5/5 seeds and the training distribution near the stream's own (0.26 vs 0.20);
> every cheaper statistic intermittently drives the system into a
> catastrophic-false-positive regime.

This does not overturn the paper's thesis — it sharpens it. The claim was always
that **where you place adaptation matters**; this is the strongest evidence yet,
because bad placement does not merely fail to help, it *actively destroys* the
model by biasing what it trains on.

## Honest scope

Ours is an *adaptation* of D3, so this is evidence that a
discriminative-separability signal did not substitute for reconstruction error
**in our pipeline** — not a claim that D3 is a poor detector in general. A
faithful `(w, ρ)` D3 with its native threshold, and HDDDM as a second comparator,
remain future work.

## Reproducing

```bash
uv run python run_experiment.py --phase C_early_d3 --seeds 0,1,2,3,4 --n-jobs 5
uv run python scripts/comparator_analysis.py
```
