# Experiment 7 — Unsupervised comparators and closed-loop stability

> **Revision history — read this first.** The first version of this experiment
> (5 seeds, 2026-07-24) reported *AE 0/5 collapses vs D3 4/5, Fisher p=0.048* and
> concluded that our autoencoder monitor was uniquely stable. **Extending to 20
> seeds on 2026-07-28 refuted that ranking.** The section below reports the
> 20-seed numbers; §"What the extension changed" states exactly what was
> retracted and why. The 5-seed asymmetry was sampling noise on a bimodal
> outcome.

## Question

Our monitor belongs to the family of **unsupervised / feature-space drift
detectors** (see `docs/paper/related-work.md`). The decisive reviewer question is
therefore:

> Given the *identical* when-to-adapt policy, does a **published** unsupervised
> drift detector do the job as well as our reconstruction monitor?

## Setup

**Comparators.** Two published detectors, chosen because their principles differ
from ours *and* from each other:

- **D3** — the Discriminative Drift Detector (Gözüaçık, Büyükçakır, Bonab, Can,
  *Unsupervised Concept Drift Detection with a Discriminative Classifier*, CIKM
  2019): the most-cited named unsupervised detector. Trains a discriminative
  classifier to separate an old window from a new one; separability (AUC) above a
  threshold means the input distribution changed.
- **HDDDM** — Hellinger Distance Drift Detection (Ditzler & Polikar, IEEE CIDUE
  2011): mean per-feature Hellinger distance between the histograms of the two
  windows. It is the informative contrast to D3, because a bin-by-bin
  distribution distance responds to a change in *dispersion at constant mean*,
  which a linear discriminator cannot see (both properties pinned in
  `tests/test_early.py`).

**Implementation.** `HDDDMMonitor`, `D3Monitor` and `D3NativeMonitor` in
`src/driftgan/early/shift_monitors.py`. river 0.23 ships neither (its detectors
are univariate over the error stream, i.e. supervised), so both are implemented
directly. Deviation carried by every arm: logistic regression by L2-regularised
batch gradient descent (no scikit-learn dependency).

**Two families of arm, answering different questions.**

- *Budget-matched* (`C_early`, `C_early_pca`, `C_early_hdddm`, `C_early_d3`,
  `C_early_mean`): each monitor's `PHT_recon` is calibrated on its cached score
  series so that it fires as often as the autoencoder (58 fires;
  `scripts/calibrate_monitor.py`). Only the shift statistic differs, so a
  difference here is a property of the *signal*.
- *Native* (`C_early_d3_native`, `C_early_d3_sized`): D3 keeps its **own**
  sliding window of size `w(1+ρ)`, its own test cadence and its own AUC
  threshold, declaring drift itself. These are deliberately **not**
  budget-matched — the fire count is whatever the published rule chooses, and
  that count is part of the result. They exist to foreclose the objection that
  the matched arm was a weakened D3.

## Results (20 seeds, budget-matched arms)

Full tables: `results/report/comparator/analysis.md`
(`scripts/comparator_analysis.py`); figure `docs/figures/fig15_closed_loop_stability.png`.

| monitor | collapsed seeds | Fisher vs AE | FPR med / worst | train atk% |
|---|---:|---:|---:|---:|
| autoencoder (ours) | 4/20 | — | 0.006 / 0.662 | 0.26 |
| HDDDM (CIDUE'11) | 5/20 | 1.000 | 0.005 / 0.603 | 0.31 |
| PCA linear | 6/20 | 0.716 | 0.006 / 0.689 | 0.37 |
| D3 (CIKM'19) | 10/20 | 0.096 | 0.225 / 0.595 | 0.36 |
| mean-shift (floor) | 12/20 | **0.022** | 0.609 / 0.700 | 0.46 |

*("collapsed" = cumulative accuracy < 0.90; the majority class alone scores
~0.80. `train atk%` = attack fraction of the samples actually fed to the learner
inside triggered windows; the stream's own rate is 0.20.)*

Pooled, the autoencoder against all four alternatives is 4/20 vs 33/80,
**p = 0.119**. Paired Wilcoxon on cumulative accuracy: only mean-shift separates
from the AE (p = 0.004); D3 p = 0.37, PCA p = 0.28, HDDDM p = 0.99.

## What survives, and what does not

**Does not survive.** *"The autoencoder is uniquely stable."* It is not. It
collapses on 4/20 seeds, HDDDM is statistically indistinguishable from it
(5/20, p = 1.00) and so is PCA (6/20, p = 0.72). D3 is worse in point estimate
(10/20) but not significantly so at this n. The only arm that separates is the
trivial mean-shift floor.

We checked that this is genuine seed variance rather than an artifact of the
seed blocks having been produced in different batches: collapse rates by block
(seeds 0–4 / 5–9 / 10–19) show no block effect (Fisher p = 0.58).

**Survives.**

1. **Closed-loop collapse is real, common and unreported.** *Every* arm,
   including ours, drives the learner into a degenerate regime on 20–60% of
   seeds. That an adaptation policy of this kind is this fragile is itself the
   finding, and we have not found it reported in the unsupervised-drift
   literature.
2. **The mechanism**, verified rather than hypothesised (below).
3. **The signal matters at the bottom end.** The trivial mean-shift floor is
   significantly worse, so this is not "any statistic will do".
4. **`final_f1` masks collapse** — the metric lesson is unaffected.
5. **Native D3 at the published defaults over-fires catastrophically** (below).

## Mechanism — two distinct failure modes

Do not conflate them; the `train atk%` and `FPR` columns separate them.

**Mode 1 — attack-biased training** (the budget-matched comparators). Badly
placed triggers open training windows over attack-dense stretches. The learner is
fed well above the stream's 20% attack rate, its prior skews toward ATTACK, the
false-positive rate on benign traffic explodes (worst case 0.60–0.70, i.e. over
half of benign traffic flagged — operationally unusable), the class-PHT then
fires constantly, and that opens more badly placed windows. Across monitors
`train atk%` (0.26 → 0.31 → 0.37 → 0.36 → 0.46) broadly tracks the collapse count
(4 → 5 → 6 → 10 → 12); the tracking is monotone except for D3, whose pooled
figure is diluted by its healthy seeds.

**Mode 2 — adaptation saturation** (native D3 at published defaults). It fires so
often that it is essentially always inside a training window: `train atk%` sits
at the base rate (0.20) and the FPR stays low, but the model degenerates toward
the majority class. The cost is the ~76× adaptation budget, not misplacement.

**Root causes.** A unit test pins D3's: its *linear* discriminator detects mean
shifts (AUC 0.93–1.00) but is **blind to a variance-only change (AUC 0.50)**,
which reconstruction error and a Hellinger histogram distance both capture. On
flow features, attack onsets often change dispersion and correlation structure
rather than means. Independently — and this drives mode 2 — at the released
defaults `w=100, ρ=0.1` the "new" window holds 10 samples, so the 2-fold CV AUC
is estimated from ~5 positives per fold and crosses τ on ~5% of tests over a
*perfectly stationary* stream (`test_d3_native_false_alarms_at_published_defaults`).
D3's own benchmarks are far narrower than 70-feature flow records; this is a
transfer limitation of the published settings, not of the method's principle.

## Native D3: the objection is foreclosed

| arm | adaptations | collapsed |
|---|---:|---:|
| D3 native, published defaults (w=100, ρ=0.1, τ=0.7) | ~12 300 | 9/20 |
| D3 native, window sized for the data (w=1000, ρ=0.2) | ~1 280 | 7/20 |
| *(reference)* autoencoder, matched budget | 188 | 4/20 |

Giving D3 its own machinery does not rescue it. The shared-protocol arm was the
**charitable** version of D3, not a weakened one — which answers the reviewer
objection that motivated these runs.

## Consequence: experiment 2's revision is reverted

Experiment 2 originally concluded "a cheap linear PCA monitor suffices; the
autoencoder is over-engineered." On 2026-07-24 that was revised on the strength
of the 5-seed stability result. **At 20 seeds the revision does not hold: PCA
matches the AE on closed-loop stability too (6/20 vs 4/20, p = 0.72), so
experiment 2's original conclusion stands and the paper should lead with the
cheap linear monitor.** The `final_f1`-masks-collapse caveat remains true — it
just does not rescue the autoencoder.

## Honest scope

These are our implementations, and the native arms are not budget-matched by
construction. The claim is about what these signals do to *this* adaptation loop
on flow data — not a verdict on the detectors in general. Read the collapse
counts as effect sizes on a modest number of seeds: **the ordering is the result;
individual pairwise p-values at this n are not.** A faithful `(w, ρ)` D3 driven
by scikit-learn's solver, and a third comparator, remain future work.

**Methodological lesson worth carrying forward:** a bimodal outcome needs far
more seeds than a mean does. Five seeds produced a clean-looking, significant and
*wrong* ranking. Do not report a collapse-count asymmetry below n = 20.

## Reproducing

```bash
uv run python run_experiment.py --phase C_early_hdddm --seeds 0,1,2,3,4 --n-jobs 5
uv run python run_experiment.py --phase C_early_d3 --seeds 0,1,2,3,4 --n-jobs 5
uv run python run_experiment.py --phase C_early_d3_native --seeds 0,1,2,3,4 --n-jobs 5
uv run python scripts/calibrate_monitor.py C_early_hdddm --reference C_early
uv run python scripts/comparator_analysis.py
```
