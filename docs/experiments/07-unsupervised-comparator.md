# Experiment 7 — Unsupervised comparators and closed-loop stability

> **All numbers here are over 20 seeds.** The collapse outcome is bimodal — a run
> either stays healthy or falls into a degenerate regime, with almost nothing in
> between — so a handful of replicates estimates its rate very badly. Collapse
> counts in this project are held to n ≥ 20 for that reason.

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

| monitor | collapsed seeds | Fisher vs AE | FPR med / worst | train atk% all (healthy / collapsed) |
|---|---:|---:|---:|---:|
| autoencoder (ours) | 4/20 | — | 0.006 / 0.662 | 0.26 (0.25 / 0.41) |
| HDDDM (CIDUE'11) | 5/20 | 1.000 | 0.005 / 0.603 | 0.31 (0.29 / 0.43) |
| PCA linear | 6/20 | 0.716 | 0.006 / 0.689 | 0.37 (0.34 / 0.48) |
| D3 (CIKM'19) | 10/20 | 0.096 | 0.225 / 0.595 | 0.36 (0.29 / 0.42) |
| mean-shift (floor) | 12/20 | **0.022** | 0.609 / 0.700 | 0.46 (0.37 / 0.46) |

*("collapsed" = cumulative accuracy < 0.90; the majority class alone scores
~0.80. `train atk%` = attack fraction of the samples actually fed to the learner
inside triggered windows, median over seeds and then over healthy / collapsed
seeds; the stream's own rate is 0.20.)*

Pooled, the autoencoder against all four alternatives is 4/20 vs 33/80,
**p = 0.119**. Paired Wilcoxon on cumulative accuracy: only mean-shift separates
from the AE (p = 0.004); D3 p = 0.37, PCA p = 0.28, HDDDM p = 0.99.

## What this establishes

**The adaptation loop is fragile, and that is the finding.** *Every* arm —
including ours — drives the learner into a degenerate regime on 20–60% of seeds.
We have not found this reported anywhere in the unsupervised-drift literature,
and it is invisible to the detection metrics that literature does report.

**Our monitor is not more stable than the published detectors.** HDDDM is
statistically indistinguishable from it (5/20 vs 4/20, p = 1.00), as is PCA
(6/20, p = 0.72). D3 is worse in point estimate (10/20) but not significantly so
at this n. The only arm that separates is the trivial mean-shift floor — so the
signal does matter at the bottom end, but not between the sophisticated ones.
This is stated plainly because the criterion, not a ranking, is the contribution.

**Detection metrics do not predict any of it**, and neither does final-window F1
(one PCA seed: final F1 0.948, cumulative accuracy 0.444, FPR 0.657). Trigger
quality has to be measured by what the trigger does to the loop.

Collapse rates are consistent across seed blocks (Fisher p = 0.58), so the spread
is genuine run-to-run variance rather than an artifact of when a batch was
produced.

## Mechanism — two distinct failure modes

Do not conflate them; the `train atk%` and `FPR` columns separate them.

**Mode 1 — attack-biased training** (the budget-matched comparators). Badly
placed triggers open training windows over attack-dense stretches. The learner is
fed well above the stream's 20% attack rate, its prior skews toward ATTACK, the
false-positive rate on benign traffic explodes (worst case 0.60–0.70, i.e. over
half of benign traffic flagged — operationally unusable), the class-PHT then
fires constantly, and that opens more badly placed windows. Within every matched
arm the collapsed seeds were fed 0.41–0.48 attack against 0.25–0.37 for the
healthy seeds. Pooled, `train atk%` (0.26 → 0.31 → 0.37 → 0.36 → 0.46) tracks the
collapse count (4 → 5 → 6 → 10 → 12) only broadly, because each pooled median
mixes the two modes in a different proportion; PCA sits above D3 because even its
healthy seeds are fed more attack (0.34 vs 0.29).

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

## Consequence for the monitor choice

PCA matches the autoencoder on closed-loop stability as well as on detection
quality (6/20 vs 4/20, p = 0.72), which agrees with
[experiment 2](02-monitor-ablation.md): the deployable configuration is the
**cheap linear monitor**. The `final_f1`-masks-collapse caveat stands on its own
and does not change that.

## Honest scope

These are our implementations, and the native arms are not budget-matched by
construction. The claim is about what these signals do to *this* adaptation loop
on flow data — not a verdict on the detectors in general. Read the collapse
counts as effect sizes on a modest number of runs: **the ordering is the result;
individual pairwise p-values at this n are not.** A faithful `(w, ρ)` D3 driven
by scikit-learn's solver, and a third published comparator, remain future work.

## Reproducing

```bash
# Calibrate a monitor's Page-Hinkley to the reference fire count (offline sweep
# over a cached score series — one run, then any threshold replays for free).
uv run python scripts/calibrate_monitor.py C_early_hdddm --reference C_early

# The collapse outcome is bimodal: run these to 20 seeds, not 5.
make run-comparators
uv run python scripts/comparator_analysis.py
```
