# Related work and the unsupervised comparator

*The section that decides whether this paper is publishable.* **PACE**
(*Proactive Adaptation via Covariate-shift Estimation*) is, at the level of its
signal, an unsupervised drift detector. If the related work does not place us
precisely against that family — and if we do not compare against a published
member of it — a reviewer closes with "this is unsupervised drift detection
triggering retraining, which already exists." This document does both:
§1–§4 position the work, §5 states the honest delta, §6 is the comparator
experiment, §7 pre-empts the reviewer.

---

## 1. Supervised (error-based) drift detection — what our *baseline* is

The classical family monitors the **classifier's error stream** and signals drift
when it degrades. It needs labels, so in an IDS it inherits the labelling delay.

| Method | Reference | Role here |
|---|---|---|
| Page-Hinkley test | Page (1954), *Biometrika* †| our reactive trigger (`B`) |
| DDM | Gama et al., SBIA (2004) † | family context |
| EDDM | Baena-García et al., ECML PKDD workshop (2006) † | family context |
| ADWIN | Bifet & Gavaldà, *SDM* (2007), doi:10.1137/1.9781611972771.42 | ran as baseline `B_adwin` |
| KSWIN | Raab, Heusinger, Schleif, *Neurocomputing* (2020) † | ran as baseline `B_kswin` |
| HDDM-A/W | Frías-Blanco et al., *TKDE* (2015) † | available in river, not run |

**Our position.** These are the *reactive* arm we improve on, not competitors to
our monitor. We already run ADWIN and KSWIN and show they do not beat plain PHT
on this problem — useful, because it forecloses "you just picked a weak reactive
detector."

---

## 2. Unsupervised / feature-space drift detection — **the family we belong to**

This is the frontier that decides our novelty. These methods detect drift from
the **unlabelled input distribution**, which is exactly what our monitor does.

- **Surveys.** Gemaque et al., *An overview of unsupervised drift detection
  methods*, WIREs DMKD 10(6), 2020, doi:10.1002/widm.1381 — the reference taxonomy
  for this family. Broader: Lu, Liu, Dong, Gu, Gama, Zhang, *Learning under
  Concept Drift: A Review*, IEEE TKDE 31:2346–2363, 2019,
  doi:10.1109/TKDE.2018.2876857 (detection / understanding / adaptation).
  Recent: Arora et al., WIREs DMKD, 2024, doi:10.1002/widm.1536.
- **D3 — Discriminative Drift Detector.** Gözüaçık, Büyükçakır, Bonab, Can,
  *Unsupervised Concept Drift Detection with a Discriminative Classifier*, CIKM
  2019. Trains a discriminative classifier to separate an old window from a new
  one; separability (AUC) above a threshold ⇒ drift. **The canonical unsupervised
  baseline, and our comparator (§6).**
- **HDDDM — Hellinger Distance Drift Detection.** Ditzler & Polikar, *Hellinger
  distance based drift detection for nonstationary environments*, IEEE CIDUE
  2011. Histogram/Hellinger distance between consecutive batches; multivariate,
  unsupervised.
- **Autoencoder reconstruction error.** Jaworski, Rutkowski et al., *Concept
  Drift Detection Using Autoencoders in Data Streams Processing*, ICAISC 2020,
  doi:10.1007/978-3-030-61401-0_12; and a 2024 framework in *Int. J. Machine
  Learning and Cybernetics*, doi:10.1007/s13042-024-02223-2.

> **The uncomfortable fact, stated plainly:** monitoring autoencoder
> reconstruction error to detect drift **already exists**. Our AE monitor is
> therefore **not** a novel detector, and the paper must not claim it is. The
> contribution has to sit on the *policy*, not the signal.
>
> **Nor do we rescue it by claiming ours is more stable.** It is not: at matched
> fire count over 20 seeds our monitor is statistically indistinguishable from
> HDDDM and from a cheap PCA statistic (experiment 7). What the comparator
> experiment establishes is different and, we think, more useful: *this style of
> adaptation loop is **fragile** — every trigger we tested, ours included,
> intermittently drives it into a catastrophic false-positive regime, through a
> mechanism we verify. That fragility is not reported by any of these papers, and
> it is invisible to the detection metrics they do report.*

---

## 3. Adaptive stream learners — the models we ride on

| Method | Reference | Role here |
|---|---|---|
| Hoeffding Adaptive Tree | Bifet & Gavaldà, IDA (2009) † | base learner |
| Adaptive Random Forest | Gomes et al., *Machine Learning* 106:1469–1495 (2017), doi:10.1007/s10994-017-5642-8 | SOTA base learner |
| Streaming Random Patches | Gomes, Read, Bifet, *ICDM* (2019) | SOTA base learner |

These self-adapt internally. Our trigger is **orthogonal**: we gate *when* they
are allowed to learn, and show the benefit survives on top of them — which is the
strongest form of the claim, since it is not a rescue of a weak model.

---

## 4. Concept drift in network intrusion detection

Drift-aware NIDS is an active applied area, and CICIDS2017 is the standard
testbed *because* its attacks arrive in temporal blocks, creating natural drift.
Representative recent work: ReCDA (representation-enhanced drift adaptation for
NIDS, 2024); PWPAE (ensemble drift adaptation for IoT streams, arXiv:2109.05013);
lightweight drift frameworks for IoT streams; and adaptive-ADWIN variants tuning
detector sensitivity for streaming IDS.

**Gap we exploit.** This literature overwhelmingly (a) triggers on **error**
(supervised) or (b) reports end accuracy without separating *what* produced the
gain — adaptation timing, adaptation frequency, or model capacity. Budget-matched
controls are rare. That methodological gap, not a new detector, is our opening.

---

## 5. The honest delta — what we actually add

**We do not claim a new drift detector.** Feature-space/unsupervised detection
(§2) exists; reconstruction-error monitoring exists. What we claim:

1. **A *when-to-adapt policy*, not a detector.** The contribution is using a cheap
   unsupervised shift signal as the *adaptation trigger* for a streaming IDS, and
   characterising what that buys — as opposed to proposing another detector and
   reporting accuracy.
2. **Controls that isolate the mechanism.** We separate adaptation **timing** from
   **frequency** (count-matched blind-clock control) and from **model capacity**
   (budget-matched ARF/SRP). This is the part §4's literature omits, and it is
   what makes the claim falsifiable rather than anecdotal.
3. **Model-agnostic transfer at a fixed budget.** The same trigger improves HAT,
   ARF and SRP across three datasets with adaptation confined to triggered
   windows — a fraction of continuous supervision.
4. **Closed-loop stability as a first-class criterion — and evidence that the
   loop is fragile.** Trigger quality has to be measured by what the trigger does
   to the adaptation loop, not by detection accuracy, because the two come apart:
   badly placed adaptation windows bias the learner's training distribution
   toward attack (up to 0.44 against a 0.20 base rate) and collapse it into a
   0.5–0.69 false-positive regime. Across 20 seeds this happens to **every**
   trigger we tested — the published D3 and HDDDM, cheap PCA and mean-shift
   statistics, and our own autoencoder — on 20–60% of runs. We report the
   criterion, the mechanism, and the honest finding that our monitor is *not*
   measurably more stable than the published alternatives (only the trivial
   mean-shift floor separates, p=0.023). None of this is reported in the §2
   literature.

In one line: **the novelty is the policy and the controlled evidence, not the
signal.** Framed that way the related work supports us; framed as "a new drift
detector" it does not survive §2.

---

## 6. The unsupervised comparators: D3 and HDDDM at identical protocol

**Why these two.** D3 is the most-cited *named* unsupervised drift detector, it
is explicitly designed to sit in front of any classifier, and its principle
(discriminative separability of old vs new windows) is orthogonal to ours
(reconstruction error). HDDDM (Ditzler & Polikar 2011) is the informative second,
because a bin-by-bin Hellinger distance responds to a change in dispersion at
constant mean, which a linear discriminator cannot see — so if instability were
merely a consequence of D3's blind spot, HDDDM would be immune. It is not.

**Implementation.** `D3Monitor`, `D3NativeMonitor` and `HDDDMMonitor` in
`src/driftgan/early/shift_monitors.py`. Deviation carried by every arm: logistic
regression is fit by L2-regularised batch gradient descent, because the project
has no scikit-learn dependency (the reference uses
`LogisticRegression(solver='liblinear')`).

**Two protocols, two questions.**

- *Budget-matched* (`C_early_d3`, `C_early_hdddm`): our shared monitor protocol
  and our Page-Hinkley, with each threshold calibrated so the fire count matches
  the autoencoder's 58. Only the shift statistic differs, so this asks whether
  the **signal** substitutes.
- *Native* (`C_early_d3_native`, `C_early_d3_sized`): D3 keeps its own `(w, ρ)`
  roll, its own cadence and its own AUC threshold, and declares drift itself.
  Deliberately not budget-matched. This asks whether the published method works
  here **out of the box**, and it forecloses the objection that the matched arm
  was a weakened D3.

**Outcome, at 20 seeds.** Collapse counts (cumulative accuracy < 0.90):
autoencoder **4/20**, HDDDM **5/20** (Fisher p = 1.00), PCA **6/20** (p = 0.72),
D3 **10/20** (p = 0.096), mean-shift **12/20** (p = 0.023). Pooled, ours against
all four alternatives is 4/20 vs 33/80, p = 0.119.

**Our monitor is not measurably more stable than the published detectors**, and
that is stated rather than buried. What the experiment does establish:

1. **The loop is fragile, and that is the reportable finding.** Every trigger —
   two published detectors, two cheap statistics and our own — collapses the
   learner on 20–60% of seeds, through a mechanism we verify: badly placed
   windows feed the learner up to 0.44 attack fraction against a 0.20 base rate,
   the prior skews, the false-positive rate reaches 0.49–0.69, the class-PHT then
   fires constantly and opens more bad windows. `train atk%` tracks the collapse
   count monotonically across monitors. Nothing in §2 reports this.
2. **Detection metrics do not predict it**, and neither does final-window F1
   (PCA seed 1: final F1 0.948, cumulative accuracy 0.444, FPR 0.657). The
   criterion has to be closed-loop.
3. **The signal still matters at the bottom end.** The trivial mean-shift floor
   is significantly worse than everything else, so this is not "any statistic
   will do".
4. **Native D3 fails differently, and worse.** At the released defaults
   (`w=100, ρ=0.1, τ=0.7`) it fires ~12 300 times against our 169 — because with
   a 10-sample "new" window the 2-fold AUC crosses τ on ~5% of tests over a
   *stationary* stream (unit-tested). That is a transfer limitation of the
   published settings to 70-feature flow data, and it saturates the loop rather
   than misplacing it.
5. **It reverses our own earlier revision of experiment 2.** PCA matches the
   autoencoder on stability too (6/20 vs 4/20), so experiment 2's original
   conclusion stands: the autoencoder is over-engineered and the paper should
   lead with the cheap linear monitor.

Full write-up: `docs/experiments/07-unsupervised-comparator.md`; numbers:
`results/report/comparator/analysis.md`; figure:
`docs/figures/fig15_closed_loop_stability.png`.

**Scope, stated honestly.** These are our implementations (gradient-descent
logistic regression rather than liblinear), and the native arms are not
budget-matched by construction. This is evidence about what these signals do to
**our** adaptation loop on flow data, not a verdict on the detectors in general.
Treat the collapse counts as effect sizes on a modest number of runs: the
ordering is the result, the individual pairwise p-values at n = 20 are not.

**Methodological note.** The collapse outcome is bimodal — a run either stays
healthy or falls into a degenerate regime, with almost nothing in between — so a
handful of replicates estimates its rate badly and can produce a clean-looking
but spurious ranking. Collapse-count comparisons here are held to n ≥ 20.

---

## 7. Reviewer objections, pre-empted

| Objection | Answer |
|---|---|
| "This is unsupervised drift detection + retraining; it exists." | Correct, and we say so (§2, §5). We claim the *policy* and the controls that isolate timing from frequency and capacity — not the detector. |
| "Why not compare against unsupervised detectors?" | We do: D3 and HDDDM at identical protocol, plus D3 with its own native window and threshold (§6), plus KS, MMD, PCA and mean-shift statistics in the ablation. |
| "So your monitor isn't better than D3 or HDDDM." | Correct, and §6 says so. The comparator experiment is not a leaderboard — it establishes that the adaptation loop is fragile for *every* trigger, identifies the mechanism, and shows detection metrics do not predict it. |
| "The autoencoder is a re-invention." | Acknowledged and cited (§2). Our own ablation shows it is unnecessary; the paper leads with the cheap linear monitor. |
| "Gains are modest." | Lead with 2018, where informed timing beats a count-matched blind clock and adapting *more* blindly actively hurts; and with transfer to ARF/SRP at a fraction of the supervision. |
| "Only 3 datasets — is the split a mechanism?" | We explicitly *decline* to claim a drift-structure mechanism: two pre-specified synthetics failed to isolate it (experiment 6). It is reported as an empirical observation. |
| "Error-based detectors would do as well." | ADWIN and KSWIN were run; neither beats plain PHT here. |

---

## 8. Citation verification status

All 25 entries of the paper's `references.bib` were checked field by field
(title, authors, year, volume, issue, pages, DOI) against their Crossref records
on 2026-10-27; every entry now carries a DOI. Two residual differences are
expected: Lu et al. is TKDE 31(12) 2019 although Crossref only records the 2018
early-access date, and Gomes et al. 2017 keeps the published misspelling
"Pfharinger".

Corrected in that pass, after a reviewer flagged placeholders and
misattributions: Arora et al. (the DOI is *A systematic review on detection and
adaptation of concept drift in streaming data using machine learning
techniques*, by Shruti Arora, Rinkle Rani and Nitin Saxena, WIREs 14(4)); Hinder
et al. (title ends *Part A: detecting concept drift*); DriftLens (published as
IEEE TKDE 37(10) 2025, not Knowledge-Based Systems; title without the "DriftLens:"
prefix); Abdel Wahab (family name "Abdel Wahab", title *... Under Data and Concept
Drifts: Online Deep Learning Approach*); AECDD (title without the "AECDD:"
prefix); DD-SAPH (GPC 2023, LNCS 14503); AEDDM (IJMLC 16(1) 2025); Souza et al.
(KAIS 63(6)); Lukats et al. (IJDSA 19(1) 2025); SSF (INFOCOM 2025).
