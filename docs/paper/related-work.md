# Related work and the unsupervised comparator

*The section that decides whether this paper is publishable.* Our monitor is,
functionally, an unsupervised drift detector. If the related work does not place
us precisely against that family — and if we do not compare against a published
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
> **But the signal is not interchangeable either** (experiment 7): at matched
> fire count, cheaper statistics (PCA, mean-shift) and the published D3 detector
> intermittently drive the adaptation loop into a catastrophic false-positive
> regime, while the AE does not. So the honest statement is: *the detector is not
> novel, but its stability inside the closed loop is what the paper measures —
> and that is not a property any of these papers report.*

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
4. **Closed-loop stability as a first-class criterion.** Monitors that look
   equivalent on detection metrics are *not* equivalent as triggers: badly placed
   adaptation windows bias the learner's training distribution toward attack
   (43–47% vs a 20% base rate) and collapse it into a 0.5–0.66 false-positive
   regime. We measure trigger quality by what it does to the loop, not by
   detection accuracy — a criterion the §2 literature does not report.

In one line: **the novelty is the policy and the controlled evidence, not the
signal.** Framed that way the related work supports us; framed as "a new
generative drift detector" it destroys us.

---

## 6. The unsupervised comparator: D3 at identical protocol

**Why D3.** It is the most-cited *named* unsupervised drift detector, it is
explicitly designed to sit in front of any classifier, and its principle
(discriminative separability of old vs new windows) is orthogonal to ours
(reconstruction error), so the comparison is informative rather than a
re-parameterisation. HDDDM is the natural second (distance between batch
histograms); we treat it as future work.

**Implementation.** `D3Monitor` in `src/driftgan/early/shift_monitors.py`, run as
phase `C_early_d3`. Deviations from the reference implementation, stated for
reproducibility:

- logistic regression is fit by L2-regularised batch gradient descent, because the
  project has no scikit-learn dependency (the reference uses
  `LogisticRegression(solver='liblinear')`);
- window bookkeeping is our shared monitor protocol (frozen reference frame vs
  current buffer) instead of D3's own `(w, ρ)` roll, so that **every** monitor in
  the ablation sees exactly the same data at exactly the same check points;
- the discriminative principle and the **2-fold stratified AUC** estimate — the
  parts that define the method — are preserved.

**Protocol.** Identical to the monitor ablation (experiment 2): same pipeline,
same base learner, same dual-PHT adaptation policy, with `PHT_recon` calibrated on
the cached D3 score series so its fire count matches the autoencoder's. The only
thing that changes is the shift statistic. This is the fairest possible reading:
*given the same when-to-adapt policy, does a published unsupervised detector do
the job as well as our monitor?*

**Outcome (the middle branch of the three we pre-committed to).** D3-triggered
adaptation proved **unstable** here: 4/5 seeds collapsed (vs 0/5 for the AE),
while the surviving seed matched it exactly. Not a quality gap — a *stability*
gap, and the mechanism is verified: badly placed triggers feed the learner
43–47% attack against a 20% base rate, skewing its prior until the false-positive
rate reaches 0.49–0.66 and the class-PHT runs away. A unit test pins the root
cause: D3's linear discriminator is blind to variance-only change (AUC 0.50),
which reconstruction error captures. Full write-up:
`docs/experiments/07-unsupervised-comparator.md`; numbers:
`results/report/comparator/analysis.md`.

Two consequences for the paper:

1. **It sharpens the thesis rather than threatening it.** The claim was always
   that *where* adaptation is placed matters; this shows bad placement does not
   merely fail to help, it actively destroys the model through training-set bias.
2. **It revises experiment 2.** "A cheap PCA monitor suffices" was read off
   *final F1*, which masks mid-stream collapse (PCA seed 1: final F1 0.948,
   cumulative accuracy 0.444, FPR 0.657). Judged on closed-loop stability, the
   autoencoder earns its place — the one arm stable in 5/5 seeds.

**Scope, stated honestly.** Ours is an *adaptation* of D3 (shared protocol,
gradient-descent logistic regression), so this is evidence about a
discriminative-separability signal **in our pipeline**, not a verdict on D3 as a
detector. A faithful `(w, ρ)` D3 with its native threshold and HDDDM as a second
comparator are future work.

---

## 7. Reviewer objections, pre-empted

| Objection | Answer |
|---|---|
| "This is unsupervised drift detection + retraining; it exists." | Correct, and we say so (§2, §5). We claim the *policy* and the controls that isolate timing from frequency and capacity — not the detector. |
| "Why not compare against unsupervised detectors?" | We do: D3 at identical protocol (§6), plus KS, MMD, PCA and mean-shift statistics in the ablation. |
| "The autoencoder is a re-invention." | Acknowledged and cited (§2). Our own ablation shows it is unnecessary; the paper leads with the cheap linear monitor. |
| "Gains are modest." | Lead with 2018, where informed timing beats a count-matched blind clock and adapting *more* blindly actively hurts; and with transfer to ARF/SRP at a fraction of the supervision. |
| "Only 3 datasets — is the split a mechanism?" | We explicitly *decline* to claim a drift-structure mechanism: two pre-specified synthetics failed to isolate it (experiment 6). It is reported as an empirical observation. |
| "Error-based detectors would do as well." | ADWIN and KSWIN were run; neither beats plain PHT here. |

---

## 8. Citation verification status

Verified against sources during this session (venue/year/DOI checked): D3 (CIKM
2019), HDDDM (Ditzler & Polikar 2011), Gemaque et al. (WIREs 2020,
doi:10.1002/widm.1381), Lu et al. (TKDE 2019, doi:10.1109/TKDE.2018.2876857),
Arora et al. (WIREs 2024, doi:10.1002/widm.1536), ARF (Machine Learning 2017,
doi:10.1007/s10994-017-5642-8), SRP (ICDM 2019), ADWIN (SDM 2007,
doi:10.1137/1.9781611972771.42), autoencoder drift detection (ICAISC 2020,
doi:10.1007/978-3-030-61401-0_12; IJMLC 2024, doi:10.1007/s13042-024-02223-2).

**† Not re-verified this session** — standard references quoted from background
knowledge; confirm the exact venue/page/DOI before submission: Page (1954), DDM
(Gama et al. 2004), EDDM (Baena-García et al. 2006), KSWIN (Raab et al. 2020),
HDDM (Frías-Blanco et al. 2015), HAT (Bifet & Gavaldà 2009).
