# Central idea of the paper — the trail of thinking

*A plain-language walk from the hypothesis we started with to the one we are
actually verifying. Written for a reader comfortable with intrusion detection and
machine learning, no prior context on this project assumed.*

---

## TL;DR (one paragraph)

We began believing the contribution was **generative**: use a GAN to *fabricate*
synthetic attack traffic at drift time so a detector could adapt faster, and use
a generative model to *anticipate* drift. Careful, budget-matched experiments
killed the generative part — fabricating traffic adds nothing over trivial
baselines. What survived, and became the real contribution, is the *opposite*
mechanism: an **unsupervised covariate-shift monitor** that watches the incoming
traffic and decides **when to adapt**, firing earlier than an error-based
reactive detector. It generates nothing, and a cheap linear version detects as well
as the deep one. The paper is therefore about **adaptation timing ("when to
adapt"), not generation ("what to generate")** — and we are now pinning down
*when* that timing actually helps.

---

## The setting (the problem in one breath)

Network intrusion detectors are trained on yesterday's traffic and then face a
stream that keeps changing: new attacks appear, old ones fade. This is **concept
drift**. When it hits, the classifier's quality drops until it is retrained
("adapted"). Two levers matter: **when** you decide to adapt, and **how** you
adapt (what data you retrain on). Our project explored both with GANs.

---

## Step 1 — The initial hypothesis (what we set out to prove)

The original idea, "DriftGAN", bundled **two** bets under one name:

- **Bet A — Generation to adapt.** When a new attack appears you only have a
  handful of real examples. A GAN could *synthesize more of them* and you retrain
  on the enlarged set, recovering faster.
- **Bet B — Generative anticipation.** A generative model could flag that a drift
  is coming *before* the classifier's error visibly rises, buying a head start.

In our heads these fused into a single slogan: *"generate traces with a GAN
before the drift, to get ahead of it."* Keep that slogan in mind — the
experiments split it apart.

---

## Step 2 — How we tested it (the controls are the point)

We built one prequential (test-then-train) engine and ran every strategy through
it, differing only in **trigger** (when to adapt) and **augmenter** (how). The
value is in the *controls*, each designed to strip away an easy explanation:

- Frozen model and a reactive baseline (adapt when classification error rises).
- The GAN augmenters (Bet A) vs. a **trivial** control (bootstrap + jitter of the
  few real samples) at the **same adaptation budget**.
- The anticipation monitor (Bet B) vs. cheaper, non-generative monitors.
- A **frequency-matched** control: adapt the *same number of times* but on a
  blind clock, to separate "adapt earlier" from "adapt more often".

Everything is multi-seed, paired per drift event, and honest about windows where
the metric is undefined.

---

## Step 3 — What broke: the generative bet is null

**Bet A failed.** Synthesizing attack traffic with the GAN to retrain does **not**
beat the trivial bootstrap+jitter control, nor the reactive baseline, at matched
budget. The literal "generate traces" idea — the thing the project was named
after — adds no value. This is a solid *negative* result, not a bug: we corrected
the known pitfalls (feature scaling) first, then measured a clean null.

---

## Step 4 — What survived, and what it really is

**Bet B works — but it is not generation, and it does not need a GAN.**

The monitor does the *reverse* of generating. It learns to *reconstruct* recent
"normal" traffic; when unfamiliar attack traffic arrives, it reconstructs it
badly and its **reconstruction error spikes**. Crucially that spike lives on the
**input features**, so it can rise **before** the classifier's labelled error
trips the reactive trigger. That lead time is the anticipation.

Two findings reframe the whole project:

1. **The signal is not a generative novelty.** An ablation shows a cheap **linear
   (PCA) reconstruction** monitor *detects* as well as the autoencoder/GAN, so
   underneath this is simply an **unsupervised covariate-shift detector** — a
   well-understood object. (Later qualified in Step 7: detecting equally well does
   *not* mean triggering equally well — only the AE keeps the adaptation loop
   stable across all seeds.)
2. **The lever is the trigger, not the model.** The benefit comes from *using
   that shift signal to decide when to adapt*, and it transfers to strong models:
   swapping a reactive trigger for the proactive shift-trigger helps not only the
   weak single-tree learner but also SOTA streaming ensembles (ARF, SRP).

So the slogan flips from *"generate traces to get ahead"* to
**"watch the input distribution to decide when to adapt."**

---

## Step 5 — The refined hypothesis (what we are verifying now)

> **An unsupervised covariate-shift monitor that triggers adaptation earlier than
> an error-based reactive detector improves sustained detection quality across
> base learners and datasets — because *when* you adapt matters more than *what*
> you generate.**

But "earlier is better" is not free of nuance, and this is exactly what the
current experiments (Tier 0 statistical hardening, Tier 1 timing-vs-frequency)
are stress-testing:

- **The effect is real for the strong models.** Across 3 datasets × 3 learners
  the proactive trigger is positive in every cell; with proper paired confidence
  intervals it is *robust* for all ensemble cells (the weak HAT is just noisy),
  and it shows up in imbalance-aware metrics (Kappa) too.
- **But "earlier" is sometimes really "more often".** On the anchor dataset
  (2017) the proactive arm wins while adapting *fewer* times — a clean timing
  win. On the external datasets it adapts *more* times, so we must separate the
  two. The frequency-matched control (`*_freq`: blind periodic clock, fire count
  calibrated to the proactive arm) shows a **split that is a property of the
  dataset, not the learner** — ARF and SRP agree cell-for-cell with tight
  confidence intervals (timing component `early − freq`, paired by seed):
  - **CIC-IDS2018 (discrete blocks): timing.** ARF +0.038, SRP +0.048, CIs
    excluding 0 — and blindly adapting *more* actively **hurts** (SRP's blind
    control 0.881 falls below the sparse reactive baseline 0.925). It is
    genuinely about *when*.
  - **UNSW-NB15 (dense bursts): frequency.** ARF +0.002, SRP +0.002, CIs
    straddling 0 — informed placement adds nothing beyond adapting more often.

  Together with the 2017 anticipation evidence, that is timing on 2 of 3 datasets
  and frequency on 1, consistent across two independent SOTA learners.

This *looked* like a sharper claim — *informed timing helps in proportion to how
structured the drift is* — but two controlled synthetic experiments could not
isolate that mechanism (Step 6), so we do **not** make it. The honest,
demonstrated statement is: **the proactive trigger robustly beats reactive/blind
adaptation at matched budget; on one of three datasets (UNSW) the benefit is
adaptation frequency rather than timing, reported as an empirical exception, not
a mechanism.**

---

## Step 6 — We tried to *earn* the structure claim; it did not hold

The block-vs-burst pattern was an **observation across three real datasets
(n=3)** — a story that fits the data. To turn it into a mechanism we ran two
**pre-specified** synthetic experiments that vary drift structure with everything
else fixed:

1. `scripts/synthetic_structure.py` — sweep drift *block size* (few large blocks
   → many small bursts). **Negative:** large blocks let a blind clock re-adapt
   mid-block, and uniformly-spaced drifts align with a count-matched uniform
   clock; the predicted monotone relationship did not appear.
2. `scripts/synthetic_clustering.py` — sweep the *temporal concentration* φ of the
   drift onsets. **Also negative for the structure claim:** the timing benefit was
   large and positive at *every* φ (~+0.2), not increasing with concentration, so
   it does not demonstrate a structure dependence.

Per the pre-registered decision rule we therefore **retreat the structure
claim** and do not present a drift-structure mechanism. Two honest takeaways
remain:

- The synthetic *does* robustly corroborate the **core** claim: informed timing
  beats a blind clock at matched adaptation frequency (+0.21 even with uniformly
  spaced drifts). "When to adapt" matters in a controlled setting, not only on the
  real data.
- Neither synthetic reproduced the real-data *exception* (UNSW, where the gain is
  frequency), so that split stays an empirical observation.

---

## Step 7 — Facing the closest literature, and the twist it produced

Our monitor belongs to the **unsupervised / feature-space drift detection**
family, so we positioned against it (`docs/paper/related-work.md`) and ran its
canonical published member, **D3** (Gözüaçık et al., CIKM 2019), through the
identical policy at a matched fire count. Two things came out:

1. **The signal is not novel — and we say so.** Autoencoder reconstruction error
   for drift detection already exists. The contribution must be the *policy* plus
   the controlled evidence, never the detector.
2. **The twist: monitors that detect equally do not *trigger* equally.** D3
   collapsed 4/5 seeds, PCA and mean-shift 2/5, the autoencoder **0/5**. The
   mechanism is verified: badly placed triggers feed the learner 43–47% attack
   against a 20% base rate, its prior skews to ATTACK, the false-positive rate
   hits 0.49–0.66, and the class-PHT runs away. `train atk%` tracks the collapse
   count monotonically across monitors.

This **revises experiment 2** ("a cheap PCA monitor suffices"), which was read off
*final F1* — a metric that masks mid-stream collapse (PCA seed 1: final F1 0.948,
cumulative accuracy 0.444, FPR 0.657). Judged on closed-loop stability, the
autoencoder earns its place after all.

It also *sharpens* the thesis instead of threatening it: the claim was always that
**where you place adaptation matters**, and this is the strongest evidence yet —
bad placement does not merely fail to help, it destroys the model by biasing what
it trains on. Trigger quality must be judged by its effect on the loop, not by
detection metrics.

---

## What we can and cannot claim (honesty guardrails)

- ✅ An unsupervised covariate-shift trigger adapts earlier than a reactive
  error-based trigger and improves sustained F1 across learners and datasets.
- ⚠️ "A cheap linear monitor is enough" — **revised** (experiment 7). True on
  detection quality *when the loop stays stable*, but only the autoencoder keeps
  it stable in 5/5 seeds; PCA and mean-shift collapse in 2/5 and the published D3
  detector in 4/5, reaching false-positive rates of 0.49–0.66. Judge a trigger by
  closed-loop stability, not by detection metrics.
- ✅ Generative augmentation for triggered adaptation does **not** beat trivial
  controls — a controlled refutation, stated as a result.
- ✅ In a controlled synthetic, informed timing beats a blind clock at matched
  adaptation frequency — the "when to adapt" mechanism, isolated from the data.
- ⚠️ The gain is *timing* on 2017/2018 (informed beats blind at matched
  frequency; blind-more even *hurts* on 2018) but *frequency* on UNSW. Report this
  split as an **empirical observation**; two pre-specified synthetics could **not**
  isolate a drift-structure mechanism, so do **not** claim one.
- ❌ "We use a GAN to generate traces to anticipate drift." False twice over: the
  generative part is what failed, and the anticipation generates nothing.

---

## Naming

The method is **PACE** — *Proactive Adaptation via Covariate-shift Estimation* —
and the project/repository is **PACE Drift** (`pace-drift`). The name encodes the
thesis: what the evidence supports is a claim about the *pacing* of adaptation
(when it fires, and that firing more often is not the same as firing better), not
about generation. The original name, *DriftGAN*, described the hypothesis we
started with rather than the result we reached.

The Python package remains `driftgan` for import stability; that is deliberate,
so `src/driftgan/...` references throughout these docs are correct.

## Working title

**When to adapt matters more than what to generate: PACE, a cheap unsupervised
covariate-shift trigger for streaming intrusion detection.**

*Contributions:* (i) a controlled refutation of generative augmentation under
triggered adaptation; (ii) an unsupervised *when-to-adapt* trigger validated
across 3 datasets × 3 base learners, beating a blind clock at matched frequency
(also in a controlled synthetic); (iii) **closed-loop stability as the criterion
for trigger quality** — monitors that detect equally, including a published
unsupervised detector, destabilize the loop very differently; (iv) an honest
characterization of when the benefit is timing vs adaptation frequency.

---

*Status: Tier 0 (statistical hardening) and Tier 1 (timing vs frequency, ARF+SRP
on 2018/UNSW) done; the empirical split holds across two learners. Two
pre-specified synthetic experiments (`scripts/synthetic_structure.py`,
`scripts/synthetic_clustering.py`) did NOT isolate a drift-structure mechanism —
the structure claim is retreated; the core "informed beats blind at matched
frequency" is corroborated. Related-work positioning and the D3 comparator are
done (`docs/paper/related-work.md`, experiment 7), and revised experiment 2's
"cheap monitor suffices" conclusion. Numbers in `results/report/tier0/`,
`tier1/`, `synthetic_clustering/` and `comparator/analysis.md`; write-ups in
`docs/experiments/06-timing-vs-frequency.md` and
`07-unsupervised-comparator.md`.*
