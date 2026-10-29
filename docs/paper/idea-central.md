# The central idea

**PACE** — *Proactive Adaptation via Covariate-shift Estimation* — treats **when
to adapt** as a design decision in its own right, rather than a consequence of
whichever drift detector is bolted onto a streaming classifier.

> **Paper framing (IEEE Euro S&P 2027, from October 2026).** The paper is now
> *"When to Retrain a Network Intrusion Detector: Budget-Matched Evidence on
> Timing, Triggers and Closed-Loop Collapse"*. After the live control (`*_pht`)
> left the monitor's contribution unresolved, PACE is presented as the
> **hypothesis under test**, not the contribution. The three headline findings
> are: timing, not frequency, carries the gain (+0.042 [+0.008, +0.077]); under
> immediate labels the label-free signal does not beat a live error test
> (+0.021 [−0.002, +0.045]); and every stream-responsive trigger can collapse a single-tree detector into an
> alert flood, which makes the trigger security-critical. The security
> evaluation ([10](../experiments/10-security.md), HAT, CIC-IDS2017, 20 seeds)
> shows an adversary can **induce** it with clean labels: feature-space chaff
> (1.6% of traffic) collapses PACE 14/20 and the live error test 15/20 (vs 4
> and 8) — label-driven triggers are **not** harder to steer — and an oracle
> that opens attack-free windows blinds the detector. **Window rebalancing**
> (replay to pin each window's attack share at the pretraining 0.134) removed
> every collapse (0/140) and lifted local F1 by +0.13 to +0.24. Do not claim
> problem-space realisability of the chaff, or robustness to an adversary who
> adapts to the defence. Label delay is the next experiment.

## The problem

A network intrusion detector trained offline meets non-stationary traffic. New
campaigns, tool variants and scanning behaviours shift the input distribution and
the input→label relation; a frozen model decays silently.

The standard remedy is to retrain when the classifier's error rises. That
inherits the delay of labelled feedback: someone has to confirm which alerts were
correct before the error signal means anything, and in the meantime the model is
already degraded. Labels are also the scarce resource in a real deployment —
continuous supervision is the one thing you cannot assume.

## The proposal

Watch the **incoming traffic** instead of the classifier's mistakes. An
unsupervised monitor tracks how far the current window of unlabelled traffic sits
from the distribution the model was fitted on, and a Page-Hinkley test over that
signal opens a short adaptation window *before* the error responds. Learning stays
confined to those triggered windows, so the policy costs a small fraction of
continuous supervision.

The monitor needs no labels, which is exactly why it can act early.

## Why this is a contribution, given that the parts exist

Feature-space drift detection exists. Monitoring an autoencoder's reconstruction
error to detect drift exists (Jaworski & Rutkowski, ICAISC 2020). If the claim
were "a new drift detector", it would be wrong. The claim is four other things.

**1. A when-to-adapt policy, not a detector.** The contribution is using a cheap
unsupervised shift signal as the *adaptation trigger* for a streaming IDS under a
bounded budget, and characterising what that buys.

**2. Controls that isolate the mechanism.** The obvious objection is that any
gain comes from adapting more often. So the proposal is measured against a blind
periodic clock firing **exactly as often**, which separates adaptation *timing*
from *frequency*, and against budget-matched ensembles, which separates it from
*model capacity*. This is the part the applied drift-in-NIDS literature routinely
omits, and it is what makes the claim falsifiable.

**3. Controls that also bound our own claim.** A live reactive control runs the
error test PACE uses alongside its monitor, on each learner's own predictions and
with no monitor. It is the control that narrowed the result (below).

**4. Closed-loop stability as the criterion for trigger quality.** Trigger quality
has to be measured by what the trigger does to the *loop*, not by how well it
detects — and applying that criterion produces the least comfortable result in
this project (below).

## What the evidence says

**Against the replayed schedule the trigger works, and it is timing.** Sustained
detection quality improves in all nine (dataset × base-learner) cells;
aggregated over cells, **+0.043 [95% CI +0.021, +0.065]**, with the CI excluding
0 in 7 of 9 cells. A blind clock firing the same number of times does not
reproduce it: over the nine cells the timing component is **+0.042
[+0.008, +0.077]** and the frequency component +0.001. The same holds on a
generated stream whose drift structure is known by construction, at an exactly
matched budget over 10 seeds (**+0.254 [+0.106, +0.401]** with PCA, **+0.288
[+0.157, +0.419]** with the AE) — where the monitor again adds only ~+0.02 over
the live error test. Caveat (third review round): treating the three datasets as
clusters (hierarchical bootstrap) widens the nine-cell timing interval to
**[−0.003, +0.084]**; the per-cell intervals (7/9, all six on 2017/2018) carry
the claim, not the aggregate. At 20 seeds the HAT anchor's PACE − schedule gain
vanishes (+0.000) while its timing contrast survives (+0.076 [+0.002, +0.149]).

**But the informing is mostly the live error test, not the monitor.** The
reactive arms replay a schedule recorded once from HAT at seed 0. Run live on
each learner's own predictions with no monitor, the error test alone matches
PACE in five of six cells on CIC-IDS2017/2018 with a third to a half of the
labels; the monitor's own contribution is **+0.021 [−0.002, +0.045] — not
resolved**. It is significant only on the UNSW ensembles, where it adds firings
and a blind clock at the same count does as well. The anticipation is real (the
monitor fires first), but under immediate labels it does not translate into
measurably better adaptation; label delay is where that should be tested.

**Honestly, timing is not the whole story everywhere.** On UNSW-NB15 the benefit
is adaptation *frequency*, not timing. This is reported as an empirical
observation across three datasets; no mechanism linking drift structure to the
size of the timing effect is claimed, because three datasets cannot separate
drift structure from everything else that differs between captures.

**The signal can be cheap.** A linear PCA reconstruction monitor nearly matches
the autoencoder, on detection quality and on closed-loop stability alike. The
deployable configuration is the cheap one; the autoencoder is over-engineered.

**The loop is fragile — and that is the finding.** At matched fire count over 20
seeds, *every* trigger tested — ours, the published D3 and HDDDM, cheap PCA and
mean-shift — drives the learner into a degenerate regime on 20–60% of runs. The
mechanism is verified rather than hypothesised: badly placed windows feed the
learner up to a 0.46 attack fraction against a 0.20 base rate, its prior skews,
the false-positive rate on benign traffic reaches 0.60–0.70, and the error-based
detector then fires constantly and opens more bad windows. Detection metrics do
not predict any of this, and final-window F1 actively conceals it — one run shows
final F1 0.948 with cumulative accuracy 0.444.

**We could not show our monitor is more stable than the published ones.** HDDDM
and PCA are not distinguishable from it at 20 seeds (low power: 0.41 for 20% vs
50%); only the trivial mean-shift floor separates. The fragility is not specific
to unsupervised triggers: the live error test alone collapses HAT on 8/20, a fixed
replayed schedule only 2/20 (no feedback), and the ensembles never. The contribution here is the criterion and the fragility it
exposes, not a ranking that favours us.

**An adversary can pull the lever, and pinning the window stops it.** Chaff
— benign-labelled flows whose features are each copied from a different benign
flow, 1.6% of the traffic, timed to the attacker's campaign — fires the monitor
near 65% of its targets and the error test near 88%, and raises collapse from
4 to 14 (PACE, p = 0.004) and 8 to 15 (live error test) of 20. Placement
alone (an oracle at the monitor's budget) is weaker and not resolved at 20
seeds; steered toward attack-free windows it blinds the detector (10/20 runs
miss a third of the attacks). Replaying stored labelled flows so each window
teaches the pretraining class ratio removed every collapse, natural and
adversarial, at no label cost; under it timing still matters (+0.024
[+0.007, +0.042]) and the monitor still does not resolve over the live error
test (+0.018 [−0.003, +0.038]).

## What we do not claim

- ❌ A novel drift detector. The signal family is prior art, and we cite it.
- ❌ That the covariate-shift monitor places adaptation better than an error test
  run live on the learner. Against that control its contribution is not resolved.
- ❌ That our monitor is more stable than published unsupervised detectors.
- ❌ A mechanism explaining why timing dominates on some datasets and frequency on
  others. It is an observation, reported as one.
- ❌ Generalisation beyond CICFlowMeter-family flow features and prequential
  evaluation with immediate labels.
- ❌ That the chaff can be realised as packets (it is built in feature space),
  that the defence resists an adversary who knows it, or that the adversary
  results hold beyond HAT on CIC-IDS2017.

## In one sentence

We treat **when** to adapt a streaming intrusion detector as a policy, show with
controls that informed placement beats blind placement — while being unable to
show that an unlabelled covariate-shift signal informs it better than a live
error test does — and show that choosing that trigger badly does not merely fail
to help: it poisons the model.
