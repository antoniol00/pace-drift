# 10 — Security evaluation: a trigger-aware adversary, and window rebalancing

The paper's threat model (Euro S&P framing) names a trigger-aware adversary
that can send traffic, knows the retraining policy and cannot alter labels.
This experiment measures it instead of discussing it, on the cell where
collapse happens: HAT on CIC-IDS2017, twenty seeds (0–19), the same seeds as
the stability study ([07](07-comparators.md)).

## Arms

| Phase | Trigger | Adversary | Defence |
|---|---|---|---|
| `C_early` | PACE (AE monitor + live error test) | — | — |
| `C_early_adv` | oracle monitor branch + live error test | oracle, attack-dense windows | — |
| `C_early_advb` | oracle monitor branch + live error test | oracle, attack-free windows | — |
| `C_early_chaff` | PACE | chaff before the oracle targets | — |
| `B_pht` | live error test | — | — |
| `B_pht_chaff` | live error test | chaff before the oracle targets | — |
| `C_early_rb`, `B_pht_rb`, `C_early_adv_rb`, `C_early_chaff_rb` | as above | as above | window rebalancing |

**Oracle** (`scripts/adversary_schedule.py` → `results/adversary/`). It fires
the monitor branch instead of a monitor, under the monitor's own constraints:
only at monitor checks (`n_pretrain + k·1,000`), no earlier than the
Page-Hinkley warm-up and at least ten checks (10,000 samples) apart, at most as
often as PACE's monitor fires (54 per run, mean over the stored `C_early`
seeds). The *attack* goal greedily picks the windows with the highest attack
share and never fires on a window below 50% attack (that would counteract the
skew), which leaves 31 firings at a mean window attack share of 0.949. The
*benign* goal opens 54 attack-free windows, preferring those followed by the
most attack traffic within the 20,000-sample lead horizon (mean 0.034: on this
capture attack-free windows are rarely followed closely by a burst). Ties are
broken by a fixed random permutation; one schedule serves every seed.

**Chaff** (`data/stream.py:inject_chaff`). Before each of the 31 attack
targets, 1,000 flows are inserted whose features are each copied from a
different benign flow among the first 10,000 benign flows of the stream:
every value is plausible, the joint structure is broken. Chaff is labelled
benign (it is not malicious traffic), is classified, feeds both tests and is
learned from when a window is open, but is kept out of every metric and does
not advance the stream index. Total: 31,000 flows, 1.6% of the evaluated stream.

**Window rebalancing** (`models/rebalance.py`). After each labelled flow learned
inside a window, stored flows of the under-represented class are replayed from
per-class reservoirs (Algorithm R, 5,000 each, fed with pretraining and window
samples) until the window's attack share is back at the prior, 0.134 — the
attack share of the 50K pretraining labels, which the defender knows without
lookahead. Replays cost no labels; the label share is that of the real windows.
The runner now records what the learner is fed (`summary.json:learned`).

## Results (HAT, CIC-IDS2017, 20 seeds)

Each cell is without / with window rebalancing (`results/report/security/analysis.md`).

| Trigger | Adversary | Collapsed | Benign FPR med | Benign FPR worst | Attack recall med | Fed attack | Local F1 med | Labels |
|---|---|---|---|---|---|---|---|---|
| PACE | none | 4 / **0** | 0.006 / 0.009 | 0.662 / 0.012 | 0.975 / 0.965 | 0.26 / 0.13 | 0.594 / 0.701 | 30.5 / 33.2% |
| PACE | oracle, attack-dense | 8 / **0** | 0.008 / 0.008 | 0.719 / 0.010 | 0.984 / 0.963 | 0.51 / 0.13 | 0.534 / 0.694 | 26.3 / 27.1% |
| PACE | oracle, attack-free | 7 / **0** | 0.007 / 0.009 | 0.461 / 0.014 | 0.808 / 0.963 | 0.24 / 0.13 | 0.418 / 0.672 | 34.3 / 37.6% |
| PACE | chaff | 14 / **0** | 0.600 / 0.008 | 0.660 / 0.010 | 0.789 / 0.964 | 0.42 / 0.13 | 0.428 / 0.700 | 37.8 / 39.3% |
| live error test | none | 8 / **0** | 0.007 / 0.008 | 0.718 / 0.013 | 0.979 / 0.963 | 0.33 / 0.13 | 0.547 / 0.697 | 21.7 / 23.6% |
| live error test | chaff | 15 / **0** | 0.717 / 0.009 | 0.718 / 0.011 | 0.982 / 0.967 | 0.53 / 0.13 | 0.461 / 0.691 | 26.8 / 22.5% |
| blind clock | none | 8 / **0** | 0.010 / 0.009 | 0.153 / 0.013 | 0.635 / 0.933 | 0.20 / 0.13 | 0.483 / 0.675 | 41.5 / 41.5% |

**Chaff steers both triggers.** Chaff fires the monitor within 2,000 flows of
65% of the targets (9% at the same moments without chaff) and PACE collapses
14/20 vs 4/20 (Fisher p = 0.004), median benign FPR 0.60. The *label-driven*
error test is no harder to steer: the detector misclassifies the chaff, and
those mistakes — against correct labels — are what it counts. It fires near
88% of the targets (19% without) and collapses 15/20 vs 8/20 (p = 0.054).
Chaff also spends labels (PACE 30.5% → 37.8% of the stream). Local F1, paired:
PACE −0.109 [−0.174, −0.044], live error test −0.060 [−0.112, −0.008].

**Placement alone is a weaker lever.** The attack oracle doubles the fed attack
share (0.51 vs 0.26) and the collapse count (8 vs 4), but neither collapse
(p = 0.30) nor local F1 (−0.017 [−0.092, +0.058]) is resolved at 20 seeds.
Chaff does more because it fires both tests and the error test keeps firing on
the false alarms (339 vs 198 firings per run). The **benign oracle blinds**:
10/20 runs end with attack recall 0.59–0.66 (7 of them also with FPR ~0.45,
3 quiet but blind), local F1 −0.124 [−0.194, −0.054], 17/20 seeds worse.

**Rebalancing removes collapse, natural and adversarial.** 0/20 in all seven
configurations (140 runs; worst benign FPR 0.014). Paired local F1 +0.133 to
+0.242, 16–20/20 seeds per configuration; the benign oracle's recall recovers
(0.808 → 0.963). Cost: 1–2 points of attack recall on healthy arms; 1.3–2.8
replayed updates per labelled flow, i.e. 2.3–3.8× the learner's update work
(wall-clock update time grew ~5–10×, but the defended runs shared a machine at
load ~100, so that figure overstates the cost); no labels. It protects the
model, not the budget: chaff still fires the monitor near 65% of its targets.

**The paper's contrasts under the defence** (paired, 20 seeds): timing (PACE −
blind clock) **+0.024 [+0.007, +0.042]**, 15/20, against +0.076 undefended —
much of the undefended gap was the clock's own collapse; monitor (PACE − live
error test) **+0.018 [−0.003, +0.038]** — still not resolved.

**Prior sensitivity** (`C_early_rb50`, `C_early_chaff_rb50`, π = 0.5, no
knowledge of the base rate): 0/20 natural and 0/20 under chaff; benign FPR
median 0.025 / 0.019, worst 0.034 / 0.024 (vs 0.009 / 0.014 at π = 0.134);
recall 0.988 / 0.986; local F1 median 0.717 / 0.718 (paired vs undefended
+0.148 / +0.258); but the extra false alarms fire the error test more, so the
label share rises to 45.2% / 48.7% (vs 33.2% / 39.3%). The prior trades false
positives and labels against recall; neither setting collapsed.

**Ensembles with the defence.** ARF (`ARF_early_rb`, 20 seeds): local F1
0.747 → **0.791** (+0.045 [+0.035, +0.056], 20/20) — above continuous ARF
(0.772, 5 seeds) — at attack recall 0.989 → 0.960, benign FPR unchanged
(0.002), 2.75 replays per labelled flow, labels 25.2% → 31.4%. SRP
(`SRP_early_rb`, 20 seeds): local F1 0.771 → **0.785** (+0.020 [+0.013, +0.026],
20/20) — below continuous SRP (0.811, 5 seeds) — at attack recall 0.993 → 0.976,
benign FPR unchanged (0.001), 2.77 replays per labelled flow, labels 21.3% →
27.7%. The label share rises on both ensembles because the error test fires
more often on the defended learner (ARF 57 → 166, SRP 37 → 79 error firings per
run). Update time grew ~6× (3,258 s → 20,140 s per run, on a loaded
machine).

## Caveats

- One capture, one learner (where collapse happens); ensembles never collapse
  undefended, so the adversary was not run on them.
- The oracle knows the attack timing exactly; the chaff is built in **feature
  space**. Whether such flows can be realised as packets that CICFlowMeter maps
  to the same features is the problem-space question (Pierazzi et al., S&P
  2020) — not shown.
- No adaptive adversary against the defence (e.g. one targeting the
  reservoirs — although chaff does enter the benign reservoir in
  `C_early_chaff_rb`), and no slow/mimicry evasion of the trigger.
- The prior 0.134 is the pretraining attack share; only one alternative (0.5) was run.
