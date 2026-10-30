# 11 — Label delay, and a linear learner

Two follow-ups the assessment for Euro S&P asked for. Every real-data cell so far
assumes labels arrive with the flow, an assumption that favours the live error
test over the label-free monitor; and closed-loop collapse has only been seen on
the single tree (HAT), which a reviewer can dismiss as an artefact of a learner
nobody deploys.

## Design

**Label delay** (`label_delay`, `experiments/runner.py`). Each flow's label is
held for `label_delay` flows after the flow is classified. Only then does the
error test count the mistake and, if the flow was selected for labelling, the
learner learn it. A firing still opens a window over the flows that *arrive*
after it, so the label budget is unchanged; their labels just land later. The
monitor needs no label and sees every flow at once; the blind clock needs none
either. Prequential scoring is unchanged (the evaluator always knows the truth).
With no delay every run is identical to before (checked on four phases at seed
99, 400K flows: metrics, learned counts and firing indices).

Delays: 1,000, 5,000 and 50,000 flows (`*_d1k`, `*_d5k`, `*_d50k`). Both evaluation windows
average about 1,100 flows per minute of capture (CIC-IDS2017: 2M flows over
30.6 capture hours; CIC-IDS2018: 3M over ~45, its timestamps are not reliable
within a day), so about 1, 5 and 45 minutes at the mean rate. A count-based delay is
shorter in time during bursts, which is where drift happens. Arms: PACE
(`C_early`/`ARF_early`), the live error test (`B_pht`/`ARF_pht`) and the blind
clock (`C_freq`/`ARF_freq`, its interval recalibrated to the delayed PACE arm's
fire count) for HAT and ARF on CIC-IDS2017 and CIC-IDS2018, ten core seeds.
SRP is left out: one SRP run takes many hours.

**Linear learner** (`LR_*`, `models/streaming.py`). A standardised online
logistic regression (river `StandardScaler | SoftmaxRegression`, defaults;
softmax over two classes is logistic regression and takes the string labels).
It is a weaker detector on flow features: learning from every flow of the first
1M anchor flows it reaches attack recall 0.99 at a benign FPR of 5.5% (F1
0.925). Log-scaling and clipping the features did not help (FPR 5.9–6.3%), so
the plain form is kept. Arms on the anchor, twenty seeds (the stability
protocol): PACE (`LR_early`), the live error test (`LR_pht`), the blind clock
(`LR_freq`, calibrated to `LR_early`'s fire count), the defence (`LR_early_rb`),
the chaff adversary (`LR_early_chaff`) and both (`LR_early_chaff_rb`).

Analyses: `scripts/delay_analysis.py` → `results/report/delay/analysis.md`;
the LR arms are in `scripts/security_analysis.py` →
`results/report/security/analysis.md`.

**Label-matched clocks** (`C_freq_lab*`, `ARF_freq_lab*`). Under delay the
error test keeps firing on mistakes the learner has not yet been taught to fix,
so PACE fires 229–445 times, but clustered firings restart windows and it learns
from only 30–41% of the stream; a clock at the same *count* never overlaps and
learns from 58–99.8%. Count matching therefore stops being a timing control, so
a second clock matches PACE's labelled share instead (interval = 5,000 / share,
pinned per dataset and delay; achieved shares within 0.2 points). It is also run
at delay 0, where the count-matched clock already learned from more labels.

## Results — label delay

`results/report/delay/analysis.md`; ten core seeds, paired local F1 with 95% t
intervals; **bold** = interval excludes zero.

| Cell | Delay | PACE − label-matched clock | PACE − count-matched clock | PACE − live error test | Labels PACE / err test |
|---|---|---|---|---|---|
| 2017 HAT | 0 | **+0.123** [+0.016, +0.230] | **+0.126** | +0.052 [−0.037, +0.140] | 29.1 / 21.7% |
| 2017 HAT | 1,000 | +0.097 [−0.035, +0.230] | +0.015 | +0.058 [−0.041, +0.157] | 30.2 / 24.1% |
| 2017 HAT | 5,000 | +0.010 [−0.085, +0.105] | **−0.173** | −0.103 [−0.227, +0.021] | 34.3 / 23.4% |
| 2017 HAT | 50,000 | **−0.077** [−0.135, −0.019] | −0.011 | −0.045 [−0.103, +0.013] | 37.4 / 30.2% |
| 2017 ARF | 0 | **+0.065** [+0.049, +0.082] | **+0.080** | −0.002 [−0.019, +0.015] | 25.1 / 13.5% |
| 2017 ARF | 1,000 | **+0.037** [+0.025, +0.049] | −0.005 | +0.006 [−0.016, +0.029] | 31.2 / 21.3% |
| 2017 ARF | 5,000 | −0.001 [−0.021, +0.019] | **−0.034** | **−0.034** [−0.056, −0.013] | 33.8 / 24.3% |
| 2017 ARF | 50,000 | **−0.042** [−0.061, −0.023] | **−0.051** | **+0.059** [+0.047, +0.070] | 38.3 / 29.0% |
| 2018 HAT | 0 | **+0.053** [+0.030, +0.077] | **+0.057** | −0.007 [−0.028, +0.014] | 32.9 / 10.5% |
| 2018 HAT | 1,000 | −0.063 [−0.218, +0.092] | −0.049 | +0.176 [−0.061, +0.414] | 33.9 / 13.1% |
| 2018 HAT | 5,000 | −0.165 [−0.373, +0.043] | −0.171 | −0.133 [−0.398, +0.132] | 35.6 / 14.2% |
| 2018 HAT | 50,000 | −0.073 [−0.210, +0.064] | −0.054 | −0.052 [−0.189, +0.085] | 38.3 / 17.4% |
| 2018 ARF | 0 | **+0.065** [+0.063, +0.066] | **+0.038** | +0.000 [+0.000, +0.001] | 32.3 / 10.5% |
| 2018 ARF | 1,000 | **+0.020** [+0.019, +0.021] | **+0.005** | **+0.002** [+0.001, +0.003] | 33.2 / 12.3% |
| 2018 ARF | 5,000 | **−0.002** [−0.003, −0.000] | **+0.002** | **−0.008** [−0.010, −0.006] | 35.7 / 14.7% |
| 2018 ARF | 50,000 | **−0.014** [−0.020, −0.007] | **−0.007** | **+0.027** [+0.021, +0.033] | 40.5 / 19.7% |

**With immediate labels the timing result gets stronger.** Against a clock with
PACE's own labelled share, informed placement wins in all four cells, +0.053 to
+0.123 — the count-matched control was conservative, as the paper says.

**One minute shrinks it, five minutes erase it; at 45 minutes a blind clock wins.**
At 1,000 flows the ensembles keep a resolved but smaller gain (+0.037 and +0.020,
from +0.065), while the tree's intervals span zero both ways (+0.097, −0.063):
its runs scatter widely once labels are late. At 5,000
flows no cell keeps a resolved timing gain (one is resolved at −0.002); at
50,000 the label-matched clock beats PACE in three of four cells with an
interval excluding zero (−0.014 to −0.077) and the fourth points the same way.
Informed placement pays only while labels come back within about a minute. We
do not isolate why.

**The monitor does not earn its place under delay either.** Where a label-free
trigger should help most, it helps only on the ensembles at 45 minutes (+0.059,
+0.027), and there PACE learns from 9–21 points more of the stream than the
error test, while a blind clock with PACE's labels does better than both
(0.448 vs 0.406 vs 0.347 on 2017 ARF). At 1 minute its only resolved gain is
+0.002 (2018 ARF); the +0.176 on 2018 HAT is unresolved and comes from the
error-test arm flooding (median benign FPR 0.41). At 5 minutes the monitor costs
a little on the ensembles (−0.034, −0.008); on HAT nothing is resolved.

**Delay degrades every arm and destabilises the tree.** Local F1 falls with
delay everywhere (2017 ARF PACE 0.747 → 0.647 → 0.406). On HAT the error test
alone at 45 minutes flags a median 37% of benign traffic on CIC-IDS2017 (6/10
runs below 0.90 accuracy), and PACE at 5 minutes 16%; at 1 minute the error test
alone floods on 5/10 (2017) and 7/10 (2018) runs against 1/10 and 4/10 at 5
minutes, so the outcome is bimodal and its medians jump between delays. On the ensembles low
cumulative accuracy under delay is missed attacks, not an alert flood (2018 ARF
at 50,000: FPR 0.08, recall 0.83), so the accuracy cut-off is not a collapse
criterion there; ten seeds do not support collapse-count comparisons anyway.

## Results — linear learner

`results/report/security/analysis.md`; twenty seeds.

| Arm | Local F1 med | Collapsed | Benign FPR med / worst | Attack recall | Fires mon / other | Fed attack share | Labels |
|---|---|---|---|---|---|---|---|
| LR, PACE | 0.597 | 0/20 | 0.035 / 0.036 | 0.987 | 54 / 200 | 0.16 | 38.9% |
| LR, live error test | 0.588 | 0/20 | 0.038 / 0.038 | 0.988 | 0 / 216 | 0.15 | 34.0% |
| LR, blind clock | 0.601 | 0/20 | 0.038 / 0.038 | 0.982 | 0 / 253 | 0.20 | 64.8% |
| LR, PACE + rebalancing | 0.590 | 0/20 | 0.043 / 0.045 | 0.958 | 54 / 317 | 0.13 (fed) | 53.7% |
| LR, PACE, chaff | 0.600 | 0/20 | 0.030 / 0.031 | 0.986 | 82 / 230 | 0.31 | 42.8% |
| LR, chaff + rebalancing | 0.587 | 0/20 | 0.045 / 0.047 | 0.959 | 82 / 363 | 0.13 (fed) | 58.1% |

**The linear model never collapsed**: 0 of 120 runs, benign FPR 3.0–4.7% in every
run, even when chaff fired the monitor near 65% of its targets (as on HAT: the
monitor never sees the learner) and the learner was fed a 0.31 attack share.
Among the four learners tested, collapse is confined to the single tree. Its
price is a false-positive floor about six times the healthy tree's (0.035 vs
0.006), at a similar local F1 (0.597 vs 0.594).

The LR is deterministic given the stream: its seeds vary only the monitor's
initialisation (and the chaff draw), and the live error test's twenty runs are
identical. So the LR timing (−0.004 [−0.006, −0.002], the count-matched clock
learning from 65% of the stream against 39%) and monitor (+0.009 [+0.007,
+0.011]) contrasts carry little evidence. The defence is unnecessary on it and
costs a little (−0.006 and −0.014 local F1, recall 0.987 → 0.958).

## Caveats

- Delay is a fixed count of flows; in time it is shorter during bursts, which is
  where drift happens. Ten seeds, two captures, two learners; no SRP, no UNSW-NB15.
- A label-matched clock is calibrated to the mean share of PACE's ten seeds, not
  per seed.
- One linear model with river's default learning rate; its stability may be a
  property of slow stochastic-gradient updates rather than of linearity — not
  isolated.
