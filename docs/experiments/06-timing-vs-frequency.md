# Experiment 6 — Timing vs frequency (statistical hardening + decisive controls)

## Question

[Experiment 4](05-external-validity.md) found the proactive early trigger positive
in all 9 dataset × base-learner cells, but with a caveat: on the external datasets
the proactive arm *fires more often* than the reactive schedule (1.4–2.5×), where
on CICIDS2017 it fires *fewer* times. Two objections follow, and this experiment
answers both from the already-persisted artifacts plus one small set of new runs:

> **(A)** Does the reported reactive→proactive gain survive proper inferential
> scrutiny, or is it seed noise?
> **(B)** On the external datasets, is the gain because the model adapts *earlier*
> (informed timing) or merely *more often* (frequency)?
> **(C)** If placement matters, is it the monitor that informs it, or the error
> test PACE runs alongside it?

## Part A — Statistical hardening (no new runs)

Overall local F1 paired by seed (**n=10**) between the reactive arm
(`B` / `*_sched`) and the proactive arm (`C_early` / `*_early`), with paired-t and
bootstrap CIs, sign pattern, event stratification, and prequential Kappa from the
traces. Script: `scripts/tier0_analysis.py`.

- **Robust for the strong models.** **7/9 cells have a paired-t CI excluding 0**
  (5 ROBUST — every seed agrees in sign as well — plus 2 CI-signif where a single
  seed dissents). The 2 remaining WEAK cells are **HAT** ones (high seed
  variance); every ensemble cell has a CI excluding 0, five of the six with
  10/10 seeds agreeing at Wilcoxon p=0.002. Aggregate over the 9 cells: mean Δ
  **+0.043**, 95% CI **[+0.021, +0.065]**, 9/9 positive (sign test p=0.0039).
- **Confirmed in imbalance-aware metrics.** Prequential Kappa-M (from per-sample
  traces) is positive for both arms in all 9 cells, and the proactive arm
  improves both Kappa-M and Kappa-Temporal in all 9. Kappa-Temporal itself is
  negative only on the degenerate HAT/2017 cell: −1.93 reactive (worse than a
  persistence classifier), improved to −1.23, with Kappa-M lifted 0.58→0.68.
- **Mechanism check (event stratification).** Splitting real (class-fire) drifts
  into anticipated vs reactive-only: anticipated drifts recover with lower AUD on
  2017 (HAT p=0.037, ARF p=0.013, SRP p<0.001) and on UNSW (ARF p<0.001, SRP
  p=0.016), but **not on 2018** (p≈0.94–0.99) nor for HAT on UNSW (p=1.00:
  anticipated drifts there have the larger median AUD, 1528 vs 882) — the first hint that the 2018 gain
  is not anticipation. Monitor false-alarm rate is high (30–80%).

## Part B — The decisive frequency-matched control (new runs)

Every one of the nine cells gets `*_freq`: the same learner adapting only in
triggered windows on a **blind periodic clock**, with the interval calibrated so
its fire count matches that cell's `*_early` (generalises
[experiment 1](01-frequency-control.md)'s C_freq to every dataset and learner).
Three arms then sit at comparable frequency, and the reported gain decomposes,
paired by seed, as:

```
early − sched (reported) = (freq − sched) frequency  +  (early − freq) timing
```

The clean, airtight statement is the **timing** component `early − freq` (same
fire count, informed vs blind placement). It is also conservative: the clock's
windows never overlap, so at a matched count it learns from **more labels** than
the proactive arm in every cell (a firing opens a 5,000-sample window and a
firing inside an open window restarts it). Script: `scripts/tier1_analysis.py`.

10 seeds per cell; labels = share of the evaluated stream learned from:

| dataset | learner | fires sched/freq/early | labels % sched/freq/early | F1 sched/freq/early | Δ frequency (freq−sched) | Δ timing (early−freq) | verdict |
|---|---|---|---|---|---|---|:---:|
| CIC-IDS2017 | HAT | 233 / 162 / 169 | 25.5 / 41.5 / 29.1 | 0.567 / 0.473 / 0.599 | −0.094 [−0.221, +0.033] | **+0.126** [+0.016, +0.236] | **timing** |
| CIC-IDS2017 | ARF | 233 / 110 / 110 | 25.5 / 28.0 / 25.1 | 0.717 / 0.667 / 0.747 | −0.050 [−0.065, −0.036] | **+0.080** [+0.064, +0.097] | **timing** |
| CIC-IDS2017 | SRP | 233 / 90 / 90 | 25.5 / 22.8 / 21.3 | 0.737 / 0.721 / 0.768 | −0.016 [−0.030, −0.001] | **+0.047** [+0.030, +0.064] | **timing** |
| CIC-IDS2018 | HAT | 124 / 306 / 306 | 8.9 / 51.7 / 32.9 | 0.828 / 0.843 / 0.899 | +0.015 [−0.059, +0.089] | **+0.057** [+0.017, +0.097] | **timing** |
| CIC-IDS2018 | ARF | 124 / 254 / 253 | 8.9 / 43.1 / 32.3 | 0.913 / 0.888 / 0.926 | −0.025 [−0.027, −0.024] | **+0.038** [+0.037, +0.038] | **timing** |
| CIC-IDS2018 | SRP | 124 / 256 / 254 | 8.9 / 43.4 / 32.3 | 0.924 / 0.881 / 0.929 | −0.042 [−0.045, −0.040] | **+0.047** [+0.047, +0.048] | **timing** |
| UNSW-NB15 | HAT | 117 / 214 / 214 | 32.4 / 64.5 / 50.4 | 0.355 / 0.475 / 0.452 | +0.120 [−0.027, +0.267] | −0.023 [−0.118, +0.072] | unresolved |
| UNSW-NB15 | ARF | 117 / 172 / 171 | 32.4 / 52.1 / 44.5 | 0.603 / 0.652 / 0.662 | +0.049 [+0.038, +0.061] | +0.010 [+0.001, +0.019] | frequency |
| UNSW-NB15 | SRP | 117 / 167 / 167 | 32.4 / 50.5 / 43.4 | 0.697 / 0.746 / 0.745 | +0.050 [+0.040, +0.059] | −0.001 [−0.008, +0.006] | frequency |

**Aggregate over the nine cells** (cell means as the unit of replication):
reported gain **+0.043** [+0.021, +0.065] (9/9 positive) = frequency **+0.001**
[−0.049, +0.050] (4/9 positive) + timing **+0.042** [+0.008, +0.077] (7/9
positive, each with a CI excluding 0). The aggregate gain is carried by timing.

On UNSW the timing component stays an order of magnitude below the other two
datasets' on the ensembles: on ARF it is a small residual (+0.010, ~17% of that
cell's +0.059 total) whose CI only just clears 0, on SRP it is indistinguishable
from 0, and on HAT neither component resolves at ten seeds. The
frequency-dominated reading of UNSW therefore stands.

> **Calibration is load-bearing here.** The blind clock must fire as often as
> `*_early` *on that dataset and learner*: 9549/9857 (ARF/SRP) on UNSW against
> 11587/11505 on 2018, 17727/21666 on 2017, and 12000/9640/7710 for HAT on
> 2017/2018/UNSW. Running UNSW at the 2017-2018 interval fires 142 times instead of
> 172, and the resulting frequency gap leaks into the timing residual — enough to
> reverse this section's conclusion. The calibrated values are pinned in
> `configs/datasets/*.yaml` under `by_phase:` and covered by a test, so they
> apply automatically; never pass them as a command-line flag.

![Timing vs frequency at matched adaptation count](../figures/fig13_timing_vs_frequency.png)

## Part C — Live reactive control (new runs)

The reactive arms replay a schedule recorded once per dataset from **HAT at seed
0**: `ARF_sched` / `SRP_sched` replay HAT's error, not their own, and `B` at seeds
1–9 replays the seed-0 run's. The error test inside `*_early` instead watches each
run's own predictions, so `early − sched` changes two things at once, and the
blind clock cannot separate them (both of PACE's tests inform its placement).
`B_pht` / `ARF_pht` / `SRP_pht` run PACE's error Page-Hinkley (δ = 0.002, river
defaults otherwise) **live on the learner's own predictions with no monitor**, so
`early − pht` is what the monitor adds.

10 seeds per cell; `monitor` = PACE firings from the monitor's Page-Hinkley
(including the ≤0.2/run that coincide with an error firing):

| dataset | learner | fires sched/pht/early | monitor | labels % sched/pht/early | F1 sched/pht/early | Δ live (pht−sched) | Δ monitor (early−pht) |
|---|---|---|---:|---|---|---|---|
| CIC-IDS2017 | HAT | 233 / 161 / 169 | 52.9 | 25.5 / 21.7 / 29.1 | 0.567 / 0.547 / 0.599 | −0.020 [−0.094, +0.055] | +0.052 [−0.037, +0.140] |
| CIC-IDS2017 | ARF | 233 / 59 / 110 | 52.9 | 25.5 / 13.5 / 25.1 | 0.717 / 0.749 / 0.747 | +0.032 [+0.018, +0.046] | −0.002 [−0.019, +0.015] |
| CIC-IDS2017 | SRP | 233 / 37 / 90 | 52.9 | 25.5 / 8.9 / 21.3 | 0.737 / 0.767 / 0.768 | +0.031 [+0.022, +0.039] | +0.001 [−0.015, +0.017] |
| CIC-IDS2018 | HAT | 124 / 139 / 306 | 161.5 | 8.9 / 10.5 / 32.9 | 0.828 / 0.906 / 0.899 | +0.078 [+0.012, +0.145] | −0.007 [−0.028, +0.014] |
| CIC-IDS2018 | ARF | 124 / 96 / 253 | 161.5 | 8.9 / 10.5 / 32.3 | 0.913 / 0.925 / 0.926 | +0.012 [+0.011, +0.014] | +0.000 [+0.000, +0.001] |
| CIC-IDS2018 | SRP | 124 / 88 / 254 | 161.5 | 8.9 / 9.7 / 32.3 | 0.924 / 0.932 / 0.929 | +0.008 [+0.004, +0.012] | −0.003 [−0.005, −0.001] |
| UNSW-NB15 | HAT | 117 / 122 / 214 | 80.9 | 32.4 / 33.4 / 50.4 | 0.355 / 0.427 / 0.452 | +0.072 [−0.090, +0.234] | +0.025 [−0.099, +0.148] |
| UNSW-NB15 | ARF | 117 / 89 / 171 | 80.9 | 32.4 / 26.6 / 44.5 | 0.603 / 0.584 / 0.662 | −0.019 [−0.032, −0.005] | **+0.078** [+0.070, +0.086] |
| UNSW-NB15 | SRP | 117 / 87 / 167 | 80.9 | 32.4 / 26.1 / 43.4 | 0.697 / 0.697 / 0.745 | +0.001 [−0.014, +0.016] | **+0.048** [+0.036, +0.060] |

**Aggregate over the nine cells:** Δ live **+0.022** [−0.006, +0.049] (7/9
positive); Δ monitor **+0.021** [−0.002, +0.045] (6/9 positive) — **not
resolved**.

- **CIC-IDS2017 and CIC-IDS2018: the live error test does the informing.** In
  five of the six cells the live control comes within 0.007 of PACE while
  learning from a third to a half as many labels; the sixth (HAT/2017) favours
  PACE by +0.052 with a CI that includes 0 at n=10. The ensembles' gain over
  `*_sched` there is the gain of a live over a replayed error test, not of the
  monitor. This also resolves the Part A tension: 2018 is where per-event
  anticipation is invisible, and it is where the monitor adds nothing.
- **UNSW-NB15: the monitor helps by adapting more.** +0.078 / +0.048 over the
  live control, but with more firings and labels (43–45% vs 26–27%), and Part B
  shows a blind clock at PACE's count doing as well.
- **The monitor's firings come on top.** The error test inside PACE fires about
  as often as it does alone (ARF/2017 57.5 vs 59, SRP/2017 37.2 vs 37, …), except
  HAT/2017 (116 vs 161). Monitor firings are identical across learners (it never
  sees the learner's output): 53 / 162 / 81 per run on 2017 / 2018 / UNSW.
  This is why PACE fires *less* than the schedule on 2017 (169 / 110 / 90 vs
  233): the 233 are one HAT run's firings; live, the ensembles raise far fewer
  error alarms (59 / 37) because they err less, and the monitor's 53 come on
  top. On HAT both act: the live error test fires 161 rather than 233, and the
  monitor's early windows cut it further to 116.
- **Replaying is a handicap in itself** (+0.022 on aggregate for the live arm).
  On 2017 the schedule was recorded from a HAT run that collapsed (seed 0,
  cumulative accuracy 0.43; `B_pht` seed 0 collapses identically), so it carries
  the firings of that collapse.

What this leaves: informed placement beats blind placement (Part B), but on real
traffic with immediate labels **we could not show that the covariate-shift
monitor places adaptation better than a live error test**. Label delay — which
slows the error test but not the monitor — is where that question should be
asked next.

> At seed 0 the live arm shares 194/233 (2017), 112/124 (2018) and 117/117 (UNSW)
> firing indices with the recorded schedule; the 2017/2018 schedules were recorded
> in July and predate later engine changes. The live arm is the arm as run now.

`B_pht` collapses (cumulative accuracy < 0.90) on 4 of its 10 seeds on 2017;
that is fewer than the 20 seeds a collapse-count comparison needs, so no
stability ranking against the other arms is drawn from it.

## Interpretation

- **The split is a property of the dataset, not the learner.** ARF and SRP agree
  cell-for-cell with tight CIs.
- **CIC-IDS2017 and CIC-IDS2018 — genuinely timing.** Informed placement beats
  a blind clock at the same frequency in all six cells (+0.038 to +0.126), and
  adapting *more* on a blind clock usually **hurts** (SRP/2018's blind control
  0.881 < the sparse reactive 0.924, on five times the labels). Here *when* is
  what matters — but Part C shows the *when* comes from the live error test, not
  from the monitor. The reported `early − sched` (+0.005 for SRP/2018) badly
  understates the timing term, which is fighting a −0.042 frequency penalty.
- **UNSW-NB15 — frequency, honestly.** The timing component is ~0 (CIs straddle
  0); the gain is adapting more often. UNSW's drift is described in its profile as
  benign↔attack bursts rather than attack-type turnover — placement matters less.
## What this does *not* claim

The per-dataset split is an **empirical observation across three real datasets**,
and it is reported as exactly that. We do **not** claim a mechanism linking drift
structure to the size of the timing effect: with three datasets there is no way to
separate drift structure from every other thing that differs between captures, so
any such explanation would be a story told after the fact.

What *is* demonstrated, on a stream whose drift structure is known by
construction rather than inferred, is the core comparison itself — informed
timing beats a blind clock at matched adaptation frequency
([experiment 8](08-synthetic-timing.md)).

## Reproducing

```bash
# Part A — statistical hardening over existing runs (no new compute)
uv run python scripts/tier0_analysis.py

# Part B — frequency-matched controls. The blind-clock interval is calibrated per
# (dataset, learner) and pinned in configs/datasets/*.yaml under `by_phase:`, so
# it is applied automatically — never pass it as a flag.
uv run python run_experiment.py --phase ARF_freq --dataset cicids2018 --seeds 0,1,2,3,4,5,6,7,8,9 --n-jobs 5
uv run python run_experiment.py --phase SRP_freq --dataset cicids2018 --seeds 0,1,2,3,4,5,6,7,8,9 --n-jobs 5
uv run python run_experiment.py --phase ARF_freq --dataset unsw       --seeds 0,1,2,3,4,5,6,7,8,9 --n-jobs 5
uv run python run_experiment.py --phase SRP_freq --dataset unsw       --seeds 0,1,2,3,4,5,6,7,8,9 --n-jobs 5
# (C_freq / ARF_freq / SRP_freq likewise on every dataset, including the 2017 anchor)

# Part C — live reactive controls (no schedule), every dataset
for d in cicids2017 cicids2018 unsw; do for p in B_pht ARF_pht SRP_pht; do
  uv run python run_experiment.py --phase $p --dataset $d --seeds 0,1,2,3,4,5,6,7,8,9 --n-jobs 10
done; done
uv run python scripts/tier1_analysis.py
```
