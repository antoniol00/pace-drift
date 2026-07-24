# Experiment 6 — Timing vs frequency (statistical hardening + decisive control)

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

## Part A — Statistical hardening (no new runs)

Overall local F1 paired by seed (n=5) between the reactive arm (`B` / `*_sched`)
and the proactive arm (`C_early` / `*_early`), with paired-t and bootstrap CIs,
sign pattern, event stratification, and prequential Kappa from the traces.
Script: `scripts/tier0_analysis.py`.

- **Robust for the strong models.** 6/9 cells are ROBUST (all seeds same sign and
  the paired-t CI excludes 0). The 3 non-robust cells are exactly the **HAT**
  cells (high seed variance); **all six ensemble cells (ARF, SRP × 3 datasets)
  are robust** with tight CIs. Aggregate over the 9 cells: mean Δ **+0.033**, 95%
  CI **[+0.015, +0.051]**, 9/9 positive.
- **Confirmed in imbalance-aware metrics.** Prequential Kappa-M and
  Kappa-Temporal (from per-sample traces) are positive in all 9 cells. On the
  degenerate reactive HAT/2017, Kappa-Temporal is −3.7 (worse than a persistence
  classifier); the proactive arm rescues it (Kappa-M 0.32→0.95).
- **Mechanism check (event stratification).** Splitting real (class-fire) drifts
  into anticipated vs reactive-only: anticipated drifts recover with lower AUD on
  2017 (HAT p=0.011, SRP p<0.001) and on UNSW/ARF (p=0.004), but **not on 2018**
  (p≈0.84–0.98) — the first hint that the 2018 gain is not anticipation. Monitor
  false-alarm rate is high (54–80%).

## Part B — The decisive frequency-matched control (new runs)

For the cells where the proactive arm fires more (ARF/SRP on CIC-IDS2018 and
UNSW-NB15) we add `*_freq`: the same ensemble adapting only in triggered windows
on a **blind periodic clock**, with the interval calibrated so its fire count
matches `*_early` (generalises [experiment 1](01-frequency-control.md)'s C_freq
to the ensembles). Three arms then sit at comparable frequency, and the reported
gain decomposes, paired by seed, as:

```
early − sched (reported) = (freq − sched) frequency  +  (early − freq) timing
```

The clean, airtight statement is the **timing** component `early − freq` (same
fire count, informed vs blind placement). Script: `scripts/tier1_analysis.py`.

| dataset | learner | fires sched/freq/early | F1 sched/freq/early | Δ timing (early−freq) | verdict |
|---|---|---|---|---|:---:|
| CIC-IDS2018 | ARF | 124 / 254 / 255 | 0.913 / 0.888 / 0.926 | **+0.038** [+0.038, +0.039] | **timing** |
| CIC-IDS2018 | SRP | 124 / 256 / 256 | 0.925 / 0.881 / 0.929 | **+0.048** [+0.046, +0.049] | **timing** |
| UNSW-NB15 | ARF | 117 / 172 / 173 | 0.597 / 0.655 / 0.658 | +0.002 [−0.009, +0.014] | ~freq |
| UNSW-NB15 | SRP | 117 / 167 / 167 | 0.695 / 0.743 / 0.746 | +0.002 [−0.005, +0.009] | ~freq |

## Interpretation

- **The split is a property of the dataset, not the learner.** ARF and SRP agree
  cell-for-cell with tight CIs.
- **CIC-IDS2018 — genuinely timing.** Informed placement beats a blind clock at
  the same frequency by +0.038 (ARF) / +0.048 (SRP), and adapting *more* on a
  blind clock actively **hurts** (SRP's blind control 0.881 < the sparse reactive
  0.925). Here "adapt more" is counter-productive; *when* is what matters. The
  reported `early − sched` (+0.004 for SRP) badly understates this, because the
  timing gain is fighting a −0.044 frequency penalty.
- **UNSW-NB15 — frequency, honestly.** The timing component is ~0 (CIs straddle
  0); the gain is adapting more often. UNSW's drift is described in its profile as
  benign↔attack bursts rather than attack-type turnover — placement matters less.
- **Sharper claim.** Informed timing helps in proportion to how *structured* the
  drift is: discrete blocks (2017, 2018) → timing; dense bursts (UNSW) →
  frequency.

## Honesty note / what this does *not* prove

The block-vs-burst pattern is an **empirical observation across three real
datasets (n=3)**, not a demonstrated mechanism. Two **pre-specified** synthetic
experiments tried to isolate a drift-structure mechanism and **both failed** to:

- `scripts/synthetic_structure.py` (sweep drift *block size*): no monotone
  relationship — large blocks let a blind clock re-adapt mid-block, and uniformly
  spaced drifts align with a count-matched uniform clock.
- `scripts/synthetic_clustering.py` (sweep *temporal concentration* φ of onsets):
  the timing benefit was large and positive at *every* φ (~+0.2), not increasing
  with concentration.

Per the pre-registered decision rule the drift-structure claim is **retreated**.
What the second synthetic *does* show is the **core** result in a controlled
setting: informed timing beats a blind clock at matched adaptation frequency
(+0.21 even with uniformly spaced drifts). So: report the real-dataset split as an
empirical observation, claim the "when to adapt" benefit, and do **not** claim a
structure mechanism in the abstract. See
`results/report/synthetic_clustering/analysis.md`.

## Reproducing

```bash
# Part A — statistical hardening over existing runs (no new compute)
uv run python scripts/tier0_analysis.py

# Part B — frequency-matched controls (calibrated interval per cell)
uv run python run_experiment.py --phase ARF_freq --dataset cicids2018 --seeds 0,1,2,3,4 --n-jobs 5 --periodic-adapt-interval 11587
uv run python run_experiment.py --phase SRP_freq --dataset cicids2018 --seeds 0,1,2,3,4 --n-jobs 5 --periodic-adapt-interval 11505
uv run python run_experiment.py --phase ARF_freq --dataset unsw       --seeds 0,1,2,3,4 --n-jobs 5 --periodic-adapt-interval 9549
uv run python run_experiment.py --phase SRP_freq --dataset unsw       --seeds 0,1,2,3,4 --n-jobs 5 --periodic-adapt-interval 9857
uv run python scripts/tier1_analysis.py
```
