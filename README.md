# PACE Drift — *When to adapt*: a cheap unsupervised covariate-shift trigger for streaming intrusion detection

**PACE** — *Proactive Adaptation via Covariate-shift Estimation*.

**Abstract.** Streaming intrusion detectors degrade as attack traffic drifts away from the distribution they were trained on, and the standard remedy — retraining once classification error rises — inherits the delay of labelled feedback. This project treats **when to adapt** as a design decision in its own right. PACE is an unsupervised trigger: a monitor watches the reconstruction error of *incoming traffic* and initiates adaptation before the error signal responds, with learning confined to short triggered windows instead of continuous supervision. The claim is validated prequentially on three intrusion-detection datasets (CICIDS2017, CIC-IDS2018, UNSW-NB15) × three streaming learners (HAT, ARF, SRP), 10 seeds each, under **budget-matched controls** that separate adaptation *timing* from adaptation *frequency* and from *model capacity*. PACE improves sustained detection quality in all nine cells, and a frequency-matched blind clock isolates timing as the cause on CICIDS2017/2018. A final experiment argues that trigger quality must be judged by **closed-loop stability** rather than detection accuracy, and reports an uncomfortable finding in that frame: at matched fire count over 20 seeds, *every* trigger tested — ours, two published unsupervised detectors and two cheap statistics — intermittently drives the learner into a catastrophic false-positive regime, through a training-distribution mechanism that detection metrics and final-window F1 both conceal. See [Results](#8-results).

> **Note on the name.** The project began as *DriftGAN*, testing whether GANs could improve drift **adaptation** (generating synthetic samples of a new attack) and **anticipation**. Under budget-matched controls the generative *augmentation* gave no measurable benefit, while the anticipation branch — which generates nothing — did. The project was renamed to **PACE Drift** to match what the evidence supports: the contribution is the *pacing* of adaptation, not generation. The full generative code path is kept so the controlled refutation stays reproducible, and the Python package is still `driftgan` for import stability. The full reasoning trail is in [docs/paper/idea-central.md](docs/paper/idea-central.md).

## Key results at a glance

- **The proactive trigger works, and it is not a crutch for a weak model** — swapping a reactive (error-based) trigger for the unsupervised covariate-shift trigger improves sustained local F1 in **all 9 (dataset × base-learner) cells** over 10 seeds each; aggregated, **+0.043 [95% CI +0.021, +0.065]** (sign test p=0.0039). The paired CI excludes 0 in **7/9 cells — every ensemble cell**, five of them with 10/10 seeds agreeing (Wilcoxon p=0.002); the two exceptions are the high-variance HAT cells. It holds under imbalance-aware κ in all nine.
- **It is timing, not just frequency** — against a blind periodic clock firing the *same number of times*, informed placement gains **+0.038 (ARF) / +0.048 (SRP)** on CIC-IDS2018, where adapting more on a blind schedule actively *hurts* (0.881 vs the sparser reactive 0.925). A controlled synthetic stream reproduces the advantage (+0.21 at matched frequency).
- **Honest scope** — on UNSW-NB15 the benefit is adaptation *frequency*, not timing (+0.002, CI straddles 0). Two **pre-specified** synthetic experiments failed to isolate a drift-structure mechanism, so the split is reported as an empirical observation and no mechanism is claimed.
- **The adaptation loop is fragile, and detection metrics do not show it** — at matched fire count over 20 seeds, *every* trigger tested drives the learner into a degenerate regime on a fifth to three-fifths of runs: autoencoder **4/20**, HDDDM 5/20, PCA 6/20, D3 (CIKM'19) 10/20, mean-shift 12/20. Badly placed windows feed it up to **0.46 attack fraction against a 0.20 base rate**, skewing its prior until the false-positive rate reaches **0.60–0.70**. Our monitor is *not* measurably more stable than the published detectors — only the trivial mean-shift floor separates (p=0.022).
- **Model capacity matters more than the augmenter** — budget-matched to ~233 sparse adaptations, ensembles still lead a single tree by ~+0.15 sustained F1, while continuous supervision (~2M updates) adds only ~0.05–0.08. The practical recipe is **ensemble + proactive triggered adaptation**.
- **Sub-millisecond cost** — the monitor is a linear-algebra pass on a small buffer; p95 latency stays under 0.5 ms/sample.

![Timing vs frequency at matched adaptation count](docs/figures/fig13_timing_vs_frequency.png)

📄 **Start here:** [the reasoning trail](docs/paper/idea-central.md) (hypothesis → findings, in plain language) · [related work & the unsupervised comparator](docs/paper/related-work.md) · [per-experiment analysis](docs/experiments/) · [architecture](docs/architecture.md) · [methodology](docs/methodology.md).

## 1. Motivation

Network intrusion detection systems trained offline face non-stationary traffic: new attack campaigns, tool variants and scanning behaviours shift the input distribution (covariate drift) and the input→label relation (real drift). A frozen model silently decays; error-based detectors (Page-Hinkley, ADWIN) react only *after* misclassifications accumulate — precisely the window in which attacks go undetected. Labels are also the scarce resource in a real deployment: continuous supervision is the one thing you cannot assume.

That makes **when to adapt** the interesting lever. A signal computed from *unlabelled input* can fire before the error signal does, and adaptation can be confined to short windows around those firings. The questions this repository answers experimentally are: does that help, is it *timing* or merely *more often*, does it survive on strong models, and what makes a good trigger.

## 2. Contributions

1. **An unsupervised *when-to-adapt* policy** — a reconstruction-error monitor over incoming traffic feeding a dual (recon + class) Page-Hinkley trigger, with learning gated to triggered windows. (`src/driftgan/early/`)
2. **Controls that isolate the mechanism** — a frequency-matched blind clock at identical fire count separates *timing* from *frequency* (`C_freq`, `ARF_freq`, `SRP_freq`); budget-matched ensembles separate *model capacity* from *continuous supervision* (`ARF_sched`, `SRP_sched`).
3. **Closed-loop stability as the criterion for trigger quality** — monitors compared by what they do to the learner (collapse rate, false-positive regime, the label composition of triggered windows), not by detection metrics. (`scripts/comparator_analysis.py`)
4. **A published unsupervised detector as an external comparator** — D3 (Gözüaçık et al., CIKM 2019) implemented and run at the identical protocol and matched fire count. (`early/shift_monitors.py:D3Monitor`)
5. **Cross-dataset validation** — the base-learner × trigger 2×2 replicated on CIC-IDS2018 and UNSW-NB15, with dataset selection driven by a label-ordering diagnostic. (`scripts/scan_drift_window.py`)
6. **Inferential rigour** — paired-t and bootstrap CIs per cell, prequential Kappa-M/Temporal, per-event stratification of anticipated vs reactive drifts. (`scripts/tier0_analysis.py`)
7. **Pre-registered synthetic experiments** — two attempts to turn an observed cross-dataset split into a mechanism, both reported despite being negative. (`scripts/synthetic_structure.py`, `synthetic_clustering.py`)
8. **Shared drift schedule and per-event recovery protocol** — identical adaptation indices across compared phases make per-event pairing valid; RT/AUD/min-F1/no-recovery with explicit handling of F1-undefined windows. (`drift/schedule.py`, `metrics/recovery.py`)
9. **Reproducibility & cost accounting** — multi-seed CLI, manifests (git/config/CSV hashes), per-component latency breakdown and RSS tracking.

## 3. Related work

The monitor belongs to the **unsupervised / feature-space drift detection** family, and the contribution is positioned as the *policy*, not the detector. Full treatment, including verification status of every citation, in **[docs/paper/related-work.md](docs/paper/related-work.md)**.

- **Supervised (error-based) detection** — the reactive arm we improve on: Page-Hinkley (Page, 1954); DDM (Gama et al., 2004); ADWIN (Bifet & Gavaldà, *SDM* 2007); KSWIN (Raab et al., *Neurocomputing* 2020). ADWIN and KSWIN are run here as baselines and do not beat plain PHT.
- **Unsupervised / feature-space detection** — the closest family: D3 (Gözüaçık et al., *CIKM* 2019), used as our comparator; HDDDM (Ditzler & Polikar, *IEEE CIDUE* 2011); autoencoder reconstruction error (Jaworski et al., *ICAISC* 2020). Surveys: Gemaque et al. (*WIREs DMKD* 2020); Lu et al. (*IEEE TKDE* 2019).
- **Adaptive stream learners** — Hoeffding Adaptive Tree (Bifet & Gavaldà, *IDA* 2009); Adaptive Random Forest (Gomes et al., *Machine Learning* 2017); Streaming Random Patches (Gomes, Read & Bifet, *ICDM* 2019).
- **Drift in network intrusion detection** — CICIDS2017 (Sharafaldin et al., 2018) is the standard testbed precisely because its attacks arrive in temporal blocks; recent drift-aware NIDS work triggers on error or reports end accuracy without separating timing, frequency and capacity — the methodological gap this work targets.

> **Stated plainly:** monitoring autoencoder reconstruction error to detect drift already exists. The signal is not novel; the *policy*, the controls that isolate its mechanism, and the closed-loop stability criterion are the contribution.

## 4. Methodology

One prequential engine (`src/driftgan/experiments/runner.py`) runs every phase; phases differ only in **model**, **trigger** and **augmenter**:

```mermaid
flowchart LR
    S[CSV stream] --> P[preprocess row]
    P --> M[model predict]
    M --> E[0/1 error]
    E --> T{trigger?}
    T -- "A: never" --> S
    T -- "B / *_sched: PHT or shared schedule" --> W[retrain window]
    T -- "*_early: PHT_recon or PHT_class" --> W
    T -- "*_freq: blind periodic clock" --> W
    T -- "ARF/SRP: none (learn every sample)" --> S
    W --> G{augmenter?}
    G -- "C / C_cond / C': synthetic ATTACK samples" --> L[learn_one synthetic]
    G -- "none" --> S
    L --> S
```

**The contribution and its controls** — the comparisons that carry the claim:

| Phase | Model | Trigger | Isolates |
|---|---|---|---|
| B / `*_sched` | HAT / ARF / SRP | PHT / shared schedule | reactive baseline (error-driven) |
| `C_early` / `ARF_early` / `SRP_early` | HAT / ARF / SRP | dual PHT (recon + class) | **the proactive policy** |
| `C_freq` / `ARF_freq` / `SRP_freq` | HAT / ARF / SRP | blind periodic clock, matched count | adaptation *timing* vs *frequency* |
| `ARF` / `SRP` | ensemble | none (continuous learning) | upper bound at full supervision |
| `C_early_pca` / `_mean` / `_d3` | HAT | dual PHT, monitor swapped | which shift signal makes a good trigger |
| `B_adwin` / `B_kswin` | HAT | ADWIN / KSWIN | drift-detector choice |
| A | HAT | none (frozen) | lower-bound control |

**Original generative phases** (kept reproducible; see the note above): `C` (vanilla GAN), `C_cond` (subtype-conditional WGAN-GP) and `C'` (bootstrap+jitter, the budget-matched augmentation control).

Details: [architecture.md](docs/architecture.md) (component design) · [methodology.md](docs/methodology.md) (protocol: schedules, recovery semantics, statistics) · [gan_design.md](docs/gan_design.md) (generative internals).

## 5. Datasets

Three CICFlowMeter-family, naturally-drifting network captures. None is included in this repository; set `csv_path` in `configs/base.yaml` or the per-dataset profile in `configs/datasets/`.

| Dataset | Window used | Why |
|---|---|---|
| **CICIDS2017** (anchor) | 2M samples from row 540,000 | attacks arrive in temporal blocks after a benign prefix |
| **CIC-IDS2018** | 3M samples from row 11,000,000 | richest attack-diversity block (selected by `scripts/scan_drift_window.py`) |
| **UNSW-NB15** | 1.7M samples from row 1,750,000 | different attack taxonomy; drift is benign↔attack bursts |

Preprocessing per row: drop identifier columns (Flow ID, IPs, Timestamp), coerce features to float, replace ±inf/NaN with 0, binarize the label to BENIGN/ATTACK while keeping the raw subtype for conditioning and ground-truth schedules. IoT candidates (CICIoT2023, Edge-IIoTset) were **rejected** by a label-ordering diagnostic — one is fully shuffled (no temporal drift), the other class-sorted (covariate shift and label change coincide, so there is nothing to anticipate).

## 6. Experimental protocol

- **Shared schedule**: pass 0 (Phase B dynamics) records PHT fire indices → scheduled phases replay them, making per-event pairing valid; a report-only PHT measures detection delay. The `*_early` phases are exempt by design (their claim is anticipation).
- **Seeds**: 5 per phase (`--seeds 0,1,2,3,4`), seeding python/numpy/torch/model/monitor.
- **Metrics**: rolling F1 (window 5,000) snapshotted every 500 samples; *overall local F1* = mean valid-window F1 (windows with no positive label are undefined and excluded, never counted as 0); per-event RT/AUD/min-F1/no-recovery; anticipation lead time and false-alarm rate; prequential Kappa-M and Kappa-Temporal; latency percentiles and peak RSS.
- **Statistics**: paired-t and percentile-bootstrap 95% CIs paired by seed; Wilcoxon signed-rank paired by drift index; Cliff's delta. Under bimodal outcomes (see §8, experiment 7) per-seed values and collapse counts are reported instead of a mean ± CI, which would be misleading.
- **Calibration discipline**: every monitor's PHT threshold is calibrated on its own cached score series to match the reference fire count, so arms differ in the *signal*, not in how often they fire.

## 7. Reproducing the results

```bash
git clone git@github.com:antoniol00/pace-drift.git && cd pace-drift
make setup                 # creates .venv from uv.lock (needs uv)
make test                  # unit + end-to-end synthetic smoke tests
# point configs/base.yaml:csv_path at your CICIDS2017 CSV, then:
make smoke                 # 10K-sample sanity run

# core experiment
make schedule && make run-all SEEDS=0,1,2,3,4
make report                                        # tables + figures

# controls and baselines
make run-C-freq         SEEDS=0,1,2,3,4            # frequency control (exp 1)
make run-sota           SEEDS=0,1,2,3,4            # ADWIN/KSWIN + ARF/SRP (exp 3)
make run-sota-matched   SEEDS=0,1,2,3,4            # budget-matched ARF/SRP (exp 3b)
make run-early-ensemble SEEDS=0,1,2,3,4            # ARF/SRP + proactive trigger (exp 3c)

# external validity (exp 4)
make schedule-2018 && make run-min-2018
make schedule-unsw && make run-min-unsw

# timing vs frequency on the external datasets (exp 6)
uv run python run_experiment.py --phase ARF_freq --dataset cicids2018 --seeds 0,1,2,3,4 --n-jobs 5 --periodic-adapt-interval 11587
uv run python run_experiment.py --phase SRP_freq --dataset cicids2018 --seeds 0,1,2,3,4 --n-jobs 5 --periodic-adapt-interval 11505
uv run python run_experiment.py --phase ARF_freq --dataset unsw       --seeds 0,1,2,3,4 --n-jobs 5 --periodic-adapt-interval 9549
uv run python run_experiment.py --phase SRP_freq --dataset unsw       --seeds 0,1,2,3,4 --n-jobs 5 --periodic-adapt-interval 9857

# unsupervised comparator (exp 7)
uv run python run_experiment.py --phase C_early_d3 --seeds 0,1,2,3,4 --n-jobs 5

# analyses and figures
uv run python scripts/tier0_analysis.py        # CIs, Kappa, event stratification
uv run python scripts/tier1_analysis.py        # timing vs frequency decomposition
uv run python scripts/comparator_analysis.py   # monitor comparison + stability
uv run python scripts/paper_figures.py         # fig1-3, 6, 8-13
uv run python scripts/reviewer_analysis.py     # fig4 resources, fig5 sensitivity
uv run python -m scripts.synthetic_clustering --all   # pre-registered synthetic
```

Each run writes `results/runs/[<dataset>/]<phase>/seed_<s>/` with `manifest.json` (git hash, config hash, CSV SHA-256, package versions), `snapshots.parquet`, `drift_events.json`, `cost.json`, `trace.csv.gz`, plus per-event recovery under `results/recovery/`. `results/` is gitignored.

## 8. Results

### 8.1 The proactive trigger across datasets and learners

Overall local F1, reactive → proactive, 10 seeds ([experiment 4](docs/experiments/05-external-validity.md)):

| Base learner | CICIDS2017 | CIC-IDS2018 | UNSW-NB15 |
|---|---|---|---|
| HAT | 0.573 → 0.595 (+0.022) | 0.830 → 0.901 (+0.071) | 0.463 → 0.474 (+0.011) |
| ARF | 0.718 → 0.748 (+0.030) | 0.913 → 0.926 (+0.013) | 0.597 → 0.658 (+0.061) |
| SRP | 0.733 → **0.769** (+0.037) | 0.925 → 0.929 (+0.004) | 0.695 → **0.746** (+0.050) |

Positive in all 9 cells. Under paired CIs over 10 seeds ([experiment 6](docs/experiments/06-timing-vs-frequency.md)) the CI excludes 0 in **7/9 cells — every ensemble cell** — and the two exceptions are the high-variance HAT ones. Aggregated over cells: **+0.043 [+0.021, +0.065]**. Prequential Kappa-M and Kappa-Temporal are positive in all nine.

### 8.2 Timing or just frequency?

A blind periodic clock, fire-count matched to the proactive arm, splits the reported gain into a frequency part and a timing part:

| Dataset | Learner | fires sched/freq/early | F1 sched/freq/early | **Δ timing** (early−freq) | verdict |
|---|---|---|---|---|:---:|
| CIC-IDS2018 | ARF | 124 / 254 / 255 | 0.913 / 0.888 / 0.926 | **+0.038** [+0.038, +0.039] | **timing** |
| CIC-IDS2018 | SRP | 124 / 256 / 256 | 0.925 / 0.881 / 0.929 | **+0.048** [+0.046, +0.049] | **timing** |
| UNSW-NB15 | ARF | 117 / 172 / 173 | 0.597 / 0.655 / 0.658 | +0.002 [−0.009, +0.014] | ~freq |
| UNSW-NB15 | SRP | 117 / 167 / 167 | 0.695 / 0.743 / 0.746 | +0.002 [−0.005, +0.009] | ~freq |

On CIC-IDS2018 adapting *more* on a blind clock is actively **worse than the sparser reactive baseline** (0.881 vs 0.925) — the benefit is genuinely *where* adaptation lands. On UNSW-NB15 the gain is frequency. The split is consistent across two independent learners, so it is a property of the dataset; two **pre-specified** synthetic experiments could not isolate a drift-structure mechanism, and per the pre-registered decision rule **no mechanism is claimed**. The second synthetic does corroborate the core result (+0.21 informed vs blind at matched frequency).

### 8.3 What makes a good trigger: closed-loop stability

All monitors share the pipeline, learner and policy, each calibrated to the same fire count, over **20 seeds** ([experiment 7](docs/experiments/07-unsupervised-comparator.md)):

| Monitor | collapsed seeds | Fisher vs AE | FPR med / worst | train atk% |
|---|---:|---:|---:|---:|
| autoencoder | 4/20 | — | 0.006 / 0.662 | 0.26 |
| HDDDM (CIDUE'11) | 5/20 | 1.000 | 0.005 / 0.603 | 0.31 |
| PCA linear | 6/20 | 0.716 | 0.006 / 0.689 | 0.37 |
| **D3** (CIKM'19) | 10/20 | 0.096 | 0.225 / 0.595 | 0.36 |
| mean-shift (floor) | 12/20 | **0.022** | 0.609 / 0.700 | 0.46 |

![Closed-loop stability by trigger](docs/figures/fig15_closed_loop_stability.png)

**The finding is fragility, not a ranking.** *Every* trigger — two published detectors, two cheap statistics and ours — collapses the loop on a substantial fraction of seeds, and our monitor is not measurably more stable than the published alternatives (pooled 4/20 vs 33/80, p=0.119). Only the trivial mean-shift floor separates, so the signal matters at the bottom end but not between the sophisticated ones.

**Mechanism (verified, not hypothesised).** Badly placed triggers open training windows over attack-dense stretches, feeding the learner up to **0.46 attack fraction against a 0.20 base rate**; its prior skews toward ATTACK, the worst-case false-positive rate on benign traffic reaches **0.60–0.70**, the class-PHT then fires constantly and adaptation runs away. `train atk%` broadly tracks the collapse count across monitors. Running D3 with its **own** window and threshold does not rescue it — 9/20 collapses at the released defaults, 7/20 with a window sized for the data — and at those defaults it fires ~12 300 times against our 188, because with a 10-sample "new" window its 2-fold AUC crosses τ on ~5% of tests over a *stationary* stream (unit-tested).

**Metric caveat that stands:** final-window F1 masks mid-stream collapse (PCA seed 1: final F1 0.948, cumulative accuracy 0.444, FPR 0.657) — but it does not rescue the autoencoder. At 20 seeds PCA matches it on stability too, so [experiment 2](docs/experiments/02-monitor-ablation.md)'s original conclusion stands and the deployable recipe is the **cheap linear monitor**. An earlier 5-seed version of this table read 0/5 vs 4/5 and was retracted; a bimodal outcome needs n ≥ 20.

### 8.4 State-of-the-art context

Self-adapting ensembles (ARF, SRP) reach 0.77–0.81 sustained F1 learning on *every* sample (~2M supervised updates). Budget-matched to the reactive baseline's ~233 triggered adaptations they still score 0.72–0.73 (vs ~0.57–0.60 for the single tree): **the ensemble lead is driven mainly by model capacity, not continuous supervision**, which adds only ~0.05–0.08. Alternative error-based detectors (ADWIN, KSWIN) do not beat plain Page-Hinkley.

Figures: `fig1` local-F1 trajectory · `fig2` recovery comparison · `fig3` lead-time & cost/benefit · `fig4` resources · `fig5` sensitivity · `fig6` frequency control · `fig7` monitor frontier · `fig8` SOTA · `fig9` monitor in-pipeline · `fig10` budget-matched SOTA · `fig11` trigger × base-learner · `fig12` external validity · `fig13` timing vs frequency · `fig14b` synthetic clustering · `fig15` closed-loop stability.

**Full per-experiment analysis** (question · setup · numbers · figures · honest interpretation · caveats): [docs/experiments/](docs/experiments/).

## 9. Discussion

- **The transferable idea is the policy, not the signal.** Reconstruction-error drift detection already exists; what this work adds is using it as a *when-to-adapt* trigger, the controls that separate timing from frequency and capacity, and the finding that trigger quality must be measured in the loop.
- **Judge triggers by closed-loop stability — and expect fragility.** Detection metrics say nothing about what a trigger does to the loop. Measured in the loop, every trigger we tested collapses the learner on 20–60% of runs into a regime that is operationally fatal for an IDS (0.49–0.69 false-positive rate), caused by the trigger biasing the learner's training distribution. No paper in the unsupervised-drift literature we surveyed reports this. We could *not* show our monitor is more stable than the published alternatives.
- **Metric choice can hide failure.** Final-window F1 masked a mid-stream collapse that cumulative accuracy and FPR expose. Report a metric that can see the whole stream.
- **Bimodal outcomes need many replicates.** At n=5 the stability comparison gave a clean, significant and wrong ranking, which we published to this repository and then had to retract at n=20. Collapse-count comparisons here are held to n ≥ 20.
- **Base learner matters more than the augmenter.** Swapping the single tree for an ensemble at the same sparse budget gains ≈ +0.15 sustained F1 — far more than any augmentation strategy did. The practical recipe is ensemble + proactive triggered adaptation.
- **Scope and limits.** The timing claim holds on CICIDS2017/2018; on UNSW-NB15 the benefit is frequency. Anticipation recall at the operating point is modest (~0.23). The shared schedule is derived from B's dynamics, which slightly favours B-like strategies (bounded by the label-derived schedule). Prequential evaluation assumes immediate labels — unrealistic in deployment, though all phases share the assumption, so comparisons stay internally valid. All three datasets are CICFlowMeter-family.
- **Negative results are reported.** Generative augmentation under triggered adaptation (§4 note), and two pre-specified synthetic experiments that failed to establish a drift-structure mechanism. Both remain reproducible in this repository.

## 10. Repository structure

```
src/driftgan/
├── config.py             # YAML resolution (base ← phase ← dataset ← CLI), run dirs, hashes
├── logging.py            # JSON-line structured logging
├── data/stream.py        # streaming CSV reader + row preprocessing
├── models/
│   ├── hat.py            # HAT factory (identical hyperparams across HAT phases)
│   └── streaming.py      # model factory: HAT vs ARF/SRP ensembles
├── drift/schedule.py     # shared schedules; dynamic-PHT / detector / periodic / scheduled triggers
├── augment/              # base protocol + FeatureScaler, bootstrap_jitter, vanilla_gan, wgan_gp, cgan
├── early/
│   ├── reconstruction_monitor.py  # autoencoder monitor (AE)
│   ├── shift_monitors.py          # PCA / KS / MMD / mean-shift + D3 comparator
│   └── dual_pht.py                # dual PHT + lead-time pairing
├── metrics/              # rolling (O(1) F1), recovery (RT/AUD), cost
├── experiments/          # runner (unified engine), seeding, manifest, stats
└── viz/plots.py          # stream comparison, boxplots, RT CDF, Pareto
configs/                  # base.yaml + one YAML per phase + datasets/<name>.yaml profiles
scripts/                  # analyses: tier0, tier1, comparator, monitor_ablation, reviewer;
                          # figures: paper_figures, make_report; synthetics; scan_drift_window
docs/                     # paper/ (idea-central, related-work), experiments/*.md,
                          # architecture.md, methodology.md, gan_design.md, figures/
tests/                    # pytest: recovery, schedule/triggers, early monitors, models, stats, augment, e2e
run_experiment.py         # multi-seed CLI (also builds the schedule: --build-schedule)
Makefile                  # setup / test / lint / smoke / schedule / run-* / report
```

## 11. Future work

- **Recover a generalizable efficiency claim** — cap the proactive arm's fire count to the reactive one per dataset, testing whether it still wins with *no more* adaptations (on CICIDS2017 it already uses fewer; on 2018/UNSW it fires more).
- **Raise anticipation recall** (~0.23 today) by fusing complementary shift statistics — the stability result suggests the *combination* matters more than any single signal.
- **Make stability a design target**: rebalance or reweight triggered windows so a mis-timed trigger cannot skew the learner's class prior.
- **A second published comparator** (HDDDM) and a faithful `(w, ρ)` D3 with its native threshold, to separate "this signal" from "our adaptation of it".
- **Label-delay simulation** (labels arriving *k* samples late) to stress the prequential assumption.
- **A fourth dataset** (CIC-DDoS2019) and randomized attack-block ordering, to decouple results from each capture's specific sequence.

## 12. Citation

```bibtex
@misc{pacedrift2026,
  title  = {PACE: when to adapt --- a cheap unsupervised covariate-shift
            trigger for streaming intrusion detection},
  author = {Lara Guti\'errez, Antonio},
  year   = {2026},
  note   = {https://github.com/antoniol00/pace-drift}
}
```

## 13. License

No license file yet — all rights reserved until one is chosen. `[TODO: pick a license]`
