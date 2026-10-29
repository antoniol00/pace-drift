# 09 — Third review round: metric, statistics and scope checks

Every number here is computed from persisted runs by `scripts/review_checks.py`
(sections A–H, written to `results/report/review/`) and
`scripts/comparator_analysis.py` (section C of its report), except where a new
run is named. Real-data cells use the ten core seeds unless stated.

## What changed in the evidence

| Check | Result | Consequence for the claim |
|---|---|---|
| Hierarchical bootstrap (datasets, then seeds) of the nine-cell aggregates | total +0.043 [+0.022, +0.075]; **timing +0.042 [−0.003, +0.084]**; monitor +0.021 [−0.003, +0.052] | The timing aggregate is fragile with three captures; the per-cell intervals (7/9, all six on 2017/2018) carry the claim. Never quote a nine-cell sign test. |
| HAT anchor at 20 seeds (B, C_freq, B_pht re-run at seeds 10–19) | PACE − schedule **+0.000 [−0.061, +0.061]** (9/20 positive); timing **+0.076 [+0.002, +0.149]** (15/20, Wilcoxon p 0.033); monitor +0.016 [−0.046, +0.077] | The anchor cell is evidence about timing, not about PACE vs the schedule. |
| Strict local F1 (benign-only windows with a false positive scored 0) | 36% of windows are benign-only on 2017/2018 (0% on UNSW); total +0.048, timing +0.031 [+0.005, +0.057], **monitor +0.056 [+0.034, +0.078], 9/9** | The monitor contrast turns positive, but on ensembles it is a few FPs per 5,000 benign flows magnified by the metric's cliff; whole-stream benign FPR does not resolve it (−0.035 [−0.094, +0.024]). Real only on HAT (live PHT FPR 29% vs 7% on 2017). Still "not resolved". |
| Lead time against the live control's error firings (no monitor window can pre-empt them) | Same recall / false alarms / lead as in-loop in every cell | The in-loop false-alarm rate is not inflated by early windows suppressing error firings. |
| Per-event AUD stratification at θ = 0.90 / 0.95 / 0.98 | 2017: anticipated recover better at all θ (ARF fails at 0.98); UNSW ensembles only at 0.90; **2018 null at every θ**, zero-AUD share falls 25–40% → 16–21% | 2018 null is not a θ ceiling. The 2018 timing win is the error test placing windows on attack onsets in the 0.55–1.05M block; the clock opens as many benign-only windows (36%) as PACE (34–40%). |
| D3 solver | 150-step GD left median gradient norm 0.44 and flipped AUC across 0.7 on 28/300 checks. Now L-BFGS; all D3 arms re-run (old runs in `results/runs_backup_d3_gd/`) | Matched D3 unchanged: 10/20 collapsed, median local F1 0.500 (was 0.479). Native sized 10/20 (was 7/20), ~1,340 fires. |
| Stability controls at 20 seeds | live error test (HAT) **8/20**, FPR worst 0.718, collapsed train atk 0.54; replayed schedule **2/20** despite 0.57 attack in every window; blind clock 8/20 but all between 0.80–0.90 cum. acc., FPR ≤ 0.15; **ARF and SRP 0/20 each, with and without the monitor (0/80)** | Collapse is a property of gating the single tree with any stream-responsive trigger, not of unsupervised triggering; it needs feedback (fixed schedule ≈ immune); ensembles immune. |
| Fisher power at 20 seeds | 20% vs 50%: 0.41; 20% vs 70%: 0.85. Published detectors only: 4/20 vs 15/40, p = 0.24 | "Not distinguishable", never "equivalent". Do not pool PCA / mean-shift with the published detectors. |
| 169 vs 188 adaptations (AE, 10 vs 20 seeds) | Monitor firings matched (42–60/run); error firings ~107 healthy vs ~245 collapsed for every arm; seeds 10–19 had 3 AE collapses vs 1 | Totals diverge with the collapse rate; only monitor firings are matched. |
| PCA 10-seed mean 0.577 vs 20-seed median 0.505 | PCA seeds 0–9 mean 0.577 (4 collapsed), 10–19 mean 0.492; local F1 only partly sees collapse | Report medians/per-seed for HAT; PCA is not distinguishable from the AE on stability (6 vs 4). |
| PCA cost (clean benchmark, seed 0, idle machine, `results/bench/`) | monitor time 34.7 s (AE 26.5 s); max latency **146 ms** (AE 1,376 ms; reactive 130 ms); peak RSS **782 MB** (AE 1,310; reactive 774) | PCA removes the tail spike and the PyTorch memory, **not** total compute. |
| Synthetic stream, 10 seeds, PCA and AE | timing PCA +0.254 [+0.106, +0.401], AE +0.288 [+0.157, +0.419]; **PACE − live reactive +0.019 / +0.015** | Mechanism reproduces with the primary monitor; the monitor adds ~+0.02 over the live error test there too. See [08](08-synthetic-timing.md). |

## Data provenance and windows (section H)

- CIC-IDS2017: CIC release, original CICFlowMeter (Flow ID / IP columns), labels
  grouped into 8 attack categories; 80 features kept (incl. source port — kept
  inconsistently with UNSW, where it is dropped). Window 2.00M from row 540K:
  attack 20.0% (capture 19.7%), 30.4 benign/attack transitions per 1,000 rows
  (21.9). Timestamps lack AM/PM.
- CIC-IDS2018: CSE-CIC-IDS2018 release, original CICFlowMeter; 78 features.
  Window 3.00M from 11M: attack **40.4% (capture 14.9%)**, 182.4 transitions
  per 1,000 (56.4). The file is **not chronological within a day** (10.4% of
  consecutive rows step back > 1 h): stream order is file order.
- UNSW-NB15: the CIC-UNSW-NB15 re-extraction (Mohammadian, Habibi Lashkari,
  Ghorbani, PST 2024; `CICFlowMeter_out.csv`) — our per-family attack counts
  match it exactly; 78 features. Window 1.70M from 1.75M: attack 4.3% (2.6%),
  82.7 transitions per 1,000 (49.0).
- A 7,000-flow lead spans a median of ~9 min of capture time on 2017 and 7 min
  on UNSW: enough to queue traffic for triage, far too little for labels.
- Windows were chosen by inspection of `scan_drift_window.py` output, committed
  (9533c33) before any arm ran on 2018/UNSW.

## Reproducibility note

Re-running B at seed 0 reproduced its 233 firings and cumulative accuracy bit for
bit. Monitor arms reproduce only in distribution (BLAS threads / GPU change
scores near the threshold and the loop amplifies one changed firing): C_early
seed 0 re-ran with 144 firings vs the stored 171, C_early_pca with 126 vs 172.
The early PCA seeds 0–4 also logged ~1,900 s of monitor time from BLAS thread
oversubscription; always set `OMP_NUM_THREADS=1` for parallel batches.

## Residual consistency fixes (full-review pass)

- Capacity contrast restated on the same 233 windows: ensembles +0.15–0.17 over
  HAT (was "+0.13–0.16", which mixed in PACE's 169-firing arm); continuous
  supervision adds +0.05–0.07 (was "+0.05–0.08").
- Positioning table: INSOMNIA and SSF marked "per sample", Abdel Wahab
  "continuous" — they do not gate learning to windows, as the text already said.
- Core table (CIC-IDS2017, HAT, seeds 0–9) gains collapse counts: replayed
  schedule 1/10, live error test 4/10, PACE 1/10.
