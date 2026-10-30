"""Label delay — do the timing and monitor contrasts change when labels are late?

Every real-data cell assumes labels arrive with the flow, which favours the
live error test: only it depends on labels for its timeliness. The ``*_d1k`` /
``*_d5k`` / ``*_d50k`` arms hold each label 1,000 / 5,000 / 50,000 flows (about
1 / 5 / 45 minutes of capture) before the error tests count it and the learner is fed it; the monitor
still sees each flow at once. For HAT and ARF on CIC-IDS2017 and CIC-IDS2018,
paired by seed over the ten core seeds:

  timing   early - freq       (blind clock, fire count matched per delay)
           early - freq_lab   (blind clock, labelled share matched per delay)
  monitor  early - pht        (the same live error test with no monitor)

Under delay PACE's firings cluster and restart windows, so the count-matched
clock, whose windows never overlap, learns from far more of the stream; the
label-matched clock is the fair timing control there.

Delay 0 is the existing immediate-label arms.

Run:  uv run python scripts/delay_analysis.py
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from tier1_analysis import (
    DSET,
    cell_seeds,
    ci_str,
    fire_sources,
    per_seed_labels,
    per_seed_local_f1,
    per_seed_n_drifts,
    run_dir,
)

OUT = Path("results/report/delay")
DELAYS = (("", "0"), ("_d1k", "1,000"), ("_d5k", "5,000"), ("_d50k", "50,000"))
LEARNERS = (("HAT", "C_early", "C_freq", "B_pht", "C_freq_lab"),
            ("ARF", "ARF_early", "ARF_freq", "ARF_pht", "ARF_freq_lab"))


def fpr_recall(dataset: str, phase: str, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed benign false-positive rate and attack recall from the traces."""
    fpr, rec = [], []
    for s in seeds:
        with gzip.open(run_dir(dataset, phase, s) / "trace.csv.gz", "rt") as fh:
            tr = pd.read_csv(fh, usecols=["y_true", "y_pred"])
        yt, yp = (tr["y_true"] == "ATTACK").to_numpy(), (tr["y_pred"] == "ATTACK").to_numpy()
        fpr.append((~yt & yp).sum() / max((~yt).sum(), 1))
        rec.append((yt & yp).sum() / max(yt.sum(), 1))
    return np.array(fpr), np.array(rec)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Label delay — timing and monitor contrasts", "",
             "Paired by seed (core seeds 0-9); local F1 with a 95% t interval. Fires are means per run, "
             "monitor / error test for the PACE arm.", "",
             "| Dataset | Learner | Delay | n | F1 PACE / clock / label clock / error test | Timing (PACE - clock) | "
             "Timing (PACE - label clock) | Monitor (PACE - error test) | Fires PACE (mon / err) / clock / err test | "
             "Labels PACE / clock / label clock / err test | FPR med PACE / err test | Recall med PACE / err test |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in ("cicids2017", "cicids2018"):
        for learner, early, freq, pht, freq_lab in LEARNERS:
            for suffix, delay in DELAYS:
                arms = (early + suffix, freq + suffix, pht + suffix, freq_lab + suffix)
                seeds = cell_seeds(ds, *arms)
                if len(seeds) < 10:
                    lines.append(f"| {DSET[ds]} | {learner} | {delay} | {len(seeds)} | (incomplete) ||||||||")
                    continue
                f1 = [per_seed_local_f1(ds, a, seeds) for a in arms]
                assert all(v is not None for v in f1)
                fe, ff, fp, fl = (np.asarray(v) for v in f1)
                fires = [per_seed_n_drifts(ds, a, seeds) for a in arms]
                src = fire_sources(ds, arms[0], seeds)
                mon = src.get("recon", 0) + src.get("recon+class", 0)
                labels = [per_seed_labels(ds, a, seeds).mean() for a in arms]
                (fpr_e, rec_e), (fpr_p, rec_p) = fpr_recall(ds, arms[0], seeds), fpr_recall(ds, arms[2], seeds)
                lines.append(
                    f"| {DSET[ds]} | {learner} | {delay} | {len(seeds)} | {fe.mean():.3f} / {ff.mean():.3f} / "
                    f"{fl.mean():.3f} / {fp.mean():.3f} | {ci_str(fe - ff)} | {ci_str(fe - fl)} | {ci_str(fe - fp)} | "
                    f"{np.mean(fires[0]):.0f} ({mon:.0f} / {np.mean(fires[0]) - mon:.0f}) / {np.mean(fires[1]):.0f} / "
                    f"{np.mean(fires[2]):.0f} | {labels[0]:.1%} / {labels[1]:.1%} / {labels[3]:.1%} / {labels[2]:.1%} | "
                    f"{np.median(fpr_e):.3f} / {np.median(fpr_p):.3f} | {np.median(rec_e):.3f} / {np.median(rec_p):.3f} |"
                )
    text = "\n".join(lines) + "\n"
    (OUT / "analysis.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
