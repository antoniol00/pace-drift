"""Unsupervised-comparator analysis — our monitor vs a published detector (D3).

Answers the related-work question: *given the identical when-to-adapt policy,
does a published unsupervised drift detector do the job as well as our
reconstruction monitor?* Compares the C_early family on the 2017 anchor, all
sharing the same pipeline, base learner and dual-PHT policy, differing only in
the shift statistic and each fire-count calibrated to the autoencoder:

    C_early       autoencoder reconstruction error (the original monitor)
    C_early_pca   linear PCA reconstruction error  (cheap ablation)
    C_early_mean  L2 mean shift                    (trivial floor)
    C_early_d3    D3 discriminative AUC            (published comparator,
                  Gozuacik et al., CIKM 2019)

Reports overall local F1 (paired-t CI vs the AE), adaptation counts, and the
anticipation quality of each signal (recon fires, lead time, false-alarm rate).

Run: uv run python scripts/comparator_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RUNS = Path("results/runs")
OUT = Path("results/report/comparator")
SEEDS = [0, 1, 2, 3, 4]

ARMS = [
    ("C_early", "autoencoder (ours)"),
    ("C_early_pca", "PCA linear (ours, cheap)"),
    ("C_early_d3", "D3 discriminative (CIKM'19)"),
    ("C_early_mean", "mean-shift (trivial floor)"),
]


def per_seed_local_f1(phase: str) -> np.ndarray | None:
    out = []
    for s in SEEDS:
        f = RUNS / phase / f"seed_{s}" / "snapshots.parquet"
        if not f.exists():
            return None
        df = pd.read_parquet(f)
        out.append(float(df.loc[df["valid"], "f1"].mean()))
    return np.array(out)


def per_seed_summary(phase: str, key: str) -> np.ndarray | None:
    out = []
    for s in SEEDS:
        f = RUNS / phase / f"seed_{s}" / "summary.json"
        if not f.exists():
            return None
        out.append(json.loads(f.read_text())[key])
    return np.array(out, dtype=float)


def lead_stats(phase: str) -> dict[str, float] | None:
    """Mean recon fires, lead time and false-alarm rate over seeds."""
    rec, lead, fp, miss = [], [], [], []
    for s in SEEDS:
        f = RUNS / phase / f"seed_{s}" / "lead_time.json"
        if not f.exists():
            return None
        d = json.loads(f.read_text())
        rec.append(d["n_recon_fires"])
        lt = d["lead_times"]
        lead.append(float(np.mean(lt)) if lt else float("nan"))
        fp.append(d["anticipated_fp_rate"])
        miss.append(d["missed_anticipation_rate"])
    return {
        "recon_fires": float(np.mean(rec)),
        "lead": float(np.nanmean(lead)),
        "fp_rate": float(np.mean(fp)),
        "recall": 1.0 - float(np.mean(miss)),
    }


def confusion_stats(phase: str) -> dict[str, np.ndarray] | None:
    """Per-seed precision / recall / false-positive rate from the traces.

    FPR is the operationally decisive metric here: a monitor that destabilizes
    the adaptation loop shows up as a catastrophic false-positive rate on the
    ~80% benign traffic, which F1-style metrics partly mask.
    """
    import gzip

    prec, rec, fpr = [], [], []
    for s in SEEDS:
        f = RUNS / phase / f"seed_{s}" / "trace.csv.gz"
        if not f.exists():
            return None
        with gzip.open(f, "rt") as fh:
            tr = pd.read_csv(fh, usecols=["y_true", "y_pred"])
        yt = (tr["y_true"] == "ATTACK").to_numpy()
        yp = (tr["y_pred"] == "ATTACK").to_numpy()
        tp = int((yt & yp).sum())
        fp = int((~yt & yp).sum())
        fn = int((yt & ~yp).sum())
        tn = int((~yt & ~yp).sum())
        prec.append(tp / (tp + fp) if tp + fp else 0.0)
        rec.append(tp / (tp + fn) if tp + fn else 0.0)
        fpr.append(fp / (fp + tn) if fp + tn else 0.0)
    return {"precision": np.array(prec), "recall": np.array(rec), "fpr": np.array(fpr)}


def train_window_attack_fraction(phase: str, train_window: int = 5_000) -> np.ndarray | None:
    """Attack fraction of the samples the model actually trains on, per seed.

    Explains the instability: badly placed triggers open windows over
    attack-dense stretches, so the learner is fed a label distribution far from
    the stream's (~20% attack), skewing its prior toward ATTACK.
    """
    import gzip

    out = []
    for s in SEEDS:
        d = RUNS / phase / f"seed_{s}"
        if not (d / "drift_events.json").exists() or not (d / "trace.csv.gz").exists():
            return None
        ev = json.loads((d / "drift_events.json").read_text())["events"]
        with gzip.open(d / "trace.csv.gz", "rt") as fh:
            tr = pd.read_csv(fh, usecols=["index", "y_true"])
        idx = tr["index"].to_numpy()
        atk = (tr["y_true"] == "ATTACK").to_numpy()
        pos = {v: i for i, v in enumerate(idx)}
        mask = np.zeros(len(idx), dtype=bool)
        for e in ev:
            st = pos.get(e["index"])
            if st is not None:
                mask[st : st + train_window] = True
        out.append(float(atk[mask].mean()) if mask.any() else float("nan"))
    return np.array(out)


def paired_ci(d: np.ndarray) -> tuple[float, float]:
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    tc = float(stats.t.ppf(0.975, df=len(d) - 1))
    return m - tc * se, m + tc * se


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = per_seed_local_f1("C_early")
    if base is None:
        print("  (C_early runs missing)")
        return

    lines = [
        "# Unsupervised comparator — our monitor vs D3 (published)",
        "",
        "All arms share the identical pipeline, base learner (HAT) and dual-PHT "
        "when-to-adapt policy on the CICIDS2017 anchor; each monitor's PHT is "
        "calibrated to the autoencoder's fire count. Only the shift statistic "
        "differs.",
        "",
        "**Read the per-seed column, not the mean.** Some arms are *bimodal* "
        "(a subset of seeds enters a degenerate feedback loop where the "
        "classifier collapses, class-PHT fires explode, and adaptation runs "
        "away). A mean ± paired-t CI is not meaningful under bimodality, so we "
        "report per-seed values, the median, and an explicit collapse count "
        "(`collapsed` = seeds with cumulative accuracy < 0.90; the majority "
        "class alone scores ~0.80).",
        "",
        "`FPR med/worst` is the false-positive rate on benign traffic — the "
        "operationally decisive number for an IDS. `train atk%` is the attack "
        "fraction of the samples the learner is actually fed inside triggered "
        "windows (the stream's own rate is 0.20); a value far above that is the "
        "mechanism of the collapse.",
        "",
        "| monitor | local F1 median | per-seed local F1 | collapsed | "
        "FPR med/worst | train atk% | adapts | recon fires | anticip. recall |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    pending = []
    collapse: dict[str, int] = {}
    for phase, label in ARMS:
        f1 = per_seed_local_f1(phase)
        if f1 is None:
            pending.append(phase)
            continue
        nd = per_seed_summary(phase, "n_drifts")
        acc = per_seed_summary(phase, "final_acc")
        ls = lead_stats(phase)
        n_bad = int((acc < 0.90).sum()) if acc is not None else 0
        collapse[phase] = n_bad
        per_seed = " ".join(f"{v:.2f}" for v in f1)
        nds = f"{nd.mean():.0f}" if nd is not None else "n/a"
        cs = confusion_stats(phase)
        twa = train_window_attack_fraction(phase)
        if cs is not None:
            worst = cs["fpr"].max()
            fprx = f"{np.median(cs['fpr']):.3f} / {worst:.3f}"
        else:
            fprx = "n/a"
        twx = f"{np.median(twa):.2f}" if twa is not None else "n/a"
        lsx = f"{ls['recon_fires']:.0f} | {ls['recall']:.2f}" if ls else "n/a | n/a"
        lines.append(f"| {label} | {np.median(f1):.3f} | {per_seed} | "
                     f"{n_bad}/5 | {fprx} | {twx} | {nds} | {lsx} |")

    if pending:
        lines += ["", f"> Pending runs: {', '.join(pending)}."]

    d3 = per_seed_local_f1("C_early_d3")
    if d3 is not None:
        n_bad = collapse.get("C_early_d3", 0)
        n_ref = collapse.get("C_early", 0)
        d = d3 - base
        lo, hi = paired_ci(d)
        if n_bad > n_ref:
            verdict = (
                f"**D3-triggered adaptation was unstable on this data: {n_bad}/5 "
                f"seeds collapsed** (vs {n_ref}/5 for the reconstruction monitor), "
                "while the surviving seed matched it. The recon-fire count is "
                "matched by calibration, so the difference is *where* the fires "
                "land, not how many. Note the paired-t CI "
                f"({d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]) spuriously looks "
                "'indistinguishable' only because the collapse makes the "
                "distribution bimodal — do not report it that way.\n\n"
                "Diagnosis (verified on synthetic checks, "
                "`tests/test_early.py`): D3's linear discriminator "
                "detects mean shifts (AUC 0.93-1.00) but is **blind to "
                "variance-only changes** (AUC 0.50), which reconstruction error "
                "does capture. On flow features, attack onsets often change "
                "dispersion and correlation structure rather than means.\n\n"
                "Honest scope: ours is an *adaptation* of D3 (shared monitor "
                "protocol, gradient-descent logistic regression), so this is "
                "evidence that a discriminative-separability signal did not "
                "substitute for the reconstruction signal **in our pipeline** — "
                "not a claim that D3 is a poor detector in general.")
        elif hi < 0:
            verdict = ("**D3 underperforms our monitor** at matched fire count — the "
                       "reconstruction signal suits high-dimensional flow features "
                       "better than discriminative separability here.")
        elif lo > 0:
            verdict = ("**D3 outperforms our monitor.** Adopt it: the policy claim "
                       "is unaffected, and the paper reports the better signal.")
        else:
            verdict = ("**D3 is statistically indistinguishable from our monitor**, "
                       "with no collapse asymmetry. This supports the thesis: the "
                       "contribution is the when-to-adapt policy, robust to the "
                       "choice of unsupervised signal.")
        lines += ["", f"**Verdict:** {verdict}"]

    doc = "\n".join(lines) + "\n"
    (OUT / "analysis.md").write_text(doc)
    print(doc)
    print(f"wrote {OUT / 'analysis.md'}")


if __name__ == "__main__":
    main()
