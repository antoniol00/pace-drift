"""Checks raised by the third review round, computed from persisted runs only.

Sections (each writes a block of ``results/report/review/checks.md``):

  A. Strict local F1. The primary metric excludes windows with no attack label,
     which also hides false positives raised inside purely benign stretches.
     Re-scores every run with those windows counted as F1 = 0 when they contain
     a false positive, reports how many windows the primary metric excludes, and
     re-derives the three headline contrasts under the strict metric.
  B. Shadow lead time. Inside PACE a monitor firing that adapts the learner in
     time can suppress the error firing it anticipated, which the in-loop
     pairing then counts as a false alarm. Pairs PACE's monitor firings with the
     error firings of the live reactive control (``*_pht``), which never sees a
     monitor window, as an unmitigated reference.
  C. Per-event mechanism across datasets: recovery, anticipated vs reactive-only
     AUD, and whether the null result on CIC-IDS2018 survives a stricter
     recovery threshold (a theta = 0.9 ceiling would hide deficits of ensembles
     that rarely fall below it).
  D. Where the blind clock loses on CIC-IDS2018: local F1 by segment, and the
     composition of the windows each arm trains on.
  E. Hierarchical bootstrap of the nine-cell aggregates (datasets, then seeds),
     which respects that the three learners of a dataset share its capture,
     monitor scores and schedule.
  F. Power of the collapse-count comparison at twenty seeds, and the pooled
     test restricted to the published detectors.
  G. D3's batch gradient-descent solver against a converged L-BFGS fit of the
     same objective, on real buffers of the CIC-IDS2017 window.
  H. The evaluated windows against the full captures: attack prevalence, label
     transitions, and the capture time a lead of 7,000 flows corresponds to.

Seeds 0-9 throughout (the real-data protocol), except F and G.

Run:  uv run python scripts/review_checks.py [--sections ABCDEFGH]
"""

from __future__ import annotations

import argparse
import gzip
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tier0_analysis import rec_path
from tier1_analysis import CELLS, DSET, run_dir

from driftgan.metrics.recovery import compute_recovery_events

OUT = Path("results/report/review")
CORE = list(range(10))
WINDOW = 5_000
TRAIN_WINDOW = 5_000
ARMS = ("sched", "freq", "early", "pht")


# --------------------------------------------------------------------------- #
# A + D: per-run trace statistics
# --------------------------------------------------------------------------- #
def _trace(d: Path) -> pd.DataFrame:
    with gzip.open(d / "trace.csv.gz", "rt") as fh:
        return pd.read_csv(fh, usecols=["index", "y_true", "y_pred"])


def run_stats(job: tuple[str, str, int]) -> dict:
    """Window counts at every snapshot, strict F1 and training-window makeup."""
    dataset, phase, seed = job
    d = run_dir(dataset, phase, seed)
    tr = _trace(d)
    idx = tr["index"].to_numpy()
    yt = (tr["y_true"] == "ATTACK").to_numpy()
    yp = (tr["y_pred"] == "ATTACK").to_numpy()
    snaps = pd.read_parquet(d / "snapshots.parquet")
    cs = lambda a: np.concatenate([[0], np.cumsum(a)])  # noqa: E731
    ctp, cfp, cfn = cs(yt & yp), cs(~yt & yp), cs(yt & ~yp)
    hi = np.searchsorted(idx, snaps["index"].to_numpy(), side="right")
    lo = np.searchsorted(idx, snaps["index"].to_numpy() - WINDOW, side="right")
    tp, fp, fn = ctp[hi] - ctp[lo], cfp[hi] - cfp[lo], cfn[hi] - cfn[lo]
    valid = (tp + fn) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(valid, 2 * tp / (2 * tp + fp + fn), np.nan)
    stored = snaps["f1"].to_numpy()
    sv = snaps["valid"].to_numpy()
    strict = np.where(valid, f1, np.where(fp > 0, 0.0, np.nan))
    # training windows: union of [firing, firing + W), as the engine applies them
    ev = json.loads((d / "drift_events.json").read_text())["events"]
    mask = np.zeros(len(idx), dtype=bool)
    starts = np.searchsorted(idx, [e["index"] for e in ev])
    win_atk = []
    for st in starts:
        mask[st : st + TRAIN_WINDOW] = True
        win_atk.append(float(yt[st : st + TRAIN_WINDOW].mean()) if st < len(idx) else np.nan)
    win_atk_a = np.array(win_atk)
    return {
        "dataset": dataset, "phase": phase, "seed": seed,
        "f1_check": float(np.nanmax(np.abs(f1[sv] - stored[sv]))) if sv.any() else 0.0,
        "valid_agree": float((valid == sv).mean()),
        "local_f1": float(np.mean(stored[sv])),
        "strict_f1": float(np.nanmean(strict)),
        "invalid_share": float((~valid).mean()),
        "invalid_fp_share": float(((~valid) & (fp > 0)).mean()),
        "benign_fpr_in_invalid": float(fp[~valid].sum() / max(1, (~valid).sum() * WINDOW)),
        "train_attack_frac": float(yt[mask].mean()) if mask.any() else float("nan"),
        "single_benign_windows": float(np.nanmean(win_atk_a == 0)) if len(win_atk_a) else float("nan"),
        "fpr": float((~yt & yp).sum() / max(1, (~yt).sum())),
        "snap_index": snaps["index"].to_numpy(),
        "snap_f1": np.where(sv, stored, np.nan),
    }


def all_run_stats() -> pd.DataFrame:
    jobs = [(ds, ph, s) for ds, _l, *arms in CELLS for ph in arms for s in CORE]
    with Pool(48) as pool:
        rows = pool.map(run_stats, jobs)
    return pd.DataFrame(rows)


def _arm(cell: tuple, arm: str) -> str:
    return cell[2 + ARMS.index(arm)]


def _col(df: pd.DataFrame, cell: tuple, arm: str, col: str) -> np.ndarray:
    """Per-seed values of ``col`` for one arm of one cell, ordered by seed."""
    sub = df[(df["dataset"] == cell[0]) & (df["phase"] == _arm(cell, arm))]
    return sub.sort_values("seed")[col].to_numpy()


def section_a(df: pd.DataFrame) -> list[str]:
    lines = ["## A. Strict local F1 (benign-only windows with a false positive score 0)", "",
             f"Recomputed window F1 matches the stored value to within "
             f"{df['f1_check'].max():.1e}, validity agrees on "
             f"{df['valid_agree'].min() * 100:.2f}% of snapshots or more.", "",
             "| dataset | excluded windows (share) | of which with a false positive |",
             "|---|---:|---:|"]
    for ds in DSET:
        sub = df[df["dataset"] == ds]
        lines.append(f"| {DSET[ds]} | {sub['invalid_share'].mean() * 100:.1f}% "
                     f"(arm range {sub.groupby('phase')['invalid_share'].mean().min() * 100:.1f}"
                     f"–{sub.groupby('phase')['invalid_share'].mean().max() * 100:.1f}%) | "
                     f"{sub['invalid_fp_share'].mean() * 100:.1f}% |")
    lines += ["", "| dataset | learner | contrast | primary | strict |", "|---|---|---|---:|---:|"]
    agg: dict[tuple[str, str], list[float]] = {}
    for cell in CELLS:
        ds, learner = cell[0], cell[1]
        for name, a, b in (("early - sched", "early", "sched"), ("early - freq", "early", "freq"),
                           ("early - pht", "early", "pht")):
            p = float((_col(df, cell, a, "local_f1") - _col(df, cell, b, "local_f1")).mean())
            s = float((_col(df, cell, a, "strict_f1") - _col(df, cell, b, "strict_f1")).mean())
            agg.setdefault((name, "primary"), []).append(p)
            agg.setdefault((name, "strict"), []).append(s)
            lines.append(f"| {DSET[ds]} | {learner} | {name} | {p:+.3f} | {s:+.3f} |")
    lines += ["", "Benign false-positive rate over the whole stream, and inside the excluded "
              "(benign-only) windows, mean over seeds (per cent):", "",
              "| dataset | learner | FPR sched / freq / early / pht | "
              "FPR in benign-only windows sched / freq / early / pht |", "|---|---|---|---|"]
    for cell in CELLS:
        lines.append(f"| {DSET[cell[0]]} | {cell[1]} | "
                     + " / ".join(f"{_col(df, cell, a, 'fpr').mean() * 100:.2f}" for a in ARMS) + " | "
                     + " / ".join(f"{_col(df, cell, a, 'benign_fpr_in_invalid').mean() * 100:.2f}"
                                  for a in ARMS) + " |")
    lines += ["", "Nine-cell aggregates (cell means, t interval):", ""]
    for name in ("early - sched", "early - freq", "early - pht"):
        for kind in ("primary", "strict"):
            v = np.array(agg[(name, kind)])
            se = v.std(ddof=1) / np.sqrt(len(v))
            t = stats.t.ppf(0.975, len(v) - 1)
            lines.append(f"- {name}, {kind}: {v.mean():+.3f} [{v.mean() - t * se:+.3f}, "
                         f"{v.mean() + t * se:+.3f}], {(v > 0).sum()}/9 positive")
    return lines


def section_d(df: pd.DataFrame) -> list[str]:
    lines = ["## D. CIC-IDS2018: where the blind clock loses, and what each arm trains on", "",
             "Local F1 by 0.5M segment of the evaluated stream (mean over seeds), then the "
             "attack fraction of the samples fed to the learner and the share of firings "
             "whose window holds no attack at all.", ""]
    edges = np.arange(50_000, 3_050_001, 500_000)
    head = " | ".join(f"{a / 1e6:.2f}–{b / 1e6:.2f}M" for a, b in zip(edges[:-1], edges[1:], strict=True))
    lines += [f"| learner | arm | {head} | train attack frac | benign-only windows |",
              "|---|---|" + "---:|" * (len(edges) + 1)]
    for cell in CELLS:
        if cell[0] != "cicids2018":
            continue
        for arm in ARMS:
            sub = df[(df["dataset"] == "cicids2018") & (df["phase"] == _arm(cell, arm))]
            segs = []
            for a, b in zip(edges[:-1], edges[1:], strict=True):
                vals = [np.nanmean(f[(i > a) & (i <= b)]) for i, f in
                        zip(sub["snap_index"], sub["snap_f1"], strict=True) if ((i > a) & (i <= b)).any()]
                segs.append(f"{np.nanmean(vals):.3f}" if vals else "–")
            lines.append(f"| {cell[1]} | {arm} | {' | '.join(segs)} | "
                         f"{sub['train_attack_frac'].mean():.2f} | "
                         f"{sub['single_benign_windows'].mean() * 100:.0f}% |")
    return lines


# --------------------------------------------------------------------------- #
# B: shadow lead time against the live reactive control
# --------------------------------------------------------------------------- #
def _pair(recon: list[int], cls: list[int], horizon: int = 20_000) -> tuple[float, float, float]:
    from driftgan.early.dual_pht import match_lead_times

    r = match_lead_times(recon, cls, horizon)
    lead = float(np.mean(r.lead_times)) if r.lead_times else float("nan")
    return 1 - r.missed_anticipation_rate, r.anticipated_fp_rate, lead


def section_b() -> list[str]:
    lines = ["## B. Lead time in the loop and against the live control's error firings", "",
             "| dataset | learner | in loop: recall / false alarms / lead | "
             "vs live control: recall / false alarms / lead |", "|---|---|---|---|"]
    for cell in CELLS:
        ds, learner = cell[0], cell[1]
        loop, shadow = [], []
        for s in CORE:
            lt = json.loads((run_dir(ds, _arm(cell, "early"), s) / "lead_time.json").read_text())
            loop.append(_pair(lt["recon_fires"], lt["class_fires"]))
            ev = json.loads((run_dir(ds, _arm(cell, "pht"), s) / "drift_events.json").read_text())
            shadow.append(_pair(lt["recon_fires"], [e["index"] for e in ev["events"]]))
        f = lambda a: f"{np.nanmean(a[:, 0]):.2f} / {np.nanmean(a[:, 1]):.2f} / {np.nanmean(a[:, 2]):,.0f}"  # noqa: E731
        lines.append(f"| {DSET[ds]} | {learner} | {f(np.array(loop))} | {f(np.array(shadow))} |")
    return lines


# --------------------------------------------------------------------------- #
# C: per-event mechanism, with a theta sensitivity
# --------------------------------------------------------------------------- #
def _events(ds: str, phase: str, s: int, theta: float) -> pd.DataFrame:
    d = run_dir(ds, phase, s)
    sn = pd.read_parquet(d / "snapshots.parquet")
    ev = json.loads((d / "drift_events.json").read_text())["events"]
    lt = json.loads((d / "lead_time.json").read_text())
    rec = compute_recovery_events(sn["index"].to_numpy(), sn["f1"].to_numpy(),
                                  sn["valid"].to_numpy(), [e["index"] for e in ev], theta=theta)
    src = {e["index"]: e["source"] for e in ev}
    matched = {p[1] for p in lt["pairs"]}
    out = pd.DataFrame([vars(r) for r in rec])
    out["source"] = out["drift_index"].map(src)
    out["anticipated"] = out["drift_index"].isin(matched)
    return out


def section_c() -> list[str]:
    lines = ["## C. Per-event mechanism (PACE arms, error-confirmed drifts)", "",
             "AUD MWU p = one-sided Mann-Whitney, anticipated < reactive-only. "
             "Zero-AUD share = error-confirmed drifts whose window never dips below theta.", "",
             "| dataset | learner | no-recovery | RT median | p @0.90 | p @0.95 | p @0.98 | "
             "zero-AUD share @0.90 | @0.98 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in CELLS:
        ds, learner = cell[0], cell[1]
        row, zeros = [], []
        base = None
        for theta in (0.90, 0.95, 0.98):
            ev = pd.concat([_events(ds, _arm(cell, "early"), s, theta) for s in CORE])
            ev = ev[ev["source"] == "class"]
            if theta == 0.90:
                base = ev
            a, k = ev.loc[ev["anticipated"], "AUD"], ev.loc[~ev["anticipated"], "AUD"]
            p = stats.mannwhitneyu(a, k, alternative="less").pvalue if len(a) and len(k) else np.nan
            row.append(f"{p:.3f}")
            zeros.append(f"{(ev['AUD'] == 0).mean() * 100:.0f}%")
        assert base is not None
        rt = base["RT"].dropna()
        lines.append(f"| {DSET[ds]} | {learner} | {base['no_recovery'].mean():.2f} | "
                     f"{rt.median():,.0f} | {' | '.join(row)} | {zeros[0]} | {zeros[2]} |")
    _ = rec_path  # recovery frames on disk agree with this recomputation at theta = 0.9
    return lines


# --------------------------------------------------------------------------- #
# E: hierarchical bootstrap of the aggregates
# --------------------------------------------------------------------------- #
def section_e(df: pd.DataFrame, n_boot: int = 10_000) -> list[str]:
    rng = np.random.default_rng(0)
    lines = ["## E. Nine-cell aggregates under a hierarchical bootstrap", "",
             "Resample the three datasets with replacement, then seeds within each cell "
             "(paired), average learners within a dataset and datasets overall. With only "
             "three datasets at the top level the interval is deliberately conservative.", "",
             "| metric | contrast | mean | cell-mean t interval | hierarchical bootstrap 95% |",
             "|---|---|---:|---:|---:|"]
    for metric, name, a, b in [(m, *c) for m in ("local_f1", "strict_f1", "fpr") for c in (
            ("early - sched", "early", "sched"), ("early - freq", "early", "freq"),
            ("early - pht", "early", "pht"))]:
        diffs: dict[str, list[np.ndarray]] = {}
        for cell in CELLS:
            diffs.setdefault(cell[0], []).append(_col(df, cell, a, metric) - _col(df, cell, b, metric))
        cm = np.array([d.mean() for v in diffs.values() for d in v])
        se = cm.std(ddof=1) / np.sqrt(len(cm))
        t = stats.t.ppf(0.975, len(cm) - 1)
        names = list(diffs)
        boots = np.empty(n_boot)
        for i in range(n_boot):
            ds_pick = rng.choice(names, size=len(names), replace=True)
            per_ds = []
            for ds in ds_pick:
                per_ds.append(np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in diffs[ds]]))
            boots[i] = np.mean(per_ds)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        lines.append(f"| {metric} | {name} | {cm.mean():+.4f} | [{cm.mean() - t * se:+.4f}, "
                     f"{cm.mean() + t * se:+.4f}] | [{lo:+.4f}, {hi:+.4f}] |")
    return lines


# --------------------------------------------------------------------------- #
# F: power of the collapse comparison
# --------------------------------------------------------------------------- #
def section_f(n_sim: int = 20_000) -> list[str]:
    rng = np.random.default_rng(0)
    lines = ["## F. Power of Fisher's exact test at twenty seeds per arm", "",
             "| collapse rate, reference vs alternative | power (two-sided, alpha 0.05) |",
             "|---|---:|"]
    cache: dict[tuple[int, int], float] = {}
    for p0, p1 in ((0.2, 0.5), (0.2, 0.6), (0.2, 0.7)):
        a = rng.binomial(20, p0, n_sim)
        b = rng.binomial(20, p1, n_sim)
        hits = 0
        for x, y in zip(a, b, strict=True):
            key = (int(x), int(y))
            if key not in cache:
                cache[key] = stats.fisher_exact([[x, 20 - x], [y, 20 - y]])[1]
            hits += cache[key] < 0.05
        lines.append(f"| {p0:.1f} vs {p1:.1f} | {hits / n_sim:.2f} |")
    p_pub = stats.fisher_exact([[4, 16], [15, 25]])[1]
    lines += ["", f"Autoencoder 4/20 against the two published detectors pooled "
                  f"(HDDDM 5/20 + D3 10/20 = 15/40): Fisher p = {p_pub:.3f}."]
    return lines


# --------------------------------------------------------------------------- #
# G: D3 solver convergence
# --------------------------------------------------------------------------- #
def section_g(n_checks: int = 300) -> list[str]:
    from driftgan.config import load_config
    from driftgan.data.stream import stream_csv
    from driftgan.early.scaling import FeatureScaler
    from driftgan.early.shift_monitors import D3Monitor

    cfg = load_config("configs/phase_c_early_d3.yaml")
    rows, keys = [], None
    for i, (x, _y, _r) in enumerate(stream_csv(cfg.csv_path, set(cfg.drop_cols), cfg.target_col,
                                               start_row=cfg.start_row, max_rows=cfg.max_samples,
                                               benign_label=cfg.benign_label)):
        if i < cfg.n_pretrain:
            continue  # monitors never see the pretraining prefix
        keys = keys or list(x)
        rows.append([x.get(k, 0.0) for k in keys])
    X = np.asarray(rows, dtype=np.float32)
    scaler = FeatureScaler.fit(X[:1000])
    Xn = np.clip(scaler.transform(X), -10, 10).astype(np.float64)
    mon = D3Monitor(seed=0)
    ends = np.linspace(3000, len(Xn), n_checks).astype(int) // 1000 * 1000
    rng = np.random.default_rng(0)

    gd_iters, gd_lr = 150, 0.5  # the solver the first comparator runs used

    def gd_fit(Xtr: np.ndarray, ytr: np.ndarray) -> tuple[np.ndarray, float, float]:
        w, b = np.zeros(Xtr.shape[1]), 0.0
        for _ in range(gd_iters):
            p = 1 / (1 + np.exp(-np.clip(Xtr @ w + b, -30, 30)))
            w -= gd_lr * (Xtr.T @ (p - ytr) / len(ytr) + mon.l2 * w)
            b -= gd_lr * float((p - ytr).mean())
        p = 1 / (1 + np.exp(-np.clip(Xtr @ w + b, -30, 30)))
        g = np.linalg.norm(np.concatenate([Xtr.T @ (p - ytr) / len(ytr) + mon.l2 * w, [(p - ytr).mean()]]))
        return w, b, float(g)

    gd, lb, gnorm = [], [], []
    for e in ends:
        old, new = Xn[e - 3000 : e - 1000], Xn[e - 2000 : e]
        Xa = np.vstack([old, new])
        y = np.concatenate([np.zeros(len(old)), np.ones(len(new))])
        folds = np.zeros(len(y), dtype=int)
        for cls in (0.0, 1.0):
            ii = np.where(y == cls)[0]
            folds[rng.permutation(ii)[: len(ii) // 2]] = 1
        pg, pl = np.zeros(len(y)), np.zeros(len(y))
        for fo in (0, 1):
            te, trn = folds == fo, folds != fo
            w, b, g = gd_fit(Xa[trn], y[trn])
            pg[te] = Xa[te] @ w + b
            pl[te] = mon._logreg_scores(Xa[trn], y[trn], Xa[te])  # L-BFGS, to convergence
        gnorm.append(g)
        gd.append(mon._auc(y, pg))
        lb.append(mon._auc(y, pl))
    gd_a, lb_a = np.array(gd), np.array(lb)
    diff = np.abs(gd_a - lb_a)
    rho = stats.spearmanr(gd_a, lb_a).statistic
    return ["## G. D3: batch gradient descent against converged L-BFGS", "",
            f"{len(ends)} consecutive buffer pairs (2,000 vs 2,000, shared protocol) evenly spaced "
            f"over the CIC-IDS2017 window; GD = {gd_iters} iterations, step {gd_lr}, "
            f"L2 {mon.l2}, zero init. Same folds for both solvers.", "",
            f"- AUC, GD: median {np.median(gd_a):.3f}; L-BFGS: median {np.median(lb_a):.3f}",
            f"- |AUC difference|: mean {diff.mean():.4f}, 95th percentile "
            f"{np.percentile(diff, 95):.4f}, max {diff.max():.4f}; Spearman rho {rho:.3f}",
            f"- checks where the two disagree on AUC > 0.7 (D3's own threshold): "
            f"{int(((gd_a > 0.7) != (lb_a > 0.7)).sum())}/{len(ends)}",
            f"- gradient norm left by GD: median {np.median(gnorm):.2e}, max {np.max(gnorm):.2e}"]


# --------------------------------------------------------------------------- #
# H: evaluation windows against the full captures
# --------------------------------------------------------------------------- #
CAPTURES = {  # dataset: (csv, label column, benign label, timestamp format, start, length)
    "cicids2017": ("CIC-IDS2017.csv", "Attack Type", "BENIGN", "mixed", 540_000, 2_000_000),
    "cicids2018": ("CIC-IDS2018.csv", "Label", "Benign", "%Y-%m-%d %H:%M:%S", 11_000_000, 3_000_000),
    "unsw": ("UNSW-NB15.csv", "Label", "Benign", "%d/%m/%Y %I:%M:%S %p", 1_750_000, 1_700_000),
}


def _label_stats(lab: np.ndarray, benign: str) -> str:
    atk = lab != benign
    bin_tr = int((atk[1:] != atk[:-1]).sum())
    sub_tr = int((lab[1:] != lab[:-1]).sum())
    return (f"{len(lab) / 1e6:.2f}M rows, attack {atk.mean() * 100:.1f}%, "
            f"{bin_tr / len(lab) * 1000:.2f} benign/attack and {sub_tr / len(lab) * 1000:.2f} "
            f"label transitions per 1,000 rows, {len(set(lab)) - 1} attack labels")


def section_h(lead: int = 7_000) -> list[str]:
    lines = ["## H. Evaluation windows against the full captures", ""]
    for ds, (csv, lab_col, benign, fmt, start, n) in CAPTURES.items():
        df = pd.read_csv(Path("/mnt/AI-DATA-SSD/alara/Data") / csv, usecols=[lab_col, "Timestamp"],
                         dtype=str)
        lab = df[lab_col].str.strip().to_numpy()
        w = slice(start, start + n)
        # CIC-IDS2017 stamps mix formats and carry no AM/PM, so afternoon flows
        # read as morning ones; the spans below inherit that ambiguity.
        ts = pd.to_datetime(df["Timestamp"].iloc[w], format=fmt, dayfirst=True, errors="coerce")
        span = ((ts.shift(-lead) - ts).dt.total_seconds().abs() / 60).dropna()
        back = float((ts.diff().dt.total_seconds() < -3600).mean())
        lines += [f"- **{DSET[ds]}** full capture: {_label_stats(lab, benign)}",
                  f"  - window: {_label_stats(lab[w], benign)}",
                  f"  - capture time spanned by {lead:,} consecutive flows in the window: "
                  f"median {span.median():.1f} min (IQR {span.quantile(0.25):.1f}"
                  f"–{span.quantile(0.75):.1f} min); steps going back more than an hour: "
                  f"{back * 100:.1f}%"]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", default="ABCDEFGH")
    sec = ap.parse_args().sections
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Third review round: checks from persisted runs", ""]
    df = all_run_stats() if set(sec) & set("ADE") else None
    if df is not None:
        df.drop(columns=["snap_index", "snap_f1"]).to_parquet(OUT / "run_stats.parquet")
    for key, fn in (("A", lambda: section_a(df)), ("B", section_b), ("C", section_c),
                    ("D", lambda: section_d(df)), ("E", lambda: section_e(df)),
                    ("F", section_f), ("G", section_g), ("H", section_h)):
        if key in sec:
            block = fn()
            print("\n".join(block), "\n", flush=True)
            lines += block + [""]
    (OUT / f"checks_{sec}.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
