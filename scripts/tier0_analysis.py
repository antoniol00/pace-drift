"""Tier 0 analysis — does the reported reactive->proactive effect survive scrutiny?

Pure post-processing over already-persisted artifacts (no re-runs). Three parts,
answering the three "free" reviewer objections:

  A. Inferential statistics on the reported 2x2 (P3).
     Per cell (dataset x base learner) the overall local F1 is paired by seed
     between the reactive arm (B / *_sched) and the proactive arm (C_early /
     *_early). We report the per-seed delta, a paired-t 95% CI, a paired
     bootstrap CI, the sign pattern, and Wilcoxon — plus the adaptation-count
     (n_drifts) context so the frequency confound is visible in the same table.
     An aggregate over the 9 cells treats each *cell mean* as one observation
     (cells, not seeds, are the unit of replication -> no seed pseudo-replication).

  B. Event stratification (P4).
     Within each proactive run, real drifts are the class-PHT fires. Using the
     matched recon->class pairs (lead_time.json) each class fire is labelled
     anticipated (a monitor window preceded it) or reactive-only. If the gain is
     anticipation, anticipated drifts should recover better (lower AUD, higher
     min_F1, fewer no_recovery). The recon false-alarm cost is reported alongside.

  C. Prequential kappa (P6).
     Kappa-M (vs majority-class) and Kappa-Temporal (vs persistence) computed
     from the per-sample traces, so the result is also stated in the metric the
     data-streams community expects given the class imbalance.

Run:  uv run python scripts/tier0_analysis.py
Writes results/report/tier0/analysis.md and prints the same tables.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats

RUNS = Path("results/runs")
REC = Path("results/recovery")
OUT = Path("results/report/tier0")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]

# (dataset, base-learner label, reactive phase, proactive phase)
CELLS = [
    ("cicids2017", "HAT", "B", "C_early"),
    ("cicids2017", "ARF", "ARF_sched", "ARF_early"),
    ("cicids2017", "SRP", "SRP_sched", "SRP_early"),
    ("cicids2018", "HAT", "B", "C_early"),
    ("cicids2018", "ARF", "ARF_sched", "ARF_early"),
    ("cicids2018", "SRP", "SRP_sched", "SRP_early"),
    ("unsw", "HAT", "B", "C_early"),
    ("unsw", "ARF", "ARF_sched", "ARF_early"),
    ("unsw", "SRP", "SRP_sched", "SRP_early"),
]
DSET = {"cicids2017": "CIC-IDS2017", "cicids2018": "CIC-IDS2018", "unsw": "UNSW-NB15"}


def run_dir(dataset: str, phase: str, seed: int) -> Path:
    base = RUNS if dataset == "cicids2017" else RUNS / dataset
    return base / phase / f"seed_{seed}"


def rec_path(dataset: str, phase: str, seed: int) -> Path:
    prefix = "" if dataset == "cicids2017" else f"{dataset}_"
    return REC / f"{prefix}{phase}_{seed}.parquet"


def per_seed_local_f1(dataset: str, phase: str) -> np.ndarray:
    """Overall local F1 (mean valid-window rolling F1) for each seed separately."""
    out = []
    for s in SEEDS:
        df = pd.read_parquet(run_dir(dataset, phase, s) / "snapshots.parquet")
        out.append(float(df.loc[df["valid"], "f1"].mean()))
    return np.array(out)


def per_seed_n_drifts(dataset: str, phase: str) -> np.ndarray:
    out = []
    for s in SEEDS:
        d = json.loads((run_dir(dataset, phase, s) / "summary.json").read_text())
        out.append(int(d["n_drifts"]))
    return np.array(out)


# --------------------------------------------------------------------------- #
# A. Inferential statistics on the reported 2x2
# --------------------------------------------------------------------------- #
class CellStat(NamedTuple):
    dataset: str
    learner: str
    f1_r: float
    f1_e: float
    deltas: np.ndarray
    mean_delta: float
    t_lo: float
    t_hi: float
    boot_lo: float
    boot_hi: float
    n_pos: int
    wilcoxon_p: float
    nd_r: float
    nd_e: float

    @property
    def verdict(self) -> str:
        # Robust: every seed agrees in sign AND the paired-t CI excludes 0.
        same_sign = self.n_pos in (0, len(self.deltas))
        ci_excludes_0 = (self.t_lo > 0) or (self.t_hi < 0)
        if same_sign and ci_excludes_0:
            return "ROBUST"
        if ci_excludes_0:
            return "CI-signif"
        return "WEAK"


def paired_ci(deltas: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    n = len(deltas)
    m = float(deltas.mean())
    se = float(deltas.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return m - tcrit * se, m + tcrit * se


def paired_bootstrap_ci(deltas: np.ndarray, n_boot: int = 10000,
                        seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def section_a() -> list[CellStat]:
    stats_out: list[CellStat] = []
    for ds, learner, r, e in CELLS:
        f1r = per_seed_local_f1(ds, r)
        f1e = per_seed_local_f1(ds, e)
        d = f1e - f1r
        t_lo, t_hi = paired_ci(d)
        b_lo, b_hi = paired_bootstrap_ci(d)
        try:
            w = stats.wilcoxon(f1e, f1r).pvalue
        except ValueError:
            w = float("nan")
        stats_out.append(CellStat(
            dataset=ds, learner=learner,
            f1_r=float(f1r.mean()), f1_e=float(f1e.mean()),
            deltas=d, mean_delta=float(d.mean()),
            t_lo=t_lo, t_hi=t_hi, boot_lo=b_lo, boot_hi=b_hi,
            n_pos=int((d > 0).sum()), wilcoxon_p=float(w),
            nd_r=float(per_seed_n_drifts(ds, r).mean()),
            nd_e=float(per_seed_n_drifts(ds, e).mean()),
        ))
    return stats_out


def render_a(rows: list[CellStat]) -> str:
    lines = ["## A. Inferential statistics on the reported reactive->proactive 2x2",
             "",
             "Overall local F1 paired by seed (n=5). `Δ` = proactive − reactive. "
             "`t-CI` = paired-t 95% CI; `boot-CI` = paired percentile bootstrap. "
             "`sign` = # seeds with Δ>0. `fires` = mean adaptation count "
             "(reactive→proactive); >1× means the proactive arm adapts *more* "
             "(frequency confound, to be removed in Tier 1).",
             "",
             "| dataset | learner | F1 react | F1 proact | Δ mean | t-CI 95% | "
             "boot-CI 95% | sign | Wilcoxon p | fires r→e | verdict |",
             "|---|---|---:|---:|---:|---|---|:---:|---:|---:|:---:|"]
    for x in rows:
        lines.append(
            f"| {DSET[x.dataset]} | {x.learner} | {x.f1_r:.3f} | {x.f1_e:.3f} | "
            f"{x.mean_delta:+.3f} | [{x.t_lo:+.3f}, {x.t_hi:+.3f}] | "
            f"[{x.boot_lo:+.3f}, {x.boot_hi:+.3f}] | {x.n_pos}/5 | "
            f"{x.wilcoxon_p:.3f} | {x.nd_r:.0f}→{x.nd_e:.0f} "
            f"({x.nd_e / x.nd_r:.1f}×) | {x.verdict} |")

    # aggregate over the 9 cells (cell means as unit of replication)
    cell_means = np.array([x.mean_delta for x in rows])
    m = float(cell_means.mean())
    se = float(cell_means.std(ddof=1) / np.sqrt(len(cell_means)))
    tcrit = float(stats.t.ppf(0.975, df=len(cell_means) - 1))
    n_pos_cells = int((cell_means > 0).sum())
    sign_p = float(stats.binomtest(n_pos_cells, len(cell_means), 0.5).pvalue)
    lines += [
        "",
        f"**Aggregate over the 9 cells** (each cell mean = 1 observation, "
        f"df={len(cell_means) - 1}): mean Δ = {m:+.3f}, "
        f"95% CI [{m - tcrit * se:+.3f}, {m + tcrit * se:+.3f}]; "
        f"{n_pos_cells}/9 cells positive (two-sided sign test p={sign_p:.4f}).",
        "",
        "> Caveat: the 9 cells share the monitor design and only 3 datasets, so "
        "they are not fully independent; the sign-test p is optimistic. The "
        "per-cell CIs are the primary evidence.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# B. Event stratification: anticipated vs reactive-only drifts
# --------------------------------------------------------------------------- #
def section_b_cell(dataset: str, proactive: str) -> dict:
    """Pool proactive class-fire recovery events, split anticipated vs not."""
    antic_rows, react_rows = [], []
    recon_fp = recon_total = 0
    for s in SEEDS:
        rd = run_dir(dataset, proactive, s)
        ev = json.loads((rd / "drift_events.json").read_text())["events"]
        lead = json.loads((rd / "lead_time.json").read_text())
        matched_class = {p[1] for p in lead["pairs"]}
        recon_fp += int(round(lead["anticipated_fp_rate"] * lead["n_recon_fires"]))
        recon_total += lead["n_recon_fires"]
        src = {e["index"]: e["source"] for e in ev}
        rec = pd.read_parquet(rec_path(dataset, proactive, s))
        for _, row in rec.iterrows():
            di = int(row["drift_index"])
            if src.get(di) != "class":  # only real (error-confirmed) drifts
                continue
            bucket = antic_rows if di in matched_class else react_rows
            bucket.append((row["AUD"], row["min_F1"], row["no_recovery"], row["RT"]))
    return {
        "antic": np.array(antic_rows, dtype=float),
        "react": np.array(react_rows, dtype=float),
        "recon_fp_rate": (recon_fp / recon_total) if recon_total else float("nan"),
    }


def _summ(arr: np.ndarray) -> str:
    if len(arr) == 0:
        return "n=0"
    aud, minf1, norec = arr[:, 0], arr[:, 1], arr[:, 2]
    return (f"n={len(arr)} · AUD med={np.nanmedian(aud):.0f} · "
            f"min_F1 med={np.nanmedian(minf1):.2f} · "
            f"no_recov={norec.mean() * 100:.0f}%")


def render_b() -> str:
    lines = ["## B. Event stratification — do anticipated drifts recover better?",
             "",
             "Real drifts = class-PHT fires in the proactive run. A drift is "
             "*anticipated* if a recon window preceded it (matched pair). Pooled "
             "across seeds. Mechanistic prediction: anticipated drifts show lower "
             "AUD / higher min_F1 / fewer no-recovery than reactive-only drifts.",
             "",
             "| dataset | learner | anticipated drifts | reactive-only drifts | "
             "AUD MWU p | recon false-alarm rate |",
             "|---|---|---|---|---:|---:|"]
    for ds, learner, _r, e in CELLS:
        c = section_b_cell(ds, e)
        a, k = c["antic"], c["react"]
        if len(a) and len(k):
            p = stats.mannwhitneyu(a[:, 0], k[:, 0], alternative="less").pvalue
            pstr = f"{p:.3f}"
        else:
            pstr = "n/a"
        lines.append(f"| {DSET[ds]} | {learner} | {_summ(a)} | {_summ(k)} | "
                     f"{pstr} | {c['recon_fp_rate'] * 100:.0f}% |")
    lines += [
        "",
        "> `AUD MWU p` = one-sided Mann-Whitney (anticipated < reactive-only on "
        "area-under-deficit). Low p ⇒ anticipated drifts recover with less "
        "deficit. The false-alarm rate is the cost: recon fires that opened a "
        "window with no imminent drift.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# C. Prequential kappa from traces
# --------------------------------------------------------------------------- #
def kappa_from_trace(path: Path) -> tuple[float, float]:
    """(Kappa-M, Kappa-Temporal) over the full evaluation stream in the trace."""
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f, usecols=["y_true", "y_pred", "correct"])
    y = df["y_true"].to_numpy()
    n = len(y)
    p0 = float(df["correct"].mean())
    # majority-class classifier accuracy = largest class prior
    _, counts = np.unique(y, return_counts=True)
    pe_m = float(counts.max() / n)
    # persistence classifier accuracy = P(y_t == y_{t-1})
    pe_t = float((y[1:] == y[:-1]).mean())
    km = (p0 - pe_m) / (1 - pe_m) if (1 - pe_m) > 1e-9 else float("nan")
    kt = (p0 - pe_t) / (1 - pe_t) if (1 - pe_t) > 1e-9 else float("nan")
    return km, kt


def section_c() -> list[dict]:
    rows = []
    for ds, learner, r, e in CELLS:
        km_r, kt_r, km_e, kt_e = [], [], [], []
        for s in SEEDS:
            for ph, kmL, ktL in ((r, km_r, kt_r), (e, km_e, kt_e)):
                tp = run_dir(ds, ph, s) / "trace.csv.gz"
                if tp.exists():
                    km, kt = kappa_from_trace(tp)
                    kmL.append(km)
                    ktL.append(kt)
        rows.append({
            "dataset": ds, "learner": learner,
            "kmR": np.nanmean(km_r), "kmE": np.nanmean(km_e),
            "ktR": np.nanmean(kt_r), "ktE": np.nanmean(kt_e),
        })
    return rows


def render_c(rows: list[dict]) -> str:
    lines = ["## C. Prequential kappa (streams-standard, imbalance-aware)",
             "",
             "Cumulative over the post-pretrain evaluation stream. Kappa-M vs "
             "majority-class, Kappa-Temporal vs persistence. `Δ` = proactive − "
             "reactive.",
             "",
             "| dataset | learner | κM react | κM proact | ΔκM | κTemp react | "
             "κTemp proact | ΔκTemp |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for x in rows:
        lines.append(
            f"| {DSET[x['dataset']]} | {x['learner']} | {x['kmR']:.3f} | "
            f"{x['kmE']:.3f} | {x['kmE'] - x['kmR']:+.3f} | {x['ktR']:.3f} | "
            f"{x['ktE']:.3f} | {x['ktE'] - x['ktR']:+.3f} |")
    return "\n".join(lines)


def main() -> None:
    a = section_a()
    md_a = render_a(a)
    md_b = render_b()
    c = section_c()
    md_c = render_c(c)

    n_robust = sum(1 for x in a if x.verdict == "ROBUST")
    n_weak = sum(1 for x in a if x.verdict == "WEAK")
    header = [
        "# Tier 0 — statistical hardening of the reported effect",
        "",
        "Post-processing only (no re-runs). Does the reactive→proactive advantage "
        "survive inferential scrutiny before we invest Tier-1 compute?",
        "",
        f"**Gate summary:** {n_robust}/9 cells ROBUST "
        f"(all seeds same sign AND paired-t CI excludes 0); {n_weak}/9 WEAK "
        "(CI straddles 0 — likely seed noise).",
        "",
    ]
    doc = "\n".join(header) + "\n" + md_a + "\n\n" + md_b + "\n\n" + md_c + "\n"
    (OUT / "analysis.md").write_text(doc)
    print(doc)
    print(f"\nwrote {OUT / 'analysis.md'}")


if __name__ == "__main__":
    main()
