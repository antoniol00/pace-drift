"""Security evaluation on the anchor: trigger-aware adversary and the defence.

Compares, over twenty seeds of HAT on CIC-IDS2017, natural PACE and the live
error test against an oracle adversary that chooses when the monitor branch
fires (toward attack-dense windows, ``adv``, or attack-free ones, ``advb``),
against chaff the real monitor sees, and each with the window-rebalancing
defence (``_rb``). Reports closed-loop stability (collapse, benign FPR), attack
recall, what the learner is fed, the label share, how often chaff fired the
trigger, and the paired effect of the defence on local F1.

Run: uv run python scripts/security_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from comparator_analysis import (
    COLLAPSE_TAU,
    RUNS,
    confusion_stats,
    fire_split,
    fisher_vs_reference,
    paired_ci,
    per_seed_local_f1,
    per_seed_summary,
    seeds_of,
    train_window_attack_fraction,
)
from tier1_analysis import per_seed_labels

OUT = Path("results/report/security")
TARGETS = Path("results/adversary/cicids2017_attack.json")
SEEDS = list(range(20))

# (phase, label, reference for the Fisher test)
ARMS = [
    ("C_early", "PACE", None),
    ("C_early_adv", "PACE, oracle adversary (attack)", "C_early"),
    ("C_early_advb", "PACE, oracle adversary (benign)", "C_early"),
    ("C_early_chaff", "PACE, chaff", "C_early"),
    ("B_pht", "live error test", None),
    ("B_pht_chaff", "live error test, chaff", "B_pht"),
    ("C_early_rb", "PACE + rebalancing", "C_early"),
    ("C_early_adv_rb", "oracle adversary (attack) + rebalancing", "C_early_adv"),
    ("C_early_chaff_rb", "chaff + rebalancing", "C_early_chaff"),
    ("B_pht_rb", "live error test + rebalancing", "B_pht"),
    ("C_early_advb_rb", "oracle adversary (benign) + rebalancing", "C_early_advb"),
    ("B_pht_chaff_rb", "live error test, chaff + rebalancing", "B_pht_chaff"),
    ("C_early_rb50", "PACE + balanced rebalancing (0.5)", "C_early"),
    ("C_early_chaff_rb50", "chaff + balanced rebalancing (0.5)", "C_early_chaff"),
    ("C_freq", "blind clock", None),
    ("C_freq_rb", "blind clock + rebalancing", "C_freq"),
    ("ARF_early", "ARF, PACE", None),
    ("ARF_early_rb", "ARF, PACE + rebalancing", "ARF_early"),
    ("SRP_early", "SRP, PACE", None),
    ("SRP_early_rb", "SRP, PACE + rebalancing", "SRP_early"),
    # Does collapse need the tree? A single online logistic regression.
    ("LR_early", "LR, PACE", None),
    ("LR_pht", "LR, live error test", "LR_early"),
    ("LR_freq", "LR, blind clock", "LR_early"),
    ("LR_early_rb", "LR, PACE + rebalancing", "LR_early"),
    ("LR_early_chaff", "LR, PACE, chaff", "LR_early"),
    ("LR_early_chaff_rb", "LR, chaff + rebalancing", "LR_early_chaff"),
]
DEFENCE_PAIRS = [
    ("C_early", "C_early_rb"),
    ("B_pht", "B_pht_rb"),
    ("C_early_adv", "C_early_adv_rb"),
    ("C_early_chaff", "C_early_chaff_rb"),
    ("C_early_advb", "C_early_advb_rb"),
    ("B_pht_chaff", "B_pht_chaff_rb"),
    ("C_early", "C_early_rb50"),
    ("C_early_chaff", "C_early_chaff_rb50"),
    ("C_freq", "C_freq_rb"),
    ("ARF_early", "ARF_early_rb"),
    ("SRP_early", "SRP_early_rb"),
    ("LR_early", "LR_early_rb"),
    ("LR_early_chaff", "LR_early_chaff_rb"),
]
# Do the paper's two contrasts survive once the training distribution is pinned?
RB_CONTRASTS = [
    ("C_early_rb", "C_freq_rb", "timing under the defence (PACE - blind clock)"),
    ("C_early_rb", "B_pht_rb", "monitor under the defence (PACE - live error test)"),
    ("C_early", "C_freq", "timing without the defence, same seeds"),
    ("C_early", "B_pht", "monitor without the defence, same seeds"),
    ("LR_early", "LR_freq", "LR timing (PACE - blind clock)"),
    ("LR_early", "LR_pht", "LR monitor (PACE - live error test)"),
    ("LR_early_chaff", "LR_early", "LR chaff (chaff - none)"),
]


def fed_shares(phase: str, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Attack share of the real samples learned in windows, and of everything fed.

    The two differ only under the defence, whose replays enter the second.
    Arms run before the runner counted what it fed fall back to the trace.
    """
    real, fed = [], []
    for s in seeds:
        learned = json.loads((RUNS / phase / f"seed_{s}" / "summary.json").read_text()).get("learned")
        if not learned:
            share = train_window_attack_fraction(phase, [s])
            assert share is not None
            real.append(float(share[0]))
            fed.append(float(share[0]))
            continue
        a, b = learned.get("ATTACK", 0), learned.get("BENIGN", 0)
        ra, rb = learned.get("replay_ATTACK", 0), learned.get("replay_BENIGN", 0)
        real.append(a / (a + b))
        fed.append((a + ra) / (a + b + ra + rb))
    return np.array(real), np.array(fed)


def replay_ratio(phase: str, seeds: list[int]) -> float:
    """Replayed samples per real sample learned (cost of the defence)."""
    rep = real = 0
    for s in seeds:
        learned = json.loads((RUNS / phase / f"seed_{s}" / "summary.json").read_text())["learned"]
        rep += learned.get("replay_ATTACK", 0) + learned.get("replay_BENIGN", 0)
        real += learned.get("ATTACK", 0) + learned.get("BENIGN", 0)
    return rep / real


def update_time(phase: str, seeds: list[int]) -> float:
    """Mean seconds per run spent in the learner's update (replays included)."""
    t = per_seed_summary(phase, "time_hat_update_s", seeds)
    assert t is not None
    return float(t.mean())


def target_hits(phase: str, seeds: list[int], key: str, after: int = 2_000) -> float:
    """Share of the adversary's targets followed by a firing within ``after`` samples.

    ``key`` picks the firings: monitor firings from lead_time.json ("recon"), or
    every firing from drift_events.json ("all"). Chaff does not advance the
    index, so a firing during the chaff before target ``t`` is logged at t - 1.
    """
    targets = json.loads(TARGETS.read_text())["indices"]
    hits = []
    for s in seeds:
        d = RUNS / phase / f"seed_{s}"
        if key == "recon":
            fires = json.loads((d / "lead_time.json").read_text())["recon_fires"]
        else:
            fires = [e["index"] for e in json.loads((d / "drift_events.json").read_text())["events"]]
        f = np.array(sorted(fires))
        hits.append(np.mean([bool(((f >= t - 1) & (f <= t + after)).any()) for t in targets]))
    return float(np.mean(hits))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Security evaluation — trigger-aware adversary and window rebalancing", "",
             f"HAT, CIC-IDS2017, seeds {SEEDS[0]}–{SEEDS[-1]}; collapsed = cumulative accuracy "
             f"< {COLLAPSE_TAU}. Fisher p two-sided against the reference arm.", ""]
    targets = json.loads(TARGETS.read_text())
    lines += [f"Oracle (attack) targets: {len(targets['indices'])} firings, mean window attack "
              f"share {targets['metadata']['window_attack_share']:.3f}.", ""]
    lines += ["| Arm | n | Local F1 med | Collapsed | Fisher p (ref) | FPR med / worst | "
              "Attack recall med | Window atk / fed atk (med) | Fires mon / other | Labels |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    collapsed: dict[str, int] = {}
    f1: dict[str, np.ndarray] = {}
    for phase, label, ref in ARMS:
        seeds = sorted(seeds_of(phase) & set(SEEDS))
        if len(seeds) < len(SEEDS):
            lines.append(f"| {label} | {len(seeds)} | (incomplete) |||||||| ")
            continue
        acc = per_seed_summary(phase, "final_acc", seeds)
        lf = per_seed_local_f1(phase, seeds)
        conf = confusion_stats(phase, seeds)
        assert acc is not None and lf is not None and conf is not None
        n_bad = int((acc < COLLAPSE_TAU).sum())
        collapsed[phase], f1[phase] = n_bad, lf
        p = f"{fisher_vs_reference(n_bad, collapsed[ref], len(seeds)):.3f} ({ref})" if ref in collapsed else "n/a"
        real, fed = fed_shares(phase, seeds)
        mon, other = fire_split(phase, seeds)
        labels = per_seed_labels("cicids2017", phase, seeds).mean()
        lines.append(
            f"| {label} | {len(seeds)} | {np.median(lf):.3f} | {n_bad}/{len(seeds)} | {p} | "
            f"{np.median(conf['fpr']):.3f} / {conf['fpr'].max():.3f} | {np.median(conf['recall']):.3f} | "
            f"{np.median(real):.2f} / {np.median(fed):.2f} | {mon:.0f} / {other:.0f} | {labels:.1%} |"
        )

    lines += ["", "## Did chaff fire the trigger?", "",
              "Share of the oracle targets followed within 2,000 samples by a firing, against the same "
              "arm without chaff (the trigger's natural rate at those moments).", ""]
    for phase, base, key in (("C_early_chaff", "C_early", "recon"), ("C_early_chaff_rb", "C_early_rb", "recon"),
                             ("B_pht_chaff", "B_pht", "all"), ("LR_early_chaff", "LR_early", "recon")):
        if phase in collapsed and base in collapsed:
            lines.append(f"- {phase} ({key} firings): {target_hits(phase, SEEDS, key):.2f} "
                         f"vs {base} {target_hits(base, SEEDS, key):.2f}")

    lines += ["", "## Effect of the defence (paired by seed)", ""]
    for base, rb in DEFENCE_PAIRS:
        if base in f1 and rb in f1:
            d = f1[rb] - f1[base]
            lo, hi = paired_ci(d)
            lines.append(f"- {rb} − {base}: local F1 {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], "
                         f"{int((d > 0).sum())}/{len(d)} seeds positive; collapsed "
                         f"{collapsed[base]} → {collapsed[rb]}; replays per real sample "
                         f"{replay_ratio(rb, SEEDS):.2f}; update time {update_time(base, SEEDS):.0f} s → "
                         f"{update_time(rb, SEEDS):.0f} s per run")

    lines += ["", "## Contrasts under the defence (paired by seed)", ""]
    for a, b, label in RB_CONTRASTS:
        if a in f1 and b in f1:
            d = f1[a] - f1[b]
            lo, hi = paired_ci(d)
            lines.append(f"- {label}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], "
                         f"{int((d > 0).sum())}/{len(d)} seeds positive")

    text = "\n".join(lines) + "\n"
    (OUT / "analysis.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
