"""Tests for the early-detection monitor and dual PHT (Improvement 6).

The monitor's contract is that a *covariate shift* raises reconstruction
error against a frozen reference frame — the redesign that replaced per-buffer
standardization (which cancelled the very signal being detected).
"""

import numpy as np

from driftgan.early.dual_pht import DualPHT, match_lead_times
from driftgan.early.reconstruction_monitor import ReconstructionMonitor


def _feed(monitor, mean, n, dim=6, seed=0, start=0):
    rng = np.random.default_rng(seed)
    errs = []
    for i in range(n):
        x = {f"f{j}": float(v) for j, v in enumerate(rng.normal(mean, 1.0, dim))}
        e = monitor.observe(x, start + i)
        if e is not None:
            errs.append(e)
    return errs


def test_recon_error_spikes_after_covariate_shift():
    m = ReconstructionMonitor(
        check_interval=200, buffer_size=400, epochs=25, device="cpu", seed=0
    )
    stable = _feed(m, mean=0.0, n=3000, seed=1)
    shifted = _feed(m, mean=8.0, n=2000, seed=2, start=3000)
    assert len(stable) >= 3 and len(shifted) >= 3
    baseline = float(np.median(stable))
    # The frozen reference frame makes the shift visible: error jumps well
    # above the stable baseline right after the distribution moves.
    assert max(shifted[:3]) > 3 * (baseline + 1e-6)


def test_recon_error_stays_low_without_shift():
    m = ReconstructionMonitor(
        check_interval=200, buffer_size=400, epochs=25, device="cpu", seed=0
    )
    warm = _feed(m, mean=0.0, n=600, seed=1)
    steady = _feed(m, mean=0.0, n=3000, seed=2, start=600)
    # No spurious spikes when the distribution is unchanged.
    assert max(steady) < 5 * (float(np.median(warm)) + 1e-6)


def test_dual_pht_recon_fires_on_sustained_rise():
    d = DualPHT(class_delta=0.005, recon_delta=0.005, recon_threshold=1.0, recon_min_instances=5)
    for i in range(20):
        d.update_recon(i, 0.1)
    for i in range(20, 45):
        fired = d.update_recon(i, 3.0)
        if fired:
            break
    assert len(d.recon_fires) >= 1


def test_dual_pht_recon_quiet_on_flat_series():
    d = DualPHT(class_delta=0.005, recon_delta=0.005, recon_threshold=1.0, recon_min_instances=5)
    for i in range(80):
        d.update_recon(i, 0.1)
    assert d.recon_fires == []


def test_match_lead_times_pairs_within_horizon():
    report = match_lead_times(recon_fires=[100, 900], class_fires=[300, 1000], horizon=500)
    assert report.n_recon_fires == 2 and report.n_class_fires == 2
    assert report.lead_times == [200, 100]
    assert report.mean_lead_time == 150.0
    assert report.missed_anticipation_rate == 0.0
    assert report.anticipated_fp_rate == 0.0


def test_match_lead_times_counts_missed_and_fp():
    # recon at 10 is too early for class at 1000 (horizon 100); class at 1000
    # is unanticipated, recon at 10 is an anticipated false positive.
    report = match_lead_times(recon_fires=[10], class_fires=[1000], horizon=100)
    assert report.pairs == []
    assert report.missed_anticipation_rate == 1.0
    assert report.anticipated_fp_rate == 1.0
