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


# --- cheap shift-monitor ablations (share the AE's frozen-reference protocol) ---
import pytest  # noqa: E402

from driftgan.early.shift_monitors import (  # noqa: E402
    HDDDMMonitor,
    KSMonitor,
    MeanShiftMonitor,
    MMDMonitor,
    PCAReconstructionMonitor,
)


def _make(kind):
    common = dict(check_interval=200, buffer_size=400, seed=0)
    if kind is PCAReconstructionMonitor:
        return kind(latent=4, **common)
    if kind is MMDMonitor:
        return kind(sample=128, **common)
    return kind(**common)


@pytest.mark.parametrize(
    "kind", [PCAReconstructionMonitor, KSMonitor, MMDMonitor, MeanShiftMonitor, HDDDMMonitor]
)
def test_shift_monitor_spikes_after_covariate_shift(kind):
    m = _make(kind)
    stable = _feed(m, mean=0.0, n=3000, seed=1)
    shifted = _feed(m, mean=8.0, n=2000, seed=2, start=3000)
    assert len(stable) >= 3 and len(shifted) >= 3
    baseline = float(np.median(stable))
    assert max(shifted[:3]) > 2 * (baseline + 1e-9)


@pytest.mark.parametrize(
    "kind", [PCAReconstructionMonitor, KSMonitor, MMDMonitor, MeanShiftMonitor, HDDDMMonitor]
)
def test_shift_monitor_quiet_without_shift(kind):
    m = _make(kind)
    _feed(m, mean=0.0, n=600, seed=1)
    steady = _feed(m, mean=0.0, n=3000, seed=2, start=600)
    shifted = _feed(m, mean=6.0, n=1500, seed=3, start=3600)
    # No-shift scores stay well below the monitor's own response to a real shift.
    assert max(steady) < 0.5 * max(shifted[:3])


def test_d3_monitor_auc_separability_semantics():
    """D3's discriminative statistic: calibrated on identical data, sharp on shift.

    Also pins the *known limitation* that explains the comparator result in
    docs/experiments/07-unsupervised-comparator.md: a linear discriminator sees
    mean shifts but is blind to a variance-only change, which reconstruction
    error does capture.
    """
    from driftgan.early.shift_monitors import D3Monitor

    m = D3Monitor(seed=0)
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((600, 20))

    m._ref = ref
    same = m._score(rng.standard_normal((600, 20)))
    assert 0.40 < same < 0.60  # no separability on identical distributions

    m._ref = ref
    shifted = m._score(rng.standard_normal((600, 20)) + 3.0)
    assert shifted > 0.95  # mean shift is trivially separable

    m._ref = ref
    variance_only = m._score(rng.standard_normal((600, 20)) * 2.5)
    assert variance_only < 0.65  # blind spot: same mean, different dispersion


def test_hdddm_sees_the_variance_change_d3_misses():
    """HDDDM's distribution distance covers D3's blind spot.

    This is why HDDDM is the informative second comparator rather than a
    repetition: a bin-by-bin Hellinger distance responds to a change in
    dispersion at a constant mean, which the linear discriminator above cannot
    see. Any closed-loop instability HDDDM shows therefore cannot be blamed on
    that blind spot.
    """
    m = HDDDMMonitor(seed=0)
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((600, 20))

    m._ref = ref
    same = m._score(rng.standard_normal((600, 20)))
    variance_only = m._score(rng.standard_normal((600, 20)) * 2.5)
    shifted = m._score(rng.standard_normal((600, 20)) + 3.0)

    assert variance_only > 2 * same  # the case D3 scores at chance
    assert shifted > 2 * same


def test_hdddm_tolerates_degenerate_features():
    """Near-constant float32 columns must not break the histogram.

    Flow data is full of features that are constant or vary by a hair over a
    window (always-zero flags, saturated counters). Their bin edges collide, so
    they carry no divergence and must be skipped rather than raising.
    """
    m = HDDDMMonitor(seed=0)
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((400, 5)).astype(np.float32)
    cur = rng.standard_normal((400, 5)).astype(np.float32)
    for col, value in ((0, 0.0), (1, 7.5)):
        ref[:, col] = value
        cur[:, col] = value
    ref[:, 2] = 3.0  # varies, but far below float32 edge resolution
    cur[:, 2] = np.float32(3.0) + rng.standard_normal(400).astype(np.float32) * 1e-9

    m._ref = ref
    score = m._score(cur)
    assert np.isfinite(score) and score >= 0.0


def _feed_native(m, n, mean, start, dim=20):
    """Push ``n`` samples through a native-window monitor, collecting tests."""
    rng = np.random.default_rng(abs(hash((n, mean, start))) % 2**32)
    out = []
    for i in range(n):
        x = {f"f{j}": float(v) for j, v in enumerate(rng.standard_normal(dim) + mean)}
        s = m.observe(x, start + i)
        if s is not None:
            out.append((s, m.fired))
    return out


def test_d3_native_uses_its_own_window_and_decision_rule():
    """Native D3 tests every w*rho samples and fires on its own AUC threshold.

    Pins the mechanics the shared-protocol version does not have: the window
    must fill before any test, tests then recur on D3's own cadence, and a fire
    is declared by the monitor itself (``declares_drift``) rather than by the
    runner's Page-Hinkley. Sized here so the AUC estimate is stable — see
    :func:`test_d3_native_false_alarms_at_published_defaults` for what happens
    when it is not.
    """
    from driftgan.early.shift_monitors import D3NativeMonitor

    m = D3NativeMonitor(w=500, rho=0.2, tau=0.7, seed=0)
    assert m.declares_drift is True

    # Window is w(1+rho) = 600: nothing can be tested before it fills.
    assert _feed_native(m, 599, 0.0, 0) == []
    stable = _feed_native(m, 1000, 0.0, 599)
    assert len(stable) == 10  # one test per w*rho = 100 samples
    assert not any(fired for _, fired in stable)  # stationary stream: no drift

    shifted = _feed_native(m, 600, 6.0, 1599)
    assert any(fired for _, fired in shifted)
    assert all(score > m.tau for score, fired in shifted if fired)


def test_d3_native_false_alarms_at_published_defaults():
    """D3's released defaults are mis-sized for high-dimensional flow features.

    At w=100, rho=0.1 the "new" window holds 10 samples, so the 2-fold CV AUC
    is estimated from ~5 positives per fold and crosses tau by chance on a
    perfectly stationary stream. This is a property of the published settings
    on data of this width, not of our reimplementation, and it is the reason
    the native-D3 arm is run both at the defaults and with a window sized for
    the feature dimension (docs/experiments/07-unsupervised-comparator.md).
    """
    from driftgan.early.shift_monitors import D3NativeMonitor

    default = D3NativeMonitor(w=100, rho=0.1, tau=0.7, seed=0)
    tests = _feed_native(default, 20_000, 0.0, 0)
    rate = sum(fired for _, fired in tests) / len(tests)
    assert rate > 0.02  # fires on stationary data at the published window size

    sized = D3NativeMonitor(w=1000, rho=0.2, tau=0.7, seed=0)
    tests_sized = _feed_native(sized, 20_000, 0.0, 0)
    assert not any(fired for _, fired in tests_sized)  # a wider window is quiet
