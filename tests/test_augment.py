"""Tests for augmenters: jitter baseline, vanilla GAN and conditional GAN."""

import numpy as np
import pytest

from driftgan.augment.base import FeatureScaler
from driftgan.augment.bootstrap_jitter import BootstrapJitterAugmenter
from driftgan.augment.cgan import ConditionalGanAugmenter, _proportional_counts
from driftgan.augment.vanilla_gan import VanillaGanAugmenter


def make_buffer(n=20, dim=4, label="DoS_Hulk", seed=0):
    rng = np.random.default_rng(seed)
    keys = [f"f{i}" for i in range(dim)]
    return [
        ({k: float(v) for k, v in zip(keys, row, strict=True)}, label)
        for row in rng.normal(5, 2, (n, dim))
    ]


def test_jitter_count_and_keys():
    buffer = make_buffer()
    aug = BootstrapJitterAugmenter(sigma_scale=0.01, seed=1)
    out = aug.fit_generate(buffer, 50)
    assert len(out) == 50
    assert set(out[0]) == set(buffer[0][0])
    assert aug.t_train == 0.0 and aug.t_generate >= 0.0


def test_jitter_zero_sigma_reproduces_buffer_values():
    buffer = make_buffer(n=5)
    aug = BootstrapJitterAugmenter(sigma_scale=0.0, seed=1)
    out = aug.fit_generate(buffer, 10)
    buffer_rows = {tuple(round(v, 5) for v in s.values()) for s, _ in buffer}
    for sample in out:
        assert tuple(round(v, 5) for v in sample.values()) in buffer_rows


def test_jitter_noise_scales_with_feature_std():
    buffer = make_buffer(n=200, seed=2)
    aug = BootstrapJitterAugmenter(sigma_scale=0.5, seed=3)
    out = aug.fit_generate(buffer, 500)
    X = np.array([[s[k] for k in buffer[0][0]] for s in out])
    orig = np.array([[s[k] for k in buffer[0][0]] for s, _ in buffer])
    # Variance should grow, roughly by (1 + scale^2) in each feature.
    assert (X.std(axis=0) > orig.std(axis=0)).all()


def test_jitter_empty_buffer():
    aug = BootstrapJitterAugmenter()
    assert aug.fit_generate([], 10) == []


def test_vanilla_gan_generates_n_samples():
    buffer = make_buffer(n=30)
    aug = VanillaGanAugmenter(noise_dim=4, epochs=3, device="cpu", seed=0)
    out = aug.fit_generate(buffer, 25)
    assert len(out) == 25
    assert set(out[0]) == set(buffer[0][0])
    assert aug.t_train > 0 and aug.t_generate > 0


def test_vanilla_gan_deterministic_given_seed():
    buffer = make_buffer(n=30)
    out1 = VanillaGanAugmenter(noise_dim=4, epochs=2, device="cpu", seed=7).fit_generate(buffer, 5)
    out2 = VanillaGanAugmenter(noise_dim=4, epochs=2, device="cpu", seed=7).fit_generate(buffer, 5)
    for a, b in zip(out1, out2, strict=True):
        for k in a:
            assert a[k] == pytest.approx(b[k])


def test_cgan_generates_proportional_mix():
    buffer = make_buffer(n=30, label="DoS_Hulk") + make_buffer(n=10, label="PortScan", seed=5)
    aug = ConditionalGanAugmenter(noise_dim=4, epochs=2, n_critic=1, device="cpu", seed=0)
    out = aug.fit_generate(buffer, 40)
    assert len(out) == 40
    assert set(aug.class_to_id) == {"DoS_Hulk", "PortScan"}


def test_cgan_bce_ablation_runs():
    buffer = make_buffer(n=20)
    aug = ConditionalGanAugmenter(
        noise_dim=4, epochs=2, loss="bce", label_smoothing=0.1, feature_matching=True,
        device="cpu", seed=0,
    )
    out = aug.fit_generate(buffer, 10)
    assert len(out) == 10


def test_proportional_counts_largest_remainder():
    from collections import Counter

    counts = _proportional_counts(Counter({"a": 2, "b": 1}), 10)
    assert sum(counts.values()) == 10
    assert counts["a"] == 7 and counts["b"] == 3


def test_small_buffer_returns_empty():
    aug = VanillaGanAugmenter(device="cpu")
    assert aug.fit_generate(make_buffer(n=1), 10) == []


# ── feature scaling (fixes the scale-collapse that made C == C_cond) ──────────

def test_feature_scaler_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(1e5, 1e4, (50, 6)).astype(np.float32)
    sc = FeatureScaler.fit(X)
    Xn = sc.transform(X)
    assert abs(float(Xn.mean())) < 1e-2 and abs(float(Xn.std()) - 1.0) < 0.2
    assert np.allclose(sc.inverse(Xn), X, rtol=1e-3, atol=1.0)


def test_feature_scaler_handles_constant_feature():
    X = np.column_stack([np.full(10, 7.0), np.arange(10.0)]).astype(np.float32)
    sc = FeatureScaler.fit(X)
    assert sc.sd[0] == 1.0  # constant feature -> unit scale, no div-by-zero
    assert np.allclose(sc.inverse(sc.transform(X)), X, atol=1e-4)


def make_large_scale_buffer(n=80, seed=0, label="DoS"):
    """Buffer whose features span ~1 to ~1e6, like raw CICIDS2017 flows."""
    rng = np.random.default_rng(seed)
    keys = ["small", "mid", "huge"]
    rows = np.column_stack([
        rng.normal(1.0, 0.5, n),
        rng.normal(1e3, 2e2, n),
        rng.normal(1e6, 3e5, n),
    ])
    return [({k: float(v) for k, v in zip(keys, row, strict=True)}, label) for row in rows]


@pytest.mark.parametrize(
    "make_aug",
    [
        lambda: VanillaGanAugmenter(noise_dim=8, epochs=25, device="cpu", seed=0),
        lambda: ConditionalGanAugmenter(noise_dim=8, epochs=25, n_critic=1, device="cpu", seed=0),
    ],
    ids=["vanilla", "cgan"],
)
def test_gan_recovers_feature_scale(make_aug):
    buffer = make_large_scale_buffer()
    real = np.array([s["huge"] for s, _ in buffer])
    out = make_aug().fit_generate(buffer, 150)
    gen = np.array([s["huge"] for s in out])
    # Without normalization the generator collapsed to ~O(1) regardless of the
    # 1e6-scale target; with it, samples are produced at the real scale.
    assert abs(float(gen.mean())) > 1e4
    assert float(gen.std()) > 0.02 * float(real.std())
