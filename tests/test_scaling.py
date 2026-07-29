"""Feature standardization used by every covariate-shift monitor."""

import numpy as np

from driftgan.early.scaling import FeatureScaler


def test_feature_scaler_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(500, 120, (64, 5)).astype(np.float32)
    sc = FeatureScaler.fit(X)
    Xn = sc.transform(X)
    assert abs(float(Xn.mean())) < 1e-2 and abs(float(Xn.std()) - 1.0) < 0.2
    assert np.allclose(sc.inverse(Xn), X, rtol=1e-3, atol=1.0)


def test_feature_scaler_handles_constant_feature():
    """Flow data is full of always-zero columns; they must not divide by zero."""
    X = np.column_stack([np.full(10, 7.0), np.arange(10.0)]).astype(np.float32)
    sc = FeatureScaler.fit(X)
    assert sc.sd[0] == 1.0  # constant feature -> unit scale
    assert np.allclose(sc.inverse(sc.transform(X)), X, atol=1e-4)
