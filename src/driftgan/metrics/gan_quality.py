"""Synthetic-sample quality metrics: multivariate energy distance and MMD.

Both compare a set of real samples against a set of synthetic samples in
feature space; lower is better. Reported per subtype and per drift event
for Phase C_cond (Improvement 5).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def _standardize_pair(real: np.ndarray, synth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize both sets with the real set's statistics so scale-heavy
    features do not dominate the distance."""
    mu = real.mean(axis=0)
    sd = real.std(axis=0)
    sd[sd == 0] = 1.0
    return (real - mu) / sd, (synth - mu) / sd


def energy_distance(real: np.ndarray, synth: np.ndarray, standardize: bool = True) -> float:
    """Multivariate energy distance (Szekely & Rizzo).

    ``E = 2*E||X-Y|| - E||X-X'|| - E||Y-Y'||`` with Euclidean norms.
    """
    if len(real) == 0 or len(synth) == 0:
        return float("nan")
    if standardize:
        real, synth = _standardize_pair(real, synth)
    d_xy = cdist(real, synth).mean()
    d_xx = cdist(real, real).mean()
    d_yy = cdist(synth, synth).mean()
    return float(2 * d_xy - d_xx - d_yy)


def mmd_rbf(real: np.ndarray, synth: np.ndarray, gamma: float | None = None, standardize: bool = True) -> float:
    """Squared MMD with an RBF kernel (biased V-statistic).

    ``gamma`` defaults to the median heuristic: ``1 / (2 * median^2)`` of
    pairwise distances over the pooled sample.
    """
    if len(real) == 0 or len(synth) == 0:
        return float("nan")
    if standardize:
        real, synth = _standardize_pair(real, synth)
    if gamma is None:
        pooled = np.concatenate([real, synth], axis=0)
        dists = cdist(pooled, pooled)
        med = np.median(dists[dists > 0]) if (dists > 0).any() else 1.0
        gamma = 1.0 / (2 * med**2)

    def k(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.exp(-gamma * cdist(a, b, "sqeuclidean")).mean())

    return k(real, real) + k(synth, synth) - 2 * k(real, synth)


def quality_by_subtype(
    real: np.ndarray,
    real_subtypes: list[str],
    synth: np.ndarray,
    synth_subtypes: list[str],
) -> dict[str, dict[str, float]]:
    """Energy distance and MMD per subtype present in both sets."""
    out: dict[str, dict[str, float]] = {}
    for subtype in sorted(set(real_subtypes) & set(synth_subtypes)):
        r = real[np.array(real_subtypes) == subtype]
        s = synth[np.array(synth_subtypes) == subtype]
        out[subtype] = {
            "energy_distance": energy_distance(r, s),
            "mmd_rbf": mmd_rbf(r, s),
            "n_real": int(len(r)),
            "n_synth": int(len(s)),
        }
    return out
