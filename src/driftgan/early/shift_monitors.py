"""Cheap covariate-shift monitors — ablations of the autoencoder monitor.

The autoencoder in :mod:`driftgan.early.reconstruction_monitor` is the novelty
of the early-detection phase, so a reviewer will ask whether a *generative*
model is doing anything a cheap statistic cannot. These monitors answer that:
they all share the AE's exact protocol — a frozen reference frame set on the
first check, then, on every later check, **score before updating the reference**
so a distribution change shows up as a leading spike — and differ only in the
shift statistic:

- ``PCAReconstructionMonitor``  linear reconstruction error (PCA via SVD)
- ``KSMonitor``                 mean per-feature Kolmogorov–Smirnov distance
- ``MMDMonitor``               RBF maximum mean discrepancy (kernel two-sample)
- ``MeanShiftMonitor``         L2 shift of the standardized feature mean (floor)
- ``HDDDMMonitor``             mean per-feature Hellinger distance (published)
- ``D3Monitor``                discriminative old-vs-new AUC (published)
- ``D3NativeMonitor``          D3 with its own window roll and decision rule

Each exposes the same interface as the AE monitor — ``observe(x, index) ->
float | None`` and an ``error_history`` list of ``(index, score)`` — so they
drop into the runner and the offline lead-time analysis unchanged.

Most monitors emit a continuous statistic and let the runner's recon
Page-Hinkley decide when to fire. A monitor that carries its own published
decision rule instead sets ``declares_drift = True`` and reports its verdict in
``fired``; see :class:`D3NativeMonitor`.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.stats import ks_2samp

from driftgan.augment.base import FeatureScaler


class ShiftMonitor:
    """Base monitor: buffering, cadence and the frozen-reference protocol.

    Subclasses implement three hooks operating on the standardized+clipped
    buffer ``Xn``: :meth:`_init_reference` (first check), :meth:`_score`
    (distance of the current buffer to the reference) and
    :meth:`_update_reference` (track the new normal, called *after* scoring).
    """

    #: When True the monitor decides drift itself (its published rule) and the
    #: runner reads :attr:`fired` instead of thresholding the score with the
    #: recon Page-Hinkley.
    declares_drift = False
    fired = False

    def __init__(
        self,
        check_interval: int = 1_000,
        buffer_size: int = 2_000,
        clip: float = 10.0,
        min_buffer: int = 32,
        seed: int = 0,
    ):
        self.check_interval = check_interval
        self.clip = clip
        self.min_buffer = min_buffer
        self.rng = np.random.default_rng(seed)
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._keys: list[str] | None = None
        self._scaler: FeatureScaler | None = None
        self._ref_ready = False
        self._n_seen = 0
        self.error_history: list[tuple[int, float]] = []

    # -- standardization (frozen after the first check) --------------------
    def _std(self, X: np.ndarray) -> np.ndarray:
        assert self._scaler is not None
        return np.clip(self._scaler.transform(X), -self.clip, self.clip)

    # -- public API --------------------------------------------------------
    def observe(self, x: dict[str, float], index: int) -> float | None:
        if self._keys is None:
            self._keys = list(x.keys())
        self._buffer.append(np.array([x.get(k, 0.0) for k in self._keys], dtype=np.float32))
        self._n_seen += 1
        if self._n_seen % self.check_interval != 0 or len(self._buffer) < self.min_buffer:
            return None
        return self._check(index)

    def _check(self, index: int) -> float:
        X = np.stack(self._buffer)
        if not self._ref_ready:
            self._scaler = FeatureScaler.fit(X)
            Xn = self._std(X)
            self._init_reference(Xn)
            self._ref_ready = True
            score = self._score(Xn)  # self-distance -> baseline ~0
        else:
            Xn = self._std(X)
            score = self._score(Xn)  # measure before adapting (leading signal)
            self._update_reference(Xn)  # then track the new normal
        self.error_history.append((index, score))
        return score

    # -- hooks -------------------------------------------------------------
    def _init_reference(self, Xn: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def _score(self, Xn: np.ndarray) -> float:  # pragma: no cover
        raise NotImplementedError

    def _update_reference(self, Xn: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError


class PCAReconstructionMonitor(ShiftMonitor):
    """Linear analogue of the AE: reconstruction error in a frozen PCA basis."""

    def __init__(self, latent: int = 16, **kw):
        super().__init__(**kw)
        self.latent = latent
        self._components: np.ndarray | None = None  # (k, d)

    def _fit(self, Xn: np.ndarray) -> None:
        # top-k right singular vectors of the centred buffer
        k = min(self.latent, min(Xn.shape) - 1)
        _, _, vt = np.linalg.svd(Xn - Xn.mean(0, keepdims=True), full_matrices=False)
        self._components = vt[:k]

    def _init_reference(self, Xn: np.ndarray) -> None:
        self._fit(Xn)

    def _score(self, Xn: np.ndarray) -> float:
        assert self._components is not None
        proj = Xn @ self._components.T @ self._components
        return float(np.mean((Xn - proj) ** 2))

    def _update_reference(self, Xn: np.ndarray) -> None:
        self._fit(Xn)


class KSMonitor(ShiftMonitor):
    """Mean per-feature two-sample Kolmogorov–Smirnov distance to the reference."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._ref: np.ndarray | None = None

    def _init_reference(self, Xn: np.ndarray) -> None:
        self._ref = Xn

    def _score(self, Xn: np.ndarray) -> float:
        assert self._ref is not None
        stats = [ks_2samp(self._ref[:, j], Xn[:, j]).statistic for j in range(Xn.shape[1])]
        return float(np.mean(stats))

    def _update_reference(self, Xn: np.ndarray) -> None:
        self._ref = Xn


class MMDMonitor(ShiftMonitor):
    """RBF maximum mean discrepancy (unbiased MMD^2) between reference and buffer.

    Both sides are subsampled to ``sample`` rows to keep the O(n^2) kernel cheap;
    the RBF bandwidth is fixed by the median heuristic on the first reference.
    """

    def __init__(self, sample: int = 256, **kw):
        super().__init__(**kw)
        self.sample = sample
        self._ref: np.ndarray | None = None
        self._gamma: float = 1.0

    def _sub(self, Xn: np.ndarray) -> np.ndarray:
        if len(Xn) <= self.sample:
            return Xn
        idx = self.rng.choice(len(Xn), self.sample, replace=False)
        return Xn[idx]

    @staticmethod
    def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aa = (a * a).sum(1)[:, None]
        bb = (b * b).sum(1)[None, :]
        return np.maximum(aa + bb - 2 * a @ b.T, 0.0)

    def _init_reference(self, Xn: np.ndarray) -> None:
        self._ref = self._sub(Xn)
        d = self._sq_dists(self._ref, self._ref)
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        self._gamma = 1.0 / (med + 1e-12)

    def _mmd2(self, a: np.ndarray, b: np.ndarray) -> float:
        kaa = np.exp(-self._gamma * self._sq_dists(a, a))
        kbb = np.exp(-self._gamma * self._sq_dists(b, b))
        kab = np.exp(-self._gamma * self._sq_dists(a, b))
        na, nb = len(a), len(b)
        # unbiased: drop the diagonal on the within-sample terms
        t1 = (kaa.sum() - np.trace(kaa)) / (na * (na - 1))
        t2 = (kbb.sum() - np.trace(kbb)) / (nb * (nb - 1))
        t3 = 2 * kab.mean()
        return float(max(t1 + t2 - t3, 0.0))

    def _score(self, Xn: np.ndarray) -> float:
        assert self._ref is not None
        return self._mmd2(self._ref, self._sub(Xn))

    def _update_reference(self, Xn: np.ndarray) -> None:
        self._ref = self._sub(Xn)


class MeanShiftMonitor(ShiftMonitor):
    """Cheapest floor: L2 distance between standardized feature means."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._ref_mean: np.ndarray | None = None

    def _init_reference(self, Xn: np.ndarray) -> None:
        self._ref_mean = Xn.mean(0)

    def _score(self, Xn: np.ndarray) -> float:
        assert self._ref_mean is not None
        return float(np.linalg.norm(Xn.mean(0) - self._ref_mean))

    def _update_reference(self, Xn: np.ndarray) -> None:
        self._ref_mean = Xn.mean(0)


class HDDDMMonitor(ShiftMonitor):
    """Hellinger distance between per-feature histograms — HDDDM.

    Ditzler & Polikar, *Hellinger distance based drift detection for
    nonstationary environments*, IEEE CIDUE 2011. The second published
    unsupervised detector we compare against, and a useful contrast to
    :class:`D3Monitor`: it compares *distributions* bin by bin rather than
    asking whether a linear boundary separates them, so unlike a discriminative
    statistic it responds to a change in dispersion at a constant mean.

    The statistic is the mean over features of

        H_j = sqrt( sum_k ( sqrt(P_jk) - sqrt(Q_jk) )^2 )

    over ``bins`` equal-width bins spanning the union of both samples, with P
    and Q the normalized histograms of the reference and current buffers. As in
    the reference method the bin count defaults to ``sqrt(n)``. Thresholding is
    left to the runner's recon Page-Hinkley, exactly as for every other
    continuous monitor here, so the arms differ only in the statistic.
    """

    def __init__(self, bins: int | None = None, **kw):
        super().__init__(**kw)
        self.bins = bins
        self._ref: np.ndarray | None = None

    def _init_reference(self, Xn: np.ndarray) -> None:
        self._ref = Xn

    def _update_reference(self, Xn: np.ndarray) -> None:
        self._ref = Xn

    def _score(self, Xn: np.ndarray) -> float:
        assert self._ref is not None
        ref = self._ref
        n_bins = self.bins or max(2, int(np.sqrt(min(len(ref), len(Xn)))))
        total = 0.0
        for j in range(Xn.shape[1]):
            a, b = ref[:, j], Xn[:, j]
            lo = float(min(a.min(), b.min()))
            hi = float(max(a.max(), b.max()))
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            # Bin edges are built in float64 and checked for collisions rather
            # than delegated to np.histogram's `range=`, which derives them in
            # the *input* dtype: on float32 flow features a near-constant column
            # (always-zero flags and the like, of which there are many) produces
            # duplicate edges and np.histogram raises. Such a feature carries no
            # distributional information, so it contributes zero divergence.
            edges = np.linspace(lo, hi, n_bins + 1, dtype=np.float64)
            if np.any(edges[:-1] >= edges[1:]):
                continue
            p, _ = np.histogram(a, bins=edges)
            q, _ = np.histogram(b, bins=edges)
            p = p / p.sum()
            q = q / q.sum()
            total += float(np.sqrt(((np.sqrt(p) - np.sqrt(q)) ** 2).sum()))
        return total / Xn.shape[1]


class D3Monitor(ShiftMonitor):
    """Discriminative drift statistic — D3 (Gozuacik et al., CIKM 2019).

    D3 is the canonical *published* unsupervised drift detector: instead of
    modelling the data, it trains a **discriminative classifier** to separate an
    old window from a new one. If the two windows are separable (AUC above a
    threshold), the input distribution has changed. It is the natural external
    comparator for the reconstruction monitors, so it runs here under this
    module's shared protocol — reference frame = D3's "old window", current
    buffer = its "new window" — making the score directly comparable.

    Deviations from the reference implementation, both forced and documented in
    ``docs/paper/related-work.md``: logistic regression is fit by plain gradient
    descent (the project has no scikit-learn dependency) and the sliding-window
    bookkeeping is the shared protocol's rather than D3's own ``(w, rho)`` roll.
    The discriminative principle and the 2-fold stratified AUC estimate — the
    parts that define the method — are preserved.
    """

    def __init__(self, iters: int = 150, lr: float = 0.5, l2: float = 1e-3, **kw):
        super().__init__(**kw)
        self.iters = iters
        self.lr = lr
        self.l2 = l2
        self._ref: np.ndarray | None = None

    def _init_reference(self, Xn: np.ndarray) -> None:
        self._ref = Xn

    def _update_reference(self, Xn: np.ndarray) -> None:
        self._ref = Xn

    def _logreg_scores(self, Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
        """Decision scores from an L2-regularized logistic regression (batch GD)."""
        w = np.zeros(Xtr.shape[1])
        b = 0.0
        n = max(len(Xtr), 1)
        for _ in range(self.iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(Xtr @ w + b, -30, 30)))
            resid = p - ytr
            w -= self.lr * (Xtr.T @ resid / n + self.l2 * w)
            b -= self.lr * float(resid.mean())
        return Xte @ w + b

    @staticmethod
    def _auc(y: np.ndarray, s: np.ndarray) -> float:
        """Rank-based (Mann-Whitney) AUC with tie-averaged ranks."""
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos == 0 or n_neg == 0:
            return 0.5
        order = np.argsort(s, kind="mergesort")
        s_sorted = s[order]
        ranks_sorted = np.arange(1, len(s) + 1, dtype=float)
        i = 0
        while i < len(s):  # average ranks within tie groups
            j = i
            while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
                j += 1
            ranks_sorted[i : j + 1] = (i + 1 + j + 1) / 2.0
            i = j + 1
        ranks = np.empty(len(s), dtype=float)
        ranks[order] = ranks_sorted
        return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    def _auc_two_sample(self, old: np.ndarray, new: np.ndarray) -> float:
        """Cross-validated AUC of separating ``old`` (label 0) from ``new`` (1).

        This is D3's statistic proper; both the shared-protocol monitor and the
        native-window one call it so that only the *window bookkeeping* differs
        between them.
        """
        X = np.vstack([old, new])
        y = np.concatenate([np.zeros(len(old)), np.ones(len(new))])
        # 2-fold stratified CV, as in the reference implementation
        preds = np.zeros(len(y))
        folds = np.zeros(len(y), dtype=int)
        for cls in (0.0, 1.0):
            idx = np.where(y == cls)[0]
            folds[self.rng.permutation(idx)[: len(idx) // 2]] = 1
        for f in (0, 1):
            te = folds == f
            tr = ~te
            if tr.sum() < 2 or te.sum() < 1 or len(np.unique(y[tr])) < 2:
                return 0.5
            preds[te] = self._logreg_scores(X[tr], y[tr], X[te])
        return self._auc(y, preds)

    def _score(self, Xn: np.ndarray) -> float:
        """AUC of separating the reference window (0) from the current one (1)."""
        assert self._ref is not None
        return self._auc_two_sample(self._ref, Xn)


class D3NativeMonitor(D3Monitor):
    """D3 with its published window mechanics *and* its own decision rule.

    :class:`D3Monitor` runs D3's statistic inside this module's shared protocol,
    which makes it directly comparable with the other monitors but leaves a
    reviewer two openings: the window bookkeeping is ours rather than D3's, and
    the fire decision is delegated to our recon Page-Hinkley. This class closes
    both. It keeps D3's own window ``W`` of size ``w(1 + rho)``, tests it every
    ``w * rho`` samples by separating the old slice ``W[:w]`` from the new one
    ``W[w:]``, and declares drift itself when the AUC exceeds ``tau`` — after
    which the old slice is discarded, exactly as in the reference algorithm
    (Gozuacik et al., CIKM 2019; defaults ``w=100, rho=0.1, tau=0.7``). When no
    drift is declared the window simply slides by ``w * rho``.

    Two deviations remain, both numerical rather than algorithmic: logistic
    regression is fit by gradient descent (the project has no scikit-learn
    dependency) and features pass through the frozen per-feature standardizer
    shared by every monitor in this module, which that solver needs in order to
    converge. Standardization is a fixed affine map, so it does not change what
    is separable — only the conditioning of the fit.
    """

    declares_drift = True

    def __init__(self, w: int = 100, rho: float = 0.1, tau: float = 0.7, **kw):
        super().__init__(**kw)
        self.w = w
        self.rho = rho
        self.tau = tau
        self.fired = False
        self._new = max(1, int(round(w * rho)))
        # maxlen gives D3's non-drift roll for free: every append past a full
        # window evicts the oldest sample, so by the next test the window has
        # slid by exactly `_new`.
        self._window: deque[np.ndarray] = deque(maxlen=w + self._new)
        self._since_test = 0

    def observe(self, x: dict[str, float], index: int) -> float | None:
        if self._keys is None:
            self._keys = list(x.keys())
        self._window.append(np.array([x.get(k, 0.0) for k in self._keys], dtype=np.float32))
        self.fired = False
        if len(self._window) < self.w + self._new:
            return None  # window not full yet: D3 cannot test
        self._since_test += 1
        if self._since_test < self._new:
            return None
        self._since_test = 0

        X = np.stack(self._window)
        if self._scaler is None:
            self._scaler = FeatureScaler.fit(X)
        Xn = self._std(X)
        score = self._auc_two_sample(Xn[: self.w], Xn[self.w :])
        self.error_history.append((index, score))
        if score > self.tau:
            self.fired = True
            for _ in range(self.w):  # drift: drop the old slice, keep the new
                self._window.popleft()
        return score


MONITOR_TYPES = {
    "pca": PCAReconstructionMonitor,
    "ks": KSMonitor,
    "mmd": MMDMonitor,
    "mean": MeanShiftMonitor,
    "hdddm": HDDDMMonitor,
    "d3": D3Monitor,
    "d3_native": D3NativeMonitor,
}
