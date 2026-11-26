"""ChaosFEX: chaos-based feature extraction with GLS neurons.

Model
-----
Each input feature (normalised to [0, 1]) drives one chaotic neuron. The
neuron starts at its initial neural activity ``q`` and iterates a chaotic
map ``T`` until its state comes within ``epsilon`` of the stimulus ``x``:

    N(x) = min{ n >= 0 : |T^n(q) - x| < epsilon }      (firing time)

The features are computed on the neural trace ``A = (T^0(q), ..., T^N(q))``:

* firing time   ``N``
* firing rate   fraction of ``A`` above the threshold ``b``
* energy        ``sum A^2``
* entropy       Shannon entropy (bits) of the binary symbolic sequence ``A > b``
* variance      ``var(A)``  (an extra feature, not in the original ChaosFEX)

Fast exact algorithm (the main contribution of this module)
-----------------------------------------------------------
Every neuron of a population starts from the *same* ``q`` and iterates the
*same* deterministic map, so all neurons follow one shared trajectory and
differ only in when they stop. We therefore

1. simulate that trajectory once (``max_iter`` map evaluations),
2. build prefix sums of ``[A > b]``, ``A`` and ``A^2``, and
3. find each stimulus's first hitting time with a blocked, vectorised
   search that only keeps working on the stimuli not resolved yet.

Every feature is then an O(1) look-up in the prefix sums. The total cost
drops from ``O(M * N_avg)`` scalar Python map evaluations (``M`` = number
of neurons) to ``O(max_iter + M * N_avg / SIMD)``, and the result is
bit-for-bit identical to the naive simulation; ``tests/`` checks this
against :func:`chaosfex_reference`.

Multi-population extension
--------------------------
``q`` may be a list. Each value defines a separate neuron population with
its own trajectory, and the features of all populations are concatenated.
Different initial activities visit the stimulus space in different orders,
so this gives a richer, less aliased encoding.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .maps import SCALAR_MAPS

__all__ = ["ChaosFEX", "gls_trajectory", "firing_times", "chaosfex_reference", "FEATURES"]

FEATURES = ("firing_time", "firing_rate", "energy", "entropy", "variance")
DEFAULT_FEATURES = ("firing_time", "firing_rate", "energy", "entropy")


def gls_trajectory(q: float, b: float, max_iter: int, chaotic_map: str = "skew_tent") -> np.ndarray:
    """The trajectory ``T^0(q), ..., T^max_iter(q)``, length ``max_iter + 1``."""
    f = SCALAR_MAPS[chaotic_map]
    b = float(b)
    x = float(q)
    out = [0.0] * (max_iter + 1)
    for n in range(max_iter + 1):
        out[n] = x
        x = f(x, b)
    return np.array(out)


def firing_times(traj: np.ndarray, stimuli: np.ndarray, epsilon: float, block: int = 256) -> np.ndarray:
    """First index ``n`` with ``|traj[n] - x| < epsilon`` for each stimulus ``x``.

    A stimulus that never fires is assigned ``len(traj) - 1``, i.e. ``max_iter``.
    """
    x = np.asarray(stimuli, float).ravel()
    out = np.full(x.size, len(traj) - 1, dtype=np.int64)
    pending = np.arange(x.size)
    for start in range(0, len(traj), block):
        seg = traj[start:start + block]
        hit = np.abs(x[pending, None] - seg[None, :]) < epsilon
        fired = hit.any(axis=1)
        out[pending[fired]] = start + hit[fired].argmax(axis=1)
        pending = pending[~fired]
        if pending.size == 0:
            break
    return out


def _binary_entropy(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.where((p <= 1e-12) | (p >= 1 - 1e-12), 0.0, h)


def _features_from_traj(traj, N, b, features):
    above = np.concatenate([[0], np.cumsum(traj > b)])
    s1 = np.concatenate([[0.0], np.cumsum(traj)])
    s2 = np.concatenate([[0.0], np.cumsum(traj ** 2)])
    length = N + 1
    fr = above[length] / length
    mean = s1[length] / length
    cols = {
        "firing_time": N.astype(float),
        "firing_rate": fr,
        "energy": s2[length],
        "entropy": _binary_entropy(fr),
        "variance": np.maximum(s2[length] / length - mean ** 2, 0.0),
    }
    return np.stack([cols[f] for f in features], axis=-1)


def chaosfex_reference(X01, q, b, epsilon, max_iter, chaotic_map="skew_tent", features=DEFAULT_FEATURES):
    """Naive per-neuron simulation in plain Python loops. Used as ground truth in tests and speed benchmarks."""
    f = SCALAR_MAPS[chaotic_map]
    b = float(b)
    m, n = X01.shape
    out = np.zeros((m, n, len(features)))
    for i in range(m):
        for j in range(n):
            x = float(q)
            trace = [x]
            k = 0
            while abs(x - X01[i, j]) >= epsilon and k < max_iter:
                x = f(x, b)
                trace.append(x)
                k += 1
            A = np.array(trace)
            fr = np.mean(A > b)
            vals = {
                "firing_time": float(k),
                "firing_rate": fr,
                "energy": float(np.sum(A ** 2)),
                "entropy": float(_binary_entropy(np.array(fr))),
                "variance": float(np.var(A)),
            }
            out[i, j] = [vals[name] for name in features]
    return out.reshape(m, -1)


class ChaosFEX(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer that maps ``X`` of shape ``(m, n)`` to chaotic-neuron features.

    Parameters
    ----------
    q : float or sequence of float
        Initial neural activity. A sequence gives one neuron population per value.
    b : float
        Skew of the GLS map, also used as the symbolic threshold.
    epsilon : float
        Radius of the firing neighbourhood around the stimulus.
    max_iter : int
        Maximum number of map iterations per neuron.
    chaotic_map : {"skew_tent", "skew_binary", "logistic"}
        Map used by the neurons.
    features : tuple of str
        Any subset of :data:`FEATURES`.
    normalize : bool
        Min-max scale the inputs to [0, 1] using the training data, clipping
        test values. Leave on unless the data is already in [0, 1].
    input_noise : float
        Standard deviation of Gaussian noise added to the normalised stimuli,
        used in the stochastic-resonance study.
    random_state : int, optional
        Seed for ``input_noise``.
    """

    def __init__(self, q=0.34, b=0.499, epsilon=0.1, max_iter=10000, chaotic_map="skew_tent",
                 features=DEFAULT_FEATURES, normalize=True, input_noise=0.0, random_state=None):
        self.q = q
        self.b = b
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.chaotic_map = chaotic_map
        self.features = features
        self.normalize = normalize
        self.input_noise = input_noise
        self.random_state = random_state

    def _qs(self):
        return list(np.atleast_1d(self.q).astype(float))

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        if self.normalize:
            self.min_ = X.min(axis=0)
            rng = X.max(axis=0) - self.min_
            self.scale_ = np.where(rng > 0, rng, 1.0)
        bad = [f for f in self.features if f not in FEATURES]
        if bad:
            raise ValueError(f"unknown features {bad}; choose from {FEATURES}")
        self.trajectories_ = [gls_trajectory(q, self.b, self.max_iter, self.chaotic_map) for q in self._qs()]
        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        return self

    def _to_unit(self, X):
        X = np.asarray(X, float)
        if self.normalize:
            X = (X - self.min_) / self.scale_
        if self.input_noise > 0:
            X = X + self._rng.normal(0.0, self.input_noise, X.shape)
        return np.clip(X, 0.0, 1.0)

    def firing_time_matrix(self, X):
        """Firing times, shape ``(m, n, n_populations)``."""
        check_is_fitted(self, "trajectories_")
        U = self._to_unit(X)
        return np.stack([firing_times(t, U, self.epsilon).reshape(U.shape) for t in self.trajectories_], -1)

    def transform(self, X):
        check_is_fitted(self, "trajectories_")
        U = self._to_unit(X)
        m, n = U.shape
        blocks = []
        for traj in self.trajectories_:
            N = firing_times(traj, U, self.epsilon)
            blocks.append(_features_from_traj(traj, N, self.b, self.features).reshape(m, n, -1))
        return np.concatenate(blocks, axis=-1).reshape(m, -1)

    def get_feature_names_out(self, input_features=None):
        names = input_features or [f"x{j}" for j in range(self.n_features_in_)]
        return np.array([f"{x}|q={q:g}|{f}" for x in names for q in self._qs() for f in self.features])
