"""Task-independent diagnostics of reservoir dynamics.

* Linear memory capacity, MC (Jaeger 2002): how well the input ``k`` steps
  ago can be linearly reconstructed from the current state.
* Quadratic information-processing capacity (Dambre et al. 2012): the same
  question for the second-order Legendre targets ``P2(u(t-k))`` and the
  cross terms ``u(t-i) u(t-j)``.
* Effective dimension: the participation ratio of the eigenvalues of the
  state covariance matrix.
* Echo-state-property (fading memory) check: two copies started from
  different random states and driven by the same input should converge
  onto each other.
"""
from __future__ import annotations

import numpy as np

from .esn import ESN, ridge_regression

__all__ = ["memory_capacity", "quadratic_capacity", "effective_dimension", "esp_convergence",
           "capacity_profile"]


def _input(T, seed):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, (T, 1))


def _capacity(X_tr, X_te, y_tr, y_te, ridge):
    F_tr = np.hstack([np.ones((len(X_tr), 1)), X_tr])
    F_te = np.hstack([np.ones((len(X_te), 1)), X_te])
    w = ridge_regression(F_tr, y_tr[:, None], ridge)
    pred = (F_te @ w).ravel()
    c = np.corrcoef(pred, y_te)[0, 1] ** 2
    return 0.0 if not np.isfinite(c) else float(c)


def capacity_profile(esn: ESN, targets_fn, T=6000, washout=500, ridge=1e-8, seed=1):
    """Generic capacity evaluation.

    ``targets_fn(u)`` must return a list of target vectors aligned with ``u``,
    with NaN marking the samples where a target is undefined.
    """
    u = _input(T, seed)
    X = esn.transform(u)
    caps = []
    for y in targets_fn(u[:, 0]):
        valid = np.arange(T) >= washout
        valid &= np.isfinite(y)
        idx = np.flatnonzero(valid)
        half = len(idx) // 2
        tr, te = idx[:half], idx[half:]
        caps.append(_capacity(X[tr], X[te], y[tr], y[te], ridge))
    return np.array(caps)


def _shift(u, k):
    y = np.full_like(u, np.nan)
    y[k:] = u[: len(u) - k] if k > 0 else u
    return y


def memory_capacity(esn: ESN, max_delay=100, **kw):
    """Return the array ``MC_k`` for ``k = 1..max_delay``. Its sum is the total memory capacity."""
    return capacity_profile(esn, lambda u: [_shift(u, k) for k in range(1, max_delay + 1)], **kw)


def quadratic_capacity(esn: ESN, max_delay=20, max_cross=8, threshold=0.01, **kw):
    """Total degree-2 capacity: Legendre ``P2(u(t-k))`` plus cross products ``u(t-i) u(t-j)``.

    Individual capacities below ``threshold`` count as zero, which suppresses
    the positive bias that finite samples give to uncorrelated targets.
    """
    def targets(u):
        ts = [_shift(1.5 * u ** 2 - 0.5, k) for k in range(1, max_delay + 1)]
        for i in range(1, max_cross + 1):
            for j in range(i + 1, max_cross + 1):
                ts.append(_shift(u, i) * _shift(u, j))
        return ts

    caps = capacity_profile(esn, targets, **kw)
    caps[caps < threshold] = 0.0
    return float(caps.sum())


def effective_dimension(states: np.ndarray) -> float:
    """Participation ratio ``(sum l)^2 / sum l^2`` of the covariance eigenvalues ``l``."""
    lam = np.clip(np.linalg.eigvalsh(np.cov(states, rowvar=False)), 0, None)
    return float(lam.sum() ** 2 / (np.sum(lam ** 2) + 1e-300))


def esp_convergence(esn: ESN, T=500, seed=0):
    """Distance between two reservoir copies with different initial states but the same input."""
    rng = np.random.default_rng(seed)
    u = _input(T, seed + 7)
    esn._build(1)
    s1 = [rng.uniform(-1, 1, esn.n_units) for _ in range(esn.depth)]
    s2 = [rng.uniform(-1, 1, esn.n_units) for _ in range(esn.depth)]
    X1, _ = esn.run(u, s1)
    X2, _ = esn.run(u, s2)
    return np.linalg.norm(X1 - X2, axis=1)
