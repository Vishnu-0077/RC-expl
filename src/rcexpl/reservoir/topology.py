"""Recurrent weight-matrix topologies for reservoirs.

Each builder returns an unscaled ``(N, N)`` matrix. Call
:func:`scale_spectral_radius` to set its spectral radius.

Topologies
----------
dense         fully connected, uniform weights (Jaeger 2001)
erdos_renyi   sparse random graph with a given connection density
small_world   Watts-Strogatz ring lattice with random rewiring
ring          simple cycle reservoir, SCR (Rodan & Tino 2011)
cycle_jumps   cycle reservoir with regular jumps, CRJ (Rodan & Tino 2012)
delay_line    feed-forward chain, i.e. a pure shift register
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

__all__ = ["build_topology", "scale_spectral_radius", "spectral_radius", "TOPOLOGIES"]


def _uniform(rng, shape):
    return rng.uniform(-1.0, 1.0, shape)


def dense(n, rng, **_):
    return _uniform(rng, (n, n))


def erdos_renyi(n, rng, density=0.05, **_):
    mask = rng.random((n, n)) < density
    return np.where(mask, _uniform(rng, (n, n)), 0.0)


def small_world(n, rng, k=6, p_rewire=0.1, **_):
    W = np.zeros((n, n))
    half = max(1, k // 2)
    for i in range(n):
        for j in range(1, half + 1):
            tgt = (i + j) % n
            if rng.random() < p_rewire:
                tgt = int(rng.integers(n))
                while tgt == i:
                    tgt = int(rng.integers(n))
            W[tgt, i] = rng.uniform(-1, 1)
            W[i, tgt] = rng.uniform(-1, 1)
    return W


def ring(n, rng, **_):
    W = np.zeros((n, n))
    idx = np.arange(n)
    W[(idx + 1) % n, idx] = 1.0
    return W


def cycle_jumps(n, rng, jump=None, jump_weight=0.5, **_):
    W = ring(n, rng)
    jump = jump or max(2, int(round(np.sqrt(n))))
    for i in range(0, n - jump + 1, jump):
        j = (i + jump) % n
        W[i, j] = W[j, i] = jump_weight
    return W


def delay_line(n, rng, **_):
    W = np.zeros((n, n))
    idx = np.arange(n - 1)
    W[idx + 1, idx] = 1.0
    return W


TOPOLOGIES = {
    "dense": dense,
    "erdos_renyi": erdos_renyi,
    "small_world": small_world,
    "ring": ring,
    "cycle_jumps": cycle_jumps,
    "delay_line": delay_line,
}


def spectral_radius(W) -> float:
    if sparse.issparse(W):
        W = W.toarray()
    return float(np.max(np.abs(np.linalg.eigvals(W)))) if W.shape[0] <= 1500 else float(
        np.abs(eigs(sparse.csr_matrix(W), k=1, which="LM", return_eigenvectors=False))[0])


def scale_spectral_radius(W, rho: float):
    """Rescale ``W`` to spectral radius ``rho``.

    A nilpotent matrix (spectral radius 0, as for ``delay_line``) is
    rescaled by its largest singular value instead, so ``rho`` still sets
    the gain of the connections.
    """
    r = spectral_radius(W)
    if r < 1e-10:
        r = float(np.linalg.norm(W, 2)) or 1.0
    return W * (rho / r)


def build_topology(name: str, n: int, rng: np.random.Generator, **kwargs) -> np.ndarray:
    try:
        return TOPOLOGIES[name](n, rng, **kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown topology {name!r}; choose from {sorted(TOPOLOGIES)}") from exc
