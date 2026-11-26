"""Generators for chaotic benchmark systems used in reservoir-computing studies.

Every generator returns a :class:`TimeSeries` holding the trajectory as an
array of shape ``(T, d)`` plus metadata: the sampling step ``dt`` and the
largest Lyapunov exponent, which we estimate numerically ourselves rather
than copy from the literature. Prediction horizons can then be given in
Lyapunov times, i.e. multiples of ``1 / lambda_max``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "TimeSeries",
    "logistic_map",
    "tent_map",
    "henon_map",
    "lorenz",
    "rossler",
    "mackey_glass",
    "narma",
    "load_system",
    "SYSTEMS",
]


@dataclass
class TimeSeries:
    """A sampled trajectory plus metadata."""

    name: str
    data: np.ndarray  # (T, d)
    dt: float = 1.0
    lyapunov: float | None = None  # largest Lyapunov exponent per unit time
    params: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return self.data.shape[1]

    def lyapunov_time_steps(self) -> float | None:
        """Number of samples in one Lyapunov time (1/lambda_max)."""
        if not self.lyapunov:
            return None
        return 1.0 / (self.lyapunov * self.dt)

    def __len__(self) -> int:
        return len(self.data)


# ----------------------------------------------------------------------------
# Discrete maps
# ----------------------------------------------------------------------------

def logistic_map(n: int = 6000, r: float = 3.95, x0: float = 0.5422, discard: int = 500) -> TimeSeries:
    x = np.empty(n + discard)
    x[0] = x0
    for t in range(n + discard - 1):
        x[t + 1] = r * x[t] * (1.0 - x[t])
    x = x[discard:]
    lam = float(np.mean(np.log(np.abs(r * (1.0 - 2.0 * x)) + 1e-300)))
    return TimeSeries("logistic", x[:, None], 1.0, lam, {"r": r})


def tent_map(n: int = 6000, mu: float = 1.9999, x0: float = 0.5422, discard: int = 500) -> TimeSeries:
    x = np.empty(n + discard)
    x[0] = x0
    for t in range(n + discard - 1):
        x[t + 1] = mu * x[t] if x[t] < 0.5 else mu * (1.0 - x[t])
    return TimeSeries("tent", x[discard:, None], 1.0, float(np.log(mu)), {"mu": mu})


def henon_map(n: int = 6000, a: float = 1.4, b: float = 0.3, discard: int = 1000) -> TimeSeries:
    N = n + discard
    s = np.empty((N, 2))
    s[0] = (0.1, 0.3)
    v = np.array([1.0, 0.0])
    log_sum = 0.0
    for t in range(N - 1):
        x, y = s[t]
        s[t + 1] = (1.0 - a * x * x + y, b * x)
        J = np.array([[-2.0 * a * x, 1.0], [b, 0.0]])
        v = J @ v
        nv = np.linalg.norm(v)
        if t >= discard:
            log_sum += np.log(nv)
        v /= nv
    lam = log_sum / (N - 1 - discard)
    return TimeSeries("henon", s[discard:], 1.0, float(lam), {"a": a, "b": b})


# ----------------------------------------------------------------------------
# Continuous flows (fixed-step RK4 with sub-stepping, plus a Benettin
# estimate of the largest Lyapunov exponent using the analytic Jacobian)
# ----------------------------------------------------------------------------

def _rk4(f, s, h):
    k1 = f(s)
    k2 = f(s + 0.5 * h * k1)
    k3 = f(s + 0.5 * h * k2)
    k4 = f(s + h * k3)
    return s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _integrate_flow(f, jac, s0, n, dt, discard, substeps=10):
    h = dt / substeps
    s = np.asarray(s0, float)
    d = s.size
    out = np.empty((n, d))
    v = np.ones(d) / np.sqrt(d)

    def tangent(z):
        x, w = z[:d], z[d:]
        return np.concatenate([f(x), jac(x) @ w])

    log_sum, count = 0.0, 0
    for t in range(n + discard):
        for _ in range(substeps):
            z = _rk4(tangent, np.concatenate([s, v]), h)
            s, v = z[:d], z[d:]
        nv = np.linalg.norm(v)
        v /= nv
        if t >= discard:
            out[t - discard] = s
            log_sum += np.log(nv)
            count += 1
    return out, log_sum / (count * dt)


def lorenz(n: int = 12000, dt: float = 0.02, sigma: float = 10.0, rho: float = 28.0,
           beta: float = 8.0 / 3.0, s0=(1.0, 1.0, 1.0), discard: int = 1000) -> TimeSeries:
    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    def jac(s):
        x, y, z = s
        return np.array([[-sigma, sigma, 0.0], [rho - z, -1.0, -x], [y, x, -beta]])

    data, lam = _integrate_flow(f, jac, s0, n, dt, discard)
    return TimeSeries("lorenz", data, dt, float(lam), {"sigma": sigma, "rho": rho, "beta": beta})


def rossler(n: int = 12000, dt: float = 0.1, a: float = 0.2, b: float = 0.2, c: float = 5.7,
            s0=(1.0, 1.0, 1.0), discard: int = 1000) -> TimeSeries:
    def f(s):
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    def jac(s):
        x, y, z = s
        return np.array([[0.0, -1.0, -1.0], [1.0, a, 0.0], [z, 0.0, x - c]])

    data, lam = _integrate_flow(f, jac, s0, n, dt, discard)
    return TimeSeries("rossler", data, dt, float(lam), {"a": a, "b": b, "c": c})


def mackey_glass(n: int = 6000, tau: float = 17.0, beta: float = 0.2, gamma: float = 0.1,
                 power: int = 10, sample_dt: float = 1.0, h: float = 0.1, discard: int = 1000,
                 x0: float = 1.2) -> TimeSeries:
    """Mackey-Glass delay differential equation (Heun's method, step ``h``).

    The delayed term is read from the stored history, so ``tau / h`` must be
    an integer.
    """
    lag = int(round(tau / h))
    every = int(round(sample_dt / h))
    total = (n + discard) * every
    x = np.empty(total + lag + 1)
    x[: lag + 1] = x0
    for i in range(lag, total + lag):
        xd, xd1 = x[i - lag], x[i - lag + 1]
        f0 = beta * xd / (1 + xd ** power) - gamma * x[i]
        xp = x[i] + h * f0
        f1 = beta * xd1 / (1 + xd1 ** power) - gamma * xp
        x[i + 1] = x[i] + 0.5 * h * (f0 + f1)
    series = x[lag + 1 :: every][discard : discard + n]
    return TimeSeries("mackey_glass", series[:, None], sample_dt, None,
                      {"tau": tau, "beta": beta, "gamma": gamma, "n": power})


def narma(n: int = 6000, order: int = 10, seed: int = 0):
    """NARMA-``order`` input/output benchmark.

    Returns ``(u, y)``, both of shape ``(n, 1)``. The input ``u`` is i.i.d.
    uniform on ``[0, 0.5]``.
    """
    rng = np.random.default_rng(seed)
    while True:  # NARMA can diverge for unlucky input draws; resample if so
        u = rng.uniform(0.0, 0.5, n + 200)
        y = np.zeros_like(u)
        for t in range(order, len(u) - 1):
            y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - order + 1 : t + 1])
                        + 1.5 * u[t - order + 1] * u[t] + 0.1)
        if np.all(np.isfinite(y)) and np.max(np.abs(y)) < 10:
            break
    return u[200:, None], y[200:, None]


SYSTEMS = {
    "logistic": logistic_map,
    "tent": tent_map,
    "henon": henon_map,
    "mackey_glass": mackey_glass,
    "lorenz": lorenz,
    "rossler": rossler,
}


def load_system(name: str, **kwargs) -> TimeSeries:
    try:
        return SYSTEMS[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown system {name!r}; choose from {sorted(SYSTEMS)}") from exc
