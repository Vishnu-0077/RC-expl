"""Echo State Networks: shallow, deep (stacked) and ensemble variants.

The leaky-integrator update for layer ``l`` is::

    x_l(t) = (1 - a) x_l(t-1) + a * tanh( W_l x_l(t-1) + W_in_l [1; z_l(t)] )

where ``z_1 = u(t)`` is the external input and ``z_l = x_{l-1}(t)`` for the
deeper layers (DeepESN, Gallicchio et al. 2017). The readout is linear in

    phi(t) = [1, u(t), x_1(t), ..., x_L(t)]            (readout="linear")
    phi(t) = [1, u(t), x(t), x(t)**2]                   (readout="quadratic")

The quadratic readout breaks the odd symmetry of ``tanh`` reservoirs
(Herteux & Raeth 2020; Pathak et al. 2018). It is trained by ridge
regression.

Autonomous ("closed-loop") forecasting feeds each prediction back as the
next input. :class:`ESNEnsemble` averages its members during closed-loop
generation and feeds that consensus back to all of them. This keeps the
members on a common trajectory and stops individual members from wandering
off the attractor.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .topology import build_topology, scale_spectral_radius

__all__ = ["ESN", "ESNEnsemble", "ridge_regression"]


def ridge_regression(F: np.ndarray, Y: np.ndarray, ridge: float) -> np.ndarray:
    """Solve ``min ||F W - Y||^2 + ridge ||W||^2`` for ``W``. Returns shape ``(n_features, n_out)``."""
    A = F.T @ F
    A[np.diag_indices_from(A)] += ridge
    try:
        return np.linalg.solve(A, F.T @ Y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, F.T @ Y, rcond=None)[0]


class ESN:
    """Leaky Echo State Network with an optional deep (stacked) reservoir.

    Parameters
    ----------
    n_units : int
        Reservoir size of each layer.
    spectral_radius : float
        Spectral radius of every recurrent matrix.
    leak_rate : float
        Leaky-integration rate ``a`` in (0, 1].
    input_scaling : float
        Input weights are drawn uniformly from ``[-input_scaling, input_scaling]``.
    bias_scaling : float
        Scale of the bias column of ``W_in``.
    topology : str
        Name of a topology in :data:`rcexpl.reservoir.topology.TOPOLOGIES`.
    topology_kwargs : dict, optional
        Extra keyword arguments for the topology builder, e.g. ``{"density": 0.05}``.
    depth : int
        Number of stacked reservoir layers.
    ridge : float
        Tikhonov regularisation of the readout.
    washout : int
        Initial transient steps discarded before training.
    readout : {"linear", "quadratic"}
        Readout feature map.
    state_noise : float
        Standard deviation of Gaussian noise added to the states during
        training only. This is a regulariser that makes closed-loop
        generation more robust.
    seed : int
        Seed of the random number generator.
    """

    def __init__(self, n_units=300, spectral_radius=0.9, leak_rate=0.5, input_scaling=0.5,
                 bias_scaling=0.2, topology="erdos_renyi", topology_kwargs=None, depth=1,
                 ridge=1e-6, washout=100, readout="linear", state_noise=0.0, seed=0):
        self.n_units = int(n_units)
        self.spectral_radius = float(spectral_radius)
        self.leak_rate = float(leak_rate)
        self.input_scaling = float(input_scaling)
        self.bias_scaling = float(bias_scaling)
        self.topology = topology
        self.topology_kwargs = dict(topology_kwargs or {})
        self.depth = int(depth)
        self.ridge = float(ridge)
        self.washout = int(washout)
        self.readout = readout
        self.state_noise = float(state_noise)
        self.seed = seed
        self._built_for = None

    # ------------------------------------------------------------------ build
    def get_params(self):
        keys = ["n_units", "spectral_radius", "leak_rate", "input_scaling", "bias_scaling",
                "topology", "topology_kwargs", "depth", "ridge", "washout", "readout",
                "state_noise", "seed"]
        return {k: getattr(self, k) for k in keys}

    def _build(self, n_in: int):
        if self._built_for == n_in:
            return
        rng = np.random.default_rng(self.seed)
        self.W_, self.W_in_ = [], []
        for layer in range(self.depth):
            W = build_topology(self.topology, self.n_units, rng, **self.topology_kwargs)
            W = scale_spectral_radius(W, self.spectral_radius)
            density = np.count_nonzero(W) / W.size
            self.W_.append(sparse.csr_matrix(W) if density < 0.1 else W)
            fan_in = n_in if layer == 0 else self.n_units
            W_in = np.empty((self.n_units, fan_in + 1))
            W_in[:, 0] = rng.uniform(-self.bias_scaling, self.bias_scaling, self.n_units)
            W_in[:, 1:] = rng.uniform(-self.input_scaling, self.input_scaling, (self.n_units, fan_in))
            self.W_in_.append(W_in)
        self._noise_rng = np.random.default_rng(None if self.seed is None else self.seed + 10_007)
        self._built_for = n_in
        self.n_inputs_ = n_in

    # --------------------------------------------------------------- dynamics
    def _zero_state(self):
        return [np.zeros(self.n_units) for _ in range(self.depth)]

    def _advance(self, states, u):
        a = self.leak_rate
        new = []
        z = u
        for W, W_in, x in zip(self.W_, self.W_in_, states):
            pre = W @ x + W_in[:, 0] + W_in[:, 1:] @ z
            x = (1.0 - a) * x + a * np.tanh(pre)
            new.append(x)
            z = x
        return new

    def run(self, U, states=None, noise: float = 0.0):
        """Drive the reservoir with ``U`` of shape ``(T, d)``.

        Returns the state history ``(T, depth * n_units)`` and the final states.
        """
        U = np.asarray(U, float).reshape(len(U), -1)
        self._build(U.shape[1])
        states = self._zero_state() if states is None else [s.copy() for s in states]
        X = np.empty((len(U), self.depth * self.n_units))
        for t, u in enumerate(U):
            states = self._advance(states, u)
            if noise > 0:
                states = [s + noise * self._noise_rng.standard_normal(s.shape) for s in states]
            X[t] = np.concatenate(states)
        return X, states

    def transform(self, U):
        """Reservoir states for ``U``, starting from the zero state."""
        return self.run(U)[0]

    def _features(self, U, X):
        U = U.reshape(len(U), -1)
        parts = [np.ones((len(U), 1)), U, X]
        if self.readout == "quadratic":
            parts.append(X ** 2)
        elif self.readout != "linear":
            raise ValueError(f"unknown readout {self.readout!r}")
        return np.hstack(parts)

    def _readout(self, states, u):
        x = np.concatenate(states)
        phi = np.concatenate([[1.0], u, x] + ([x ** 2] if self.readout == "quadratic" else []))
        return phi @ self.W_out_

    # ---------------------------------------------------------------- training
    def fit(self, U, Y):
        """Teacher-forced training: ``Y[t]`` is the target given inputs up to ``U[t]``."""
        U = np.asarray(U, float).reshape(len(U), -1)
        Y = np.asarray(Y, float).reshape(len(Y), -1)
        X, states = self.run(U, noise=self.state_noise)
        F = self._features(U, X)[self.washout:]
        self.W_out_ = ridge_regression(F, Y[self.washout:], self.ridge)
        self.state_ = states
        self.last_input_ = U[-1].copy()
        self.n_outputs_ = Y.shape[1]
        return self

    def predict(self, U, warmup=None):
        """One-step (teacher-forced) predictions for the inputs ``U``.

        If ``warmup`` is given, the reservoir first restarts from zero and is
        synchronised on it. Otherwise it continues from the end of training,
        so ``U`` should follow the training data directly.
        """
        U = np.asarray(U, float).reshape(len(U), -1)
        states = self._sync(warmup)
        X, self.state_ = self.run(U, states)
        return self._features(U, X) @ self.W_out_

    def _sync(self, warmup):
        if warmup is None:
            return self.state_
        warmup = np.asarray(warmup, float).reshape(len(warmup), -1)
        _, states = self.run(warmup)
        self.last_input_ = warmup[-1].copy()
        return states

    def generate(self, n_steps: int, warmup=None):
        """Autonomous closed-loop forecast of ``n_steps`` samples.

        Requires the input and output dimensions to match. With ``warmup``,
        the reservoir is first synchronised on the true trajectory and the
        forecast starts at the sample right after its last point.
        """
        states = self._sync(warmup)
        u = self.last_input_.copy()
        out = np.empty((n_steps, self.n_outputs_))
        for t in range(n_steps):
            y = self._readout(states, u)
            out[t] = y
            states = self._advance(states, y)
            u = y
        return out

    def lyapunov_exponent(self, warmup, n_steps=3000, transient=500, delta=1e-8, renorm=10,
                          seed=0) -> float:
        """Largest Lyapunov exponent of the trained closed-loop system, in units of 1/step.

        Uses the Benettin two-trajectory method on the joint state
        ``(reservoir states, fed-back output)``. A well-trained surrogate of a
        chaotic system should reproduce the true exponent.
        """
        rng = np.random.default_rng(seed)
        states = self._sync(warmup)
        u = self.last_input_.copy()

        def flat(st, uu):
            return np.concatenate(st + [uu])

        def unflat(v):
            st = [v[i * self.n_units:(i + 1) * self.n_units] for i in range(self.depth)]
            return st, v[self.depth * self.n_units:]

        def step(v):
            st, uu = unflat(v)
            y = self._readout(st, uu)
            return flat(self._advance(st, y), y)

        v = flat(states, u)
        for _ in range(transient):
            v = step(v)
        d0 = rng.standard_normal(v.shape)
        w = v + delta * d0 / np.linalg.norm(d0)
        log_sum, count = 0.0, 0
        for t in range(1, n_steps + 1):
            v, w = step(v), step(w)
            if t % renorm == 0:
                d = np.linalg.norm(w - v)
                if not np.isfinite(d) or d == 0:
                    return float("nan")
                log_sum += np.log(d / delta)
                count += renorm
                w = v + (w - v) * (delta / d)
        return log_sum / count


class ESNEnsemble:
    """Ensemble of ESNs with different random seeds.

    During closed-loop generation the members' outputs are averaged and the
    average (the consensus) is fed back to every member.
    """

    def __init__(self, n_members=5, seed=0, **esn_kwargs):
        self.n_members = n_members
        self.seed = seed
        self.esn_kwargs = esn_kwargs
        self.members = [ESN(seed=seed + 101 * i, **esn_kwargs) for i in range(n_members)]

    def get_params(self):
        return {"n_members": self.n_members, "seed": self.seed, **self.esn_kwargs}

    def fit(self, U, Y):
        for m in self.members:
            m.fit(U, Y)
        self.n_outputs_ = self.members[0].n_outputs_
        return self

    def predict(self, U, warmup=None):
        return np.mean([m.predict(U, warmup) for m in self.members], axis=0)

    def generate(self, n_steps: int, warmup=None):
        states = [m._sync(warmup) for m in self.members]
        u = self.members[0].last_input_.copy()
        out = np.empty((n_steps, self.n_outputs_))
        for t in range(n_steps):
            y = np.mean([m._readout(s, u) for m, s in zip(self.members, states)], axis=0)
            out[t] = y
            states = [m._advance(s, y) for m, s in zip(self.members, states)]
            u = y
        return out
