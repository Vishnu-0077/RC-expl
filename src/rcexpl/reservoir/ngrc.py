"""Next-Generation Reservoir Computing (NG-RC), after Gauthier et al., Nat. Commun. 12, 5564 (2021).

NG-RC replaces the random recurrent reservoir with an explicit feature
vector made of

* ``k`` time-delayed copies of the input, spaced ``s`` samples apart
  (the linear part), and
* all unique monomials of the linear part of degree 2 to ``degree``
  (the nonlinear part),

plus a constant. A ridge readout predicts the one-step increment
``y(t) - u(t)``, or the next value directly if ``predict_increment=False``.
"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

from .esn import ridge_regression

__all__ = ["NGRC"]


class NGRC:
    def __init__(self, k=2, s=1, degree=2, ridge=1e-6, predict_increment=True, washout=0):
        self.k = int(k)
        self.s = int(s)
        self.degree = int(degree)
        self.ridge = float(ridge)
        self.predict_increment = predict_increment
        self.washout = int(washout)

    def get_params(self):
        return {k: getattr(self, k) for k in ["k", "s", "degree", "ridge", "predict_increment", "washout"]}

    @property
    def span(self) -> int:
        """Number of past samples needed to form one feature vector."""
        return (self.k - 1) * self.s + 1

    def _monomial_index(self, n_lin):
        idx = []
        for p in range(2, self.degree + 1):
            idx.extend(combinations_with_replacement(range(n_lin), p))
        return idx

    def _linear(self, U):
        """Delay-embedded linear features for every t with full history: shape (T - span + 1, k*d)."""
        T = len(U)
        cols = [U[(self.k - 1 - j) * self.s: T - j * self.s] for j in range(self.k)]
        return np.hstack(cols)

    def _nonlinear(self, L):
        out = [np.ones((len(L), 1)), L]
        for combo in self._mono_:
            out.append(np.prod(L[:, combo], axis=1, keepdims=True))
        return np.hstack(out)

    def features(self, U):
        U = np.asarray(U, float).reshape(len(U), -1)
        if not hasattr(self, "_mono_"):
            self._mono_ = self._monomial_index(self.k * U.shape[1])
        return self._nonlinear(self._linear(U))

    @property
    def n_features_(self) -> int:
        return 1 + self.k * self.d_ + len(self._mono_)

    def fit(self, U, Y):
        U = np.asarray(U, float).reshape(len(U), -1)
        Y = np.asarray(Y, float).reshape(len(Y), -1)
        self.d_ = U.shape[1]
        self._mono_ = self._monomial_index(self.k * self.d_)
        F = self.features(U)
        target = Y[self.span - 1:]
        if self.predict_increment:
            target = target - U[self.span - 1:]
        F, target = F[self.washout:], target[self.washout:]
        self.W_out_ = ridge_regression(F, target, self.ridge)
        self.history_ = U[-self.span:].copy()
        self.n_outputs_ = Y.shape[1]
        return self

    def _one(self, hist):
        lin = np.concatenate([hist[-1 - j * self.s] for j in range(self.k)])
        phi = np.concatenate([[1.0], lin, [np.prod(lin[list(c)]) for c in self._mono_]])
        y = phi @ self.W_out_
        return hist[-1] + y if self.predict_increment else y

    def predict(self, U, warmup=None):
        """Teacher-forced one-step predictions. The history comes from ``warmup`` or the end of training."""
        U = np.asarray(U, float).reshape(len(U), -1)
        hist = self.history_ if warmup is None else np.asarray(warmup, float).reshape(len(warmup), -1)
        full = np.vstack([hist[-(self.span - 1):] if self.span > 1 else hist[:0], U])
        F = self.features(full)
        y = F @ self.W_out_
        if self.predict_increment:
            y = y + full[self.span - 1:]
        self.history_ = full[-self.span:].copy()
        return y

    def generate(self, n_steps: int, warmup=None):
        hist = self.history_ if warmup is None else np.asarray(warmup, float).reshape(len(warmup), -1)
        if len(hist) < self.span:
            raise ValueError(f"warmup needs at least {self.span} samples")
        hist = list(hist[-self.span:])
        out = np.empty((n_steps, self.n_outputs_))
        for t in range(n_steps):
            y = self._one(hist)
            out[t] = y
            hist.append(y)
            hist.pop(0)
        return out
