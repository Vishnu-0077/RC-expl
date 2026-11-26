"""1-D chaotic maps used as neurons.

The Generalised Lüroth Series (GLS) maps are the skew-tent and skew-binary
maps. Both are piecewise linear, preserve the Lebesgue measure on [0, 1]
and are topologically transitive for any skew ``0 < b < 1``. Their
Lyapunov exponent equals their Shannon entropy,
``-b ln b - (1-b) ln(1-b)``, which peaks at ``ln 2`` when ``b = 0.5``.

Floating-point caveat: with ``b = 0.5`` exactly, each iteration of these
maps discards one mantissa bit, so every orbit collapses to a fixed point
within about 53 steps. Use a skew such as ``b = 0.499`` instead.
"""
from __future__ import annotations

import numpy as np

__all__ = ["skew_tent", "skew_binary", "logistic", "MAPS", "SCALAR_MAPS", "gls_lyapunov"]


def skew_tent(x, b):
    x = np.asarray(x, float)
    return np.where(x < b, x / b, (1.0 - x) / (1.0 - b))


def skew_binary(x, b):
    x = np.asarray(x, float)
    return np.where(x < b, x / b, (x - b) / (1.0 - b))


def logistic(x, b=None):
    """Fully chaotic logistic map ``4x(1-x)``. The skew ``b`` only sets the symbolic threshold."""
    x = np.asarray(x, float)
    return 4.0 * x * (1.0 - x)


MAPS = {"skew_tent": skew_tent, "skew_binary": skew_binary, "logistic": logistic}

# Scalar (pure-Python float) versions for iterating a single orbit. They give
# bit-identical IEEE-754 results to the vectorised versions, without NumPy's
# per-call overhead.
SCALAR_MAPS = {
    "skew_tent": lambda x, b: x / b if x < b else (1.0 - x) / (1.0 - b),
    "skew_binary": lambda x, b: x / b if x < b else (x - b) / (1.0 - b),
    "logistic": lambda x, b: 4.0 * x * (1.0 - x),
}


def gls_lyapunov(b):
    b = np.asarray(b, float)
    return -b * np.log(b) - (1 - b) * np.log(1 - b)
