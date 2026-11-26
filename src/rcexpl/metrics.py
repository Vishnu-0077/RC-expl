"""Evaluation metrics for chaotic time-series forecasting."""
from __future__ import annotations

import numpy as np

__all__ = [
    "nrmse",
    "valid_prediction_time",
    "power_spectrum",
    "spectrum_distance",
    "local_maxima",
]


def nrmse(y_true, y_pred) -> float:
    """RMSE normalised by the standard deviation of the target, averaged over dimensions."""
    y_true = np.atleast_2d(np.asarray(y_true, float).T).T
    y_pred = np.atleast_2d(np.asarray(y_pred, float).T).T
    err = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    std = np.std(y_true, axis=0) + 1e-12
    return float(np.mean(err / std))


def valid_prediction_time(y_true, y_pred, threshold: float = 0.4, scale=None) -> int:
    """Number of steps before the normalised error first exceeds ``threshold``.

    Uses the criterion of Pathak et al. (2018):
    ``||y - y_hat|| / sqrt(<||y - <y>||^2>) > threshold``.
    ``scale`` overrides the denominator; pass the attractor's spread
    computed on training data when the test window is short.
    """
    y_true = np.asarray(y_true, float).reshape(len(y_true), -1)
    y_pred = np.asarray(y_pred, float).reshape(len(y_pred), -1)
    if scale is None:
        scale = np.sqrt(np.mean(np.sum((y_true - y_true.mean(0)) ** 2, axis=1)))
    with np.errstate(all="ignore"):  # diverged forecasts may contain inf/NaN
        err = np.linalg.norm(y_true - y_pred, axis=1) / (scale + 1e-12)
    bad = np.flatnonzero(~(err <= threshold))  # also catches NaN/inf
    return int(bad[0]) if bad.size else len(err)


def power_spectrum(x, dt: float = 1.0):
    """One-sided power spectral density of a 1-D signal (Hann-windowed periodogram)."""
    x = np.asarray(x, float).ravel()
    x = x - x.mean()
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    psd = (np.abs(X) ** 2) / np.sum(w ** 2)
    freqs = np.fft.rfftfreq(len(x), dt)
    return freqs, psd


def spectrum_distance(x_true, x_pred, dt: float = 1.0) -> float:
    """Mean absolute difference between the log power spectra of two signals."""
    _, p1 = power_spectrum(x_true, dt)
    _, p2 = power_spectrum(x_pred, dt)
    return float(np.mean(np.abs(np.log10(p1 + 1e-12) - np.log10(p2 + 1e-12))))


def local_maxima(x):
    """Values of the strict local maxima of a 1-D signal, e.g. for the Lorenz return map."""
    x = np.asarray(x).ravel()
    idx = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])) + 1
    return x[idx]
