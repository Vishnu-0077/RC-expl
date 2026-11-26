"""Classification datasets for the neurochaos-learning experiments.

The real-world datasets ship with scikit-learn, so no download is needed.
The synthetic ones test nonlinear decision boundaries.
"""
from __future__ import annotations

import numpy as np
from sklearn import datasets as skd

__all__ = ["load_classification", "CLASSIFICATION_DATASETS"]


def _xor(n=600, noise=0.15, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X += rng.normal(0, noise, X.shape)
    return X, y


def _spirals(n=600, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    m = n // 2
    t = np.sqrt(rng.uniform(0, 1, m)) * 3 * np.pi
    a = np.c_[t * np.cos(t), t * np.sin(t)]
    b = -a
    X = np.vstack([a, b]) + rng.normal(0, noise, (2 * m, 2))
    y = np.r_[np.zeros(m, int), np.ones(m, int)]
    return X, y


CLASSIFICATION_DATASETS = {
    "iris": lambda: skd.load_iris(return_X_y=True),
    "wine": lambda: skd.load_wine(return_X_y=True),
    "breast_cancer": lambda: skd.load_breast_cancer(return_X_y=True),
    "digits": lambda: skd.load_digits(return_X_y=True),
    "moons": lambda: skd.make_moons(600, noise=0.25, random_state=0),
    "circles": lambda: skd.make_circles(600, noise=0.1, factor=0.5, random_state=0),
    "xor": _xor,
    "spirals": _spirals,
}


def load_classification(name: str):
    try:
        X, y = CLASSIFICATION_DATASETS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(CLASSIFICATION_DATASETS)}") from exc
    return np.asarray(X, float), np.asarray(y)
