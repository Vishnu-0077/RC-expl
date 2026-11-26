"""Parallel random-search hyperparameter optimisation for reservoir models.

The objective is evaluated by walk-forward validation: the model is trained
on the training block, then several closed-loop forecasts are launched from
different start points inside a validation block that the model never saw
during training. The score is the mean valid prediction time (VPT), which
we maximise. This measures what we care about, autonomous forecasting,
rather than one-step error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from ..metrics import valid_prediction_time

__all__ = ["LogUniform", "Uniform", "Choice", "IntUniform", "random_search", "closed_loop_score"]


@dataclass
class LogUniform:
    low: float
    high: float

    def sample(self, rng):
        return float(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))


@dataclass
class Uniform:
    low: float
    high: float

    def sample(self, rng):
        return float(rng.uniform(self.low, self.high))


@dataclass
class IntUniform:
    low: int
    high: int

    def sample(self, rng):
        return int(rng.integers(self.low, self.high + 1))


@dataclass
class Choice:
    options: list

    def sample(self, rng):
        return self.options[int(rng.integers(len(self.options)))]


def closed_loop_score(model, train, val, n_windows=5, warmup=200, horizon=300, threshold=0.4,
                      scale=None):
    """Fit on ``train``, then return the mean VPT (in steps) over ``n_windows`` forecasts launched inside ``val``."""
    try:
        model.fit(train[:-1], train[1:])
    except Exception:
        return 0.0, []
    max_start = len(val) - warmup - horizon
    starts = np.linspace(0, max_start, n_windows).astype(int)
    vpts = []
    for s0 in starts:
        warm = val[s0: s0 + warmup]
        truth = val[s0 + warmup: s0 + warmup + horizon]
        with np.errstate(all="ignore"):
            pred = model.generate(horizon, warmup=warm)
        vpts.append(valid_prediction_time(truth, pred, threshold, scale))
    return float(np.mean(vpts)), vpts


def _trial(builder, params, train, val, score_kw):
    model = builder(**params)
    score, _ = closed_loop_score(model, train, val, **score_kw)
    return params, score


def random_search(builder, space: dict, train, val, n_trials=40, seed=0, n_jobs=-1, **score_kw):
    """Random search over ``space``, a dict mapping parameter names to samplers or constants.

    Returns ``(best_params, best_score, history)``.
    """
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n_trials):
        trials.append({k: (v.sample(rng) if hasattr(v, "sample") else v) for k, v in space.items()})
    results = Parallel(n_jobs=n_jobs)(delayed(_trial)(builder, p, train, val, score_kw) for p in trials)
    best = max(results, key=lambda r: r[1])
    return best[0], best[1], results
