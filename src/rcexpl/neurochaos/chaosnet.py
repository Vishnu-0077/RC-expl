"""ChaosNet classifier and ChaosFEX + ML pipelines.

ChaosNet (Balakrishnan, Kathpalia, Saha & Nagaraj, Chaos 29, 2019) keeps
one *mean representation vector* per class, the average ChaosFEX feature
vector of the training samples of that class. It predicts the class whose
representation has the highest cosine similarity with the test sample.
Training is a single pass with no gradient descent. Its behaviour with very
few samples is tested in experiment NL-02.

The "ChaosFEX + classical ML" pipelines (Harikrishnan & Nagaraj) feed the
same features to any scikit-learn estimator.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .chaosfex import ChaosFEX

__all__ = ["CosineMeanClassifier", "ChaosNet", "chaosfex_pipeline"]


class CosineMeanClassifier(BaseEstimator, ClassifierMixin):
    """Nearest class-mean classifier under cosine similarity.

    If ``standardize`` is true, each feature is z-scored using training
    statistics before the class means are computed. Without this, large
    features such as firing time and energy dominate the cosine, so this
    option is an improvement over the original ChaosNet.
    """

    def __init__(self, standardize=False):
        self.standardize = standardize

    def _prep(self, X):
        X = np.asarray(X, float)
        if self.standardize:
            X = (X - self.mu_) / self.sd_
        return X

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        if self.standardize:
            self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        X = self._prep(X)
        self.classes_ = np.unique(y)
        M = np.stack([X[y == c].mean(axis=0) for c in self.classes_])
        self.means_ = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        return self

    def decision_function(self, X):
        check_is_fitted(self, "means_")
        X = self._prep(X)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return X @ self.means_.T

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class ChaosNet(Pipeline):
    """ChaosFEX feature extraction followed by cosine-similarity class means, as one estimator."""

    def __init__(self, q=0.34, b=0.499, epsilon=0.1, max_iter=10000, chaotic_map="skew_tent",
                 features=("firing_time", "firing_rate", "energy", "entropy"), standardize=False):
        self.q, self.b, self.epsilon, self.max_iter = q, b, epsilon, max_iter
        self.chaotic_map, self.features, self.standardize = chaotic_map, features, standardize
        super().__init__([
            ("chaosfex", ChaosFEX(q=q, b=b, epsilon=epsilon, max_iter=max_iter,
                                  chaotic_map=chaotic_map, features=features)),
            ("cosine", CosineMeanClassifier(standardize=standardize)),
        ])

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in
                ["q", "b", "epsilon", "max_iter", "chaotic_map", "features", "standardize"]}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        self.__init__(**self.get_params())
        return self


def chaosfex_pipeline(estimator, scale=True, **chaosfex_kwargs) -> Pipeline:
    """Build ``ChaosFEX -> [StandardScaler] -> estimator``."""
    steps = [("chaosfex", ChaosFEX(**chaosfex_kwargs))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", clone(estimator)))
    return Pipeline(steps)
