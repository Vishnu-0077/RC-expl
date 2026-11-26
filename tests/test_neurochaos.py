import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from rcexpl.datasets import load_classification
from rcexpl.neurochaos import (FEATURES, ChaosFEX, ChaosNet, chaosfex_reference, firing_times,
                               gls_lyapunov, gls_trajectory, skew_binary, skew_tent)


@pytest.mark.parametrize("cmap", ["skew_tent", "skew_binary", "logistic"])
@pytest.mark.parametrize("eps", [0.2, 0.05, 0.01])
def test_fast_chaosfex_matches_reference(cmap, eps):
    rng = np.random.default_rng(0)
    X = rng.random((25, 3))
    f = ChaosFEX(q=0.34, b=0.499, epsilon=eps, max_iter=3000, chaotic_map=cmap, features=FEATURES,
                 normalize=False).fit(X)
    ref = chaosfex_reference(X, 0.34, 0.499, eps, 3000, cmap, FEATURES)
    np.testing.assert_allclose(f.transform(X), ref, rtol=1e-9, atol=1e-9)


def test_firing_time_definition():
    traj = np.array([0.1, 0.5, 0.9, 0.3])
    np.testing.assert_array_equal(firing_times(traj, [0.12, 0.88, 0.31, 0.7], 0.05), [0, 2, 3, 3])


def test_maps_stay_in_unit_interval():
    x = np.linspace(0, 1, 1001)
    for f in (skew_tent, skew_binary):
        y = f(x, 0.37)
        assert y.min() >= 0 and y.max() <= 1 + 1e-12


def test_trajectory_does_not_collapse_for_skewed_b():
    traj = gls_trajectory(0.34, 0.499, 5000)
    assert np.std(traj[-1000:]) > 0.2


def test_gls_lyapunov_max_at_half():
    b = np.linspace(0.05, 0.95, 91)
    assert abs(b[np.argmax(gls_lyapunov(b))] - 0.5) < 1e-9
    assert abs(gls_lyapunov(0.5) - np.log(2)) < 1e-12


def test_multi_population_shape():
    X = np.random.default_rng(1).random((10, 4))
    f = ChaosFEX(q=[0.2, 0.4, 0.6], features=("firing_time", "entropy")).fit(X)
    assert f.transform(X).shape == (10, 4 * 3 * 2)
    assert len(f.get_feature_names_out()) == 24


def test_chaosnet_beats_chance_on_iris():
    X, y = load_classification("iris")
    acc = cross_val_score(ChaosNet(q=0.34, b=0.499, epsilon=0.25, standardize=True), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
    assert acc > 0.8
