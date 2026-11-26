import numpy as np
import pytest

from rcexpl.datasets import load_system, narma
from rcexpl.metrics import nrmse, valid_prediction_time
from rcexpl.reservoir import (ESN, ESNEnsemble, NGRC, TOPOLOGIES, build_topology, esp_convergence,
                              memory_capacity, scale_spectral_radius, spectral_radius)


@pytest.mark.parametrize("name", sorted(TOPOLOGIES))
def test_topology_spectral_radius(name):
    W = scale_spectral_radius(build_topology(name, 60, np.random.default_rng(0)), 0.8)
    if name != "delay_line":  # nilpotent: scaled by its norm instead
        assert abs(spectral_radius(W) - 0.8) < 1e-6


def test_esn_fits_sine_and_generates():
    t = np.arange(3000) * 0.05
    s = np.sin(t)[:, None]
    m = ESN(n_units=100, spectral_radius=0.8, leak_rate=0.5, ridge=1e-8, seed=1).fit(s[:-1], s[1:])
    pred = m.generate(200, warmup=s[2000:2300])
    assert nrmse(s[2300:2500], pred) < 0.05


def test_esn_is_deterministic():
    u = np.random.default_rng(0).standard_normal((300, 2))
    a = ESN(n_units=50, seed=3).transform(u)
    b = ESN(n_units=50, seed=3).transform(u)
    np.testing.assert_array_equal(a, b)


def test_deep_esn_state_shape():
    u = np.zeros((10, 1))
    assert ESN(n_units=20, depth=3).transform(u).shape == (10, 60)


def test_echo_state_property_contracting():
    d = esp_convergence(ESN(n_units=100, spectral_radius=0.8, leak_rate=1.0), T=300)
    assert d[-1] < 1e-6 * d[0]


def test_memory_capacity_bounded_by_units():
    mc = memory_capacity(ESN(n_units=40, spectral_radius=0.9, input_scaling=0.1, seed=0), max_delay=80,
                         T=3000)
    assert 5 < mc.sum() <= 40 + 1


def test_ngrc_recovers_logistic_map_exactly():
    # x_{t+1} = r x_t - r x_t^2 lies exactly in the span of NG-RC features with k=1, degree=2
    x = load_system("logistic", n=800).data
    m = NGRC(k=1, degree=2, ridge=1e-12, predict_increment=False).fit(x[:500], x[1:501])
    p = m.predict(x[500:-1], warmup=x[:500])
    assert nrmse(x[501:], p) < 1e-6


def test_ensemble_and_ngrc_forecast_lorenz():
    ts = load_system("lorenz", n=5000)
    z = (ts.data - ts.data.mean(0)) / ts.data.std(0)
    tr, te = z[:3000], z[3000:]
    lt = ts.lyapunov_time_steps()
    for m in [ESNEnsemble(n_members=2, n_units=300, spectral_radius=0.9, leak_rate=0.6, input_scaling=0.3,
                          topology_kwargs={"density": 0.03}),
              NGRC(k=2, degree=2, ridge=1e-6)]:
        m.fit(tr[:-1], tr[1:])
        g = m.generate(500, warmup=te[:200])
        assert valid_prediction_time(te[200:700], g) / lt > 2.0


def test_narma_shapes():
    u, y = narma(1000)
    assert u.shape == y.shape == (1000, 1) and np.isfinite(y).all()


def test_lyapunov_estimates_close_to_literature():
    assert abs(load_system("lorenz", n=3000).lyapunov - 0.906) < 0.1
    assert abs(load_system("henon").lyapunov - 0.419) < 0.02
