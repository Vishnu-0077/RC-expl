"""RC-05: Does the trained reservoir reproduce the Lorenz attractor's long-term climate?

Short-term accuracy (VPT) is bounded by chaos. A good surrogate model
should nonetheless keep generating trajectories with the right
*statistics* long after it has diverged from the true one. We run each
model autonomously for 20 000 steps (about 360 Lyapunov times) and
compare:

* phase portraits, as x-z projections;
* the Lorenz return map z_max(n) -> z_max(n+1) (Lorenz 1963);
* the power spectrum of x(t);
* the invariant density of z, compared with the Wasserstein distance;
* the largest Lyapunov exponent of the trained closed-loop ESN, estimated
  with Benettin's method in reservoir space, against the true exponent.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PALETTE, QUICK, RESULTS, outdir, plt, savefig, standardize, write_table  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

from rcexpl.datasets import load_system  # noqa: E402
from rcexpl.metrics import local_maxima, power_spectrum, spectrum_distance  # noqa: E402
from rcexpl.reservoir import ESN, NGRC, ESNEnsemble  # noqa: E402

OUT = outdir("reservoir", "rc05_climate")
STEPS = 4000 if QUICK else 20000


def load_best():
    p = RESULTS / "reservoir" / "rc01_benchmark" / "best_params.json"
    default_esn = dict(spectral_radius=0.9, leak_rate=0.6, input_scaling=0.3, ridge=1e-7, n_units=400,
                       topology_kwargs={"density": 0.02}, washout=200)
    default_ng = dict(k=2, s=1, degree=2, ridge=1e-6)
    if p.exists():
        best = json.loads(p.read_text())["lorenz"]
        return best["ESN-Q"], best["NG-RC"]
    return default_esn, default_ng


def main():
    ts = load_system("lorenz", n=8000 + STEPS)
    z, mu, sd = standardize(ts.data, 8000)
    train, ref = z[:8000], z[8000:]
    esn_p, ng_p = load_best()
    esn_p = dict(esn_p)
    for k in ("predict_increment",):
        esn_p.pop(k, None)
    models = {
        "ESN-Q": ESN(readout="quadratic", **esn_p),
        "ESN-Ensemble": ESNEnsemble(n_members=5, readout="quadratic", **esn_p),
        "NG-RC": NGRC(**ng_p),
    }
    gens = {}
    for name, m in models.items():
        m.fit(train[:-1], train[1:])
        with np.errstate(all="ignore"):
            g = m.generate(STEPS, warmup=train[-500:])
        gens[name] = g * sd + mu
        print(f"{name}: generated {STEPS} steps, finite={np.isfinite(g).all()}")
    truth = ref * sd + mu

    true_lle = ts.lyapunov
    esn_lle = models["ESN-Q"].lyapunov_exponent(train[-500:], n_steps=3000 if QUICK else 10000) / ts.dt
    rows = []
    zmax_true = local_maxima(truth[:, 2])
    for name, g in gens.items():
        ok = np.isfinite(g).all() and np.abs(g).max() < 1e3
        zm = local_maxima(g[:, 2]) if ok else np.array([np.nan])
        rows.append({
            "model": name,
            "stays_on_attractor": bool(ok),
            "W1_density_z": wasserstein_distance(truth[:, 2], g[:, 2]) if ok else np.nan,
            "W1_zmax": wasserstein_distance(zmax_true, zm) if ok else np.nan,
            "log_spectrum_dist_x": spectrum_distance(truth[:, 0], g[:, 0], ts.dt) if ok else np.nan,
            "largest_LE": esn_lle if name == "ESN-Q" else np.nan,
        })
    rows.append({"model": "ground truth", "stays_on_attractor": True, "W1_density_z": 0.0, "W1_zmax": 0.0,
                 "log_spectrum_dist_x": 0.0, "largest_LE": true_lle})
    write_table(rows, OUT / "climate")
    print(f"largest Lyapunov exponent: true {true_lle:.3f}, ESN closed loop {esn_lle:.3f}")

    names = ["ground truth"] + list(gens)
    series = [truth] + list(gens.values())
    colors = ["#191919"] + PALETTE[:3]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, n, s, c in zip(axes[0], names, series, colors):
        ax.plot(s[:, 0], s[:, 2], lw=0.25, color=c, alpha=0.8)
        ax.set_title(n); ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_xlim(-25, 25); ax.set_ylim(0, 55)
    ax = axes[1, 0]
    for n, s, c in zip(names, series, colors):
        zm = local_maxima(s[:, 2])
        ax.scatter(zm[:-1], zm[1:], s=3, color=c, alpha=0.6, label=n, linewidths=0)
    ax.set_xlabel("z_max(n)"); ax.set_ylabel("z_max(n+1)"); ax.set_title("Lorenz return map")
    ax.legend(fontsize=7, markerscale=4); ax.set_xlim(28, 50); ax.set_ylim(28, 50)
    ax = axes[1, 1]
    for n, s, c in zip(names, series, colors):
        f, p = power_spectrum(s[:, 0], ts.dt)
        k = np.convolve(p, np.ones(15) / 15, mode="same")
        ax.semilogy(f, k, color=c, lw=1.1, label=n)
    ax.set_xlim(0, 5); ax.set_ylim(1e-2, None); ax.set_xlabel("frequency (1/time)"); ax.set_ylabel("PSD of x (smoothed)")
    ax.set_title("Power spectrum"); ax.legend(fontsize=7)
    ax = axes[1, 2]
    bins = np.linspace(0, 50, 60)
    for n, s, c in zip(names, series, colors):
        ax.hist(s[:, 2], bins=bins, histtype="step", density=True, color=c, lw=1.4, label=n)
    ax.set_xlabel("z"); ax.set_ylabel("density"); ax.set_title("Invariant density of z"); ax.legend(fontsize=7)
    ax = axes[1, 3]
    ax.bar([0, 1], [true_lle, esn_lle], color=["#191919", PALETTE[0]], width=0.6, edgecolor="white", linewidth=2)
    ax.set_xticks([0, 1], ["true system", "trained ESN\n(closed loop)"])
    for i, v in enumerate([true_lle, esn_lle]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, color="#191919")
    ax.set_ylabel("largest Lyapunov exponent"); ax.set_title("Learned dynamics: Lyapunov exponent")
    fig.suptitle(f"Long-term climate replication: {STEPS} autonomous steps "
                 f"(~{STEPS / ts.lyapunov_time_steps():.0f} Lyapunov times)", fontweight="bold")
    fig.tight_layout()
    savefig(fig, OUT / "lorenz_climate.png")


if __name__ == "__main__":
    main()
