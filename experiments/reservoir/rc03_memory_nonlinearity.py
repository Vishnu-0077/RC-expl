"""RC-03: The memory–nonlinearity trade-off and the edge of stability.

We sweep the spectral radius rho for several input scalings and measure:

* the total linear memory capacity (Jaeger 2002),
* the total quadratic information-processing capacity (Dambre et al. 2012),
* the effective dimension of the reservoir state (participation ratio).

The sweep uses bias scaling 0.2. As a control we repeat it with zero bias:
the reservoir is then odd-symmetric (x(-u) = -x(u)), and its quadratic
capacity must be exactly zero, because every degree-2 target is even.

We also plot the memory curve MC_k for a few values of rho, and the
echo-state-property test (do two different initial states converge under
the same input?) as rho crosses 1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PALETTE, QUICK, outdir, plt, savefig, write_table  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from rcexpl.reservoir import ESN, effective_dimension, esp_convergence, memory_capacity, quadratic_capacity  # noqa: E402

OUT = outdir("reservoir", "rc03_memory")
N = 100
RHOS = np.linspace(0.1, 1.5, 8 if QUICK else 15)
SCALINGS = [0.05, 0.3, 1.0, 3.0]
SEEDS = range(2 if QUICK else 4)


BIAS = 0.2


def measure(rho, sigma, seed, bias=BIAS):
    esn = ESN(n_units=N, spectral_radius=rho, leak_rate=1.0, input_scaling=sigma, bias_scaling=bias,
              topology="erdos_renyi", topology_kwargs={"density": 0.1}, seed=seed)
    mc = memory_capacity(esn, max_delay=150, T=5000).sum()
    qc = quadratic_capacity(esn, max_delay=30, max_cross=10, T=5000)
    X = esn.transform(np.random.default_rng(seed).uniform(-1, 1, (3000, 1)))[500:]
    return rho, sigma, seed, mc, qc, effective_dimension(X), bias


def main():
    jobs = [(r, s, k, BIAS) for r in RHOS for s in SCALINGS for k in SEEDS]
    jobs += [(r, 0.3, k, 0.0) for r in RHOS for k in SEEDS]  # odd-symmetric control
    res = Parallel(n_jobs=-1)(delayed(measure)(*j) for j in jobs)
    rows = []
    for sigma, bias in [(s, BIAS) for s in SCALINGS] + [(0.3, 0.0)]:
        for rho in RHOS:
            sel = [r for r in res if r[0] == rho and r[1] == sigma and r[6] == bias]
            rows.append({"input_scaling": sigma, "bias_scaling": bias, "spectral_radius": float(rho),
                         "linear_MC": np.mean([r[3] for r in sel]), "quadratic_capacity": np.mean([r[4] for r in sel]),
                         "effective_dim": np.mean([r[5] for r in sel])})
    write_table(rows, OUT / "capacity_sweep")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for key, ax, title in [("linear_MC", axes[0], "Linear memory capacity"),
                           ("quadratic_capacity", axes[1], "Quadratic (nonlinear) capacity"),
                           ("effective_dim", axes[2], "Effective state dimension")]:
        for c, sigma in zip(PALETTE, SCALINGS):
            sel = [r for r in rows if r["input_scaling"] == sigma and r["bias_scaling"] == BIAS]
            ax.plot([r["spectral_radius"] for r in sel], [r[key] for r in sel], "-o", ms=4, color=c,
                    label=f"input scaling {sigma}")
        sel = [r for r in rows if r["bias_scaling"] == 0.0]
        ax.plot([r["spectral_radius"] for r in sel], [r[key] for r in sel], "--", color="#666", lw=1.2,
                label="input scaling 0.3, zero bias (odd-symmetric)")
        ax.axvline(1.0, color="#999", ls=":", lw=1)
        ax.set_xlabel("spectral radius ρ")
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Memory–nonlinearity trade-off (N = {N}, bias scaling {BIAS}, mean of {len(SEEDS)} reservoirs)",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, OUT / "capacity_tradeoff.png")

    # Memory curves and ESP check
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for c, rho in zip(PALETTE, [0.5, 0.9, 1.1, 1.4]):
        esn = ESN(n_units=N, spectral_radius=rho, leak_rate=1.0, input_scaling=0.05, bias_scaling=0.0,
                  topology_kwargs={"density": 0.1}, seed=0)
        mck = memory_capacity(esn, max_delay=150, T=5000)
        axes[0].plot(np.arange(1, 151), mck, color=c, label=f"ρ = {rho}  (MC = {mck.sum():.1f})")
        d = esp_convergence(ESN(n_units=N, spectral_radius=rho, leak_rate=1.0, input_scaling=0.05,
                                bias_scaling=0.0, topology_kwargs={"density": 0.1}, seed=0), T=600)
        axes[1].semilogy(d + 1e-16, color=c, label=f"ρ = {rho}")
    axes[0].set_xlabel("delay k"); axes[0].set_ylabel("MC_k  (squared correlation)")
    axes[0].set_title("Forgetting curves"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("time step"); axes[1].set_ylabel("‖x₁(t) − x₂(t)‖")
    axes[1].set_title("Echo-state property: do two initial states converge?"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, OUT / "memory_curves_esp.png")


if __name__ == "__main__":
    main()
