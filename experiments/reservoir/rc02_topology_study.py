"""RC-02: Does the reservoir topology matter?

We compare six recurrent topologies (dense, Erdős–Rényi, small-world,
simple cycle, cycle-with-jumps, delay line) on three axes:

1. closed-loop Lorenz forecasting, measured as VPT;
2. NARMA-10, an input-driven task measured by teacher-forced NRMSE;
3. linear memory capacity.

Every topology gets its own small random search on a validation block for
tasks 1 and 2. It is then evaluated over 10 independent random seeds, each
seed giving a new reservoir instance, so the spread across instances is
visible.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PALETTE, QUICK, mean_ci, outdir, plt, savefig, standardize, write_table  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from rcexpl.datasets import load_system, narma  # noqa: E402
from rcexpl.metrics import nrmse, valid_prediction_time  # noqa: E402
from rcexpl.reservoir import ESN, TOPOLOGIES, memory_capacity  # noqa: E402
from rcexpl.reservoir.tuning import LogUniform, Uniform, random_search  # noqa: E402

OUT = outdir("reservoir", "rc02_topology")
SEEDS = range(3 if QUICK else 10)
N_TRIALS = 6 if QUICK else 30
UNITS = 300
TOPO_KW = {"erdos_renyi": {"density": 0.03}, "small_world": {"k": 8, "p_rewire": 0.1}}


def lorenz_eval(topo, params, seed, data):
    train, val, test, scale, lt = data
    m = ESN(topology=topo, topology_kwargs=TOPO_KW.get(topo, {}), readout="quadratic", seed=seed,
            n_units=UNITS, **params)
    m.fit(np.vstack([train, val])[:-1], np.vstack([train, val])[1:])
    vpts = []
    for s0 in np.linspace(0, len(test) - 1500, 8).astype(int):
        with np.errstate(all="ignore"):
            g = m.generate(1200, warmup=test[s0:s0 + 300])
        vpts.append(valid_prediction_time(test[s0 + 300:s0 + 1500], g, 0.4, scale) / lt)
    return float(np.mean(vpts))


def narma_score(topo, params, seed, u, y, split):
    a, b = split
    m = ESN(topology=topo, topology_kwargs=TOPO_KW.get(topo, {}), seed=seed, n_units=UNITS, **params)
    m.fit(u[:a], y[:a])
    p = m.predict(u[a:b], warmup=None)
    return nrmse(y[a + 50:b], p[50:])


def main():
    ts = load_system("lorenz", n=12000)
    z, _, _ = standardize(ts.data, 5000)
    train, val, test = z[:5000], z[5000:8000], z[8000:]
    scale = float(np.sqrt(3.0))
    lt = ts.lyapunov_time_steps()
    lor = (train, val, test, scale, lt)
    u, y = narma(8000, seed=0)
    space = dict(spectral_radius=Uniform(0.1, 1.3), leak_rate=Uniform(0.1, 1.0),
                 input_scaling=LogUniform(0.02, 2.0), ridge=LogUniform(1e-10, 1e-3), washout=200)

    rows, lor_scores, narma_scores, mcs = [], {}, {}, {}
    for topo in TOPOLOGIES:
        print(f"--- {topo}")
        builder = lambda **p: ESN(topology=topo, topology_kwargs=TOPO_KW.get(topo, {}),  # noqa: E731
                                  readout="quadratic", n_units=UNITS, **p)
        best, _, _ = random_search(builder, space, train, val, n_trials=N_TRIALS, seed=2,
                                   warmup=300, horizon=1200, scale=scale)
        lor_scores[topo] = Parallel(n_jobs=-1)(delayed(lorenz_eval)(topo, best, s, lor) for s in SEEDS)

        # NARMA-10: tune on a validation split with a small random search
        rng = np.random.default_rng(3)
        cands = [{k: (v.sample(rng) if hasattr(v, "sample") else v) for k, v in space.items()}
                 for _ in range(N_TRIALS)]
        val_scores = Parallel(n_jobs=-1)(delayed(narma_score)(topo, c, 0, u, y, (4000, 6000)) for c in cands)
        nbest = cands[int(np.nanargmin(val_scores))]
        narma_scores[topo] = Parallel(n_jobs=-1)(
            delayed(narma_score)(topo, nbest, s, u, y, (6000, 8000)) for s in SEEDS)

        mcs[topo] = [memory_capacity(ESN(topology=topo, topology_kwargs=TOPO_KW.get(topo, {}), n_units=100,
                                         spectral_radius=0.95, leak_rate=1.0, input_scaling=0.1, seed=s),
                                     max_delay=200, T=5000).sum() for s in SEEDS]
        lm, lci = mean_ci(lor_scores[topo])
        nm, nci = mean_ci(narma_scores[topo])
        mm, mci = mean_ci(mcs[topo])
        rows.append({"topology": topo, "lorenz_VPT_LT": lm, "lorenz_VPT_ci95": lci,
                     "narma10_NRMSE": nm, "narma10_ci95": nci, "memory_capacity_N100": mm, "MC_ci95": mci})
        print(f"  Lorenz VPT {lm:.2f}±{lci:.2f} LT | NARMA10 NRMSE {nm:.3f}±{nci:.3f} | MC {mm:.1f}")

    write_table(rows, OUT / "topology")
    topos = list(TOPOLOGIES)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, data, title, ylab in [
        (axes[0], lor_scores, "Lorenz: closed-loop VPT (higher is better)", "VPT (Lyapunov times)"),
        (axes[1], narma_scores, "NARMA-10: NRMSE (lower is better)", "NRMSE"),
        (axes[2], mcs, "Linear memory capacity (N = 100)", "MC = Σ_k MC_k"),
    ]:
        for i, t in enumerate(topos):
            mu, ci = mean_ci(data[t])
            ax.bar(i, mu, color=PALETTE[0], width=0.7, edgecolor="white", linewidth=2)
            ax.errorbar(i, mu, yerr=ci, color="#444", capsize=3, lw=1)
            ax.scatter(i + np.random.default_rng(i).uniform(-0.2, 0.2, len(data[t])), data[t], s=9,
                       color="#333", alpha=0.4, zorder=3, linewidths=0)
        ax.set_xticks(range(len(topos)), [t.replace("_", "\n") for t in topos], fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylab)
    fig.suptitle("Reservoir topology study (mean ± 95% CI over independent reservoir instances)",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, OUT / "topology.png")


if __name__ == "__main__":
    main()
