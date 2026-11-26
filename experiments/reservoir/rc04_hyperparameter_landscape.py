"""RC-04: Hyperparameter landscape and reservoir-size scaling.

(a) A grid over spectral radius x leak rate on Mackey-Glass (tau = 17),
    scored by closed-loop VPT and by one-step NRMSE. Across the whole grid
    the two scores are strongly rank-correlated, but among the
    configurations with the best one-step error the VPT still varies by
    about 2x. One-step error is therefore only a coarse proxy for
    forecasting skill.
(b) Lorenz closed-loop VPT as a function of reservoir size N, for the
    linear and quadratic readouts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PALETTE, QUICK, mean_ci, outdir, plt, savefig, standardize, write_table  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from rcexpl.datasets import load_system  # noqa: E402
from rcexpl.metrics import nrmse, valid_prediction_time  # noqa: E402
from rcexpl.reservoir import ESN  # noqa: E402

OUT = outdir("reservoir", "rc04_landscape")
G = 6 if QUICK else 14
RHOS = np.linspace(0.1, 1.5, G)
LEAKS = np.linspace(0.05, 1.0, G)


def mg_cell(rho, leak, data):
    train, val = data
    vp, ns = [], []
    for seed in range(3):
        m = ESN(n_units=300, spectral_radius=rho, leak_rate=leak, input_scaling=0.5, ridge=1e-7,
                readout="quadratic", topology_kwargs={"density": 0.03}, seed=seed)
        m.fit(train[:-1], train[1:])
        ns.append(nrmse(val[201:], m.predict(val[200:-1], warmup=val[:200])))
        for s0 in (0, 1000):
            with np.errstate(all="ignore"):
                g = m.generate(1200, warmup=val[s0:s0 + 200])
            vp.append(valid_prediction_time(val[s0 + 200:s0 + 1400], g, 0.4, 1.0))
    return rho, leak, float(np.mean(vp)), float(np.mean(ns))


def size_cell(n, readout, seed, data):
    train, test, lt = data
    m = ESN(n_units=n, spectral_radius=0.9, leak_rate=0.6, input_scaling=0.3, ridge=1e-7, readout=readout,
            topology_kwargs={"density": min(1.0, 8.0 / n)}, seed=seed)
    m.fit(train[:-1], train[1:])
    v = []
    for s0 in np.linspace(0, len(test) - 1700, 6).astype(int):
        with np.errstate(all="ignore"):
            g = m.generate(1400, warmup=test[s0:s0 + 300])
        v.append(valid_prediction_time(test[s0 + 300:s0 + 1700], g, 0.4, np.sqrt(3)) / lt)
    return n, readout, seed, float(np.mean(v))


def main():
    ts = load_system("mackey_glass", n=9000)
    z, _, _ = standardize(ts.data, 5000)
    res = Parallel(n_jobs=-1)(delayed(mg_cell)(r, a, (z[:5000], z[5000:])) for r in RHOS for a in LEAKS)
    V = np.array([r[2] for r in res]).reshape(G, G)
    E = np.array([r[3] for r in res]).reshape(G, G)
    rows = [{"spectral_radius": r[0], "leak_rate": r[1], "VPT_steps": r[2], "one_step_NRMSE": r[3]} for r in res]
    write_table(rows, OUT / "mackey_glass_grid", floatfmt="{:.4g}")
    from scipy.stats import spearmanr
    vp, ne = np.array([r[2] for r in res]), np.array([r[3] for r in res])
    good = ne <= np.quantile(ne, 0.25)
    summary = [{"spearman_VPT_vs_minus_log_NRMSE": float(spearmanr(vp, -np.log(ne))[0]),
                "best_VPT": float(vp.max()), "VPT_at_best_NRMSE": float(vp[np.argmin(ne)]),
                "VPT_min_in_top_quartile_NRMSE": float(vp[good].min()),
                "VPT_max_in_top_quartile_NRMSE": float(vp[good].max())}]
    write_table(summary, OUT / "proxy_summary", floatfmt="{:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ext = [LEAKS[0], LEAKS[-1], RHOS[0], RHOS[-1]]
    im = axes[0].imshow(V, origin="lower", aspect="auto", extent=ext, cmap="seq_blue")
    fig.colorbar(im, ax=axes[0], label="VPT (steps)")
    axes[0].set_title("Closed-loop VPT  (higher is better)")
    im = axes[1].imshow(np.log10(E), origin="lower", aspect="auto", extent=ext, cmap="seq_blue_r")
    fig.colorbar(im, ax=axes[1], label="log₁₀ one-step NRMSE")
    axes[1].set_title("One-step NRMSE  (darker = lower = better)")
    i, j = np.unravel_index(np.argmax(V), V.shape)
    k, l = np.unravel_index(np.argmin(E), E.shape)
    for ax in axes:
        ax.plot(LEAKS[j], RHOS[i], marker="*", ms=14, color=PALETTE[1], mec="white", label="best VPT")
        ax.plot(LEAKS[l], RHOS[k], marker="P", ms=11, color=PALETTE[2], mec="white", label="best NRMSE")
        ax.set_xlabel("leak rate a"); ax.set_ylabel("spectral radius ρ"); ax.grid(False)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Mackey-Glass (τ = 17): one-step error is only a coarse proxy for forecasting skill "
                 f"(VPT varies {vp[good].min():.0f}–{vp[good].max():.0f} steps within the best NRMSE quartile)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    savefig(fig, OUT / "mackey_glass_landscape.png")

    # (b) size scaling on Lorenz
    ts = load_system("lorenz", n=12000)
    z, _, _ = standardize(ts.data, 6000)
    sizes = [50, 100, 200, 400] if QUICK else [25, 50, 100, 200, 400, 800, 1200]
    seeds = range(2 if QUICK else 5)
    jobs = [(n, ro, s) for n in sizes for ro in ("linear", "quadratic") for s in seeds]
    res = Parallel(n_jobs=-1)(delayed(size_cell)(*j, (z[:6000], z[6000:], ts.lyapunov_time_steps())) for j in jobs)
    rows = []
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for c, ro in zip(PALETTE, ("linear", "quadratic")):
        mus, cis = [], []
        for n in sizes:
            m, ci = mean_ci([r[3] for r in res if r[0] == n and r[1] == ro])
            mus.append(m); cis.append(ci)
            rows.append({"readout": ro, "n_units": n, "VPT_LT": m, "ci95": ci})
        ax.errorbar(sizes, mus, yerr=cis, fmt="-o", ms=5, color=c, capsize=3, label=f"{ro} readout")
    ax.set_xscale("log"); ax.set_xlabel("reservoir size N"); ax.set_ylabel("VPT (Lyapunov times)")
    ax.set_title("Lorenz: forecasting horizon vs reservoir size"); ax.legend()
    fig.tight_layout()
    savefig(fig, OUT / "size_scaling.png")
    write_table(rows, OUT / "size_scaling")


if __name__ == "__main__":
    main()
