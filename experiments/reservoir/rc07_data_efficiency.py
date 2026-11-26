"""RC-07: Data efficiency. How much training data does each model need?

We vary the Lorenz training length from 100 to 8000 samples (about 2 to
145 Lyapunov times) and measure the closed-loop VPT on a fixed test block.
Each point averages 5 different training segments; the ESN members also
differ in their random seed. The ridge parameter is chosen per length on
a validation block.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PALETTE, QUICK, mean_ci, outdir, plt, savefig, standardize, write_table  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from rcexpl.datasets import load_system  # noqa: E402
from rcexpl.metrics import valid_prediction_time  # noqa: E402
from rcexpl.reservoir import ESN, NGRC, ESNEnsemble  # noqa: E402

OUT = outdir("reservoir", "rc07_data_efficiency")
LENGTHS = [200, 1000, 4000] if QUICK else [100, 200, 400, 800, 1600, 3200, 6400, 8000]
REPS = range(2 if QUICK else 5)
RIDGES = [1e-8, 1e-6, 1e-4, 1e-2]
ESN_KW = dict(n_units=400, spectral_radius=0.9, leak_rate=0.6, input_scaling=0.3,
              topology_kwargs={"density": 0.02}, readout="quadratic", washout=50)


def make(kind, ridge, seed):
    if kind == "NG-RC":
        return NGRC(k=2, s=1, degree=2, ridge=ridge)
    if kind == "ESN-Ensemble":
        return ESNEnsemble(n_members=5, ridge=ridge, seed=seed, **ESN_KW)
    return ESN(ridge=ridge, seed=seed, **ESN_KW)


def score(m, block, lt, n_win):
    out = []
    for s0 in np.linspace(0, len(block) - 1500, n_win).astype(int):
        with np.errstate(all="ignore"):
            g = m.generate(1200, warmup=block[s0:s0 + 300])
        out.append(valid_prediction_time(block[s0 + 300:s0 + 1500], g, 0.4, np.sqrt(3)) / lt)
    return float(np.mean(out))


def run(kind, L, rep, z, lt):
    start = rep * 1500
    train = z[start:start + L]
    val, test = z[16000:19000], z[19000:]
    best = max(RIDGES, key=lambda r: score(make(kind, r, rep).fit(train[:-1], train[1:]), val, lt, 3))
    return kind, L, rep, score(make(kind, best, rep).fit(train[:-1], train[1:]), test, lt, 8)


def main():
    ts = load_system("lorenz", n=24000)
    z, _, _ = standardize(ts.data, 16000)
    lt = ts.lyapunov_time_steps()
    kinds = ["ESN-Q", "ESN-Ensemble", "NG-RC"]
    res = Parallel(n_jobs=-1)(delayed(run)(k, L, r, z, lt) for k in kinds for L in LENGTHS for r in REPS)
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c, k in zip(PALETTE, kinds):
        mus, cis = [], []
        for L in LENGTHS:
            m, ci = mean_ci([r[3] for r in res if r[0] == k and r[1] == L])
            mus.append(m); cis.append(ci)
            rows.append({"model": k, "train_length": L, "train_lyapunov_times": L / lt, "VPT_LT": m, "ci95": ci})
        ax.errorbar(np.array(LENGTHS) / lt, mus, yerr=cis, fmt="-o", ms=5, capsize=3, color=c, label=k)
    ax.set_xscale("log"); ax.set_xlabel("training length (Lyapunov times)")
    ax.set_ylabel("VPT (Lyapunov times)"); ax.set_title("Lorenz: data efficiency"); ax.legend()
    fig.tight_layout()
    savefig(fig, OUT / "data_efficiency.png")
    write_table(rows, OUT / "data_efficiency")


if __name__ == "__main__":
    main()
