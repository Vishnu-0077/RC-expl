"""RC-06: Robustness to observation noise, and noise as a regulariser.

The training data are corrupted with Gaussian observation noise at
relative levels sigma in {0, 0.1%, 1%, 2%, 5%, 10%} of the signal standard
deviation. Each model is evaluated by closed-loop VPT against the *clean*
Lorenz trajectory, so we measure how well it recovers the underlying
dynamics from noisy measurements.

Models: ESN-Q, ESN-Q trained with state-noise injection (a regulariser
that hardens the closed loop against its own errors), and NG-RC. For each
noise level the ridge parameter (and the state-noise amplitude) is picked
on a validation block, so the comparison is between best achievable
configurations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PALETTE, QUICK, mean_ci, outdir, plt, savefig, standardize, write_table  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from rcexpl.datasets import load_system  # noqa: E402
from rcexpl.metrics import valid_prediction_time  # noqa: E402
from rcexpl.reservoir import ESN, NGRC  # noqa: E402

OUT = outdir("reservoir", "rc06_noise")
LEVELS = [0.0, 0.01, 0.05] if QUICK else [0.0, 0.001, 0.01, 0.02, 0.05, 0.1]
RIDGES = [1e-8, 1e-6, 1e-4, 1e-2] if QUICK else [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1.0]
STATE_NOISE = [1e-4, 1e-3, 1e-2]
SEEDS = range(2 if QUICK else 5)
ESN_BASE = dict(n_units=400, spectral_radius=0.9, leak_rate=0.6, input_scaling=0.3,
                topology_kwargs={"density": 0.02}, readout="quadratic")


def vpt(m, clean, lt, n_win=6):
    out = []
    for s0 in np.linspace(0, len(clean) - 1500, n_win).astype(int):
        with np.errstate(all="ignore"):
            g = m.generate(1200, warmup=clean[s0:s0 + 300])
        out.append(valid_prediction_time(clean[s0 + 300:s0 + 1500], g, 0.4, np.sqrt(3)) / lt)
    return float(np.mean(out))


def build(kind, ridge, sn, seed):
    if kind == "NG-RC":
        return NGRC(k=2, s=1, degree=2, ridge=ridge)
    return ESN(ridge=ridge, state_noise=sn, seed=seed, **ESN_BASE)


def run(kind, level, seed, data):
    train, val, test, lt = data
    rng = np.random.default_rng(100 + seed)
    noisy = train + level * rng.standard_normal(train.shape)
    sns = STATE_NOISE if kind == "ESN-Q + state noise" else [0.0]
    best, best_score = None, -1
    for r in RIDGES:
        for sn in sns:
            m = build(kind, r, sn, seed).fit(noisy[:-1], noisy[1:])
            sc = vpt(m, val, lt, 3)
            if sc > best_score:
                best, best_score = (r, sn), sc
    m = build(kind, *best, seed).fit(noisy[:-1], noisy[1:])
    return kind, level, seed, vpt(m, test, lt), best


def main():
    ts = load_system("lorenz", n=14000)
    z, _, _ = standardize(ts.data, 6000)
    data = (z[:6000], z[6000:9000], z[9000:], ts.lyapunov_time_steps())
    kinds = ["ESN-Q", "ESN-Q + state noise", "NG-RC"]
    res = Parallel(n_jobs=-1)(delayed(run)(k, lv, s, data) for k in kinds for lv in LEVELS for s in SEEDS)
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for c, k in zip(PALETTE, kinds):
        mus, cis = [], []
        for lv in LEVELS:
            sel = [r for r in res if r[0] == k and r[1] == lv]
            m, ci = mean_ci([r[3] for r in sel])
            mus.append(m); cis.append(ci)
            rows.append({"model": k, "noise_level": lv, "VPT_LT": m, "ci95": ci,
                         "chosen_ridge": sel[0][4][0], "chosen_state_noise": sel[0][4][1]})
        x = np.array(LEVELS) + 3e-4
        ax.errorbar(x, mus, yerr=cis, fmt="-o", ms=5, capsize=3, color=c, label=k)
    ax.set_xscale("log"); ax.set_xlabel("observation noise σ (fraction of signal std; 0 plotted at 3e-4)")
    ax.set_ylabel("VPT on clean data (Lyapunov times)")
    ax.set_title("Lorenz: learning dynamics from noisy measurements"); ax.legend()
    fig.tight_layout()
    savefig(fig, OUT / "noise_robustness.png")
    write_table(rows, OUT / "noise_robustness", floatfmt="{:.4g}")


if __name__ == "__main__":
    main()
