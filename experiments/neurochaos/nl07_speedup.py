"""NL-07: Speed and exactness of the shared-trajectory ChaosFEX algorithm.

We time the naive per-neuron simulation, which is how ChaosFEX is usually
implemented, against our shared-trajectory / prefix-sum algorithm, as a
function of the number of neurons (samples x features) and of epsilon.
We also report the maximum absolute difference between the two outputs.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PALETTE, QUICK, outdir, plt, savefig, write_table  # noqa: E402

import numpy as np  # noqa: E402

from rcexpl.neurochaos import FEATURES, ChaosFEX, chaosfex_reference  # noqa: E402

OUT = outdir("neurochaos", "nl07_speedup")
SIZES = [100, 1000] if QUICK else [100, 300, 1000, 3000, 10000, 30000]
EPS = [0.1, 0.01, 0.001]
MAX_ITER = 20000


def best_time(fn, reps=3):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t)
    return min(ts), out


def main():
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for c, eps in zip(PALETTE, EPS):
        tf, tr = [], []
        for n in SIZES:
            X = np.random.default_rng(n).random((n, 1))
            f = ChaosFEX(q=0.34, b=0.499, epsilon=eps, max_iter=MAX_ITER, features=FEATURES, normalize=False)
            t_fast, A = best_time(lambda: f.fit(X).transform(X))
            # The reference is slow: time it once, on at most 3000 neurons, and extrapolate linearly.
            n_ref = min(n, 3000)
            t0 = time.perf_counter()
            B = chaosfex_reference(X[:n_ref], 0.34, 0.499, eps, MAX_ITER, features=FEATURES)
            t_ref = (time.perf_counter() - t0) * n / n_ref
            err = float(np.abs(A[:n_ref] - B).max())
            tf.append(t_fast); tr.append(t_ref)
            rows.append({"epsilon": eps, "neurons": n, "naive_s": t_ref, "fast_s": t_fast,
                         "speedup": t_ref / t_fast, "max_abs_diff": err,
                         "naive_extrapolated": n > n_ref})
            print(f"eps={eps:g} n={n:6d} naive={t_ref:8.3f}s fast={t_fast:8.4f}s x{t_ref / t_fast:7.1f} err={err:.1e}")
        ax.loglog(SIZES, tr, "--o", ms=4, color=c, label=f"naive, ε={eps:g}")
        ax.loglog(SIZES, tf, "-o", ms=4, color=c, label=f"shared-trajectory, ε={eps:g}")
    ax.set_xlabel("number of neurons (samples x features)"); ax.set_ylabel("wall time (s)")
    ax.set_title("ChaosFEX feature extraction: naive vs shared-trajectory (exact)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    savefig(fig, OUT / "speedup.png")
    write_table(rows, OUT / "speedup", floatfmt="{:.4g}")


if __name__ == "__main__":
    main()
