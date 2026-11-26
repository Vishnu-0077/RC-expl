"""NL-08: Inside the chaotic neuron. Visualising GLS dynamics and the ChaosFEX encoding.

Panels:
 (a) the skew-tent GLS map for several skews b;
 (b) the Lyapunov exponent as a function of b: the analytic expression
     -b ln b - (1-b) ln(1-b) against a numerical trajectory estimate;
 (c) a neural trace from q until it fires in the epsilon-neighbourhood of a stimulus;
 (d) the firing-time landscape N(x) over the stimulus range for two values of epsilon;
 (e) the five ChaosFEX features as functions of the stimulus x;
 (f) the invariant density of a long trajectory, uniform as the theory predicts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PALETTE, outdir, plt, savefig, write_table  # noqa: E402

import numpy as np  # noqa: E402

from rcexpl.neurochaos import FEATURES, ChaosFEX, firing_times, gls_lyapunov, gls_trajectory, skew_tent  # noqa: E402

OUT = outdir("neurochaos", "nl08_dynamics")


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x = np.linspace(0, 1, 1001)
    ax = axes[0, 0]
    for c, b in zip(PALETTE, [0.2, 0.499, 0.8]):
        ax.plot(x, skew_tent(x, b), color=c, label=f"b = {b}")
    ax.plot(x, x, ":", color="#999", lw=1)
    ax.set_title("(a) Skew-tent GLS map T_b(x)"); ax.set_xlabel("x"); ax.set_ylabel("T(x)"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    bs = np.linspace(0.02, 0.98, 49)
    num = []
    for b in bs:
        tr = gls_trajectory(0.1234, b, 20000)[1000:]
        num.append(np.mean(np.where(tr < b, -np.log(b), -np.log(1 - b))))
    ax.plot(bs, gls_lyapunov(bs), color="#191919", label="analytic  −b ln b − (1−b) ln(1−b)")
    ax.plot(bs, num, "o", ms=4, color=PALETTE[0], label="numerical (20k-step trajectory)")
    ax.axhline(np.log(2), ls=":", color="#999", lw=1)
    ax.set_title("(b) Lyapunov exponent vs skew b"); ax.set_xlabel("b"); ax.set_ylabel("λ"); ax.legend(fontsize=8)
    write_table([{"b": b, "lyapunov_analytic": float(gls_lyapunov(b)), "lyapunov_numerical": n}
                 for b, n in zip(bs, num)], OUT / "lyapunov_vs_b", floatfmt="{:.4f}")

    ax = axes[0, 2]
    q, b, eps, stim = 0.34, 0.499, 0.02, 0.71
    traj = gls_trajectory(q, b, 400)
    N = int(firing_times(traj, [stim], eps)[0])
    ax.axhspan(stim - eps, stim + eps, color=PALETTE[1], alpha=0.25, label=f"stimulus ± ε  (x = {stim})")
    ax.plot(np.arange(N + 1), traj[:N + 1], "-o", ms=3, color=PALETTE[0], lw=1, label="neural trace")
    ax.axhline(b, ls=":", color="#999", lw=1)
    ax.text(0, b + 0.02, "threshold b", fontsize=8, color="#666")
    ax.set_title(f"(c) Neuron fires after N = {N} iterations"); ax.set_xlabel("iteration n")
    ax.set_ylabel("neural activity"); ax.legend(fontsize=8, loc="lower right")

    ax = axes[1, 0]
    xs = np.linspace(0, 1, 4000)
    for c, e in zip(PALETTE, [0.1, 0.01]):
        tr = gls_trajectory(0.34, 0.499, 50000)
        ax.semilogy(xs, firing_times(tr, xs, e) + 1, lw=0.8, color=c, label=f"ε = {e}")
    ax.set_title("(d) Firing-time landscape N(x)"); ax.set_xlabel("stimulus x"); ax.set_ylabel("N(x) + 1")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    f = ChaosFEX(q=0.34, b=0.499, epsilon=0.05, features=FEATURES, normalize=False).fit(xs[:, None])
    F = f.transform(xs[:, None])
    for k, (name, c) in enumerate(zip(FEATURES, PALETTE)):
        v = F[:, k]
        ax.plot(xs, (v - v.min()) / (np.ptp(v) + 1e-12) + 1.15 * k, lw=0.8, color=c)
        ax.text(1.01, 1.15 * k + 0.4, name, fontsize=8, color="#191919", va="center")
    ax.set_yticks([]); ax.set_xlabel("stimulus x"); ax.set_xlim(0, 1.25)
    ax.set_title("(e) ChaosFEX features vs stimulus (ε = 0.05, min-max scaled)")

    ax = axes[1, 2]
    tr = gls_trajectory(0.34, 0.499, 200000)[100:]
    ax.hist(tr, bins=50, density=True, color=PALETTE[0], edgecolor="white", linewidth=1)
    ax.axhline(1.0, color="#191919", ls="--", lw=1, label="uniform (Lebesgue) density")
    ax.set_title("(f) Invariant density of the GLS neuron"); ax.set_xlabel("x"); ax.set_ylabel("density")
    ax.legend(fontsize=8, loc="lower center")
    fig.suptitle("Inside the GLS chaotic neuron", fontweight="bold")
    fig.tight_layout()
    savefig(fig, OUT / "neuron_dynamics.png")


if __name__ == "__main__":
    main()
