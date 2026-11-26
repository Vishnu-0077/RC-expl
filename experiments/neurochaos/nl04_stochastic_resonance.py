"""NL-04: Stochastic-resonance-like behaviour of the firing neighbourhood epsilon.

Harikrishnan & Nagaraj (2021) interpret the neighbourhood epsilon as a
noise intensity and report that classification performance peaks at an
intermediate value. We test this across datasets:

(a) macro-F1 as a function of epsilon, on a log grid from 1e-3 to 0.5;
(b) the mean firing time as a function of epsilon, i.e. the computational
    cost, which scales roughly as 1/epsilon;
(c) explicit additive Gaussian noise on the input stimulus at a fixed
    epsilon. Does injected noise ever help?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PALETTE, QUICK, outdir, plt, savefig, write_table  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from rcexpl.datasets import load_classification  # noqa: E402
from rcexpl.neurochaos import ChaosFEX, ChaosNet  # noqa: E402

OUT = outdir("neurochaos", "nl04_resonance")
DATASETS = ["iris", "wine", "breast_cancer", "moons", "spirals"]
EPS = np.logspace(-3, np.log10(0.5), 8 if QUICK else 26)
NOISE = np.r_[0, np.logspace(-3, np.log10(0.3), 6 if QUICK else 14)]
CV = StratifiedKFold(5, shuffle=True, random_state=0)


def f1_eps(ds, eps, X, y):
    a = cross_val_score(ChaosNet(q=0.34, b=0.499, epsilon=eps, standardize=True), X, y, cv=CV,
                        scoring="f1_macro").mean()
    fex = ChaosFEX(q=0.34, b=0.499, epsilon=eps).fit(X)
    ft = fex.firing_time_matrix(X).mean()
    return ds, eps, a, ft


def f1_noise(ds, sigma, X, y):
    # Noise is injected in both training and test transforms (a fresh draw each time).
    pipe = Pipeline([("fex", ChaosFEX(q=0.34, b=0.499, epsilon=0.05, input_noise=sigma, random_state=0)),
                     ("sc", StandardScaler()), ("svm", SVC())])
    return ds, sigma, cross_val_score(pipe, X, y, cv=CV, scoring="f1_macro").mean()


def main():
    data = {ds: load_classification(ds) for ds in DATASETS}
    r1 = Parallel(n_jobs=-1)(delayed(f1_eps)(ds, e, *data[ds]) for ds in DATASETS for e in EPS)
    r2 = Parallel(n_jobs=-1)(delayed(f1_noise)(ds, s, *data[ds]) for ds in DATASETS for s in NOISE)
    rows = [{"dataset": d, "epsilon": e, "macro_F1": a, "mean_firing_time": ft} for d, e, a, ft in r1]
    write_table(rows, OUT / "epsilon_sweep", floatfmt="{:.4g}")
    write_table([{"dataset": d, "input_noise": s, "macro_F1": a} for d, s, a in r2], OUT / "input_noise",
                floatfmt="{:.4g}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c, ds in zip(PALETTE, DATASETS):
        sel = [r for r in r1 if r[0] == ds]
        f1 = np.array([r[2] for r in sel])
        axes[0].plot(EPS, f1, "-o", ms=3, color=c, label=ds)
        k = int(np.argmax(f1))
        axes[0].plot(EPS[k], f1[k], "*", ms=12, color=c, mec="white")
        axes[1].loglog(EPS, [r[3] for r in sel], "-", color=c, label=ds)
        sel2 = [r for r in r2 if r[0] == ds]
        axes[2].plot(np.array(NOISE) + 5e-4, [r[2] for r in sel2], "-o", ms=3, color=c, label=ds)
    axes[1].loglog(EPS, 0.5 / EPS, ":", color="#666", label="∝ 1/ε")
    axes[0].set_xscale("log"); axes[0].set_xlabel("neighbourhood ε"); axes[0].set_ylabel("macro-F1 (5-fold CV)")
    axes[0].set_title("Accuracy peaks at an intermediate ε  (★ = optimum)"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("neighbourhood ε"); axes[1].set_ylabel("mean firing time (iterations)")
    axes[1].set_title("Computational cost of the neurons"); axes[1].legend(fontsize=8)
    axes[2].set_xscale("log"); axes[2].set_xlabel("input noise σ (0 plotted at 5e-4)")
    axes[2].set_ylabel("macro-F1 (ChaosFEX ε=0.05 + SVM)"); axes[2].set_title("Additive stimulus noise")
    fig.suptitle("Stochastic-resonance-like behaviour in neurochaos learning", fontweight="bold")
    fig.tight_layout()
    savefig(fig, OUT / "stochastic_resonance.png")


if __name__ == "__main__":
    main()
