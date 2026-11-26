"""NL-06: Choice of chaotic map and number of neuron populations.

We compare three chaotic neurons (skew tent, skew binary and the logistic
map at r = 4) and 1, 2, 4, 8 or 16 neuron populations per input feature,
each population with its own initial activity q spread evenly over (0, 1).
The pipeline is ChaosFEX -> SVM with 5-fold CV.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PALETTE, QUICK, mean_ci, outdir, plt, savefig, write_table  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from rcexpl.datasets import load_classification  # noqa: E402
from rcexpl.neurochaos import chaosfex_pipeline  # noqa: E402

OUT = outdir("neurochaos", "nl06_maps_populations")
DATASETS = ["iris", "wine", "breast_cancer", "moons", "circles", "spirals"]
MAPS = ["skew_tent", "skew_binary", "logistic"]
POPS = [1, 2, 4] if QUICK else [1, 2, 4, 8, 16]
CV = StratifiedKFold(5, shuffle=True, random_state=0)


def qset(n, offset):
    # evenly spread, shifted by an irrational offset to avoid special points
    return list(((np.arange(n) + 0.5) / n + offset * 0.6180339887) % 1.0 * 0.9 + 0.05)


def run(ds, cmap, n, rep, X, y):
    pipe = chaosfex_pipeline(SVC(), q=qset(n, rep), b=0.499, epsilon=0.1, chaotic_map=cmap)
    return ds, cmap, n, rep, cross_val_score(pipe, X, y, cv=CV, scoring="f1_macro").mean()


def main():
    data = {d: load_classification(d) for d in DATASETS}
    reps = range(2 if QUICK else 5)
    res = Parallel(n_jobs=-1)(delayed(run)(d, m, n, r, *data[d])
                              for d in DATASETS for m in MAPS for n in POPS for r in reps)
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, d in zip(axes.ravel(), DATASETS):
        for c, m in zip(PALETTE, MAPS):
            mus, cis = [], []
            for n in POPS:
                mu, ci = mean_ci([r[4] for r in res if r[0] == d and r[1] == m and r[2] == n])
                mus.append(mu); cis.append(ci)
                rows.append({"dataset": d, "map": m, "populations": n, "macro_F1": mu, "ci95": ci})
            ax.errorbar(POPS, mus, yerr=cis, fmt="-o", ms=4, capsize=2, color=c, label=m)
        ax.set_xscale("log", base=2); ax.set_xticks(POPS, [str(p) for p in POPS]); ax.set_title(d)
    for ax in axes[-1]:
        ax.set_xlabel("neuron populations per input")
    for ax in axes[:, 0]:
        ax.set_ylabel("macro-F1")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Chaotic map choice x multi-population encoding (ChaosFEX + SVM)", fontweight="bold")
    fig.tight_layout()
    savefig(fig, OUT / "maps_populations.png")
    write_table(rows, OUT / "maps_populations")


if __name__ == "__main__":
    main()
