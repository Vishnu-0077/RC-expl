"""NL-03: Hyperparameter landscape of ChaosNet.

Macro-F1 (5-fold CV) over a grid of initial activity q x map skew b, at a
fixed neighbourhood epsilon, for several datasets. The landscape is very
rugged: neighbouring (q, b) pairs can differ by tens of F1 points, because
the firing-time encoding x -> N(x) is a highly discontinuous function of
q and b. This motivates the multi-population encoding, which averages
over several values of q and is far less sensitive (right column).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PALETTE, QUICK, outdir, plt, savefig, write_table  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from rcexpl.datasets import load_classification  # noqa: E402
from rcexpl.neurochaos import ChaosNet, chaosfex_pipeline  # noqa: E402

OUT = outdir("neurochaos", "nl03_landscape")
DATASETS = ["iris", "wine", "breast_cancer"]
G = 6 if QUICK else 19
QS = np.linspace(0.05, 0.95, G)
BS = np.linspace(0.05, 0.95, G)
EPS = 0.18
CV = StratifiedKFold(5, shuffle=True, random_state=0)


def cell(ds, q, b, X, y):
    return cross_val_score(ChaosNet(q=q, b=b, epsilon=EPS, standardize=True), X, y, cv=CV,
                           scoring="f1_macro").mean()


def mp_cell(ds, b, X, y, n_pop, seed):
    qs = np.random.default_rng(seed).uniform(0.05, 0.95, n_pop)
    return cross_val_score(chaosfex_pipeline(SVC(), q=list(qs), b=b, epsilon=EPS), X, y, cv=CV,
                           scoring="f1_macro").mean()


def main():
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(11, 3.7 * len(DATASETS)),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    rows = []
    for row, ds in enumerate(DATASETS):
        X, y = load_classification(ds)
        res = Parallel(n_jobs=-1)(delayed(cell)(ds, q, b, X, y) for q in QS for b in BS)
        F = np.array(res).reshape(G, G)
        for (i, q) in enumerate(QS):
            for (j, b) in enumerate(BS):
                rows.append({"dataset": ds, "q": q, "b": b, "macro_F1": F[i, j]})
        ax = axes[row, 0]
        im = ax.imshow(F, origin="lower", extent=[BS[0], BS[-1], QS[0], QS[-1]], aspect="auto",
                       cmap="seq_blue", vmin=0.3, vmax=1.0)
        fig.colorbar(im, ax=ax, label="macro-F1")
        i, j = np.unravel_index(np.argmax(F), F.shape)
        ax.plot(BS[j], QS[i], "*", ms=13, color=PALETTE[1], mec="white")
        ax.set_xlabel("map skew b"); ax.set_ylabel("initial activity q"); ax.grid(False)
        ax.set_title(f"{ds}: ChaosNet-Std, ε = {EPS}  (best {F.max():.3f}, median {np.median(F):.3f})")

        # Robustness of multi-population encoding: F1 distribution across random q-sets
        ax = axes[row, 1]
        b_fix = 0.499
        single = Parallel(n_jobs=-1)(delayed(mp_cell)(ds, b_fix, X, y, 1, s) for s in range(30))
        multi = Parallel(n_jobs=-1)(delayed(mp_cell)(ds, b_fix, X, y, 4, s) for s in range(30))
        multi8 = Parallel(n_jobs=-1)(delayed(mp_cell)(ds, b_fix, X, y, 8, s) for s in range(30))
        parts = ax.violinplot([single, multi, multi8], showmedians=True)
        for body, c in zip(parts["bodies"], PALETTE):
            body.set_facecolor(c); body.set_alpha(0.55)
        ax.set_xticks([1, 2, 3], ["1 population", "4 populations", "8 populations"])
        ax.set_ylabel("macro-F1 (ChaosFEX + SVM)")
        ax.set_title(f"{ds}: sensitivity to random initial activities q")
        for n, v in zip([1, 4, 8], [single, multi, multi8]):
            rows.append({"dataset": ds, "q": f"random x{n}", "b": b_fix, "macro_F1": float(np.mean(v))})
    fig.tight_layout()
    savefig(fig, OUT / "q_b_landscape.png")
    write_table(rows, OUT / "landscape")


if __name__ == "__main__":
    main()
