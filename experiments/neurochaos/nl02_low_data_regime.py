"""NL-02: Learning from very few examples.

A central claim of neurochaos learning is that it works with very few
training samples. We test it: for k in {1, 2, 3, 5, 10, 20, 50} training
samples per class, drawn at random 30 times, every model is trained on the
k * n_classes samples and evaluated on all the remaining samples.

With k = 1 no hyperparameter tuning is possible, so every model uses one
fixed, reasonable configuration: sklearn defaults for the baselines, and
q = 0.34, b = 0.499, epsilon = 0.18 for the neurochaos models.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PALETTE, QUICK, mean_ci, outdir, plt, savefig, write_table  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from rcexpl.datasets import load_classification  # noqa: E402
from rcexpl.neurochaos import ChaosNet, chaosfex_pipeline  # noqa: E402

OUT = outdir("neurochaos", "nl02_low_data")
DATASETS = ["iris", "wine", "breast_cancer", "digits"]
KS = [1, 3, 10] if QUICK else [1, 2, 3, 5, 10, 20, 50]
TRIALS = 5 if QUICK else 30
CH = dict(q=0.34, b=0.499, epsilon=0.18)

MODELS = {
    "ChaosNet": ChaosNet(**CH),
    "ChaosNet-Std": ChaosNet(standardize=True, **CH),
    "MP-ChaosFEX+SVM": chaosfex_pipeline(SVC(), q=[0.13, 0.37, 0.61, 0.83], b=0.499, epsilon=0.18),
    "SVM-RBF": make_pipeline(StandardScaler(), SVC()),
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
    "1-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(1)),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=0),
}


def trial(ds, k, t, X, y):
    rng = np.random.default_rng(1000 * k + t)
    tr = np.concatenate([rng.choice(np.flatnonzero(y == c), k, replace=False) for c in np.unique(y)])
    te = np.setdiff1d(np.arange(len(y)), tr)
    out = []
    for name, m in MODELS.items():
        try:
            pred = clone(m).fit(X[tr], y[tr]).predict(X[te])
            out.append((ds, k, t, name, f1_score(y[te], pred, average="macro")))
        except ValueError:  # e.g. LogReg/SVM cannot fit when a class has too few samples
            out.append((ds, k, t, name, np.nan))
    return out


def main():
    jobs = []
    for ds in DATASETS:
        X, y = load_classification(ds)
        kmax = np.bincount(y).min() - 1
        jobs += [(ds, k, t, X, y) for k in KS if k <= kmax for t in range(TRIALS)]
    res = [r for chunk in Parallel(n_jobs=-1)(delayed(trial)(*j) for j in jobs) for r in chunk]

    rows = []
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.2 * len(DATASETS), 3.9), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        for c, name in zip(PALETTE, MODELS):
            ks, mus, cis = [], [], []
            for k in KS:
                v = [r[4] for r in res if r[0] == ds and r[1] == k and r[3] == name and np.isfinite(r[4])]
                if not v:
                    continue
                m, ci = mean_ci(v)
                ks.append(k); mus.append(m); cis.append(ci)
                rows.append({"dataset": ds, "samples_per_class": k, "model": name, "macro_F1": m, "ci95": ci})
            ls = "-" if "Chaos" in name else "--"
            ax.errorbar(ks, mus, yerr=cis, fmt=ls + "o", ms=4, capsize=2, color=c, label=name, lw=1.4)
        ax.set_xscale("log"); ax.set_xlabel("training samples per class"); ax.set_title(ds)
        ax.set_xticks(KS, [str(k) for k in KS])
    axes[0].set_ylabel("macro-F1 on held-out data")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Low-data regime: mean ± 95% CI over {TRIALS} random draws "
                 "(solid = neurochaos, dashed = baselines)", fontweight="bold")
    fig.tight_layout()
    savefig(fig, OUT / "low_data.png")
    write_table(rows, OUT / "low_data")


if __name__ == "__main__":
    main()
