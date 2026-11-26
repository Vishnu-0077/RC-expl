"""NL-05: Which ChaosFEX features carry the information?

We evaluate all 31 non-empty subsets of the five features {firing time,
firing rate, energy, entropy, variance} with a ChaosFEX -> SVM pipeline,
using 5-fold CV on four datasets.

All five features are deterministic functions of a single integer, the
firing time N(x), so each stimulus is encoded through a 1-D quantity. The
features differ only in how nonlinearly they re-express N, which matters
for the downstream classifier's geometry. The ablation measures how much
that matters.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PALETTE, outdir, plt, savefig, write_table  # noqa: E402

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from rcexpl.datasets import load_classification  # noqa: E402
from rcexpl.neurochaos import FEATURES, chaosfex_pipeline  # noqa: E402

OUT = outdir("neurochaos", "nl05_ablation")
DATASETS = ["iris", "wine", "breast_cancer", "digits"]
CV = StratifiedKFold(5, shuffle=True, random_state=0)
SHORT = {"firing_time": "FT", "firing_rate": "FR", "energy": "En", "entropy": "Ent", "variance": "Var"}


def run(ds, feats, X, y):
    pipe = chaosfex_pipeline(SVC(), q=0.34, b=0.499, epsilon=0.1, features=feats)
    return ds, feats, cross_val_score(pipe, X, y, cv=CV, scoring="f1_macro").mean()


def main():
    subsets = [s for r in range(1, 6) for s in itertools.combinations(FEATURES, r)]
    data = {d: load_classification(d) for d in DATASETS}
    res = Parallel(n_jobs=-1)(delayed(run)(d, s, *data[d]) for d in DATASETS for s in subsets)
    score = {(d, s): a for d, s, a in res}
    rows = []
    for s in subsets:
        row = {"features": "+".join(SHORT[f] for f in s), "n_features": len(s)}
        for d in DATASETS:
            row[d] = score[(d, s)]
        row["mean"] = float(np.mean([score[(d, s)] for d in DATASETS]))
        rows.append(row)
    rows.sort(key=lambda r: -r["mean"])
    write_table(rows, OUT / "all_subsets")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    singles = [(f,) for f in FEATURES]
    w = 0.8 / len(DATASETS)
    for k, (d, c) in enumerate(zip(DATASETS, PALETTE)):
        axes[0].bar(np.arange(5) + (k - 1.5) * w, [score[(d, s)] for s in singles], w, color=c,
                    edgecolor="white", linewidth=1.5, label=d)
        loo = [tuple(f for f in FEATURES if f != g) for g in FEATURES]
        full = score[(d, tuple(FEATURES))]
        axes[1].bar(np.arange(5) + (k - 1.5) * w, [full - score[(d, s)] for s in loo], w, color=c,
                    edgecolor="white", linewidth=1.5, label=d)
    axes[0].set_xticks(range(5), [SHORT[f] for f in FEATURES]); axes[0].set_ylim(0.4, 1.0)
    axes[0].set_ylabel("macro-F1"); axes[0].set_title("Single feature only"); axes[0].legend(fontsize=8, ncol=2)
    axes[1].axhline(0, color="#999", lw=1)
    axes[1].set_xticks(range(5), [f"−{SHORT[f]}" for f in FEATURES])
    axes[1].set_ylabel("F1 drop vs all five features"); axes[1].set_title("Leave-one-feature-out")
    fig.suptitle("ChaosFEX feature ablation (ChaosFEX + SVM, 5-fold CV)", fontweight="bold")
    fig.tight_layout()
    savefig(fig, OUT / "feature_ablation.png")


if __name__ == "__main__":
    main()
