"""NL-01: Neurochaos-learning classification benchmark.

Protocol: 5-fold stratified outer cross-validation, repeated twice with
different shuffles, reporting macro-F1 and accuracy. Each model's
hyperparameters (chaos parameters q, b, epsilon and the downstream
classifier's parameters) are tuned *inside* each training fold by
randomised 3-fold CV. This is nested cross-validation, so no test data
leaks into tuning. Classical baselines on the raw features get the same
tuning budget.

Neurochaos models
  ChaosNet            original: ChaosFEX + cosine similarity to class means
  ChaosNet-Std        ours: the same, with standardised features before the cosine
  ChaosFEX+SVM        ChaosFEX features -> RBF-SVM
  ChaosFEX+LR         ChaosFEX features -> logistic regression
  MP-ChaosFEX+SVM     ours: multi-population ChaosFEX (4 initial activities) -> RBF-SVM
Baselines (raw features): SVM-RBF, LogReg, kNN, RandomForest, MLP
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import PALETTE, QUICK, Timer, mean_ci, outdir, plt, savefig, write_json, write_table  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import loguniform, uniform  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from rcexpl.datasets import load_classification  # noqa: E402
from rcexpl.neurochaos import ChaosNet, chaosfex_pipeline  # noqa: E402

OUT = outdir("neurochaos", "nl01_benchmark")
DATASETS = ["iris", "wine", "breast_cancer", "digits", "moons", "circles", "xor", "spirals"]
if QUICK:
    DATASETS = ["iris", "wine", "moons"]
N_ITER = 5 if QUICK else 25
REPEATS = 1 if QUICK else 2

CHAOS_SPACE = {"q": uniform(0.05, 0.9), "b": uniform(0.05, 0.9), "epsilon": loguniform(0.01, 0.35)}


def chaos_params(prefix):
    return {f"{prefix}{k}": v for k, v in CHAOS_SPACE.items()}


def models():
    mq = [0.13, 0.37, 0.61, 0.83]
    return {
        "ChaosNet": (ChaosNet(), chaos_params("")),
        "ChaosNet-Std": (ChaosNet(standardize=True), chaos_params("")),
        "ChaosFEX+SVM": (chaosfex_pipeline(SVC()), {**chaos_params("chaosfex__"),
                                                    "clf__C": loguniform(0.1, 1e3), "clf__gamma": ["scale"]}),
        "ChaosFEX+LR": (chaosfex_pipeline(LogisticRegression(max_iter=3000)),
                        {**chaos_params("chaosfex__"), "clf__C": loguniform(1e-2, 1e3)}),
        "MP-ChaosFEX+SVM": (chaosfex_pipeline(SVC(), q=mq),
                            {"chaosfex__b": uniform(0.05, 0.9), "chaosfex__epsilon": loguniform(0.01, 0.35),
                             "clf__C": loguniform(0.1, 1e3)}),
        "SVM-RBF": (make_pipeline(StandardScaler(), SVC()),
                    {"svc__C": loguniform(0.1, 1e3), "svc__gamma": loguniform(1e-4, 10)}),
        "LogReg": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                   {"logisticregression__C": loguniform(1e-3, 1e3)}),
        "kNN": (make_pipeline(StandardScaler(), KNeighborsClassifier()),
                {"kneighborsclassifier__n_neighbors": list(range(1, 26)),
                 "kneighborsclassifier__weights": ["uniform", "distance"]}),
        "RandomForest": (RandomForestClassifier(n_estimators=200, random_state=0),
                         {"max_depth": [None, 4, 8, 16], "max_features": ["sqrt", 0.5, 1.0]}),
        "MLP": (make_pipeline(StandardScaler(), MLPClassifier(max_iter=2000, random_state=0)),
                {"mlpclassifier__hidden_layer_sizes": [(32,), (64,), (128,), (64, 64)],
                 "mlpclassifier__alpha": loguniform(1e-5, 1e-1)}),
    }


def main():
    rows, best = [], {}
    raw = {}
    for ds in DATASETS:
        X, y = load_classification(ds)
        print(f"\n=== {ds}: X{X.shape}, {len(np.unique(y))} classes")
        outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=0)
        for name, (est, space) in models().items():
            f1s, accs, params = [], [], []
            with Timer() as tm:
                for k, (tr, te) in enumerate(outer.split(X, y)):
                    search = RandomizedSearchCV(est, space, n_iter=N_ITER, scoring="f1_macro", n_jobs=-1,
                                                cv=StratifiedKFold(3, shuffle=True, random_state=k),
                                                random_state=k, error_score=0.0)
                    search.fit(X[tr], y[tr])
                    pred = search.predict(X[te])
                    f1s.append(f1_score(y[te], pred, average="macro"))
                    accs.append(accuracy_score(y[te], pred))
                    params.append(search.best_params_)
            fm, fci = mean_ci(f1s)
            am, _ = mean_ci(accs)
            family = "neurochaos" if "Chaos" in name else "baseline"
            rows.append({"dataset": ds, "model": name, "family": family, "macro_F1": fm, "F1_ci95": fci,
                         "accuracy": am, "time_s": tm.elapsed})
            raw[(ds, name)] = f1s
            best.setdefault(ds, {})[name] = params
            print(f"  {name:16s} F1={fm:.3f} ± {fci:.3f}  acc={am:.3f}  ({tm.elapsed:.0f}s)")
    write_table(rows, OUT / "benchmark")
    write_json(best, OUT / "best_params.json")

    # Rank summary across datasets
    names = list(models())
    ranks = {n: [] for n in names}
    for ds in DATASETS:
        scores = {r["model"]: r["macro_F1"] for r in rows if r["dataset"] == ds}
        order = sorted(names, key=lambda n: -scores[n])
        for i, n in enumerate(order):
            ranks[n].append(i + 1)
    rank_rows = sorted([{"model": n, "mean_rank": float(np.mean(ranks[n])),
                         "mean_F1": float(np.mean([r["macro_F1"] for r in rows if r["model"] == n]))}
                        for n in names], key=lambda r: r["mean_rank"])
    write_table(rank_rows, OUT / "ranking")

    # Heatmap of F1 (models x datasets)
    M = np.array([[next(r["macro_F1"] for r in rows if r["dataset"] == d and r["model"] == n) for d in DATASETS]
                  for n in names])
    fig, ax = plt.subplots(figsize=(1.15 * len(DATASETS) + 3, 5.2))
    im = ax.imshow(M, cmap="seq_blue", vmin=max(0.4, M.min()), vmax=1.0, aspect="auto")
    for i in range(len(names)):
        for j in range(len(DATASETS)):
            best_col = M[:, j].max()
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.8 * 1.0 + 0.2 * max(0.4, M.min()) - 0.1 else "#191919",
                    fontweight="bold" if M[i, j] == best_col else "normal")
    ax.set_xticks(range(len(DATASETS)), DATASETS, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.axhline(4.5, color="white", lw=3)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="macro-F1 (nested CV)")
    ax.set_title("Neurochaos learning vs classical baselines (bold = best per dataset)")
    fig.tight_layout()
    savefig(fig, OUT / "benchmark_heatmap.png")

    # ChaosNet vs ChaosNet-Std improvement
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.38
    for k, (n, c) in enumerate([("ChaosNet", PALETTE[1]), ("ChaosNet-Std", PALETTE[0])]):
        mus = [mean_ci(raw[(d, n)])[0] for d in DATASETS]
        cis = [mean_ci(raw[(d, n)])[1] for d in DATASETS]
        ax.bar(np.arange(len(DATASETS)) + (k - 0.5) * w, mus, w, yerr=cis, capsize=2, color=c,
               edgecolor="white", linewidth=2, label=n)
    ax.set_xticks(range(len(DATASETS)), DATASETS, rotation=20)
    ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1.05); ax.legend(ncol=2, loc="lower right")
    ax.set_title("Effect of feature standardisation on the cosine-similarity ChaosNet")
    fig.tight_layout()
    savefig(fig, OUT / "chaosnet_standardization.png")


if __name__ == "__main__":
    main()
