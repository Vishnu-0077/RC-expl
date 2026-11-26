"""RC-01: Autonomous forecasting benchmark on six chaotic systems.

Each (system, model) pair is tuned by random search, maximising the mean
valid prediction time (VPT) on a validation block. The tuned model is then
evaluated on 20 closed-loop forecasts started inside a held-out test block.

Models
  ESN           leaky ESN with a linear readout
  ESN-Q         leaky ESN with the quadratic readout [1, u, x, x^2]
  DeepESN       2 stacked reservoirs (same total number of units)
  ESN-Ensemble  5 ESN-Q members with consensus feedback
  NG-RC         next-generation RC (delays plus polynomial monomials)

Outputs go to results/reservoir/rc01_benchmark/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (PALETTE, QUICK, Timer, mean_ci, outdir, plt, savefig, standardize,  # noqa: E402
                     write_json, write_table)

import numpy as np  # noqa: E402

from rcexpl.datasets import load_system  # noqa: E402
from rcexpl.metrics import nrmse, valid_prediction_time  # noqa: E402
from rcexpl.reservoir import ESN, NGRC, ESNEnsemble  # noqa: E402
from rcexpl.reservoir.tuning import Choice, IntUniform, LogUniform, Uniform, random_search  # noqa: E402

OUT = outdir("reservoir", "rc01_benchmark")
N_TRIALS = 6 if QUICK else 40
N_TEST = 5 if QUICK else 20
UNITS = 400

# system -> (generator kwargs, n_train, n_val, warmup, horizon)
SETUP = {
    "logistic": (dict(n=8000), 4000, 1500, 100, 300),
    "tent": (dict(n=8000), 4000, 1500, 100, 300),
    "henon": (dict(n=8000), 4000, 1500, 100, 300),
    "mackey_glass": (dict(n=12000), 5000, 3000, 300, 1500),
    "lorenz": (dict(n=14000), 6000, 3000, 300, 1200),
    "rossler": (dict(n=22000), 8000, 5000, 300, 3000),
}

ESN_SPACE = dict(spectral_radius=Uniform(0.05, 1.4), leak_rate=Uniform(0.05, 1.0),
                 input_scaling=LogUniform(0.02, 2.0), bias_scaling=Uniform(0.0, 1.0),
                 ridge=LogUniform(1e-10, 1e-2), n_units=UNITS, topology_kwargs={"density": 0.02},
                 washout=200)
NGRC_SPACE = dict(k=IntUniform(1, 4), s=IntUniform(1, 5), degree=Choice([2, 3]),
                  ridge=LogUniform(1e-10, 1e-1), predict_increment=Choice([True, False]))

MODELS = {
    "ESN": (lambda **p: ESN(readout="linear", **p), ESN_SPACE),
    "ESN-Q": (lambda **p: ESN(readout="quadratic", **p), ESN_SPACE),
    "DeepESN": (lambda **p: ESN(readout="quadratic", depth=2, **p),
                {**ESN_SPACE, "n_units": UNITS // 2, "topology_kwargs": {"density": 0.04}}),
    "NG-RC": (lambda **p: NGRC(**p), NGRC_SPACE),
}


def evaluate(model, test, warmup, horizon, scale, lt):
    starts = np.linspace(0, len(test) - warmup - horizon, N_TEST).astype(int)
    vpts, example = [], None
    for i, s0 in enumerate(starts):
        with np.errstate(all="ignore"):
            pred = model.generate(horizon, warmup=test[s0:s0 + warmup])
        truth = test[s0 + warmup:s0 + warmup + horizon]
        vpts.append(valid_prediction_time(truth, pred, 0.4, scale) / lt)
        if i == 0:
            example = (truth, pred)
    one_step = model.predict(test[warmup:-1], warmup=test[:warmup])
    return vpts, nrmse(test[warmup + 1:], one_step), example


def main():
    rows, best_params, examples, all_vpts = [], {}, {}, {}
    for system, (gen_kw, n_tr, n_val, warmup, horizon) in SETUP.items():
        ts = load_system(system, **gen_kw)
        z, _, _ = standardize(ts.data, n_tr)
        train, val, test = z[:n_tr], z[n_tr:n_tr + n_val], z[n_tr + n_val:]
        scale = float(np.sqrt(np.mean(np.sum((train - train.mean(0)) ** 2, 1))))
        lt = ts.lyapunov_time_steps() or 1.0
        unit = "LT" if ts.lyapunov else "steps"
        print(f"\n=== {system}: lambda={ts.lyapunov}, 1 LT = {lt:.1f} steps, VPT unit = {unit}")
        best_params[system] = {}
        for name, (builder, space) in MODELS.items():
            with Timer() as tm:
                params, score, _ = random_search(builder, space, train, val, n_trials=N_TRIALS, seed=1,
                                                 warmup=warmup, horizon=horizon, scale=scale, n_windows=5)
                model = builder(**params).fit(np.vstack([train, val])[:-1], np.vstack([train, val])[1:])
                vpts, one, ex = evaluate(model, test, warmup, horizon, scale, lt)
            best_params[system][name] = params
            m, ci = mean_ci(vpts)
            rows.append({"system": system, "model": name, f"VPT_mean": m, "VPT_ci95": ci,
                         "VPT_median": float(np.median(vpts)), "unit": unit, "one_step_NRMSE": one,
                         "time_s": tm.elapsed})
            all_vpts[(system, name)] = vpts
            examples[(system, name)] = ex
            print(f"  {name:12s} VPT={m:7.2f} ± {ci:5.2f} {unit}  1-step NRMSE={one:.2e}  ({tm.elapsed:.0f}s)")

        # Ensemble of the tuned ESN-Q configuration
        p = dict(best_params[system]["ESN-Q"])
        with Timer() as tm:
            ens = ESNEnsemble(n_members=5, readout="quadratic", **p)
            ens.fit(np.vstack([train, val])[:-1], np.vstack([train, val])[1:])
            vpts, one, ex = evaluate(ens, test, warmup, horizon, scale, lt)
        m, ci = mean_ci(vpts)
        rows.append({"system": system, "model": "ESN-Ensemble", "VPT_mean": m, "VPT_ci95": ci,
                     "VPT_median": float(np.median(vpts)), "unit": unit, "one_step_NRMSE": one,
                     "time_s": tm.elapsed})
        all_vpts[(system, "ESN-Ensemble")] = vpts
        examples[(system, "ESN-Ensemble")] = ex
        print(f"  {'ESN-Ensemble':12s} VPT={m:7.2f} ± {ci:5.2f} {unit}  1-step NRMSE={one:.2e}  ({tm.elapsed:.0f}s)")

    write_table(rows, OUT / "benchmark")
    write_json(best_params, OUT / "best_params.json")
    plot(rows, all_vpts, examples)


def plot(rows, all_vpts, examples):
    systems = list(SETUP)
    models = ["ESN", "ESN-Q", "DeepESN", "ESN-Ensemble", "NG-RC"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, system in zip(axes.ravel(), systems):
        unit = next(r["unit"] for r in rows if r["system"] == system)
        data = [all_vpts[(system, m)] for m in models]
        for i, (d, c) in enumerate(zip(data, PALETTE)):
            mu, ci = mean_ci(d)
            ax.bar(i, mu, width=0.7, color=c, edgecolor="white", linewidth=2)
            ax.errorbar(i, mu, yerr=ci, color="#444", capsize=3, lw=1)
            jitter = np.random.default_rng(0).uniform(-0.18, 0.18, len(d))
            ax.scatter(i + jitter, d, s=9, color="#333", alpha=0.35, zorder=3, linewidths=0)
        ax.set_xticks(range(len(models)), models, rotation=25, ha="right")
        ax.set_title(system.replace("_", "-").title())
        ax.set_ylabel(f"valid prediction time ({unit})")
    fig.suptitle("Closed-loop forecasting: valid prediction time (mean ± 95% CI, dots = test windows)",
                 fontweight="bold")
    fig.tight_layout()
    savefig(fig, OUT / "vpt_bars.png")

    # Example forecasts: best model per system, first coordinate
    fig, axes = plt.subplots(len(systems), 1, figsize=(11, 2.0 * len(systems)))
    for ax, system in zip(axes, systems):
        sys_rows = [r for r in rows if r["system"] == system]
        best = max(sys_rows, key=lambda r: r["VPT_mean"])["model"]
        truth, pred = examples[(system, best)]
        n = 100 if system in ("logistic", "tent", "henon") else len(truth)
        ax.plot(truth[:n, 0], color="#191919", lw=1.3, label="ground truth")
        ax.plot(pred[:n, 0], color=PALETTE[0], lw=1.3, ls="--", label=f"{best} forecast")
        vpt = valid_prediction_time(truth, pred)
        ax.axvline(vpt, color=PALETTE[1], lw=1, ls=":")
        ax.text(vpt, 1.02, "VPT", color="#666", fontsize=8, ha="center", transform=ax.get_xaxis_transform())
        ax.set_ylabel(system.replace("_", "-"))
        ax.set_xlim(0, n)
        ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.25), fontsize=8)
    axes[-1].set_xlabel("steps after warm-up (autonomous)")
    fig.suptitle("Example autonomous forecasts (best model per system, first coordinate)", fontweight="bold")
    fig.tight_layout()
    savefig(fig, OUT / "example_forecasts.png")


if __name__ == "__main__":
    main()
