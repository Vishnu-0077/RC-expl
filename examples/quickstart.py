"""Five-minute tour of rcexpl. Run with:  python examples/quickstart.py"""
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from rcexpl.datasets import load_classification, load_system
from rcexpl.metrics import valid_prediction_time
from rcexpl.neurochaos import ChaosNet, chaosfex_pipeline
from rcexpl.reservoir import ESN, NGRC, ESNEnsemble

# ---------------------------------------------------------------- Reservoir computing
ts = load_system("lorenz", n=10000)                 # RK4-integrated, with Lyapunov exponent
z = (ts.data - ts.data[:6000].mean(0)) / ts.data[:6000].std(0)
train, test = z[:6000], z[6000:]
lt = ts.lyapunov_time_steps()                       # samples per Lyapunov time
print(f"Lorenz: lambda_max = {ts.lyapunov:.3f}, 1 Lyapunov time = {lt:.0f} steps")

models = {
    "ESN (quadratic readout)": ESN(n_units=400, spectral_radius=0.9, leak_rate=0.6, input_scaling=0.3,
                                   readout="quadratic", topology_kwargs={"density": 0.02}),
    "ESN ensemble (consensus)": ESNEnsemble(n_members=5, n_units=400, spectral_radius=0.9, leak_rate=0.6,
                                            input_scaling=0.3, readout="quadratic",
                                            topology_kwargs={"density": 0.02}),
    "NG-RC": NGRC(k=2, s=1, degree=2, ridge=1e-6),
}
for name, model in models.items():
    model.fit(train[:-1], train[1:])
    forecast = model.generate(1000, warmup=test[:300])  # autonomous closed-loop forecast
    vpt = valid_prediction_time(test[300:1300], forecast) / lt
    print(f"  {name:26s} valid prediction time = {vpt:5.2f} Lyapunov times")

# ---------------------------------------------------------------- Neurochaos learning
X, y = load_classification("wine")
for name, clf in {
    "ChaosNet": ChaosNet(q=0.34, b=0.499, epsilon=0.18),
    "ChaosNet-Std": ChaosNet(q=0.34, b=0.499, epsilon=0.18, standardize=True),
    "Multi-population ChaosFEX + SVM": chaosfex_pipeline(SVC(), q=[0.13, 0.37, 0.61, 0.83], epsilon=0.1),
}.items():
    f1 = cross_val_score(clf, X, y, cv=5, scoring="f1_macro").mean()
    print(f"  wine | {name:32s} macro-F1 = {f1:.3f}")
