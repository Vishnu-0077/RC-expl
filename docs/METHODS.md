# Methods

This document gives the mathematical definitions behind every model, metric
and diagnostic in `rcexpl`. The two research tracks, **reservoir computing**
for forecasting dynamical systems and **neurochaos learning** for
classification, are independent: they share only the dataset and plotting
utilities.

---

## Part A: Reservoir computing

### A.1 Echo State Network (leaky integrator)

For an input `u(t) ∈ ℝ^d` and a reservoir state `x(t) ∈ ℝ^N`:

```
x(t) = (1 − a) · x(t−1) + a · tanh( W x(t−1) + W_in [1; u(t)] )
```

* `W` (N×N) is a fixed random recurrent matrix, rescaled to a spectral radius `ρ`.
* `W_in` holds input weights drawn from `U[−σ, σ]` (input scaling `σ`), plus a bias column.
* `a ∈ (0, 1]` is the leak rate. It sets the reservoir's intrinsic time scale.

**Readout.** `ŷ(t) = W_out φ(t)`, with `φ(t) = [1, u(t), x(t)]` (linear) or
`φ(t) = [1, u(t), x(t), x(t)²]` (quadratic). Only `W_out` is trained, by ridge
regression:

```
W_out = argmin_W ‖Φ W − Y‖² + β ‖W‖²   ⇒   W_out = (ΦᵀΦ + βI)⁻¹ ΦᵀY
```

The first `washout` states are discarded so that the arbitrary initial state
does not bias the fit.

**Why a quadratic readout?** A `tanh` reservoir with zero bias is odd-symmetric:
`x(−u) = −x(u)`. A linear readout then cannot represent even functions of the
input, for example the `z` coordinate of Lorenz, which depends on `x·y`.
Squaring the states breaks this symmetry (Herteux & Räth 2020).

**Closed-loop (autonomous) forecasting.** After synchronising the reservoir on
a warm-up segment of true data, the prediction is fed back as the next input:
`u(t+1) := ŷ(t)`. The trained reservoir then acts as an autonomous dynamical
system, a learned surrogate of the true one.

### A.2 Deep ESN

`L` reservoirs are stacked. Layer 1 receives `u(t)` and layer `l > 1` receives
`x_{l−1}(t)`. The readout sees all the layers: `φ = [1, u, x_1, …, x_L]`. The
layers develop progressively slower, more abstract time scales
(Gallicchio, Micheli & Pedrelli 2017).

### A.3 ESN ensemble with consensus feedback (contribution)

`M` ESNs with different random seeds are trained independently. In closed loop,
member `m` produces `ŷ_m(t)`, and the **consensus** `ȳ(t) = (1/M) Σ_m ŷ_m(t)` is
fed back to *every* member:

```
x_m(t+1) = F_m(x_m(t), ȳ(t))
```

Averaging the outputs of independent closed loops would let each member drift
onto its own trajectory. Consensus feedback instead keeps all members
synchronised on a common trajectory, and their independent readout errors
partially cancel at each step.

### A.4 Next-Generation RC (NG-RC)

Gauthier et al. (2021) replace the random reservoir with an explicit feature
vector:

```
O_lin(t)    = [u(t), u(t−s), …, u(t−(k−1)s)]           (k delay taps, spacing s)
O_nonlin(t) = all unique monomials of O_lin of degree 2..p
φ(t)        = [1, O_lin(t), O_nonlin(t)]
```

The readout predicts the increment `u(t+1) − u(t)`, or the next value directly.
NG-RC is equivalent to a nonlinear vector autoregression (NVAR), so it is
exact whenever the true update rule is a low-degree polynomial in the delay
taps, as for the logistic and Hénon maps.

### A.5 Reservoir topologies

| name | construction |
|---|---|
| `dense` | all N² weights `U[−1, 1]` |
| `erdos_renyi` | each weight non-zero with probability `density` |
| `small_world` | Watts–Strogatz ring with degree k, each edge rewired with probability `p` |
| `ring` | simple cycle reservoir, SCR: `W[i+1, i] = r` (Rodan & Tiňo 2011) |
| `cycle_jumps` | SCR plus bidirectional jumps of length ≈ √N (CRJ, Rodan & Tiňo 2012) |
| `delay_line` | feed-forward chain (nilpotent, so it is scaled by its operator norm) |

Every matrix is rescaled to spectral radius `ρ`.

### A.6 Metrics

* **NRMSE** = `RMSE / std(y)`, averaged over dimensions.
* **Valid prediction time (VPT)**: the first time `t` at which
  `‖y(t) − ŷ(t)‖ / sqrt(⟨‖y − ⟨y⟩‖²⟩) > 0.4` (Pathak et al. 2018).
  We report it in **Lyapunov times** `t · λ_max · Δt`, so the numbers are
  comparable across systems. One Lyapunov time is the horizon over which errors
  grow by a factor `e`.
* **Largest Lyapunov exponent `λ_max`**: computed by us for every system with
  Benettin's method, propagating a tangent vector through the analytic Jacobian
  with RK4 integration. For maps we average `log|Df|` along the orbit. For a
  *trained ESN* we run the two-trajectory Benettin method on the joint closed-loop
  state `(x, ŷ)`.
* **Climate metrics**: the Wasserstein-1 distance between invariant densities,
  the Wasserstein-1 distance between the distributions of the Lorenz-map maxima
  `z_max`, and the mean absolute difference of log power spectra.

### A.7 Reservoir diagnostics

* **Linear memory capacity** (Jaeger 2002):
  `MC = Σ_k r²( u(t−k), ŷ_k(t) )`, where `ŷ_k` is a linear readout trained to
  recall the input `k` steps back from i.i.d. `U[−1, 1]` inputs. Bound: `MC ≤ N`.
* **Quadratic capacity** (Dambre et al. 2012): the same procedure for the targets
  `P₂(u(t−k)) = (3u² − 1)/2` and the cross terms `u(t−i) u(t−j)`. Capacities are
  measured on held-out data and thresholded at 0.01 to remove finite-sample bias.
* **Effective dimension**: the participation ratio `(Σλ_i)² / Σλ_i²` of the
  eigenvalues of the state covariance matrix.
* **Echo-state-property test**: `‖x₁(t) − x₂(t)‖` for two different initial
  states driven by the same input. Exponential decay means fading memory.

### A.8 Hyperparameter search

Random search, parallelised with joblib, with log-uniform priors on scale
parameters. **The objective is the closed-loop VPT** on a validation block
disjoint from both the training and test data, averaged over 5 forecast
windows. Experiment RC-04 shows that the usual one-step NRMSE objective is only
a coarse proxy: it correlates with VPT overall, but cannot tell apart
configurations whose forecasting horizons differ by about 2×.

---

## Part B: Neurochaos learning

### B.1 GLS chaotic neuron

The Generalised Lüroth Series (GLS) skew-tent map with skew `b ∈ (0, 1)` is

```
T_b(x) = x / b                 if x < b
         (1 − x) / (1 − b)     if x ≥ b
```

It is chaotic for every `b`, with Lyapunov exponent
`λ(b) = −b ln b − (1−b) ln(1−b)` (maximal, `ln 2`, at `b = ½`), and it preserves
the uniform density on `[0, 1]`. We also provide the skew-binary map
`(x − b)/(1 − b)` on the right branch, and the logistic map `4x(1−x)`.

*Floating-point note.* With `b = 0.5` exactly, each iteration of the map shifts
one bit out of the mantissa, so every orbit reaches 0 within about 53 steps.
We use `b = 0.499` by default (NL-08 panel f checks the uniform invariant
density).

### B.2 ChaosFEX feature extraction

Each input feature, min-max normalised to `[0, 1]`, is a stimulus `x` for one
GLS neuron with initial activity `q`. The neuron iterates until it fires,
which happens when it enters the ε-neighbourhood of the stimulus:

```
N(x) = min{ n ≥ 0 : |T_b^n(q) − x| < ε }      (firing time; capped at max_iter)
A    = (T^0(q), …, T^N(q))                    (neural trace)
```

| feature | definition |
|---|---|
| firing time | `N` |
| firing rate | `#{a ∈ A : a > b} / |A|` |
| energy | `Σ_{a∈A} a²` |
| entropy | binary Shannon entropy (bits) of the symbolic sequence `𝟙[a > b]` |
| variance | `var(A)` (an extra feature we add) |

The topological transitivity of `T_b` guarantees that the neuron fires
eventually, for any stimulus and any `ε > 0`.

### B.3 Shared-trajectory algorithm (contribution)

Every neuron in a population starts at the **same** `q` and iterates the
**same** deterministic map, so all neurons follow *one* trajectory
`τ_n = T^n(q)` and differ only in their stopping index `N(x)`. Therefore:

1. Simulate `τ_0 … τ_{max_iter}` once.
2. Precompute the prefix sums `S_>[n] = Σ_{i<n} 𝟙[τ_i > b]`, `S_1[n] = Σ τ_i`
   and `S_2[n] = Σ τ_i²`.
3. Find `N(x)` for all stimuli at once with a blocked, vectorised first-hit
   search that only revisits unresolved stimuli.
4. Every feature becomes an O(1) look-up: for example, energy = `S_2[N+1]` and
   variance = `S_2[N+1]/(N+1) − (S_1[N+1]/(N+1))²`.

This is mathematically identical to the naive simulation. The unit tests check
agreement to `1e-9` for all three maps and several values of ε. The naive cost
is `Σ_i N(x_i)` interpreted map evaluations. The fast cost is `max_iter`
evaluations plus vectorised array work. NL-07 measures the speed-up.

A consequence worth stating explicitly: **every ChaosFEX feature is a
deterministic function of the single integer `N(x)`.** The encoding is
therefore a fixed, highly non-monotone scalar function of each input
coordinate, a "chaotic hash" of `x`, whose resolution is set by `ε`.

### B.4 Multi-population ChaosFEX (contribution)

Using `P` initial activities `q_1 … q_P` gives `P` independent trajectories,
and therefore `P` different encodings `N_p(x)` of each input. Their features
are concatenated. Because each `N_p` aliases different stimuli together, the
joint code `(N_1(x), …, N_P(x))` is far more injective than any single one.
NL-03 and NL-06 show that this both raises accuracy and removes most of the
sensitivity to the choice of `q`.

### B.5 Classifiers

* **ChaosNet** (Balakrishnan et al. 2019) computes the class mean of the
  ChaosFEX vectors and predicts the class with maximal cosine similarity.
* **ChaosNet-Std** (contribution) z-scores the features with training
  statistics before taking the class means and the cosine. Without this, the
  large-magnitude features (firing time, energy) dominate the angle.
* **ChaosFEX + ML**: ChaosFEX → StandardScaler → SVM, logistic regression, or
  any scikit-learn estimator (Harikrishnan & Nagaraj 2021).

### B.6 Evaluation protocol

* NL-01 uses **nested cross-validation**: an outer 5-fold stratified CV
  repeated twice, with an inner randomised 3-fold search (25 candidates) over
  `(q, b, ε)` and the classifier's parameters. Baselines get the same budget.
* The metric is macro-F1, as in the neurochaos-learning literature, plus
  accuracy.
* NL-02 (low data) uses **fixed** hyperparameters for all models, because
  tuning is impossible with one sample per class.
