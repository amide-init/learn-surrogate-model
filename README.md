# learn-surrogate

A Python project for learning surrogate-assisted optimisation from absolute zero — with a **neural network as the primary surrogate model**. Built so that completing all lessons gives you enough theory, code, and experiments to write a research paper.

No prior knowledge of machine learning or optimisation is assumed.

---

## Primary Goal

Build and evaluate a **neural network surrogate** for expensive black-box optimisation, then benchmark it against Gaussian Processes and CMA-ES on the COCO/BBOB suite.

> Gaussian Processes are taught first because they build the intuition you need to understand why and how a neural network works as a surrogate. They are the baseline, not the contribution.

---

## Paper Angle

**"Neural Network Surrogates with Uncertainty Estimation for Bayesian Optimisation on BBOB: MC Dropout vs. Deep Ensembles"**

- Surrogate models: feedforward neural network (PyTorch) with two uncertainty methods:
  - **MC Dropout** (Gal & Ghahramani 2016) — single network, dropout at inference
  - **Deep Ensembles** (Lakshminarayanan 2017) — 5 independently trained networks
- Acquisition function: Expected Improvement using NN predictive mean and variance
- Benchmark: COCO BBOB 24-function noiseless suite, dimensions 2 and 5
- Baselines: GP-BO, CMA-ES, random search

> Both uncertainty methods are taught and compared. The comparison between MC Dropout and Deep Ensembles is itself a contribution.

---

## How to Use This Roadmap

Each lesson has:
- **What you will learn** — the concept in plain English
- **What you will build** — a concrete deliverable
- **Why it matters** — which part of your paper it feeds

Work through lessons in order. Do not skip — each one builds on the last.

---

## Lesson 1 — Python and Scientific Computing Basics

**What you will learn:** The tools every lesson depends on.

- NumPy: arrays, vectorised math, `linspace`, `meshgrid`
- SciPy: built-in optimisers, `minimize`
- Matplotlib: line plots, contour plots, subplots

**What you will build:** `lesson-1/main.py`
- Plot `sin(x)` and `x²` with NumPy
- 2D contour + 3D surface of `f(x,y) = x² + y²`
- Use `scipy.optimize.minimize` to find a minimum and mark it on the plot

**Why it matters:** Every script in every lesson uses these three libraries.

---

## Lesson 2 — What is Optimisation?

**What you will learn:** The problem surrogates are designed to solve.

- Gradient descent: update rule, step size, convergence
- Why gradient methods fail: black-box functions, noise, discontinuities
- What "expensive" means: each evaluation costs real time, money, or compute

**What you will build:** `lesson-2/main.py`
- Implement gradient descent by hand on `f(x) = x²`
- Run `scipy.optimize.minimize` on Rosenbrock and Ackley
- Add noise to a function and show that gradient descent breaks

**Why it matters:** Your paper's introduction explains why standard optimisers are not enough. This lesson gives you that argument.

---

## Lesson 3 — What is a Surrogate?

**What you will learn:** The central idea of the whole project.

- A surrogate is a cheap model that approximates an expensive function
- The loop: evaluate a few points → fit model → use model to pick next point
- First surrogate: polynomial regression (the simplest possible case)

**What you will build:** `lesson-3/main.py`
- Sample 5 points from `f(x) = sin(x) + 0.1x²`
- Fit a degree-3 polynomial with NumPy `polyfit`
- Plot: true function vs. surrogate — show where it fails

**Why it matters:** Seeing a simple surrogate fail motivates everything that follows.

---

## Lesson 4 — Statistics Refresher

**What you will learn:** The math behind uncertainty estimates — needed for both GP and neural network surrogates.

- Mean, variance, standard deviation
- Multivariate Gaussian: mean vector, covariance matrix
- Conditional distributions: P(Y | X = x)
- Sampling from a Gaussian

**What you will build:** `lesson-4/main.py`
- Plot 1D Gaussians with different means and variances
- Sample from a 2D Gaussian, plot scatter and confidence ellipse
- Visualise a conditional distribution

**Why it matters:** Both GP uncertainty and MC Dropout uncertainty are Gaussian. You need this to understand what your neural network is outputting.

---

## Lesson 5 — Kernels and Similarity

**What you will learn:** How GPs measure similarity — and why neural networks learn this automatically.

- What is a kernel?
- RBF, Matern-3/2, Matern-5/2 kernels
- Length-scale hyperparameter
- Sampling functions from a GP prior

**What you will build:** `lesson-5/main.py`
- Implement RBF kernel from scratch
- Plot kernel matrices as heatmaps
- Sample and plot random functions from different kernels

**Why it matters:** Understanding kernels explains the GP baseline you will compare your NN against. It also helps you justify why NN can generalise better in higher dimensions.

---

## Lesson 6 — Gaussian Process Regression (Baseline)

**What you will learn:** How to build the baseline model your neural network will compete against.

- GP as a distribution over functions
- GP posterior: predictive mean and variance
- Hyperparameter optimisation via log marginal likelihood
- Strengths: data-efficient, principled uncertainty. Weaknesses: scales as O(n³), struggles in high dimensions

**What you will build:** `lesson-6/main.py`
- GP class from scratch: `fit(X, y)` and `predict(X_new)`
- Test on Forrester function (standard 1D benchmark)
- Plot: data, true function, GP mean, ±2σ uncertainty band

**Why it matters:** This is your primary baseline. Your paper's claim is that a neural network surrogate is competitive with or better than GP in specific settings.

---

## Lesson 7 — Acquisition Functions

**What you will learn:** How a surrogate tells the optimiser where to evaluate next.

- Exploration vs. exploitation trade-off
- Probability of Improvement (PI)
- Expected Improvement (EI) — the acquisition function your paper will use
- Upper Confidence Bound (UCB)

| Function | Key property |
|---|---|
| PI | Conservative, fast convergence |
| EI | Balanced — most common in papers |
| UCB | Explicit exploration control via κ |

**What you will build:** `lesson-7/main.py`
- Implement PI, EI, UCB from the GP mean and variance
- Plot all three alongside the surrogate on a 1D example

**Why it matters:** EI is the acquisition function you will plug your neural network into. This lesson makes the connection clear.

---

## Lesson 8 — Bayesian Optimisation Loop

**What you will learn:** How surrogate + acquisition combine into a full algorithm.

```
1. Evaluate f at initial points (Latin Hypercube Sampling)
2. Fit surrogate to observations
3. Optimise acquisition function → find next point x*
4. Evaluate f(x*)  ← expensive step
5. Add (x*, f(x*)) to data → go to 2
6. Stop when budget is exhausted
```

**What you will build:** `lesson-8/main.py`
- Full BO loop using GP as the surrogate
- LHS initial design, multi-start L-BFGS-B for acquisition optimisation
- Test on Forrester (1D), Branin (2D), Hartmann-3 (3D)
- Convergence plot: best value found vs. number of evaluations

**Why it matters:** This is the shell your neural network will slot into. In Lesson 10 you replace the GP with a neural network and everything else stays the same.

---

## Lesson 9 — PyTorch Basics for Surrogate Models

**What you will learn:** The minimum PyTorch needed to build a neural network surrogate.

- Tensors vs. NumPy arrays
- Building a feedforward network with `nn.Module`
- Training loop: forward pass, loss, backward pass, optimiser step
- Saving and loading a model

**What you will build:** `lesson-9/main.py`
- Feedforward network that fits a 1D regression curve
- Training loop with MSE loss and Adam optimiser
- Plot: true function vs. NN prediction after training

**Why it matters:** Direct preparation for Lesson 10. No surrogate context yet — just the tool.

---

## Lesson 10 — Neural Network Surrogate + MC Dropout

**What you will learn:** The first uncertainty method — a single network that estimates its own uncertainty using dropout at inference time.

**Why plain NN fails as a surrogate:**
- A standard NN gives one number per input — no uncertainty → pure exploitation → gets stuck

**Solution — MC Dropout (Gal & Ghahramani 2016):**
- Add `Dropout(p=0.1)` layers to the network
- At training time: dropout randomly zeros activations (regularisation)
- At prediction time: keep dropout ON and run N=50 forward passes
- Mean of N predictions = surrogate mean
- Variance of N predictions = surrogate uncertainty

```
Architecture:
Input(d) → Linear(64) → ReLU → Dropout(0.1)
         → Linear(64) → ReLU → Dropout(0.1)
         → Linear(1)

Prediction:
Run 50 forward passes → mean μ, variance σ²
Feed μ, σ² into EI acquisition
```

**What you will build:** `lesson-10/mc_dropout.py` and `lesson-10/main.py`
- `MCDropoutSurrogate` class: `fit(X, y)` and `predict(X_new, n_samples=50)`
- Plug into the EI acquisition from Lesson 7
- Run the full BO loop with MC Dropout NN instead of GP
- Plot: surrogate mean, ±2σ uncertainty band, EI landscape

**Why it matters:** MC Dropout is simple, cheap, and well-cited. It is the first NN uncertainty method in your paper.

---

## Lesson 11 — Neural Network Surrogate + Deep Ensembles

**What you will learn:** The second uncertainty method — train multiple independent networks and use their disagreement as uncertainty.

**Solution — Deep Ensembles (Lakshminarayanan 2017):**
- Train K=5 identical networks independently with different random seeds
- At prediction time: run all 5 networks on the same input
- Mean of 5 predictions = surrogate mean
- Variance of 5 predictions = surrogate uncertainty (ensemble disagreement)

```
Architecture (×5 independent networks):
Input(d) → Linear(64) → ReLU
         → Linear(64) → ReLU
         → Linear(1)

Prediction:
5 networks → 5 outputs → mean μ, variance σ²
Feed μ, σ² into EI acquisition
```

**MC Dropout vs. Deep Ensembles:**

| Property | MC Dropout | Deep Ensembles |
|---|---|---|
| Networks trained | 1 | 5 |
| Training cost | Low | 5× higher |
| Prediction cost | N forward passes | 5 forward passes |
| Uncertainty quality | Approximate | Better calibrated |
| Implementation | Simple | Straightforward |
| Paper citation | Gal 2016 | Lakshminarayanan 2017 |

**What you will build:** `lesson-11/deep_ensemble.py` and `lesson-11/main.py`
- `DeepEnsembleSurrogate` class: trains 5 networks, returns mean + variance
- Run the full BO loop with Deep Ensemble NN
- Side-by-side comparison: MC Dropout uncertainty vs. Deep Ensemble uncertainty on the same data
- Plot: both uncertainty bands on the same function

**Why it matters:** Deep Ensembles typically produce better-calibrated uncertainty. Comparing them gives you a second result in your paper.

---

## Lesson 12 — Comparing Uncertainty Methods

**What you will learn:** Which uncertainty method works better as a surrogate, and when.

**What you will compare:**
- MC Dropout (Lesson 10) vs. Deep Ensembles (Lesson 11) vs. GP (Lesson 6)
- On the same test functions, same budget, same acquisition function (EI)

**Metrics:**
- Best value found (optimisation quality)
- Surrogate RMSE (how accurate is the fit)
- Uncertainty calibration (does high variance correspond to high error?)
- Wall-clock time per iteration

**What you will build:** `lesson-12/main.py`
- Run all three on Forrester (1D), Branin (2D), Hartmann-3 (3D)
- Calibration plot: predicted σ vs. actual error
- Convergence curves: best value vs. evaluations for all three
- Summary table: RMSE, best value, time

**Why it matters:** This comparison is the core result of your paper — it directly answers "which NN uncertainty method works best as a surrogate?"

---

## Lesson 13 — Surrogate Control Strategies (Critical Concept)

**What you will learn:** The most commonly misunderstood concept in surrogate-assisted optimisation — how many evaluations should actually hit the expensive function?

**The core rule:**
> A surrogate is only useful if it **reduces** the number of real function evaluations. If every candidate still gets evaluated with the real function, the surrogate saves nothing.

**Three control strategies:**

| Strategy | How it works | Real eval ratio |
|---|---|---|
| **No control (wrong)** | Surrogate scores candidates but all get real evals | ~100% |
| **Generation-based** | Every k-th generation uses real evals; others use surrogate only | ~1/k |
| **Individual-based** | Only top-μ candidates per generation get real evals | μ/λ (e.g. 10%) |
| **Pre-selection** | Generate large pool → surrogate filters → small set gets real evals | ~1-5% |

**The AFN-CMA-ES bug (Issue #1 from the reviewed paper):**
```python
# BROKEN — surrogate reorders 8 candidates, all 8 still get real evals
solutions = es.ask()                        # returns 8 candidates
top_indices = np.argsort(scores)[-8:]       # picks 8 FROM 8 — no-op!
for x in solutions:
    f = objective_function(x)               # 100% real evaluations!

# CORRECT — surrogate filters 1000 down to 8, only 8 get real evals
large_pool = es.ask(number=1000)            # 1000 candidates
mean, std = surrogate.predict(large_pool)
top_8 = select_top_by_ucb(mean, std, k=8)
for x in top_8:
    f = objective_function(x)               # 0.8% real evaluations ✓
```

**What you will build:** `lesson-13/main.py`
- Implement the correct pre-selection loop from scratch (no CMA-ES dependency yet)
- Track and plot: surrogate evaluations vs. real evaluations per iteration
- Compare convergence: no-control vs. pre-selection on Forrester and Branin
- Compute surrogate control ratio (SCR = real evals / total candidates) per run

**Why it matters:** This is the exact issue the paper reviewer flagged. Understanding SCR is essential before running COCO experiments — without it, your algorithm is not a surrogate-assisted method at all.

---

## Lesson 14 — Surrogate Model Comparison (All Methods)

**What you will learn:** How all surrogate models compare on a common benchmark.

| Model | Uncertainty method | Package |
|---|---|---|
| **NN + MC Dropout** | Dropout at inference | `torch` |
| **NN + Deep Ensembles** | Ensemble disagreement | `torch` |
| Gaussian Process | Exact posterior | `sklearn` |
| Random Forest | Tree variance | `sklearn` |
| RBF Interpolation | None | `scipy` |

**What you will build:** `lesson-14/main.py`
- All five models, same test functions, same budget, same EI acquisition
- Correct pre-selection loop from Lesson 13 applied to all models
- Metrics: best value found, RMSE, SCR, wall-clock time
- Box plots across 10 independent runs, LaTeX-ready comparison table

**Why it matters:** Positions NN methods in the full landscape of surrogate models. Strengthens your related work section.

---

## Lesson 15 — Noise and High Dimensions

**What you will learn:** Where NN surrogates beat GP — the "when to use NN" claim.

**Noise:**
- GP with noise: add σ_n² to kernel diagonal — exact but still O(n³)
- NN with noise: train on noisy data — scales better, no explicit noise model needed
- Compare: NN-BO vs. GP-BO on noisy Forrester at σ = 0.1, 0.5, 1.0

**High dimensions:**
- GP cost: O(n³) in data, kernel matrix grows with dimension → breaks at d ≥ 20
- NN cost: O(n) in data, scales naturally with input dimension
- Experiment: d = 2, 5, 10, 20 — plot best value vs. dimension for NN and GP

**What you will build:** `lesson-15/main.py`
- Noisy BO: MC Dropout vs. Deep Ensemble vs. GP on noisy Forrester
- Scalability: NN vs. GP across dimensions

**Why it matters:** This is the "advantages" section of your paper — where your contribution clearly wins.

---

## Lesson 16 — COCO / BBOB Benchmark

**What you will learn:** How to evaluate on the community-standard benchmark for fair comparison with published work.

- COCO framework: `cocoex` (run) + `cocopp` (plot)
- BBOB suite: 24 noiseless functions, dimensions 2 and 5
- Budget: `500 × dimension` evaluations
- Metrics: ERT (Expected Running Time), ECDF curves

**What you will build:** `lesson-16/main.py`
- Wrap MC Dropout NN-BO and Deep Ensemble NN-BO in COCO-compatible interfaces
- Apply correct SCR pre-selection — surrogate must save real evaluations
- Run on all 24 BBOB functions, dimensions 2 and 5
- Run GP-BO and CMA-ES as baselines
- Generate ECDF plots with `cocopp`

**COCO key concepts:**

| Term | Meaning |
|---|---|
| `budget_multiplier` | Max evaluations = multiplier × dimension |
| Target precision | f(x) − f\_opt < ∆f |
| ECDF | Fraction of (function, instance, target) triples solved |
| ERT | Expected evaluations to reach a target |
| SCR | Surrogate control ratio — must be < 20% |

**Why it matters:** ECDF plots are mandatory for GECCO/CEC papers. Your paper needs these to be accepted.

---

## Lesson 17 — Statistical Analysis and Paper Figures

**What you will learn:** How to turn raw results into defensible claims.

- Ablation: MC Dropout → deterministic NN (remove uncertainty)
- Ablation: Deep Ensembles → single network (remove ensemble)
- Ablation: EI → random acquisition
- Wilcoxon signed-rank test: MC Dropout vs. Deep Ensemble vs. GP
- Friedman test across all algorithms
- Publication-quality figures

**What you will build:** `lesson-17/main.py`
- All ablation runs including: pre-selection vs. no pre-selection (SCR ablation)
- Statistical significance tests
- LaTeX-ready table with means, std, significance symbols (†, ‡)

**Why it matters:** Without statistical tests, claims of "better" will be rejected by reviewers.

---

## Lesson 18 — Writing the Research Paper

**Paper structure:**

```
1. Introduction
   ├── Problem: expensive black-box optimisation
   ├── Gap: GPs scale poorly; plain NNs have no uncertainty
   ├── Contribution: compare MC Dropout vs. Deep Ensembles as NN surrogates
   └── Paper outline

2. Background
   ├── Black-box optimisation (Lesson 2)
   ├── Surrogate-assisted optimisation (Lesson 3)
   ├── Gaussian Processes (Lesson 6) — baseline
   └── Bayesian Optimisation loop (Lesson 8)

3. Related Work
   ├── GP-based BO
   ├── Neural network surrogates (prior work)
   ├── MC Dropout — Gal & Ghahramani (2016)
   └── Deep Ensembles — Lakshminarayanan et al. (2017)

4. Proposed Methods
   ├── Shared: network architecture, EI acquisition, BO loop
   ├── Method A: MC Dropout (Lesson 10)
   └── Method B: Deep Ensembles (Lesson 11)

5. Experimental Setup
   ├── BBOB test suite (Lesson 15)
   ├── Baselines: GP-BO, CMA-ES, random search
   └── Metrics: ERT, ECDF, best-so-far

6. Results and Discussion
   ├── MC Dropout vs. Deep Ensembles — uncertainty quality (Lesson 12)
   ├── All surrogates comparison (Lesson 13)
   ├── Noise and scalability (Lesson 14)
   ├── ECDF plots (Lesson 15)
   ├── Ablation study (Lesson 16)
   ├── Statistical significance (Lesson 16)
   └── Limitations

7. Conclusion and Future Work

References
```

**Target venues:**
- GECCO Workshop — most accessible, good for early work
- CEC (Congress on Evolutionary Computation) — standard venue
- GECCO Main Track — needs strong COCO results
- PPSN (Parallel Problem Solving from Nature) — biennial, rigorous

---

## Setup

### 1. Create and activate the virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your terminal prompt — this confirms the environment is active.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

This installs everything needed for all 18 lessons.

### 3. Verify the setup

```bash
python -c "import numpy, scipy, matplotlib, sklearn, torch; print('All good')"
```

### 4. Launch Jupyter for notebook lessons

```bash
jupyter notebook
```

> **Note:** Always activate the virtual environment (`source .venv/bin/activate`) before running any lesson script or launching Jupyter. You need to do this once per terminal session.

### Deactivate when done

```bash
deactivate
```

---

## Dependencies

| Package | Purpose | First used |
|---|---|---|
| `numpy` | Arrays, linear algebra | Lesson 1 |
| `scipy` | Optimisers, statistics, LHS | Lesson 2 |
| `matplotlib` | All plots | Lesson 1 |
| `scikit-learn` | GP baseline, Random Forest | Lesson 6 |
| `torch` | Neural network surrogate (main model) | Lesson 9 |
| `scikit-optimize` | Additional BO baseline | Lesson 11 |
| `coco-experiment` | COCO experiment runner | Lesson 13 |
| `cocopp` | COCO post-processing, ECDF plots | Lesson 13 |
| `cma` | CMA-ES baseline optimiser | Lesson 13 |

---

## Directory Structure

```
learn-surrogate/
├── lesson-1/          # Python and scientific computing basics
├── lesson-2/          # Optimisation fundamentals
├── lesson-3/          # First surrogate (polynomial)
├── lesson-4/          # Statistics refresher
├── lesson-5/          # Kernels
├── lesson-6/          # GP regression (baseline)
├── lesson-7/          # Acquisition functions
├── lesson-8/          # BO loop with GP
├── lesson-9/          # PyTorch basics
├── lesson-10/         # NN surrogate + MC Dropout
├── lesson-11/         # NN surrogate + Deep Ensembles
├── lesson-12/         # MC Dropout vs. Deep Ensembles vs. GP
├── lesson-13/         # All surrogates comparison
├── lesson-14/         # Noise and high dimensions
├── lesson-15/         # COCO benchmark
├── lesson-16/         # Statistical analysis and paper figures
├── lesson-17/         # Paper writing guide
├── results/           # Benchmark outputs (gitignored)
└── requirements.txt
```

---

## Key References

**Neural Network Surrogates**
- Snoek et al. (2015). Scalable and Accurate Deep Learning with Stochastic Depth
- Springenberg et al. (2016). Bayesian Optimization with Robust Bayesian Neural Networks

**MC Dropout for Uncertainty**
- Gal & Ghahramani (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning

**Deep Ensembles**
- Lakshminarayanan et al. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

**Bayesian Optimisation**
- Frazier (2018). [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811)

**Gaussian Processes (Baseline)**
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning* — free PDF at gaussianprocess.org

**COCO Benchmark**
- Hansen et al. (2021). [COCO: A Platform for Comparing Continuous Optimizers](https://arxiv.org/abs/1603.08785)
