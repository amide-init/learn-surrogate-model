# learn-surrogate

A Python project for learning surrogate-assisted optimisation from absolute zero — with a **neural network as the primary surrogate model**. Built so that completing all lessons gives you enough theory, code, and experiments to write a research paper.

No prior knowledge of machine learning or optimisation is assumed.

---

## Primary Goal

Build and evaluate a **neural network surrogate** for expensive black-box optimisation, then benchmark it against Gaussian Processes and CMA-ES on the COCO/BBOB suite.

> Gaussian Processes are taught first because they build the intuition you need to understand why and how a neural network works as a surrogate. They are the baseline, not the contribution.

---

## Paper Angle

**"Neural Network Surrogate with Uncertainty Estimation for Bayesian Optimisation on BBOB"**

- Surrogate model: feedforward neural network (PyTorch) with MC Dropout for uncertainty
- Acquisition function: Expected Improvement using NN predictive mean and variance
- Benchmark: COCO BBOB 24-function noiseless suite, dimensions 2 and 5
- Baselines: GP-BO, CMA-ES, random search

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

## Lesson 10 — Neural Network Surrogate (Main Contribution)

**What you will learn:** How to use a neural network as a surrogate model with uncertainty estimates.

**Why plain NN fails as a surrogate:**
- NN gives a point prediction — no uncertainty → pure exploitation → gets stuck

**Solution — MC Dropout:**
- Add `Dropout(p=0.1)` layers to the network
- At prediction time, keep dropout ON and run N forward passes
- Mean of N predictions = surrogate mean
- Variance of N predictions = surrogate uncertainty

**Network architecture:**
```
Input(d) → Linear(64) → ReLU → Dropout(0.1)
         → Linear(64) → ReLU → Dropout(0.1)
         → Linear(1)
```

**What you will build:** `lesson-10/nn_surrogate.py` and `lesson-10/main.py`
- `NNSurrogate` class with `fit(X, y)` and `predict(X_new, n_samples=50)`
- Returns mean and variance from MC Dropout samples
- Plug into the EI acquisition function from Lesson 7
- Run the full BO loop from Lesson 8 with NN instead of GP
- Plot: NN surrogate fit, uncertainty band, EI landscape

**Why it matters:** This is the core contribution of your paper. Everything before leads here; everything after evaluates this.

---

## Lesson 11 — Surrogate Model Comparison

**What you will learn:** How your neural network compares to other surrogates.

| Model | Uncertainty | Scales to high-d | Package |
|---|---|---|---|
| **NN + MC Dropout** | Yes (approximate) | Yes | `torch` |
| Gaussian Process | Yes (exact) | No (O(n³)) | `sklearn` |
| RBF Interpolation | No | Yes | `scipy` |
| Random Forest | Yes (empirical) | Yes | `sklearn` |

**What you will build:** `lesson-11/main.py`
- Run all four models on the same test functions
- Same initial data, same budget, same acquisition function (EI)
- Metrics: best value found, surrogate RMSE, wall-clock time
- Output: comparison table and box plots

**Why it matters:** Model comparison results are a core section of your paper. The NN should win or tie in mid-to-high dimensions where GP struggles.

---

## Lesson 12 — Handling Noise and High Dimensions

**What you will learn:** How to make your neural network surrogate more robust.

**Noise:**
- Add a noise output head or increase dropout during training
- Compare: NN-BO with and without noise handling on noisy test functions

**High dimensions:**
- NN naturally handles many inputs (no O(n³) bottleneck)
- Add a larger first layer or use an embedding for high-d inputs
- Experiment: run at d = 2, 5, 10, 20 and plot best value vs. dimension

**What you will build:** `lesson-12/main.py`
- Noisy BO experiment: NN vs. GP on noisy Forrester
- Scalability experiment: NN vs. GP across dimensions

**Why it matters:** Shows the cases where NN beats GP — the "when to use NN" claim in your paper.

---

## Lesson 13 — COCO / BBOB Benchmark

**What you will learn:** How to evaluate your algorithm on the community-standard benchmark.

- COCO framework: `cocoex` (run) + `cocopp` (plot)
- BBOB suite: 24 noiseless functions across different landscape types
- Budget: `100 × dimension` evaluations
- Metrics: ERT, ECDF

**What you will build:** `lesson-13/main.py`
- Wrap your NN-BO from Lesson 10 in a COCO-compatible interface
- Run on all 24 BBOB functions, dimensions 2 and 5
- Run the same for GP-BO and CMA-ES as baselines
- Generate ECDF plots with `cocopp`

**COCO key concepts:**

| Term | Meaning |
|---|---|
| `budget_multiplier` | Max evaluations = multiplier × dimension |
| Target precision | f(x) − f\_opt < ∆f |
| ECDF | Fraction of (function, instance, target) triples solved |
| ERT | Expected evaluations to reach a target |

**Why it matters:** ECDF plots are mandatory for GECCO/CEC papers. This is the experiment your paper's results section is built on.

---

## Lesson 14 — Statistical Analysis and Paper Figures

**What you will learn:** How to turn raw results into defensible scientific claims.

- Ablation study: remove MC Dropout → compare to deterministic NN
- Ablation: swap EI → random acquisition
- Wilcoxon signed-rank test between NN-BO and GP-BO
- Friedman test across all algorithms
- Publication-quality figures: ECDF, convergence curves, box plots

**What you will build:** `lesson-14/main.py`
- All ablation runs
- Statistical significance tests
- LaTeX-ready results table with significance symbols (†, ‡)

**Why it matters:** Reviewers check this section first. Without statistical tests, claims of "better" will be rejected.

---

## Lesson 15 — Writing the Research Paper

**Paper structure:**

```
1. Introduction
   ├── Problem: expensive black-box optimisation
   ├── Gap: GPs scale poorly; NNs lack uncertainty
   ├── Contribution: NN surrogate with MC Dropout for BO
   └── Paper outline

2. Background
   ├── Black-box optimisation (Lesson 2)
   ├── Surrogate-assisted optimisation (Lesson 3)
   ├── Gaussian Processes (Lesson 6) — for comparison
   └── Bayesian Optimisation loop (Lesson 8)

3. Related Work
   ├── GP-based BO (standard literature)
   ├── Neural network surrogates (prior work)
   ├── MC Dropout for uncertainty (Gal & Ghahramani 2016)
   └── Deep ensembles (Lakshminarayanan 2017)

4. Proposed Method — NN Surrogate with MC Dropout
   ├── Network architecture
   ├── MC Dropout uncertainty derivation
   ├── EI acquisition with NN
   └── Full algorithm pseudocode

5. Experimental Setup
   ├── BBOB benchmark description (Lesson 13)
   ├── Baselines: GP-BO, CMA-ES, random search
   └── Metrics: ERT, ECDF, best-so-far

6. Results and Discussion
   ├── ECDF plots — NN vs. GP vs. CMA-ES
   ├── Ablation: MC Dropout vs. deterministic NN
   ├── Scalability: NN vs. GP across dimensions
   ├── Statistical significance (Lesson 14)
   └── Limitations: NN needs more data than GP in low-d

7. Conclusion and Future Work
   └── Future: deep ensembles, Bayesian neural networks, multi-fidelity

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

This installs everything needed for all 15 lessons.

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
├── lesson-10/         # NN surrogate — main contribution
├── lesson-11/         # Model comparison
├── lesson-12/         # Noise and high dimensions
├── lesson-13/         # COCO benchmark
├── lesson-14/         # Statistical analysis
├── lesson-15/         # Paper writing guide
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
