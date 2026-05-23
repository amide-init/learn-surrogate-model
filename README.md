# learn-surrogate

A Python project for learning surrogate-assisted optimisation from absolute zero — built so that completing all stages gives you enough theory, code, and experiments to write a research paper.

No prior knowledge of machine learning or optimisation is assumed.

---

## How to use this roadmap

Each stage has:
- **What you will learn** — the concept in plain English
- **What you will build** — a concrete Python script
- **Why it matters** — how it connects to your research paper

Work through stages in order. Do not skip — each one builds on the last.

---

## Stage 0 — Python and Scientific Computing Basics

**What you will learn:** The tools every stage depends on.

- NumPy: arrays, broadcasting, vectorised math
- SciPy: built-in optimisers, statistics
- Matplotlib: 1D line plots, 2D contour plots, subplots

**What you will build:** `basics/s0_tools.py`
- Plot y = sin(x) and y = x² with NumPy
- Draw a 2D contour plot of f(x, y) = x² + y²
- Use `scipy.optimize.minimize` to find the minimum of a simple function

**Why it matters:** Every surrogate experiment you run will use these three libraries. Getting comfortable with them now saves hours of confusion later.

**Resources:**
- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [Matplotlib tutorials](https://matplotlib.org/stable/tutorials/index.html)

---

## Stage 1 — What is Optimisation?

**What you will learn:** The problem surrogates are designed to solve.

- What does "optimise a function" mean?
- Types of optimisation: unconstrained, bounded, constrained
- Gradient-based methods: gradient descent, L-BFGS-B
- Why gradient-based methods fail: noisy functions, discontinuities, black boxes
- What "expensive" means: each evaluation costs real time, money, or compute

**What you will build:** `basics/s1_optimisation.py`
- Run gradient descent on f(x) = x² by hand (update rule, step size)
- Use `scipy.optimize.minimize` on Rosenbrock and Ackley
- Show that gradient descent fails when you add noise to the function

**Why it matters:** Your paper's introduction must explain why standard optimisers are not enough. This stage gives you that argument.

---

## Stage 2 — What is a Surrogate?

**What you will learn:** The central idea of the whole project — in its simplest form.

- A surrogate is a cheap model trained on a few evaluations of an expensive function
- The loop: evaluate a few points → fit a model → use the model to decide where to evaluate next
- Your first surrogate: polynomial regression

**What you will build:** `basics/s2_first_surrogate.py`
- Sample 5 points from f(x) = sin(x) + 0.1·x²
- Fit a degree-3 polynomial (NumPy `polyfit`)
- Plot: true function vs. surrogate approximation
- Show what happens with 3 points vs. 10 points

**Why it matters:** Polynomial regression is the simplest possible surrogate. Seeing it succeed and fail motivates why we need something smarter (Gaussian Processes).

---

## Stage 3 — Statistics Refresher

**What you will learn:** The mathematical language used in every GP paper.

- Random variables, mean, variance, standard deviation
- Covariance and correlation
- The Gaussian (normal) distribution: PDF, CDF, sampling
- Multivariate Gaussian: mean vector, covariance matrix
- Conditional distributions: "given that X = x, what is Y?"

**What you will build:** `basics/s3_statistics.py`
- Plot 1D Gaussians with different means and variances
- Sample from a 2D Gaussian and plot the scatter + ellipse
- Compute and visualise the conditional distribution P(Y | X = x)

**Why it matters:** Gaussian Processes are just multivariate Gaussians. If you understand this stage, GPs will feel natural.

---

## Stage 4 — Kernels and Similarity

**What you will learn:** How to measure similarity between inputs — the engine of GP models.

- What is a kernel (covariance function)?
- Intuition: points that are close in input space should have similar outputs
- Common kernels:
  - RBF / Squared Exponential: very smooth functions
  - Matern-3/2, Matern-5/2: rough to smooth (most used in practice)
  - Linear kernel: recovers linear regression
- Hyperparameters: length-scale (how quickly similarity drops), signal variance

**What you will build:** `basics/s4_kernels.py`
- Implement RBF kernel: k(x, x') = σ² · exp(−‖x−x'‖² / 2l²)
- Plot the kernel matrix (heatmap) for a set of 1D points
- Show how length-scale changes the kernel matrix
- Sample random functions from different kernels

**Why it matters:** Kernel choice is one of the most cited design decisions in surrogate papers. You need to be able to justify your choice.

---

## Stage 5 — Gaussian Process Regression

**What you will learn:** The standard surrogate model used across most of the literature.

- A Gaussian Process (GP) = a distribution over functions
- GP prior: before seeing data, what functions do we think are plausible?
- GP posterior: after seeing data, what functions are still plausible?
- Predictive mean and variance at new points
- Fitting hyperparameters by maximising the log marginal likelihood

**What you will build:** `basics/s5_gp.py`
- Build a GP class from scratch:
  - `fit(X, y)`: compute kernel matrix, store Cholesky factor
  - `predict(X_new)`: return posterior mean and variance
- Test on the Forrester function (standard 1D benchmark)
- Plot: data points, true function, GP mean, ±2σ uncertainty band

**Why it matters:** GP regression is the most common surrogate model. This is the core of your paper's methodology.

---

## Stage 6 — Acquisition Functions

**What you will learn:** How surrogates decide where to evaluate next.

- The exploration vs. exploitation trade-off
- Probability of Improvement (PI): how likely is this point to beat the best so far?
- Expected Improvement (EI): how much improvement do we expect on average?
- Upper Confidence Bound (UCB): mean − κ · std (κ controls exploration)

| Function | Formula (informal) | Behaviour |
|---|---|---|
| PI | P(f(x) < f\*) | Conservative, fast convergence |
| EI | E[max(f\* − f(x), 0)] | Balanced, most common in papers |
| UCB | μ(x) − κ σ(x) | Explicit exploration control |

**What you will build:** `basics/s6_acquisition.py`
- Implement PI, EI, UCB as functions of GP mean and variance
- Plot all three on the same 1D example alongside the GP surrogate
- Show how κ in UCB changes the recommended next point

**Why it matters:** Acquisition function choice and comparison is a common research contribution. Understanding all three lets you run ablation studies.

---

## Stage 7 — The Bayesian Optimisation Loop

**What you will learn:** How everything wires together into a complete algorithm.

```
1. Evaluate f at an initial set of points (design of experiments)
2. Fit a GP surrogate to the observations
3. Optimise the acquisition function to find the next point x*
4. Evaluate f(x*) — this is the expensive step
5. Add (x*, f(x*)) to the data and go to step 2
6. Stop when the budget is exhausted
```

**What you will build:** `basics/s7_bo_loop.py`
- Implement the full loop as above
- Initial design: Latin Hypercube Sampling (LHS) — better than random
- Acquisition optimiser: multi-start L-BFGS-B (avoid local optima)
- Test on: Forrester (1D), Branin (2D), Hartmann-3 (3D)
- Plot convergence: best value found vs. number of evaluations

**Why it matters:** This is your baseline algorithm. Every comparison in your paper starts here.

---

## Stage 8 — Surrogate Model Comparison

**What you will learn:** When is a GP better or worse than other surrogate choices?

| Model | Package | Key property |
|---|---|---|
| Gaussian Process | `sklearn.gaussian_process` | Uncertainty estimates, data-efficient |
| Radial Basis Function | `scipy.interpolate` | Fast, deterministic, no uncertainty |
| Random Forest | `sklearn.ensemble` | Handles discrete variables |
| Polynomial Response Surface | `numpy.polyfit` | Interpretable, fast, low-dim only |

**What you will build:** `surrogate/s8_model_comparison.py`
- Run all four models on the same test functions
- Same initial data, same evaluation budget
- Metrics: best value found, RMSE of surrogate fit, wall-clock time
- Output: comparison table and box plots

**Why it matters:** A model comparison experiment is a publishable contribution on its own. It also strengthens your justification for choosing GP.

---

## Stage 9 — Handling Noise

**What you will learn:** Real experiments are noisy — your surrogate must handle this.

- Noise in observations: f(x) = f_true(x) + ε, ε ~ N(0, σ_n²)
- GP with noise: add σ_n² to the diagonal of the kernel matrix
- Estimating noise level from data
- Noisy vs. noiseless acquisition (EI with noise)

**What you will build:** `surrogate/s9_noisy_gp.py`
- Add Gaussian noise to a test function
- Fit a noisy GP and compare to a noiseless GP
- Show how the uncertainty band grows with noise
- Run BO with and without noise handling

**Why it matters:** The COCO benchmark has a noisy variant. Papers that handle noise are more general and more publishable.

---

## Stage 10 — Scalability: High-Dimensional Surrogates

**What you will learn:** Why GPs break in high dimensions and what to do about it.

- The curse of dimensionality: GP kernel matrices become poorly conditioned
- Automatic Relevance Determination (ARD): one length-scale per dimension, prunes irrelevant ones
- Variable selection: identify which inputs actually matter
- REMBO (Random EMbedding BO): project high-dim input into low-dim subspace

**What you will build:** `surrogate/s10_high_dim.py`
- Run BO with RBF kernel (single length-scale) vs. ARD kernel across d = 2, 5, 10, 20
- Plot: best value found vs. dimension for both kernels
- Add a function where only 3 of 10 inputs matter — show ARD finds them

**Why it matters:** Most real engineering problems have many inputs. Scalability is a natural limitations section and future-work direction.

---

## Stage 11 — Surrogate-Assisted Evolutionary Algorithms (SAEA)

**What you will learn:** A different paradigm — combining surrogates with population-based search.

- Why pure BO (sequential, single-point) is slow at scale
- SAEA: use the surrogate to pre-screen a population, only evaluate the best candidates
- Offline vs. online surrogates
- Simple (1+1)-ES with GP-based fitness approximation

**What you will build:** `surrogate/s11_saea.py`
- Implement a (1+λ)-ES with surrogate pre-selection
- Compare: vanilla ES vs. SAEA on Rastrigin and Hartmann-6
- Plot: evaluations saved vs. solution quality

**Why it matters:** SAEA is a major branch of surrogate literature (Jin 2011, Liu 2020). Including it widens your related work and shows breadth.

---

## Stage 12 — COCO / BBOB Benchmark

**What you will learn:** How to evaluate your algorithm on the community-standard benchmark so your results are comparable to published work.

- COCO framework: `cocoex` (run experiments) + `cocopp` (generate plots)
- BBOB suite: 24 noiseless test functions covering different landscape types
- Dimensions: 2, 3, 5, 10, 20, 40
- Budget: typically 10 × dimension to 1000 × dimension function evaluations
- Metrics: ERT (Expected Running Time), ECDF curves

**What you will build:** `benchmark/s12_coco_run.py`
- Wrap your Stage 7 BO algorithm in a COCO-compatible interface
- Run on all 24 BBOB functions, dimensions 2 and 5, budget = 100 × dimension
- Generate ECDF plots with `cocopp`
- Compare to CMA-ES (included in COCO archive) as a baseline

**COCO concepts:**

| Term | Meaning |
|---|---|
| `budget_multiplier` | Max evaluations = multiplier × dimension |
| Target precision | f(x) − f\_opt < ∆f (solved when this holds) |
| ECDF | Fraction of (function, instance, target) triples solved within budget |
| ERT | Expected number of evaluations to hit a target, averaged over instances |

**Why it matters:** ECDF plots are the standard figure in GECCO / CEC papers. Without COCO results, most surrogate papers will not pass peer review.

---

## Stage 13 — Analysis, Ablation, and Statistical Testing

**What you will learn:** How to turn raw results into defensible scientific claims.

- Ablation study: disable one component at a time, measure the drop in performance
- Statistical significance: Wilcoxon signed-rank test (pairwise), Friedman test (multiple algorithms)
- Effect size: not just "better" but "how much better"
- Sensitivity analysis: how sensitive is performance to budget, kernel choice, acquisition function?

**What you will build:** `analysis/s13_stats.py`
- Run each ablation (remove LHS → random init, swap EI → random, remove GP → random search)
- Run Wilcoxon tests between each pair of algorithms
- Output: LaTeX-ready table of means, standard deviations, and significance symbols (†, ‡)

**Why it matters:** This is the section reviewers check to decide if your claims are credible. Without it, a paper will be rejected.

---

## Stage 14 — Writing the Research Paper

**Suggested paper structure:**

```
1. Introduction
   ├── Problem: expensive black-box optimisation
   ├── Motivation: each evaluation costs significant resources
   ├── Proposed approach: surrogate-assisted optimisation
   ├── Contribution: (what is new — your specific angle)
   └── Paper outline

2. Background
   ├── Problem statement and notation (Stage 1)
   ├── Gaussian Processes (Stages 3–5)
   └── Bayesian Optimisation (Stages 6–7)

3. Related Work
   ├── Surrogate model comparison (Stage 8)
   ├── Noisy optimisation (Stage 9)
   ├── High-dimensional surrogates (Stage 10)
   └── SAEA methods (Stage 11)

4. Proposed Method
   ├── Algorithm pseudocode
   ├── Design choices and justification
   └── Theoretical properties (if any)

5. Experimental Setup
   ├── BBOB test suite description (Stage 12)
   ├── Algorithms compared and configurations
   └── Performance metrics: ERT, ECDF, best-so-far

6. Results and Discussion
   ├── ECDF plots (Stage 12)
   ├── Ablation study (Stage 13)
   ├── Statistical significance (Stage 13)
   └── Limitations

7. Conclusion and Future Work

References
```

**Possible paper angles** (pick one for a focused contribution):
- New acquisition function or acquisition ensemble
- Surrogate model comparison on BBOB (Stages 8 + 12)
- BO vs. SAEA comparison (Stages 7, 11, 12)
- Handling noise with GP (Stages 9 + 12 noisy variant)
- ARD kernels for high-dimensional BBOB (Stages 10 + 12)

**Target venues (ascending difficulty):**
- GECCO Workshop — most accessible, good for early work
- CEC (Congress on Evolutionary Computation) — standard venue
- GECCO Main Track — competitive, needs strong COCO results
- PPSN (Parallel Problem Solving from Nature) — biennial, rigorous
- NeurIPS / ICML — requires theoretical contribution

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

This installs everything needed for all lessons including the COCO benchmark (Lesson 13).

### 3. Verify the setup

```bash
python -c "import numpy, scipy, matplotlib, sklearn; print('All good')"
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

## Dependencies

| Package | Purpose | First used |
|---|---|---|
| `numpy` | Arrays, linear algebra | Lesson 1 |
| `scipy` | Optimisers, statistics, LHS | Lesson 2 |
| `matplotlib` | All plots | Lesson 1 |
| `scikit-learn` | GP, Random Forest | Lesson 6 |
| `coco-experiment` | COCO experiment runner | Lesson 13 |
| `cocopp` | COCO post-processing, ECDF | Lesson 13 |
| `cma` | CMA-ES baseline optimiser | Lesson 13 |
| `scikit-optimize` | Additional BO baseline | Lesson 8 |
| `torch` | Neural network surrogate | Lesson 9 |

## Directory Structure

```
learn-surrogate/
├── basics/        # Stages 0–7:  foundations through BO loop
├── surrogate/     # Stages 8–11: model comparison, noise, high-dim, SAEA
├── benchmark/     # Stage 12:   COCO experiment runner
├── analysis/      # Stage 13:   statistical tests and paper figures
├── results/       # Plots and outputs (gitignored)
└── requirements.txt
```

---

## Key References

**Bayesian Optimisation**
- Frazier (2018). [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811)
- Mockus (1975). On Bayesian Methods for Seeking the Extremum — original EI derivation

**Gaussian Processes**
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning* — free PDF at gaussianprocess.org

**Acquisition Functions**
- Srinivas et al. (2010). Gaussian Process Optimization in the Bandit Setting — UCB theory

**Surrogate-Assisted Evolutionary Algorithms**
- Jin (2011). Surrogate-assisted evolutionary computation: Recent advances and future challenges. *Swarm and Evolutionary Computation*
- Liu et al. (2020). A Gaussian Process Surrogate Model Assisted Evolutionary Algorithm for Medium Scale Expensive Optimization Problems

**COCO Benchmark**
- Hansen et al. (2021). [COCO: A Platform for Comparing Continuous Optimizers](https://arxiv.org/abs/1603.08785)
- Hansen et al. BBOB function definitions — numbbo.github.io/coco/bbob
