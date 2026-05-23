# Lesson 6 — Gaussian Process Regression

## Objective

Build a GP surrogate from scratch by combining the conditional Gaussian formula (Lesson 4) with the kernel matrix (Lesson 5) — and use it to predict the Forrester and Branin benchmark functions with calibrated uncertainty.

---

## Concepts

- **GP posterior** — the distribution over functions after conditioning on data: μ* = Ks K⁻¹ y, σ*² = Kss − Ks K⁻¹ Ks^T
- **Noise model** — K + σ²_n I makes the GP robust to noisy observations
- **Cholesky decomposition** — numerically stable way to solve K⁻¹ y without explicit matrix inversion
- **Log marginal likelihood** — scores how well a set of kernel hyperparameters explains the data: log p(y|X) = −½ yᵀ K⁻¹ y − ½ log|K| − N/2 log 2π
- **Hyperparameter optimisation** — sweep (or gradient-ascend) over length-scale and noise to maximise the log marginal likelihood
- **Epistemic uncertainty** — GP uncertainty decreases near training points and grows far from them
- **GP as a baseline** — exact, well-calibrated, but O(n³) training cost; our NN surrogates scale better

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `notebook.ipynb` | Interactive — tweak hyperparameters, observe posterior changes |
| `main.py` | Standalone script — saves all plots to `output/` |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-6/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-6/notebook.ipynb
```

---

## What You Will Build

### Part 1 — Prior → Posterior
Sample functions from a GP prior, then condition on a few data points and show how the posterior collapses around the observations.

### Part 2 — GP on the Forrester function
Fit a GP to 5, 10, and 20 points sampled from the Forrester benchmark. Plot the posterior mean and ±2σ uncertainty band alongside the true function.

### Part 3 — Effect of the noise hyperparameter
Keep the same data; vary σ²_n from 0.001 to 0.5. Show how a larger noise parameter smooths the posterior and widens the uncertainty band.

### Part 4 — Kernel comparison on Forrester
Fit RBF, Matern-5/2, and Matern-3/2 kernels to the same data. Show that Matern-5/2 often fits engineering functions better than the infinitely smooth RBF.

### Part 5 — Log marginal likelihood sweep
Sweep length-scale from 0.05 to 2.0 and plot the log marginal likelihood. Identify the peak — that is the hyperparameter the GP "prefers" given the data.

### Part 6 — 2D GP on the Branin function
Fit a GP to 20 points in 2D. Plot the true Branin surface, the GP mean prediction, and the GP uncertainty on a 20×20 grid.

---

## Exercises

1. In Part 2, reduce the training set to 3 points. Where does the uncertainty grow largest? Does the GP still recover the true function shape?
2. In Part 5, also sweep over `signal_var` (0.5, 1.0, 2.0, 4.0). Plot a 2D heatmap of log marginal likelihood over (length_scale, signal_var).
3. Replace the Cholesky solver in `gp_predict` with `np.linalg.solve(K, y_train)` directly. When does this fail numerically, and why does Cholesky avoid it?

---

## What's Next

**Lesson 7** — Acquisition functions (PI, EI, UCB): given the GP posterior, where should we evaluate the expensive function next?
