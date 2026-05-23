# Lesson 4 — Statistics Refresher

## Objective

Build the statistical foundation needed for Gaussian Processes (Lesson 6) and neural network uncertainty estimation (Lessons 10–11). Every surrogate model in this project outputs a **mean** and a **variance** — this lesson explains what those numbers mean.

---

## Concepts

- **Random variable** — a quantity whose value is uncertain
- **Mean (μ)** — the expected value, centre of the distribution
- **Variance (σ²)** — how spread out the distribution is
- **Standard deviation (σ)** — square root of variance, same units as the data
- **Gaussian distribution** — the bell curve; defined entirely by μ and σ²
- **PDF and CDF** — probability density function and cumulative distribution function
- **Multivariate Gaussian** — Gaussian over multiple variables at once; uses a mean **vector** and covariance **matrix**
- **Covariance** — how much two variables move together
- **Correlation** — normalised covariance, always between −1 and +1
- **Conditional distribution** — P(Y | X = x): what Y looks like once we know X

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `notebook.ipynb` | Interactive — change parameters, observe distributions |
| `main.py` | Standalone script — saves all plots to `output/` |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-4/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-4/notebook.ipynb
```

---

## What You Will Build

### Part 1 — The 1D Gaussian
Plot Gaussians with different means and variances. Understand the 68-95-99.7 rule.

### Part 2 — Sampling from a Gaussian
Draw samples, compute their mean and variance, show convergence to the true parameters.

### Part 3 — Covariance and Correlation
Visualise what positive, negative, and zero correlation look like in 2D scatter plots.

### Part 4 — The 2D (Multivariate) Gaussian
Plot a 2D Gaussian as a contour and compute the covariance matrix from data.

### Part 5 — Conditional Distributions
Given a 2D Gaussian, compute P(Y | X = x). This is the exact operation a Gaussian Process performs at every prediction step.

### Part 6 — Why This Matters for Surrogates
Show that a GP prediction (mean + variance) is exactly a conditional Gaussian — connecting this lesson directly to Lesson 6.

---

## Exercises

1. In Part 1, change `sigma=2.0` to `sigma=0.1`. What happens to the bell curve? What does this mean for a surrogate model?
2. In Part 3, change the covariance to `[[1, -0.9], [-0.9, 1]]`. What shape do the samples form?
3. In Part 5, compute P(Y | X = 1.5) and P(Y | X = -1.5). How does the conditional mean change?

---

## What's Next

**Lesson 5** — Kernels and similarity functions. You will use the covariance intuition from this lesson to understand how a Gaussian Process measures similarity between inputs.
