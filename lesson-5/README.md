# Lesson 5 — Kernels and Similarity Functions

## Objective

Understand what a kernel is, implement the most common kernels from scratch, and see how they define the shape of functions a Gaussian Process can model. Kernels are the engine of GP surrogates — and understanding them helps you explain why neural networks can generalise better in higher dimensions where GP kernels struggle.

---

## Concepts

- **Kernel (covariance function)** — a function k(x, x') that measures how similar two inputs are
- **Core intuition** — if x and x' are close, f(x) and f(x') should be similar; the kernel quantifies "close"
- **Kernel matrix (Gram matrix)** — K[i,j] = k(xᵢ, xⱼ); the full matrix of pairwise similarities
- **Length-scale (l)** — how quickly similarity drops with distance; small l = wiggly, large l = smooth
- **Signal variance (σ²)** — overall scale of the function values
- **RBF kernel** — infinitely smooth; k(x,x') = σ² exp(−‖x−x'‖²/2l²)
- **Matern kernels** — rougher than RBF; Matern-5/2 is the most used in practice
- **Linear kernel** — recovers linear regression; k(x,x') = σ² x·x'
- **GP prior samples** — random functions drawn from a Gaussian Process before seeing any data
- **Stationary vs. non-stationary** — stationary kernels depend only on distance |x−x'|, not on position

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `notebook.ipynb` | Interactive — tweak length-scale, kernel type, observe resulting functions |
| `main.py` | Standalone script — saves all plots to `output/` |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-5/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-5/notebook.ipynb
```

---

## What You Will Build

### Part 1 — What is a kernel?
Implement the RBF kernel from scratch. Compute k(x, x') for a few pairs of points and show that close points get high similarity, far points get low similarity.

### Part 2 — The kernel matrix
Build the full kernel matrix for a set of 1D points. Plot it as a heatmap. Show how it changes with length-scale.

### Part 3 — Effect of length-scale
Plot RBF kernel similarity curves for l = 0.1, 0.5, 1.0, 2.0. Show what each implies about function smoothness.

### Part 4 — Sampling functions from a GP prior
Use the kernel matrix to sample random functions before seeing any data. Show how different kernels produce different "styles" of function.

### Part 5 — Kernel comparison
Compare RBF, Matern-3/2, Matern-5/2, and Linear kernels side by side — matrix heatmap and sample functions.

### Part 6 — 2D kernel matrix
Show how kernels work in 2D input space (needed for Lesson 6 with the Branin function).

---

## Exercises

1. In Part 3, change `length_scale=0.1`. What do the sampled functions look like? Would this be a good surrogate for a smooth engineering function?
2. In Part 4, add noise `K += 0.01 * np.eye(n)` before sampling. What changes? (This is exactly what a noisy GP does.)
3. In Part 5, implement the **periodic kernel**: `k(x,x') = σ² exp(−2 sin²(π|x−x'|/p) / l²)`. What shape do sampled functions take?

---

## What's Next

**Lesson 6** — Gaussian Process regression. You will combine the conditional distribution from Lesson 4 with the kernel matrix from this lesson to build a full GP surrogate from scratch.
