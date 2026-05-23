# Lesson 3 — What is a Surrogate?

## Objective

Build your first surrogate model — a cheap approximation of an expensive function — using polynomial regression, and understand the core surrogate loop that every lesson from here onward follows.

---

## Concepts

- **Surrogate model** — a cheap model trained on a few evaluations of an expensive function
- **The surrogate loop** — evaluate → fit surrogate → use surrogate to pick next point → repeat
- **Training data** — the small set of points where we actually evaluated the true function
- **Polynomial regression** — fitting a degree-N polynomial through observed points
- **Overfitting vs. underfitting** — too many degrees fits noise; too few misses the shape
- **Surrogate failure modes** — where polynomial surrogates break (multimodal, extrapolation)
- **Why we need something smarter** — motivation for Gaussian Processes and neural networks

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `notebook.ipynb` | Interactive — tweak number of points, polynomial degree, observe results |
| `main.py` | Standalone script — runs all experiments, saves plots to `output/` |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-3/main.py
```

Plots are saved to `lesson-3/output/`.

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-3/notebook.ipynb
```

---

## What You Will Build

### Part 1 — The surrogate concept
Sample 5 points from a hidden expensive function. Fit a polynomial. Show the surrogate vs. the true function.

### Part 2 — Effect of number of training points
Compare surrogates built from 3, 6, and 12 points. Show how more data improves the fit.

### Part 3 — Effect of polynomial degree
Compare degree 1, 3, and 8 on the same data. Show underfitting and overfitting.

### Part 4 — The surrogate loop
Run 10 iterations of: fit surrogate → find surrogate minimum → evaluate true function there → add to data. Watch the surrogate improve over time.

### Part 5 — Where polynomial surrogates fail
Test on a multimodal function (many peaks). Show the surrogate completely misses the landscape — motivation for GP and NN.

---

## Exercises

1. Change the polynomial degree in Part 3 from 8 to 15. What happens? Why is this a problem for optimisation?
2. In Part 4, instead of picking the surrogate minimum as the next point, pick a random point. How does convergence change?
3. Try the surrogate loop on `f(x) = sin(3x) + x` (multimodal). Does it find the global minimum?

---

## What's Next

**Lesson 4** — Statistics refresher. You will learn about Gaussian distributions and covariance — the math needed to understand why Gaussian Processes give better surrogates than polynomials.
