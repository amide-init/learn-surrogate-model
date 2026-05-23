# Lesson 2 — What is Optimisation?

## Objective

Understand what optimisation means, how gradient descent works, and — crucially — why it fails on black-box and noisy functions. This failure is the reason surrogates exist.

---

## Concepts

- **Optimisation** — finding the input `x` that minimises a function `f(x)`
- **Gradient descent** — follow the slope downhill: `x = x - lr * f'(x)`
- **Learning rate** — step size; too small = slow, too large = diverges
- **Black-box function** — you can evaluate `f(x)` but cannot compute its gradient
- **Noisy function** — each evaluation returns `f(x) + noise`, corrupting the gradient
- **Expensive function** — each evaluation costs real time, money, or compute

---

## Files

| File | Purpose |
|---|---|
| `notebook.ipynb` | Interactive — tweak learning rate, noise level, observe results |
| `main.py` | Standalone script — runs all experiments, saves plots to `output/` |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-2/main.py
```

Plots are saved to `lesson-2/output/`.

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-2/notebook.ipynb
```

---

## What You Will Build

### Part 1 — Gradient descent from scratch
Implement the update rule by hand on `f(x) = x²`. Plot every step taken toward the minimum.

### Part 2 — Effect of learning rate
Run gradient descent with three different learning rates. Show how too small = slow, too large = diverges.

### Part 3 — SciPy on hard functions
Use `scipy.optimize.minimize` on Rosenbrock and Ackley. Show that even library optimisers struggle on these.

### Part 4 — Gradient descent fails on noise
Add Gaussian noise to `f(x) = x²`. Watch gradient descent struggle or fail entirely.

### Part 5 — Black-box functions have no gradient
Show a discontinuous function (step function). Explain that if you cannot compute `f'(x)`, gradient descent is impossible — and this is exactly when you need a surrogate.

---

## Exercises

1. Change the learning rate from `0.1` to `1.5` in Part 2. What happens? Why does it diverge?
2. Try gradient descent on `f(x) = |x|` (absolute value, no gradient at 0). What goes wrong?
3. Increase noise from `sigma=0.5` to `sigma=2.0` in Part 4. At what level does descent completely break?

---

## What's Next

**Lesson 3** — What is a surrogate? You will build your first surrogate model (polynomial regression) to approximate an expensive function using only a few evaluations.
