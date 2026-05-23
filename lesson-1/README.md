# Lesson 1 — Python and Scientific Computing Basics

## Objective

Get comfortable with NumPy, SciPy, and Matplotlib — the three tools every lesson in this project depends on.

---

## Concepts

- **NumPy** — creating arrays, vectorised math, `linspace`, `meshgrid`
- **Matplotlib** — 1D line plots, 2D contour plots, subplots, labels, legends
- **SciPy** — using `scipy.optimize.minimize` to find a function minimum

---

## Files

| File | Purpose |
|---|---|
| `notebook.ipynb` | Interactive — run cells one by one, tweak values, observe results |
| `main.py` | Standalone script — runs everything and saves plots to `output/` |

---

## Instructions

### Run the script

```bash
# From the project root
source .venv/bin/activate
python lesson-1/main.py
```

Plots are saved to `lesson-1/output/`.

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-1/notebook.ipynb
```

Run cells top to bottom with `Shift + Enter`.

---

## What You Will Build

### Part 1 — Plotting basic functions
Plot `y = sin(x)` and `y = x²` on the same figure using NumPy arrays.

### Part 2 — 2D contour plot
Visualise the bowl-shaped function `f(x, y) = x² + y²` as a contour plot and a 3D surface.

### Part 3 — Finding a minimum with SciPy
Use `scipy.optimize.minimize` to find the minimum of `f(x) = (x - 2)² + 1` and mark it on the plot.

---

## Exercises

Try these after going through the notebook:

1. Change `sin(x)` to `sin(2x) + 0.5 * cos(3x)` and replot. What changes?
2. Replace `f(x, y) = x² + y²` with the Rosenbrock function: `f(x, y) = (1 - x)² + 100(y - x²)²`. Plot its contour — notice how the valley curves.
3. Use `method='Nelder-Mead'` instead of the default in `scipy.optimize.minimize`. Does it find the same minimum?

---

## What's Next

**Lesson 2** — What is optimisation? You will learn why gradient-based methods fail on black-box functions and why we need a different approach.
