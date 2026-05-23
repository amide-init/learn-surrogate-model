# Lesson 12 — MC Dropout vs Deep Ensembles vs GP

## Objective

Run a head-to-head comparison of all three surrogates on the same benchmarks (Forrester 1D and Branin 2D) — mean fit quality, uncertainty calibration, BO convergence speed, and computational cost.

---

## Concepts

- **Calibration** — a surrogate is calibrated when σ(x) is large where the prediction error is large and small where it is small; poorly-calibrated σ leads to bad acquisition decisions
- **Gap to optimum** — `gap = best_f_found − f*`; plot on log scale to compare convergence speed across surrogates
- **Convergence curve** — best-so-far vs BO iteration, averaged across random seeds to reduce noise
- **Cost per refit** — how long each surrogate takes to refit when a new point is added; this scales with BO budget
- **GP advantage** — analytic posterior, calibrated σ, cheap; but cubic cost in n and struggles above ~20D
- **MC Dropout advantage** — single network, cheap to train; but noisy σ, stochastic at inference
- **Deep Ensemble advantage** — smooth, structured σ; but N× training cost per refit

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `notebook.ipynb` | Interactive — change seeds, iter count, compare methods |
| `main.py` | Standalone script — saves all plots to `output/` |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-12/main.py
```

> **Runtime warning**: the convergence experiment trains each surrogate from scratch at every BO step.  
> Expected runtime: 8–15 minutes depending on hardware.  
> Progress is printed to the terminal.

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-12/notebook.ipynb
```

---

## What You Will Build

### Part 1 — Fit comparison on Forrester
Three panels side by side: GP | MC Dropout | Deep Ensemble, each fitted on the same 15 training points. Show mean ± 2σ. Are the mean fits equally good? Where does σ differ?

### Part 2 — Uncertainty calibration on Forrester
Plot σ(x) for all three methods on the same axis. Then compare |μ(x) − f(x)| (prediction error) with σ(x) — a well-calibrated surrogate has large σ where its error is large.

### Part 3 — BO convergence on Forrester
Run 3 seeds × 20 BO iterations for each surrogate. Plot: (1) best f found so far, and (2) gap to optimum on log scale. Shows which surrogate converges fastest in 1D.

### Part 4 — Fit comparison on Branin (2D)
Contour plots of μ and σ for all three surrogates on the 2D Branin domain. 2×3 grid: top row = mean, bottom row = uncertainty. Shows how uncertainty spreads in 2D.

### Part 5 — BO convergence on Branin
Same as Part 3 but on Branin (2D). Which surrogate handles the higher-dimensional search best?

### Part 6 — Computational cost
Measure and plot the wall-clock time per surrogate refit. Bar chart: GP vs MC Dropout vs Deep Ensemble at 1000 epochs. The cost ratio is the main practical consideration when scaling BO.

---

## Exercises

1. In Part 3, increase `N_ITER = 30`. At what iteration do all three surrogates converge? Does GP maintain its lead as the budget grows?
2. In Part 4, inspect the σ contour for each surrogate in the corners of the domain (far from training points). Which surrogate shows the most uncertainty there? Which shows the least?
3. In Part 6, add `N_MEMBERS = 3` and re-run Part 3. Does a smaller ensemble converge similarly to `N_MEMBERS = 5` while being cheaper?

---

## What's Next

**Lesson 13** — Surrogate Control Ratio (SCR): deciding when to use the surrogate vs. the true function. Without SCR, the surrogate may dominate too many evaluations, violating the budget constraint needed for valid COCO results.
