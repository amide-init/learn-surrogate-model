# Lesson 8 — Bayesian Optimisation Loop with GP

## Objective

Combine the GP surrogate from Lesson 6 and the EI acquisition function from Lesson 7 into a full Bayesian Optimisation loop — initialise with Latin Hypercube Sampling, run 20 iterations, and track convergence on the Forrester and Branin benchmarks.

---

## Concepts

- **BO loop** — initialise → fit GP → maximise acquisition → evaluate f → update → repeat
- **Latin Hypercube Sampling (LHS)** — stratified random design that guarantees one point per stratum in each dimension; better initial coverage than random sampling
- **Budget** — total number of real function evaluations (initial + BO iterations)
- **Best-so-far curve** — tracks the best f value found at each iteration; the standard BO convergence metric
- **Regret** — gap between current best and the true global minimum; decreases as BO converges
- **Random search baseline** — evaluates f at random points; no model; shows the baseline BO must beat
- **Candidate set** — dense grid used to maximise EI in 1D/2D; in high dimensions replaced by gradient-based optimisers or random restarts

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `notebook.ipynb` | Interactive — run the BO loop, watch the GP evolve |
| `main.py` | Standalone script — saves all plots to `output/` |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-8/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-8/notebook.ipynb
```

---

## What You Will Build

### Part 1 — Latin Hypercube Sampling
Compare LHS vs random sampling in 2D. Show that LHS places exactly one point per stratum in each axis — better coverage with the same number of points.

### Part 2 — GP snapshots during the BO loop
Run the full BO loop on Forrester. Snapshot the GP and EI at iterations 1, 5, 10, and 20. Watch the uncertainty collapse and the mean improve with each evaluation.

### Part 3 — Convergence curve
Plot best-found value vs iteration number averaged over 10 independent runs. Add error bands (mean ± std). Compare directly against random search.

### Part 4 — BO vs random search
Run both BO (EI) and random search for 30 evaluations across 10 seeds. Plot convergence curves side-by-side. Quantify how many fewer evaluations BO needs to reach the same quality.

### Part 5 — 2D BO on Branin
Run the BO loop in 2D on the Branin function (n_init=10, budget=40). Plot training points overlaid on the true Branin surface and the convergence curve.

### Part 6 — Effect of initial design size
Run BO with n_init = 3, 5, 10 on Forrester (fixed total budget = 25). Show how the initial design size affects early vs late convergence.

---

## Exercises

1. In Part 2, replace EI with PI (ξ=0). Does the loop get stuck? Compare convergence with EI (ξ=0.01).
2. In Part 3, what happens if you set `noise=1e-6`? Does a near-noise-free GP help or hurt convergence? Why?
3. In Part 5, increase the budget to 60. Does BO reliably find the Branin minimum (≈ 0.397) within that budget?

---

## What's Next

**Lesson 9** — PyTorch basics: tensors, `nn.Module`, autograd, and the training loop. These are the building blocks of the NN surrogates in Lessons 10 and 11.
