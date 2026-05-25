# Lesson 15 — Noise Handling and High-Dimensional Inputs

## Objective

Test how each surrogate behaves when observations are noisy (f(x) + ε) and when the input dimension increases to d=5 and d=10.

---

## Concepts

- **Observation noise** — evaluations corrupted by ε ~ N(0, σ²); surrogates must smooth rather than interpolate
- **GP nugget** — small diagonal term (alpha) added to K_train for numerical stability; absorbs observation noise
- **Noise robustness** — GP and Random Forest average over noise naturally; RBF interpolates exactly (bad under noise)
- **Curse of dimensionality** — in high-d, nearest neighbours are far away; EI landscape is harder to optimise
- **Ackley benchmark** — standard multimodal function on [−5, 5]^d (mapped from [0, 1]^d); f* = 0 at the origin
- **SCR ≤ 20%** — same enforcer as Lessons 13–14; surrogate steps are always less than 20% of total steps
- **Isotropic GP** — single shared length scale used in high-d; ARD would need far more data to estimate reliably

---

## Surrogates

| Model | Noise handling | High-dim behaviour |
|---|---|---|
| GP | Nugget absorbs noise; posterior smooths over data | Degrades above d~10 (length scale estimation unreliable) |
| MC Dropout | Averaging over passes provides some robustness | Generalises if enough training data |
| Deep Ensembles | Ensemble averaging reduces noise impact | Better calibrated but slow; same data requirements as MC Dropout |
| RBF | Interpolates exactly — highly sensitive to noise | Distance proxy breaks in high-d (curse of dimensionality) |
| Random Forest | Tree averaging is naturally noise-robust | Struggles with smooth functions in high-d |

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `main.py` | Standalone script — saves all plots to `output/` |
| `notebook.ipynb` | Interactive — explore noise and high-dim effects |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-15/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-15/notebook.ipynb
```

---

## What You Will Build

### Part 1 — Noisy Forrester fit
All 5 surrogates fitted to 15 noisy training points (σ_noise = 0.5). Visual comparison of how each handles noise in μ ± 2σ. RBF should interpolate through the noise; GP should smooth over it.

### Part 2 — Noise level sweep
Vary σ_noise ∈ {0.0, 0.1, 0.5, 1.0, 2.0}. For each surrogate, fit to noisy data and compute RMSE on a noise-free test set. Line plot reveals which surrogates degrade gracefully.

### Part 3 — BO convergence under noise (Forrester 1D)
SCR-enforced BO (SCR ≤ 20%) on noisy Forrester (σ = 0.5). Convergence curves show how noise affects each surrogate's BO performance.

### Part 4 — BO convergence on Ackley (d=5)
5D Ackley with 20 initial points and 15 BO iterations per seed. Tests surrogate quality when the input dimension is moderate.

### Part 5 — BO convergence on Ackley (d=10)
10D Ackley with 30 initial points and 10 BO iterations per seed. Tests surrogate behaviour at the onset of the curse of dimensionality.

### Part 6 — Leaderboard
Final gap to optimum across all three experiments (noisy 1D, 5D, 10D). Bar charts reveal which surrogate is the most robust overall.

---

## Exercises

1. In Part 2, which surrogate has the steepest RMSE rise as σ_noise increases? What property of that surrogate causes this?
2. In Part 4, try `N_INIT_5D = 10` (fewer initial points). Does the ranking of surrogates change?
3. In Part 5 (d=10), the GP often degrades. Does switching to `n_restarts_optimizer=5` help, or does the problem lie elsewhere?

---

## What's Next

**Lesson 16** — COCO / BBOB benchmark: evaluate all surrogates on the official noiseless BBOB suite with budget = 100 × d evaluations.
