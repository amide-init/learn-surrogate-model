# Lesson 14 — All Surrogates Comparison

## Objective

Compare GP, MC Dropout, Deep Ensembles, RBF, and Random Forest inside the SCR ≤ 20% enforcer BO loop on Forrester (1D) and Branin (2D).

---

## Concepts

- **Unified surrogate interface** — all surrogates expose `predict(X_cand) → μ, σ`; the SCR enforcer loop is identical for all
- **SCR ≤ 20% enforced** — `SCR = surrogate_only_calls / (true_calls + surrogate_only_calls)` capped at 20% using the enforcer from Lesson 13
- **Rank correlation** — Spearman ρ measures how well each surrogate ranks candidates; high ρ → surrogate-only steps are trustworthy
- **Convergence per true evaluation** — COCO-correct comparison; x-axis = true function calls only
- **Calibration** — σ should track |μ(x) − f(x)|; GP is best-calibrated, RBF uses a distance proxy, RF uses tree variance
- **Speed vs. quality trade-off** — GP and RBF are fast; Deep Ensembles are slow but better calibrated

---

## Surrogates

| Model | Uncertainty source | Speed |
|---|---|---|
| GP | Posterior variance (Matérn-5/2) | Fast |
| MC Dropout | Variance across T=50 forward passes (dropout ON at inference) | Moderate |
| Deep Ensembles | Variance across 5 independently trained networks | Slow |
| RBF | Distance to nearest training point (proxy) | Fast |
| Random Forest | Variance across 100 tree predictions | Fast |

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `main.py` | Standalone script — saves all plots to `output/` |
| `notebook.ipynb` | Interactive — explore each surrogate and compare |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-14/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-14/notebook.ipynb
```

---

## What You Will Build

### Part 1 — Fit quality on Forrester
All 5 surrogates fitted to the same 15 training points. Plot μ ± 2σ for each. Compare uncertainty shape qualitatively.

### Part 2 — Rank correlation
Generate 40 candidates. Compute Spearman ρ between true f and surrogate μ for all 5 models. Bar chart of ρ values — determines which surrogates are safe for surrogate-only steps.

### Part 3 — BO convergence on Forrester (1D)
Run SCR-enforced BO (SCR ≤ 20%, 30 iterations, 3 seeds) for all 5 surrogates. X-axis = true function calls.

### Part 4 — BO convergence on Branin (2D)
Same loop on Branin. Tests whether surrogate quality generalises to higher dimensions.

### Part 5 — Speed comparison
Time the fit + predict step for each surrogate. Bar chart of wall-clock time per BO iteration.

### Part 6 — Leaderboard
Summary: final gap to optimum + speed for each surrogate × benchmark. Identify the best speed-quality trade-off.

---

## Exercises

1. In Part 2, which surrogate has the lowest Spearman ρ? Does that explain its BO convergence in Part 3?
2. In Part 3, increase `EPOCHS_BO` from 500 to 1000 for MC Dropout and Deep Ensembles. Does convergence improve?
3. In Part 4 on Branin, try `N_INIT_2D = 5` (very few initial points). Which surrogate degrades most?

---

## What's Next

**Lesson 15** — Noise handling and high-dimensional inputs: how each surrogate behaves when f(x) is noisy or d > 5.
