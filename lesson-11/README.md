# Lesson 11 — NN Surrogate with Deep Ensembles

## Objective

Build the second main surrogate model: train 5 independent MLPs from different random seeds and compute mean and uncertainty across their predictions — a better-calibrated alternative to MC Dropout that is used as the primary comparison in Lesson 12.

---

## Concepts

- **Deep Ensembles** — N independent MLPs, each trained from a different random seed; no dropout; diversity comes from random initialisation and stochastic gradient descent
- **Ensemble mean** — μ(x) = (1/N) Σᵢ fᵢ(x); average across all members
- **Ensemble uncertainty** — σ²(x) = (1/N) Σᵢ (fᵢ(x) − μ(x))²; variance across members
- **Disagreement = uncertainty** — where members agree, σ is small; where they disagree, σ is large
- **`model.eval()` at inference** — unlike MC Dropout, each member is deterministic; stochasticity comes from having different weights
- **Reference** — Lakshminarayanan et al. (2017): *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*
- **Trade-off vs MC Dropout** — better calibrated uncertainty, but N× more training cost and memory

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `notebook.ipynb` | Interactive — change N members, compare calibration |
| `main.py` | Standalone script — saves all plots to `output/` |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-11/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-11/notebook.ipynb
```

---

## What You Will Build

### Part 1 — Ensemble diversity
Plot all 5 trained member predictions individually. Show that each learned a slightly different function — this disagreement is the source of the uncertainty estimate.

### Part 2 — Combined mean and uncertainty
Average the 5 members to get μ(x). Compute std across members to get σ(x). Plot the ±2σ band alongside the true function.

### Part 3 — Uncertainty calibration
Show σ(x) beneath the fit — peaks in data-sparse regions, small near observations. Compare calibration quality with MC Dropout from Lesson 10.

### Part 4 — Training: 5 loss curves
Plot all 5 member training loss curves. Show that they converge to similar but slightly different final values — the diversity that makes ensembles work.

### Part 5 — Deep Ensembles vs MC Dropout
Fit both surrogates to the same 15 training points. Compare mean and uncertainty side-by-side. Deep Ensembles typically gives wider, better-calibrated uncertainty.

### Part 6 — One BO step with Deep Ensembles
Replace the surrogate in the BO loop with the Deep Ensemble. Compute EI from ensemble μ and σ, pick x_next, evaluate, and update — same loop structure as Lessons 8 and 10.

---

## Exercises

1. In Part 1, reduce `n_members=2`. Does the uncertainty estimate become less reliable? Try `n_members=10` — does the uncertainty improve significantly?
2. In Part 3, compare the σ(x) curve from Deep Ensembles with the one from MC Dropout (Lesson 10 Part 3) on the same training data. Which gives larger uncertainty in the gaps?
3. In Part 6, run 15 BO iterations with Deep Ensembles on Forrester. Does it converge at the same rate as GP-BO from Lesson 8?

---

## What's Next

**Lesson 12** — Head-to-head comparison: MC Dropout vs Deep Ensembles vs GP on the same benchmarks (Forrester and Branin). Convergence curves, uncertainty calibration, and computational cost — the core results of the paper.
