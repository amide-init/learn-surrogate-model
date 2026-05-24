# Lesson 13 — Surrogate Control Ratio (SCR)

## Objective

Understand what SCR is, why it must stay below 20% for valid surrogate-assisted optimisation, and implement an SCR enforcer inside the BO loop.

---

## Core definition

```
SCR = surrogate_only_calls / (true_calls + surrogate_only_calls)
```

- **SCR = 0%** — pure BO: every accepted point is truly evaluated
- **SCR = 20%** — 1 in 5 evaluations uses the surrogate prediction as fitness (saves 1 true eval per 5 steps)
- **SCR > 20%** — risky: training on too many surrogate-predicted y values causes surrogate drift

### Why SCR must be controlled
If the surrogate's prediction `μ(x)` is used as the fitness value `y` and that point is added back to the training dataset, the surrogate then trains on its own errors. Over many steps this causes **surrogate drift** — the model becomes increasingly inaccurate while appearing confident.

---

## Concepts

- **SCR enforcer** — before each BO step, check: "if I use the surrogate now, will SCR exceed the ceiling?" If yes → call the true function
- **Projected SCR** — `(surr_calls + 1) / (true_calls + surr_calls + 1)`; compare to `scr_max` before deciding
- **Rank correlation** — Spearman ρ between surrogate ranking and true ranking; high ρ means prescreening is trustworthy
- **Surrogate drift** — if surrogate-predicted y values are added back as training data, the surrogate trains on its own errors
- **COCO budget** — measured in true function calls only; SCR controls how efficiently each true call is used

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `notebook.ipynb` | Interactive — change SCR ceiling, compare convergence per true eval |
| `main.py` | Standalone script — saves all plots to `output/` |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-13/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-13/notebook.ipynb
```

---

## What You Will Build

### Part 1 — The SCR concept
Illustrate how the enforcer distributes true vs. surrogate calls across 25 BO steps for three SCR ceilings: 0%, 20%, 40%. Show which steps call the true function (green) and which use the surrogate (orange).

### Part 2 — Surrogate ranking quality
Generate 40 candidates, evaluate all with both the true function and the GP. Plot Spearman rank correlation ρ. High ρ is the prerequisite for trusting the surrogate's ranking when SCR > 0.

### Part 3 — SCR-controlled BO
Run a single BO with SCR ceiling = 20%. Track and plot the running SCR at each step. Show the enforcer keeping it at or below the ceiling.

### Part 4 — SCR sensitivity: convergence per true evaluation
Run the BO at SCR = 0%, 10%, 20%, 40%. X-axis = true function calls (COCO budget). Show that moderate SCR (10–20%) is nearly as good as pure BO but uses fewer true evaluations.

### Part 5 — Budget breakdown
After 30 iterations, compare how many evaluations were true vs. surrogate-only for each SCR setting. Show the trade-off between budget saved and SCR level.

### Part 6 — COCO-compatible BO
Full BO with hard true-evaluation budget. X-axis = true function calls only (COCO budget counter). Compare SCR=0% vs. SCR=20% convergence on Forrester.

---

## Exercises

1. In Part 2, reduce training data from 15 to 5 points. Does rank correlation ρ drop? What SCR is safe when ρ is low?
2. In Part 4, add SCR = 60% and SCR = 80%. At what SCR does convergence clearly worsen due to surrogate drift?
3. In Part 6, increase `TRUE_BUDGET = 80`. With SCR = 20%, does the BO converge to f* ≈ -6.021?

---

## What's Next

**Lesson 14** — All surrogates comparison with SCR < 20% enforced: GP, MC Dropout, Deep Ensembles, RBF, and Random Forest side-by-side on Forrester and Branin.
