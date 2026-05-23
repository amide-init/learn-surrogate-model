# Lesson 13 — Surrogate Control Ratio (SCR)

## Objective

Understand the correct definition of SCR, implement a generation-based prescreening loop that keeps SCR below 20%, and see why evaluating the true function in more than 20–50% of points is unacceptable in surrogate-assisted black-box optimisation.

---

## Core definition

```
SCR = true_function_calls / (true_function_calls + surrogate_calls)
```

- **SCR must be < 20%** — the expensive true function is called in at most 1 in 5 evaluations
- The surrogate handles the remaining 80%+ of evaluations (cheap predictions)
- The surrogate **saves** real evaluations; without control it does not

### Common mistake
A naive BO loop that calls the true function for every accepted point has SCR ≈ 100% — the surrogate guides search but saves nothing. This is "surrogate-assisted" in name only.

---

## Concepts

- **Prescreening** — generate a large pool of λ candidates, evaluate ALL with the surrogate (cheap), select only the top k by surrogate ranking for true evaluation; SCR = k / (k + λ)
- **Pool size λ** — how many candidates are generated and surrogate-evaluated per generation
- **Selection count k** — how many candidates are truly evaluated; k = ceil(SCR_target × λ)
- **Rank correlation** — Spearman ρ between surrogate ranking and true ranking; high ρ means prescreening is trustworthy; low ρ means the surrogate may discard the true best candidate
- **Surrogate drift** — if surrogate-predicted y values are added back as training data, the surrogate trains on its own errors; avoid by only adding truly-evaluated points to the dataset
- **COCO budget** — measured in true function calls only; SCR controls how efficiently each true call is used; lower SCR = more budget saved

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `notebook.ipynb` | Interactive — change SCR and λ, compare convergence per true eval |
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

### Part 1 — The prescreening model
Illustrate the generation-based loop: λ candidates → all surrogate-evaluated → top k truly evaluated. Show SCR for three settings: 10%, 20%, 50% (bad). Visualise which candidates receive true vs. surrogate evaluation.

### Part 2 — Surrogate ranking quality
Generate 50 candidates, evaluate all with both the true function and the GP. Plot Spearman rank correlation ρ. High ρ is the prerequisite for low SCR: the surrogate must correctly identify the top k without true evaluations.

### Part 3 — SCR-controlled prescreening loop
Implement the prescreening BO: λ=20 candidates per generation, k = ceil(SCR × λ) true evaluations. Track true calls, surrogate calls, and running SCR per generation. Show the enforcer keeping SCR ≤ 20%.

### Part 4 — SCR sensitivity: convergence per true evaluation
Run the prescreening loop at SCR = 5%, 10%, 20%, 50%, 100%. X-axis = true function calls. Show that SCR = 50–100% wastes budget; SCR = 10–20% is the efficient operating zone.

### Part 5 — Budget breakdown
After 40 generations, plot true vs. surrogate evaluations for each SCR setting. Show the dramatic difference in true-eval budget consumption.

### Part 6 — COCO-compatible prescreening BO
Full prescreening loop with hard SCR ≤ 20% enforcement. X-axis = true function calls only (COCO budget counter). Convergence on Forrester with the corrected SCR accounting.

---

## Exercises

1. In Part 2, reduce training data from 15 to 5 points. Does rank correlation ρ drop? What SCR is safe when ρ is low?
2. In Part 4, compare SCR = 50% with SCR = 10% on the same true-eval budget. How many more surrogate calls does SCR = 10% make? Does it converge better?
3. In Part 6, increase the pool size λ from 20 to 50. With SCR = 10%, how many true evals per generation does that give? Does a larger pool improve convergence?

---

## What's Next

**Lesson 14** — All surrogates comparison with SCR < 20% enforced: GP, MC Dropout, Deep Ensembles, RBF, and Random Forest side-by-side on Forrester and Branin.
