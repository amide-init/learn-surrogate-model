# Lesson 16 — SCR Sensitivity: Varying Surrogate Usage from 0% to 80%

## Objective

Systematically vary the SCR ceiling from 0% (pure true evaluations) to 80% (heavy surrogate reliance) and measure the impact on optimization quality and true-evaluation savings for every surrogate.

---

## Concepts

- **SCR ceiling** — the maximum allowed fraction of surrogate-only steps; `SCR = surrogate_only / (true + surrogate_only)`
- **True evaluation savings** — extra true calls avoided vs SCR=0%; higher SCR = more savings but higher risk
- **Quality penalty** — degradation in final gap vs the SCR=0% baseline; acceptable if small
- **Quality–efficiency frontier** — scatter of (true calls used, final gap): the goal is the lower-left corner
- **Surrogate trustworthiness** — determines the safe SCR ceiling; well-calibrated surrogates tolerate higher SCR
- **Recommended SCR** — highest ceiling where quality penalty stays below 50% of the baseline gap

---

## SCR Levels Tested

| SCR ceiling | Surrogate-only steps | True calls saved (approx.) |
|---|---|---|
| 0% | 0 out of 15 | 0 |
| 20% | 3 out of 15 | 3 |
| 40% | 6 out of 15 | 6 |
| 60% | 9 out of 15 | 9 |
| 80% | 12 out of 15 | 12 |

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `main.py` | Standalone script — saves all plots to `output/` |
| `notebook.ipynb` | Interactive — explore the SCR trade-off |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-16/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-16/notebook.ipynb
```

---

## What You Will Build

### Part 1 — GP convergence curves per SCR level (Forrester 1D)
One convergence curve per SCR level (x-axis = true calls). Shows that low SCR levels are indistinguishable from the baseline while high SCR levels cause slower convergence.

### Part 2 — True calls saved per SCR level
Bar chart: average true function calls used across seeds for each SCR level (GP on Forrester). Annotations show percentage savings vs SCR=0%.

### Part 3 — Quality–efficiency frontier (all surrogates)
Scatter plot: x = true calls used, y = final gap to optimum. Each surrogate is one colored line with one point per SCR level. The ideal position is the lower-left corner (few true calls AND low gap).

### Part 4 — Final gap per surrogate × SCR level
Grouped bar chart: final gap for every (surrogate, SCR) combination. Reveals which surrogates maintain quality at high SCR and which degrade.

### Part 5 — Branin (2D) validation
Repeat Part 4 on the 2D Branin function using SCR ∈ {0%, 40%, 80%}. Tests whether the per-surrogate SCR tolerance generalises to higher dimensions.

### Part 6 — Leaderboard and recommendations
Summary: recommended max SCR per surrogate (highest ceiling where quality penalty < 50%), savings at that ceiling, and quality penalty.

---

## Exercises

1. In Part 3, which surrogate has the best lower-left position? At what SCR level does it achieve this?
2. In Part 4, at SCR=80%, which surrogate degrades the most? Can you explain why based on its uncertainty estimate?
3. Change `N_ITER = 30` and re-run Parts 1 and 2. Does a longer budget make high SCR safer (more data → better surrogate)?

---

## What's Next

**Lesson 17** — COCO / BBOB benchmark: evaluate all surrogates on the official noiseless BBOB suite with budget = 100 × d evaluations.
