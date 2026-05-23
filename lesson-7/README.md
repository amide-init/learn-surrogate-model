# Lesson 7 — Acquisition Functions: PI, EI, UCB

## Objective

Given the GP posterior from Lesson 6, implement the three most common acquisition functions — Probability of Improvement (PI), Expected Improvement (EI), and Upper Confidence Bound (UCB) — and show how they decide where to evaluate the expensive function next.

---

## Concepts

- **Acquisition function** — a cheap surrogate of "how valuable is evaluating f at x?" computed from the GP mean μ(x) and std σ(x)
- **Exploration vs exploitation** — exploitation: evaluate where μ is low (likely improvement); exploration: evaluate where σ is high (uncertain region)
- **Probability of Improvement (PI)** — P(f(x) < f_best − ξ) = Φ(Z), Z = (f_best − ξ − μ) / σ
- **Expected Improvement (EI)** — E[max(f_best − ξ − f(x), 0)] = (f_best − ξ − μ) Φ(Z) + σ φ(Z); most used in practice
- **Upper/Lower Confidence Bound (UCB/LCB)** — LCB = μ(x) − κ σ(x); κ directly controls exploration; theoretically grounded
- **ξ (xi)** — jitter parameter in PI and EI; ξ > 0 forces exploration beyond the current best
- **κ (kappa)** — exploration weight in UCB; small κ exploits, large κ explores
- **x_next = argmax α(x)** — the next evaluation point is wherever the acquisition function peaks
- **One BO step** — GP posterior → acquisition → argmax → evaluate f → add to data → refit GP

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `notebook.ipynb` | Interactive — tweak ξ and κ, watch where x_next moves |
| `main.py` | Standalone script — saves all plots to `output/` |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-7/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-7/notebook.ipynb
```

---

## What You Will Build

### Part 1 — Why acquisition functions?
Annotate a GP posterior with the exploitation zone (low μ) and exploration zone (high σ). Show that neither alone is enough — we need to balance both.

### Part 2 — Probability of Improvement (PI)
Implement PI from scratch. Show the acquisition curve for ξ = 0, 0.05, 0.1. See how ξ = 0 is greedy and ξ > 0 encourages exploration.

### Part 3 — Expected Improvement (EI)
Implement EI from scratch. EI measures the expected size of improvement, not just the probability — so it avoids getting stuck near the current best. Compare ξ = 0, 0.01, 0.1.

### Part 4 — Upper Confidence Bound (UCB / LCB)
LCB = μ − κσ for minimization. Argmin of LCB = point that is most likely to be low OR most uncertain. Show κ = 0.5, 2.0, 4.0.

### Part 5 — All three compared
Overlay normalized PI, EI, and UCB on the same GP. Show where each function suggests evaluating next and how they differ.

### Part 6 — One complete BO step
Pick x_next using EI, evaluate f(x_next), add to training set, refit GP. Show before/after. This is the one-step loop that Lesson 8 repeats.

---

## Exercises

1. In Part 2, set ξ = 0 for PI. What happens when the current best is already very close to the true minimum? Does the algorithm keep exploring or get stuck?
2. In Part 3, implement a variant of EI that uses the **GP mean** instead of f_best as the baseline. How does the acquisition surface change?
3. In Part 4, increase κ to 10.0. Where does the UCB point? Is this still a sensible next evaluation? What does this tell you about tuning κ?

---

## What's Next

**Lesson 8** — Bayesian Optimisation loop: combine the GP from Lesson 6 and the acquisition functions from this lesson into a full iterative optimisation loop on the Forrester function.
