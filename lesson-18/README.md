# Lesson 17 — COCO / BBOB Benchmark

## Objective

Evaluate all five surrogates on four representative BBOB functions (Sphere, Ellipsoidal, Rosenbrock, Rastrigin) using the cocoex test suite, COCO-correct true-call counting, and a simplified performance profile.

---

## Concepts

- **BBOB suite** — 24 noiseless black-box functions from the COCO framework, each with instance-specific rotations and offsets; covers unimodal, ill-conditioned, multimodal landscapes
- **cocoex** — Python library providing BBOB functions as callable objects; inputs on [−5, 5]^d
- **Budget** — total number of true function evaluations; COCO standard = 100 × d; here we use a smaller budget for speed
- **Performance profile** — fraction of (seed, target) pairs solved within a given budget; x = true calls, y = fraction solved
- **Target** — a gap threshold τ; a run is "solved" when gap = f(x_best) − f* < τ
- **f_opt** — the true function minimum; estimated via multi-start scipy.minimize since BBOB optima are instance-specific
- **Dimension scaling** — how each surrogate's final gap grows as d increases from 2 to 5
- **SCR ≤ 20%** — same enforcer as all previous lessons

---

## BBOB Functions Used

| ID | Name | Character |
|---|---|---|
| f1 | Sphere | Unimodal, separable, easy |
| f2 | Ellipsoidal | Unimodal, ill-conditioned (λ_max/λ_min = 10⁶) |
| f8 | Rosenbrock | Narrow banana-shaped valley, deceptive |
| f15 | Rastrigin | Highly multimodal, many local minima |

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `main.py` | Standalone script — saves all plots to `output/` |
| `notebook.ipynb` | Interactive — explore each BBOB function and compare surrogates |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-17/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-17/notebook.ipynb
```

---

## What You Will Build

### Part 1 — BBOB function gallery
2D landscape contour plots of all four BBOB functions at d=2. Shows the landscape character before any optimisation.

### Part 2 — Convergence on Sphere (f1, d=2)
All 5 surrogates × 3 seeds × SCR≤20% BO. Convergence curves (true calls vs gap). Sphere is the easiest — every surrogate should converge well.

### Part 3 — Performance profile (f1, d=2)
For each surrogate: fraction of (seed, target) combinations solved within budget. Targets = {10, 1, 0.1, 0.01}. A faster rising curve = better anytime performance.

### Part 4 — Multi-function heatmap
Run all surrogates on all 4 BBOB functions. Heatmap: rows = surrogates, columns = functions, cell = log₁₀(final gap). Reveals which surrogate is best for which landscape type.

### Part 5 — Dimension scaling (Sphere: d=2 vs d=5)
Compare final gap at d=2 and d=5 for each surrogate. Shows how performance degrades as dimension increases.

### Part 6 — Leaderboard
Rank surrogates by average performance across all 4 BBOB functions. Shows the overall best surrogate for BBOB-style benchmarks.

---

## Exercises

1. In Part 4, which function is hardest for all surrogates? Is this consistent with the landscape character shown in Part 1?
2. In Part 3, does GP hit the tightest target (τ=0.01) more reliably than neural network surrogates? What does this say about calibration?
3. In Part 5, try `DIM_HD = 10`. Which surrogate degrades least? Does this match the finding from Lesson 15?

---

## What's Next

**Lesson 18** — Statistical analysis and paper figures: aggregate results across all lessons, compute significance tests, and produce publication-quality plots.
