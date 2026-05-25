# Lesson 12 — CMA-ES as the Acquisition Optimiser (Deep Ensembles)

## Objective
Replace random candidate sampling with CMA-ES to maximise the Expected Improvement (EI) acquisition function over a Deep Ensemble surrogate, and show why this matters as problem dimension grows.

## Concepts
- **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy): a derivative-free evolutionary optimiser that adapts its search distribution to the landscape of the objective.
- **Acquisition optimisation**: finding the input x that maximises EI (or UCB). In low dimensions random sampling works; in higher dimensions it fails badly.
- **Curse of dimensionality**: 2000 random candidates cover a negligible fraction of a 6D unit cube — CMA-ES explores systematically instead.
- **EI via Deep Ensembles**: EI needs μ(x) and σ(x); the ensemble provides both without dropout or GP fitting.
- **BO loop with CMA-ES**: fit ensemble → CMA-ES maximises EI → evaluate → repeat.

## Architecture (Deep Ensemble — unchanged from Lesson 11)
```
5 × [Input(d) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(1)]
μ = mean of 5 outputs,  σ = std of 5 outputs
EI(x) = (f_best − μ) · Φ(Z) + σ · φ(Z),   Z = (f_best − μ) / σ
```

## CMA-ES role
```
x_next = argmax_{x ∈ [0,1]^d} EI(x)

Random:  sample 2000 random candidates → pick highest EI   (works in 1D–2D)
CMA-ES:  start from x0 ∈ [0,1]^d, σ=0.3
         iterate: ask → evaluate neg_EI → tell → adapt covariance
         converges in ~100–200 function evaluations of EI
```

## Instructions

```bash
source .venv/bin/activate

# Script (~15–25 min depending on hardware)
python lesson-12/main.py

# Notebook
jupyter notebook lesson-12/notebook.ipynb
```

## Outputs (lesson-12/output/)
| File | Description |
|---|---|
| `part1_cmaes_mechanics.png` | CMA-ES convergence on Forrester EI |
| `part2_acq_quality.png` | CMA-ES vs random EI quality across 1D / 2D / 6D |
| `part3_convergence_forrester.png` | BO convergence — Forrester 1D |
| `part4_convergence_branin.png` | BO convergence — Branin 2D |
| `part5_convergence_hartmann6.png` | BO convergence — Hartmann-6 (6D) |
| `part6_trace_hartmann6.png` | Detailed BO trace, best seed, Hartmann-6 |

## Exercises
1. Change `N_MEMBERS = 5` to `N_MEMBERS = 1` (single network). Does CMA-ES still outperform random? Why might σ quality matter?
2. Replace EI with UCB (`μ − β·σ`, β = 2) and repeat Part 3. Does CMA-ES advantage change?
3. Increase `n_cand = 2000` to `20000` for random search. At what dimension does CMA-ES still win?

## What's next
Lesson 13 — head-to-head comparison: GP vs MC Dropout vs Deep Ensembles (all using CMA-ES acquisition).
