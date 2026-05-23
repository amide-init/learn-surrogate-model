# Lesson 10 — NN Surrogate with MC Dropout

## Objective

Build the first main surrogate model: a neural network with MC Dropout that produces both a mean prediction and an uncertainty estimate — making it usable as a drop-in replacement for the GP inside the Bayesian Optimisation loop.

---

## Concepts

- **MC Dropout** — keep `nn.Dropout` active at inference time; each forward pass gives a slightly different prediction because different neurons are randomly zeroed
- **Monte Carlo estimation** — run T stochastic forward passes, compute mean μ and std σ across outputs
- **`model.train()` at inference** — the critical implementation detail: calling `train()` instead of `eval()` keeps dropout ON
- **Epistemic uncertainty** — uncertainty from lack of data; σ(x) grows where training data is sparse; captured by MC Dropout
- **Reference** — Gal & Ghahramani (2016): *Dropout as a Bayesian Approximation*
- **Architecture** — `Input(d) → Linear(64) → ReLU → Dropout(0.1) → Linear(64) → ReLU → Dropout(0.1) → Linear(1)`
- **T=50 passes** — standard choice; more passes → smoother uncertainty but slower

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `notebook.ipynb` | Interactive — change dropout rate and T, watch uncertainty change |
| `main.py` | Standalone script — saves all plots to `output/` |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-10/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-10/notebook.ipynb
```

---

## What You Will Build

### Part 1 — Dropout: eval() vs train() mode
Train an MCDropoutMLP. Call `model.eval()` five times on the same input → identical output each time. Call `model.train()` five times → different output each time. This one-line difference is the entire mechanism.

### Part 2 — 50 MC passes
Run 50 stochastic forward passes. Plot each one as a thin line. Compute the mean (solid line) and ±2σ band. This is the uncertainty estimate.

### Part 3 — Uncertainty scales with data sparsity
Show that σ(x) is small near training points and large in unexplored regions — exactly the behaviour needed for an acquisition function.

### Part 4 — Training and surrogate fit
Train MCDropoutMLP on 15 Forrester points with standardisation. Compare training loss with a plain MLP. Show the final fit with the MC Dropout uncertainty band.

### Part 5 — MC Dropout vs GP surrogate
Fit both MC Dropout and GP to the same 15 training points. Plot mean and uncertainty side-by-side. Identify where they agree and where they differ.

### Part 6 — One BO step with MC Dropout
Replace the GP with MCDropoutMLP as the surrogate in one complete BO step: fit → compute EI from MC Dropout μ and σ → evaluate → update.

---

## Exercises

1. In Part 2, change `n_passes=5`. Does the uncertainty estimate become noisier? What about `n_passes=200`?
2. In Part 3, change `dropout_rate=0.5`. Does the uncertainty band widen? Is it still calibrated near training points?
3. In Part 6, run 10 BO iterations with MCDropoutMLP. Does it find the Forrester minimum? Compare the convergence curve to the GP-BO curve from Lesson 8.

---

## What's Next

**Lesson 11** — Deep Ensembles surrogate: train 5 independent MLPs (no dropout) and compute mean/variance across their predictions. Better calibrated than MC Dropout but 5× more expensive to train.
