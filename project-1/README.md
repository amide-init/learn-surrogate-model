# Project 1 — Deep Ensemble Surrogate-Assisted CMA-ES on COCO/BBOB

## Objective

Benchmark a Deep Ensemble surrogate paired with CMA-ES on the COCO/BBOB suite and produce convergence data for comparison against lq-CMA-ES, LMM-CMA-ES, and DTS-CMA-ES.

## Algorithm

1. **Initialise** — LHS sampling of `N_INIT_FACTOR × dim` points
2. **Train** — fit a Deep Ensemble (5 networks) on the initial data
3. **Search** — CMA-ES loop:
   - Pre-screen all candidates with the surrogate
   - Evaluate the top `SCR` fraction on the real function
   - Append real evaluations to the dataset; retrain the surrogate
4. **Save** — convergence history per function/dimension instance

## Key Parameters (`main.py`)

| Variable | Default | Description |
|---|---|---|
| `BUDGET_FACTOR` | 100 | Real evaluations = `BUDGET_FACTOR × dim` |
| `SCR` | 0.2 | Fraction of CMA-ES candidates evaluated on real function |
| `N_INIT_FACTOR` | 5 | LHS initial points = `N_INIT_FACTOR × dim` |
| `N_INSTANCES` | 15 | COCO instances per (function, dimension) |
| `DIMENSIONS` | [2, 5, 10] | Search space dimensions |
| `FUNCTION_IDS` | range(1, 25) | BBOB function indices to run |

## Running

```bash
source .venv/bin/activate
python project-1/main.py
```

## Results

| Path | Format | Description |
|---|---|---|
| `results/dim_{d}_fun_{f}.npy` | `(N_INSTANCES, budget)` float array | Best-so-far per real evaluation, one row per instance |
| `results/coco_output/` | COCO format | For post-processing with `cocopp` |

## Comparison

Results will be compared against:
- **lq-CMA-ES** — local quadratic surrogate
- **LMM-CMA-ES** — limited memory matrix surrogate
- **DTS-CMA-ES** — distance-based tournament selection
