# CLAUDE.md

## Project

A step-by-step Python learning project for surrogate-assisted optimisation, structured as numbered lessons. The end goal is a research paper using a **neural network (PyTorch + MC Dropout) as the primary surrogate model**.

---

## Primary Models: Two NN Uncertainty Methods

Both methods are main contributions. GP, RBF, and Random Forest are baselines.

### Method A — MC Dropout (Lesson 10)
```
Input(d) → Linear(64) → ReLU → Dropout(0.1)
         → Linear(64) → ReLU → Dropout(0.1)
         → Linear(1)

Predict: run 50 forward passes with dropout ON → mean μ, variance σ²
```
- Single network, cheap to train
- Reference: Gal & Ghahramani (2016)

### Method B — Deep Ensembles (Lesson 11)
```
5 × [Input(d) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(1)]
Each trained independently with a different random seed.

Predict: run all 5 networks → mean μ, variance σ² across outputs
```
- Better calibrated uncertainty than MC Dropout
- Reference: Lakshminarayanan et al. (2017)

**Lesson 12 compares both methods head-to-head** — this comparison is a core paper result.

**When implementing surrogates:** implement both NN methods. GP is always a baseline.

---

## Lesson Structure

Every lesson lives in its own folder named `lesson-N`.

Each lesson folder must contain:

```
lesson-N/
├── README.md        # Objective, concepts, instructions, exercises, what's next
├── notebook.ipynb   # Jupyter notebook — interactive exploration and visualisations
├── main.py          # Standalone runnable script
└── output/          # Saved plots (gitignored)
```

Additional Python files are allowed inside a lesson folder if the lesson needs helper modules (e.g. `nn_surrogate.py`, `gp.py`). Keep them flat — no sub-folders within lessons.

### README.md per lesson
1. **Objective** — one sentence
2. **Concepts** — bullet list
3. **Instructions** — how to run notebook and script
4. **Exercises** — 2–3 tasks for the student
5. **What's next** — one line to next lesson

### notebook.ipynb per lesson
- Markdown cell explaining the concept before every code cell
- Every plot: title, axis labels, legend
- Cells run top-to-bottom without error
- One idea per cell

### main.py per lesson
- Runnable standalone: `python lesson-N/main.py`
- Saves plots to `lesson-N/output/` — never call `plt.show()`
- No interactive input
- `RANDOM_SEED = 42` at the top

---

## Full Lesson Map

| Folder | Topic | Role |
|---|---|---|
| `lesson-1` | Python, NumPy, SciPy, Matplotlib | Tooling |
| `lesson-2` | Optimisation, gradient descent, black-box problems | Background |
| `lesson-3` | First surrogate — polynomial regression | Intuition |
| `lesson-4` | Statistics: Gaussians, covariance, conditional distributions | Math foundation |
| `lesson-5` | Kernels and similarity functions | GP background |
| `lesson-6` | Gaussian Process regression | Baseline model |
| `lesson-7` | Acquisition functions: PI, EI, UCB | Core algorithm |
| `lesson-8` | Bayesian Optimisation loop with GP | Baseline loop |
| `lesson-9` | PyTorch basics: tensors, `nn.Module`, training loop | NN tooling |
| `lesson-10` | **NN surrogate + MC Dropout** | **Main contribution A** |
| `lesson-11` | **NN surrogate + Deep Ensembles** | **Main contribution B** |
| `lesson-12` | MC Dropout vs. Deep Ensembles vs. GP | Core comparison |
| `lesson-13` | **Surrogate control strategies (SCR)** | **Critical concept** |
| `lesson-14` | All surrogates comparison (NN, GP, RBF, RF) | Experiments |
| `lesson-15` | Noise handling and high-dimensional inputs | Robustness |
| `lesson-16` | **SCR sensitivity — varying surrogate usage from 0% to 80%** | **SCR tuning** |
| `lesson-17` | COCO / BBOB benchmark | Evaluation |
| `lesson-18` | Statistical analysis and paper figures | Results |
| `lesson-19` | Writing the research paper | Paper |

---

## Directory Layout

```
learn-surrogate/
├── lesson-1/
│   ├── README.md
│   ├── notebook.ipynb
│   ├── main.py
│   └── output/
├── lesson-2/ ... lesson-15/
├── results/             # Shared benchmark outputs (gitignored)
└── requirements.txt
```

---

## Language and Style

- Python only
- PyTorch for all neural network code (not TensorFlow or Keras)
- No type annotations unless asked
- No docstrings unless asked — good naming is enough
- Use `numpy`/`scipy`/`scikit-learn` for non-NN code
- `RANDOM_SEED = 42` at the top of every script
- Save all plots to `lesson-N/output/` — never call `plt.show()` in `main.py`
- Notebooks may use `plt.show()`

---

## Running a Lesson

```bash
source .venv/bin/activate

# Run the script
python lesson-N/main.py

# Launch the notebook
jupyter notebook lesson-N/notebook.ipynb
```

---

## COCO-specific Notes (Lesson 13)

- Use `coco-experiment` (`import cocoex`) for running experiments, `cocopp` for post-processing
- COCO output goes in `results/exdata/` (gitignored)
- Default budget: `100 * dimension` evaluations for early runs
- Baselines to include: GP-BO, CMA-ES, random search
- Always pass `random_seed` for reproducibility

---

## What to Avoid

- Do not skip lessons — each one introduces a concept the next depends on
- Do not use GP as the main model — it is always a baseline
- Do not skip Lesson 10 before Lesson 11 — Deep Ensembles build on the same BO loop
- Do not skip Lesson 13 (SCR) before Lesson 16 (COCO) — without correct surrogate control, COCO results are invalid
- Surrogate control ratio (SCR) must be < 20% in all experiments — the surrogate must save real evaluations
- Do not add abstractions before they are needed
- Do not introduce PyTorch before Lesson 9
- Do not mock expensive function evaluations — run on small budgets instead
- Do not put shared code outside lesson folders until Lesson 14 makes it necessary
