# CLAUDE.md

## Project

A step-by-step Python learning project for surrogate-assisted optimisation, structured as numbered lessons. The end goal is to have enough theory, code, and experiments to write a research paper. Each lesson is self-contained and builds on the previous one.

---

## Lesson Structure

Every lesson lives in its own folder named `lesson-N` (e.g. `lesson-1`, `lesson-2`).

Each lesson folder must contain:

```
lesson-N/
├── README.md        # What this lesson teaches, concepts, instructions, exercises
├── notebook.ipynb   # Jupyter notebook — interactive exploration and visualisations
└── main.py          # Clean Python script — runnable, no notebook overhead
```

Additional Python files are allowed inside a lesson folder if the lesson needs helper modules (e.g. `gp.py`, `kernels.py`). Keep them flat inside the lesson folder — no sub-folders within lessons.

### README.md per lesson

Each lesson README must cover:
1. **Objective** — one sentence on what the student will learn
2. **Concepts** — bullet list of topics covered
3. **Instructions** — how to run the notebook and the script
4. **Exercises** — 2–3 small tasks for the student to try on their own
5. **What's next** — one line pointing to the next lesson

### notebook.ipynb per lesson

- Use markdown cells to explain concepts before code cells
- Every plot must have a title, axis labels, and legend
- Cells must run top-to-bottom without error
- Keep cells short — one idea per cell

### main.py per lesson

- Runnable standalone: `python lesson-N/main.py`
- Produces the same key outputs as the notebook (plots saved to `lesson-N/output/`)
- No interactive input required
- Set `RANDOM_SEED = 42` at the top of every script

---

## Full Lesson Map

| Folder | Stage | Topic |
|---|---|---|
| `lesson-1` | 0 | Python and scientific computing basics (NumPy, SciPy, Matplotlib) |
| `lesson-2` | 1 | What is optimisation? Gradient descent and black-box problems |
| `lesson-3` | 2 | What is a surrogate? Polynomial regression as a first surrogate |
| `lesson-4` | 3 | Statistics refresher: Gaussians, covariance, conditional distributions |
| `lesson-5` | 4 | Kernels and similarity functions |
| `lesson-6` | 5 | Gaussian Process regression from scratch |
| `lesson-7` | 6 | Acquisition functions: PI, EI, UCB |
| `lesson-8` | 7 | The Bayesian Optimisation loop |
| `lesson-9` | 8 | Surrogate model comparison (GP, RBF, RF, polynomial) |
| `lesson-10` | 9 | Handling noisy observations |
| `lesson-11` | 10 | High-dimensional surrogates and ARD kernels |
| `lesson-12` | 11 | Surrogate-Assisted Evolutionary Algorithms (SAEA) |
| `lesson-13` | 12 | COCO / BBOB benchmark evaluation |
| `lesson-14` | 13 | Statistical analysis and paper figures |
| `lesson-15` | 14 | Writing the research paper |

---

## Directory Layout

```
learn-surrogate/
├── lesson-1/
│   ├── README.md
│   ├── notebook.ipynb
│   ├── main.py
│   └── output/          # Saved plots (gitignored)
├── lesson-2/
│   ├── README.md
│   ├── notebook.ipynb
│   └── main.py
│   └── output/
├── ...
├── lesson-15/
│   └── README.md        # Writing guide only — no code
├── results/             # Shared benchmark outputs (gitignored)
└── requirements.txt
```

---

## Language and Style

- Python only
- No type annotations unless asked
- No docstrings unless asked — good naming is enough
- Use `numpy`/`scipy`/`scikit-learn` before any heavier library
- Always set `RANDOM_SEED = 42` at the top of scripts
- Save all plots to `lesson-N/output/` — never call `plt.show()` in `main.py`
- Notebooks may use `plt.show()`

---

## Running a Lesson

```bash
source .venv/bin/activate

# Run the script
python lesson-1/main.py

# Launch the notebook
jupyter notebook lesson-1/notebook.ipynb
```

---

## COCO-specific Notes (Lesson 13 onward)

- Use `coco-experiment` (`import cocoex`) for running experiments, `cocopp` for post-processing
- COCO output goes in `results/exdata/` (gitignored)
- Default budget: `100 * dimension` function evaluations for early runs
- Always pass `random_seed` to the experiment for reproducibility

---

## What to Avoid

- Do not skip lessons — each one introduces a concept the next one depends on
- Do not add abstractions before they are needed
- Do not use heavy frameworks (PyTorch, TensorFlow) before Lesson 9
- Do not mock expensive function evaluations — run on small budgets instead
- Do not put shared code outside lesson folders until Lesson 9 (model comparison) makes it necessary
