# Lesson 19 — Writing the Research Paper

## Objective

Assemble all experimental results into publication-ready figures and a complete LaTeX paper scaffold that you can submit to a venue.

---

## Concepts

- **Paper Figure 1** — main convergence comparison (Forrester 1D + Branin 2D), bootstrap 95% CI, B&W-safe line styles
- **Paper Figure 2** — SCR sensitivity: quality vs. true-evaluation savings scatter for all surrogates
- **Paper Figure 3** — noise robustness: RMSE vs. σ_noise for each surrogate
- **Paper Figure 4** — BBOB leaderboard: average rank bar chart + rank heatmap
- **Paper Figure 5** — SCR enforcer timeline: running SCR trace showing how the budget is allocated
- **LaTeX scaffold** — complete compilable paper template with Abstract, Introduction, Method, Experiments, Results, and Conclusion sections pre-filled with result placeholders

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `main.py` | Standalone script — runs all experiments, saves 5 figures + `paper_scaffold.tex` |
| `notebook.ipynb` | Interactive — reproduce each figure with guidance on where each fits in the paper |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-19/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-19/notebook.ipynb
```

---

## What You Will Build

### Part 1 — Paper Figure 1: Main convergence comparison
2-panel figure (Forrester 1D left, Branin 2D right). Median convergence + 95% bootstrap CI bands. Larger fonts, B&W-safe line styles, tight layout, ready for `\includegraphics`.

### Part 2 — Paper Figure 2: SCR sensitivity scatter
One point per (surrogate, SCR level). X-axis = true evaluations saved (%), Y-axis = median final gap. Shows the quality-savings frontier.

### Part 3 — Paper Figure 3: Noise robustness
RMSE vs. σ_noise curves for all surrogates on noisy Forrester. Reveals which surrogate degrades gracefully.

### Part 4 — Paper Figure 4: BBOB leaderboard
Top panel: average rank bar chart (lower is better). Bottom panel: rank heatmap (surrogate × BBOB function).

### Part 5 — Paper Figure 5: SCR enforcer timeline
Single run trace: green bars = true evaluation, orange bars = surrogate evaluation. Running SCR line overlaid. Shows the enforcer keeping SCR ≤ 20%.

### Part 6 — LaTeX scaffold
Saves `lesson-19/output/paper_scaffold.tex` — a compilable NeurIPS/ICML-style template with all sections pre-filled with `[PLACEHOLDER]` markers that you replace with your final numbers and prose.

---

## Exercises

1. In Part 1, try `PAPER_COLORS = ['black'] * 5` and rely solely on `PAPER_STYLES`. Is the figure still readable in grayscale? Which surrogates become hardest to distinguish?
2. In Part 2, add a third axis showing wall-clock time saved. Does the Pareto frontier change when you account for surrogate training cost?
3. Open `paper_scaffold.tex` and fill in the `[PLACEHOLDER]` fields using the printed results table. Try compiling it with `pdflatex paper_scaffold.tex`.

---

## What's Next

You have completed the full surrogate-assisted optimisation curriculum. The research paper is the final deliverable.
