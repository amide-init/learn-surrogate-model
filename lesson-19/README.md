# Lesson 18 — Statistical Analysis and Paper Figures

## Objective

Re-run all key experiments with enough seeds for statistical power, apply non-parametric significance tests, compute bootstrap confidence intervals, and produce publication-quality figures and a results table for the research paper.

---

## Concepts

- **Bootstrap confidence interval** — resample seeds with replacement to estimate the CI on the median convergence curve without assuming normality
- **Wilcoxon signed-rank test** — non-parametric pairwise test comparing two surrogates across matched seeds; p < 0.05 indicates a significant difference
- **Friedman test** — non-parametric extension of repeated-measures ANOVA; tests whether any surrogate is significantly different from the others across multiple functions
- **Effect size (rank-biserial r)** — how large the difference is, independent of sample size; r > 0.5 = large effect
- **Publication figure** — clean typography, line-style-safe for B&W printing, 95% CI bands, correct axis labels
- **Results table** — median gap ± IQR, win rate vs GP baseline, recommended SCR — the main table for the paper

---

## Files

| File | Purpose |
|---|---|
| `README.md` | This file |
| `main.py` | Standalone script — saves all plots and prints the results table |
| `notebook.ipynb` | Interactive — reproduce each figure and inspect the statistics |

---

## Instructions

### Run the script

```bash
source .venv/bin/activate
python lesson-18/main.py
```

### Run the notebook

```bash
source .venv/bin/activate
jupyter notebook lesson-18/notebook.ipynb
```

---

## What You Will Build

### Part 1 — Bootstrap convergence curves (Forrester 1D)
5 seeds per surrogate. Median + 95% bootstrap CI bands. This is the primary convergence figure — used directly in the paper.

### Part 2 — Final gap distribution (box plots)
Box plots of final gap across all seeds. Reveals variance and outliers that the convergence curve hides. Identifies which surrogates are consistent vs volatile.

### Part 3 — Wilcoxon pairwise significance (Forrester 1D)
All 5×5 surrogate pairs tested with the Wilcoxon signed-rank test. Heatmap of p-values with significance markers (✓ = p < 0.05). Note how small N_SEEDS makes many differences non-significant.

### Part 4 — BBOB multi-function bar chart with error bars
Final gap (median ± IQR) across 3 BBOB functions for each surrogate. Grouped bar chart with error bars. Friedman test result printed below the chart.

### Part 5 — Paper Figure 1: main convergence comparison
Publication-quality 2-panel figure: Forrester (1D) and Branin (2D). Larger fonts, line styles safe for B&W printing, tight layout, ready for LaTeX \includegraphics.

### Part 6 — Results table
Summary table: surrogate × {median gap Forrester, median gap Branin, BBOB avg rank, recommended SCR}. Printed in Markdown and LaTeX format for copy-paste into the paper.

---

## Exercises

1. In Part 3, increase `N_SEEDS = 10` and re-run. How many more surrogate pairs become statistically significant? What does this tell you about experiment design?
2. In Part 1, does the 95% CI for GP and Deep Ensemble overlap? If so, can you claim GP is better in the paper?
3. In Part 5, try switching to grayscale by setting `PAPER_COLORS = ['black']*5` and relying solely on `PAPER_STYLES`. Is the figure still readable?

---

## What's Next

**Lesson 19** — Writing the research paper: structure, abstract, related work, and how to present surrogate-assisted optimisation results to reviewers.
