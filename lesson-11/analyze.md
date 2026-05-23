# Lesson 11 — Deep Ensembles Analysis

What does the output actually tell us about Deep Ensembles as a surrogate?
Reading every plot from the run.

---

## What happened in this run

- 15 random training points on Forrester `f(x) = (6x-2)² sin(12x-4)`
- 5 members × 3000 epochs each
- Seed per member: 0, 100, 200, 300, 400

**Final training losses:**

| Member | Final MSE |
|--------|-----------|
| 1 | 0.000000 (perfect fit) |
| 2 | 0.000023 |
| 3 | 0.000014 |
| 4 | 0.000014 |
| 5 | 0.000023 |

All five members trained to near-zero loss — essentially memorised the 15 points.

---

## Plot-by-plot observations

### Part 1 — Diversity (part1_diversity.png)

**What you see**: 5 coloured curves almost perfectly overlapping each other and the training data. The shaded "member range" band is nearly invisible.

**What it means**: Members have very low diversity. They all converged to nearly the same function. The source of ensemble uncertainty — disagreement between members — is almost absent here.

**Why**: With 15 points in 1D and 3000 epochs, every member over-fit the training data. When all members memorise the same 15 points, they end up with very similar weights → very similar predictions everywhere.

---

### Part 2 — Mean and uncertainty band (part2_ensemble_prediction.png)

**What you see**: The ensemble mean follows the true function closely. The ±2σ band is extremely thin — almost invisible — even in the gap region around x ≈ 0.5.

**What it means**: The ensemble is highly confident everywhere. It has nearly no uncertainty to report.

**The problem**: Confidence is not the same as accuracy. Look at x ≈ 0.4–0.6: the mean slightly deviates from the true function (grey dashed), but σ is still tiny. A well-calibrated surrogate should have larger σ where its mean is wrong — this one does not.

---

### Part 3 — Uncertainty calibration (part3_uncertainty_calibration.png)

**What you see**: σ(x) has one visible peak around x ≈ 0.5 reaching ~0.45. It is near zero everywhere else, including in the left extrapolation region (x < 0.1) where there is no training data.

**Good**: The peak at x ≈ 0.5 is correct — there is a real gap in the training data there and the ensemble does detect it.

**Bad**: 
- x < 0.1: no training points, but σ ≈ 0.07, barely above zero. A GP would show large uncertainty here.
- x > 0.9: one data point at the edge, σ ≈ 0.22. Still low.
- σ drops to zero between dense clusters even where the true function curves (e.g. x ≈ 0.3). The ensemble does not know it may be wrong there.

**Summary of calibration**: Partial. It detects the largest gap but misses uncertainty in extrapolation regions and underestimates uncertainty in smaller gaps.

---

### Part 4 — Training curves (part4_training_curves.png)

**What you see**: All 5 loss curves descend steeply in the first ~500 epochs, then continue declining slowly to near zero. The final losses differ by less than 0.000025.

**What it means**: Very small spread between members. The diversity that makes ensembles work comes from different local minima, but when all members converge to near-zero loss, they have all found essentially the same function — just encoded differently in weights. Low final-loss spread = low prediction diversity = low σ.

**Takeaway**: Ensemble diversity depends on training budget and model capacity. With a small dataset and many epochs, over-fitting collapses diversity.

---

### Part 5 — Deep Ensembles vs MC Dropout (part5_ensemble_vs_mcdropout.png, part5b_sigma_comparison.png)

**What you see in part5**: Both mean fits look similar and track the training data well. But MC Dropout's ±2σ band (red) is visibly wider than Deep Ensemble's band (blue).

**What you see in part5b**: 
- MC Dropout σ (red) grows almost continuously from left to right, reaching ~1.75 at x = 1.0.
- Deep Ensemble σ (blue) has one clean peak at x ≈ 0.5 and stays low everywhere else.
- MC Dropout σ is noisy/jagged. Deep Ensemble σ is smooth.

**Interpretation**:

| | MC Dropout | Deep Ensembles |
|---|---|---|
| σ shape | Noisy, grows toward x=1.0 | Smooth, single peak at data gap |
| σ magnitude | Much larger (0–1.75) | Smaller (0–0.5) |
| Sensitivity to data | Low — σ even large near training points | High — σ drops sharply near data |
| Noise in σ | Yes — stochastic at inference | No — deterministic, smooth |

**Which is better here?**
Neither is perfectly calibrated, but they fail differently:
- MC Dropout **over-estimates** uncertainty everywhere, especially toward the boundary. The noisy σ curve makes EI unreliable.
- Deep Ensembles **under-estimates** uncertainty in extrapolation but gives a clean, structured signal in the training region.

For a Bayesian Optimisation acquisition function, a smooth and data-aware σ is preferable over a noisy over-inflated one — so Deep Ensembles is easier to work with here, even if it's over-confident.

---

### Part 6 — One BO step (part6_bo_step.png)

**What you see**: EI has a single sharp spike at x ≈ 0.756. The ensemble picked x_next = 0.7559, f(x_next) = -6.020. The Forrester global minimum is at x ≈ 0.757, f ≈ -6.021.

**The ensemble found the global minimum in one step.**

**Why this worked**: The training data already had several points clustered near x ≈ 0.75–0.85, so the surrogate had a very accurate fit in that region. EI exploited the low-uncertainty region with the lowest predicted value — pure exploitation, no exploration needed.

**The concern**: EI produces a knife-edge spike with no width. There is no exploration of other parts of the domain. If the true minimum had been in x ≈ 0.5 (the high-uncertainty gap), the ensemble would have missed it — σ is low there and EI is zero.

---

## Summary: Deep Ensembles as a surrogate — Good and Bad

### What Deep Ensembles does well

1. **Smooth, structured uncertainty**: σ(x) is smooth and peaks where data is genuinely absent. MC Dropout produces a noisy σ that is harder to use in EI.

2. **Accurate mean fit**: With 15 points, the ensemble mean matches the true function very well. The averaging of 5 members smooths out individual member errors.

3. **Deterministic at inference**: No randomness at prediction time. Running the ensemble twice gives the same answer. This makes EI computation clean and reproducible.

4. **Scales to higher dimensions**: Each member trains like a plain neural network — no special inference trick. This scales naturally to 10D, 20D inputs where GPs struggle.

5. **BO compatibility**: Works as a drop-in surrogate. μ and σ feed directly into EI, PI, or UCB. Part 6 shows it immediately finds the minimum on a well-sampled problem.

---

### What Deep Ensembles does badly

1. **Over-confidence when members over-fit**: This is the most important problem visible in the plots. When training loss → 0, members converge to the same function → σ → 0 → the ensemble thinks it is certain everywhere. A GP would show growing uncertainty in the gaps.

2. **Uncertainty does not grow outside training range**: At x < 0.05 (extrapolation), σ ≈ 0.07. This is wrong. A good surrogate should be uncertain outside the training region. GPs handle this automatically via the kernel; ensembles do not.

3. **N× training cost**: 5 members = 5 complete training runs. In Part 6 we re-train the full ensemble after each BO step. With N=5, each BO iteration is 5× more expensive than a single MC Dropout or GP fit.

4. **No diversity without regularisation**: The ensemble only works if members disagree. With small data + many epochs, all members over-fit to the same points → zero diversity → σ = 0. Solutions: (a) fewer epochs, (b) weight decay, (c) dropout inside members, (d) different architectures.

5. **EI collapse to a spike**: Because σ is near zero almost everywhere, EI ≈ 0 almost everywhere and spikes sharply in one place. This produces pure exploitation with no exploration. In a longer BO loop, this leads to getting stuck in a local minimum.

---

## Key numbers from this run

| Metric | Value |
|--------|-------|
| Members | 5 |
| Training points | 15 |
| Epochs per member | 3000 |
| Mean final loss | 0.000011 |
| Max σ observed | ~0.45 (at x ≈ 0.5) |
| MC Dropout max σ | ~1.75 (at x ≈ 1.0) |
| x_next from EI | 0.7559 |
| f(x_next) | -6.020 |
| Forrester global min | ≈ -6.021 at x ≈ 0.757 |

---

## When to use Deep Ensembles

**Use Deep Ensembles when:**
- You have enough budget to train N networks (N× wall-clock time is acceptable)
- You need smooth, clean σ for reliable EI computation
- The input dimension is high (>5D) where GPs become slow
- You want deterministic, reproducible predictions at inference

**Do NOT use Deep Ensembles when:**
- Training data is very small (< ~10 points) — members collapse to the same solution
- Each surrogate refit must be fast — N× cost per BO step adds up quickly
- You need calibrated extrapolation uncertainty — ensembles are blind outside the training range
- Diversity matters more than speed — MC Dropout is cheaper for similar (or better) raw uncertainty coverage

---

## What Lesson 12 will compare

Lesson 12 runs both MC Dropout and Deep Ensembles against the GP baseline on Forrester and Branin. The plots here give you a prediction:

- **Deep Ensembles mean**: probably comparable to GP mean, slightly worse calibration
- **Deep Ensembles σ**: under-estimated compared to GP, especially in extrapolation
- **MC Dropout σ**: over-estimated and noisy, but at least it explores
- **BO convergence**: GP likely wins in early iterations; NN surrogates may catch up with more data
