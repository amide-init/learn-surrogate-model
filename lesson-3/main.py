import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

X_PLOT = np.linspace(0, 1, 300)


# ── True expensive functions ──────────────────────────────────────────────────

def forrester(x):
    """Standard 1D benchmark. Pretend each call costs 1 hour of compute."""
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

def multimodal(x):
    return np.sin(8 * np.pi * x) + 0.5 * x


# ── Surrogate: polynomial regression ─────────────────────────────────────────

def fit_poly(X_train, y_train, degree):
    coeffs = np.polyfit(X_train, y_train, degree)
    return np.poly1d(coeffs)

def surrogate_min(poly, x_grid):
    """Find the minimum of the surrogate on a dense grid."""
    y_hat = poly(x_grid)
    return x_grid[np.argmin(y_hat)]


# ── Part 1: The surrogate concept ─────────────────────────────────────────────

n_init = 5
X_train = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
y_train = forrester(X_train)

poly = fit_poly(X_train, y_train, degree=3)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(X_PLOT, forrester(X_PLOT), color="steelblue", linewidth=2,
             label="True f(x)  [hidden — expensive]")
axes[0].plot(X_PLOT, poly(X_PLOT), color="tomato", linewidth=2,
             linestyle="--", label="Surrogate (degree-3 polynomial)")
axes[0].scatter(X_train, y_train, color="black", s=80, zorder=5, label="Observed points")
axes[0].set_title("A surrogate approximates f(x) from a few evaluations")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residual error
error = np.abs(forrester(X_PLOT) - poly(X_PLOT))
axes[1].fill_between(X_PLOT, 0, error, color="tomato", alpha=0.4, label="Surrogate error |f − ŝ|")
axes[1].plot(X_PLOT, error, color="tomato", linewidth=1.5)
axes[1].scatter(X_train, np.zeros_like(X_train), color="black", s=80, zorder=5,
                label="Training points (error = 0)")
axes[1].set_title("Surrogate error across the input space")
axes[1].set_xlabel("x")
axes[1].set_ylabel("|True − Surrogate|")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part1_surrogate_concept.png", dpi=150)
plt.close()
print("Saved: part1_surrogate_concept.png")


# ── Part 2: Effect of number of training points ───────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

rng = np.random.default_rng(42)
for ax, n in zip(axes, [3, 6, 12]):
    X_n = np.sort(rng.uniform(0, 1, n))
    y_n = forrester(X_n)
    poly_n = fit_poly(X_n, y_n, degree=min(n - 1, 5))

    ax.plot(X_PLOT, forrester(X_PLOT), color="steelblue", linewidth=2, label="True f(x)")
    ax.plot(X_PLOT, poly_n(X_PLOT), color="tomato", linewidth=2,
            linestyle="--", label="Surrogate")
    ax.scatter(X_n, y_n, color="black", s=80, zorder=5, label=f"{n} points")
    ax.set_title(f"{n} training points")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-8, 18)

plt.suptitle("More training points → better surrogate", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part2_num_points.png", dpi=150)
plt.close()
print("Saved: part2_num_points.png")


# ── Part 3: Effect of polynomial degree ──────────────────────────────────────

X_fixed = np.sort(rng.uniform(0, 1, 8))
y_fixed  = forrester(X_fixed)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
configs   = [(1, "Degree 1 — underfitting"), (3, "Degree 3 — good fit"), (7, "Degree 7 — overfitting")]

for ax, (deg, title) in zip(axes, configs):
    poly_d = fit_poly(X_fixed, y_fixed, degree=deg)
    y_hat  = poly_d(X_PLOT)

    rmse = np.sqrt(np.mean((forrester(X_PLOT) - y_hat) ** 2))

    ax.plot(X_PLOT, forrester(X_PLOT), color="steelblue", linewidth=2, label="True f(x)")
    ax.plot(X_PLOT, np.clip(y_hat, -20, 30), color="tomato", linewidth=2,
            linestyle="--", label=f"Degree {deg}  (RMSE={rmse:.2f})")
    ax.scatter(X_fixed, y_fixed, color="black", s=80, zorder=5, label="Training data")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_ylim(-10, 20)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Polynomial degree: underfitting vs. overfitting", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part3_poly_degree.png", dpi=150)
plt.close()
print("Saved: part3_poly_degree.png")


# ── Part 4: The surrogate loop ────────────────────────────────────────────────

X_obs = np.array([0.1, 0.5, 0.9])
y_obs = forrester(X_obs)
n_iterations = 8

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(n_iterations):
    degree  = min(len(X_obs) - 1, 5)
    poly_i  = fit_poly(X_obs, y_obs, degree)
    x_next  = surrogate_min(poly_i, X_PLOT)
    y_next  = forrester(x_next)

    ax = axes[i]
    ax.plot(X_PLOT, forrester(X_PLOT), color="steelblue", linewidth=1.5,
            label="True f(x)", alpha=0.6)
    ax.plot(X_PLOT, np.clip(poly_i(X_PLOT), -15, 20), color="tomato",
            linewidth=1.5, linestyle="--", label="Surrogate")
    ax.scatter(X_obs, y_obs, color="black", s=50, zorder=5, label="Observed")
    ax.scatter([x_next], [y_next], color="lime", s=120, zorder=6,
               marker="*", label=f"Next: x={x_next:.2f}")
    ax.set_title(f"Iteration {i + 1}  |  {len(X_obs)} points")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_ylim(-10, 18)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    X_obs = np.append(X_obs, x_next)
    y_obs = np.append(y_obs, y_next)

plt.suptitle("The surrogate loop: fit → find minimum → evaluate → repeat", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part4_surrogate_loop.png", dpi=150)
plt.close()
print("Saved: part4_surrogate_loop.png")
print(f"  Best found: f({X_obs[np.argmin(y_obs)]:.4f}) = {y_obs.min():.4f}")
print(f"  True minimum of Forrester ≈ f(0.757) = -6.02")


# ── Part 5: Where polynomial surrogates fail ──────────────────────────────────

X_sparse = np.array([0.1, 0.3, 0.6, 0.9])
y_sparse  = multimodal(X_sparse)

poly_fail = fit_poly(X_sparse, y_sparse, degree=3)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(X_PLOT, multimodal(X_PLOT), color="steelblue", linewidth=2, label="True f(x)")
axes[0].plot(X_PLOT, poly_fail(X_PLOT), color="tomato", linewidth=2,
             linestyle="--", label="Polynomial surrogate")
axes[0].scatter(X_sparse, y_sparse, color="black", s=80, zorder=5, label="Training points")
x_true_min = X_PLOT[np.argmin(multimodal(X_PLOT))]
x_surr_min = surrogate_min(poly_fail, X_PLOT)
axes[0].axvline(x_true_min, color="steelblue", linestyle=":", linewidth=1.5,
                label=f"True min x≈{x_true_min:.2f}")
axes[0].axvline(x_surr_min, color="tomato", linestyle=":", linewidth=1.5,
                label=f"Surrogate min x≈{x_surr_min:.2f}")
axes[0].set_title("Multimodal function — polynomial surrogate fails")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Show what a good surrogate should do — many peaks need many points
n_dense  = 20
X_dense  = np.sort(rng.uniform(0, 1, n_dense))
y_dense  = multimodal(X_dense)
poly_ok  = fit_poly(X_dense, y_dense, degree=min(n_dense - 1, 12))

axes[1].plot(X_PLOT, multimodal(X_PLOT), color="steelblue", linewidth=2, label="True f(x)")
axes[1].plot(X_PLOT, np.clip(poly_ok(X_PLOT), -3, 3), color="tomato",
             linewidth=2, linestyle="--", label=f"Polynomial ({n_dense} pts, deg 12)")
axes[1].scatter(X_dense, y_dense, color="black", s=40, zorder=5, label=f"{n_dense} training points")
axes[1].set_title(f"Even with {n_dense} points, polynomial still struggles")
axes[1].set_xlabel("x")
axes[1].set_ylabel("f(x)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.suptitle("Polynomial surrogates fail on multimodal functions → need GP or NN", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part5_failure_modes.png", dpi=150)
plt.close()
print("Saved: part5_failure_modes.png")

print("\nAll done. Check lesson-3/output/ for plots.")
print("\nKey takeaway:")
print("  Polynomial surrogates work on simple smooth functions.")
print("  They fail on multimodal functions and give no uncertainty estimate.")
print("  Lesson 6 introduces Gaussian Processes — a much better surrogate.")
print("  Lesson 10 introduces neural networks — our main contribution.")
