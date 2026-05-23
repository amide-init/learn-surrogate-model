import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def f_quadratic(x):
    return x ** 2

def f_quadratic_grad(x):
    return 2 * x

def f_noisy(x, sigma=0.5):
    return x ** 2 + np.random.normal(0, sigma)

def f_rosenbrock(xy):
    x, y = xy
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def f_ackley(xy):
    x, y = xy
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.e + 20
    )

def f_step(x):
    return np.where(x < 0, 1.0, 0.0)


# ── Part 1: Gradient descent from scratch ────────────────────────────────────

def gradient_descent(start, lr, n_steps):
    x = start
    history = [x]
    for _ in range(n_steps):
        grad = f_quadratic_grad(x)
        x = x - lr * grad
        history.append(x)
    return np.array(history)

history = gradient_descent(start=4.0, lr=0.1, n_steps=30)

x_plot = np.linspace(-5, 5, 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(x_plot, f_quadratic(x_plot), color="steelblue", linewidth=2, label="f(x) = x²")
axes[0].scatter(history, f_quadratic(history), c=range(len(history)), cmap="plasma",
                zorder=5, s=40)
axes[0].plot(history[0], f_quadratic(history[0]), "go", markersize=10, label=f"start x={history[0]:.1f}")
axes[0].plot(history[-1], f_quadratic(history[-1]), "r*", markersize=14, label=f"end x={history[-1]:.4f}")
axes[0].set_title("Gradient descent on f(x) = x²")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(len(history)), f_quadratic(history), color="tomato", linewidth=2, marker="o",
             markersize=4)
axes[1].set_title("Convergence: f(x) vs. iteration")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("f(x)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part1_gradient_descent.png", dpi=150)
plt.close()
print("Saved: part1_gradient_descent.png")


# ── Part 2: Effect of learning rate ──────────────────────────────────────────

learning_rates = [0.01, 0.1, 0.95]
labels = ["lr=0.01 (too small — slow)", "lr=0.1 (good)", "lr=0.95 (too large — diverges)"]
colors = ["steelblue", "green", "tomato"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for lr, label, color in zip(learning_rates, labels, colors):
    h = gradient_descent(start=4.0, lr=lr, n_steps=40)
    f_vals = np.clip(f_quadratic(h), -10, 200)
    axes[0].plot(range(len(h)), f_vals, label=label, color=color, linewidth=2)

axes[0].set_title("Effect of learning rate on convergence")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("f(x)")
axes[0].set_ylim(-1, 50)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for lr, label, color in zip(learning_rates, labels, colors):
    h = gradient_descent(start=4.0, lr=lr, n_steps=40)
    axes[1].plot(x_plot, f_quadratic(x_plot), color="lightgray", linewidth=1.5)
    axes[1].scatter(h[:15], np.clip(f_quadratic(h[:15]), -10, 200),
                    label=label, color=color, s=30, zorder=5)

axes[1].set_title("Steps taken (first 15 iterations)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("f(x)")
axes[1].set_ylim(-1, 30)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part2_learning_rate.png", dpi=150)
plt.close()
print("Saved: part2_learning_rate.png")


# ── Part 3: SciPy on hard functions ──────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Rosenbrock
grid = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(grid, grid)
Z_rosenbrock = (1 - X)**2 + 100 * (Y - X**2)**2

result_rb = minimize(f_rosenbrock, x0=[-1.5, 1.5], method="L-BFGS-B")

cf = axes[0].contourf(X, Y, np.log1p(Z_rosenbrock), levels=30, cmap="viridis")
fig.colorbar(cf, ax=axes[0], label="log(1 + f)")
axes[0].plot(-1.5, 1.5, "go", markersize=10, label="start (-1.5, 1.5)")
axes[0].plot(result_rb.x[0], result_rb.x[1], "r*", markersize=14,
             label=f"found ({result_rb.x[0]:.2f}, {result_rb.x[1]:.2f})")
axes[0].plot(1, 1, "w^", markersize=10, label="true minimum (1, 1)")
axes[0].set_title("Rosenbrock — f(x,y) = (1−x)² + 100(y−x²)²")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend(fontsize=8)

# Ackley
grid_a = np.linspace(-4, 4, 200)
Xa, Ya = np.meshgrid(grid_a, grid_a)
Za = (-20 * np.exp(-0.2 * np.sqrt(0.5 * (Xa**2 + Ya**2)))
      - np.exp(0.5 * (np.cos(2 * np.pi * Xa) + np.cos(2 * np.pi * Ya)))
      + np.e + 20)

result_ack = minimize(f_ackley, x0=[2.0, 2.0], method="L-BFGS-B")

cf2 = axes[1].contourf(Xa, Ya, Za, levels=30, cmap="plasma")
fig.colorbar(cf2, ax=axes[1], label="f(x, y)")
axes[1].plot(2.0, 2.0, "go", markersize=10, label="start (2, 2)")
axes[1].plot(result_ack.x[0], result_ack.x[1], "r*", markersize=14,
             label=f"found ({result_ack.x[0]:.2f}, {result_ack.x[1]:.2f})")
axes[1].plot(0, 0, "w^", markersize=10, label="true minimum (0, 0)")
axes[1].set_title("Ackley — multimodal, many local minima")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part3_hard_functions.png", dpi=150)
plt.close()
print("Saved: part3_hard_functions.png")
print(f"  Rosenbrock: found {result_rb.x}, f={result_rb.fun:.6f}, success={result_rb.success}")
print(f"  Ackley:     found {result_ack.x}, f={result_ack.fun:.6f}, success={result_ack.success}")


# ── Part 4: Gradient descent fails on noise ───────────────────────────────────

def gradient_descent_noisy(start, lr, n_steps, sigma):
    x = start
    history_x = [x]
    history_f = [f_quadratic(x)]
    for _ in range(n_steps):
        noisy_val_left  = f_noisy(x - 1e-5, sigma)
        noisy_val_right = f_noisy(x + 1e-5, sigma)
        noisy_grad = (noisy_val_right - noisy_val_left) / (2e-5)
        x = x - lr * noisy_grad
        history_x.append(x)
        history_f.append(f_quadratic(x))
    return np.array(history_x), np.array(history_f)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

noise_levels = [0.0, 0.5, 2.0]
colors_n = ["steelblue", "orange", "tomato"]
labels_n = ["σ=0.0 (clean)", "σ=0.5 (mild noise)", "σ=2.0 (heavy noise)"]

clean_h = gradient_descent(start=4.0, lr=0.1, n_steps=50)
axes[0].plot(range(len(clean_h)), f_quadratic(clean_h), color="steelblue",
             linewidth=2, label="σ=0.0 (clean)")

for sigma, color, label in zip([0.5, 2.0], ["orange", "tomato"], ["σ=0.5", "σ=2.0"]):
    hx, hf = gradient_descent_noisy(start=4.0, lr=0.1, n_steps=50, sigma=sigma)
    axes[0].plot(range(len(hf)), hf, color=color, linewidth=1.5, alpha=0.8, label=label)

axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--", label="true minimum f=0")
axes[0].set_title("Gradient descent: clean vs. noisy")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("f(x)  [true value, no noise]")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-1, 20)

# Show scatter of final x positions across multiple noisy runs
final_xs_clean = []
final_xs_noisy = []
for _ in range(30):
    h = gradient_descent(start=4.0, lr=0.1, n_steps=50)
    final_xs_clean.append(h[-1])
    hx, _ = gradient_descent_noisy(start=4.0, lr=0.1, n_steps=50, sigma=1.5)
    final_xs_noisy.append(hx[-1])

axes[1].hist(final_xs_clean, bins=15, alpha=0.7, color="steelblue", label="clean (σ=0)")
axes[1].hist(final_xs_noisy, bins=15, alpha=0.7, color="tomato", label="noisy (σ=1.5)")
axes[1].axvline(0, color="black", linewidth=1.5, linestyle="--", label="true minimum x=0")
axes[1].set_title("Where does gradient descent end up? (30 runs)")
axes[1].set_xlabel("Final x value")
axes[1].set_ylabel("Count")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part4_noisy_gradient.png", dpi=150)
plt.close()
print("Saved: part4_noisy_gradient.png")


# ── Part 5: Black-box — no gradient available ─────────────────────────────────

x_step = np.linspace(-3, 3, 600)
y_step = f_step(x_step)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(x_step, y_step, color="steelblue", linewidth=2.5, label="f(x) = step function")
axes[0].axvline(0, color="tomato", linestyle="--", linewidth=1.5, label="discontinuity at x=0")
axes[0].set_title("Black-box function — no gradient exists")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].set_ylim(-0.2, 1.4)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Show evaluation budget: 5 evaluations vs. 1000 evaluations
x_expensive = np.linspace(-3, 3, 300)
true_curve = np.sin(x_expensive) + 0.5 * x_expensive

n_evals_few = 6
n_evals_many = 300
x_few = np.linspace(-3, 3, n_evals_few)
x_many = np.linspace(-3, 3, n_evals_many)

axes[1].plot(x_expensive, true_curve, color="lightgray", linewidth=2, label="true f(x) — hidden")
axes[1].scatter(x_few, np.sin(x_few) + 0.5 * x_few, color="tomato", s=80, zorder=5,
                label=f"{n_evals_few} evaluations (cheap budget)")
axes[1].scatter(x_many, np.sin(x_many) + 0.5 * x_many, color="steelblue", s=10, alpha=0.5,
                zorder=4, label=f"{n_evals_many} evaluations (expensive!)")
axes[1].set_title("Expensive functions: every evaluation has a cost")
axes[1].set_xlabel("x")
axes[1].set_ylabel("f(x)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part5_blackbox.png", dpi=150)
plt.close()
print("Saved: part5_blackbox.png")

print("\nAll done. Check lesson-2/output/ for plots.")
print("\nKey takeaway:")
print("  Gradient descent works on smooth, clean, cheap functions.")
print("  Real problems are noisy, black-box, and expensive to evaluate.")
print("  This is exactly why we need surrogates — Lesson 3 onwards.")
