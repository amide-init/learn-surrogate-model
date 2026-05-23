import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Part 1: Plotting basic functions ─────────────────────────────────────────

x = np.linspace(-2 * np.pi, 2 * np.pi, 300)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x, np.sin(x), label="sin(x)", color="steelblue")
axes[0].axhline(0, color="black", linewidth=0.5)
axes[0].set_title("Sine function")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(x, x**2, label="x²", color="tomato")
axes[1].set_title("Quadratic function")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part1_basic_functions.png", dpi=150)
plt.close()
print("Saved: part1_basic_functions.png")


# ── Part 2: 2D contour plot ───────────────────────────────────────────────────

grid_x = np.linspace(-3, 3, 200)
grid_y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(grid_x, grid_y)
Z = X**2 + Y**2

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Filled contour
cf = axes[0].contourf(X, Y, Z, levels=20, cmap="viridis")
axes[0].contour(X, Y, Z, levels=20, colors="white", linewidths=0.4, alpha=0.5)
fig.colorbar(cf, ax=axes[0], label="f(x, y)")
axes[0].set_title("f(x, y) = x² + y²  —  contour")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].plot(0, 0, "r*", markersize=12, label="minimum (0, 0)")
axes[0].legend()

# 3D surface
ax3d = fig.add_subplot(1, 2, 2, projection="3d")
ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, linewidth=0)
ax3d.set_title("f(x, y) = x² + y²  —  surface")
ax3d.set_xlabel("x")
ax3d.set_ylabel("y")
ax3d.set_zlabel("f(x, y)")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part2_contour_surface.png", dpi=150)
plt.close()
print("Saved: part2_contour_surface.png")


# ── Part 3: Finding a minimum with SciPy ─────────────────────────────────────

def f(x):
    return (x[0] - 2) ** 2 + 1


x0 = [0.0]
result = minimize(f, x0, method="L-BFGS-B")

x_plot = np.linspace(-1, 5, 300)
y_plot = (x_plot - 2) ** 2 + 1

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_plot, y_plot, label="f(x) = (x − 2)² + 1", color="steelblue", linewidth=2)
ax.plot(x0, f(x0), "go", markersize=10, label=f"start x₀ = {x0[0]}")
ax.plot(
    result.x,
    result.fun,
    "r*",
    markersize=14,
    label=f"minimum at x = {result.x[0]:.4f}, f = {result.fun:.4f}",
)
ax.set_title("Minimising f(x) = (x − 2)² + 1 with SciPy")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part3_minimisation.png", dpi=150)
plt.close()
print("Saved: part3_minimisation.png")

print(f"\nSciPy result:")
print(f"  Minimum at x = {result.x[0]:.6f}")
print(f"  f(x)        = {result.fun:.6f}")
print(f"  Converged   = {result.success}")
print(f"  Message     = {result.message}")
