import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Kernel implementations ────────────────────────────────────────────────────

def rbf(X1, X2, length_scale=1.0, signal_var=1.0):
    """RBF / Squared Exponential kernel."""
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]          # (n, m, d)
    sq_dist = np.sum(diff ** 2, axis=-1)              # (n, m)
    return signal_var * np.exp(-0.5 * sq_dist / length_scale ** 2)

def matern52(X1, X2, length_scale=1.0, signal_var=1.0):
    """Matern 5/2 kernel — most used in practice."""
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    s = np.sqrt(5) * r / length_scale
    return signal_var * (1 + s + s ** 2 / 3) * np.exp(-s)

def matern32(X1, X2, length_scale=1.0, signal_var=1.0):
    """Matern 3/2 kernel — rougher than 5/2."""
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    s = np.sqrt(3) * r / length_scale
    return signal_var * (1 + s) * np.exp(-s)

def linear(X1, X2, signal_var=1.0, bias=0.0):
    """Linear kernel — recovers linear regression."""
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    return signal_var * (X1 @ X2.T) + bias

def sample_gp_prior(K, n_samples=5, jitter=1e-6):
    """Sample functions from a GP prior using the kernel matrix K."""
    K_jit = K + jitter * np.eye(len(K))
    L = np.linalg.cholesky(K_jit)
    z = rng.standard_normal((len(K), n_samples))
    return L @ z                                      # (n_points, n_samples)


# ── Part 1: What is a kernel? ─────────────────────────────────────────────────

x_ref   = np.array([[0.5]])
x_query = np.linspace(0, 1, 200).reshape(-1, 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ls, color, label in [(0.1, "tomato", "l=0.1 (short)"),
                          (0.3, "orange", "l=0.3"),
                          (0.5, "steelblue", "l=0.5 (medium)"),
                          (1.0, "green", "l=1.0 (long)")]:
    k_vals = rbf(x_query, x_ref, length_scale=ls).squeeze()
    axes[0].plot(x_query.squeeze(), k_vals, color=color, linewidth=2, label=label)

axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1, label="reference x=0.5")
axes[0].set_title("RBF kernel k(x, 0.5) — similarity to x=0.5")
axes[0].set_xlabel("x")
axes[0].set_ylabel("k(x, 0.5)")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Show specific values
x_pairs = np.array([[0.5, 0.5], [0.5, 0.6], [0.5, 0.8], [0.5, 1.0]])
ls_demo = 0.3
print("RBF kernel values (l=0.3):")
for x1, x2 in x_pairs:
    k = rbf(np.array([[x1]]), np.array([[x2]]), length_scale=ls_demo).item()
    print(f"  k({x1}, {x2}) = {k:.4f}   distance = {abs(x1-x2):.1f}")

# Bar chart of similarity values
distances = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
for ls, color in [(0.2, "tomato"), (0.5, "steelblue")]:
    k_vals = [rbf(np.array([[0.0]]), np.array([[d]]), length_scale=ls).item()
              for d in distances]
    axes[1].plot(distances, k_vals, "o-", color=color, linewidth=2,
                 markersize=7, label=f"l={ls}")

axes[1].set_title("Similarity drops with distance — rate controlled by length-scale")
axes[1].set_xlabel("Distance |x − x'|")
axes[1].set_ylabel("k(x, x')")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part1_kernel_similarity.png", dpi=150)
plt.close()
print("\nSaved: part1_kernel_similarity.png")


# ── Part 2: Kernel matrix ─────────────────────────────────────────────────────

x_pts = np.linspace(0, 1, 10).reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, ls, title in zip(axes,
                          [0.1, 0.4, 1.0],
                          ["l=0.1 (short — wiggly)", "l=0.4 (medium)", "l=1.0 (long — smooth)"]):
    K = rbf(x_pts, x_pts, length_scale=ls)
    im = ax.imshow(K, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"RBF kernel matrix\n{title}")
    ax.set_xlabel("Point index j")
    ax.set_ylabel("Point index i")
    ticks = range(len(x_pts))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{v:.1f}" for v in x_pts.squeeze()], rotation=45, fontsize=7)
    ax.set_yticklabels([f"{v:.1f}" for v in x_pts.squeeze()], fontsize=7)

plt.suptitle("Kernel matrix K[i,j] = k(xᵢ, xⱼ) — diagonal is always 1.0", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part2_kernel_matrix.png", dpi=150)
plt.close()
print("Saved: part2_kernel_matrix.png")


# ── Part 3: Effect of length-scale on sampled functions ──────────────────────

x_fine = np.linspace(0, 1, 200).reshape(-1, 1)
n_samples = 4

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, ls in zip(axes, [0.05, 0.2, 0.5, 1.5]):
    K = rbf(x_fine, x_fine, length_scale=ls)
    samples = sample_gp_prior(K, n_samples=n_samples)
    for i in range(n_samples):
        ax.plot(x_fine.squeeze(), samples[:, i], linewidth=1.5, alpha=0.8)
    ax.set_title(f"l = {ls}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.3)

plt.suptitle("GP prior samples — RBF kernel with different length-scales\n"
             "Short l = wiggly functions,  Long l = smooth functions", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part3_length_scale.png", dpi=150)
plt.close()
print("Saved: part3_length_scale.png")


# ── Part 4: Kernel comparison — matrix + samples ──────────────────────────────

kernels = [
    ("RBF",        lambda X1, X2: rbf(X1, X2, length_scale=0.3)),
    ("Matern-5/2", lambda X1, X2: matern52(X1, X2, length_scale=0.3)),
    ("Matern-3/2", lambda X1, X2: matern32(X1, X2, length_scale=0.3)),
    ("Linear",     lambda X1, X2: linear(X1, X2, signal_var=1.0, bias=0.1)),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for col, (name, kern_fn) in enumerate(kernels):
    K_mat = kern_fn(x_pts, x_pts)
    im = axes[0, col].imshow(K_mat, cmap="viridis")
    plt.colorbar(im, ax=axes[0, col])
    axes[0, col].set_title(f"{name}\nKernel matrix")
    axes[0, col].set_xlabel("j")
    axes[0, col].set_ylabel("i")

    K_fine = kern_fn(x_fine, x_fine)
    try:
        samples = sample_gp_prior(K_fine, n_samples=4)
        for i in range(4):
            axes[1, col].plot(x_fine.squeeze(), samples[:, i], linewidth=1.5, alpha=0.8)
    except np.linalg.LinAlgError:
        axes[1, col].text(0.5, 0.5, "Not PSD\n(skip)", ha="center", transform=axes[1, col].transAxes)

    axes[1, col].set_title(f"{name}\nPrior samples")
    axes[1, col].set_xlabel("x")
    axes[1, col].set_ylabel("f(x)")
    axes[1, col].set_ylim(-3, 3)
    axes[1, col].grid(True, alpha=0.3)

plt.suptitle("Kernel comparison — matrix structure (top) and sample functions (bottom)",
             fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part4_kernel_comparison.png", dpi=150)
plt.close()
print("Saved: part4_kernel_comparison.png")


# ── Part 5: Smoothness comparison on the same data ───────────────────────────

x_obs  = np.array([[0.1], [0.35], [0.6], [0.85]])
y_obs  = np.sin(2 * np.pi * x_obs.squeeze())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
kern_compare = [
    ("RBF (infinitely smooth)",  lambda X1, X2: rbf(X1, X2, length_scale=0.3)),
    ("Matern-5/2 (twice diff.)", lambda X1, X2: matern52(X1, X2, length_scale=0.3)),
    ("Matern-3/2 (once diff.)",  lambda X1, X2: matern32(X1, X2, length_scale=0.3)),
]

for ax, (name, kern_fn) in zip(axes, kern_compare):
    K   = kern_fn(x_obs, x_obs) + 1e-6 * np.eye(len(x_obs))
    Ks  = kern_fn(x_fine, x_obs)
    Kss = kern_fn(x_fine, x_fine)
    L   = np.linalg.cholesky(K)
    mu  = Ks @ np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
    v   = np.linalg.solve(L, Ks.T)
    std = np.sqrt(np.maximum(np.diag(Kss) - np.sum(v ** 2, axis=0), 0))

    ax.plot(x_fine.squeeze(), np.sin(2 * np.pi * x_fine.squeeze()),
            color="gray", linewidth=1.5, linestyle="--", label="True f(x)", alpha=0.5)
    ax.plot(x_fine.squeeze(), mu, color="steelblue", linewidth=2, label="GP mean")
    ax.fill_between(x_fine.squeeze(), mu - 2 * std, mu + 2 * std,
                    alpha=0.2, color="steelblue", label="±2σ")
    ax.scatter(x_obs.squeeze(), y_obs, color="black", s=80, zorder=5, label="Data")
    ax.set_title(name)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_ylim(-2.5, 2.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("Same data, different kernel → different surrogate behaviour", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part5_smoothness.png", dpi=150)
plt.close()
print("Saved: part5_smoothness.png")


# ── Part 6: 2D kernel matrix ──────────────────────────────────────────────────

x1_grid = np.linspace(0, 1, 8)
x2_grid = np.linspace(0, 1, 8)
X1g, X2g = np.meshgrid(x1_grid, x2_grid)
X2d = np.column_stack([X1g.ravel(), X2g.ravel()])  # (64, 2)

K_2d = rbf(X2d, X2d, length_scale=0.3)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

im = axes[0].imshow(K_2d, cmap="viridis")
plt.colorbar(im, ax=axes[0], label="k(xᵢ, xⱼ)")
axes[0].set_title("2D RBF kernel matrix (8×8 grid = 64 points)")
axes[0].set_xlabel("Point index j")
axes[0].set_ylabel("Point index i")

# Sample one 2D function from the prior
samples_2d = sample_gp_prior(K_2d, n_samples=1)
Z_sample   = samples_2d[:, 0].reshape(8, 8)

cf = axes[1].contourf(X1g, X2g, Z_sample, levels=15, cmap="viridis")
plt.colorbar(cf, ax=axes[1], label="f(x₁, x₂)")
axes[1].set_title("One sample function from 2D GP prior (RBF, l=0.3)")
axes[1].set_xlabel("x₁")
axes[1].set_ylabel("x₂")
axes[1].grid(True, alpha=0.3)

plt.suptitle("2D kernels — needed for Branin, Hartmann, and BBOB functions", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part6_2d_kernel.png", dpi=150)
plt.close()
print("Saved: part6_2d_kernel.png")

print("\nAll done. Check lesson-5/output/ for plots.")
print("\nKey takeaways:")
print("  k(x, x') measures how similar two inputs are.")
print("  Short length-scale → wiggly surrogate. Long → smooth.")
print("  RBF is the default; Matern-5/2 is preferred in practice.")
print("  Neural networks learn their own similarity — no kernel needed.")
print("  This is why NNs generalise better in high dimensions (Lesson 10-11).")
