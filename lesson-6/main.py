import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Benchmark functions ───────────────────────────────────────────────────────

def forrester(x):
    """Standard 1D benchmark: (6x-2)² sin(12x-4), domain [0,1]."""
    x = np.asarray(x).ravel()
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

def branin_norm(X):
    """Branin on [0,1]²."""
    X = np.atleast_2d(X)
    x1 = X[:, 0] * 15 - 5
    x2 = X[:, 1] * 15
    a, b, c = 1.0, 5.1 / (4 * np.pi ** 2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


# ── Kernels ───────────────────────────────────────────────────────────────────

def rbf(X1, X2, length_scale=1.0, signal_var=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]
    sq_dist = np.sum(diff ** 2, axis=-1)
    return signal_var * np.exp(-0.5 * sq_dist / length_scale ** 2)

def matern52(X1, X2, length_scale=1.0, signal_var=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    s = np.sqrt(5) * r / length_scale
    return signal_var * (1 + s + s ** 2 / 3) * np.exp(-s)

def matern32(X1, X2, length_scale=1.0, signal_var=1.0):
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    s = np.sqrt(3) * r / length_scale
    return signal_var * (1 + s) * np.exp(-s)


# ── GP core ───────────────────────────────────────────────────────────────────

def gp_predict(X_train, y_train, X_test, kernel_fn, noise=1e-4):
    """
    GP posterior mean and standard deviation.
    Uses Cholesky for numerical stability.
    """
    K    = kernel_fn(X_train, X_train) + noise * np.eye(len(X_train))
    Ks   = kernel_fn(X_test, X_train)
    Kss  = kernel_fn(X_test, X_test)
    L    = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu   = Ks @ alpha
    v    = np.linalg.solve(L, Ks.T)
    var  = np.diag(Kss) - np.sum(v ** 2, axis=0)
    std  = np.sqrt(np.maximum(var, 0))
    return mu, std

def log_marginal_likelihood(X_train, y_train, kernel_fn, noise=1e-4):
    K = kernel_fn(X_train, X_train) + noise * np.eye(len(X_train))
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return -np.inf
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    lml  = -0.5 * y_train @ alpha
    lml -= np.sum(np.log(np.diag(L)))
    lml -= 0.5 * len(X_train) * np.log(2 * np.pi)
    return float(lml)

def sample_gp_prior(K, n_samples=5, jitter=1e-6):
    K_jit = K + jitter * np.eye(len(K))
    L = np.linalg.cholesky(K_jit)
    z = rng.standard_normal((len(K), n_samples))
    return L @ z


# ── Part 1: Prior → Posterior ─────────────────────────────────────────────────

x_fine = np.linspace(0, 1, 200).reshape(-1, 1)
kern   = lambda X1, X2: rbf(X1, X2, length_scale=0.3, signal_var=1.0)

K_prior = kern(x_fine, x_fine)
prior_samples = sample_gp_prior(K_prior, n_samples=5)

x_obs = np.array([[0.1], [0.4], [0.75]])
y_obs = np.sin(2 * np.pi * x_obs.squeeze())
mu_post, std_post = gp_predict(x_obs, y_obs, x_fine, kern, noise=1e-6)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for i in range(5):
    axes[0].plot(x_fine.squeeze(), prior_samples[:, i], linewidth=1.5, alpha=0.7)
axes[0].set_title("GP prior — 5 functions sampled before seeing data")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].set_ylim(-3, 3)
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_fine.squeeze(), np.sin(2 * np.pi * x_fine.squeeze()),
             color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True f(x)")
axes[1].plot(x_fine.squeeze(), mu_post, color="steelblue", linewidth=2, label="GP mean")
axes[1].fill_between(x_fine.squeeze(),
                     mu_post - 2 * std_post, mu_post + 2 * std_post,
                     alpha=0.2, color="steelblue", label="±2σ")
axes[1].scatter(x_obs.squeeze(), y_obs, color="black", s=80, zorder=5, label="Data")
axes[1].set_title("GP posterior — conditioned on 3 observations")
axes[1].set_xlabel("x")
axes[1].set_ylabel("f(x)")
axes[1].set_ylim(-3, 3)
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.suptitle("Bayesian update: prior → posterior\nUncertainty collapses near data, grows elsewhere",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part1_prior_to_posterior.png", dpi=150)
plt.close()
print("Saved: part1_prior_to_posterior.png")


# ── Part 2: GP on Forrester — effect of training set size ────────────────────

x_all  = rng.uniform(0, 1, 20).reshape(-1, 1)
y_all  = forrester(x_all)
x_true = np.linspace(0, 1, 300).reshape(-1, 1)
y_true = forrester(x_true)

kern_f = lambda X1, X2: matern52(X1, X2, length_scale=0.3, signal_var=4.0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, n in zip(axes, [5, 10, 20]):
    X_tr = x_all[:n]
    y_tr = y_all[:n]
    mu, std = gp_predict(X_tr, y_tr, x_true, kern_f, noise=0.01)

    ax.plot(x_true.squeeze(), y_true, color="gray", linestyle="--",
            linewidth=1.5, alpha=0.6, label="True Forrester")
    ax.plot(x_true.squeeze(), mu, color="steelblue", linewidth=2, label="GP mean")
    ax.fill_between(x_true.squeeze(), mu - 2 * std, mu + 2 * std,
                    alpha=0.2, color="steelblue", label="±2σ")
    ax.scatter(X_tr.squeeze(), y_tr, color="black", s=50, zorder=5, label=f"Data (n={n})")
    ax.set_title(f"n = {n} training points")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("GP on the Forrester function — more data → less uncertainty", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part2_forrester_gp.png", dpi=150)
plt.close()
print("Saved: part2_forrester_gp.png")


# ── Part 3: Effect of noise hyperparameter ────────────────────────────────────

X_tr5 = x_all[:8]
y_tr5 = y_all[:8]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, noise in zip(axes, [1e-6, 0.01, 0.1, 0.5]):
    mu, std = gp_predict(X_tr5, y_tr5, x_true, kern_f, noise=noise)
    ax.plot(x_true.squeeze(), y_true, color="gray", linestyle="--",
            linewidth=1.5, alpha=0.5, label="True")
    ax.plot(x_true.squeeze(), mu, color="steelblue", linewidth=2, label="GP mean")
    ax.fill_between(x_true.squeeze(), mu - 2 * std, mu + 2 * std,
                    alpha=0.2, color="steelblue", label="±2σ")
    ax.scatter(X_tr5.squeeze(), y_tr5, color="black", s=50, zorder=5)
    ax.set_title(f"noise = {noise}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("Effect of noise hyperparameter\nLow noise → interpolation.  High noise → smoothing.",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part3_noise_effect.png", dpi=150)
plt.close()
print("Saved: part3_noise_effect.png")


# ── Part 4: Kernel comparison on Forrester ────────────────────────────────────

X_tr10 = x_all[:10]
y_tr10 = y_all[:10]

kernels = [
    ("RBF (∞-smooth)",  lambda X1, X2: rbf(X1, X2, length_scale=0.3, signal_var=4.0)),
    ("Matern-5/2",      lambda X1, X2: matern52(X1, X2, length_scale=0.3, signal_var=4.0)),
    ("Matern-3/2",      lambda X1, X2: matern32(X1, X2, length_scale=0.3, signal_var=4.0)),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, kern_fn) in zip(axes, kernels):
    mu, std = gp_predict(X_tr10, y_tr10, x_true, kern_fn, noise=0.01)
    ax.plot(x_true.squeeze(), y_true, color="gray", linestyle="--",
            linewidth=1.5, alpha=0.5, label="True")
    ax.plot(x_true.squeeze(), mu, color="steelblue", linewidth=2, label="GP mean")
    ax.fill_between(x_true.squeeze(), mu - 2 * std, mu + 2 * std,
                    alpha=0.2, color="steelblue", label="±2σ")
    ax.scatter(X_tr10.squeeze(), y_tr10, color="black", s=50, zorder=5, label="Data")
    lml = log_marginal_likelihood(X_tr10, y_tr10, kern_fn, noise=0.01)
    ax.set_title(f"{name}\nlog p(y|X) = {lml:.1f}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("Kernel comparison — same data, different smoothness assumption\n"
             "Higher log p(y|X) = better fit to data", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part4_kernel_comparison.png", dpi=150)
plt.close()
print("Saved: part4_kernel_comparison.png")


# ── Part 5: Log marginal likelihood sweep ─────────────────────────────────────

X_tr8 = x_all[:8]
y_tr8 = y_all[:8]

ls_values = np.logspace(-1.5, 0.5, 60)   # 0.03 … 3.0
lml_rbf    = [log_marginal_likelihood(X_tr8, y_tr8,
               lambda X1, X2, ls=ls: rbf(X1, X2, length_scale=ls, signal_var=4.0))
              for ls in ls_values]
lml_m52    = [log_marginal_likelihood(X_tr8, y_tr8,
               lambda X1, X2, ls=ls: matern52(X1, X2, length_scale=ls, signal_var=4.0))
              for ls in ls_values]

best_ls_rbf = ls_values[np.argmax(lml_rbf)]
best_ls_m52 = ls_values[np.argmax(lml_m52)]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ls_values, lml_rbf, color="tomato",    linewidth=2, label="RBF")
ax.plot(ls_values, lml_m52, color="steelblue", linewidth=2, label="Matern-5/2")
ax.axvline(best_ls_rbf, color="tomato",    linestyle="--", linewidth=1,
           label=f"RBF best l = {best_ls_rbf:.3f}")
ax.axvline(best_ls_m52, color="steelblue", linestyle="--", linewidth=1,
           label=f"Matern-5/2 best l = {best_ls_m52:.3f}")
ax.set_xscale("log")
ax.set_title("Log marginal likelihood vs length-scale\nPeak = hyperparameter preferred by the data")
ax.set_xlabel("length-scale (log scale)")
ax.set_ylabel("log p(y | X, l)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part5_log_marginal_likelihood.png", dpi=150)
plt.close()
print(f"Saved: part5_log_marginal_likelihood.png")
print(f"  Best RBF length-scale: {best_ls_rbf:.3f}")
print(f"  Best Matern-5/2 length-scale: {best_ls_m52:.3f}")


# ── Part 6: 2D GP on Branin ──────────────────────────────────────────────────

n_train_2d = 20
X_2d = rng.uniform(0, 1, (n_train_2d, 2))
y_2d = branin_norm(X_2d)

# Prediction grid
grid_res = 20
g = np.linspace(0, 1, grid_res)
G1, G2 = np.meshgrid(g, g)
X_grid = np.column_stack([G1.ravel(), G2.ravel()])   # (400, 2)
y_true_grid = branin_norm(X_grid).reshape(grid_res, grid_res)

kern_2d = lambda X1, X2: matern52(X1, X2, length_scale=0.3, signal_var=50.0)
mu_grid, std_grid = gp_predict(X_2d, y_2d, X_grid, kern_2d, noise=0.1)
mu_grid  = mu_grid.reshape(grid_res, grid_res)
std_grid = std_grid.reshape(grid_res, grid_res)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

cf0 = axes[0].contourf(G1, G2, y_true_grid, levels=20, cmap="viridis")
plt.colorbar(cf0, ax=axes[0])
axes[0].set_title("True Branin surface")
axes[0].set_xlabel("x₁ (normalised)")
axes[0].set_ylabel("x₂ (normalised)")

cf1 = axes[1].contourf(G1, G2, mu_grid, levels=20, cmap="viridis")
plt.colorbar(cf1, ax=axes[1])
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], color="white", s=30,
                edgecolors="black", linewidths=0.7, zorder=5, label=f"Training (n={n_train_2d})")
axes[1].set_title("GP mean prediction")
axes[1].set_xlabel("x₁")
axes[1].set_ylabel("x₂")
axes[1].legend(fontsize=8)

cf2 = axes[2].contourf(G1, G2, std_grid, levels=20, cmap="Reds")
plt.colorbar(cf2, ax=axes[2], label="σ (uncertainty)")
axes[2].scatter(X_2d[:, 0], X_2d[:, 1], color="white", s=30,
                edgecolors="black", linewidths=0.7, zorder=5)
axes[2].set_title("GP uncertainty (σ)")
axes[2].set_xlabel("x₁")
axes[2].set_ylabel("x₂")

plt.suptitle(f"2D GP on Branin — {n_train_2d} training points\n"
             "High uncertainty = regions the surrogate has not seen yet", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part6_2d_branin_gp.png", dpi=150)
plt.close()
print("Saved: part6_2d_branin_gp.png")

print("\nAll done. Check lesson-6/output/ for plots.")
print("\nKey takeaways:")
print("  GP posterior = conditional Gaussian from Lesson 4 + kernel from Lesson 5.")
print("  Cholesky gives numerical stability for K⁻¹ y.")
print("  Log marginal likelihood selects kernel hyperparameters from data alone.")
print("  GP uncertainty is largest far from training points — perfect for guiding search.")
print("  Lesson 7: we turn this uncertainty into an acquisition function.")
