import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm, multivariate_normal
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Part 1: The 1D Gaussian ───────────────────────────────────────────────────

x = np.linspace(-6, 6, 400)

configs = [
    (0.0, 0.5, "steelblue",  "μ=0, σ=0.5"),
    (0.0, 1.0, "tomato",     "μ=0, σ=1.0  (standard normal)"),
    (0.0, 2.0, "green",      "μ=0, σ=2.0"),
    (2.0, 1.0, "purple",     "μ=2, σ=1.0"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for mu, sigma, color, label in configs:
    axes[0].plot(x, norm.pdf(x, mu, sigma), color=color, linewidth=2, label=label)

axes[0].set_title("Gaussian PDF with different μ and σ")
axes[0].set_xlabel("x")
axes[0].set_ylabel("Probability density")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# 68-95-99.7 rule on the standard normal
mu, sigma = 0.0, 1.0
pdf = norm.pdf(x, mu, sigma)
axes[1].plot(x, pdf, color="steelblue", linewidth=2)
for n_sigma, alpha, label in [(1, 0.5, "±1σ: 68.3%"), (2, 0.3, "±2σ: 95.4%"), (3, 0.15, "±3σ: 99.7%")]:
    mask = np.abs(x - mu) <= n_sigma * sigma
    axes[1].fill_between(x, pdf, where=mask, alpha=alpha, color="steelblue", label=label)
axes[1].set_title("68-95-99.7 Rule (standard normal)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("Probability density")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part1_gaussian_1d.png", dpi=150)
plt.close()
print("Saved: part1_gaussian_1d.png")


# ── Part 2: Sampling and convergence ─────────────────────────────────────────

true_mu, true_sigma = 1.5, 0.8
sample_sizes = [5, 20, 100, 500]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for n, color in zip(sample_sizes, ["tomato", "orange", "steelblue", "green"]):
    samples = rng.normal(true_mu, true_sigma, n)
    axes[0].plot(range(1, n + 1), np.cumsum(samples) / np.arange(1, n + 1),
                 color=color, linewidth=1.5, label=f"n={n}")

axes[0].axhline(true_mu, color="black", linestyle="--", linewidth=1.5, label=f"true μ={true_mu}")
axes[0].set_title("Sample mean converges to true μ (Law of Large Numbers)")
axes[0].set_xlabel("Number of samples")
axes[0].set_ylabel("Running mean")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Histogram vs. true PDF for different n
n_large = 1000
samples_large = rng.normal(true_mu, true_sigma, n_large)
x_range = np.linspace(true_mu - 4 * true_sigma, true_mu + 4 * true_sigma, 300)

axes[1].hist(samples_large, bins=40, density=True, color="steelblue",
             alpha=0.6, label=f"n={n_large} samples")
axes[1].plot(x_range, norm.pdf(x_range, true_mu, true_sigma),
             color="tomato", linewidth=2.5, label=f"True PDF (μ={true_mu}, σ={true_sigma})")
axes[1].set_title("Histogram of samples vs. true Gaussian PDF")
axes[1].set_xlabel("x")
axes[1].set_ylabel("Density")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part2_sampling.png", dpi=150)
plt.close()
print("Saved: part2_sampling.png")


# ── Part 3: Covariance and correlation ───────────────────────────────────────

cov_configs = [
    (np.array([[1.0, 0.8], [0.8, 1.0]]),  "Positive correlation (ρ=0.8)",  "steelblue"),
    (np.array([[1.0, 0.0], [0.0, 1.0]]),  "No correlation (ρ=0.0)",        "green"),
    (np.array([[1.0, -0.8], [-0.8, 1.0]]), "Negative correlation (ρ=-0.8)", "tomato"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (cov, title, color) in zip(axes, cov_configs):
    samples = rng.multivariate_normal([0, 0], cov, size=300)
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.4, s=20, color=color)

    # Draw confidence ellipse
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    for n_std, alpha in [(1, 0.4), (2, 0.2)]:
        w, h = 2 * n_std * np.sqrt(vals)
        ell = Ellipse((0, 0), width=w, height=h, angle=angle,
                      edgecolor=color, facecolor="none", linewidth=2, alpha=alpha + 0.4)
        ax.add_patch(ell)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title(title)
    ax.set_xlabel("X₁")
    ax.set_ylabel("X₂")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

plt.suptitle("Covariance shapes — 300 samples + 1σ / 2σ ellipses", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part3_correlation.png", dpi=150)
plt.close()
print("Saved: part3_correlation.png")


# ── Part 4: 2D Multivariate Gaussian ─────────────────────────────────────────

mu_2d  = np.array([1.0, -0.5])
cov_2d = np.array([[1.2, 0.7], [0.7, 0.8]])

grid_x = np.linspace(-3, 5, 200)
grid_y = np.linspace(-4, 3, 200)
X2, Y2 = np.meshgrid(grid_x, grid_y)
pos    = np.dstack((X2, Y2))
Z2     = multivariate_normal.pdf(pos, mean=mu_2d, cov=cov_2d)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cf = axes[0].contourf(X2, Y2, Z2, levels=15, cmap="viridis")
axes[0].contour(X2, Y2, Z2, levels=15, colors="white", linewidths=0.4, alpha=0.5)
fig.colorbar(cf, ax=axes[0], label="Probability density")
axes[0].plot(*mu_2d, "r*", markersize=14, label=f"mean ({mu_2d[0]}, {mu_2d[1]})")
axes[0].set_title("2D Gaussian — contour plot")
axes[0].set_xlabel("X₁")
axes[0].set_ylabel("X₂")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

samples_2d = rng.multivariate_normal(mu_2d, cov_2d, size=400)
axes[1].scatter(samples_2d[:, 0], samples_2d[:, 1], alpha=0.4, s=20, color="steelblue",
                label="400 samples")
axes[1].plot(*mu_2d, "r*", markersize=14, label=f"true mean ({mu_2d[0]}, {mu_2d[1]})")
axes[1].plot(*samples_2d.mean(axis=0), "g^", markersize=10,
             label=f"sample mean ({samples_2d.mean(axis=0)[0]:.2f}, {samples_2d.mean(axis=0)[1]:.2f})")
axes[1].set_title("2D Gaussian — samples")
axes[1].set_xlabel("X₁")
axes[1].set_ylabel("X₂")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

print(f"\nTrue covariance matrix:\n{cov_2d}")
print(f"Sample covariance matrix:\n{np.cov(samples_2d.T).round(3)}")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part4_multivariate_gaussian.png", dpi=150)
plt.close()
print("Saved: part4_multivariate_gaussian.png")


# ── Part 5: Conditional distributions ────────────────────────────────────────

# Joint: [X, Y] ~ N([0,0], [[1, rho],[rho, 1]])
# Conditional: Y | X=x ~ N(rho*x, 1 - rho^2)
rho   = 0.85
mu_xy = np.array([0.0, 0.0])
cov_xy = np.array([[1.0, rho], [rho, 1.0]])

x_vals   = [-1.5, 0.0, 1.5]
colors_c = ["tomato", "steelblue", "green"]
y_range  = np.linspace(-4, 4, 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Joint scatter + conditioning lines
joint_samples = rng.multivariate_normal(mu_xy, cov_xy, size=500)
axes[0].scatter(joint_samples[:, 0], joint_samples[:, 1],
                alpha=0.3, s=15, color="gray", label="joint samples")

for xv, color in zip(x_vals, colors_c):
    axes[0].axvline(xv, color=color, linewidth=2, linestyle="--",
                    label=f"X = {xv}")
    cond_mu  = rho * xv
    axes[0].plot(xv, cond_mu, "o", color=color, markersize=10)

axes[0].set_title("Joint distribution — conditioning on X = x")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Conditional PDFs
for xv, color in zip(x_vals, colors_c):
    cond_mu    = rho * xv
    cond_sigma = np.sqrt(1 - rho ** 2)
    pdf_cond   = norm.pdf(y_range, cond_mu, cond_sigma)
    axes[1].plot(y_range, pdf_cond, color=color, linewidth=2,
                 label=f"P(Y|X={xv}):  μ={cond_mu:.2f}, σ={cond_sigma:.2f}")
    axes[1].axvline(cond_mu, color=color, linestyle=":", linewidth=1.2)

axes[1].set_title("Conditional distributions P(Y | X = x)")
axes[1].set_xlabel("y")
axes[1].set_ylabel("Density")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.suptitle(f"Conditional Gaussian  (ρ = {rho})", fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part5_conditional.png", dpi=150)
plt.close()
print("Saved: part5_conditional.png")


# ── Part 6: Connection to GP predictions ─────────────────────────────────────

# A GP predicts: for a new point x*, the output f(x*) is Gaussian
# Mean = best guess, Variance = how uncertain we are
# This is EXACTLY a conditional Gaussian

np.random.seed(RANDOM_SEED)
x_obs   = np.array([0.1, 0.4, 0.7, 0.9])
y_obs   = np.sin(2 * np.pi * x_obs)
x_star  = np.linspace(0, 1, 200)

# Minimal RBF GP for illustration only
def rbf(a, b, length=0.2, var=1.0):
    return var * np.exp(-0.5 * ((a[:, None] - b[None, :]) / length) ** 2)

K_obs  = rbf(x_obs, x_obs) + 1e-6 * np.eye(len(x_obs))
K_star = rbf(x_star, x_obs)
K_ss   = rbf(x_star, x_star)
L      = np.linalg.cholesky(K_obs)
alpha  = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
mu_gp  = K_star @ alpha
v      = np.linalg.solve(L, K_star.T)
var_gp = np.diag(K_ss) - np.sum(v ** 2, axis=0)
std_gp = np.sqrt(np.maximum(var_gp, 0))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(x_star, np.sin(2 * np.pi * x_star), color="gray",
             linewidth=1.5, linestyle="--", label="True f(x)", alpha=0.6)
axes[0].plot(x_star, mu_gp, color="steelblue", linewidth=2, label="GP mean")
axes[0].fill_between(x_star, mu_gp - 2 * std_gp, mu_gp + 2 * std_gp,
                     alpha=0.25, color="steelblue", label="±2σ (95% credible)")
axes[0].scatter(x_obs, y_obs, color="black", s=80, zorder=5, label="Observed")
axes[0].set_title("GP prediction = conditional Gaussian at each x*")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# Plot the predictive Gaussian at two specific points
x_query = [0.3, 0.6]
colors_q = ["tomato", "green"]
y_range2 = np.linspace(-2.5, 2.5, 300)

for xq, color in zip(x_query, colors_q):
    idx    = np.argmin(np.abs(x_star - xq))
    mu_q   = mu_gp[idx]
    std_q  = std_gp[idx]
    pdf_q  = norm.pdf(y_range2, mu_q, std_q)
    axes[1].plot(y_range2, pdf_q, color=color, linewidth=2,
                 label=f"x*={xq}: μ={mu_q:.2f}, σ={std_q:.2f}")
    axes[1].axvline(mu_q, color=color, linestyle=":", linewidth=1.2)

axes[1].set_title("GP output is a Gaussian: P(f(x*) | data)")
axes[1].set_xlabel("f(x*)")
axes[1].set_ylabel("Density")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.suptitle("GP predictions are conditional Gaussians — connecting Lesson 4 to Lesson 6",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part6_gp_connection.png", dpi=150)
plt.close()
print("Saved: part6_gp_connection.png")

print("\nAll done. Check lesson-4/output/ for plots.")
print("\nKey takeaway:")
print("  Every surrogate in this project outputs a Gaussian: (mean, variance).")
print("  Mean = best prediction.  Variance = how uncertain we are.")
print("  Lesson 5 shows how GPs build covariance from kernel functions.")
print("  Lessons 10-11 show how NNs learn to output the same thing.")
