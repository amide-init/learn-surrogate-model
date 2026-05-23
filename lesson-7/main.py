import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Benchmark, kernel, GP (reused from Lesson 6) ─────────────────────────────

def forrester(x):
    x = np.asarray(x).ravel()
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

def matern52(X1, X2, length_scale=1.0, signal_var=1.0):
    X1 = np.atleast_2d(X1); X2 = np.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    s = np.sqrt(5) * r / length_scale
    return signal_var * (1 + s + s ** 2 / 3) * np.exp(-s)

def gp_predict(X_train, y_train, X_test, kernel_fn, noise=1e-4):
    K    = kernel_fn(X_train, X_train) + noise * np.eye(len(X_train))
    Ks   = kernel_fn(X_test, X_train)
    Kss  = kernel_fn(X_test, X_test)
    L    = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu   = Ks @ alpha
    v    = np.linalg.solve(L, Ks.T)
    std  = np.sqrt(np.maximum(np.diag(Kss) - np.sum(v ** 2, axis=0), 0))
    return mu, std


# ── Acquisition functions ─────────────────────────────────────────────────────

def acq_pi(mu, std, f_best, xi=0.0):
    """Probability of Improvement (minimization)."""
    Z = (f_best - xi - mu) / (std + 1e-9)
    return norm.cdf(Z)

def acq_ei(mu, std, f_best, xi=0.01):
    """Expected Improvement (minimization)."""
    Z = (f_best - xi - mu) / (std + 1e-9)
    return np.maximum((f_best - xi - mu) * norm.cdf(Z) + std * norm.pdf(Z), 0)

def acq_lcb(mu, std, kappa=2.0):
    """Negative Lower Confidence Bound — argmax gives the LCB minimizer."""
    return -(mu - kappa * std)


# ── Shared data and GP fit ────────────────────────────────────────────────────

kern = lambda X1, X2: matern52(X1, X2, length_scale=0.25, signal_var=4.0)

x_fine = np.linspace(0, 1, 300).reshape(-1, 1)
xs     = x_fine.squeeze()
y_true = forrester(xs)

X_train = np.array([[0.1], [0.3], [0.5], [0.7], [0.9]])
y_train = forrester(X_train)
f_best  = y_train.min()

mu, std = gp_predict(X_train, y_train, x_fine, kern, noise=0.01)


# ── Helper: 2-row (GP + acquisition) panel ───────────────────────────────────

def draw_panel(ax_gp, ax_acq, mu, std, acq_vals, title, acq_label):
    x_next = x_fine[np.argmax(acq_vals)].item()

    ax_gp.plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.4, label="True")
    ax_gp.plot(xs, mu, color="steelblue", linewidth=2, label="GP μ")
    ax_gp.fill_between(xs, mu - 2*std, mu + 2*std, alpha=0.2, color="steelblue", label="±2σ")
    ax_gp.scatter(X_train.squeeze(), y_train, color="black", s=50, zorder=5)
    ax_gp.axhline(f_best, color="gray", linewidth=1, linestyle=":", alpha=0.6)
    ax_gp.axvline(x_next, color="red", linewidth=2, linestyle=":",
                  label=f"x_next={x_next:.3f}")
    ax_gp.set_title(title)
    ax_gp.set_ylabel("f(x)")
    ax_gp.legend(fontsize=6)
    ax_gp.grid(True, alpha=0.3)

    ax_acq.plot(xs, acq_vals, color="darkorange", linewidth=2)
    ax_acq.fill_between(xs, 0, acq_vals, alpha=0.25, color="darkorange")
    ax_acq.axvline(x_next, color="red", linewidth=2, linestyle=":")
    ax_acq.set_ylabel(acq_label)
    ax_acq.set_xlabel("x")
    ax_acq.grid(True, alpha=0.3)


# ── Part 1: Why acquisition functions? ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True Forrester")
ax.plot(xs, mu, color="steelblue", linewidth=2, label="GP mean μ(x)")
ax.fill_between(xs, mu - 2*std, mu + 2*std, alpha=0.2, color="steelblue", label="±2σ(x)")
ax.scatter(X_train.squeeze(), y_train, color="black", s=80, zorder=5, label="Observations")
ax.axhline(f_best, color="tomato", linestyle=":", linewidth=1.5, label=f"f_best = {f_best:.2f}")

low_mu_idx = np.argmin(mu)
high_std_idx = np.argmax(std)

ax.annotate("Exploitation zone\n(low μ — likely improvement)",
            xy=(xs[low_mu_idx], mu[low_mu_idx]),
            xytext=(xs[low_mu_idx] + 0.12, mu[low_mu_idx] + 4),
            arrowprops=dict(arrowstyle="->", color="green"),
            color="green", fontsize=8)
ax.annotate("Exploration zone\n(high σ — uncertain region)",
            xy=(xs[high_std_idx], mu[high_std_idx]),
            xytext=(xs[high_std_idx] - 0.28, mu[high_std_idx] - 5),
            arrowprops=dict(arrowstyle="->", color="purple"),
            color="purple", fontsize=8)

ax.set_title("GP posterior on Forrester\n"
             "Acquisition functions balance exploitation (low μ) and exploration (high σ)")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part1_motivation.png", dpi=150)
plt.close()
print("Saved: part1_motivation.png")


# ── Part 2: Probability of Improvement ───────────────────────────────────────

xi_vals = [0.0, 0.05, 0.1]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for col, xi in enumerate(xi_vals):
    pi_vals = acq_pi(mu, std, f_best, xi=xi)
    draw_panel(axes[0, col], axes[1, col], mu, std, pi_vals,
               title=f"PI   ξ = {xi}", acq_label="PI(x)")

plt.suptitle("Probability of Improvement (PI)\n"
             "PI(x) = Φ((f_best − ξ − μ) / σ)   |   ξ=0 is greedy, ξ>0 explores",
             fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part2_pi.png", dpi=150)
plt.close()
print("Saved: part2_pi.png")


# ── Part 3: Expected Improvement ─────────────────────────────────────────────

xi_vals = [0.0, 0.01, 0.1]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for col, xi in enumerate(xi_vals):
    ei_vals = acq_ei(mu, std, f_best, xi=xi)
    draw_panel(axes[0, col], axes[1, col], mu, std, ei_vals,
               title=f"EI   ξ = {xi}", acq_label="EI(x)")

plt.suptitle("Expected Improvement (EI)\n"
             "EI(x) = (f_best−ξ−μ)Φ(Z) + σφ(Z)   |   Weights improvement by its probability",
             fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part3_ei.png", dpi=150)
plt.close()
print("Saved: part3_ei.png")


# ── Part 4: Lower Confidence Bound (UCB) ─────────────────────────────────────

kappa_vals = [0.5, 2.0, 4.0]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for col, kappa in enumerate(kappa_vals):
    lcb_vals = acq_lcb(mu, std, kappa=kappa)
    draw_panel(axes[0, col], axes[1, col], mu, std, lcb_vals,
               title=f"LCB   κ = {kappa}", acq_label="−LCB(x) = κσ−μ")

plt.suptitle("Lower Confidence Bound (LCB / UCB)\n"
             "LCB(x) = μ(x) − κσ(x)   |   κ=0.5 exploits, κ=4.0 explores",
             fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part4_lcb.png", dpi=150)
plt.close()
print("Saved: part4_lcb.png")


# ── Part 5: All three compared ────────────────────────────────────────────────

pi_vals  = acq_pi(mu, std, f_best, xi=0.01)
ei_vals  = acq_ei(mu, std, f_best, xi=0.01)
lcb_vals = acq_lcb(mu, std, kappa=2.0)

x_next_pi  = x_fine[np.argmax(pi_vals)].item()
x_next_ei  = x_fine[np.argmax(ei_vals)].item()
x_next_lcb = x_fine[np.argmax(lcb_vals)].item()

def normalise(v):
    v = v - v.min()
    return v / (v.max() + 1e-9)

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

axes[0].plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
axes[0].plot(xs, mu, color="steelblue", linewidth=2, label="GP mean")
axes[0].fill_between(xs, mu - 2*std, mu + 2*std, alpha=0.15, color="steelblue")
axes[0].scatter(X_train.squeeze(), y_train, color="black", s=60, zorder=5, label="Data")
axes[0].axvline(x_next_pi,  color="tomato",  linewidth=2, linestyle="--",
                label=f"PI next = {x_next_pi:.3f}")
axes[0].axvline(x_next_ei,  color="green",   linewidth=2, linestyle="--",
                label=f"EI next = {x_next_ei:.3f}")
axes[0].axvline(x_next_lcb, color="purple",  linewidth=2, linestyle="--",
                label=f"LCB next = {x_next_lcb:.3f}")
axes[0].set_title("GP posterior — where does each acquisition function suggest next?")
axes[0].set_ylabel("f(x)")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(xs, normalise(pi_vals),  color="tomato",  linewidth=2, label="PI  (ξ=0.01)")
axes[1].plot(xs, normalise(ei_vals),  color="green",   linewidth=2, label="EI  (ξ=0.01)")
axes[1].plot(xs, normalise(lcb_vals), color="purple",  linewidth=2, label="−LCB (κ=2.0)")
axes[1].axvline(x_next_pi,  color="tomato",  linewidth=1.5, linestyle="--")
axes[1].axvline(x_next_ei,  color="green",   linewidth=1.5, linestyle="--")
axes[1].axvline(x_next_lcb, color="purple",  linewidth=1.5, linestyle="--")
axes[1].set_title("Normalised acquisition functions — overlaid for comparison")
axes[1].set_xlabel("x")
axes[1].set_ylabel("Acquisition (normalised)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.suptitle("PI vs EI vs LCB — same GP, different next-point suggestions", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part5_comparison.png", dpi=150)
plt.close()
print("Saved: part5_comparison.png")
print(f"  PI next: {x_next_pi:.3f}   EI next: {x_next_ei:.3f}   LCB next: {x_next_lcb:.3f}")


# ── Part 6: One complete BO step ─────────────────────────────────────────────

ei_for_step = acq_ei(mu, std, f_best, xi=0.01)
x_next      = x_fine[np.argmax(ei_for_step)]
y_next      = forrester(x_next)

X_new    = np.vstack([X_train, x_next])
y_new    = np.append(y_train, y_next)
f_best_new = y_new.min()

mu_new, std_new = gp_predict(X_new, y_new, x_fine, kern, noise=0.01)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(xs, y_true, "gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
axes[0].plot(xs, mu, color="steelblue", linewidth=2, label="GP mean")
axes[0].fill_between(xs, mu-2*std, mu+2*std, alpha=0.2, color="steelblue", label="±2σ")
axes[0].scatter(X_train.squeeze(), y_train, color="black", s=60, zorder=5,
                label=f"Data (n={len(X_train)})")
axes[0].set_title("Step 1: GP posterior before new evaluation")
axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

axes[1].plot(xs, ei_for_step, color="darkorange", linewidth=2, label="EI(x)")
axes[1].fill_between(xs, 0, ei_for_step, alpha=0.25, color="darkorange")
axes[1].axvline(x_next.item(), color="red", linewidth=2.5, linestyle=":",
                label=f"x_next = {x_next.item():.3f}")
axes[1].set_title(f"Step 2: EI → x_next = {x_next.item():.3f}\nf(x_next) = {y_next.item():.3f}")
axes[1].set_xlabel("x"); axes[1].set_ylabel("EI(x)")
axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

axes[2].plot(xs, y_true, "gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
axes[2].plot(xs, mu_new, color="steelblue", linewidth=2, label="GP mean (updated)")
axes[2].fill_between(xs, mu_new-2*std_new, mu_new+2*std_new,
                     alpha=0.2, color="steelblue", label="±2σ")
axes[2].scatter(X_train.squeeze(), y_train, color="black", s=60, zorder=5, label="Old data")
axes[2].scatter([x_next.item()], [y_next.item()], color="red", s=150, zorder=6,
                marker="*", label=f"New point")
axes[2].set_title(f"Step 3: GP updated (n={len(X_new)})\n"
                  f"f_best: {f_best:.3f} → {f_best_new:.3f}")
axes[2].set_xlabel("x"); axes[2].set_ylabel("f(x)")
axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

plt.suptitle("One complete Bayesian Optimisation step\n"
             "GP posterior → EI → evaluate → update — repeat this loop in Lesson 8",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part6_one_bo_step.png", dpi=150)
plt.close()
print("Saved: part6_one_bo_step.png")
print(f"  f_best before: {f_best:.4f}   after: {f_best_new:.4f}")

print("\nAll done. Check lesson-7/output/ for plots.")
print("\nKey takeaways:")
print("  PI = P(improvement): simple, tends to over-exploit.")
print("  EI = expected size of improvement: best default choice.")
print("  LCB = mu - kappa*sigma: tunable, theoretically grounded.")
print("  EI (xi=0.01) is the standard starting point for GP-BO.")
print("  Lesson 8: wrap this into a full iterative optimisation loop.")
