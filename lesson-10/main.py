import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def forrester(x):
    x = np.asarray(x).ravel()
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


# ── MC Dropout MLP — Method A (Gal & Ghahramani 2016) ─────────────────────────

class MCDropoutMLP(nn.Module):
    """
    Two-layer MLP with Dropout after each ReLU.
    At inference, call model.train() to keep dropout active,
    then run T forward passes to estimate μ and σ.
    """
    def __init__(self, input_dim=1, hidden=64, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

    def mc_predict(self, x, n_passes=50):
        """
        Run n_passes stochastic forward passes with dropout ON.
        Returns (mu, std) in the network's output space.
        """
        self.train()                                             # keep dropout ON
        with torch.no_grad():
            preds = torch.stack([self(x) for _ in range(n_passes)])  # (T, N, 1)
        mu  = preds.mean(0).squeeze()
        std = preds.std(0).squeeze()
        return mu, std


# ── Plain MLP (no dropout) — for comparison ───────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# ── GP (baseline) ─────────────────────────────────────────────────────────────

def matern52(X1, X2, length_scale=0.25, signal_var=4.0):
    X1 = np.atleast_2d(X1); X2 = np.atleast_2d(X2)
    diff = X1[:, None, :] - X2[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    s = np.sqrt(5) * r / length_scale
    return signal_var * (1 + s + s ** 2 / 3) * np.exp(-s)

def gp_predict(X_train, y_train, X_test, noise=0.01):
    kern = lambda a, b: matern52(a, b)
    K    = kern(X_train, X_train) + noise * np.eye(len(X_train))
    Ks   = kern(X_test, X_train)
    Kss  = kern(X_test, X_test)
    L    = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu   = Ks @ alpha
    v    = np.linalg.solve(L, Ks.T)
    std  = np.sqrt(np.maximum(np.diag(Kss) - np.sum(v ** 2, axis=0), 0))
    return mu, std


# ── Helpers ───────────────────────────────────────────────────────────────────

def train_model(model, X_tr, y_tr, n_epochs=3000, lr=1e-3):
    opt     = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses  = []
    for _ in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr).squeeze(), y_tr)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

def to_tensor(arr):
    return torch.from_numpy(arr.astype(np.float32))

def lhs(n, d, seed=0):
    r = np.random.default_rng(seed)
    result = np.zeros((n, d))
    for j in range(d):
        result[:, j] = (r.permutation(n) + r.uniform(0, 1, n)) / n
    return result

def acq_ei(mu, std, f_best, xi=0.01):
    Z = (f_best - xi - mu) / (std + 1e-9)
    return np.maximum((f_best - xi - mu) * norm.cdf(Z) + std * norm.pdf(Z), 0)


# ── Training data and model (shared across parts) ─────────────────────────────

n_train  = 15
X_np     = rng.uniform(0, 1, (n_train, 1)).astype(np.float32)
y_np     = forrester(X_np).astype(np.float32)

Xm, Xs   = float(X_np.mean()), float(X_np.std())
ym, ys   = float(y_np.mean()), float(y_np.std())

X_tr     = to_tensor((X_np - Xm) / Xs)
y_tr     = to_tensor((y_np - ym) / ys)

x_dense  = np.linspace(0, 1, 300).astype(np.float32)
xs       = x_dense.squeeze()
y_true   = forrester(xs)
x_dense_t = to_tensor(((x_dense.reshape(-1, 1)) - Xm) / Xs)

torch.manual_seed(RANDOM_SEED)
model = MCDropoutMLP(input_dim=1, hidden=64, dropout_rate=0.1)
train_losses = train_model(model, X_tr, y_tr, n_epochs=3000)

# MC predictions on dense grid
N_PASSES = 50
mu_std_t, std_std_t = model.mc_predict(x_dense_t, n_passes=N_PASSES)
mu_mc  = mu_std_t.numpy()  * ys + ym
std_mc = std_std_t.numpy() * ys


# ── Part 1: eval() vs train() — the key difference ───────────────────────────

model.eval()
preds_eval = []
for _ in range(5):
    with torch.no_grad():
        out = model(x_dense_t).squeeze().numpy() * ys + ym
    preds_eval.append(out)

model.train()   # MC Dropout mode
preds_train = []
for _ in range(5):
    with torch.no_grad():
        out = model(x_dense_t).squeeze().numpy() * ys + ym
    preds_train.append(out)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for p in preds_eval:
    axes[0].plot(xs, p, linewidth=1.5, alpha=0.8)
axes[0].scatter(X_np.squeeze(), y_np, color="black", s=60, zorder=5)
axes[0].set_title("model.eval() — dropout OFF\n5 calls → identical output (deterministic)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
axes[0].grid(True, alpha=0.3)
axes[0].text(0.02, 0.95, "All 5 lines overlap", transform=axes[0].transAxes,
             fontsize=9, color="gray", va="top")

for p in preds_train:
    axes[1].plot(xs, p, linewidth=1.5, alpha=0.8)
axes[1].scatter(X_np.squeeze(), y_np, color="black", s=60, zorder=5)
axes[1].set_title("model.train() — dropout ON\n5 calls → different outputs (stochastic)")
axes[1].set_xlabel("x"); axes[1].set_ylabel("f(x)")
axes[1].grid(True, alpha=0.3)
axes[1].text(0.02, 0.95, "Lines differ = uncertainty", transform=axes[1].transAxes,
             fontsize=9, color="green", va="top")

plt.suptitle("MC Dropout: one line of code changes inference from deterministic to stochastic\n"
             "model.eval()  →  standard NN       model.train()  →  MC Dropout", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part1_eval_vs_train.png", dpi=150)
plt.close()
print("Saved: part1_eval_vs_train.png")


# ── Part 2: 50 MC passes ──────────────────────────────────────────────────────

model.train()
all_passes = []
with torch.no_grad():
    for _ in range(N_PASSES):
        out = model(x_dense_t).squeeze().numpy() * ys + ym
        all_passes.append(out)
all_passes = np.array(all_passes)   # (50, 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for i, p in enumerate(all_passes):
    axes[0].plot(xs, p, color="steelblue", linewidth=0.6, alpha=0.3)
axes[0].plot(xs, all_passes.mean(0), color="steelblue", linewidth=2.5, label="Mean μ(x)")
axes[0].scatter(X_np.squeeze(), y_np, color="black", s=60, zorder=5, label="Data")
axes[0].set_title(f"{N_PASSES} MC forward passes (thin lines)\nSpread of lines = uncertainty")
axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

mu_all  = all_passes.mean(0)
std_all = all_passes.std(0)
axes[1].plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
axes[1].plot(xs, mu_all, color="steelblue", linewidth=2, label="MC mean μ(x)")
axes[1].fill_between(xs, mu_all - 2*std_all, mu_all + 2*std_all,
                     alpha=0.25, color="steelblue", label="±2σ (MC Dropout)")
axes[1].scatter(X_np.squeeze(), y_np, color="black", s=60, zorder=5, label="Data")
axes[1].set_title("MC Dropout posterior: mean ± 2σ")
axes[1].set_xlabel("x"); axes[1].set_ylabel("f(x)")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

plt.suptitle(f"T={N_PASSES} stochastic forward passes → mean μ(x) and std σ(x)", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part2_mc_passes.png", dpi=150)
plt.close()
print("Saved: part2_mc_passes.png")


# ── Part 3: Uncertainty scales with data sparsity ─────────────────────────────

# Show σ(x) alongside the training point density
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
axes[0].plot(xs, mu_mc, color="steelblue", linewidth=2, label="MC mean μ(x)")
axes[0].fill_between(xs, mu_mc - 2*std_mc, mu_mc + 2*std_mc,
                     alpha=0.25, color="steelblue", label="±2σ")
axes[0].scatter(X_np.squeeze(), y_np, color="black", s=70, zorder=5, label="Training data")
axes[0].set_ylabel("f(x)"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
axes[0].set_title("MC Dropout surrogate — mean and uncertainty")

axes[1].fill_between(xs, 0, std_mc, alpha=0.4, color="tomato", label="σ(x)")
axes[1].plot(xs, std_mc, color="tomato", linewidth=2)
for xi in X_np.squeeze():
    axes[1].axvline(xi, color="black", linewidth=0.8, alpha=0.4)
axes[1].set_ylabel("σ(x) — uncertainty")
axes[1].set_xlabel("x")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_title("σ(x) is small near data (black ticks), large in sparse regions")

plt.suptitle("Uncertainty calibration: MC Dropout uncertainty reflects data coverage",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part3_uncertainty_calibration.png", dpi=150)
plt.close()
print("Saved: part3_uncertainty_calibration.png")


# ── Part 4: Training — MC Dropout vs plain MLP ───────────────────────────────

torch.manual_seed(RANDOM_SEED)
model_plain = MLP(input_dim=1, hidden=64)
losses_plain = train_model(model_plain, X_tr, y_tr, n_epochs=3000)

model_plain.eval()
with torch.no_grad():
    y_plain = model_plain(x_dense_t).squeeze().numpy() * ys + ym

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

epochs = np.arange(1, 3001)
axes[0].semilogy(epochs, train_losses,  color="steelblue", linewidth=2,
                 label="MCDropoutMLP (p=0.1)")
axes[0].semilogy(epochs, losses_plain,  color="tomato",    linewidth=2,
                 label="Plain MLP (no dropout)")
axes[0].set_title("Training loss comparison")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE loss (log)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
axes[1].plot(xs, mu_mc,  color="steelblue", linewidth=2, label="MC Dropout mean")
axes[1].fill_between(xs, mu_mc - 2*std_mc, mu_mc + 2*std_mc,
                     alpha=0.2, color="steelblue", label="±2σ (MC Dropout)")
axes[1].plot(xs, y_plain, color="tomato", linewidth=2, linestyle="--",
             label="Plain MLP (no uncertainty)")
axes[1].scatter(X_np.squeeze(), y_np, color="black", s=60, zorder=5, label="Data")
axes[1].set_title("Fit comparison: MC Dropout adds uncertainty at no extra training cost")
axes[1].set_xlabel("x"); axes[1].set_ylabel("f(x)")
axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

plt.suptitle("MC Dropout vs plain MLP — same architecture, dropout adds uncertainty estimation",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part4_training_comparison.png", dpi=150)
plt.close()
print("Saved: part4_training_comparison.png")
print(f"  Final loss — MC Dropout: {train_losses[-1]:.5f}  Plain MLP: {losses_plain[-1]:.5f}")


# ── Part 5: MC Dropout vs GP surrogate ───────────────────────────────────────

mu_gp, std_gp = gp_predict(X_np, y_np, x_dense.reshape(-1, 1), noise=0.01)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (mu, std, label, color) in zip(axes, [
    (mu_mc,  std_mc,  "MC Dropout (NN)",    "steelblue"),
    (mu_gp,  std_gp,  "GP (Matern-5/2)",    "green"),
]):
    ax.plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
    ax.plot(xs, mu, color=color, linewidth=2, label=f"{label} mean")
    ax.fill_between(xs, mu - 2*std, mu + 2*std,
                    alpha=0.2, color=color, label="±2σ")
    ax.scatter(X_np.squeeze(), y_np, color="black", s=60, zorder=5, label="Data")
    ax.set_title(label)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle(f"MC Dropout vs GP surrogate — same {n_train} training points\n"
             "Both give mean + uncertainty; NN scales to high dimensions, GP does not",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part5_mcdropout_vs_gp.png", dpi=150)
plt.close()
print("Saved: part5_mcdropout_vs_gp.png")


# ── Part 6: One BO step with MC Dropout ──────────────────────────────────────

x_cand     = np.linspace(0, 1, 500).astype(np.float32).reshape(-1, 1)
x_cand_t   = to_tensor((x_cand - Xm) / Xs)

f_best     = float(y_np.min())
ei_mc      = acq_ei(mu_mc, std_mc, f_best, xi=0.01)
idx_next   = int(np.argmax(ei_mc))
x_next_val = float(x_cand[idx_next, 0])
y_next_val = forrester(np.array([x_next_val])).item()

# Update training set and refit
X_new    = np.vstack([X_np, [[x_next_val]]]).astype(np.float32)
y_new    = np.append(y_np, y_next_val).astype(np.float32)
Xm2, Xs2 = float(X_new.mean()), float(X_new.std())
ym2, ys2 = float(y_new.mean()), float(y_new.std())

X_tr2    = to_tensor((X_new - Xm2) / Xs2)
y_tr2    = to_tensor((y_new - ym2) / ys2)

torch.manual_seed(RANDOM_SEED + 1)
model2   = MCDropoutMLP(input_dim=1, hidden=64, dropout_rate=0.1)
train_model(model2, X_tr2, y_tr2, n_epochs=3000)

x_dense_t2   = to_tensor((x_dense.reshape(-1, 1) - Xm2) / Xs2)
mu2_s, std2_s = model2.mc_predict(x_dense_t2, n_passes=N_PASSES)
mu2  = mu2_s.numpy()  * ys2 + ym2
std2 = std2_s.numpy() * ys2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
axes[0].plot(xs, mu_mc, color="steelblue", linewidth=2, label="MC Dropout μ")
axes[0].fill_between(xs, mu_mc-2*std_mc, mu_mc+2*std_mc, alpha=0.2, color="steelblue", label="±2σ")
axes[0].scatter(X_np.squeeze(), y_np, color="black", s=60, zorder=5, label=f"Data (n={n_train})")
axes[0].set_title(f"Step 1: MC Dropout surrogate\n(n={n_train} training points)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

axes[1].plot(xs, ei_mc, color="darkorange", linewidth=2, label="EI(x)")
axes[1].fill_between(xs, 0, ei_mc, alpha=0.25, color="darkorange")
axes[1].axvline(x_next_val, color="red", linewidth=2.5, linestyle=":",
                label=f"x_next = {x_next_val:.3f}")
axes[1].set_title(f"Step 2: EI from MC Dropout μ, σ\nx_next = {x_next_val:.3f}  "
                  f"f(x_next) = {y_next_val:.3f}")
axes[1].set_xlabel("x"); axes[1].set_ylabel("EI(x)")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

axes[2].plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
axes[2].plot(xs, mu2, color="steelblue", linewidth=2, label="MC Dropout μ (updated)")
axes[2].fill_between(xs, mu2-2*std2, mu2+2*std2, alpha=0.2, color="steelblue", label="±2σ")
axes[2].scatter(X_np.squeeze(), y_np, color="black", s=50, zorder=5, label="Old data")
axes[2].scatter([x_next_val], [y_next_val], color="red", s=150,
                zorder=6, marker="*", label="New point")
axes[2].set_title(f"Step 3: surrogate updated (n={len(X_new)})\n"
                  f"f_best: {f_best:.3f} → {float(y_new.min()):.3f}")
axes[2].set_xlabel("x"); axes[2].set_ylabel("f(x)")
axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

plt.suptitle("One BO step with MC Dropout: fit → EI → evaluate → update\n"
             "Same loop as Lesson 8 — only the surrogate changed", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part6_bo_step.png", dpi=150)
plt.close()
print("Saved: part6_bo_step.png")
print(f"  x_next = {x_next_val:.4f}  y_next = {y_next_val:.4f}")

print("\nAll done. Check lesson-10/output/ for plots.")
print("\nKey takeaways:")
print("  MC Dropout = standard dropout + model.train() at inference.")
print("  T forward passes → mean μ(x) and std σ(x) — no extra training cost.")
print("  σ(x) grows in regions far from training data — calibrated uncertainty.")
print("  Plug μ, σ into EI/LCB — same acquisition functions as GP-BO.")
print("  Lesson 11: Deep Ensembles gives better calibrated σ at 5× training cost.")
