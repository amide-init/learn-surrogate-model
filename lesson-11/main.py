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


# ── Plain MLP (one ensemble member) ──────────────────────────────────────────

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


# ── Deep Ensemble — Method B (Lakshminarayanan et al. 2017) ──────────────────

class DeepEnsemble:
    """
    N independent MLPs trained from different random seeds.
    Diversity comes from random initialisation — no dropout needed.
    Each member is deterministic at inference (model.eval()).
    """
    def __init__(self, n_members=5, input_dim=1, hidden=64):
        self.n_members  = n_members
        self.members    = []
        self.all_losses = []
        for i in range(n_members):
            torch.manual_seed(i * 100)       # different init per member
            self.members.append(MLP(input_dim, hidden))

    def fit(self, X_tr, y_tr, n_epochs=3000, lr=1e-3):
        loss_fn = nn.MSELoss()
        self.all_losses = []
        for i, model in enumerate(self.members):
            opt    = optim.Adam(model.parameters(), lr=lr)
            losses = []
            for _ in range(n_epochs):
                model.train()
                opt.zero_grad()
                loss = loss_fn(model(X_tr).squeeze(), y_tr)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            self.all_losses.append(losses)
            print(f"  Member {i+1}/{self.n_members}  final loss: {losses[-1]:.6f}")
        return self.all_losses

    def predict(self, x):
        """
        Deterministic forward pass through all members.
        Returns (mu, std) in network output space.
        """
        preds = []
        for model in self.members:
            model.eval()
            with torch.no_grad():
                preds.append(model(x).squeeze())
        preds = torch.stack(preds)        # (N, n_points)
        return preds.mean(0), preds.std(0)

    def predict_each(self, x):
        """Return individual member predictions as numpy array (N, n_points)."""
        preds = []
        for model in self.members:
            model.eval()
            with torch.no_grad():
                preds.append(model(x).squeeze().numpy())
        return np.array(preds)


# ── MC Dropout MLP (for comparison in Part 5) ────────────────────────────────

class MCDropoutMLP(nn.Module):
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
        self.train()
        with torch.no_grad():
            preds = torch.stack([self(x) for _ in range(n_passes)])
        return preds.mean(0).squeeze(), preds.std(0).squeeze()


# ── Helpers ───────────────────────────────────────────────────────────────────

def train_single(model, X_tr, y_tr, n_epochs=3000, lr=1e-3):
    opt, loss_fn, losses = optim.Adam(model.parameters(), lr=lr), nn.MSELoss(), []
    for _ in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr).squeeze(), y_tr)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return losses

def to_tensor(arr):
    return torch.from_numpy(arr.astype(np.float32))

def acq_ei(mu, std, f_best, xi=0.01):
    Z = (f_best - xi - mu) / (std + 1e-9)
    return np.maximum((f_best - xi - mu) * norm.cdf(Z) + std * norm.pdf(Z), 0)


# ── Shared training data (same as Lesson 10 for fair comparison) ──────────────

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
x_dense_t = to_tensor((x_dense.reshape(-1, 1) - Xm) / Xs)

# Train ensemble
N_MEMBERS = 5
print(f"Training Deep Ensemble ({N_MEMBERS} members × 3000 epochs)...")
ensemble = DeepEnsemble(n_members=N_MEMBERS, input_dim=1, hidden=64)
all_losses = ensemble.fit(X_tr, y_tr, n_epochs=3000)

# Ensemble predictions on dense grid
mu_s, std_s = ensemble.predict(x_dense_t)
mu_de  = mu_s.numpy()  * ys + ym
std_de = std_s.numpy() * ys

# Individual member predictions
member_preds = ensemble.predict_each(x_dense_t)         # (N, 300)
member_preds = member_preds * ys + ym                   # un-standardize


# ── Part 1: Ensemble diversity — 5 members individually ──────────────────────

colors = ["steelblue", "tomato", "green", "purple", "orange"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for i, (pred, color) in enumerate(zip(member_preds, colors)):
    axes[0].plot(xs, pred, color=color, linewidth=1.8, alpha=0.8,
                 label=f"Member {i+1}")
axes[0].plot(xs, y_true, color="black", linestyle="--", linewidth=1.5,
             alpha=0.4, label="True")
axes[0].scatter(X_np.squeeze(), y_np, color="black", s=60, zorder=5)
axes[0].set_title("5 ensemble members — individual predictions\n"
                  "Each trained from a different random seed")
axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

# Zoom into a sparse region to show disagreement
axes[1].fill_between(xs, member_preds.min(0), member_preds.max(0),
                     alpha=0.2, color="steelblue", label="Member range")
for i, (pred, color) in enumerate(zip(member_preds, colors)):
    axes[1].plot(xs, pred, color=color, linewidth=1.5, alpha=0.7)
axes[1].plot(xs, mu_de, color="black", linewidth=2.5, label="Ensemble mean μ")
axes[1].scatter(X_np.squeeze(), y_np, color="black", s=60, zorder=5, label="Data")
axes[1].set_title("Disagreement between members = uncertainty\n"
                  "Shaded band spans min–max of member predictions")
axes[1].set_xlabel("x"); axes[1].set_ylabel("f(x)")
axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

plt.suptitle("Deep Ensemble diversity — the source of uncertainty estimation\n"
             "Different random initialisations → different learned functions",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part1_diversity.png", dpi=150)
plt.close()
print("Saved: part1_diversity.png")


# ── Part 2: Combined mean and uncertainty ─────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Mean only
axes[0].plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
axes[0].plot(xs, mu_de, color="steelblue", linewidth=2.5, label="Ensemble mean μ(x)")
for i, (pred, color) in enumerate(zip(member_preds, colors)):
    axes[0].plot(xs, pred, color=color, linewidth=0.8, alpha=0.4)
axes[0].scatter(X_np.squeeze(), y_np, color="black", s=60, zorder=5, label="Data")
axes[0].set_title("Individual members (faded) + ensemble mean (bold)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

# Mean + uncertainty band
axes[1].plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
axes[1].plot(xs, mu_de, color="steelblue", linewidth=2, label="Ensemble mean μ(x)")
axes[1].fill_between(xs, mu_de - 2*std_de, mu_de + 2*std_de,
                     alpha=0.25, color="steelblue", label="±2σ (ensemble std)")
axes[1].scatter(X_np.squeeze(), y_np, color="black", s=60, zorder=5, label="Data")
axes[1].set_title("Deep Ensemble posterior: mean ± 2σ")
axes[1].set_xlabel("x"); axes[1].set_ylabel("f(x)")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

plt.suptitle("Combining 5 members: μ = mean of outputs,  σ = std of outputs",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part2_ensemble_prediction.png", dpi=150)
plt.close()
print("Saved: part2_ensemble_prediction.png")


# ── Part 3: Uncertainty calibration ──────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
axes[0].plot(xs, mu_de, color="steelblue", linewidth=2, label="Ensemble mean μ(x)")
axes[0].fill_between(xs, mu_de - 2*std_de, mu_de + 2*std_de,
                     alpha=0.25, color="steelblue", label="±2σ")
axes[0].scatter(X_np.squeeze(), y_np, color="black", s=70, zorder=5, label="Training data")
axes[0].set_ylabel("f(x)"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
axes[0].set_title("Deep Ensemble surrogate — mean and uncertainty")

axes[1].fill_between(xs, 0, std_de, alpha=0.4, color="steelblue", label="σ(x) — ensemble")
axes[1].plot(xs, std_de, color="steelblue", linewidth=2)
for xi in X_np.squeeze():
    axes[1].axvline(xi, color="black", linewidth=0.8, alpha=0.4)
axes[1].set_ylabel("σ(x) — uncertainty")
axes[1].set_xlabel("x")
axes[1].set_title("σ(x): small near training points (ticks), large in gaps")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle("Uncertainty calibration: Deep Ensemble σ reflects data coverage",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part3_uncertainty_calibration.png", dpi=150)
plt.close()
print("Saved: part3_uncertainty_calibration.png")


# ── Part 4: Training — 5 loss curves ─────────────────────────────────────────

epochs = np.arange(1, 3001)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for i, (losses, color) in enumerate(zip(all_losses, colors)):
    axes[0].semilogy(epochs, losses, color=color, linewidth=1.5,
                     alpha=0.8, label=f"Member {i+1}")
axes[0].set_title("5 independent training runs\nEach starts from a different random seed")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE loss (log)")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

# Final losses bar chart
final_losses = [l[-1] for l in all_losses]
axes[1].bar(range(1, N_MEMBERS+1), final_losses, color=colors, alpha=0.8, edgecolor="black")
axes[1].axhline(np.mean(final_losses), color="black", linestyle="--", linewidth=1.5,
                label=f"Mean = {np.mean(final_losses):.5f}")
axes[1].set_title("Final training loss per member\nSmall spread = members trained to similar quality")
axes[1].set_xlabel("Member"); axes[1].set_ylabel("Final MSE loss")
axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")
for i, v in enumerate(final_losses):
    axes[1].text(i+1, v + max(final_losses)*0.01, f"{v:.4f}", ha="center", fontsize=8)

plt.suptitle("Deep Ensemble training: 5 independent optimisations\n"
             "Different initialisations → different local minima → diversity",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part4_training_curves.png", dpi=150)
plt.close()
print("Saved: part4_training_curves.png")


# ── Part 5: Deep Ensembles vs MC Dropout ──────────────────────────────────────

# Train MC Dropout on same data
torch.manual_seed(RANDOM_SEED)
mc_model = MCDropoutMLP(input_dim=1, hidden=64, dropout_rate=0.1)
train_single(mc_model, X_tr, y_tr, n_epochs=3000)

mu_mc_s, std_mc_s = mc_model.mc_predict(x_dense_t, n_passes=50)
mu_mc  = mu_mc_s.numpy()  * ys + ym
std_mc = std_mc_s.numpy() * ys

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (mu, std, label, color) in zip(axes, [
    (mu_mc, std_mc, "MC Dropout (Lesson 10)",    "tomato"),
    (mu_de, std_de, "Deep Ensembles (Lesson 11)", "steelblue"),
]):
    ax.plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
    ax.plot(xs, mu, color=color, linewidth=2, label=f"{label} mean")
    ax.fill_between(xs, mu - 2*std, mu + 2*std, alpha=0.2, color=color, label="±2σ")
    ax.scatter(X_np.squeeze(), y_np, color="black", s=60, zorder=5, label="Data")
    ax.set_title(label)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle(f"MC Dropout vs Deep Ensembles — {n_train} training points\n"
             "Ensembles typically have wider, better-calibrated uncertainty",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part5_ensemble_vs_mcdropout.png", dpi=150)
plt.close()
print("Saved: part5_ensemble_vs_mcdropout.png")

# σ comparison side-by-side
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(xs, std_mc, color="tomato",    linewidth=2, label="MC Dropout σ(x)")
ax.plot(xs, std_de, color="steelblue", linewidth=2, label="Deep Ensemble σ(x)")
for xi in X_np.squeeze():
    ax.axvline(xi, color="black", linewidth=0.6, alpha=0.3)
ax.set_title("Uncertainty comparison — σ(x) for both methods\n"
             "Black ticks = training point locations")
ax.set_xlabel("x"); ax.set_ylabel("σ(x)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part5b_sigma_comparison.png", dpi=150)
plt.close()
print("Saved: part5b_sigma_comparison.png")


# ── Part 6: One BO step with Deep Ensembles ───────────────────────────────────

f_best   = float(y_np.min())
ei_de    = acq_ei(mu_de, std_de, f_best, xi=0.01)
idx_next = int(np.argmax(ei_de))
x_next   = float(xs[idx_next])
y_next   = forrester(np.array([x_next])).item()

# Update and refit ensemble
X_new    = np.vstack([X_np, [[x_next]]]).astype(np.float32)
y_new    = np.append(y_np, y_next).astype(np.float32)
Xm2, Xs2 = float(X_new.mean()), float(X_new.std())
ym2, ys2 = float(y_new.mean()), float(y_new.std())
X_tr2    = to_tensor((X_new - Xm2) / Xs2)
y_tr2    = to_tensor((y_new - ym2) / ys2)

print(f"Refitting ensemble on {len(X_new)} points...")
ensemble2 = DeepEnsemble(n_members=N_MEMBERS, input_dim=1, hidden=64)
ensemble2.fit(X_tr2, y_tr2, n_epochs=3000)

x_dense_t2    = to_tensor((x_dense.reshape(-1, 1) - Xm2) / Xs2)
mu2_s, std2_s = ensemble2.predict(x_dense_t2)
mu2  = mu2_s.numpy()  * ys2 + ym2
std2 = std2_s.numpy() * ys2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
axes[0].plot(xs, mu_de, color="steelblue", linewidth=2, label="Ensemble μ")
axes[0].fill_between(xs, mu_de-2*std_de, mu_de+2*std_de, alpha=0.2, color="steelblue", label="±2σ")
axes[0].scatter(X_np.squeeze(), y_np, color="black", s=60, zorder=5, label=f"Data (n={n_train})")
axes[0].set_title(f"Step 1: Deep Ensemble surrogate (n={n_train})")
axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

axes[1].plot(xs, ei_de, color="darkorange", linewidth=2, label="EI(x)")
axes[1].fill_between(xs, 0, ei_de, alpha=0.25, color="darkorange")
axes[1].axvline(x_next, color="red", linewidth=2.5, linestyle=":",
                label=f"x_next = {x_next:.3f}")
axes[1].set_title(f"Step 2: EI from ensemble μ, σ\nx_next={x_next:.3f}  f(x_next)={y_next:.3f}")
axes[1].set_xlabel("x"); axes[1].set_ylabel("EI(x)")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

axes[2].plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.5, label="True")
axes[2].plot(xs, mu2, color="steelblue", linewidth=2, label="Ensemble μ (updated)")
axes[2].fill_between(xs, mu2-2*std2, mu2+2*std2, alpha=0.2, color="steelblue", label="±2σ")
axes[2].scatter(X_np.squeeze(), y_np, color="black", s=50, zorder=5, label="Old data")
axes[2].scatter([x_next], [y_next], color="red", s=150, zorder=6, marker="*", label="New point")
axes[2].set_title(f"Step 3: ensemble refitted (n={len(X_new)})\n"
                  f"f_best: {f_best:.3f} → {float(y_new.min()):.3f}")
axes[2].set_xlabel("x"); axes[2].set_ylabel("f(x)")
axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

plt.suptitle("BO step with Deep Ensembles: fit → EI → evaluate → update\n"
             "Same loop as Lessons 8 and 10 — only the surrogate changed",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part6_bo_step.png", dpi=150)
plt.close()
print("Saved: part6_bo_step.png")
print(f"  x_next = {x_next:.4f}  y_next = {y_next:.4f}")

print("\nAll done. Check lesson-11/output/ for plots.")
print("\nKey takeaways:")
print("  Deep Ensemble = N plain MLPs trained from different random seeds.")
print("  Diversity comes from random init + SGD stochasticity — no dropout needed.")
print("  μ = mean across members,  σ = std across members.")
print("  Each member uses model.eval() at inference — fully deterministic per member.")
print("  Better calibrated than MC Dropout; N× more expensive to train.")
print("  Lesson 12: head-to-head comparison MC Dropout vs Deep Ensembles vs GP.")
