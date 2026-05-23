import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

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


# ── MLP — the base architecture for Lessons 10 and 11 ────────────────────────

class MLP(nn.Module):
    """
    Two-layer MLP.
    Lesson 10 adds Dropout after each ReLU.
    Lesson 11 trains 5 of these independently (Deep Ensembles).
    """
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


# ── Training loop — reused in Lessons 10 and 11 ──────────────────────────────

def train_model(model, X_tr, y_tr, X_val=None, y_val=None,
                n_epochs=3000, lr=1e-3):
    """
    Standard training loop.
    Returns train_losses list and val_losses list (empty if no val data).
    """
    opt     = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_tr).squeeze()
        loss = loss_fn(pred, y_tr)
        loss.backward()
        opt.step()
        train_losses.append(loss.item())

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(X_val).squeeze(), y_val)
            val_losses.append(val_loss.item())

    return train_losses, val_losses


# ── Part 1: Tensors and NumPy interop ────────────────────────────────────────

# NumPy → Tensor → NumPy
x_np  = np.linspace(0, 1, 200)
x_t   = torch.from_numpy(x_np).float()          # numpy → tensor
x_back = x_t.numpy()                             # tensor → numpy (shared memory)

# Verify round-trip is lossless
assert np.allclose(x_np, x_back)

# Tensor math is identical to NumPy math
y_torch = torch.sin(2 * np.pi * x_t)
y_numpy = np.sin(2 * np.pi * x_np)

# requires_grad: track operations for gradient computation
x_grad = torch.linspace(0, 1, 200, requires_grad=True)
f_x    = (x_grad ** 2).sum()                     # f(x) = x²
f_x.backward()                                    # compute df/dx = 2x
grad_vals = x_grad.grad.detach().numpy()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(x_np, y_numpy, color="steelblue", linewidth=2.5, label="np.sin(2πx)")
axes[0].plot(x_np, y_torch.detach().numpy(), color="tomato",
             linewidth=1.5, linestyle="--", label="torch.sin(2πx)")
axes[0].set_title("NumPy and PyTorch: identical operations\n"
                  "torch.from_numpy() ↔ .numpy() convert between them")
axes[0].set_xlabel("x"); axes[0].set_ylabel("sin(2πx)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(x_np, x_np ** 2, color="steelblue", linewidth=2, label="f(x) = x²")
axes[1].plot(x_np, grad_vals, color="tomato", linewidth=2, label="autograd: ∂f/∂x")
axes[1].plot(x_np, 2 * x_np, color="green", linestyle="--", linewidth=1.5,
             label="analytical: 2x")
axes[1].set_title("requires_grad=True + backward()\nAutograd computes ∂f/∂x automatically")
axes[1].set_xlabel("x"); axes[1].set_ylabel("value")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle("PyTorch tensors — like NumPy arrays with built-in autodiff", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part1_tensors.png", dpi=150)
plt.close()
print("Saved: part1_tensors.png")


# ── Part 2: Autograd — gradient of Forrester ─────────────────────────────────

x_ag = torch.linspace(0, 1, 200, requires_grad=True)
y_ag = ((6 * x_ag - 2) ** 2 * torch.sin(12 * x_ag - 4)).sum()
y_ag.backward()
grad_ag = x_ag.grad.detach().numpy()

# Numerical gradient via finite differences
h = 1e-5
grad_fd = (forrester(x_np + h) - forrester(x_np - h)) / (2 * h)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(x_np, forrester(x_np), color="steelblue", linewidth=2)
axes[0].set_title("Forrester function f(x) = (6x−2)² sin(12x−4)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_np, grad_ag, color="tomato", linewidth=2.5, label="Autograd ∂f/∂x")
axes[1].plot(x_np, grad_fd, color="steelblue", linestyle="--", linewidth=1.5,
             label="Finite differences ∂f/∂x")
max_err = np.abs(grad_ag - grad_fd).max()
axes[1].set_title(f"Autograd vs finite differences\nMax error = {max_err:.2e}")
axes[1].set_xlabel("x"); axes[1].set_ylabel("∂f/∂x")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle("Autograd computes exact gradients — used internally during training", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part2_autograd.png", dpi=150)
plt.close()
print("Saved: part2_autograd.png")


# ── Part 3: MLP architecture and NN prior ─────────────────────────────────────

x_plot = torch.linspace(0, 1, 200).unsqueeze(1)   # (200, 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Show 5 randomly-initialised networks (the "NN prior")
for seed in range(5):
    torch.manual_seed(seed)
    net = MLP(input_dim=1, hidden=64)
    with torch.no_grad():
        out = net(x_plot).squeeze().numpy()
    axes[0].plot(x_np, out, linewidth=1.5, alpha=0.8)

axes[0].set_title("5 randomly-initialised MLPs (NN 'prior')\n"
                  "No training yet — output depends only on random weight initialisation")
axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
axes[0].grid(True, alpha=0.3)

# Print architecture
torch.manual_seed(RANDOM_SEED)
model_demo = MLP(input_dim=1, hidden=64)
layers = []
for name, module in model_demo.named_modules():
    if isinstance(module, (nn.Linear, nn.ReLU)):
        layers.append(f"  {name}: {module}")

axes[1].axis("off")
arch_text = (
    "MLP Architecture\n"
    "─────────────────────────────\n"
    "Input:          (batch, 1)\n"
    "\n"
    "Linear(1 → 64)  +  ReLU\n"
    "Linear(64 → 64) +  ReLU\n"
    "Linear(64 → 1)\n"
    "\n"
    "─────────────────────────────\n"
    f"Total params: {sum(p.numel() for p in model_demo.parameters()):,}\n"
    "\n"
    "Lesson 10:  add Dropout(0.1) after each ReLU\n"
    "Lesson 11:  train 5 of these independently"
)
axes[1].text(0.05, 0.5, arch_text, transform=axes[1].transAxes,
             fontsize=11, verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.suptitle("MLP: the base architecture for all NN surrogates in this course", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part3_architecture.png", dpi=150)
plt.close()
print("Saved: part3_architecture.png")


# ── Part 4: Training loop — fit to Forrester ─────────────────────────────────

n_train = 20
X_np  = rng.uniform(0, 1, (n_train, 1)).astype(np.float32)
y_np  = forrester(X_np).astype(np.float32)

# Standardize inputs and outputs
X_mean, X_std = X_np.mean(), X_np.std()
y_mean, y_std = y_np.mean(), y_np.std()

X_tr = torch.from_numpy((X_np - X_mean) / X_std)
y_tr = torch.from_numpy((y_np - y_mean) / y_std)

torch.manual_seed(RANDOM_SEED)
model = MLP(input_dim=1, hidden=64)
train_losses, _ = train_model(model, X_tr, y_tr, n_epochs=3000)

# Predict on dense grid
x_test_np = np.linspace(0, 1, 300).astype(np.float32)
x_test_std = torch.from_numpy((x_test_np.reshape(-1, 1) - X_mean) / X_std)

model.eval()
with torch.no_grad():
    y_pred_std = model(x_test_std).squeeze().numpy()
y_pred = y_pred_std * y_std + y_mean    # un-standardize

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].semilogy(train_losses, color="steelblue", linewidth=1.5)
axes[0].set_title("Training loss (MSE) — log scale")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log scale)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(x_test_np, forrester(x_test_np), color="gray", linestyle="--",
             linewidth=1.5, alpha=0.6, label="True Forrester")
axes[1].plot(x_test_np, y_pred, color="steelblue", linewidth=2, label="MLP prediction")
axes[1].scatter(X_np.squeeze(), y_np, color="black", s=60, zorder=5,
                label=f"Training data (n={n_train})")
axes[1].set_title(f"MLP fit after 3000 epochs\nFinal loss = {train_losses[-1]:.5f}")
axes[1].set_xlabel("x"); axes[1].set_ylabel("f(x)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle("Training loop: zero_grad → forward → loss → backward → step",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part4_training.png", dpi=150)
plt.close()
print("Saved: part4_training.png")
print(f"  Final train loss: {train_losses[-1]:.6f}")


# ── Part 5: Standardization + train/val split ─────────────────────────────────

n_total = 25
X_all_np = rng.uniform(0, 1, (n_total, 1)).astype(np.float32)
y_all_np  = forrester(X_all_np).astype(np.float32)

n_tr  = 18
X_tr_np, X_val_np = X_all_np[:n_tr], X_all_np[n_tr:]
y_tr_np, y_val_np = y_all_np[:n_tr], y_all_np[n_tr:]

# Standardize using training stats only
Xm, Xs = X_tr_np.mean(), X_tr_np.std()
ym, ys = y_tr_np.mean(), y_tr_np.std()

def to_tensor(arr): return torch.from_numpy(arr)

X_tr_t  = to_tensor((X_tr_np  - Xm) / Xs)
X_val_t = to_tensor((X_val_np - Xm) / Xs)
y_tr_t  = to_tensor((y_tr_np  - ym) / ys)
y_val_t = to_tensor((y_val_np - ym) / ys)

# Train WITH standardization
torch.manual_seed(RANDOM_SEED)
model_std = MLP()
tl_std, vl_std = train_model(model_std, X_tr_t, y_tr_t,
                              X_val_t, y_val_t, n_epochs=3000)

# Train WITHOUT standardization
torch.manual_seed(RANDOM_SEED)
model_raw = MLP()
tl_raw, vl_raw = train_model(model_raw,
                              to_tensor(X_tr_np), to_tensor(y_tr_np),
                              to_tensor(X_val_np), to_tensor(y_val_np),
                              n_epochs=3000)

epochs = np.arange(1, 3001)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].semilogy(epochs, tl_raw, color="steelblue", linewidth=1.5, label="Train")
axes[0].semilogy(epochs, vl_raw, color="tomato",   linewidth=1.5, label="Val")
axes[0].set_title("Without standardization\nLarge loss magnitudes, slow convergence")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (log)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].semilogy(epochs, tl_std, color="steelblue", linewidth=1.5, label="Train")
axes[1].semilogy(epochs, vl_std, color="tomato",   linewidth=1.5, label="Val")
axes[1].set_title("With standardization\nFaster convergence, stable training")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss (log)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle("Why standardization matters for NN training\n"
             "Normalize inputs and outputs using training-set mean and std",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part5_standardization.png", dpi=150)
plt.close()
print("Saved: part5_standardization.png")


# ── Part 6: Save and load a model ────────────────────────────────────────────

model_path = OUTPUT_DIR / "mlp_forrester.pt"
torch.save(model_std.state_dict(), model_path)

# Load into a fresh model
torch.manual_seed(99)               # different seed — fresh random weights
model_loaded = MLP()
model_loaded.load_state_dict(torch.load(model_path, weights_only=True))
model_loaded.eval()

# Predictions before and after loading
x_dense = np.linspace(0, 1, 300).astype(np.float32)
x_dense_t = to_tensor((x_dense.reshape(-1, 1) - Xm) / Xs)

model_std.eval()
with torch.no_grad():
    pred_orig   = model_std(x_dense_t).squeeze().numpy() * ys + ym
    pred_loaded = model_loaded(x_dense_t).squeeze().numpy() * ys + ym

max_diff = np.abs(pred_orig - pred_loaded).max()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(x_dense, forrester(x_dense), color="gray", linestyle="--",
             linewidth=1.5, alpha=0.6, label="True")
axes[0].plot(x_dense, pred_orig,   color="steelblue", linewidth=2.5, label="Original model")
axes[0].plot(x_dense, pred_loaded, color="tomato",   linewidth=1.5,
             linestyle=":", label="Loaded model")
axes[0].scatter(X_tr_np.squeeze(), y_tr_np, color="black", s=40, zorder=5)
axes[0].set_title(f"Predictions before and after loading\nMax difference = {max_diff:.2e}")
axes[0].set_xlabel("x"); axes[0].set_ylabel("f(x)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].axis("off")
save_text = (
    "How to save and load a PyTorch model\n"
    "─────────────────────────────────────────\n"
    "\n"
    "# Save (weights only — recommended)\n"
    "torch.save(model.state_dict(), 'model.pt')\n"
    "\n"
    "# Load\n"
    "model = MLP()   # same architecture\n"
    "model.load_state_dict(\n"
    "    torch.load('model.pt', weights_only=True)\n"
    ")\n"
    "model.eval()    # switch off dropout/batchnorm\n"
    "\n"
    "─────────────────────────────────────────\n"
    "state_dict = all learned weights and biases\n"
    "Architecture must match — weights only,\n"
    "not the full object."
)
axes[1].text(0.05, 0.5, save_text, transform=axes[1].transAxes,
             fontsize=10, verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.suptitle("Saving and loading a PyTorch model", fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part6_save_load.png", dpi=150)
plt.close()
print("Saved: part6_save_load.png")
print(f"  Max prediction difference after load: {max_diff:.2e}")

print("\nAll done. Check lesson-9/output/ for plots.")
print("\nKey takeaways:")
print("  Tensor = NumPy array + autograd. Convert with from_numpy() / .numpy().")
print("  backward() computes all gradients in one pass — exact, not numerical.")
print("  Training loop: zero_grad → forward → loss → backward → step.")
print("  Always standardize inputs and outputs before training an NN.")
print("  MLP(input_dim, hidden=64) is the base for MC Dropout and Deep Ensembles.")
print("  Lesson 10: add Dropout(0.1) to this MLP for uncertainty estimation.")
