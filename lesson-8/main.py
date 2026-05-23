import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Benchmark functions ───────────────────────────────────────────────────────

def forrester(x):
    x = np.asarray(x).ravel()
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

def branin_norm(X):
    X = np.atleast_2d(X)
    x1 = X[:, 0] * 15 - 5
    x2 = X[:, 1] * 15
    a, b, c = 1.0, 5.1 / (4 * np.pi ** 2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

FORRESTER_MIN = forrester(np.linspace(0, 1, 10000)).min()
BRANIN_MIN    = 0.397887   # known global minimum of Branin


# ── GP and kernel ─────────────────────────────────────────────────────────────

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


# ── Acquisition and LHS ───────────────────────────────────────────────────────

def acq_ei(mu, std, f_best, xi=0.01):
    Z = (f_best - xi - mu) / (std + 1e-9)
    return np.maximum((f_best - xi - mu) * norm.cdf(Z) + std * norm.pdf(Z), 0)

def lhs(n, d, seed=None):
    r = np.random.default_rng(seed)
    result = np.zeros((n, d))
    for j in range(d):
        result[:, j] = (r.permutation(n) + r.uniform(0, 1, n)) / n
    return result


# ── BO loop ───────────────────────────────────────────────────────────────────

def run_bo_1d(n_init, budget, kern_fn, noise=0.01, xi=0.01, seed=0,
              snapshot_at=None):
    """
    BO loop on 1D Forrester.
    Returns best_history (length = budget - n_init + 1) and snapshots dict.
    """
    r   = np.random.default_rng(seed)
    X   = lhs(n_init, 1, seed=seed)
    y   = forrester(X)
    x_cand = np.linspace(0, 1, 500).reshape(-1, 1)

    best_history = [float(y.min())]
    snapshots    = {}
    snap_set     = set(snapshot_at or [])

    for i in range(budget - n_init):
        mu, std = gp_predict(X, y, x_cand, kern_fn, noise)
        f_best  = float(y.min())
        ei_vals = acq_ei(mu, std, f_best, xi=xi)
        idx     = int(np.argmax(ei_vals))

        if i in snap_set:
            snapshots[i] = dict(X=X.copy(), y=y.copy(),
                                mu=mu, std=std, ei=ei_vals,
                                x_next=float(x_cand[idx, 0]),
                                y_next=float(forrester(x_cand[[idx]]).item()),
                                f_best=f_best, n_eval=n_init + i)

        x_next = x_cand[[idx]]
        X      = np.vstack([X, x_next])
        y      = np.append(y, forrester(x_next).item())
        best_history.append(float(y.min()))

    return X, y, best_history, snapshots


def run_bo_2d(n_init, budget, kern_fn, noise=0.1, xi=0.01, seed=0):
    r      = np.random.default_rng(seed)
    X      = lhs(n_init, 2, seed=seed)
    y      = branin_norm(X)
    g      = np.linspace(0, 1, 30)
    G1, G2 = np.meshgrid(g, g)
    x_cand = np.column_stack([G1.ravel(), G2.ravel()])

    best_history = [float(y.min())]

    for _ in range(budget - n_init):
        mu, std = gp_predict(X, y, x_cand, kern_fn, noise)
        ei_vals = acq_ei(mu, std, float(y.min()), xi=xi)
        idx     = int(np.argmax(ei_vals))
        x_next  = x_cand[[idx]]
        X       = np.vstack([X, x_next])
        y       = np.append(y, branin_norm(x_next).item())
        best_history.append(float(y.min()))

    return X, y, best_history


# ── Kernel config ─────────────────────────────────────────────────────────────

kern_1d = lambda X1, X2: matern52(X1, X2, length_scale=0.25, signal_var=4.0)
kern_2d = lambda X1, X2: matern52(X1, X2, length_scale=0.3,  signal_var=50.0)


# ── Part 1: LHS vs random sampling ───────────────────────────────────────────

n_pts      = 20
lhs_pts    = lhs(n_pts, 2, seed=0)
random_pts = np.random.default_rng(1).uniform(0, 1, (n_pts, 2))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, pts, title in zip(axes,
    [random_pts, lhs_pts],
    ["Random sampling — gaps and clusters", "Latin Hypercube — one point per stratum"]):
    ax.scatter(pts[:, 0], pts[:, 1], s=80, color="steelblue",
               edgecolors="black", linewidths=0.7)
    for k in range(n_pts + 1):
        v = k / n_pts
        ax.axvline(v, color="gray", linewidth=0.4, alpha=0.5)
        ax.axhline(v, color="gray", linewidth=0.4, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.set_aspect("equal")

plt.suptitle(f"Initial design: n={n_pts} points in 2D\n"
             "LHS guarantees exactly one point per row and column in the stratification grid",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part1_lhs.png", dpi=150)
plt.close()
print("Saved: part1_lhs.png")


# ── Part 2: GP snapshots during BO loop ──────────────────────────────────────

n_init  = 5
budget  = 25
snap_at = [0, 4, 9, 19]   # iteration indices (0-based after init)

_, _, _, snaps = run_bo_1d(n_init, budget, kern_1d, seed=0, snapshot_at=snap_at)

x_fine = np.linspace(0, 1, 500).reshape(-1, 1)
xs     = x_fine.squeeze()
y_true = forrester(xs)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for col, it in enumerate(snap_at):
    if it not in snaps:
        continue
    s = snaps[it]
    ax_gp, ax_ei = axes[0, col], axes[1, col]

    ax_gp.plot(xs, y_true, color="gray", linestyle="--", linewidth=1.5, alpha=0.4, label="True")
    ax_gp.plot(xs, s["mu"], color="steelblue", linewidth=2, label="GP μ")
    ax_gp.fill_between(xs, s["mu"] - 2*s["std"], s["mu"] + 2*s["std"],
                       alpha=0.2, color="steelblue", label="±2σ")
    ax_gp.scatter(s["X"].squeeze(), s["y"], color="black", s=50, zorder=5)
    ax_gp.axvline(s["x_next"], color="red", linewidth=2, linestyle=":",
                  label=f"x_next={s['x_next']:.3f}")
    ax_gp.set_title(f"After {s['n_eval']} evals\nf_best={s['f_best']:.3f}")
    ax_gp.set_ylabel("f(x)"); ax_gp.legend(fontsize=6); ax_gp.grid(True, alpha=0.3)

    ax_ei.plot(xs, s["ei"], color="darkorange", linewidth=2)
    ax_ei.fill_between(xs, 0, s["ei"], alpha=0.25, color="darkorange")
    ax_ei.axvline(s["x_next"], color="red", linewidth=2, linestyle=":")
    ax_ei.set_ylabel("EI(x)"); ax_ei.set_xlabel("x"); ax_ei.grid(True, alpha=0.3)

plt.suptitle("GP-BO on Forrester — GP and EI at four stages\n"
             "Uncertainty collapses around evaluated points; EI shifts toward unexplored regions",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part2_bo_snapshots.png", dpi=150)
plt.close()
print("Saved: part2_bo_snapshots.png")


# ── Part 3: Convergence curve (multiple runs) ─────────────────────────────────

n_runs  = 10
budget  = 30
n_init  = 5

bo_curves = []
for seed in range(n_runs):
    _, _, hist, _ = run_bo_1d(n_init, budget, kern_1d, seed=seed)
    bo_curves.append(hist)

bo_arr = np.array(bo_curves)               # (n_runs, budget - n_init + 1)
iters  = np.arange(bo_arr.shape[1]) + n_init

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(iters, bo_arr.mean(axis=0), color="steelblue", linewidth=2.5, label="GP-BO mean")
ax.fill_between(iters,
                bo_arr.mean(axis=0) - bo_arr.std(axis=0),
                bo_arr.mean(axis=0) + bo_arr.std(axis=0),
                alpha=0.2, color="steelblue", label="±1 std")
for curve in bo_curves:
    ax.plot(iters, curve, color="steelblue", linewidth=0.6, alpha=0.3)

ax.axhline(FORRESTER_MIN, color="tomato", linestyle="--", linewidth=1.5,
           label=f"True minimum = {FORRESTER_MIN:.3f}")
ax.set_title(f"GP-BO convergence on Forrester ({n_runs} independent runs)\n"
             f"Budget = {budget} evaluations, n_init = {n_init}")
ax.set_xlabel("Number of evaluations")
ax.set_ylabel("Best f found")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part3_convergence.png", dpi=150)
plt.close()
print("Saved: part3_convergence.png")


# ── Part 4: BO vs random search ──────────────────────────────────────────────

random_curves = []
for seed in range(n_runs):
    r = np.random.default_rng(seed + 100)
    X_rand = r.uniform(0, 1, (budget, 1))
    y_rand = forrester(X_rand)
    random_curves.append([float(y_rand[:k+1].min()) for k in range(budget)])

rand_arr = np.array(random_curves)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(np.arange(1, budget + 1), rand_arr.mean(axis=0),
        color="tomato", linewidth=2.5, label="Random search")
ax.fill_between(np.arange(1, budget + 1),
                rand_arr.mean(axis=0) - rand_arr.std(axis=0),
                rand_arr.mean(axis=0) + rand_arr.std(axis=0),
                alpha=0.2, color="tomato")

bo_iters_full = np.arange(bo_arr.shape[1]) + n_init
ax.plot(bo_iters_full, bo_arr.mean(axis=0),
        color="steelblue", linewidth=2.5, label="GP-BO (EI, ξ=0.01)")
ax.fill_between(bo_iters_full,
                bo_arr.mean(axis=0) - bo_arr.std(axis=0),
                bo_arr.mean(axis=0) + bo_arr.std(axis=0),
                alpha=0.2, color="steelblue")

ax.axhline(FORRESTER_MIN, color="black", linestyle="--", linewidth=1.2,
           label=f"True minimum = {FORRESTER_MIN:.3f}")
ax.set_title(f"GP-BO vs Random Search on Forrester\n({n_runs} runs, budget={budget})")
ax.set_xlabel("Number of evaluations")
ax.set_ylabel("Best f found")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part4_bo_vs_random.png", dpi=150)
plt.close()
print("Saved: part4_bo_vs_random.png")


# ── Part 5: 2D BO on Branin ──────────────────────────────────────────────────

budget_2d = 40
n_init_2d = 10
n_runs_2d = 5

bo_curves_2d = []
X_final_2d, y_final_2d = None, None
for seed in range(n_runs_2d):
    X_f, y_f, hist = run_bo_2d(n_init_2d, budget_2d, kern_2d, seed=seed)
    bo_curves_2d.append(hist)
    if seed == 0:
        X_final_2d, y_final_2d = X_f, y_f

bo_2d_arr  = np.array(bo_curves_2d)
iters_2d   = np.arange(bo_2d_arr.shape[1]) + n_init_2d

g = np.linspace(0, 1, 80)
G1, G2 = np.meshgrid(g, g)
z_true = branin_norm(np.column_stack([G1.ravel(), G2.ravel()])).reshape(80, 80)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cf = axes[0].contourf(G1, G2, z_true, levels=25, cmap="viridis")
plt.colorbar(cf, ax=axes[0])
axes[0].scatter(X_final_2d[:n_init_2d, 0], X_final_2d[:n_init_2d, 1],
                color="white", s=40, edgecolors="gray", linewidths=0.7,
                zorder=5, label=f"Init (n={n_init_2d})")
axes[0].scatter(X_final_2d[n_init_2d:, 0], X_final_2d[n_init_2d:, 1],
                color="red", s=60, edgecolors="black", linewidths=0.7,
                zorder=6, label="BO evaluations")
axes[0].set_title("Branin surface — training points from one BO run")
axes[0].set_xlabel("x₁ (normalised)"); axes[0].set_ylabel("x₂ (normalised)")
axes[0].legend(fontsize=8)

axes[1].plot(iters_2d, bo_2d_arr.mean(axis=0), color="steelblue", linewidth=2.5,
             label="GP-BO mean")
axes[1].fill_between(iters_2d,
                     bo_2d_arr.mean(axis=0) - bo_2d_arr.std(axis=0),
                     bo_2d_arr.mean(axis=0) + bo_2d_arr.std(axis=0),
                     alpha=0.2, color="steelblue")
axes[1].axhline(BRANIN_MIN, color="tomato", linestyle="--", linewidth=1.5,
                label=f"True min ≈ {BRANIN_MIN:.3f}")
axes[1].set_title(f"Convergence on Branin ({n_runs_2d} runs, budget={budget_2d})")
axes[1].set_xlabel("Number of evaluations"); axes[1].set_ylabel("Best f found")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle("2D GP-BO on Branin function\nBO concentrates evaluations near the three global minima",
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part5_branin_bo.png", dpi=150)
plt.close()
print("Saved: part5_branin_bo.png")


# ── Part 6: Effect of initial design size ────────────────────────────────────

total_budget = 25
n_runs_n6    = 8
n_inits      = [3, 5, 10]
colors       = ["tomato", "steelblue", "green"]

fig, ax = plt.subplots(figsize=(9, 5))

for n_init_val, color in zip(n_inits, colors):
    curves = []
    for seed in range(n_runs_n6):
        _, _, hist, _ = run_bo_1d(n_init_val, total_budget, kern_1d, seed=seed)
        # Pad or truncate to total_budget steps
        curves.append(hist[:total_budget - n_init_val + 1])

    # All curves have different lengths; align on total evaluations used
    max_len = max(len(c) for c in curves)
    padded  = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
    evals   = np.arange(max_len) + n_init_val

    ax.plot(evals, padded.mean(axis=0), color=color, linewidth=2.5,
            label=f"n_init = {n_init_val}")
    ax.fill_between(evals,
                    padded.mean(axis=0) - padded.std(axis=0),
                    padded.mean(axis=0) + padded.std(axis=0),
                    alpha=0.15, color=color)

ax.axhline(FORRESTER_MIN, color="black", linestyle="--", linewidth=1.2,
           label=f"True min = {FORRESTER_MIN:.3f}")
ax.set_title(f"Effect of initial design size on GP-BO convergence\n"
             f"Total budget = {total_budget} evaluations, {n_runs_n6} runs each")
ax.set_xlabel("Total evaluations used")
ax.set_ylabel("Best f found")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "part6_n_init_effect.png", dpi=150)
plt.close()
print("Saved: part6_n_init_effect.png")

print("\nAll done. Check lesson-8/output/ for plots.")
print("\nKey takeaways:")
print("  LHS gives better initial coverage than random — important with small budgets.")
print("  Each BO iteration: fit GP → maximise EI → evaluate f → update.")
print("  BO consistently outperforms random search on smooth low-dim functions.")
print("  The surrogate must be cheap vs f — otherwise BO overhead is not worth it.")
print("  Lesson 9: replace the GP surrogate with a neural network (PyTorch).")
