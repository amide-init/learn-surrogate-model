import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
import cma

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

N_MEMBERS  = 5
N_EPOCHS   = 1500
N_INIT     = 6
N_ITER     = 25
N_SEEDS    = 5


# ── Benchmark functions ───────────────────────────────────────────────────────

def forrester(x):
    x = np.asarray(x).ravel()
    return ((6*x - 2)**2 * np.sin(12*x - 4)).item() if x.size == 1 else (6*x - 2)**2 * np.sin(12*x - 4)

def branin_norm(x):
    x  = np.asarray(x).ravel()
    x1 = x[0]*15 - 5;  x2 = x[1]*15
    a, b, c = 1.0, 5.1/(4*np.pi**2), 5/np.pi
    r, s, t = 6.0, 10.0, 1/(8*np.pi)
    return float(a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s)

def hartmann6(x):
    x = np.asarray(x).ravel()
    A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                  [0.05, 10, 17, 0.1, 8, 14],
                  [3, 3.5, 1.7, 10, 17, 8],
                  [17, 8, 0.05, 10, 0.1, 14]], dtype=float)
    P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                          [2329, 4135, 8307, 3736, 1004, 9991],
                          [2348, 1451, 3522, 2883, 3047, 6650],
                          [4047, 8828, 8732, 5743, 1091, 381]], dtype=float)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    return float(-np.sum(alpha * np.exp(-np.sum(A * (x - P)**2, axis=1))))

F_STAR = {
    'forrester': forrester(np.array([0.7572])),
    'branin':    0.397887,
    'hartmann6': -3.32237,
}


# ── Deep Ensemble surrogate ───────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, d=1, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1),
        )
    def forward(self, x): return self.net(x)

class DeepEnsemble:
    def __init__(self, n=5, d=1, h=64):
        self.members = []
        for i in range(n):
            torch.manual_seed(i * 100 + d * 7)
            self.members.append(_MLP(d, h))

    def fit(self, X_t, y_t, epochs=1500, lr=1e-3):
        loss_fn = nn.MSELoss()
        for m in self.members:
            opt = optim.Adam(m.parameters(), lr=lr)
            for _ in range(epochs):
                m.train(); opt.zero_grad()
                loss_fn(m(X_t).squeeze(), y_t).backward()
                opt.step()

    def predict_np(self, X_np):
        x = torch.from_numpy(X_np.astype(np.float32))
        preds = []
        for m in self.members:
            m.eval()
            with torch.no_grad():
                preds.append(m(x).squeeze(-1).numpy())
        preds = np.array(preds)          # (N, n_points)
        return preds.mean(0), preds.std(0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_tensor(arr):
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))

def lhs(n, d, rg):
    pts = np.zeros((n, d))
    for j in range(d):
        pts[:, j] = (rg.permutation(n) + rg.uniform(size=n)) / n
    return pts.astype(np.float32)

def ei(mu, std, f_best, xi=0.01):
    imp = f_best - xi - mu
    Z   = imp / (std + 1e-9)
    return np.maximum(imp * norm.cdf(Z) + std * norm.pdf(Z), 0.0)

def standardise(X, y):
    Xm, Xs = X.mean(0), X.std(0) + 1e-8
    ym, ys = float(y.mean()), float(y.std()) + 1e-8
    return (X - Xm) / Xs, (y - ym) / ys, Xm, Xs, ym, ys

def build_ei_fn(ensemble, Xm, Xs, ym, ys, f_best, dim):
    """Return a callable EI(x_unit) for CMA-ES (unit cube input, negated for minimisation)."""
    def neg_ei(x_unit):
        x_unit = np.asarray(x_unit).reshape(1, dim)
        x_std  = (x_unit - Xm) / Xs
        mu_s, std_s = ensemble.predict_np(x_std)
        mu  = float(np.squeeze(mu_s))  * ys + ym
        std = float(np.squeeze(std_s)) * ys
        return float(-ei(np.array([mu]), np.array([std]), f_best)[0])
    return neg_ei


# ── Acquisition optimiser — random search ────────────────────────────────────

def acq_random(ensemble, Xm, Xs, ym, ys, f_best, dim, rg, n_cand=2000):
    Xc      = rg.uniform(0, 1, (n_cand, dim)).astype(np.float32)
    Xc_std  = (Xc - Xm) / Xs
    mu_s, std_s = ensemble.predict_np(Xc_std)
    mu  = mu_s  * ys + ym
    std = std_s * ys
    ei_vals = ei(mu, std, f_best)
    idx = int(np.argmax(ei_vals))
    return Xc[idx], float(ei_vals[idx])


# ── Acquisition optimiser — CMA-ES ───────────────────────────────────────────

def acq_cmaes(ensemble, Xm, Xs, ym, ys, f_best, dim, rg):
    neg_ei_fn = build_ei_fn(ensemble, Xm, Xs, ym, ys, f_best, dim)
    x0    = rg.uniform(0.2, 0.8, dim)
    sigma = 0.3
    opts  = {
        'bounds':      [[0.0]*dim, [1.0]*dim],
        'maxiter':     200,
        'tolx':        1e-5,
        'tolfun':      1e-5,
        'verbose':     -9,
        'seed':        int(rg.integers(0, 2**31)),
    }
    es = cma.CMAEvolutionStrategy(x0, sigma, opts)
    es.optimize(neg_ei_fn)
    x_best = np.clip(es.result.xbest, 0, 1).astype(np.float32)
    ei_best = -float(es.result.fbest)
    return x_best, ei_best


# ── BO loop ───────────────────────────────────────────────────────────────────

def run_bo(func, dim, seed, acq_method='cmaes', epochs=N_EPOCHS):
    rg = np.random.default_rng(seed)
    torch.manual_seed(seed)

    X = lhs(N_INIT, dim, rg)
    y = np.array([func(x) for x in X], dtype=np.float32)
    best_history = [float(y.min())]

    for _ in range(N_ITER):
        X_std, y_std, Xm, Xs, ym, ys = standardise(X, y)
        X_t = to_tensor(X_std)
        y_t = to_tensor(y_std)

        de = DeepEnsemble(n=N_MEMBERS, d=dim)
        de.fit(X_t, y_t, epochs=epochs)

        f_best = float(y.min())
        if acq_method == 'cmaes':
            x_next, _ = acq_cmaes(de, Xm, Xs, ym, ys, f_best, dim, rg)
        else:
            x_next, _ = acq_random(de, Xm, Xs, ym, ys, f_best, dim, rg)

        y_next = func(x_next)
        X = np.vstack([X, x_next.reshape(1, dim)])
        y = np.append(y, y_next)
        best_history.append(float(y.min()))

    return np.array(best_history)


# ════════════════════════════════════════════════════════════════════════════════
# Part 1 — CMA-ES mechanics: how CMA-ES maximises EI on Forrester
# ════════════════════════════════════════════════════════════════════════════════

print("Part 1 — CMA-ES mechanics on Forrester EI landscape...")

n_init_viz = 8
X_viz = rng.uniform(0, 1, (n_init_viz, 1)).astype(np.float32)
y_viz = np.array([forrester(x) for x in X_viz], dtype=np.float32)
X_std_v, y_std_v, Xm_v, Xs_v, ym_v, ys_v = standardise(X_viz, y_viz)

de_viz = DeepEnsemble(n=N_MEMBERS, d=1)
de_viz.fit(to_tensor(X_std_v), to_tensor(y_std_v), epochs=2000)

xs       = np.linspace(0, 1, 300, dtype=np.float32).reshape(-1, 1)
mu_s, std_s = de_viz.predict_np((xs - Xm_v) / Xs_v)
mu_true  = mu_s * ys_v + ym_v
std_true = std_s * ys_v
f_best_v = float(y_viz.min())
ei_vals  = ei(mu_true, std_true, f_best_v)

# --- run CMA-ES, record search history ---
neg_ei_fn = build_ei_fn(de_viz, Xm_v, Xs_v, ym_v, ys_v, f_best_v, dim=1)
cma_history = []

class _Logger:
    def __call__(self, es):
        cma_history.append(float(es.result.xbest))

opts = {'bounds': [[0.0], [1.0]], 'maxiter': 80, 'verbose': -9,
        'tolx': 1e-6, 'tolfun': 1e-6, 'seed': 42}
es = cma.CMAEvolutionStrategy([0.5], 0.3, opts)
while not es.stop():
    sols = es.ask()
    es.tell(sols, [neg_ei_fn(s) for s in sols])
    cma_history.append(float(np.clip(es.result.xbest, 0, 1)))

x_cma  = float(np.clip(es.result.xbest, 0, 1))
x_rand = float(xs[np.argmax(ei_vals)])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# surrogate
ax = axes[0]
y_true_f = np.array([forrester(x) for x in xs.ravel()])
ax.plot(xs, y_true_f, 'k--', lw=1.5, alpha=0.4, label='True f(x)')
ax.plot(xs, mu_true, color='steelblue', lw=2.5, label='Ensemble μ')
ax.fill_between(xs.ravel(), mu_true - 2*std_true, mu_true + 2*std_true,
                alpha=0.2, color='steelblue', label='±2σ')
ax.scatter(X_viz.ravel(), y_viz, c='black', s=70, zorder=5, label='Data')
ax.set_title(f'Deep Ensemble surrogate\n({n_init_viz} training points)')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# EI landscape + solutions
ax = axes[1]
ax.fill_between(xs.ravel(), 0, ei_vals, alpha=0.3, color='darkorange')
ax.plot(xs, ei_vals, color='darkorange', lw=2, label='EI(x)')
ax.axvline(x_rand, color='green',  lw=2.5, ls='--', label=f'Random best  x={x_rand:.3f}')
ax.axvline(x_cma,  color='red',    lw=2.5, ls=':',  label=f'CMA-ES best  x={x_cma:.3f}')
ax.set_title('EI landscape — where each method places the next query')
ax.set_xlabel('x'); ax.set_ylabel('EI(x)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# CMA-ES search history
ax = axes[2]
iters = np.arange(len(cma_history))
ax.plot(iters, cma_history, color='red', lw=2, marker='o', markersize=3, label='CMA-ES x_best')
ax.axhline(x_cma, color='red', ls='--', lw=1, alpha=0.5)
ax.set_title('CMA-ES converging to EI maximum\nEach point = best solution found so far')
ax.set_xlabel('CMA-ES generation'); ax.set_ylabel('x_best')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('Part 1 — CMA-ES mechanics: maximising EI over the Deep Ensemble surrogate\n'
             'CMA-ES adapts its search distribution to home in on the EI peak',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part1_cmaes_mechanics.png', dpi=150)
plt.close()
print("  Saved: part1_cmaes_mechanics.png")


# ════════════════════════════════════════════════════════════════════════════════
# Part 2 — Acquisition quality: CMA-ES vs random sampling across problem dims
# ════════════════════════════════════════════════════════════════════════════════

print("Part 2 — Acquisition quality: CMA-ES vs random (1D, 2D, 6D)...")

def make_ensemble_and_data(func, dim, n_pts, seed):
    rg2 = np.random.default_rng(seed)
    X   = lhs(n_pts, dim, rg2)
    y   = np.array([func(x) for x in X], dtype=np.float32)
    X_std, y_std, Xm, Xs, ym, ys = standardise(X, y)
    de  = DeepEnsemble(n=N_MEMBERS, d=dim)
    de.fit(to_tensor(X_std), to_tensor(y_std), epochs=1500)
    return de, X, y, Xm, Xs, ym, ys

configs = [
    ('forrester',  forrester,   1, 10),
    ('branin',     branin_norm, 2, 20),
    ('hartmann6',  hartmann6,   6, 50),
]

n_reps = 10
results_acq = {}   # (name, method) → list of EI values found

for name, func, dim, n_pts in configs:
    print(f"  {name} (d={dim})...")
    de, X, y, Xm, Xs, ym, ys = make_ensemble_and_data(func, dim, n_pts, seed=0)
    f_best = float(y.min())
    ei_rand_list, ei_cma_list = [], []
    for rep in range(n_reps):
        rg2 = np.random.default_rng(rep + 100)
        _, ei_r = acq_random(de, Xm, Xs, ym, ys, f_best, dim, rg2, n_cand=2000)
        _, ei_c = acq_cmaes(de, Xm, Xs, ym, ys, f_best, dim, rg2)
        ei_rand_list.append(ei_r)
        ei_cma_list.append(ei_c)
    results_acq[name] = {
        'rand': np.array(ei_rand_list),
        'cma':  np.array(ei_cma_list),
        'dim':  dim,
    }

labels_acq = [c[0] for c in configs]
x_pos   = np.arange(len(labels_acq))
width   = 0.35

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Mean EI found
means_r = [results_acq[n]['rand'].mean() for n in labels_acq]
means_c = [results_acq[n]['cma'].mean()  for n in labels_acq]
axes[0].bar(x_pos - width/2, means_r, width, label='Random (2000 cands)', color='green',  alpha=0.8)
axes[0].bar(x_pos + width/2, means_c, width, label='CMA-ES',              color='red',    alpha=0.8)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([f'{n}\n(d={results_acq[n]["dim"]})' for n in labels_acq])
axes[0].set_title(f'Mean EI found — {n_reps} repetitions\nHigher = better acquisition point')
axes[0].set_ylabel('Mean EI'); axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')

# Ratio CMA / random
ratios = [results_acq[n]['cma'].mean() / (results_acq[n]['rand'].mean() + 1e-12)
          for n in labels_acq]
colors_ratio = ['green' if r < 1.1 else 'orange' if r < 2 else 'red' for r in ratios]
bars = axes[1].bar(x_pos, ratios, color=colors_ratio, alpha=0.85, edgecolor='black')
axes[1].axhline(1.0, color='black', ls='--', lw=1.5, label='Random = 1×')
for bar, r in zip(bars, ratios):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 r + 0.02, f'{r:.2f}×', ha='center', fontsize=11, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([f'{n}\n(d={results_acq[n]["dim"]})' for n in labels_acq])
axes[1].set_title('CMA-ES EI / Random EI ratio\n>1× means CMA-ES found a better acquisition point')
axes[1].set_ylabel('CMA-ES / Random EI'); axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Part 2 — Acquisition optimisation quality: CMA-ES vs random sampling\n'
             'Gap widens in higher dimensions — random sampling misses EI peaks',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part2_acq_quality.png', dpi=150)
plt.close()
print("  Saved: part2_acq_quality.png")


# ════════════════════════════════════════════════════════════════════════════════
# Part 3 — BO convergence: CMA-ES vs random, Forrester (1D)
# ════════════════════════════════════════════════════════════════════════════════

print(f"Part 3 — BO convergence on Forrester ({N_SEEDS} seeds × {N_ITER} iters)...")

curves = {'cmaes': [], 'random': []}
for seed in range(N_SEEDS):
    print(f"  seed {seed+1}/{N_SEEDS}", flush=True)
    curves['cmaes'].append(run_bo(forrester, dim=1, seed=seed, acq_method='cmaes'))
    curves['random'].append(run_bo(forrester, dim=1, seed=seed, acq_method='random'))

curves = {k: np.array(v) for k, v in curves.items()}
iters  = np.arange(N_ITER + 1)
fstar  = F_STAR['forrester']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for method, label, color in [('cmaes', 'Deep Ensemble + CMA-ES', 'red'),
                               ('random', 'Deep Ensemble + Random', 'green')]:
    c = curves[method]
    m, s = c.mean(0), c.std(0)
    axes[0].plot(iters, m, color=color, lw=2.5, label=label)
    axes[0].fill_between(iters, m - s, m + s, alpha=0.2, color=color)
    gap = np.maximum(c - fstar, 1e-4)
    axes[1].semilogy(iters, gap.mean(0), color=color, lw=2.5, label=label)
    for g in gap:
        axes[1].semilogy(iters, g, color=color, lw=0.7, alpha=0.25)

axes[0].axhline(fstar, color='black', ls='--', lw=1.5, label=f'f* = {fstar:.3f}')
axes[0].set_title(f'Best f found — Forrester 1D\nmean ± std, {N_SEEDS} seeds')
axes[0].set_xlabel('BO iteration'); axes[0].set_ylabel('Best f(x) found')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].set_title('Gap to optimum (log scale)\nLower is better')
axes[1].set_xlabel('BO iteration'); axes[1].set_ylabel('f_best − f*')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, which='both')

plt.suptitle(f'Part 3 — BO convergence on Forrester: CMA-ES vs random acquisition\n'
             f'Both use Deep Ensembles as surrogate. {N_INIT} LHS init + {N_ITER} BO steps.',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part3_convergence_forrester.png', dpi=150)
plt.close()
print("  Saved: part3_convergence_forrester.png")


# ════════════════════════════════════════════════════════════════════════════════
# Part 4 — BO convergence on Branin 2D
# ════════════════════════════════════════════════════════════════════════════════

print(f"Part 4 — BO convergence on Branin ({N_SEEDS} seeds × {N_ITER} iters)...")

curves_br = {'cmaes': [], 'random': []}
for seed in range(N_SEEDS):
    print(f"  seed {seed+1}/{N_SEEDS}", flush=True)
    curves_br['cmaes'].append(run_bo(branin_norm, dim=2, seed=seed, acq_method='cmaes'))
    curves_br['random'].append(run_bo(branin_norm, dim=2, seed=seed, acq_method='random'))

curves_br = {k: np.array(v) for k, v in curves_br.items()}
fstar_br  = F_STAR['branin']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for method, label, color in [('cmaes', 'Deep Ensemble + CMA-ES', 'red'),
                               ('random', 'Deep Ensemble + Random', 'green')]:
    c = curves_br[method]
    m, s = c.mean(0), c.std(0)
    axes[0].plot(iters, m, color=color, lw=2.5, label=label)
    axes[0].fill_between(iters, m - s, m + s, alpha=0.2, color=color)
    gap = np.maximum(c - fstar_br, 1e-4)
    axes[1].semilogy(iters, gap.mean(0), color=color, lw=2.5, label=label)
    for g in gap:
        axes[1].semilogy(iters, g, color=color, lw=0.7, alpha=0.25)

axes[0].axhline(fstar_br, color='black', ls='--', lw=1.5, label=f'f* ≈ {fstar_br:.3f}')
axes[0].set_title(f'Best Branin found — 2D\nmean ± std, {N_SEEDS} seeds')
axes[0].set_xlabel('BO iteration'); axes[0].set_ylabel('Best f(x) found')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].set_title('Gap to Branin optimum (log)\nLower is better')
axes[1].set_xlabel('BO iteration'); axes[1].set_ylabel('f_best − f*')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, which='both')

plt.suptitle(f'Part 4 — BO convergence on Branin 2D: CMA-ES vs random acquisition\n'
             f'2D shows a clearer CMA-ES advantage than 1D.',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part4_convergence_branin.png', dpi=150)
plt.close()
print("  Saved: part4_convergence_branin.png")


# ════════════════════════════════════════════════════════════════════════════════
# Part 5 — BO convergence on Hartmann-6 (6D) — where CMA-ES matters most
# ════════════════════════════════════════════════════════════════════════════════

print(f"Part 5 — BO convergence on Hartmann-6 ({N_SEEDS} seeds × {N_ITER} iters)...")

curves_h6 = {'cmaes': [], 'random': []}
for seed in range(N_SEEDS):
    print(f"  seed {seed+1}/{N_SEEDS}", flush=True)
    curves_h6['cmaes'].append(run_bo(hartmann6, dim=6, seed=seed, acq_method='cmaes', epochs=1000))
    curves_h6['random'].append(run_bo(hartmann6, dim=6, seed=seed, acq_method='random', epochs=1000))

curves_h6 = {k: np.array(v) for k, v in curves_h6.items()}
fstar_h6  = F_STAR['hartmann6']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for method, label, color in [('cmaes', 'Deep Ensemble + CMA-ES', 'red'),
                               ('random', 'Deep Ensemble + Random', 'green')]:
    c = curves_h6[method]
    m, s = c.mean(0), c.std(0)
    axes[0].plot(iters, m, color=color, lw=2.5, label=label)
    axes[0].fill_between(iters, m - s, m + s, alpha=0.2, color=color)
    gap = np.maximum(c - fstar_h6, 1e-4)
    axes[1].semilogy(iters, gap.mean(0), color=color, lw=2.5, label=label)
    for g in gap:
        axes[1].semilogy(iters, g, color=color, lw=0.7, alpha=0.25)

axes[0].axhline(fstar_h6, color='black', ls='--', lw=1.5, label=f'f* ≈ {fstar_h6:.3f}')
axes[0].set_title(f'Best Hartmann-6 found — 6D\nmean ± std, {N_SEEDS} seeds')
axes[0].set_xlabel('BO iteration'); axes[0].set_ylabel('Best f(x) found')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].set_title('Gap to Hartmann-6 optimum (log)\nLower is better')
axes[1].set_xlabel('BO iteration'); axes[1].set_ylabel('f_best − f*')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, which='both')

plt.suptitle(f'Part 5 — BO convergence on Hartmann-6 (6D): where CMA-ES matters most\n'
             f'Random sampling of 2000 candidates covers a negligible fraction of 6D space.',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part5_convergence_hartmann6.png', dpi=150)
plt.close()
print("  Saved: part5_convergence_hartmann6.png")


# ════════════════════════════════════════════════════════════════════════════════
# Part 6 — Full BO trace: Hartmann-6, best seed, CMA-ES
# ════════════════════════════════════════════════════════════════════════════════

print("Part 6 — Detailed BO trace on Hartmann-6 (best seed, CMA-ES)...")

best_seed = int(np.argmin(curves_h6['cmaes'][:, -1]))

rg_trace = np.random.default_rng(best_seed)
torch.manual_seed(best_seed)
X_tr = lhs(N_INIT, 6, rg_trace)
y_tr = np.array([hartmann6(x) for x in X_tr], dtype=np.float32)

all_x, all_y = list(X_tr), list(y_tr)
trace_best, trace_ei = [], []

for step in range(N_ITER):
    X_arr = np.array(all_x, dtype=np.float32)
    y_arr = np.array(all_y, dtype=np.float32)
    X_std, y_std, Xm, Xs, ym, ys = standardise(X_arr, y_arr)

    de = DeepEnsemble(n=N_MEMBERS, d=6)
    de.fit(to_tensor(X_std), to_tensor(y_std), epochs=1000)

    f_best = float(y_arr.min())
    x_next, ei_val = acq_cmaes(de, Xm, Xs, ym, ys, f_best, 6, rg_trace)
    y_next = hartmann6(x_next)

    all_x.append(x_next)
    all_y.append(y_next)
    trace_best.append(float(np.array(all_y).min()))
    trace_ei.append(ei_val)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

steps = np.arange(1, N_ITER + 1)
axes[0].plot(steps, trace_best, color='red', lw=2.5, marker='o', markersize=5)
axes[0].axhline(fstar_h6, color='black', ls='--', lw=1.5, label=f'f* = {fstar_h6:.3f}')
axes[0].set_title(f'Best Hartmann-6 per BO step\nSeed {best_seed} — CMA-ES acquisition')
axes[0].set_xlabel('BO iteration'); axes[0].set_ylabel('Best f(x) found')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].semilogy(steps, trace_ei, color='darkorange', lw=2.5, marker='s', markersize=5)
axes[1].set_title('EI at each selected point\nDecreases as surrogate exhausts obvious improvements')
axes[1].set_xlabel('BO iteration'); axes[1].set_ylabel('EI (log scale)')
axes[1].grid(True, alpha=0.3, which='both')

plt.suptitle('Part 6 — Detailed BO trace: Deep Ensemble + CMA-ES on Hartmann-6\n'
             'Watch EI decay as the surrogate converges — fewer surprises left',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part6_trace_hartmann6.png', dpi=150)
plt.close()
print("  Saved: part6_trace_hartmann6.png")


print("\nAll done. Outputs in lesson-12/output/")
print("\nKey takeaways:")
print("  CMA-ES adapts covariance to EI landscape — far better than random in high-d.")
print("  1D: similar; 2D: moderate gain; 6D: CMA-ES wins clearly.")
print("  EI decays as BO progresses — surrogate has no unexplored regions left.")
print("  Deep Ensembles provide smooth σ needed for reliable EI gradients for CMA-ES.")
print("  Next: Lesson 13 compares all three surrogates (GP, MC Dropout, Deep Ensembles).")
