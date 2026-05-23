import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm, spearmanr
from scipy.optimize import minimize

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Prescreening settings
LAMBDA   = 20    # candidates generated per generation
N_INIT   = 10    # initial LHS true evaluations
N_ITER   = 40    # generations per run
N_SEEDS  = 5     # seeds for sensitivity analysis


# ── Benchmark ──────────────────────────────────────────────────────────────────

def forrester(x):
    x = np.asarray(x).ravel()
    return (6*x - 2)**2 * np.sin(12*x - 4)

F_STAR = forrester(np.array([0.7572])).item()   # ≈ -6.021


# ── Helpers ────────────────────────────────────────────────────────────────────

def lhs(n, d, rg):
    pts = np.zeros((n, d))
    for j in range(d):
        pts[:, j] = (rg.permutation(n) + rg.uniform(size=n)) / n
    return pts.astype(np.float32)

def acq_ei(mu, std, f_best, xi=0.01):
    Z = (f_best - xi - mu) / (std + 1e-9)
    return np.maximum((f_best - xi - mu)*norm.cdf(Z) + std*norm.pdf(Z), 0.0)

def eval_single(x):
    r = forrester(x.ravel())
    return r.item() if isinstance(r, np.ndarray) else float(r)


# ── GP surrogate ───────────────────────────────────────────────────────────────

def matern52(X1, X2, l=1.0, sf=1.0):
    diff = X1[:, None, :] - X2[None, :, :]
    r    = np.sqrt((diff**2).sum(-1) + 1e-12)
    k    = np.sqrt(5) * r / l
    return sf**2 * (1 + k + k**2/3) * np.exp(-k)

def gp_predict(X_tr, y_tr, X_te, l=1.0, sf=1.0, sn=0.01):
    K = matern52(X_tr, X_tr, l, sf) + (sn**2 + 1e-6)*np.eye(len(X_tr))
    L = np.linalg.cholesky(K)
    Ks  = matern52(X_tr, X_te, l, sf)
    Kss = np.diag(matern52(X_te, X_te, l, sf))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
    mu    = Ks.T @ alpha
    v     = np.linalg.solve(L, Ks)
    std   = np.sqrt(np.maximum(Kss - (v**2).sum(0), 0.0))
    return mu, std

def fit_gp(X_tr, y_tr):
    def nll(lp):
        l, sf, sn = np.exp(lp)
        K = matern52(X_tr, X_tr, l, sf) + (sn**2 + 1e-6)*np.eye(len(X_tr))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
        return float(0.5*(y_tr @ a) + np.log(np.diag(L)).sum())
    res = minimize(nll, [0., 0., -3.], method='L-BFGS-B',
                   bounds=[(-3,3), (-2,2), (-5,0)])
    return tuple(np.exp(res.x))


# ── Prescreening BO loop ───────────────────────────────────────────────────────

def run_prescreening_bo(seed, scr_target=0.10, lambda_=LAMBDA, n_iter=N_ITER):
    """
    Generation-based BO with prescreening.

    Each generation:
      1. Generate lambda_ candidate points (the pool)
      2. Evaluate ALL lambda_ with the surrogate  → surrogate_calls += lambda_
      3. Rank by EI; select top k = max(1, ceil(scr_target * lambda_))
      4. Evaluate the top k with the TRUE function → true_calls += k
      5. Add ONLY those k truly-evaluated points to the training dataset

    SCR = true_calls / (true_calls + surrogate_calls)   (target < 20%)

    Only truly-evaluated points enter the dataset — no surrogate drift.
    """
    rg   = np.random.default_rng(seed)
    X    = lhs(N_INIT, 1, rg)
    y    = np.array([eval_single(x) for x in X])

    true_calls = N_INIT
    surr_calls = 0
    best_true  = float(y.min())

    best_history = [best_true]
    true_history = [true_calls]
    scr_history  = [true_calls / (true_calls + 1)]  # initial SCR ≈ 1 (all LHS are true)
    k_per_gen    = []

    k = max(1, int(np.ceil(scr_target * lambda_)))   # true evals per generation

    for _ in range(n_iter):
        # Standardise on current true dataset
        Xm, Xs = float(X.mean()), float(X.std()) + 1e-8
        ym, ys = float(y.mean()), float(y.std()) + 1e-8
        X_std  = (X - Xm) / Xs
        y_std  = (y - ym) / ys

        l, sf, sn = fit_gp(X_std, y_std)

        # Step 1–2: generate pool and evaluate ALL with surrogate
        pool     = rg.uniform(0, 1, (lambda_, 1)).astype(np.float32)
        pool_std = (pool - Xm) / Xs
        mu_s, std_s = gp_predict(X_std, y_std, pool_std, l, sf, sn)
        mu  = mu_s * ys + ym
        std = std_s * ys
        surr_calls += lambda_

        # Step 3: rank by EI, select top k
        ei      = acq_ei(mu, std, float(y.min()))
        top_idx = np.argsort(ei)[-k:][::-1]   # top k by EI (descending)

        # Step 4–5: truly evaluate top k and add to dataset
        for idx in top_idx:
            x_next = pool[idx]
            y_next = eval_single(x_next)
            true_calls += 1
            best_true = min(best_true, y_next)
            X = np.vstack([X, x_next])
            y = np.append(y, y_next)

        k_per_gen.append(k)
        best_history.append(best_true)
        true_history.append(true_calls)
        scr_history.append(true_calls / (true_calls + surr_calls))

    return (np.array(best_history), np.array(true_history),
            np.array(scr_history), k_per_gen)


# ════════════════════════════════════════════════════════════════════════════════
# Part 1 — The prescreening model
# ════════════════════════════════════════════════════════════════════════════════

def draw_prescreening_panel(ax, scr, lambda_=LAMBDA, title=''):
    k = max(1, int(np.ceil(scr * lambda_)))
    # Simulate surrogate ranking with some noise
    rng2 = np.random.default_rng(0)
    true_vals = rng2.uniform(-6, 5, lambda_)
    noise     = rng2.normal(0, 1.0, lambda_)
    surr_vals = true_vals + noise
    ranked    = np.argsort(surr_vals)        # ascending = best first (minimisation)
    colors    = ['seagreen' if i in ranked[:k] else '#cccccc' for i in range(lambda_)]
    xs        = np.arange(1, lambda_ + 1)
    ax.bar(xs, -surr_vals + surr_vals.max() + 1, color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(ranked[k-1] + 1, color='red', lw=2, ls='--', alpha=0.7, label=f'cutoff (top {k})')
    scr_real = k / (k + lambda_)
    ax.set_title(f'{title}\n'
                 f'λ={lambda_}, k={k} true evals\nSCR={scr_real*100:.0f}%  '
                 f'{"✓ acceptable" if scr_real < 0.20 else "✗ TOO HIGH"}',
                 fontsize=9)
    ax.set_xlabel('Candidate (sorted by pool index)')
    ax.set_yticks([])
    ax.set_xlim(0, lambda_ + 1)
    if scr_real >= 0.20:
        ax.set_facecolor('#fff0f0')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
from matplotlib.patches import Patch

for ax, scr, title in zip(axes,
    [0.10, 0.20, 0.50],
    ['SCR ≈ 10%\n(target)', 'SCR = 20%\n(ceiling)', 'SCR = 50%\n(professor\'s complaint)']):
    draw_prescreening_panel(ax, scr, title=title)

axes[0].legend(handles=[Patch(color='seagreen', label='True function call'),
                         Patch(color='#cccccc', label='Surrogate-only (discarded)')],
               fontsize=8)

plt.suptitle('Prescreening: generate λ candidates → evaluate ALL with surrogate → '
             'select top k for true evaluation\n'
             'SCR = true_calls / (true_calls + surrogate_calls)   must be < 20%',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part1_prescreening_concept.png', dpi=150)
plt.close()
print("Saved: part1_prescreening_concept.png")


# ════════════════════════════════════════════════════════════════════════════════
# Part 2 — Surrogate rank correlation
# ════════════════════════════════════════════════════════════════════════════════

n_tr    = 15
X_tr_np = rng.uniform(0, 1, (n_tr, 1)).astype(np.float32)
y_tr_np = forrester(X_tr_np).astype(np.float32)
Xm_t    = float(X_tr_np.mean()); Xs_t = float(X_tr_np.std()) + 1e-8
ym_t    = float(y_tr_np.mean()); ys_t = float(y_tr_np.std()) + 1e-8
X_tr_s  = (X_tr_np - Xm_t) / Xs_t
y_tr_s  = (y_tr_np - ym_t) / ys_t
l_t, sf_t, sn_t = fit_gp(X_tr_s, y_tr_s)

n_pool   = 50
X_pool   = rng.uniform(0, 1, (n_pool, 1)).astype(np.float32)
y_true_p = forrester(X_pool.ravel())
Xp_s     = (X_pool - Xm_t) / Xs_t
mu_p_s, _ = gp_predict(X_tr_s, y_tr_s, Xp_s, l_t, sf_t, sn_t)
mu_p     = mu_p_s * ys_t + ym_t

rho, pval    = spearmanr(y_true_p, mu_p)
true_ranks   = y_true_p.argsort().argsort() + 1
surr_ranks   = mu_p.argsort().argsort() + 1

# What fraction of the true top-10 does the surrogate find if k=5 or k=10?
k_vals   = [2, 5, 10]
true_top = set(np.argsort(y_true_p)[:10])
surr_sorted = np.argsort(mu_p)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_true_p, mu_p, color='steelblue', s=60, alpha=0.8, zorder=3)
lims = [min(y_true_p.min(), mu_p.min())-0.5, max(y_true_p.max(), mu_p.max())+0.5]
axes[0].plot(lims, lims, 'k--', lw=1.5, alpha=0.5, label='Perfect prediction')
axes[0].set_title(f'Surrogate μ vs. true f(x)\nSpearman ρ = {rho:.3f}  (p = {pval:.2e})')
axes[0].set_xlabel('True f(x)'); axes[0].set_ylabel('Surrogate μ(x)')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

axes[1].scatter(true_ranks, surr_ranks, color='tomato', s=60, alpha=0.8, zorder=3)
axes[1].plot([1, n_pool], [1, n_pool], 'k--', lw=1.5, alpha=0.5, label='Perfect ranking')
for k_show in k_vals:
    selected = set(surr_sorted[:k_show])
    hit = len(selected & true_top)
    axes[1].axvline(k_show, color='gray', lw=0.8, ls=':', alpha=0.6)
    axes[1].text(k_show+0.3, n_pool-3, f'k={k_show}: {hit}/10 hits',
                 fontsize=7, color='gray')
axes[1].set_title(f'Rank comparison — {n_pool} pool candidates\n'
                  f'ρ = {rho:.3f}  →  prescreening is '
                  f'{"trustworthy" if rho > 0.7 else "UNRELIABLE"}')
axes[1].set_xlabel('True rank (1 = best)'); axes[1].set_ylabel('Surrogate rank')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

plt.suptitle('Rank correlation: can the surrogate correctly rank candidates?\n'
             'High ρ → low SCR is safe; low ρ → the surrogate may discard the true best',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part2_rank_correlation.png', dpi=150)
plt.close()
print(f"Saved: part2_rank_correlation.png  (Spearman ρ = {rho:.3f})")


# ════════════════════════════════════════════════════════════════════════════════
# Part 3 — Live SCR tracker (single run, SCR target = 10%)
# ════════════════════════════════════════════════════════════════════════════════

best3, true3, scr3, k3 = run_prescreening_bo(seed=RANDOM_SEED, scr_target=0.10)
gens = np.arange(len(scr3))

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# True evaluations added per generation
k3_arr = np.array([0] + k3)
axes[0].bar(gens[1:], k3, color='seagreen', edgecolor='white', linewidth=0.5, label='True evals this gen')
axes[0].bar(gens[1:], [LAMBDA]*N_ITER, bottom=k3, color='#d0d0d0',
            edgecolor='white', linewidth=0.5, label='Surrogate-only (pool)')
axes[0].set_title(f'Evaluations per generation: {LAMBDA} total, {k3[0]} truly evaluated\n'
                  f'Surrogate handles {LAMBDA - k3[0]}/{LAMBDA} = '
                  f'{(LAMBDA - k3[0])/LAMBDA*100:.0f}% of each generation')
axes[0].set_ylabel('Evaluations')
axes[0].legend(fontsize=8)

# Cumulative true calls
axes[1].plot(gens, true3, color='steelblue', lw=2.5, label='Cumulative true calls')
axes[1].fill_between(gens, 0, true3, alpha=0.2, color='steelblue')
axes[1].set_title('Cumulative true function calls (COCO budget counter)')
axes[1].set_ylabel('True calls'); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

# Running SCR
axes[2].plot(gens, scr3 * 100, color='tomato', lw=2.5, label='Running SCR (%)')
axes[2].axhline(20, color='black', ls='--', lw=1.5, label='20% ceiling')
axes[2].fill_between(gens, 0, scr3 * 100, alpha=0.2, color='tomato')
axes[2].set_ylabel('SCR (%)'); axes[2].set_xlabel('Generation')
axes[2].set_title('Running SCR converges well below 20% ceiling')
axes[2].set_ylim(0, 60); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

plt.suptitle(f'Prescreening BO with SCR target = 10%\n'
             f'λ = {LAMBDA} pool, k = {k3[0]} true evals/gen  →  final SCR = {scr3[-1]*100:.1f}%',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part3_scr_tracker.png', dpi=150)
plt.close()
print(f"Saved: part3_scr_tracker.png  (final SCR = {scr3[-1]*100:.1f}%)")


# ════════════════════════════════════════════════════════════════════════════════
# Part 4 — SCR sensitivity: convergence per TRUE evaluation
# ════════════════════════════════════════════════════════════════════════════════

print("Running SCR sensitivity analysis...")
# Note: SCR=1.0 means k=lambda_ → every candidate truly evaluated (no surrogate saving)
scr_settings = [0.05, 0.10, 0.20, 0.50, 1.00]
scr_names    = ['SCR 5%', 'SCR 10%', 'SCR 20%', 'SCR 50% (bad)', 'SCR 100% (no surrogate)']
colors5      = ['steelblue', 'seagreen', 'darkorange', 'tomato', 'purple']

results5 = {}
for scr in scr_settings:
    k = max(1, int(np.ceil(scr * LAMBDA)))
    runs = []
    for seed in range(N_SEEDS):
        best_h, true_h, _, _ = run_prescreening_bo(seed=seed, scr_target=scr)
        runs.append((best_h, true_h))
    results5[scr] = runs
    print(f"  SCR={scr*100:.0f}%  k={k}  done")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: best true f vs. generation
for scr, label, color in zip(scr_settings, scr_names, colors5):
    bests = np.array([r[0] for r in results5[scr]])
    m = bests.mean(0)
    ls = '--' if scr >= 0.50 else '-'
    axes[0].plot(range(N_ITER+1), m, color=color, lw=2.5, ls=ls, label=label)
axes[0].axhline(F_STAR, color='black', ls=':', lw=1.5, label=f'f* ≈ {F_STAR:.3f}')
axes[0].set_title(f'Best true f found vs. generation\nmean over {N_SEEDS} seeds')
axes[0].set_xlabel('Generation'); axes[0].set_ylabel('Best f(x) found')
axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

# Right: best true f vs. TRUE evaluations (the COCO-correct comparison)
for scr, label, color in zip(scr_settings, scr_names, colors5):
    ls = '--' if scr >= 0.50 else '-'
    for i, (best_h, true_h) in enumerate(results5[scr]):
        axes[1].step(true_h, best_h, color=color, lw=1.5, alpha=0.4, where='post',
                     label=label if i == 0 else None, ls=ls)
axes[1].axhline(F_STAR, color='black', ls=':', lw=1.5, label=f'f* ≈ {F_STAR:.3f}')
axes[1].set_title('Convergence per TRUE evaluation (COCO-correct x-axis)\n'
                  'SCR 50–100% wastes true-eval budget; SCR 10–20% is efficient')
axes[1].set_xlabel('True function calls (COCO budget)'); axes[1].set_ylabel('Best true f(x)')
axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

# Annotate the "bad zone"
axes[1].axvspan(axes[1].get_xlim()[0], axes[1].get_xlim()[1], alpha=0)  # force lim first
plt.suptitle('SCR sensitivity on Forrester — lower SCR = more surrogate saves per true eval\n'
             'SCR ≥ 50% is the professor\'s complaint: too many expensive evaluations',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part4_scr_sensitivity.png', dpi=150)
plt.close()
print("Saved: part4_scr_sensitivity.png")


# ════════════════════════════════════════════════════════════════════════════════
# Part 5 — Budget breakdown after N_ITER generations
# ════════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

true_counts = []
surr_counts = []
realized_scrs = []

for scr in scr_settings:
    k = max(1, int(np.ceil(scr * LAMBDA)))
    tc = N_INIT + k * N_ITER      # true calls: N_INIT LHS + k per generation
    sc = LAMBDA * N_ITER           # surrogate calls: lambda_ per generation
    true_counts.append(tc)
    surr_counts.append(sc)
    realized_scrs.append(tc / (tc + sc) * 100)

x5 = np.arange(len(scr_settings))
axes[0].bar(x5, true_counts, color='seagreen', edgecolor='black', alpha=0.85, label='True f calls')
axes[0].bar(x5, surr_counts, bottom=true_counts, color='#d0d0d0',
            edgecolor='black', alpha=0.85, label=f'Surrogate calls (λ×{N_ITER}={LAMBDA*N_ITER})')
axes[0].set_xticks(x5); axes[0].set_xticklabels(scr_names, fontsize=8)
axes[0].set_title(f'Budget after {N_ITER} generations\nSurrogate pool always = {LAMBDA*N_ITER} calls')
axes[0].set_ylabel('Total evaluations'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3, axis='y')
for i, (t, s) in enumerate(zip(true_counts, surr_counts)):
    axes[0].text(i, t/2, str(t), ha='center', va='center', fontsize=9, fontweight='bold', color='white')

bar_colors = ['seagreen' if r < 20 else ('darkorange' if r < 40 else 'tomato') for r in realized_scrs]
bars = axes[1].bar(x5, realized_scrs, color=bar_colors, edgecolor='black', alpha=0.9)
axes[1].axhline(20, color='black', ls='--', lw=2, label='20% ceiling')
axes[1].axhspan(20, 110, alpha=0.07, color='red', label='Unacceptable zone')
axes[1].set_xticks(x5); axes[1].set_xticklabels(scr_names, fontsize=8)
axes[1].set_title('Realized SCR — anything ≥ 20% is unacceptable\n'
                  'Professor\'s point: >50% is far too high')
axes[1].set_ylabel('Realized SCR (%)'); axes[1].set_ylim(0, 110)
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, realized_scrs):
    axes[1].text(bar.get_x()+bar.get_width()/2, v+1.5, f'{v:.1f}%',
                 ha='center', fontsize=9, fontweight='bold')

plt.suptitle('Budget allocation: the lower the SCR, the more the surrogate saves\n'
             f'All runs use the same surrogate pool ({LAMBDA*N_ITER} calls); only true calls change',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part5_budget_breakdown.png', dpi=150)
plt.close()
print("Saved: part5_budget_breakdown.png")


# ════════════════════════════════════════════════════════════════════════════════
# Part 6 — COCO-compatible prescreening BO (true-eval budget, SCR ≤ 20%)
# ════════════════════════════════════════════════════════════════════════════════

def run_coco_prescreening(seed, true_budget=50, scr_target=0.10, lambda_=LAMBDA):
    """
    Prescreening BO with a hard TRUE-EVALUATION budget.
    Stops when true_budget true function calls have been made.
    SCR target enforced by k = ceil(scr_target * lambda_).
    """
    rg = np.random.default_rng(seed)
    X  = lhs(N_INIT, 1, rg)
    y  = np.array([eval_single(x) for x in X])

    true_calls = N_INIT
    surr_calls = 0
    best_true  = float(y.min())

    true_axis = [true_calls]
    best_axis = [best_true]

    k = max(1, int(np.ceil(scr_target * lambda_)))

    while true_calls < true_budget:
        Xm, Xs = float(X.mean()), float(X.std()) + 1e-8
        ym, ys = float(y.mean()), float(y.std()) + 1e-8
        X_std  = (X - Xm) / Xs
        y_std  = (y - ym) / ys
        l, sf, sn = fit_gp(X_std, y_std)

        pool     = rg.uniform(0, 1, (lambda_, 1)).astype(np.float32)
        pool_std = (pool - Xm) / Xs
        mu_s, std_s = gp_predict(X_std, y_std, pool_std, l, sf, sn)
        mu  = mu_s * ys + ym
        std = std_s * ys
        surr_calls += lambda_

        ei      = acq_ei(mu, std, float(y.min()))
        top_idx = np.argsort(ei)[-k:][::-1]

        for idx in top_idx:
            if true_calls >= true_budget:
                break
            x_next = pool[idx]
            y_next = eval_single(x_next)
            true_calls += 1
            best_true = min(best_true, y_next)
            X = np.vstack([X, x_next])
            y = np.append(y, y_next)
            true_axis.append(true_calls)
            best_axis.append(best_true)

    return np.array(true_axis), np.array(best_axis), surr_calls

TRUE_BUDGET = 50
print(f"\nRunning COCO-compatible prescreening BO (budget = {TRUE_BUDGET} true evals)...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

compare_scrs = [0.10, 0.20, 1.00]
compare_labels = ['SCR 10% (target)', 'SCR 20% (ceiling)', 'SCR 100% (no surrogate)']
compare_colors = ['seagreen', 'darkorange', 'tomato']

for scr, label, color in zip(compare_scrs, compare_labels, compare_colors):
    for seed in range(N_SEEDS):
        t_ax, b_ax, sc = run_coco_prescreening(seed=seed, true_budget=TRUE_BUDGET, scr_target=scr)
        ls = '--' if scr >= 1.0 else '-'
        axes[0].step(t_ax, b_ax, color=color, lw=1.5, alpha=0.5, where='post',
                     label=label if seed == 0 else None, ls=ls)

axes[0].axhline(F_STAR, color='black', ls=':', lw=1.5, label=f'f* ≈ {F_STAR:.3f}')
axes[0].axvline(TRUE_BUDGET, color='gray', ls=':', lw=1.2, label=f'Budget = {TRUE_BUDGET}')
axes[0].set_title(f'COCO-compatible convergence\nBudget = {TRUE_BUDGET} TRUE function calls')
axes[0].set_xlabel('True function calls (COCO budget counter)')
axes[0].set_ylabel('Best f(x) found'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

# Gap to optimum log scale
for scr, label, color in zip(compare_scrs, compare_labels, compare_colors):
    all_gaps = []
    for seed in range(N_SEEDS):
        t_ax, b_ax, _ = run_coco_prescreening(seed=seed, true_budget=TRUE_BUDGET, scr_target=scr)
        gap = np.maximum(b_ax - F_STAR, 1e-4)
        ls  = '--' if scr >= 1.0 else '-'
        axes[1].step(t_ax, gap, color=color, lw=1.2, alpha=0.4, where='post', ls=ls,
                     label=label if seed == 0 else None)
axes[1].set_yscale('log')
axes[1].set_title('Gap to optimum — log scale\n'
                  'SCR 10% finds near-optimal solution using far fewer true calls')
axes[1].set_xlabel('True function calls'); axes[1].set_ylabel('gap = best_f − f* (log)')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3, which='both')

plt.suptitle(f'COCO-compatible prescreening BO\n'
             f'λ = {LAMBDA} pool per generation, hard budget = {TRUE_BUDGET} true evaluations',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part6_coco_prescreening.png', dpi=150)
plt.close()
print("Saved: part6_coco_prescreening.png")

# Print final SCR summary
print("\nFinal SCR summary:")
print(f"  {'Setting':<20}  {'k/gen':>5}  {'True calls':>10}  {'Surr calls':>10}  {'SCR':>7}")
for scr in scr_settings:
    k = max(1, int(np.ceil(scr * LAMBDA)))
    tc = N_INIT + k * N_ITER
    sc = LAMBDA * N_ITER
    r  = tc / (tc + sc) * 100
    ok = "✓" if r < 20 else "✗"
    print(f"  {scr*100:.0f}%{'':<17}  {k:>5}  {tc:>10}  {sc:>10}  {r:>6.1f}% {ok}")

print("\nAll done. Check lesson-13/output/ for plots.")
print("\nKey takeaways:")
print("  CORRECT:  SCR = true_calls / (true_calls + surrogate_calls) < 20%")
print("  The surrogate must handle 80%+ of evaluations — not 20%")
print("  Prescreening: generate λ pool, evaluate ALL with surrogate,")
print("                select top k (k/λ < 20%) for true evaluation")
print("  Only truly-evaluated points enter the training dataset (no drift)")
print("  SCR ≥ 50% is the professor's complaint — far too many true evals")
