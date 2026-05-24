import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm, spearmanr
from scipy.optimize import minimize
from matplotlib.patches import Patch

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

N_INIT  = 5
N_ITER  = 30
N_SEEDS = 5


# ── Benchmark ──────────────────────────────────────────────────────────────────

def forrester(x):
    x = np.asarray(x).ravel()
    return (6*x - 2)**2 * np.sin(12*x - 4)

F_STAR = forrester(np.array([0.7572])).item()


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


# ── SCR-enforced BO loop ───────────────────────────────────────────────────────

def run_scr_bo(seed, scr_max=0.0, n_iter=N_ITER):
    """
    BO with SCR ceiling enforced at each step.

    Before accepting a surrogate prediction, the loop computes projected SCR:
        projected_scr = (surr_calls + 1) / (true_calls + surr_calls + 1)
    If projected_scr > scr_max, call the true function instead.

    SCR = surrogate_only_calls / (true_calls + surrogate_only_calls)
    """
    rg = np.random.default_rng(seed)
    X  = lhs(N_INIT, 1, rg)
    y  = np.array([eval_single(x) for x in X])

    true_calls = N_INIT
    surr_calls = 0
    best_true  = float(y.min())

    best_history = [best_true]
    true_history = [true_calls]
    scr_history  = [0.0]
    eval_types   = []

    for _ in range(n_iter):
        Xm, Xs = float(X.mean()), float(X.std()) + 1e-8
        ym, ys = float(y.mean()), float(y.std()) + 1e-8
        X_std  = (X - Xm) / Xs
        y_std  = (y - ym) / ys
        l, sf, sn = fit_gp(X_std, y_std)

        Xc     = rg.uniform(0, 1, (500, 1)).astype(np.float32)
        Xc_std = (Xc - Xm) / Xs
        mu_s, std_s = gp_predict(X_std, y_std, Xc_std, l, sf, sn)
        mu  = mu_s * ys + ym
        std = std_s * ys

        ei      = acq_ei(mu, std, float(y.min()))
        idx     = int(np.argmax(ei))
        x_next  = Xc[idx]
        mu_next = float(mu[idx])

        # SCR enforcer: use surrogate only if projected SCR stays within ceiling
        proj_scr = (surr_calls + 1) / (true_calls + surr_calls + 1)
        use_surr = (scr_max > 0.0) and (proj_scr <= scr_max)

        if use_surr:
            y_next = mu_next
            surr_calls += 1
            eval_types.append('surrogate')
        else:
            y_next = eval_single(x_next)
            true_calls += 1
            eval_types.append('true')
            best_true = min(best_true, y_next)

        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
        best_history.append(best_true)
        true_history.append(true_calls)
        scr_history.append(surr_calls / (true_calls + surr_calls))

    return (np.array(best_history), np.array(true_history),
            np.array(scr_history), eval_types)


# ════════════════════════════════════════════════════════════════════════════════
# Part 1 — SCR concept illustration
# ════════════════════════════════════════════════════════════════════════════════

def illustrate_scr(scr, n_steps=25, seed=0):
    rng2 = np.random.default_rng(seed)
    true_c = 0; surr_c = 0; types = []
    for _ in range(n_steps):
        total = true_c + surr_c
        proj  = (surr_c + 1) / (total + 1) if total > 0 else 0.0
        if scr > 0 and proj <= scr:
            surr_c += 1; types.append('surrogate')
        else:
            true_c += 1; types.append('true')
    return types, true_c, surr_c

n_show = 25
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, scr, title in zip(axes,
    [0.0, 0.20, 0.40],
    ['SCR = 0%\n(pure BO)', 'SCR = 20%', 'SCR = 40%']):
    types, n_true, n_surr = illustrate_scr(scr, n_show)
    colors = ['seagreen' if t == 'true' else 'darkorange' for t in types]
    ax.bar(range(1, n_show+1), [1]*n_show, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_title(f'{title}\n{n_true} true  |  {n_surr} surrogate', fontsize=10)
    ax.set_xlabel('BO iteration')
    ax.set_yticks([])
    ax.set_xlim(0, n_show + 1)

axes[0].legend(handles=[Patch(color='seagreen', label='True function call'),
                         Patch(color='darkorange', label='Surrogate-only')],
               fontsize=8)
plt.suptitle('SCR = surrogate_only_calls / (true_calls + surrogate_only_calls)\n'
             'Green = true evaluation,  Orange = surrogate-only (no f call made)',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part1_scr_concept.png', dpi=150)
plt.close()
print("Saved: part1_scr_concept.png")


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

n_cand   = 40
X_cand   = rng.uniform(0, 1, (n_cand, 1)).astype(np.float32)
y_true_c = forrester(X_cand.ravel())
Xc_s     = (X_cand - Xm_t) / Xs_t
mu_c_s, _ = gp_predict(X_tr_s, y_tr_s, Xc_s, l_t, sf_t, sn_t)
mu_c     = mu_c_s * ys_t + ym_t

rho, pval    = spearmanr(y_true_c, mu_c)
true_ranks   = y_true_c.argsort().argsort() + 1
surr_ranks   = mu_c.argsort().argsort() + 1

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_true_c, mu_c, color='steelblue', s=60, alpha=0.8, zorder=3)
lims = [min(y_true_c.min(), mu_c.min())-0.5, max(y_true_c.max(), mu_c.max())+0.5]
axes[0].plot(lims, lims, 'k--', lw=1.5, alpha=0.5, label='Perfect prediction')
axes[0].set_title(f'Surrogate μ vs. true f(x)\nSpearman ρ = {rho:.3f}  (p = {pval:.2e})')
axes[0].set_xlabel('True f(x)'); axes[0].set_ylabel('Surrogate μ(x)')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

axes[1].scatter(true_ranks, surr_ranks, color='tomato', s=60, alpha=0.8, zorder=3)
axes[1].plot([1, n_cand], [1, n_cand], 'k--', lw=1.5, alpha=0.5, label='Perfect ranking')
axes[1].set_title(f'Rank comparison — {n_cand} candidates\n'
                  f'ρ = {rho:.3f}  → {"safe to prescreen" if rho > 0.7 else "ranking unreliable"}')
axes[1].set_xlabel('True rank (1 = best)'); axes[1].set_ylabel('Surrogate rank')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

plt.suptitle('Part 2 — Rank correlation: can the surrogate correctly order candidates?\n'
             'High ρ → prescreening is trustworthy → moderate SCR is safer',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part2_rank_correlation.png', dpi=150)
plt.close()
print(f"Saved: part2_rank_correlation.png  (Spearman ρ = {rho:.3f})")


# ════════════════════════════════════════════════════════════════════════════════
# Part 3 — Live SCR tracker during BO
# ════════════════════════════════════════════════════════════════════════════════

best3, true3, scr3, etypes3 = run_scr_bo(seed=RANDOM_SEED, scr_max=0.20)

step_colors = ['seagreen' if t == 'true' else 'darkorange' for t in etypes3]
iters       = np.arange(1, N_ITER + 1)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].bar(iters, [1]*N_ITER, color=step_colors, edgecolor='white', linewidth=0.5)
axes[0].set_yticks([])
axes[0].set_title(f'Evaluation pattern: SCR ceiling = 20%  '
                  f'({etypes3.count("true")} true,  {etypes3.count("surrogate")} surrogate-only)')
axes[0].legend(handles=[Patch(color='seagreen', label='True call'),
                         Patch(color='darkorange', label='Surrogate-only')],
               fontsize=8)

axes[1].plot(range(N_ITER+1), scr3*100, color='tomato', lw=2.5, label='Running SCR (%)')
axes[1].axhline(20, color='black', ls='--', lw=1.5, label='20% ceiling')
axes[1].fill_between(range(N_ITER+1), 0, scr3*100, alpha=0.2, color='tomato')
axes[1].set_ylabel('SCR (%)'); axes[1].set_xlabel('BO iteration')
axes[1].set_title('Running SCR — enforcer keeps it at or below 20%')
axes[1].set_ylim(0, 35); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

plt.suptitle('Part 3 — SCR-controlled BO: the enforcer decides at each step\n'
             '"Can I use the surrogate this time, or must I call the true function?"',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part3_scr_tracker.png', dpi=150)
plt.close()
print(f"Saved: part3_scr_tracker.png  (final SCR = {scr3[-1]*100:.1f}%)")


# ════════════════════════════════════════════════════════════════════════════════
# Part 4 — SCR sensitivity: convergence per TRUE evaluation
# ════════════════════════════════════════════════════════════════════════════════

scr_settings = [0.0, 0.10, 0.20, 0.40]
scr_names    = ['SCR 0%', 'SCR 10%', 'SCR 20%', 'SCR 40%']
colors4      = ['steelblue', 'seagreen', 'darkorange', 'tomato']

print("Running SCR sensitivity analysis...")
results4 = {}
for scr in scr_settings:
    runs = []
    for seed in range(N_SEEDS):
        best_h, true_h, _, _ = run_scr_bo(seed=seed, scr_max=scr, n_iter=N_ITER)
        runs.append((best_h, true_h))
    results4[scr] = runs
print("Done.")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for scr, label, color in zip(scr_settings, scr_names, colors4):
    bests = np.array([r[0] for r in results4[scr]])
    m = bests.mean(0); s = bests.std(0)
    axes[0].plot(range(N_ITER+1), m, color=color, lw=2.5, label=label)
    axes[0].fill_between(range(N_ITER+1), m-s, m+s, alpha=0.15, color=color)
axes[0].axhline(F_STAR, color='black', ls='--', lw=1.5, label=f'f* ≈ {F_STAR:.3f}')
axes[0].set_title(f'Best true f found — mean ± std over {N_SEEDS} seeds')
axes[0].set_xlabel('BO iteration'); axes[0].set_ylabel('Best f(x) found')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

for scr, label, color in zip(scr_settings, scr_names, colors4):
    for i, (best_h, true_h) in enumerate(results4[scr]):
        axes[1].step(true_h, best_h, color=color, lw=1.5,
                     alpha=0.3 if i > 0 else 0.9, where='post',
                     label=label if i == 0 else None)
axes[1].axhline(F_STAR, color='black', ls='--', lw=1.5, label=f'f* ≈ {F_STAR:.3f}')
axes[1].set_title('Same convergence — x-axis = TRUE evaluations only\n'
                  'High SCR = fewer true evals = curve shifts right')
axes[1].set_xlabel('True function calls (COCO budget)')
axes[1].set_ylabel('Best true f(x) found')
axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

plt.suptitle('Part 4 — SCR sensitivity: convergence per true evaluation\n'
             'SCR 0% is the baseline; moderate SCR is nearly free; high SCR can drift',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part4_scr_sensitivity.png', dpi=150)
plt.close()
print("Saved: part4_scr_sensitivity.png")


# ════════════════════════════════════════════════════════════════════════════════
# Part 5 — Budget allocation for each SCR
# ════════════════════════════════════════════════════════════════════════════════

true_counts = []
surr_counts = []
for scr in scr_settings:
    _, _, _, etypes = run_scr_bo(seed=0, scr_max=scr, n_iter=N_ITER)
    true_counts.append(etypes.count('true'))
    surr_counts.append(etypes.count('surrogate'))

x4 = np.arange(len(scr_settings))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(x4, true_counts, color='seagreen', edgecolor='black', alpha=0.85,
            label='True function calls')
axes[0].bar(x4, surr_counts, bottom=true_counts, color='darkorange',
            edgecolor='black', alpha=0.85, label='Surrogate-only')
axes[0].set_xticks(x4); axes[0].set_xticklabels(scr_names)
axes[0].set_title(f'Budget after {N_ITER} BO iterations\nTotal iterations = same for all')
axes[0].set_ylabel('Evaluations'); axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3, axis='y')
for i, (t, s) in enumerate(zip(true_counts, surr_counts)):
    axes[0].text(i, t/2, str(t), ha='center', va='center',
                 fontsize=11, fontweight='bold', color='white')
    if s > 0:
        axes[0].text(i, t + s/2, str(s), ha='center', va='center',
                     fontsize=11, fontweight='bold', color='white')

realized_scr = [s/(t+s)*100 if (t+s) > 0 else 0
                for t, s in zip(true_counts, surr_counts)]
bars = axes[1].bar(x4, realized_scr, color=colors4, edgecolor='black', alpha=0.85)
axes[1].axhline(20, color='black', ls='--', lw=2, label='20% ceiling (COCO limit)')
axes[1].set_xticks(x4); axes[1].set_xticklabels(scr_names)
axes[1].set_title('Realized SCR — must stay ≤ 20% for valid experiments')
axes[1].set_ylabel('Realized SCR (%)')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, realized_scr):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.5, f'{v:.1f}%',
                 ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Part 5 — Budget allocation: how many evaluations are truly expensive?',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part5_budget_breakdown.png', dpi=150)
plt.close()
print("Saved: part5_budget_breakdown.png")


# ════════════════════════════════════════════════════════════════════════════════
# Part 6 — COCO-compatible BO with true-eval budget
# ════════════════════════════════════════════════════════════════════════════════

def run_coco_bo(seed, true_budget=30, scr_max=0.20):
    """
    BO with a hard TRUE-EVALUATION budget.
    Stops when true_budget true function calls have been made.
    SCR enforcer applied at each step.
    """
    rg = np.random.default_rng(seed)
    X  = lhs(N_INIT, 1, rg)
    y  = np.array([eval_single(x) for x in X])
    true_calls = N_INIT; surr_calls = 0
    best_true  = float(y.min())
    true_axis  = [true_calls]; best_axis = [best_true]

    while true_calls < true_budget:
        Xm, Xs = float(X.mean()), float(X.std()) + 1e-8
        ym, ys = float(y.mean()), float(y.std()) + 1e-8
        X_std  = (X - Xm) / Xs
        y_std  = (y - ym) / ys
        l, sf, sn = fit_gp(X_std, y_std)

        Xc     = rg.uniform(0, 1, (500, 1)).astype(np.float32)
        Xc_std = (Xc - Xm) / Xs
        mu_s, std_s = gp_predict(X_std, y_std, Xc_std, l, sf, sn)
        mu  = mu_s * ys + ym
        std = std_s * ys

        ei  = acq_ei(mu, std, float(y.min()))
        idx = int(np.argmax(ei))
        x_next  = Xc[idx]
        mu_next = float(mu[idx])

        proj_scr = (surr_calls + 1) / (true_calls + surr_calls + 1)
        use_surr = (scr_max > 0.0) and (proj_scr <= scr_max)

        if use_surr:
            y_next = mu_next; surr_calls += 1
        else:
            y_next = eval_single(x_next); true_calls += 1
            best_true = min(best_true, y_next)
            true_axis.append(true_calls); best_axis.append(best_true)

        X = np.vstack([X, x_next]); y = np.append(y, y_next)

    return np.array(true_axis), np.array(best_axis)

TRUE_BUDGET = 30
print(f"\nRunning COCO-compatible BO (budget = {TRUE_BUDGET} true evals)...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for scr, label, color in zip([0.0, 0.20], ['SCR 0% (pure BO)', 'SCR 20%'],
                               ['steelblue', 'darkorange']):
    for seed in range(N_SEEDS):
        t_ax, b_ax = run_coco_bo(seed=seed, true_budget=TRUE_BUDGET, scr_max=scr)
        axes[0].step(t_ax, b_ax, color=color, lw=1.5, alpha=0.5, where='post',
                     label=label if seed == 0 else None)
axes[0].axhline(F_STAR, color='black', ls='--', lw=1.5, label=f'f* ≈ {F_STAR:.3f}')
axes[0].axvline(TRUE_BUDGET, color='gray', ls=':', lw=1.2, label=f'Budget = {TRUE_BUDGET}')
axes[0].set_title(f'COCO-compatible convergence\nBudget = {TRUE_BUDGET} TRUE evaluations')
axes[0].set_xlabel('True function calls (COCO budget counter)')
axes[0].set_ylabel('Best f(x) found')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

for scr, label, color in zip([0.0, 0.20], ['SCR 0%', 'SCR 20%'],
                               ['steelblue', 'darkorange']):
    for seed in range(N_SEEDS):
        t_ax, b_ax = run_coco_bo(seed=seed, true_budget=TRUE_BUDGET, scr_max=scr)
        gap = np.maximum(b_ax - F_STAR, 1e-4)
        axes[1].step(t_ax, gap, color=color, lw=1, alpha=0.4, where='post',
                     label=label if seed == 0 else None)
axes[1].set_yscale('log')
axes[1].set_title('Gap to optimum — log scale')
axes[1].set_xlabel('True function calls')
axes[1].set_ylabel('gap = best_f − f* (log)')
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3, which='both')

plt.suptitle(f'Part 6 — COCO-compatible BO — x-axis counts ONLY true function calls\n'
             f'This is the correct form for Lesson 16 COCO benchmarking',
             fontsize=11)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'part6_coco_bo.png', dpi=150)
plt.close()
print("Saved: part6_coco_bo.png")

print("\nAll done. Check lesson-13/output/ for plots.")
print("\nKey takeaways:")
print("  SCR = surrogate_only_calls / (true_calls + surrogate_only_calls)")
print("  SCR must stay ≤ 20% to avoid surrogate drift")
print("  The enforcer checks projected SCR before each step")
print("  COCO x-axis = true function calls only (surrogate calls are free)")
