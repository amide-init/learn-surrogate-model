RANDOM_SEED = 42

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import norm as sp_norm
from scipy.stats.qmc import LatinHypercube
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.ensemble import RandomForestRegressor
from scipy.interpolate import RBFInterpolator

warnings.filterwarnings('ignore')
os.makedirs('lesson-16/output', exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
SCR_LEVELS    = [0.0, 0.2, 0.4, 0.6, 0.8]
SCR_LABELS    = ['0%', '20%', '40%', '60%', '80%']
SCR_COLORS    = ['#1d3557', '#457b9d', '#2a9d8f', '#f4a261', '#e63946']
SCR_LEVELS_2D = [0.0, 0.4, 0.8]
N_INIT        = 10
N_INIT_2D     = 15
N_ITER        = 15
N_SEEDS       = 3
EPOCHS_BO     = 200
N_MEMBERS     = 5
T_DROPOUT     = 50

SURROGATES = ['gp', 'mc_dropout', 'deep_ensemble', 'rbf', 'rf']
LABELS     = ['GP', 'MC Dropout', 'Deep Ensemble', 'RBF', 'Random Forest']
COLORS     = ['steelblue', 'seagreen', 'darkorange', 'tomato', 'purple']

FORRESTER_OPT = -6.020740
BRANIN_OPT    =  0.397887


# ── Benchmark functions ────────────────────────────────────────────────────────
def forrester(x):
    x = float(np.asarray(x).ravel()[0])
    return (6*x - 2)**2 * np.sin(12*x - 4)


def branin(x):
    x  = np.asarray(x).ravel()
    x1 = x[0] * 15 - 5
    x2 = x[1] * 15
    return float((x2 - 5.1 / (4*np.pi**2) * x1**2 + 5/np.pi * x1 - 6)**2
                 + 10*(1 - 1/(8*np.pi)) * np.cos(x1) + 10)


# ── Neural network ─────────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self, d_in, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 64),   nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_net(X, y, epochs, seed):
    torch.manual_seed(seed)
    model = Net(X.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt    = torch.tensor(X, dtype=torch.float32)
    yt    = torch.tensor(y, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        nn.MSELoss()(model(Xt), yt).backward()
        opt.step()
    return model


# ── Surrogate predict ─────────────────────────────────────────────────────────
def surrogate_predict(stype, X_tr, y_tr, X_cand, seed=0, epochs=200):
    torch.manual_seed(seed)

    if stype == 'gp':
        kernel = Matern(nu=2.5, length_scale_bounds=(1e-3, 10.0))
        gpr = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-4,
            n_restarts_optimizer=2, normalize_y=True, random_state=seed,
        )
        gpr.fit(X_tr, y_tr)
        mu, std = gpr.predict(X_cand, return_std=True)
        return mu, np.maximum(std, 1e-8)

    if stype == 'mc_dropout':
        model = train_net(X_tr, y_tr, epochs, seed)
        model.train()
        Xc = torch.tensor(X_cand, dtype=torch.float32)
        with torch.no_grad():
            preds = np.stack([model(Xc).numpy() for _ in range(T_DROPOUT)])
        return preds.mean(0), np.maximum(preds.std(0), 1e-8)

    if stype == 'deep_ensemble':
        preds = []
        for i in range(N_MEMBERS):
            m  = train_net(X_tr, y_tr, epochs, seed + i * 17)
            m.eval()
            Xc = torch.tensor(X_cand, dtype=torch.float32)
            with torch.no_grad():
                preds.append(m(Xc).numpy())
        preds = np.stack(preds)
        return preds.mean(0), np.maximum(preds.std(0), 1e-8)

    if stype == 'rbf':
        rbf = RBFInterpolator(X_tr, y_tr, kernel='thin_plate_spline', smoothing=1e-3)
        mu  = rbf(X_cand)
        diffs = X_cand[:, None, :] - X_tr[None, :, :]
        min_d = np.linalg.norm(diffs, axis=-1).min(axis=1)
        std   = min_d / (min_d.max() + 1e-8) + 0.05
        return mu, std

    if stype == 'rf':
        rf = RandomForestRegressor(n_estimators=100, random_state=seed)
        rf.fit(X_tr, y_tr)
        preds = np.stack([t.predict(X_cand) for t in rf.estimators_])
        return preds.mean(0), np.maximum(preds.std(0), 1e-8)

    raise ValueError(f'Unknown surrogate: {stype}')


# ── EI acquisition ────────────────────────────────────────────────────────────
def ei(mu, std, best, xi=0.01):
    imp = best - mu - xi
    z   = imp / (std + 1e-8)
    return imp * sp_norm.cdf(z) + std * sp_norm.pdf(z)


# ── SCR-enforced BO ───────────────────────────────────────────────────────────
def run_bo(stype, func, dim, seed, f_opt,
           n_init, n_iter, scr_max, epochs=EPOCHS_BO):
    rng = np.random.RandomState(seed)

    sampler = LatinHypercube(d=dim, seed=seed)
    X_obs   = sampler.random(n_init)
    y_obs   = np.array([func(x) for x in X_obs])

    true_calls = n_init
    surr_calls = 0
    best_true  = y_obs.min()
    history    = [(true_calls, max(best_true - f_opt, 0.0))]

    for _ in range(n_iter):
        y_mean = y_obs.mean()
        y_sc   = y_obs.std() + 1e-8
        y_std  = (y_obs - y_mean) / y_sc

        X_cand      = rng.rand(200, dim)
        mu_s, sig_s = surrogate_predict(stype, X_obs, y_std, X_cand, seed, epochs)
        mu          = mu_s  * y_sc + y_mean
        sig         = sig_s * y_sc

        acq             = ei(mu, sig, best_true)
        idx             = np.argmax(acq)
        x_next, mu_next = X_cand[idx], mu[idx]

        proj     = (surr_calls + 1) / (true_calls + surr_calls + 1)
        use_surr = (scr_max > 0) and (proj <= scr_max)

        if use_surr:
            y_next = mu_next
            surr_calls += 1
        else:
            y_next = func(x_next)
            true_calls += 1
            best_true = min(best_true, y_next)

        X_obs = np.vstack([X_obs, x_next])
        y_obs = np.append(y_obs, y_next)
        history.append((true_calls, max(best_true - f_opt, 0.0)))

    return history


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_seeds(stype, func, dim, f_opt, n_init, n_iter, scr_max, epochs=EPOCHS_BO):
    return [
        run_bo(stype, func, dim, RANDOM_SEED + s, f_opt,
               n_init=n_init, n_iter=n_iter, scr_max=scr_max, epochs=epochs)
        for s in range(N_SEEDS)
    ]


def final_gap(hists):
    return float(np.median([h[-1][1] for h in hists]))


def true_calls_used(hists, n_init):
    return float(np.median([h[-1][0] - n_init for h in hists]))


def plot_convergence_scr(ax, results_by_scr, n_init, title):
    for scr, slabel, scolor in zip(SCR_LEVELS, SCR_LABELS, SCR_COLORS):
        if scr not in results_by_scr:
            continue
        hists  = results_by_scr[scr]
        max_tc = max(h[-1][0] for h in hists)
        grid   = np.arange(n_init, max_tc + 1)
        runs   = np.array([
            np.interp(grid, [h[0] for h in hist], [h[1] for h in hist])
            for hist in hists
        ])
        med = np.median(runs, axis=0)
        lo  = np.percentile(runs, 25, axis=0)
        hi  = np.percentile(runs, 75, axis=0)
        ax.semilogy(grid, np.maximum(med, 1e-6), color=scolor,
                    label=f'SCR≤{slabel}', lw=2)
        ax.fill_between(grid, np.maximum(lo, 1e-6), np.maximum(hi, 1e-6),
                        alpha=0.15, color=scolor)
    ax.set_xlabel('True function calls')
    ax.set_ylabel('Gap to optimum (log scale)')
    ax.set_title(title)
    ax.legend(fontsize=9)


# ── Part 1: GP convergence per SCR level ──────────────────────────────────────
def part1_gp_convergence(gp_forr):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_convergence_scr(ax, gp_forr, N_INIT,
                         'Part 1 — GP convergence per SCR ceiling (Forrester 1D)')
    plt.tight_layout()
    plt.savefig('lesson-16/output/part1_gp_convergence.png', dpi=120)
    plt.close()
    print('Part 1 done.')


# ── Part 2: True calls saved per SCR level ────────────────────────────────────
def part2_true_calls_saved(gp_forr):
    tc_baseline = true_calls_used(gp_forr[0.0], N_INIT)
    tcs  = [true_calls_used(gp_forr[s], N_INIT) for s in SCR_LEVELS]
    savs = [100 * (tc_baseline - tc) / (tc_baseline + 1e-8) for tc in tcs]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(SCR_LABELS, tcs, color=SCR_COLORS)
    for bar, tc, sav in zip(bars, tcs, savs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f'{tc:.1f}\n(−{sav:.0f}%)', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('SCR ceiling')
    ax.set_ylabel('Avg true calls beyond initial LHS')
    ax.set_title('Part 2 — True calls saved per SCR level (GP, Forrester 1D)')
    ax.set_ylim(0, tc_baseline * 1.25)
    plt.tight_layout()
    plt.savefig('lesson-16/output/part2_true_calls_saved.png', dpi=120)
    plt.close()
    print('Part 2 done.')


# ── Part 3: Quality–efficiency frontier (all surrogates) ─────────────────────
def part3_quality_efficiency(all_forr):
    fig, ax = plt.subplots(figsize=(8, 6))
    for stype, label, color in zip(SURROGATES, LABELS, COLORS):
        tcs  = [true_calls_used(all_forr[stype][s], N_INIT) for s in SCR_LEVELS]
        gaps = [final_gap(all_forr[stype][s])              for s in SCR_LEVELS]
        ax.plot(tcs, gaps, 'o-', color=color, label=label, lw=2, markersize=7)
        for tc, gap, slabel in zip(tcs, gaps, SCR_LABELS):
            ax.annotate(slabel, (tc, gap), textcoords='offset points',
                        xytext=(4, 3), fontsize=7, color=color)
    ax.set_xlabel('Avg true calls beyond initial LHS (fewer = cheaper)')
    ax.set_ylabel('Final gap to optimum (lower = better)')
    ax.set_title('Part 3 — Quality–efficiency frontier (Forrester 1D)')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig('lesson-16/output/part3_quality_efficiency.png', dpi=120)
    plt.close()
    print('Part 3 done.')


# ── Part 4: Final gap per surrogate × SCR level ───────────────────────────────
def part4_gap_by_scr(all_forr):
    n_sur = len(SURROGATES)
    n_scr = len(SCR_LEVELS)
    x     = np.arange(n_sur)
    width = 0.8 / n_scr

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (scr, slabel, scolor) in enumerate(zip(SCR_LEVELS, SCR_LABELS, SCR_COLORS)):
        gaps   = [final_gap(all_forr[s][scr]) for s in SURROGATES]
        offset = (i - n_scr / 2 + 0.5) * width
        bars   = ax.bar(x + offset, gaps, width, label=f'SCR≤{slabel}',
                        color=scolor, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=15)
    ax.set_ylabel('Final gap to optimum')
    ax.set_title('Part 4 — Final gap per surrogate × SCR ceiling (Forrester 1D)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('lesson-16/output/part4_gap_by_scr.png', dpi=120)
    plt.close()
    print('Part 4 done.')


# ── Part 5: Branin 2D validation ──────────────────────────────────────────────
def part5_branin(all_bran):
    n_sur = len(SURROGATES)
    n_scr = len(SCR_LEVELS_2D)
    x     = np.arange(n_sur)
    width = 0.7 / n_scr
    bran_colors = [SCR_COLORS[SCR_LEVELS.index(s)] for s in SCR_LEVELS_2D]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (scr, scolor) in enumerate(zip(SCR_LEVELS_2D, bran_colors)):
        gaps   = [final_gap(all_bran[s][scr]) for s in SURROGATES]
        offset = (i - n_scr / 2 + 0.5) * width
        slabel = SCR_LABELS[SCR_LEVELS.index(scr)]
        ax.bar(x + offset, gaps, width, label=f'SCR≤{slabel}',
               color=scolor, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=15)
    ax.set_ylabel('Final gap to optimum')
    ax.set_title('Part 5 — Branin (2D) validation: gap per surrogate × SCR ceiling')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('lesson-16/output/part5_branin.png', dpi=120)
    plt.close()
    print('Part 5 done.')


# ── Part 6: Leaderboard and recommendations ───────────────────────────────────
def part6_leaderboard(all_forr):
    def recommend(gaps_by_scr):
        baseline  = gaps_by_scr[0.0] + 1e-6
        threshold = baseline * 1.5
        best_scr  = 0.0
        for scr in SCR_LEVELS:
            if gaps_by_scr[scr] <= threshold:
                best_scr = scr
        return best_scr

    rec_scrs  = []
    savings   = []
    penalties = []

    tc_at_scr0 = true_calls_used(all_forr[SURROGATES[0]][0.0], N_INIT)

    for stype in SURROGATES:
        gaps  = {scr: final_gap(all_forr[stype][scr]) for scr in SCR_LEVELS}
        r     = recommend(gaps)
        rec_scrs.append(r)
        tc_r  = true_calls_used(all_forr[stype][r], N_INIT)
        sav   = 100 * (tc_at_scr0 - tc_r) / (tc_at_scr0 + 1e-8)
        savings.append(max(sav, 0))
        pen   = 100 * (gaps[r] - gaps[0.0]) / (gaps[0.0] + 1e-6)
        penalties.append(max(pen, 0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(LABELS, [s * 100 for s in rec_scrs], color=COLORS)
    axes[0].set_ylabel('Recommended max SCR (%)')
    axes[0].set_title('Recommended SCR ceiling')
    axes[0].tick_params(axis='x', rotation=20)
    for i, v in enumerate(rec_scrs):
        axes[0].text(i, v * 100 + 0.5, f'{int(v*100)}%', ha='center', va='bottom', fontsize=10)

    axes[1].bar(LABELS, savings, color=COLORS)
    axes[1].set_ylabel('True call savings (%)')
    axes[1].set_title('Savings at recommended SCR')
    axes[1].tick_params(axis='x', rotation=20)
    for i, v in enumerate(savings):
        axes[1].text(i, v + 0.3, f'{v:.0f}%', ha='center', va='bottom', fontsize=10)

    axes[2].bar(LABELS, penalties, color=COLORS)
    axes[2].set_ylabel('Quality penalty (%)')
    axes[2].set_title('Gap increase vs SCR=0%')
    axes[2].tick_params(axis='x', rotation=20)
    for i, v in enumerate(penalties):
        axes[2].text(i, v + 0.1, f'{v:.0f}%', ha='center', va='bottom', fontsize=10)

    fig.suptitle('Part 6 — Leaderboard: recommended SCR per surrogate', fontsize=13)
    plt.tight_layout()
    plt.savefig('lesson-16/output/part6_leaderboard.png', dpi=120)
    plt.close()

    print('\nPart 6 — Recommendations (criterion: gap ≤ 1.5 × baseline gap)')
    print(f"{'Surrogate':<18} {'Rec. SCR':>10} {'Savings':>10} {'Gap penalty':>12}")
    print('-' * 54)
    for label, r, sav, pen in zip(LABELS, rec_scrs, savings, penalties):
        print(f'{label:<18} {int(r*100):>9}% {sav:>9.1f}% {pen:>11.1f}%')
    print('Part 6 done.')


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Lesson 16 — SCR sensitivity: varying surrogate usage from 0% to 80%')
    print('=' * 70)

    print('Running GP SCR sweep on Forrester (Parts 1–2)...')
    gp_forr = {scr: run_seeds('gp', forrester, 1, FORRESTER_OPT,
                               N_INIT, N_ITER, scr)
               for scr in SCR_LEVELS}

    part1_gp_convergence(gp_forr)
    part2_true_calls_saved(gp_forr)

    print('Running all surrogates × SCR levels on Forrester (Parts 3–4, 6)...')
    all_forr = {}
    for stype, label in zip(SURROGATES, LABELS):
        print(f'  {label}...')
        all_forr[stype] = {scr: run_seeds(stype, forrester, 1, FORRESTER_OPT,
                                          N_INIT, N_ITER, scr)
                           for scr in SCR_LEVELS}

    part3_quality_efficiency(all_forr)
    part4_gap_by_scr(all_forr)

    print('Running Branin 2D validation (Part 5)...')
    all_bran = {}
    for stype, label in zip(SURROGATES, LABELS):
        print(f'  {label}...')
        all_bran[stype] = {scr: run_seeds(stype, branin, 2, BRANIN_OPT,
                                          N_INIT_2D, N_ITER, scr)
                           for scr in SCR_LEVELS_2D}

    part5_branin(all_bran)
    part6_leaderboard(all_forr)

    print('\nAll plots saved to lesson-16/output/')
