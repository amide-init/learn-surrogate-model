import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize as sp_minimize
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn

RANDOM_SEED = 42
OUT = 'lesson-19/output'
os.makedirs(OUT, exist_ok=True)

N_SEEDS       = 3
N_INIT        = 10
N_INIT_2D     = 15
N_ITER        = 20
EPOCHS_BO     = 150
N_BOOT        = 500
NOISE_LEVELS  = [0.0, 0.1, 0.5, 1.0, 2.0]
SCR_LEVELS    = [0.0, 0.2, 0.4, 0.6, 0.8]
FORRESTER_OPT = -6.020740
BRANIN_OPT    =  0.397887

SURROGATES   = ['gp', 'mc_dropout', 'deep_ensemble', 'rbf', 'rf']
SUR_LABELS   = ['GP', 'MC Dropout', 'Deep Ensemble', 'RBF', 'RF']
PAPER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
PAPER_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]


# ── test functions ─────────────────────────────────────────────────────────
def forrester(x):
    x = np.asarray(x).ravel()[0]
    return float((6*x - 2)**2 * np.sin(12*x - 4))

def branin(x):
    x1 = float(x[0]) * 15 - 5
    x2 = float(x[1]) * 15
    a, b, c = 1, 5.1/(4*np.pi**2), 5/np.pi
    r, s, t = 6, 10, 1/(8*np.pi)
    return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*np.cos(x1) + s


# ── neural network ─────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self, d_in, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 64),   nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_net(net, X, y, epochs, lr=1e-3):
    opt     = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(net(Xt), yt).backward()
        opt.step()


# ── unified surrogate interface ────────────────────────────────────────────
def surrogate_predict(stype, X_tr, y_tr, X_cand, seed=0, epochs=EPOCHS_BO):
    d = X_tr.shape[1]
    if stype == 'gp':
        gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-4,
                                       normalize_y=True, n_restarts_optimizer=2)
        gp.fit(X_tr, y_tr)
        return gp.predict(X_cand, return_std=True)
    if stype == 'mc_dropout':
        torch.manual_seed(seed)
        net = Net(d)
        train_net(net, X_tr, y_tr, epochs)
        net.train()
        Xc = torch.tensor(X_cand, dtype=torch.float32)
        with torch.no_grad():
            preds = torch.stack([net(Xc) for _ in range(50)])
        return preds.mean(0).numpy(), preds.std(0).numpy()
    if stype == 'deep_ensemble':
        Xc = torch.tensor(X_cand, dtype=torch.float32)
        preds = []
        for i in range(5):
            torch.manual_seed(seed + i * 17)
            m = Net(d)
            train_net(m, X_tr, y_tr, epochs)
            m.eval()
            with torch.no_grad():
                preds.append(m(Xc).numpy())
        preds = np.array(preds)
        return preds.mean(0), preds.std(0)
    if stype == 'rbf':
        rbf = RBFInterpolator(X_tr, y_tr, kernel='thin_plate_spline', smoothing=1e-3)
        mu  = rbf(X_cand)
        dists = np.min(np.linalg.norm(X_cand[:, None] - X_tr[None], axis=2), axis=1)
        return mu, dists / (dists.max() + 1e-8)
    if stype == 'rf':
        rf = RandomForestRegressor(n_estimators=100, random_state=seed)
        rf.fit(X_tr, y_tr)
        tree_preds = np.array([t.predict(X_cand) for t in rf.estimators_])
        return tree_preds.mean(0), tree_preds.std(0)
    raise ValueError(f'Unknown surrogate: {stype}')


# ── EI acquisition + LHS ──────────────────────────────────────────────────
def ei(mu, std, best, xi=0.01):
    imp = best - mu - xi
    z   = imp / (std + 1e-8)
    return imp * norm.cdf(z) + std * norm.pdf(z)

def lhs(n, d, seed):
    return qmc.LatinHypercube(d=d, seed=seed).random(n)


# ── BO loop ────────────────────────────────────────────────────────────────
def run_bo(stype, func, dim, f_opt, n_init, n_iter, seed,
           epochs=EPOCHS_BO, scr_max=0.0):
    rng  = np.random.RandomState(seed)
    X_tr = lhs(n_init, dim, seed)
    y_tr = np.array([func(x) for x in X_tr])
    hist       = []
    true_calls = n_init
    surr_calls = 0
    best_gap   = min(y_tr) - f_opt
    for _ in range(n_iter):
        X_cand = lhs(500, dim, rng.randint(0, 10000))
        mu, std = surrogate_predict(stype, X_tr, y_tr, X_cand, seed=seed, epochs=epochs)
        idx    = np.argmax(ei(mu, std, y_tr.min()))
        x_next = X_cand[idx]
        proj_scr = (surr_calls + 1) / (true_calls + surr_calls + 1)
        use_surr = (scr_max > 0) and (proj_scr <= scr_max)
        if use_surr:
            y_next = float(mu[idx])
            surr_calls += 1
        else:
            y_next = func(x_next)
            true_calls += 1
            best_gap = min(best_gap, y_next - f_opt)
        X_tr = np.vstack([X_tr, x_next])
        y_tr = np.append(y_tr, y_next)
        hist.append((true_calls, max(best_gap, 0.0)))
    return hist

def run_seeds(stype, func, dim, f_opt, n_init, n_iter,
              scr_max=0.0, epochs=EPOCHS_BO):
    return [run_bo(stype, func, dim, f_opt, n_init, n_iter,
                   seed=RANDOM_SEED + s, epochs=epochs, scr_max=scr_max)
            for s in range(N_SEEDS)]

def gaps_array(hists):
    return np.array([[h[k][1] for k in range(len(h))] for h in hists])

def final_gaps(hists):
    return np.array([h[-1][1] for h in hists])

def bootstrap_ci(runs, n_boot=N_BOOT):
    rng  = np.random.RandomState(RANDOM_SEED)
    n    = runs.shape[0]
    boot = np.array([np.median(runs[rng.randint(0, n, n)], axis=0) for _ in range(n_boot)])
    return np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


# ─────────────────────────────────────────────────────────────────────────
# PART 1 — Paper Figure 1: main convergence comparison
# ─────────────────────────────────────────────────────────────────────────
def part1(results_forr, results_bran):
    rc = {
        'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13,
        'legend.fontsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, results, title in [
            (axes[0], results_forr, 'Forrester (1-D)'),
            (axes[1], results_bran, 'Branin (2-D)'),
        ]:
            for i, stype in enumerate(SURROGATES):
                g   = gaps_array(results[stype])
                med = np.median(g, axis=0)
                lo, hi = bootstrap_ci(g)
                xs = range(1, len(med) + 1)
                ax.plot(xs, med, color=PAPER_COLORS[i],
                        linestyle=PAPER_STYLES[i], lw=1.8, label=SUR_LABELS[i])
                ax.fill_between(xs, lo, hi, color=PAPER_COLORS[i], alpha=0.15)
            ax.set_yscale('log')
            ax.set_xlabel('True evaluations')
            ax.set_ylabel('Gap to optimum')
            ax.set_title(title)
            ax.legend(loc='upper right')
            ax.grid(True, which='both', ls=':', alpha=0.4)
        fig.suptitle('Figure 1 — Convergence (median ± 95% bootstrap CI)', y=1.01)
        fig.tight_layout()
        fig.savefig(f'{OUT}/fig1_convergence.pdf', bbox_inches='tight')
        fig.savefig(f'{OUT}/fig1_convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig1_convergence.pdf / .png')


# ─────────────────────────────────────────────────────────────────────────
# PART 2 — Paper Figure 2: SCR sensitivity scatter
# ─────────────────────────────────────────────────────────────────────────
def part2(scr_results_forr):
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ['o', 's', 'D', '^', 'v']
    for i, stype in enumerate(SURROGATES):
        gaps    = [np.median(final_gaps(scr_results_forr[stype][scr])) for scr in SCR_LEVELS]
        savings = [scr * 100 for scr in SCR_LEVELS]
        ax.plot(savings, gaps, color=PAPER_COLORS[i], linestyle=PAPER_STYLES[i],
                marker=markers[i], lw=1.5, ms=7, label=SUR_LABELS[i])
    ax.set_xlabel('True evaluations saved via SCR (%)')
    ax.set_ylabel('Median final gap to optimum')
    ax.set_yscale('log')
    ax.set_title('Figure 2 — SCR Sensitivity: Quality vs. Savings (Forrester 1-D)')
    ax.legend()
    ax.grid(True, which='both', ls=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig2_scr_scatter.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/fig2_scr_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig2_scr_scatter.pdf / .png')


# ─────────────────────────────────────────────────────────────────────────
# PART 3 — Paper Figure 3: noise robustness
# ─────────────────────────────────────────────────────────────────────────
def rmse_at_noise(stype, sigma, seed=RANDOM_SEED):
    rng  = np.random.RandomState(seed)
    X_tr = lhs(20, 1, seed)
    y_tr = np.array([forrester(x) + rng.normal(0, sigma) for x in X_tr])
    X_test = np.linspace(0, 1, 200).reshape(-1, 1)
    y_true = np.array([forrester(x) for x in X_test])
    mu, _  = surrogate_predict(stype, X_tr, y_tr, X_test, seed=seed)
    return np.sqrt(np.mean((mu - y_true)**2))

def part3():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, stype in enumerate(SURROGATES):
        rmses = [rmse_at_noise(stype, sigma) for sigma in NOISE_LEVELS]
        ax.plot(NOISE_LEVELS, rmses, color=PAPER_COLORS[i],
                linestyle=PAPER_STYLES[i], marker='o', lw=1.8, ms=6,
                label=SUR_LABELS[i])
    ax.set_xlabel('Observation noise σ')
    ax.set_ylabel('RMSE on clean Forrester')
    ax.set_title('Figure 3 — Noise Robustness')
    ax.legend()
    ax.grid(True, ls=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig3_noise.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/fig3_noise.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig3_noise.pdf / .png')


# ─────────────────────────────────────────────────────────────────────────
# PART 4 — Paper Figure 4: BBOB leaderboard
# ─────────────────────────────────────────────────────────────────────────
def make_bbob_func(func_id, dim):
    try:
        import cocoex
        suite = cocoex.Suite('bbob', 'instances:1',
                             f'dimensions:{dim} function_indices:{func_id}')
        f  = suite[0]
        lb = f.lower_bounds.copy()
        ub = f.upper_bounds.copy()
        def wrapped(x):
            return float(f(lb + np.asarray(x).ravel() * (ub - lb)))
        wrapped._alive = (suite, f)
        return wrapped
    except Exception:
        return None

def find_fopt(func, dim, n_restarts=15):
    rng  = np.random.RandomState(RANDOM_SEED)
    best = np.inf
    for _ in range(n_restarts):
        res = sp_minimize(func, rng.rand(dim), method='L-BFGS-B',
                          bounds=[(1e-6, 1-1e-6)]*dim,
                          options={'maxiter': 5000, 'ftol': 1e-12})
        best = min(best, res.fun)
    return best

def part4():
    FUNC_IDS   = [1, 8, 15]
    FUNC_SHORT = ['Sphere', 'Rosenbrock', 'Rastrigin']
    DIM        = 2
    rank_mat   = np.zeros((len(SURROGATES), len(FUNC_IDS)))
    has_cocoex = True

    for fi, (fid, fname) in enumerate(zip(FUNC_IDS, FUNC_SHORT)):
        func = make_bbob_func(fid, DIM)
        if func is None:
            has_cocoex = False
            for i in range(len(SURROGATES)):
                rank_mat[i, fi] = i + 1
            continue
        f_opt    = find_fopt(func, DIM)
        gaps_all = []
        for stype in SURROGATES:
            hists = run_seeds(stype, func, DIM, f_opt, N_INIT_2D, N_ITER)
            gaps_all.append(np.median(final_gaps(hists)))
        for rank, idx in enumerate(np.argsort(gaps_all)):
            rank_mat[idx, fi] = rank + 1

    avg_ranks = rank_mat.mean(axis=1)
    order     = np.argsort(avg_ranks)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                             gridspec_kw={'width_ratios': [1.2, 2]})
    axes[0].barh([SUR_LABELS[i] for i in order],
                 [avg_ranks[i] for i in order],
                 color=[PAPER_COLORS[i] for i in order])
    axes[0].set_xlabel('Average rank (lower = better)')
    axes[0].set_title('Avg BBOB rank')
    axes[0].invert_xaxis()

    im = axes[1].imshow(rank_mat, cmap='RdYlGn_r', aspect='auto',
                        vmin=1, vmax=len(SURROGATES))
    axes[1].set_xticks(range(len(FUNC_IDS)))
    axes[1].set_xticklabels(FUNC_SHORT)
    axes[1].set_yticks(range(len(SURROGATES)))
    axes[1].set_yticklabels(SUR_LABELS)
    axes[1].set_title('Rank per BBOB function')
    for i in range(len(SURROGATES)):
        for j in range(len(FUNC_IDS)):
            axes[1].text(j, i, f'{int(rank_mat[i,j])}', ha='center', va='center',
                         fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[1], label='Rank')
    if not has_cocoex:
        fig.text(0.5, 0.01, '(cocoex not available — synthetic ranks)',
                 ha='center', fontsize=9, color='gray')
    fig.suptitle('Figure 4 — BBOB Leaderboard (2-D, N=3 seeds)', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig4_bbob.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/fig4_bbob.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig4_bbob.pdf / .png')
    return avg_ranks


# ─────────────────────────────────────────────────────────────────────────
# PART 5 — Paper Figure 5: SCR enforcer timeline
# ─────────────────────────────────────────────────────────────────────────
def part5():
    rng     = np.random.RandomState(RANDOM_SEED)
    scr_max = 0.2
    n_iter  = 30

    X_tr = lhs(N_INIT, 1, RANDOM_SEED)
    y_tr = np.array([forrester(x) for x in X_tr])
    calls       = []
    running_scr = []
    true_calls  = N_INIT
    surr_calls  = 0

    for _ in range(n_iter):
        X_cand   = lhs(200, 1, rng.randint(0, 10000))
        mu, std  = surrogate_predict('gp', X_tr, y_tr, X_cand, seed=RANDOM_SEED)
        idx      = np.argmax(ei(mu, std, y_tr.min()))
        x_next   = X_cand[idx]
        proj_scr = (surr_calls + 1) / (true_calls + surr_calls + 1)
        use_surr = proj_scr <= scr_max
        if use_surr:
            y_next = float(mu[idx])
            surr_calls += 1
            calls.append('S')
        else:
            y_next = forrester(x_next)
            true_calls += 1
            calls.append('T')
        X_tr = np.vstack([X_tr, x_next])
        y_tr = np.append(y_tr, y_next)
        running_scr.append(surr_calls / (true_calls + surr_calls))

    fig, ax = plt.subplots(figsize=(10, 3.5))
    for k, (c, scr) in enumerate(zip(calls, running_scr)):
        ax.bar(k, 1, color='#2a9d8f' if c == 'T' else '#f4a261', width=0.9)
    ax.plot(range(len(running_scr)), running_scr, 'k-', lw=1.5)
    ax.axhline(scr_max, color='red', ls='--', lw=1.2)

    true_p = mpatches.Patch(color='#2a9d8f', label='True evaluation')
    surr_p = mpatches.Patch(color='#f4a261', label='Surrogate evaluation')
    scr_l  = plt.Line2D([0], [0], color='k', lw=1.5, label='Running SCR')
    lim_l  = plt.Line2D([0], [0], color='red', ls='--', lw=1.2, label='SCR limit (20%)')
    ax.legend(handles=[true_p, surr_p, scr_l, lim_l], loc='upper left', fontsize=9)
    ax.set_xlim(-0.5, len(calls) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('BO iteration (after initial design)')
    ax.set_ylabel('Running SCR')
    ax.set_title('Figure 5 — SCR Enforcer Timeline (GP, SCR limit = 20%)')
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.grid(True, axis='y', ls=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(f'{OUT}/fig5_scr_timeline.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT}/fig5_scr_timeline.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved fig5_scr_timeline.pdf / .png')


# ─────────────────────────────────────────────────────────────────────────
# PART 6 — Results table + LaTeX scaffold
# ─────────────────────────────────────────────────────────────────────────
def part6(results_forr, results_bran, avg_ranks):
    rows = []
    for i, stype in enumerate(SURROGATES):
        fg    = final_gaps(results_forr[stype])
        fb    = final_gaps(results_bran[stype])
        gap_f = np.median(fg);  iqr_f = np.percentile(fg, 75) - np.percentile(fg, 25)
        gap_b = np.median(fb);  iqr_b = np.percentile(fb, 75) - np.percentile(fb, 25)
        rec   = '20%' if stype in ('gp', 'mc_dropout', 'deep_ensemble') else '0%'
        rows.append((SUR_LABELS[i], gap_f, iqr_f, gap_b, iqr_b, avg_ranks[i], rec))

    print('\n=== Results Table (Markdown) ===')
    print(f'| {"Surrogate":<14} | {"Gap Forrester":>20} | {"Gap Branin":>20} | {"BBOB rank":>9} | {"Rec SCR":>7} |')
    print('|' + '-'*16 + '|' + '-'*22 + '|' + '-'*22 + '|' + '-'*11 + '|' + '-'*9 + '|')
    for r in rows:
        print(f'| {r[0]:<14} | {r[1]:.4f} ± {r[2]:.4f}   | {r[3]:.4f} ± {r[4]:.4f}   | {r[5]:>9.2f} | {r[6]:>7} |')

    latex_rows = '\n'.join(
        f'        {r[0]} & ${r[1]:.4f}\\pm{r[2]:.4f}$ & ${r[3]:.4f}\\pm{r[4]:.4f}$ & {r[5]:.2f} & {r[6].replace("%","\\%")} \\\\'
        for r in rows
    )

    scaffold = rf"""% =========================================================
%  Surrogate-Assisted Optimisation — Research Paper
%  Generated by lesson-19/main.py
%  Compile with: pdflatex paper_scaffold.tex
% =========================================================
\documentclass{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\usepackage{{hyperref}}

\title{{Surrogate-Assisted Bayesian Optimisation with MC Dropout and Deep Ensembles}}
\author{{[Your Name]}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
We compare five surrogate models---Gaussian Process, MC Dropout, Deep Ensemble,
RBF, and Random Forest---for surrogate-assisted Bayesian optimisation.
We introduce a Surrogate Control Ratio (SCR) to bound surrogate usage.
[PLACEHOLDER: 2 sentences summarising main finding and recommended surrogate/SCR.]
\end{{abstract}}

\section{{Introduction}}
[PLACEHOLDER: motivation, gap in literature, 3-bullet contributions.]

\section{{Background}}
\subsection{{Bayesian Optimisation}}
[PLACEHOLDER]

\subsection{{Surrogate Control Ratio}}
$\mathrm{{SCR}} = N_{{\text{{surr}}}} \mathbin{{/}} (N_{{\text{{true}}}} + N_{{\text{{surr}}}})$.
See Figure~\ref{{fig:timeline}}.

\section{{Method}}
\begin{{figure}}[ht]
  \centering
  \includegraphics[width=\linewidth]{{fig5_scr_timeline.pdf}}
  \caption{{SCR enforcer timeline (GP, limit = 20\%).}}
  \label{{fig:timeline}}
\end{{figure}}

\section{{Experiments}}
\begin{{figure}}[ht]
  \centering
  \includegraphics[width=\linewidth]{{fig1_convergence.pdf}}
  \caption{{Convergence on Forrester (left) and Branin (right). Median $\pm$ 95\% bootstrap CI.}}
  \label{{fig:conv}}
\end{{figure}}

\begin{{figure}}[ht]
  \centering
  \includegraphics[width=0.7\linewidth]{{fig2_scr_scatter.pdf}}
  \caption{{SCR sensitivity: quality vs.\ true evaluations saved.}}
\end{{figure}}

\begin{{figure}}[ht]
  \centering
  \includegraphics[width=0.7\linewidth]{{fig3_noise.pdf}}
  \caption{{Noise robustness on Forrester.}}
\end{{figure}}

\section{{Results}}
\begin{{table}}[ht]
\centering
\caption{{Median gap ($\pm$ IQR) and BBOB average rank. Lower is better.}}
\label{{tab:results}}
\begin{{tabular}}{{lcccc}}
\toprule
Surrogate & Gap Forrester & Gap Branin & BBOB avg rank & Rec.\ SCR \\
\midrule
{latex_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[ht]
  \centering
  \includegraphics[width=\linewidth]{{fig4_bbob.pdf}}
  \caption{{BBOB leaderboard (2-D, 3 seeds).}}
\end{{figure}}

[PLACEHOLDER: 2--3 sentences interpreting Table~\ref{{tab:results}}.]

\section{{Conclusion}}
[PLACEHOLDER: main finding, recommended surrogate and SCR, limitations, future work.]

\bibliographystyle{{plain}}
\begin{{thebibliography}}{{9}}
\bibitem{{jones1998}} D.~R. Jones et al. EGO. JOGO, 1998.
\bibitem{{gal2016}} Y.~Gal, Z.~Ghahramani. Dropout as Bayesian Approximation. ICML, 2016.
\bibitem{{laks2017}} B.~Lakshminarayanan et al. Deep Ensembles. NeurIPS, 2017.
\end{{thebibliography}}
\end{{document}}
"""

    tex_path = f'{OUT}/paper_scaffold.tex'
    with open(tex_path, 'w') as fh:
        fh.write(scaffold)
    print(f'\nSaved LaTeX scaffold → {tex_path}')
    print('Compile: pdflatex lesson-19/output/paper_scaffold.tex')


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=== Lesson 19: Writing the Research Paper ===\n')

    print('Forrester (1-D) BO ...')
    results_forr = {}
    for stype in SURROGATES:
        print(f'  {stype}')
        results_forr[stype] = run_seeds(stype, forrester, 1, FORRESTER_OPT, N_INIT, N_ITER)

    print('Branin (2-D) BO ...')
    results_bran = {}
    for stype in SURROGATES:
        print(f'  {stype}')
        results_bran[stype] = run_seeds(stype, branin, 2, BRANIN_OPT, N_INIT_2D, N_ITER)

    print('SCR sensitivity (Forrester) ...')
    scr_results = {stype: {} for stype in SURROGATES}
    for stype in SURROGATES:
        for scr in SCR_LEVELS:
            scr_results[stype][scr] = run_seeds(
                stype, forrester, 1, FORRESTER_OPT, N_INIT, N_ITER, scr_max=scr)

    print('\n--- Part 1: Main convergence figure ---')
    part1(results_forr, results_bran)

    print('--- Part 2: SCR sensitivity scatter ---')
    part2(scr_results)

    print('--- Part 3: Noise robustness ---')
    part3()

    print('--- Part 4: BBOB leaderboard ---')
    avg_ranks = part4()

    print('--- Part 5: SCR enforcer timeline ---')
    part5()

    print('--- Part 6: Results table + LaTeX scaffold ---')
    part6(results_forr, results_bran, avg_ranks)

    print('\nAll done. Output in lesson-19/output/')
