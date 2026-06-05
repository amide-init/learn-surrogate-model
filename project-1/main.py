"""
AFN-CMA-ES  —  clean AFN(4) base version
=========================================

Key design rules
----------------
1. Normal surrogate-assisted generations ALWAYS call es.tell() with the
   FULL CMA population using hybrid scores (surrogate rank for unevaluated
   candidates, empirical true-fitness rank for evaluated ones).

2. es.tell() is NEVER called with only the k evaluated candidates alone.

3. The final incomplete true-CMA generation is handled safely:
   if fewer than one full CMA population remains, leftover points are
   evaluated to fill the COCO budget and es.tell() is skipped.

4. fun_num is used only for logging / seeding — never for algorithmic decisions.
"""

import os
import re
from typing import Dict, List, Tuple

import cma
import cocoex
import numpy as np
from scipy.stats import kendalltau
from scipy.stats.qmc import LatinHypercube

from surrogate import DeepEnsemble


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_SEED = 42

BUDGET_FACTOR    = 250       # COCO budget = BUDGET_FACTOR * dimension
N_INIT_FACTOR    = 20        # 20*d initial LHS points; for d=2 → 40
RETRAIN_EVERY    = 6         # retrain after this many real evaluations
LOCAL_TRAIN_SIZE = 250       # local training subset around CMA mean

N_INSTANCES  = 15
DIMENSIONS   = [2]
FUNCTION_IDS = list(range(1, 25))

# At most 50% of a CMA generation uses true evaluations.
TRUE_EVAL_FRACTION = 0.40

# Polishing: automatic trigger — no function-number checks.
POLISH_START_FRAC     = 0.70   # don't polish before 70% budget used
POLISH_BEST_THRESHOLD = 1e-4   # trigger if best is already very small
POLISH_SIGMA_FRAC     = 3e-3   # trigger if CMA sigma < 0.3% of domain
SURROGATE_BAD_TAU     = 0.35   # tau threshold for "unreliable surrogate"

# Restart: stagnation limit (no function-specific values).
STAGNATION_LIMIT = 45

np.random.seed(RANDOM_SEED)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# UTILITIES
# =========================================================

def lhs_sample(n: int, lb: np.ndarray, ub: np.ndarray, seed: int) -> np.ndarray:
    sampler = LatinHypercube(d=len(lb), seed=seed)
    return lb + sampler.random(n) * (ub - lb)


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return np.zeros_like(values, dtype=float)
    return np.argsort(np.argsort(values)).astype(float) / float(values.size - 1)


def empirical_rank01(y_ref: np.ndarray, f: float) -> float:
    y_ref = np.sort(np.asarray(y_ref, dtype=float))
    if y_ref.size <= 1:
        return 0.0
    return float(np.searchsorted(y_ref, f, side="right") / y_ref.size)


def make_rank_targets(y: np.ndarray) -> np.ndarray:
    return rank01(np.asarray(y, dtype=float)).astype(np.float32)


def get_local_dataset(
    X: np.ndarray, y: np.ndarray, center: np.ndarray, max_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    dist = np.linalg.norm(X - center, axis=1)
    idx  = np.argsort(dist)[: min(max_size, len(X))]
    return X[idx], y[idx]


def make_cma(x0: np.ndarray, sigma0: float, lb: np.ndarray, ub: np.ndarray, seed: int):
    return cma.CMAEvolutionStrategy(
        x0.tolist(),
        float(sigma0),
        {
            "bounds":      [lb.tolist(), ub.tolist()],
            "seed":        int(seed),
            "verbose":     -9,
            "CMA_mirrors": 0,
        },
    )


# =========================================================
# POLISHING TRIGGER  (automatic, no fun_num)
# =========================================================

def should_use_polish_mode(
    n_real: int,
    budget: int,
    best: float,
    es,
    domain_range: float,
    surrogate_tau: float,
) -> bool:
    if n_real < int(POLISH_START_FRAC * budget):
        return False
    close_to_target = best    < POLISH_BEST_THRESHOLD
    small_sigma     = float(es.sigma) < POLISH_SIGMA_FRAC * domain_range
    bad_surrogate   = surrogate_tau   < SURROGATE_BAD_TAU
    return close_to_target or small_sigma or bad_surrogate


# =========================================================
# RESCUE TRIGGER  (automatic, no fun_num)
# =========================================================

def should_protect_local_convergence(best: float, es, domain_range: float) -> bool:
    """
    True when the run is already close to target or CMA has converged tightly.
    Prevents probe-rescue from destroying a near-solved local descent.
    """
    close_enough      = best                < 1e-3
    very_small_sigma  = float(es.sigma)     < 1e-3 * domain_range
    return close_enough or very_small_sigma


def probe_rescue(
    fun,
    lb: np.ndarray,
    ub: np.ndarray,
    dim: int,
    seed: int,
    n_probe: int,
    best: float,
    budget: int,
    n_real: int,
):
    """
    Evaluate n_probe random global points.  If any beats the current best,
    return the best probe as a restart candidate; otherwise return None.

    Only restarts CMA when a probe genuinely improves the run — safer than
    unconditional global restart.
    """
    rng     = np.random.default_rng(seed)
    probe_X: List[np.ndarray] = []
    probe_y: List[float]      = []

    for _ in range(min(n_probe, budget - n_real)):
        x = lb + rng.random(dim) * (ub - lb)
        f = float(fun(x))
        probe_X.append(np.asarray(x, dtype=float))
        probe_y.append(f)
        n_real += 1

    if not probe_y:
        return None, None, n_real

    probe_y_arr = np.asarray(probe_y)
    probe_X_arr = np.asarray(probe_X)
    best_idx    = int(np.argmin(probe_y_arr))

    if probe_y_arr[best_idx] < best:
        return probe_X_arr[best_idx], float(probe_y_arr[best_idx]), n_real

    return None, None, n_real


# =========================================================
# CANDIDATE SELECTION  (automatic beta from tau + stagnation)
# =========================================================

def select_real_candidates(
    mu: np.ndarray,
    sig: np.ndarray,
    candidates: np.ndarray,
    es,
    k: int,
    surrogate_tau: float,
    no_improve_gens: int,
) -> Tuple[List[int], np.ndarray]:
    # Beta increases as surrogate quality falls or search stagnates.
    if surrogate_tau < 0.35:
        beta = 0.70
    elif surrogate_tau < 0.55:
        beta = 0.35
    else:
        beta = 0.15

    if no_improve_gens >= 12:
        beta = max(beta, 1.00)   # strong exploration after 12 stagnant gens

    acq      = mu - beta * sig
    selected: List[int] = []

    def add(index: int) -> None:
        index = int(index)
        if index not in selected and len(selected) < k:
            selected.append(index)

    add(int(np.argmin(acq)))   # best acquisition
    add(int(np.argmin(mu)))    # best predicted mean

    # Uncertainty / diversity probes when model is doubtful or stagnant.
    if len(selected) < k and (surrogate_tau < 0.55 or no_improve_gens >= 10):
        add(int(np.argmax(sig)))

    if len(selected) < k and no_improve_gens >= 15:
        dists = np.linalg.norm(candidates - np.asarray(es.mean), axis=1)
        add(int(np.argmax(dists)))

    # Fill remaining slots by acquisition value.
    for idx in np.argsort(acq):
        add(int(idx))
        if len(selected) == k:
            break

    return selected, acq


# =========================================================
# MAIN OPTIMISATION LOOP
# =========================================================

def run_instance(
    fun, dim: int, budget: int, seed: int, fun_num: int | None = None
) -> np.ndarray:

    lb           = np.asarray(fun.lower_bounds, dtype=float)
    ub           = np.asarray(fun.upper_bounds, dtype=float)
    domain_range = float(np.mean(ub - lb))

    # ----------------------------------------------------------
    # Phase 1 — LHS initial design
    # ----------------------------------------------------------
    n_init = min(N_INIT_FACTOR * dim, max(2 * dim + 4, budget // 5))
    X      = lhs_sample(n_init, lb, ub, seed)
    y      = np.asarray([float(fun(x)) for x in X], dtype=float)

    best    = float(np.min(y))
    history: List[float] = list(np.minimum.accumulate(y))
    n_real  = int(n_init)

    model = DeepEnsemble(d=dim)
    model.fit(X, make_rank_targets(y))

    sigma0 = domain_range / 6.0
    x0     = X[int(np.argmin(y))]
    es     = make_cma(x0, sigma0, lb, ub, seed)

    n_since_retrain = 0
    no_improve_gens = 0
    restart_count   = 0
    surrogate_tau   = 0.50

    # ==========================================================
    # Main loop
    # ==========================================================

    while n_real < budget:

        # ------------------------------------------------------
        # Restart: CMA converged naturally OR stagnation limit
        # ------------------------------------------------------
        if es.stop() or no_improve_gens >= STAGNATION_LIMIT:

            restart_count   += 1
            no_improve_gens  = 0
            rng              = np.random.default_rng(seed + 1000 * restart_count)
            x_best           = X[int(np.argmin(y))]

            if restart_count % 3 == 0:
                x_start       = lb + rng.random(dim) * (ub - lb)
                sigma_restart = sigma0
            else:
                x_start       = x_best
                sigma_restart = max(0.03 * domain_range, 0.35 * sigma0)

            es = make_cma(x_start, sigma_restart, lb, ub, seed + restart_count)

        # ------------------------------------------------------
        # Polish mode — pure CMA-ES (automatic trigger, no fun_num)
        # ------------------------------------------------------
        if should_use_polish_mode(n_real, budget, best, es, domain_range, surrogate_tau):

            solutions = es.ask()
            lam       = len(solutions)
            n_eval    = min(lam, budget - n_real)

            true_vals: List[float] = []
            prev_best = best
            new_X: List[np.ndarray] = []
            new_y: List[float]      = []

            for x in solutions[:n_eval]:
                f = float(fun(x))
                true_vals.append(f)
                best = min(best, f)
                history.append(best)
                new_X.append(np.asarray(x, dtype=float))
                new_y.append(f)
                n_real          += 1
                n_since_retrain += 1

            # Safe tell: only call when a FULL generation was evaluated.
            # Final incomplete generation (n_eval < lam) skips tell()
            # to avoid the pycma "population size too small" crash.
            if n_eval == lam:
                es.tell(solutions, true_vals)

            if best < prev_best:
                no_improve_gens = 0
            else:
                no_improve_gens += 1

            if new_X:
                X = np.vstack([X, np.asarray(new_X)])
                y = np.append(y, np.asarray(new_y, dtype=float))

            if n_since_retrain >= RETRAIN_EVERY and n_real < budget:
                X_local, y_local = get_local_dataset(
                    X, y, np.asarray(es.mean), LOCAL_TRAIN_SIZE
                )
                model.fit(X_local, make_rank_targets(y_local))
                n_since_retrain = 0

            continue

        # ------------------------------------------------------
        # Probe-rescue — fires only when strongly stuck and not near solution
        #
        # Trigger (all must hold):
        #   - 60% of budget used
        #   - no improvement for >= 35 gens
        #   - best > 1e-2  (not already close)
        #   - NOT in local convergence (sigma not tiny, best not < 1e-3)
        #
        # Evaluates 6 random probe points.  CMA restarts ONLY if a probe
        # actually beats the current best — never destroys a converging run.
        # ------------------------------------------------------
        if (
            n_real          >= int(0.60 * budget) and
            no_improve_gens >= 35                 and
            best            >  1e-2               and
            not should_protect_local_convergence(best, es, domain_range)
        ):
            x_probe, f_probe, n_real = probe_rescue(
                fun=fun, lb=lb, ub=ub, dim=dim,
                seed=seed + 9000 + restart_count,
                n_probe=6, best=best, budget=budget, n_real=n_real,
            )

            if x_probe is not None:
                restart_count += 1
                X    = np.vstack([X, x_probe.reshape(1, -1)])
                y    = np.append(y, f_probe)
                best = min(best, f_probe)
                history.append(best)

                es = make_cma(
                    x_probe, domain_range / 4.0, lb, ub,
                    seed + 9000 + restart_count,
                )

                no_improve_gens = 0
                surrogate_tau   = 0.50
                continue

        # ------------------------------------------------------
        # Normal AFN-CMA-ES generation
        # ------------------------------------------------------
        solutions  = es.ask()
        candidates = np.asarray(solutions, dtype=float)
        lam        = len(solutions)

        k = max(2, int(np.floor(TRUE_EVAL_FRACTION * lam)))
        k = min(k, budget - n_real, lam)

        mu, sig = model.predict(candidates)

        top_k_idx, acq = select_real_candidates(
            mu=mu, sig=sig, candidates=candidates, es=es,
            k=k, surrogate_tau=surrogate_tau, no_improve_gens=no_improve_gens,
        )

        prev_best = best
        new_X: List[np.ndarray] = []
        new_y: List[float]      = []
        eval_map: Dict[int, float] = {}
        pred_for_tau: List[float]  = []

        for idx in top_k_idx:
            if n_real >= budget:
                break
            f              = float(fun(candidates[idx]))
            eval_map[idx]  = f
            pred_for_tau.append(float(mu[idx]))
            best           = min(best, f)
            history.append(best)
            new_X.append(candidates[idx])
            new_y.append(f)
            n_real          += 1
            n_since_retrain += 1

        # ----------------------------------------------------------
        # Full-population hybrid CMA update  (AFN(4) rule)
        #
        # Unevaluated candidates → surrogate acquisition rank
        # Evaluated candidates   → empirical true-fitness rank
        #
        # es.tell() ALWAYS receives the full population (lam solutions).
        # We never call tell() with only the k evaluated candidates.
        # ----------------------------------------------------------
        # Use predicted mean rank for CMA update — safer for covariance
        # adaptation than acquisition rank (acq = mu - beta*sig).
        # Acquisition is used only to SELECT which candidates to evaluate.
        cma_scores = rank01(mu)

        for idx, f in eval_map.items():
            cma_scores[idx] = empirical_rank01(y, f)

        if surrogate_tau < SURROGATE_BAD_TAU and len(eval_map) > 0:
            unevaluated                = np.ones(lam, dtype=bool)
            unevaluated[list(eval_map.keys())] = False
            cma_scores[unevaluated]   += 0.25   # mild penalty; 0.50 distorted CMA ranking

        es.tell(solutions, cma_scores.tolist())

        # Stagnation tracking
        if best < prev_best:
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        if new_X:
            if len(new_y) >= 3:
                tau_val, _ = kendalltau(
                    np.asarray(pred_for_tau), np.asarray(new_y)
                )
                if np.isfinite(tau_val):
                    surrogate_tau = float(0.7 * surrogate_tau + 0.3 * tau_val)

            X = np.vstack([X, np.asarray(new_X)])
            y = np.append(y, np.asarray(new_y, dtype=float))

        if n_since_retrain >= RETRAIN_EVERY and n_real < budget:
            X_local, y_local = get_local_dataset(
                X, y, np.asarray(es.mean), LOCAL_TRAIN_SIZE
            )
            model.fit(X_local, make_rank_targets(y_local))
            n_since_retrain = 0

    # Pad to exactly `budget` entries for COCO.
    if len(history) < budget:
        history.extend([history[-1]] * (budget - len(history)))

    return np.asarray(history[:budget], dtype=float)


# =========================================================
# COCO ENTRY POINT
# =========================================================

def main() -> None:

    suite_opts = (
        f"function_indices: {','.join(map(str, FUNCTION_IDS))} "
        f"dimensions: {','.join(map(str, DIMENSIONS))} "
        f"instance_indices: 1-{N_INSTANCES}"
    )

    # year: 2010 gives instances 1–15 consecutively (verified).
    suite = cocoex.Suite("bbob", "year: 2010", suite_opts)

    result_folder = (
        f"AFN4_clean_safeTell_"
        f"{BUDGET_FACTOR}d_"
        f"dim{'_'.join(map(str, DIMENSIONS))}"
    )

    observer = cocoex.Observer(
        "bbob",
        f"result_folder: {result_folder} algorithm_name: AFN-CMA-ES",
    )

    collected: Dict[Tuple[int, int], List[np.ndarray]] = {}

    for fun in suite:

        fun.add_observer(observer)

        dim      = int(fun.dimension)
        match    = re.search(r"f(\d+)_i(\d+)", fun.id)
        fun_num  = int(match.group(1))
        inst_num = int(match.group(2))
        budget   = BUDGET_FACTOR * dim
        seed     = RANDOM_SEED + fun_num * 1000 + inst_num

        print(f"f{fun_num:02d} d{dim:02d} i{inst_num:03d} ", end="", flush=True)

        history = run_instance(
            fun=fun, dim=dim, budget=budget, seed=seed,
            fun_num=fun_num,  # logging/seeding only — not used for decisions
        )

        print(f"best={history[-1]:.3e}")
        collected.setdefault((fun_num, dim), []).append(history)

    for (fun_num, dim), histories in collected.items():
        out_path = os.path.join(RESULTS_DIR, f"dim_{dim}_fun_{fun_num}.npy")
        np.save(out_path, np.asarray(histories, dtype=float))

    print("\n===================================")
    print(f"Results saved to:\n{RESULTS_DIR}")
    print(f"\nCOCO output saved to:\n{result_folder}")
    print("===================================")


if __name__ == "__main__":
    main()
