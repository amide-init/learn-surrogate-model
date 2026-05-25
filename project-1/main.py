import os
import re
import numpy as np
import cocoex
import cma

from scipy.stats.qmc import LatinHypercube
from surrogate import DeepEnsemble

RANDOM_SEED = 42

# =========================================================
# CONFIG
# =========================================================

BUDGET_FACTOR = 1000

# Initial real samples
N_INIT_FACTOR = 20

# Retrain surrogate every N new real points
RETRAIN_EVERY = 10

# Number of BBOB instances
N_INSTANCES = 15

DIMENSIONS = [2]
FUNCTION_IDS = [1, 2, 3]

# Exploration coefficient for LCB
KAPPA = 2.0

np.random.seed(RANDOM_SEED)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# LHS SAMPLING
# =========================================================

def lhs_sample(n, lb, ub, seed):
    sampler = LatinHypercube(d=len(lb), seed=seed)
    unit = sampler.random(n)
    return lb + unit * (ub - lb)


# =========================================================
# NORMALIZATION HELPERS
# =========================================================

def normalize_y(y):
    mean = np.mean(y)
    std = np.std(y) + 1e-8
    y_norm = (y - mean) / std
    return y_norm, mean, std


def denormalize_y(y_norm, mean, std):
    return y_norm * std + mean


# =========================================================
# DYNAMIC SCR
# =========================================================

def get_scr(n_real, budget):
    """
    Early stage:
        more real evaluations

    Later:
        more surrogate usage
    """

    progress = n_real / budget

    if progress < 0.3:
        return 0.8

    elif progress < 0.6:
        return 0.5

    else:
        return 0.2


# =========================================================
# MAIN OPTIMIZATION LOOP
# =========================================================

def run_instance(fun, dim, budget, seed):

    lb = fun.lower_bounds
    ub = fun.upper_bounds

    # -----------------------------------------------------
    # INITIAL DESIGN
    # -----------------------------------------------------

    n_init = N_INIT_FACTOR * dim

    X = lhs_sample(n_init, lb, ub, seed)

    y = np.array([fun(x) for x in X])

    best = float(y.min())

    history = [best] * n_init

    # -----------------------------------------------------
    # NORMALIZE TARGETS
    # -----------------------------------------------------

    y_norm, y_mean, y_std = normalize_y(y)

    # -----------------------------------------------------
    # TRAIN INITIAL SURROGATE
    # -----------------------------------------------------

    model = DeepEnsemble(d=dim)

    model.fit(X, y_norm)

    # -----------------------------------------------------
    # CMA-ES
    # -----------------------------------------------------

    x0 = X[y.argmin()]

    sigma0 = 0.3 * np.mean(ub - lb)

    es = cma.CMAEvolutionStrategy(
        x0.tolist(),
        sigma0,
        {
            "bounds": [lb.tolist(), ub.tolist()],
            "seed": seed,
            "verbose": -9,
            "tolx": 1e-12,
            "tolfun": 1e-12,
        }
    )

    n_real = n_init
    n_since_retrain = 0

    # =====================================================
    # LOOP
    # =====================================================

    while not es.stop() and n_real < budget:

        # -------------------------------------------------
        # ASK CMA
        # -------------------------------------------------

        solutions = es.ask()

        candidates = np.array(solutions)

        # -------------------------------------------------
        # SURROGATE PREDICTION
        # -------------------------------------------------

        mu_norm, sigma_norm = model.predict(candidates)

        mu = denormalize_y(mu_norm, y_mean, y_std)

        sigma = sigma_norm * y_std

        # -------------------------------------------------
        # LOWER CONFIDENCE BOUND
        # -------------------------------------------------

        acquisition = mu - KAPPA * sigma

        # -------------------------------------------------
        # DYNAMIC SCR
        # -------------------------------------------------

        SCR = get_scr(n_real, budget)

        n_eval = max(1, round(SCR * len(candidates)))

        # Evaluate best acquisition candidates
        real_idx = np.argsort(acquisition)[:n_eval]

        # -------------------------------------------------
        # IMPORTANT:
        # Use surrogate predictions for ALL candidates
        # instead of harsh penalties
        # -------------------------------------------------

        fitvals = mu.tolist()

        new_X = []
        new_y = []

        # -------------------------------------------------
        # REAL EVALUATIONS
        # -------------------------------------------------

        for i in real_idx:

            if n_real >= budget:
                break

            f = float(fun(candidates[i]))

            fitvals[i] = f

            n_real += 1

            best = min(best, f)

            history.append(best)

            new_X.append(candidates[i])

            new_y.append(f)

            n_since_retrain += 1

        # -------------------------------------------------
        # TELL CMA
        # -------------------------------------------------

        es.tell(solutions, fitvals)

        # -------------------------------------------------
        # UPDATE DATASET
        # -------------------------------------------------

        if len(new_X) > 0:

            X = np.vstack([X, new_X])

            y = np.append(y, new_y)

            # ---------------------------------------------
            # RETRAIN SURROGATE
            # ---------------------------------------------

            if n_since_retrain >= RETRAIN_EVERY:

                y_norm, y_mean, y_std = normalize_y(y)

                model.fit(X, y_norm)

                n_since_retrain = 0

    # =====================================================
    # PAD HISTORY
    # =====================================================

    history += [history[-1]] * (budget - len(history))

    return np.array(history[:budget])


# =========================================================
# MAIN
# =========================================================

def main():

    suite_opts = (
        f"function_indices: {','.join(map(str, FUNCTION_IDS))} "
        f"dimensions: {' '.join(map(str, DIMENSIONS))} "
        f"instance_indices: 1-{N_INSTANCES}"
    )

    suite = cocoex.Suite("bbob", "", suite_opts)

    observer = cocoex.Observer(
        "bbob",
        "result_folder: coco_output"
    )

    collected = {}

    for fun in suite:

        fun.add_observer(observer)

        dim = fun.dimension

        m = re.search(r"f(\d+)_i(\d+)", fun.id)

        fun_num = int(m.group(1))
        inst_num = int(m.group(2))

        budget = BUDGET_FACTOR * dim

        seed = RANDOM_SEED + fun_num * 1000 + inst_num

        print(
            f"f{fun_num:02d} "
            f"d{dim:02d} "
            f"i{inst_num:02d} ",
            end="",
            flush=True
        )

        history = run_instance(
            fun=fun,
            dim=dim,
            budget=budget,
            seed=seed
        )

        print(f"best={history[-1]:.3e}")

        collected.setdefault((fun_num, dim), []).append(history)

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    for (fun_num, dim), histories in collected.items():

        path = os.path.join(
            RESULTS_DIR,
            f"dim_{dim}_fun_{fun_num}.npy"
        )

        np.save(path, np.array(histories))

    print(f"\nResults saved to {RESULTS_DIR}/")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    main()