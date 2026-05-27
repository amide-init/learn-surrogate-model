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

BUDGET_FACTOR = 250

N_INIT_FACTOR = 10

RETRAIN_EVERY = 10

N_INSTANCES = 15

DIMENSIONS = [2]

FUNCTION_IDS = [2]

KAPPA = 1.0

PRESCREEN_POP = 15

LOCAL_TRAIN_SIZE = 200

np.random.seed(RANDOM_SEED)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# LHS
# =========================================================

def lhs_sample(n, lb, ub, seed):

    sampler = LatinHypercube(
        d=len(lb),
        seed=seed
    )

    unit = sampler.random(n)

    return lb + unit * (ub - lb)


# =========================================================
# SCR
# =========================================================

def get_scr(n_real, budget):

    progress = n_real / budget

    if progress < 0.3:
        return 0.5

    elif progress < 0.7:
        return 0.3

    else:
        return 0.1


# =========================================================
# LOCAL DATASET
# =========================================================

def get_local_dataset(X, y, center, k):

    dist = np.linalg.norm(X - center, axis=1)

    idx = np.argsort(dist)[:min(k, len(X))]

    return X[idx], y[idx]


# =========================================================
# MAIN LOOP
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
    # MODEL
    # -----------------------------------------------------

    model = DeepEnsemble(d=dim)

    # -----------------------------------------------------
    # RANK-BASED TRAINING TARGET
    # -----------------------------------------------------

    ranks = np.argsort(np.argsort(y)).astype(np.float32)

    y_train = ranks / max(ranks.max(), 1)

    model.fit(X, y_train)

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
        }
    )

    n_real = n_init

    n_since_retrain = 0

    refit_count = 0

    # =====================================================
    # LOOP
    # =====================================================

    while not es.stop() and n_real < budget:

        # -------------------------------------------------
        # PRESCREENING
        # -------------------------------------------------

        solutions = es.ask(PRESCREEN_POP)

        candidates = np.array(solutions)

        # -------------------------------------------------
        # SURROGATE PREDICTION
        # -------------------------------------------------

        mu, sigma = model.predict(candidates)

        # ---------------------------------------------
        # DYNAMIC KAPPA
        # ---------------------------------------------

        progress = n_real / budget

        if progress < 0.5:
            kappa = 1.0
        else:
            kappa = 0.3

        acquisition = mu - kappa * sigma

        # -------------------------------------------------
        # SCR
        # -------------------------------------------------

        SCR = get_scr(n_real, budget)

        n_eval = max(
            1,
            round(SCR * len(candidates))
        )

        # -------------------------------------------------
        # SELECT BEST CANDIDATES ONLY
        # -------------------------------------------------

        best_idx = np.argsort(acquisition)[:n_eval // 2]

        uncertain_idx = np.argsort(-sigma)[:n_eval // 2]

        real_idx = np.unique(
            np.concatenate([
                best_idx,
                uncertain_idx
            ])
        )

        # -------------------------------------------------
        # SAFE CMA FITNESS
        # -------------------------------------------------

        # ---------------------------------------------
# SURROGATE FITNESS
# ---------------------------------------------

        fitvals = mu.copy().tolist()

        new_X = []

        new_y = []

        # ---------------------------------------------
        # REAL EVALUATIONS
        # ---------------------------------------------

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
        # CMA UPDATE
        # -------------------------------------------------

        es.tell(solutions, fitvals)

        # -------------------------------------------------
        # UPDATE DATA
        # -------------------------------------------------

        if len(new_X) > 0:

            X = np.vstack([X, new_X])

            y = np.append(y, new_y)

            # ---------------------------------------------
            # LOCAL RETRAINING
            # ---------------------------------------------

            if n_since_retrain >= RETRAIN_EVERY:

                center = np.array(es.mean)

                X_local, y_local = get_local_dataset(
                    X,
                    y,
                    center,
                    LOCAL_TRAIN_SIZE
                )

                # -----------------------------------------
                # RANK TRAINING
                # -----------------------------------------

                local_ranks = np.argsort(
                    np.argsort(y_local)
                ).astype(np.float32)

                y_local_train = (
                    local_ranks /
                    max(local_ranks.max(), 1)
                )

                model.fit(
                    X_local,
                    y_local_train
                )

                n_since_retrain = 0

                refit_count += 1

                # print(
                #     f"refit : {refit_count}"
                # )

    # =====================================================
    # PAD HISTORY
    # =====================================================

    history += [history[-1]] * (
        budget - len(history)
    )

    return np.array(history[:budget])


# =========================================================
# MAIN
# =========================================================

def main():

    suite_opts = (
        f"function_indices: "
        f"{','.join(map(str, FUNCTION_IDS))} "
        f"dimensions: "
        f"{' '.join(map(str, DIMENSIONS))} "
        f"instance_indices: "
        f"1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    )

    suite = cocoex.Suite(
        "bbob",
        "",
        suite_opts
    )

    result_folder = (
        f"deepensemble_rank_"
        f"{BUDGET_FACTOR}d_"
        f"dim{'_'.join(map(str, DIMENSIONS))}"
    )

    observer = cocoex.Observer(
        "bbob",
        f"result_folder: {result_folder}"
    )

    collected = {}

    for fun in suite:

        fun.add_observer(observer)

        dim = fun.dimension

        m = re.search(
            r"f(\d+)_i(\d+)",
            fun.id
        )

        fun_num = int(m.group(1))

        inst_num = int(m.group(2))

        budget = BUDGET_FACTOR * dim

        seed = (
            RANDOM_SEED
            + fun_num * 1000
            + inst_num
        )

        print(
            f"f{fun_num:02d} "
            f"d{dim:02d} "
            f"i{inst_num:03d} ",
            end="",
            flush=True
        )

        history = run_instance(
            fun=fun,
            dim=dim,
            budget=budget,
            seed=seed
        )

        print(
            f"best={history[-1]:.3e}"
        )

        collected.setdefault(
            (fun_num, dim),
            []
        ).append(history)

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    for (fun_num, dim), histories in collected.items():

        path = os.path.join(
            RESULTS_DIR,
            f"dim_{dim}_fun_{fun_num}.npy"
        )

        np.save(
            path,
            np.array(histories)
        )

    print("\n===================================")

    print(
        f"Results saved to:\n"
        f"{RESULTS_DIR}"
    )

    print(
        f"\nCOCO output saved to:\n"
        f"{result_folder}"
    )

    print("===================================")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    main()