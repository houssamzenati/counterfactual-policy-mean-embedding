# %%
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_kernels, pairwise_distances
import os
import jsonlines
from joblib import Parallel, delayed
from scipy.optimize import minimize
from embeddings import plugin_embedding_pi, dr_embedding_pi, mmd2_biased, mmd2_unbiased
from environment import (
    logistic_logging_policy,
    reward_nonlinear,
    reward_quadratic,
    find_best_params,
    importance_weights,
    pi0_proba,
    pi_proba,
)


# --- Kernel herding from CME embedding ---


def kernel_herding(Y_support, weights, sigma, num_samples):
    y0 = np.random.randn()
    res = minimize(
        lambda y: -np.mean(
            np.dot(
                weights.T,
                pairwise_kernels(
                    Y_support.reshape(-1, 1),
                    np.atleast_2d(y),
                    metric="rbf",
                    gamma=1.0 / (2 * sigma),
                ),
            )
        ),
        y0,
        method="CG",
        options={"gtol": 1e-6, "disp": False},
    )
    yt = res.x.ravel()[0]
    samples = [yt]
    for t in range(2, num_samples + 1):
        yt_hist = np.array(samples)
        res = minimize(
            lambda y: -np.mean(
                np.dot(
                    weights.T,
                    pairwise_kernels(
                        Y_support.reshape(-1, 1),
                        np.atleast_2d(y),
                        metric="rbf",
                        gamma=1.0 / (2 * sigma),
                    ),
                )
            )
            + np.mean(
                pairwise_kernels(
                    yt_hist.reshape(-1, 1),
                    np.atleast_2d(y),
                    metric="rbf",
                    gamma=1.0 / (2 * sigma),
                )
            ),
            y0,
            method="CG",
            options={"gtol": 1e-6, "disp": False},
        )
        yt = res.x.ravel()[0]
        samples.append(yt)
    return np.array(samples)


def run_experiment(seed, logging_type, reward_type):
    rng = np.random.RandomState(seed)
    d = 5
    beta = np.linspace(0.1, 0.5, d)
    n = 1000
    X_log = rng.randn(n, d)

    # Choose logging policy
    if logging_type == "uniform":
        probs_log = np.full(n, 0.5)
        pi0_fn = lambda a, X: np.full(X.shape[0], 0.5)
    elif logging_type == "logistic":
        probs_log = logistic_logging_policy(X_log, beta)
        pi0_fn = lambda a, X: pi0_proba(a, X)
    else:
        raise ValueError("Unknown logging policy.")

    # Choose action sampling and target policy
    A_log = rng.binomial(1, probs_log)
    probs_tgt = 1 / (1 + np.exp(-(X_log @ beta)))
    A_tgt = rng.binomial(1, probs_tgt)
    pi_fn = lambda a, X: pi_proba(a, X)

    # Sample outcomes
    if reward_type == "quadratic":
        Y_log = reward_quadratic(X_log, A_log, beta)
        Y_tgt = reward_quadratic(X_log, A_tgt, beta)
    elif reward_type == "nonlinear":
        Y_log = reward_nonlinear(X_log, A_log, beta)
        Y_tgt = reward_nonlinear(X_log, A_tgt, beta)
    else:
        raise ValueError("Unknown reward type")

    # Importance weights
    w_pi = importance_weights(A_log, X_log, pi_fn, pi0_fn)

    # Regularization and kernel params
    reg_lambda = find_best_params(X_log, A_log, Y_log)
    sigmaKX = np.median(pairwise_distances(X_log)) ** 2 + 1e-8
    sigma = np.median(pairwise_distances(Y_log[:, None])) ** 2 + 1e-8

    # Embedding and kernel herding
    phi_plugin = plugin_embedding_pi(Y_log, X_log, A_log, A_tgt, reg_lambda, sigmaKX)
    phi_dr = dr_embedding_pi(Y_log, X_log, A_log, w_pi, A_tgt, reg_lambda, sigmaKX)
    Y_plugin = kernel_herding(Y_log, phi_plugin, sigma, 500)
    Y_dr = kernel_herding(Y_log, phi_dr, sigma, 500)

    return (
        {
            "seed": seed,
            "logging_type": logging_type,
            "reward_type": reward_type,
            "mmd_unbiased_dr": mmd2_unbiased(Y_dr, Y_tgt, "rbf", sigma),
            "mmd_unbiased_plugin": mmd2_unbiased(Y_plugin, Y_tgt, "rbf", sigma),
            "mmd_biased_dr": mmd2_biased(Y_dr, Y_tgt, "rbf", sigma),
            "mmd_biased_plugin": mmd2_biased(Y_plugin, Y_tgt, "rbf", sigma),
            "wass_dr": wasserstein_distance(Y_dr, Y_tgt),
            "wass_plugin": wasserstein_distance(Y_plugin, Y_tgt),
        },
        Y_plugin,
        Y_dr,
        Y_log,
        Y_tgt,
    )


def run_and_save(logging_type, reward_type, output_dir="results", nb_seeds=100):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/results_{logging_type}_{reward_type}.jsonl"

    with jsonlines.open(filename, mode="w") as writer:
        for seed in range(nb_seeds):
            result, _, _, _, _ = run_experiment(seed, logging_type, reward_type)
            writer.write(result)


logging_types = ["uniform", "logistic"]
reward_types = ["quadratic", "nonlinear"]

jobs = [(log, rew) for log in logging_types for rew in reward_types]

# Run 8 jobs in parallel (1 per configuration, each handling 100 seeds)
Parallel(n_jobs=8)(delayed(run_and_save)(log, rew) for (log, rew) in jobs)
