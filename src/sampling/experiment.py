# %%
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_kernels, pairwise_distances
import os
import jsonlines
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns


# === Evaluation metrics ===
def mmd2_unbiased(X, Y, kernel, gamma):
    K_XX = pairwise_kernels(X[:, None], X[:, None], metric=kernel, gamma=gamma)
    K_YY = pairwise_kernels(Y[:, None], Y[:, None], metric=kernel, gamma=gamma)
    K_XY = pairwise_kernels(X[:, None], Y[:, None], metric=kernel, gamma=gamma)
    n = len(X)
    m = len(Y)
    return (
        (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1))
        + (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1))
        - 2 * K_XY.mean()
    )


def mmd2_biased(X, Y, kernel, gamma):
    K_XX = pairwise_kernels(X[:, None], X[:, None], metric=kernel, gamma=gamma)
    K_YY = pairwise_kernels(Y[:, None], Y[:, None], metric=kernel, gamma=gamma)
    K_XY = pairwise_kernels(X[:, None], Y[:, None], metric=kernel, gamma=gamma)
    return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()


def uniform_logging_policy(X):
    return np.full(X.shape[0], 0.5)


def logistic_logging_policy(X, beta):
    logits = -2 * (X @ beta)
    return 1 / (1 + np.exp(-logits))


def reward_quadratic(X, A, beta):
    return (X @ beta) + 2.0 * (A**2) + 0.1 * np.random.randn(len(X))


def reward_nonlinear(X, A, beta):
    return np.sin(X @ beta) + A * np.cos(X @ beta) + 0.1 * np.random.randn(len(X))


def reward_linear(X, A, beta):
    return (X @ beta) + A + 0.1 * np.random.randn(len(X))


def gaussian_policy_mean(X, beta):
    return X @ beta  # mean of Gaussian


def gaussian_pdf(a, X, beta, scale=0.5):
    mu = gaussian_policy_mean(X, beta)
    return (1 / (np.sqrt(2 * np.pi) * scale)) * np.exp(-0.5 * ((a - mu) / scale) ** 2)


def find_best_params(
    X_log, A_log, Y_log, reg_grid=[1e1, 1e0, 0.1, 1e-2, 1e-3, 1e-4], num_cv=3
):
    kr = GridSearchCV(
        KernelRidge(kernel="rbf", gamma=0.1),
        cv=num_cv,
        param_grid={"alpha": reg_grid},
    )
    features = np.concatenate([X_log, A_log.reshape(-1, 1)], axis=1)
    kr.fit(features, Y_log)
    reg_param = kr.best_params_["alpha"]
    return reg_param


def pi0_proba(a, X):
    """
    Logging policy: P(a=1|x) = sigmoid(-2 * x · beta), P(a=0|x) = 1 - P(a=1|x)
    """
    logits = -2 * (X @ np.linspace(0.1, 0.5, X.shape[1]))
    probs = 1 / (1 + np.exp(-logits))  # P(a=1 | x)
    a = np.asarray(a)
    return probs * (a == 1) + (1 - probs) * (a == 0)


def pi_proba(a, X):
    """
    Target policy: logistic over X @ beta, returns pi(a|x)
    """
    logits = 4 * (X @ np.linspace(0.1, 0.5, X.shape[1]))
    probs = 1 / (1 + np.exp(-logits))  # P(a=1|x)
    a = np.asarray(a)
    return probs * (a == 1) + (1 - probs) * (a == 0)


def importance_weights(A, X, pi, pi0, eps=1e-8):
    """
    Compute importance weights w = pi(a|x) / pi0(a|x)

    Parameters
    ----------
    A : np.ndarray of shape (n,)
        Actions taken (can be discrete or continuous)
    X : np.ndarray of shape (n, d)
        Contexts
    pi : callable
        Target policy. Should return pi(a|x) — either probability mass or density.
    pi0 : callable
        Logging policy. Same interface as pi.
    eps : float
        Clipping constant to avoid division by zero or instability.

    Returns
    -------
    w : np.ndarray of shape (n,)
        Importance sampling weights.
    """
    numer = pi(A, X)
    denom = pi0(A, X)

    # clip to avoid division by zero or exploding weights
    numer = np.clip(numer, eps, None)
    denom = np.clip(denom, eps, None)

    return numer / denom


def plugin_embedding_pi(
    Y, X, logging_T, pi_samples, reg_lambda, sigmaKX=1.0, sigmaKT=0.5
):
    N = len(Y)
    KX = pairwise_kernels(X, metric="rbf", gamma=1.0 / (sigmaKX))
    KT = pairwise_kernels(logging_T[:, None], metric="rbf", gamma=1.0 / (sigmaKT))
    KT_pi = pairwise_kernels(
        logging_T[:, None], pi_samples[:, None], metric="rbf", gamma=1.0 / (sigmaKT)
    )
    mu_pi = np.linalg.solve(
        np.multiply(KX, KT) + reg_lambda * np.eye(N), np.multiply(KX, KT_pi)
    )
    return mu_pi


def dr_embedding_pi(
    Y, X, logging_T, w_pi, pi_samples, reg_lambda, sigmaKX=1.0, sigmaKT=0.5
):
    N = len(Y)
    KX = pairwise_kernels(X, metric="rbf", gamma=1.0 / (sigmaKX))
    KT = pairwise_kernels(logging_T[:, None], metric="rbf", gamma=1.0 / (sigmaKT))
    KT_pi = pairwise_kernels(
        logging_T[:, None], pi_samples[:, None], metric="rbf", gamma=1.0 / (sigmaKT)
    )
    mu_log = np.linalg.solve(
        np.multiply(KX, KT) + reg_lambda * np.eye(N), np.multiply(KX, KT)
    )
    mu_pi = np.linalg.solve(
        np.multiply(KX, KT) + reg_lambda * np.eye(N), np.multiply(KX, KT_pi)
    )
    phi = mu_pi + w_pi[:, None] * (np.eye(N) - mu_log)
    return phi


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


def run_experiment(seed, logging_type, reward_type, action_type):
    rng = np.random.RandomState(seed)
    d = 5
    beta = np.linspace(0.1, 0.5, d)
    n = 500
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
    if action_type == "binary":
        A_log = rng.binomial(1, probs_log)
        probs_tgt = 1 / (1 + np.exp(-(X_log @ beta)))
        A_tgt = rng.binomial(1, probs_tgt)
        pi_fn = lambda a, X: pi_proba(a, X)
    elif action_type == "continuous":
        A_log = gaussian_policy_mean(X_log, beta) + 0.5 * rng.randn(n)
        A_tgt = gaussian_policy_mean(X_log, beta) + 0.5 * rng.randn(n)
        pi_fn = lambda a, X: gaussian_pdf(a, X, beta)
        pi0_fn = lambda a, X: gaussian_pdf(a, X, -2 * beta)
    else:
        raise ValueError("Unknown action type")

    # Sample outcomes
    if reward_type == "quadratic":
        Y_log = reward_quadratic(X_log, A_log, beta)
        Y_tgt = reward_quadratic(X_log, A_tgt, beta)
    elif reward_type == "nonlinear":
        Y_log = reward_nonlinear(X_log, A_log, beta)
        Y_tgt = reward_nonlinear(X_log, A_tgt, beta)
    elif reward_type == "linear":
        Y_log = reward_linear(X_log, A_log, beta)
        Y_tgt = reward_linear(X_log, A_tgt, beta)
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
            "action_type": action_type,
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


def run_and_save(
    logging_type, reward_type, action_type, output_dir="results", nb_seeds=100
):
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/results_{logging_type}_{reward_type}_{action_type}.jsonl"

    with jsonlines.open(filename, mode="w") as writer:
        for seed in range(nb_seeds):
            result, _, _, _, _ = run_experiment(
                seed, logging_type, reward_type, action_type
            )
            writer.write(result)


logging_types = ["uniform", "logistic"]
reward_types = ["quadratic", "nonlinear", "linear"]
action_types = ["binary"]

jobs = [
    (log, rew, act)
    for log in logging_types
    for rew in reward_types
    for act in action_types
]

# Run 8 jobs in parallel (1 per configuration, each handling 100 seeds)
Parallel(n_jobs=8)(delayed(run_and_save)(log, rew, act) for (log, rew, act) in jobs)
# %%

# _, Y_plugin, Y_dr, Y_log, Y_tgt = run_experiment(42, "logistic", "nonlinear", "binary")
# # %%
# def plot_herding_vs_true(Y_log, Y_tgt, Y_herded_dr_cme, Y_herded_plug_in_cme):
#     plt.figure(figsize=(8, 5))
#     sns.histplot(Y_log, color="red", kde=True, stat="density", label="Logged", alpha=0.3)
#     sns.histplot(Y_tgt, color="blue", kde=True, stat="density", label="True $\pi$", alpha=0.3)
#     sns.histplot(Y_herded_plug_in_cme, color="orange", kde=True, stat="density", label="Herded PI-CPME", alpha=0.3)
#     sns.histplot(Y_herded_dr_cme, color="green", kde=True, stat="density", label="Herded DR-CPME", alpha=0.3)
#     plt.xlim([-4, 4])
#     plt.xlabel("Y")
#     plt.ylabel("Density")
#     plt.legend()
#     plt.title("Counterfactual Outcome Distribution via Kernel Herding")
#     plt.tight_layout()
#     plt.show()

# plot_herding_vs_true(Y_log, Y_tgt, Y_dr, Y_plugin)
# %%
