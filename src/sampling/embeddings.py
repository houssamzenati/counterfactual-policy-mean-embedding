import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.metrics import pairwise_kernels, pairwise_distances


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
