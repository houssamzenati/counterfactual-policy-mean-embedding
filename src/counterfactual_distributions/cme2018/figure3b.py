# FAST Mode Settings
bootstrap_iterations = 10000  # Kernel Two-Sample Test
num_experiments = 1000  # Monte Carlo Repeats
sample_sizes = [10, 50, 100, 150, 200]  # Sample Sizes
num_herding = 1000  # Number of kernel herding samples (set inside the loop)
num_cv = 5  # Cross-validation folds for CME parameter search

# FINAL Mode Settings (for publication)
# bootstrap_iterations = 10000
# num_experiments = 1000
# sample_sizes = [10, 50, 100, 150, 200]
# num_herding = 1000
# num_cv = 5

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from tqdm import tqdm

np.random.seed(2)

significance_level = 0.01

# Helper functions


def gauss_rbf(X1, X2, sigma=1):
    return np.exp(-cdist(X1, X2, "sqeuclidean") / (2 * sigma))


def generate_data(ns):
    d = 5
    X0 = np.random.randn(ns, d)
    X1 = np.random.laplace(0, 0.2, size=(ns, d))
    beta_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    b_shift = 1.0
    Y0 = np.dot(beta_vec, X0.T) + 0.1 * np.random.randn(ns)
    Y1 = np.dot(beta_vec, X1.T) + b_shift + 0.1 * np.random.randn(ns)
    return X0, X1, Y0, Y1


def estimate_cme_real_inv(X0, X1):
    ns = X0.shape[0]
    reg_param = 1e-6
    sigma = np.median(pairwise_distances(X1, X0, metric="euclidean")) ** 2
    K1 = gauss_rbf(X1, X1, sigma)
    K2 = gauss_rbf(X1, X0, sigma)
    inverse = np.linalg.inv(K1 + ns * reg_param * np.eye(ns))
    b_vec = inverse @ K2 @ (np.ones((ns, 1)) / ns)
    b_vec = np.maximum(b_vec.flatten(), 0)
    b_vec /= b_vec.sum()
    return b_vec


def generate_kernel_herding_samples(Y0, b_vec, sigma, n_samples):
    selected = []
    mean_embedding = np.dot(b_vec, gauss_rbf(Y0[:, None], Y0[:, None], sigma))
    current_sum = np.zeros_like(mean_embedding)
    for _ in range(n_samples):
        scores = mean_embedding - current_sum / (len(selected) + 1e-8)
        idx = np.argmax(scores)
        selected.append(idx)
        current_sum += gauss_rbf(Y0[:, None], Y0[idx : idx + 1, None], sigma).flatten()
    return Y0[selected]


def kernel_two_sample_test(X, Y, kernel_function="rbf", gamma=None):
    XY = np.vstack([X, Y])
    if kernel_function == "rbf":
        K = np.exp(-gamma * pairwise_distances(XY, metric="sqeuclidean"))
    elif kernel_function == "linear":
        K = np.dot(XY, XY.T)
    else:
        raise ValueError("Unknown kernel")

    m = X.shape[0]
    n = Y.shape[0]
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]

    mmd2u = (
        (Kx.sum() - np.trace(Kx)) / (m * (m - 1))
        + (Ky.sum() - np.trace(Ky)) / (n * (n - 1))
        - 2 * Kxy.sum() / (m * n)
    )

    perm_stats = []
    for _ in range(10):
        idx = np.random.permutation(m + n)
        X_perm = XY[idx[:m]]
        Y_perm = XY[idx[m:]]
        Kp = (
            np.exp(
                -gamma
                * pairwise_distances(np.vstack([X_perm, Y_perm]), metric="sqeuclidean")
            )
            if kernel_function == "rbf"
            else np.dot(np.vstack([X_perm, Y_perm]), np.vstack([X_perm, Y_perm]).T)
        )
        Kx_p = Kp[:m, :m]
        Ky_p = Kp[m:, m:]
        Kxy_p = Kp[:m, m:]
        mmd2u_p = (
            (Kx_p.sum() - np.trace(Kx_p)) / (m * (m - 1))
            + (Ky_p.sum() - np.trace(Ky_p)) / (n * (n - 1))
            - 2 * Kxy_p.sum() / (m * n)
        )
        perm_stats.append(mmd2u_p)

    p_value = np.mean(np.array(perm_stats) > mmd2u)
    return mmd2u, perm_stats, p_value


# Main experiment loop

results = []

for ns in sample_sizes:
    lin_rejects = 0
    rbf_rejects = 0
    num_herding = ns if num_herding is None else num_herding

    for _ in tqdm(range(num_experiments), desc=f"Sample Size {ns}"):
        X0, X1, Y0, Y1 = generate_data(ns)
        sigma_Y = (
            np.median(pairwise_distances(Y0[:, None], Y1[:, None], metric="euclidean"))
            ** 2
        )

        b_vec = estimate_cme_real_inv(X0, X1)
        Y_samples = generate_kernel_herding_samples(Y0, b_vec, sigma_Y, num_herding)

        # Generate true counterfactual outcomes for comparison
        beta_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        Y0_T1 = np.dot(beta_vec, X1.T) + 0.1 * np.random.randn(X1.shape[0])

        _, _, p_rbf = kernel_two_sample_test(
            Y_samples[:, None],
            Y0_T1[:, None],
            kernel_function="rbf",
            gamma=1.0 / sigma_Y,
        )
        if p_rbf < significance_level:
            rbf_rejects += 1

        _, _, p_lin = kernel_two_sample_test(
            Y_samples[:, None], Y0_T1[:, None], kernel_function="linear", gamma=None
        )
        if p_lin < significance_level:
            lin_rejects += 1

    results.append(
        {"Kernel": "ATE", "Power": lin_rejects / num_experiments, "Sample Size": ns}
    )
    results.append(
        {"Kernel": "DATE", "Power": rbf_rejects / num_experiments, "Sample Size": ns}
    )

df_res = pd.DataFrame(results)

# Plotting
sns.set(style="whitegrid")
g = sns.FacetGrid(
    df_res,
    hue="Kernel",
    height=3.4 * 0.8,
    aspect=1.2,
    sharey=False,
    hue_kws={"marker": ["o", "s"]},
)
(g.map(sns.lineplot, "Sample Size", "Power", markers=True, dashes=False))

g.ax.legend()
g.ax.set_title("Power of Test (Real Inverse)")
plt.tight_layout()
plt.show()
