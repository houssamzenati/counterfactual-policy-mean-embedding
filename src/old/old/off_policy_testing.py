# ==============================
#  Distributional Testing under Stochastic Policies with Doubly Robust Kernel Statistic
# ==============================

# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.stats import laplace, bernoulli
from sklearn.metrics import pairwise_kernels
import scipy.stats as st

np.random.seed(42)

# Create directories
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)


# ==============================
# 1. Define Policy Classes
# ==============================
class GaussianPolicy:
    def __init__(self, w):
        self.w = w

    def sample_treatments(self, X):
        mean = X @ self.w
        return np.random.normal(mean, 1.0)

    def get_propensities(self, X, t):
        mean = X @ self.w
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (t - mean) ** 2)


class LaplacePolicy:
    def __init__(self, w):
        self.w = w

    def sample_treatments(self, X):
        mean = X @ self.w
        return laplace.rvs(loc=mean, scale=0.5)

    def get_propensities(self, X, t):
        mean = X @ self.w
        return (1 / (2 * 0.5)) * np.exp(-np.abs(t - mean) / 0.5)


class BernoulliPolicy:
    def __init__(self, w):
        self.w = w

    def sample_treatments(self, X):
        prob = 1 / (1 + np.exp(-(X @ self.w)))
        return 2 * bernoulli.rvs(prob) - 1

    def get_propensities(self, X, t):
        prob = 1 / (1 + np.exp(-(X @ self.w)))
        t = (t + 1) // 2  # map {-1, 1} to {0, 1}
        return prob if t == 1 else 1 - prob


# ==============================
# 2. Define Outcome Model
# ==============================
def outcome_model(X, T, beta, gamma):
    noise = 0.1 * np.random.randn(len(T))
    return X @ beta + gamma * T + noise


# ==============================
# 3. DR Kernel Statistic
# ==============================
def kernel_dr_two_sample_test(Y, X, T, kernel_function="rbf"):
    N = len(Y)
    N2 = N // 2

    Xa = X[:N2]
    Xb = X[N2:]
    Ta = T[:N2]
    Tb = T[N2:]
    Ya = Y[:N2]
    Yb = Y[N2:]

    w_model = LogisticRegression(C=1e6, max_iter=1000)
    w_model.fit(X, T)
    propensity_scores = w_model.predict_proba(X)[:, 1]

    w = np.zeros_like(T, dtype=float)
    w[T == 1] = 1.0 / propensity_scores[T == 1]
    w[T == 0] = 1.0 / (1.0 - propensity_scores[T == 0])

    sigma = np.median(pairwise_distances(X, metric="euclidean")) ** 2
    KX = pairwise_kernels(X, metric="rbf", gamma=1.0 / sigma)
    KY = pairwise_kernels(Y.reshape(-1, 1), metric=kernel_function)

    left = w[:N2] * (KX[:N2, :N2].mean(axis=1) - KX[:N2, N2:].mean(axis=1))
    right = w[N2:] * (KX[N2:, :N2].mean(axis=1) - KX[N2:, N2:].mean(axis=1))

    test_stat = left.mean() + right.mean()

    p_value = 1 - st.norm.cdf(test_stat)
    return test_stat, p_value


# ==============================
# 4. MMD Two-Sample Test
# ==============================
def MMD2u(K, m, n):
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return (
        (Kx.sum() - np.trace(Kx)) / (m * (m - 1))
        + (Ky.sum() - np.trace(Ky)) / (n * (n - 1))
        - 2 * Kxy.sum() / (m * n)
    )


def compute_null_distribution(K, m, n, iterations=500):
    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        idx = np.random.permutation(m + n)
        K_i = K[idx, :][:, idx]
        mmd2u_null[i] = MMD2u(K_i, m, n)
    return mmd2u_null


def kernel_two_sample_test(X, Y, kernel_function="rbf", gamma=None, iterations=500):
    XY = np.vstack([X, Y])
    if kernel_function == "rbf":
        if gamma is None:
            gamma = 1.0 / (np.median(pairwise_distances(XY)) ** 2)
        K = np.exp(-gamma * pairwise_distances(XY, metric="sqeuclidean"))
    elif kernel_function == "linear":
        K = XY @ XY.T
    else:
        raise ValueError("Unknown kernel")

    m, n = len(X), len(Y)
    mmd2u_stat = MMD2u(K, m, n)
    null_dist = compute_null_distribution(K, m, n, iterations)
    p_value = np.mean(null_dist > mmd2u_stat)
    return mmd2u_stat, p_value


# ==============================
# 5. Experiment Runner
# ==============================
def run_experiment(
    pi, pi_prime, beta, gamma, sample_sizes, num_experiments, scenario_name
):
    results = []
    for ns in sample_sizes:
        lin_rejects, rbf_rejects, dr_rejects = 0, 0, 0
        for _ in tqdm(
            range(num_experiments), desc=f"Sample size {ns} - {scenario_name}"
        ):
            X = np.random.randn(ns, len(beta))
            X_prime = np.random.randn(ns, len(beta))
            T_pi = pi.sample_treatments(X)
            T_pi_prime = pi_prime.sample_treatments(X_prime)
            pi_propensities = pi.get_propensities()
            Y_pi = outcome_model(X, T_pi, beta, gamma)
            Y_pi_prime = outcome_model(X_prime, T_pi_prime, beta, gamma)

            Y_pi = Y_pi.reshape(-1, 1)
            Y_pi_prime = Y_pi_prime.reshape(-1, 1)

            start = time.perf_counter()
            _, p_lin = kernel_two_sample_test(
                Y_pi, Y_pi_prime, kernel_function="linear"
            )
            time_lin = time.perf_counter() - start

            start = time.perf_counter()
            _, p_rbf = kernel_two_sample_test(Y_pi, Y_pi_prime, kernel_function="rbf")
            time_rbf = time.perf_counter() - start

            Y_all = np.vstack([Y_pi, Y_pi_prime])
            X_all = np.vstack([X, X_prime])
            T_all = np.hstack([np.zeros(ns), np.ones(ns)])

            start = time.perf_counter()
            _, p_dr = kernel_dr_two_sample_test(Y_all, X_all, T_all)
            time_dr = time.perf_counter() - start

            if p_lin < 0.01:
                lin_rejects += 1
            if p_rbf < 0.01:
                rbf_rejects += 1
            if p_dr < 0.01:
                dr_rejects += 1

        results.append(
            {
                "Sample Size": ns,
                "Scenario": scenario_name,
                "ATE Power (Linear Kernel)": lin_rejects / num_experiments,
                "DTE Power (RBF Kernel)": rbf_rejects / num_experiments,
                "DR Kernel Power": dr_rejects / num_experiments,
            }
        )
    return pd.DataFrame(results)


# ==============================
# 6. Main Execution and Plotting
# ==============================
if __name__ == "__main__":

    beta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    gamma = 1.0
    w = np.random.randn(5)
    w_prime = np.random.randn(5)

    sample_sizes = [50, 100, 200, 250, 300, 350]
    num_experiments = 200

    all_results = []

    # Scenario i) No Treatment Effect
    pi = GaussianPolicy(w)
    pi_prime = GaussianPolicy(w)
    df1 = run_experiment(
        pi,
        pi_prime,
        beta,
        gamma,
        sample_sizes,
        num_experiments,
        scenario_name="Scenario i) No Treatment Effect",
    )
    all_results.append(df1)

    # Scenario ii) Mean Shift Effect
    pi = GaussianPolicy(w)
    pi_prime = LaplacePolicy(w)
    df2 = run_experiment(
        pi,
        pi_prime,
        beta,
        gamma,
        sample_sizes,
        num_experiments,
        scenario_name="Scenario ii) Mean Shift Effect",
    )
    all_results.append(df2)

    # Scenario iii) High-Order Treatment Effect
    pi = GaussianPolicy(w)
    pi_prime = BernoulliPolicy(w_prime)
    df3 = run_experiment(
        pi,
        pi_prime,
        beta,
        gamma,
        sample_sizes,
        num_experiments,
        scenario_name="Scenario iii) High-Order Treatment Effect",
    )
    all_results.append(df3)

    df_final = pd.concat(all_results, ignore_index=True)
    df_final.to_csv("tables/results_table.csv", index=False)

    import seaborn as sns

    sns.set(style="whitegrid")

    for scenario in df_final["Scenario"].unique():
        df_scenario = df_final[df_final["Scenario"] == scenario]
        plt.figure(figsize=(6, 4))
        plt.plot(
            df_scenario["Sample Size"],
            df_scenario["ATE Power (Linear Kernel)"],
            marker="o",
            label="Linear Kernel (APE)",
        )
        plt.plot(
            df_scenario["Sample Size"],
            df_scenario["DTE Power (RBF Kernel)"],
            marker="s",
            label="Gaussian Kernel (DPE, permutation)",
        )
        plt.plot(
            df_scenario["Sample Size"],
            df_scenario["DR Kernel Power"],
            marker="^",
            label="Gaussian Kernel (DPE, DR-asymp-norm)",
        )
        plt.xlabel("Sample Size")
        plt.ylabel("Power")
        plt.title(f"Power Comparison - {scenario}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"figures/power_curve_{scenario.replace(' ', '_').replace(')', '').replace('(', '').lower()}.png"
        )
        plt.close()
