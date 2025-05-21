# ==============================
#  Distributional Testing under Stochastic Policies
# ==============================

# Imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from scipy.stats import laplace, bernoulli

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


class LaplacePolicy:
    def __init__(self, w):
        self.w = w

    def sample_treatments(self, X):
        mean = X @ self.w
        return laplace.rvs(loc=mean, scale=0.5)


class BernoulliPolicy:
    def __init__(self, w):
        self.w = w

    def sample_treatments(self, X):
        prob = 1 / (1 + np.exp(-(X @ self.w)))
        return 2 * bernoulli.rvs(prob) - 1


# ==============================
# 2. Define Outcome Model
# ==============================
def outcome_model(X, T, beta, gamma):
    noise = 0.1 * np.random.randn(len(T))
    return X @ beta + gamma * T + noise


# ==============================
# 3. MMD Two-Sample Test
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
# 4. Experiment Runner
# ==============================
def run_experiment(
    pi, pi_prime, beta, gamma, sample_sizes, num_experiments, scenario_name
):
    results = []
    for ns in sample_sizes:
        lin_rejects, rbf_rejects = 0, 0
        for _ in tqdm(
            range(num_experiments), desc=f"Sample size {ns} - {scenario_name}"
        ):
            X = np.random.randn(ns, len(beta))
            T_pi = pi.sample_treatments(X)
            T_pi_prime = pi_prime.sample_treatments(X)
            Y_pi = outcome_model(X, T_pi, beta, gamma)
            Y_pi_prime = outcome_model(X, T_pi_prime, beta, gamma)

            Y_pi = Y_pi.reshape(-1, 1)
            Y_pi_prime = Y_pi_prime.reshape(-1, 1)

            _, p_lin = kernel_two_sample_test(
                Y_pi, Y_pi_prime, kernel_function="linear"
            )
            _, p_rbf = kernel_two_sample_test(Y_pi, Y_pi_prime, kernel_function="rbf")

            if p_lin < 0.01:
                lin_rejects += 1
            if p_rbf < 0.01:
                rbf_rejects += 1

        results.append(
            {
                "Sample Size": ns,
                "Scenario": scenario_name,
                "ATE Power (Linear Kernel)": lin_rejects / num_experiments,
                "DTE Power (RBF Kernel)": rbf_rejects / num_experiments,
            }
        )
    return pd.DataFrame(results)


# ==============================
# 5. Main Execution
# ==============================
if __name__ == "__main__":

    # Constants
    beta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    gamma = 1.0
    w = np.random.randn(5)
    w_prime = np.random.randn(5)

    sample_sizes = [50, 100, 200]
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

    # Concatenate all results
    df_final = pd.concat(all_results, ignore_index=True)
    print(df_final)
    df_final.to_csv("tables/results_table.csv", index=False)

    # Save Table in Wide Format
    table = df_final.pivot(index="Scenario", columns="Sample Size")
    table.columns = [f"N={col[1]} {col[0]}" for col in table.columns]
    table = table.reset_index()
    table.to_csv("tables/results_table_formatted.csv", index=False)

    # Plot Power Curves per Scenario
    sns.set(style="whitegrid")
    for scenario in df_final["Scenario"].unique():
        df_scenario = df_final[df_final["Scenario"] == scenario]
        plt.figure(figsize=(6, 4))
        plt.plot(
            df_scenario["Sample Size"],
            df_scenario["ATE Power (Linear Kernel)"],
            marker="o",
            label="Linear Kernel",
        )
        plt.plot(
            df_scenario["Sample Size"],
            df_scenario["DTE Power (RBF Kernel)"],
            marker="s",
            label="Gaussian Kernel",
        )
        plt.xlabel("Sample Size")
        plt.ylabel("Power")
        plt.title(f"Power of Test - {scenario}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"figures/power_curve_{scenario.replace(' ', '_').replace(')', '').replace('(', '').lower()}.png"
        )
        plt.close()

    # Plot Histograms for Three Scenarios
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    scenarios = [
        ("Scenario i) No Treatment Effect", GaussianPolicy(w), GaussianPolicy(w)),
        ("Scenario ii) Mean Shift Effect", GaussianPolicy(w), LaplacePolicy(w)),
        (
            "Scenario iii) High-Order Treatment Effect",
            GaussianPolicy(w),
            BernoulliPolicy(w_prime),
        ),
    ]

    for ax, (scenario_name, pi, pi_prime) in zip(axs, scenarios):
        X = np.random.randn(500, 5)
        T_pi = pi.sample_treatments(X)
        T_pi_prime = pi_prime.sample_treatments(X)
        Y_pi = outcome_model(X, T_pi, beta, gamma)
        Y_pi_prime = outcome_model(X, T_pi_prime, beta, gamma)

        ax.hist(
            Y_pi, bins=30, density=True, alpha=0.5, label="$Y^{\\pi}$", color="green"
        )
        ax.hist(
            Y_pi_prime,
            bins=30,
            density=True,
            alpha=0.5,
            label="$Y^{\\pi'}$",
            color="blue",
        )
        ax.set_title(scenario_name)
        ax.set_xlabel("Y")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    plt.savefig("figures/histogram_three_scenarios.png")
    plt.close()
