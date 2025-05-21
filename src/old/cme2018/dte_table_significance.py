import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import bernoulli
from sklearn.metrics import pairwise_distances
from kernel_two_sample_test_nonuniform import kernel_two_sample_test_nonuniform
from tqdm import tqdm


def run_table1_experiment(
    ns_values=[50, 100], num_experiments=100, significance_level=0.01
):
    results = []

    for ns in ns_values:
        for scenario in ["I", "II", "III"]:
            lin_num_rejects = 0
            rbf_num_rejects = 0

            for _ in tqdm(
                range(num_experiments), desc=f"Sample Size {ns}, Scenario {scenario}"
            ):
                d = 5
                noise_var = 0.1
                beta_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
                alpha_vec = np.array([0.05, 0.04, 0.03, 0.02, 0.01])
                alpha_0 = 0.05

                X = np.random.randn(ns, d)
                Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
                T = bernoulli.rvs(Prob_vec)

                if scenario == "I":
                    b = 0
                    Y0 = np.dot(beta_vec, X[T == 0, :].T) + noise_var * np.random.randn(
                        X[T == 0, :].shape[0]
                    )
                    Y1 = (
                        np.dot(beta_vec, X[T == 1, :].T)
                        + b
                        + noise_var * np.random.randn(X[T == 1, :].shape[0])
                    )
                elif scenario == "II":
                    b = 2
                    Y0 = np.dot(beta_vec, X[T == 0, :].T) + noise_var * np.random.randn(
                        X[T == 0, :].shape[0]
                    )
                    Y1 = (
                        np.dot(beta_vec, X[T == 1, :].T)
                        + b
                        + noise_var * np.random.randn(X[T == 1, :].shape[0])
                    )
                elif scenario == "III":
                    b = 2
                    Z = bernoulli.rvs(0.5, size=len(T[T == 1]))
                    Y0 = np.dot(beta_vec, X[T == 0, :].T) + noise_var * np.random.randn(
                        X[T == 0, :].shape[0]
                    )
                    Y1 = (
                        np.dot(beta_vec, X[T == 1, :].T)
                        + (2 * Z - 1)
                        + noise_var * np.random.randn(X[T == 1, :].shape[0])
                    )
                else:
                    raise ValueError("Unknown scenario")

                YY0 = Y0[:, np.newaxis]
                YY1 = Y1[:, np.newaxis]

                # DTE: RBF Kernel
                sigma2 = (
                    np.median(pairwise_distances(YY0, YY1, metric="euclidean")) ** 2
                )
                mmd2u_rbf, mmd2u_null_rbf, p_value_rbf = (
                    kernel_two_sample_test_nonuniform(
                        YY0,
                        YY1,
                        Prob_vec,
                        kernel_function="rbf",
                        gamma=1.0 / sigma2,
                        verbose=False,
                        iterations=10000,
                    )
                )
                if p_value_rbf < significance_level:
                    rbf_num_rejects += 1

                # ATE: Linear Kernel
                mmd2u_lin, mmd2u_null_lin, p_value_lin = (
                    kernel_two_sample_test_nonuniform(
                        YY0,
                        YY1,
                        Prob_vec,
                        kernel_function="linear",
                        verbose=False,
                        iterations=10000,
                    )
                )
                if p_value_lin < significance_level:
                    lin_num_rejects += 1

            results.append(
                {
                    "Sample Size": ns,
                    "Scenario": scenario,
                    "ATE": lin_num_rejects / num_experiments,
                    "DTE": rbf_num_rejects / num_experiments,
                }
            )

    return pd.DataFrame(results)


# Example usage:
df_results = run_table1_experiment()
print(df_results)
# df_results.to_csv('table1_results.csv', index=False)
# with open('table1_results.tex', 'w') as f:
#     f.write(df_results.to_latex(index=False))
