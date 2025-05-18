# Corrected Context Dim Experiment for Figure 4b

# FAST Mode Settings
num_experiments = 10
context_dims = [5, 10, 15, 20, 25, 30]
num_cv = 3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist
from Environment import AvgEnvironment
from Estimator import (
    IPSEstimator,
    SlateEstimator,
    DirectEstimator,
    DoublyRobustEstimator,
    CMEstimator,
)
from Policy import MultinomialPolicy
from ParameterSelector import ParameterSelector

np.random.seed(2)


# Helper function for simulation
def simulate_context_dim(context_dim, config, num_iter):
    results = []

    user_vectors = np.random.normal(0, 1, size=(config["n_users"], context_dim))
    target_user_vectors = user_vectors * np.random.binomial(
        1, 0.5, size=user_vectors.shape
    )
    item_vectors = np.random.normal(0, 1, size=(config["n_items"], context_dim))

    alpha = -0.3
    null_user_vectors = alpha * target_user_vectors

    null_policy = MultinomialPolicy(
        item_vectors,
        null_user_vectors,
        config["n_items"],
        config["n_reco"],
        temperature=0.5,
        cal_gamma=True,
    )
    target_policy = MultinomialPolicy(
        item_vectors,
        target_user_vectors,
        config["n_items"],
        config["n_reco"],
        temperature=1.0,
        cal_gamma=False,
    )
    environment = AvgEnvironment(item_vectors, user_vectors)

    reg_pow = -1
    reg_params = (10.0**reg_pow) / config["n_observation"]
    bw_params = 10.0**0
    params = [reg_params, bw_params, bw_params]

    estimators = [
        IPSEstimator(config["n_reco"], null_policy, target_policy),
        SlateEstimator(config["n_reco"], null_policy),
        DirectEstimator(),
        DoublyRobustEstimator(config["n_reco"], null_policy, target_policy),
        CMEstimator(rbf_kernel, rbf_kernel, params),
    ]

    seeds = np.random.randint(np.iinfo(np.int32).max, size=num_iter)

    for seed in tqdm(seeds, desc=f"Context dim {context_dim}"):
        np.random.seed(seed)

        sim_data = []
        for _ in range(config["n_observation"]):
            user = environment.get_context()
            null_reco, null_multinomial, null_user_vector = null_policy.recommend(user)
            null_reco_vec = np.concatenate(item_vectors[null_reco])
            null_reward = environment.get_reward(user, null_reco)

            target_reco, target_multinomial, _ = target_policy.recommend(user)
            target_reco_vec = np.concatenate(item_vectors[target_reco])
            target_reward = environment.get_reward(user, target_reco)

            observation = {
                "null_context_vec": null_user_vector,
                "target_context_vec": null_user_vector,
                "null_reco": tuple(null_reco),
                "null_reco_vec": null_reco_vec,
                "null_reward": null_reward,
                "target_reco": tuple(target_reco),
                "null_multinomial": null_multinomial,
                "target_multinomial": target_multinomial,
                "target_reco_vec": target_reco_vec,
                "target_reward": target_reward,
                "user": user,
            }

            sim_data.append(observation)

        sim_data = pd.DataFrame(sim_data)

        direct_selector = ParameterSelector(estimators[2])
        params_grid = [(n_hiddens, 1024, 100) for n_hiddens in [50, 100, 150, 200]]
        direct_selector.select_from_propensity(
            sim_data, params_grid, null_policy, target_policy
        )
        estimators[2] = direct_selector.estimator

        estimators[3].params = direct_selector.parameters

        cme_selector = ParameterSelector(estimators[4])
        params_grid = [
            [(10.0**p) / config["n_observation"], 1.0, 1.0] for p in np.arange(-6, 0, 1)
        ]
        cme_selector.select_from_propensity(
            sim_data, params_grid, null_policy, target_policy
        )
        estimators[4] = cme_selector.estimator

        actual_value = np.mean(sim_data["target_reward"])

        for estimator in estimators:
            est_value = estimator.estimate(sim_data)
            mse = (est_value - actual_value) ** 2
            results.append(
                {"Estimator": estimator.name, "MSE": mse, "Context Dim": context_dim}
            )

    return pd.DataFrame(results)


# Updated Simulation configuration
config = {
    "n_users": 50,
    "n_items": 20,
    "n_reco": 4,
    "n_observation": 5000,
}

# Running the simulation
full_results = pd.concat(
    [simulate_context_dim(d, config, num_experiments) for d in context_dims]
)

# Plotting results
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.lineplot(data=full_results, x="Context Dim", y="MSE", hue="Estimator", marker="o")
plt.yscale("log")
plt.title("Context Dimension vs MSE (FAST Mode)")
plt.tight_layout()
plt.show()
