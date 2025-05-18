# Corrected Number of Observation Experiment for Figure 4f

# FAST Mode Settings
num_experiments = 10
observation_sizes = [100, 1000, 2000]
num_cv = 5

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
    DRCMEstimator,
)
from Policy import MultinomialPolicy
from ParameterSelector import ParameterSelector

np.random.seed(2)


# Helper function for simulation
def simulate_observation_size(obs_size, config, num_iter):
    results = []

    user_vectors = np.random.normal(
        0, 1, size=(config["n_users"], config["context_dim"])
    )
    target_user_vectors = user_vectors * np.random.binomial(
        1, 0.5, size=user_vectors.shape
    )
    item_vectors = np.random.normal(
        0, 1, size=(config["n_items"], config["context_dim"])
    )

    alpha = -0.3
    logging_user_vectors = alpha * target_user_vectors

    logging_policy = MultinomialPolicy(
        item_vectors,
        logging_user_vectors,
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
    reg_params = (10.0**reg_pow) / obs_size
    bw_params = 10.0**0
    params = [reg_params, bw_params, bw_params]

    estimators = [
        # IPSEstimator(config["n_reco"], logging_policy, target_policy),
        # SlateEstimator(config["n_reco"], logging_policy),
        # DirectEstimator(),
        # DoublyRobustEstimator(config["n_reco"], logging_policy, target_policy),
        CMEstimator(rbf_kernel, rbf_kernel, params),
        DRCMEstimator(rbf_kernel, rbf_kernel, params, logging_policy, target_policy),
    ]

    seeds = np.random.randint(np.iinfo(np.int32).max, size=num_iter)

    for seed in tqdm(seeds, desc=f"Obs size {obs_size}"):
        np.random.seed(seed)

        sim_data = []
        for _ in range(obs_size):
            user = environment.get_context()
            logging_reco, logging_multinomial, logging_user_vector = (
                logging_policy.recommend(user)
            )
            logging_reco_vec = np.concatenate(item_vectors[logging_reco])
            logging_reward = environment.get_reward(user, logging_reco)

            target_reco, target_multinomial, _ = target_policy.recommend(user)
            target_reco_vec = np.concatenate(item_vectors[target_reco])
            target_reward = environment.get_reward(user, target_reco)

            observation = {
                "logging_context_vec": logging_user_vector,
                "target_context_vec": logging_user_vector,
                "logging_reco": tuple(logging_reco),
                "logging_reco_vec": logging_reco_vec,
                "logging_reward": logging_reward,
                "target_reco": tuple(target_reco),
                "target_multinomial": target_multinomial,
                "target_reco_vec": target_reco_vec,
                "target_reward": target_reward,
                "logging_multinomial": logging_multinomial,
                "user": user,
            }

            sim_data.append(observation)

        sim_data = pd.DataFrame(sim_data)

        # direct_selector = ParameterSelector(estimators[2])
        # params_grid = [(n_hiddens, 1024, 100) for n_hiddens in [50, 100, 150, 200]]
        # direct_selector.select_from_propensity(
        #     sim_data, params_grid, logging_policy, target_policy
        # )
        # estimators[2] = direct_selector.estimator

        # estimators[3].params = direct_selector.parameters

        cme_selector = ParameterSelector(estimators[0])
        params_grid = [(10.0**p, 1e-4, 1e-4) for p in np.arange(-6, 1)]

        cme_selector.select_from_propensity(
            sim_data, params_grid, logging_policy, target_policy
        )
        estimators[0] = cme_selector.estimator
        estimators[1].params = cme_selector.parameters

        actual_value = np.mean(sim_data["target_reward"])

        for estimator in estimators:
            est_value = estimator.estimate(sim_data)
            mse = (est_value - actual_value) ** 2
            results.append(
                {"Estimator": estimator.name, "MSE": mse, "Observation Size": obs_size}
            )

    return pd.DataFrame(results)


# Updated Simulation configuration
config = {
    "n_users": 10,
    "n_items": 20,
    "context_dim": 1,
    "n_reco": 4,
}

# Running the simulation
full_results = pd.concat(
    [simulate_observation_size(n, config, num_experiments) for n in observation_sizes]
)

# Plotting results
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.lineplot(
    data=full_results, x="Observation Size", y="MSE", hue="Estimator", marker="o"
)
plt.yscale("log")
plt.title("Number of Observations vs MSE")
plt.tight_layout()
plt.show()
