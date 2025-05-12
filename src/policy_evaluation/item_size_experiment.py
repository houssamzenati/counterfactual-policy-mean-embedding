# Corrected Item Size Experiment for Figure 4c
import warnings
warnings.filterwarnings("ignore")
# FAST Mode Settings
num_experiments = 10
item_sizes = [20, 40, 60, 80]
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
    # IPSEstimator,
    SlateEstimator,
    DirectEstimator,
    DoublyRobustEstimator,
    CMEstimator,
)
from Estimator_CPME import (
    IPSEstimator,
    CMEbis,
    DoublyRobustbis,
    BehaviorPolicyEstimator
)
from Policy import MultinomialPolicy
from ParameterSelector import ParameterSelector

np.random.seed(2)


# Helper function for simulation
def simulate_item_size(item_size, config, num_iter):
    results = []

    user_vectors = np.random.normal(
        0, 1, size=(config["n_users"], config["context_dim"])
    )
    target_user_vectors = user_vectors * np.random.binomial(
        1, 0.5, size=user_vectors.shape
    )
    item_vectors = np.random.normal(0, 1, size=(item_size, config["context_dim"]))

    alpha = -0.3
    null_user_vectors = alpha * target_user_vectors

    null_policy = MultinomialPolicy(
        item_vectors,
        null_user_vectors,
        item_size,
        config["n_reco"],
        temperature=0.5,
        cal_gamma=True,
    )
    logging_policy = null_policy
    target_policy = MultinomialPolicy(
        item_vectors,
        target_user_vectors,
        item_size,
        config["n_reco"],
        temperature=1.0,
        cal_gamma=False,
    )
    environment = AvgEnvironment(item_vectors, user_vectors)

    reg_pow = -1
    reg_params = (10.0**reg_pow) / config["n_observation"]
    bw_params = 10.0**0
    params = [reg_params, bw_params, bw_params]
    
    seeds = np.random.randint(np.iinfo(np.int32).max, size=num_iter)

    for seed in tqdm(seeds, desc=f"Item size {item_size}"):
        np.random.seed(seed)

        sim_data = []
        for _ in range(config["n_observation"]):
            user = environment.get_context()

            logging_reco, logging_multinomial, logging_user_vector = logging_policy.recommend(user)
            target_reco, target_multinomial, _ = target_policy.recommend(user)

            observation = {
                "logging_context_vec": logging_user_vector,
                "target_context_vec": logging_user_vector,
                "logging_reco": tuple(logging_reco),
                "logging_reco_vec": np.concatenate(item_vectors[logging_reco]),
                "logging_reward": environment.get_reward(user, logging_reco),
                "target_reco": tuple(target_reco),
                "target_multinomial": target_multinomial,
                "target_reco_vec": np.concatenate(item_vectors[target_reco]),
                "target_reward": environment.get_reward(user, target_reco),
                "logging_multinomial": logging_multinomial,
                "user": user,
            }

            sim_data.append(observation)

        sim_data = pd.DataFrame(sim_data)
        
        # === Prepare estimators ===
        behavior_estimator = BehaviorPolicyEstimator(item_size)
        user_features = np.vstack(sim_data["logging_context_vec"].values)
        actions = [r[0] for r in sim_data["logging_reco"].values]  # Taking first item as action

        behavior_estimator.fit(user_features, actions)


        estimators = [
            IPSEstimator(behavior_estimator, target_policy),
            SlateEstimator(config["n_reco"], null_policy),
            DirectEstimator(),
            DoublyRobustEstimator(config["n_reco"], null_policy, target_policy),
            # CMEstimator(rbf_kernel, rbf_kernel, params),
            CMEbis(rbf_kernel, rbf_kernel, params),
            DoublyRobustbis(rbf_kernel, rbf_kernel, params, behavior_estimator, target_policy)
        ]
    
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
            sim_data, params_grid, behavior_estimator, target_policy
        )
        estimators[4] = cme_selector.estimator

        drcme_selector = ParameterSelector(estimators[5])
        params_grid = [
            [(10.0**p) / config["n_observation"], 1.0, 1.0] for p in np.arange(-6, 0, 1)
        ]
        drcme_selector.select_from_propensity(
            sim_data, params_grid, behavior_estimator, target_policy
        )
        estimators[5] = drcme_selector.estimator

        actual_value = np.mean(sim_data["target_reward"])

        for estimator in estimators:
            est_value = estimator.estimate(sim_data)
            mse = (est_value - actual_value) ** 2
            results.append(
                {"Estimator": estimator.name, "MSE": mse, "Item Size": item_size}
            )

    return pd.DataFrame(results)


# Updated Simulation configuration
config = {
    "n_users": 50,
    "context_dim": 10,
    "n_reco": 4,
    "n_observation": 5000,
}

# Running the simulation
full_results = pd.concat(
    [simulate_item_size(s, config, num_experiments) for s in item_sizes]
)

# Plotting results
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.lineplot(data=full_results, x="Item Size", y="MSE", hue="Estimator", marker="o")
plt.yscale("log")
plt.title("Item Size vs MSE (FAST Mode)")
plt.tight_layout()
plt.show()
