import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel
# from scipy.spatial.distance import pdist
from Environment import AvgEnvironment
from Policy import MultinomialPolicy
from ParameterSelector import ParameterSelectorWithBehaviorEstimator
from Estimator_CPME import *
# import joblib
import os

if not os.path.exists("./Results_UnknownProps"):
    os.mkdir("./Results_UnknownProps")

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

config = {
    "n_users": 50,
    "n_items": 20,
    "context_dim": 10,
    # "n_reco": 4,
    "n_observation": 2000,
}

num_iter = 30
# observation_sizes = [100, 1000, 5000]
# num_items_list = [20, 40, 60, 80]
# reco_sizes_list = [2, 3, 4, 5, 6, 7]
reco_sizes_list = [5, 6, 7]

def simulate_recom_size(n_reco, config, num_iter):
    config['n_reco'] = n_reco
    obs_size = config['n_observation']
    results = []

    # === Generate environment ===
    user_vectors = np.random.normal(0, 1, size=(config["n_users"], config["context_dim"]))
    target_user_vectors = user_vectors * np.random.binomial(1, 0.5, size=user_vectors.shape)
    item_vectors = np.random.normal(0, 1, size=(config["n_items"], config["context_dim"]))

    alpha = -0.3
    logging_user_vectors = alpha * target_user_vectors

    logging_policy = MultinomialPolicy(item_vectors, logging_user_vectors, config["n_items"], config["n_reco"], temperature=0.5, cal_gamma=True)
    target_policy = MultinomialPolicy(item_vectors, target_user_vectors, config["n_items"], config["n_reco"], temperature=1.0, cal_gamma=False)
    environment = AvgEnvironment(item_vectors, user_vectors)

    seeds = np.random.randint(np.iinfo(np.int32).max, size=num_iter)

    for seed in tqdm(seeds, desc=f"Recommendation size {n_reco}"):
        np.random.seed(seed)

        # === Generate simulation data ===
        sim_data = []
        for _ in range(obs_size):
            user = environment.get_context()

            logging_reco, logging_multinomial, logging_user_vector = logging_policy.recommend(user)
            target_reco, target_multinomial, _ = target_policy.recommend(user)

            sim_data.append({
                "null_context_vec": logging_user_vector,
                "target_context_vec": logging_user_vector,
                "null_reco": tuple(logging_reco),
                "null_reco_vec": np.concatenate(item_vectors[logging_reco]),
                "null_reward": environment.get_reward(user, logging_reco),
                "target_reco": tuple(target_reco),
                "target_multinomial": target_multinomial,
                "target_reco_vec": np.concatenate(item_vectors[target_reco]),
                "target_reward": environment.get_reward(user, target_reco),
                "null_multinomial": logging_multinomial,
                "user": user,
            })

        sim_data = pd.DataFrame(sim_data)

        # === Prepare estimators ===
        behavior_estimator = BehaviorPolicyEstimator(config["n_items"])
        user_features = np.vstack(sim_data["null_context_vec"].values)
        actions = [r[0] for r in sim_data["null_reco"].values]  # Taking first item as action

        behavior_estimator.fit(user_features, actions)

        estimators = [
            IPSEstimator(behavior_estimator, target_policy, null_propensity_known = False),
            DirectEstimator(),
            DoublyRobustEstimator(behavior_estimator, target_policy, null_propensity_known = False),
            # CMEstimator(rbf_kernel, rbf_kernel, params=[5e-5, 1.0, 1.0]),
            # DRCMEstimator(rbf_kernel, rbf_kernel, [1e-3, 1.0, 1.0], logging_policy, target_policy), 
            CMEbis(rbf_kernel, rbf_kernel, params=[5e-5, 1.0, 1.0]),
            DoublyRobustbis(rbf_kernel, rbf_kernel, [5e-5, 1.0, 1.0], behavior_estimator, target_policy, null_propensity_known = False)
        ]

        # parameter selection
        direct_selector = ParameterSelectorWithBehaviorEstimator(estimators[1])  # direct estimator
        params_grid = [(n_hiddens, 1024, 100) for n_hiddens in [50, 100, 150, 200]]
        direct_selector.select_from_propensity(sim_data, params_grid, behavior_estimator, target_policy)
        estimators[1] = direct_selector.estimator
        
        estimators[2].params = direct_selector.parameters  # doubly robust estimator
        
        cme_selector = ParameterSelectorWithBehaviorEstimator(estimators[3])  # cme estimator
        # params_grid = [[(10.0 ** p) / config['n_observation'], 1.0, 1.0] for p in np.arange(-7, 0, 1)]
        params_grid = [[(10.0 ** p), 1.0, 1.0] for p in np.arange(-8, -3, 1)]
        cme_selector.select_from_propensity(sim_data, params_grid, behavior_estimator, target_policy)
        estimators[3] = cme_selector.estimator
        
        estimators[4].params = estimators[3]._params
        # === Prepare features for reward estimators ===
        logging_context_vec = np.vstack(sim_data["null_context_vec"].dropna().values)
        logging_reco_vec = np.vstack(sim_data["null_reco_vec"].dropna().values)
        logging_reward = sim_data["null_reward"].dropna().values

        X_logging = np.hstack([logging_context_vec, logging_reco_vec])

        # === Train reward models where needed ===
        for estimator in estimators:
            if isinstance(estimator, DirectEstimator):
                estimator.fit(X_logging, logging_reward, n_hidden_units=estimator.params[0], batch_size=1024, epochs=100)
            if isinstance(estimator, DoublyRobustEstimator):
                estimator.fit(X_logging, logging_reward)

        # === Calculate results ===
        actual_value = np.mean(sim_data["target_reward"])

        for estimator in estimators:
            est_value = estimator.estimate(sim_data)
            mse = (est_value - actual_value) ** 2
            results.append({
                "Estimator": estimator.name,
                "MSE": mse,
                "n_reco": n_reco
            })

    return pd.DataFrame(results)


# Running the simulation
full_results = pd.concat(
    [simulate_recom_size(n, config, num_iter) for n in reco_sizes_list]
)

# full_results = joblib.Parallel(n_jobs=-1, verbose=0)(
#             joblib.delayed(simulate_observation_size)(n, config, num_iter) for n in observation_sizes
#         )

# Save results
full_results.to_csv("Results_UnknownProps/OPE_n_recommendations_result_n_reco_5_to_7_Unknown_Propensities.csv", index=False)