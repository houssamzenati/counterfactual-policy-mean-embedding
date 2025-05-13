from abc import abstractmethod

import numpy as np
import itertools
import scipy
import scipy.linalg
import pandas as pd
from Policy import *
from scipy.optimize import lsq_linear, nnls
from scipy.spatial.distance import pdist
import tensorflow as tf
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from scipy.stats.mstats import winsorize

import joblib
from sklearn.model_selection import StratifiedKFold

"""
Classes that represent different policy estimators for simulated experiments
"""


class Estimator(object):
    @abstractmethod
    def estimate(self, sim_data):
        """
        calculate expected reward from an observation
        :param sim_data: a data frame consists of {context, logging_reco, logging_reward, target_reco, target_reward}
        :return: expected reward (double)
        """
        pass

    @property
    @abstractmethod
    def name(self):
        pass


class IPSEstimator(Estimator):
    def __init__(self, behavior_estimator, target_policy):
        self.behavior_estimator = behavior_estimator
        self.target_policy = target_policy

    @property
    def name(self):
        return "ips_estimator"

    def calculate_weight(self, row):
        logging_prob = self.behavior_estimator.predict_proba(row.logging_context_vec, row.logging_reco[0])
        if not self.target_policy.greedy:
            target_prob = self.target_policy.get_propensity(row.target_multinomial, row.logging_reco)
        else:
            target_prob = 1.0 if row.logging_reco == row.target_reco else 0.0

        if logging_prob == 0:
            return 0.0
        return np.clip(target_prob / logging_prob, 0, 10)  # Clip for stability

    def estimate(self, sim_data):
        sim_data["ips_w"] = sim_data.apply(self.calculate_weight, axis=1)
        exp_reward = np.mean(sim_data["ips_w"] * sim_data["logging_reward"])
        exp_weight = np.mean(sim_data["ips_w"])
        return exp_reward / exp_weight

        
class CMEbis(Estimator):

    @property
    def name(self):
        return "cmebis"

    def __init__(self, context_kernel, recom_kernel, params):
        """
        :param context_kernel: the kernel function for the context variable
        :param recom_kernel: the kernel function for the recommendation
        :param params: all parameters including regularization parameter and kernel parameters
        """
        self.context_kernel = context_kernel
        self.recom_kernel = recom_kernel
        self._params = params

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value
        
    def estimate(self, sim_data):
        settings = {
        'kernel': 'gauss',
        'bandwidth': 0.1,
        }
        reg_param = self.params[0]
        context_param = self.params[1]
        recom_param = self.params[2]

        logging_reward = sim_data.logging_reward.dropna(axis=0)

        logging_context_vec = np.stack(
            sim_data["logging_context_vec"].dropna(axis=0).to_numpy()
        )
        logging_reco_vec = np.stack(
            sim_data["logging_reco_vec"].dropna(axis=0).to_numpy()
        )
        target_context_vec = np.stack(
            sim_data["target_context_vec"].dropna(axis=0).to_numpy()
        )
        target_reco_vec = np.stack(
            sim_data["target_reco_vec"].dropna(axis=0).to_numpy()
        )

        context_dists = pdist(
            np.vstack([logging_context_vec, target_context_vec]), "sqeuclidean"
        )
        recom_dists = pdist(
            np.vstack([logging_reco_vec, target_reco_vec]), "sqeuclidean"
        )
        context_median = np.median(context_dists)
        recom_median = np.median(recom_dists)

        if context_median == 0.0:
            context_param = 1.0
        else:
            context_param = (0.5 * context_param) / context_median

        if recom_median == 0.0:
            recom_param = 1.0
        else:
            recom_param = (0.5 * recom_param) / recom_median


        n = logging_context_vec.shape[0]

        K_X = self.context_kernel(
            logging_context_vec, logging_context_vec, context_param
        )
        K_T = self.recom_kernel(logging_reco_vec, logging_reco_vec, recom_param)
        K_X_tilde = self.context_kernel(
            logging_context_vec, target_context_vec, context_param
        )
        K_T_tilde = self.recom_kernel(
            logging_reco_vec, target_reco_vec, recom_param
        )

        A_logging = np.multiply(K_X, K_T) + n * reg_param * np.eye(n)
        b_logging = np.multiply(K_T_tilde, K_X_tilde)
        inv_K_tilde_K =  np.linalg.solve(A_logging, b_logging)
        return np.mean(np.dot(logging_reward,inv_K_tilde_K))


class DoublyRobustbis(Estimator):
    def __init__(self, context_kernel, recom_kernel, params, behavior_estimator, target_policy):
        self.context_kernel = context_kernel
        self.recom_kernel = recom_kernel
        self.behavior_estimator = behavior_estimator
        self.target_policy = target_policy
        self.params = params
        self.model = None

    @property
    def name(self):
        return "dr_bis"


    def calculate_weight(self, row):
        logging_prob = self.behavior_estimator.predict_proba(row.logging_context_vec, row.logging_reco[0])
        if not self.target_policy.greedy:
            target_prob = self.target_policy.get_propensity(row.target_multinomial, row.logging_reco)
        else:
            target_prob = 1.0 if row.logging_reco == row.target_reco else 0.0

        if logging_prob == 0:
            return 0.0
        return np.clip(target_prob / logging_prob, 0, 10)

    def estimate(self, sim_data):
        sim_data = sim_data.copy()

        settings = {
        'kernel': 'gauss',
        'bandwidth': 0.1,
        }
        reg_param = self.params[0]
        context_param = self.params[1]
        recom_param = self.params[2]

        logging_reward = sim_data.logging_reward.dropna(axis=0)

        logging_context_vec = np.stack(
            sim_data["logging_context_vec"].dropna(axis=0).to_numpy()
        )
        logging_reco_vec = np.stack(
            sim_data["logging_reco_vec"].dropna(axis=0).to_numpy()
        )
        target_context_vec = np.stack(
            sim_data["target_context_vec"].dropna(axis=0).to_numpy()
        )
        target_reco_vec = np.stack(
            sim_data["target_reco_vec"].dropna(axis=0).to_numpy()
        )

        context_dists = pdist(
            np.vstack([logging_context_vec, target_context_vec]), "sqeuclidean"
        )
        recom_dists = pdist(
            np.vstack([logging_reco_vec, target_reco_vec]), "sqeuclidean"
        )
        context_median = np.median(context_dists)
        recom_median = np.median(recom_dists)

        if context_median == 0.0:
            context_param = 1.0
        else:
            context_param = (0.5 * context_param) / context_median

        if recom_median == 0.0:
            recom_param = 1.0
        else:
            recom_param = (0.5 * recom_param) / recom_median


        n = logging_context_vec.shape[0]

        K_X = self.context_kernel(
            logging_context_vec, logging_context_vec, context_param
        )
        K_T = self.recom_kernel(logging_reco_vec, logging_reco_vec, recom_param)
        K_X_tilde = self.context_kernel(
            logging_context_vec, target_context_vec, context_param
        )
        K_T_tilde = self.recom_kernel(
            logging_reco_vec, target_reco_vec, recom_param
        )

        n = logging_context_vec.shape[0]
        A_logging = np.multiply(K_T, K_X) + n * reg_param * np.eye(n)
        b_target = np.multiply(K_T_tilde, K_X_tilde)
        inv_K_tilde_K =  np.linalg.solve(A_logging, b_target)
        target_predictions = np.dot(logging_reward,inv_K_tilde_K)

        b_logging = np.multiply(K_T, K_X)
        inv_K_K =  np.linalg.solve(A_logging, b_logging)
        logging_predictions = np.dot(logging_reward, inv_K_K)
    
        ips_w = sim_data.apply(self.calculate_weight, axis=1)

        estimated_reward = target_predictions + (logging_reward.values - logging_predictions) * ips_w
        return np.mean(estimated_reward)


import numpy as np
from sklearn.linear_model import LogisticRegression


class BehaviorPolicyEstimator:
    def __init__(self, n_items):
        self.model = None
        self.n_items = n_items

    def fit(self, user_features, action_indices):
        user_features = np.asarray(user_features)
        action_indices = np.asarray(action_indices)

        if user_features.ndim != 2:
            raise ValueError(f"user_features must be 2D, got {user_features.shape}")
        if action_indices.ndim != 1:
            raise ValueError(f"action_indices must be 1D, got {action_indices.shape}")
        if user_features.shape[0] != action_indices.shape[0]:
            raise ValueError(f"Mismatch: {user_features.shape[0]} samples vs {action_indices.shape[0]} actions")

        self.model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            random_state=42,
        )
        self.model.fit(user_features, action_indices)

    def predict_proba(self, user_feature, action_index):
        user_feature = np.asarray(user_feature)
        if user_feature.ndim != 1:
            raise ValueError(f"user_feature must be 1D, got {user_feature.shape}")

        user_feature = user_feature.reshape(1, -1)
        proba = self.model.predict_proba(user_feature)[0]

        # Manually expand
        full_proba = np.zeros(self.n_items)
        for cls_idx, cls in enumerate(self.model.classes_):
            full_proba[cls] = proba[cls_idx]

        return full_proba[action_index]
    
    def get_propensity(self, user_feature, action_indices):
        return self.predict_proba(user_feature, action_indices)