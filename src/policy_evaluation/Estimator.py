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


class DirectEstimator:
    def __init__(self):
        self.model = None
        self.params = None
        self.name = "Direct"

    def fit(self, features, rewards, n_hidden_units=100, batch_size=64, epochs=10):
        input_dim = features.shape[1]
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    n_hidden_units, activation="relu", input_shape=(input_dim,)
                ),
                tf.keras.layers.Dense(1),
            ]
        )

        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(
            features, rewards, batch_size=batch_size, epochs=epochs, verbose=0
        )

    def estimate(self, sim_data):
        # Prepare feature matrix (context + recommendation)
        sim_data = sim_data.copy()

        logging_context_vec = sim_data["logging_context_vec"].dropna(axis=0)
        logging_reco_vec = sim_data["logging_reco_vec"].dropna(axis=0)
        logging_reward = sim_data["logging_reward"].dropna(axis=0)
        target_context_vec = sim_data["target_context_vec"].dropna(axis=0)
        target_reco_vec = sim_data["target_reco_vec"].dropna(axis=0)

        X_logging = np.hstack(
            [np.vstack(logging_context_vec.values), np.vstack(logging_reco_vec.values)]
        )
        self.fit(X_logging, logging_reward.values)
        X_target = np.hstack(
            [np.vstack(target_context_vec.values), np.vstack(target_reco_vec.values)]
        )
        pred_rewards = self.model.predict(X_target, batch_size=256, verbose=0)
        return np.mean(pred_rewards)

    def set_parameters(self, params):
        self.params = params


class IPSEstimator(Estimator):
    @property
    def name(self):
        return "ips_estimator"

    def __init__(
        self,
        n_reco: int,
        logging_policy: MultinomialPolicy,
        target_policy: MultinomialPolicy,
    ):
        """
        :param n_reco: number of recommendation
        :param logging_policy: a policy used to generate data
        :param target_policy: a policy that we want to estimate its reward
        """
        self.n_reco = n_reco
        self.logging_policy = logging_policy
        self.target_policy = target_policy

    def calculate_weight(self, row):
        loggingProb = self.logging_policy.get_propensity(
            row.logging_multinomial, row.logging_reco
        )
        if not self.target_policy.greedy:
            targetProb = self.target_policy.get_propensity(
                row.target_multinomial, row.logging_reco
            )
        else:
            targetProb = 1.0 if row.logging_reco == row.target_reco else 0

        return targetProb / loggingProb

    def estimate(self, sim_data):
        sim_data["ips_w"] = sim_data.apply(self.calculate_weight, axis=1)
        exp_reward = np.mean(sim_data["ips_w"] * sim_data["logging_reward"])
        exp_weight = np.mean(sim_data["ips_w"])

        return exp_reward / exp_weight


class DoublyRobustEstimator:
    @property
    def name(self):
        return "doubly robust estimator"

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    def __init__(self, n_reco, logging_policy, target_policy, params=(40, 1024, 100)):
        super().__init__()
        self.n_reco = n_reco
        self.logging_policy = logging_policy
        self.target_policy = target_policy
        self.params = params
        self.model = None

    def fit(self, features, rewards):
        input_dim = features.shape[1]
        n_hidden_units = self.params[0]

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(n_hidden_units, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(
            features,
            rewards,
            batch_size=self.params[1],
            epochs=self.params[2],
            verbose=0,
        )

    def estimate(self, sim_data):
        sim_data = sim_data.copy()

        logging_context_vec = sim_data["logging_context_vec"].dropna(axis=0)
        logging_reco_vec = sim_data["logging_reco_vec"].dropna(axis=0)
        logging_reward = sim_data["logging_reward"].dropna(axis=0)
        target_context_vec = sim_data["target_context_vec"].dropna(axis=0)
        target_reco_vec = sim_data["target_reco_vec"].dropna(axis=0)

        X_logging = np.hstack(
            [np.vstack(logging_context_vec.values), np.vstack(logging_reco_vec.values)]
        )
        X_target = np.hstack(
            [np.vstack(target_context_vec.values), np.vstack(target_reco_vec.values)]
        )

        # Fit model on logging data
        self.fit(X_logging, logging_reward.values)

        # Predict
        logging_predictions = self.model.predict(X_logging, batch_size=256).flatten()
        target_predictions = self.model.predict(X_target, batch_size=256).flatten()

        ips_w = self.calculate_weights(sim_data)

        estimated_reward = (
            target_predictions + (logging_reward.values - logging_predictions) * ips_w
        )

        return np.mean(estimated_reward)

    def calculate_weight(self, row):
        logging_multinomial = row["logging_multinomial"]
        target_multinomial = row["target_multinomial"]

        if logging_multinomial is None or target_multinomial is None:
            return 0.0

        # both are arrays, take their product
        num = np.prod(target_multinomial)
        den = np.prod(logging_multinomial)

        if den == 0:
            return 0.0
        return num / den

    def calculate_weights(self, sim_data):
        w = sim_data.apply(self.calculate_weight, axis=1)
        return np.clip(w, 0, np.quantile(w, 0.99))


class SlateEstimator(Estimator):
    @property
    def name(self):
        return "slate_estimator"

    def __init__(self, n_reco, logging_policy):
        self.n_reco = n_reco
        self.logging_policy = logging_policy

    def calculate_weight(self, row):
        n_items = self.logging_policy.n_items
        n_reco = self.n_reco
        n_dim = n_reco * n_items
        temp_range = range(n_reco)

        exploredMatrix = np.zeros((n_reco, n_items), dtype=np.longdouble)
        exploredMatrix[temp_range, list(row.logging_reco)] = 1.0

        targetMatrix = np.zeros((n_reco, n_items), dtype=np.longdouble)
        targetMatrix[temp_range, list(row.target_reco)] = 1.0

        posRelVector = exploredMatrix.reshape(n_dim)
        targetSlateVector = targetMatrix.reshape(n_dim)

        estimatedPhi = np.dot(self.logging_policy.gammas[row.user], posRelVector)

        return np.dot(estimatedPhi, targetSlateVector)

    def estimate(self, sim_data):
        sim_data["ips_w"] = sim_data.apply(self.calculate_weight, axis=1)
        exp_reward = np.mean(sim_data["ips_w"] * sim_data["logging_reward"])
        exp_weight = np.mean(sim_data["ips_w"])

        return exp_reward / exp_weight


"""
The counterfactual mean embedding estimator 
"""


class CMEstimator(Estimator):
    @property
    def name(self):
        return "cme_estimator"

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
        """
        Calculate and return a coefficient vector (beta) of the counterfactual mean embedding of reward distribution.
        """

        # extract the regularization and kernel parameters
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

        # safe median heuristic for the bandwidth parameters
        context_dists = pdist(
            np.vstack([logging_context_vec, target_context_vec]), "sqeuclidean"
        )
        context_median = np.median(context_dists)
        if context_median == 0.0:
            context_param = 1.0
        else:
            context_param = (0.5 * context_param) / context_median

        recom_dists = pdist(
            np.vstack([logging_reco_vec, target_reco_vec]), "sqeuclidean"
        )
        recom_median = np.median(recom_dists)
        if recom_median == 0.0:
            recom_param = 1.0
        else:
            recom_param = (0.5 * recom_param) / recom_median

        contextMatrix = self.context_kernel(
            logging_context_vec, logging_context_vec, context_param
        )
        recomMatrix = self.recom_kernel(logging_reco_vec, logging_reco_vec, recom_param)

        targetContextMatrix = self.context_kernel(
            logging_context_vec, target_context_vec, context_param
        )
        targetRecomMatrix = self.recom_kernel(
            logging_reco_vec, target_reco_vec, recom_param
        )

        # calculate the coefficient vector using the pointwise product kernel L_ij = K_ij.G_ij
        m = sim_data["target_reco"].dropna(axis=0).shape[0]
        n = sim_data["logging_reco"].dropna(axis=0).shape[0]
        b = np.dot(
            np.multiply(targetContextMatrix, targetRecomMatrix),
            np.repeat(1.0 / m, m, axis=0),
        )

        # solve a linear least-square
        A = np.multiply(contextMatrix, recomMatrix) + np.diag(
            np.repeat(n * reg_param, n)
        )
        beta_vec, _ = scipy.sparse.linalg.cg(A, b, tol=1e-06)

        # return the expected reward as an average of the rewards, obtained from the logging policy,
        # weighted by the coefficients beta from the counterfactual mean estimator.
        return np.dot(beta_vec, logging_reward) / beta_vec.sum()


class DRCMEstimator(Estimator):
    @property
    def name(self):
        return "dr_cme_estimator"

    def __init__(
        self, context_kernel, recom_kernel, params, logging_policy, target_policy
    ):
        self.context_kernel = context_kernel
        self.recom_kernel = recom_kernel
        self._params = params
        self.logging_policy = logging_policy
        self.target_policy = target_policy
        self.model = None
        print(self._params)

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    def set_parameters(self, params):
        self.params = params

    def estimate(self, sim_data):
        # Extract data
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
        logging_reward = sim_data["logging_reward"].dropna(axis=0).values

        # Kernel median heuristic
        reg_param = self.params[0]
        context_param = self.params[1]
        recom_param = self.params[2]
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

        contextMatrix = self.context_kernel(
            logging_context_vec, logging_context_vec, context_param
        )
        recomMatrix = self.recom_kernel(logging_reco_vec, logging_reco_vec, recom_param)
        targetContextMatrix = self.context_kernel(
            logging_context_vec, target_context_vec, context_param
        )
        targetRecomMatrix = self.recom_kernel(
            logging_reco_vec, target_reco_vec, recom_param
        )

        m = target_context_vec.shape[0]
        n = logging_context_vec.shape[0]

        b = np.dot(
            np.multiply(targetContextMatrix, targetRecomMatrix),
            np.repeat(1.0 / m, m, axis=0),
        )
        A = np.multiply(contextMatrix, recomMatrix) + np.diag(
            np.repeat(n * reg_param, n)
        )

        beta_vec, _ = scipy.sparse.linalg.cg(A, b, tol=1e-06)
        cme_prediction = np.dot(beta_vec, logging_reward) / beta_vec.sum()

        # Estimate IPS weights
        def calculate_weight(row):
            loggingProb = self.logging_policy.get_propensity(
                row.logging_multinomial, row.logging_reco
            )
            if not self.target_policy.greedy:
                targetProb = self.target_policy.get_propensity(
                    row.target_multinomial, row.logging_reco
                )
            else:
                targetProb = 1.0 if row.logging_reco == row.target_reco else 0.0
            return targetProb / loggingProb

        ips_weights = sim_data.apply(calculate_weight, axis=1).values

        # CME estimated rewards at logging points (reweighting)
        b_logging = np.dot(
            np.multiply(contextMatrix, contextMatrix),
            ips_weights,
        )

        # solve a linear least-square
        A_logging = np.multiply(contextMatrix, recomMatrix) + np.diag(
            np.repeat(n * reg_param, n)
        )
        # beta_vec_logging, _ = scipy.sparse.linalg.cg(A_logging, b_logging, tol=1e-06)
        beta_vec_logging = np.linalg.solve(A_logging, b_logging)

        cme_pred_logging = (
            np.dot(beta_vec_logging, logging_reward) / beta_vec_logging.sum()
        )
        # cme_pred_logging = np.dot(beta_vec_logging, logging_reward)
        # DR-CME estimation
        dr_terms = ips_weights * logging_reward / np.mean(
            ips_weights
        ) - cme_pred_logging / np.mean(ips_weights)
        correction = np.mean(dr_terms)

        return correction + cme_prediction


import numpy as np
from sklearn.linear_model import LogisticRegression


class BehaviorPolicyEstimator:
    def __init__(self):
        self.model = None

    def fit(self, user_features, action_indices):
        """
        Fit a multinomial logistic regression model on (user_feature, action) pairs.

        Parameters
        ----------
        user_features: array-like of shape (n_samples, d_context)
        action_indices: array-like of shape (n_samples,)
        """
        user_features = np.asarray(user_features)
        action_indices = np.asarray(action_indices)

        if user_features.ndim != 2:
            raise ValueError(f"user_features must be 2D, got {user_features.shape}")
        if action_indices.ndim != 1:
            raise ValueError(f"action_indices must be 1D, got {action_indices.shape}")
        if user_features.shape[0] != action_indices.shape[0]:
            raise ValueError(
                f"Mismatch: {user_features.shape[0]} samples vs {action_indices.shape[0]} actions"
            )

        self.model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        self.model.fit(user_features, action_indices)

    def predict_proba(self, user_feature, action_index):
        """
        Predict estimated propensity score \hat{\pi}_0(a | u).

        Parameters
        ----------
        user_feature: array-like of shape (d_context,)
        action_index: int

        Returns
        -------
        float: probability \hat{\pi}_0(action_index | user_feature)
        """
        user_feature = np.asarray(user_feature)
        if user_feature.ndim != 1:
            raise ValueError(f"user_feature must be 1D, got {user_feature.shape}")

        user_feature = user_feature.reshape(1, -1)
        proba = self.model.predict_proba(user_feature)[0]

        return proba[action_index]
