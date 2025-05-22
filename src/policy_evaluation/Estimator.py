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
        :param sim_data: a data frame consists of {context, null_reco, null_reward, target_reco, target_reward}
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

        logging_context_vec = sim_data["null_context_vec"].dropna(axis=0)
        logging_reco_vec = sim_data["null_reco_vec"].dropna(axis=0)
        logging_reward = sim_data["null_reward"].dropna(axis=0)
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



class SlateEstimator(Estimator):
    @property
    def name(self):
        return "slate_estimator"

    def __init__(self, n_reco, null_policy):
        self.n_reco = n_reco
        self.null_policy = null_policy

    def calculate_weight(self, row):
        n_items = self.null_policy.n_items
        n_reco = self.n_reco
        n_dim = n_reco * n_items
        temp_range = range(n_reco)

        exploredMatrix = np.zeros((n_reco, n_items), dtype=np.longdouble)
        exploredMatrix[temp_range, list(row.null_reco)] = 1.

        targetMatrix = np.zeros((n_reco, n_items), dtype=np.longdouble)
        targetMatrix[temp_range, list(row.target_reco)] = 1.

        posRelVector = exploredMatrix.reshape(n_dim)
        targetSlateVector = targetMatrix.reshape(n_dim)

        estimatedPhi = np.dot(self.null_policy.gammas[row.user], posRelVector)

        return np.dot(estimatedPhi, targetSlateVector)

    def estimate(self, sim_data):
        sim_data['ips_w'] = sim_data.apply(self.calculate_weight, axis=1)
        exp_reward = np.mean(sim_data['ips_w'] * sim_data['null_reward'])
        exp_weight = np.mean(sim_data['ips_w'])

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
        self.params = params

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

        null_reward = sim_data.null_reward.dropna(axis=0)

        null_context_vec = np.stack(sim_data['null_context_vec'].dropna(axis=0).values)
        null_reco_vec = np.stack(sim_data['null_reco_vec'].dropna(axis=0).values)
        target_context_vec = np.stack(sim_data['target_context_vec'].dropna(axis=0).values)
        target_reco_vec = np.stack(sim_data['target_reco_vec'].dropna(axis=0).values)

        # use median heuristic for the bandwidth parameters
        context_param = (0.5 * context_param) / np.median(pdist(np.vstack([null_context_vec, target_context_vec]), 'sqeuclidean'))
        recom_param = (0.5 * recom_param) / np.median(pdist(np.vstack([null_reco_vec, target_reco_vec]), 'sqeuclidean'))

        contextMatrix = self.context_kernel(null_context_vec, null_context_vec, context_param)
        recomMatrix = self.recom_kernel(null_reco_vec, null_reco_vec, recom_param)  #

        targetContextMatrix = self.context_kernel(null_context_vec, target_context_vec, context_param)
        targetRecomMatrix = self.recom_kernel(null_reco_vec, target_reco_vec, recom_param)

        # calculate the coefficient vector using the pointwise product kernel L_ij = K_ij.G_ij
        m = sim_data["target_reco"].dropna(axis=0).shape[0]
        n = sim_data["null_reco"].dropna(axis=0).shape[0]
        b = np.dot(np.multiply(targetContextMatrix, targetRecomMatrix), np.repeat(1.0 / m, m, axis=0))

        # solve a linear least-square
        A = np.multiply(contextMatrix, recomMatrix) + np.diag(np.repeat(n * reg_param, n))
        beta_vec, _ = scipy.sparse.linalg.cg(A, b)

        # return the expected reward as an average of the rewards, obtained from the null policy,
        # weighted by the coefficients beta from the counterfactual mean estimator.
        return np.dot(beta_vec, null_reward) / beta_vec.sum()