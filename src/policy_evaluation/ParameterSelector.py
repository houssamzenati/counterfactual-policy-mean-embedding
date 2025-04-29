from Estimator import *
from Policy import *

from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import copy

"""
The class for selecting the best parameters of the reward estimators
"""

from sklearn.model_selection import KFold


class ParameterSelector(object):
    """A Class for Parameter Selection"""

    def __init__(self, estimator=None, n_splits=5):
        self._estimator = estimator
        self._parameters = None
        self._score = None
        self.parameters = None
        self.n_splits = n_splits 

    @property
    def name(self):
        if self.estimator is None:
            return "Empty ParameterSelector"
        else:
            return "".join(["ParameterSelector for ", self.estimator.name])

    @property
    def estimator(self):
        return self._estimator

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @estimator.setter
    def estimator(self, value):
        self._estimator = value

    def select_from_propensity(
        self, sim_data, params_grid, logging_policy, target_policy
    ):
        X = np.vstack(sim_data["logging_reco_vec"].values)
        y = sim_data["logging_reward"].values

        best_mse = np.inf
        best_params = None

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for params in params_grid:
            try:
                fold_mse = []

                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    if isinstance(self.estimator, DirectEstimator):
                        n_hidden_units, batch_size, epochs = params
                        self.estimator.fit(
                            X_train,
                            y_train,
                            n_hidden_units=n_hidden_units,
                            batch_size=batch_size,
                            epochs=epochs,
                        )
                        pred_rewards = self.estimator.model.predict(
                            X_test, batch_size=256, verbose=0
                        )
                        mse = np.mean((pred_rewards.flatten() - y_test) ** 2)

                    elif isinstance(self.estimator, CMEstimator) or isinstance(
                        self.estimator, DRCMEstimator
                    ):
                        reg_param, context_bw, reco_bw = params
                        self.estimator.params = [reg_param, context_bw, reco_bw]
                        pred_rewards = np.full_like(
                            y_test, np.mean(y_train)
                        )  # Dummy prediction
                        mse = np.mean((pred_rewards.flatten() - y_test) ** 2)

                    else:
                        raise ValueError(
                            "Unsupported estimator type for parameter selection."
                        )

                    fold_mse.append(mse)

                avg_mse = np.mean(fold_mse)

                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_params = params

            except Exception as e:
                continue  # skip params that fail

        if best_params is not None:
            if isinstance(self.estimator, DirectEstimator):
                n_hidden_units, batch_size, epochs = best_params
                self.estimator.fit(
                    X,
                    y,
                    n_hidden_units=n_hidden_units,
                    batch_size=batch_size,
                    epochs=epochs,
                )
            elif isinstance(self.estimator, CMEstimator) or isinstance(
                self.estimator, DRCMEstimator
            ):
                reg_param, context_bw, reco_bw = best_params
                self.estimator.params = [reg_param, context_bw, reco_bw]
            else:
                raise ValueError("Unsupported estimator type for final assignment.")

            self.parameters = best_params
