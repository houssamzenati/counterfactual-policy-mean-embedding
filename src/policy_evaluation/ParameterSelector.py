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

class ParameterSelector(object):

    """ A Class for Parameter Selection """

    def __init__(self, estimator=None):
        self._estimator = estimator
        self._parameters = None
        self._score = None

    @property
    def name(self):
        if self.estimator is None:
            return "Empty ParamterSelector"
        else:
            return "".join["ParameterSelector for ", self.estimator.name]

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

    def select_from_propensity(self, data, params_grid, null_policy, target_policy, n_splits=5):
        """ Select the best parameter using the propensity score """

        num_params = len(params_grid)
        num_data = len(data)

        kfold = StratifiedKFold(n_splits=n_splits)
        errors = np.zeros(num_params)

        # create estimators using parameter grid
        estimators = [copy.deepcopy(self.estimator) for _ in params_grid]
        for params, e in zip(params_grid, estimators):
            e.params = params

        with joblib.Parallel(n_jobs=num_params, max_nbytes=1e6) as parallel:

            for train, test in kfold.split(np.zeros(num_data), data.logging_reward):

                # split the data
                new_data = pd.concat([pd.DataFrame({'logging_reward': data.logging_reward.iloc[train],
                                                    'logging_context_vec': data.logging_context_vec.iloc[train],
                                                    'logging_reco_vec': data.logging_reco_vec.iloc[train],
                                                    'logging_reco': data.logging_reco.iloc[train],
                                                    'logging_multinomial': data.logging_multinomial.iloc[train]}),
                                      pd.DataFrame({'target_context_vec': data.logging_context_vec.iloc[test],
                                                    'target_reco_vec': data.logging_reco_vec.iloc[test],
                                                    'target_reco': data.logging_reco.iloc[test],
                                                    'target_multinomial': data.target_multinomial.iloc[test]})],
                                     axis=1)
                
                # evaluate the estimator on each split
                validate_data = data.iloc[test]
                validate_reward = data["logging_reward"].iloc[test].values
                print(null_policy)
                nullProb = [null_policy.get_propensity(row.logging_context_vec, row.logging_reco[0]) for _,row in validate_data.iterrows()]
                if not target_policy.greedy:
                    targetProb = [target_policy.get_propensity(row.target_multinomial, row.logging_reco) for _,row in validate_data.iterrows()]
                else:
                    targetProb = [1.0 if row.logging_reco == row.target_reco else 0 for _,row in validate_data.iterrows()]

                ips_w = np.divide(targetProb, nullProb)
                actual_value = np.mean(ips_w * validate_reward) / np.mean(ips_w)

                estimated_values = parallel(joblib.delayed(e.estimate)(new_data) for e in estimators)
                errors += [(est - actual_value) ** 2 for est in estimated_values]

            errors /= n_splits

        # update the best parameter setting and the new estimator
        self.parameters = params_grid[np.argmin(errors)]
        self.estimator = estimators[np.argmin(errors)]
        self.score = np.min(errors)

    def select_from_covariate_matching(self, data, params_grid):
        """ Select the best parameter using the covariate matching """