import os
import sys

# Libraries
import os
import numpy as np
import jax.numpy as jnp
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, RidgeCV


base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

import numpy as np

#
from data.base import Dataset


class Advertising(Dataset):

    def __init__(self, name, **kw):
        """Initializes the class

        Attributes:
            name (str): name of the dataset
            n_samples (int): number of samples
            start_mean (float): starting mean of the logging policy
            start_std (float): starting std of the logging policy
            start_sigma (float): starting parameter sigma of the logging policy
            start_mu (float): starting parameter mu of the logging policy
            mus (list): list of means of the potential group labels
            potentials_sigma (float): variance of the potential group labels

        Note:
            Setup done in auxiliary private method
        """
        super(Advertising, self).__init__(**kw)
        self.name = name
        self.dimension = (2, 1)
        # self.n_samples = n_samples
        self.start_mean = 2.0
        self.start_std = 0.3
        self.start_sigma = np.sqrt(np.log(self.start_std**2 / self.start_mean**2 + 1))
        self.start_mu = np.log(self.start_mean) - self.start_sigma**2 / 2
        self.mus = [3, 1, 0.1]
        self.targets_sigma = 0.5
        self.evaluation_offline = False
        self.test_data = self.sample_data(10000, 0)
        self.logging_scale = 0.3
        self.parameter_scale = 1
        self.start = 0
        self.param_dimension = (3, 1)

    def sample_contexts_targets(self, n_samples):

        X, y = datasets.make_moons(
            n_samples=n_samples, noise=0.05, random_state=self.rng
        )
        v = self._get_targets(y)
        return X, v

    def _get_targets(self, y):
        """
        Args
            y (np.array): group labels

        """
        n_samples = y.shape[0]
        groups = [
            self.rng.normal(loc=mu, scale=self.targets_sigma, size=n_samples)
            for mu in self.mus
        ]
        targets = np.ones_like(y, dtype=np.float64)
        for y_value, group in zip(np.unique(y), groups):
            targets[y == y_value] = group[y == y_value]

        return np.abs(targets)


    def sample_logging_actions(self, n_samples):
        actions = self.rng.normal(
            loc=self.start_mean, scale=self.start_std, size=n_samples
        )

        return actions

    def generate_data(self, n_samples):
        """
        Sets up experiments and generates data.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        tuple of np.ndarray
            - Actions sampled by the logging policy.
            - Contextual features.
            - Loss values.
            - Propensity scores for the actions.
            - Target values.
        """

        contexts, targets = self.sample_contexts_targets(n_samples)

        actions = self.sample_logging_actions(n_samples)
        losses = self.get_losses_from_actions(targets, actions)
        propensities = self.logging_policy(actions, self.start_mean, self.start_std)
        return actions, contexts, losses, propensities, targets

    @staticmethod
    # def logging_policy(action, mu, sigma):
    #     """Log-normal distribution PDF policy

    #     Args:
    #         action (np.array)
    #         mu (np.array): parameter of log normal pdf
    #         sigma (np.array): parameter of log normal pdf
    #     """
    #     return jnp.exp(-((jnp.log(action) - mu) ** 2) / (2 * sigma**2)) / (
    #         action * sigma * jnp.sqrt(2 * jnp.pi)
    #     )
    def logging_policy(action, mu, sigma):
        """Normal distribution PDF policy

        Args:
            action (np.array)
            mu (np.array): parameter of normal pdf
            sigma (np.array): parameter of normal pdf
        """
        return jnp.exp(-((action - mu) ** 2) / (2 * sigma**2)) / (
            sigma * jnp.sqrt(2 * jnp.pi)
        )

    def get_logging_data(self, n_samples):
        """_summary_

        Args:
            n_samples (_type_): _description_

        Returns:
            _type_: _description_
        """

        actions, contexts, losses, propensities, _ = self.generate_data(n_samples)
        return actions, contexts, losses, propensities

    def sample_data(self, n_samples, index):

        _, contexts, _, _, targets = self.generate_data(n_samples)
        return contexts, targets

    @staticmethod
    def get_losses_from_actions(targets, actions):
        return -np.maximum(
            np.where(
                actions < targets,
                actions / targets,
                -0.5 * actions + 1 + 0.5 * targets,
            ),
            -0.1,
        )

    def sample_losses(self, actions, contexts, targets):
        return self.get_losses_from_actions(targets, actions)

    def get_optimal_parameter(self, contextual_modelling):
        features, targets = self.sample_data(10000, 0)
        pistar_determinist = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])

        if contextual_modelling == "linear":
            embedding = features
        elif contextual_modelling == "polynomial":
            quadra_features = np.einsum("ij,ih->ijh", features, features).reshape(
                features.shape[0], -1
            )
            embedding = np.hstack([features, quadra_features])
        else:
            return
        pistar_determinist.fit(embedding, targets)
        return (
            np.concatenate(
                [
                    np.array([pistar_determinist.intercept_]).reshape(1, 1),
                    pistar_determinist.coef_.reshape(2, 1),
                ],
                axis=0,
            ),
            pistar_determinist,
        )

    class LoggingPolicy:
        """Logging policy for the environment"""

        def __init__(self, env, random_seed):

            self.env = env
            self.rng = np.random.RandomState(random_seed)
            self.propensity_type = "normal"

        def get_propensity(self, actions, features, targets):
            propensities = self.env.logging_policy(
                actions, self.env.start_mean, self.env.start_std
            )
            return propensities

        def get_actions_and_propensities(self, features, targets):
            n_samples = len(targets)
            actions = self.env.sample_logging_actions(n_samples)

            propensities = self.get_propensity(actions, features, targets)
            return actions, propensities

        def online_evaluation(self, env, random_seed):

            rng = np.random.RandomState(random_seed)
            contexts, targets = env.test_data
            size = contexts.shape[0]
            losses = []

            for i in range(10):
                sampled_actions = self.env.sample_logging_actions(len(targets))
                losses += [env.get_losses_from_actions(targets, sampled_actions)]

            losses_array = np.stack(losses, axis=0)
            return losses_array.mean()
