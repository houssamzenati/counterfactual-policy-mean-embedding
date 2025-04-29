import os
import sys

# Libraries
import os
import numpy as np
from scipy.stats import norm
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, RidgeCV


# Get the current working directory
base_dir = os.path.join(os.getcwd(), "../..")
sys.path.append(base_dir)

import numpy as np

#
from data.base import Dataset


class Pricing(Dataset):
    """Parent class for Data"""

    def __init__(self, name, mode="quadratic", **kw):
        """Initializes the class

        Attributes:
            name (str): name of the dataset
            n_samples (int): number of samples
            start_mean (float): starting mean of the logging policy
            start_std (float): starting std of the logging policy
            start_sigma (float): starting parameter sigma of the logging policy
            start_mu (float): starting parameter mu of the logging policy
            mus (list): list of means of the potential group labels
            targets_sigma (float): variance of the potential group labels

        Note:
            Setup done in auxiliary private method
        """
        super(Pricing, self).__init__(**kw)
        self.name = name
        self.policy_type = 'continuous'
        self.dimension = (10, 1)
        self.l = 3
        self.start_std = 0.5
        self.start_mean = 1.5
        self.start_sigma = np.sqrt(np.log(self.start_std**2 / self.start_mean**2 + 1))
        self.start_mu = np.log(self.start_mean) - self.start_sigma**2 / 2
        self.mode = mode
        self.a, self.b = self.get_functions(self.mode)
        self.test_data = self.sample_data(10000, 0)
        self.parameter_scale = 0.01
        self.param_dimension = (11, 1)
        self.start_sigma = None
        self.logging_scale = 0.5

    def get_functions(self, mode):
        """
        Returns reward functions based on the specified mode.

        Parameters
        ----------
        mode : str
            Mode of the environment (e.g., "quadratic").

        Returns
        -------
        tuple of callables
            Functions `a` and `b` defining the reward structure.
        """

        a = lambda z: 2 * z**2
        b = lambda z: 0.6 * z
        return a, b

    def sample_contexts_targets(self, n_samples):

        X = self.rng.uniform(low=1, high=2, size=(n_samples, self.dimension[0]))
        v = self._get_targets(X)
        return X, v

    def sample_logging_actions(self, targets):
        p = self.rng.normal(loc=targets, scale=self.start_std)
        return p

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
        p = self.sample_logging_actions(targets)
        losses = self.get_losses_from_actions(targets, p)
        propensities = norm(loc=targets, scale=self.start_std).pdf(p)

        return p, contexts, losses, propensities, targets

    def sample_data(self, n_samples, index):

        _, contexts, _, _, targets = self.generate_data(n_samples)
        return contexts, targets

    def _get_targets(self, z):
        """
        Computes target values from context features.

        Parameters
        ----------
        z : np.ndarray
            Contextual features.

        Returns
        -------
        np.ndarray
            Target values computed as the mean of the first `l` dimensions.
        """

        return np.mean(z[:, : self.l], axis=1)

    def get_losses_from_actions(self, z_bar, actions):
        """
        Computes losses based on actions and target values.

        Parameters
        ----------
        z_bar : np.ndarray
            Target values.
        actions : np.ndarray
            Actions taken by the policy.

        Returns
        -------
        np.ndarray
            Loss values for each action-target pair.
        """

        epsilon_noise = self.rng.normal(loc=np.zeros_like(z_bar), scale=1)
        losses = -(actions * (self.a(z_bar) - self.b(z_bar) * actions) + epsilon_noise)
        return np.minimum(losses, np.zeros_like(losses))

    def sample_losses(self, actions, contexts, targets):
        """
        Computes losses based on actions and target values.
        """
        return self.get_losses_from_actions(targets, actions)

    def get_optimal_parameter(self, contextual_modelling):
        """
        Computes the optimal parameter for a given contextual modeling method.

        Parameters
        ----------
        contextual_modelling : str
            The type of contextual modeling to use (e.g., "linear",
            "polynomial").

        Returns
        -------
        tuple
            - Optimal parameters as a numpy array.
            - Fitted RidgeCV model.
        """

        z, z_bar = self.sample_data(10000, 0)
        pistar_determinist = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
        optimal_prices = self.a(z_bar) / (2 * self.b(z_bar))
        if contextual_modelling == "linear":
            embedding = z
        elif contextual_modelling == "polynomial":
            quadra_z = np.einsum("ij,ih->ijh", z, z).reshape(z.shape[0], -1)
            embedding = np.hstack([z, quadra_z])
        else:
            return
        pistar_determinist.fit(embedding, optimal_prices)
        return (
            np.concatenate(
                [np.array([pistar_determinist.intercept_]), pistar_determinist.coef_]
            ).reshape(-1, 1),
            pistar_determinist,
        )

    def online_evaluation(self, optimized_param, contextual_modelling, random_seed):

        rng = np.random.RandomState(random_seed)
        contexts, potentials = self.test_data
        contextual_param = contextual_modelling.get_parameter(optimized_param, contexts)
        contextual_param = contextual_param.reshape(
            -1,
        )
        size = contexts.shape[0]
        losses = []

        for i in range(10):
            sampled_actions = rng.normal(contextual_param, self.logging_scale, size)
            losses += [self.get_losses_from_actions(potentials, sampled_actions)]

        losses_array = np.stack(losses, axis=0)
        return np.mean(losses_array)

    def get_optimal_loss(self, contextual_model):
        optimal_theta, pistar_determinist = self.get_optimal_parameter(
        contextual_model.name
        )
        optimal_loss = self.online_evaluation(
            optimal_theta, contextual_model
        )
        return optimal_loss