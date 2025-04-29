import os
import sys
from abc import ABCMeta, abstractmethod

import numpy as np

import jax
import jax.numpy as jnp
from policies.base import Policy


def log_normal(action, mu, sigma):
    """Log-normal distribution PDF policy

    Args:
        action (np.array)
        mu (np.array): parameter of log normal pdf
        sigma (np.array): parameter of log normal pdf
    """
    return jnp.exp(-((jnp.log(action) - mu) ** 2) / (2 * sigma**2)) / (
        action * sigma * jnp.sqrt(2 * jnp.pi)
    )


def normal(action, mu, sigma):
    """Normal distribution PDF policy

    Args:
        action (np.array)
        mu (np.array): parameter of normal pdf
        sigma (np.array): parameter of normal pdf
    """
    return jnp.exp(-((action - mu) ** 2) / (2 * sigma**2)) / (
        sigma * jnp.sqrt(2 * jnp.pi)
    )


class ContinuousPolicy(Policy):
    """Class representing a continuous policy.

    This class supports log-normal and normal distribution policies.

    Args:
        parameter (np.array): Policy parameters.
        type (str): The distribution the policy is following  ('log_normal' or 'normal').
        contextual_model (object): Contextual model for parameter estimation.
        random_seed (int): Seed for reproducibility.
        epsilon (float): Exploration Parameter.
        propensity_type (str):  Are we using a log trick to treat the propensity ('normal' or 'logarithmic')
    """

    def __init__(
        self,
        parameter,
        type,
        sigma,
        log_sigma,
        contextual_model,
        random_seed,
        epsilon=None,
    ):

        self.type = type
        self.contextual_model = contextual_model
        self.sigma = sigma
        self.log_sigma = log_sigma
        self.parameter = parameter
        self.epsilon = epsilon
        self.rng = np.random.RandomState(random_seed)
        self.propensity_type = "normal"

    def create_start_parameter(self, env):
        """Initialize the starting policy parameter based on the environment.

        Args:
            env (Environment): Environment object with dimensionality and
            scaling.
        """
        d = env.dimension
        if self.contextual_model.name == "linear":
            self.parameter = np.concatenate(
                [np.array([1.0]), self.rng.normal(scale=env.parameter_scale, size=d)]
            )
        elif self.contextual_model.name == "polynomial":
            self.parameter = self.rng.normal(
                scale=env.parameter_scale, size=d**2 + d + 1
            )
        else:
            raise NotImplementedError

    def context_modelling(self, features):
        """Model the context based on features and policy parameters.

        Args:
            features (np.array): Contextual features.

        Returns:
            np.array: Modeled context parameters.
        """
        parameters = self.parameter
        return self.contextual_model.get_parameter(parameters, features)

    def evaluate(self, actions, features):
        """Calculates the density for given actions and features.

        Args:
            actions (np.array): Actions taken.
            features (np.array): Contextual features.

        Returns:
            jnp.array: Propensity scores.
        """

        if self.type == "log_normal":
            return log_normal(actions, self.context_modelling(features), self.log_sigma)
        elif self.type == "normal":
            return normal(actions, self.context_modelling(features), self.sigma)

    def get_actions_and_propensities(self, features):
        """Generate actions and calculate their densities.

        Args:
            features (np.array): Contextual features.

        Returns:
            tuple: A tuple containing actions and their densities.
        """

        contexted_features = self.context_modelling(features)

        if self.type == "log_normal":

            actions = self.rng.lognormal(
                mean=contexted_features, sigma=self.log_sigma, size=features.shape[0]
            )
            propensities = self.evaluate(actions, features)
            return actions, propensities
        elif self.type == "normal":
            actions = self.rng.normal(
                loc=contexted_features, scale=self.sigma, size=features.shape[0]
            )
            propensities = self.evaluate(actions, features)
            return actions, propensities

    def online_evaluation(self, env):
        """Evaluate the policy online with test data from the environment.

        Args:
            env (Environment): Environment object with test data.

        Returns:
            Array: Losses of the policy on the test data.
        """

        contexts, potentials = env.test_data
        contextual_param = self.context_modelling(contexts)

        size = contexts.shape[0]
        losses = []

        for i in range(10):
            sampled_actions = self.rng.normal(
                loc=contextual_param, scale=env.logging_scale, size=size
            )
            losses += [env.get_losses_from_actions(potentials, sampled_actions)]

        losses_array = np.stack(losses, axis=0)
        return np.mean(losses_array)
