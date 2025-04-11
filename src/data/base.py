import os
import sys
from abc import ABCMeta, abstractmethod

import numpy as np
import jax.numpy as jnp

base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(base_dir)

from sklearn.model_selection import GridSearchCV
from sklearn.base import clone


class Dataset(object):
    """
    General abstract class for policy learning environments.

    Notes
    -----
    This class is intended as a base for specific environment implementations. 
    It requires subclassing and implementing abstract methods.
    """


    __metaclass__ = ABCMeta

    def __init__(self, random_seed=42):
        """
        Initializes the environment.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility (default is 42).

        Attributes
        ----------
        rng : numpy.RandomState
            Random number generator for reproducibility.
        """

        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

 
    @abstractmethod
    def sample_logging_actions(self, n_samples):
        """
        Samples logging actions for the environment.

        Parameters
        ----------
        n_samples : int
            Number of actions to generate.
        """

        pass

    @abstractmethod
    def sample_contexts_targets(self, n_samples):
        """
        Samples context-target pairs for the environment.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        """

        pass

    @abstractmethod
    def sample_losses(self, actions, contexts, targets):
        """Estimates causal effect on data.

        Parameters
        ----------
        actions : array-like, shape (n_samples,)
            Treatment values, binary.

        contexts : array-like, shape (n_samples, n_features_covariates)
            Covariates (potential confounders).
        targets : array-like, shape (n_samples,)
            Outcome values, continuous.

        """
        pass

