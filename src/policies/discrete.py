from policies.base import Policy

import numpy as np

import jax
import jax.numpy as jnp

from jax.scipy.special import expit as jexpit


class DiscretePolicy(Policy):
    """A class representing a discrete policy.

    Args:
        parameter (ndarray): Policy parameters.
        random_seed (int): Seed for random number generation.
        epsilon (float, optional): Exploration-exploitation trade-off.
        type (str, optional): Distribution Law of the policy, in a discrete setting the distribution isn't defined analytically .
        contextual_model (object, optional): Model for contextual parameters, also None in a discrete setting.
        propensity_type (str, optional):  Are we using a log trick to treat the propensity ('normal' or 'logarithmic')
    """

    def __init__(
        self,
        parameter,
        random_seed,
        epsilon=0,
        sigma=None,
        log_sigma=None,
        type=None,
        contextual_model=None,
        propensity_type="logarithmic",
    ):
        super().__init__()
        self.parameter = parameter
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.epsilon = epsilon
        self.type = type
        self.contextual_model = contextual_model
        self.propensity_type = propensity_type
        self.start_sigma = None
        self.sigma = None

    def create_start_parameter(self, env):
        d = env.dimension
        beta = jnp.array(np.zeros(d))
        self.parameter = beta

    def context_modelling(self, features):
        """Model the context to get the parameter

        Args:
            features (array): array of feature vector
        """
        parameter = self.parameter
        return self.contextual_model._linear_modelling(parameter, features)

    def evaluate(self, actions, features, targets):
        """Returns the probability of selecting an action given the state

        Args:
            features (array): array of feature vector
            actions (array): array of actions
        """
        # Implement the logic to select an action based on the state

        wx = self.context_modelling(features)
        unif_wx = jnp.zeros(wx.shape)
        actions_sign = 2 * actions - 1
        sigm = (1 - self.epsilon) * jexpit(actions_sign * wx) + self.epsilon * jexpit(
            actions_sign * unif_wx
        )
        log_propensities = jnp.log(sigm).sum(axis=1)
        if self.propensity_type == "logarithmic":

            return log_propensities
        else:
            propensities = jnp.exp(log_propensities)
            return propensities

    def get_actions_and_propensities(self, features, targets):
        """Generate actions and their associated propensities.

        Args:
            features (ndarray): Contextual features.
            targets (ndarray): Target values.

        Returns:
            tuple: Actions and their log-propensities.
        """

        n, k = features.shape[0], targets.shape[1]
        wx = self.context_modelling(features)
        unif_wx = jnp.zeros(wx.shape)
        actions_sign = jnp.array(2 * targets - 1)
        sigm = (1 - self.epsilon) * jexpit(actions_sign * wx) + self.epsilon * jexpit(
            actions_sign * unif_wx
        )

        # get actions
        actions = (self.rng.uniform(size=(n, k)) < sigm).astype(int)
        zero_chosen = np.where(actions == 0)

        propensities = np.array(sigm)
        propensities[zero_chosen] = 1 - sigm[zero_chosen]

        log_propensities = jnp.log(propensities).sum(axis=1)

        if self.propensity_type == "logarithmic":

            return actions, log_propensities
        else:
            propensities = jnp.exp(log_propensities)
            return actions, propensities

    def online_evaluation(self, env):
        """Perform online evaluation of the policy.

        Args:
            env (Environment): Environment object with test data.

        Returns:
            float: Average propensity across the test data.
        """

        contexts, targets = env.X_test, env.y_test
        y_invert = 1 - targets

        wx = self.context_modelling(contexts)
        unif_wx = jnp.zeros(wx.shape)
        actions_sign = 2 * y_invert - 1
        sigm = (1 - self.epsilon) * jexpit(actions_sign * wx) + self.epsilon * jexpit(
            actions_sign * unif_wx
        )

        return sigm.sum() / (targets.shape[1] * targets.shape[0])
