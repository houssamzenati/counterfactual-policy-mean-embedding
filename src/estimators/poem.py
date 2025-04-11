import numpy as np
import jax
import jax.numpy as jnp

from estimators.base import Estimator
from policies.base import Policy


class ImportanceSamplingEstimator(Estimator):
    """Estimator for the classical Multiple importance sampling method

    Args:
        BaseEstimator (object): Base class for all estimators
    """

    def __init__(
        self, policy, clipping_parameter=1e-6, lambda_=1e-4, variance_penalty=True
    ):

        super().__init__(policy, clipping_parameter, lambda_, variance_penalty)
        # Add any additional initialization code here

    def loss_function(
        self,
        policy,
        actions, contexts, losses, logged_propensities
    ):
        """return the loss function for the mixture estimator

        Args:
            theta (Array):parameter of the policy to optimize
            features_list (list): list of the past context sampled from the previous round
            actions_list (list): list of the past actions sampled from the previous round
            logged_propensities_matrix (Array): matrix of the logged propensities for all the previous rounds
            losses_list (list): list of the past losses sampled from the previous round

        Returns:
            float: the mixture estimator objective
        """

        estimates = losses * self.transform_weights(
            policy.evaluate(actions, contexts),
            logged_propensities,
            policy.propensity_type,
        )

        return self.objective_function(estimates)
