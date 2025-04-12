import numpy as np
import jax
import jax.numpy as jnp

from estimators.base import Estimator
from policies.base import Policy
from utils.kernels import Gaussian


class CPME(Estimator):
    """Estimator using the counterfactual policy mean embedding method

    Args:
        BaseEstimator (object): Base class for all estimators
    """

    def __init__(
        self, policy, clipping_parameter=1e-6, lambda_=1e-4, variance_penalty=True
    ):

        super().__init__(policy, clipping_parameter, lambda_, variance_penalty)
        self.ridge_lambda = lambda_

    def loss_function(self, policy, actions, contexts, losses, logged_propensities):
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

        W_pi = self.transform_weights(
            policy.evaluate(actions, contexts),
            logged_propensities,
            policy.propensity_type,
        )
        n = actions.shape[0]
        K_A_X, reg_inv_K_A_X = self.ridge_regression(actions, contexts)
        ridge_K = jnp.dot(losses, reg_inv_K_A_X)
        cpme = jnp.dot(K_A_X, W_pi)

        return 1 / n * jnp.dot(ridge_K, cpme)

    def objective_function(self, estimate):
        if self.variance_penalty:

            return jnp.mean(estimate) + self.lambda_ * jnp.sqrt(
                jnp.sum(jnp.cov(estimate))
            )
        else:
            return jnp.mean(estimate)

    def ridge_regression(self, actions, contexts):

        n = actions.shape[0]
        settings = {"bandwidth": 0.1}
        kernel_actions = Gaussian(settings)
        kernel_covariates = Gaussian(settings)

        kernel_actions.fit(actions)
        kernel_covariates.fit(contexts)

        K_A = kernel_actions.gram_matrix(actions)
        K_X = kernel_covariates.gram_matrix(contexts)

        K_A_X = K_A * K_X
        reg_inv_K_A_X = jnp.linalg.inv(K_A_X + n * self.ridge_lambda * jnp.eye(n))

        return K_A_X, reg_inv_K_A_X
