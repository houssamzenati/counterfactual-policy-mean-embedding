import os
import sys
from abc import ABCMeta, abstractmethod


import jaxopt

import jax.numpy as jnp
import numpy as np

from policies.base import Policy
from policies.continuous import ContinuousPolicy
from policies.discrete import DiscretePolicy


class Estimator(object):
    """base class for all estimators

    Args:
        object (class): base class for all estimators
    """

    def __init__(
        self, policy, clipping_parameter=1e-6, lambda_=1e-4, variance_penalty=True
    ):
        """Initializes the Estimator.

        Parameters
        ----------
        settings : dict
            Estimator configuration settings.
        verbose : bool
            Whether to enable verbose output.
        args, kwargs : optional
            Additional arguments.

        """
        self.rng = np.random.RandomState(123)
        self.clipping_parameter = clipping_parameter
        self.policy = policy
        self.list_of_parameters = []
        self.lambda_ = lambda_
        self.variance_penalty = variance_penalty  # do you want regular ERM or CRM

    @abstractmethod
    def loss_function(self, policy, actions, contexts, losses, propensities, targets):
        pass

    def transform_weights(self, pi_propensities, logged_propensities, propensity_type):
        if propensity_type == "normal":
            if self.clipping_parameter:
                return pi_propensities / (
                    logged_propensities + self.clipping_parameter * pi_propensities
                )
            else:
                return pi_propensities / logged_propensities
        elif propensity_type == "logarithmic":
            if self.clipping_parameter:
                a = pi_propensities - (
                    logged_propensities + self.clipping_parameter * pi_propensities
                )
                b = jnp.clip(a, -50, 50)

                return jnp.exp(b)
            else:
                a = pi_propensities - logged_propensities
                b = jnp.clip(a, -50, 50)
                return jnp.exp(b)

    def objective_function(self, estimate):
        if self.variance_penalty:

            return jnp.mean(estimate) + self.lambda_ * jnp.sqrt(
                jnp.sum(jnp.cov(estimate))
            )
        else:
            return jnp.mean(estimate)

    def optimize(
        self,
        actions,
        contexts,
        losses,
        propensities,
        max_iter=500,
        tol=None,
        seed=42,
    ):

        def _loss(theta):
            self.policy.parameter = theta
            return self.loss_function(
                self.policy, actions, contexts, losses, propensities
            )

        optimizer = jaxopt.ScipyMinimize(
            method="L-BFGS-B", fun=_loss, maxiter=max_iter, tol=tol
        )

        # Run the optimizer and capture the result
        result = optimizer.run(self.policy.parameter)
        parameter_opt = result.params

        self.list_of_parameters.append(parameter_opt)

        fun_val = result.state.fun_val

        return parameter_opt, fun_val
