import os
import sys
from abc import ABCMeta, abstractmethod

import numpy as np

import jax
import jax.numpy as jnp


class Policy(metaclass=ABCMeta):
    @abstractmethod
    def create_start_parameter(self, env):
        pass

    @abstractmethod
    def context_modelling(self, features):
        pass

    @abstractmethod
    def evaluate(self, actions, features, targets):
        pass
