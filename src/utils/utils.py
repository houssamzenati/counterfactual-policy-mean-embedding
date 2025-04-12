import numpy as np
import os
import sys

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier


def display_experiment(random_seed, dataset, name):
    print(
        "***",
        "EXPERIMENT",
        name,
        "Random seed: %i" % random_seed,
        "Dataset: %s" % dataset.name,
        "***",
        file=sys.stderr,
    )


def display_metrics(offline_loss, online_loss, regret):
    print(
        "***",
        "Offline loss: ",
        offline_loss,
        "Online loss: %f" % online_loss,
        "Regret: %f" % regret,
        "***",
        file=sys.stderr,
    )


def get_logging_data(n_samples, dataset):

    actions, contexts, losses, propensities, potentials = dataset.sample_logged_data(
        n_samples
    )
    logging_data = actions, contexts, losses, propensities

    return logging_data


def dataset_split(contexts, actions, losses, propensities, random_seed, ratio=0.25):
    rng = np.random.RandomState(random_seed)
    idx = rng.permutation(contexts.shape[0])
    contexts, actions, losses, propensities = (
        contexts[idx],
        actions[idx],
        losses[idx],
        propensities[idx],
    )

    size = int(contexts.shape[0] * ratio)
    contexts_train, contexts_valid = contexts[:size, :], contexts[size:, :]
    actions_train, actions_valid = actions[:size], actions[size:]
    losses_train, losses_valid = losses[:size], losses[size:]
    propensities_train, propensities_valid = propensities[:size], propensities[size:]
    #     potentials_train, potentials_valid = potentials[:size], potentials[size:]

    logged_train = actions_train, contexts_train, losses_train, propensities_train
    logged_valid = actions_valid, contexts_valid, losses_valid, propensities_valid

    return logged_train, logged_valid


def online_evaluation(optimized_param, contextual_modelling, dataset, random_seed):

    rng = np.random.RandomState(random_seed)
    contexts, potentials = dataset.test_data
    contextual_param = contextual_modelling.get_parameter(optimized_param, contexts)
    contextual_param = contextual_param.reshape(
        -1,
    )
    size = contexts.shape[0]
    losses = []

    for i in range(10):
        sampled_actions = rng.normal(contextual_param, dataset.logging_scale, size)
        losses += [dataset.get_losses_from_actions(potentials, sampled_actions)]

    losses_array = np.stack(losses, axis=0)
    return np.mean(losses_array)
