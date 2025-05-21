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


