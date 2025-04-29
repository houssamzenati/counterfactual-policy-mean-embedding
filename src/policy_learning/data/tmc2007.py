import numpy as np
import numpy as np
import json
import copy
import os
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.preprocessing import add_dummy_feature

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier

import jax.numpy as jnp
from data.base import Dataset


class tmc2007(Dataset):
    """Parent class for Data"""

    def __init__(self, name, mode="quadratic", test_size=0.25, **kw):
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
        super(tmc2007, self).__init__(**kw)
        self.name = name
        self.policy_type = 'discrete'
        self.test_size = test_size
        self.l = 3
        self.start = 0
        self.mode = mode

        self.logging_scale = 0.5
        self.parameter_scale = 0.01
        self.X_train, self.y_train, self.X_test, self.y_test, _ = load_dataset(
            self.name, self.test_size, reduce_dim=300
        )
        self.X_all, self.y_all = np.vstack([self.X_train, self.X_test]), np.vstack(
            [self.y_train, self.y_test]
        )
        self.dimension = (self.X_train.shape[1], self.y_train.shape[1])
        self.param_dimension = self.dimension
        self.n_samples = self.X_train.shape[0]
        self.start_sigma = None

    def sample_contexts_targets(self, n_samples):
        b = min(self.start + n_samples, self.n_samples)

        X = self.X_train[self.start : b]
        y = self.y_train[self.start : b]

        return X, y

    def sample_losses(self, actions, contexts, targets):
        """
        Computes losses based on actions and target values.

        Parameters
        ----------
        actions : np.ndarray
            Actions taken by the policy.
        contexts : np.ndarray
            Contextual features corresponding to the actions.
        targets : np.ndarray
            Target values to compare against actions.

        Returns
        -------
        np.ndarray
            Loss values for each action-target pair.
        """

        n = actions.shape[0]
        k = actions.shape[1]
        rewards = (1 - np.logical_xor(actions, targets)).sum(axis=1).reshape((-1,))

        return k - rewards

    def online_evaluation_star(self, pi_star):
        contexts, targets = self.X_test, self.y_test
        predictions = pi_star.predict_proba(contexts)
        predictions = np.array([_[:, 1] for _ in predictions]).T
        idx = np.where(targets == 0)
        fp = predictions[idx].sum()
        idx = np.where(targets == 1)
        fn = (1 - predictions[idx]).sum()
        return (fn + fp) / (targets.shape[0] * targets.shape[1])
    
    def get_optimal_loss(self, contextual_model):
        pistar = make_baselines_skylines(self.X_train, self.y_train)
        return self.online_evaluation_star(pistar)

    class LoggingPolicy:
        def __init__(self, env, random_seed):
            self.env = env
            self.pi0, _ = make_baselines_skylines(
                env.X_train,
                env.y_train,
                bonus=None,
                mlp=False,
                n_jobs=4,
                skip_skyline=False,
            )
            self.rng = np.random.RandomState(random_seed)
            self.propensity_type = "logarithmic"

        def get_propensity(self, actions, features, targets):
            """Returns the probability of selecting an action given the state

            Args:
                features (array): array of feature vector
                actions (array): array of actions
            """
            # Implement the logic to select an action based on the state

            sampling_probas = np.array(
                [_[:, 1] for _ in self.pi0.predict_proba(features)]
            ).T
            zero_chosen = np.where(actions == 0)
            propensities = np.array(sampling_probas)
            propensities[zero_chosen] = 1 - sampling_probas[zero_chosen]
            log_propensities = jnp.log(propensities).sum(axis=1)

            if self.propensity_type == "logarithmic":
                return log_propensities
            else:
                propensities = jnp.exp(log_propensities)
                return propensities

        def get_actions_and_propensities(self, features, targets):
            """Return actions and associated propensities given features and targets

            Args:
                features (_type_): _description_

                targets (_type_): _description_
            """
            # Implement the logic to update the policy based on the experience

            n = features.shape[0]
            k = targets.shape[1]
            probas = np.array([_[:, 1] for _ in self.pi0.predict_proba(features)]).T
            actions = (self.rng.uniform(size=(n, k)) < probas).astype(int)
            zero_chosen = np.where(actions == 0)
            propensities = np.array(probas)
            propensities[zero_chosen] = 1 - probas[zero_chosen]
            log_propensities = jnp.log(propensities).sum(axis=1)

            if self.propensity_type == "logarithmic":
                return actions, log_propensities
            else:
                propensities = jnp.exp(log_propensities)

                return actions, propensities

        def online_evaluation(self, env):
            """Evaluate the Logging Policy

            Args:
                env (Environment): Environment object with test data.

            Returns:
                _type_: the baseline evaluation
            """
            contexts, targets = env.X_test, env.y_test
            predictions = self.pi0.predict_proba(contexts)
            predictions = np.array([_[:, 1] for _ in predictions]).T
            idx = np.where(targets == 0)
            fp = predictions[idx].sum()
            idx = np.where(targets == 1)
            fn = (1 - predictions[idx]).sum()
            return (fn + fp) / (targets.shape[0] * targets.shape[1])
        

def load_dataset(
    dataset_name,
    test_size=0.25,
    seed=0,
    add_intercept=True,
    scale=False,
    reduce_dim: int = None,
):

    dataset_path = os.path.join("data", dataset_name, dataset_name + "_train.svm")
    X_train, y_train_ = load_svmlight_file(dataset_path, multilabel=True)

    dataset_path = os.path.join("data", dataset_name, dataset_name + "_test.svm")
    X_test, y_test_ = load_svmlight_file(dataset_path, multilabel=True)

    if reduce_dim is None and dataset_name in (
        "tmc2007",
        "rcv1_topics",
    ):
        reduce_dim = 300
    if reduce_dim:
        print("reducing dimension for %s dataset" % dataset_name)
        fh = GaussianRandomProjection(n_components=reduce_dim)
        X_train = fh.fit_transform(X_train)
        X_test = fh.transform(X_test)
    try:
        X_train = np.array(X_train.todense())
        X_test = np.array(X_test.todense())
    except AttributeError:
        pass

    onehot_labeller = MultiLabelBinarizer()
    y_train = onehot_labeller.fit_transform(y_train_).astype(int)
    y_test = onehot_labeller.transform(y_test_).astype(int)

    X_all = np.vstack([X_train, X_test])
    if add_intercept:
        X_all = add_dummy_feature(X_all)
    y_all = np.vstack([y_train, y_test])

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    labels = onehot_labeller.classes_.astype(int)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)

    return X_train, y_train, X_test, y_test, labels


def skyline_evaluation(pi_star_determinist, dataset):

    contexts, potentials = dataset.test_data
    predictions = pi_star_determinist.predict(contexts)

    losses = dataset.get_losses_from_actions(potentials, predictions)

    return np.mean(losses)

class OneClassRobustLogisticRegression(LogisticRegression):
    def fit(self, X, y, sample_weight=None):
        try:
            LogisticRegression.fit(self, X, y, sample_weight)
            return self
        except ValueError as exc:
            print("WARN: training set has only positive examples")
            self.coef_ = np.zeros(((1, X.shape[1])))
            return self

    def predict_proba(self, X):
        if len(self.classes_) == 1:
            return np.ones((X.shape[0], 2))
        return LogisticRegression.predict_proba(self, X)
    

def make_baselines_skylines(
    X_train,
    y_train,
    bonus: float = None,
    mlp=False,
    n_jobs=4,
    skip_skyline = False):
    """
        Creates baseline and skyline models for policy evaluation.

        Parameters
        ----------
        X_train : np.ndarray
            Training data for contexts.
        y_train : np.ndarray
            Training data for targets.
        bonus : float, optional
            Adjustment factor for baseline model coefficients (default is None).
        mlp : bool, optional
            Whether to use a multi-layer perceptron for modeling (default is 
            False).
        n_jobs : int, optional
            Number of parallel jobs for model training (default is 4).
        skip_skyline : bool, optional
            Whether to skip the skyline model creation (default is False).

        Returns
        -------
        tuple
            - Baseline model (`pi0`).
            - Skyline model (`pistar`), or None if `skip_skyline` is True.
        """
    
    if mlp:
        base_clf = MLPClassifier(
            hidden_layer_sizes=(
                500,
                100,
                40,
                10,
            )
        )
        pistar = MultiOutputClassifier(base_clf, n_jobs=n_jobs)
    else:
        base_clf = LogisticRegressionCV(max_iter=10000, n_jobs=6)
        pistar = MultiOutputClassifier(base_clf, n_jobs=n_jobs)
    if skip_skyline:
        pistar = None
    else:
        try:
            pistar.fit(X_train, y_train)
        except ValueError as exc:
            base_clf = OneClassRobustLogisticRegression()
            pistar = MultiOutputClassifier(base_clf, n_jobs=n_jobs)
            pistar.fit(X_train, y_train)

    return pistar
