import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV


def uniform_logging_policy(X):
    return np.full(X.shape[0], 0.5)


def logistic_logging_policy(X, beta):
    logits = -2 * (X @ beta)
    return 1 / (1 + np.exp(-logits))


def reward_quadratic(X, A, beta):
    return (X @ beta) + 2.0 * (A**2) + 0.1 * np.random.randn(len(X))


def reward_nonlinear(X, A, beta):
    return np.sin(X @ beta) + A * np.cos(X @ beta) + 0.1 * np.random.randn(len(X))


def reward_linear(X, A, beta):
    return (X @ beta) + A + 0.1 * np.random.randn(len(X))


def gaussian_policy_mean(X, beta):
    return X @ beta  # mean of Gaussian


def gaussian_pdf(a, X, beta, scale=0.5):
    mu = gaussian_policy_mean(X, beta)
    return (1 / (np.sqrt(2 * np.pi) * scale)) * np.exp(-0.5 * ((a - mu) / scale) ** 2)


def find_best_params(
    X_log, A_log, Y_log, reg_grid=[1e1, 1e0, 0.1, 1e-2, 1e-3, 1e-4], num_cv=3
):
    kr = GridSearchCV(
        KernelRidge(kernel="rbf", gamma=0.1),
        cv=num_cv,
        param_grid={"alpha": reg_grid},
    )
    features = np.concatenate([X_log, A_log.reshape(-1, 1)], axis=1)
    kr.fit(features, Y_log)
    reg_param = kr.best_params_["alpha"]
    return reg_param


def pi0_proba(a, X):
    """
    Logging policy: P(a=1|x) = sigmoid(-2 * x · beta), P(a=0|x) = 1 - P(a=1|x)
    """
    logits = -2 * (X @ np.linspace(0.1, 0.5, X.shape[1]))
    probs = 1 / (1 + np.exp(-logits))  # P(a=1 | x)
    a = np.asarray(a)
    return probs * (a == 1) + (1 - probs) * (a == 0)


def pi_proba(a, X):
    """
    Target policy: logistic over X @ beta, returns pi(a|x)
    """
    logits = 4 * (X @ np.linspace(0.1, 0.5, X.shape[1]))
    probs = 1 / (1 + np.exp(-logits))  # P(a=1|x)
    a = np.asarray(a)
    return probs * (a == 1) + (1 - probs) * (a == 0)


def importance_weights(A, X, pi, pi0, eps=1e-8):
    """
    Compute importance weights w = pi(a|x) / pi0(a|x)

    Parameters
    ----------
    A : np.ndarray of shape (n,)
        Actions taken (can be discrete or continuous)
    X : np.ndarray of shape (n, d)
        Contexts
    pi : callable
        Target policy. Should return pi(a|x) — either probability mass or density.
    pi0 : callable
        Logging policy. Same interface as pi.
    eps : float
        Clipping constant to avoid division by zero or instability.

    Returns
    -------
    w : np.ndarray of shape (n,)
        Importance sampling weights.
    """
    numer = pi(A, X)
    denom = pi0(A, X)

    # clip to avoid division by zero or exploding weights
    numer = np.clip(numer, eps, None)
    denom = np.clip(denom, eps, None)

    return numer / denom
