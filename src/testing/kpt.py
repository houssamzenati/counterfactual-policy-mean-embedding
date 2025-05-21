from __future__ import division
import numpy as np
from sys import stdout
from sklearn.metrics import pairwise_kernels
import scipy.stats as st
from sklearn.metrics import pairwise_distances


def MMD2u_reweight(Y, w_pi, w_pi_prime, kernel_function="rbf", **kwargs):
    """
    Compute the MMD^2_u statistic on a single dataset Y,
    with two different sets of importance weights.
    """
    w_pi = w_pi.reshape(-1, 1)
    w_pi_prime = w_pi_prime.reshape(-1, 1)
    K = pairwise_kernels(Y, metric=kernel_function, **kwargs)

    K_pi = np.outer(w_pi, w_pi) * K
    K_pi_prime = np.outer(w_pi_prime, w_pi_prime) * K
    K_cross = np.outer(w_pi, w_pi_prime) * K

    mmd2u = (
        (K_pi.sum() - np.trace(K_pi)) / (len(Y) * (len(Y) - 1))
        + (K_pi_prime.sum() - np.trace(K_pi_prime)) / (len(Y) * (len(Y) - 1))
        - 2 * K_cross.sum() / (len(Y) ** 2)
    )

    return mmd2u, K


def compute_null_distribution_reweight_single_K(
    K, w_pi, w_pi_prime, iterations=1000, random_state=None
):
    rng = np.random.RandomState(random_state)
    n = len(w_pi)
    mmd2_null = np.zeros(iterations)
    for i in range(iterations):
        idx = rng.permutation(n)
        w_pi_perm = w_pi[idx]
        w_pi_prime_perm = w_pi_prime[idx]
        mmd2_null[i] = MMD2u_reweight_given_K(K, w_pi_perm, w_pi_prime_perm)
    return mmd2_null


def MMD2u_reweight_given_K(K, w_pi, w_pi_prime):
    w_pi = w_pi.reshape(-1, 1)
    w_pi_prime = w_pi_prime.reshape(-1, 1)

    K_pi = np.outer(w_pi, w_pi) * K
    K_pi_prime = np.outer(w_pi_prime, w_pi_prime) * K
    K_cross = np.outer(w_pi, w_pi_prime) * K

    n = len(w_pi)
    return (
        (K_pi.sum() - np.trace(K_pi)) / (n * (n - 1))
        + (K_pi_prime.sum() - np.trace(K_pi_prime)) / (n * (n - 1))
        - 2 * K_cross.sum() / (n**2)
    )


def kernel_two_sample_test_reweight(
    Y,
    w_pi,
    w_pi_prime,
    kernel_function="rbf",
    iterations=1000,
    random_state=None,
    **kwargs
):
    mmd2, K = MMD2u_reweight(
        Y, w_pi, w_pi_prime, kernel_function=kernel_function, **kwargs
    )
    mmd2_null = compute_null_distribution_reweight_single_K(
        K, w_pi, w_pi_prime, iterations=iterations, random_state=random_state
    )
    pval = np.mean(mmd2_null > mmd2)
    return mmd2, mmd2_null, pval


def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic."""
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return (
        1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum())
        + 1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum())
        - 2.0 / (m * n) * Kxy.sum()
    )


def compute_null_distribution(
    K, m, n, iterations=10000, verbose=False, random_state=None, marker_interval=1000
):
    """Compute the bootstrap null-distribution of MMD2u."""
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        if verbose and (i % marker_interval) == 0:
            print(i),
            stdout.flush()
        idx = rng.permutation(m + n)
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    if verbose:
        print("")

    return mmd2u_null


def kernel_two_sample_test(
    X,
    Y,
    kernel_function="rbf",
    iterations=10000,
    verbose=False,
    random_state=None,
    **kwargs
):
    """Compute MMD^2_u, its null distribution and the p-value of the
    kernel two-sample test.
    Note that extra parameters captured by **kwargs will be passed to
    pairwise_kernels() as kernel parameters. E.g. if
    kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
    then this will result in getting the kernel through
    kernel_function(metric='rbf', gamma=0.1).
    """
    m = len(X)
    n = len(Y)
    XY = np.vstack([X, Y])
    K = pairwise_kernels(XY, metric=kernel_function, **kwargs)
    mmd2u = MMD2u(K, m, n)
    if verbose:
        print("MMD^2_u = %s" % mmd2u)
        print("Computing the null distribution.")

    mmd2u_null = compute_null_distribution(
        K, m, n, iterations, verbose=verbose, random_state=random_state
    )
    p_value = max(1.0 / iterations, (mmd2u_null > mmd2u).sum() / float(iterations))
    if verbose:
        print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0 / iterations))

    return mmd2u, mmd2u_null, p_value
