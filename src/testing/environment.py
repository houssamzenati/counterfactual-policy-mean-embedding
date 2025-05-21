from scipy.stats import laplace, bernoulli
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np


class GaussianPolicy:
    def __init__(self, w, scale=1.0):
        self.w = w
        self.scale = scale

    def sample_treatments(self, X):
        return np.random.normal(self.get_mean(X), self.scale)

    def get_mean(self, X):
        return X @ self.w

    def get_propensities(self, X, t):
        mean = self.get_mean(X)
        return (1 / (self.scale * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((t - mean) / self.scale) ** 2
        )


class LaplacePolicy:
    def __init__(self, w, scale=0.5):
        self.w = w
        self.scale = scale

    def sample_treatments(self, X):
        return laplace.rvs(loc=self.get_mean(X), scale=self.scale)

    def get_mean(self, X):
        return X @ self.w

    def get_propensities(self, X, t):
        mean = self.get_mean(X)
        return (1 / (2 * self.scale)) * np.exp(-np.abs(t - mean) / self.scale)


class BernoulliPolicy:
    def __init__(self, w):
        self.w = w

    def sample_treatments(self, X):
        prob = self.get_probs(X)
        return 2 * bernoulli.rvs(prob) - 1  # outputs in {-1, +1}

    def get_probs(self, X):
        return 1 / (1 + np.exp(-(X @ self.w)))

    def get_mean(self, X):
        return self.get_probs(X)  # expected value of treatment is in [-1, 1]

    def get_propensities(self, X, t):
        prob = self.get_probs(X)
        t = ((t + 1) // 2).astype(int)  # map {-1, +1} to {0, 1}
        return np.where(t == 1, prob, 1 - prob)


class EstimatedLoggingPolicy:
    def __init__(self, X, T):
        if np.issubdtype(T.dtype, np.integer) and np.unique(T).size == 2:
            self.model_type = "binary"
            self.model = LogisticRegression()
            self.model.fit(X, ((T + 1) // 2))  # map {-1, +1} to {0, 1}
        else:
            self.model_type = "continuous"
            self.model = LinearRegression()
            self.model.fit(X, T)

    def get_propensities(self, X, t):
        if self.model_type == "binary":
            prob = self.model.predict_proba(X)[:, 1]
            t = ((t + 1) // 2).astype(int)
            return np.where(t == 1, prob, 1 - prob)
        else:
            mean = self.model.predict(X)
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (t - mean) ** 2)

    def get_mean(self, X):
        return self.model.predict(X)


def outcome_model(X, T, beta, gamma, noise_std=0.1):
    noise = noise_std * np.random.randn(len(T))
    return X @ beta + gamma * T + noise


def generate_ope_data(
    X, policy_logging, policy_pi, policy_pi_prime, beta, gamma, noise_std=0.1
):
    """
    Simulate logged bandit data and prepare inputs for counterfactual evaluation.

    Parameters:
        X                : Covariates (n, d)
        policy_logging   : Policy that generated the logged data (π₀)
        policy_pi        : Target policy π
        policy_pi_prime  : Alternative policy π′
        beta, gamma      : Outcome model parameters
        noise_std        : Std of additive Gaussian noise

    Returns:
        dict with {X, T, Y, w_pi, w_pi_prime, pi_samples, pi_prime_samples}
    """
    T = policy_logging.sample_treatments(X)
    Y = outcome_model(X, T, beta, gamma, noise_std)
    estimate_logging_propensities = EstimatedLoggingPolicy(X, T).get_propensities(X, T)
    w_pi = (
        policy_pi.get_propensities(X, T)[:, np.newaxis]
        / estimate_logging_propensities[:, np.newaxis]
    )
    w_pi_prime = (
        policy_pi_prime.get_propensities(X, T)[:, np.newaxis]
        / estimate_logging_propensities[:, np.newaxis]
    )

    pi_samples = policy_pi.sample_treatments(X)
    pi_prime_samples = policy_pi_prime.sample_treatments(X)

    return dict(
        X=X,
        T=T,
        Y=Y,
        w_pi=w_pi,
        w_pi_prime=w_pi_prime,
        pi_samples=pi_samples,
        pi_prime_samples=pi_prime_samples,
    )


def make_scenario_continuous(scenario_id, d=5, seed=None):
    """
    Returns:
        - X: covariates
        - policy_logging: logging policy π₀
        - policy_pi: target policy π
        - policy_pi_prime: alternative policy π′
        - beta, gamma: for outcome model
    """
    if seed is not None:
        np.random.seed(seed)

    n = 1000  # number of samples
    X = np.random.normal(0, 1, size=(n, d))

    beta = np.linspace(0.1, 0.5, d)  # outcome main effect
    gamma = 1.0  # treatment effect coefficient

    w_base = np.ones(d) / np.sqrt(d)

    # Logging policy: fixed
    policy_logging = GaussianPolicy(w_base, scale=1)

    if scenario_id == "I":
        # Null scenario: π = π′ (no difference)
        policy_pi = GaussianPolicy(w_base, scale=1)
        policy_pi_prime = GaussianPolicy(w_base, scale=1)

    elif scenario_id == "II":
        # Mean shift: π′ mean shifted along one direction
        shift = 2 * np.ones(d)
        policy_pi = GaussianPolicy(w_base, scale=1.0)
        policy_pi_prime = GaussianPolicy(w_base + shift, scale=1.0)

    elif scenario_id == "III":
        # Mixture policy π′: bimodal
        w1 = w_base + np.ones(d)
        w2 = w_base - np.ones(d)

        class MixturePolicy:
            def __init__(self, w1, w2):
                self.p1 = GaussianPolicy(w1)
                self.p2 = GaussianPolicy(w2)

            def sample_treatments(self, X):
                mask = np.random.binomial(1, 0.5, size=X.shape[0])
                T1 = self.p1.sample_treatments(X)
                T2 = self.p2.sample_treatments(X)
                return mask * T1 + (1 - mask) * T2

            def get_propensities(self, X, t):
                return 0.5 * self.p1.get_propensities(
                    X, t
                ) + 0.5 * self.p2.get_propensities(X, t)

        policy_pi = GaussianPolicy(w_base)
        policy_pi_prime = MixturePolicy(w1, w2)

    elif scenario_id == "IV":
        # Support shift: π′ outputs treatments in opposite direction for half the population
        # class FlippingPolicy:
        #     def __init__(self, base_policy):
        #         self.base = base_policy

        #     def sample_treatments(self, X):
        #         T = self.base.sample_treatments(X)
        #         mask = (X[:, 0] > 0).astype(float)
        #         return T * (1 - 2 * mask)  # flip sign

        #     def get_propensities(self, X, t):
        #         flipped_t = np.where(X[:, 0] > 0, -t, t)
        #         return self.base.get_propensities(X, flipped_t)

        # base_pi = GaussianPolicy(w_base)
        # policy_pi = base_pi
        # policy_pi_prime = FlippingPolicy(base_pi)
        w1 = w_base + 2 * np.ones(d)
        w2 = w_base

        class MixturePolicy:
            def __init__(self, w1, w2):
                self.p1 = GaussianPolicy(w1)
                self.p2 = GaussianPolicy(w2)

            def sample_treatments(self, X):
                mask = np.random.binomial(1, 0.5, size=X.shape[0])
                T1 = self.p1.sample_treatments(X)
                T2 = self.p2.sample_treatments(X)
                return mask * T1 + (1 - mask) * T2

            def get_propensities(self, X, t):
                return 0.5 * self.p1.get_propensities(
                    X, t
                ) + 0.5 * self.p2.get_propensities(X, t)

        policy_pi = GaussianPolicy(w_base)
        policy_pi_prime = MixturePolicy(w1, w2)

    else:
        raise ValueError(f"Unknown scenario {scenario_id}")

    return X, policy_logging, policy_pi, policy_pi_prime, beta, gamma


def make_binary_scenario(scenario_id, d=5, seed=None):
    if seed is not None:
        np.random.seed(seed)

    n = 1000
    X = np.random.normal(0, 1, size=(n, d))

    beta = np.linspace(0.1, 0.5, d)
    gamma = 1.0

    w_base = np.ones(d) / np.sqrt(d)

    if scenario_id == "I":
        policy_logging = BernoulliPolicy(w_base)
        policy_pi = BernoulliPolicy(w_base)
        policy_pi_prime = BernoulliPolicy(w_base)

    elif scenario_id == "II":
        shift = np.ones(d)
        policy_logging = BernoulliPolicy(w_base)
        policy_pi = BernoulliPolicy(w_base - shift)
        policy_pi_prime = BernoulliPolicy(w_base + shift)

    elif scenario_id == "III":
        w1 = w_base + np.ones(d) / 2
        w2 = w_base - np.ones(d) / 2

        class MixtureBinaryPolicy:
            def __init__(self, w1, w2):
                self.p1 = BernoulliPolicy(w1)
                self.p2 = BernoulliPolicy(w2)

            def sample_treatments(self, X):
                mask = np.random.binomial(1, 0.5, size=X.shape[0])
                T1 = self.p1.sample_treatments(X)
                T2 = self.p2.sample_treatments(X)
                return mask * T1 + (1 - mask) * T2

            def get_propensities(self, X, t):
                return 0.5 * self.p1.get_propensities(
                    X, t
                ) + 0.5 * self.p2.get_propensities(X, t)

        policy_logging = BernoulliPolicy(w_base)
        policy_pi = BernoulliPolicy(w_base)
        policy_pi_prime = MixtureBinaryPolicy(w1, w2)

    elif scenario_id == "IV":

        class FlippingBinaryPolicy:
            def __init__(self, base_policy):
                self.base = base_policy

            def sample_treatments(self, X):
                T = self.base.sample_treatments(X)
                mask = (X[:, 0] > 0).astype(float)
                return T * (1 - 2 * mask)

            def get_propensities(self, X, t):
                flipped_t = np.where(X[:, 0] > 0, -t, t)
                return self.base.get_propensities(X, flipped_t)

        base_pi = BernoulliPolicy(w_base)
        policy_logging = base_pi
        policy_pi = base_pi
        policy_pi_prime = FlippingBinaryPolicy(base_pi)

    else:
        raise ValueError(f"Unknown scenario {scenario_id}")

    return X, policy_logging, policy_pi, policy_pi_prime, beta, gamma
