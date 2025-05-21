import numpy as np


class SyntheticOPEEnvironment:
    def __init__(self, d=5, seed=0):
        self.d = d
        np.random.seed(seed)
        self.beta = np.linspace(0.1, 0.5, d)

    # --- Logging policy (anti-aligned with target) ---
    def logging_proba(self, X):
        logits = -2 * (X @ self.beta)
        return 1 / (1 + np.exp(-logits))  # P(a=1|x)

    def logging_sample(self, X):
        return np.random.binomial(1, self.logging_proba(X))

    # --- Target policy ---
    def target_proba(self, X):
        logits = X @ self.beta
        return 1 / (1 + np.exp(-logits))  # P(a=1|x)

    def target_sample(self, X):
        return np.random.binomial(1, self.target_proba(X))

    # --- Outcome model ---
    def true_outcome(self, X, A):
        return (X @ self.beta) + 2.0 * (A**2) + 0.1 * np.random.randn(len(X))

    # --- Unified sampling interface ---
    def sample(self, n, policy="logging"):
        X = np.random.randn(n, self.d)
        if policy == "logging":
            A = self.logging_sample(X)
        elif policy == "target":
            A = self.target_sample(X)
        else:
            raise ValueError("Unknown policy: must be 'logging' or 'target'")
        Y = self.true_outcome(X, A)
        return X, A, Y

    # --- Importance weight computation ---
    def importance_weights(self, A, X, eps=1e-8):
        """
        Computes importance weights: w = pi(a|x) / pi0(a|x)
        """
        p_pi = self.eval_policy("target", A, X)
        p_pi0 = self.eval_policy("logging", A, X)

        p_pi = np.clip(p_pi, eps, None)
        p_pi0 = np.clip(p_pi0, eps, None)

        return p_pi / p_pi0

    # --- Evaluate pi(a|x) for a given policy ---
    def eval_policy(self, policy, A, X):
        A = np.asarray(A)
        if policy == "logging":
            p = self.logging_proba(X)
        elif policy == "target":
            p = self.target_proba(X)
        else:
            raise ValueError("Unknown policy: must be 'logging' or 'target'")

        return p * (A == 1) + (1 - p) * (A == 0)
