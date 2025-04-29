import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import bernoulli
from sklearn.metrics import pairwise_distances

# Parameters
ns = 500
d = 5
noise_var = 0.1
beta_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
alpha_vec = np.array([0.05, 0.04, 0.03, 0.02, 0.01])
alpha_0 = 0.05

# Create figure
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Scenario I
b = 0
X = np.random.randn(ns, d)
Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
T = bernoulli.rvs(Prob_vec)
Y0 = np.dot(beta_vec, X[T==0, :].T) + noise_var*np.random.randn(X[T==0, :].shape[0])
Y1 = np.dot(beta_vec, X[T==1, :].T) + b + noise_var*np.random.randn(X[T==1, :].shape[0])

axs[0].hist(Y0, 30, facecolor='green', weights=Prob_vec[T==0], density=True, alpha=0.6, label="$Y_0$")
axs[0].hist(Y1, 30, facecolor='blue', weights=Prob_vec[T==1], density=True, alpha=0.6, label="$Y_1$")
axs[0].set_title('Scenario I', fontsize=16)
axs[0].set_ylabel('P(Y)', fontsize=14)
axs[0].set_xlabel('Y', fontsize=14)
axs[0].legend()

# Scenario II
b = 2
X = np.random.randn(ns, d)
Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
T = bernoulli.rvs(Prob_vec)
Y0 = np.dot(beta_vec, X[T==0, :].T) + noise_var*np.random.randn(X[T==0, :].shape[0])
Y1 = np.dot(beta_vec, X[T==1, :].T) + b + noise_var*np.random.randn(X[T==1, :].shape[0])

axs[1].hist(Y0, 30, facecolor='green', weights=Prob_vec[T==0], density=True, alpha=0.6, label="$Y_0$")
axs[1].hist(Y1, 30, facecolor='blue', weights=Prob_vec[T==1], density=True, alpha=0.6, label="$Y_1$")
axs[1].set_title('Scenario II', fontsize=16)
axs[1].set_ylabel('P(Y)', fontsize=14)
axs[1].set_xlabel('Y', fontsize=14)
axs[1].legend()

# Scenario III
X = np.random.randn(ns, d)
Prob_vec = expit(np.dot(alpha_vec, X.T) + alpha_0)
T = bernoulli.rvs(Prob_vec)
Z = bernoulli.rvs(0.5, size=len(T[T==1]))

Y0 = np.dot(beta_vec, X[T==0, :].T) + noise_var*np.random.randn(X[T==0, :].shape[0])
Y1 = np.dot(beta_vec, X[T==1, :].T) + (2*Z - 1) + noise_var*np.random.randn(X[T==1, :].shape[0])

axs[2].hist(Y0, 30, facecolor='green', weights=Prob_vec[T==0], density=True, alpha=0.6, label="$Y_0$")
axs[2].hist(Y1, 30, facecolor='blue', weights=Prob_vec[T==1], density=True, alpha=0.6, label="$Y_1$")
axs[2].set_title('Scenario III', fontsize=16)
axs[2].set_ylabel('P(Y)', fontsize=14)
axs[2].set_xlabel('Y', fontsize=14)
axs[2].legend()

# Final adjustments
fig.tight_layout()
plt.show()
