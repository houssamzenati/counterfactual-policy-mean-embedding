# Script to generate Figure 3(a) (Observed vs True vs Generated samples)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import pairwise_distances

# Set random seed for reproducibility
np.random.seed(2)

# Parameters
ns = 500  # number of samples (can adjust later)
d = 5  # dimension


# Mixture of Gaussians for X1
def get_mixture_gaussian_samples(n, d, locs, weights):
    p = weights / weights.sum()
    samples = []
    for _ in range(n):
        Z = np.random.choice(np.arange(len(p)), p=p)
        samples.append(
            np.random.multivariate_normal(mean=locs[Z], cov=np.eye(d), size=1)
        )
    return np.array(samples).squeeze()


# Generate covariates
X0 = np.random.randn(ns, d)
locs = np.array([[-5, 2.5, 0, 0, 2.5], [2.5, 2.5, 0, 0, -5], [2.5, -5, 0, 0, 2.5]])
weights = np.array([1, 1, 1])
X1 = get_mixture_gaussian_samples(ns, d, locs, weights)

# Generate outcomes
beta_vec = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
Y0 = np.dot(beta_vec, X0.T) + 0.1 * np.random.randn(X0.shape[0])
Y1 = np.dot(beta_vec, X1.T) + 0.1 * np.random.randn(X1.shape[0])

YY0 = Y0[:, np.newaxis]
YY1 = Y1[:, np.newaxis]


# Kernel functions
def gauss_rbf(X1, X2, sigma=1):
    K = np.exp(-np.divide(cdist(X1, X2, "sqeuclidean"), 2 * sigma))
    return K


def find_best_params(
    X1, Y1, reg_grid=[1e1, 1e0, 0.1, 1e-2], gamma_grid=[1e1, 1e0, 0.1, 1e-2], num_cv=3
):
    kr = GridSearchCV(
        KernelRidge(kernel="rbf", gamma=0.1),
        cv=num_cv,
        param_grid={"alpha": reg_grid, "gamma": gamma_grid},
    )
    kr.fit(X1, Y1)
    sg = 1.0 / kr.best_params_["gamma"]
    reg_param = kr.best_params_["alpha"]
    return sg, reg_param


def estimate_cme(X0, X1, Y1):
    sg, reg_param = find_best_params(X1, Y1)
    K1 = gauss_rbf(X0, X0, sg)
    K2 = gauss_rbf(X0, X1, sg)
    b = np.dot(
        np.dot(np.linalg.inv(K1 + reg_param * np.eye(X0.shape[0])), K2),
        np.ones((X1.shape[0], 1)) / X1.shape[0],
    )
    b = b / b.sum()
    return b


def generate_herding_samples(num_herding, Y0, sigma, weights):
    y0 = np.random.randn()
    res = minimize(
        lambda y: -np.dot(weights.T, np.exp(-((Y0 - y) ** 2) / (2 * sigma))),
        y0,
        method="CG",
        options={"gtol": 1e-6, "disp": False},
    )
    yt = res.x.ravel()[0]
    yt_samples = [yt]
    for t in range(2, num_herding + 1):
        yt_hist = np.array(yt_samples)
        res = minimize(
            lambda y: -np.dot(weights.T, np.exp(-((Y0 - y) ** 2) / (2 * sigma)))
            + np.exp(-((yt_hist - y) ** 2) / (2 * sigma)).mean(),
            y0,
            method="CG",
            options={"gtol": 1e-6, "disp": False},
        )
        yt = res.x.ravel()[0]
        yt_samples.append(yt)
    return np.array(yt_samples)


# Estimate CME and generate counterfactual samples
sigma = np.median(pairwise_distances(YY0, YY1, metric="sqeuclidean"))
weights = estimate_cme(X0, X1, Y1)

# NOTE: Here, herding is done for 'ns' samples (200 samples). To improve quality,
# in the final version, you can set 'num_herding = 1000' below.
generated_cf_samples = generate_herding_samples(
    num_herding=ns, Y0=YY0, sigma=sigma, weights=weights
)
generated_cf_samples = generated_cf_samples[:, np.newaxis]

# ============ Plot ============
fig, ax = plt.subplots(figsize=(7, 5))

# Histograms
ax.hist(
    YY0,
    bins=30,
    density=True,
    alpha=0.3,
    color="red",
    label=r"$Y_0^* \mid T =0$ ",
)
ax.hist(
    YY1,
    bins=30,
    density=True,
    alpha=0.3,
    color="blue",
    label=r"$Y_0^* \mid T =1$",
)
ax.hist(
    generated_cf_samples,
    bins=30,
    density=True,
    alpha=0.3,
    color="green",
    label=r"Y herding",
)

# KDE curves
sns.kdeplot(YY0.ravel(), ax=ax, color="red", lw=2)
sns.kdeplot(YY1.ravel(), ax=ax, color="blue", lw=2)
sns.kdeplot(generated_cf_samples.ravel(), ax=ax, color="green", lw=2)

ax.set_xlim([-4, 4])
ax.set_xlabel("Y")
ax.set_ylabel("Density")
ax.set_title("Comparison of Observed vs True vs Generated (Figure 3(a))")
ax.legend()
fig.tight_layout()

# Save figure
fig.savefig("figure3a_with_kde.png", dpi=100)
print("Figure saved as 'figure3a_with_kde.png'")
