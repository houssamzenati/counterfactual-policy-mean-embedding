# Real Data Experiment (Figure 5) - Minimal Extension Based on Provided Code

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics.pairwise import rbf_kernel
from Environment import AvgEnvironment
from Estimator import (
    IPSEstimator,
    SlateEstimator,
    DirectEstimator,
    DoublyRobustEstimator,
    CMEstimator,
)
from Policy import MultinomialPolicy
from ParameterSelector import ParameterSelector

# FAST Mode Settings
bootstrap_iterations = 1000
num_experiments = 10
num_cv = 3

np.random.seed(2)


# Step 1: Download MSLR-WEB30K if missing
def download_mslr(destination_folder="mslr_data"):
    os.makedirs(destination_folder, exist_ok=True)
    data_url = "https://archive.org/download/mslr-web30k/MSLR-WEB30K.zip"
    zip_path = os.path.join(destination_folder, "MSLR-WEB30K.zip")

    if not os.path.exists(zip_path):
        print("Downloading MSLR-WEB30K dataset...")
        urllib.request.urlretrieve(data_url, zip_path)

    if not os.path.exists(os.path.join(destination_folder, "MSLR-WEB30K")):
        print("Extracting MSLR-WEB30K dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(destination_folder)

    print("Dataset ready.")


# Step 2: Simple Parser for MSLR Data
def parse_mslr_file(filepath, num_features=136):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            features = np.zeros(num_features)
            for item in parts[2:]:
                idx, val = item.split(":")
                features[int(idx) - 1] = float(val)
            data.append((label, features))
    labels, features = zip(*data)
    return np.array(features), np.array(labels)


# Step 3: ERR Calculation
def compute_ERR(labels, max_rank=10):
    max_relevance = 4
    R = [(2**rel - 1) / (2**max_relevance) for rel in labels]
    ERR = 0.0
    p = 1.0
    for k in range(min(max_rank, len(R))):
        ERR += p * R[k] / (k + 1)
        p *= 1 - R[k]
    return ERR


# Step 4: Main Real Data Experiment

# Download and prepare dataset
download_mslr()

# Assume using Fold1 training and validation sets
train_features, train_labels = parse_mslr_file("mslr_data/MSLR-WEB30K/Fold1/train.txt")
vali_features, vali_labels = parse_mslr_file("mslr_data/MSLR-WEB30K/Fold1/vali.txt")

# Step 5: Fit Logging and Target Policies

# Logging policy: Decision Tree
tree_model = DecisionTreeRegressor(max_depth=5)
tree_model.fit(train_features, train_labels)

# Target policy: Lasso Regression
lasso_model = Lasso(alpha=0.001)
lasso_model.fit(train_features, train_labels)

# Step 6: Create Logged Data from Logging Policy

n_samples = 5000  # Adjust depending on speed
logged_data = []
for _ in range(n_samples):
    idx = np.random.randint(0, vali_features.shape[0])
    features = vali_features[idx]
    pred_score = tree_model.predict([features])[0]
    logged_data.append((features, pred_score, vali_labels[idx]))

# Step 7: Estimation

# Prepare sim_data format
sim_data = []
for features, pred_score, true_label in logged_data:
    observation = {
        "null_context_vec": features,
        "target_context_vec": features,
        "null_reco": tuple([0]),  # Dummy recommendation
        "null_reco_vec": features,
        "null_reward": compute_ERR([true_label]),
        "target_reco": tuple([0]),
        "target_multinomial": 1.0,
        "target_reco_vec": features,
        "target_reward": compute_ERR([true_label]),
        "null_multinomial": 1.0,
        "user": features,
    }
    sim_data.append(observation)

sim_data = pd.DataFrame(sim_data)

# Step 8: Set Estimators

reg_pow = -1
reg_params = (10.0**reg_pow) / n_samples
bw_params = 10.0**0
params = [reg_params, bw_params, bw_params]

estimators = [
    IPSEstimator(1, None, None),
    SlateEstimator(1, None),
    DirectEstimator(),
    DoublyRobustEstimator(1, None, None),
    CMEstimator(rbf_kernel, rbf_kernel, params),
]

# Step 9: Estimate and Plot Results

actual_value = np.mean(sim_data["target_reward"])
results = []

for estimator in estimators:
    est_value = estimator.estimate(sim_data)
    mse = (est_value - actual_value) ** 2
    results.append({"Estimator": estimator.name, "MSE": mse})

results_df = pd.DataFrame(results)

sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x="Estimator", y="MSE")
plt.yscale("log")
plt.title("Real Data Experiment on MSLR-WEB30K (FAST Mode)")
plt.tight_layout()
plt.show()
