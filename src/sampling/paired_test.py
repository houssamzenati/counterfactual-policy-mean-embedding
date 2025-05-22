from scipy.stats import ttest_rel
import pandas as pd
import json
from pathlib import Path

# Load all .jsonl files
folder = Path("results")  # Adjust if needed
jsonl_files = sorted(folder.glob("results_*.jsonl"))

# Parse each line into records
all_data = []
for file in jsonl_files:
    with open(file) as f:
        for line in f:
            entry = json.loads(line)
            entry["scenario"] = f"{entry['logging_type']}_{entry['reward_type']}"
            all_data.append(entry)

# Create DataFrame
df = pd.DataFrame(all_data)

# Paired t-tests for each scenario
scenarios = df["scenario"].unique()
results = []

for scenario in scenarios:
    sub = df[df["scenario"] == scenario]
    stat_wass, pval_wass = ttest_rel(sub["wass_dr"], sub["wass_plugin"])
    stat_mmd, pval_mmd = ttest_rel(sub["mmd_unbiased_dr"], sub["mmd_unbiased_plugin"])
    results.append(
        {
            "Scenario": scenario,
            "t-stat (Wass)": stat_wass,
            "p-val (Wass)": pval_wass,
            "t-stat (MMD)": stat_mmd,
            "p-val (MMD)": pval_mmd,
        }
    )

# Convert to DataFrame and show results
results_df = pd.DataFrame(results)
print(results_df)

# Optionally save
results_df.to_csv("paired_ttest_sampling_results.csv", index=False)
