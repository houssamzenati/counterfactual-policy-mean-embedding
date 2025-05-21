import os
import numpy as np

scenario_list_alt = ["II", "III", "IV"]
ns_list_alt = np.arange(100, 400, 50)
methods_alt = ["PE-linear", "KPE", "DR-CF", "DR-CF"]
data_folder = "data/"

for scenario in scenario_list_alt:
    for method in methods_alt:
        for ns in ns_list_alt:
            old_name = f"{data_folder}ns{ns}_scenario{scenario}_{method}.csv"
            new_name = f"{data_folder}ns{ns}_treatments_continuous_scenario{scenario}_{method}.csv"

            if os.path.exists(old_name):
                os.rename(old_name, new_name)
                print(f"Renamed: {old_name} -> {new_name}")
            else:
                print(f"File not found: {old_name}")
