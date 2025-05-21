import numpy as np
import pandas as pd
import os

# === Scenario I (null)
scenario_list_null = ["I"]
ns_list_null = np.arange(100, 1050, 50)
methods_null = ["PE-linear", "KPE", "DR-CF"]

# === Scenarios II–IV (non-null)
scenario_list_alt = ["II", "III", "IV"]
ns_list_alt = np.arange(100, 450, 50)
methods_alt = ["PE-linear", "KPE", "DR-CF"]

# === Common folder
results_folder = "results/"
method_map = {"DR-CF": "DR-KPT", "KPE": "KPT", "PE-linear": "PT-linear"}

# === Load all files into dictionary
d_results = {}

# Null case (Scenario I)
for scenario in scenario_list_null:
    for method in methods_null:
        for ns in ns_list_null:
            fname = f"{results_folder}ns{ns}_scenario{scenario}_{method}.csv"
            if os.path.exists(fname):
                d_results[fname] = pd.read_csv(fname)
            else:
                print(f"Missing file: {fname}")

# Alternative cases (Scenarios II–IV)
for scenario in scenario_list_alt:
    for method in methods_alt:
        for ns in ns_list_alt:
            fname = f"{results_folder}ns{ns}_scenario{scenario}_{method}.csv"
            if os.path.exists(fname):
                d_results[fname] = pd.read_csv(fname)
            else:
                print(f"Missing file: {fname}")


# === Build LaTeX runtime tables per scenario
def build_scenario_table(scenario, sample_size_list):
    table_rows = []
    for idx, (method_key, method_latex) in enumerate(method_map.items()):
        row = [method_latex]
        for ns in sample_size_list:
            fname = f"{results_folder}ns{ns}_scenario{scenario}_{method_key}.csv"
            if fname in d_results:
                df = d_results[fname]
                mean_time = df["time"].mean()
                std_time = df["time"].std()
                formatted = f"{mean_time:.3f} $\\pm$ {std_time:.3f}"
            else:
                formatted = "---"
            row.append(formatted)
        table_rows.append(row)
    columns = ["Method"] + [str(ns) for ns in sample_size_list]
    return pd.DataFrame(table_rows, columns=columns)


# === Convert DataFrame to LaTeX
def dataframe_to_latex_table(df, scenario):
    header = (
        f"\\begin{{table}}[h]\n"
        f"\\centering\n"
        f"\\caption{{Average runtime (in seconds) for Scenario {scenario}. Values are reported as mean $\\pm$ std over 100 runs.}}\n"
        f"\\label{{tab:runtime_scenario_{scenario.lower()}}}\n"
        f"\\resizebox{{\\textwidth}}{{!}}{{\n"
        f"\\begin{{tabular}}{{l" + "c" * (df.shape[1] - 1) + "}\n"
        f"\\toprule\n"
        + " & ".join(f"\\textbf{{{col}}}" for col in df.columns)
        + " \\\\\n"
        f"\\midrule\n"
    )
    body = ""
    for _, row in df.iterrows():
        row_str = " & ".join(str(cell) for cell in row) + " \\\\\n"
        body += row_str
    footer = "\\bottomrule\n\\end{tabular}}\n\\end{table}"

    return header + body + footer


# === Generate and save all 4 scenario tables
os.makedirs("tables", exist_ok=True)
scenario_lists = {
    "I": list(np.arange(100, 450, 50)),
    "II": list(np.arange(100, 450, 50)),
    "III": list(np.arange(100, 450, 50)),
    "IV": list(np.arange(100, 450, 50)),
}

for scenario, ns_list in scenario_lists.items():
    df_scenario = build_scenario_table(scenario, ns_list)
    df_scenario.to_csv(f"tables/runtime_scenario_{scenario}.csv", index=False)

    latex_code = dataframe_to_latex_table(df_scenario, scenario)
    with open(f"tables/runtime_scenario_{scenario}.tex", "w") as f:
        f.write(latex_code)

    print(f"Saved Scenario {scenario} LaTeX table.")
