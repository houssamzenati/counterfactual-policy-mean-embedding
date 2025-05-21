import os
import jsonlines
import pandas as pd


def load_all_results(results_dir="results"):
    records = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(results_dir, filename)
            with jsonlines.open(filepath) as reader:
                records.extend(reader)
    return pd.DataFrame(records)


def make_metric_table(df, action_type, metric_base):
    """
    Build a summary table for a given metric (e.g. 'mmd_unbiased', 'mmd_biased', or 'wass'),
    restricted to reward types 'nonlinear' and 'quadratic'.
    """
    df = df[
        (df["action_type"] == action_type)
        & (df["reward_type"].isin(["nonlinear", "quadratic"]))
    ]
    methods = ["plugin", "dr"]

    rows = []
    for method in methods:
        row = {"Method": "Plug-in" if method == "plugin" else "DR"}
        for (log, rew), group in df.groupby(["logging_type", "reward_type"]):
            metric_col = f"{metric_base}_{method}"
            mean = group[metric_col].mean()
            std = group[metric_col].std()
            colname = f"{log}-{rew}"
            row[colname] = f"{mean:.2e} ± {std:.1e}"
        rows.append(row)

    return pd.DataFrame(rows)


def save_latex_table(df, filename):
    with open(filename, "w") as f:
        f.write(
            df.to_latex(
                index=False, escape=False, column_format="l" + "c" * (df.shape[1] - 1)
            )
        )
    print(f"LaTeX table saved to {filename}")


def save_csv_table(df, filename):
    df.to_csv(filename, index=False)
    print(f"CSV table saved to {filename}")


if __name__ == "__main__":
    df = load_all_results("results")
    action_type = "binary"  # or "continuous" if you wish

    for metric in ["mmd_unbiased", "mmd_biased", "wass"]:
        table = make_metric_table(df, action_type, metric)
        print(f"\n=== {metric.upper()} ===")
        print(table.to_string(index=False))

        save_latex_table(table, f"tables/results_summary_{metric}_{action_type}.tex")
        save_csv_table(table, f"tables/results_summary_{metric}_{action_type}.csv")
