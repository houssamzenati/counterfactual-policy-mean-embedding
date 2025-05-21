from experiment import run_experiment
import matplotlib.pyplot as plt
import seaborn as sns

_, Y_plugin, Y_dr, Y_log, Y_tgt = run_experiment(42, "logistic", "nonlinear", "binary")

def plot_herding_vs_true(Y_log, Y_tgt, Y_herded_dr_cme, Y_herded_plug_in_cme):
    plt.figure(figsize=(8, 5))
    sns.histplot(Y_log, color="red", kde=True, stat="density", label="Logged", alpha=0.3)
    sns.histplot(Y_tgt, color="blue", kde=True, stat="density", label="True $\pi$", alpha=0.3)
    sns.histplot(Y_herded_plug_in_cme, color="orange", kde=True, stat="density", label="Herded PI-CPME", alpha=0.3)
    sns.histplot(Y_herded_dr_cme, color="green", kde=True, stat="density", label="Herded DR-CPME", alpha=0.3)
    plt.xlim([-4, 4])
    plt.xlabel("Y")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Counterfactual Outcome Distribution via Kernel Herding")
    plt.tight_layout()
    plt.show()

plot_herding_vs_true(Y_log, Y_tgt, Y_dr, Y_plugin)