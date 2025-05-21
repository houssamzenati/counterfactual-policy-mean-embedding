# Off-Policy Evaluation Experiments

This folder (policy_evaluation) contains Python scripts to reproduce the off-policy evaluation experiments presented in the paper, which are adapted from [1]. We consider five experimental settings, varying the following parameters:
(i) number of observations ($n$),
(ii) number of recommendations ($K$),
(iii) number of users ($N$),
(iv) context dimension ($d$),
(v) policy similarity parameter ($\alpha$).

* The Python scripts are named according to the parameter they vary. For example, to run experiments varying the number of observations, use the following commands (or run the corresponding bash scripts on a cluster):

    ``` python OPE_n_observation_experiments_100.py```
    ``` python OPE_n_observation_experiments_1000.py```
    ``` python OPE_n_observation_experiments_2000.py```
    ``` python OPE_n_observation_experiments_5000.py```

* Upon completion, these scripts generate CSV files containing the results. For the number of observations experiments, the output files are:
  * "Results/Results/OPE_n_observations_result_100_observations.csv"
  * "Results/Results/OPE_n_observations_result_1000_observations.csv"
  * "Results/Results/OPE_n_observations_result_2000_observations.csv"
  * "Results/Results/OPE_n_observations_result_5000_observations.csv"

* The Jupyter notebook Plot_Simulation_Results_Known_Propensities.ipynb reads these CSV files and visualizes the results.

## Python Scripts and their contents

Python Script         |  Explanation
:--------------------:|:-------------------------:
Estimator_CPME.py | Implements IPS, Direct method NN, Doubly robust NN, and our proposed CPME and DR-CPME methods.
Environment.py | Environment classes to define how rewards are generated.
Policy.py | Policy functions and classes.
ParameterSelector.py | Cross-Validation class to select optimal hyperparameters. This procedure is introduced in [1].
visualization_utils.py | utility functions for visualization.

# References
[1] Krikamol Muandet, Motonobu Kanagawa, Sorawit Saengkyongam, and Sanparith Marukatat.364
Counterfactual mean embeddings. Journal of Machine Learning Research, 22(162):1–71, 2021.365