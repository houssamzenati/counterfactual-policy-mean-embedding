# counterfactual-policy-mean-embedding
Counterfactual Policy Mean Embeddings 

Implementation for the experiments on Doubly Robust Estimation of Counterfactual Policy Mean Embeddings. 

## Structure

The repository contains three folders for the experiment sections on testing, sampling, and policy evaluation (OPE). Each folders contains: 
- Python scripts, which define the CPME estimators and related benchmark tests. 
- Python notebooks, which make use of the scripts to set the experiments presented in the paper and parse the results.

### Testing

**environment.py** contains the implementation of simulated setting
**dr_kpt.py** contains the implementation of DR-KPT.
**kpt.py** contains the implementation of the KPT test adapted from (Muandet et al 2020) with permutations test.
**experiments.ipynb** stores the results of the different tests for the synthetic data and plots the figures.
**runtime_tables.py** contains the code to generate the time tables to compare computational times.

### Sampling

**environment.py** contains simulated logging policies and target policies
**embeddings.py** contains the plug-in and DR estimators of CPME.
**experiment.py** runs and saves the experiment.
**analyze_results.py** generates tables with distance metrics between herded samples and samples from the true counterfactual distribution. 
**plots.py** generates the plots to see histograms of samples.

## Policy Evaluation 

