# AN EFFICIENT DOUBLY-ROBUST TEST FOR THE KERNEL TREATMENT EFFECT

Implementation of the AIPW-xKTE test and related benchmarks. 

## Structure

The repository contains 
- Python scripts, which define the AIPW-xKTE test and related benchmark tests. 
- Python notebooks, which make use of the scripts to set the experiments presented in the paper. 

## Scripts

**xkte.py** contains the implementation of AIPW-xKTE and IPW-xKTE.
**kte.py** contains the implementation of the Kernel Treatment Effect (KTE) test.
**baselines.py** contains a set of tests designed for the testing differences in the average treatment effect. 
**dr_kte.py** contains the DR test presented in Fawkes et al. 

## Notebooks

**generate_data.ipynb** stores the results of the different tests for the synthetic data.
**ihdp_experiment.ipynb** stores the results of the different tests for the ihdp dataset.
**gen_plots.ipynb** generates the plots exhibited in the paper for the synthetic data.  
**read_data_ihdp.ipynb** generates the tables exhibited in the paper for the ihdp dataset.

