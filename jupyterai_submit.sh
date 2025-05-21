#!/bin/bash
# 
# CompecTA (c) 2018
#
# Jupyter job submission script
#
# TODO:
#   - Set name of the job below changing "JupiterNotebook" value.
#   - Set the requested number of nodes (servers) with --nodes parameter.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter. (Total accross all nodes)
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - mid   : For jobs that have maximum run time of 1 day..
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input/output file names below.
#   - If you do not want mail please remove the line that has --mail-type and --mail-user. If you do want to get notification emails, set your email address.
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch jupyter_submit.sh
#
# -= Resources =-
#

#SBATCH --job-name=JupiterNotebook
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu
##SBATCH --constraint=a100
# #SBATCH --gres=gpu:A100
#SBATCH  --mem=10GB
#SBATCH --time=24:00:00
#SBATCH --output=jupyter-%J.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brscn_bzkrt@hotmail.com

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

# Load Anaconda
echo "======================="
echo "Loading Anaconda Module..."
module load miniconda/4.9.2
module load cuda/11.8
source activate tensorenv
echo "======================="


echo
echo "============================== ENVIRONMENT VARIABLES ==============================="
env
echo "===================================================================================="
echo
echo


################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i 6000-6999 -n1)
node=$(hostname -s)
user=$(whoami)

# print tunneling instructions jupyter-log
echo -e "
====================================================================================
 For more info and how to connect from windows, 

 Here is the MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ssh.swc.ucl.ac.uk
SSH login: $user
SSH port: 22

====================================================================================
 MacOS or linux terminal command to create your ssh tunnel on your local machine:

ssh -N -L ${port}:${node}:${port} ${user}@ssh.swc.ucl.ac.uk
====================================================================================

WAIT 1 MINUTE, WILL BE CONNECT ADDRESS APPEARS!

"

# DON'T USE ADDRESS BELOW. 
# DO USE TOKEN BELOW
jupyter-lab --no-browser --port=${port} --ip="*"
