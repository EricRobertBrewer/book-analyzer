#!/bin/bash

#SBATCH --time=01:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1   # number of GPUs
#SBATCH --mem-per-cpu=32G   # memory per CPU core
#SBATCH -J "test"   # job name
#SBATCH --qos=test

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load python/3.6
module load tensorflow-gpu/1.9

# Run.
python3 test.py
