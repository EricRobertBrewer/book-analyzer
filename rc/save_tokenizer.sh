#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=64G   # memory per CPU core
#SBATCH -J "tok"   # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load python/3.6
module load tensorflow/1.9

# Run.
python3 -m python.text.save_tokenizer paragraph true
python3 -m python.text.save_tokenizer paragraph false
