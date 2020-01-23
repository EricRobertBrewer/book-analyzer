#!/bin/bash

#SBATCH --time=40:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1   # number of GPUs
#SBATCH --mem-per-cpu=64G   # memory per CPU core
#SBATCH -J "han"   # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load python/3.6
module load tensorflow-gpu/1.9

# Run.
# Usage: <source_mode> <net_mode> <agg_mode> <label_mode> <batch_size> <steps_per_epoch> <epochs> [note]
python3 book_han.py 1 0 64 1024-pmax
