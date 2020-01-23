#!/bin/bash

#SBATCH --time=20:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1   # number of GPUs
#SBATCH --mem-per-cpu=64G   # memory per CPU core
#SBATCH -J "net"   # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load python/3.6
module load tensorflow-gpu/1.9

# Run.
# Usage: <source_mode> <net_mode> <agg_mode> <label_mode> <category_index> <steps_per_epoch> <epochs> [note]
python3 book_net.py paragraph rnn rnn ordinal -1 0 16 normalagg
