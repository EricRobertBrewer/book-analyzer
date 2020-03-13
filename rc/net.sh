#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1   # number of GPUs
#SBATCH --mem-per-cpu=64G   # memory per CPU core
#SBATCH -J "net"   # job name

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load python/3.6
module load tensorflow-gpu/1.9

# Run.
python3 -m python.classifiers.book_net 0 \
  --source_mode paragraph \
  --net_mode cnn \
  --agg_mode maxavg \
  --book_dense_units 128 \
  --book_dropout 0.5 \
  --label_mode ordinal \
  --epochs 256
