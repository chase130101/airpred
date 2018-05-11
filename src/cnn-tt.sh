#!/bin/bash
#SBATCH -p general
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 100000 # memory pool for all cores
#SBATCH -t 0-15:00 # time (D-HH:MM)
#SBATCH -o slurm.cnnTT.%N.%j.out # STDOUT
#SBATCH -e slurm.cnnTT.%N.%j.err # STDERR
python CNN_train_test.py
