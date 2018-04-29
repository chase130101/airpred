#!/bin/bash
#SBATCH -p shared
#SBATCH -N 1 # number of nodes
#SBATCH -n 10 # number of cores
#SBATCH --mem 128000 # memory pool for all cores
#SBATCH -t 0-05:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
python rf_impute_trainTest.py
