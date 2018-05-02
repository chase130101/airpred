#!/bin/bash
#SBATCH -p shared
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 80000 # memory pool for all cores
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -o slurm.olsTT.%N.%j.out # STDOUT
#SBATCH -e slurm.olsTT.%N.%j.err # STDERR
python ols_trainTest.py
