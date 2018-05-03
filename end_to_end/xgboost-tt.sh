#!/bin/bash
#SBATCH -p shared
#SBATCH -N 1 # number of nodes
#SBATCH -n 32 # number of cores
#SBATCH --mem 100000 # memory pool for all cores
#SBATCH -t 0-03:00 # time (D-HH:MM)
#SBATCH -o slurm.xgboostTT.%N.%j.out # STDOUT
#SBATCH -e slurm.xgboostTT.%N.%j.err # STDERR
python xgboost_trainTest.py
