#!/bin/bash
#SBATCH -p general
#SBATCH -N 1 # number of nodes
#SBATCH -n 64 # number of cores
#SBATCH --mem 128000 # memory pool for all cores
#SBATCH -t 0-48:00 # time (D-HH:MM)
#SBATCH -o slurm.xgboostCrossVal.%N.%j.out # STDOUT
#SBATCH -e slurm.xgboostCrossVal.%N.%j.err # STDERR
python xgboost_crossVal.py