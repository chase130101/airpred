#!/bin/bash
#SBATCH -p bigmem
#SBATCH -N 1 # number of nodes
#SBATCH -n 40 # number of cores
#SBATCH --mem 400000 # memory pool for all cores
#SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH -o slurm.rfImputeTVT.%N.%j.out # STDOUT
#SBATCH -e slurm.rfImputeTVT.%N.%j.err # STDERR
python rf_fit_impute_eval_trainValTest.py
