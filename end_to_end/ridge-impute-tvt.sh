#!/bin/bash
#SBATCH -p shared
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 100000 # memory pool for all cores
#SBATCH -t 0-02:00 # time (D-HH:MM)
#SBATCH -o slurm.ridgeImputeTVT.%N.%j.out # STDOUT
#SBATCH -e slurm.ridgeImputeTVT.%N.%j.err # STDERR
python ridge_impute_eval_trainValTest.py
