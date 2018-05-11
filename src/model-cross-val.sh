#!/bin/bash
#SBATCH -p shared
#SBATCH -N 1 # number of nodes
#SBATCH -n 32 # number of cores
#SBATCH --mem 100000 # memory pool for all cores
#SBATCH -t 0-24:00 # time (D-HH:MM)
#SBATCH -o slurm.crossVal.%N.%j.out # STDOUT
#SBATCH -e slurm.crossVal.%N.%j.err # STDERR
python model_cross_validation.py ridge 4 ridgeImp
