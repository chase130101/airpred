#!/bin/bash
#SBATCH -p shared
#SBATCH -N 1 # number of nodes
#SBATCH -n 20 # number of cores
#SBATCH --mem 100000 # memory pool for all cores
#SBATCH -t 0-05:00 # time (D-HH:MM)
#SBATCH -o slurm.rfCrossVal.%N.%j.out # STDOUT
#SBATCH -e slurm.rfCrossVal.%N.%j.err # STDERR
python rf_crossVal.py
